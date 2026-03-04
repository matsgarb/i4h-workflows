
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift a teddy bear with a robotic arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp
import time

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401
from robotic.surgery.tasks.surgical.lift.lift_env_cfg import LiftEnvCfg
# from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    OPEN_GRIPPER = wp.constant(5)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.6)
    LIFT_OBJECT = wp.constant(1.0)
    OPEN_GRIPPER = wp.constant(0.0)

# NEW
PICK_REST = 0
PICK_APPROACH_ABOVE = 1
PICK_APPROACH = 2
PICK_GRASP = 3
PICK_LIFT = 4
PICK_OPEN = 5


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.OPEN_GRIPPER
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.OPEN_GRIPPER:
        # des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.OPEN_GRIPPER:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.OPEN_GRIPPER
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.2
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Deformable-PSM-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env_cfg.viewer.eye = (2.1, 1.0, 1.3)
    
    # create environment
    env = gym.make("Isaac-Lift-Deformable-PSM-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # ## attach deformable object to robot end-effector
    # from pxr import Usd, Sdf, Vt, PhysxSchema
    # import omni.usd
    # stage = omni.usd.get_context().get_stage()
    # deformable_prim_path = Sdf.Path("/World/envs/env_0/Object")
    # rigid_actor_prim_path = Sdf.Path("/World/envs/env_0/Robot/psm_tool_tip_link")
    # attachment_node_indices = [0, 1, 2]
    # attach_prim_path = deformable_prim_path.AppendChild("PhysicsAttachment")
    # attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attach_prim_path)
    # prim = attachment.GetPrim()
    # # Do not pre-assign actor targets here; the runtime attachment block will set
    # # `physx:actor0` and `physx:actor1` to the correct mesh/finger prims at grasp time.
    # try:
    #     if hasattr(attachment, "CreateIndicesAttr"):
    #         attachment.CreateIndicesAttr().Set(Vt.IntArray(attachment_node_indices))
    #     else:
    #         prim.CreateAttribute("physx:indices", Sdf.ValueTypeNames.IntArray).Set(
    #             Vt.IntArray(attachment_node_indices)
    #         )
    # except Exception as e:
    #     print("Failed to set attachment indices:", e)
    #     print("attachment helpers:", [m for m in dir(attachment) if 'Index' in m or 'Indices' in m or 'Actor' in m])


    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    # print("Action space:", actions.shape)
    actions[:, 3] = 1.0

    # desired rotation after grasping
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    object_grasp_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    # z-axis pointing down and 45 degrees rotation
    object_grasp_orientation[:, 1] = 0.9238795
    object_grasp_orientation[:, 2] = -0.3826834
    object_local_grasp_position = torch.tensor([0.0, 0.0, -0.005], device=env.unwrapped.device) # [0.02, -0.08, 0.0] [0.0, 0.0, -0.015]

    # create state machine
    pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    # NEW
    object = env.unwrapped.scene["object"]
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    object_nodal_kinematic_target = object.data.nodal_kinematic_target.clone()
    grasp_vertex_idx = torch.zeros((num_envs,), dtype=torch.long, device=device)
    grasp_active = torch.zeros(num_envs, dtype=torch.bool, device=device)


    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            ee_pos_world = ee_frame_sensor.data.target_pos_w[..., 0, :].clone()
            tcp_rest_position = ee_pos_world - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            object_position += object_local_grasp_position

            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, object_grasp_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            # NEW
            sm_state_now = pick_sm.sm_state
            grasp_envs = (sm_state_now >= PICK_GRASP) & (sm_state_now <= PICK_LIFT)
            release_envs = sm_state_now >= PICK_OPEN 
            if grasp_envs.any():
                ee_pos = ee_pos_world[grasp_envs]
                v_idx = grasp_vertex_idx[grasp_envs]
                object_nodal_kinematic_target[grasp_envs, v_idx, 0] = ee_pos[:, 0]
                object_nodal_kinematic_target[grasp_envs, v_idx, 1] = ee_pos[:, 1]
                object_nodal_kinematic_target[grasp_envs, v_idx, 2] = ee_pos[:, 2] + 0.03
                object_nodal_kinematic_target[grasp_envs, v_idx, 3] = 0.0  # 0 => constrained
                grasp_active[grasp_envs] = True
            
            if release_envs.any():
                # v_idx = grasp_vertex_idx[release_envs]
                # object_nodal_kinematic_target[release_envs, v_idx, 3] = 1.0  # 1 => free
                print("Kinematic flags after release:", object.data.nodal_kinematic_target[0,:,3].unique())
                nodal_state = object.data.nodal_state_w
                object_nodal_kinematic_target[release_envs, :, :3] = nodal_state[release_envs, :, :3]  # 1 => free
                object_nodal_kinematic_target[release_envs, :, 3] = 1.0
                grasp_active[release_envs] = False
            
            object.write_nodal_kinematic_target_to_sim(object_nodal_kinematic_target)

            # # --- RUNTIME ATTACHMENT BLOCK (DROPABLE) ---
            # # This block attempts to create a deformable->rigid attachment at the
            # # moment the gripper closes and is near the object. To disable, comment
            # # out or remove this entire block (search for the header above).
            # try:
            #     ATTACH_ON_GRASP = True
            #     if ATTACH_ON_GRASP:
            #         # only create once
            #         if not hasattr(env.unwrapped, "_attachment_done") or not env.unwrapped._attachment_done:
            #             # read current gripper command from newly computed actions
            #             try:
            #                 gripper_cmd = float(actions.reshape(-1, actions.shape[-1])[0, -1])
            #             except Exception:
            #                 try:
            #                     gripper_cmd = float(actions[..., -1].cpu().numpy().reshape(-1)[0])
            #                 except Exception:
            #                     gripper_cmd = None

            #             import numpy as _np
            #             ee_pos = tcp_rest_position[0].cpu().numpy() if hasattr(tcp_rest_position, "cpu") else tcp_rest_position[0]
            #             obj_pos = object_position[0].cpu().numpy() if hasattr(object_position, "cpu") else object_position[0]
            #             dist = _np.linalg.norm(_np.asarray(ee_pos) - _np.asarray(obj_pos))
            #             contact_thresh = 0.02

            #             if (gripper_cmd is not None) and (gripper_cmd < 0.0) and (dist < contact_thresh):
            #                 print("[attachment] Gripper closed + proximity detected (dist=", dist, ") — creating attachment")

            #                 # Read deformable nodal positions using the canonical API
            #                 deformable_data = None
            #                 try:
            #                     deformable_data = env.unwrapped.scene["object"].data
            #                 except Exception as e:
            #                     print("[attachment] deformable_data lookup failed:", e)

            #                 pts = None
            #                 if deformable_data is not None and hasattr(deformable_data, "nodal_pos_w"):
            #                     try:
            #                         nodal = deformable_data.nodal_pos_w
            #                         # nodal shape: (num_envs, N, 3)
            #                         if hasattr(nodal, "cpu"):
            #                             nodal_np = nodal.cpu().numpy()
            #                         else:
            #                             nodal_np = _np.asarray(nodal)
            #                         if nodal_np.ndim == 3:
            #                             pts = nodal_np[0]
            #                         elif nodal_np.ndim == 2:
            #                             pts = nodal_np
            #                         print(f"[attachment] read nodal_pos_w -> pts.shape={None if pts is None else pts.shape}")
            #                     except Exception as e:
            #                         print("[attachment] reading nodal_pos_w failed:", e)

            #                 # USD fallback (rare) — not preferred for runtime attachments
            #                 if pts is None:
            #                     try:
            #                         from pxr import UsdGeom
            #                         prim_obj = stage.GetPrimAtPath(deformable_prim_path)
            #                         if prim_obj and prim_obj.IsValid():
            #                             pts_attr = UsdGeom.Points(prim_obj).GetPointsAttr()
            #                             pts_vt = pts_attr.Get()
            #                             if pts_vt is not None:
            #                                 pts = _np.array([[p[0], p[1], p[2]] for p in pts_vt])
            #                                 print(f"[attachment] USD fallback read {pts.shape[0]} points")
            #                     except Exception as e:
            #                         print("[attachment] USD fallback failed:", e)

            #                 if pts is None:
            #                     print("[attachment] No deformable nodal positions available — cannot attach now")
            #                 else:
            #                     # pick nearest nodes
            #                     dists = _np.linalg.norm(pts - _np.reshape(ee_pos, (1, 3)), axis=1)
            #                     k = min(8, pts.shape[0])
            #                     idxs = _np.argsort(dists)[:k].tolist()
            #                     print("[attachment] selected node indices:", idxs)

            #                     # compute local Vec3 positions for the selected nodes (relative to deformable root if available)
            #                     try:
            #                         root = None
            #                         if hasattr(deformable_data, "root_pos_w"):
            #                             root = deformable_data.root_pos_w
            #                             if hasattr(root, "cpu"):
            #                                 root_np = root.cpu().numpy()
            #                                 if root_np.ndim == 2:
            #                                     root_np = root_np[0]
            #                                 root = root_np
            #                         local_pts = pts[idxs]
            #                         if root is not None:
            #                             local_pts = local_pts - root
            #                         from pxr import Gf
            #                         vecs = [Gf.Vec3f(float(x), float(y), float(z)) for (x, y, z) in local_pts]
            #                     except Exception as e:
            #                         print("[attachment] failed computing local Vec3 positions:", e)
            #                         vecs = None

            #                     # set indices via schema if possible and ensure points arrays exist
            #                     try:
            #                         if hasattr(attachment, "CreateIndicesAttr"):
            #                             try:
            #                                 attachment.CreateIndicesAttr().Set(Vt.IntArray(idxs))
            #                                 print("[attachment] wrote indices via CreateIndicesAttr")
            #                             except Exception:
            #                                 # still ensure a prim attribute is present for visibility
            #                                 prim.CreateAttribute("physx:indices", Sdf.ValueTypeNames.IntArray).Set(Vt.IntArray(idxs))
            #                                 print("[attachment] wrote physx:indices attribute as fallback")
            #                         else:
            #                             prim.CreateAttribute("physx:indices", Sdf.ValueTypeNames.IntArray).Set(Vt.IntArray(idxs))
            #                             print("[attachment] wrote physx:indices attribute")

            #                         # Always write points0 and points1 Vec3 arrays when vecs available
            #                         if vecs is not None:
            #                             try:
            #                                 attr0 = prim.GetAttribute("points0")
            #                                 if not attr0 or not attr0.IsValid():
            #                                     prim.CreateAttribute("points0", Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray(vecs))
            #                                     print("[attachment] created 'points0' with Vec3f positions")
            #                                 else:
            #                                     try:
            #                                         cur0 = attr0.Get()
            #                                         if not cur0:
            #                                             attr0.Set(Vt.Vec3fArray(vecs))
            #                                             print("[attachment] set existing 'points0' attribute")
            #                                     except Exception:
            #                                         attr0.Set(Vt.Vec3fArray(vecs))
            #                                         print("[attachment] overwrote 'points0' attribute with Vec3f positions")

            #                                 attr1 = prim.GetAttribute("points1")
            #                                 if not attr1 or not attr1.IsValid():
            #                                     prim.CreateAttribute("points1", Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray(vecs))
            #                                     print("[attachment] created 'points1' with Vec3f positions")
            #                                 else:
            #                                     try:
            #                                         cur1 = attr1.Get()
            #                                         if not cur1:
            #                                             attr1.Set(Vt.Vec3fArray(vecs))
            #                                             print("[attachment] set existing 'points1' attribute")
            #                                     except Exception:
            #                                         attr1.Set(Vt.Vec3fArray(vecs))
            #                                         print("[attachment] overwrote 'points1' attribute with Vec3f positions")
            #                             except Exception as e:
            #                                 print("[attachment] failed writing points attributes:", e)
            #                         else:
            #                             # fallback: write world-space positions (pts[idxs]) if local vecs unavailable
            #                             try:
            #                                 from pxr import Gf
            #                                 fb_pts = pts[idxs]
            #                                 fb_vecs = [Gf.Vec3f(float(x), float(y), float(z)) for (x, y, z) in fb_pts]
            #                                 try:
            #                                     attr0 = prim.GetAttribute("points0")
            #                                     if not attr0 or not attr0.IsValid():
            #                                         prim.CreateAttribute("points0", Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray(fb_vecs))
            #                                         print("[attachment] created 'points0' (world-space) with Vec3f positions")
            #                                     else:
            #                                         try:
            #                                             cur0 = attr0.Get()
            #                                             if not cur0:
            #                                                 attr0.Set(Vt.Vec3fArray(fb_vecs))
            #                                                 print("[attachment] set existing 'points0' attribute (world-space)")
            #                                         except Exception:
            #                                             attr0.Set(Vt.Vec3fArray(fb_vecs))
            #                                             print("[attachment] overwrote 'points0' attribute with world-space Vec3f positions")

            #                                     attr1 = prim.GetAttribute("points1")
            #                                     if not attr1 or not attr1.IsValid():
            #                                         prim.CreateAttribute("points1", Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray(fb_vecs))
            #                                         print("[attachment] created 'points1' (world-space) with Vec3f positions")
            #                                     else:
            #                                         try:
            #                                             cur1 = attr1.Get()
            #                                             if not cur1:
            #                                                 attr1.Set(Vt.Vec3fArray(fb_vecs))
            #                                                 print("[attachment] set existing 'points1' attribute (world-space)")
            #                                         except Exception:
            #                                             attr1.Set(Vt.Vec3fArray(fb_vecs))
            #                                             print("[attachment] overwrote 'points1' attribute with world-space Vec3f positions")
            #                                 except Exception as e:
            #                                     print("[attachment] failed writing fallback points attributes:", e)
            #                             except Exception as e:
            #                                 print("[attachment] vecs not available; skipping points0/points1 write", e)

            #                     except Exception as e:
            #                         print("[attachment] failed to set attachment indices/points:", e)

            #                     # mark so we don't repeatedly try to attach
            #                     # Create actor relationships now that indices/points exist
            #                     try:
            #                         # prefer object mesh under the deformable prim
            #                         try:
            #                             obj_mesh_path = deformable_prim_path.AppendChild("mesh")
            #                             if not stage.GetPrimAtPath(obj_mesh_path) or not stage.GetPrimAtPath(obj_mesh_path).IsValid():
            #                                 obj_mesh_path = deformable_prim_path
            #                         except Exception:
            #                             obj_mesh_path = deformable_prim_path

            #                         # prefer explicit finger visual prim, fallback search under Robot
            #                         try:
            #                             finger_path = Sdf.Path("/World/envs/env_0/Robot/panda_leftfinger/visuals/panda_leftfinger")
            #                             if not stage.GetPrimAtPath(finger_path) or not stage.GetPrimAtPath(finger_path).IsValid():
            #                                 # BFS search for likely finger prim names
            #                                 robot_root = Sdf.Path("/World/envs/env_0/Robot")
            #                                 def _find_prim_by_name(stage, base_path, tokens):
            #                                     base = stage.GetPrimAtPath(base_path)
            #                                     if not base or not base.IsValid():
            #                                         return None
            #                                     queue = list(base.GetChildren())
            #                                     while queue:
            #                                         c = queue.pop(0)
            #                                         try:
            #                                             name = c.GetName().lower()
            #                                         except Exception:
            #                                             name = ""
            #                                         if any(t in name for t in tokens):
            #                                             return c.GetPath()
            #                                         try:
            #                                             queue.extend(list(c.GetChildren()))
            #                                         except Exception:
            #                                             pass
            #                                     return None

            #                                 candidate_f = _find_prim_by_name(stage, robot_root, ["panda_leftfinger", "leftfinger", "finger"])
            #                                 if candidate_f:
            #                                     finger_path = candidate_f
            #                                 else:
            #                                     finger_path = rigid_actor_prim_path
            #                         except Exception:
            #                             finger_path = rigid_actor_prim_path

            #                         try:
            #                             r0 = prim.CreateRelationship("physx:actor0")
            #                             r0.AddTarget(obj_mesh_path)
            #                             r1 = prim.CreateRelationship("physx:actor1")
            #                             r1.AddTarget(finger_path)
            #                             # also set non-namespaced variants
            #                             try:
            #                                 r0b = prim.CreateRelationship("actor0")
            #                                 r0b.AddTarget(obj_mesh_path)
            #                                 r1b = prim.CreateRelationship("actor1")
            #                                 r1b.AddTarget(finger_path)
            #                             except Exception:
            #                                 pass
            #                             print(f"[attachment] created actor relationships: actor0->{obj_mesh_path}, actor1->{finger_path}")
            #                             # Try enabling the attachment flag and set filter types to vertex (best-effort)
            #                             try:
            #                                 attr_enabled = prim.GetAttribute("attachmentEnabled")
            #                                 if attr_enabled and attr_enabled.IsValid():
            #                                     attr_enabled.Set(True)
            #                                     print("[attachment] set 'attachmentEnabled' = True")
            #                                 try:
            #                                     a0 = prim.GetAttribute("filterType0")
            #                                     if a0 and a0.IsValid():
            #                                         a0.Set("vertex")
            #                                         print("[attachment] set 'filterType0' = 'vertex'")
            #                                 except Exception:
            #                                     pass
            #                                 try:
            #                                     a1 = prim.GetAttribute("filterType1")
            #                                     if a1 and a1.IsValid():
            #                                         a1.Set("vertex")
            #                                         print("[attachment] set 'filterType1' = 'vertex'")
            #                                 except Exception:
            #                                     pass
            #                             except Exception as e:
            #                                 print("[attachment] failed to enable attachment flag:", e)
            #                             # Search for physics-specific prims under the mesh and finger targets
            #                             try:
            #                                 from pxr import PhysxSchema

            #                                 def _find_physics_candidates(stage, base_path, max_results=8):
            #                                     res = []
            #                                     base = stage.GetPrimAtPath(base_path)
            #                                     if not base or not base.IsValid():
            #                                         return res
            #                                     queue = [base]
            #                                     while queue and len(res) < max_results:
            #                                         p = queue.pop(0)
            #                                         try:
            #                                             tname = str(p.GetTypeName()).lower()
            #                                         except Exception:
            #                                             tname = ""
            #                                         name = p.GetName().lower() if p.GetName() else ""
            #                                         has_physx_api = False
            #                                         try:
            #                                             has_physx_api = PhysxSchema.PhysxRigidBodyAPI.HasAPI(p)
            #                                         except Exception:
            #                                             has_physx_api = False
            #                                         if has_physx_api or any(x in tname for x in ("rigid", "phys", "shape", "actor")) or any(x in name for x in ("rigid", "phys", "shape", "actor")):
            #                                             res.append(p.GetPath())
            #                                         try:
            #                                             queue.extend(list(p.GetChildren()))
            #                                         except Exception:
            #                                             pass
            #                                     return res

            #                                 obj_candidates = _find_physics_candidates(stage, obj_mesh_path)
            #                                 finger_candidates = _find_physics_candidates(stage, finger_path)
            #                                 print("[attachment][diagnostic] object physics candidates:", obj_candidates)
            #                                 print("[attachment][diagnostic] finger physics candidates:", finger_candidates)

            #                                 # If better physics prims were found, repoint actor relationships
            #                                 try:
            #                                     if obj_candidates:
            #                                         new_obj = obj_candidates[0]
            #                                         # replace actor0 targets
            #                                         try:
            #                                             r0.SetTargets([new_obj])
            #                                             print(f"[attachment] repointed physx:actor0 -> {new_obj}")
            #                                         except Exception:
            #                                             try:
            #                                                 rel = prim.CreateRelationship("physx:actor0")
            #                                                 rel.AddTarget(new_obj)
            #                                             except Exception:
            #                                                 pass
            #                                     if finger_candidates:
            #                                         new_f = finger_candidates[0]
            #                                         try:
            #                                             r1.SetTargets([new_f])
            #                                             print(f"[attachment] repointed physx:actor1 -> {new_f}")
            #                                         except Exception:
            #                                             try:
            #                                                 rel = prim.CreateRelationship("physx:actor1")
            #                                                 rel.AddTarget(new_f)
            #                                             except Exception:
            #                                                 pass
            #                                 except Exception as e:
            #                                     print("[attachment][diagnostic] failed to repoint actor relationships:", e)
            #                             except Exception as e:
            #                                 print("[attachment][diagnostic] physics candidate search failed:", e)
            #                             # If the USD-level attachment did not produce a physical bond,
            #                             # attempt a physics-level weld/D6 constraint between the
            #                             # object mesh actor and the finger rigid body as a fallback.
            #                             try:
            #                                 import omni.physx as _omni_physx
            #                                 physx_iface = None
            #                                 try:
            #                                     physx_iface = _omni_physx.get_physx_interface()
            #                                 except Exception:
            #                                     try:
            #                                         physx_iface = _omni_physx.acquire_physx_interface()
            #                                     except Exception:
            #                                         physx_iface = None

            #                                 if physx_iface is not None:
            #                                 ee_frame_sensor.data.target_pos_w[..., 0, :].clone()    # Find actors for the two prims
            #                                     try:
            #                                         # helper to get physx actor from prim path
            #                                         def _get_actor_for_prim(stage, prim_path):
            #                                             try:
            #                                                 from omni.physx.scripts import utils as _physx_utils
            #                                                 prim = stage.GetPrimAtPath(prim_path)
            #                                                 if not prim or not prim.IsValid():
            #                                                     return None
            #                                                 actor = _physx_utils.get_rigid_body_actor(prim)
            #                                                 return actor
            #                                             except Exception:
            #                                                 return None

            #                                         actor_obj = _get_actor_for_prim(stage, str(obj_mesh_path))
            #                                 ee_frame_sensor.data.target_pos_w[..., 0, :].clone()        actor_finger = _get_actor_for_prim(stage, str(finger_path))

            #                                         if actor_obj is None or actor_finger is None:
            #                                             print("[attachment][constraint] could not resolve one or both PhysX actors; skipping constraint fallback")
            #                                         else:
            #                                             # Create a fixed (weld) joint between actor_finger and actor_obj
            #                                             try:
            #                                                 # Use physx_iface to create constraint if available
            #                                                 # Best-effort: try D6 joint via API helper
            #                                                 from omni.physx.scripts import joint as _joint
            #                                                 # compute world frames at current transforms
            #                                                 obj_world_pose = actor_obj.get_global_pose()
            #                                                 finger_world_pose = actor_finger.get_global_pose()
            #                                                 # create a fixed-like D6 with all DOF locked
            #                                                 j = _joint.create_fixed_joint_between_actors(actor_finger, actor_obj)
            #                                                 if j is not None:
            #                                                     print("[attachment][constraint] created fixed joint between finger and object (weld)")
            #                                                 else:
            #                                                     print("[attachment][constraint] joint creation returned None")
            #                                             except Exception as e:
            #                                                 print("[attachment][constraint] failed creating joint:", e)
            #                                     except Exception as e:
            #                                         print("[attachment][constraint] resolving actors failed:", e)
            #                                 else:
            #                                     print("[attachment][constraint] PhysX interface unavailable; skipping constraint fallback")
            #                             except Exception as e:
            #                                 print("[attachment][constraint] unexpected error:", e)
            #                         except Exception as e:
            #                             print("[attachment] failed creating actor relationships:", e)

            #                     except Exception:
            #                         pass

            #                     try:
            #                         env.unwrapped._attachment_done = True
            #                     except Exception:
            #                         pass
            # except Exception as e:
            #     print("[attachment] error:", e)
            # # --- END RUNTIME ATTACHMENT BLOCK ---

            # reset state machine
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()