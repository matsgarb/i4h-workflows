
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Reach and grasp state machine of a deformable sheet for psm platform environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# if use_fabric = False -> Fabric disabled, uses USD read/write I/O instead
# if use_fabric = True -> Fabric enabled: high-performance. GPU-backed path
parser.add_argument("--world_ref_vis", action="store_true", default=False, help="Enable visualization of world reference frame.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.") 
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, livestream=args_cli.livestream) # HEADLESS = runs a program or browser without a graphical user interface (GUI), it operates in the background without a visual window
simulation_app = app_launcher.app


# Libraries import
from collections.abc import Sequence
import gymnasium as gym # to create physical env
import robotic.surgery.tasks  # noqa: F401
import torch, math
import warp as wp # for writing high-performance simulation and graphics code
import robotic.surgery.tasks.surgical.tissue_retraction.config.psm
from isaaclab.assets import RigidObject
from isaaclab.utils.math import subtract_frame_transforms # to have pose in different referent systems
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from robotic.surgery.tasks.surgical.lift.lift_env_cfg import LiftEnvCfg
from robotic.surgery.tasks.surgical.tissue_retraction.tissue_retraction_env_cfg import TissueRetractionEnvCfg


# STATES for the state machine
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


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.5)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(0.7)
    GRASP_OBJECT = wp.constant(0.5)
    LIFT_OBJECT = wp.constant(2.0)

PICK_REST = 0
PICK_APPROACH_ABOVE_OBJECT = 1
PICK_APPROACH_OBJECT = 2
PICK_GRASP_OBJECT = 3
PICK_LIFT_OBJECT = 4


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
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
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
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

# Tissue Grasping State Machine
class TissueGraspSM:
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
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
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.05
        self.offset[:, -1] = 1.0 

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
        # self.contact_z = -0.05
        # self.contact_x = 0.0
        # self.contact_y = 0.0
        # object_pose[:,2] += self.contact_z
        # object_pose[:,0] += self.contact_x
        # object_pose[:,1] += self.contact_y
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
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg: TissueRetractionEnvCfg = parse_env_cfg(
        "Isaac-TissueRetraction-PSM-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Lift-Deformable-PSM-IK-Abs-v0",cfg=env_cfg)
    # reset environment at start
    env.reset()

    env_cfg.episode_length_s = 8.0

    robot: RigidObject = env.unwrapped.scene["robot"]

    # new
    tissue = env.unwrapped.scene["object"]
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    grasp_vertex_idx = torch.zeros((num_envs,), dtype=torch.long, device=device)
    ee_local_offset = torch.tensor([0.0, 0.0, 0.01], device=device)

    # visualize global/world reference frame at origin (only when GUI enabled)
    if args_cli.world_ref_vis and not args_cli.headless:
        cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/WorldFrame")
        cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        world_frame_vis = VisualizationMarkers(cfg)
        world_pos = torch.tensor([[0.0, 0.0, 0.0]], device=env.unwrapped.device)
        world_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)

    # create action buffers (position + quaternion) -> [x,y,z,qw,qx,qy,qz]
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0 # nuovo
    print("Action space:", actions.shape)
    print("tissue root_pos_w[0]:", tissue.data.root_pos_w[0].cpu().numpy())
    print("env_origins[0]:", env.unwrapped.scene.env_origins[0].cpu().numpy())
    

    # state machine
    tissue_grasp_sm = TissueGraspSM(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
    # object_local_grasp_position = torch.tensor([0.0, 0.30, 0.0],device=env.unwrapped.device,dtype=env.unwrapped.scene["object"].data.root_pos_w.dtype,)
    object_local_grasp_position = torch.tensor([0.0, 0.0, 0.0],device=env.unwrapped.device,dtype=env.unwrapped.scene["object"].data.root_pos_w.dtype,)
    # object_local_grasp_position = torch.tensor([0.005, 0.0, 0.001],device=env.unwrapped.device,dtype=env.unwrapped.scene["object"].data.root_pos_w.dtype,) deformable block

    # NEW
    tissue = env.unwrapped.scene["object"]
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    tissue_nodal_kinematic_target = tissue.data.nodal_kinematic_target.clone() 
    grasp_vertex_idx = torch.zeros((num_envs,), dtype=torch.long, device=device)
    grasp_offset_z = 0.003


    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]
            # observations
            robot: RigidObject = env.unwrapped.scene["robot"]
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            ee_pos_world = ee_frame_sensor.data.target_pos_w[..., 0, :].clone()
            tcp_rest_position = ee_pos_world - env.unwrapped.scene.env_origins
            tcp_rest_position_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], tcp_rest_position
            )
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame (tissue barycenter -> base frame), build 7D pose
            tissue = env.unwrapped.scene["object"]
            object_position = tissue.data.root_pos_w - env.unwrapped.scene.env_origins         
            object_position += object_local_grasp_position     
            # object_position = torch.tensor([0.0, -0.08, 0.03],device=env.unwrapped.device)           
            object_position_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_position
            )

            # single quaternion (w, x, y, z)
            # q = torch.tensor(
            #     [math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0],
            #     device=env.unwrapped.device,
            #     dtype=tcp_rest_orientation.dtype,
            # )
            # expand to (num_envs, 4)
            # object_orientation = q.unsqueeze(0).expand(int(env.unwrapped.num_envs), -1).contiguous()
            object_orientation = tcp_rest_orientation # nuovo

            # obj_quat_b = tcp_rest_orientation # vecchio
            # object_orientation = torch.cat([object_position_b, obj_quat_b], dim=-1)     # vecchio
            

            # -- target object pose 
            desired_pose = env.unwrapped.command_manager.get_command("object_pose")
            if desired_pose.shape[-1] == 3:
                desired_pose = torch.cat([desired_pose, tcp_rest_orientation], dim=-1)
            
            # advance state machine
            actions = tissue_grasp_sm.compute(
                torch.cat([tcp_rest_position_b, tcp_rest_orientation], dim=-1),
                torch.cat([object_position_b, object_orientation], dim=-1),
                desired_pose,
            )

            ## NEW
            # stato corrente della state machine (tensor int32, shape: (num_envs,))
            sm_state_now = tissue_grasp_sm.sm_state

            # env in cui siamo in GRASP o LIFT
            grasp_envs = (sm_state_now >= PICK_GRASP_OBJECT) & (sm_state_now <= PICK_LIFT_OBJECT)

            if grasp_envs.any():
                # posizione dell'EE in world per quegli env
                ee_pos = ee_pos_world[grasp_envs]  # (N_grasp, 3)

                # indice vertice (per ora 0) per quegli env
                v_idx = grasp_vertex_idx[grasp_envs]  # tutti zeri

                # aggiorna kinematic targets: pos = (x,y,z), flag=0 -> kinematic
                tissue_nodal_kinematic_target[grasp_envs, v_idx, 0] = ee_pos[:, 0]
                tissue_nodal_kinematic_target[grasp_envs, v_idx, 1] = ee_pos[:, 1]
                tissue_nodal_kinematic_target[grasp_envs, v_idx, 2] = ee_pos[:, 2] + grasp_offset_z
                tissue_nodal_kinematic_target[grasp_envs, v_idx, 3] = 0.0  # 0 => constrained

            # (per ora niente release: il nodo resta sempre vincolato dopo il grasp)
            tissue.write_nodal_kinematic_target_to_sim(tissue_nodal_kinematic_target)


            # ALWAYS KEEP GRIPPER CLOSED
            # sm_out = tissue_grasp_sm.compute(
            #     torch.cat([tcp_rest_position_b, tcp_rest_orientation], dim=-1),
            #     torch.cat([object_position_b, object_orientation], dim=-1),
            #     desired_pose,
            # )
            # # force gripper always closed: set gripper column (last column) to closed value (-1.0)
            # # sm_out is a torch tensor shaped (N, 8) -> last column is gripper
            # sm_out[:, -1] = -1.0
            # actions = sm_out


            '''# state machine -> desired EE pose (7) + gripper (1)
            sm_out = tissue_grasp_sm.compute(
                torch.cat([tcp_rest_position_b, tcp_rest_orientation], dim=-1),  # (N,7)
                object_pose_b,                                                   # (N,7)
                desired_pose,                                                    # (N,7)
            )
            des_ee_pose_b, grip = sm_out[:, :7], sm_out[:, 7]

            # pack to full action vector [6 body | 1 gripper | 7 ik_abs]
            actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
            print("Action space:", actions.shape)
            actions[:, 6] = grip
            actions[:, 7:14] = des_ee_pose_b''' # vecchio

            # reset state machine
            if dones.any():
                env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                tissue_grasp_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                print("Resetting the state machine.")
                nodal_state = tissue.data.nodal_state_w.clone()
                tissue_nodal_kinematic_target[env_ids, :, :3] = nodal_state[env_ids, :, :3]
                tissue_nodal_kinematic_target[env_ids, :, 3] = 1.0
                tissue.write_nodal_kinematic_target_to_sim(tissue_nodal_kinematic_target, env_ids=env_ids)


   
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


    # "Isaac-Lift-Deformable-PSM-IK-Abs-v0"