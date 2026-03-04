import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Reach state machine for psm platform environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--world_ref_vis", action="store_true", default=False, help="Enable visualization of world reference frame.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.") # default = num of envs
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, livestream=args_cli.livestream, enable_cameras=args_cli.enable_cameras) # HEADLESS = runs a program or browser without a graphical user interface (GUI), it operates in the background without a visual window
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence

import gymnasium as gym # to create physical env
import robotic.surgery.tasks  # noqa: F401
import torch
import math
import warp as wp # for writing high-performance simulation and graphics code
from isaaclab.assets import RigidObject # rigid body is described by its pose, velocity and mass distribution
from isaaclab.utils.math import combine_frame_transforms, quat_mul, subtract_frame_transforms, quat_error_magnitude, quat_apply
import numpy as np
from PIL import Image
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from robotic.surgery.tasks.surgical.liver_retraction.reach_env_cfg import ReachEnvCfg # task: Reach
import isaaclab.sim.utils as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.utils.math import quat_from_matrix
from pxr import UsdGeom, Gf, Usd
import isaacsim.core.utils.stage as stage_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
# initialize warp
wp.init()
import robotic.surgery.tasks.surgical.liver_retraction.mdp as mdp
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import liver_target_pose_world


class ReachSmState:
    """States for the reach state machine (rest position or action/reach the target)."""

    REST = wp.constant(0)
    REACH = wp.constant(1)
    # insert and lift
    INSERT = wp.constant(2)
    LIFT = wp.constant(3)


class ReachSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(1.0)
    REACH = wp.constant(8.0)  # Increased from 2.0 to 5.0 to allow more time for IK convergence
    # insert and lift
    INSERT = wp.constant(2.0)
    LIFT = wp.constant(2.0)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),  # time-step for each env
    sm_state: wp.array(dtype=int),  # current state
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    
    des_reach_pose: wp.array(dtype=wp.transform), 
    # insert and lift
    des_insert_pose: wp.array(dtype=wp.transform), 
    des_lift_pose: wp.array(dtype=wp.transform), 

    des_ee_pose: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid() 
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == ReachSmState.REST:
        des_ee_pose[tid] = ee_pose[tid] 
        if sm_wait_time[tid] >= ReachSmWaitTime.REST:
            sm_state[tid] = ReachSmState.REACH
            sm_wait_time[tid] = 0.0
            
    elif state == ReachSmState.REACH:
        des_ee_pose[tid] = des_reach_pose[tid] 
        if sm_wait_time[tid] >= ReachSmWaitTime.REACH:
            sm_state[tid] = ReachSmState.INSERT # ReachSmState.REACH, insert and lift
            sm_wait_time[tid] = 0.0

    # insert and lift
    elif state == ReachSmState.INSERT:
        des_ee_pose[tid] = des_insert_pose[tid] 
        if sm_wait_time[tid] >= ReachSmWaitTime.INSERT:
            sm_state[tid] = ReachSmState.LIFT
            sm_wait_time[tid] = 0.0

    elif state == ReachSmState.LIFT:
        des_ee_pose[tid] = des_lift_pose[tid]  
        if sm_wait_time[tid] >= ReachSmWaitTime.LIFT:
            sm_state[tid] = ReachSmState.LIFT
            sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class ReachSm:
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine (for each env create a torch vector dt, state and timer)
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device) # dt between 2 control steps
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device) # time passed

        # desired state (pose expressed as quaternion 4 + 3 position x,y,z)
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    # def compute(self, ee_pose: torch.Tensor, des_reach_pose: torch.Tensor):
    # insert and lift
    def compute(self, ee_pose: torch.Tensor, des_reach_pose: torch.Tensor, des_insert_pose: torch.Tensor, des_lift_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector."""
        
        # convert transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_reach_pose = des_reach_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        des_reach_pose_wp = wp.from_torch(des_reach_pose.contiguous(), wp.transform)
              
        # insert and lift
        des_insert_pose = des_insert_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_lift_pose = des_lift_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_insert_pose_wp = wp.from_torch(des_insert_pose.contiguous(), wp.transform)
        des_lift_pose_wp = wp.from_torch(des_lift_pose.contiguous(), wp.transform)
        
        # run state machine kernel
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                des_reach_pose_wp,
                des_insert_pose_wp,
                des_lift_pose_wp,
                self.des_ee_pose_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torchimport
        return des_ee_pose


def main():
    # parse configuration
    env_cfg: ReachEnvCfg = parse_env_cfg(
        "Isaac-Liver-PSM-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Liver-PSM-v0", cfg=env_cfg) # Isaac-Liver-PSM-IK-Abs-v0
    
    # print action info
    print("Action space: ", env.unwrapped.action_space)
    print("Action space shape: ", env.unwrapped.action_space.shape)
    print("Num envs: ", env.unwrapped.num_envs)
    
    # reset environment at start
    env.reset()
    
    # camera
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    # # Check if camera exists in scene
    # try:
    #     camera = env.unwrapped.scene.sensors["camera"]
    #     print("✓ Camera found in scene")
    # except (KeyError, AttributeError) as e:
    #     print(f"✗ Camera not found: {e}")
    #     camera = None
    
    pos_threshold = 0.002
    or_threshold = 0.1
    step_idx = 0
    
    # Create output directory for images
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = f"camera_images_{timestamp}"
    # os.makedirs(output_dir, exist_ok=True)

    # # RGB images
    # output_dir = os.path.join("logs","rgb_observations_"+timestamp)
    # os.makedirs(output_dir, exist_ok=True)
    
    # liver
    liver = env.unwrapped.scene["liver"]
    nodal_pos_env = liver.data.nodal_pos_w[0] # n_vertices x 3 [i_vertice, [xi yi zi]]]
    anchor_idx = 319 # OLD MESH (onlyliver.usd), resolution = 8
    anchor_idx = 3 # final_organs_5
    # anchor_idx = 335 # NEW MESH (liver_and_gallbladder.usd, resolution = 10)
    # anchor_idx = 26 # NEW MESH (liver_and_gallbladder.usd, resolution = 8), target point on the surface of the liver

    # anchor_idx = 331 # NEW MESH (liver_and_gallbladder.usd, resolution = 9), target point below the liver
    # anchor_idx = 286 # NEW MESH (liver_and_gallbladder.usd, resolution = 9), targer point on the surface of the liver
    robot = env.unwrapped.scene["robot"]
    print("Joint names: ", robot.data.joint_names)
    print("Joint pos limits (lower, upper): ")
    print(robot.data.joint_pos_limits)

    # print("Anchor node index:",nodal_pos_env.size())
    num_nodes = nodal_pos_env.shape[0]

    # FRAME VISUALIZATION
    if not args_cli.headless and args_cli.world_ref_vis:
        # World frame
        world_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/WorldFrame")
        world_frame_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
        world_frame_vis = VisualizationMarkers(world_frame_cfg)

        # # Robot base frame
        # robot_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/RobotFrame")
        # robot_frame_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        # robot_frame_vis = VisualizationMarkers(robot_frame_cfg)

        # EE (TCP) frame
        ee_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame")
        ee_frame_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        ee_frame_vis = VisualizationMarkers(ee_frame_cfg)

        # # Object frame
        # object_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/ObjectFrame")
        # object_frame_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        # object_frame_vis = VisualizationMarkers(object_frame_cfg)

        # # TO VISUALIZE LIVER NODES (simulation mesh) OLD 
        # desired_pose_vis = VisualizationMarkers(
        #     POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/DesiredPose"))
        # nodes_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/LiverNodes")
        # liver_nodes_vis = VisualizationMarkers(nodes_cfg)

        # TO VISUALIZE LIVER NODES (simulation mesh) UPDATED
        if not args_cli.headless and args_cli.world_ref_vis:
            import isaacsim.core.utils.prims as prim_utils
            from pxr import UsdGeom, Sdf, Gf
            stage = stage_utils.get_current_stage()
            parent_path = "/Visuals/LiverNodesList"
            if not stage.GetPrimAtPath(parent_path):
                UsdGeom.Scope.Define(stage, parent_path)
            for i in range(num_nodes):
                node_path = f"{parent_path}/Node_{i}"
                prim = prim_utils.create_prim(prim_path=node_path, prim_type="Sphere")
                sphere_geom = UsdGeom.Sphere(prim)
                sphere_geom.GetRadiusAttr().Set(0.0015)
                color_attr = sphere_geom.GetDisplayColorAttr()
                if not color_attr.HasValue():
                    color_attr.Set([Gf.Vec3f(1.0, 0.0, 0.0)])
        
        # Desired EE Frame (target pose)
        desired_ee_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/ReachTarget")
        desired_ee_frame_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        desired_ee_frame_vis = VisualizationMarkers(desired_ee_frame_cfg)
        
        # Insert target marker
        insert_target_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/InsertTarget")
        insert_target_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        insert_target_vis = VisualizationMarkers(insert_target_cfg)
        # insert_target_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/InsertTarget")
        # insert_target_vis = VisualizationMarkers(insert_target_cfg)

        # Lift target marker
        lift_target_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/LiftTarget")
        lift_target_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        lift_target_vis = VisualizationMarkers(lift_target_cfg)

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    
    # create state machine
    reach_sm = ReachSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            # dones = env.step(actions)[-2] # send the desired EE pose (actions) as long as the app is open (the robot move forward the target)
            step_out = env.step(actions)
            env.unwrapped.scene.update(dt=env_cfg.sim.dt)
            terminated = step_out[2]
            truncated = step_out[3]
            dones = terminated | truncated

            # Log joint positions at the first step of each (per-env) episode
            try:
                if hasattr(env.unwrapped, "episode_length_buf"):
                    just_reset_mask = env.unwrapped.episode_length_buf == 1
                    if torch.any(just_reset_mask):
                        reset_env_ids = torch.nonzero(just_reset_mask, as_tuple=False).squeeze(-1)
                        robot: RigidObject = env.unwrapped.scene["robot"]
                        jp = robot.data.joint_pos[reset_env_ids].detach().cpu().numpy()
                        print(f"[episode reset] env_ids={reset_env_ids.tolist()} joint_pos={jp}")
            except Exception as e:
                print(f"[DEBUG] joint reset log error: {e}")

            # DEBUG: Print joint positions and limits every N steps
            try:
                if step_idx % 50 == 0:  # Print every 50 steps to avoid spam
                    robot: RigidObject = env.unwrapped.scene["robot"]
                    joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()  # First env
                    joint_pos_limits = robot.data.joint_pos_limits[0].detach().cpu().numpy()  # (N_joints, 2) -> [lower, upper]
                    
                    print(f"\n[Step {step_idx}] Joint Positions (env 0):")
                    print(f"  Joint names: {robot.data.joint_names}")
                    for i, (name, pos) in enumerate(zip(robot.data.joint_names, joint_pos)):
                        lower, upper = joint_pos_limits[i]
                        in_limits = lower <= pos <= upper
                        status = "✓" if in_limits else "✗ OUT OF LIMITS"
                        print(f"    {name:30s} pos={pos:8.4f}  limits=[{lower:8.4f}, {upper:8.4f}]  {status}")
            except Exception as e:
                print(f"[DEBUG] joint debug print error: {e}")

            # simplest runtime phase print (works for tensor/int)
            phase = getattr(env.unwrapped, "_phase", None)
            # print(f"Phase={phase}")
            
            # world
            env_origins = env.unwrapped.scene.env_origins  # shape (num_envs, 3)
            world_pos = env_origins
            world_quat = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
            world_quat[:, 0] = 1.0  
            
            # robot
            robot: RigidObject = env.unwrapped.scene["robot"]
            robot_pos_w = robot.data.root_state_w[:, :3]
            robot_quat_w = robot.data.root_state_w[:, 3:7]
            
            # end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            ee_pos_w = ee_frame_sensor.data.target_pos_w[..., 0, :].clone()
            ee_quat_w = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            
            tcp_rest_position = ee_pos_w - env_origins
            tcp_rest_position_b, _ = subtract_frame_transforms(
                robot_pos_w, robot_quat_w, tcp_rest_position
            )
            tcp_rest_orientation = ee_quat_w
            
            # Desired position
            liver = env.unwrapped.scene["liver"]
            nodal_pos_w = liver.data.nodal_pos_w
            anchor_pos_w = nodal_pos_w[:, anchor_idx, :]
            env_origins = env.scene.env_origins
            
            # OLD: anchor_pos_env = anchor_pos_w - env_origins
            # OLD: dz = 0.008 #0.007
            # OLD: approach_pos_w = anchor_pos_w + torch.tensor([0.0, 0.0, -dz], device = env.unwrapped.device)
            # OLD: approach_pos_env = approach_pos_w - env_origins
            # OLD: anchor_pos_env = approach_pos_env
            # NEW: keep reach z fixed to the initial/default nodal height, allow x/y to follow the deformable
            
            # dz = 0.007 # |0.008| ##0.007
            dz = -0.025
            default_anchor_pos = liver.data.default_nodal_state_w[:, anchor_idx, :3].to(anchor_pos_w.device)
            reach_pos_w = anchor_pos_w.clone()
            
            reach_pos_w[:, 2] = default_anchor_pos[:, 2] - dz
            # reach_pos_w[:, 2] = reach_pos_w[:, 2] - dz 
            
            approach_pos_w = reach_pos_w
            approach_pos_env = approach_pos_w - env_origins
            anchor_pos_env = approach_pos_env
            
            
            # Desired orientation defined as pitch and yaw of world rf orientation
            yaw = math.radians(45.0) # 0.0
            pitch = math.radians(50.0) # psm_pitch_end_joint. 50.0
            roll = math.radians(-45.0) # 0.0
            
            w_yaw = math.cos(yaw / 2.0)
            z_yaw = math.sin(yaw / 2.0)
            
            w_pitch = math.cos(pitch / 2.0)
            y_pitch = math.sin(pitch / 2.0)
            
            w_roll = math.cos(roll / 2.0)
            x_roll = math.sin(roll / 2.0)
            
            q_yaw = torch.tensor(
            	[w_yaw, 0.0, 0.0, z_yaw],
            	device = env.unwrapped.device,
            	dtype = torch.float32,
            )
            q_pitch = torch.tensor(
            	[w_pitch, 0.0, y_pitch, 0.0],
            	device = env.unwrapped.device,
            	dtype = torch.float32,
            )
            q_roll = torch.tensor(
            	[w_roll, x_roll, 0.0, 0.0],
            	device = env.unwrapped.device,
            	dtype = torch.float32,
            )
            
            # desired_quat_w = q_pitch.repeat(env.unwrapped.num_envs, 1)
            quat_w_single = quat_mul(q_pitch, q_roll)
            desired_quat_w_single = quat_mul(quat_w_single, q_yaw)
            desired_quat_w_single = liver_target_pose_world(env.unwrapped, object_cfg=SceneEntityCfg("liver"))[0, 3:7]  # Override with orientation from rewards.py
            desired_quat_w = desired_quat_w_single.repeat(env.unwrapped.num_envs, 1)
            
            '''
            # Desired orientation
            # nodal_pos_w = liver.data.nodal_pos_w
            # bary_pos_w = nodal_pos_w.mean(dim=1)
            bary_pos_w = liver.data.root_pos_w
            z_axis = - (bary_pos_w - approach_pos_w)
            z_axis = z_axis / (torch.norm(z_axis, dim = 1, keepdim = True) + 1e-8)
            world_x = torch.tensor([1.0, 0.0, 0.0], device = env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            world_y = torch.tensor([0.0, 1.0, 0.0], device = env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            dot_x = torch.abs((z_axis*world_x).sum(dim=1, keepdim=True))
            ref_axis = torch.where(dot_x > 0.95, world_y, world_x)
            x_axis = ref_axis - (ref_axis*z_axis).sum(dim=1, keepdim=True)*z_axis
            x_axis = x_axis / (torch.norm(x_axis, dim=1, keepdim=True) + 1e-8)
            y_axis = torch.cross(z_axis, x_axis, dim=1)
            y_axis = y_axis / (torch.norm(y_axis, dim=1, keepdim=True) + 1e-8)
            R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
            desired_quat_w = quat_from_matrix(R)
            '''
            
            desired_pos_b, desired_quat_b = subtract_frame_transforms(
                robot_pos_w, robot_quat_w, anchor_pos_env, desired_quat_w
            )
            
            desired_pose = torch.cat([desired_pos_b, desired_quat_b], dim=-1)
            pose_reach = desired_pose
            
            # insert and lift: use horizontal component of EE z (projected onto XY plane)
            d_insert = 0.0 # 0.03
            ee_z_w = quat_apply(desired_quat_w, torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32).repeat(num_envs, 1))
            ee_z_h = ee_z_w.clone()
            ee_z_h[:, 2] = 0.0
            norms = torch.norm(ee_z_h, dim=1, keepdim=True)
            eps = 1e-6
            fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32).repeat(num_envs, 1)
            ee_dir_h = torch.where(norms > eps, ee_z_h / norms, fallback)
            
            insert_pos_env = anchor_pos_env - d_insert * (ee_dir_h) # ee_z_h / norms instead of ee_dir_h
            insert_pos_b, insert_quat_b = subtract_frame_transforms(robot_pos_w, robot_quat_w, insert_pos_env, desired_quat_w)
            pose_insert = torch.cat([insert_pos_b, insert_quat_b], dim=-1)
            
            d_lift = 0.03 # 0.08
            # # LIFT OLD
            # lift_pos_env = insert_pos_env + torch.tensor([0.0, 0.0, d_lift], device=device, dtype=torch.float32).repeat(num_envs,1)
            # lift_pos_b, lift_quat_b = subtract_frame_transforms(robot_pos_w, robot_quat_w, lift_pos_env, desired_quat_w)
            # pose_lift = torch.cat([lift_pos_b, lift_quat_b], dim=-1)
            
            # LIFT NEW
            lift_pos_w_fixed = default_anchor_pos.clone()
            lift_pos_w_fixed[:, 2] += d_lift
            lift_pos_env_fixed = lift_pos_w_fixed - env_origins # This is the correct variable
            lift_pos_b, lift_quat_b = subtract_frame_transforms(
                robot_pos_w, robot_quat_w, lift_pos_env_fixed, desired_quat_w
            )
            pose_lift = torch.cat([lift_pos_b, lift_quat_b], dim=-1)

            # DEBUG: Print distance and orientation errors every 50 steps
            if step_idx % 50 == 0:
                sm_state_names = {0: "REST", 1: "REACH", 2: "INSERT", 3: "LIFT"}
                current_state = reach_sm.sm_state[0].item()
                wait_time = reach_sm.sm_wait_time[0].item()
                state_name = sm_state_names.get(current_state, "UNKNOWN")
                print(f"  State machine: {state_name} (wait_time={wait_time:.3f}s)")
                
                try:
                    # Distance (meters)
                    robot_body_idx = robot.find_bodies("psm_tool_tip_link")[0]
                    r_dist = mdp.object_ee_distance(
                        env.unwrapped,
                        asset_cfg=SceneEntityCfg("robot", body_ids=robot_body_idx),
                        object_cfg=SceneEntityCfg("liver"),
                    )
                    print(f"  Liver distance: {r_dist}")
                except Exception as e:
                    print("[DEBUG] distance reward error:", e)
                
                try:
                    # Orientation error (radians)
                    robot_body_idx = robot.find_bodies("psm_tool_tip_link")[0]
                    r_orient = mdp.object_ee_orientation_error(
                        env.unwrapped,
                        asset_cfg=SceneEntityCfg("robot", body_ids=robot_body_idx),
                        object_cfg=SceneEntityCfg("liver"),
                    )
                    print(f"  Liver orientation error: {r_orient}")
                except Exception as e:
                    print("[DEBUG] orientation reward error:", e)   
            
            try:
                # Insert lift (meters + constant offset)
                r_lift = mdp.insert_lift_reward(env.unwrapped)
                # print(f"insert_lift   : {r_lift}")
            except Exception as e:
                print("[DEBUG] insert lift reward error:", e)
            
            try:
                # Reach → Insert transition bonus
                r_transition = mdp.reach_to_insert_transition_bonus(env.unwrapped)
                # print(f"transition    : {r_transition}")
            except Exception as e:
                print("[DEBUG] transition reward error:", e)

            # try:
            #     # Final success
            #     robot_body_idx = robot.find_bodies("psm_tool_tip_link")[0]
            #     r_success = mdp.final_success_reward(
            #         env.unwrapped,
            #         asset_cfg=SceneEntityCfg("robot", body_ids=robot_body_idx),
            #         object_cfg=SceneEntityCfg("liver"),
            #     )
            #     print(f"success       : {r_success}")
            # except Exception as e:
            #     print("[DEBUG] success reward error:", e)

            try:
                # Action rate penalty 
                r_action_rate = mdp.action_rate_l2(env.unwrapped)
                # print(f"action_rate   : {r_action_rate}")
            except Exception as e:
                print("[DEBUG] action rate reward error:", e)   
            
            try:
                # Joint velocity penalty 
                r_joint_vel = mdp.joint_vel_l2(
                    env.unwrapped,
                    asset_cfg=SceneEntityCfg("robot"),
                )
                # print(f"joint_vel     : {r_joint_vel}")
            except Exception as e:
                print("[DEBUG] joint vel reward error:", e)
            
            
            insert_pos_w, insert_quat_w = combine_frame_transforms(
                robot_pos_w, robot_quat_w, insert_pos_b, insert_quat_b,
            )
            
            des_pos_w, des_quat_w = combine_frame_transforms(
                robot_pos_w, robot_quat_w, desired_pos_b, desired_quat_b
            )


            # Visualization of RF (if python ... --world_ref_vis)
            if not args_cli.headless and args_cli.world_ref_vis:
                
                world_frame_vis.visualize(world_pos, world_quat)
                # robot_frame_vis.visualize(robot_pos_w, robot_quat_w)
                ee_frame_vis.visualize(ee_pos_w, ee_quat_w)
                
                # NEW
                # object_pos_w = liver.data.root_pos_w
                # object_quat_w = world_quat
                # object_frame_vis.visualize(object_pos_w, object_quat_w)
                # marker_indices = torch.zeros(env.unwrapped.num_envs, dtype=torch.int32, device=env.unwrapped.device)
                # desired_pose_vis.visualize(anchor_pos_w, object_quat_w, marker_indices=marker_indices)
                
                
                # VISUALIZE LIVER NODES (simulation mesh)
                nodal_pos_w = liver.data.nodal_pos_w[0]
                num_nodes = nodal_pos_w.shape[0]
                node_quats = torch.zeros((num_nodes,4),device=env.unwrapped.device)
                node_quats[:,0] = 1.0
                nodes_indices = torch.zeros(num_nodes,dtype=torch.int32,device=env.unwrapped.device)
                # # OLD
                # # liver_nodes_vis.visualize(nodal_pos_w, node_quats, marker_indices=nodes_indices)

                # UPDATED
                if not args_cli.headless and args_cli.world_ref_vis and step_idx % 2 == 0:
                    from pxr import Gf, UsdGeom
                    curr_stage = stage_utils.get_current_stage() # Assicuriamoci di avere lo stage
                    for i in range(num_nodes):
                        node_path = f"/Visuals/LiverNodesList/Node_{i}"
                        pos = nodal_pos_w[i].tolist()
                        prim = curr_stage.GetPrimAtPath(node_path)
                        if prim.IsValid():
                            xformable = UsdGeom.Xformable(prim)
                            translate_op = next((op for op in xformable.GetOrderedXformOps() 
                                               if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
                            if not translate_op:
                                translate_op = xformable.AddTranslateOp()
                            translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))

                desired_ee_frame_vis.visualize(des_pos_w, des_quat_w)
                insert_target_vis.visualize(insert_pos_w, insert_quat_w)
                lift_target_vis.visualize(lift_pos_env_fixed + env_origins, desired_quat_w)
            
            
            # # INSTANCE SEGMENTATION IMAGES ([liver, gallbladder, robot, table, background])
            # if camera is not None:
            #     should_save = (step_idx % 30 == 0)
            #     if should_save:
            #         try:
            #             cam_out = camera.data.output
            #             image = cam_out.get("instance_id_segmentation_fast", None) # "rgb", "depth", "semantic_segmentation", "instance_segmentation_fast", "instance_id_segmentation_fast"
            #             if image is not None:
            #                 # Save from first env only
            #                 img = image[0]  # (H, W, C)
            #                 np = img.detach().cpu().numpy()

            #                 # Convert to uint8
            #                 if np.max() <= 1.0:
            #                     np = (np * 255).astype(np.uint8)

            #                 final_img = Image.fromarray(np)
            #                 filename = f"{output_dir}/env0_step{step_idx:05d}.png"
            #                 final_img.save(filename)
            #                 print(f"[Step {step_idx}] Saved: {filename}")
            #             else:
            #                 print(f"[Step {step_idx}] Camera output has no data")
            #         except Exception as e:
            #             print(f"[Step {step_idx}] Error saving image: {e}")


            # ## BINARY MASK FOR GALLBLADDER AREA EXTRACTION ([gallbladder vs non-gallbladder])
            # if camera is not None:
            #     should_save = (step_idx % 10 == 0)

            #     if should_save:
            #         try:
            #             cam_out = camera.data.output
            #             image_data = cam_out.get("instance_segmentation_fast", None)
            #             all_info_list = camera.data.info
                        
            #             if image_data is not None and len(all_info_list) > 0:
            #                 info_env0 = all_info_list[0]
            #                 seg_info = info_env0.get("instance_segmentation_fast", {})
            #                 id_to_labels = seg_info.get("idToLabels", {})

            #                 # RGBA --> (H, W, 4) 
            #                 seg_rgba = image_data[0].detach().cpu().numpy()

            #                 # Find the color tuple for Gallbladder                                      
            #                 gb_color = None
            #                 for color_tuple, path in id_to_labels.items():
            #                     if "Gallbladder" in str(path):
            #                         gb_color = color_tuple 
            #                         break

            #                 if gb_color is not None:
            #                     # Mask creation
            #                     mask_boolean = (seg_rgba == gb_color).all(axis=-1)
            #                     binary_mask = (mask_boolean).astype(np.uint8) * 255
                                
            #                     # Save binary mask image
            #                     final_img = Image.fromarray(binary_mask)
            #                     # filename = f"{output_dir}/gb_mask_step{step_idx:05d}.png"
            #                     # final_img.save(filename)
                                
            #                     num_pixels = (binary_mask == 255).sum()
            #                     print(f"[Step {step_idx}] Mask Saved! GB Pixels: {num_pixels}")
            #                 else:
            #                     print(f"[Step {step_idx}] GB path not found in labels: {list(id_to_labels.values())}")
                        
            #         except Exception as e:
            #             print(f"[Step {step_idx}] Error: {e}")
            #             import traceback
            #             traceback.print_exc()

            #     step_idx += 1

            # # ## RGB images
            # if camera is not None and step_idx % 1 == 0:
            #     try:
            #         rgb_tensor = mdp.camera_rgb_observation(env.unwrapped)
            #         img_tensor = rgb_tensor[0].permute(1,2,0)
            #         img_tensor = (img_tensor * 255.0).clamp(0,255).to(torch.uint8)
            #         img_np = img_tensor.cpu().numpy()
            #         final_img = Image.fromarray(img_np)
            #         # if final_img.size != (256,256):
            #         #     final_img = final_img.resize((256,256), Image.Resampling.LANCZOS)
            #         filename = os.path.join(output_dir, f"env0_step{step_idx:05d}.png")
            #         final_img.save(filename)
            #     # try:
            #     #     rgb_tensor = mdp.camera_rgb_stack_observation(env.unwrapped)
            #     #     frames = rgb_tensor[0].chunk(4, dim=0)
            #     #     frames_np = []
            #     #     for f in frames:
            #     #         f_proc = f.permute(1, 2, 0).clamp(0, 1) * 255.0
            #     #         frames_np.append(f_proc.detach().cpu().to(torch.uint8).numpy())
            #     #     img_combined_np = np.concatenate(frames_np, axis=1)
            #     #     final_img = Image.fromarray(img_combined_np)
            #     #     if final_img.width > 1024:
            #     #         final_img = final_img.resize((1024, int(1024 * final_img.height / final_img.width)), Image.Resampling.LANCZOS)
            #     #     filename = os.path.join(output_dir, f"env0_step{step_idx:05d}.png")
            #     #     final_img.save(filename)
            #     except Exception as e:
            #         print(f"[Step {step_idx}] Error saving RGB image: {e}")
            
            # if camera is not None and step_idx % 1 == 0:
            #     try:
            #         rgb_tensor = mdp.camera_rgb_observation(env.unwrapped)
                    
            #         img_tensor = rgb_tensor[0].permute(1, 2, 0)
                    
            #         img_tensor = (img_tensor * 255.0).clamp(0, 255).to(torch.uint8)
                    
            #         img_np = img_tensor.cpu().numpy()
            #         final_img = Image.fromarray(img_np)
            #         filename = os.path.join(output_dir, f"env0_step{step_idx:05d}_raw_84x84.png")
            #         final_img.save(filename)

            #         # vis_img = final_img.resize((336, 336), Image.Resampling.NEAREST) # 84 * 4 = 336
            #         # filename_vis = os.path.join(output_dir, f"env0_step{step_idx:05d}_vis_zoom.png")
            #         # vis_img.save(filename_vis)
                    
            #     except Exception as e:

            #         print(f"[Step {step_idx}] Error saving RGB image: {e}")

            step_idx += 1

            # DEBUG: Check state transitions
            prev_state = reach_sm.sm_state[0].item()
            prev_wait_time = reach_sm.sm_wait_time[0].item()
            
            # insert and lift
            # actions = reach_sm.compute(torch.cat([tcp_rest_position_b, tcp_rest_orientation], dim=-1), desired_pose)
            actions = reach_sm.compute(torch.cat([tcp_rest_position_b, tcp_rest_orientation], dim=-1), pose_reach, pose_insert, pose_lift)
            
            # Check if state changed (REACH -> INSERT)
            new_state = reach_sm.sm_state[0].item()
            if prev_state == 1 and new_state == 2:  # REACH (1) -> INSERT (2)
                try:
                    robot_body_idx = robot.find_bodies("psm_tool_tip_link")[0]
                    dist_at_transition = mdp.object_ee_distance(
                        env.unwrapped,
                        asset_cfg=SceneEntityCfg("robot", body_ids=robot_body_idx),
                        object_cfg=SceneEntityCfg("liver"),
                    )
                    orient_at_transition = mdp.object_ee_orientation_error(
                        env.unwrapped,
                        asset_cfg=SceneEntityCfg("robot", body_ids=robot_body_idx),
                        object_cfg=SceneEntityCfg("liver"),
                    )
                    print(f"\n!!! TRANSITION REACH->INSERT at step {step_idx}")
                    print(f"    BEFORE compute: wait_time={prev_wait_time:.4f}s (should be >= 2.0)")
                    print(f"    Expected wait_time >= 2.0s, but got {prev_wait_time:.4f}s!")
                    print(f"    Distance: {dist_at_transition}")
                    print(f"    Orientation error: {orient_at_transition}\n")
                except Exception as e:
                    print(f"[DEBUG] transition error: {e}")
            
            # reset state machine
            if dones.any():
                reach_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                print("Resetting the state machine.", reach_sm.sm_state[0].item(),"wait: ", reach_sm.sm_wait_time[0].item())

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
