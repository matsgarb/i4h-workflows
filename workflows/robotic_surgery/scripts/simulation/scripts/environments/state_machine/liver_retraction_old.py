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
app_launcher = AppLauncher(headless=args_cli.headless, livestream=args_cli.livestream) # HEADLESS = runs a program or browser without a graphical user interface (GUI), it operates in the background without a visual window
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence

import gymnasium as gym # to create physical env
import robotic.surgery.tasks  # noqa: F401
import torch
import math
import warp as wp # for writing high-performance simulation and graphics code
from isaaclab.assets import RigidObject # rigid body is described by its pose, velocity and mass distribution
from isaaclab.utils.math import subtract_frame_transforms # to have pose in different referent systems
from isaaclab.utils.math import combine_frame_transforms, quat_mul
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from robotic.surgery.tasks.surgical.liver_retraction.reach_env_cfg import ReachEnvCfg # task: Reach
import isaaclab.sim.utils as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from pxr import UsdGeom, Gf, Usd
import isaacsim.core.utils.stage as stage_utils
# initialize warp
wp.init()

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class ReachSmState:
    """States for the reach state machine (rest position or action/reach the target)."""

    REST = wp.constant(0)
    REACH = wp.constant(1)


class ReachSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.5)
    REACH = wp.constant(2.0)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),  # time-step for each env
    sm_state: wp.array(dtype=int),  # current state
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    des_final_pose: wp.array(dtype=wp.transform),  # target posefrom isaacsim.core.prims import TextPrimView

    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
):
    # retrieve thread id
    tid = wp.tid() 
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == ReachSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]  # if REST: des pose = current pose (stay still)
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= ReachSmWaitTime.REST: # when REST time passes, next state 
            # move to next state and reset wait time
            sm_state[tid] = ReachSmState.REACH
            sm_wait_time[tid] = 0.0
    elif state == ReachSmState.REACH:
        des_ee_pose[tid] = des_final_pose[tid]  # move to the desired pose
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= ReachSmWaitTime.REACH:
            # keep in REACH state and reset timer
            sm_state[tid] = ReachSmState.REACH
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class ReachSm:
    """A simple state machine in a robot's task space for a reach task.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector, and outputs the desired state of the robot's end-effectorso the visual centroid is centered at the environment origin. Values computed from
        #.
    The state machine is implemented as a finite state machine with the following states:

    1. REST: The robot is at rest.
    2. REACH: The robot reaches to the desired pose. This is the final state.
    """

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
        # initialize state machine (for each env create a torch vector dt, state and timer)
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device) # dt between 2 control steps
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device) # time passed

        # desired state (pose expressed as quaternion 4 + 3 position x,y,z)
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, des_final_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector."""
        # convert transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_final_pose = des_final_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        des_final_pose_wp = wp.from_torch(des_final_pose.contiguous(), wp.transform)

        # run state machine kernel
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                des_final_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torchimport
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg: ReachEnvCfg = parse_env_cfg(
        "Isaac-Liver-PSM-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Liver-PSM-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()
    
    obj = env.unwrapped.scene["object"]
    # obj.data.nodal_pos_w = n_instances x n_vertices x 3 [i_istance, i_vertice, [xi yi zi]]
    nodal_pos_env = obj.data.nodal_pos_w[0] # n_vertices x 3 [i_vertice, [xi yi zi]]]
    # anchor_idx = torch.argmax(nodal_pos_env[:,2]).item() 
    anchor_idx = 319
    
    robot = env.unwrapped.scene["robot"]
    print("Joint names: ", robot.data.joint_names)
    print("Joint pos limits (lower, upper): ")
    print(robot.data.joint_pos_limits)

    print("Anchor node index:",nodal_pos_env.size())
    num_nodes = nodal_pos_env.shape[0]

    # FRAME VISUALIZATION
    if not args_cli.headless and args_cli.world_ref_vis:
        # World frame
        world_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/WorldFrame")
        world_frame_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
        world_frame_vis = VisualizationMarkers(world_frame_cfg)

        # Robot base frame
        robot_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/RobotFrame")
        robot_frame_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        robot_frame_vis = VisualizationMarkers(robot_frame_cfg)

        # EE (TCP) frame
        ee_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame")
        ee_frame_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        ee_frame_vis = VisualizationMarkers(ee_frame_cfg)

        # Object frame
        object_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/ObjectFrame")
        object_frame_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        object_frame_vis = VisualizationMarkers(object_frame_cfg)

        # Desired target position marker
        desired_pose_vis = VisualizationMarkers(
            POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/DesiredPose"))
        nodes_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/LiverNodes")
        liver_nodes_vis = VisualizationMarkers(nodes_cfg)
        
        # Desired EE Frame (target pose)
        desired_ee_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/DesiredEEFrame")
        desired_ee_frame_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        desired_ee_frame_vis = VisualizationMarkers(desired_ee_frame_cfg)
        

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    
    # create state machine
    reach_sm = ReachSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2] # send the desired EE pose (actions) as long as the app is open (the robot move forward the target)
            
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
            
            # # object pose wrt robot + offset
            # obj = env.unwrapped.scene["object"]
            # object_pos_w = obj.data.root_pos_w
            # object_quat_w = tcp_rest_orientation # obj.data.root_quat_w NO because root_quat_w defined only for Rigid Bodies
            # object_position_world = object_pos_w - env_origins  # (N,3)
            # object_position_b, _ = subtract_frame_transforms(
            #     robot_pos_w, robot_quat_w, object_position_world
            # )
            # print("Object position: ",object_position_b)
            # object_orientation = object_quat_w  # (N,4) quaternion as (w,x,y,z)
            # object_local_offset = torch.tensor([0.03, 0.09, 0.008], device=env.unwrapped.device, dtype=object_position_b.dtype)
            # desired_pose = torch.cat([object_position_b + object_local_offset, object_orientation], dim=-1)  # (N,7)
            # # desired_pose = env.unwrapped.command_manager.get_command("object_pose")
            # print("Desired pose: ",desired_pose)

            # NEW
            obj = env.unwrapped.scene["object"]
            nodal_pos_w = obj.data.nodal_pos_w
            anchor_pos_w = nodal_pos_w[:, anchor_idx, :]
            anchor_pos_env = anchor_pos_w - env_origins
            # desired_pos_b, _ = subtract_frame_transforms(
            #     robot_pos_w, robot_quat_w, anchor_pos_env
            # )
            
            # desired_quat_b = tcp_rest_orientation
            yaw = math.radians(90.0)
            pitch = math.radians(50.0) # psm_pitch_end_joint
            
            w_yaw = math.cos(yaw / 2.0)
            z_yaw = math.sin(yaw / 2.0)
            w_pitch = math.cos(pitch / 2.0)
            y_pitch = math.sin(pitch / 2.0)
            
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
            
            # desired_quat_w = q_pitch.repeat(env.unwrapped.num_envs, 1)
            desired_quat_w_single = quat_mul(q_pitch, q_yaw)
            desired_quat_w = desired_quat_w_single.repeat(env.unwrapped.num_envs, 1)
            
            desired_pos_b, desired_quat_b = subtract_frame_transforms(
                robot_pos_w, robot_quat_w, anchor_pos_env, desired_quat_w
            )
            
            desired_pose = torch.cat([desired_pos_b, desired_quat_b], dim=-1)


            # # Target position in world RF
            # desired_pos_b = desired_pose[:, :3]
            # desired_quat_b = desired_pose[:, 3:7]
            des_pos_w, des_quat_w = combine_frame_transforms(
                robot_pos_w, robot_quat_w, desired_pos_b, desired_quat_b
            )


            # Visualization of RF (if python ... --world_ref_vis)
            if not args_cli.headless and args_cli.world_ref_vis:
                world_frame_vis.visualize(world_pos, world_quat)
                robot_frame_vis.visualize(robot_pos_w, robot_quat_w)
                ee_frame_vis.visualize(ee_pos_w, ee_quat_w)
                
                # NEW
                object_pos_w = obj.data.root_pos_w
                object_quat_w = world_quat
                object_frame_vis.visualize(object_pos_w, object_quat_w)
                marker_indices = torch.zeros(env.unwrapped.num_envs, dtype=torch.int32, device=env.unwrapped.device)
                desired_pose_vis.visualize(anchor_pos_w, object_quat_w, marker_indices=marker_indices)
                
                # NEW NEW
                nodal_pos_w = obj.data.nodal_pos_w[0]
                num_nodes = nodal_pos_w.shape[0]
                node_quats = torch.zeros((num_nodes,4),device=env.unwrapped.device)
                node_quats[:,0] = 1.0
                nodes_indices = torch.zeros(num_nodes,dtype=torch.int32,device=env.unwrapped.device)
                liver_nodes_vis.visualize(nodal_pos_w, node_quats, marker_indices=nodes_indices)
                desired_ee_frame_vis.visualize(des_pos_w, des_quat_w)



                # object_frame_vis.visualize(object_pos_w, object_quat_w)
                # object_frame_vis.visualize(object_position_world, object_quat_w)
                # marker_indices = torch.zeros(env.unwrapped.num_envs, dtype=torch.int32, device=env.unwrapped.device)
                # desired_pose_vis.visualize(des_pos_w, des_quat_w, marker_indices=marker_indices)

                
            actions = reach_sm.compute(torch.cat([tcp_rest_position_b, tcp_rest_orientation], dim=-1), desired_pose)
            
            # reset state machine
            if dones.any():
                reach_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                print("Resetting the state machine.")

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
