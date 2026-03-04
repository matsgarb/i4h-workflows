# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a reach state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Reach state machine for psm platform environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
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
import warp as wp # for writing high-performance simulation and graphics code
from isaaclab.assets import RigidObject # rigid body is described by its pose, velocity and mass distribution
from isaaclab.utils.math import subtract_frame_transforms # to have pose in different referent systems
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from robotic.surgery.tasks.surgical.reach.reach_env_cfg import ReachEnvCfg # task: Reach

# initialize warp
wp.init()


class ReachSmState:
    """States for the reach state machine (rest position or action/reach the target)."""

    REST = wp.constant(0)
    REACH = wp.constant(1)


class ReachSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.5)
    REACH = wp.constant(1.0)


@wp.kernel 
def infer_state_machine(
    dt: wp.array(dtype=float), # time-step for each env
    sm_state: wp.array(dtype=int), # current state
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    des_final_pose: wp.array(dtype=wp.transform), # target pose 
    des_ee_pose: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid() 
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == ReachSmState.REST:
        des_ee_pose[tid] = ee_pose[tid] # if REST: des pose = current pose (stay still, stay in the REST position)
        # wait for a while
        if sm_wait_time[tid] >= ReachSmWaitTime.REST: # when REST time passes, next state 
            # move to next state and reset wait time
            sm_state[tid] = ReachSmState.REACH
            sm_wait_time[tid] = 0.0
    elif state == ReachSmState.REACH:
        des_ee_pose[tid] = des_final_pose[tid] # move to the desired pose
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= ReachSmWaitTime.REACH:
            # move to next state and reset wait time
            sm_state[tid] = ReachSmState.REACH
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class ReachSm:
    """A simple state machine in a robot's task space for a reach task.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector, and outputs the desired state of the robot's end-effector.
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

    def compute(self, ee_pose: torch.Tensor, des_final_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_final_pose = des_final_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        des_final_pose_wp = wp.from_torch(des_final_pose.contiguous(), wp.transform)

        # run state machine
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
        "Isaac-Reach-PSM-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Reach-PSM-IK-Abs-v0", cfg=env_cfg)
    
    # print action info
    print("Action space: ", env.unwrapped.action_space)
    print("Action space shape: ", env.unwrapped.action_space.shape)
    print("Num envs: ", env.unwrapped.num_envs)
    # reset environment at start
    env.reset()

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
            # observations
            robot: RigidObject = env.unwrapped.scene["robot"]
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_position_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], tcp_rest_position
            )
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- target end-effector frame
            desired_pose = env.unwrapped.command_manager.get_command("ee_pose")

            # advance state machine
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
