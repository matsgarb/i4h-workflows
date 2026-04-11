# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL.
   Receives real robot state via UDP and sends actions back via UDP.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------
# All imports AFTER Isaac Sim is launched (Isaac Sim must init first)
# -----------------------------------------------------------------------
import os
import time
import socket
import struct
import numpy as np
import gymnasium as gym
import robotic.surgery.tasks  # noqa: F401
import torch
from tensordict import TensorDict
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner


def main():

    ### (1) ENV AND AGENT CONFIGURATION ###
    # Parse env config and agent config, locate the checkpoint, create the gym env.

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    ### (2) TENSORDICT COMPATIBILITY PATCH ###
    # RSL-RL's OnPolicyRunner expects observations as a TensorDict with a "policy" key.
    # This patch wraps get_observations and step so their outputs are always in that format,
    # without modifying the original environment code.

    def ensure_tensordict(data):
        """Convert any observation format to TensorDict with 'policy' key."""
        obs = data[0] if isinstance(data, tuple) else data
        if isinstance(obs, TensorDict): return obs
        if isinstance(obs, dict): return TensorDict(obs, batch_size=env.num_envs)
        return TensorDict({"policy": obs.to(agent_cfg.device)}, batch_size=env.num_envs)

    original_get_obs = env.get_observations
    original_step = env.step
    env.get_observations = lambda: ensure_tensordict(original_get_obs())

    def patched_step(actions):
        obs, rewards, dones, extras = original_step(actions)
        return ensure_tensordict(obs), rewards, dones, extras

    env.step = patched_step

    ### (3) POLICY LOADING AND EXPORTING ###
    # Load the PPO runner from checkpoint, extract the inference policy,
    # and export it to JIT/ONNX for deployment.

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO] Model checkpoint loaded from: {resume_path}")

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    obs_normalizer = getattr(ppo_runner.alg, "obs_normalizer", None)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    try:
        export_policy_as_jit(ppo_runner.alg.policy, obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(ppo_runner.alg.policy, normalizer=obs_normalizer, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Policy exported successfully to: {export_model_dir}")
    except Exception as e:
        print(f"[WARNING] Error during export (playback will continue): {e}")

    ### (4) SCENE ASSET REFERENCES ###
    # Grab handles to the robot and EE frame from the unwrapped env scene.

    base_env = getattr(env, "unwrapped", env)
    robot_asset = base_env.scene["robot"]
    ee_frame = base_env.scene["ee_frame"]

    ### (5) USER-TUNABLE FLAGS AND VARIABLES ###
    # --- UDP communication ---
    UDP_IP_RECEIVE_OBS = "0.0.0.0"     # listen on all interfaces
    PORT_RECEIVE_OBS   = 5006           # port to receive robot state (pos + vel)
    UDP_IP_ROBOT       = "10.41.53.28"  # IP of the real robot arm computer
    PORT_ROBOT         = 5005           # port to send actions to the real robot

    # --- Action filtering (EMA: alpha * raw + (1-alpha) * prev_filtered) ---
    USE_ACTION_FILTER   = False
    action_filter_alpha = 0.1  # higher = more responsive, lower = smoother
    filtered_actions    = None

    # --- Manual reset pose (robot is driven here before inference starts) ---
    # Joint positions (rad / m) for the lift_liver task starting pose.
    MANUAL_RESET_POS = torch.tensor(
        [[0.8836, -0.3286, 0.1045, 0.9168, 0.3883, 0.4726]],
        device=agent_cfg.device
    )

    ### (6) VARIABLE INITIALIZATION ###
    timestep = 0
    initial_pos = None  # set on first UDP packet received
    previous_actions = torch.zeros((1, 6), device=agent_cfg.device, dtype=torch.float32)
    last_step_time = time.perf_counter()
    last_final_pos = None  

    joint_names = ["Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"]
    joint_names_obs = ["Yaw", "Pitch", "Insert", "Wrist_Roll", "Wrist_Pitch", "Wrist_Yaw", "Gripper1", "Gripper2"]

    ### (7) UDP SOCKETS SETUP ###
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind((UDP_IP_RECEIVE_OBS, PORT_RECEIVE_OBS))
    sock_recv.settimeout(0.1)  # skip iteration if no packet arrives within 100 ms
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    ### (8) MANUAL ROBOT RESET ###
    zero_vel = torch.zeros_like(MANUAL_RESET_POS)
    robot_asset.set_joint_position_target(MANUAL_RESET_POS, joint_ids=slice(0, 6))
    robot_asset.write_joint_state_to_sim(MANUAL_RESET_POS, zero_vel, joint_ids=slice(0, 6))
    robot_asset.write_data_to_sim()
    simulation_app.update()
    print(f"\n[INFO] Robot reset to manual pose: {MANUAL_RESET_POS[0].cpu().numpy()}")

    print("\n" + "="*80)
    print(f"[READY] Listening for robot state on port {PORT_RECEIVE_OBS} ...")
    print("="*80 + "\n")

    ###### MAIN SIMULATION LOOP - INFERENCE ######
    while simulation_app.is_running():
        with torch.inference_mode():
            print("\n" + "="*80)
            print(f"STEP {timestep:04d}")
            print("="*80)

            ### (9) RECEIVE ROBOT STATE VIA UDP ###
            # Wait for a 64-byte packet: 16 floats = [pos(8), vel(8)].
            try:
                data_obs, _ = sock_recv.recvfrom(1024)
                if len(data_obs) == 64:
                    unpacked_obs = struct.unpack('16f', data_obs)
                    ext_pos = np.array(unpacked_obs[:8])   
                    ext_vel = np.array(unpacked_obs[8:16]) 
                    ext_pos_tensor = torch.tensor(ext_pos, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                    ext_vel_tensor = torch.tensor(ext_vel, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                else:
                    print(f"[WARNING] Unexpected packet length: {len(data_obs)} bytes (expected 64). Skipping.")
                    continue
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[ERROR] UDP receive error: {e}")
                continue

            # Print initial joint state once (on the very first packet received)
            if initial_pos is None:
                initial_pos = ext_pos[:6].copy()
                print("\n" + "="*80)
                print("INITIAL STATE (first UDP packet from robot)")
                print("="*80)
                header = f"{'JOINT':<15} | {'POSITION':>12}"
                print(header)
                print("-" * len(header))
                for i in range(6):
                    print(f"{joint_names[i]:<15} | {initial_pos[i]:>12.5f}")
                print("="*80 + "\n")

            pos_before = ext_pos[:6]

            ### (10) BUILD OBSERVATION FROM REAL ROBOT STATE ###
            # Observation vector: [pos(8), vel(8), prev_actions(6)] = 22 components.
            obs_tensor = torch.cat([ext_pos_tensor, ext_vel_tensor, previous_actions], dim=1)
            obs_before = {"policy": obs_tensor}

            ### (11) GET ACTION FROM POLICY ###
            actions    = policy(obs_before)
            raw_policy = actions[0].cpu().numpy()

            ### (12) APPLY ACTION FILTER (EMA) ###
            if USE_ACTION_FILTER:
                if filtered_actions is None:
                    filtered_actions = actions.clone()
                else:
                    filtered_actions = action_filter_alpha * actions + (1.0 - action_filter_alpha) * filtered_actions
                actions_to_use = filtered_actions
            else:
                actions_to_use = actions

            ### (13) STEP SIMULATION + SEND ACTION TO REAL ROBOT ###
            previous_actions = actions_to_use.clone()
            obs, rewards, dones, extras = env.step(actions_to_use)

            target_raw = actions_to_use[0].cpu().numpy()[:6]
            data_send  = struct.pack('6f', *target_raw)
            sock_send.sendto(data_send, (UDP_IP_ROBOT, PORT_ROBOT))

            ### (14) EXTRACT FLAT OBSERVATION ARRAY ###
            # obs_before was built manually, extract obs_flat for display.
            # Structure: pos(8) + vel(8) + prev_actions(6) = 22

            obs_data = obs_before.get("policy") if isinstance(obs_before, dict) else obs_before
            obs_flat = obs_data[0].cpu().numpy() if torch.is_tensor(obs_data) and obs_data.dim() > 1 else np.array(obs_data)

            ### (15) CHECK DONE FLAG ###

            done_flag = False
            if torch.is_tensor(dones):
                done_flag = bool(torch.any(dones))
            elif isinstance(dones, (np.ndarray, list, tuple)):
                done_flag = bool(np.any(dones))
            elif isinstance(dones, bool):
                done_flag = dones

            ### (16) TIMING ###

            now = time.perf_counter()
            dt  = now - last_step_time
            last_step_time = now
            print(f"STEP DT: {dt*1000:.2f} ms  ({1.0/dt if dt > 0 else float('inf'):.1f} Hz)")

            ### (17) PRINT OBSERVATIONS AND ACTION TABLE ###

            obs_size      = len(obs_flat)
            joint_pos_obs = obs_flat[:8]
            joint_vel_obs = obs_flat[8:16]
            actions_obs   = obs_flat[16:22]

            print(f"\nOBSERVATIONS (size={obs_size}):")
            print(f"  joint_pos (8):    {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_pos_obs))}")
            print(f"  joint_vel (8):    {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_vel_obs))}")
            print(f"  last_action (6):  {' | '.join(f'{i:8.5f}' for i in actions_obs)}")

            cmd_term          = base_env.action_manager.get_term("arm_action")
            processed_actions = cmd_term.processed_actions[0].cpu().numpy()
            pos_after         = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            real_move         = pos_after - pos_before

            header = f"{'JOINT':<10} | {'POSITION':>10} | {'RAW POLICY':>10} | {'PROCESSED':>10} | {'REAL MOVE':>10}"
            if USE_ACTION_FILTER:
                header += f" | {'FILTERED':>10}"
            print(header)
            print("-" * len(header))
            for i in range(6):
                row = (f"{joint_names[i]:<10} | "
                       f"{pos_after[i]:>10.5f} | "
                       f"{raw_policy[i]:>10.5f} | "
                       f"{processed_actions[i]:>10.5f} | "
                       f"{real_move[i]:>10.5f}")
                if USE_ACTION_FILTER:
                    fv = filtered_actions[0, i].cpu().item() if filtered_actions is not None else float('nan')
                    row += f" | {fv:>10.5f}"
                print(row)
            print("=" * len(header))
            print("="*80 + "\n")

            if not done_flag:
                last_final_pos = pos_after.copy()

        ### (18) EPISODE TERMINATION CHECK AND FINAL SUMMARY ###
        # Re-evaluate done flag outside inference_mode for the termination branch.

        done_flag = False
        if torch.is_tensor(dones):
            done_flag = bool(torch.any(dones))
        elif isinstance(dones, (np.ndarray, list, tuple)):
            done_flag = bool(np.any(dones))
        elif isinstance(dones, bool):
            done_flag = dones

        if done_flag:
            # determine reason for termination from extras log
            episode_reason = "UNKNOWN"
            if isinstance(extras, dict) and "log" in extras:
                log = extras["log"]
                time_out_value = log.get("Episode_Termination/time_out", 0)
                success_value  = log.get("Episode_Termination/success", 0)
                if hasattr(time_out_value, 'item'): time_out_value = time_out_value.item()
                if hasattr(success_value,  'item'): success_value  = success_value.item()
                if success_value   == 1: episode_reason = "ACHIEVED SUCCESS"
                elif time_out_value == 1: episode_reason = "TIME OUT"

            print("\n" + "="*80)
            print(f"EPISODE TERMINATED — {episode_reason}")
            print("="*80)

            # print initial vs final joint positions side by side
            if initial_pos is not None:
                header = f"{'JOINT':<15} | {'INITIAL':>12} | {'FINAL':>12}"
                print(header)
                print("-" * len(header))
                for i in range(6):
                    final_val = last_final_pos[i] if last_final_pos is not None else float('nan')
                    print(f"{joint_names[i]:<15} | {initial_pos[i]:>12.5f} | {final_val:>12.5f}")
                print("="*80 + "\n")

            # stop the robot by zeroing velocity target
            try:
                zero_vel_stop = torch.zeros((1, 6), device=robot_asset.device)
                robot_asset.set_joint_velocity_target(zero_vel_stop, joint_ids=slice(0, 6))
                robot_asset.write_data_to_sim()
                print("[INFO] Robot stopped (velocity target = 0).")
            except Exception as e:
                print(f"[WARN] Unable to stop robot: {e}")

            print("[INFO] Stopping simulation.")
            break

        else:
            timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            break

    ### (19) CLOSE ENVIRONMENT ###
    env.close()


### (20) ENTRY POINT ###
if __name__ == "__main__":
    main()
    simulation_app.close()