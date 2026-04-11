# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL.
   IMAGE-BASED version: receives raw RGB frames from the real robot camera via UDP,
   builds a stacked-frame observation, and sends actions back via UDP.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (image-based).")
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
import cv2
import time
import socket
import struct
import numpy as np
import gymnasium as gym
import robotic.surgery.tasks  # noqa: F401
import torch
from collections import deque
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
    # For image-based RL, we also:
    #   - permute image tensors from (B, H, W, C) → (B, C, H, W) and normalize to [0, 1]
    #   - inject a "dummy_state" key of shape (B, 0) required by some RSL-RL actor configs

    def ensure_tensordict(data):
        """Convert any observation format to TensorDict with 'policy' key.
        Handles image tensors: permutes (B,H,W,C) → (B,C,H,W) and normalizes to [0,1].
        """
        obs = data[0] if isinstance(data, tuple) else data
        if isinstance(obs, TensorDict):
            td = obs.to(agent_cfg.device)
        elif isinstance(obs, dict):
            td = TensorDict(obs, batch_size=env.num_envs).to(agent_cfg.device)
        else:
            td = TensorDict({"policy": obs.to(agent_cfg.device)}, batch_size=env.num_envs)

        # normalize image tensors: (B, H, W, C) → (B, C, H, W), [0,255] → [0,1]
        for key in list(td.keys()):
            if len(td[key].shape) == 4 and td[key].shape[-1] in [3, 4]:
                td[key] = td[key].permute(0, 3, 1, 2).float() / 255.0

        # inject dummy_state if not present (required by some RSL-RL actor configs)
        if "dummy_state" not in td.keys():
            td["dummy_state"] = torch.zeros((env.num_envs, 0), device=agent_cfg.device)

        return td

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
    PORT_RECEIVE_OBS   = 5006           # port to receive raw RGB frames from robot camera
    UDP_IP_ROBOT       = "10.41.53.28"  # IP of the real robot arm computer
    PORT_ROBOT         = 5005           # port to send actions to the real robot

    # --- Image dimensions ---
    # Received frame: raw RGB image from the real camera, resized to (84, 84, 3).
    # Policy input:   stacked frames upsampled to (168, 168), N_STACK frames deep.
    IMG_H, IMG_W   = 84, 84
    IMG_CHANNELS   = 3
    IMG_SIZE       = IMG_H * IMG_W * IMG_CHANNELS  # 21168 bytes per frame
    POLICY_H, POLICY_W = 168, 168                  # spatial resolution fed to the policy
    N_STACK        = 4                              # number of consecutive frames stacked

    # --- Action filtering (EMA: alpha * raw + (1-alpha) * prev_filtered) ---
    USE_ACTION_FILTER   = False 
    action_filter_alpha = 0.1  
    filtered_actions    = None

    # --- Observation saving (saves each stacked frame to disk for debugging) ---
    SAVE_OBS = False
    obs_save_dir = os.path.join(log_dir, "obs_images")
    if SAVE_OBS:
        os.makedirs(obs_save_dir, exist_ok=True)
        print(f"[INFO] Saving stacked observations to: {obs_save_dir}")

    # --- Manual reset pose (robot is driven here before inference starts) ---
    # Joint positions (rad / m) for the lift_liver task starting pose.
    MANUAL_RESET_POS = torch.tensor(
        [[0.91613, -0.34843, 0.08892, 0.97344, 0.01925, 0.08353]],
        device=agent_cfg.device
    )

    ### (6) VARIABLE INITIALIZATION ###

    obs              = env.get_observations()
    timestep         = 0
    initial_pos      = None  # set once on the first received frame
    previous_actions = torch.zeros((1, 6), device=agent_cfg.device, dtype=torch.float32)
    last_step_time   = time.perf_counter()
    last_final_pos   = None  # last joint positions before a reset (for final summary)

    # rolling buffer holding the last N_STACK frames (oldest → newest)
    frame_buffer = deque(maxlen=N_STACK)

    joint_names = ["Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"]

    ### (7) UDP SOCKETS SETUP ###
    # sock_recv: listens for raw RGB frame packets (IMG_SIZE bytes = 84*84*3 uint8).
    # sock_send: sends 6-float action packets to the real robot arm (24 bytes).

    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind((UDP_IP_RECEIVE_OBS, PORT_RECEIVE_OBS))
    sock_recv.settimeout(0.1)  # skip iteration if no packet arrives within 100 ms

    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    ### (8) MANUAL ROBOT RESET ###
    # Drive the simulated robot to MANUAL_RESET_POS by directly writing joint
    # positions and velocities into the simulator.
    # This ensures a known, consistent starting pose regardless of the previous episode.

    zero_vel = torch.zeros_like(MANUAL_RESET_POS)
    robot_asset.set_joint_position_target(MANUAL_RESET_POS, joint_ids=slice(0, 6))
    robot_asset.write_joint_state_to_sim(MANUAL_RESET_POS, zero_vel, joint_ids=slice(0, 6))
    robot_asset.write_data_to_sim()
    simulation_app.update()
    print(f"\n[INFO] Robot reset to manual pose: {MANUAL_RESET_POS[0].cpu().numpy()}")

    print("\n" + "="*80)
    print(f"[READY] Listening for RGB frames on port {PORT_RECEIVE_OBS} (expected {IMG_SIZE} bytes/frame) ...")
    print("="*80 + "\n")

    ###### MAIN SIMULATION LOOP - INFERENCE ######
    while simulation_app.is_running():
        with torch.inference_mode():
            print("\n" + "="*80)
            print(f"STEP {timestep:04d}")
            print("="*80)

            ### (9) RECEIVE RGB FRAME VIA UDP ###
            # Wait for a packet of exactly IMG_SIZE bytes = 84*84*3 uint8 (RGB, 0-255).
            # If no packet arrives within the timeout, skip this iteration.
            # If the buffer is empty (first frame), pre-fill it with N_STACK copies
            # of the first frame to avoid a cold-start with zeros.

            try:
                data_obs, _ = sock_recv.recvfrom(IMG_SIZE + 1024)
                if len(data_obs) == IMG_SIZE:
                    frame = np.frombuffer(data_obs, dtype=np.uint8).reshape(IMG_H, IMG_W, IMG_CHANNELS).copy()
                else:
                    print(f"[WARNING] Unexpected packet size: {len(data_obs)} bytes (expected {IMG_SIZE}). Skipping.")
                    continue
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[ERROR] UDP receive error: {e}")
                continue

            # pre-fill frame buffer on the very first frame
            if len(frame_buffer) == 0:
                for _ in range(N_STACK):
                    frame_buffer.append(frame)
            else:
                frame_buffer.append(frame)

            # print initial state once (on the very first frame received)
            pos_before = np.zeros(6)
            if initial_pos is None:
                initial_pos = pos_before.copy()
                print("\n" + "="*80)
                print("INITIAL STATE (first frame received from robot)")
                print("="*80)
                header = f"{'JOINT':<15} | {'POSITION':>12}"
                print(header)
                print("-" * len(header))
                for i in range(6):
                    print(f"{joint_names[i]:<15} | {initial_pos[i]:>12.5f}")
                print("="*80 + "\n")

            ### (10) BUILD STACKED OBSERVATION FROM FRAME BUFFER ###
            # Stack N_STACK frames along the channel axis: (H, W, C*N_STACK).
            # Normalize to [0,1], convert to tensor (1, C*N_STACK, H, W),
            # then upsample to (1, C*N_STACK, POLICY_H, POLICY_W).

            stacked       = np.concatenate(list(frame_buffer), axis=-1)   # (84, 84, 12)
            stacked_norm  = stacked.astype(np.float32) / 255.0
            stacked_tensor = (
                torch.tensor(stacked_norm, device=base_env.device)
                .permute(2, 0, 1)          # (12, 84, 84)
                .unsqueeze(0)              # (1, 12, 84, 84)
            )
            stacked_tensor = torch.nn.functional.interpolate(
                stacked_tensor, size=(POLICY_H, POLICY_W), mode='bilinear', align_corners=False
            )  # (1, 12, 168, 168)

            ### (11) OPTIONAL: SAVE STACKED FRAMES TO DISK ###
            # Saves each of the N_STACK frames at policy resolution as PNG files.
            # Useful for debugging what the policy actually sees.

            if SAVE_OBS:
                for fi in range(N_STACK):
                    ch_start = fi * IMG_CHANNELS
                    ch_end   = ch_start + IMG_CHANNELS
                    frame_vis = stacked_tensor[0, ch_start:ch_end, :, :]   # (3, 168, 168)
                    frame_vis = (frame_vis.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    fname = os.path.join(obs_save_dir, f"step{timestep:04d}_frame{fi}.png")
                    cv2.imwrite(fname, cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))

            ### (12) GET ACTION FROM POLICY ###
            # Wrap the stacked tensor in a TensorDict with the "policy" key as expected by RSL-RL.

            obs_before = ensure_tensordict({"policy": stacked_tensor})
            actions    = policy(obs_before)
            raw_policy = actions[0].cpu().numpy()

            ### (13) APPLY ACTION FILTER (EMA) ###
            # filtered = alpha * raw + (1 - alpha) * previous_filtered
            # Higher alpha = more responsive; lower alpha = smoother commands to the robot.

            if USE_ACTION_FILTER:
                if filtered_actions is None:
                    filtered_actions = actions.clone()
                else:
                    filtered_actions = action_filter_alpha * actions + (1.0 - action_filter_alpha) * filtered_actions
                actions_to_use = filtered_actions
            else:
                actions_to_use = actions

            ### (14) STEP SIMULATION + SEND ACTION TO REAL ROBOT ###
            # (a) Update previous_actions buffer (not used in obs for image policy,
            #     kept for symmetry with state-based scripts).
            # (b) Step the Isaac Sim environment with the chosen action.
            # (c) Send the 6 raw action floats to the real robot arm via UDP (24 bytes).

            previous_actions = actions_to_use.clone()
            obs, rewards, dones, extras = env.step(actions_to_use)

            target_raw = actions_to_use[0].cpu().numpy()[:6]
            data_send  = struct.pack('6f', *target_raw)
            sock_send.sendto(data_send, (UDP_IP_ROBOT, PORT_ROBOT))

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

            ### (17) PRINT OBSERVATION SUMMARY AND ACTION TABLE ###

            print(f"OBSERVATION: stacked image tensor shape={tuple(stacked_tensor.shape)} "
                  f"(1, {IMG_CHANNELS*N_STACK}, {POLICY_H}, {POLICY_W})")

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

            # save the last "good" final positions (avoids printing reset-state in summary)
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
    simulation_app.close()# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL.
   IMAGE-BASED version: receives raw RGB frames from the real robot camera via UDP,
   builds a stacked-frame observation, and sends actions back via UDP.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (image-based).")
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
import cv2
import time
import socket
import struct
import numpy as np
import gymnasium as gym
import robotic.surgery.tasks  # noqa: F401
import torch
from collections import deque
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
    # For image-based RL, we also:
    #   - permute image tensors from (B, H, W, C) → (B, C, H, W) and normalize to [0, 1]
    #   - inject a "dummy_state" key of shape (B, 0) required by some RSL-RL actor configs

    def ensure_tensordict(data):
        """Convert any observation format to TensorDict with 'policy' key.
        Handles image tensors: permutes (B,H,W,C) → (B,C,H,W) and normalizes to [0,1].
        """
        obs = data[0] if isinstance(data, tuple) else data
        if isinstance(obs, TensorDict):
            td = obs.to(agent_cfg.device)
        elif isinstance(obs, dict):
            td = TensorDict(obs, batch_size=env.num_envs).to(agent_cfg.device)
        else:
            td = TensorDict({"policy": obs.to(agent_cfg.device)}, batch_size=env.num_envs)

        # normalize image tensors: (B, H, W, C) → (B, C, H, W), [0,255] → [0,1]
        for key in list(td.keys()):
            if len(td[key].shape) == 4 and td[key].shape[-1] in [3, 4]:
                td[key] = td[key].permute(0, 3, 1, 2).float() / 255.0

        # inject dummy_state if not present (required by some RSL-RL actor configs)
        if "dummy_state" not in td.keys():
            td["dummy_state"] = torch.zeros((env.num_envs, 0), device=agent_cfg.device)

        return td

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
    PORT_RECEIVE_OBS   = 5006           # port to receive raw RGB frames from robot camera
    UDP_IP_ROBOT       = "10.41.53.28"  # IP of the real robot arm computer
    PORT_ROBOT         = 5005           # port to send actions to the real robot

    # --- Image dimensions ---
    # Received frame: raw RGB image from the real camera, resized to (84, 84, 3).
    # Policy input:   stacked frames upsampled to (168, 168), N_STACK frames deep.
    IMG_H, IMG_W   = 84, 84
    IMG_CHANNELS   = 3
    IMG_SIZE       = IMG_H * IMG_W * IMG_CHANNELS  # 21168 bytes per frame
    POLICY_H, POLICY_W = 168, 168                  # spatial resolution fed to the policy
    N_STACK        = 4                              # number of consecutive frames stacked

    # --- Action filtering (EMA: alpha * raw + (1-alpha) * prev_filtered) ---
    USE_ACTION_FILTER   = True
    action_filter_alpha = 0.3  # higher = more responsive, lower = smoother
    filtered_actions    = None

    # --- Observation saving (saves each stacked frame to disk for debugging) ---
    SAVE_OBS = False
    obs_save_dir = os.path.join(log_dir, "obs_images")
    if SAVE_OBS:
        os.makedirs(obs_save_dir, exist_ok=True)
        print(f"[INFO] Saving stacked observations to: {obs_save_dir}")

    # --- Manual reset pose (robot is driven here before inference starts) ---
    # Joint positions (rad / m) for the lift_liver task starting pose.
    MANUAL_RESET_POS = torch.tensor(
        [[0.91613, -0.34843, 0.08892, 0.97344, 0.01925, 0.08353]],
        device=agent_cfg.device
    )

    ### (6) VARIABLE INITIALIZATION ###

    obs              = env.get_observations()
    timestep         = 0
    initial_pos      = None  # set once on the first received frame
    previous_actions = torch.zeros((1, 6), device=agent_cfg.device, dtype=torch.float32)
    last_step_time   = time.perf_counter()
    last_final_pos   = None  # last joint positions before a reset (for final summary)

    # rolling buffer holding the last N_STACK frames (oldest → newest)
    frame_buffer = deque(maxlen=N_STACK)

    joint_names = ["Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"]

    ### (7) UDP SOCKETS SETUP ###
    # sock_recv: listens for raw RGB frame packets (IMG_SIZE bytes = 84*84*3 uint8).
    # sock_send: sends 6-float action packets to the real robot arm (24 bytes).

    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind((UDP_IP_RECEIVE_OBS, PORT_RECEIVE_OBS))
    sock_recv.settimeout(0.1)  # skip iteration if no packet arrives within 100 ms

    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    ### (8) MANUAL ROBOT RESET ###
    # Drive the simulated robot to MANUAL_RESET_POS by directly writing joint
    # positions and velocities into the simulator.
    # This ensures a known, consistent starting pose regardless of the previous episode.

    zero_vel = torch.zeros_like(MANUAL_RESET_POS)
    robot_asset.set_joint_position_target(MANUAL_RESET_POS, joint_ids=slice(0, 6))
    robot_asset.write_joint_state_to_sim(MANUAL_RESET_POS, zero_vel, joint_ids=slice(0, 6))
    robot_asset.write_data_to_sim()
    simulation_app.update()
    print(f"\n[INFO] Robot reset to manual pose: {MANUAL_RESET_POS[0].cpu().numpy()}")

    print("\n" + "="*80)
    print(f"[READY] Listening for RGB frames on port {PORT_RECEIVE_OBS} (expected {IMG_SIZE} bytes/frame) ...")
    print("="*80 + "\n")

    ###### MAIN SIMULATION LOOP - INFERENCE ######
    while simulation_app.is_running():
        with torch.inference_mode():
            print("\n" + "="*80)
            print(f"STEP {timestep:04d}")
            print("="*80)

            ### (9) RECEIVE RGB FRAME VIA UDP ###
            # Wait for a packet of exactly IMG_SIZE bytes = 84*84*3 uint8 (RGB, 0-255).
            # If no packet arrives within the timeout, skip this iteration.
            # If the buffer is empty (first frame), pre-fill it with N_STACK copies
            # of the first frame to avoid a cold-start with zeros.

            try:
                data_obs, _ = sock_recv.recvfrom(IMG_SIZE + 1024)
                if len(data_obs) == IMG_SIZE:
                    frame = np.frombuffer(data_obs, dtype=np.uint8).reshape(IMG_H, IMG_W, IMG_CHANNELS).copy()
                else:
                    print(f"[WARNING] Unexpected packet size: {len(data_obs)} bytes (expected {IMG_SIZE}). Skipping.")
                    continue
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[ERROR] UDP receive error: {e}")
                continue

            # pre-fill frame buffer on the very first frame
            if len(frame_buffer) == 0:
                for _ in range(N_STACK):
                    frame_buffer.append(frame)
            else:
                frame_buffer.append(frame)

            # print initial state once (on the very first frame received)
            pos_before = np.zeros(6)
            if initial_pos is None:
                initial_pos = pos_before.copy()
                print("\n" + "="*80)
                print("INITIAL STATE (first frame received from robot)")
                print("="*80)
                header = f"{'JOINT':<15} | {'POSITION':>12}"
                print(header)
                print("-" * len(header))
                for i in range(6):
                    print(f"{joint_names[i]:<15} | {initial_pos[i]:>12.5f}")
                print("="*80 + "\n")

            ### (10) BUILD STACKED OBSERVATION FROM FRAME BUFFER ###
            # Stack N_STACK frames along the channel axis: (H, W, C*N_STACK).
            # Normalize to [0,1], convert to tensor (1, C*N_STACK, H, W),
            # then upsample to (1, C*N_STACK, POLICY_H, POLICY_W).

            stacked       = np.concatenate(list(frame_buffer), axis=-1)   # (84, 84, 12)
            stacked_norm  = stacked.astype(np.float32) / 255.0
            stacked_tensor = (
                torch.tensor(stacked_norm, device=base_env.device)
                .permute(2, 0, 1)          # (12, 84, 84)
                .unsqueeze(0)              # (1, 12, 84, 84)
            )
            stacked_tensor = torch.nn.functional.interpolate(
                stacked_tensor, size=(POLICY_H, POLICY_W), mode='bilinear', align_corners=False
            )  # (1, 12, 168, 168)

            ### (11) OPTIONAL: SAVE STACKED FRAMES TO DISK ###
            # Saves each of the N_STACK frames at policy resolution as PNG files.
            # Useful for debugging what the policy actually sees.

            if SAVE_OBS:
                for fi in range(N_STACK):
                    ch_start = fi * IMG_CHANNELS
                    ch_end   = ch_start + IMG_CHANNELS
                    frame_vis = stacked_tensor[0, ch_start:ch_end, :, :]   # (3, 168, 168)
                    frame_vis = (frame_vis.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    fname = os.path.join(obs_save_dir, f"step{timestep:04d}_frame{fi}.png")
                    cv2.imwrite(fname, cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))

            ### (12) GET ACTION FROM POLICY ###
            # Wrap the stacked tensor in a TensorDict with the "policy" key as expected by RSL-RL.

            obs_before = ensure_tensordict({"policy": stacked_tensor})
            actions    = policy(obs_before)
            raw_policy = actions[0].cpu().numpy()

            ### (13) APPLY ACTION FILTER (EMA) ###
            # filtered = alpha * raw + (1 - alpha) * previous_filtered
            # Higher alpha = more responsive; lower alpha = smoother commands to the robot.

            if USE_ACTION_FILTER:
                if filtered_actions is None:
                    filtered_actions = actions.clone()
                else:
                    filtered_actions = action_filter_alpha * actions + (1.0 - action_filter_alpha) * filtered_actions
                actions_to_use = filtered_actions
            else:
                actions_to_use = actions

            ### (14) STEP SIMULATION + SEND ACTION TO REAL ROBOT ###
            # (a) Update previous_actions buffer (not used in obs for image policy,
            #     kept for symmetry with state-based scripts).
            # (b) Step the Isaac Sim environment with the chosen action.
            # (c) Send the 6 raw action floats to the real robot arm via UDP (24 bytes).

            previous_actions = actions_to_use.clone()
            obs, rewards, dones, extras = env.step(actions_to_use)

            target_raw = actions_to_use[0].cpu().numpy()[:6]
            data_send  = struct.pack('6f', *target_raw)
            sock_send.sendto(data_send, (UDP_IP_ROBOT, PORT_ROBOT))

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

            ### (17) PRINT OBSERVATION SUMMARY AND ACTION TABLE ###

            print(f"OBSERVATION: stacked image tensor shape={tuple(stacked_tensor.shape)} "
                  f"(1, {IMG_CHANNELS*N_STACK}, {POLICY_H}, {POLICY_W})")

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

            # save the last "good" final positions (avoids printing reset-state in summary)
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