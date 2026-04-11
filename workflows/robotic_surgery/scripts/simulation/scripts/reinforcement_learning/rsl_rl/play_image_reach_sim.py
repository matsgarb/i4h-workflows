# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL.
   IMAGE-BASED version for the REACH task.
   Success criteria: distance error < 3 mm AND orientation error < 0.3 rad.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (image-based, reach task).")
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
import numpy as np
import gymnasium as gym
import robotic.surgery.tasks  # noqa: F401
import torch
import pandas as pd
from tensordict import TensorDict
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import liver_target_pose_world
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG


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
    # Grab handles to the robot, EE frame, and camera sensor from the unwrapped env scene.

    base_env = getattr(env, "unwrapped", env)
    robot_asset = base_env.scene["robot"]
    ee_frame    = base_env.scene["ee_frame"]

    ### (5) USER-TUNABLE FLAGS AND VARIABLES ###

    # liver stabilization warm-up: number of steps with zero actions before inference starts
    STABILIZATION_STEPS = 20

    # action filtering: disabled for reach task (policy outputs are used directly)
    USE_ACTION_FILTER  = False
    action_filter_alpha = 0.1
    filtered_actions   = None

    # observation frame saving (saves the image observation fed to the policy at each step)
    SAVE_OBS_FRAMES = False
    obs_frames_dir  = None
    if SAVE_OBS_FRAMES:
        obs_frames_dir = os.path.join(log_dir, "obs_frames")
        os.makedirs(obs_frames_dir, exist_ok=True)
        print(f"[INFO] Observation frames will be saved to: {obs_frames_dir}")

    # data saving: saves per-timestep joint data + errors to Excel at episode end
    SAVE_DATA          = False
    timestep_data_list = []
    if SAVE_DATA:
        print(f"[INFO] Data saving enabled. Data will be saved as Excel file at episode end.")

    # success thresholds for the reach task
    DIST_THRESHOLD   = 0.003  # 3 mm
    ORIENT_THRESHOLD = 0.3    # rad

    ### (6) VARIABLE INITIALIZATION ###

    obs      = env.get_observations()
    timestep = 0

    joint_names     = ["Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"]
    joint_names_obs = ["Yaw", "Pitch", "Insert", "Wrist_Roll", "Wrist_Pitch", "Wrist_Yaw", "Gripper1", "Gripper2"]

    observation_data = []  # per-step obs data for optional Excel export

    # "last good" values: updated every non-terminal step so the final summary
    # always shows pre-reset values rather than the reset state
    last_dist_error   = None
    last_orient_error = None

    ### (7) INITIAL STATE PRINT ###
    # Print joint positions, velocities, and EE pose (robot RF) before any action.

    print("\n" + "="*80)
    print("INITIAL STATE (STEP 0 - RESET)")
    print("="*80)
    initial_pos = robot_asset.data.joint_pos[0][:6].cpu().numpy()
    initial_vel = robot_asset.data.joint_vel[0][:6].cpu().numpy()
    header = f"{'JOINT':<15} | {'POSITION':>12} | {'VELOCITY':>12}"
    print(header)
    print("-" * len(header))
    for i in range(6):
        print(f"{joint_names[i]:<15} | {initial_pos[i]:>12.5f} | {initial_vel[i]:>12.5f}")

    robot_root_pos  = robot_asset.data.root_state_w[:, :3]
    robot_root_quat = robot_asset.data.root_state_w[:, 3:7]
    ee_pos_w  = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w)
    e_pos_b  = ee_pos_b[0].cpu().numpy()
    e_quat_b = ee_quat_b[0].cpu().numpy()
    print(f"\nEE frame (robot RF): X={e_pos_b[0]:.5f}, Y={e_pos_b[1]:.5f}, Z={e_pos_b[2]:.5f}, "
          f"w={e_quat_b[0]:.5f}, x={e_quat_b[1]:.5f}, y={e_quat_b[2]:.5f}, z={e_quat_b[3]:.5f}")
    print("="*80 + "\n")

    ### (8) LIVER STABILIZATION WARM-UP ###
    # Run STABILIZATION_STEPS steps with zero actions so the deformable liver
    # settles into its rest pose before the policy starts acting.

    print(f"\n[INFO] Running {STABILIZATION_STEPS} stabilization steps (zero actions) to let liver settle...")
    action_dim   = base_env.action_manager.total_action_dim
    zero_actions = torch.zeros(env.num_envs, action_dim, device=agent_cfg.device)
    with torch.inference_mode():
        for stab_step in range(STABILIZATION_STEPS):
            obs, _, _, _ = env.step(zero_actions)
            if stab_step % 5 == 0:
                print(f"  [STABILIZE] step {stab_step+1}/{STABILIZATION_STEPS}")
    print("[INFO] Stabilization complete. Starting policy inference.\n")

    ###### MAIN SIMULATION LOOP - INFERENCE ######
    while simulation_app.is_running():
        with torch.inference_mode():
            print("\n" + "="*80)
            print(f"STEP {timestep:04d}")
            print("="*80)

            ### (9) CAPTURE STATE BEFORE ACTION ###
            # Save the current observation and joint positions before stepping
            # so we can compute real_move = pos_after - pos_before.

            obs_before = obs
            pos_before = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            true_pos   = robot_asset.data.joint_pos[0].cpu().numpy()
            true_vel   = robot_asset.data.joint_vel[0].cpu().numpy()

            ### (10) OPTIONAL: SAVE OBSERVATION FRAME TO DISK ###
            # Saves the exact camera_rgb_observation used during training
            # (resized to 168x168, normalized to [0,1]) as a PNG file.

            if SAVE_OBS_FRAMES:
                try:
                    camera = base_env.scene.sensors["camera"]
                    rgb = camera.data.output["rgb"][..., :3].float() / 255.0   # [B, H, W, 3]
                    rgb = rgb.permute(0, 3, 1, 2)                               # [B, 3, H, W]
                    if rgb.shape[-2:] != (168, 168):
                        rgb = torch.nn.functional.interpolate(
                            rgb, size=(168, 168), mode='bilinear', align_corners=False
                        )
                    img_hwc  = rgb[0].permute(1, 2, 0).cpu().numpy()           # [168, 168, 3]
                    img_uint8 = (img_hwc * 255.0).astype(np.uint8)
                    frame_path = os.path.join(obs_frames_dir, f"obs_frame_step_{timestep:04d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
                    if timestep % 20 == 0:
                        print(f"[OBS FRAME] Saved: {frame_path}")
                except Exception as e:
                    print(f"[OBS FRAME WARNING] {type(e).__name__}: {e}")

            ### (11) GET ACTION FROM POLICY ###

            actions    = policy(obs_before)
            raw_policy = actions[0].cpu().numpy()

            ### (12) APPLY FILTER OR USE RAW ACTIONS ###
            # For the reach task, filtering is disabled (USE_ACTION_FILTER = False).
            # EMA formula if enabled: filtered = alpha * raw + (1 - alpha) * prev_filtered

            if USE_ACTION_FILTER:
                if filtered_actions is None:
                    filtered_actions = actions.clone()
                else:
                    filtered_actions = action_filter_alpha * actions + (1.0 - action_filter_alpha) * filtered_actions
                actions_to_use = filtered_actions
            else:
                actions_to_use = actions

            ### (13) EXECUTE ACTION IN ENVIRONMENT (SIMULATION STEP) ###

            obs, rewards, dones, extras = env.step(actions_to_use)

            ### (14) EXTRACT FLAT OBSERVATION ARRAY FOR LOGGING ###
            # obs structure: pos(8) + vel(8) + last_actions(6) = 22 components.
            # For image-based policy the obs TensorDict also contains the image,
            # but we extract the "policy" key which is the flattened state used for logging.

            obs_data = obs_before.get("policy") if isinstance(obs_before, dict) or hasattr(obs_before, 'get') else obs_before
            if torch.is_tensor(obs_data):
                obs_flat = obs_data[0].cpu().numpy() if obs_data.dim() > 1 else obs_data.cpu().numpy()
            else:
                obs_flat = np.array(obs_data)

            noisy_pos = obs_flat[:8]
            noisy_vel = obs_flat[8:16]
            obs_row   = {'step': timestep}
            for j in range(8):
                name = joint_names_obs[j] if j < len(joint_names_obs) else f"joint_{j}"
                obs_row[f'true_pos_{name}']  = true_pos[j]
                obs_row[f'noisy_pos_{name}'] = noisy_pos[j]
                obs_row[f'pos_error_{name}'] = true_pos[j] - noisy_pos[j]
                obs_row[f'true_vel_{name}']  = true_vel[j]
                obs_row[f'noisy_vel_{name}'] = noisy_vel[j]
                obs_row[f'vel_error_{name}'] = true_vel[j] - noisy_vel[j]
            observation_data.append(obs_row)

            ### (15) DONE FLAG CHECK ###

            done_flag = False
            if torch.is_tensor(dones):
                done_flag = bool(torch.any(dones))
            elif isinstance(dones, (np.ndarray, list, tuple)):
                done_flag = bool(np.any(dones))
            elif isinstance(dones, bool):
                done_flag = dones

            ### (16) ACTION LOGGING ###

            cmd_term          = base_env.action_manager.get_term("arm_action")
            processed_actions = cmd_term.processed_actions[0].cpu().numpy()
            pos_after         = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            real_move         = pos_after - pos_before

            ### (17) POSE EXTRACTION, ERROR COMPUTATION, AND PRINTING (only if not done) ###
            # Skip on the final step when done=True: the env has already reset the robot,
            # so poses would reflect the reset state rather than the terminal state.
            # On that step we use the last stored values instead (see section 20).

            if not done_flag:
                robot_root_pos  = robot_asset.data.root_state_w[:, :3]
                robot_root_quat = robot_asset.data.root_state_w[:, 3:7]

                # --- target liver pose (robot RF) ---
                target_pose_w  = liver_target_pose_world(base_env)
                target_pos_w   = target_pose_w[:, :3]
                target_quat_w  = target_pose_w[:, 3:7]
                target_pos_b, target_quat_b = subtract_frame_transforms(
                    robot_root_pos, robot_root_quat, target_pos_w, target_quat_w
                )
                t_pos_b  = target_pos_b[0].cpu().numpy()
                t_quat_b = target_quat_b[0].cpu().numpy()

                # --- EE pose (robot RF) ---
                ee_pos_w  = ee_frame.data.target_pos_w[..., 0, :]
                ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w
                )
                e_pos_b  = ee_pos_b[0].cpu().numpy()
                e_quat_b = ee_quat_b[0].cpu().numpy()

                # --- distance and orientation errors ---
                ee_pos_b_t  = torch.tensor(e_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                ee_quat_b_t = torch.tensor(e_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                t_pos_b_t = torch.tensor(t_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                t_quat_b_t = torch.tensor(t_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)

                dist_error   = torch.norm(ee_pos_b_t - t_pos_b_t, dim=1)[0].cpu().item()
                orient_error = quat_error_magnitude(ee_quat_b_t, t_quat_b_t)[0].cpu().item()

                # store for final summary (overwritten every non-terminal step)
                last_dist_error   = dist_error
                last_orient_error = orient_error

                ### (19) PRINT STEP INFO ###

                print("EE frame (robot RF):    X={:.4f}, Y={:.4f}, Z={:.4f}, "
                      "w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                      e_pos_b[0], e_pos_b[1], e_pos_b[2],
                      e_quat_b[0], e_quat_b[1], e_quat_b[2], e_quat_b[3]))
                print("TARGET pose (robot RF): X={:.4f}, Y={:.4f}, Z={:.4f}, "
                      "w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                      t_pos_b[0], t_pos_b[1], t_pos_b[2],
                      t_quat_b[0], t_quat_b[1], t_quat_b[2], t_quat_b[3]))
                print(f"DISTANCE ERROR: {dist_error:.6f} m | ORIENTATION ERROR: {orient_error:.6f} rad")

                # in-loop success notification (does NOT stop the episode — termination
                # is handled by the environment's termination manager)
                if dist_error < DIST_THRESHOLD and orient_error < ORIENT_THRESHOLD:
                    print("\n" + "=" * 80)
                    print("TARGET REACHED!")
                    print("=" * 80)
                    print(f"DISTANCE ERROR:    {dist_error:.6f} m    (threshold: < {DIST_THRESHOLD})")
                    print(f"ORIENTATION ERROR: {orient_error:.6f} rad (threshold: < {ORIENT_THRESHOLD})")
                    print("=" * 80 + "\n")

                # observations printout
                obs_size      = len(obs_flat)
                joint_pos_obs = obs_flat[:8]
                joint_vel_obs = obs_flat[8:16]
                actions_obs   = obs_flat[16:22]

                print(f"\nOBSERVATIONS (size={obs_size}):")
                print(f"  joint_pos_rel (8):  {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_pos_obs))}")
                print(f"  joint_vel_rel (8):  {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_vel_obs))}")
                print(f"  last_action (6):    {' | '.join(f'{i:8.5f}' for i in actions_obs)}")

                header = f"{'JOINT':<10} | {'POSITION':>10} | {'RAW POLICY':>10} | {'PROCESSED':>10} | {'REAL MOVE':>10}"
                print(header)
                print("-" * len(header))
                for i in range(6):
                    print(f"{joint_names[i]:<10} | "
                          f"{pos_after[i]:>10.5f} | "
                          f"{raw_policy[i]:>10.5f} | "
                          f"{processed_actions[i]:>10.5f} | "
                          f"{real_move[i]:>10.5f}")
                print("=" * len(header))
                print("="*80 + "\n")

                ### (20) SAVE TIMESTEP DATA ###
                if SAVE_DATA:
                    row_data = {'timestep': timestep}
                    for j in range(6):
                        row_data[f'position_j{j}']         = pos_after[j]
                        row_data[f'raw_policy_j{j}']       = raw_policy[j]
                        row_data[f'processed_actions_j{j}'] = processed_actions[j]
                        row_data[f'real_movement_j{j}']    = real_move[j]
                    row_data['distance_error']    = dist_error
                    row_data['orientation_error'] = orient_error
                    timestep_data_list.append(row_data)

        ### (21) SAVE FINAL METRICS BEFORE TERMINATION CHECK ###
        # Snapshot of errors from the last non-terminal step.
        # If done_flag is True these will already have been set correctly above.

        final_dist_error   = last_dist_error
        final_orient_error = last_orient_error
        final_timestep     = timestep
        final_episode_reason = "UNKNOWN"

        ### (22) EPISODE TERMINATION CHECK AND LOGGING ###

        done_flag = False
        if torch.is_tensor(dones):
            done_flag = bool(torch.any(dones))
        elif isinstance(dones, (np.ndarray, list, tuple)):
            done_flag = bool(np.any(dones))
        elif isinstance(dones, bool):
            done_flag = dones

        if done_flag:
            # determine termination reason from extras log
            if isinstance(extras, dict) and "log" in extras:
                log = extras["log"]
                time_out_value = log.get("Episode_Termination/time_out", 0)
                success_value  = log.get("Episode_Termination/success", 0)
                if hasattr(time_out_value, 'item'): time_out_value = time_out_value.item()
                if hasattr(success_value,  'item'): success_value  = success_value.item()
                if success_value   == 1: final_episode_reason = "ACHIEVED SUCCESS"
                elif time_out_value == 1: final_episode_reason = "TIME OUT"

            print(f"[INFO] Episode terminated: {final_episode_reason}")
            pos_str = " ".join([f"{p:.5f}".replace(".", ",") for p in initial_pos])
            print(f"Initial position of the joints: {pos_str}")

            # save Excel data if enabled
            if SAVE_DATA and timestep_data_list:
                df = pd.DataFrame(timestep_data_list)
                excel_filename = os.path.join(log_dir, f"episode_{timestep}_timesteps.xlsx")
                df.to_excel(excel_filename, index=False)
                print(f"[INFO] Timestep data saved to: {excel_filename}")

            if observation_data:
                df_obs   = pd.DataFrame(observation_data)
                obs_path = os.path.join(log_dir, f"observations_{timestep}_data.xlsx")
                # df_obs.to_excel(obs_path, index=False)  # uncomment to enable

            # stop robot by zeroing velocity target
            try:
                zero_vel = torch.zeros((1, 6), device=robot_asset.device)
                robot_asset.set_joint_velocity_target(zero_vel, joint_ids=slice(0, 6))
                robot_asset.write_data_to_sim()
                print("[INFO] Stopping robot (velocity = 0).")
            except Exception as e:
                print(f"[WARN] Unable to stop robot: {e}")

            print("[INFO] Stopping simulation.")
            timestep_data_list = []
            observation_data   = []
            break

        else:
            timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            break

    ### (23) FINAL SUMMARY (using pre-termination values) ###

    try:
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Episode terminated: {final_episode_reason}")
        print(f"Total steps to completion: {final_timestep}")
        if final_dist_error is not None:
            print(f"DISTANCE ERROR:    {final_dist_error*1000:.4f} mm (threshold: {DIST_THRESHOLD*1000:.1f} mm)")
            print(f"ORIENTATION ERROR: {final_orient_error*180/np.pi:.4f}° "
                  f"({final_orient_error:.6f} rad, threshold: {ORIENT_THRESHOLD} rad)")
        print("=" * 80)
    except Exception as e:
        print(f"[WARN] Unable to compute final summary: {e}")

    ### (24) CLOSE ENVIRONMENT ###
    env.close()


### (25) ENTRY POINT ###
if __name__ == "__main__":
    main()
    simulation_app.close()