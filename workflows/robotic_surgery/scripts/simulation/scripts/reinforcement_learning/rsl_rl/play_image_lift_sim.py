# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL.
   IMAGE-BASED version for the LIFT task.
   Success criteria: gallbladder visible pixels > GALLBLADDER_PIXEL_THRESHOLD.
   Action filtering enabled (EMA).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (image-based, lift task).")
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
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import (
    gallbladder_pixel_count,
    gallbladder_visibility_success,
    gallbladder_mask_tensor,
)

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
    # RSL-RL's OnPolicyRunner expects observations as a TensorDict with a "policy" key
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
    robot_asset  = base_env.scene["robot"]
    ee_frame     = base_env.scene["ee_frame"]
    camera_sensor = base_env.scene.sensors["camera"]

    ### (5) USER-TUNABLE FLAGS AND VARIABLES ###

    # liver stabilization warm-up: number of steps with zero actions before inference starts
    STABILIZATION_STEPS = 20


    # action filtering: enabled for lift task to smooth commands
    # EMA formula: filtered = alpha * raw + (1 - alpha) * prev_filtered
    USE_ACTION_FILTER   = True
    action_filter_alpha = 0.1   # higher = more responsive, lower = smoother
    filtered_actions    = None

    # observation frame saving (saves the image observation fed to the policy at each step)
    SAVE_OBS_FRAMES = False
    obs_frames_dir  = None
    if SAVE_OBS_FRAMES:
        obs_frames_dir = os.path.join(log_dir, "obs_frames")
        os.makedirs(obs_frames_dir, exist_ok=True)
        print(f"[INFO] Observation frames will be saved to: {obs_frames_dir}")

    # data saving: saves per-timestep joint data + gallbladder pixels to Excel at episode end
    SAVE_DATA          = False
    timestep_data_list = []
    if SAVE_DATA:
        print(f"[INFO] Data saving enabled. Data will be saved as Excel file at episode end.")

    # mask saving: saves binary gallbladder mask + final RGB frame at episode end
    SAVE_MASK = False
    if SAVE_MASK:
        print(f"[INFO] Mask saving enabled. Binary mask and final frame will be saved at episode end.")

    # gallbladder exposure success threshold (pixels)
    GALLBLADDER_PIXEL_THRESHOLD = 27000

    ### (6) VARIABLE INITIALIZATION ###

    obs      = env.get_observations()
    timestep = 0

    joint_names     = ["Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"]
    joint_names_obs = ["Yaw", "Pitch", "Insert", "Wrist_Roll", "Wrist_Pitch", "Wrist_Yaw", "Gripper1", "Gripper2"]

    observation_data = []  # per-step obs data for optional Excel export

    # "last good" value: updated every non-terminal step so the final summary
    # always shows pre-reset values rather than the reset state
    last_visible_pixels = 0

    ### (7) INITIAL STATE PRINT ###
    # Print joint positions and velocities before any action.

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
            # Save current obs and joint positions before stepping so we can
            # compute real_move = pos_after - pos_before.

            obs_before = obs
            pos_before = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            true_pos   = robot_asset.data.joint_pos[0].cpu().numpy()
            true_vel   = robot_asset.data.joint_vel[0].cpu().numpy()

            ### (10) OPTIONAL: SAVE OBSERVATION FRAME TO DISK ###
            # Saves the exact camera_rgb_observation used during training
            # (resized to 168x168, normalized to [0,1]) as a PNG file.

            if SAVE_OBS_FRAMES:
                try:
                    rgb = camera_sensor.data.output["rgb"][..., :3].float() / 255.0  # [B, H, W, 3]
                    rgb = rgb.permute(0, 3, 1, 2)                                     # [B, 3, H, W]
                    if rgb.shape[-2:] != (168, 168):
                        rgb = torch.nn.functional.interpolate(
                            rgb, size=(168, 168), mode='bilinear', align_corners=False
                        )
                    img_hwc   = rgb[0].permute(1, 2, 0).cpu().numpy()
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

            ### (12) APPLY ACTION FILTER (EMA) ###
            # filtered = alpha * raw + (1 - alpha) * previous_filtered
            # Higher alpha = more responsive; lower alpha = smoother commands.

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
            # obs_before is the TensorDict from the previous step.
            # We extract the "policy" key which holds the image tensor.
            # obs_flat is used only for the joint-level logging section.

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

            ### (17) GALLBLADDER EXPOSURE MONITORING (replaces dist/orient error) ###
            # Every non-terminal step: read RGB from camera, count visible gallbladder pixels,
            # check success condition, print result.

            if not done_flag:
                visible_pixels = 0
                success_bool   = False

                try:
                    cam_out    = camera_sensor.data.output
                    rgb_tensor = cam_out.get("rgb", None)
                    if rgb_tensor is not None:
                        img_rgb        = rgb_tensor[0, :, :, :3].cpu().numpy()
                        visible_pixels = gallbladder_pixel_count(img_rgb)

                        # success: gallbladder exposure above threshold
                        success_status = gallbladder_visibility_success(
                            base_env, pixel_threshold=GALLBLADDER_PIXEL_THRESHOLD
                        )
                        if isinstance(success_status, torch.Tensor):
                            success_bool = success_status[0].item() if success_status.shape[0] > 0 else False
                        else:
                            success_bool = bool(success_status[0]) if hasattr(success_status, '__getitem__') else bool(success_status)
                    else:
                        if timestep == 0:
                            print("[DEBUG] RGB tensor is None")
                except Exception as e:
                    if timestep == 0:
                        print(f"[DEBUG] Camera error: {e}")

                # store for final summary
                last_visible_pixels = visible_pixels

                success_text = "✓ SUCCESS" if success_bool else "✗ NO SUCCESS"
                print(f"[GALLBLADDER] Visible pixels: {visible_pixels:6d} | "
                      f"Threshold: {GALLBLADDER_PIXEL_THRESHOLD} | {success_text}")

                if success_bool:
                    print("\n" + "=" * 80)
                    print("GALLBLADDER EXPOSED — TARGET REACHED!")
                    print("=" * 80)
                    print(f"Visible pixels: {visible_pixels} (threshold: > {GALLBLADDER_PIXEL_THRESHOLD})")
                    print("=" * 80 + "\n")


                ### (19) PRINT OBSERVATIONS AND ACTION TABLE ###

                obs_size      = len(obs_flat)
                joint_pos_obs = obs_flat[:8]
                joint_vel_obs = obs_flat[8:16]
                actions_obs   = obs_flat[16:22]

                print(f"\nOBSERVATIONS (size={obs_size}):")
                print(f"  joint_pos_rel (8):  {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_pos_obs))}")
                print(f"  joint_vel_rel (8):  {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_vel_obs))}")
                print(f"  last_action (6):    {' | '.join(f'{i:8.5f}' for i in actions_obs)}")

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

                ### (20) SAVE TIMESTEP DATA ###
                if SAVE_DATA:
                    row_data = {'timestep': timestep}
                    for j in range(6):
                        row_data[f'position_j{j}']          = pos_after[j]
                        row_data[f'raw_policy_j{j}']        = raw_policy[j]
                        row_data[f'processed_actions_j{j}'] = processed_actions[j]
                        row_data[f'real_movement_j{j}']     = real_move[j]
                    row_data['visible_pixels'] = visible_pixels
                    row_data['success']        = int(success_bool)
                    timestep_data_list.append(row_data)

        ### (21) SNAPSHOT OF FINAL METRICS BEFORE TERMINATION CHECK ###
        final_visible_pixels = last_visible_pixels
        final_timestep       = timestep
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

            # ===== SAVE BINARY MASK AND FINAL FRAME AT EPISODE END =====
            if SAVE_MASK:
                try:
                    if camera_sensor is not None and hasattr(camera_sensor, "data") and hasattr(camera_sensor.data, "output"):
                        camera_output = camera_sensor.data.output
                        if "rgb" in camera_output:
                            camera_rgb_raw = camera_output["rgb"][0, ..., :3]
                            camera_rgb = camera_rgb_raw.cpu().numpy() if torch.is_tensor(camera_rgb_raw) else np.array(camera_rgb_raw)
                            if camera_rgb.max() > 1.0:
                                camera_rgb = camera_rgb / 255.0

                            mask_output_dir = os.path.join(log_dir, "masks")
                            os.makedirs(mask_output_dir, exist_ok=True)

                            # save final RGB frame
                            frame_rgb_uint8 = (camera_rgb * 255.0).astype(np.uint8)
                            frame_bgr = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)
                            frame_path = os.path.join(mask_output_dir, f"final_frame_step{timestep:04d}.png")
                            cv2.imwrite(frame_path, frame_bgr)
                            print(f"[INFO] Final RGB frame saved: {frame_path}")
                            print(f"       Shape: {frame_rgb_uint8.shape}, dtype: {frame_rgb_uint8.dtype}")

                            # create and save binary gallbladder mask
                            camera_tensor = torch.from_numpy(camera_rgb).unsqueeze(0).to(torch.float32)
                            mask          = gallbladder_mask_tensor(camera_tensor)
                            mask_binary   = (mask[0].cpu().numpy() > 0.5).astype(np.uint8) * 255

                            mask_path = os.path.join(mask_output_dir, f"gallbladder_mask_step{timestep:04d}.png")
                            cv2.imwrite(mask_path, mask_binary)
                            print(f"[INFO] Binary mask saved: {mask_path}")
                            print(f"       Mask shape: {mask_binary.shape}, unique values: {np.unique(mask_binary)}")
                            print(f"       White pixels (mask=255): {np.sum(mask_binary == 255)}, "
                                  f"Black pixels (mask=0): {np.sum(mask_binary == 0)}")
                        else:
                            print("[WARNING] No RGB data in camera output")
                    else:
                        print("[WARNING] Camera sensor not available for mask/frame save")
                except Exception as e:
                    print(f"[WARNING] Could not save mask/frame at step {timestep}: {type(e).__name__}: {e}")
            # ===== END MASK/FRAME SAVE =====

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
            # reset buffers for next episode
            filtered_actions   = None
            timestep_data_list = []
            observation_data   = []
            break

        else:
            timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            break

    ### (23) FINAL SUMMARY ###

    try:
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Episode terminated: {final_episode_reason}")
        print(f"Total steps to completion: {final_timestep}")
        print(f"Final visible gallbladder pixels: {final_visible_pixels} "
              f"(threshold: > {GALLBLADDER_PIXEL_THRESHOLD})")
        print("=" * 80)
    except Exception as e:
        print(f"[WARN] Unable to compute final summary: {e}")

    ### (24) CLOSE ENVIRONMENT ###
    env.close()


### (25) ENTRY POINT ###
if __name__ == "__main__":
    main()
    simulation_app.close()