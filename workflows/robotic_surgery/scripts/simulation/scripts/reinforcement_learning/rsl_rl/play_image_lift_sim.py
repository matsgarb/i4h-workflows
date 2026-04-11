# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL.
   IMAGE-BASED version for the LIFT task.
   Success criteria: gallbladder visible pixels > GALLBLADDER_PIXEL_THRESHOLD.
   Action filtering enabled (EMA). No dist/orient error monitoring.
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
args_cli = parser.parse_args()
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
import numpy as np
import gymnasium as gym
import robotic.surgery.tasks  # noqa: F401
import torch
import cv2
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import (
    gallbladder_pixel_count,
    gallbladder_visibility_success,
    gallbladder_mask_tensor,
)
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


def main():

    ### (1) ENV AND AGENT CONFIGURATION ###
    # Parse env config and agent config, locate the checkpoint, create the gym env.

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
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
    # For image-based RL:
    #   - permute image tensors from (B, H, W, C) -> (B, C, H, W) and normalize to [0, 1]
    #   - inject "dummy_state" key of shape (B, 0) required by some RSL-RL actor configs

    from tensordict import TensorDict

    def ensure_tensordict(data):
        obs = data[0] if isinstance(data, tuple) else data
        if isinstance(obs, TensorDict):
            td = obs.to(agent_cfg.device)
        elif isinstance(obs, dict):
            td = TensorDict(obs, batch_size=env.num_envs).to(agent_cfg.device)
        else:
            td = TensorDict({"policy": obs.to(agent_cfg.device)}, batch_size=env.num_envs)

        # normalize image tensors: (B, H, W, C) -> (B, C, H, W), [0,255] -> [0,1]
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
    # Grab handles to the robot and camera sensor from the unwrapped env scene.

    base_env      = getattr(env, "unwrapped", env)
    robot_asset   = base_env.scene["robot"]
    camera_sensor = base_env.scene.sensors["camera"]

    ### (5) USER-TUNABLE FLAGS AND VARIABLES ###

    # liver stabilization warm-up: number of steps with zero actions before inference starts
    STABILIZATION_STEPS = 20

    # observation frame saving (saves the exact camera image fed to the policy at each step)
    SAVE_OBS_FRAMES = False
    obs_frames_dir  = None
    if SAVE_OBS_FRAMES:
        obs_frames_dir = os.path.join(log_dir, "obs_frames")
        os.makedirs(obs_frames_dir, exist_ok=True)
        print(f"[INFO] Observation frames will be saved to: {obs_frames_dir}")

    # reference frame visualization toggle (robot root, EE, camera frames)
    SHOW_REF_FRAMES         = False
    camera_frame_marker     = None
    robot_root_frame_marker = None
    ee_marker               = None

    # action filtering: enabled for lift task to smooth commands
    # EMA formula: filtered = alpha * raw + (1 - alpha) * prev_filtered
    USE_ACTION_FILTER   = True
    action_filter_alpha = 0.1  # higher = more responsive, lower = smoother
    filtered_actions    = None

    # mask saving: saves binary gallbladder mask + final RGB frame at episode end
    SAVE_MASK = True
    if SAVE_MASK:
        print(f"[INFO] Mask saving enabled. Binary mask and final frame will be saved at episode end.")

    # gallbladder exposure success threshold (pixels)
    GALLBLADDER_PIXEL_THRESHOLD = 27000

    ### (6) VARIABLE INITIALIZATION ###

    obs      = env.get_observations()
    timestep = 0

    joint_names = ["Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"]

    # "last good" pixel count: updated every non-terminal step for the final summary
    last_visible_pixels = 0

    # for the final summary, tracked at outer scope
    episode_reason = "UNKNOWN"

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

            ### (10) OPTIONAL: SAVE OBSERVATION FRAME TO DISK ###
            # Saves the exact camera image used during training (168x168, [0,1]) as a PNG.

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

            ### (14) DONE FLAG CHECK ###

            done_flag = False
            if torch.is_tensor(dones):
                done_flag = bool(torch.any(dones))
            elif isinstance(dones, (np.ndarray, list, tuple)):
                done_flag = bool(np.any(dones))
            elif isinstance(dones, bool):
                done_flag = dones

            ### (15) ACTION LOGGING ###

            cmd_term          = base_env.action_manager.get_term("arm_action")
            processed_actions = cmd_term.processed_actions[0].cpu().numpy()
            pos_after         = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            real_move         = pos_after - pos_before

            ### (16) GALLBLADDER EXPOSURE MONITORING (every non-terminal step) ###
            # Read RGB from camera, count visible gallbladder pixels, check success condition.
            # This replaces the dist/orient error monitoring used in the reach task.

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

                # store for final summary (overwritten every non-terminal step)
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

                ### (17) REFERENCE FRAME VISUALIZATION (optional) ###
                if SHOW_REF_FRAMES:
                    try:
                        robot_root_pos  = robot_asset.data.root_state_w[:, :3]
                        robot_root_quat = robot_asset.data.root_state_w[:, 3:7]
                        ee_frame        = base_env.scene["ee_frame"]
                        ee_pos_w        = ee_frame.data.target_pos_w[..., 0, :]
                        ee_quat_w       = ee_frame.data.target_quat_w[..., 0, :]
                        num_envs        = robot_root_pos.shape[0]
                        marker_indices  = torch.zeros(num_envs, dtype=torch.int32, device=base_env.device)

                        if robot_root_frame_marker is None:
                            robot_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/RobotRootFrame_Play")
                            robot_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
                            robot_root_frame_marker = VisualizationMarkers(robot_cfg)

                        if camera_frame_marker is None:
                            cam_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/CameraFrame_Play")
                            cam_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
                            camera_frame_marker = VisualizationMarkers(cam_cfg)

                        if ee_marker is None:
                            ee_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame_Play")
                            ee_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
                            ee_marker = VisualizationMarkers(ee_marker_cfg)

                        robot_root_frame_marker.visualize(robot_root_pos, robot_root_quat, marker_indices=marker_indices)
                        ee_marker.visualize(ee_pos_w, ee_quat_w, marker_indices=marker_indices)

                        if hasattr(camera_sensor.data, "pos_w"):
                            cam_pos_w = camera_sensor.data.pos_w
                            if hasattr(camera_sensor.data, "quat_w"):
                                cam_quat_w_vis = camera_sensor.data.quat_w
                            else:
                                cam_quat_w_vis = torch.zeros((num_envs, 4), device=base_env.device, dtype=cam_pos_w.dtype)
                                cam_quat_w_vis[:, 0] = 1.0
                            camera_frame_marker.visualize(cam_pos_w, cam_quat_w_vis, marker_indices=marker_indices)
                    except Exception as e:
                        print(f"[DEBUG] ref-frame visualization error: {e}")

                ### (18) PRINT JOINT TABLE ###

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

        ### (19) EPISODE TERMINATION CHECK AND LOGGING ###

        done_flag = False
        if torch.is_tensor(dones):
            done_flag = bool(torch.any(dones))
        elif isinstance(dones, (np.ndarray, list, tuple)):
            done_flag = bool(np.any(dones))
        elif isinstance(dones, bool):
            done_flag = dones

        if done_flag:
            # determine termination reason from extras log
            filtered_actions = None
            if isinstance(extras, dict) and "log" in extras:
                log = extras["log"]
                time_out_value = log.get("Episode_Termination/time_out", 0)
                success_value  = log.get("Episode_Termination/success", 0)
                if hasattr(time_out_value, 'item'): time_out_value = time_out_value.item()
                if hasattr(success_value,  'item'): success_value  = success_value.item()
                if success_value   == 1: episode_reason = "ACHIEVED SUCCESS"
                elif time_out_value == 1: episode_reason = "TIME OUT"

            print(f"[INFO] Episode terminated: {episode_reason}")

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

            # print initial joint positions at termination on one line
            pos_str = " ".join([f"{p:.5f}".replace(".", ",") for p in initial_pos])
            print(f"JOINT RESET POSITION: {pos_str}")

            # stop robot by zeroing velocity target
            try:
                zero_vel = torch.zeros((1, 6), device=robot_asset.device)
                robot_asset.set_joint_velocity_target(zero_vel, joint_ids=slice(0, 6))
                robot_asset.write_data_to_sim()
                print("[INFO] Stopping robot (velocity = 0).")
            except Exception as e:
                print(f"[WARN] Unable to stop robot: {e}")

            print("[INFO] Stopping simulation.")
            break

        else:
            timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            break

    ### (20) FINAL SUMMARY (using pre-termination values) ###

    try:
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Episode terminated: {episode_reason}")
        print(f"Total steps to completion: {timestep}")
        print(f"Final visible gallbladder pixels: {last_visible_pixels} "
              f"(threshold: > {GALLBLADDER_PIXEL_THRESHOLD})")
        print("=" * 80)
    except Exception as e:
        print(f"[WARN] Unable to compute final summary: {e}")

    ### (21) CLOSE ENVIRONMENT ###
    env.close()


### (22) ENTRY POINT ###
if __name__ == "__main__":
    main()
    simulation_app.close()