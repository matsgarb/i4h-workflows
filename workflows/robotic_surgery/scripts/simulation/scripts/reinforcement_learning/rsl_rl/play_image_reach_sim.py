# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL.
   IMAGE-BASED version for the REACH task.
   Success criteria: distance error < 3 mm AND orientation error < 0.3 rad.
   No action filtering. No gallbladder monitoring.
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
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
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
    #   - permute image tensors from (B, H, W, C) → (B, C, H, W) and normalize to [0, 1]
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
    ee_frame    = base_env.scene["ee_frame"]

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

    # reference frame visualization toggle
    SHOW_REF_FRAMES         = False
    camera_frame_marker     = None
    robot_root_frame_marker = None
    target_frame_marker     = None
    target_point_marker     = None
    marker_container        = {"ee_marker": None}

    # action filtering: disabled for reach task (policy outputs used directly)
    USE_ACTION_FILTER   = False
    action_filter_alpha = 0.1
    filtered_actions    = None

    # success thresholds for the reach task
    DIST_THRESHOLD   = 0.003  # 3 mm
    ORIENT_THRESHOLD = 0.3    # rad (~17°)

    ### (6) VARIABLE INITIALIZATION ###

    obs      = env.get_observations()
    timestep = 0

    joint_names = ["Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"]

    # "last good" values: updated every non-terminal step so the final summary
    # always shows pre-reset values rather than the reset state after done=True
    last_ee_pos_b     = None
    last_ee_quat_b    = None
    last_ee_pos_w     = None
    last_ee_quat_w    = None
    last_dist_error   = None
    last_orient_error = None

    # for the final summary, we also track these at the outer scope
    episode_reason = "UNKNOWN"

    ### (7) INITIAL STATE PRINT ###
    # Print joint positions, velocities, and initial EE pose (robot RF) before any action.

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
    ee_pos_w_init  = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w_init = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b_init, ee_quat_b_init = subtract_frame_transforms(robot_root_pos, robot_root_quat, ee_pos_w_init, ee_quat_w_init)
    e_pos_b_init  = ee_pos_b_init[0].cpu().numpy()
    e_quat_b_init = ee_quat_b_init[0].cpu().numpy()
    print(f"\nEE frame (robot RF): X={e_pos_b_init[0]:.5f}, Y={e_pos_b_init[1]:.5f}, Z={e_pos_b_init[2]:.5f}, "
          f"w={e_quat_b_init[0]:.5f}, x={e_quat_b_init[1]:.5f}, y={e_quat_b_init[2]:.5f}, z={e_quat_b_init[3]:.5f}")
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
                    camera = base_env.scene.sensors["camera"]
                    rgb = camera.data.output["rgb"][..., :3].float() / 255.0  # [B, H, W, 3]
                    rgb = rgb.permute(0, 3, 1, 2)                              # [B, 3, H, W]
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

            ### (12) APPLY FILTER OR USE RAW ACTIONS ###
            # For reach, filtering is disabled. EMA formula if enabled:
            # filtered = alpha * raw + (1 - alpha) * prev_filtered

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

            ### (16) POSE EXTRACTION, ERROR COMPUTATION, AND PRINTING (only if not done) ###
            # Skip on the terminal step when done=True: env has already reset the robot,
            # so poses would reflect the reset state. Use last stored values in summary instead.

            if not done_flag:
                robot_root_pos  = robot_asset.data.root_state_w[:, :3]
                robot_root_quat = robot_asset.data.root_state_w[:, 3:7]

                # --- target liver pose (robot RF) ---
                target_pose_w = liver_target_pose_world(base_env)
                target_pos_w  = target_pose_w[:, :3]
                target_quat_w = target_pose_w[:, 3:7]
                target_pos_b, target_quat_b = subtract_frame_transforms(
                    robot_root_pos, robot_root_quat, target_pos_w, target_quat_w
                )
                t_pos_b  = target_pos_b[0].detach().cpu().numpy()
                t_quat_b = target_quat_b[0].detach().cpu().numpy()

                # --- EE pose (world frame and robot RF) ---
                ee_pos_w  = ee_frame.data.target_pos_w[..., 0, :]
                ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w
                )
                e_pos_w  = ee_pos_w[0].cpu().numpy()
                e_quat_w = ee_quat_w[0].cpu().numpy()
                e_pos_b  = ee_pos_b[0].cpu().numpy()
                e_quat_b = ee_quat_b[0].cpu().numpy()

                # --- distance and orientation errors ---
                ee_pos_b_t  = torch.tensor(e_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                ee_quat_b_t = torch.tensor(e_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                t_pos_b_t   = torch.tensor(t_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                t_quat_b_t  = torch.tensor(t_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)

                dist_error   = torch.norm(ee_pos_b_t - t_pos_b_t, dim=1)[0].cpu().item()
                orient_error = quat_error_magnitude(ee_quat_b_t, t_quat_b_t)[0].cpu().item()

                # store for final summary (overwritten every non-terminal step)
                last_ee_pos_b     = e_pos_b.copy()
                last_ee_quat_b    = e_quat_b.copy()
                last_ee_pos_w     = e_pos_w.copy()
                last_ee_quat_w    = e_quat_w.copy()
                last_dist_error   = dist_error
                last_orient_error = orient_error

                ### (17) REFERENCE FRAME VISUALIZATION (optional) ###
                if SHOW_REF_FRAMES:
                    try:
                        num_envs       = robot_root_pos.shape[0]
                        marker_indices = torch.zeros(num_envs, dtype=torch.int32, device=base_env.device)

                        if robot_root_frame_marker is None:
                            robot_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/RobotRootFrame_Play")
                            robot_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
                            robot_root_frame_marker = VisualizationMarkers(robot_cfg)

                        if target_frame_marker is None:
                            target_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TargetFrame_Play")
                            target_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
                            target_frame_marker = VisualizationMarkers(target_cfg)

                        if camera_frame_marker is None:
                            cam_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/CameraFrame_Play")
                            cam_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
                            camera_frame_marker = VisualizationMarkers(cam_cfg)

                        if target_point_marker is None:
                            point_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/TargetPoint_Play")
                            target_point_marker = VisualizationMarkers(point_cfg)

                        if marker_container["ee_marker"] is None:
                            ee_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame_Play")
                            ee_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
                            marker_container["ee_marker"] = VisualizationMarkers(ee_marker_cfg)

                        robot_root_frame_marker.visualize(robot_root_pos, robot_root_quat, marker_indices=marker_indices)
                        target_frame_marker.visualize(target_pos_w, target_quat_w, marker_indices=marker_indices)

                        quat_id = torch.zeros((num_envs, 4), device=base_env.device, dtype=target_pos_w.dtype)
                        quat_id[:, 0] = 1.0
                        target_point_marker.visualize(target_pos_w, quat_id, marker_indices=marker_indices)

                        ee_pos_w_t  = torch.tensor(e_pos_w, dtype=torch.float32, device=base_env.device).unsqueeze(0)
                        ee_quat_w_t = torch.tensor(e_quat_w, dtype=torch.float32, device=base_env.device).unsqueeze(0)
                        marker_container["ee_marker"].visualize(
                            ee_pos_w_t, ee_quat_w_t,
                            marker_indices=torch.zeros(1, dtype=torch.int32, device=base_env.device)
                        )

                        try:
                            camera_sensor = base_env.scene.sensors["camera"]
                        except Exception:
                            camera_sensor = None
                        if camera_sensor is not None and hasattr(camera_sensor.data, "pos_w"):
                            cam_pos_w = camera_sensor.data.pos_w
                            if hasattr(camera_sensor.data, "quat_w"):
                                cam_quat_w_vis = camera_sensor.data.quat_w
                            else:
                                cam_quat_w_vis = torch.zeros((num_envs, 4), device=base_env.device, dtype=cam_pos_w.dtype)
                                cam_quat_w_vis[:, 0] = 1.0
                            camera_frame_marker.visualize(cam_pos_w, cam_quat_w_vis, marker_indices=marker_indices)
                    except Exception as e:
                        print(f"[DEBUG] ref-frame visualization error: {e}")

                ### (18) PRINT STEP INFO ###
                # Prints EE pose, target pose, dist/orient errors, and joint table.

                print("EE frame (robot RF):    X={:.4f}, Y={:.4f}, Z={:.4f}, "
                      "w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                      e_pos_b[0], e_pos_b[1], e_pos_b[2],
                      e_quat_b[0], e_quat_b[1], e_quat_b[2], e_quat_b[3]))
                print("TARGET pose (robot RF): X={:.4f}, Y={:.4f}, Z={:.4f}, "
                      "w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                      t_pos_b[0], t_pos_b[1], t_pos_b[2],
                      t_quat_b[0], t_quat_b[1], t_quat_b[2], t_quat_b[3]))
                print(f"DISTANCE ERROR: {dist_error:.6f} m | ORIENTATION ERROR: {orient_error:.6f} rad")

                # in-loop success notification
                # (the environment's termination manager is the one that actually ends the episode)
                if dist_error < DIST_THRESHOLD and orient_error < ORIENT_THRESHOLD:
                    print("\n" + "=" * 80)
                    print("TARGET REACHED!")
                    print("=" * 80)
                    print(f"DISTANCE ERROR:    {dist_error:.6f} m    (threshold: < {DIST_THRESHOLD})")
                    print(f"ORIENTATION ERROR: {orient_error:.6f} rad (threshold: < {ORIENT_THRESHOLD})")
                    print("=" * 80 + "\n")

                # joint table
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

            # print the last good EE pose and errors (pre-reset values)
            if last_ee_pos_b is not None:
                print("\n" + "="*80)
                print("STEP before reset")
                print("="*80)
                print(f"EE frame (world frame): X={last_ee_pos_w[0]:.4f}, Y={last_ee_pos_w[1]:.4f}, "
                      f"Z={last_ee_pos_w[2]:.4f}, w={last_ee_quat_w[0]:.4f}, x={last_ee_quat_w[1]:.4f}, "
                      f"y={last_ee_quat_w[2]:.4f}, z={last_ee_quat_w[3]:.4f}")
                print(f"EE frame (robot RF):   X={last_ee_pos_b[0]:.4f}, Y={last_ee_pos_b[1]:.4f}, "
                      f"Z={last_ee_pos_b[2]:.4f}, w={last_ee_quat_b[0]:.4f}, x={last_ee_quat_b[1]:.4f}, "
                      f"y={last_ee_quat_b[2]:.4f}, z={last_ee_quat_b[3]:.4f}")
                print(f"DISTANCE ERROR: {last_dist_error*1000:.4f} mm | "
                      f"ORIENTATION ERROR: {last_orient_error*180/np.pi:.4f}° ({last_orient_error:.6f} rad)")
                print("="*80)

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
        if last_dist_error is not None:
            print(f"DISTANCE ERROR:    {last_dist_error*1000:.4f} mm (threshold: {DIST_THRESHOLD*1000:.1f} mm)")
            print(f"ORIENTATION ERROR: {last_orient_error*180/np.pi:.4f}° "
                  f"({last_orient_error:.6f} rad, threshold: {ORIENT_THRESHOLD} rad)")
        print("=" * 80)
    except Exception as e:
        print(f"[WARN] Unable to compute final summary: {e}")

    ### (21) CLOSE ENVIRONMENT ###
    env.close()


### (22) ENTRY POINT ###
if __name__ == "__main__":
    main()
    simulation_app.close()