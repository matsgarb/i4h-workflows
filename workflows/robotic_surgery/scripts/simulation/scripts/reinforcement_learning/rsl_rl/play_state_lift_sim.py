# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# import libraries
import os
import numpy as np
import gymnasium as gym
import robotic.surgery.tasks
import torch
import cv2
import struct
import pandas as pd
from tensordict import TensorDict
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import liver_target_pose_world, gallbladder_pixel_count, gallbladder_visibility_success, visual_exposure_reward, vertical_lifting_reward
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import gallbladder_mask_tensor
from isaaclab.managers.reward_manager import RewardTermCfg
from isaaclab.envs.mdp.rewards import joint_vel_l2
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG



def main():
    ### (1) ENV and AGENT CONFIGURATION ###
    
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording 
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl compatibility 
    env = RslRlVecEnvWrapper(env)

    ### (2)TENSORDICT COMPATIBILITY PATCH ###
    # RSL-RL's OnPolicyRunner expects observations as a TensorDict with a "policy" key.
    # This patch ensures that the environment's observations are converted to the expected
    # format without modifying the original environment code. 
    # It wraps the original get_observations and step functions to convert outputs to TensorDicts as needed.
    
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
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
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
    base_env = getattr(env, "unwrapped", env)
    robot_asset = base_env.scene["robot"]
    ee_frame = base_env.scene["ee_frame"]
    camera_sensor = base_env.scene.sensors["camera"]
    
    ### (5) USER-TUNABLE FLAGS AND VARIABLES ###
    # liver stabilization warm-up (number of steps to run with zero actions to let the liver settle before starting policy inference)
    STABILIZATION_STEPS = 20 

    # reference frame visualization (markers) variables
    SHOW_REF_FRAMES = False
    camera_frame_marker = None
    robot_root_frame_marker = None
    target_frame_marker = None
    target_point_marker = None
    ee_marker = None

    # action filtering 
    USE_ACTION_FILTER = False
    action_filter_alpha = 0.1
    filtered_actions = None

    # data saving
    SAVE_DATA = False
    SAVE_MASK = False  
    timestep_data_list = []
    if SAVE_DATA:
        print(f"[INFO] Data saving enabled. Data will be saved as Excel file at episode end.")
    if SAVE_MASK:
        print(f"[INFO] Mask saving enabled. Binary mask and final frame will be saved at episode end.")
    
    ### (6) VARIABLE INITIALIZATION ###
    obs = env.get_observations() 
    timestep = 0
    joint_names = [
        "Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"
    ]
    joint_names_obs = [
        "Yaw", "Pitch", "Insert", "Wrist_Roll", "Wrist_Pitch", "Wrist_Yaw", "Gripper1", "Gripper2"
    ]
    
    
    ### (7) INITIAL STATE PRINT ###
    print("\n" + "="*80)
    print(f"INITIAL STATE (STEP 0 - RESET)")
    print("="*80)
    initial_pos = robot_asset.data.joint_pos[0][:6].cpu().numpy()
    initial_vel = robot_asset.data.joint_vel[0][:6].cpu().numpy()
    header = f"{'JOINT':<15} | {'POSITION':>12} | {'VELOCITY':>12}"
    print(header)
    print("-" * len(header))
    for i in range(6):
        print(f"{joint_names[i]:<15} | {initial_pos[i]:>12.5f} | {initial_vel[i]:>12.5f}")    
    # Print initial EE frame (robot reference frame)
    robot_root_pos = robot_asset.data.root_state_w[:, :3]
    robot_root_quat = robot_asset.data.root_state_w[:, 3:7]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w
    )
    e_pos_b = ee_pos_b[0].cpu().numpy()
    e_quat_b = ee_quat_b[0].cpu().numpy()
    print(f"\nEE frame (robot RF): X={e_pos_b[0]:.5f}, Y={e_pos_b[1]:.5f}, Z={e_pos_b[2]:.5f}, w={e_quat_b[0]:.5f}, x={e_quat_b[1]:.5f}, y={e_quat_b[2]:.5f}, z={e_quat_b[3]:.5f}")
    print("="*80 + "\\n")

    ### (8) LIVER STABILIZATION WARM-UP ###
    print(f"\n[INFO] Running {STABILIZATION_STEPS} stabilization steps (zero actions) to let liver settle...")
    action_dim = base_env.action_manager.total_action_dim
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
            obs_before = obs  # Observation at current timestep (s_t)
            pos_before = robot_asset.data.joint_pos[0][:6].cpu().numpy()  # Position at current timestep
            true_pos = robot_asset.data.joint_pos[0].cpu().numpy() 
            true_vel = robot_asset.data.joint_vel[0].cpu().numpy()
            
            
            ### (10) GET ACTION FROM POLICY BASED ON CURRENT OBSERVATION ###
            actions = policy(obs_before)
            raw_policy = actions[0].cpu().numpy()

            ### (11) APPLY FILTER OR USE RAW ACTIONS BASED ON FLAG ###
            # ---------> filtered action = alpha * raw_action + (1 - alpha) * previous_filtered_action
            if USE_ACTION_FILTER:
                if filtered_actions is None:
                    filtered_actions = actions.clone()
                else:
                    filtered_actions = action_filter_alpha * actions + (1.0 - action_filter_alpha) * filtered_actions
                actions_to_use = filtered_actions
                raw_val = actions[0, 2].cpu().item()  # Joint 2 (Insertion)
                filtered_val = actions_to_use[0, 2].cpu().item()
                print(f"[FILTER DEBUG] Joint 2 (Insertion): RAW={raw_val:+.4f} → FILTERED={filtered_val:+.4f} (alpha={action_filter_alpha})")
            else:
                actions_to_use = actions
                raw_val = actions[0, 2].cpu().item()
                print(f"[NO FILTER] Joint 2 (Insertion): RAW={raw_val:+.4f} → USED AS-IS={raw_val:+.4f}")

            ### (12) EXECUTE ACTION IN ENVIRONMENT AND GET NEW STATE (SIMULATION STEP) ###
            obs, rewards, dones, extras = env.step(actions_to_use) 

            ### (13) EXTRACT AND PRINT REWARD COMPONENTS AND SUCCESS METRICS ###
            try:
                camera = env.unwrapped.scene.sensors["camera"]
                if camera is not None:
                    cam_out = camera.data.output
                    rgb_tensor = cam_out.get("rgb", None)
                    if rgb_tensor is not None:
                        img_rgb = rgb_tensor[0, :, :, :3].cpu().numpy()
                        visible_pixels = gallbladder_pixel_count(img_rgb)
                        
                        # Check success condition
                        success_status = gallbladder_visibility_success(env.unwrapped, pixel_threshold=2700)
                        # Handle tensor output - ensure we get a scalar
                        if isinstance(success_status, torch.Tensor):
                            success_bool = success_status[0].item() if success_status.shape[0] > 0 else False
                        else:
                            success_bool = bool(success_status[0]) if hasattr(success_status, '__getitem__') else bool(success_status)
                        
                        # Compute visual exposure reward
                        vis_reward = visual_exposure_reward(env.unwrapped)
                        # Handle tensor output
                        if isinstance(vis_reward, torch.Tensor):
                            vis_reward_value = vis_reward[0].item() if vis_reward.shape[0] > 0 else 0.0
                        else:
                            vis_reward_value = float(vis_reward[0]) if hasattr(vis_reward, '__getitem__') else float(vis_reward)
                        
                        success_text = "✓ SUCCESS" if success_bool else "✗ NO SUCCESS"
                        print(f"[GALLBLADDER] Visible pixels: {visible_pixels} | Threshold: 2700 | {success_text} | visibility reward: {vis_reward_value:.4f}")
                        try:
                            lift_reward_tensor = vertical_lifting_reward(env.unwrapped, lateral_penalty_coef=1.0)
                            lift_reward_value = lift_reward_tensor[0].item()
                            ee_pos_now = env.unwrapped.scene["ee_frame"].data.target_pos_w[0, 0, :].cpu()
                            reset_pos  = env.unwrapped._ee_reset_pos[0].cpu()
                            dx = (ee_pos_now[0] - reset_pos[0]).item()
                            dy = (ee_pos_now[1] - reset_pos[1]).item()
                            dz = (ee_pos_now[2] - reset_pos[2]).item()

                            print(f"[LIFTING]     Δx={dx:+.5f}  Δy={dy:+.5f}  Δz={dz:+.5f} | lifting reward: {lift_reward_value:+.4f}")
                        except Exception as e_lift:
                            print(f"[LIFTING]     (not available: {e_lift})")
                    else:
                        if timestep == 0:
                            print("[DEBUG] RGB tensor is None")
                else:
                    if timestep == 0:
                        print("[DEBUG] Camera is None")
            except Exception as e:
                if timestep == 0:
                    print(f"[DEBUG] Camera error: {e}")  # Print error only on first step

            # Extract obs_flat correctly (same logic as later in the code)
            if isinstance(obs_before, dict):
                obs_data = obs_before.get("policy", obs_before)
            elif hasattr(obs_before, 'get'):
                obs_data = obs_before.get("policy", obs_before)
            else:
                obs_data = obs_before
            
            if torch.is_tensor(obs_data):
                obs_flat = obs_data[0].cpu().numpy() if obs_data.dim() > 1 else obs_data.cpu().numpy()
            else:
                obs_flat = np.array(obs_data)
            
            noisy_pos = obs_flat[:8] # obs file
            noisy_vel = obs_flat[8:16] # obs file
            obs_row = {'step': timestep} # obs file
            
            for j in range(8): # obs file
                name = joint_names_obs[j] if j < len(joint_names_obs) else f"joint_{j}" # obs file
                obs_row[f'true_pos_{name}'] = true_pos[j] # obs file
                obs_row[f'noisy_pos_{name}'] = noisy_pos[j] # obs file
                obs_row[f'pos_error_{name}'] = true_pos[j] - noisy_pos[j] # obs file

                obs_row[f'true_vel_{name}'] = true_vel[j] # obs file
                obs_row[f'noisy_vel_{name}'] = noisy_vel[j] # obs file
                obs_row[f'vel_error_{name}'] = true_vel[j] - noisy_vel[j] # obs file
            observation_data.append(obs_row) # obs file


            # compute done flag early
            done_flag = False
            if torch.is_tensor(dones):
                done_flag = bool(torch.any(dones))
            elif isinstance(dones, (np.ndarray, list, tuple)):
                done_flag = bool(np.any(dones))
            elif isinstance(dones, bool):
                done_flag = dones

            # # timing
            # now = time.perf_counter()
            # dt = now - last_step_time
            # last_step_time = now
            # print(f"STEP DT: {dt*1000:.2f} ms ({1.0/dt if dt > 0 else float('inf'):.1f} Hz)")

            # # ######## ACTION LOGGING ########
            # (1) JOINT SPACE
            cmd_term = base_env.action_manager.get_term("arm_action")
            processed_actions = cmd_term.processed_actions[0].cpu().numpy()

            pos_after = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            real_move = pos_after - pos_before

            
            # target pose wrt robot root frame
            robot_root_pos = robot_asset.data.root_state_w[:, :3]
            robot_root_quat = robot_asset.data.root_state_w[:, 3:7]
            r_pos_w = robot_root_pos[0].cpu().numpy()
            r_quat_w = robot_root_quat[0].cpu().numpy()
            target_pose_w = liver_target_pose_world(base_env)
            target_pos_w = target_pose_w[:, :3]
            target_quat_w = target_pose_w[:, 3:7]
            target_pos_b, target_quat_b = subtract_frame_transforms(
                robot_root_pos, robot_root_quat, target_pos_w, target_quat_w
            )
            t_pos_w = target_pos_w[0].cpu().numpy()
            t_quat_w = target_quat_w[0].cpu().numpy()
            t_pos_b = target_pos_b[0].cpu().numpy()
            t_quat_b = target_quat_b[0].cpu().numpy()
            ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
            ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w
            )
            e_pos_w = ee_pos_w[0].cpu().numpy()
            e_quat_w = ee_quat_w[0].cpu().numpy()
            e_pos_b = ee_pos_b[0].cpu().numpy()
            e_quat_b = ee_quat_b[0].cpu().numpy()

            # ===== OPTIONAL REF FRAME VISUALIZATION (COMMENT/DECOMMENT) =====
            # Shows: camera frame, robot root frame, target frame + target point.
            # Toggle with SHOW_REF_FRAMES above.
            if SHOW_REF_FRAMES:
                try:
                    num_envs = robot_root_pos.shape[0]
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

                    if "ee_marker" not in locals():
                        ee_marker = None
                    if ee_marker is None:
                        ee_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame_Play")
                        ee_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
                        ee_marker = VisualizationMarkers(ee_marker_cfg)

                    # Robot root frame
                    robot_root_frame_marker.visualize(robot_root_pos, robot_root_quat, marker_indices=marker_indices)

                    # Target frame + point (world frame)
                    target_frame_marker.visualize(target_pos_w, target_quat_w, marker_indices=marker_indices)
                    quat_id = torch.zeros((num_envs, 4), device=base_env.device, dtype=target_pos_w.dtype)
                    quat_id[:, 0] = 1.0
                    target_point_marker.visualize(target_pos_w, quat_id, marker_indices=marker_indices)

                    # Camera frame (world frame)
                    camera_sensor = None
                    try:
                        camera_sensor = base_env.scene.sensors["camera"]
                    except Exception:
                        try:
                            camera_sensor = base_env.scene["camera"]
                        except Exception:
                            camera_sensor = None

                    if camera_sensor is not None and hasattr(camera_sensor, "data") and hasattr(camera_sensor.data, "pos_w"):
                        cam_pos_w = camera_sensor.data.pos_w
                        if hasattr(camera_sensor.data, "quat_w"):
                            cam_quat_w = camera_sensor.data.quat_w
                        else:
                            cam_quat_w = torch.zeros((num_envs, 4), device=base_env.device, dtype=cam_pos_w.dtype)
                            cam_quat_w[:, 0] = 1.0
                        camera_frame_marker.visualize(cam_pos_w, cam_quat_w, marker_indices=marker_indices)

                    # EE frame (world frame)
                    ee_marker.visualize(ee_pos_w, ee_quat_w, marker_indices=marker_indices)
                except Exception as e:
                    print(f"[DEBUG] ref-frame visualization error: {e}")
            # ===== END VISUALIZATION (COMMENT/DECOMMENT) =====
            
            print("ROBOT frame (world):   X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                r_pos_w[0], r_pos_w[1], r_pos_w[2], r_quat_w[0], r_quat_w[1], r_quat_w[2], r_quat_w[3]))
            print("EE frame (world):      X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                e_pos_w[0], e_pos_w[1], e_pos_w[2], e_quat_w[0], e_quat_w[1], e_quat_w[2], e_quat_w[3]))
            print("EE frame (robot RF):   X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                e_pos_b[0], e_pos_b[1], e_pos_b[2], e_quat_b[0], e_quat_b[1], e_quat_b[2], e_quat_b[3]))
            # print("TARGET pose (world):    X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
            #     t_pos_w[0], t_pos_w[1], t_pos_w[2], t_quat_w[0], t_quat_w[1], t_quat_w[2], t_quat_w[3]))
            print("TARGET pose (robot RF): X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                t_pos_b[0], t_pos_b[1], t_pos_b[2], t_quat_b[0], t_quat_b[1], t_quat_b[2], t_quat_b[3]))
            
            # Compute and print distance and orientation errors directly from robot RF poses
            ee_pos_b_tensor = torch.tensor(e_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            ee_quat_b_tensor = torch.tensor(e_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            t_pos_b_tensor = torch.tensor(t_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            t_quat_b_tensor = torch.tensor(t_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            
            dist_error = torch.norm(ee_pos_b_tensor - t_pos_b_tensor, dim=1)[0].cpu().item()
            orient_error = quat_error_magnitude(ee_quat_b_tensor, t_quat_b_tensor)[0].cpu().item()
            
            # Compute joint velocity penalty (L2 squared norm)
            try:
                joint_vel_penalty = joint_vel_l2(base_env, asset_cfg=SceneEntityCfg("robot"))[0].cpu().item()
                print(f"DISTANCE ERROR: {dist_error:.6f} m | ORIENTATION ERROR: {orient_error:.6f} rad | JOINT VEL L2: {joint_vel_penalty:.6f}")
            except Exception as e:
                print(f"DISTANCE ERROR: {dist_error:.6f} m | ORIENTATION ERROR: {orient_error:.6f} rad | [joint vel error: {e}]")
            
            # Check if stopping conditions are met
            if dist_error < 0.003 and orient_error < 0.3:
                print("\n" + "=" * 80)
                print("TARGET REACHED!")
                print("=" * 80)
                print(f"DISTANCE ERROR: {dist_error:.6f} m (threshold: < 0.003)")
                print(f"ORIENTATION ERROR: {orient_error:.6f} rad (threshold: < 0.3)")
                print("=" * 80 + "\n")
            
            # STEP 4: Get position after action execution (pos_{t+1})
            pos_after = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            real_move = pos_after - pos_before

            # Print observations (extract from obs_before, which is the state at the beginning of this step)
            # Obs structure: [joint_pos(6), joint_vel(6), target_pose(7:3pos+4quat), actions(6)]
            # Total = 25 components
            obs_size = len(obs_flat)
            
            # Structure: joint_pos(8: 6arm+2gripper) + joint_vel(8) + target_pose(7) + actions(6) = 29
            joint_pos_obs = obs_flat[:8] if obs_size >= 8 else obs_flat[:obs_size]
            joint_vel_obs = obs_flat[8:16] if obs_size >= 16 else obs_flat[8:obs_size]
            
            
            # target_pose: last 13 = target_pose(7) + actions(6)
            target_pose_start = obs_size - 13 if obs_size >= 13 else -1
            if target_pose_start >= 0:
                target_pose_obs = obs_flat[target_pose_start:target_pose_start+7]
                actions_obs = obs_flat[target_pose_start+7:]
            else:
                target_pose_obs = []
                actions_obs = []
            
            # ===== PRINT OBSERVATIONS =====
            # Observation structure: joint_pos(8) + joint_vel(8) + actions(6) = 22
            print(f"\nOBSERVATIONS (size={obs_size}):")
            print(f"  joint_pos_rel (8):    {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_pos_obs))}")
            print(f"  joint_vel_rel (8):    {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_vel_obs))}")
            print(f"  last_action (6):      {' | '.join(f'{i:8.5f}' for i in actions_obs)}")
            # ===============================
            

            
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
            
            # ===== DATA SAVING =====
            if SAVE_DATA:
                # Create a row for this timestep
                row_data = {'timestep': timestep}
                
                # Add position (6 joints)
                for j in range(6):
                    row_data[f'position_j{j}'] = pos_after[j]
                
                # Add raw policy (6 values)
                for j in range(6):
                    row_data[f'raw_policy_j{j}'] = raw_policy[j]
                
                # Add processed actions (6 values)
                for j in range(6):
                    row_data[f'processed_actions_j{j}'] = processed_actions[j]
                
                # Add real movement (6 values)
                for j in range(6):
                    row_data[f'real_movement_j{j}'] = real_move[j]
                
                # Add distance and orientation errors
                row_data['distance_error'] = dist_error
                row_data['orientation_error'] = orient_error
                
                # Append to list
                timestep_data_list.append(row_data)
            # ======================

        done_flag = False
        if torch.is_tensor(dones):
            done_flag = bool(torch.any(dones))
        elif isinstance(dones, (np.ndarray, list, tuple)):
            done_flag = bool(np.any(dones))
        elif isinstance(dones, bool):
            done_flag = dones

        if done_flag:
            # Determine the reason for episode termination
            episode_reason = "UNKNOWN"
            
            # Check extras for termination reason
            if isinstance(extras, dict) and "log" in extras:
                log = extras["log"]
                time_out_value = log.get("Episode_Termination/time_out", 0)
                success_value = log.get("Episode_Termination/success", 0)
                
                # Convert tensor to scalar if needed
                if hasattr(time_out_value, 'item'):
                    time_out_value = time_out_value.item()
                if hasattr(success_value, 'item'):
                    success_value = success_value.item()
                
                if success_value == 1:
                    episode_reason = "ACHIEVED SUCCESS"
                elif time_out_value == 1:
                    episode_reason = "TIME OUT"
            
            print(f"[INFO] Episode terminated: {episode_reason}")
            # Print initial joint positions at termination on one line
            pos_str = " ".join([f"{p:.5f}".replace(".", ",") for p in initial_pos])
            print(f"Initial position of the joints: {pos_str}")
            
            # ===== SAVE BINARY MASK AND FINAL FRAME AT EPISODE END =====
            if SAVE_MASK:
                try:                    
                    if camera_sensor is not None and hasattr(camera_sensor, "data") and hasattr(camera_sensor.data, "output"):
                        # Get RGB data from camera (fresh data at final step)
                        camera_output = camera_sensor.data.output
                        
                        if "rgb" in camera_output:
                            camera_rgb_raw = camera_output["rgb"][0, ..., :3]  # [H, W, 3]
                            
                            # Convert to numpy
                            if torch.is_tensor(camera_rgb_raw):
                                camera_rgb = camera_rgb_raw.cpu().numpy()
                            else:
                                camera_rgb = np.array(camera_rgb_raw)
                            
                            # Normalize to [0, 1] if needed
                            if camera_rgb.max() > 1.0:
                                camera_rgb = camera_rgb / 255.0
                            
                            # Create output directory for masks
                            mask_output_dir = os.path.join(log_dir, "masks")
                            os.makedirs(mask_output_dir, exist_ok=True)
                            
                            # === SAVE FINAL RGB FRAME ===
                            frame_rgb_uint8 = (camera_rgb * 255.0).astype(np.uint8)
                            frame_bgr = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)
                            frame_path = os.path.join(mask_output_dir, f"final_frame_step{timestep:04d}.png")
                            cv2.imwrite(frame_path, frame_bgr)
                            print(f"[INFO] Final RGB frame saved: {frame_path}")
                            print(f"       Frame shape: {frame_rgb_uint8.shape}, dtype: {frame_rgb_uint8.dtype}")
                            
                            # === CREATE AND SAVE BINARY MASK ===
                            camera_tensor = torch.from_numpy(camera_rgb).unsqueeze(0).to(torch.float32)  # [1, H, W, C]
                            
                            # Get the mask (shape: [1, H, W])
                            mask = gallbladder_mask_tensor(camera_tensor)
                            mask_binary = (mask[0].cpu().numpy() > 0.5).astype(np.uint8) * 255  # Convert to 0-255 binary
                            
                            # Save the binary mask
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
                except ImportError:
                    print("[WARNING] gallbladder_mask_tensor function not available")
                except Exception as e:
                    print(f"[WARNING] Could not save mask/frame at step {timestep}: {type(e).__name__}: {e}")
            # ===== END MASK/FRAME SAVE =====
            
            
            # Save timestep data to Excel if enabled
            if SAVE_DATA and timestep_data_list:
                df_timesteps = pd.DataFrame(timestep_data_list)
                excel_filename = os.path.join(log_dir, f"episode_{timestep}_timesteps.xlsx")
                df_timesteps.to_excel(excel_filename, index=False)
                print(f"[INFO] Timestep data saved to: {excel_filename}")
            
            # Save episode data to Excel
            if episode_data:
                df = pd.DataFrame(episode_data)
                excel_filename = os.path.join(log_dir, f"episode_{timestep}_data.xlsx")
                # df.to_excel(excel_filename, index=False)
                # print(f"[INFO] Episode data saved to: {excel_filename}")
            
            if observation_data: # obs file
                df_obs = pd.DataFrame(observation_data) # obs file
                # excel_obs_filename = os.path.join(log_dir, f"observations_{timestep}_data.xlsx") # obs file
                # df_obs.to_excel(excel_obs_filename, index=False) # obs file
                # print(f"[INFO] Observation data saved to: {excel_obs_filename}") # obs file
            
            # Stop the robot by setting velocity targets to zero
            try:
                zero_vel = torch.zeros((1, 6), device=robot_asset.device)
                robot_asset.set_joint_velocity_target(zero_vel, joint_ids=slice(0, 6))
                robot_asset.write_data_to_sim()
                print("[INFO] Stopping robot (velocity = 0).")
            except Exception as e:
                print(f"[WARN] Unable to stop robot: {e}")
            print("[INFO] Stopping simulation.")
            
            timestep_data_list = []  # Reset timestep data for next episode
            # timestep = 0
            break
        else:
            timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break


    # ===== FINAL POSE SUMMARY =====
    try:
        robot_root_pos = robot_asset.data.root_state_w[:, :3]
        robot_root_quat = robot_asset.data.root_state_w[:, 3:7]

        # Target pose w.r.t. robot reference frame
        target_pose_w = liver_target_pose_world(base_env)
        target_pos_w = target_pose_w[:, :3]
        target_quat_w = target_pose_w[:, 3:7]
        target_pos_b, target_quat_b = subtract_frame_transforms(
            robot_root_pos, robot_root_quat, target_pos_w, target_quat_w
        )

        t_pos_b = target_pos_b[0].detach().cpu().numpy()
        t_quat_b = target_quat_b[0].detach().cpu().numpy()

        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(
            "TARGET pose (robot RF): "
            f"X={t_pos_b[0]:+.4f}, Y={t_pos_b[1]:+.4f}, Z={t_pos_b[2]:+.4f}, "
            f"w={t_quat_b[0]:+.4f}, x={t_quat_b[1]:+.4f}, y={t_quat_b[2]:+.4f}, z={t_quat_b[3]:+.4f}"
        )

        # Camera pose in world + robot frames
        camera_sensor = None
        try:
            camera_sensor = base_env.scene.sensors["camera"]
        except Exception:
            try:
                camera_sensor = base_env.scene["camera"]
            except Exception:
                camera_sensor = None

        if camera_sensor is not None and hasattr(camera_sensor, "data") and hasattr(camera_sensor.data, "pos_w"):
            cam_pos_w_tensor = camera_sensor.data.pos_w
            cam_pos_w = cam_pos_w_tensor[0].detach().cpu().numpy()

            # Some camera data objects do not expose quaternion; fallback to identity.
            if hasattr(camera_sensor.data, "quat_w"):
                cam_quat_w_tensor = camera_sensor.data.quat_w
            else:
                cam_quat_w_tensor = torch.zeros((cam_pos_w_tensor.shape[0], 4), device=base_env.device)
                cam_quat_w_tensor[:, 0] = 1.0

            cam_pos_b, cam_quat_b = subtract_frame_transforms(
                robot_root_pos, robot_root_quat, cam_pos_w_tensor, cam_quat_w_tensor
            )
            cam_pos_b_np = cam_pos_b[0].detach().cpu().numpy()
            cam_quat_b_np = cam_quat_b[0].detach().cpu().numpy()
            cam_quat_w_np = cam_quat_w_tensor[0].detach().cpu().numpy()

            print(
                "CAMERA pose (world):   "
                f"X={cam_pos_w[0]:+.4f}, Y={cam_pos_w[1]:+.4f}, Z={cam_pos_w[2]:+.4f}, "
                f"w={cam_quat_w_np[0]:+.4f}, x={cam_quat_w_np[1]:+.4f}, y={cam_quat_w_np[2]:+.4f}, z={cam_quat_w_np[3]:+.4f}"
            )
            print(
                "CAMERA pose (robot RF):"
                f" X={cam_pos_b_np[0]:+.4f}, Y={cam_pos_b_np[1]:+.4f}, Z={cam_pos_b_np[2]:+.4f}, "
                f"w={cam_quat_b_np[0]:+.4f}, x={cam_quat_b_np[1]:+.4f}, y={cam_quat_b_np[2]:+.4f}, z={cam_quat_b_np[3]:+.4f}"
            )
        else:
            print("[WARNING] Camera sensor not available for final pose summary.")

        print("=" * 80)
    except Exception as e:
        print(f"[WARNING] Could not print final pose summary: {e}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
