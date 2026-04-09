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
import pandas as pd
from tensordict import TensorDict
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
    timestep_data_list = []
    if SAVE_DATA:
        print(f"[INFO] Data saving enabled. Data will be saved as Excel file at episode end.")
    
    ### (6) VARIABLE INITIALIZATION ###
    obs = env.get_observations() 
    timestep = 0
    joint_names = [
        "Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"
    ]
    joint_names_obs = [
        "Yaw", "Pitch", "Insert", "Wrist_Roll", "Wrist_Pitch", "Wrist_Yaw", "Gripper1", "Gripper2"
    ]
    observation_data = [] # obs file
    
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
            else:
                actions_to_use = actions

            ### (12) EXECUTE ACTION IN ENVIRONMENT AND GET NEW STATE (SIMULATION STEP) ###
            obs, rewards, dones, extras = env.step(actions_to_use) 
            
            ### (13) EXTRACT AND PRINT OBSERVATIONS, ACTIONS, AND POSE INFORMATION ###
            # obs structure: 
            # - [0:8] = joint positions (6 arm joints + 2 gripper joints)
            # - [8:16] = joint velocities (6 arm joints + 2 gripper joints)
            # - [16:22] = last actions (6 arm joints)
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
            
            noisy_pos = obs_flat[:8]
            noisy_vel = obs_flat[8:16]
            obs_row = {'step': timestep}
            
            for j in range(8):
                name = joint_names_obs[j] if j < len(joint_names_obs) else f"joint_{j}" 
                obs_row[f'true_pos_{name}'] = true_pos[j] 
                obs_row[f'noisy_pos_{name}'] = noisy_pos[j] 
                obs_row[f'pos_error_{name}'] = true_pos[j] - noisy_pos[j]
                obs_row[f'true_vel_{name}'] = true_vel[j]
                obs_row[f'noisy_vel_{name}'] = noisy_vel[j]
                obs_row[f'vel_error_{name}'] = true_vel[j] - noisy_vel[j]
            observation_data.append(obs_row)


            ### (14) DONE FLAG CHECKING (for episode termination) ###
            done_flag = False
            if torch.is_tensor(dones):
                done_flag = bool(torch.any(dones))
            elif isinstance(dones, (np.ndarray, list, tuple)):
                done_flag = bool(np.any(dones))
            elif isinstance(dones, bool):
                done_flag = dones

            ### (15) ACTION LOGGING AND OBSERVATION PRINTING ###
            cmd_term = base_env.action_manager.get_term("arm_action")
            processed_actions = cmd_term.processed_actions[0].cpu().numpy()
            pos_after = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            real_move = pos_after - pos_before

            ### (16) POSE EXTRACTION AND ERROR COMPUTATION (ONLY IF NOT DONE) ###
            # Skip error calculation on the final step when done=True, because the environment 
            # has already reset the robot to initial position. Use the previously calculated errors instead.
            if not done_flag:
                # computation of EE and target poses, and calculation of distance and orientation errors
                robot_root_pos = robot_asset.data.root_state_w[:, :3]
                robot_root_quat = robot_asset.data.root_state_w[:, 3:7]
                
                # target liver pose
                target_pose_w = liver_target_pose_world(base_env)
                target_pos_w, target_quat_w = target_pose_w[:, :3], target_pose_w[:, 3:7]
                target_pos_b, target_quat_b = subtract_frame_transforms(
                    robot_root_pos, robot_root_quat, target_pos_w, target_quat_w
                )

                # ee pose 
                ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
                ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w
                )

                # convert all poses to numpy for printing and visualization
                t_pos_b = target_pos_b[0].cpu().numpy()
                t_quat_b = target_quat_b[0].cpu().numpy()
                e_pos_b = ee_pos_b[0].cpu().numpy()
                e_quat_b = ee_quat_b[0].cpu().numpy()

                # compute distance error (Euclidean distance between EE position and target position in robot reference frame)
                # and orientation error (angle between EE orientation and target orientation in robot reference frame)
                ee_pos_b_tensor = torch.tensor(e_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                ee_quat_b_tensor = torch.tensor(e_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                t_pos_b_tensor = torch.tensor(t_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                t_quat_b_tensor = torch.tensor(t_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
                
                dist_error = torch.norm(ee_pos_b_tensor - t_pos_b_tensor, dim=1)[0].cpu().item()
                orient_error = quat_error_magnitude(ee_quat_b_tensor, t_quat_b_tensor)[0].cpu().item()

                ### (17) REFERENCE FRAME VISUALIZATION BASED ON FLAG ###
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

                ### (18) PRINT INFO ABOUT POSES, ERRORS, AND ACTIONS FOR EACH SIMULATION STEP ###
                print("EE frame (robot RF):   X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                    e_pos_b[0], e_pos_b[1], e_pos_b[2], e_quat_b[0], e_quat_b[1], e_quat_b[2], e_quat_b[3]))
                print("TARGET pose (robot RF): X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                    t_pos_b[0], t_pos_b[1], t_pos_b[2], t_quat_b[0], t_quat_b[1], t_quat_b[2], t_quat_b[3]))
                print(f"DISTANCE ERROR: {dist_error:.6f} m | ORIENTATION ERROR: {orient_error:.6f} rad")
                
                # success criteria check (when dist_error < 3mm and orient_error < 0.3 rad, we consider the target reached)
                if dist_error < 0.003 and orient_error < 0.3:
                    print("\n" + "=" * 80)
                    print("TARGET REACHED!")
                    print("=" * 80)
                    print(f"DISTANCE ERROR: {dist_error:.6f} m (threshold: < 0.003)")
                    print(f"ORIENTATION ERROR: {orient_error:.6f} rad (threshold: < 0.3)")
                    print("=" * 80 + "\n")

                # print observations (joint_pos(8) + joint_vel(8) + actions(6) = 22)
                obs_size = len(obs_flat)
                joint_pos_obs = obs_flat[:8] 
                joint_vel_obs = obs_flat[8:16] 
                actions_obs = obs_flat[16:22]

                print(f"\nOBSERVATIONS (size={obs_size}):")
                print(f"  joint_pos_rel (8):    {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_pos_obs))}")
                print(f"  joint_vel_rel (8):    {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_vel_obs))}")
                print(f"  last_action (6):      {' | '.join(f'{i:8.5f}' for i in actions_obs)}")
                
                # print actions and pose info for each joint in a table format
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
                
                ### (19) SAVE TIMESTEP DATA TO LIST FOR EXCEL EXPORT AT EPISODE END ###
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



        ### (20) SAVE FINAL ERROR METRICS BEFORE TERMINATION CHECK ###
        final_dist_error = dist_error
        final_orient_error = orient_error
        final_timestep = timestep
        final_episode_reason = "UNKNOWN"

        ### (21) CHECK FOR EPISODE TERMINATION AND LOGGING ###
        done_flag = False
        if torch.is_tensor(dones):
            done_flag = bool(torch.any(dones))
        elif isinstance(dones, (np.ndarray, list, tuple)):
            done_flag = bool(np.any(dones))
        elif isinstance(dones, bool):
            done_flag = dones

        if done_flag:
            # determine the reason for episode termination
            if isinstance(extras, dict) and "log" in extras:
                log = extras["log"]
                time_out_value = log.get("Episode_Termination/time_out", 0)
                success_value = log.get("Episode_Termination/success", 0)
                if hasattr(time_out_value, 'item'):
                    time_out_value = time_out_value.item()
                if hasattr(success_value, 'item'):
                    success_value = success_value.item()
                
                if success_value == 1:
                    final_episode_reason = "ACHIEVED SUCCESS"
                elif time_out_value == 1:
                    final_episode_reason = "TIME OUT"
            
            print(f"[INFO] Episode terminated: {final_episode_reason}")
            # print initial joint positions at termination on one line
            pos_str = " ".join([f"{p:.5f}".replace(".", ",") for p in initial_pos])
            print(f"Initial position of the joints: {pos_str}")
            
            
            # save timestep data to Excel if enabled
            if SAVE_DATA and timestep_data_list:
                df_timesteps = pd.DataFrame(timestep_data_list)
                excel_filename = os.path.join(log_dir, f"episode_{timestep}_timesteps.xlsx")
                df_timesteps.to_excel(excel_filename, index=False)
                print(f"[INFO] Timestep data saved to: {excel_filename}")
            
            
            if observation_data: # obs file
                df_obs   = pd.DataFrame(observation_data)
                obs_path = os.path.join(log_dir, f"observations_{timestep}_data.xlsx")
                # df_obs.to_excel(obs_path, index=False)   # uncomment to enable
                # print(f"[INFO] Observation data saved to: {obs_path}")
            
            try:
                zero_vel = torch.zeros((1, 6), device=robot_asset.device)
                robot_asset.set_joint_velocity_target(zero_vel, joint_ids=slice(0, 6))
                robot_asset.write_data_to_sim()
                print("[INFO] Stopping robot (velocity = 0).")
            except Exception as e:
                print(f"[WARN] Unable to stop robot: {e}")
            print("[INFO] Stopping simulation.")
            timestep_data_list = [] 
            observation_data = []
            break
        else:
            timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break


    ### (22) FINAL SUMMARY PRINTING (using pre-termination values) ###
    try:
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Episode terminated: {final_episode_reason}")
        print(f"Total steps to completion: {final_timestep}")
        print(f"DISTANCE ERROR: {final_dist_error*1000:.4f} mm (threshold: 3.0 mm)")
        print(f"ORIENTATION ERROR: {final_orient_error*180/np.pi:.4f}° ({final_orient_error:.6f} rad, threshold: 0.3 rad)")
        print("=" * 80)
    except Exception as e:
        print(f"[WARN] Unable to compute final summary: {e}")
    
    ### (22) CLOSE ENVIRONMENT AND SIMULATION APP ###
    env.close()

### (23) ENTRY POINT CHECK TO RUN MAIN FUNCTION ###
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
