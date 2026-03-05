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
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import numpy as np
import gymnasium as gym
import robotic.surgery.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


from rsl_rl.runners import OnPolicyRunner
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import liver_target_pose_world, gallbladder_pixel_count, gallbladder_visibility_success, visual_exposure_reward
from isaaclab.utils.math import quat_error_magnitude
import socket
import struct
import time
import pandas as pd



def main():
    """Play with RSL-RL agent."""
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
   
    ########## NEW RSL_RL STATE BASED RL ##########
    from tensordict import TensorDict
    def ensure_tensordict(data):
        # Estrae i dati se è una tupla (obs, extras)
        obs = data[0] if isinstance(data, tuple) else data
        if isinstance(obs, TensorDict): return obs
        if isinstance(obs, dict): return TensorDict(obs, batch_size=env.num_envs)
        # Impacchetta il tensore sotto la chiave "policy"
        return TensorDict({"policy": obs.to(agent_cfg.device)}, batch_size=env.num_envs)

    original_get_obs = env.get_observations
    original_step = env.step

    env.get_observations = lambda: ensure_tensordict(original_get_obs())
    
    def patched_step(actions):
        obs, rewards, dones, extras = original_step(actions)
        return ensure_tensordict(obs), rewards, dones, extras
    
    env.step = patched_step
   

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    ########## NEW RSL_RL STATE BASED RL ##########
    obs_normalizer = getattr(ppo_runner.alg, "obs_normalizer", None)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    try:
        export_policy_as_jit(ppo_runner.alg.policy, obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(ppo_runner.alg.policy, normalizer=obs_normalizer, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Policy exported successfully to: {export_model_dir}")
    except Exception as e:
        print(f"[WARNING] Error during export (playback will continue): {e}")
    ##############################################

    obs = env.get_observations() 
    timestep = 0

    base_env = getattr(env, "unwrapped", env)
    robot_asset = base_env.scene["robot"]
    ee_frame = base_env.scene["ee_frame"]
 
    # # PRINT ACTUATOR CONFIG FOR DEBUGGING
    # drive_props = getattr(robot_asset, "drive_properties", None)
    # act_cfg = getattr(robot_asset, "cfg", None)
    # actuators = getattr(act_cfg, "actuators", None) if act_cfg is not None else None
    # for name, cfg in actuators.items():
    #     stiff = getattr(cfg, "stiffness", None)
    #     damp = getattr(cfg, "damping", None)
    #     print(f"  {name}: stiffness={stiff}, damping={damp}")

    # ROBOT_IP = "127.0.0.1" # localhost
    # ROBOT_IP = "10.168.129.217"
    # ROBOT_IP = "10.41.197.104"
    ROBOT_IP = "10.41.50.246" # new arm computer
    ROBOT_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   
    
    joint_names = [
        "Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"
    ]
    prev_pos = None
    last_step_time = time.perf_counter()
    last_pos_after = None
    
    # Buffer for Insertion MAE calculation (last 100 steps)
    insertion_real_moves = []  # real_move for Insertion joint (index 2)
    insertion_processed = []   # processed_actions for Insertion joint (index 2)
    
    # Data logging for Excel export
    episode_data = []  # List to accumulate data for current episode
    observation_data = []
    
    # Print initial robot state (STEP 0)
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
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            print("\n" + "="*80)
            print(f"STEP {timestep:04d}")
            print("="*80)
            
            # STEP 1: Capture state BEFORE action execution (s_t, pos_t)
            obs_before = obs  # Observation at current timestep (s_t)
            pos_before = robot_asset.data.joint_pos[0][:6].cpu().numpy()  # Position at current timestep
            true_pos = robot_asset.data.joint_pos[0].cpu().numpy() # obs file
            true_vel = robot_asset.data.joint_vel[0].cpu().numpy() # obs file
            
            # STEP 2: Get action from policy based on current observation
            actions = policy(obs_before)
            raw_policy = actions[0].cpu().numpy()
            
            # # Keep robot still - override actions with zeros
            # actions = torch.zeros_like(actions)

            # STEP 3: Execute action and get next state (s_{t+1})
            obs, rewards, dones, extras = env.step(actions) 

            # Get camera sensor and compute gallbladder visibility
            try:
                camera = env.unwrapped.scene.sensors["camera"]
                if camera is not None:
                    cam_out = camera.data.output
                    rgb_tensor = cam_out.get("rgb", None)
                    if rgb_tensor is not None:
                        img_rgb = rgb_tensor[0, :, :, :3].cpu().numpy()
                        visible_pixels = gallbladder_pixel_count(img_rgb)
                        
                        # Check success condition
                        success_status = gallbladder_visibility_success(env.unwrapped, pixel_threshold=2500)
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
                        print(f"[GALLBLADDER] Visible pixels: {visible_pixels} | Threshold: 2500 | {success_text} | visibility reward: {vis_reward_value:.4f}")
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
            joint_names_obs = ["Yaw", "Pitch", "Insert", "Wrist_Roll", "Wrist_Pitch", "Wrist_Yaw", "Gripper1", "Gripper2"]
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

            # data = struct.pack('6f', *processed_actions)
            # sock.sendto(data, (ROBOT_IP, ROBOT_PORT))

            pos_after = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            real_move = pos_after - pos_before

            if not done_flag:
                last_pos_after = pos_after.copy()
                # Send UDP only if not done
                # data = struct.pack('6f', *real_move)
                target_jp = processed_actions[:6] 
                data = struct.pack('6f', *target_jp)
                sock.sendto(data, (ROBOT_IP, ROBOT_PORT))
                
                # Track Insertion joint (index 2) for MAE calculation
                insertion_real_moves.append(real_move[2])
                insertion_processed.append(processed_actions[2])
                
                # Keep only last 100 steps
                if len(insertion_real_moves) > 100:
                    insertion_real_moves.pop(0)
                    insertion_processed.pop(0)
                
                # Every 100 steps, print MAE
                if timestep > 0 and timestep % 100 == 0:
                    mae = np.mean(np.abs(np.array(insertion_real_moves) - np.array(insertion_processed)))
                    print(f"[MAE] Insertion (last 100 steps): {mae:.6f}")
            
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
            # print("ROBOT frame (world):   X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
            #     r_pos_w[0], r_pos_w[1], r_pos_w[2], r_quat_w[0], r_quat_w[1], r_quat_w[2], r_quat_w[3]))
            # print("EE frame (world):      X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
            #     e_pos_w[0], e_pos_w[1], e_pos_w[2], e_quat_w[0], e_quat_w[1], e_quat_w[2], e_quat_w[3]))
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
            print(f"DISTANCE ERROR: {dist_error:.6f} m | ORIENTATION ERROR: {orient_error:.6f} rad")
            
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
            
            # # PRINT OBSERVATIONS
            # print(f"OBSERVATIONS (size={obs_size}, struct: pos[0:8] vel[8:16] target[{target_pose_start}:{target_pose_start+7}] actions[{target_pose_start+7}:{obs_size}]):")
            # print(f"  joint_pos (rel, 8):   {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_pos_obs))}")
            # print(f"  joint_vel (rel, 8):   {' | '.join(f'{n:12s}: {x:8.5f}' for n, x in zip(joint_names_obs, joint_vel_obs))}")
            # if len(target_pose_obs) == 7:
            #     print(f"  target_pose (world, 7): pos=[{target_pose_obs[0]:.5f}, {target_pose_obs[1]:.5f}, {target_pose_obs[2]:.5f}], quat=[{target_pose_obs[3]:.5f}, {target_pose_obs[4]:.5f}, {target_pose_obs[5]:.5f}, {target_pose_obs[6]:.5f}]")
            # print(f"  actions (last, 6):    {' | '.join(f'{i:8.5f}' for i in actions_obs)}")
            

            # # Collect data for Excel export (only arm joints 0-5, not gripper)
            # row_data = {
            #     'step': timestep,
            # }
            # # obs_joint_pos (only first 6: arm)
            # for j in range(6):
            #     row_data[f'obs_joint_pos_j{j}'] = joint_pos_obs[j]
            # # obs_joint_vel (only first 6: arm)
            # for j in range(6):
            #     row_data[f'obs_joint_vel_j{j}'] = joint_vel_obs[j]
            # # obs_last_action
            # for j in range(6):
            #     row_data[f'obs_last_action_j{j}'] = actions_obs[j] if j < len(actions_obs) else 0.0
            # # raw_policy
            # for j in range(6):
            #     row_data[f'raw_policy_j{j}'] = raw_policy[j]
            # # processed_actions
            # for j in range(6):
            #     row_data[f'processed_action_j{j}'] = processed_actions[j]
            # # real_move
            # for j in range(6):
            #     row_data[f'real_move_j{j}'] = real_move[j]
            # # position (current)
            # for j in range(6):
            #     row_data[f'position_j{j}'] = pos_after[j]
            
            # episode_data.append(row_data)
            
            
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
            
            # # Print MAE for Insertion if episode ended before 100 steps
            # if len(insertion_real_moves) > 0:
            #     mae = np.mean(np.abs(np.array(insertion_real_moves) - np.array(insertion_processed)))
            #     print(f"[MAE] Insertion (episode total, {len(insertion_real_moves)} steps): {mae:.6f}")
            
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
            
            # Reset buffers for next episode
            insertion_real_moves = []
            insertion_processed = []
            episode_data = []  # Reset data for next episode
            observation_data = [] # obs file
            # timestep = 0
            break
        else:
            timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
