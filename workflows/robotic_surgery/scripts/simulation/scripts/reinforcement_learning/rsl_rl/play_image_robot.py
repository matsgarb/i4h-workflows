

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
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import liver_target_pose_world
from isaaclab.utils.math import quat_error_magnitude
import socket
import struct
import time
import pandas as pd
from collections import deque
import cv2


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
        obs = data[0] if isinstance(data, tuple) else data
        if isinstance(obs, TensorDict):
            td = obs.to(agent_cfg.device)
        elif isinstance(obs, dict):
            td = TensorDict(obs, batch_size=env.num_envs).to(agent_cfg.device)
        else:
            td = TensorDict({"policy": obs.to(agent_cfg.device)}, batch_size=env.num_envs)
        for key in list(td.keys()):
            if len(td[key].shape) == 4 and td[key].shape[-1] in [3, 4]:
                td[key] = td[key].permute(0, 3, 1, 2).float() / 255.0
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
 

    drive_props = getattr(robot_asset, "drive_properties", None)
    act_cfg = getattr(robot_asset, "cfg", None)
    actuators = getattr(act_cfg, "actuators", None) if act_cfg is not None else None

    # ROBOT_IP = "127.0.0.1" # localhost
    # ROBOT_IP = "10.168.129.217"
    # ROBOT_IP = "10.41.197.104"
    
    ############ NEW
    UDP_IP_RECEIVE_OBS = "0.0.0.0"
    PORT_RECEIVE_OBS = 5006
    
    # UDP_IP_ROBOT = "127.0.0.1" # localhost
    UDP_IP_ROBOT = "10.41.53.28" # new arm computer
    PORT_ROBOT = 5005
    
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind((UDP_IP_RECEIVE_OBS, PORT_RECEIVE_OBS))
    sock_recv.settimeout(0.1)  # timeout of 0.1 second for receiving data
    
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ACK_PORT = 5007
    # sock_ack = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    IMG_H, IMG_W = 84, 84
    IMG_CHANNELS = 3
    IMG_SIZE = IMG_H * IMG_W * IMG_CHANNELS  # 84672 bytes
    POLICY_H, POLICY_W = 168, 168
    N_STACK = 4
    SAVE_OBS = False  # set False to disable saving
    obs_save_dir = os.path.join(log_dir, "obs_images")
    if SAVE_OBS:
        os.makedirs(obs_save_dir, exist_ok=True)
        print(f"[INFO] Saving stacked observations to: {obs_save_dir}")
    frame_buffer = deque(maxlen=N_STACK)
    joint_names = [
        "Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"
    ]
    prev_pos = None
    initial_pos = None  # Store initial reset positions
    last_step_time = time.perf_counter()
    last_pos_after = None
    last_final_pos = None  # Store final joint positions from last step
    last_ee_pos_b = None  # Store EE position wrt robot RF from last step
    last_ee_quat_b = None  # Store EE quaternion wrt robot RF from last step
    last_target_pos_b = None  # Store target position wrt robot RF from last step
    last_target_quat_b = None  # Store target quaternion wrt robot RF from last step
    last_dist_error = None  # Store distance error from last step
    last_orient_error = None  # Store orientation error from last step
    
    # Buffer for Insertion MAE calculation (last 100 steps)
    insertion_real_moves = []  # real_move for Insertion joint (index 2)
    insertion_processed = []   # processed_actions for Insertion joint (index 2)

    print("\n" + "="*50)
    print(f"[READY] Starting policy playback. Listening for robot state on {ppo_runner}:{PORT_RECEIVE_OBS}...") # NEW
    print("="*50 + "\n")
    
    # ========== ACTION FILTERING SETUP ==========
    USE_ACTION_FILTER = True # LIFT
    action_filter_alpha = 0.3
    filtered_actions = None
    # ==========================================

    previous_actions = torch.zeros((1, 6), device=agent_cfg.device, dtype=torch.float32)
    MANUAL_RESET_POS = torch.tensor([[0.91613, -0.34843, 0.08892, 0.97344, 0.01925, 0.08353]], device=agent_cfg.device) # LIFT
    zero_vel = torch.zeros_like(MANUAL_RESET_POS)
    robot_asset.set_joint_position_target(MANUAL_RESET_POS, joint_ids=slice(0, 6))
    robot_asset.write_joint_state_to_sim(MANUAL_RESET_POS, zero_vel, joint_ids=slice(0, 6))
    robot_asset.write_data_to_sim()
    simulation_app.update()

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            try:
                data_obs, addr_obs = sock_recv.recvfrom(IMG_SIZE + 1024)
                if len(data_obs) == IMG_SIZE:
                    # frame: (84, 84, 3) uint8, 0-255, RGB
                    frame = np.frombuffer(data_obs, dtype=np.uint8).reshape(IMG_H, IMG_W, IMG_CHANNELS).copy()
                else:
                    print(f"[WARNING] Received unexpected data length: {len(data_obs)} bytes (expected {IMG_SIZE})")
                    continue
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[ERROR] Exception while receiving UDP data: {e}")
                continue
            if len(frame_buffer) == 0:
                for _ in range(N_STACK):
                    frame_buffer.append(frame)
            else:
                frame_buffer.append(frame)
            stacked = np.concatenate(list(frame_buffer), axis=-1)  # (84, 84, 12)
            stacked_norm = stacked.astype(np.float32) / 255.0
            stacked_tensor = torch.tensor(stacked_norm, device=base_env.device).permute(2, 0, 1).unsqueeze(0)  # (1, 12, 84, 84)
            stacked_tensor = torch.nn.functional.interpolate(
                stacked_tensor, size=(POLICY_H, POLICY_W), mode='bilinear', align_corners=False
            )  # (1, 12, 168, 168)

            if SAVE_OBS:
                frames_list = list(frame_buffer)  # [oldest, ..., newest]
                for fi in range(N_STACK):
                    ch_start = fi * 3
                    ch_end   = ch_start + 3
                    frame_168 = stacked_tensor[0, ch_start:ch_end, :, :]  # (3, 168, 168)
                    frame_168 = (frame_168.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)  # (168, 168, 3) uint8
                    fname = os.path.join(obs_save_dir, f"step{timestep:04d}_frame{fi}.png")
                    cv2.imwrite(fname, cv2.cvtColor(frame_168, cv2.COLOR_RGB2BGR))

            pos_before = np.zeros(6)
            if initial_pos is None:
                initial_pos = pos_before.copy()
                print("\n" + "="*80)
                print(f"INITIAL STATE (RESET POSITIONS FROM ROBOT)")
                print("="*80)
                header = f"{'JOINT':<15} | {'POSITION':>12}"
                print(header)
                print("-" * len(header))
                for i in range(6):
                    print(f"{joint_names[i]:<15} | {initial_pos[i]:>12.5f}")
                print("="*80 + "\n")

            obs_before = ensure_tensordict({"policy": stacked_tensor})
            actions = policy(obs_before)
            
            raw_policy = actions[0].cpu().numpy()

            # STEP 3: Execute action and get next state (s_{t+1})
            # Apply filter or use raw actions based on USE_ACTION_FILTER flag
            if USE_ACTION_FILTER:
                if filtered_actions is None:
                    filtered_actions = actions.clone()
                else:
                    filtered_actions = action_filter_alpha * actions + (1.0 - action_filter_alpha) * filtered_actions
                actions_to_use = filtered_actions
                raw_val = actions[0, 2].cpu().item()
                filtered_val = actions_to_use[0, 2].cpu().item()
                print(f"[FILTER DEBUG] Joint 2 (Insertion): RAW={raw_val:+.4f} → FILTERED={filtered_val:+.4f} (alpha={action_filter_alpha})")
            else:
                actions_to_use = actions
                raw_val = actions[0, 2].cpu().item()
                print(f"[NO FILTER] Joint 2 (Insertion): RAW={raw_val:+.4f} → USED AS-IS={raw_val:+.4f}")

            # STEP 3: Execute action and get next state (s_{t+1})
            previous_actions = actions_to_use.clone()
            obs, rewards, dones, extras = env.step(actions_to_use)

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
            
            # compute done flag early
            done_flag = False
            if torch.is_tensor(dones):
                done_flag = bool(torch.any(dones))
            elif isinstance(dones, (np.ndarray, list, tuple)):
                done_flag = bool(np.any(dones))
            elif isinstance(dones, bool):
                done_flag = dones

            # timing
            now = time.perf_counter()
            dt = now - last_step_time
            last_step_time = now
            print(f"STEP DT: {dt*1000:.2f} ms ({1.0/dt if dt > 0 else float('inf'):.1f} Hz)")

            # # ######## ACTION LOGGING ########
            # (1) JOINT SPACE
            cmd_term = base_env.action_manager.get_term("arm_action")
            processed_actions = cmd_term.processed_actions[0].cpu().numpy()

            pos_after = np.zeros(6)
            real_move = np.zeros(6)

            if not done_flag:
                last_pos_after = pos_after.copy()
                target_raw = actions_to_use[0].cpu().numpy()[:6]
                data = struct.pack('6f', *target_raw)
                sock_send.sendto(data, (UDP_IP_ROBOT, PORT_ROBOT)) # NEW
                
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
            
            # Compute and print distance and orientation errors directly from robot RF poses
            ee_pos_b_tensor = torch.tensor(e_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            ee_quat_b_tensor = torch.tensor(e_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            t_pos_b_tensor = torch.tensor(t_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            t_quat_b_tensor = torch.tensor(t_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            
            dist_error = torch.norm(ee_pos_b_tensor - t_pos_b_tensor, dim=1)[0].cpu().item()
            orient_error = quat_error_magnitude(ee_quat_b_tensor, t_quat_b_tensor)[0].cpu().item()
            
            pos_after = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            real_move = pos_after - pos_before
            
            # Save final position and frame data only if episode is NOT ending (to avoid reset positions)
            if not done_flag:
                last_final_pos = pos_after.copy()
                last_ee_pos_b = e_pos_b.copy()
                last_ee_quat_b = e_quat_b.copy()
                last_target_pos_b = t_pos_b.copy()
                last_target_quat_b = t_quat_b.copy()
                last_dist_error = dist_error
                last_orient_error = orient_error

            print(f"OBSERVATIONS: stacked image tensor shape={tuple(stacked_tensor.shape)} (1, 12, 168, 168)")

            print(f"\nSTEP: {timestep:04d}")
            
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
        
        # sock_ack.sendto(b"OK", (ROBOT_IP, ACK_PORT)) # NEW
        # sock_ack.sendto(b"OK", (LOCAL_IP, ACK_PORT)) 

        done_flag = False
        if torch.is_tensor(dones):
            done_flag = bool(torch.any(dones))
        elif isinstance(dones, (np.ndarray, list, tuple)):
            done_flag = bool(np.any(dones))
        elif isinstance(dones, bool):
            done_flag = dones

        if done_flag:
            # Print MAE for Insertion if episode ended before 100 steps
            if len(insertion_real_moves) > 0:
                mae = np.mean(np.abs(np.array(insertion_real_moves) - np.array(insertion_processed)))
                print(f"[MAE] Insertion (episode total, {len(insertion_real_moves)} steps): {mae:.6f}")
            
            print("\n" + "="*80)
            print("EPISODE TERMINATED - SUCCESS")
            print("="*80)
            
            # Print initial positions
            if initial_pos is not None:
                header = f"{'JOINT':<15} | {'INITIAL POSITION':>15}"
                print(header)
                print("-" * len(header))
                for i in range(6):
                    print(f"{joint_names[i]:<15} | {initial_pos[i]:>15.5f}")
                print("-" * len(header))
            
            # Use final positions from last step
            if last_final_pos is not None:
                header = f"{'JOINT':<15} | {'FINAL POSITION':>15}"
                print(header)
                print("-" * len(header))
                for i in range(6):
                    print(f"{joint_names[i]:<15} | {last_final_pos[i]:>15.5f}")
                print("-" * len(header))
            
            # Print EE frame wrt robot RF
            if last_ee_pos_b is not None:
                print(f"\nEE FRAME (robot RF):")
                print(f"  Position: X={last_ee_pos_b[0]:>10.6f}, Y={last_ee_pos_b[1]:>10.6f}, Z={last_ee_pos_b[2]:>10.6f} (m)")
                print(f"  Quat:     w={last_ee_quat_b[0]:>10.6f}, x={last_ee_quat_b[1]:>10.6f}, y={last_ee_quat_b[2]:>10.6f}, z={last_ee_quat_b[3]:>10.6f}")
            
            # Print target frame wrt robot RF
            if last_target_pos_b is not None:
                print(f"\nTARGET FRAME (robot RF):")
                print(f"  Position: X={last_target_pos_b[0]:>10.6f}, Y={last_target_pos_b[1]:>10.6f}, Z={last_target_pos_b[2]:>10.6f} (m)")
                print(f"  Quat:     w={last_target_quat_b[0]:>10.6f}, x={last_target_quat_b[1]:>10.6f}, y={last_target_quat_b[2]:>10.6f}, z={last_target_quat_b[3]:>10.6f}")
            
            # Print errors
            if last_dist_error is not None:
                print(f"\nERROR METRICS:")
                print(f"  Distance Error:    {last_dist_error*1000:>10.4f} mm (threshold: 3.0 mm)")
                print(f"  Orientation Error: {last_orient_error*180/np.pi:>10.4f}° (threshold: 11.46°)")
                print(f"  Orientation Error: {last_orient_error:>10.6f} rad (threshold: 0.2 rad)")
            print("="*80 + "\n")
            
            # Stop the robot by setting velocity targets to zero
            try:
                zero_vel = torch.zeros((1, 6), device=robot_asset.device)
                robot_asset.set_joint_velocity_target(zero_vel, joint_ids=slice(0, 6))
                robot_asset.write_data_to_sim()
                print("[INFO] Target reached; stopping robot (velocity = 0).")
            except Exception as e:
                print(f"[WARN] Unable to stop robot: {e}")
            print("[INFO] Stopping simulation.")
            
            # Reset buffers for next episode
            insertion_real_moves = []
            insertion_processed = []
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