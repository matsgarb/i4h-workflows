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
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import liver_target_pose_world
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
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
    
    # Container for EE marker
    marker_container = {"ee_marker": None}

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
   
    # ########## NEW RSL_RL STATE BASED RL ##########
    # from tensordict import TensorDict
    # def ensure_tensordict(data):
    #     # Estrae i dati se è una tupla (obs, extras)
    #     obs = data[0] if isinstance(data, tuple) else data
    #     if isinstance(obs, TensorDict): return obs
    #     if isinstance(obs, dict): return TensorDict(obs, batch_size=env.num_envs)
    #     # Impacchetta il tensore sotto la chiave "policy"
    #     return TensorDict({"policy": obs.to(agent_cfg.device)}, batch_size=env.num_envs)

    # original_get_obs = env.get_observations
    # original_step = env.step

    # # Sovrascriviamo i metodi dell'ambiente
    # env.get_observations = lambda: ensure_tensordict(original_get_obs())
    
    # def patched_step(actions):
    #     obs, rewards, dones, extras = original_step(actions)
    #     return ensure_tensordict(obs), rewards, dones, extras
    
    # env.step = patched_step
    # ###################################################################
    
    ########## TENSORDICT WRAPPER - SUPPORTS ALL RL TYPES ##########
    from tensordict import TensorDict
    def ensure_tensordict(data):
        obs = data[0] if isinstance(data, tuple) else data
        if isinstance(obs, TensorDict):
            td = obs.to(agent_cfg.device)
        elif isinstance(obs, dict):
            td = TensorDict(obs, batch_size=env.num_envs).to(agent_cfg.device)
        else:
            td = TensorDict({"policy": obs.to(agent_cfg.device)}, batch_size=env.num_envs)
        
        # Process image tensors: permute NHWC -> NCHW and normalize to [0, 1]
        for key in list(td.keys()):
            if len(td[key].shape) == 4 and td[key].shape[-1] in [3, 4]:
                # This is an image tensor: (B, H, W, C) -> (B, C, H, W)
                td[key] = td[key].permute(0, 3, 1, 2).float() / 255.0
        
        # Ensure dummy_state exists for compatibility
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

    #################################################

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ########## OLD RSL_RL STATE BASED RL ##########
    # # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(
    #     ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )
    # ##############################################

    # ########## NEW RSL_RL STATE BASED RL ##########
    # obs_normalizer = getattr(ppo_runner.alg, "obs_normalizer", None)
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # os.makedirs(export_model_dir, exist_ok=True)

    # try:
    #     export_policy_as_jit(ppo_runner.alg.policy, obs_normalizer, path=export_model_dir, filename="policy.pt")
    #     export_policy_as_onnx(ppo_runner.alg.policy, normalizer=obs_normalizer, path=export_model_dir, filename="policy.onnx")
    #     print(f"[INFO] Policy exported successfully to: {export_model_dir}")
    # except Exception as e:
    #     print(f"[WARNING] Error during export (playback will continue): {e}")
    # ##############################################

    # reset environment
    # obs, _ = env.get_observations() # OLD RSL_RL
    obs = env.get_observations() # NEW RSL_RL
    timestep = 0

    # # ######## ACTION LOGGING ########
    # (1) JOINT SPACE
    # robot_asset = env.unwrapped.scene["robot"]
    base_env = getattr(env, "unwrapped", env)
    robot_asset = base_env.scene["robot"]
    ee_frame = base_env.scene["ee_frame"]
 
    try:
        drive_props = getattr(robot_asset, "drive_properties", None)
        if drive_props is None:
            getter = getattr(robot_asset.root_physx_view, "get_drive_properties", None)
            if getter is not None:
                drive_props = getter()

        if drive_props is not None:
            if isinstance(drive_props, dict):
                stiffness = drive_props.get("stiffness")
                damping = drive_props.get("damping")
            elif isinstance(drive_props, (list, tuple)) and len(drive_props) > 0 and isinstance(drive_props[0], dict):
                stiffness = drive_props[0].get("stiffness")
                damping = drive_props[0].get("damping")
            else:
                stiffness = getattr(drive_props, "stiffness", None)
                damping = getattr(drive_props, "damping", None)
            print(f"Stiffness: {stiffness}")
            print(f"Damping:   {damping}")
        else:
            act_cfg = getattr(robot_asset, "cfg", None)
            actuators = getattr(act_cfg, "actuators", None) if act_cfg is not None else None
            if actuators:
                print("[WARN] drive properties unavailable; reporting actuator cfg values instead:")
                for name, cfg in actuators.items():
                    stiff = getattr(cfg, "stiffness", None)
                    damp = getattr(cfg, "damping", None)
                    print(f"  {name}: stiffness={stiff}, damping={damp}")
            else:
                print("[WARN] drive properties unavailable on robot asset")
    except Exception as e:
        print(f"[WARN] Unable to read drive stiffness/damping: {e}")
   
   
    joint_names = [
        "Yaw", "Pitch", "Insertion", "Wrist Roll", "Wrist Pitch", "Wrist Yaw"
    ]
    prev_pos = None
    
    # Low-pass filter for smooth action trajectories 
    action_filter_alpha = 0.2
    filtered_actions = None
    
    # ===== FILTER TOGGLE: Set to True to enable, False to disable =====
    USE_ACTION_FILTER = False
    # ====================================================================
    
    # Variables to save penultimate step data
    last_ee_pos_b = None
    last_ee_quat_b = None
    last_dist_error = None
    last_orient_error = None
    last_liver_offset = None
    last_liver_pos = None
    
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
    print("="*80 + "\n")

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            print("\n" + "="*80)
            print(f"STEP {timestep:04d}")
            print("="*80)
            
            pos_before = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            # # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            
            # Apply filter or use raw actions based on USE_ACTION_FILTER flag
            if USE_ACTION_FILTER:
                # Low-pass filter for smooth trajectories
                if filtered_actions is None:
                    filtered_actions = actions.clone()
                else:
                    filtered_actions = action_filter_alpha * actions + (1.0 - action_filter_alpha) * filtered_actions
                actions_to_use = filtered_actions
                
                # DEBUG: Show the difference
                raw_val = actions[0, 2].cpu().item()  # Joint 2 (Insertion)
                filtered_val = actions_to_use[0, 2].cpu().item()
                print(f"[FILTER DEBUG] Joint 2 (Insertion): RAW={raw_val:+.4f} → FILTERED={filtered_val:+.4f} (alpha={action_filter_alpha})")
            else:
                # Raw actions without filtering
                actions_to_use = actions
                raw_val = actions[0, 2].cpu().item()
                print(f"[NO FILTER] Joint 2 (Insertion): RAW={raw_val:+.4f} → USED AS-IS={raw_val:+.4f}")

            # # ######## ACTION LOGGING ########
            # # (full mapping: policy -> clipped -> affine -> clip -> target)
            # (1) JOINT SPACE
            raw_policy = actions[0].cpu().numpy()

            # all_joint_pos = robot_asset.data.joint_pos[0]
            # current_pos = all_joint_pos[:6].cpu().numpy()
            # if prev_pos is None:
            #     prev_pos = current_pos.copy()
            # obs_debug = obs["policy"][0].cpu().numpy()
            # print(f"\n[BRAIN CHECK] Step {timestep:04d}")
            # print(f"  COSA VEDE (Obs primi 6 val): {obs_debug[:6]}")
            # actions = policy(obs) 
            # raw_action_debug = actions[0].cpu().numpy()
            # print(f"  COSA PENSA (Raw Action J2) : {raw_action_debug[2]:.5f}")
            # print(f"  REALTA' FISICA (Pos J2)    : {current_pos[2]:.5f}")
            # if raw_action_debug[2] > 0.9 and current_pos[2] > 0.09:
            #      print("  >>> ALLARME LOGICO: Il robot è esteso (>0.09) ma la policy spinge ancora al massimo (+1.0)!")
            #      print("      CAUSA PROBABILE: L'osservazione (obs) dice al robot che è ancora indietro.")
            # cmd = base_env.action_manager.get_term("arm_action")
            # policy_raw = actions[0].cpu().numpy()
            # policy_clipped = torch.clamp(actions, -1.0, 1.0)[0].cpu().numpy()
            # scale_np = cmd._scale[0].cpu().numpy() if isinstance(cmd._scale, torch.Tensor) else np.array([cmd._scale] * 6)
            # offset_np = cmd._offset[0].cpu().numpy() if isinstance(cmd._offset, torch.Tensor) else np.array([cmd._offset] * 6)
            # affine_pre_clip = policy_clipped * scale_np + offset_np
            # affine_final = np.clip(affine_pre_clip, cmd._clip[0,:,0].cpu().numpy(), cmd._clip[0,:,1].cpu().numpy()) if cmd.cfg.clip is not None else affine_pre_clip
            
            # # (2) CARTESIAN SPACE
            # ee_pos_curr = ee_frame.data.target_pos_w[0, 0].detach().cpu().numpy()
            # ee_quat_curr = ee_frame.data.target_quat_w[0, 0].detach().cpu().numpy()
            # obs_debug = obs["policy"][0].cpu().numpy()
            # raw_action = actions[0].cpu().numpy()
            # scale = 0.02
            # delta_cmd = raw_action[:3] * scale
            # ee_pos_cmd = ee_pos_curr + delta_cmd
            # rot_cmd = raw_action[3:6] if raw_action.shape[0] >= 6 else None
            # rot_scale = None
            # if rot_cmd is not None:
            #     if isinstance(cmd_term._scale, torch.Tensor):
            #         rot_scale = cmd_term._scale[0, 3:6].detach().cpu().numpy()
            #     else:
            #         rot_scale = np.array(cmd_term._scale)[3:6]
            #     delta_rot = rot_cmd * rot_scale
            # target_pose = liver_target_pose_world(base_env)
            # target_pos = target_pose[0, :3].detach().cpu().numpy()
            # target_quat = target_pose[0, 3:7].detach().cpu().numpy()
            # dist_pre = np.linalg.norm(ee_pos_curr - target_pos)
            # print(f"\n[IK RELATIVE CHECK] Step {timestep:04d}")
            # print(f"  OBSERVATIONS [:6]: {obs_debug[:6]}")
            # print(f"  RAW POLICY  : dX={raw_action[0]:+.4f}, dY={raw_action[1]:+.4f}, dZ={raw_action[2]:+.4f}, dRX={rot_cmd[0]:+.4f}, dRY={rot_cmd[1]:+.4f}, dRZ={rot_cmd[2]:+.4f}")
            # print(f"  SCALED POLICY (RAW x 0.02)    : dX={delta_cmd[0]:+.4f}, dY={delta_cmd[1]:+.4f}, dZ={delta_cmd[2]:+.4f}, dRX={delta_rot[0]:+.4f}, dRY={delta_rot[1]:+.4f}, dRZ={delta_rot[2]:+.4f}")
            # print(f"  EE before the step    : X={ee_pos_curr[0]:.4f}, Y={ee_pos_curr[1]:.4f}, Z={ee_pos_curr[2]:.4f},w={ee_quat_curr[0]:.4f}, x={ee_quat_curr[1]:.4f}, y={ee_quat_curr[2]:.4f}, z={ee_quat_curr[3]:.4f}")
            # print(f"  EE target (EE + scaled policy)    : X={ee_pos_cmd[0]:.4f}, Y={ee_pos_cmd[1]:.4f}, Z={ee_pos_cmd[2]:.4f}")
            # print(f"  TARGET (liver_target_pose_world): X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}, w={target_quat[0]:.4f}, x={target_quat[1]:.4f}, y={target_quat[2]:.4f}, z={target_quat[3]:.4f}")
            # print(f"    Dist EE->target before the step: {dist_pre:.4f} m")
            # ########################################

            # step env
            
            # env stepping
            obs, rewards, dones, extras = env.step(actions_to_use) # NEW RSL_RL - use filtered actions
            
            
            # ######## ACTION LOGGING ########
            # # (2) CARTESIAN SPACE
            # ee_pos_next = ee_frame.data.target_pos_w[0, 0].detach().cpu().numpy()
            # ee_quat_next = ee_frame.data.target_quat_w[0, 0].detach().cpu().numpy()
            # dist_post = np.linalg.norm(ee_pos_next - target_pos)
            # ee_move = ee_pos_next - ee_pos_curr
            # print(f"  EE after the step          : X={ee_pos_next[0]:.4f}, Y={ee_pos_next[1]:.4f}, Z={ee_pos_next[2]:.4f},w={ee_quat_next[0]:.4f}, x={ee_quat_next[1]:.4f}, y={ee_quat_next[2]:.4f}, z={ee_quat_next[3]:.4f}")
            # print(f"  REAL MOVEMENT: dX={ee_move[0]:+.4f}, dY={ee_move[1]:+.4f}, dZ={ee_move[2]:+.4f}")
            # print(f"  Dist EE->target after  : {dist_post:.4f} m (before {dist_pre:.4f})")
            # ########################################
            
            # obs, _, _, _ = env.step(actions) # OLD RSL_RL

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
            # print("ROBOT frame (world):   X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
            #     r_pos_w[0], r_pos_w[1], r_pos_w[2], r_quat_w[0], r_quat_w[1], r_quat_w[2], r_quat_w[3]))
            # print("EE frame (world):      X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
            #     e_pos_w[0], e_pos_w[1], e_pos_w[2], e_quat_w[0], e_quat_w[1], e_quat_w[2], e_quat_w[3]))
            print("EE frame (robot RF):   X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                e_pos_b[0], e_pos_b[1], e_pos_b[2], e_quat_b[0], e_quat_b[1], e_quat_b[2], e_quat_b[3]))
            
            # # Draw small frame at EE position
            # if marker_container["ee_marker"] is None:
            #     ee_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame")
            #     # Scale down the frame marker to be very small
            #     ee_marker_cfg.markers["frame"].scale = (0.003, 0.003, 0.003)
            #     marker_container["ee_marker"] = VisualizationMarkers(ee_marker_cfg)
            
            # # Visualize the EE frame
            # ee_pos_w_tensor = torch.tensor(e_pos_w, dtype=torch.float32, device=base_env.device).unsqueeze(0)
            # ee_quat_w_tensor = torch.tensor(e_quat_w, dtype=torch.float32, device=base_env.device).unsqueeze(0)
            # marker_indices = torch.zeros(1, dtype=torch.int32, device=base_env.device)
            # marker_container["ee_marker"].visualize(ee_pos_w_tensor, ee_quat_w_tensor, marker_indices=marker_indices)
            
            print("TARGET pose (robot RF): X={:.4f}, Y={:.4f}, Z={:.4f}, w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}".format(
                t_pos_b[0], t_pos_b[1], t_pos_b[2], t_quat_b[0], t_quat_b[1], t_quat_b[2], t_quat_b[3]))
            
            # Compute and print distance and orientation errors
            ee_pos_b_tensor = torch.tensor(e_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            ee_quat_b_tensor = torch.tensor(e_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            t_pos_b_tensor = torch.tensor(t_pos_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            t_quat_b_tensor = torch.tensor(t_quat_b, device=base_env.device, dtype=torch.float32).unsqueeze(0)
            
            dist_error = torch.norm(ee_pos_b_tensor - t_pos_b_tensor, dim=1)[0].cpu().item()
            orient_error = quat_error_magnitude(ee_quat_b_tensor, t_quat_b_tensor)[0].cpu().item()
            print(f"DISTANCE ERROR: {dist_error:.6f} m | ORIENTATION ERROR: {orient_error:.6f} rad")
            
            # Save penultimate step data (only if episode is NOT ending)
            done_flag = False
            if torch.is_tensor(dones):
                done_flag = bool(torch.any(dones))
            elif isinstance(dones, (np.ndarray, list, tuple)):
                done_flag = bool(np.any(dones))
            elif isinstance(dones, bool):
                done_flag = dones
            
            if not done_flag:
                last_ee_pos_b = e_pos_b.copy()
                last_ee_quat_b = e_quat_b.copy()
                last_dist_error = dist_error
                last_orient_error = orient_error
                # Get liver offset from environment config
                try:
                    liver_cfg = base_env.cfg.object.liver
                    if hasattr(liver_cfg, 'init_state'):
                        init_pos = liver_cfg.init_state.pos
                        last_liver_offset = init_pos
                except:
                    last_liver_offset = None
                # Get liver current position
                try:
                    liver_asset = base_env.scene["liver"]
                    liver_pos = liver_asset.data.root_state_w[0, :3].cpu().numpy()
                    last_liver_pos = liver_pos
                except:
                    last_liver_pos = None
            
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



            # all_joint_pos_after = robot_asset.data.joint_pos[0]
            # current_pos_after = all_joint_pos_after[:6].cpu().numpy()
            # cmd_raw = cmd.raw_actions[0].cpu().numpy()
            # cmd_processed = cmd.processed_actions[0].cpu().numpy()
            # # Parametri Fisici Costanti
            # dt_policy = (1.0 / 60.0) * 2  # 0.0333s
            # v_limit = 0.1                 # velocity_limit (rad/s o m/s)
            # max_phys_step = v_limit * dt_policy # 0.003333
            # # Calcoli Delta
            # delta_needed = cmd_processed - prev_pos
            # delta_applied = current_pos_after - prev_pos
            # print("\n" + "=" * 100)
            # print(f"STEP: {timestep:04d} | LIMIT FISICO VELOCITÀ: +/- {max_phys_step:.6f} per step")
            # # Analizziamo SOLO i primi 3 giunti per chiarezza (Yaw, Pitch, Insertion)
            # joint_labels = ["J0 (Yaw - Rot)", "J1 (Pitch - Rot)", "J2 (Insert - Lin)"]
            # for i in [2]:# range(3):
            #     name = joint_labels[i]
            #     raw = policy_raw[i]
            #     clipped = policy_clipped[i]
            #     s = scale_np[i]
            #     o = offset_np[i]
            #     target = cmd_processed[i]
            #     p_prev = prev_pos[i]
            #     p_curr = current_pos_after[i]
            #     d_need = delta_needed[i]
            #     d_real = delta_applied[i]
            #     if abs(d_need) > max_phys_step:
            #         theoretical_move = max_phys_step * np.sign(d_need)
            #         limit_hit = True
            #     else:
            #         theoretical_move = d_need
            #         limit_hit = False
            #     efficiency = (d_real / theoretical_move) * 100 if abs(theoretical_move) > 1e-6 else 100.0
            #     print("-" * 100)
            #     print(f" {name.upper()}")
            #     delta_cmd = clipped * s
            #     target_est = p_prev + delta_cmd + o
            #     print(f"  1. POLICY : {raw:8.5f} (Raw) -> Clip[-1,1] -> {clipped:8.5f}")
            #     print(f"  2. FORMULA: prev {p_prev:8.5f} + ({clipped:8.5f} * {s:.3f}) + {o:.5f} (Offset) = {target_est:8.5f} (Atteso) | cmd.processed={target:8.5f}")
            #     if cmd.cfg.clip is not None:
            #         clip_lo = cmd._clip[0, i, 0].item()
            #         clip_hi = cmd._clip[0, i, 1].item()
            #         print(f"  2b. CLIP   : pre={affine_pre_clip[i]:+.5f} -> clip[{clip_lo:+.3f},{clip_hi:+.3f}] -> {affine_final[i]:+.5f}")
            #     print(f"  3. SPAZIO : {p_prev:8.5f} (Prev) ....................... -> {target:8.5f} (Target)")
            #     print(f"  4. DELTA  : Serve uno spostamento di ................... : {d_need:8.5f}")
            #     print(f"  5. LIMITE : Il motore consente max ..................... : {max_phys_step:8.5f}")
            #     if limit_hit:
            #         msg_limit = f"SATURATO (Richiesto {abs(d_need):.4f} > Limit {max_phys_step:.4f})"
            #     else:
            #         msg_limit = "NEL RANGE (Velocità OK)"
            #     print(f"  6. TEORIA : Dovrebbe muoversi di ....................... : {theoretical_move:8.5f} [{msg_limit}]")
            #     print(f"  7. REALTÀ : Si è mosso effettivamente di ............... : {d_real:8.5f}")
            #     if abs(d_real - theoretical_move) < 1e-4:
            #         print(f"  >> ESITO  : PERFETTO. Il robot segue esattamente la fisica.")
            #     elif abs(d_real) < abs(theoretical_move):
            #         print(f"  >> ESITO  : IN RITARDO (Eff {efficiency:.0f}%). Causa: Inerzia/Damping (Sta accelerando).")
            #     else:
            #         print(f"  >> ESITO  : OVERSHOOT. Il robot è andato oltre (raro).")
            # print("=" * 100 + "\n")
            # prev_pos = current_pos_after.copy()

            # # ##################################################################


        # Determine done flag
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
            
            # Print penultimate step data
            if last_ee_pos_b is not None:
                print("\n" + "="*80)
                print("PENULTIMATE STEP (before reset)")
                print("="*80)
                print(f"EE frame (robot RF):   X={last_ee_pos_b[0]:.4f}, Y={last_ee_pos_b[1]:.4f}, Z={last_ee_pos_b[2]:.4f}, w={last_ee_quat_b[0]:.4f}, x={last_ee_quat_b[1]:.4f}, y={last_ee_quat_b[2]:.4f}, z={last_ee_quat_b[3]:.4f}")
                print(f"DISTANCE ERROR: {last_dist_error*1000:.4f} mm | ORIENTATION ERROR: {last_orient_error*180/np.pi:.4f}° ({last_orient_error:.6f} rad)")
                if last_liver_pos is not None:
                    print(f"LIVER POSITION (world): X={last_liver_pos[0]:.4f}, Y={last_liver_pos[1]:.4f}, Z={last_liver_pos[2]:.4f} (m)")
                print("="*80)
            
            # Print liver offset
            if last_liver_offset is not None:
                print(f"\nLIVER OFFSET: X={last_liver_offset[0]:+.4f}, Y={last_liver_offset[1]:+.4f}, Z={last_liver_offset[2]:+.4f} (m)")
            
            # Print initial joint positions at termination on one line
            pos_str = " ".join([f"{p:.5f}".replace(".", ",") for p in initial_pos])
            print(pos_str)
            print("[INFO] Stopping simulation.")
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
