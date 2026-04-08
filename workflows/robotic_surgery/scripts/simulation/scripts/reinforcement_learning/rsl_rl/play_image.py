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
import cv2  # For saving mask images
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import liver_target_pose_world, ee_below_target_penalty, gallbladder_pixel_count, gallbladder_visibility_success, visual_exposure_reward, vertical_lifting_reward
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf
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
    obs = env.get_observations() # NEW RSL_RL OBS FROM SIM
    timestep = 0
    
    # ========== CONFIGURATION FOR OBSERVATION FRAME SAVING ==========
    # Set SAVE_OBS_FRAMES to True to enable saving observation images
    SAVE_OBS_FRAMES = False  # Set to True to save model input observations
    obs_frames_dir = None
    if SAVE_OBS_FRAMES:
        obs_frames_dir = os.path.join(log_dir, "obs_frames")
        os.makedirs(obs_frames_dir, exist_ok=True)
        print(f"[INFO] Observation frames will be saved to: {obs_frames_dir}")
    # ==================================================================
    
    # ========== LIVER STABILIZATION WARM-UP ==========
    # Number of steps to run with zero actions before the policy starts
    STABILIZATION_STEPS = 20  # Adjust as needed (10-30 steps is usually enough)
    # ==================================================
    
    # ===== REF FRAME VISUALIZATION TOGGLE =====
    SHOW_REF_FRAMES = False
    camera_frame_marker = None
    robot_root_frame_marker = None
    target_frame_marker = None
    target_point_marker = None
    # ===== END VISUALIZATION (COMMENT/DECOMMENT) =====

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
    
    
    # Variables to save penultimate step data
    last_ee_pos_b = None
    last_ee_quat_b = None
    last_ee_pos_w = None
    last_ee_quat_w = None
    last_dist_error = None
    last_orient_error = None
    last_liver_offset = None
    last_liver_pos = None

    filtered_actions = None
    
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
    
    # Store initial liver position offset (robot reference frame) for computing relative offset
    liver_pos_b_initial = None
    liver_pos_w_initial = None
    liver_offset_w = None
    liver_offset_b = None
    liver_expected_offset = None  # Track expected offset from reset_nodal_state_uniform
    try:
        liver_asset = base_env.scene["liver"]
        # Deformable objects use nodal_pos_w for node positions
        liver_nodes_w = liver_asset.data.nodal_pos_w[:, :, :3]  # [batch, num_nodes, 3]
        liver_pos_w = liver_nodes_w[0].mean(dim=0)  # Average position of all nodes
        
        # For rotation, we'll use identity (deformable objects don't have a clear rotation)
        liver_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=base_env.device)
        
        # Transform to robot reference frame
        liver_pos_b, _ = subtract_frame_transforms(
            robot_root_pos, robot_root_quat, 
            liver_pos_w.unsqueeze(0), liver_quat_w.unsqueeze(0)
        )
        liver_pos_b_initial = liver_pos_b[0].cpu().numpy()
        
        # Get expected offset from config
        try:
            position_range = None
            if hasattr(base_env, 'event_manager') and hasattr(base_env.event_manager, 'reset_liver_position'):
                reset_event = base_env.event_manager.reset_liver_position
                if hasattr(reset_event, 'params'):
                    position_range = reset_event.params.get("position_range", None)
            
            if position_range is None and hasattr(base_env, 'cfg') and hasattr(base_env.cfg, 'events'):
                if hasattr(base_env.cfg.events, 'reset_liver_position'):
                    reset_event_cfg = base_env.cfg.events.reset_liver_position
                    if hasattr(reset_event_cfg, 'params'):
                        position_range = reset_event_cfg.params.get("position_range", None)
            
            if position_range is not None:
                liver_expected_offset = np.array([
                    position_range.get("x", (0, 0))[0],
                    position_range.get("y", (0, 0))[0],
                    position_range.get("z", (0, 0))[0]
                ])
                print(f"LIVER POSITION (robot RF):   X={liver_pos_b_initial[0]:+.4f}, Y={liver_pos_b_initial[1]:+.4f}, Z={liver_pos_b_initial[2]:+.4f}")
                print(f"EXPECTED OFFSET FROM RANGE: X={liver_expected_offset[0]:+.4f}, Y={liver_expected_offset[1]:+.4f}, Z={liver_expected_offset[2]:+.4f}")
            else:
                print(f"LIVER POSITION (robot RF):   X={liver_pos_b_initial[0]:+.4f}, Y={liver_pos_b_initial[1]:+.4f}, Z={liver_pos_b_initial[2]:+.4f}")
        except Exception as e:
            print(f"LIVER POSITION (robot RF):   X={liver_pos_b_initial[0]:+.4f}, Y={liver_pos_b_initial[1]:+.4f}, Z={liver_pos_b_initial[2]:+.4f}")
            print(f"[DEBUG] Could not get position_range: {e}")
    except Exception as e:
        print(f"[WARNING] Could not get liver position: {e}")
    
    print("="*80 + "\n")

    # ============================================================
    # ===== INITIALIZE CAMERA RF POSITION MARKER (ROBOT RF) =====
    # ============================================================
    camera_rf_marker = None  # Will be initialized on first step
    target_pos_robot_rf = torch.tensor(
        [-0.1182, -0.0434, -0.1000],
        dtype=torch.float32,
        device=base_env.device
    )

    # ========== LIVER STABILIZATION WARM-UP ==========
    print(f"\n[INFO] Running {STABILIZATION_STEPS} stabilization steps (zero actions) to let liver settle...")
    # Get correct action dimension from the environment
    action_dim = base_env.action_manager.total_action_dim
    zero_actions = torch.zeros(env.num_envs, action_dim, device=agent_cfg.device)
    with torch.inference_mode():
        for stab_step in range(STABILIZATION_STEPS):
            obs, _, _, _ = env.step(zero_actions)
            if stab_step % 5 == 0:
                print(f"  [STABILIZE] step {stab_step+1}/{STABILIZATION_STEPS}")
    print("[INFO] Stabilization complete. Starting policy inference.\n")
    # ==================================================

    # ========== MASK SAVING SETUP ==========
    SAVE_MASK = False  # Save binary mask and final frame when episode ends
    if SAVE_MASK:
        print(f"[INFO] Mask saving enabled. Binary mask and final frame will be saved at episode end.")
    # ========================================

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            print("\n" + "="*80)
            print(f"STEP {timestep:04d}")
            print("="*80)
            
            # DEBUG: Very early check
            if timestep == 0:
                print(f"[EARLY DEBUG] SAVE_OBS_FRAMES = {SAVE_OBS_FRAMES}")
                print(f"[EARLY DEBUG] obs_frames_dir = {obs_frames_dir}")
                print(f"[EARLY DEBUG] obs is None: {obs is None}")
                if obs is not None:
                    print(f"[EARLY DEBUG] type(obs) = {type(obs)}")
            
            # ===== SAVE OBSERVATION FRAME (if enabled) =====
            if SAVE_OBS_FRAMES and obs is not None:
                try:
                    # DEBUG: Print obs structure
                    if timestep == 0 or timestep % 10 == 0:
                        print(f"[OBS DEBUG STEP {timestep}] obs type: {type(obs)}")
                        # TensorDict has a keys() method
                        if hasattr(obs, 'keys'):
                            print(f"[OBS DEBUG STEP {timestep}] obs keys: {list(obs.keys())}")
                            for key in obs.keys():
                                val = obs[key]
                                if torch.is_tensor(val):
                                    print(f"[OBS DEBUG STEP {timestep}]   {key}: tensor shape={val.shape}, dtype={val.dtype}")
                                else:
                                    print(f"[OBS DEBUG STEP {timestep}]   {key}: {type(val)}")
                    
                    # Extract the image observation tensor (should be NCHW format [B, 3, 168, 168])
                    # Handle both dict and TensorDict
                    obs_tensor = None
                    
                    # Try to access as TensorDict or dict with "policy" key
                    if hasattr(obs, '__getitem__'):
                        try:
                            obs_tensor = obs["policy"]
                            if timestep == 0 or timestep % 10 == 0:
                                print(f"[OBS DEBUG STEP {timestep}] Found obs['policy'], shape: {obs_tensor.shape}")
                        except (KeyError, TypeError):
                            pass
                    
                    # Fallback: if obs itself is a tensor
                    if obs_tensor is None and torch.is_tensor(obs):
                        obs_tensor = obs
                        if timestep == 0 or timestep % 10 == 0:
                            print(f"[OBS DEBUG STEP {timestep}] obs is tensor, shape: {obs_tensor.shape}")
                    
                    if obs_tensor is not None:
                        if timestep == 0 or timestep % 10 == 0:
                            print(f"[OBS DEBUG STEP {timestep}] obs_tensor shape: {obs_tensor.shape}, checking if shape[1]==3...")
                        
                        if len(obs_tensor.shape) == 4:
                            if obs_tensor.shape[1] == 3:
                                print(f"[OBS DEBUG STEP {timestep}] ✓ Shape is 4D and channels=3, saving...")
                                # Extract first batch, assume NCHW format
                                img_nchw = obs_tensor[0]  # [3, 168, 168]
                                
                                # Convert from NCHW [3, 168, 168] to HWC [168, 168, 3]
                                img_hwc = img_nchw.permute(1, 2, 0)  # [168, 168, 3]
                                
                                # Denormalize from [0, 1] to [0, 255]
                                img_rgb = (img_hwc.cpu().numpy() * 255.0).astype(np.uint8)
                                
                                # Save as PNG
                                frame_path = os.path.join(obs_frames_dir, f"obs_frame_step_{timestep:04d}.png")
                                cv2.imwrite(frame_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                                print(f"[OBS FRAME] ✓ Saved observation: {frame_path} | Shape: {img_nchw.shape}")
                            else:
                                print(f"[OBS DEBUG STEP {timestep}] ✗ Shape is 4D but channels={obs_tensor.shape[1]} (expected 3)")
                        else:
                            print(f"[OBS DEBUG STEP {timestep}] ✗ Shape is {len(obs_tensor.shape)}D (expected 4D)")
                    else:
                        if timestep == 0 or timestep % 10 == 0:
                            print(f"[OBS DEBUG STEP {timestep}] ✗ obs_tensor is None!")
                except Exception as e:
                    print(f"[OBS FRAME WARNING] Could not save observation frame: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
            elif timestep == 0:
                print(f"[OBS DEBUG] SAVE_OBS_FRAMES={SAVE_OBS_FRAMES}, obs is None: {obs is None}")
            # ================================================
            
            # ===== SAVE EXACT camera_rgb_observation (replica of training observation) =====
            # This replicates the EXACT observation used during training
            if SAVE_OBS_FRAMES:
                try:
                    camera = base_env.scene.sensors["camera"]
                    rgb = camera.data.output["rgb"][..., :3].float()  # [B, H, W, 3]
                    rgb /= 255.0  # Normalize to [0, 1]
                    rgb = rgb.permute(0, 3, 1, 2)  # NCHW: [B, 3, H, W]
                    
                    # Resize to 168x168 if needed
                    if rgb.shape[-2:] != (168, 168):
                        rgb = torch.nn.functional.interpolate(
                            rgb, size=(168, 168), mode='bilinear', align_corners=False
                        )
                    
                    # Extract batch 0: [3, 168, 168]
                    img_nchw = rgb[0].contiguous()  # Exact replica of camera_rgb_observation output
                    
                    # Convert to HWC for saving
                    img_hwc = img_nchw.permute(1, 2, 0)  # [168, 168, 3]
                    img_rgb = (img_hwc.cpu().numpy() * 255.0).astype(np.uint8)
                    
                    # Save
                    frame_path = os.path.join(obs_frames_dir, f"obs_frame_step_{timestep:04d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                    
                    if timestep % 20 == 0:
                        print(f"[OBS FRAME] ✓ Saved EXACT camera_rgb_observation: {frame_path} | Shape: {img_nchw.shape}")
                except Exception as e:
                    print(f"[OBS FRAME ERROR] {type(e).__name__}: {e}")
            # ============================================================================
            
            pos_before = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            # # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)

            
            cmd_term = base_env.action_manager.get_term("arm_action")
            print(f"scale: {cmd_term._scale}")
            print(f"offset: {cmd_term._offset}") 
            print(f"processed_actions: {cmd_term.processed_actions[0]}")
            print(f"current_pos: {robot_asset.data.joint_pos[0][:6]}")
            print(f"raw action [2]: {actions[0,2]}")

            # ##################
            # ------> desired_target=raw_actions×scale+offset --> desired_delta=desired_target-current_pos --> filtered_delta=alpha×desired_delta+(1−alpha)×previous_filtered_delta --> filtered_target=current_pos+filtered_delta --> filtered_raw=(filtered_target-offset)/scale
            USE_PROCESSED_FILTER = False
            PROCESSED_FILTER_ALPHA = 0.02
            if USE_PROCESSED_FILTER:
                scale = 0.02
                offset = cmd_term._offset[0]
                current_pos = robot_asset.data.joint_pos[0][:6]
                # raw_clipped = torch.clamp(actions[0], -1.0, 1.0)
                raw_actions = actions[0]
                desired_target = raw_actions * scale + offset
                desired_delta = desired_target - current_pos
                if not hasattr(cmd_term, '_filtered_delta'):
                    cmd_term._filtered_delta = desired_delta.clone()
                cmd_term._filtered_delta = (
                    PROCESSED_FILTER_ALPHA * desired_delta +
                    (1.0 - PROCESSED_FILTER_ALPHA) * cmd_term._filtered_delta
                )
                filtered_target = current_pos + cmd_term._filtered_delta
                filtered_raw = (filtered_target - offset) / scale
                actions_to_use = filtered_raw.unsqueeze(0)
                print(f"[FILTER] J2: RAW={actions[0,2].item():+.5f} | DESIRED_DELTA={desired_delta[2].item():+.5f} | FILTERED_DELTA={cmd_term._filtered_delta[2].item():+.5f}")
            else:
                actions_to_use = actions
            ###################

            ###########################
            # Apply filter or use raw actions based on USE_ACTION_FILTER flag
            # ---------> filtered action = alpha * raw_action + (1 - alpha) * previous_filtered_action
            USE_ACTION_FILTER = False
            action_filter_alpha = 0.1 # python play_image.py --task Isaac-Liver-PSM-Play-v0 --num_envs 1 --experiment_name psm_reach --load_run 2026-03-13_10-03-18 --checkpoint model_6000.pt --enable_cameras
            # action_filter_alpha = 0.2 # python play_image.py --task Isaac-Liver-PSM-Play-v0 --num_envs 1 --experiment_name psm_reach --load_run 2026-03-13_10-03-18 --checkpoint model_8000.pt --enable_cameras
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
            ############################

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

            cmd_term = base_env.action_manager.get_term("arm_action")
            processed_actions = cmd_term.processed_actions[0].cpu().numpy()
            pos_after = robot_asset.data.joint_pos[0][:6].cpu().numpy()
            real_move = pos_after - pos_before
            # print(f"[VERIFY] J2: PROCESSED={processed_actions[2]:+.5f} | FILTERED_DELTA={cmd_term._filtered_delta[2].item():+.5f}")
            # ========== PRINT ee_below_target_penalty VALUE ==========
            try:
                # Compute ee_below_target_penalty: 1.0 if EE Z is more than 1 cm below target Z
                # Using already computed e_pos_b and t_pos_b (robot reference frame)
                ee_z = e_pos_b[2]
                target_z = t_pos_b[2]
                threshold = 0.01  # 1 cm threshold
                penalty_value = 1.0 if ee_z < (target_z - threshold) else 0.0
                z_diff = target_z - ee_z
                status = "BELOW THRESHOLD" if penalty_value > 0.5 else "OK"
                print(f"[ee_below_target_penalty] EE_Z={ee_z:.4f}, TARGET_Z={target_z:.4f}, Diff={z_diff:.4f}, Threshold={threshold:.4f}, Value={penalty_value:.1f} ({status})")
            except Exception as e:
                print(f"[WARNING] Could not compute ee_below_target_penalty: {type(e).__name__}: {e}")
            # =====================================================
            
            # ========== SAVE OBSERVATION FRAME IMAGES ==========
            # Note: Single frame observations are already saved above in the obs["policy"] section
            # Frame stack code disabled - the policy receives processed observations directly
            # =====================================================
            
            
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
            t_pos_w = target_pos_w[0].detach().cpu().numpy()
            t_quat_w = target_quat_w[0].detach().cpu().numpy()
            t_pos_b = target_pos_b[0].detach().cpu().numpy()
            t_quat_b = target_quat_b[0].detach().cpu().numpy()
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
            
            # Draw and visualize EE frame (only if SHOW_REF_FRAMES is True)
            if SHOW_REF_FRAMES:
                try:
                    if marker_container["ee_marker"] is None:
                        ee_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame")
                        # Scale down the frame marker to be small
                        ee_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
                        marker_container["ee_marker"] = VisualizationMarkers(ee_marker_cfg)
                    
                    # Visualize the EE frame in world coordinates
                    ee_pos_w_tensor = torch.tensor(e_pos_w, dtype=torch.float32, device=base_env.device).unsqueeze(0)
                    ee_quat_w_tensor = torch.tensor(e_quat_w, dtype=torch.float32, device=base_env.device).unsqueeze(0)
                    marker_indices = torch.zeros(1, dtype=torch.int32, device=base_env.device)
                    marker_container["ee_marker"].visualize(ee_pos_w_tensor, ee_quat_w_tensor, marker_indices=marker_indices)
                except Exception as e:
                    print(f"[WARNING] Could not visualize EE frame: {type(e).__name__}: {e}")
            
            try:
                liver_asset = base_env.scene["liver"]
                # Deformable objects use nodal_pos_w for node positions
                liver_nodes_w = liver_asset.data.nodal_pos_w[:, :, :3]  # [batch, num_nodes, 3]
                liver_pos_w = liver_nodes_w[0].mean(dim=0)  # Average position of all nodes
                
                # For rotation, we'll use identity (deformable objects don't have a clear rotation)
                liver_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=base_env.device)
                
                # Transform to robot reference frame
                liver_pos_b, _ = subtract_frame_transforms(
                    robot_root_pos, robot_root_quat, 
                    liver_pos_w.unsqueeze(0), liver_quat_w.unsqueeze(0)
                )
                liver_pos_b = liver_pos_b[0].cpu().numpy()
                liver_pos_w_np = liver_pos_w.cpu().numpy()
                
                # If this is the first step of a new episode, reinitialize the initial position
                if liver_pos_b_initial is None:
                    liver_pos_b_initial = liver_pos_b.copy()
                    liver_pos_w_initial = liver_pos_w_np.copy()
                    
                    # Print initial positions in both frames
                    print(f"LIVER POSITION (world frame):   X={liver_pos_w_initial[0]:+.4f}, Y={liver_pos_w_initial[1]:+.4f}, Z={liver_pos_w_initial[2]:+.4f}")
                    print(f"LIVER POSITION (robot RF):     X={liver_pos_b_initial[0]:+.4f}, Y={liver_pos_b_initial[1]:+.4f}, Z={liver_pos_b_initial[2]:+.4f}")
                    try:
                        # Access events through the environment's manager
                        position_range = None
                        
                        # Check if event manager has the reset_liver_position event
                        if hasattr(base_env, 'event_manager'):
                            event_mgr = base_env.event_manager
                            if hasattr(event_mgr, 'reset_liver_position'):
                                reset_event = event_mgr.reset_liver_position
                                if hasattr(reset_event, 'params'):
                                    position_range = reset_event.params.get("position_range", None)
                        
                        # Alternative: try accessing through cfg.events
                        if position_range is None and hasattr(base_env, 'cfg'):
                            cfg = base_env.cfg
                            if hasattr(cfg, 'events') and hasattr(cfg.events, 'reset_liver_position'):
                                reset_event_cfg = cfg.events.reset_liver_position
                                if hasattr(reset_event_cfg, 'params'):
                                    position_range = reset_event_cfg.params.get("position_range", None)
                        
                        if position_range is not None:
                            liver_expected_offset = np.array([
                                position_range.get("x", (0, 0))[0],
                                position_range.get("y", (0, 0))[0],
                                position_range.get("z", (0, 0))[0]
                            ])
                            print(f"EXPECTED OFFSET (world frame): X={liver_expected_offset[0]:+.4f}, Y={liver_expected_offset[1]:+.4f}, Z={liver_expected_offset[2]:+.4f}")
                        else:
                            print(f"[DEBUG] position_range not found in event manager or cfg")
                    except Exception as e:
                        print(f"[DEBUG] Could not get position_range: {type(e).__name__}: {e}")

                
                # Compute relative offsets from initial position (for termination output)
                if liver_pos_w_initial is not None and liver_pos_b_initial is not None:
                    liver_offset_w = liver_pos_w_np - liver_pos_w_initial
                    liver_offset_b = liver_pos_b - liver_pos_b_initial
                else:
                    # Initialize if they're None
                    if liver_pos_w_initial is None and liver_pos_b_initial is not None:
                        liver_pos_w_initial = liver_pos_w_np.copy()
                    if liver_offset_w is None and liver_offset_b is None:
                        liver_offset_w = np.array([0.0, 0.0, 0.0])
                        liver_offset_b = np.array([0.0, 0.0, 0.0])
            except Exception as e:
                print(f"[WARNING] Could not get liver position: {e}")
            
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
                last_ee_pos_w = e_pos_w.copy()
                last_ee_quat_w = e_quat_w.copy()
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
            
            # ============================================================
            # ===== VISUALIZE CAMERA RF POSITION MARKER (ROBOT RF) =====
            # ============================================================
            try:
                '''
                # Initialize marker on first step
                if camera_rf_marker is None:
                    stage = stage_utils.get_current_stage()
                    marker_path = "/Visuals/CameraRFPositionMarker"
                    prim = stage.GetPrimAtPath(marker_path)
                    if not prim.IsValid():
                        prim = prim_utils.create_prim(prim_path=marker_path, prim_type="Sphere")
                    
                    sphere_geom = UsdGeom.Sphere(prim)
                    sphere_geom.GetRadiusAttr().Set(0.005)  # 5mm radius
                    color_attr = sphere_geom.GetDisplayColorAttr()
                    if not color_attr.HasValue():
                        color_attr.Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # Red color
                    
                    camera_rf_marker = marker_path  # Store path for update
                    '''
                # Update marker position each step
                robot_root_pos = robot_asset.data.root_state_w[:, :3]
                robot_root_quat = robot_asset.data.root_state_w[:, 3:7]
                
                # Transform target position from robot RF to world frame
                from isaaclab.utils.math import combine_frame_transforms
                batch_size = robot_root_pos.shape[0]
                target_pos_robot_rf_batch = target_pos_robot_rf.unsqueeze(0).expand(batch_size, -1)
                
                # Identity quaternion
                quat_id = torch.zeros((batch_size, 4), device=base_env.device, dtype=torch.float32)
                quat_id[:, 0] = 1.0
                
                # Transform to world frame
                target_pos_w, _ = combine_frame_transforms(
                    robot_root_pos, robot_root_quat,
                    target_pos_robot_rf_batch, quat_id
                )
                '''
                # Update marker position in USD
                stage = stage_utils.get_current_stage()
                prim = stage.GetPrimAtPath(camera_rf_marker)
                if prim.IsValid():
                    xformable = UsdGeom.Xformable(prim)
                    translate_op = next((op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
                    if not translate_op:
                        translate_op = xformable.AddTranslateOp()
                    
                    # Use first environment position
                    pos = target_pos_w[0].cpu().numpy()
                    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                '''
            except Exception as e:
                print(f"[WARNING] Could not visualize camera RF marker: {type(e).__name__}: {e}")
            
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

                    # Robot root frame
                    robot_root_frame_marker.visualize(robot_root_pos, robot_root_quat, marker_indices=marker_indices)

                    # Target pose w.r.t. robot reference frame
                    target_pose_w = liver_target_pose_world(base_env)
                    target_pos_w_vis = target_pose_w[:, :3]
                    target_quat_w_vis = target_pose_w[:, 3:7]
                    target_pos_b_vis, target_quat_b_vis = subtract_frame_transforms(
                        robot_root_pos, robot_root_quat, target_pos_w_vis, target_quat_w_vis
                    )
                    
                    t_pos_b_vis = target_pos_b_vis[0].detach().cpu().numpy()
                    t_quat_b_vis = target_quat_b_vis[0].detach().cpu().numpy()

                    # Target frame + point (world frame)
                    target_frame_marker.visualize(target_pos_w_vis, target_quat_w_vis, marker_indices=marker_indices)
                    quat_id = torch.zeros((num_envs, 4), device=base_env.device, dtype=target_pos_w_vis.dtype)
                    quat_id[:, 0] = 1.0
                    target_point_marker.visualize(target_pos_w_vis, quat_id, marker_indices=marker_indices)

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
                except Exception as e:
                    print(f"[DEBUG] ref-frame visualization error: {e}")
            # ===== END VISUALIZATION (COMMENT/DECOMMENT) =====
            
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
            filtered_actions = None
            
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
            
            # ===== SAVE BINARY MASK AND FINAL FRAME AT EPISODE END =====
            if SAVE_MASK:
                try:
                    from robotic.surgery.tasks.surgical.liver_retraction.mdp.rewards import gallbladder_mask_tensor
                    
                    camera = base_env.scene.sensors["camera"]
                    if camera is not None and hasattr(camera, "data") and hasattr(camera.data, "output"):
                        # Get RGB data from camera (fresh data at final step)
                        camera_output = camera.data.output
                        
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
            
            # Print camera intrinsics at episode end
            try:
                camera = base_env.scene.sensors["camera"]
                cfg = camera.cfg.spawn          # PinholeCameraCfg
                w   = camera.cfg.width
                h   = camera.cfg.height
                f   = cfg.focal_length          # mm
                ap  = cfg.horizontal_aperture   # mm (default Omniverse = 20.955)

                fx = f / ap * w
                fy = f / ap * h                 # se sensore non quadrato
                cx = w / 2.0
                cy = h / 2.0

                print(f"\n[CAMERA INTRINSICS]")
                print(f"  focal_length={f} mm,  aperture={ap} mm")
                print(f"  fx={fx:.2f} px,  fy={fy:.2f} px")
                print(f"  cx={cx:.1f} px,  cy={cy:.1f} px")
                print(f"  resolution={w}x{h}")
            except Exception as e:
                print(f"[WARNING] intrinsics: {e}")
            #########################################
            
            # Save liver position info before resetting
            saved_liver_pos_b_initial = liver_pos_b_initial
            saved_liver_pos_w_initial = liver_pos_w_initial
            saved_liver_offset_w = liver_offset_w
            saved_liver_offset_b = liver_offset_b
            
            # Reset liver position tracking for next episode
            liver_pos_b_initial = None
            liver_pos_w_initial = None
            liver_offset_w = None
            liver_offset_b = None
            
            # Print penultimate step data
            if last_ee_pos_b is not None:
                print("\n" + "="*80)
                print("STEP before reset")
                print("="*80)
                print(f"EE frame (world frame): X={last_ee_pos_w[0]:.4f}, Y={last_ee_pos_w[1]:.4f}, Z={last_ee_pos_w[2]:.4f}, w={last_ee_quat_w[0]:.4f}, x={last_ee_quat_w[1]:.4f}, y={last_ee_quat_w[2]:.4f}, z={last_ee_quat_w[3]:.4f}")
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
            print(f"JOINT RESET POSITION: {pos_str}")
            
            # Print liver reset position info in both frames
            if saved_liver_pos_w_initial is not None:
                print(f"LIVER RESET POSITION (world frame): X={saved_liver_pos_w_initial[0]:+.4f}, Y={saved_liver_pos_w_initial[1]:+.4f}, Z={saved_liver_pos_w_initial[2]:+.4f}")
            if saved_liver_pos_b_initial is not None:
                print(f"LIVER RESET POSITION (robot RF):   X={saved_liver_pos_b_initial[0]:+.4f}, Y={saved_liver_pos_b_initial[1]:+.4f}, Z={saved_liver_pos_b_initial[2]:+.4f}")
            
            # Print offsets in both frames
            if saved_liver_offset_w is not None:
                print(f"LIVER POSITION OFFSET (world frame): X={saved_liver_offset_w[0]:+.4f}, Y={saved_liver_offset_w[1]:+.4f}, Z={saved_liver_offset_w[2]:+.4f}")
            if saved_liver_offset_b is not None:
                print(f"LIVER POSITION OFFSET (robot RF):   X={saved_liver_offset_b[0]:+.4f}, Y={saved_liver_offset_b[1]:+.4f}, Z={saved_liver_offset_b[2]:+.4f}")
            
            # Print robot base position in world frame
            robot_root_pos_np = robot_root_pos.cpu().numpy() if torch.is_tensor(robot_root_pos) else robot_root_pos
            if robot_root_pos_np.ndim > 1:
                robot_root_pos_np = robot_root_pos_np[0]
            print(f"\nROBOT BASE POSITION (world frame): X={robot_root_pos_np[0]:+.4f}, Y={robot_root_pos_np[1]:+.4f}, Z={robot_root_pos_np[2]:+.4f}")
            
            # Print camera pose in robot reference frame
            try:
                # SOLUTION: READ ACTUAL CAMERA POSITION DIRECTLY FROM SENSOR
                # The camera is parented to env_origin (not to the end-effector),
                # so we must query the sensor, not calculate it from EE position
                camera_sensor = base_env.scene["camera"]
                
                # Get real camera position directly from IsaacLab sensor buffers
                real_cam_pos_w = camera_sensor.data.pos_w[0].cpu().numpy()
                
                # Camera orientation: try to get from sensor, fallback to identity
                # CameraData may not have quat_w, so we use identity quaternion
                real_cam_quat_w = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion (w, x, y, z)
                
                # Transform to robot reference frame
                cam_pos_w_tensor = camera_sensor.data.pos_w
                cam_quat_w_tensor = torch.tensor(real_cam_quat_w, dtype=torch.float32, device=base_env.device).unsqueeze(0)
                cam_pos_b, cam_quat_b = subtract_frame_transforms(
                    robot_root_pos, robot_root_quat,
                    cam_pos_w_tensor, cam_quat_w_tensor
                )
                cam_pos_b = cam_pos_b[0].cpu().numpy()
                cam_quat_b = cam_quat_b[0].cpu().numpy()
                
                print(f"\n--- ACTUAL CAMERA POSE (FROM SENSOR) ---")
                print(f"CAMERA POSITION (world frame):   X={real_cam_pos_w[0]:+.4f}, Y={real_cam_pos_w[1]:+.4f}, Z={real_cam_pos_w[2]:+.4f}")
                print(f"CAMERA POSITION (robot RF):      X={cam_pos_b[0]:+.4f}, Y={cam_pos_b[1]:+.4f}, Z={cam_pos_b[2]:+.4f}")
                print(f"CAMERA ORIENTATION (world frame): w={real_cam_quat_w[0]:+.4f}, x={real_cam_quat_w[1]:+.4f}, y={real_cam_quat_w[2]:+.4f}, z={real_cam_quat_w[3]:+.4f} (identity)")
                print(f"CAMERA ORIENTATION (robot RF):   w={cam_quat_b[0]:+.4f}, x={cam_quat_b[1]:+.4f}, y={cam_quat_b[2]:+.4f}, z={cam_quat_b[3]:+.4f}")
            except Exception as e:
                print(f"[WARNING] Could not read camera sensor: {type(e).__name__}: {e}")
            
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
