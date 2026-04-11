# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING
import math
import numpy as np
from isaaclab.assets import RigidObject, DeformableObject
from isaaclab.sensors import FrameTransformer
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prim_utils
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def __tissue_barycenter_w(liver: DeformableObject) -> torch.Tensor:
    return liver.data.root_pos_w

def object_ee_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
) -> torch.Tensor:
    
    robot: RigidObject = env.scene[asset_cfg.name]
    robot_root_pos_w = robot.data.root_state_w[:, :3]
    robot_root_quat_w = robot.data.root_state_w[:, 3:7]
    target_pose_w = liver_target_pose_world(env, object_cfg)
    target_pos_w = target_pose_w[:, :3]
    ee_pos_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    ee_pos_b, _ = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, ee_pos_w
    )
    target_pos_b, _ = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, target_pos_w
    )
    dist = torch.norm(ee_pos_b - target_pos_b, dim=1)
    return dist


def object_ee_orientation_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[asset_cfg.name]
    robot_root_pos_w = robot.data.root_state_w[:, :3]
    robot_root_quat_w = robot.data.root_state_w[:, 3:7]
    target_pose_w = liver_target_pose_world(env, object_cfg)
    des_quat_w = target_pose_w[:, 3:7]
    curr_quat_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    _, des_quat_b = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, torch.zeros_like(robot_root_pos_w), des_quat_w
    )
    _, curr_quat_b = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, torch.zeros_like(robot_root_pos_w), curr_quat_w
    )
    orientation_error = quat_error_magnitude(curr_quat_b, des_quat_b)
    return orientation_error


def ee_below_target_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
) -> torch.Tensor:
    """Penalty when EE Z position goes more than 1 cm below target Z position (in robot RF).
    
    Returns:
        torch.Tensor: 1.0 if EE is more than 1 cm below target (Z_ee < Z_target - 0.01), 0.0 otherwise
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    robot_root_pos_w = robot.data.root_state_w[:, :3]
    robot_root_quat_w = robot.data.root_state_w[:, 3:7]
    
    # Get target pose
    target_pose_w = liver_target_pose_world(env, object_cfg)
    target_pos_w = target_pose_w[:, :3]
    
    # Get EE pose
    ee_pos_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    
    # Transform to robot RF (Z axis)
    ee_pos_b, _ = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, ee_pos_w
    )
    target_pos_b, _ = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, target_pos_w
    )
    
    # Check if EE Z is more than 1 cm below target Z
    # Return 1.0 where Z_ee < (Z_target - 0.01), 0.0 otherwise
    threshold = 0.01  # 1 cm threshold
    penalty = (ee_pos_b[:, 2] < target_pos_b[:, 2] - threshold).float()
    
    return penalty


def success_reached(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
    pos_threshold: float = 0.003,
    or_threshold: float = 0.2,
) -> torch.Tensor:
    """Reward when the end-effector reaches the target within thresholds.
    
    Returns:
        torch.Tensor: 1.0 if both position and orientation are within thresholds, 0.0 otherwise
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    robot_root_pos_w = robot.data.root_state_w[:, :3]
    robot_root_quat_w = robot.data.root_state_w[:, 3:7]
    
    # Get target pose
    target_pose_w = liver_target_pose_world(env, object_cfg)
    target_pos_w = target_pose_w[:, :3]
    target_quat_w = target_pose_w[:, 3:7]
    
    # Get EE pose
    ee_pos_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    ee_quat_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    
    # Transform to robot RF
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, ee_pos_w, ee_quat_w
    )
    target_pos_b, target_quat_b = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, target_pos_w, target_quat_w
    )
    
    # Check position and orientation errors
    pos_error = torch.norm(ee_pos_b - target_pos_b, dim=1)
    or_error = quat_error_magnitude(ee_quat_b, target_quat_b)
    
    # Success when both conditions are met
    success = ((pos_error < pos_threshold) & (or_error < or_threshold)).float()
    
    return success



def liver_target_pose_world(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
) -> torch.Tensor:
    deformable: DeformableObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene["robot"]

    # # ============== OPTION 0: Dynamic target from deformable node ==============
    # # REACH
    # # target position
    # nodal_pos_w = deformable.data.nodal_pos_w          # (N_env, N_nodes, 3)
    # # anchor_idx = 319
    # # anchor_idx = 3 # when res = 9
    # anchor_idx = 433 # when res = 12 final_organs_5 and final_organs_6
    # # anchor_idx = 26 # 26: (final_organs_4), 319
    # anchor_pos_w = nodal_pos_w[:, anchor_idx, :]       # (N,3)
    
    # # dz = 0.007 
    # # dz = - 0.040 #  0.007 new
    # dz = 0.0
    # # dz = -0.015
    # default_anchor_pos = deformable.data.default_nodal_state_w[:, anchor_idx, :3].to(anchor_pos_w.device)
    # reach_pos_w = anchor_pos_w.clone()
    
    # reach_pos_w[:, 2] = reach_pos_w[:, 2] - dz
    # # reach_pos_w[:, 2] = default_anchor_pos[:, 2] - dz
    
    # # reach follows the deformable node (including Z) so we can measure lift on the reach point
    # # reach_pos_w = anchor_pos_w + torch.tensor([0.0, 0.0, -dz], device=anchor_pos_w.device, dtype=anchor_pos_w.dtype)
    
    # '''env_origins = env.scene.env_origins
    # approach_pos_env = target_pos_w - env_origins
    # target_pos_w = approach_pos_env'''
    
    # # # desired orientation (wrt world ref)
    # # yaw_deg = 45.0 # 0.0 ---------- 45.0 (3)
    # # pitch_deg = 50.0 # (1)
    # # roll_deg = -45.0 # 0.0 ---------- -45.0 (2)
    # # desired orientation (wrt world ref)
    # yaw_deg = 180.0 - 0.0
    # pitch_deg = 20.0 
    # roll_deg = 20.0 
    
    # yaw = math.radians(yaw_deg)
    # pitch = math.radians(pitch_deg)
    # roll = math.radians(roll_deg)
    
    # num_envs = reach_pos_w.shape[0] # anchor_pos_w
    # device = reach_pos_w.device # anchor_pos_w
    # dtype = reach_pos_w.dtype # anchor_pos_w
    
    # w_yaw = math.cos(yaw / 2.0)
    # z_yaw = math.sin(yaw / 2.0)
    # q_yaw = torch.tensor(
    #     [w_yaw, 0.0, 0.0, z_yaw],
    #     device=device,
    #     dtype=dtype,
    # )
    
    # w_pitch = math.cos(pitch / 2.0)
    # y_pitch = math.sin(pitch / 2.0)
    # q_pitch = torch.tensor(
    #     [w_pitch, 0.0, y_pitch, 0.0],
    #     device=device,
    #     dtype=dtype,
    # )
    
    # w_roll = math.cos(roll / 2.0)
    # x_roll = math.sin(roll / 2.0)
    # q_roll = torch.tensor(
    #     [w_roll, x_roll, 0.0, 0.0],
    #     device=device,
    #     dtype=dtype,
    # )
    
    
    # # quat_w_single = quat_mul(q_pitch, q_roll)          # q_pitch, q_yaw --------- q_pitch, q_roll
    # # q_des_world_single = quat_mul(quat_w_single, q_yaw)
    
    # quat_w_single = quat_mul(q_pitch, q_yaw)
    # q_des_world_single = quat_mul(quat_w_single, q_roll)
    
    # desired_quat_w = q_des_world_single.repeat(num_envs, 1)  
    
    # # [pos, quat] in world frame
    
    
    # # # REACH + INSERT
    # # # INSERT (compute insert target using horizontal component of EE z)
    # # ee_z_w = quat_apply(desired_quat_w, torch.tensor([0.0, 0.0, 1.0], device = device, dtype=dtype).repeat(num_envs, 1))
    # # # project onto horizontal plane (remove vertical component)
    # # ee_z_h = ee_z_w.clone()
    # # ee_z_h[:, 2] = 0.0
    # # norms = torch.norm(ee_z_h, dim=1, keepdim=True)
    # # eps = 1e-6
    # # fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).repeat(num_envs, 1)
    # # ee_dir_h = torch.where(norms > eps, ee_z_h / norms, fallback)
    # # d_insert = 0.03
    # # # compute insert pos using anchor x/y but fixed initial z so the insert target does not rise
    # # default_anchor_pos = deformable.data.default_nodal_state_w[:, anchor_idx, :3].to(anchor_pos_w.device)
    # # insert_base = anchor_pos_w.clone()
    # # insert_base[:, 2] = default_anchor_pos[:, 2] - dz
    # # insert_pos_w = insert_base - d_insert * ee_dir_h
    # # ============== END OPTION 0 ==============
    
    # ============== OPTION 1: Fixed target in robot frame ==============
    num_envs = env.scene.num_envs
    device = robot.data.root_state_w.device
    dtype = robot.data.root_state_w.dtype
    target_pos_b = torch.tensor([0.0792, 0.0270, -0.065], device=device, dtype=dtype).repeat(num_envs, 1) # NEWNEW REACH TARGET

    # qw qx qy qz
    target_quat_b = torch.tensor([0.7071068, 0.0, -0.5, -0.5], device=device, dtype=dtype).repeat(num_envs, 1) # NEWNEW REACH TARGET
    
    # Transform to world frame
    robot_root_pos = robot.data.root_state_w[:, :3]
    robot_root_quat = robot.data.root_state_w[:, 3:7]
    reach_pos_w, desired_quat_w = combine_frame_transforms(
        robot_root_pos, robot_root_quat, target_pos_b, target_quat_b
    )
    # ============== END OPTION 1 ==============

    target_pos_w = reach_pos_w
    
    return torch.cat([target_pos_w, desired_quat_w], dim=-1)


def gallbladder_mask_tensor(image_tensor):
    """Optimized PyTorch version for gallbladder color detection mask.
    
    Args:
        image_tensor: (B, H, W, 3) with values in [0, 255] or [0, 1]
    
    Returns:
        Boolean mask tensor with same spatial dimensions as input.
    """
    # Normalize to [0, 255] range if needed
    # Check dtype and value range: if it's float and max is small, it's likely [0, 1]
    if image_tensor.dtype in [torch.float32, torch.float64]:
        # Check if values are in [0, 1] range (allowing small tolerance for floating point)
        max_val = torch.max(image_tensor)
        if max_val <= 1.5:  # Values > 1.5 definitely mean [0, 255] range
            image_tensor = image_tensor * 255.0
    
    # Extract RGB channels
    r = image_tensor[..., 0]
    g = image_tensor[..., 1]
    b = image_tensor[..., 2]

    # 1. Basic color range thresholds for gallbladder detection (matching NumPy version)
    r_range = (r >= 0) & (r <= 100)
    g_range = (g >= 30) & (g <= 255)
    b_range = (b >= 30) & (b <= 255)
    
    # 2. Ensure it's a green-blue tone (not red-dominant)
    is_greenish_blue = (g > r + 10) & (b > r + 10) & (torch.abs(g - b) <= 30)
    
    # 3. Exclude grays (where all channels are similar)
    is_not_gray = (torch.abs(g - r) > 8) | (torch.abs(b - r) > 8)
    
    # 4. Exclude blacks (too dark)
    is_not_black = (r + g + b) > 30
    
    # 5. Exclude whites (too bright)
    is_not_white = (r + g + b) < 500

    # Combine all conditions into final mask (True/False)
    final_mask = r_range & g_range & b_range & is_greenish_blue & is_not_gray & is_not_black & is_not_white
    return final_mask


def gallbladder_pixel_count(rgb_image):
    """Detect gallbladder pixels using color range (cyan-green tones) - NumPy version.
    
    Args:
        rgb_image: NumPy array (H, W, 3) with values in [0, 255]
    
    Returns:
        int: Number of gallbladder pixels detected
    """
    img = rgb_image.astype(np.float32)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # # Basic range for the colors (including lighter tones)
    r_range = (r >= 0) & (r <= 100)
    g_range = (g >= 30) & (g <= 255)
    b_range = (b >= 30) & (b <= 255)

    # Ensure it's a green-blue tone (not red-dominant)
    is_greenish_blue = (g > r + 10) & (b > r + 10) & (np.abs(g - b) <= 30)
    
    # Exclude grays (where all channels are similar)
    is_not_gray = (np.abs(g - r) > 8) | (np.abs(b - r) > 8)
    
    # Exclude blacks (too dark)
    is_not_black = (r + g + b) > 30
    
    # Exclude whites (too bright)
    is_not_white = (r + g + b) < 500
    
    # Combine all conditions
    mask_range = r_range & g_range & b_range & is_greenish_blue & is_not_gray & is_not_black & is_not_white
    visible_pixels = np.sum(mask_range > 0)
    return int(visible_pixels)


def visual_exposure_reward(env):
    """Visual exposure reward based on gallbladder pixel visibility.
    
    Uses the same NumPy-based detection as play_state_1.py for consistency.
    Applies a 2-step warmup period (steps 0-1) where reward is zeroed to avoid noisy scene initialization.
    
    Args:
        env: ManagerBasedRLEnv instance.
    
    Returns:
        torch.Tensor: Reward per environment (shape: num_envs), scaled as visible_pixels / 1000.
                      Returns 0 for first 2 steps of each episode (warmup period).
    """
    # Get camera sensor from scene
    cam_sensor = env.scene["camera"] 
    
    # RGB output shape: (num_envs, H, W, 3)
    rgb_data = cam_sensor.data.output["rgb"]
    
    # Convert to NumPy and scale to [0, 255] range
    rgb_numpy = rgb_data.cpu().numpy()
    
    # Initialize reward tensor
    num_envs = rgb_data.shape[0]
    device = rgb_data.device
    rewards = []
    
    # Process each environment separately using the NumPy version for consistency
    for i in range(num_envs):
        # Extract frame for this environment (H, W, 3)
        frame = rgb_numpy[i]
        
        # Scale to [0, 255] if needed (IsaacLab typically outputs [0, 1])
        if frame.max() <= 1.0:
            frame = (frame * 255.0).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        
        # Count gallbladder pixels using the NumPy version
        pixel_count = gallbladder_pixel_count(frame)
        
        # Normalize by 1000
        reward_value = pixel_count / 1000.0
        rewards.append(reward_value)
    
    # Convert list to tensor
    reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
    
    # Apply warmup
    if hasattr(env, 'episode_length_buf'):
        warmup_mask = env.episode_length_buf <= 0  # Steps 0 and 1 (first two steps) 
        # warmup_mask = env.episode_length_buf <= 2
        reward_tensor[warmup_mask] = 0.0
    
    return reward_tensor

def gallbladder_visibility_success(env: ManagerBasedRLEnv, pixel_threshold: float = 2500) -> torch.Tensor:
    """Termination condition: episode succeeds when gallbladder visibility exceeds threshold.
    
    Uses the same NumPy-based detection as visual_exposure_reward for consistency.
    
    Args:
        env: The environment instance
        pixel_threshold: Number of visible gallbladder pixels required for success (default: 2500)
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating success for each environment
    """
    try:
        # Get camera sensor
        camera = env.unwrapped.scene.sensors["camera"]
        if camera is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
        # Get RGB data from camera
        cam_out = camera.data.output
        rgb_tensor = cam_out.get("rgb", None)
        if rgb_tensor is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
        # Convert to NumPy and scale to [0, 255] range
        rgb_numpy = rgb_tensor.cpu().numpy()
        
        # Initialize success tensor
        num_envs = rgb_tensor.shape[0]
        device = rgb_tensor.device
        success_list = []
        
        # Process each environment separately using the NumPy version
        for i in range(num_envs):
            # Extract frame for this environment (H, W, 3)
            frame = rgb_numpy[i]
            
            # Scale to [0, 255] if needed
            if frame.max() <= 1.0:
                frame = (frame * 255.0).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            
            # Count gallbladder pixels using the NumPy version
            pixel_count = gallbladder_pixel_count(frame)
            
            # Check if exceeds threshold
            success = pixel_count >= pixel_threshold
            success_list.append(success)
        
        # Convert list to tensor
        success_tensor = torch.tensor(success_list, device=device, dtype=torch.bool)
        
        return success_tensor
        
    except Exception as e:
        # If camera is not available or error occurs, return False for all environments
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)



def vertical_lifting_reward(env, lateral_penalty_coef: float = 1.0) -> torch.Tensor:
    """Reward that encourages vertical lifting (Z axis) and penalizes lateral movement.
    
    Computes the TCP displacement relative to the reset position and rewards the positive
    Z component while penalizing lateral components (X, Y).
    
    Args:
        env: ManagerBasedRLEnv instance.
        lateral_penalty_coef: Weight of lateral penalty (α). Default 1.0.
    
    Returns:
        torch.Tensor: Reward per environment (shape: num_envs).
    """
    ee_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]  # (num_envs, 3)
    
    if not hasattr(env, '_ee_reset_pos'):
        env._ee_reset_pos = ee_pos_w.clone()
    
    # At reset (episode_length_buf == 1), update the reference position
    reset_mask = env.episode_length_buf <= 1
    env._ee_reset_pos[reset_mask] = ee_pos_w[reset_mask].clone()
    
    # Displacement from reset
    delta = ee_pos_w - env._ee_reset_pos  # (num_envs, 3)
    
    delta_z = delta[:, 2]   # vertical component (positive = lifting)
    delta_x = delta[:, 0]
    delta_y = delta[:, 1]
    
    # Reward: rewards positive z, penalizes lateral norm
    lateral_norm = torch.sqrt(delta_x**2 + delta_y**2 + 1e-8)
    
    reward = torch.clamp(delta_z, min=0.0) - lateral_penalty_coef * lateral_norm
    
    return reward