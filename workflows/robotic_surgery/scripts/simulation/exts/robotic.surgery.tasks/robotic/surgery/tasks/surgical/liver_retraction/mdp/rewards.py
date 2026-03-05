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
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG, FRAME_MARKER_CFG
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf
# _target_marker = None
_ee_frame_marker = None
_curr_ee_marker = None
_robot_root_marker = None
_world_frame_marker = None
_target_node_marker = None
_prova_rf_marker = None
# _insert_marker = None # REACH + INSERT
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def __tissue_barycenter_w(liver: DeformableObject) -> torch.Tensor:
    return liver.data.root_pos_w

# ####### WORLD RF
# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
# ) -> torch.Tensor:
    
#     asset: RigidObject = env.scene[asset_cfg.name]
#     target_pose_w = liver_target_pose_world(env, object_cfg)
#     des_pos_w = target_pose_w[:, :3]
#     curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    
#     dist = torch.norm(curr_pos_w - des_pos_w, dim=1)
    
    
#     # # REACH + INSERT
#     # phase = getattr(env, "_phase", None)
#     # if phase is not None:
#     #     # # In second phase (INSERT: phase==1), emphasize distance by x2
#     #     # dist = dist * (1.0 + (phase == 1).to(dist.dtype)) # !!!
#     #     # Small bias during REACH (phase==0)
#     #     dist += 0.03 * (phase == 0).to(dist.dtype)
    
    
#     # phase = getattr(env, "_phase", None)
#     # if phase is not None:
#     #     dist += 0.03 * (phase == 0).to(dist.dtype)
#     # else:
#     #     dist += 0.03

#     return dist


######### ROBOT RF
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


# ###### WORLD RF
# def object_ee_orientation_error(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
# ) -> torch.Tensor:
    
#     asset: RigidObject = env.scene[asset_cfg.name]
#     target_pose_w = liver_target_pose_world(env, object_cfg)
#     des_quat_w = target_pose_w[:, 3:7]
#     curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
#     orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
#     return orientation_error


###### ROBOT RF
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
    """Penalty when EE Z position goes below target Z position (in robot RF).
    
    Returns:
        torch.Tensor: 1.0 if EE is below target (Z_ee < Z_target), 0.0 otherwise
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
    
    # Check if EE Z is below target Z
    # Return 1.0 where Z_ee < Z_target, 0.0 otherwise
    penalty = (ee_pos_b[:, 2] < target_pos_b[:, 2]).float()
    
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


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    liver: DeformableObject = env.scene[object_cfg.name]
    object_barycenter_w = __tissue_barycenter_w(liver)
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_barycenter_w
    )
    return object_pos_b

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
    
    # Fixed target position and quaternion in robot frame
    # target_pos_b = torch.tensor([-0.0014, 0.0491, -0.0314], device=device, dtype=dtype).repeat(num_envs, 1)
    # target_pos_b = torch.tensor([-0.0048, 0.0394, -0.0913], device=device, dtype=dtype).repeat(num_envs, 1)
    # target_pos_b = torch.tensor([-0.002, 0.07, -0.085], device=device, dtype=dtype).repeat(num_envs, 1)
    # target_pos_b = torch.tensor([-0.0017, 0.0319, -0.0808], device=device, dtype=dtype).repeat(num_envs, 1)
    
    target_pos_b = torch.tensor([-0.01, 0.06, -0.09], device=device, dtype=dtype).repeat(num_envs, 1) # NEW LIFT TARGET
    # target_pos_b = torch.tensor([-0.0018, 0.0452, -0.1378], device=device, dtype=dtype).repeat(num_envs, 1) # NEW REACH TARGET
   
    # target_quat_b = torch.tensor([0.6002, 0.4988, -0.4274, 0.4564], device=device, dtype=dtype).repeat(num_envs, 1)
    # target_quat_b = torch.tensor([0.7058, 0.3179, -0.2798, 0.5679], device=device, dtype=dtype).repeat(num_envs, 1)
    # target_quat_b = torch.tensor([-0.5231,-0.5274, 0.5553, -0.3739], device=device, dtype=dtype).repeat(num_envs, 1)
    # target_quat_b = torch.tensor([0.936, 0.2062, -0.1264, 0.2557], device=device, dtype=dtype).repeat(num_envs, 1)
    target_quat_b = torch.tensor([0.7423, 0.1969, -0.1670, 0.6184], device=device, dtype=dtype).repeat(num_envs, 1) # NEW LIFT TARGET
    # target_quat_b = torch.tensor([0.7423, 0.1969, -0.1670, 0.6184], device=device, dtype=dtype).repeat(num_envs, 1) # NEW REACH TARGET
    
    # Transform to world frame
    robot_root_pos = robot.data.root_state_w[:, :3]
    robot_root_quat = robot.data.root_state_w[:, 3:7]
    reach_pos_w, desired_quat_w = combine_frame_transforms(
        robot_root_pos, robot_root_quat, target_pos_b, target_quat_b
    )
    # ============== END OPTION 1 ==============
    
    
    # # IMAGE
    # # VISUALIZATION
    # # global _ee_frame_marker, _curr_ee_marker, _insert_marker # REACH + INSERT
    # global _ee_frame_marker, _curr_ee_marker, _robot_root_marker, _world_frame_marker, _target_node_marker, _prova_rf_marker
    # try:
    #     if _target_node_marker is None:
    #         stage = stage_utils.get_current_stage()
    #         target_marker_path = "/Visuals/TargetNode"
    #         prim = stage.GetPrimAtPath(target_marker_path)
    #         if not prim.IsValid():
    #             prim = prim_utils.create_prim(prim_path=target_marker_path, prim_type="Sphere")
    #         sphere_geom = UsdGeom.Sphere(prim)
    #         sphere_geom.GetRadiusAttr().Set(0.0015)  # 1.5mm sphere
    #         color_attr = sphere_geom.GetDisplayColorAttr()
    #         if not color_attr.HasValue():
    #             color_attr.Set([Gf.Vec3f(1.0, 0.85, 0.0)])  # Bright yellow color

    #     # Update target marker position (yellow sphere) every step
    #     stage = stage_utils.get_current_stage()
    #     target_marker_path = "/Visuals/TargetNode"
    #     prim = stage.GetPrimAtPath(target_marker_path)
    #     if prim.IsValid():
    #         xformable = UsdGeom.Xformable(prim)
    #         translate_op = next((op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    #         if not translate_op:
    #             translate_op = xformable.AddTranslateOp()
    #         # reach_pos_w is the target position in world frame
    #         pos = reach_pos_w[0].cpu().numpy() if torch.is_tensor(reach_pos_w) else reach_pos_w[0]
    #         translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    #     if _ee_frame_marker is None:
    #         ee_frame_cfg_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/DesiredEEFrame_REACH")
    #         ee_frame_cfg_vis.markers["frame"].scale = (0.01, 0.01, 0.01)
    #         _ee_frame_marker = VisualizationMarkers(ee_frame_cfg_vis)
        
    #     if _curr_ee_marker is None:
    #         _curr_ee_marker_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/CurrentEEFrame")
    #         _curr_ee_marker_vis.markers["frame"].scale = (0.01, 0.01, 0.01)
    #         _curr_ee_marker = VisualizationMarkers(_curr_ee_marker_vis)

    #     if _robot_root_marker is None:
    #         _robot_root_marker_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/RobotRootFrame")
    #         _robot_root_marker_vis.markers["frame"].scale = (0.01, 0.01, 0.01)
    #         _robot_root_marker = VisualizationMarkers(_robot_root_marker_vis)

    #     if _world_frame_marker is None:
    #         _world_frame_marker_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/WorldFrame")
    #         _world_frame_marker_vis.markers["frame"].scale = (0.01, 0.01, 0.01)
    #         _world_frame_marker = VisualizationMarkers(_world_frame_marker_vis)

    #     # if _prova_rf_marker is None:
    #     #     _prova_rf_marker_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/ProvaRF")
    #     #     _prova_rf_marker_vis.markers["frame"].scale = (0.015, 0.015, 0.015)  # Slightly larger for visibility
    #     #     _prova_rf_marker = VisualizationMarkers(_prova_rf_marker_vis)

    #     quat_id = torch.zeros((num_envs, 4), device=device, dtype=dtype)
    #     quat_id[:, 0] = 1.0
    #     marker_indices = torch.zeros(num_envs, dtype=torch.int32, device=device)

    #     # Visualize target node (red sphere) - now managed via USD Prim directly
    #     # _target_node_marker.visualize(reach_pos_w, quat_id, marker_indices=marker_indices)
        
    #     _ee_frame_marker.visualize(reach_pos_w, desired_quat_w, marker_indices=marker_indices) # anchor_pos_w
    
    # # # # #     
    # # # # #     # visualize insertion goal REACH + INSERT
    # # # # #     if _insert_marker is None:
    # # # # #         insert_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/DesiredEEFrame_INSERT")
    # # # # #         _insert_marker = VisualizationMarkers(insert_cfg)
    # # # # #     _insert_marker.visualize(insert_pos_w, quat_id, marker_indices=marker_indices)
    # # # # #    

    #     ee_frame: FrameTransformer = env.scene["ee_frame"]
    #     ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    #     ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    #     _curr_ee_marker.visualize(ee_pos_w, ee_quat_w, marker_indices=marker_indices)

    #     # DEBUG: Print transformation between RobotRootFrame and CurrentEEFrame
    #     robot_root_pos = robot.data.root_state_w[:, :3]
    #     robot_root_quat = robot.data.root_state_w[:, 3:7]
    #     # if num_envs > 0:
    #     #     ee_pos_b, ee_quat_b = subtract_frame_transforms(
    #     #         robot_root_pos, robot_root_quat, ee_pos_w, ee_quat_w
    #     #     )
    #     #     pos_rel = ee_pos_b[0]
    #     #     print(f"[Transformation] RobotRootFrame -> CurrentEEFrame:")
    #     #     print(f"  Position (robot frame): [{pos_rel[0]:.6f}, {pos_rel[1]:.6f}, {pos_rel[2]:.6f}]")
    #     #     print(f"  Orientation (wxyz): [{ee_quat_b[0, 0]:.6f}, {ee_quat_b[0, 1]:.6f}, {ee_quat_b[0, 2]:.6f}, {ee_quat_b[0, 3]:.6f}]")
        
    #     _robot_root_marker.visualize(robot_root_pos, robot_root_quat, marker_indices=marker_indices)

    #     # ProvaRF: Robot root frame translated by -0.0565 along Z (in robot frame coordinates)
    #     # Same orientation as robot root, only position is offset
    #     prova_rf_offset_b = torch.tensor([0.0, 0.0, -0.0565], device=device, dtype=dtype).repeat(num_envs, 1)
    #     quat_identity = torch.zeros((num_envs, 4), device=device, dtype=dtype)
    #     quat_identity[:, 0] = 1.0
    #     # Transform offset to world frame, then add to robot root position
    #     prova_rf_pos_w, _ = combine_frame_transforms(
    #         robot_root_pos, robot_root_quat, prova_rf_offset_b, quat_identity
    #     )
    #     prova_rf_quat_w = robot_root_quat  # Same orientation as robot root frame
    #     # _prova_rf_marker.visualize(prova_rf_pos_w, prova_rf_quat_w, marker_indices=marker_indices)
        
    #     # # DEBUG: Print transformation between RobotRootFrame and ProvaRF
    #     # if num_envs > 0:
    #     #     pos_rel = prova_rf_pos_w[0] - robot_root_pos[0]  # Relative position (world frame)
    #     #     print(f"[Transformation] RobotRootFrame -> ProvaRF:")
    #     #     print(f"  Position offset (world): [{pos_rel[0]:.6f}, {pos_rel[1]:.6f}, {pos_rel[2]:.6f}]")
    #     #     print(f"  Relative orientation: Same as RobotRootFrame (identity)")

    #     world_pos = getattr(env.scene, "env_origins", None)
    #     if world_pos is None:
    #         world_pos = torch.zeros_like(robot_root_pos)
    #     _world_frame_marker.visualize(world_pos, quat_id, marker_indices=marker_indices)
    # except Exception as e:
    #     print("[DEBUG] liver target marker error:", e)
    # # END VISUALIZATION
    
        
    
    
    # # REACH + INSERT
    # phase = getattr(env, "_phase", None)
    # if phase is None:
    #     phase = torch.zeros(num_envs, device = device, dtype = torch.int64)
    
    # target_pos_w = torch.where(phase.unsqueeze(-1) == 0, reach_pos_w, insert_pos_w)

    target_pos_w = reach_pos_w
    
    return torch.cat([target_pos_w, desired_quat_w], dim=-1)   # anchor_pos_w, desired_quat_w # INSERT reach_pos_w --> target_pos_w


def insert_lift_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
) -> torch.Tensor:
    """Positive reward equal to vertical lift of the target relative to initial reach position.

    - Uses the same anchor index and dz as `liver_target_pose_world` (anchor_idx=319, dz=0.008).
    - Reward is active only when the environment `phase` is not zero (i.e., insert phase).
    """
    deformable: DeformableObject = env.scene[object_cfg.name]

    # compute current reach position (follow node, including Z)
    anchor_idx = 319
    dz = 0.008
    nodal_pos_w = deformable.data.nodal_pos_w[:, anchor_idx, :3]
    reach_pos_w = nodal_pos_w + torch.tensor([0.0, 0.0, -dz], device=nodal_pos_w.device, dtype=nodal_pos_w.dtype)

    # initial reach position computed from default nodal state (fixed)
    default_anchor_pos = deformable.data.default_nodal_state_w[:, anchor_idx, :3].to(nodal_pos_w.device)
    initial_reach_z = (default_anchor_pos + torch.tensor([0.0, 0.0, -dz], device=default_anchor_pos.device, dtype=default_anchor_pos.dtype))[:, 2]

    # REWARD PROPORTIONAL TO VERTICAL DISPLACEMENT OF REACH POINT:
    # vertical lift (only positive) measured on the reach point
    lift = reach_pos_w[:, 2] - initial_reach_z - 0.0025
    lift = torch.clamp(lift, min=0.0)

    # apply only when in insert phase (phase != 0)
    phase = getattr(env, "_phase", None)
    if phase is None:
        return torch.zeros_like(lift)

    mask = (phase != 0).to(lift.dtype)
    return (lift) * mask # + 0.0012 * mask # 0.03*0.04, lift - 0.0025


    # # CONSTANT REWARD VERSION:
    # # vertical lift measured on the reach point
    # lift = reach_pos_w[:, 2] - initial_reach_z

    # # boolean: did it rise above baseline? (strictly greater)
    # positive = (lift > 0.0).to(lift.dtype)

    # # apply only when in insert phase (phase != 0)
    # phase = getattr(env, "_phase", None)
    # if phase is None:
    #     return torch.zeros_like(lift)

    # mask = (phase != 0).to(lift.dtype)

    # # constant reward when any positive lift is observed
    # constant_reward_value = 1.0
    # return constant_reward_value * positive * mask

def insert_phase_drive_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
) -> torch.Tensor:
    """
    Additional penalty on distance, active ONLY during the Insert phase (phase == 1).
    Used to create a much steeper gradient to push the agent away from the Reach point.
    """
    # Calculates the standard distance (which internally already points to the correct target based on phase)
    dist = object_ee_distance(env, asset_cfg, object_cfg)
    
    phase = getattr(env, "_phase", None)
    if phase is None:
        return torch.zeros_like(dist)
    
    # Mask active only for phase 1
    mask = (phase == 1).float()
    
    # Returns the distance only if we are in insert phase
    return dist * mask

def final_success_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
) -> torch.Tensor:
    from .terminations import achieved_insert_target

    # Se usiamo la funzione di terminazione con la soglia a 0.003 (come sopra)
    # success will be detected even at 0.0021.
    success_mask = achieved_insert_target(env, asset_cfg=asset_cfg, object_cfg=object_cfg).to(torch.float32)

    return 1.0 * success_mask

# # REACH + INSERT
# def final_success_reward(
#     env: "ManagerBasedRLEnv",
#     asset_cfg: SceneEntityCfg,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
# ) -> torch.Tensor:
#     """Constant positive reward when the final insert success termination is achieved.

#     Returns 1.0 for envs where `achieved_insert_target` is True, else 0.0.
#     """
#     from .terminations import achieved_insert_target

#     # compute success mask
#     success_mask = achieved_insert_target(env, asset_cfg=asset_cfg, object_cfg=object_cfg).to(torch.float32)

#     # reward value
#     reward_value = 1.0

#     return reward_value * success_mask


def reach_phase_transition_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
    pos_threshold: float = 0.002,
    or_threshold: float = 0.2,
) -> torch.Tensor:
    """One-time positive reward when the env transitions into the reach-achieved phase (phase==1).

    - Rewards only the first step where position and orientation are within thresholds and phase==1.
    - Resets the per-env bookkeeping when phase returns to 0.
    """
    # compute current errors
    pos_err = object_ee_distance(env, asset_cfg=asset_cfg, object_cfg=object_cfg)
    or_err = object_ee_orientation_error(env, asset_cfg=asset_cfg, object_cfg=object_cfg)

    phase = getattr(env, "_phase", None)
    if phase is None:
        # no phase tracking available, return zeros
        return torch.zeros_like(pos_err)

    device = pos_err.device

    # ensure per-env bookkeeping exists on the env
    if not hasattr(env, "_phase_reach_rewarded"):
        env._phase_reach_rewarded = torch.zeros(env.scene.num_envs, device=device, dtype=torch.bool)

    # reset bookkeeping for envs that returned to phase 0
    reset_mask = (phase == 0)
    if reset_mask.any():
        env._phase_reach_rewarded[reset_mask] = False

    # detect first-time success: phase==1, errors below thresholds, and not yet rewarded
    cond = (phase == 1) & (pos_err < pos_threshold) & (or_err < or_threshold) & (~env._phase_reach_rewarded)

    # create reward tensor (constant when triggered)
    reward_value = 1.0
    reward = reward_value * cond.to(torch.float32)

    # mark rewarded envs so we don't double count
    if cond.any():
        env._phase_reach_rewarded[cond] = True

    return reward


def reach_to_insert_transition_bonus(
    env: "ManagerBasedRLEnv",
    bonus: float = 0.03,
) -> torch.Tensor:
    """
    One-shot bonus reward when phase transitions from REACH (0) to INSERT (1).
    Used to compensate the reward drop due to target discontinuity.
    """

    phase = getattr(env, "_phase", None)
    if phase is None:
        return torch.zeros(env.scene.num_envs, device=env.device)

    device = phase.device
    num_envs = phase.shape[0]

    # bookkeeping tensor (per-env)
    if not hasattr(env, "_reach_insert_bonus_given"):
        env._reach_insert_bonus_given = torch.zeros(
            num_envs, device=device, dtype=torch.bool
        )

    # detect first step in INSERT
    transition = (phase == 1) & (~env._reach_insert_bonus_given)

    reward = bonus * transition.to(torch.float32)

    # mark as given
    if transition.any():
        env._reach_insert_bonus_given[transition] = True

    # reset bookkeeping if env goes back to REACH
    reset = (phase == 0)
    if reset.any():
        env._reach_insert_bonus_given[reset] = False

    return reward


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
    r_range = (r >= 0) & (r <= 50)
    g_range = (g >= 40) & (g <= 165)
    b_range = (b >= 45) & (b <= 165)
    
    # 2. Ensure it's a green-blue tone (not red-dominant)
    is_greenish_blue = (g > r + 20) & (b > r + 20) & (torch.abs(g - b) <= 25)
    
    # 3. Exclude grays (where all channels are similar)
    is_not_gray = (torch.abs(g - r) > 15) | (torch.abs(b - r) > 15)
    
    # 4. Exclude blacks (too dark)
    is_not_black = (r + g + b) > 40
    
    # 5. Exclude whites (too bright)
    is_not_white = (r + g + b) < 380

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
    
    # Basic range for the colors (including lighter tones)
    r_range = (r >= 0) & (r <= 50)
    g_range = (g >= 40) & (g <= 165)
    b_range = (b >= 45) & (b <= 165)
    
    # Ensure it's a green-blue tone (not red-dominant)
    is_greenish_blue = (g > r + 20) & (b > r + 20) & (np.abs(g - b) <= 25)
    
    # Exclude grays (where all channels are similar)
    is_not_gray = (np.abs(g - r) > 15) | (np.abs(b - r) > 15)
    
    # Exclude blacks (too dark)
    is_not_black = (r + g + b) > 40
    
    # Exclude whites (too bright)
    is_not_white = (r + g + b) < 380
    
    # Combine all conditions
    mask_range = r_range & g_range & b_range & is_greenish_blue & is_not_gray & is_not_black & is_not_white
    visible_pixels = np.sum(mask_range > 0)
    return int(visible_pixels)


# def visual_exposure_reward(env):
#     """Visual exposure reward based on gallbladder pixel visibility with warmup period.
    
#     During the first 50 steps of each episode (warmup), returns 0 to avoid learning from
#     initial scene loading artifacts when the deformable object is inserted into the scene.
    
#     Args:
#         env: ManagerBasedRLEnv instance.
    
#     Returns:
#         torch.Tensor: Reward per environment (shape: num_envs), scaled by 1/1000.
#     """
#     # Initialize warmup tracking on first call
#     if not hasattr(visual_exposure_reward, 'warmup_step'):
#         visual_exposure_reward.warmup_step = torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.long)
#         visual_exposure_reward.warmup_duration = 20
#         visual_exposure_reward.baseline = torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32)
#         visual_exposure_reward.baseline_set = torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)
    
#     # Increment warmup counter for all environments
#     visual_exposure_reward.warmup_step += 1
    
#     # Detect episode reset: when env.episode_length_buf == 1, we are in first step of new episode
#     # Note: episode_length_buf increments every step, resets to 1 after termination
#     if hasattr(env, 'episode_length_buf'):
#         reset_happened = env.episode_length_buf == 1
#         visual_exposure_reward.warmup_step[reset_happened] = 1
#         visual_exposure_reward.baseline_set[reset_happened] = False
    
#     # Get camera sensor from scene
#     cam_sensor = env.scene["camera"] 
    
#     # RGB output shape: (num_envs, H, W, 3)
#     rgb_data = cam_sensor.data.output["rgb"]
    
#     # Compute gallbladder detection mask
#     mask = gallbladder_mask_tensor(rgb_data)
    
#     # Count positive pixels for each environment
#     # Sum over spatial dimensions (H, W) -> result shape: (num_envs,)
#     pixel_count = torch.sum(mask, dim=[-1, -2]).float()

#     # Capture baseline at the end of warmup (step == warmup_duration)
#     at_baseline_step = visual_exposure_reward.warmup_step == visual_exposure_reward.warmup_duration
#     if at_baseline_step.any():
#         visual_exposure_reward.baseline[at_baseline_step] = pixel_count[at_baseline_step]
#         visual_exposure_reward.baseline_set[at_baseline_step] = True
    
#     # Normalize reward by dividing by 1000 (e.g., ~600 pixels -> reward ~0.6)
#     # After warmup, reward = current pixels - baseline pixels (captured at warmup end)
#     reward_raw = pixel_count - visual_exposure_reward.baseline
#     reward = reward_raw / 1000.0 
    
#     # Apply warmup: during first 50 steps, zero out the reward
#     in_warmup = visual_exposure_reward.warmup_step <= visual_exposure_reward.warmup_duration
#     reward = reward * (~in_warmup).float()
    
#     return reward



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
    
    # Apply warmup: zero out rewards for first 2 steps of each episode
    # episode_length_buf starts at 1 for the first step, 2 for the second, etc.
    if hasattr(env, 'episode_length_buf'):
        warmup_mask = env.episode_length_buf <= 2  # Steps 0 and 1 (first two steps)
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