# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING
import math
from isaaclab.assets import RigidObject, DeformableObject
from isaaclab.sensors import FrameTransformer
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG, FRAME_MARKER_CFG
_target_marker = None
_ee_frame_marker = None
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def __tissue_barycenter_w(object: DeformableObject) -> torch.Tensor:
    """Compute the barycenter of the tissue in world frame."""
    return object.data.root_pos_w


'''
def object_ee_distance( env: ManagerBasedRLEnv, std: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object"), ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"), robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name] 
    deformable: DeformableObject = env.scene[object_cfg.name] 
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name] 
    
    # --- posedel robot in world --- 
    robot_root_pos_w = robot.data.root_state_w[:, :3]      # (N,3) 
    robot_root_quat_w = robot.data.root_state_w[:, 3:7]    # (N,4) 
    
    # --- punto di riferimento del tessuto (barycenter) --- 
    tissue_pos_w = deformable.data.root_pos_w              # (N,3) 
    
    # --- end-effector in world --- 
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]       # (N,3) 
    
    # --- converti entrambe nel frame root del robot --- 
    tissue_pos_b, _ = subtract_frame_transforms( 
    robot_root_pos_w, robot_root_quat_w, tissue_pos_w 
    ) 
    ee_pos_b, _ = subtract_frame_transforms( 
    robot_root_pos_w, robot_root_quat_w, ee_pos_w 
    ) 
    
    # --- offset locale (nel frame root del robot) --- 
    offset = torch.tensor( 
    [0.03, 0.09, 0.008], 
    device=tissue_pos_b.device, 
    dtype=tissue_pos_b.dtype, 
    ).unsqueeze(0).expand_as(tissue_pos_b) 
    
    # punto target da raggiungere 
    target_pos_b = tissue_pos_b + offset     # (N,3) 
    
    # --- distanza EE–target --- 
    dist = torch.norm(ee_pos_b - target_pos_b, dim=1)   # (N,) 
    
    # --- reward tanh-kernel --- 
    return 1.0 - torch.tanh(dist / std)
'''

'''
def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    deformable: DeformableObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    #  pose robot wrt world RF
    robot_root_pos_w = robot.data.root_state_w[:, :3]      # (N,3)
    robot_root_quat_w = robot.data.root_state_w[:, 3:7]    # (N,4)

    # barycenter of deformable object
    tissue_pos_w = deformable.data.root_pos_w              # (N,3)

    # ee frame
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]       # (N,3)
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    # ee frame and deformable object wrt robot
    tissue_pos_b, _ = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, tissue_pos_w
    )
    ee_pos_b, _ = subtract_frame_transforms(
        robot_root_pos_w, robot_root_quat_w, ee_pos_w
    )

    # offset
    offset = torch.tensor(
        [0.03, 0.09, 0.008],
        device=tissue_pos_b.device,
        dtype=tissue_pos_b.dtype,
    ).unsqueeze(0).expand_as(tissue_pos_b)

    # target pos (wrt robot)
    target_pos_b = tissue_pos_b + offset     # (N,3)

    # # Target visualization
    # global _target_marker
    # try:
    #     if _target_marker is None:
    #         print("[DEBUG] Creating target marker")
    #         marker_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/TargetPoint")

    #         print("[DEBUG] marker keys:", list(marker_cfg.markers.keys()))
    #         for m in marker_cfg.markers.values():
    #             m.scale = (0.02, 0.02, 0.02)

    #         _target_marker = VisualizationMarkers(marker_cfg)
        
    #     target_pos_b_0 = target_pos_b[0:1, :]
    #     robot_pos_w_0 = robot_root_pos_w[0:1, :]
    #     robot_quat_w_0 = robot_root_quat_w[0:1, :]

    #     quat_b_0 = torch.zeros((1, 4), device=tissue_pos_b.device, dtype=tissue_pos_b.dtype)
    #     quat_b_0[:, 0] = 1.0

    #     target_pos_w_0, target_quat_w_0 = combine_frame_transforms(
    #         robot_pos_w_0, robot_quat_w_0, target_pos_b_0, quat_b_0
    #     )

    #     marker_indices = torch.zeros(1, dtype=torch.int32, device=tissue_pos_b.device)
    #     _target_marker.visualize(target_pos_w_0, target_quat_w_0, marker_indices=marker_indices)
    # except Exception as e:
        
    #     print("[DEBUG] marker error:", e)
    #     pass
    
    
    # global _ee_frame_marker
    # if _ee_frame_marker is None:
    #     print("[DEBUG] Creating EE marker")
    #     ee_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame")

    #     for m in ee_marker_cfg.markers.values():
    #         m.scale = (0.02, 0.02, 0.02)

    #     _ee_frame_marker = VisualizationMarkers(ee_marker_cfg)
    
    # ee_pos_w_0 = ee_pos_w[0:1, :]
    # ee_quat_w_0 = ee_quat_w[0:1, :]
    # ee_marker_indices = torch.zeros(1, dtype=torch.int32, device=ee_pos_w.device)
    # _ee_frame_marker.visualize(ee_pos_w_0, ee_quat_w_0, marker_indices=ee_marker_indices)
    # distance ee - target
    dist = torch.norm(ee_pos_b - target_pos_b, dim=1)   # (N,)

    # # contact threshold
    # contact_eps = 0.002
    # contact_mask = dist < contact_eps
    # if contact_mask.any():
    #     idx = contact_mask.nonzero(as_tuple=False)[0, 0].item()
    #     print(f"[DEBUG] EE close to target in env {idx}, dist = {dist[idx].item():.5f} m")

    return 1.0 - torch.tanh(dist / std)
'''

### DISTANCE EE-TARGET: 1 - tanh (distance)
# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     robot: RigidObject = env.scene[robot_cfg.name]
#     deformable: DeformableObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

#     # pose robot wrt world RF
#     robot_root_pos_w = robot.data.root_state_w[:, :3]      # (N,3)
#     robot_root_quat_w = robot.data.root_state_w[:, 3:7]    # (N,4)

#     # EE frame in world
#     ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]       # (N,3)
#     ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]

#     # selection of a node of the obj as target position
#     # nodal_pos_w: (N_env, N_nodes, 3)
#     nodal_pos_w = deformable.data.nodal_pos_w
#     anchor_idx = 319 

#     # target wrt world rf
#     anchor_pos_w = nodal_pos_w[:, anchor_idx, :]

#     # 2) ee and target wrt robot frame
#     anchor_pos_b, _ = subtract_frame_transforms(
#         robot_root_pos_w, robot_root_quat_w, anchor_pos_w
#     )
#     ee_pos_b, _ = subtract_frame_transforms(
#         robot_root_pos_w, robot_root_quat_w, ee_pos_w
#     )
#     target_pos_b = anchor_pos_b     # (N,3)

#     # 3) Target visualization
#     global _target_marker
#     try:
#         if _target_marker is None:
#             marker_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/LiverTargetNode")
#             _target_marker = VisualizationMarkers(marker_cfg)
#         num_envs = target_pos_b.shape[0]
#         quat_b = torch.zeros((num_envs, 4), device = target_pos_b.device, dtype=target_pos_b.dtype)
#         quat_b[:,0] = 1.0

#         target_pos_w, target_quat_w = combine_frame_transforms(
#             robot_root_pos_w, robot_root_quat_w, target_pos_b, quat_b
#         )

#         marker_indices = torch.zeros(num_envs, dtype=torch.int32, device=target_pos_b.device)
#         _target_marker.visualize(target_pos_w, target_quat_w, marker_indices=marker_indices)

#     except Exception as e:
#         print("[DEBUG] target marker error:", e)

#     # 4) distance ee - target
#     dist = torch.norm(ee_pos_b - target_pos_b, dim=1)   

#     # reward: 1 - tanh(dist / std)
#     return 1.0 - torch.tanh(dist / std)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    deformable: DeformableObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # EE frame in world
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :] 

    # selection of a node of the obj as target position
    nodal_pos_w = deformable.data.nodal_pos_w              # (N_env, N_nodes, 3)
    anchor_idx = 319 
    anchor_pos_w = nodal_pos_w[:, anchor_idx, :]           # (N,3)

    # visualization of target node position
    global _target_marker
    try:
        if _target_marker is None:
            marker_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/LiverTargetNode")
            _target_marker = VisualizationMarkers(marker_cfg)
        num_envs = anchor_pos_w.shape[0]
        quat_w = torch.zeros((num_envs, 4), device=anchor_pos_w.device, dtype=anchor_pos_w.dtype)
        quat_w[:, 0] = 1.0
        marker_indices = torch.zeros(num_envs, dtype=torch.int32, device=anchor_pos_w.device)
        _target_marker.visualize(anchor_pos_w, quat_w, marker_indices=marker_indices)
    except Exception as e:
        print("[DEBUG] target marker error:", e)

    # distance error ee position - desired position
    dist = torch.norm(ee_pos_w - anchor_pos_w, dim=1) 
    return dist


def object_ee_orientation_error(
    env: ManagerBasedRLEnv,
    yaw_deg: float = 90.0,
    pitch_deg: float = 50.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    robot: RigidObject = env.scene[robot_cfg.name]
    deformable: DeformableObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # EE frame in world
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]     # (N,4)
    
    num_envs = ee_quat_w.shape[0]
    device = ee_quat_w.device
    dtype = ee_quat_w.dtype

    # target position
    nodal_pos_w = deformable.data.nodal_pos_w              # (N_env, N_nodes, 3)
    anchor_idx = 319
    anchor_pos_w = nodal_pos_w[:, anchor_idx, :]           # (N,3)

    # desired orientation (wrt world ref)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    w_yaw = math.cos(yaw / 2.0)
    z_yaw = math.sin(yaw / 2.0)
    q_yaw = torch.tensor(
        [w_yaw, 0.0, 0.0, z_yaw],
        device=device,
        dtype=dtype,
    )

    w_pitch = math.cos(pitch / 2.0)
    y_pitch = math.sin(pitch / 2.0)
    q_pitch = torch.tensor(
        [w_pitch, 0.0, y_pitch, 0.0],
        device=device,
        dtype=dtype,
    )

    q_des_world_single = quat_mul(q_pitch, q_yaw)        
    desired_quat_w = q_des_world_single.repeat(num_envs, 1)  

    global _ee_frame_marker
    try:
        if _ee_frame_marker is None:
            ee_frame_cfg_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/DesiredEEFrame")
            ee_frame_cfg_vis.markers["frame"].scale = (0.01, 0.01, 0.01)
            _ee_frame_marker = VisualizationMarkers(ee_frame_cfg_vis)

        marker_indices = torch.zeros(num_envs, dtype=torch.int32, device=anchor_pos_w.device)
        _ee_frame_marker.visualize(anchor_pos_w, desired_quat_w, marker_indices=marker_indices)
    except Exception as e:
        print("[DEBUG] desired EE frame marker error:", e)

    ori_err = quat_error_magnitude(ee_quat_w, desired_quat_w)  

    return ori_err



def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: DeformableObject = env.scene[object_cfg.name]
    object_barycenter_w = __tissue_barycenter_w(object)
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_barycenter_w
    )
    return object_pos_b

def liver_target_pose_world(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    deformable: DeformableObject = env.scene[object_cfg.name]

    # target position
    nodal_pos_w = deformable.data.nodal_pos_w          # (N_env, N_nodes, 3)
    anchor_idx = 319
    anchor_pos_w = nodal_pos_w[:, anchor_idx, :]       # (N,3)

    # desired orientation (wrt world ref)
    yaw_deg = 90.0
    pitch_deg = 50.0

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    num_envs = anchor_pos_w.shape[0]
    device = anchor_pos_w.device
    dtype = anchor_pos_w.dtype

    w_yaw = math.cos(yaw / 2.0)
    z_yaw = math.sin(yaw / 2.0)
    q_yaw = torch.tensor(
        [w_yaw, 0.0, 0.0, z_yaw],
        device=device,
        dtype=dtype,
    )

    w_pitch = math.cos(pitch / 2.0)
    y_pitch = math.sin(pitch / 2.0)
    q_pitch = torch.tensor(
        [w_pitch, 0.0, y_pitch, 0.0],
        device=device,
        dtype=dtype,
    )

    q_des_world_single = quat_mul(q_pitch, q_yaw)          # (4,)
    desired_quat_w = q_des_world_single.repeat(num_envs, 1)  # (N,4)
    
    # visualization
    global _target_marker, _ee_frame_marker
    try:
        if _target_marker is None:
            marker_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/LiverTargetNode")
            _target_marker = VisualizationMarkers(marker_cfg)

        if _ee_frame_marker is None:
            ee_frame_cfg_vis = FRAME_MARKER_CFG.replace(prim_path="/Visuals/DesiredEEFrame")
            ee_frame_cfg_vis.markers["frame"].scale = (0.01, 0.01, 0.01)
            _ee_frame_marker = VisualizationMarkers(ee_frame_cfg_vis)

        quat_id = torch.zeros((num_envs, 4), device=device, dtype=dtype)
        quat_id[:, 0] = 1.0

        marker_indices = torch.zeros(num_envs, dtype=torch.int32, device=device)

        _target_marker.visualize(anchor_pos_w, quat_id, marker_indices=marker_indices)
        _ee_frame_marker.visualize(anchor_pos_w, desired_quat_w, marker_indices=marker_indices)
    except Exception as e:
        print("[DEBUG] liver target marker error:", e)
    return torch.cat([anchor_pos_w, desired_quat_w], dim=-1)  # (N,7)


# def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking of the position error using L2-norm.

#     The function computes the position error between the desired position (from the command) and the
#     current position of the asset's body (in world frame). The position error is computed as the L2-norm
#     of the difference between the desired and current positions.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current positions
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
#     curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
#     return torch.norm(curr_pos_w - des_pos_w, dim=1)

