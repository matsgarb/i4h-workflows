# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event MDP functions for liver lifting task."""

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, DeformableObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from typing import Sequence

# Global storage for initial liver state
_liver_initial_state = {}
_liver_snapshot_state = {"captured": False, "step_counter": 0, "snapshot": None}


def attach_liver_node_to_tcp(env, env_ids, node_index: int, asset_cfg: SceneEntityCfg, tcp_cfg: SceneEntityCfg):
    """Attach a liver node to the EE position.
    """
    liver = env.scene[asset_cfg.name]
    ee_frame = env.scene[tcp_cfg.name]
    
    # Get current EE position (in world frame) for specified envs 
    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :3] # (1, 3) , 1 = reset env, 3 = x y z
    
    # Get current kinematic target
    kin_target = liver.data.nodal_kinematic_target.clone() # (num_envs, num_nodes, 4), 4 = x y z is_not_kinematic
    
    # Set the node position to EE position
    kin_target[env_ids, node_index, 0:3] = ee_pos_w
    kin_target[env_ids, node_index, 3] = 0.0 # is_not_kinematic = 0.0 (kinematic)
    
    # Write to sim
    liver.write_nodal_kinematic_target_to_sim(kin_target)

def drive_liver_node_to_tcp(env, env_ids, node_index: int, asset_cfg: SceneEntityCfg, tcp_cfg: SceneEntityCfg):
    """Update liver node position to follow the EE at every step.
    """

    liver = env.scene[asset_cfg.name]
    ee_frame = env.scene[tcp_cfg.name]
    
    # Get current EE position (in world frame) for all environments
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :3] # (num_envs, 3) , 3 = x y z
    
    # Get current kinematic target
    kin_target = liver.data.nodal_kinematic_target.clone() 
    
    # Update the node position to follow EE
    kin_target[:, node_index, 0:3] = ee_pos_w
    kin_target[:, node_index, 3] = 0.0 # is_not_kinematic = 0.0 (kinematic)
    
    # Write to sim
    liver.write_nodal_kinematic_target_to_sim(kin_target)

def reset_joints_by_offset_per_joint(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_ranges: Sequence[tuple[float, float]],
    velocity_ranges: Sequence[tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot joints with per-joint offset ranges around defaults.

    Args:
        position_ranges: sequence of (min, max) per joint, len must match dof
        velocity_ranges: optional sequence of (min, max) per joint, same length (defaults to zeros)
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    dof = joint_pos.shape[-1]
    if len(position_ranges) != dof:
        raise ValueError(f"position_ranges len {len(position_ranges)} != dof {dof}")
    if velocity_ranges is None:
        velocity_ranges = [(0.0, 0.0)] * dof
    if len(velocity_ranges) != dof:
        raise ValueError(f"velocity_ranges len {len(velocity_ranges)} != dof {dof}")

    pos_low = torch.tensor([r[0] for r in position_ranges], device=asset.device)
    pos_high = torch.tensor([r[1] for r in position_ranges], device=asset.device)
    vel_low = torch.tensor([r[0] for r in velocity_ranges], device=asset.device)
    vel_high = torch.tensor([r[1] for r in velocity_ranges], device=asset.device)

    offsets_pos = pos_low + (pos_high - pos_low) * torch.rand((len(env_ids), dof), device=asset.device)
    offsets_vel = vel_low + (vel_high - vel_low) * torch.rand((len(env_ids), dof), device=asset.device)

    joint_pos += offsets_pos
    joint_vel += offsets_vel

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def capture_liver_snapshot_once(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
    capture_step: int = 25,
    node_index: int = 491,
):
    """Capture nodal_state_w once at a given step of the first episode.
    """
    if _liver_snapshot_state["captured"]:
        return

    _liver_snapshot_state["step_counter"] += 1
    if _liver_snapshot_state["step_counter"] < capture_step:
        return

    liver: DeformableObject = env.scene[asset_cfg.name]
    if hasattr(liver, "update"):
        try:
            dt = getattr(env.scene, "physics_dt", None) or getattr(env, "physics_dt", None)
            if dt is not None:
                liver.update(dt)
        except Exception:
            pass

    _liver_snapshot_state["snapshot"] = liver.data.nodal_state_w.clone()
    _liver_snapshot_state["captured"] = True


def reset_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_noise_m: float = 0.01,  # ± 1 cm
) -> None:
    """Randomize camera position by ±position_noise_m along each axis."""
    camera = env.scene[asset_cfg.name]

    # Base position (the one defined in OffsetCfg)
    base_pos = torch.tensor([0.1, -0.0, 0.05], device=env.device)  # same as your offset position

    # Uniform noise: shape (len(env_ids), 3)
    noise = (torch.rand((len(env_ids), 3), device=env.device) * 2 - 1) * position_noise_m

    new_pos = base_pos.unsqueeze(0) + noise  # (len(env_ids), 3)

    # Apply the new position while keeping rotation unchanged
    camera.set_world_poses(
        positions=new_pos,
        env_ids=env_ids,
    )


def reset_liver_to_snapshot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
):
    """Reset liver to previously captured snapshot (velocities zero, free kinematic targets)."""
    snap = _liver_snapshot_state.get("snapshot")
    if snap is None:
        return

    liver: DeformableObject = env.scene[asset_cfg.name]
    state_snapshot = snap[env_ids].clone()
    state_snapshot[..., 3:] = 0.0
    vel_zero = torch.zeros_like(liver.data.nodal_vel_w[env_ids])

    liver.write_nodal_state_to_sim(state_snapshot, env_ids=env_ids)
    liver.write_nodal_velocity_to_sim(vel_zero, env_ids=env_ids)

    kin = liver.data.nodal_kinematic_target.clone()
    kin[env_ids, :, :3] = state_snapshot[:, :, :3]
    kin[env_ids, :, 3] = 1.0
    liver.write_nodal_kinematic_target_to_sim(kin[env_ids], env_ids=env_ids)

