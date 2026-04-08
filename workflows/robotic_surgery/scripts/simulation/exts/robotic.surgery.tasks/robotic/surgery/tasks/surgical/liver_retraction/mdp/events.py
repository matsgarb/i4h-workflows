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


# lift_liver
def attach_liver_node_to_tcp(env, env_ids, node_index: int, asset_cfg: SceneEntityCfg, tcp_cfg: SceneEntityCfg):
    """Attach a liver node to the EE position.
    
    Called at reset to set the nodal kinematic target to the current EE position.
    IMPORTANT: specify env_ids to only update the reset envs.
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
    
    This keeps the node attached and moving with the end-effector.
    EventTerm signature requires env_ids parameter even though we update all envs.
    IMPORTANT: not specify env_ids since we update all environments.
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


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    asset = env.scene[asset_cfg.name]

    # DeformableObject does not expose default_root_state; fall back to nodal reset
    if isinstance(asset, DeformableObject):
        # positions
        pos_ranges = torch.tensor(
            [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]], device=asset.device
        )
        rand_pos = math_utils.sample_uniform(pos_ranges[:, 0], pos_ranges[:, 1], (len(env_ids), 3), device=asset.device)
        nodal_state = asset.data.default_nodal_state_w[env_ids].clone()
        nodal_state[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
        nodal_state[..., :3] += rand_pos.unsqueeze(1)
        # velocities (only linear parts make sense for nodal velocities)
        vel_ranges = torch.tensor(
            [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]], device=asset.device
        )
        rand_vel = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 3), device=asset.device)
        nodal_vel = asset.data.nodal_vel_w[env_ids].clone()
        nodal_vel[..., :3] = rand_vel.unsqueeze(1)

        asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
        asset.write_nodal_velocity_to_sim(nodal_vel, env_ids=env_ids)
        return

    # extract the used quantities (to enable type-hinting)
    asset = cast(RigidObject | Articulation, asset)
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_nodal_state_and_targets_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
):
    """Reset deformable nodal state *and* kinematic targets to default (no pinning).

    - Positions sampled within ``position_range`` around ``default_nodal_state_w``.
    - Velocities sampled within ``velocity_range``.
    - Kinematic flags set to 1.0 (free) and positions set to the reset nodal positions.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]

    nodal_state = asset.data.default_nodal_state_w[env_ids].clone()
    # position offsets
    pos_ranges = torch.tensor(
        [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]], device=asset.device
    )
    rand_pos = math_utils.sample_uniform(pos_ranges[:, 0], pos_ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)
    nodal_state[..., :3] += rand_pos

    # velocity offsets
    vel_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]], device=asset.device
    )
    rand_vel = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)
    nodal_vel = asset.data.nodal_vel_w[env_ids].clone()
    nodal_vel[..., :3] = rand_vel

    # write state
    asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
    asset.write_nodal_velocity_to_sim(nodal_vel, env_ids=env_ids)

    # reset kinematic targets: positions to nodal_state, flag=1 (free)
    kin = asset.data.nodal_kinematic_target.clone()
    kin[env_ids, :, :3] = nodal_state[..., :3]
    kin[env_ids, :, 3] = 1.0
    asset.write_nodal_kinematic_target_to_sim(kin, env_ids=env_ids)


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

    # # Debug: log reset joint positions per episode/env
    # try:
    #     env_list = env_ids.tolist()
    #     jp = joint_pos.detach().cpu().numpy()
    #     print(f"[reset_joints] envs={env_list} joint_pos={jp}")
    # except Exception:
    #     pass


def capture_liver_snapshot_once(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
    capture_step: int = 25,
    node_index: int = 491,
):
    """Capture nodal_state_w once at a given step of the first episode.

    Stores the snapshot in a module-global dict so that reset can restore it.
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
    # node_z = liver.data.nodal_pos_w[0, node_index, 2].item()
    # print(f"[SNAPSHOT] Salvato nodal_state_w al passo {capture_step} (altezza nodo{node_index} = {node_z:.6f} m)")



def reset_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_noise_m: float = 0.01,  # ± 1 cm
) -> None:
    """Randomizza la posizione della camera di ± position_noise_m in ogni direzione."""
    camera = env.scene[asset_cfg.name]

    # Posizione base (quella definita nell'OffsetCfg)
    base_pos = torch.tensor([0.1, -0.0, 0.05], device=env.device)  # stessa del tuo offset pos

    # Rumore uniforme: shape (len(env_ids), 3)
    noise = (torch.rand((len(env_ids), 3), device=env.device) * 2 - 1) * position_noise_m

    new_pos = base_pos.unsqueeze(0) + noise  # (len(env_ids), 3)

    # Applica la nuova posizione mantenendo la rotazione invariata
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
    # write only the slice corresponding to env_ids to avoid shape mismatch
    liver.write_nodal_kinematic_target_to_sim(kin[env_ids], env_ids=env_ids)


def reset_liver_to_snapshot_with_randomization(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("liver"),
    position_range: dict[str, tuple[float, float]] = None,
    rotation_range_deg: tuple[float, float] = None,
):
    """Reset liver to snapshot with optional position and rotation randomization.
    
    Args:
        position_range: dict with keys "x", "y", "z" and values (min, max) offset ranges
        rotation_range_deg: tuple of (min_deg, max_deg) for rotation around Z-axis
    """
    snap = _liver_snapshot_state.get("snapshot")
    if snap is None:
        return

    liver: DeformableObject = env.scene[asset_cfg.name]
    state_snapshot = snap[env_ids].clone()
    state_snapshot[..., 3:] = 0.0  # zero velocities
    
    # Apply position randomization
    if position_range is not None:
        pos_ranges = torch.tensor(
            [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]], 
            device=liver.device
        )
        rand_pos = math_utils.sample_uniform(
            pos_ranges[:, 0], pos_ranges[:, 1], 
            (len(env_ids), 1, 3), 
            device=liver.device
        )
        state_snapshot[..., :3] += rand_pos
    
    # Apply rotation randomization (around Z-axis)
    if rotation_range_deg is not None:
        # Convert degrees to radians
        rot_range_rad = (rotation_range_deg[0] * 3.14159 / 180.0, 
                         rotation_range_deg[1] * 3.14159 / 180.0)
        rand_yaw = math_utils.sample_uniform(
            torch.tensor(rot_range_rad[0], device=liver.device),
            torch.tensor(rot_range_rad[1], device=liver.device),
            (len(env_ids),),
            device=liver.device
        )
        # Create quaternions for Z-axis rotation
        sin_half_yaw = torch.sin(rand_yaw / 2.0)
        cos_half_yaw = torch.cos(rand_yaw / 2.0)
        rot_quat = torch.stack([
            cos_half_yaw,
            torch.zeros_like(sin_half_yaw),
            torch.zeros_like(sin_half_yaw),
            sin_half_yaw
        ], dim=-1)  # [w, x, y, z]
        
        # Get current nodal positions (relative to their centroid)
        centroid = state_snapshot[..., :3].mean(dim=1, keepdim=True)
        pos_rel = state_snapshot[..., :3] - centroid
        
        # Apply rotation to each nodal position
        for i in range(len(env_ids)):
            # Apply quaternion rotation using isaaclab's quat_mul
            pos_rotated = _rotate_points_by_quat(pos_rel[i], rot_quat[i], liver.device)
            state_snapshot[i, :, :3] = pos_rotated + centroid[i]
    
    vel_zero = torch.zeros_like(liver.data.nodal_vel_w[env_ids])

    liver.write_nodal_state_to_sim(state_snapshot, env_ids=env_ids)
    liver.write_nodal_velocity_to_sim(vel_zero, env_ids=env_ids)

    kin = liver.data.nodal_kinematic_target.clone()
    kin[env_ids, :, :3] = state_snapshot[..., :3]
    kin[env_ids, :, 3] = 1.0
    liver.write_nodal_kinematic_target_to_sim(kin[env_ids], env_ids=env_ids)


def _rotate_points_by_quat(points: torch.Tensor, quat: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply quaternion rotation to 3D points.
    
    Args:
        points: (N, 3) tensor of 3D points
        quat: (4,) quaternion [w, x, y, z]
    
    Returns:
        rotated_points: (N, 3) tensor of rotated points
    """
    # Convert points to quaternion format (0, x, y, z)
    points_quat = torch.zeros(points.shape[0], 4, device=device)
    points_quat[:, 1:] = points
    
    # Normalize quaternion
    quat_norm = quat / (torch.norm(quat) + 1e-8)
    
    # Compute conjugate
    quat_conj = torch.tensor([quat_norm[0], -quat_norm[1], -quat_norm[2], -quat_norm[3]], device=device)
    
    # Apply rotation: q * p * q^-1
    # First: q * p
    qp = math_utils.quat_mul(quat_norm.unsqueeze(0).expand(points.shape[0], -1), points_quat)
    # Then: (q*p) * q^-1
    qpq_inv = math_utils.quat_mul(qp, quat_conj.unsqueeze(0).expand(points.shape[0], -1))
    
    return qpq_inv[:, 1:4]