from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def __tissue_barycenter_w(object: DeformableObject) -> torch.Tensor:
    """Compute the barycenter of the tissue in world frame."""
    return object.data.root_pos_w

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
    
    
def camera_rgbd_observation(
   env: ManagerBasedRLEnv,
   robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
   """Get RGBD data from the front camera."""
   # Get the camera from the scene
   camera = env.scene.sensors["camera"]
   output = camera.data.output  # Shape: (N, H, W, 3)
   rgb_data = output["rgb"]
   depth_data = output["distance_to_image_plane"]  # Shape: (N, H, W, 1)
   rgbd_data = torch.cat([rgb_data, depth_data], dim=-1)  # Shape: (N, H, W, 4)
   return rgbd_data
