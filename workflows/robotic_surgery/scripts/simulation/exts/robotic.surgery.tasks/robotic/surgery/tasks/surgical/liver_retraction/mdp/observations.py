from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from collections import deque
import torch.nn.functional as F

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


################################################################################
# SINGLE FRAME OBSERVATION 
################################################################################

def camera_rgb_observation(
   env: ManagerBasedRLEnv,
   robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
   """
   Output shape: [batch, 3, 168, 168]
   """
   camera = env.scene.sensors["camera"]
   rgb = camera.data.output["rgb"][..., :3].float()
   rgb /= 255.0
   rgb = rgb.permute(0, 3, 1, 2)  # NCHW
   
   if rgb.shape[-2:] != (168, 168): # resize to (84 x 84) or (168 x 168)
      rgb = torch.nn.functional.interpolate(
         rgb, size=(168, 168), mode='bilinear', align_corners=False # (84 x 84) or (168 x 168)
      )
   return rgb.contiguous()


################################################################################
# ===== FRAME STACKING OBSERVATION =====
################################################################################

# Global buffer to store the last 4 frames per environment
_FRAME_BUFFERS = {}


def _get_or_create_buffer(env_id: int, batch_size: int, device: torch.device) -> dict:
    """
    Create or retrieve the frame buffer for a given environment.
    
    Args:
        env_id: Unique environment ID
        batch_size: Number of parallel environments
        device: Device (cuda/cpu) to work on
    
    Returns:
        dict: Buffer with deque and metadata
    """
    if env_id not in _FRAME_BUFFERS:
        _FRAME_BUFFERS[env_id] = {
            "deque": deque(maxlen=4),  # Store max 4 frames
            "batch_size": batch_size,
            "device": device,
        }
    return _FRAME_BUFFERS[env_id]


def camera_rgb_frame_stack_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Frame stacking camera observation - 4 concatenated frames.
    
    Paper Reference: "Sim-To-Real Transfer for Visual Reinforcement Learning of 
    Deformable Object Manipulation"
    - Section II.B.3: "The four most recent images are concatenated"
    - Output: 168x168x12 (4 RGB images = 12 channels)
    
    Why frame stacking?
    - The task is a POMDP (Partially Observable Markov Decision Process)
    - Without temporal history, the model cannot infer dynamics
    - With 4 frames: the model sees the direction and velocity of tissue movement
    
    Output shape: [batch_size, 12, 168, 168]
        - 12 channels = 4 frames * 3 RGB channels
        - 168x168 = consistent with single-frame observation
    
    Buffer initialization:
        - At first reset: buffer is filled by repeating the first frame
        - Subsequently: new frames are appended (maxlen=4 maintains order)
    """
    camera = env.scene.sensors["camera"]
    
    # Extract current frame in normalized RGB (0-1)
    rgb_current = camera.data.output["rgb"][..., :3].float() / 255.0  # [batch, H, W, 3]
    batch_size = rgb_current.shape[0]
    device = rgb_current.device
    
    # Create or retrieve buffer for this environment
    env_id = id(env)
    buffer_dict = _get_or_create_buffer(env_id, batch_size, device)
    frame_buffer = buffer_dict["deque"]
    
    # Append current frame to buffer
    frame_buffer.append(rgb_current)
    
    # If buffer is not yet full (first reset), fill with current frame
    while len(frame_buffer) < 4:
        frame_buffer.appendleft(rgb_current)
    
    # Concatenate 4 frames along channel dimension
    # Order: [oldest, ..., newest] → [batch, H, W, 12]
    frames_list = list(frame_buffer)  # [frame_t-3, frame_t-2, frame_t-1, frame_t]
    stacked_rgb = torch.cat(frames_list, dim=-1)  # Concatenate along last dimension (channels)
    
    # Convert from NHWC to NCHW
    stacked_rgb = stacked_rgb.permute(0, 3, 1, 2)  # [batch, 12, H, W]
    
    # Resize to 168x168 (consistent with single-frame observation)
    if stacked_rgb.shape[-2:] != (168, 168):
        stacked_rgb = torch.nn.functional.interpolate(
            stacked_rgb, 
            size=(168, 168), 
            mode='bilinear', 
            align_corners=False
        )
    
    return stacked_rgb.contiguous()  # [batch, 12, 168, 168]

