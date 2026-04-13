from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.managers import SceneEntityCfg
from .rewards import object_ee_orientation_error, object_ee_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def achieved_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pos_threshold: float = 0.002,
    or_threshold: float = 0.2,
) -> torch.Tensor:
    
    pos_err = object_ee_distance(env, asset_cfg=asset_cfg, object_cfg=object_cfg)
    below_th_pos = pos_err < pos_threshold
    
    or_err = object_ee_orientation_error(env, asset_cfg=asset_cfg, object_cfg=object_cfg)
    below_th_or = or_err < or_threshold
    
    done = below_th_pos & below_th_or

    return done
