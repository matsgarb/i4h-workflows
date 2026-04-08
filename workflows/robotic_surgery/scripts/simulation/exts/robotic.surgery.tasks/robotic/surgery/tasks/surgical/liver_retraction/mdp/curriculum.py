# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum learning functions for the liver retraction task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight after a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)


def modify_reward_weight_after_success_steps(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    weight: float,
    num_steps: int,
) -> None:
    """Curriculum that modifies a reward weight num_steps after the first episode success.

    This version tracks success through dones and applies weight changes accordingly.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The new weight of the reward term.
        num_steps: The number of steps after first success when the change should be applied.
    """
    # Initialize tracking if not present
    if not hasattr(env, "_curriculum_success_step"):
        env._curriculum_success_step = None
    
    if not hasattr(env, "_curriculum_weight_applied"):
        env._curriculum_weight_applied = False
    
    # If we haven't recorded first success yet, check the done conditions
    if env._curriculum_success_step is None:
        # Check if any of the success terminations have been triggered
        # This will be true after an episode ends with success
        try:
            # Access the termination manager to check for success
            dones = env.termination_manager._term_values
            
            # Check if "success" termination has been triggered
            if "success" in dones:
                success_dones = dones["success"]
                if torch.any(success_dones):
                    env._curriculum_success_step = env.common_step_counter
        except Exception:
            pass
    
    # Apply weight change if conditions are met
    if env._curriculum_success_step is not None and not env._curriculum_weight_applied:
        steps_since_success = env.common_step_counter - env._curriculum_success_step
        
        if steps_since_success >= num_steps:
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, term_cfg)
            env._curriculum_weight_applied = True
