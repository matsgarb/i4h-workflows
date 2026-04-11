# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


from . import agents, ik_abs_env_cfg_reach, ik_rel_env_cfg_reach, joint_pos_env_cfg_reach, ik_abs_env_cfg_lift, ik_rel_env_cfg_lift, joint_pos_env_cfg_lift

###### REACH ENV CONFIGS ######
##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Liver-Reach-PSM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg_reach.PSMReachEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMReachPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Liver-Reach-PSM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg_reach.PSMReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMReachPPORunnerCfg,
    },
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Liver-Reach-PSM-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg_reach.PSMReachEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMReachPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Liver-Reach-PSM-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg_reach.PSMReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMReachPPORunnerCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Liver-Reach-PSM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg_reach.PSMReachEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMReachPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Liver-Reach-PSM-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg_reach.PSMReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMReachPPORunnerCfg,
    },
    disable_env_checker=True,
)



###### LIFT ENV CONFIGS ######
##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Liver-Lift-PSM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg_lift.PSMLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMLiftPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Liver-Lift-PSM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg_lift.PSMLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMLiftPPORunnerCfg,
    },
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Liver-Lift-PSM-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg_lift.PSMLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMLiftPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Liver-Lift-PSM-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg_lift.PSMLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMLiftPPORunnerCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Liver-Lift-PSM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg_lift.PSMLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMLiftPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Liver-Lift-PSM-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg_lift.PSMLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PSMLiftPPORunnerCfg,
    },
    disable_env_checker=True,
)
