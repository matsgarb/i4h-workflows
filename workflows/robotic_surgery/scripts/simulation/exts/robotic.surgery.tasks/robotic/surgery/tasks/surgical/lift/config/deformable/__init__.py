
import gymnasium as gym
import os

# from . import agents



gym.register(
    id="Isaac-Lift-Deformable-PSM-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaDeformableLiftEnvCfg", # FrankaCubeLiftEnvCfg, FrankaDeformableLiftEnvCfg, FrankaTeddyBearLiftEnvCfg
    },
    disable_env_checker=True,
)
