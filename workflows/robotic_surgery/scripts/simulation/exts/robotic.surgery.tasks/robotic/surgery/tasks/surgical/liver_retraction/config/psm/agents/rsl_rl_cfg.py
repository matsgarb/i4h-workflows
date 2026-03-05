# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCNNCfg

'''
@configclass
class PSMReachPpoActorCriticCNNCfg(RslRlPpoActorCriticCfg):
    actor_cnn_cfg: dict | None = None
    critic_cnn_cfg: dict | None = None
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False

    # IMAGE BASED RL
    policy = PSMReachPpoActorCriticCNNCfg(
        class_name="ActorCriticCNN",
        init_noise_std=0.5,
        actor_cnn_cfg = {
            "nf": [32, 64, 64], # number of filters
            "k": [8, 4, 3], # kernel sizes
            "s": [4, 2, 1], # strides
        },
        critic_cnn_cfg = {
            "nf": [32, 64, 64], # number of filters
            "k": [8, 4, 3], # kernel sizes
            "s": [4, 2, 1], # strides
        },
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
'''

# # IMAGE BASED RL
# @configclass
# class PSMReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 128
#     max_iterations = 15000
#     save_interval = 2000
#     experiment_name = "psm_reach"
#     run_name = ""
#     resume = False
#     empirical_normalization = False
#     # # IMAGE BASED RL
#     # obs_groups = {
#     #     "policy": ["policy", "dummy_state"],
#     #     "critic": ["policy", "dummy_state"],
#     # }
    
#     # HYBRID IMAGE + STATE BASED RL
#     obs_groups = {
#         "policy": ["camera_rgb", "actions"],  # 2D image + 1D vector (actions, not last_action)
#         "critic": ["camera_rgb", "actions"],
#     }
    
#     policy = RslRlPpoActorCriticCNNCfg(
#         class_name="ActorCriticCNN",
#         init_noise_std=1.0,
#         actor_cnn_cfg={
#             "output_channels": [32, 64, 64], 
#             "kernel_size": [8, 4, 3],       
#             "stride": [4, 2, 1],             
#             "activation": "elu",
#         },
#         critic_cnn_cfg={
#             "output_channels": [32, 64, 64], 
#             "kernel_size": [8, 4, 3],        
#             "stride": [4, 2, 1],            
#             "activation": "elu",
#         },
#         actor_hidden_dims=[512],
#         critic_hidden_dims=[512],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         class_name="PPO",
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=8,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )

# STATE BASED RL
@configclass
class PSMReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 8000
    save_interval = 2000
    experiment_name = "psm_reach"
    run_name = ""
    resume = False
    empirical_normalization = False
    
    ### NEW RSL_RL
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    ###

    policy = RslRlPpoActorCriticCfg(

        ### NEW RSL_RL
        class_name="rsl_rl.modules.actor_critic.ActorCritic",
        ###

        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(

        ## NEW RSL_RL
        class_name="rsl_rl.algorithms.ppo.PPO",
        ###

        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
