# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the da Vinci Research Kit (dVRK) Patient Side Manipulator (PSM) robots.

The following configurations are available:

* :obj:`PSM_CFG`: dVRK PSM robot arm
* :obj:`PSM_HIGH_PD_CFG`: dVRK PSM robot arm with stiffer PD control

Reference: https://github.com/med-air/SurRoL
           https://github.com/WPI-AIM/dvrk_env
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from simulation.utils.assets import robotic_surgery_assets
from isaaclab.utils.math import quat_mul
import math
import torch


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

# 45° around Z, 60° around Y (w, x, y, z)
q_z = (math.cos(math.radians(45)/2), 0.0, 0.0, math.sin(math.radians(45)/2))
q_y = (math.cos(math.radians(60)/2), 0.0, math.sin(math.radians(60)/2), 0.0)
q_total = quat_mul_wxyz(q_z, q_y)


PSM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=robotic_surgery_assets.dVRK_PSM,
        activate_contact_sensors=False,
        semantic_tags=[("class","robot")],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # reach_liver
        joint_pos={
            "psm_yaw_joint":  0.0, # -0.18915,#
            "psm_pitch_end_joint": 0.0, # -0.48114, #
            "psm_main_insertion_joint":  0.0565, # 0.08451, #
            "psm_tool_roll_joint": 0.0, # -0.48192, #
            "psm_tool_pitch_joint": 0.0, # 0.19863, #
            "psm_tool_yaw_joint":  0.0, # 0.37562, #
            "psm_tool_gripper1_joint": -0.09, 
            "psm_tool_gripper2_joint": 0.09, 
        },

        # # lifting liver 
        # joint_pos={
        #     "psm_yaw_joint":  -0.056,
        #     "psm_pitch_end_joint": -0.36, 
        #     "psm_main_insertion_joint":  0.095,
        #     "psm_tool_roll_joint": -1.4,
        #     "psm_tool_pitch_joint": -0.0256,
        #     "psm_tool_yaw_joint": 1.3992,
        #     "psm_tool_gripper1_joint": -0.09, 
        #     "psm_tool_gripper2_joint": 0.09, 
        # },

       
        # joint_pos={
        #     "psm_yaw_joint": 0.01,
        #     "psm_pitch_end_joint": 0.01,
        #     "psm_main_insertion_joint": 0.07,
        #     "psm_tool_roll_joint": 0.01,
        #     "psm_tool_pitch_joint": 0.01,
        #     "psm_tool_yaw_joint": 0.01,
        #     "psm_tool_gripper1_joint": -0.09, 
        #     "psm_tool_gripper2_joint": 0.09, 
        #},
        
        
        # lift_liver (reference values kept for manual tweaking)
        # joint_pos={
        #     "psm_yaw_joint": 0.115, # final_organs_2: 0.20, # final_organs_1: 0.18
        #     "psm_pitch_end_joint": -0.0145, # final_organs_1: 0.1,
        #     "psm_main_insertion_joint": 0.105, # final_organs_1: 0.105, # 0.11 for state machine, 0.1 for RL
        #     "psm_tool_roll_joint": 0.0,
        #     "psm_tool_pitch_joint": 0.0,
        #     "psm_tool_yaw_joint": 0.0,
        #     "psm_tool_gripper1_joint": -0.09,
        #     "psm_tool_gripper2_joint": 0.09,
        # },
        # pos=(0.02, 0.02, 0.08), # original pos: 0.0, 0.0, 0.15 ------- 0.02, 0.02, 0.08
        pos=(-0.03, 0.0, 0.1), 
        # pos=(0.0, 0.0, 0.1),
        # rot=(q_total), # original orient: (1.0, 0.0, 0.0, 0.0) ------- q_total 
        rot=(0.2334, 0.0, 0.0, 0.9723),  # 153° rotation around Z (63° + 90°) 
        # pos=(0.0, 0.0, 0.15),
        # rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "psm": ImplicitActuatorCfg(
            joint_names_expr=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            effort_limit_sim=50.0, # |12.0|
            effort_limit=50.0, # |12.0|
            velocity_limit_sim=0.1, # |0.1|
            velocity_limit=0.1, # |0.1|
            # velocity_limit_sim=0.5,
            stiffness=50000.0, # |800.0|
            damping=100, # |40.0|
        ),
        "psm_tool": ImplicitActuatorCfg(
            joint_names_expr=["psm_tool_gripper.*"],
            effort_limit_sim=0.1, # |0.1| 
            effort_limit=0.1, # |0.1| 
            velocity_limit_sim=0.1, # |0.1|
            stiffness=500, # |500|
            damping=0.1, # |0.1|
            # friction=20, #10
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of dVRK PSM robot arm."""


PSM_HIGH_PD_CFG = PSM_CFG.copy()
PSM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# PSM_HIGH_PD_CFG.actuators["psm"].stiffness = 800.0
# PSM_HIGH_PD_CFG.actuators["psm"].damping = 40.0 
# PSM_HIGH_PD_CFG.actuators["psm"].stiffness = 5000.0 
# PSM_HIGH_PD_CFG.actuators["psm"].damping = 500.0 
"""Configuration of dVRK PSM robot arm with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
