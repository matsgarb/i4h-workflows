# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms 
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from simulation.utils.assets import robotic_surgery_assets
import torch
from . import mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.95)),
    )

    table: AssetBaseCfg = MISSING

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # end-effectors sensor
    ee_frame: FrameTransformerCfg = MISSING

    # target objects
    object: DeformableObjectCfg = MISSING


##
# MDP settings
##


# @configclass
# class CommandsCfg:
#     """Command terms for the MDP."""
#     object_pose=mdp.UniformPoseCommandCfg(
#         asset_name="robot",
#         body_name=MISSING,
#         resampling_time_range=(1.0,1.0),
#         debug_vis=False, 
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             pos_x=(-0.05, -0.05),
#             pos_y=(-0.05, -0.05),
#             pos_z=(-0.12, -0.12),
#             roll=(0.0, 0.0),
#             pitch=(0.0, 0.0),
#             yaw=(0.0, 0.0),
#         ),
#     )
#     object_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
#     object_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_pose = ObsTerm(func=mdp.liver_target_pose_world)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # reset_robot_joints: EventTerm = MISSING # IN THIS WAY THE LIVER IS NOT RESET EVERY TIME
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # reset_object_position = EventTerm(
    #     func=mdp.reset_nodal_state_uniform, # reset_root_state_uniform
    #     mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("object"),"velocity_range": {},"position_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},}, )
    


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    # action penalty
    reaching_object = RewTerm(func=mdp.object_ee_distance, weight=-0.2)
    ee_orientation = RewTerm(func=mdp.object_ee_orientation_error, params={"yaw_deg": 90, "pitch_deg": 50}, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )
    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )

##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 1.0 / 60.0
