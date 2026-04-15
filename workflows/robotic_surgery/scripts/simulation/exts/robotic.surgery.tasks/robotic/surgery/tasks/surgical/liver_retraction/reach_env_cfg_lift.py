# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
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
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz 
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from simulation.utils.assets import robotic_surgery_assets
import torch
from . import mdp

# Light orientation target: X = -90°, Y = 75°, Z = 90° (XYZ convention)
q_light = quat_from_euler_xyz(
    torch.tensor(math.radians(-90.0)), 
    torch.tensor(math.radians(90.0)),  
    torch.tensor(math.radians(-75.0))  
)
q_light = q_light.tolist()

##
# Scene definition
##
@configclass
class LiftSceneCfg(InteractiveSceneCfg):
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
    
    endoscope_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/endoscope_light",
        spawn=sim_utils.DiskLightCfg(
            color=(1.0, 0.97, 0.93),
            intensity=5000.0,
            radius=0.015,
            enable_color_temperature=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.02, -0.035, 0.018),
            rot=q_light,
        )
    )
    
    # target objects
    liver: DeformableObjectCfg = MISSING

    

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # ===== OPTION 1: STATE BASED RL =====
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
        
        # # ===== OPTION 2: IMAGE BASED RL - SINGLE FRAME (OLD) =====
        # camera_rgb = ObsTerm(func=mdp.camera_rgb_observation)
        # def __post_init__(self):
        #     self.enable_corruption = False
        #     self.concatenate_terms = True
        
        # # ===== OPTION 3: IMAGE BASED RL - FRAME STACKING (NEW) =====
        # camera_rgbd = ObsTerm(func=mdp.camera_rgb_frame_stack_observation)
        # def __post_init__(self):
        #     self.enable_corruption = False
        #     self.concatenate_terms = True
        
        # # ===== OPTION 4: HYBRID RL - IMAGE + STATE (CURRENT) =====
        # camera_rgb = ObsTerm(func=mdp.camera_rgb_observation)
        # actions = ObsTerm(func=mdp.last_action)
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # def __post_init__(self):
        #     self.enable_corruption = False
        #     self.concatenate_terms = False  # image (3,168,168) + state vectors are incompatible shapes, CNN handles this internally

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints: EventTerm = MISSING
    reset_liver_position: EventTerm = MISSING 
    attach_liver_node: EventTerm = MISSING
    drive_liver_node: EventTerm = MISSING
    

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # lift_liver
    visual_exposure = RewTerm(func=mdp.visual_exposure_reward, weight=0.005)

    vertical_lifting = RewTerm(
        func=mdp.vertical_lifting_reward,
        weight=10.0,                        
        params={"lateral_penalty_coef": 1.0}, 
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    success = DoneTerm(
        func=mdp.gallbladder_visibility_success,
        params={"pixel_threshold": 27000},
    )


@configclass
class CurriculumCfg:

    """Curriculum terms for the MDP."""
    # visual_exposure = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "visual_exposure", "weight": 0.01, "num_steps": 320000}
    # )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.05, "num_steps": 640000}
    )
    


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: LiftSceneCfg = LiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        # STATE BASED
        self.episode_length_s = 12
        # # IMAGE BASED 
        # self.episode_length_s = 15
        # simulation settings
        self.sim.dt = 1.0 / 80.0
