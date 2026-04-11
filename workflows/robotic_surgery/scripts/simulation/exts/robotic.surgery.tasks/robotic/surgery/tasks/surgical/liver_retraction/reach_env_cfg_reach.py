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
    
    endoscope_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/endoscope_light",
        spawn=sim_utils.DiskLightCfg(
            color=(1.0, 0.97, 0.93),
            intensity=5000.0,
            radius=0.015,
            enable_color_temperature=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # pos=(-0.03, -0.03, 0.017),
            pos=(-0.02, -0.035, 0.018),
            rot=q_light, # q_light,  # align with camera view, pitched further down to illuminate liver underside
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
        
        # # ===== OPTION 1: STATE BASED RL =====
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # actions = ObsTerm(func=mdp.last_action)
        # def __post_init__(self):
        #     self.enable_corruption = True
        #     self.concatenate_terms = True
        
        # # ===== OPTION 2: IMAGE BASED RL - SINGLE FRAME (OLD) =====
        # camera_rgb = ObsTerm(func=mdp.camera_rgb_observation)
        # def __post_init__(self):
        #     self.enable_corruption = False
        #     self.concatenate_terms = True

        # ===== OPTION 3: IMAGE BASED RL - FRAME STACKING (NEW) =====
        camera_rgbd = ObsTerm(func=mdp.camera_rgb_frame_stack_observation)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
        
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

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    reaching_object = RewTerm(func=mdp.object_ee_distance, weight=-0.8, params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "object_cfg": SceneEntityCfg("liver")})
    ee_orientation = RewTerm(func=mdp.object_ee_orientation_error, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "object_cfg": SceneEntityCfg("liver")})
    
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01, # -0.02, -0.001
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    ee_below_target_penalty = RewTerm(
        func=mdp.ee_below_target_penalty,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "object_cfg": SceneEntityCfg("liver")},
    )
    
    success_bonus = RewTerm(
        func=mdp.success_reached,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "object_cfg": SceneEntityCfg("liver"), "pos_threshold": 0.003, "or_threshold": 0.3},
    )
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
    	func=mdp.achieved_target,
    	params = {
    	"asset_cfg": SceneEntityCfg("robot", body_names = MISSING),
    	"object_cfg": SceneEntityCfg("liver"),
    	"pos_threshold": 0.003, 
    		"or_threshold": 0.3,
    	},
	)


@configclass
class CurriculumCfg:

    """Curriculum terms for the MDP."""
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 25600}
    )
    


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
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
        # # STATE BASED
        # self.episode_length_s = 12
        # IMAGE BASED
        self.episode_length_s = 15
        # simulation settings
        self.sim.dt = 1.0 / 80.0
