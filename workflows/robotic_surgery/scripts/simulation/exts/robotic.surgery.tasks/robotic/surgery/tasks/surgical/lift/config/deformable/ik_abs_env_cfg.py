# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg, DeformableBodyMaterialCfg
from isaaclab.sim.schemas import DeformableBodyPropertiesCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR


import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


##
# Rigid object lift environment.
##


@configclass
class FrankaCubeLiftEnvCfg(joint_pos_env_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


##spawn = UsdFileCfg(usd_path="/home/dvrkteam/Downloads/Object.usdc", scale=(0.01, 0.01, 0.01), deformable_props=DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),visual_material=PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2),),)
# Deformable object lift environment.
##


@configclass
class FrankaDeformableLiftEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.2, 0), rot=(0.9238795, 0, 0, 0.3826834)), # (pos=(0.5, 0.2, 0), rot=(0.9238795, 0, 0, 0.3826834)),  (pos=(2.2, -0.4, 1), rot=(0.9238795, 0, 0, 0.3826834)),
            # spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/DeformableTube/tube.usd", scale=(1, 1, 1),),
            # spawn=UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd", scale=(0.01, 0.01, 0.01),),
            # spawn=UsdFileCfg(usd_path="/home/dvrkteam/Downloads/teddy_bear.usd", scale=(0.01, 0.01, 0.01),),
            # spawn = sim_utils.MeshSphereCfg(radius = 0.03,deformable_props=DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),visual_material=PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2),),physics_material=DeformableBodyMaterialCfg(youngs_modulus=5000.0, damping_scale=1.0, dynamic_friction=40.0, elasticity_damping=0.005, poissons_ratio=0.4,density=1000.0),)
            # spawn = sim_utils.MeshCuboidCfg(size=(0.05, 0.05, 0.05),deformable_props=DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),visual_material=PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2),),physics_material=DeformableBodyMaterialCfg(youngs_modulus=5000.0, damping_scale=1.0, dynamic_friction=40.0, elasticity_damping=0.005, poissons_ratio=0.4,),)
            # spawn = UsdFileCfg(usd_path="/home/dvrkteam/Downloads/Torus.usd", scale=(0.03, 0.03, 0.06), deformable_props=DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),)
            spawn=UsdFileCfg(usd_path="/home/dvrkteam/Downloads/final_organs_2.usd", scale=(0.1, 0.1, 0.1),deformable_props=DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001, simulation_hexahedral_resolution=9),
            physics_material=DeformableBodyMaterialCfg(damping_scale=0.5, density=5.0, dynamic_friction=40, elasticity_damping=0.01, poissons_ratio=0.45, youngs_modulus=100,),)
        )


        # self.scene.object = DeformableObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0), rot=(0.9238795, 0, 0, 0.3826834)),
        #     # spawn = sim_utils.MeshCuboidCfg(size=(0.05, 0.05, 0.05),
        #     spawn = sim_utils.MeshSphereCfg(radius = 0.03,
        #         deformable_props=DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
        #         visual_material=PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2),),
        #         physics_material=DeformableBodyMaterialCfg(
        #             youngs_modulus=5000.0,
        #             damping_scale=1.0,
        #             dynamic_friction=40.0,
        #             elasticity_damping=0.005,
        #             poissons_ratio=0.4,),
        #         ),
        # )


        # self.scene.object = DeformableObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0), rot=(0.9238795, 0, 0, 0.3826834)),
        #     spawn = UsdFileCfg(
        #         usd_path="/home/dvrkteam/Downloads/Torus.usd",
        #         scale=(0.03, 0.03, 0.06),
        #         deformable_props=DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001))
        # )

        # self.scene.robot.actuators["panda_hand"].effort_limit = 50.0 # default: 200
        self.scene.robot.actuators["panda_hand"].stiffness = 2e4 # 40.0 # default: 2e3
        # self.scene.robot.actuators["panda_hand"].damping = 10.0 # default: 1e2

 
        self.scene.replicate_physics = False

        # Set events for the specific object type (deformable cube)
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # Remove all the terms for the state machine demo
        # TODO: Computing the root pose of deformable object from nodal positions is expensive.
        #       We need to fix that part before enabling these terms for the training.
        self.terminations.object_dropping = None
        self.rewards.reaching_object = None
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.observations.policy.object_position = None
