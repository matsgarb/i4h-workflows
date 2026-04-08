# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from isaaclab.assets import DeformableObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
import robotic.surgery.tasks.surgical.tissue_retraction.mdp as mdp
from robotic.surgery.tasks.surgical.tissue_retraction.tissue_retraction_env_cfg import TissueRetractionEnvCfg
from simulation.utils.assets import robotic_surgery_assets
from isaaclab.sim.spawners.materials import DeformableBodyMaterialCfg, PreviewSurfaceCfg
from isaaclab.sim.spawners.meshes import MeshSphereCfg, MeshCuboidCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import DeformableObjectCfg
from robotic.surgery.tasks.surgical.liver_retraction.reach_env_cfg import ReachEnvCfg
from simulation.utils.assets import robotic_surgery_assets
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
from isaaclab.sim.spawners.materials import DeformableBodyMaterialCfg, PreviewSurfaceCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotic.surgery.assets.psm import PSM_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class PSMTissueRetractionEnvCfg(TissueRetractionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set PSM as robot
        self.scene.robot = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (PSM)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.15, "psm_tool_gripper2_joint": 0.15}, # prima 0.1 e -0.1
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "psm_tool_tip_link"

        # Set Tissue as object (SPHERE or CUBOID)
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.1, 0.1, 0), rot=(0.9238795, 0, 0, 0.3826834)), # (pos=(0.5, 0.2, 0), rot=(0.9238795, 0, 0, 0.3826834)),  (pos=(2.2, -0.4, 1), rot=(0.9238795, 0, 0, 0.3826834)),
            spawn = sim_utils.MeshSphereCfg(radius = 0.006,deformable_props=DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),visual_material=PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2),),physics_material=DeformableBodyMaterialCfg(youngs_modulus=5000.0, damping_scale=1.0, dynamic_friction=40.0, elasticity_damping=0.005, poissons_ratio=0.4,density=1000.0),)
        )
        
        # self.scene.object = DeformableObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.2, 0, 0), rot=(1, 0, 0, 0)),
        #     spawn=UsdFileCfg(
        #         usd_path="/home/dvrkteam/Downloads/liver_and_gallbladder.usd",
        #         scale=(0.15, 0.15, 0.15),
        #         deformable_props=DeformableBodyPropertiesCfg(
        #             rest_offset=0.0,
        #             contact_offset=0.0001,
        #             simulation_hexahedral_resolution=6,
        #         ),
        #         physics_material=DeformableBodyMaterialCfg(
        #             # Example material params — tune these to change behavior
        #             density=10.0,
        #             damping_scale=0.5,
        #             elasticity_damping=0.2,
        #             dynamic_friction=0.25,
        #             youngs_modulus=4000,
        #             poissons_ratio=0.45,
        #         ),
        #     ),
        # )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )


@configclass
class PSMTissueRetractionEnvCfg_PLAY(PSMTissueRetractionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
