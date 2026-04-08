from __future__ import annotations

import isaaclab.sim as sim_utils
import robotic.surgery.tasks.surgical.liver_retraction.mdp as mdp
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
from isaaclab.managers import SceneEntityCfg
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotic.surgery.assets.psm import PSM_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class PSMReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # simulation settings
        self.viewer.eye = (0.2, 0.2, 0.1)
        self.viewer.lookat = (0.0, 0.0, 0.04)

        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=robotic_surgery_assets.Table,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        )

        # switch robot to PSM
        self.scene.robot = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
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
            close_command_expr={"psm_tool_gripper1_joint": -0.02, "psm_tool_gripper2_joint": 0.02}, # prima 0.1 e -0.1
        )
        # self.commands.object_pose.body_name = "psm_tool_tip_link"
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.03, 0.12, 0), rot=(1, 0, 0, 0)), # pos=(-0.15, 0, 0) liver_and_gallbladder
            spawn=UsdFileCfg(
                usd_path="/home/dvrkteam/Downloads/onlyliver.usd",
                scale=(0.15, 0.15, 0.15),
                deformable_props=DeformableBodyPropertiesCfg(

                    rest_offset=0.0,
                    contact_offset=0.0001,
                    simulation_hexahedral_resolution=8, # if I want my node mesh more similar to object ---> increase this!
                ),
                physics_material=DeformableBodyMaterialCfg(
                    density=10.0,
                    damping_scale=0.5,
                    elasticity_damping=0.2,
                    dynamic_friction=0.25,
                    youngs_modulus=4000,
                    poissons_ratio=0.45,
                ),
            ),
        )

        self.scene.replicate_physics = False
        
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )

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
class PSMReachEnvCfg_PLAY(PSMReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
