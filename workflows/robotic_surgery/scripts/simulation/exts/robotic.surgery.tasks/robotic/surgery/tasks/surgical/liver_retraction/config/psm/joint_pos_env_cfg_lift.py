from __future__ import annotations

from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
import torch
import isaaclab.sim as sim_utils
import robotic.surgery.tasks.surgical.liver_retraction.mdp as mdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import FrameTransformerCfg, CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import DeformableObjectCfg
from robotic.surgery.tasks.surgical.liver_retraction.reach_env_cfg_lift import LiftEnvCfg
from simulation.utils.assets import robotic_surgery_assets
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
from isaaclab.sim.spawners.materials import DeformableBodyMaterialCfg, PreviewSurfaceCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotic.surgery.assets.psm import PSM_CFG_LIFT  # isort: skip
##
# Environment configuration
##


@configclass
class PSMLiftEnvCfg(LiftEnvCfg):
    """PSM-based liver retraction environment configuration."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # =======================================================
        # ===================== SCENE SETUP =====================
        # =======================================================
        
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
        self.scene.robot = PSM_CFG_LIFT.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # reach_liver
        q_pitch = torch.tensor((0.9962, 0.0, 0.0872, 0.0))  
        q_yaw = torch.tensor((0.1736, 0.0, 0.0, -0.9848))
        q_final = quat_mul(q_yaw, q_pitch).tolist()


        # # intrinsic camera parameters
        fx, fy   = 1188.15, 1187.41
        cx, cy   = 562.09,  408.09
        W,  H    = 1280,    720
        intrinsic_matrix = [fx, 0, cx,
                     0, fy, cy,
                     0,  0,  1]

        # camera
        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0, # update period of the sensor buffers (in seconds), if = 0.0 -> update at every step
            history_length = 0, # number of past frames to store in the sensor buffers, defaults to 0 (only the current data is stored)
            height= H, # height of the image in pixels
            width= W, # width of the image in pixels
            data_types=["rgb"], # , "distance_to_image_plane","semantic_segmentation","instance_segmentation_fast","instance_id_segmentation_fast"], # types of data to be produced by the camera
            spawn = PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix = intrinsic_matrix,
                width            = W,
                height           = H,
                clipping_range   = (0.001, 1.0e5),
                focal_length     = 24.0,
            ),
            offset = TiledCameraCfg.OffsetCfg(pos=(0.1, -0.0, 0.05), rot=q_final, convention="world"),
        )
        
        
        ############ LIVER + GALLBLADDER | 1 mesh #############
        # deformable liver in the scene
        self.scene.liver = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Liver",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.09, -0.07, -0.06), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path="/home/dvrkteam/Downloads/final_organs_8.usd", # final_organs_4 # final_organs_6, final_organs_7
                # AFTER CHECKING PHYSICAL PARAMETERS OF THE MESH MADE OF SYLICON
                scale=(0.1, 0.1, 0.1), 
                deformable_props=DeformableBodyPropertiesCfg(
                    rest_offset=0.0,
                    contact_offset=0.001,
                    simulation_hexahedral_resolution=12, 
                ),
                physics_material=DeformableBodyMaterialCfg(
                    damping_scale=0.1,
                    density=1070.0,
                    dynamic_friction=1, 
                    elasticity_damping=0.05,
                    poissons_ratio=0.49,
                    youngs_modulus=82600, 
                ),
            ),
        )
        
        self.scene.replicate_physics = False
       
        # ============================================================
        # ===================== ACTIONS OVERRIDE =====================
        # ============================================================
        # # override actions
        # STATE BASED
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale = 0.02,
            use_default_offset=True,
        )
        
        
        # # IMAGE BASED
        # self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=[
        #         "psm_yaw_joint",
        #         "psm_pitch_end_joint",
        #         "psm_main_insertion_joint",
        #         "psm_tool_roll_joint",
        #         "psm_tool_pitch_joint",
        #         "psm_tool_yaw_joint",
        #     ],
        #     scale = 0.001,
        #     use_zero_offset=True,
        # )

        # =============================================================
        # ===================== EVENTS MANAGEMENT =====================
        # =============================================================
        
        # "reset" or "interval" (/home/dvrkteam/i4h-workflows/third_party/IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py)
        
        # ############### RESET ROBOT JOINT POSITION
        # reach_and_lift_liver
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset_per_joint, 
            mode="reset",
            params={
                # per-joint ranges: [yaw, pitch, insertion, roll, pitch, yaw, gripper 1, gripper 2]
                # (-0.2, 0.2) for reach STATE based, (-0.1, 0.1) for reach IMAGE based,  (0, 0) for lift
                "position_ranges": [
                    (-0.0, -0.0),  # yaw
                    (0.0, 0.0),  # pitch
                    (0, 0),  # insertion
                    (0, 0),  # tool roll
                    (0.0, 0.0),  # tool pitch
                    (-0.0, -0.0),  # tool yaw
                    (0, 0),  # gripper 1
                    (0, 0),  # gripper 2
                ],
                "velocity_ranges": [
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
            },
        )

        ################# RESET LIVER POSITION
        self.events.reset_liver_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("liver"),
            },
        )
        
        # ========== ATTACH LIVER NODE TO EE POSITION ==========
        # lift_liver - attach node at reset
        self.events.attach_liver_node = EventTerm(
            func=mdp.attach_liver_node_to_tcp,
            mode="reset",
            params={
                "node_index": 433, # 433 final_organs_7 # 3, # 26, # 491, 323 final_organs_1
                "asset_cfg": SceneEntityCfg("liver"),
                "tcp_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        self.events.drive_liver_node = EventTerm( # keep node attached to EE at every step
            func=mdp.drive_liver_node_to_tcp,
            mode="interval",
            interval_range_s=(0.001, 0.001),
            params={
                "node_index": 433, # 3 #26, # 491, 323 final_organs_1
                "asset_cfg": SceneEntityCfg("liver"),
                "tcp_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        
        # =============================================================
        # ===================== FRAME TRANSFORMER =====================
        # =============================================================

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
class PSMLiftEnvCfg_PLAY(PSMLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
