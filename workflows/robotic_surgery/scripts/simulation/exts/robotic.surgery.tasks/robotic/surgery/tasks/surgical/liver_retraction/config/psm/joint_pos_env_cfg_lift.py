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
from robotic.surgery.tasks.surgical.liver_retraction.reach_env_cfg_reach import ReachEnvCfg
from simulation.utils.assets import robotic_surgery_assets
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
from isaaclab.sim.spawners.materials import DeformableBodyMaterialCfg, PreviewSurfaceCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotic.surgery.assets.psm import PSM_CFG_REACH  # isort: skip
##
# Environment configuration
##


@configclass
class PSMReachEnvCfg(ReachEnvCfg):
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
        self.scene.robot = PSM_CFG_REACH.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Rotation quaternion for camera
        q_init = torch.tensor((1.0, 0.0, 0.0, 0.0))  # identity
        
        # reach_liver
        q_roll = torch.tensor((0.7071, -0.7071, 0.0, 0.0))  # -90 degrees around X
        q_pitch = torch.tensor((0.9962, 0.0, 0.0872, 0.0))  # 10 degrees around Y
        q_yaw = torch.tensor((0.1736, 0.0, 0.0, -0.9848)) # -160 degrees around Z

        # # lift_liver
        # q_roll = torch.tensor((0.7071, -0.7071, 0.0, 0.0))  # -90 degrees around X
        # q_pitch = torch.tensor((0.9848078, 0, 0.1736482, 0))  # 20 degrees around Y
        # q_yaw = torch.tensor((0.422618, 0, 0, -0.906308)) # -130 degrees around Z
        
        q_final = quat_mul(q_yaw, q_pitch).tolist()


        # # intrinsic camera parameters
        # fx, fy   = 366.50, 274.87 
        # cx, cy   = 160.0, 120.0 
        # W,  H    = 320, 240 
        # intrinsic_matrix = [fx, 0, cx,
        #              0, fy, cy,
        #              0,  0,  1]

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
            # height=240, # height of the image in pixels # 480
            # width=320, # width of the image in pixels. # 640
            height= H, # height of the image in pixels
            width= W, # width of the image in pixels
            data_types=["rgb"], # , "distance_to_image_plane","semantic_segmentation","instance_segmentation_fast","instance_id_segmentation_fast"], # types of data to be produced by the camera
            # spawn=sim_utils.PinholeCameraCfg(
            #     focal_length = 24.0, clipping_range=(0.001, 1.0e5)
            # ),
            spawn = PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix = intrinsic_matrix,
                width            = W,
                height           = H,
                clipping_range   = (0.001, 1.0e5),
                focal_length     = 24.0,
            ),
            # reach_liver
            # offset = TiledCameraCfg.OffsetCfg(pos=(0.1, -0.0, 0.05), rot=q_final, convention="world"), 
            # offset = TiledCameraCfg.OffsetCfg(pos=(0.08, -0.0, 0.048), rot=q_final, convention="world"), # from_intrinsic_matrix, focal_length = 1
            offset = TiledCameraCfg.OffsetCfg(pos=(0.1, -0.0, 0.05), rot=q_final, convention="world"), # from_intrinsic_matrix, focal_length = 24
            # # lift_liver
            # offset = TiledCameraCfg.OffsetCfg(pos=(0.02, -0.0, 0.04), rot=q_final, convention="world"),
        )
        

        # # ############# ONLY LIVER #############
        # # deformable object in the scene
        # self.scene.liver = DeformableObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Liver",
        #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.03, 0.12, 0.0), rot=(1, 0, 0, 0)),
        #     spawn=UsdFileCfg(
        #         usd_path="/home/dvrkteam/Downloads/onlyliver.usd",
        #         scale=(0.15, 0.15, 0.15),
        #         semantic_tags=[("class","liver")],
        #         deformable_props=DeformableBodyPropertiesCfg(

        #             rest_offset=0.0,
        #             contact_offset=0.0001,
        #             simulation_hexahedral_resolution=8, # if I want my node mesh more similar to object ---> increase this!
        #         ),
        #         physics_material=DeformableBodyMaterialCfg(
        #             density=10.0, # 10.0
        #             damping_scale=0.5, # 0.5
        #             elasticity_damping=0.1, # 0.2
        #             dynamic_friction=0.25, # 0.25
        #             youngs_modulus=500, # 4000
        #             poissons_ratio=0.45, # 0.45
        #         ),
        #     ),
        # )
        
        # ############ LIVER + GALLBLADDER | 2 meshes #############
        # # deformable liver in the scene
        # self.scene.liver = DeformableObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Liver",
        #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.1, -0.1, -0.0), rot=(1, 0, 0, 0)),
        #     spawn=UsdFileCfg(
        #         usd_path="/home/dvrkteam/Downloads/liver.usd",
        #         scale=(0.15, 0.15, 0.15),
        #         semantic_tags=[("class","liver")],
        #         deformable_props=DeformableBodyPropertiesCfg(
        #             rest_offset=0.0,
        #             contact_offset=0.001,
        #             simulation_hexahedral_resolution=8, # |9| ## 8 node 319
        #         ),
        #         physics_material=DeformableBodyMaterialCfg(
        #             damping_scale=0.5, # |0.1|
        #             density=10.0, # |10.0| ## 100
        #             dynamic_friction=40, # |40|
        #             elasticity_damping=0.02, # |0.01| ##try 0.05
        #             poissons_ratio=0.45, # |0.45|
        #             youngs_modulus=500, # |300| ## 5000
        #         ),
        #     ),
        # )
        # # deformable gallbladder in the scene
        # self.scene.gallbladder = DeformableObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Gallbladder",
        #     init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.1, -0.1, 0.0), rot=(1, 0, 0, 0)),
        #     spawn=UsdFileCfg(
        #         usd_path="/home/dvrkteam/Downloads/gallbladder.usd",
        #         scale=(0.15, 0.15, 0.15),
        #         semantic_tags=[("class","gallbladder")],
        #         deformable_props=DeformableBodyPropertiesCfg(
        #             rest_offset=0.0,
        #             contact_offset=0.0001,
        #             simulation_hexahedral_resolution=18, # |16| ## 18
        #         ),
        #         physics_material=DeformableBodyMaterialCfg(
        #             damping_scale=1.0, # |1.0|
        #             density=50, # |10|
        #             dynamic_friction=40, # |40|
        #             elasticity_damping=0.001, # |0.001|
        #             poissons_ratio=0.2, # |0.49|
        #             youngs_modulus=2000, # |2000|
        #         ),
        #     ),
        # )
        
        
        ############ LIVER + GALLBLADDER | 1 mesh #############
        # deformable liver in the scene
        self.scene.liver = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Liver",
            # lift_liver = pos=(-0.055, -0.051, -0.0), rot=(1, 0, 0, 0) with final_organs_1.usd
            # (-0.08, -0.08, -0.0), final_organs_4.usd
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.09, -0.07, -0.06), rot=(1, 0, 0, 0)), # pos=(-0.1, -0.1, 0.0), final_organs_1:(-0.07, -0.07, -0.0), final_organs_1:(-0.053, -0.051, -0.0)
            # init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.1, -0.08, -0.06), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path="/home/dvrkteam/Downloads/final_organs_8.usd", # final_organs_4 # final_organs_6, final_organs_7
                # # BEFORE CHECKING PHYSICAL PARAMETERS OF THE MESH MADE OF SYLICON
                # # scale=(0.12, 0.12, 0.12), 
                # scale=(0.1, 0.1, 0.1), # final_organs_5.usd
                # deformable_props=DeformableBodyPropertiesCfg(
                #     rest_offset=0.0,
                #     contact_offset=0.001,
                #     simulation_hexahedral_resolution=9, # |9 -> node 323|, |11 -> node 555|
                # ),
                # physics_material=DeformableBodyMaterialCfg(
                #     damping_scale=0.5,
                #     density=20.0, # |10|, |50, |
                #     dynamic_friction=40, 
                #     elasticity_damping=0.05,
                #     poissons_ratio=0.45,
                #     youngs_modulus=10000, 
                # ),
                
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
        # ===================== REWARDS OVERRIDE =====================
        # ============================================================

        # override rewards and terminations
        # self.rewards.insert_drive.params["asset_cfg"].body_names = ["psm_tool_tip_link"]
        # self.rewards.reaching_object.params["asset_cfg"].body_names = ["psm_tool_tip_link"]
        # self.rewards.ee_orientation.params["asset_cfg"].body_names = ["psm_tool_tip_link"]
        # self.rewards.ee_below_target_penalty.params["asset_cfg"].body_names = ["psm_tool_tip_link"]
        # self.rewards.success_bonus.params["asset_cfg"].body_names = ["psm_tool_tip_link"]
        self.terminations.success.params["asset_cfg"].body_names = ["psm_tool_tip_link"]
        # self.rewards.success_reward.params["asset_cfg"].body_names = ["psm_tool_tip_link"] # REACH + INSERT
        # self.rewards.reach_phase_reward.params["asset_cfg"].body_names = ["psm_tool_tip_link"]
        
        # ============================================================
        # ===================== ACTIONS OVERRIDE =====================
        # ============================================================
        # # override actions
        
        
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
            scale = 0.02, # STATE BASED RL
            # scale=0.015, # IMAGE BASED RL
            use_default_offset=True,
        )
        
        
        '''
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            # scale = 0.015,
            scale = 0.001,
            use_zero_offset=True,
        )
        '''

        # =============================================================
        # ===================== EVENTS MANAGEMENT =====================
        # =============================================================
        
        # "reset" or "interval" (/home/dvrkteam/i4h-workflows/third_party/IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py)
        
        # ############### RESET ROBOT JOINT POSITION

        
        # # reach_liver
        # self.events.reset_robot_joints = EventTerm(
        #     func=mdp.reset_joints_by_scale,
        #     mode="reset",
        #     params={
        #         "position_range": (0.01, 0.1), # before: (0.01, 0.1)
        #         "velocity_range": (0.0, 0.0),
        #     },
        # )
        
        
        # self.events.reset_robot_joints = EventTerm(
        #     func=mdp.reset_joints_by_scale,
        #     mode="reset",
        #     params={
        #         "position_range": (1, 1), # before: (0.01, 0.1)
        #         "velocity_range": (0.0, 0.0),
        #     },
        # )
        
        
        

        # lift_liver
        # self.events.reset_robot_joints = EventTerm(
        #     func=mdp.reset_joints_by_offset, 
        #     mode="reset",
        #     params={
        #         "position_range": (0.0, 0.0), 
        #         "velocity_range": (0.0, 0.0),
        #     },
        # )
               

        # reach_and_lift_liver
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset_per_joint, 
            mode="reset",
            params={
                # per-joint ranges: [yaw, pitch, insertion, roll, pitch, yaw, gripper 1, gripper 2]
                # (-0.2, 0.2) for reach STATE based, (-0.1, 0.1) for reach IMAGE based,  (0, 0) for lift
                "position_ranges": [
                    (-0.2, 0.2),  # yaw
                    (-0.2, 0.2),  # pitch
                    (0, 0.01),  # insertion
                    (-0.2, 0.2),  # tool roll
                    (-0.2, 0.2),  # tool pitch
                    (-0.2, 0.2),  # tool yaw
                    (0, 0),  # gripper 1
                    (0, 0),  # gripper 2
                    # (-0.0, -0.0),  # yaw
                    # (0.0, 0.0),  # pitch
                    # (0, 0),  # insertion
                    # (0, 0),  # tool roll
                    # (0.0, 0.0),  # tool pitch
                    # (-0.0, -0.0),  # tool yaw
                    # (0, 0),  # gripper 1
                    # (0, 0),  # gripper 2
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

        # # # # # # ################# RESET LIVER POSITION
        self.events.reset_liver_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("liver"),
            },
        )
        
        # self.events.reset_gallbladder_position = EventTerm(
        #     func=mdp.reset_nodal_state_uniform,
        #     mode="reset",
        #     params={
        #         "position_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        #         "velocity_range": {},
        #         "asset_cfg": SceneEntityCfg("gallbladder"),
        #     },
        # )
        
        # self.events.capture_liver_snapshot = EventTerm(
        #     func=mdp.capture_liver_snapshot_once,
        #     mode="interval",
        #     interval_range_s=(0.0, 0.0),
        #     params={
        #         "asset_cfg": SceneEntityCfg("liver"),
        #         "capture_step": 25,
        #         "node_index": 433, # 26, # 491 # 3 # 433 final_organs_6
        #     },
        # )
        # self.events.reset_liver_position = EventTerm(
        #     func=mdp.reset_liver_to_snapshot,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("liver"),
        #     },
        # )

        # self.events.reset_camera_pose = EventTerm(
        #     func=mdp.reset_camera_pose,
        #     mode="reset",                      # si attiva ad ogni reset episodio
        #     params={
        #         "asset_cfg": SceneEntityCfg("camera"),
        #         "position_noise_m": 0.01,      # ± 1 cm
        #     },
        # )
            
        
        # # Randomize liver position and rotation at each reset
        # self.events.reset_liver_position = EventTerm(
        #     func=mdp.reset_nodal_state_uniform,
        #     mode="reset",
        #     params={
        #         "position_range": {"x": (-0.0, -0.0), "y": (-0.02, -0.02), "z": (0.0, 0.0)},
        #         "velocity_range": {},
        #         "asset_cfg": SceneEntityCfg("liver"),
        #     },
        # )
        
        
        # # ========== ATTACH LIVER NODE TO EE POSITION ==========
        # # lift_liver - attach node at reset
        # self.events.attach_liver_node = EventTerm(
        #     func=mdp.attach_liver_node_to_tcp,
        #     mode="reset",
        #     params={
        #         "node_index": 433, # 433 final_organs_7 # 3, # 26, # 491, 323 final_organs_1
        #         "asset_cfg": SceneEntityCfg("liver"),
        #         "tcp_cfg": SceneEntityCfg("ee_frame"),
        #     },
        # )
        # self.events.drive_liver_node = EventTerm( # keep node attached to EE at every step
        #     func=mdp.drive_liver_node_to_tcp,
        #     mode="interval",
        #     interval_range_s=(0.001, 0.001),
        #     params={
        #         "node_index": 433, # 3 #26, # 491, 323 final_organs_1
        #         "asset_cfg": SceneEntityCfg("liver"),
        #         "tcp_cfg": SceneEntityCfg("ee_frame"),
        #     },
        # )
        

         

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
class PSMReachEnvCfg_PLAY(PSMReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
