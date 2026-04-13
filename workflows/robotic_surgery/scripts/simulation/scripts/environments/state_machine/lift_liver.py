import argparse
from isaaclab.app import AppLauncher

# Argparse configuration
parser = argparse.ArgumentParser(description="REST and LIFT state machine: Hard-coded reset override.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--world_ref_vis", action="store_true", default=False, help="Enable reference frame visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(headless=args_cli.headless, livestream=args_cli.livestream, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import warp as wp
import robotic.surgery.tasks  # noqa: F401
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from robotic.surgery.tasks.surgical.liver_retraction.reach_env_cfg import ReachEnvCfg
from robotic.surgery.tasks.surgical.liver_retraction import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from PIL import Image
import os
from datetime import datetime
wp.init()

# --- WARP STATE MACHINE KERNEL ---
class ReachSmState:
    REST = wp.constant(0)
    LIFT = wp.constant(1)

class ReachSmWaitTime:
    REST = wp.constant(2.0)
    LIFT = wp.constant(2.0)

@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    des_rest_pose: wp.array(dtype=wp.transform),
    des_lift_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    state = sm_state[tid]
    
    if state == ReachSmState.REST:
        des_ee_pose[tid] = des_rest_pose[tid]
        if sm_wait_time[tid] >= ReachSmWaitTime.REST:
            sm_state[tid] = ReachSmState.LIFT
            sm_wait_time[tid] = 0.0
            
    elif state == ReachSmState.LIFT:
        des_ee_pose[tid] = des_lift_pose[tid]

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

class ReachSm:
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)

    def reset_idx(self, env_ids=None):
        if env_ids is None: env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, rest_pose: torch.Tensor, lift_pose: torch.Tensor):
        rest_pose_wp = wp.from_torch(rest_pose[:, [0, 1, 2, 4, 5, 6, 3]].contiguous(), wp.transform)
        lift_pose_wp = wp.from_torch(lift_pose[:, [0, 1, 2, 4, 5, 6, 3]].contiguous(), wp.transform)
        wp.launch(kernel=infer_state_machine, dim=self.num_envs, inputs=[self.sm_dt_wp, self.sm_state_wp, self.sm_wait_time_wp, rest_pose_wp, lift_pose_wp, self.des_ee_pose_wp], device=self.device)
        return self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]

# def get_gallbladder_mask(rgb_image, step_idx):
#     if not hasattr(get_gallbladder_mask, "prev_pixels"):
#         get_gallbladder_mask.prev_pixels = 0
#     img = rgb_image.astype(np.float32)
#     r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
#     is_not_gray = (np.abs(g - r) > 5) | (np.abs(g - b) > 5)
#     mask_range = (r >= 5) & (r <= 110) & \
#                  (g >= 15) & (g <= 125) & \
#                  (b >= 3)  & (b <= 95)
#     green_dominant = (g > r) & (g > b)

#     final_mask = (is_not_gray & mask_range & green_dominant).astype(np.uint8) * 255
#     current_pixels = np.sum(final_mask > 0)
#     WARMUP_STEPS = 50  
#     if step_idx <= WARMUP_STEPS:
#         get_gallbladder_mask.prev_pixels = current_pixels
#         delta = 0 
#     else:
#         delta = current_pixels - get_gallbladder_mask.prev_pixels
#         get_gallbladder_mask.prev_pixels = current_pixels
#     return delta, final_mask

def get_gallbladder_mask(rgb_image):
    img = rgb_image.astype(np.float32)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    r_range = (r >= 0) & (r <= 50)
    g_range = (g >= 40) & (g <= 165)
    b_range = (b >= 45) & (b <= 165)
    
    # Ensure it's a green-blue tone (not red-dominant)
    is_greenish_blue = (g > r + 20) & (b > r + 20) & (np.abs(g - b) <= 25)
    
    # Exclude grays (where all channels are similar)
    is_not_gray = (np.abs(g - r) > 15) | (np.abs(b - r) > 15)
    
    # Exclude blacks (too dark)
    is_not_black = (r + g + b) > 40
    
    # Exclude whites (too bright)
    is_not_white = (r + g + b) < 380
    
    # Combine all conditions
    mask_range = r_range & g_range & b_range & is_greenish_blue & is_not_gray & is_not_black & is_not_white
    final_mask = mask_range.astype(np.uint8) * 255
    visible_pixels = np.sum(final_mask > 0)
    return visible_pixels, final_mask


def main():
    # 1. Load config
    env_cfg: ReachEnvCfg = parse_env_cfg("Isaac-Liver-PSM-IK-Abs-v0", device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    
    # # 2. Define the exact joint positions we want (8 joints = yaw, pitch, insertion, tool_roll, tool_pitch, tool_yaw, gripper1, gripper2)
    # NEW_RESET_JOINTS = [0.18, 0.1, 0.11, 0.0, 0.0, 0.0, -0.09, 0.09] 
    # # This overrides the 'default' position used by mdp.reset_joints_by_scale
    # # ----- DIRECT KINEMATIC SETTING OF JOINT POSITIONS -----
    # new_reset_tensor = torch.tensor(NEW_RESET_JOINTS, device=args_cli.device).repeat(args_cli.num_envs, 1) 
    # robot_asset.data.default_joint_pos[:] = new_reset_tensor
    # zeros_vel = torch.zeros_like(new_reset_tensor)
    # robot_asset.write_joint_state_to_sim(new_reset_tensor, zeros_vel)
    # # --------------------------------------------------
    # # Manually move the robot
    # zeros_vel = torch.zeros_like(new_reset_tensor)
    # robot_asset.write_joint_state_to_sim(new_reset_tensor, zeros_vel)
    
    # 3. Create Environment
    env = gym.make("Isaac-Liver-PSM-v0", cfg=env_cfg)
   
    # # ----------- visualize liver nodes
    # import isaacsim.core.utils.stage as stage_utils
    # liver = env.unwrapped.scene["liver"]
    # nodal_pos_initial = liver.data.nodal_pos_w[0]
    # num_nodes = nodal_pos_initial.shape[0]
    # # ---------------------------------
    env.reset()

    # # create output directory for camera images
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = f"camera_images_{timestamp}"
    # os.makedirs(output_dir, exist_ok=True)

    camera = env.unwrapped.scene.sensors["camera"]

    robot_asset = env.unwrapped.scene["robot"]

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    
    if not args_cli.headless and args_cli.world_ref_vis:
        m_cfg = FRAME_MARKER_CFG.copy()
        m_cfg.markers["frame"].scale = (0.005, 0.005, 0.005)
        rest_vis = VisualizationMarkers(m_cfg.replace(prim_path="/Visuals/RestFrame"))
        lift_vis = VisualizationMarkers(m_cfg.replace(prim_path="/Visuals/LiftFrame"))
        ee_vis = VisualizationMarkers(m_cfg.replace(prim_path="/Visuals/EEFrameActual"))
        
        # # --- visualize liver nodes ---
        # import isaacsim.core.utils.prims as prim_utils
        # from pxr import UsdGeom, Gf
        # stage = stage_utils.get_current_stage()
        # parent_path = "/Visuals/LiverNodesList"
        # if not stage.GetPrimAtPath(parent_path):
        #     UsdGeom.Scope.Define(stage, parent_path)
        # for i in range(num_nodes):
        #     node_path = f"{parent_path}/Node_{i}"
        #     prim = prim_utils.create_prim(prim_path=node_path, prim_type="Sphere")
        #     sphere_geom = UsdGeom.Sphere(prim)
        #     sphere_geom.GetRadiusAttr().Set(0.0015)
        #     color_attr = sphere_geom.GetDisplayColorAttr()
        #     if not color_attr.HasValue():
        #         color_attr.Set([Gf.Vec3f(1.0, 0.0, 0.0)]) 
        # # -----------------------------------------------

    reach_sm = ReachSm(env_cfg.sim.dt * env_cfg.decimation, num_envs, device)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=device)
    actions[:, 3] = 1.0 

    step_idx = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            step_out = env.step(actions)
            env.unwrapped.scene.update(dt=env_cfg.sim.dt)
            
            # Keep liver node attached to EE
            # mdp.drive_liver_node_to_tcp(env.unwrapped, 323, SceneEntityCfg("liver"), SceneEntityCfg("ee_frame"))
            
            dones = step_out[2] | step_out[3]

            robot_pos_w = robot_asset.data.root_state_w[:, :3]
            robot_quat_w = robot_asset.data.root_state_w[:, 3:7]
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            ee_pos_w = ee_frame_sensor.data.target_pos_w[..., 0, :].clone()
            ee_quat_w = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            env_origins = env.unwrapped.scene.env_origins

            # # ========== ATTACH LIVER NODE TO EE POSITION ==========
            # tissue_nodal_kinematic_target = liver.data.nodal_kinematic_target.clone()
            # target_node_idx = 323
            # tissue_nodal_kinematic_target[:, target_node_idx, 0] = ee_pos_w[:, 0]
            # tissue_nodal_kinematic_target[:, target_node_idx, 1] = ee_pos_w[:, 1]
            # tissue_nodal_kinematic_target[:, target_node_idx, 2] = ee_pos_w[:, 2]

            # tissue_nodal_kinematic_target[:, target_node_idx, 3] = 0.0
            # liver.write_nodal_kinematic_target_to_sim(tissue_nodal_kinematic_target)
            # # ======================================================

            # # --- visualize liver nodes ---
            # if not args_cli.headless and args_cli.world_ref_vis and step_idx % 2 == 0:
            #     from pxr import Gf, UsdGeom
            #     current_nodal_pos_w = liver.data.nodal_pos_w[0] 
            #     curr_stage = stage_utils.get_current_stage()
            #     for i in range(num_nodes):
            #         node_path = f"/Visuals/LiverNodesList/Node_{i}"
            #         pos = current_nodal_pos_w[i].tolist()
            #         prim = curr_stage.GetPrimAtPath(node_path)
            #         if prim.IsValid():
            #             xformable = UsdGeom.Xformable(prim)
            #             translate_op = next((op for op in xformable.GetOrderedXformOps() 
            #                                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
            #             if not translate_op:
            #                 translate_op = xformable.AddTranslateOp()
            #             translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
            # # ----------------------------------------------------

            if step_idx == 0:
                rest_pos_env = ee_pos_w - env_origins
                rest_quat_w = ee_quat_w.clone()
                
                # # Auto-generated lift position (offset from rest position)
                # lift_pos_env = rest_pos_env.clone()
                # lift_pos_env[:, 2] += 0.05 
                # lift_quat_w = rest_quat_w.clone()

                # Precise lift position (hard-coded in robot reference frame)
                # Convert from robot RF to world frame using combine_frame_transforms
                from isaaclab.utils.math import combine_frame_transforms
                lift_pos_b = torch.tensor([[-0.01, 0.06, -0.09]], device=device).expand(num_envs, -1)
                # Quaternion in (qw, qx, qy, qz) format
                lift_quat_b = torch.tensor([[0.7423, 0.1969, -0.1670, 0.6184]], device=device).expand(num_envs, -1)
                lift_pos_env, lift_quat_w = combine_frame_transforms(robot_pos_w, robot_quat_w, lift_pos_b, lift_quat_b)

            rest_pos_b, rest_quat_b = subtract_frame_transforms(robot_pos_w, robot_quat_w, rest_pos_env, rest_quat_w)
            lift_pos_b, lift_quat_b = subtract_frame_transforms(robot_pos_w, robot_quat_w, lift_pos_env, lift_quat_w)
            pose_rest = torch.cat([rest_pos_b, rest_quat_b], dim=-1)
            pose_lift = torch.cat([lift_pos_b, lift_quat_b], dim=-1)
            
            # Print lift position and quaternion in robot reference frame at each step
            if step_idx % 10 == 0:
                print(f"[Step {step_idx}] LIFT position (robot RF): x={lift_pos_b[0,0]:.4f}, y={lift_pos_b[0,1]:.4f}, z={lift_pos_b[0,2]:.4f}")
                print(f"[Step {step_idx}] LIFT quaternion (robot RF): qx={lift_quat_b[0,0]:.4f}, qy={lift_quat_b[0,1]:.4f}, qz={lift_quat_b[0,2]:.4f}, qw={lift_quat_b[0,3]:.4f}")

            actions = reach_sm.compute(pose_rest, pose_lift)

            if not args_cli.headless and args_cli.world_ref_vis:
                rest_vis.visualize(rest_pos_env + env_origins, rest_quat_w)
                lift_vis.visualize(lift_pos_env + env_origins, lift_quat_w)
                ee_vis.visualize(ee_pos_w, ee_quat_w)

            if dones.any():
                reach_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                step_idx = -1
            

            if camera is not None:
                should_save = (step_idx % 50 == 0)
                if should_save:
                    cam_out = camera.data.output
                    rgb_tensor = cam_out.get("rgb", None)
                    
                    if rgb_tensor is not None:
                        img_rgb = rgb_tensor[0, :, :, :3].cpu().numpy()
                        visible_pixels, gb_mask = get_gallbladder_mask(img_rgb)
                        print(f"[Step {step_idx}] Visible gallbladder pixels: {visible_pixels}")
                        
                        # Print joint positions
                        joint_pos = robot_asset.data.joint_pos[0].cpu().numpy()
                        print(f"[Step {step_idx}] Joint positions: yaw={joint_pos[0]:.4f}, pitch={joint_pos[1]:.4f}, insertion={joint_pos[2]:.4f}, roll={joint_pos[3]:.4f}, pitch_tool={joint_pos[4]:.4f}, yaw_tool={joint_pos[5]:.4f}, gripper1={joint_pos[6]:.4f}, gripper2={joint_pos[7]:.4f}")
                        
                        # Compute and print EE position and quaternion in robot reference frame
                        ee_pos_b, ee_quat_b = subtract_frame_transforms(robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w)
                        print(f"[Step {step_idx}] EE position (robot RF): x={ee_pos_b[0,0]:.4f}, y={ee_pos_b[0,1]:.4f}, z={ee_pos_b[0,2]:.4f}")
                        print(f"[Step {step_idx}] EE quaternion (robot RF): qx={ee_quat_b[0,0]:.4f}, qy={ee_quat_b[0,1]:.4f}, qz={ee_quat_b[0,2]:.4f}, qw={ee_quat_b[0,3]:.4f}")
                        
                        # delta, gb_mask = get_gallbladder_mask(img_rgb, step_idx)
                        # print(f"[Step {step_idx}] Pixel delta: {delta:+4d}")
                        
                        # Test visual_exposure_reward function
                        try:
                            reward = mdp.visual_exposure_reward(env.unwrapped)
                            print(f"[Step {step_idx}] Visual Exposure Reward: {reward.item():.6f}")
                        except Exception as e:
                            print(f"[Step {step_idx}] Error computing visual_exposure_reward: {e}")
                        
                        # mask_img = Image.fromarray(gb_mask)
                        # mask_filename = f"{output_dir}/step{step_idx:05d}_mask_gb.png"
                        # mask_img.save(mask_filename)

                        # rgb_img = Image.fromarray(img_rgb.astype(np.uint8))
                        # rgb_filename = f"{output_dir}/step{step_idx:05d}_rgb.png"
                        # rgb_img.save(rgb_filename)
                        # print(f"[Step {step_idx}] Saved images to {output_dir}")
            step_idx += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()