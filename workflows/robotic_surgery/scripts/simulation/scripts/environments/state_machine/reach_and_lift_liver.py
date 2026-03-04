import argparse
from isaaclab.app import AppLauncher

# --- Configurazione Argparse ---
parser = argparse.ArgumentParser(description="Surgical Robotics: Reach and Lift Liver with clean reset.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--world_ref_vis", action="store_true", default=False, help="Enable reference frame visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Lancio dell'app Isaac Sim
app_launcher = AppLauncher(headless=args_cli.headless, livestream=args_cli.livestream, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from datetime import datetime
from PIL import Image

import robotic.surgery.tasks  # noqa: F401
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from robotic.surgery.tasks.surgical.liver_retraction.reach_env_cfg import ReachEnvCfg
from robotic.surgery.tasks.surgical.liver_retraction import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

wp.init()

# --- TARGET NODE INDEX ---
TARGET_NODE_IDX = 433 # 12 res --> 433, 9 res --> 3

# --- WARP STATE MACHINE ---
class ReachSmState:
    REST = wp.constant(0)
    REACH = wp.constant(1)
    LIFT = wp.constant(2)

class ReachSmWaitTime:
    REST = wp.constant(0.5)
    REACH = wp.constant(10.0)
    LIFT = wp.constant(2.0)

@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    des_rest_pose: wp.array(dtype=wp.transform),
    des_reach_pose: wp.array(dtype=wp.transform),    
    des_lift_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    ee_pos: wp.array(dtype=wp.vec3),
    reach_target_pos: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    state = sm_state[tid]
    pos_diff = ee_pos[tid] - reach_target_pos[tid]
    distance_to_target = wp.length(pos_diff)
    
    if state == ReachSmState.REST:
        des_ee_pose[tid] = des_rest_pose[tid]
        if sm_wait_time[tid] >= ReachSmWaitTime.REST:
            sm_state[tid] = ReachSmState.REACH
            sm_wait_time[tid] = 0.0
    elif state == ReachSmState.REACH:
        des_ee_pose[tid] = des_reach_pose[tid]
        if distance_to_target < 0.005 or sm_wait_time[tid] >= ReachSmWaitTime.REACH:
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

    def compute(self, rest_pose: torch.Tensor, reach_pose: torch.Tensor, lift_pose: torch.Tensor, ee_pos: torch.Tensor, reach_target_pos: torch.Tensor):
        rest_pose_wp = wp.from_torch(rest_pose[:, [0, 1, 2, 4, 5, 6, 3]].contiguous(), wp.transform)
        reach_pose_wp = wp.from_torch(reach_pose[:, [0, 1, 2, 4, 5, 6, 3]].contiguous(), wp.transform)
        lift_pose_wp = wp.from_torch(lift_pose[:, [0, 1, 2, 4, 5, 6, 3]].contiguous(), wp.transform)
        ee_pos_wp = wp.from_torch(ee_pos, wp.vec3)
        reach_target_pos_wp = wp.from_torch(reach_target_pos, wp.vec3)
        wp.launch(kernel=infer_state_machine, dim=self.num_envs, inputs=[self.sm_dt_wp, self.sm_state_wp, self.sm_wait_time_wp, rest_pose_wp, reach_pose_wp, lift_pose_wp, self.des_ee_pose_wp, ee_pos_wp, reach_target_pos_wp], device=self.device)
        return self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]

# --- Utility per Maschera Colecisti ---
def get_gallbladder_mask(rgb_image):
    img = rgb_image.astype(np.float32)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    is_not_gray = (np.abs(g - r) > 5) | (np.abs(g - b) > 5)
    mask_range = (r >= 5) & (r <= 110) & (g >= 15) & (g <= 125) & (b >= 3) & (b <= 95)
    green_dominant = (g > r) & (g > b)
    final_mask = (is_not_gray & mask_range & green_dominant).astype(np.uint8) * 255
    return np.sum(final_mask > 0), final_mask

def main():
    # 1. Caricamento Ambiente
    env_cfg: ReachEnvCfg = parse_env_cfg("Isaac-Liver-PSM-IK-Abs-v0", device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make("Isaac-Liver-PSM-v0", cfg=env_cfg)
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # 2. Configurazione Robot (Reset Joints)
    robot_asset = env.unwrapped.scene["robot"]
    NEW_RESET_JOINTS = [0.18, 0.1, 0.01, 0.0, 0.0, 0.0, -0.09, 0.09]
    new_reset_tensor = torch.tensor(NEW_RESET_JOINTS, device=device).repeat(num_envs, 1)
    robot_asset.data.default_joint_pos[:] = new_reset_tensor
    robot_asset.write_joint_state_to_sim(new_reset_tensor, torch.zeros_like(new_reset_tensor))

    # 3. Inizializzazione Fegato (Deformabile)
    liver = env.unwrapped.scene["liver"]
    num_nodes = liver.data.nodal_pos_w.shape[1]
    # Buffer per i target cinematici, verrà aggiornato ogni step partendo dallo stato attuale
    nodal_kinematic_target = liver.data.nodal_kinematic_target.clone()
    # Snapshot nodale da catturare una sola volta (ora gestito via EventTerm in joint_pos_env_cfg)
    # liver_rest_snapshot = None
    
    env.reset()

    # Visualizzazione nodi: se world_ref_vis mostra tutti i nodi, altrimenti solo il 491
    if not args_cli.headless:
        import isaacsim.core.utils.stage as stage_utils
        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdGeom, Gf

        stage = stage_utils.get_current_stage()
        parent_path = "/Visuals/LiverNodesList"
        if not stage.GetPrimAtPath(parent_path):
            UsdGeom.Scope.Define(stage, parent_path)

        nodes_to_spawn = range(num_nodes) if args_cli.world_ref_vis else [TARGET_NODE_IDX]
        for i in nodes_to_spawn:
            node_path = f"{parent_path}/Node_{i}"
            prim = stage.GetPrimAtPath(node_path)
            if not prim.IsValid():
                prim = prim_utils.create_prim(prim_path=node_path, prim_type="Sphere")
            sphere_geom = UsdGeom.Sphere(prim)
            radius = 0.003 if i == TARGET_NODE_IDX else 0.0015
            sphere_geom.GetRadiusAttr().Set(radius)
            color_attr = sphere_geom.GetDisplayColorAttr()
            if not color_attr.HasValue():
                color = Gf.Vec3f(1.0, 0.0, 0.0) if i == TARGET_NODE_IDX else Gf.Vec3f(0.2, 0.2, 0.8)
                color_attr.Set([color])

    # 4. State Machine e Variabili Loop
    reach_sm = ReachSm(env_cfg.sim.dt * env_cfg.decimation, num_envs, device)
    step_idx = 0
    episode_idx = 1
    lift_pos_env_fixed = None
    actions = torch.zeros(env.unwrapped.action_space.shape, device=device)
    actions[:, 3] = 1.0 # Gripper chiuso

    while simulation_app.is_running():
        with torch.inference_mode():
            # --- AGGIORNAMENTO FISICA ---
            step_out = env.step(actions)
            env.unwrapped.scene.update(dt=env_cfg.sim.dt)
            dones = step_out[2] | step_out[3]

            # --- RECUPERO STATO EE E NODI ---
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            ee_pos_w = ee_frame_sensor.data.target_pos_w[..., 0, :].clone()
            ee_quat_w = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            node_491_pos_w = liver.data.nodal_pos_w[:, TARGET_NODE_IDX, :].clone()
            env_origins = env.unwrapped.scene.env_origins

            # --- LOGICA RESET EPISODIO ---
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                env.unwrapped._reset_idx(done_ids)

                # Log joint positions post-reset (episodio corrente)
                joint_pos = robot_asset.data.joint_pos[done_ids].detach().cpu().tolist()
                print(f"EPISODE {episode_idx}, JOINT POS: {joint_pos}")

                # Reset liver da snapshot ora gestito da EventTerm; teniamo il codice per riferimento
                # if liver_rest_snapshot is not None:
                #     state_snapshot = liver_rest_snapshot[done_ids].clone()  # shape: (B, N, 6)
                #     state_snapshot[..., 3:] = 0.0
                #     vel_zero = torch.zeros_like(liver.data.nodal_vel_w[done_ids])
                #     liver.write_nodal_state_to_sim(state_snapshot, done_ids)
                #     liver.write_nodal_velocity_to_sim(vel_zero, done_ids)
                #     kin = liver.data.nodal_kinematic_target[done_ids].clone()
                #     kin[:, :, :3] = state_snapshot[:, :, :3]
                #     kin[:, :, 3] = 1.0
                #     liver.write_nodal_kinematic_target_to_sim(kin, done_ids)

                reach_sm.reset_idx(done_ids)
                lift_pos_env_fixed = None
                actions.zero_()
                step_idx = -1
                episode_idx += 1
                continue

            # --- CAPTURE TARGETS (Step 0) ---
            if step_idx == 0:
                rest_pos_env = ee_pos_w - env_origins
                rest_quat_w = ee_quat_w.clone()
                lift_pos_env_fixed = node_491_pos_w - env_origins
                lift_pos_env_fixed[:, 2] += 0.03 # Lift 3cm

                # Debug: stato liver immediatamente dopo il reset/cattura target
                liver.update(env_cfg.sim.dt)
                pos_err0 = torch.norm(liver.data.nodal_pos_w - liver.data.default_nodal_state_w[..., :3], dim=-1).max().item()
                vel_max0 = torch.norm(liver.data.nodal_vel_w, dim=-1).max().item()
                flag_min0 = liver.data.nodal_kinematic_target[..., 3].min().item()
                flag_max0 = liver.data.nodal_kinematic_target[..., 3].max().item()
                node491_curr = liver.data.nodal_pos_w[:, 26, :3]
                node491_def = liver.data.default_nodal_state_w[:, 26, :3]
                node491_err = torch.norm(node491_curr - node491_def, dim=-1).max().item()
                dist_ee_node491 = torch.norm(ee_pos_w - node491_curr, dim=-1).max().item()
                # print(f"[DEBUG STEP0] pos_err_max={pos_err0:.6e} vel_max={vel_max0:.6e} flag_min={flag_min0:.1f} flag_max={flag_max0:.1f} node491_err={node491_err:.6e} dist_ee_node491={dist_ee_node491:.6e}")

            nodal_kinematic_target = liver.data.nodal_kinematic_target.clone()
            
            distance_to_node = torch.norm(ee_pos_w - node_491_pos_w, dim=-1)
            current_state = reach_sm.sm_state[0].item()
            ATTACHMENT_THRESHOLD = 0.005

            if (distance_to_node < ATTACHMENT_THRESHOLD).any() and current_state > 0:
                nodal_kinematic_target[:, TARGET_NODE_IDX, :3] = ee_pos_w
                nodal_kinematic_target[:, TARGET_NODE_IDX, 3] = 0.0
            
            liver.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
            liver.write_data_to_sim() # Applica i target al solutore PhysX FEM
            if step_idx <= 2:
                liver.update(env_cfg.sim.dt)
                flag_min_step = nodal_kinematic_target[..., 3].min().item()
                flag_max_step = nodal_kinematic_target[..., 3].max().item()
                pos_err_step = torch.norm(liver.data.nodal_pos_w - liver.data.default_nodal_state_w[..., :3], dim=-1).max().item()
                vel_max_step = torch.norm(liver.data.nodal_vel_w, dim=-1).max().item()
                node491_curr = liver.data.nodal_pos_w[:, TARGET_NODE_IDX, :3]
                node491_def = liver.data.default_nodal_state_w[:, TARGET_NODE_IDX, :3]
                node491_err = torch.norm(node491_curr - node491_def, dim=-1).max().item()
                dist_ee_node491 = torch.norm(ee_pos_w - node491_curr, dim=-1).max().item()
                # print(f"[DEBUG STEP {step_idx}] pos_err_max={pos_err_step:.6e} vel_max={vel_max_step:.6e} flag_min={flag_min_step:.1f} flag_max={flag_max_step:.1f} node491_err={node491_err:.6e} dist_ee_node491={dist_ee_node491:.6e}")

            # Snapshot ora gestito da EventTerm; blocco lasciato commentato per riferimento
            # if liver_rest_snapshot is None and step_idx == 25:
            #     liver.update(env_cfg.sim.dt)
            #     liver_rest_snapshot = liver.data.nodal_state_w.clone()
            #     node491_z = liver.data.nodal_pos_w[0, 491, 2].item()
            #     print(f"[SNAPSHOT] Salvato nodal_state_w al passo 25 (altezza nodo491 = {node491_z:.6f} m)")

            # Log altezza nodo 491 rispetto al default (displacement verticale)
            node491_curr = liver.data.nodal_pos_w[:, TARGET_NODE_IDX, 2]
            node491_default = liver.data.default_nodal_state_w[:, TARGET_NODE_IDX, 2]
            disp_z = (node491_curr - node491_default).mean().item()
            # print(f"EPISODIO {episode_idx} PASSO {step_idx+1}: disp_z={disp_z:.6f} m")

            # --- CALCOLO AZIONI (IK) ---
            robot_pos_w = robot_asset.data.root_state_w[:, :3]
            robot_quat_w = robot_asset.data.root_state_w[:, 3:7]
            
            reach_pos_env = node_491_pos_w - env_origins
            
            # Trasformazioni in frame robot
            rest_pos_b, rest_quat_b = subtract_frame_transforms(robot_pos_w, robot_quat_w, rest_pos_env, rest_quat_w)
            reach_pos_b, reach_quat_b = subtract_frame_transforms(robot_pos_w, robot_quat_w, reach_pos_env, rest_quat_w)
            lift_pos_b, lift_quat_b = subtract_frame_transforms(robot_pos_w, robot_quat_w, lift_pos_env_fixed, rest_quat_w)

            pose_rest = torch.cat([rest_pos_b, rest_quat_b], dim=-1)
            pose_reach = torch.cat([reach_pos_b, reach_quat_b], dim=-1)
            pose_lift = torch.cat([lift_pos_b, lift_quat_b], dim=-1)

            actions = reach_sm.compute(pose_rest, pose_reach, pose_lift, ee_pos_w, node_491_pos_w[0])

            # Debugging
            if step_idx % 50 == 0:
                state_names = {0: "REST", 1: "REACH", 2: "LIFT"}
                # print(f"[Step {step_idx}] Stato: {state_names[current_state]} | Dist EE-Nodo491: {distance_to_node[0].item():.5f}m")

            # Aggiorna la posa dei marker nodali (se non headless; tutti se world_ref_vis, altrimenti solo 26)
            if not args_cli.headless and step_idx % 2 == 0:
                from pxr import UsdGeom, Gf
                import isaacsim.core.utils.stage as stage_utils
                stage = stage_utils.get_current_stage()
                liver.update(env_cfg.sim.dt)
                node_ids = range(num_nodes) if args_cli.world_ref_vis else [TARGET_NODE_IDX]
                curr_pos = liver.data.nodal_pos_w[0]
                for i in node_ids:
                    node_path = f"/Visuals/LiverNodesList/Node_{i}"
                    prim = stage.GetPrimAtPath(node_path)
                    if prim.IsValid():
                        xformable = UsdGeom.Xformable(prim)
                        translate_op = next((op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
                        if not translate_op:
                            translate_op = xformable.AddTranslateOp()
                        pos = curr_pos[i].tolist()
                        translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))

            step_idx += 1

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()