#!/usr/bin/env python

import argparse
import numpy as np
import sys
import socket
import struct
import csv
import time
from datetime import datetime

# OPTION 1: Pandas needed to read Excel
import pandas as pd 

# =========================================================================
# ROS/CRTK LIBRARIES FOR OPTION 2
# Uncomment when you move to real robot
# =========================================================================
# import crtk
# import dvrk

if sys.version_info.major < 3:
    input = raw_input

class udp_psm_rl_controller:
    def __init__(self, isaac_ip, port_send_obs, port_recv_act, port_ack_recv=5007, log_file=None, arm_name='PSM3', ral=None):
        self.isaac_ip = isaac_ip
        self.port_send_obs = port_send_obs  # 5006 (To Isaac)
        self.port_recv_act = port_recv_act  # 5005 (From Isaac)
        self.port_ack_recv = port_ack_recv  # 5007 (ACK, Option 1 only)

        # RL Action Processing Parameters
        self.default_position = np.array([0.0, 0.0, 0.0565, 0.0, 0.0, 0.0])
        self.scale = 0.02
        self.max_step_delta = 0.0025

        # Base Socket Initialization (common to both options)
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(('', self.port_recv_act))
        
        # ACK Socket (used only in Option 1)
        self.sock_ack = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_ack.bind(('', self.port_ack_recv))

        # =========================================================================
        # ARM INITIALIZATION FOR OPTION 2 
        # =========================================================================
        """
        print(f'> Configuring RL Controller for {arm_name}')
        self.ral = ral
        self.arm = dvrk.psm(ral=ral, arm_name=arm_name)
        self.sock_recv.settimeout(None) # Safety on real robot -> 0.5
        self.log_file = log_file
        if self.log_file:
            self.csv_file = open(self.log_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'timestamp', 'loop_dt',
                'raw_act_1', 'raw_act_2', 'raw_act_3', 'raw_act_4', 'raw_act_5', 'raw_act_6',
                'target_1', 'target_2', 'target_3', 'target_4', 'target_5', 'target_6',
                'meas_1', 'meas_2', 'meas_3', 'meas_4', 'meas_5', 'meas_6'
            ])
        """

    def run(self):
        # =========================================================================
        # OPTION (1): Observations from Excel, Printed Actions
        # =========================================================================
        FILE_PATH = "logs/rsl_rl/psm_reach/2026-02-16_07-56-52/observations_192_data.xlsx"
        print(f"\n=======================================================")
        print(f"> [OPTION 1 ACTIVE] Loading file: {FILE_PATH}...")
        
        try:
            df = pd.read_excel(FILE_PATH)
        except FileNotFoundError:
            print("ERROR: File not found.")
            sys.exit()

        print(f"> Found {len(df)} rows. Synchronized testing (Step-by-Step).")
        joint_names = ["Yaw", "Pitch", "Insert", "Wrist_Roll", "Wrist_Pitch", "Wrist_Yaw", "Gripper1", "Gripper2"]
        current_position = self.default_position.copy()
        for index, row in df.iterrows():
            # 1. DATA EXTRACTION
            pos_values = []
            vel_values = []
            for name in joint_names:
                pos_values.append(row[f"true_pos_{name}"])
                vel_values.append(row[f"true_vel_{name}"])

            data_list = pos_values + vel_values
            udp_packet = struct.pack('16f', *data_list)
            
            # 2. SEND OBSERVATION TO ISAAC
            self.sock_send.sendto(udp_packet, (self.isaac_ip, self.port_send_obs))
            
            # 3. RECEIVE INFERENCE (Raw Action)
            try:
                data_act, _ = self.sock_recv.recvfrom(1024)
                raw_actions = np.array(struct.unpack('6f', data_act))
            except Exception as e:
                print(f"Error receiving action: {e}")
                break

            # 4. ACTION PROCESSING (Your test_receiver)
            desired_target = self.default_position + (raw_actions * self.scale)
            requested_delta = desired_target - current_position
            clipped_delta = np.clip(requested_delta, -self.max_step_delta, self.max_step_delta)
            safe_target = current_position + clipped_delta
            
            # Update virtual position
            current_position = safe_target.copy()

            # 5. PRINT (Instead of sending to motors)
            np.set_printoptions(precision=5, suppress=True)
            print(f"\n[Row {index}]")
            print(f"  Sent Obs Pos        : {np.array(pos_values[:6])}")
            print(f"  Received Raw Action : {raw_actions}")
            print(f"  Applied Safe Delta  : {clipped_delta}")
            print(f"  Executed Target     : {safe_target}")
            print(f"  -> Waiting for OK (ACK) from Sim...")
            print("-" * 60)

            # 6. SYNCHRONIZATION (Wait for ACK)
            try:
                ack_data, _ = self.sock_ack.recvfrom(1024)
            except Exception as e:
                print(f"Error waiting for ACK: {e}")
                break

        print("> End of Excel simulation.")
        self.sock_send.close()
        self.sock_recv.close()
        self.sock_ack.close()


        # =========================================================================
        # OPTION (2): Observations and Actions on Real Robot (dVRK)
        # =========================================================================
        """
        self.home() # Initialize real robot
        print('\n> [OPTION 2] RL LOOP ACTIVE on REAL ROBOT. Press Ctrl+C to terminate')
        
        loop_rate = self.ral.rate(40.0)
        
        try:
            while not self.ral.is_shutdown():
                t_start = time.time()
                
                # 1. READ STATE FROM REAL ROBOT
                meas_pos = self.arm.measured_jp()
                meas_vel = self.arm.measured_jv()
                jaw_pos = self.arm.measured_jaw()
                jaw_vel = self.arm.measured_jaw_velocity() if hasattr(self.arm, 'measured_jaw_velocity') else np.array([0.0])
                
                pos_8 = np.zeros(8)
                vel_8 = np.zeros(8)
                
                pos_8[0:6] = meas_pos
                pos_8[6] = -jaw_pos[0] if len(jaw_pos)>0 else 0.0
                pos_8[7] = jaw_pos[0] if len(jaw_pos)>0 else 0.0
                
                vel_8[0:6] = meas_vel
                vel_8[6] = -jaw_vel[0] if len(jaw_vel)>0 else 0.0
                vel_8[7] = jaw_vel[0] if len(jaw_vel)>0 else 0.0

                # 2. SEND OBSERVATION TO ISAAC
                obs_data = pos_8.tolist() + vel_8.tolist()
                udp_packet = struct.pack('16f', *obs_data)
                self.sock_send.sendto(udp_packet, (self.isaac_ip, self.port_send_obs))

                # 3. WAIT FOR INFERENCE
                try:
                    data_act, _ = self.sock_recv.recvfrom(1024)
                    raw_actions = np.array(struct.unpack('6f', data_act))
                except socket.timeout:
                    print("  ! Timeout waiting for Isaac Sim. Holding position.")
                    continue
                
                # 4. ACTION PROCESSING
                desired_target = self.default_position + (raw_actions * self.scale)
                requested_delta = desired_target - meas_pos
                clipped_delta = np.clip(requested_delta, -self.max_step_delta, self.max_step_delta)
                safe_target = meas_pos + clipped_delta

                # 5. COMMAND REAL ROBOT
                self.arm.servo_jp(safe_target)

                # 6. LOGGING & TIMING
                dt = time.time() - t_start
                if self.log_file:
                    row = [time.time(), dt]
                    row.extend(raw_actions.tolist())
                    row.extend(safe_target.tolist())
                    row.extend(meas_pos.tolist())
                    self.csv_writer.writerow(row)
                
                np.set_printoptions(precision=4, suppress=True)
                print(f"Meas: {meas_pos} | Act: {raw_actions} | Cmd: {safe_target}", end='\r')
                
                loop_rate.sleep()

        except KeyboardInterrupt:
            print('\n> Keyboard interrupt, stopping...')
        finally:
            self.sock_send.close()
            self.sock_recv.close()
            if self.log_file:
                self.csv_file.close()
                print(f'\n> Log saved to {self.log_file}')
        """

    # =========================================================================
    # ROBOT METHODS
    # =========================================================================
    """
    def home(self):
        self.ral.check_connections()
        print('> Enabling arm')
        if not self.arm.enable(10):
            sys.exit('  ! Unable to enable within 10 seconds')
        print('> Homing')
        if not self.arm.home(10):
            sys.exit('  ! Unable to complete homing within 10 seconds')

        print('> Moving to Default RL Position [0, 0, 0.0565, 0, 0, 0]')
        self.arm.move_jp(self.default_position).wait()
        self.arm.move_jaw(np.array([0.0])).wait()
        print('> Ready for RL Control Loop')
    """

if __name__ == '__main__':
    # =========================================================================
    # MAIN - OPTION (1) simple launch (No ROS)
    # =========================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--isaac-ip', type=str, default='127.0.0.1', help='IP where Isaac Sim is running')
    args = parser.parse_args()

    controller = udp_psm_rl_controller(isaac_ip=args.isaac_ip, port_send_obs=5006, port_recv_act=5005)
    controller.run()

    # =========================================================================
    # MAIN - OPTION (2): launch with ROS/CRTK
    # =========================================================================
    """
    argv = crtk.ral.parse_argv(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, default='PSM3', choices=['PSM1', 'PSM2', 'PSM3'])
    parser.add_argument('-i', '--isaac-ip', type=str, default='127.0.0.1', help='IP where Isaac Sim is running')
    parser.add_argument('-l', '--log-file', type=str, default=None)
    args = parser.parse_args(argv)

    if args.log_file is None:
        args.log_file = 'rl_dvrk_log_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

    ral = crtk.ral('udp_rl_controller')
    controller = udp_psm_rl_controller(isaac_ip=args.isaac_ip, port_send_obs=5006, port_recv_act=5005, port_ack_recv=5007, log_file=args.log_file, arm_name=args.arm, ral=ral)
    ral.spin_and_execute(controller.run)
    """