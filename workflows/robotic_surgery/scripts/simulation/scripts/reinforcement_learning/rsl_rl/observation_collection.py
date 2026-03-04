import pandas as pd
import socket
import struct
import numpy as np

UDP_IP_SEND_OBS = "127.0.0.1" # local host
PORT_SEND_OBS = 5006 # NEW

UDP_OK = "0.0.0.0" # NEW "0.0.0.0"
PORT_OK = 5007  # NEW 

# FILE_PATH = "logs/rsl_rl/psm_reach/2026-02-16_07-56-52/observations_192_data.xlsx"
FILE_PATH = "logs/rsl_rl/psm_reach/2026-02-17_21-21-55/observations_11_data.xlsx"

# Setup Sockets
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # NEW
sock_recv.bind((UDP_OK, PORT_OK)) # NEW Listens for OK from Isaac

print(f"> Loading file: {FILE_PATH}...")  
try:
    df = pd.read_excel(FILE_PATH)
except FileNotFoundError:
    print("ERROR: File not found.")
    exit()

print(f"> Found {len(df)} rows. Synchronized sending (Step-by-Step).")
joint_names = ["Yaw", "Pitch", "Insert", "Wrist_Roll", "Wrist_Pitch", "Wrist_Yaw", "Gripper1", "Gripper2"]

for index, row in df.iterrows():
    # Extraction
    pos_values = []
    vel_values = []
    for name in joint_names:
        pos_values.append(row[f"true_pos_{name}"])
        vel_values.append(row[f"true_vel_{name}"])

    data_list = pos_values + vel_values
    udp_packet = struct.pack('16f', *data_list)
    
    # 1. SEND OBSERVATION
    sock_send.sendto(udp_packet, (UDP_IP_SEND_OBS, PORT_SEND_OBS)) # NEW
    
    # Debug to ensure we're sending the correct data
    np.set_printoptions(precision=5, suppress=True)
    print(f"[Row {index}] Sent pos: {np.array(pos_values[:6])} -> Waiting for OK from Sim...")
    
    # 2. WAIT FOR ISAAC TO FINISH (This guarantees the logic you requested!)
    try:
        ack_data, _ = sock_recv.recvfrom(1024) # NEW
        # If it receives this, Isaac finished the step. The cycle restarts with the next observation.
    except Exception as e:
        print(f"Error: {e}")
        break

print("> End of file.")
sock_send.close()
sock_recv.close() # NEW