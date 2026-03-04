import socket
import struct
import numpy as np

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

UDP_OK = "127.0.0.1"
PORT_OK = 5007
sock_ok = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Listening on {UDP_IP}:{UDP_PORT}...")
default_position = np.array([0.0, 0.0, 0.0565, 0.0, 0.0, 0.0])
current_position = default_position.copy()
MAX_STEP_DELTA = 0.0025
SCALE = 0.02

while True:
    data, addr = sock.recvfrom(1024)
    # received_target = np.array(struct.unpack('6f', data))
    raw_actions = np.array(struct.unpack('6f', data))
    received_target = default_position + raw_actions * SCALE
    
    # 1. Calculate requested delta
    requested_delta = received_target - current_position
    
    # 2. Clip the delta
    clipped_delta = np.clip(requested_delta, -MAX_STEP_DELTA, MAX_STEP_DELTA)
    
    # 3. Calculate final safe target
    safe_target = current_position + clipped_delta
    
    # 4. Update current position
    current_position = safe_target.copy()
    
    # --- PRINTS ---
    np.set_printoptions(precision=4, suppress=True)
    print(f"Requested Target (Isaac)  : {received_target}")
    print(f"Requested Delta           : {requested_delta}")
    print(f"Applied Delta             : {clipped_delta}")
    print(f"Executed Target (Safe)    : {safe_target}")
    print("-" * 50)
    sock_ok.sendto(b"OK", (UDP_OK, PORT_OK))

'''
import socket
import struct
import dvrk
import rospy
import numpy as np
rospy.init_node('isaac_bridge')
arm = dvrk.psm('PSM3') 
UDP_IP = "0.0.0.0" 
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Waiting for actions from Isaac Lab...")

while not rospy.is_shutdown():
    data, addr = sock.recvfrom(1024)
    joint_deltas = np.array(struct.unpack('6f', data))
    current_pos = np.array(arm.get_current_joint_position())
    target_pos = current_pos + joint_deltas
    arm.servo_jp(target_pos)
'''

# measured_js()