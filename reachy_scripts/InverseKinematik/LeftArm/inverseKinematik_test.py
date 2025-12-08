from reachy_sdk import ReachySDK
import numpy as np
import time

# -------------------------------------------
# 1. Mit Reachy verbinden
# -------------------------------------------
reachy = ReachySDK(host='localhost')   # oder IP von Reachy

arm = reachy.right_arm   # oder left_arm


# -------------------------------------------
# 2. Zielpose definieren (Beispiel: cartesian x,y,z + orientation)
# -------------------------------------------

# Startpunkt = aktuelle Pose
start_pose = arm.forward_kinematics()

# Zielpunkt (Beispielwerte)
target_pose = {
    'x': 0.25,
    'y': -0.15,
    'z': 0.10,
    'roll': 0.0,
    'pitch': 0.0,
    'yaw': 0.0,
}


# -------------------------------------------
# 3. Inverse Kinematik NUR für Start & Ziel aufrufen
#    (→ NICHT für jeden Zwischenpunkt!)
# -------------------------------------------

print("Berechne IK…")
start_joints = arm.inverse_kinematics(
    x=start_pose['x'],
    y=start_pose['y'],
    z=start_pose['z'],
    roll=start_pose['roll'],
    pitch=start_pose['pitch'],
    yaw=start_pose['yaw'],
)

goal_joints = arm.inverse_kinematics(
    x=target_pose['x'],
    y=target_pose['y'],
    z=target_pose['z'],
    roll=target_pose['roll'],
    pitch=target_pose['pitch'],
    yaw=target_pose['yaw'],
)


# -------------------------------------------
# 4. Gelenkwinkel interpolieren
# -------------------------------------------

num_steps = 200               # Anzahl der Zwischenpunkte
duration = 2.0                # Sekunden, wie lange es dauern soll

joint_names = list(start_joints.keys())

# Array von Winkeltrajektorien erzeugen
trajectory = []

for t in np.linspace(0, 1, num_steps):
    step = {}
    for joint in joint_names:
        step[joint] = (1 - t) * start_joints[joint] + t * goal_joints[joint]
    trajectory.append(step)


# -------------------------------------------
# 5. Trajektorie an Reachy senden (smooth movement)
# -------------------------------------------

print("Starte Bewegung…")

interval = duration / num_steps

for step_joints in trajectory:
    for joint, angle in step_joints.items():
        arm.joints[joint].goal_position = angle
    time.sleep(interval)

print("Bewegung abgeschlossen!")
