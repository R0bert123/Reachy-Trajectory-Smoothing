#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import time
from ikpy.chain import Chain
from ikpy.link import URDFLink

# --------------------------
# 1. IK-Kette für linken Arm Reachy 2023
# --------------------------
L_SHOULDER_ELBOW = 0.25  # Oberarmlänge
L_ELBOW_WRIST = 0.25     # Unterarmlänge

left_arm_chain = Chain(name='left_arm', links=[
    URDFLink(
        name='shoulder_pitch', 
        origin_translation=[0,0,0], 
        origin_orientation=[0,0,0], 
        rotation=[0,1,0]),
    URDFLink(
        name='shoulder_roll', 
        origin_translation=[0,0,0], 
        origin_orientation=[0,0,0], 
        rotation=[1,0,0]),
    URDFLink(
        name='shoulder_yaw', 
        origin_translation=[0,0,0.1], 
        origin_orientation=[0,0,0], 
        rotation=[0,0,1]),
    URDFLink(
        name='elbow_pitch', 
        origin_translation=[L_SHOULDER_ELBOW,0,0], 
        origin_orientation=[0,0,0], 
        rotation=[0,1,0]),
    URDFLink(
        name='wrist_pitch', 
        origin_translation=[L_ELBOW_WRIST,0,0], 
        origin_orientation=[0,0,0], 
        rotation=[0,1,0]),
    URDFLink(
        name='wrist_roll', 
        origin_translation=[0,0,0], 
        origin_orientation=[0,0,0], 
        rotation=[1,0,0]),
    URDFLink(
        name='gripper', 
        origin_translation=[0,0,0], 
        origin_orientation=[0,0,0], 
        rotation=[0,0,1])
])

# --------------------------
# 2. ROS2 Node für Gazebo Publisher
# --------------------------
class LeftArmIKTrajectory(Node):
    def __init__(self):
        super().__init__('left_arm_ik_trajectory')
        self.pub = self.create_publisher(Float64MultiArray, '/l_arm_forward_position_controller/commands', 10)
        self.current_joints = [0.0]*7
        self.sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

    def joint_state_callback(self, msg):
        # linke Arm-Gelenke sind die ersten 7
        self.current_joints = list(msg.position[:7])

    # IK berechnen für Ziel [x, y, z] mit Handfläche nach unten
    def compute_ik(self, x, y, z):
        # 4x4 Homogenmatrix
        target_frame = np.eye(4)
        target_frame[:3, 3] = [x, y, z]  # Position

        # Rotation um X-Achse: Handfläche nach unten
        angle = -np.pi/2
        Rx = np.array([
            [1,0,0,0],
            [0,np.cos(angle),-np.sin(angle),0],
            [0,np.sin(angle), np.cos(angle),0],
            [0,0,0,1]
        ])
        target_frame = target_frame @ Rx

        # IK berechnen
        joints = left_arm_chain.inverse_kinematics_frame(target_frame)
        target_joints = np.append(joints[1:], 0.0)  # Anzahl Gelenke = 7 + Dummy Gripper 0.0
        return target_joints

    # Sanfte Interpolation
    def move_to_pose(self, x, y, z, steps=50, delay=0.05):
        target_joints = self.compute_ik(x, y, z)
        start_joints = np.array(self.current_joints)
        for alpha in np.linspace(0, 1, steps):
            interp = (1-alpha)*start_joints + alpha*target_joints
            msg = Float64MultiArray()
            msg.data = interp.tolist()
            self.pub.publish(msg)
            time.sleep(delay)

    # Trajektorie abfahren
    def follow_trajectory(self, points, steps_per_segment=50, delay=0.05):
        for x, y, z in points:
            self.move_to_pose(x, y, z, steps=steps_per_segment, delay=delay)

# --------------------------
# 3. Hauptprogramm
# --------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LeftArmIKTrajectory()

    # kurze Wartezeit, bis joint_states empfangen wurden
    time.sleep(1.0)

    # Trajektorie: kleine Kurve vor dem Körper
    trajectory_points = [
        (0.25, 0.1, -0.15),
        (0.26, 0.12, -0.18),
        (0.27, 0.08, -0.20),
        (0.25, 0.1, -0.15)
    ]

    node.follow_trajectory(trajectory_points, steps_per_segment=100, delay=0.05)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
