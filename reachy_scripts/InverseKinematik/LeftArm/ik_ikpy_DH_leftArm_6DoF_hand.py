#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import ikpy.chain
import numpy as np
import time

from scipy.spatial.transform import Rotation as R


class ReachyIKNode(Node):
    def __init__(self):
        super().__init__("reachy_left_arm_ik")

        # Publisher für den linken Arm
        self.pub = self.create_publisher(
            Float64MultiArray,
            "/l_arm_forward_position_controller/commands",
            10
        )

        # ---------------------------------------
        # IK Chain vollständig aus URDF laden
        # ---------------------------------------
        self.chain = ikpy.chain.Chain.from_urdf_file(
            "/home/robert/reachy_2023/reachy_description/urdf/reachy.urdf",
            active_links_mask=[True] * 8  # 8 "bewegte" Einträge, 7 Gelenke + Base
        )

        # 7 Gelenke des linken Arms in der Reihenfolge des Controllers:
        self.joint_names = [
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_arm_yaw",
            "l_elbow_pitch",
            "l_forearm_yaw",
            "l_wrist_pitch",
            "l_wrist_roll"
        ]

        self.num_joints = len(self.joint_names)

        # ---------------------------------------
        # NATÜRLICHE HOME-POSITION (aus Gazebo)
        # ---------------------------------------
        self.home = np.array([
            0.0,     # shoulder_pitch
            -0.25,   # shoulder_roll
            0.0,     # arm_yaw
            0.4,     # elbow_pitch
            0.0,     # forearm_yaw
            0.0,     # wrist_pitch
            0.0      # wrist_roll
        ])

        self.current_joints = self.home.copy()

        self.get_logger().info("Initialized IK node with 6-DoF IK + 7 joint Reachy arm.")

    # -------------------------------------------------------------
    # Orientation Helper: convert Roll/Pitch/Yaw → 3×3 matrix
    # -------------------------------------------------------------
    def rpy_to_matrix(self, roll, pitch, yaw):
        rot = R.from_euler("xyz", [roll, pitch, yaw])
        return rot.as_matrix()

    # -------------------------------------------------------------
    # Compute IK: full 6DoF IK (XYZ + orientation)
    # -------------------------------------------------------------
    def compute_ik(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        # Convert orientation to matrix
        rot_matrix = self.rpy_to_matrix(roll, pitch, yaw)

        # Call IKPy inverse kinematics
        full_solution = self.chain.inverse_kinematics(
            target_position=[x, y, z],
            target_orientation=rot_matrix,
            orientation_mode="all"  # <- essentiell für 6DoF IK
        )

        # Extract exactly the 7 arm joints
        joints = full_solution[1:1 + self.num_joints]

        return joints

    # -------------------------------------------------------------
    # Send joint vector to controller
    # -------------------------------------------------------------
    def send_joints(self, joints):
        msg = Float64MultiArray()
        msg.data = joints.tolist()
        self.pub.publish(msg)

    # -------------------------------------------------------------
    # Smooth linear interpolation movement
    # -------------------------------------------------------------
    def move_to_joints(self, target, steps=80, delay=0.02):
        current = self.current_joints.copy()

        for i in range(steps):
            alpha = (i + 1) / steps
            interp = current * (1 - alpha) + target * alpha
            self.send_joints(interp)
            time.sleep(delay)

        self.current_joints = target.copy()

    # -------------------------------------------------------------
    # High-level pose movement via IK
    # -------------------------------------------------------------
    def move_to_pose(self, x, y, z, roll=0, pitch=0, yaw=0, steps=80, delay=0.02):
        target_joints = self.compute_ik(x, y, z, roll, pitch, yaw)
        self.move_to_joints(target_joints, steps=steps, delay=delay)

    # -------------------------------------------------------------
    # Execute a list of xyz (+ orientation) points
    # -------------------------------------------------------------
    def follow_trajectory(self, trajectory, steps=80, delay=0.02):
        for (x, y, z, roll, pitch, yaw) in trajectory:
            self.get_logger().info(
                f"→ Trajektorienpunkt: {x:.3f}, {y:.3f}, {z:.3f}  "
                f"RPY: {roll:.2f}, {pitch:.2f}, {yaw:.2f}"
            )
            self.move_to_pose(x, y, z, roll, pitch, yaw,
                              steps=steps, delay=delay)

        self.get_logger().info("→ Rückkehr in Home-Pose…")
        self.move_to_joints(self.home, steps=steps, delay=delay)


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    rclpy.init()
    node = ReachyIKNode()

    node.get_logger().info("Bewegung: Start in natürliche Home-Pose…")
    node.move_to_joints(node.home, steps=80, delay=0.02)

    # Example trajectory including orientation (roll, pitch, yaw)
    trajectory = [
        (0.30, 0.18, -0.10,  0.0,  1.0,  0.0),
        (0.33, 0.10, -0.15,  0.0,  1.2,  0.3),
        (0.28, 0.14, -0.22,  0.0,  1.5,  0.0),
        (0.26, 0.20, -0.12,  0.0,  1.0, -0.3),
    ]

    node.get_logger().info("Starte Trajektorie…")
    node.follow_trajectory(trajectory, steps=80, delay=0.02)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
