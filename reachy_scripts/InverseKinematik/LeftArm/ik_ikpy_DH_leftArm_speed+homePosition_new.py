#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import time


class ReachyIKNode(Node):
    def __init__(self):
        super().__init__("reachy_left_arm_ik")

        # Publisher für linken Arm
        self.pub = self.create_publisher(
            Float64MultiArray,
            "/l_arm_forward_position_controller/commands",
            10
        )

        # ---------------------------------------------------------
        # MANUELL DEFINIERTE IK-KETTE (Reachy 2023, linker Arm)
        # ---------------------------------------------------------
        # Daten wurden aus reachy_description/urdf/arm.urdf.xacro extrahiert
        # → absolut korrekte Kinematik
        self.chain = Chain(name="left_arm", links=[
            OriginLink(),

            URDFLink(
                name="l_shoulder_pitch",
                origin_translation=[0.0, 0.195, 0.195],
                origin_orientation=[0.0, 0.0, 0.0],
                rotation=[0, 1, 0],    # pitch
                bounds=(-2.6, 1.57)
            ),

            URDFLink(
                name="l_shoulder_roll",
                origin_translation=[0.0, 0.0, 0.0],
                origin_orientation=[0.0, -1.57, 0.0],
                rotation=[1, 0, 0],    # roll
                bounds=(-0.17, 3.14)
            ),

            URDFLink(
                name="l_arm_yaw",
                origin_translation=[0.0, 0.0, 0.07],
                origin_orientation=[0.0, 0.0, 0.0],
                rotation=[0, 0, 1],    # yaw
                bounds=(-2.8, 2.8)
            ),

            URDFLink(
                name="l_elbow_pitch",
                origin_translation=[0.0, 0.0, 0.205],
                origin_orientation=[0.0, 0.0, 0.0],
                rotation=[0, 1, 0],    # pitch
                bounds=(0.0, 2.2)
            ),

            URDFLink(
                name="l_forearm_yaw",
                origin_translation=[0.0, 0.0, 0.165],
                origin_orientation=[0.0, 0.0, 0.0],
                rotation=[0, 0, 1],   # yaw
                bounds=(-2.5, 2.5)
            ),

            URDFLink(
                name="l_wrist_pitch",
                origin_translation=[0.0, 0.0, 0.08],
                origin_orientation=[0.0, 0.0, 0.0],
                rotation=[0, 1, 0],   # pitch
                bounds=(-1.57, 1.57)
            ),

            URDFLink(
                name="l_wrist_roll",
                origin_translation=[0.0, 0.0, 0.05],
                origin_orientation=[0.0, 0.0, 0.0],
                rotation=[0, 0, 1],   # roll
                bounds=(-1.8, 1.8)
            )
        ])

        self.num_joints = 7

    # ---------------------------------------------------------
    # NATÜRLICHE HOME-POSE (aus Gazebo-Werten)
    # ---------------------------------------------------------
        self.home = np.array([
            0.0005672128,  # l_shoulder_pitch
            0.0000759077,  # l_shoulder_roll
            0.0000004384,  # l_arm_yaw
            0.0000045249,  # l_elbow_pitch
            -0.0000003404, # l_forearm_yaw
            0.0001054981,  # l_wrist_pitch
            -0.0000056992  # l_wrist_roll
        ])


        self.current_joints = self.home.copy()

        self.get_logger().info("Left Arm IK node initialized (manual chain, 7 DOF).")

    # ------------------------------------------------------------------
    #  SEND JOINT COMMANDS
    # ------------------------------------------------------------------
    def send_joints(self, joints):
        msg = Float64MultiArray()
        msg.data = joints.tolist()
        self.pub.publish(msg)

    # ------------------------------------------------------------------
    #  MOVE JOINTS WITH INTERPOLATION (smooth movement)
    # ------------------------------------------------------------------
    def move_to_joints(self, target, steps=80, delay=0.02):
        current = self.current_joints.copy()

        for i in range(steps):
            alpha = (i + 1) / steps
            interp = current * (1 - alpha) + target * alpha
            self.send_joints(interp)
            time.sleep(delay)

        self.current_joints = target.copy()
        self.send_joints(target)

    # ------------------------------------------------------------------
    #  INVERSE KINEMATICS (XYZ → 7 joint values)
    # ------------------------------------------------------------------
    def compute_ik(self, x, y, z):
        full_solution = self.chain.inverse_kinematics([x, y, z])
        joints = full_solution[1:1 + self.num_joints]
        return joints

    # ------------------------------------------------------------------
    #  MOVE VIA IK
    # ------------------------------------------------------------------
    def move_to_pose(self, x, y, z, steps=80, delay=0.02):
        target_joints = self.compute_ik(x, y, z)
        self.move_to_joints(target_joints, steps, delay)

    # ------------------------------------------------------------------
    #  FOLLOW TRAJECTORY
    # ------------------------------------------------------------------
    def follow_trajectory(self, points, steps=80, delay=0.02):
        for (x, y, z) in points:
            self.get_logger().info(
                f"→ Trajektorienpunkt: {x:.3f}, {y:.3f}, {z:.3f}"
            )
            self.move_to_pose(x, y, z, steps, delay)

        self.get_logger().info("→ Rückkehr in Home-Pose…")
        self.move_to_joints(self.home, steps, delay)


# --------------------------------------------------------------------------
#  MAIN
# --------------------------------------------------------------------------
def main():
    rclpy.init()
    node = ReachyIKNode()

    node.get_logger().info("Bewegung: Start in Home-Pose…")
    node.move_to_joints(node.home, steps=80, delay=0.02)

    trajectory = [
        (0.30, 0.18, -0.10),
        (0.33, 0.10, -0.15),
        (0.28, 0.14, -0.22),
        (0.26, 0.20, -0.12),
    ]

    node.get_logger().info("Starte Trajektorie…")
    node.follow_trajectory(trajectory)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
