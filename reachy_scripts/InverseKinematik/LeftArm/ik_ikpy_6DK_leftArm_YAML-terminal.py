#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import ikpy.chain
import numpy as np
import yaml
import time
import os
import glob
from math import sin, cos


def choose_yaml_trajectory():
    folder = os.path.expanduser("~/reachy_scripts/YAML-Files/*.yaml")
    files = sorted(glob.glob(folder))

    if not files:
        print("Keine YAML-Dateien gefunden.")
        return None

    print("\nGefundene Trajektorien:")
    for i, f in enumerate(files, start=1):
        print(f"[{i}] {os.path.basename(f)}")

    choice = int(input("\nWelche YAML-Trajektorie ausführen? "))
    return files[choice - 1]


def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(roll), -np.sin(roll), 0],
                   [0, np.sin(roll), np.cos(roll), 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch), 0],
                   [0, 1, 0, 0],
                   [-np.sin(pitch), 0, np.cos(pitch), 0],
                   [0, 0, 0, 1]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                   [np.sin(yaw), np.cos(yaw), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    return Rz @ Ry @ Rx


class ReachyIKNode(Node):
    def __init__(self):
        super().__init__("reachy_left_arm_ik")

        # Publisher
        self.pub = self.create_publisher(
            Float64MultiArray,
            "/l_arm_forward_position_controller/commands",
            10
        )

        urdf_path = "/home/robert/reachy_2023/reachy_description/urdf/left_arm.urdf"

        print(f"\nLade URDF: {urdf_path}")
        full_chain = ikpy.chain.Chain.from_urdf_file(urdf_path)

        link_names = [link.name for link in full_chain.links]
        print("\nGefundene Links im URDF:")
        for i, name in enumerate(link_names):
            print(f"[{i}] {name}")

        print("\nBitte Start- & End-Link auswählen")
        start_idx = int(input("Start-Link Index: "))
        end_idx = int(input("End-Link Index: "))

        self.chain = ikpy.chain.Chain(
            links=full_chain.links[start_idx:end_idx + 1],
            name="left_arm_chain"
        )

        print(f"\nExtrahierte Armkette: {len(self.chain.links)} Links")
        for l in self.chain.links:
            print(" -", l.name)

        self.num_joints = 7

        self.home = np.array([0.0, -0.25, 0.0, 0.40, 0.0, 0.0, 0.0])
        self.current = self.home.copy()

        self.get_logger().info("6D IK Node gestartet.")


    def send_joints(self, joints):
        msg = Float64MultiArray()
        msg.data = joints.tolist()
        self.pub.publish(msg)


    def move_to_joints(self, target, steps=80, delay=0.02):
        start = self.current.copy()

        for i in range(steps):
            a = (i + 1) / steps
            interp = start * (1 - a) + target * a
            self.send_joints(interp)
            time.sleep(delay)

        self.current = target.copy()


    def compute_ik(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z

        R = rpy_to_matrix(roll, pitch, yaw)
        T[:3, :3] = R[:3, :3]

        full = self.chain.inverse_kinematics_frame(target=T)

        joints = full[1:1 + self.num_joints]
        return np.array(joints)



    def move_to_pose(self, pose, steps, delay):
        x, y, z = pose["x"], pose["y"], pose["z"]

        roll = pose.get("roll", 0.0)
        pitch = pose.get("pitch", 0.0)
        yaw = pose.get("yaw", 0.0)

        target_joints = self.compute_ik(x, y, z, roll, pitch, yaw)
        self.move_to_joints(target_joints, steps=steps, delay=delay)


    def follow_trajectory(self, trajectory):
        for p in trajectory:
            steps = p.get("steps", 80)
            delay = p.get("delay", 0.02)

            self.move_to_pose(p, steps, delay)

        self.move_to_joints(self.home, steps=80, delay=0.02)


def main():
    rclpy.init()
    node = ReachyIKNode()

    yaml_file = choose_yaml_trajectory()
    if yaml_file is None:
        return

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    trajectory = data["trajectory"]

    node.move_to_joints(node.home, steps=80, delay=0.02)
    node.follow_trajectory(trajectory)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
