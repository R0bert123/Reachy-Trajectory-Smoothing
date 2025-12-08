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


# -----------------------------------------------------------
#  Terminal-Auswahl: YAML-Trajektorie auswählen
# -----------------------------------------------------------
def choose_yaml_trajectory():
    folder = os.path.expanduser("~/reachy_scripts/YAML-Files/*.yaml")
    files = sorted(glob.glob(folder))

    if not files:
        print("Fehler: Keine YAML-Dateien gefunden in ~/reachy_scripts/YAML-Files/")
        return None

    print("\nVerfügbare Trajektorien:")
    for i, f in enumerate(files, start=1):
        print(f"[{i}] {os.path.basename(f)}")

    choice = int(input("\nWelche YAML-Datei soll ausgeführt werden? (Nummer): "))
    selected = files[choice - 1]

    print(f"\nLade Trajektorie: {os.path.basename(selected)}\n")
    return selected


# -----------------------------------------------------------
#                Reachy IK Node
# -----------------------------------------------------------
class ReachyIKNode(Node):
    def __init__(self):
        super().__init__("reachy_left_arm_ik")

        self.pub = self.create_publisher(
            Float64MultiArray,
            "/l_arm_forward_position_controller/commands",
            10
        )

        # URDF laden (komplette Reachy 2023 URDF)
        self.chain = ikpy.chain.Chain.from_urdf_file(
            "/home/robert/reachy_2023/reachy_description/urdf/reachy.urdf",

            # Aktive Gelenke (9 Links total, 7 echte Gelenke)
            active_links_mask=[
                False,  # base_link
                True,   # shoulder_pitch
                True,   # shoulder_roll
                True,   # arm_yaw
                True,   # elbow_pitch
                True,   # forearm_yaw
                True,   # wrist_pitch
                True,   # wrist_roll
                False   # end-effector dummy
            ]
        )

        # Anzahl echter Arm-Gelenke
        self.num_joints = 7

        # Natürliche Home-Pose (feinjustierbar)
        self.home = np.array([
            0.0,    # shoulder_pitch
            -0.25,  # shoulder_roll
            0.0,    # arm_yaw
            0.4,    # elbow_pitch
            0.0,    # forearm_yaw
            0.0,    # wrist_pitch
            0.0     # wrist_roll
        ])

        self.current = self.home.copy()

        self.get_logger().info("Reachy IK Node gestartet (7 DOF Arm, URDF erfolgreich geladen).")


    # -----------------------------------------------------------
    #  Gelenke an Controller senden
    # -----------------------------------------------------------
    def send_joints(self, joints):
        msg = Float64MultiArray()
        msg.data = joints.tolist()
        self.pub.publish(msg)


    # -----------------------------------------------------------
    #  Lineare Gelenkinterpolation
    # -----------------------------------------------------------
    def move_to_joints(self, target, steps=80, delay=0.02):
        start = self.current.copy()

        for i in range(steps):
            alpha = (i + 1) / steps
            interpolated = start * (1 - alpha) + target * alpha
            self.send_joints(interpolated)
            time.sleep(delay)

        self.current = target.copy()


    # -----------------------------------------------------------
    #  Inverse Kinematik: XYZ -> 7 Gelenke
    # -----------------------------------------------------------
    def compute_ik(self, x, y, z):
        target_xyz = [x, y, z]

        full_solution = self.chain.inverse_kinematics(target_xyz)

        # Gelenke extrahieren (Index 1 bis 7)
        joints = full_solution[1:1 + self.num_joints]

        return np.array(joints)


    # -----------------------------------------------------------
    #  IK-Bewegung im Raum
    # -----------------------------------------------------------
    def move_to_pose(self, x, y, z, steps=60, delay=0.02):
        target_joints = self.compute_ik(x, y, z)
        self.move_to_joints(target_joints, steps=steps, delay=delay)


    # -----------------------------------------------------------
    #  Trajektorie aus YAML abspielen
    # -----------------------------------------------------------
    def follow_trajectory(self, trajectory):
        for p in trajectory:
            x, y, z = p["x"], p["y"], p["z"]
            steps = p.get("steps", 80)
            delay = p.get("delay", 0.02)

            self.get_logger().info(f"XYZ: {x:.3f}, {y:.3f}, {z:.3f}")
            self.move_to_pose(x, y, z, steps=steps, delay=delay)

        self.get_logger().info("Rückkehr zur Home-Pose")
        self.move_to_joints(self.home, steps=80, delay=0.02)


# -----------------------------------------------------------
#  MAIN
# -----------------------------------------------------------
def main():
    rclpy.init()
    node = ReachyIKNode()

    yaml_path = choose_yaml_trajectory()
    if yaml_path is None:
        return

    # YAML laden
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    trajectory = yaml_data["trajectory"]

    node.get_logger().info("Fahre Home-Pose an")
    node.move_to_joints(node.home, steps=70, delay=0.02)

    node.get_logger().info("Starte Trajektorie")
    node.follow_trajectory(trajectory)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
