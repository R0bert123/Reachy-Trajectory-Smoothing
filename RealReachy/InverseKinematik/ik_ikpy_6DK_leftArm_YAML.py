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
import math
import xml.etree.ElementTree as ET

#Heutige IP von Reachy: 192.168.68.50
# -----------------------------------------------------------
# KONFIG
# -----------------------------------------------------------

YAML_DIR = os.path.expanduser("~/reachy_scripts/YAML-Files")
LEFT_ARM_URDF = os.path.expanduser(
    "~/reachy_2023/reachy_description/urdf/left_arm.urdf"
)

LEFT_ARM_JOINT_NAMES = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_arm_yaw",
    "l_elbow_pitch",
    "l_forearm_yaw",
    "l_wrist_pitch",
    "l_wrist_roll",
]
NUM_JOINTS = len(LEFT_ARM_JOINT_NAMES)


# -----------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------

def rpy_to_matrix(roll, pitch, yaw):
    """4x4 Homogenmatrix aus Roll/Pitch/Yaw (wie in deinen früheren Skripten)."""
    Rx = np.array([
        [1, 0, 0, 0],
        [0, math.cos(roll), -math.sin(roll), 0],
        [0, math.sin(roll),  math.cos(roll), 0],
        [0, 0, 0, 1],
    ])

    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch), 0],
        [ 0,               1, 0,               0],
        [-math.sin(pitch), 0, math.cos(pitch), 0],
        [ 0,               0, 0,               1],
    ])

    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0, 0],
        [math.sin(yaw),  math.cos(yaw), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1],
    ])

    return Rz @ Ry @ Rx


def choose_yaml_trajectory():
    """Lässt dich im Terminal eine YAML-Trajektorie auswählen."""
    pattern = os.path.join(YAML_DIR, "*.yaml")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Keine YAML-Dateien in {YAML_DIR} gefunden.")
        return None

    print("\nGefundene Trajektorien:")
    for i, f in enumerate(files, start=1):
        print(f"[{i}] {os.path.basename(f)}")

    while True:
        try:
            choice = int(input("\nWelche YAML-Trajektorie ausführen? (Nummer): "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
        except ValueError:
            pass
        print("Ungültige Eingabe, bitte erneut.")


def validate_yaml_trajectory(yaml_data):
    """Einfacher Check der Struktur der YAML-Trajektorie."""
    if not isinstance(yaml_data, dict):
        raise ValueError("YAML-Wurzel muss ein Dict sein.")

    if "trajectory" not in yaml_data:
        raise ValueError('YAML muss das Feld "trajectory" enthalten.')

    traj = yaml_data["trajectory"]
    if not isinstance(traj, list):
        raise ValueError('"trajectory" muss eine Liste sein.')

    for idx, p in enumerate(traj):
        if not isinstance(p, dict):
            raise ValueError(f"trajectory[{idx}] ist kein Dict.")
        for key in ["x", "y", "z"]:
            if key not in p:
                raise ValueError(f"trajectory[{idx}] fehlt Schlüssel '{key}'.")

        for opt in ["roll", "pitch", "yaw"]:
            if opt in p and not isinstance(p[opt], (int, float)):
                raise ValueError(f"trajectory[{idx}]['{opt}'] muss Zahl sein.")

        if "steps" in p and not isinstance(p["steps"], int):
            raise ValueError(f"trajectory[{idx}]['steps'] muss int sein.")
        if "delay" in p and not isinstance(p["delay"], (int, float)):
            raise ValueError(f"trajectory[{idx}]['delay'] muss float/int sein.")


def load_joint_limits_from_urdf(urdf_path, joint_names):
    """Liest min/max/velocity Limits aus der URDF für die angegebenen Gelenke."""
    limits = {name: {"min": None, "max": None, "velocity": None}
              for name in joint_names}

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for joint in root.findall("joint"):
        jname = joint.get("name")
        if jname not in limits:
            continue

        limit_elem = joint.find("limit")
        if limit_elem is None:
            continue

        jmin = limit_elem.get("lower")
        jmax = limit_elem.get("upper")
        jvel = limit_elem.get("velocity")

        if jmin is not None:
            limits[jname]["min"] = float(jmin)
        if jmax is not None:
            limits[jname]["max"] = float(jmax)
        if jvel is not None:
            limits[jname]["velocity"] = float(jvel)

    return limits


def clip_joints_with_limits(joint_vec, joint_names, limits):
    """Clipt Gelenke auf Limits und sammelt Warnungen."""
    joint_vec = np.array(joint_vec, dtype=float)
    warnings = []

    for i, name in enumerate(joint_names):
        if name not in limits:
            continue
        lim = limits[name]
        v = joint_vec[i]

        if lim["min"] is not None and v < lim["min"]:
            warnings.append(f"{name}: {v:.3f} < min {lim['min']:.3f}")
            joint_vec[i] = lim["min"]

        if lim["max"] is not None and v > lim["max"]:
            warnings.append(f"{name}: {v:.3f} > max {lim['max']:.3f}")
            joint_vec[i] = lim["max"]

    return joint_vec, warnings


def print_joint_limits(limits):
    print("\nGelenk-Limits aus URDF:")
    for name, lim in limits.items():
        print(
            f"  {name:16s}  min={lim['min']}  max={lim['max']}  "
            f"vel={lim['velocity']}"
        )


# -----------------------------------------------------------
# IK-Knoten für linken Arm (6D + Limits)
# -----------------------------------------------------------

class ReachyLeftArmIK(Node):
    def __init__(self):
        super().__init__("reachy_left_arm_ik_extended")

        # Publisher
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            "/l_arm_forward_position_controller/commands",
            10,
        )

        # IK-Kette aus URDF
        self.get_logger().info(f"Lade URDF: {LEFT_ARM_URDF}")
        self.chain = ikpy.chain.Chain.from_urdf_file(LEFT_ARM_URDF)
        self.get_logger().info(f"IK-Kette geladen: {len(self.chain.links)} Links.")

        # Gelenke & Limits
        self.joint_names = LEFT_ARM_JOINT_NAMES
        self.num_joints = NUM_JOINTS
        self.limits = load_joint_limits_from_urdf(LEFT_ARM_URDF, self.joint_names)
        print_joint_limits(self.limits)

        # Home-Pose (natürliche Pose neben dem Körper, feinjustierbar)
        self.home = np.array([0.0, -0.25, 0.0, 0.40, 0.0, 0.0, 0.0], dtype=float)
        self.current = self.home.copy()

        self.get_logger().info("Left Arm 6D IK Node gestartet.")

    # -------------------------------------------------------
    # Senden der Gelenke (mit Limit-Check)
    # -------------------------------------------------------
    def send_joints(self, joints):
        joints = np.array(joints, dtype=float)
        clipped, warnings = clip_joints_with_limits(
            joints, self.joint_names, self.limits
        )

        if warnings:
            self.get_logger().warn(
                "Limitverletzung IK / Trajektorie:\n  " +
                "\n  ".join(warnings)
            )

        msg = Float64MultiArray()
        msg.data = clipped.tolist()
        self.cmd_pub.publish(msg)

    # -------------------------------------------------------
    # Steps an Velocity-Limits anpassen
    # -------------------------------------------------------
    def plan_steps_with_velocity_limits(self, target,
                                        base_steps=80,
                                        delay=0.02,
                                        speed_scale=1.0):
        target = np.array(target, dtype=float)
        current = np.array(self.current, dtype=float)
        delta = np.abs(target - current)

        max_steps_required = base_steps

        for i, name in enumerate(self.joint_names):
            v_limit = self.limits.get(name, {}).get("velocity", None)
            if v_limit is None or v_limit <= 0:
                continue

            allowed = v_limit * speed_scale
            if allowed <= 0:
                continue

            if delta[i] > 0:
                steps_needed = delta[i] / (allowed * delay)
                if steps_needed > max_steps_required:
                    max_steps_required = steps_needed

        steps = max(base_steps, int(math.ceil(max_steps_required)))

        if steps != base_steps:
            self.get_logger().info(
                f"Steps von {base_steps} auf {steps} erhöht (Velocity-Limits)."
            )

        return steps

    # -------------------------------------------------------
    # Gelenkinterpolation
    # -------------------------------------------------------
    def move_to_joints(self, target,
                       base_steps=80,
                       delay=0.02,
                       speed_scale=1.0):
        target = np.array(target, dtype=float)
        steps = self.plan_steps_with_velocity_limits(
            target,
            base_steps=base_steps,
            delay=delay,
            speed_scale=speed_scale,
        )

        start = self.current.copy()

        for i in range(steps):
            a = (i + 1) / steps
            interp = start * (1 - a) + target * a
            self.send_joints(interp)
            time.sleep(delay)

        self.current = target.copy()

    # -------------------------------------------------------
    # 6D-IK: Position + Orientierung
    # -------------------------------------------------------
    def compute_ik_6d(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z

        R = rpy_to_matrix(roll, pitch, yaw)
        T[:3, :3] = R[:3, :3]

        # ikpy erwartet hier target_frame=T
        full = self.chain.inverse_kinematics(target_frame=T)
        joints = full[1:1 + self.num_joints]  # 0 = Base in der Chain

        return np.array(joints, dtype=float)

    # -------------------------------------------------------
    # Einzelne Pose aus YAML-Eintrag
    # -------------------------------------------------------
    def move_to_pose(self, pose,
                     base_steps=80,
                     delay=0.02,
                     speed_scale=1.0):
        x, y, z = pose["x"], pose["y"], pose["z"]
        roll = pose.get("roll", 0.0)
        pitch = pose.get("pitch", 0.0)
        yaw = pose.get("yaw", 0.0)

        # Optional: Grad → Rad
        if pose.get("use_degrees", False):
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)

        joints = self.compute_ik_6d(x, y, z, roll, pitch, yaw)
        self.move_to_joints(joints,
                            base_steps=base_steps,
                            delay=delay,
                            speed_scale=speed_scale)

    # -------------------------------------------------------
    # Komplette Trajektorie ausführen
    # -------------------------------------------------------
    def follow_trajectory(self, trajectory):
        for idx, p in enumerate(trajectory):
            steps = p.get("steps", 80)
            delay = p.get("delay", 0.02)
            speed_scale = p.get("speed_scale", 1.0)

            self.get_logger().info(
                f"Pose {idx}: x={p['x']:.3f}, y={p['y']:.3f}, z={p['z']:.3f}, "
                f"roll={p.get('roll', 0.0):.3f}, "
                f"pitch={p.get('pitch', 0.0):.3f}, "
                f"yaw={p.get('yaw', 0.0):.3f}"
            )

            self.move_to_pose(p,
                              base_steps=steps,
                              delay=delay,
                              speed_scale=speed_scale)

        self.get_logger().info("Rückkehr in Home-Pose.")
        self.move_to_joints(self.home, base_steps=80, delay=0.02)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    rclpy.init()
    node = ReachyLeftArmIK()

    yaml_file = choose_yaml_trajectory()
    if yaml_file is None:
        node.get_logger().error("Keine Trajektorie gewählt, beende.")
        rclpy.shutdown()
        return

    node.get_logger().info(f"Lade YAML-Trajektorie: {yaml_file}")
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    try:
        validate_yaml_trajectory(data)
    except Exception as e:
        node.get_logger().error(f"YAML-Validierung fehlgeschlagen: {e}")
        rclpy.shutdown()
        return

    trajectory = data["trajectory"]

    node.get_logger().info("Fahre Home-Pose an...")
    node.move_to_joints(node.home, base_steps=80, delay=0.02)

    node.get_logger().info("Starte Trajektorie...")
    node.follow_trajectory(trajectory)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
