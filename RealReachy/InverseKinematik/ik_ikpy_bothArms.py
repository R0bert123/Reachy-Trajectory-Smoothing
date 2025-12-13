#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import ikpy.chain
from ikpy.chain import Chain
from ikpy.urdf import URDF
from ikpy.link import OriginLink

import numpy as np
import yaml
import time
import os
import glob
import math
import xml.etree.ElementTree as ET

from time import sleep

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

# ------------------------------------------------------------------
# REACHY SDK: Motoren an/aus (ROS macht die Positionskommandos)
# ------------------------------------------------------------------
reachy = ReachySDK(host='localhost')
reachy.turn_on('reachy')

# -----------------------------------------------------------
# KONFIG
# -----------------------------------------------------------

YAML_DIR = os.path.expanduser("/home/reachy/Codes/Robert/YAML-Files")

# Pfad zur URDF Datei "reachy.urdf"
REACHY_URDF = os.path.expanduser(
    "~/Codes/Robert/reachy.urdf"
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

RIGHT_ARM_JOINT_NAMES = [
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_arm_yaw",
    "r_elbow_pitch",
    "r_forearm_yaw",
    "r_wrist_pitch",
    "r_wrist_roll",
]

NUM_LEFT_JOINTS = len(LEFT_ARM_JOINT_NAMES)
NUM_RIGHT_JOINTS = len(RIGHT_ARM_JOINT_NAMES)


# -----------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------

def get_current_joint_positions_from_reachy(arm: str):
    """
    Liest die aktuellen Gelenkwinkel direkt vom realen Reachy-SDK aus.
    Der SDK liefert Winkel in Grad --> Für IKPY wird Radiant verwendet.
    """
    if arm == "left":
        joints_deg = [
            reachy.l_arm.l_shoulder_pitch.present_position,
            reachy.l_arm.l_shoulder_roll.present_position,
            reachy.l_arm.l_arm_yaw.present_position,
            reachy.l_arm.l_elbow_pitch.present_position,
            reachy.l_arm.l_forearm_yaw.present_position,
            reachy.l_arm.l_wrist_pitch.present_position,
            reachy.l_arm.l_wrist_roll.present_position,
        ]
    else:  # right arm
        joints_deg = [
            reachy.r_arm.r_shoulder_pitch.present_position,
            reachy.r_arm.r_shoulder_roll.present_position,
            reachy.r_arm.r_arm_yaw.present_position,
            reachy.r_arm.r_elbow_pitch.present_position,
            reachy.r_arm.r_forearm_yaw.present_position,
            reachy.r_arm.r_wrist_pitch.present_position,
            reachy.r_arm.r_wrist_roll.present_position,
        ]

    # Reachy liefert Grad → IKPY braucht Radiant
    return np.radians(joints_deg)


def rpy_to_matrix(roll, pitch, yaw):
    """4x4 Homogenmatrix aus Roll/Pitch/Yaw."""
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
        print("Ungültige Eingabe, bitte erneut eine Zahl eingeben.")


def _validate_pose_dict(pose, label):
    """Hilfsfunktion: prüft left/right-Pose in YAML."""
    if not isinstance(pose, dict):
        raise ValueError(f"{label} muss ein Dict sein.")
    for key in ["x", "y", "z"]:
        if key not in pose:
            raise ValueError(f"{label} fehlt Schlüssel '{key}'.")
    for opt in ["roll", "pitch", "yaw"]:
        if opt in pose and not isinstance(pose[opt], (int, float)):
            raise ValueError(f"{label}['{opt}'] muss Zahl sein.")
    if "use_degrees" in pose and not isinstance(pose["use_degrees"], bool):
        raise ValueError(f"{label}['use_degrees'] muss bool sein.")


def validate_yaml_trajectory(yaml_data):
    """
    Struktur:
      trajectory:
        - left:
            x, y, z, roll, pitch, yaw, use_degrees (optional)
          right:
            ...
          steps: 80
          delay: 0.02
          speed_scale: 1.0
    """
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

        left = p.get("left")
        right = p.get("right")

        if left is None and right is None:
            raise ValueError(
                f"trajectory[{idx}] muss mindestens 'left' oder 'right' enthalten."
            )

        if left is not None:
            _validate_pose_dict(left, f"trajectory[{idx}]['left']")
        if right is not None:
            _validate_pose_dict(right, f"trajectory[{idx}]['right']")

        if "steps" in p and not isinstance(p["steps"], int):
            raise ValueError(f"trajectory[{idx}]['steps'] muss int sein.")
        if "delay" in p and not isinstance(p["delay"], (int, float)):
            raise ValueError(f"trajectory[{idx}]['delay'] muss float/int sein.")
        if "speed_scale" in p and not isinstance(p["speed_scale"], (int, float)):
            raise ValueError(f"trajectory[{idx}]['speed_scale'] muss float/int sein.")


def load_joint_limits_from_urdf(urdf_path, joint_names):
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
    """
    Clipt Gelenke auf Limits und sammelt Warnungen.
    Diese Schutzlogik verhindert, dass der Arm außerhalb der
    URDF-Gelenkgrenzen bewegt wird.
    """
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


def print_joint_limits(limits, arm_label):
    print(f"\nGelenk-Limits aus URDF ({arm_label}):")
    for name, lim in limits.items():
        print(
            f"  {name:16s}  min={lim['min']}  max={lim['max']}  "
            f"vel={lim['velocity']}"
        )


def build_arm_chain_from_reachy_urdf(urdf_path, joint_names, chain_name):
    """
    Erzeuge eine IK-Kette nur für die angegebenen Gelenke aus reachy.urdf.
    (Derzeit nicht benutzt, könnte aber bei Bedarf verwendet werden.)
    """
    links = URDF.get_chain_from_joints(urdf_path, joint_names)
    all_links = [OriginLink()] + links
    chain = Chain(name=chain_name, links=all_links)
    return chain


def minimum_jerk(alpha: float) -> float:
    """
    Minimum-Jerk-Polynom für alpha in [0,1]:
        s(alpha) = 10 a^3 - 15 a^4 + 6 a^5
    """
    return 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5


# -----------------------------------------------------------
# IK-Knoten für BEIDE Arme
# -----------------------------------------------------------

class ReachyDualArmIK(Node):
    def __init__(self):
        super().__init__("reachy_dual_arm_ik")

        # Publisher für beide Arme
        self.l_pub = self.create_publisher(
            Float64MultiArray,
            "/l_arm_forward_position_controller/commands",
            10,
        )
        self.r_pub = self.create_publisher(
            Float64MultiArray,
            "/r_arm_forward_position_controller/commands",
            10,
        )

    # ---------------------------------------------------
    # IK-Ketten nur aus den echten Arm-Gelenken aufbauen
    #---------------------------------------------------
        self.get_logger().info(f"Lade Arm-IK-Ketten aus URDF: {REACHY_URDF}")

    # Nur die relevanten Links der Arme extrahieren
        left_links = URDF.get_chain_from_joints(REACHY_URDF, LEFT_ARM_JOINT_NAMES)
        right_links = URDF.get_chain_from_joints(REACHY_URDF, RIGHT_ARM_JOINT_NAMES)

    # IKPy benötigt einen OriginLink als Start
        self.left_chain = Chain(name="left_arm_chain", links=[OriginLink()] + left_links)
        self.right_chain = Chain(name="right_arm_chain", links=[OriginLink()] + right_links)

        self.get_logger().info(
           f"Left arm IK chain loaded with {len(self.left_chain.links)} links."
)
        self.get_logger().info(
            f"Right arm IK chain loaded with {len(self.right_chain.links)} links."
)

        # Limits aus derselben URDF laden
        self.left_limits = load_joint_limits_from_urdf(
            REACHY_URDF, LEFT_ARM_JOINT_NAMES
        )
        self.right_limits = load_joint_limits_from_urdf(
            REACHY_URDF, RIGHT_ARM_JOINT_NAMES
        )

        print_joint_limits(self.left_limits, "Left Arm")
        print_joint_limits(self.right_limits, "Right Arm")



        # ---------------------------------------------------
        # Home-Pose dynamisch von Reachy realer Pose lesen
        # ---------------------------------------------------
        self.get_logger().info("Lese aktuelle Gelenkstellung als Home-Pose...")

        # Gelenke aus SDK lesen (Radiant)
        self.left_home = get_current_joint_positions_from_reachy("left")
        self.right_home = get_current_joint_positions_from_reachy("right")

        self.left_current = self.left_home.copy()
        self.right_current = self.right_home.copy()

        # Zusätzlich: Ausgabe in Grad
        left_home_deg = np.degrees(self.left_home)
        right_home_deg = np.degrees(self.right_home)

        self.get_logger().info("----------------------------------------------------")
        self.get_logger().info("     REALE HOME-POSE VON REACHY (LINKER ARM)")
        self.get_logger().info("----------------------------------------------------")
        for name, rad, deg in zip(LEFT_ARM_JOINT_NAMES, self.left_home, left_home_deg):
            self.get_logger().info(f"  {name:17s}:  {rad:8.4f} rad   |   {deg:8.2f}°")

        self.get_logger().info("----------------------------------------------------")
        self.get_logger().info("     REALE HOME-POSE VON REACHY (RECHTER ARM)")
        self.get_logger().info("----------------------------------------------------")
        for name, rad, deg in zip(RIGHT_ARM_JOINT_NAMES, self.right_home, right_home_deg):
            self.get_logger().info(f"  {name:17s}:  {rad:8.4f} rad   |   {deg:8.2f}°")

        self.get_logger().info("----------------------------------------------------")
        self.get_logger().info("Home-Pose erfolgreich eingelesen.")


    # -------------------------------------------------------
    # IK für linken/rechten Arm
    # -------------------------------------------------------
    def _ik_chain_for_arm(self, arm):
        if arm == "left":
            return self.left_chain, LEFT_ARM_JOINT_NAMES, NUM_LEFT_JOINTS
        else:
            return self.right_chain, RIGHT_ARM_JOINT_NAMES, NUM_RIGHT_JOINTS

    def compute_ik_6d(self, arm, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        # Homogenmatrix
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(roll, pitch, yaw)[:3, :3]
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z

        chain, joint_names, nj = self._ik_chain_for_arm(arm)

        # IKPy hat je nach Version unterschiedliche Funktionsnamen
        try:
            full = chain.inverse_kinematics(target_frame=T)
        except TypeError:
            full = chain.inverse_kinematics_frame(T)

        # Nur aktive Gelenke extrahieren
        if hasattr(chain, "active_from_full"):
            joints = chain.active_from_full(full)
        else:
            # Fallback: erstes Element (Origin) droppen
            joints = np.array(full[1:1 + nj])

        return np.array(joints, dtype=float)

    # -------------------------------------------------------
    # Geschwindigkeit/Steps an Limits anpassen
    # -------------------------------------------------------
    def plan_steps_with_velocity_limits(self, current, target,
                                        joint_names, limits,
                                        base_steps=80,
                                        delay=0.02,
                                        speed_scale=1.0):
        target = np.array(target, dtype=float)
        current = np.array(current, dtype=float)
        delta = np.abs(target - current)

        max_steps_required = base_steps

        for i, name in enumerate(joint_names):
            v_limit = limits.get(name, {}).get("velocity", None)
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
        return steps

    # -------------------------------------------------------
    # Einzelarm senden (mit Limit-Check)
    # -------------------------------------------------------
    def send_arm(self, arm, joints):
        if arm == "left":
            joint_names = LEFT_ARM_JOINT_NAMES
            limits = self.left_limits
        else:
            joint_names = RIGHT_ARM_JOINT_NAMES
            limits = self.right_limits

        clipped, warnings = clip_joints_with_limits(joints, joint_names, limits)

        if warnings:
            self.get_logger().warn(
                f"Limitverletzung {arm} IK / Trajektorie:\n  " +
                "\n  ".join(warnings)
            )

        msg = Float64MultiArray()
        msg.data = clipped.tolist()

        if arm == "left":
            self.l_pub.publish(msg)
        else:
            self.r_pub.publish(msg)

    # -------------------------------------------------------
    # Gelenkinterpolation für EINEN Arm (Minimum Jerk)
    # -------------------------------------------------------
    def move_arm_to_joints(self, arm, target,
                           base_steps=80,
                           delay=0.02,
                           speed_scale=1.0):
        target = np.array(target, dtype=float)

        if arm == "left":
            current = self.left_current
            joint_names = LEFT_ARM_JOINT_NAMES
            limits = self.left_limits
        else:
            current = self.right_current
            joint_names = RIGHT_ARM_JOINT_NAMES
            limits = self.right_limits

        steps = self.plan_steps_with_velocity_limits(
            current, target, joint_names, limits,
            base_steps=base_steps,
            delay=delay,
            speed_scale=speed_scale,
        )

        if steps <= 0:
            return

        start = current.copy()
        delta = target - start

        for i in range(steps):
            alpha = (i + 1) / steps  # 0..1
            s = minimum_jerk(alpha)  # Minimum-Jerk-Skalierung
            interp = start + s * delta
            self.send_arm(arm, interp)
            time.sleep(delay)

        if arm == "left":
            self.left_current = target.copy()
        else:
            self.right_current = target.copy()

    # -------------------------------------------------------
    # Gleichzeitig beide Arme bewegen (Minimum Jerk)
    # -------------------------------------------------------
    def move_both_arms(self, left_pose=None, right_pose=None,
                       base_steps=80, delay=0.02, speed_scale=1.0):
        # Wenn kein Pose-Dict übergeben, bleibt der Arm in aktueller Stellung.
        left_target = self.left_current.copy()
        right_target = self.right_current.copy()

        if left_pose is not None:
            x = left_pose["x"]
            y = left_pose["y"]
            z = left_pose["z"]
            roll = left_pose.get("roll", 0.0)
            pitch = left_pose.get("pitch", 0.0)
            yaw = left_pose.get("yaw", 0.0)
            if left_pose.get("use_degrees", False):
                roll = math.radians(roll)
                pitch = math.radians(pitch)
                yaw = math.radians(yaw)
            left_target = self.compute_ik_6d("left", x, y, z, roll, pitch, yaw)

        if right_pose is not None:
            x = right_pose["x"]
            y = right_pose["y"]
            z = right_pose["z"]
            roll = right_pose.get("roll", 0.0)
            pitch = right_pose.get("pitch", 0.0)
            yaw = right_pose.get("yaw", 0.0)
            if right_pose.get("use_degrees", False):
                roll = math.radians(roll)
                pitch = math.radians(pitch)
                yaw = math.radians(yaw)
            right_target = self.compute_ik_6d("right", x, y, z, roll, pitch, yaw)

        # Steps pro Arm bestimmen
        left_steps = self.plan_steps_with_velocity_limits(
            self.left_current, left_target,
            LEFT_ARM_JOINT_NAMES, self.left_limits,
            base_steps=base_steps, delay=delay, speed_scale=speed_scale
        )
        right_steps = self.plan_steps_with_velocity_limits(
            self.right_current, right_target,
            RIGHT_ARM_JOINT_NAMES, self.right_limits,
            base_steps=base_steps, delay=delay, speed_scale=speed_scale
        )

        steps = max(left_steps, right_steps)
        if steps <= 0:
            return

        left_start = self.left_current.copy()
        right_start = self.right_current.copy()

        left_delta = left_target - left_start
        right_delta = right_target - right_start

        for i in range(steps):
            alpha = (i + 1) / steps
            s = minimum_jerk(alpha)

            # Links
            if left_pose is not None:
                l_interp = left_start + s * left_delta
            else:
                l_interp = left_start
            self.send_arm("left", l_interp)

            # Rechts
            if right_pose is not None:
                r_interp = right_start + s * right_delta
            else:
                r_interp = right_start
            self.send_arm("right", r_interp)

            time.sleep(delay)

        self.left_current = left_target.copy()
        self.right_current = right_target.copy()

    # -------------------------------------------------------
    # Komplette Trajektorie ausführen
    # -------------------------------------------------------
    def follow_trajectory(self, trajectory):
        for idx, p in enumerate(trajectory):
            steps = p.get("steps", 80)
            delay = p.get("delay", 0.02)
            speed_scale = p.get("speed_scale", 1.0)

            left = p.get("left")
            right = p.get("right")

            self.get_logger().info(
                f"Pose {idx}: "
                f"left={left if left else '---'}, "
                f"right={right if right else '---'}, "
                f"steps={steps}, delay={delay}, speed_scale={speed_scale}"
            )

            self.move_both_arms(left_pose=left,
                                right_pose=right,
                                base_steps=steps,
                                delay=delay,
                                speed_scale=speed_scale)

        self.get_logger().info("Rückkehr in Home-Pose (erste reale Pose).")
        self.move_arm_to_joints("left", self.left_home, base_steps=80, delay=0.02)
        self.move_arm_to_joints("right", self.right_home, base_steps=80, delay=0.02)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    rclpy.init()
    node = ReachyDualArmIK()

    yaml_file = choose_yaml_trajectory()
    if yaml_file is None:
        node.get_logger().error("Keine Trajektorie gewählt. Prozess wird beendet.")
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

    # Home-Pose ist bereits die reale Startpose, hier wird nur
    # nochmal sichergestellt, dass der ROS-Controller diesen Zustand kennt.
    node.get_logger().info("Synchronisiere Home-Pose (reale Startpose)...")
    node.move_arm_to_joints("left", node.left_home, base_steps=20, delay=0.02)
    node.move_arm_to_joints("right", node.right_home, base_steps=20, delay=0.02)

    node.get_logger().info("Starte Trajektorie...")
    node.follow_trajectory(trajectory)

    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Reachy-Motoren sauber ausschalten
        reachy.turn_off_smoothly('reachy')

