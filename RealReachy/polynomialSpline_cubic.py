import os
import sys
import glob
import time
import math
import yaml
import numpy as np
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import ikpy
from ikpy.chain import Chain
from ikpy.urdf import URDF
from ikpy.link import OriginLink

# ============================================================
#  MODE: GAZEBO oder REAL
# ============================================================

# Wenn du Gazebo nutzt: True
# Wenn du echten Reachy (ReachySDK) nutzt: False
USE_GAZEBO = True

# Reachy SDK Imports nur wenn REAL gebraucht wird
if not USE_GAZEBO:
    from reachy_sdk import ReachySDK
    from reachy_sdk.trajectory import goto
    from reachy_sdk.trajectory.interpolation import InterpolationMode


# ============================================================
#  KONFIG
# ============================================================

# YAML Ordner
YAML_DIR = os.path.expanduser("~/reachy_scripts/YAML-Files")

# URDFs (2 separate)
LEFT_ARM_URDF = os.path.expanduser(
    "~/reachy_ws/src/reachy_2023/reachy_description/urdf/left_arm.urdf"
)
RIGHT_ARM_URDF = os.path.expanduser(
    "~/reachy_ws/src/reachy_2023/reachy_description/urdf/right_arm.urdf"
)

# Joint Names (aktiv)
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

# Base / End Links in deinen ARM-URDFs
# Achtung: in deinem Beispiel-URDF existiert base_link und r_tool / l_tool.
LEFT_BASE_LINK = "base_link"
LEFT_END_LINK = "l_tool"

RIGHT_BASE_LINK = "base_link"
RIGHT_END_LINK = "r_tool"

# ROS Controller Topics
LEFT_CMD_TOPIC = "/l_arm_forward_position_controller/commands"
RIGHT_CMD_TOPIC = "/r_arm_forward_position_controller/commands"

# Default Timing
DEFAULT_STEPS = 80
DEFAULT_DELAY = 0.02
DEFAULT_SPEED_SCALE = 1.0

# ============================================================
#  Reachy SDK Setup (nur REAL)
# ============================================================

reachy = None
if not USE_GAZEBO:
    try:
        reachy = ReachySDK(host="localhost")
        reachy.turn_on("reachy")
        print("ReachySDK verbunden & Motoren an")
    except Exception as e:
        print("ReachySDK nicht erreichbar. Wenn du Gazebo willst: USE_GAZEBO=True")
        raise


# ============================================================
#  HILFSFUNKTIONEN
# ============================================================

def assert_file_exists(path: str, label: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} nicht gefunden: {path}")

def joints_dict_to_vector(self, arm, joints_dict, use_degrees=False):
    if arm == "left":
        names = LEFT_ARM_JOINT_NAMES
    else:
        names = RIGHT_ARM_JOINT_NAMES

    vec = []
    for n in names:
        v = joints_dict.get(n, 0.0)
        if use_degrees:
            v = math.radians(v)
        vec.append(v)

    return np.array(vec, dtype=float)

def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """4x4 Homogenmatrix (Rotation) aus Roll/Pitch/Yaw."""
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


def choose_yaml_trajectory_interactive(yaml_dir: str) -> str | None:
    """Interaktive YAML Auswahl."""
    pattern = os.path.join(yaml_dir, "*.yaml")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Keine YAML-Dateien in {yaml_dir} gefunden.")
        return None

    print("\nGefundene Trajektorien:")
    for i, f in enumerate(files, start=1):
        print(f"[{i}] {os.path.basename(f)}")

    while True:
        try:
            choice = input("\nWelche YAML-Trajektorie ausführen? (Nummer oder q=quit): ").strip()
            if choice.lower() in ("q", "quit", "exit"):
                return None
            idx = int(choice)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        except ValueError:
            pass
        print("Ungültige Eingabe, bitte erneut versuchen.")


def _validate_pose_dict(pose, label):
    if not isinstance(pose, dict):
        raise ValueError(f"{label} muss ein Dict sein.")

    has_cartesian = all(k in pose for k in ("x", "y", "z"))
    has_joints = "joints" in pose

    if not has_cartesian and not has_joints:
        raise ValueError(
            f"{label} muss entweder "
            f"(x,y,z) ODER 'joints' enthalten."
        )

    # Cartesian prüfen (nur wenn vorhanden)
    if has_cartesian:
        for key in ["x", "y", "z"]:
            if not isinstance(pose[key], (int, float)):
                raise ValueError(f"{label}['{key}'] muss Zahl sein.")

        for opt in ["roll", "pitch", "yaw"]:
            if opt in pose and not isinstance(pose[opt], (int, float)):
                raise ValueError(f"{label}['{opt}'] muss Zahl sein.")

    # Joint-Space prüfen (nur wenn vorhanden)
    if has_joints:
        if not isinstance(pose["joints"], dict):
            raise ValueError(f"{label}['joints'] muss ein Dict sein.")

        for j, v in pose["joints"].items():
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"{label}['joints']['{j}'] muss Zahl sein."
                )

    # Optional
    if "use_degrees" in pose and not isinstance(pose["use_degrees"], bool):
        raise ValueError(f"{label}['use_degrees'] muss bool sein.")



def validate_yaml_trajectory_allow_single_arm(yaml_data: dict):
    """
    Erlaubt:
      trajectory:
        - left: {x,y,z,...}   # right optional
          right: {x,y,z,...}  # left optional
          steps: int optional
          delay: float optional
          speed_scale: float optional
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
            raise ValueError(f"trajectory[{idx}] muss mindestens 'left' oder 'right' enthalten.")

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


def load_joint_limits_from_urdf(urdf_path: str, joint_names: list[str]):
    """
    Limits: lower/upper + velocity (wenn vorhanden).
    """
    limits = {n: {"min": None, "max": None, "velocity": None} for n in joint_names}
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in limits:
            continue
        lim = joint.find("limit")
        if lim is None:
            continue

        lo = lim.get("lower")
        hi = lim.get("upper")
        vel = lim.get("velocity")

        if lo is not None:
            limits[name]["min"] = float(lo)
        if hi is not None:
            limits[name]["max"] = float(hi)
        if vel is not None:
            limits[name]["velocity"] = float(vel)

    return limits


def clip_joints_with_limits(joint_vec, joint_names, limits):
    joint_vec = np.array(joint_vec, dtype=float)
    warnings = []
    for i, name in enumerate(joint_names):
        lim = limits.get(name, None)
        if lim is None:
            continue
        v = joint_vec[i]
        if lim["min"] is not None and v < lim["min"]:
            warnings.append(f"{name}: {v:.3f} < min {lim['min']:.3f}")
            joint_vec[i] = lim["min"]
        if lim["max"] is not None and v > lim["max"]:
            warnings.append(f"{name}: {v:.3f} > max {lim['max']:.3f}")
            joint_vec[i] = lim["max"]
    return joint_vec, warnings


def print_joint_limits(limits, label: str):
    print(f"\nGelenk-Limits aus URDF ({label}):")
    for name, lim in limits.items():
        print(f"  {name:18s} min={lim['min']}  max={lim['max']}  vel={lim['velocity']}")


def get_current_joint_positions_from_reachy_or_zero(arm: str) -> np.ndarray:
    """
    REAL: aus ReachySDK lesen (Grad -> Radiant)
    GAZEBO: 0 Pose
    """
    if USE_GAZEBO or reachy is None:
        if arm == "left":
            return np.zeros(NUM_LEFT_JOINTS, dtype=float)
        return np.zeros(NUM_RIGHT_JOINTS, dtype=float)

    # REAL
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
    else:
        joints_deg = [
            reachy.r_arm.r_shoulder_pitch.present_position,
            reachy.r_arm.r_shoulder_roll.present_position,
            reachy.r_arm.r_arm_yaw.present_position,
            reachy.r_arm.r_elbow_pitch.present_position,
            reachy.r_arm.r_forearm_yaw.present_position,
            reachy.r_arm.r_wrist_pitch.present_position,
            reachy.r_arm.r_wrist_roll.present_position,
        ]
    return np.radians(joints_deg)


# ============================================================
#  Interpolator API (keyword-only)
# ============================================================

import abc

class TrajectoryInterpolator(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        *,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        dt: float,
    ):
        pass


class MinimumJerkInterpolator(TrajectoryInterpolator):
    @staticmethod
    def _minimum_jerk(alpha: float) -> float:
        return 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5

    def generate(
        self,
        *,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        dt: float,
    ):
        start = np.array(start, dtype=float)
        target = np.array(target, dtype=float)

        if duration <= 0.0:
            yield target
            return

        steps = max(1, int(duration / dt))
        delta = target - start

        for i in range(steps):
            alpha = (i + 1) / steps
            s = self._minimum_jerk(alpha)
            q = start + s * delta
            yield q

        yield target


###################################
#Polynomial Spline Interpolator (2. Methode für Trajektorienglättung)
#Ersetzt eingebauten Minimum Jerk von Reachy
####################################

class PolynomialSplineInterpolator(TrajectoryInterpolator):
    """
    Polynomialische Splines (cubic / quintic)
    Drop-in Ersatz für MinimumJerkInterpolator
    """

    def __init__(self, mode: str = "quintic"):
        self.mode = mode

    def generate(self, *, start, target, duration, dt):
        start = np.array(start, dtype=float)
        target = np.array(target, dtype=float)

        if duration <= 0.0:
            yield target
            return

        # sanfter Start / Stopp (wichtig für Vergleichbarkeit!)
        v0 = np.zeros_like(start)
        v1 = np.zeros_like(start)
        a0 = np.zeros_like(start)
        a1 = np.zeros_like(start)

        steps = max(1, int(duration / dt))

        for i in range(steps):
            t = (i + 1) * dt
            t = min(t, duration)

            if self.mode == "cubic":
                q = self._cubic(start, target, v0, v1, duration, t)
            else:
                q = self._quintic(start, target, v0, v1, a0, a1, duration, t)

            print("Spline step:", q[:3]) #Debug 

            yield q

        yield target

    # ---------------- mathematics ----------------

    def _cubic(self, q0, q1, v0, v1, T, t):
        s = t / T
        s2 = s * s
        s3 = s2 * s

        h00 =  2*s3 - 3*s2 + 1
        h10 =      s3 - 2*s2 + s
        h01 = -2*s3 + 3*s2
        h11 =      s3 -   s2

        return h00*q0 + h10*T*v0 + h01*q1 + h11*T*v1

    def _quintic(self, q0, q1, v0, v1, a0, a1, T, t):
        T2, T3, T4, T5 = T*T, T**3, T**4, T**5

        c0 = q0
        c1 = v0
        c2 = 0.5 * a0

        c3 = (20*(q1-q0) - (8*v1+12*v0)*T - (3*a0-a1)*T2) / (2*T3)
        c4 = (30*(q0-q1) + (14*v1+16*v0)*T + (3*a0-2*a1)*T2) / (2*T4)
        c5 = (12*(q1-q0) - (6*v1+6*v0)*T - (a0-a1)*T2) / (2*T5)

        return (
            c0
            + c1*t
            + c2*t*t
            + c3*t**3
            + c4*t**4
            + c5*t**5
        )


# ============================================================
#  IKPy Chain Builder (ikpy version robust)
# ============================================================

def build_arm_chain_from_urdf(
        urdf_path: str, 
        joint_names: list, 
        chain_name: str
        ):
    """
    Baut eine IKPy-Chain aus einer ARM-spezifischen URDF (IKPy 3.4.2 kompatibel).
    """

    chain = Chain.from_urdf_file(urdf_path)

    # ---- aktive Gelenke setzen (nur die echten Arm-Gelenke) ----
    active_mask = []

    for link in chain.links:
        if link.name in joint_names:
            active_mask.append(True)
        else:
            active_mask.append(False)

    chain.active_links_mask = active_mask
    chain.name = chain_name

    return chain



# ============================================================
#  ROS2 Node: Dual Arm IK
# ============================================================

class ReachyDualArmIK(Node):
    def __init__(self):
        super().__init__("reachy_dual_arm_ik")

        # --- Publisher
        self.l_pub = self.create_publisher(Float64MultiArray, LEFT_CMD_TOPIC, 10)
        self.r_pub = self.create_publisher(Float64MultiArray, RIGHT_CMD_TOPIC, 10)

        # --- Interpolator (Methode 2 = PolynomialSplineInterpolator im Modus "cubic")
        self.interpolator = PolynomialSplineInterpolator(mode="cubic")

        # --- URDFs prüfen
        assert_file_exists(LEFT_ARM_URDF, "LEFT_ARM_URDF")
        assert_file_exists(RIGHT_ARM_URDF, "RIGHT_ARM_URDF")

        self.get_logger().info("Lade Arm-IK-Ketten aus separaten URDFs:")
        self.get_logger().info(f"  LEFT : {LEFT_ARM_URDF}")
        self.get_logger().info(f"  RIGHT: {RIGHT_ARM_URDF}")

        # --- IK Chains
        self.left_chain = build_arm_chain_from_urdf(
            urdf_path=LEFT_ARM_URDF,
            joint_names=LEFT_ARM_JOINT_NAMES,
            chain_name="left_arm_chain",
            
        )
        self.right_chain = build_arm_chain_from_urdf(
            urdf_path=RIGHT_ARM_URDF,
            joint_names=RIGHT_ARM_JOINT_NAMES,
            chain_name="right_arm_chain",
        )

        self.get_logger().info(f"Left chain links : {len(self.left_chain.links)}")
        self.get_logger().info(f"Right chain links: {len(self.right_chain.links)}")

        # --- Limits aus URDF
        self.left_limits = load_joint_limits_from_urdf(LEFT_ARM_URDF, LEFT_ARM_JOINT_NAMES)
        self.right_limits = load_joint_limits_from_urdf(RIGHT_ARM_URDF, RIGHT_ARM_JOINT_NAMES)

        print_joint_limits(self.left_limits, "Left Arm")
        print_joint_limits(self.right_limits, "Right Arm")

        # --- Home Pose (Real oder Gazebo)
        self.get_logger().info("Lese aktuelle Gelenkstellung als Home-Pose...")

        self.left_home = get_current_joint_positions_from_reachy_or_zero("left")
        self.right_home = get_current_joint_positions_from_reachy_or_zero("right")

        self.left_current = self.left_home.copy()
        self.right_current = self.right_home.copy()

        # Log in Grad
        self.get_logger().info("----------------------------------------------------")
        self.get_logger().info("HOME-POSE (LINKER ARM)")
        for n, rad in zip(LEFT_ARM_JOINT_NAMES, self.left_home):
            self.get_logger().info(f"  {n:17s}: {rad:8.4f} rad | {math.degrees(rad):8.2f}°")

        self.get_logger().info("----------------------------------------------------")
        self.get_logger().info("HOME-POSE (RECHTER ARM)")
        for n, rad in zip(RIGHT_ARM_JOINT_NAMES, self.right_home):
            self.get_logger().info(f"  {n:17s}: {rad:8.4f} rad | {math.degrees(rad):8.2f}°")

        self.get_logger().info("----------------------------------------------------")

    # -------------------------------------------------------
    # IK Compute
    # -------------------------------------------------------

    def _ik_chain_for_arm(self, arm: str):
        if arm == "left":
            return self.left_chain, LEFT_ARM_JOINT_NAMES, NUM_LEFT_JOINTS
        return self.right_chain, RIGHT_ARM_JOINT_NAMES, NUM_RIGHT_JOINTS

    def compute_ik_6d(self, arm: str, x: float, y: float, z: float,
                      roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(roll, pitch, yaw)[:3, :3]
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z

        chain, joint_names, nj = self._ik_chain_for_arm(arm)

        # IKPy Versionen: inverse_kinematics(target_frame=...) oder inverse_kinematics_frame
        try:
            full = chain.inverse_kinematics(target_frame=T)
        except TypeError:
            full = chain.inverse_kinematics_frame(T)

        # IKPy liefert full-array über alle Links -> aktive Joints extrahieren
        # Häufig: full[1:1+nj] (0 ist base)
        # Wenn du active_links_mask sauber gesetzt hast, passt das.
        joints = np.array(full[1:1 + nj], dtype=float)
        return joints

    # -------------------------------------------------------
    # Velocity-limits -> Schritte planen
    # -------------------------------------------------------

    def plan_steps_with_velocity_limits(self, current, target,
                                        joint_names, limits,
                                        base_steps=DEFAULT_STEPS,
                                        delay=DEFAULT_DELAY,
                                        speed_scale=DEFAULT_SPEED_SCALE) -> int:
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

        return max(base_steps, int(math.ceil(max_steps_required)))

    # -------------------------------------------------------
    # Send commands (with limit clipping)
    # -------------------------------------------------------

    def send_arm(self, arm: str, joints: np.ndarray):
        if arm == "left":
            joint_names = LEFT_ARM_JOINT_NAMES
            limits = self.left_limits
        else:
            joint_names = RIGHT_ARM_JOINT_NAMES
            limits = self.right_limits

        clipped, warnings = clip_joints_with_limits(joints, joint_names, limits)
        if warnings:
            self.get_logger().warn(
                f"Limitverletzung {arm}:\n  " + "\n  ".join(warnings)
            )
         
        # Debug, ob rad korrekt gesendet wird
        self.get_logger().info(
            f"SEND {arm.upper()} JOINTS [rad]: "
            f"{[round(v, 4) for v in clipped.tolist()]}"
        )

        msg = Float64MultiArray()
        msg.data = clipped.tolist()

        if arm == "left":
            self.l_pub.publish(msg)
        else:
            self.r_pub.publish(msg)

    # -------------------------------------------------------
    # Move single arm using Interpolator (Lösung A)
    # -------------------------------------------------------

    def move_arm_to_joints(self, arm: str, target: np.ndarray,
                           base_steps=DEFAULT_STEPS,
                           delay=DEFAULT_DELAY,
                           speed_scale=DEFAULT_SPEED_SCALE):

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
            current=current,
            target=target,
            joint_names=joint_names,
            limits=limits,
            base_steps=base_steps,
            delay=delay,
            speed_scale=speed_scale
        )
        if steps <= 0:
            return

        duration = steps * delay

        for q in self.interpolator.generate(
            start=current,
            target=target,
            duration=duration,
            dt=delay,
        ):
            self.send_arm(arm, q)
            time.sleep(delay)

        if arm == "left":
            self.left_current = target.copy()
        else:
            self.right_current = target.copy()

    # -------------------------------------------------------
    # Move both arms simultaneously using Interpolator (Lösung A)
    # -------------------------------------------------------

    def move_both_arms(self, left_pose=None, right_pose=None,
                       base_steps=DEFAULT_STEPS, delay=DEFAULT_DELAY,
                       speed_scale=DEFAULT_SPEED_SCALE):

        left_target = self.left_current.copy()
        right_target = self.right_current.copy()

        # LEFT
        if left_pose is not None:

            # ----------------------------------------
            # Joint-YAML -> DIREKT FAHREN
            # ----------------------------------------
            if "joints" in left_pose:
                left_target = self.joints_dict_to_vector(
                    "left",
                    left_pose["joints"],
                    left_pose.get("use_degrees", False),
                )

            # ----------------------------------------
            # Cartesian-YAML -> IK
            # ----------------------------------------
            else:
                left_target = self.compute_ik_6d(
                    "left",
                    left_pose["x"],
                    left_pose["y"],
                    left_pose["z"],
                    left_pose.get("roll", 0.0),
                    left_pose.get("pitch", 0.0),
                    left_pose.get("yaw", 0.0),
                )


        # RIGHT
        if right_pose is not None:

            # ----------------------------------------
            # Joint-YAML -> DIREKT FAHREN
            # ----------------------------------------
            if "joints" in right_pose:
                right_target = self.joints_dict_to_vector(
                    "right",
                    right_pose["joints"],
                    right_pose.get("use_degrees", False),
                )

            # ----------------------------------------
            # Cartesian-YAML -> IK
            # ----------------------------------------
            else:
                right_target = self.compute_ik_6d(
                    "right",
                    right_pose["x"],
                    right_pose["y"],
                    right_pose["z"],
                    right_pose.get("roll", 0.0),
                    right_pose.get("pitch", 0.0),
                    right_pose.get("yaw", 0.0),
                )


        # Steps pro Arm
        left_steps = self.plan_steps_with_velocity_limits(
            current=self.left_current,
            target=left_target,
            joint_names=LEFT_ARM_JOINT_NAMES,
            limits=self.left_limits,
            base_steps=base_steps,
            delay=delay,
            speed_scale=speed_scale,
        )
        right_steps = self.plan_steps_with_velocity_limits(
            current=self.right_current,
            target=right_target,
            joint_names=RIGHT_ARM_JOINT_NAMES,
            limits=self.right_limits,
            base_steps=base_steps,
            delay=delay,
            speed_scale=speed_scale,
        )

        steps = max(left_steps, right_steps)
        if steps <= 0:
            return

        duration = steps * delay

        left_start = self.left_current.copy()
        right_start = self.right_current.copy()

        # Wenn Arm nicht gesetzt ist, bleibt er konstant (Generator mit start==target)
        if left_pose is None:
            left_target = left_start
        if right_pose is None:
            right_target = right_start

        left_gen = self.interpolator.generate(
            start=left_start,
            target=left_target,
            duration=duration,
            dt=delay,
        )
        right_gen = self.interpolator.generate(
            start=right_start,
            target=right_target,
            duration=duration,
            dt=delay,
        )

        for ql, qr in zip(left_gen, right_gen):
            self.send_arm("left", ql)
            self.send_arm("right", qr)
            time.sleep(delay)

        self.left_current = left_target.copy()
        self.right_current = right_target.copy()

    # -------------------------------------------------------
    # Trajectory execution
    # -------------------------------------------------------

    def follow_trajectory(self, trajectory: list[dict]):
        for idx, p in enumerate(trajectory):
            steps = p.get("steps", DEFAULT_STEPS)
            delay = p.get("delay", DEFAULT_DELAY)
            speed_scale = p.get("speed_scale", DEFAULT_SPEED_SCALE)

            left = p.get("left", None)
            right = p.get("right", None)

            self.get_logger().info(
                f"Pose {idx}: left={'yes' if left else '---'}, right={'yes' if right else '---'}, "
                f"steps={steps}, delay={delay}, speed_scale={speed_scale}"
            )

            self.move_both_arms(
                left_pose=left,
                right_pose=right,
                base_steps=steps,
                delay=delay,
                speed_scale=speed_scale,
            )

        self.get_logger().info("Rückkehr in Home-Pose.")
        self.move_arm_to_joints("left", self.left_home, base_steps=DEFAULT_STEPS, delay=DEFAULT_DELAY)
        self.move_arm_to_joints("right", self.right_home, base_steps=DEFAULT_STEPS, delay=DEFAULT_DELAY)

#############################################
#Umwandeln der Joint Werte aus YAML in Gelenkvektoren (radianten für IKPY)
#############################################

    def joints_dict_to_vector(self, arm: str, joints_dict: dict, use_degrees: bool):

        if arm == "left":
            joint_names = LEFT_ARM_JOINT_NAMES
        else:
            joint_names = RIGHT_ARM_JOINT_NAMES

        vec = []

        for name in joint_names:
            if name not in joints_dict:
                raise ValueError(
                    f"Joint '{name}' fehlt im joints-Dict für Arm '{arm}'"
                )

            val = joints_dict[name]
            if use_degrees:
                val = math.radians(val)

            vec.append(val)

        return np.array(vec, dtype=float)
    

    # ==========================================================
    # Forward Kinematics: Joint-Vektor -> Cartesian Pose
    # ==========================================================
    def fk_to_cartesian(self, arm: str, joint_vec: np.ndarray):
        if arm == "left":
            chain = self.left_chain
        else:
            chain = self.right_chain

        joint_vec = np.array(joint_vec, dtype=float)

        # IKPy erwartet einen Wert pro Link
        n_links = len(chain.links)

        # IKPy-Convention:
        # joints[0] = base (dummy)
        # danach aktive + fixed joints
        full = np.zeros(n_links)

        # aktive Gelenke ab Index 1 einsetzen
        full[1:1 + len(joint_vec)] = joint_vec

        # Forward Kinematics
        T = chain.forward_kinematics(full)

        x = float(T[0, 3])
        y = float(T[1, 3])
        z = float(T[2, 3])

        R = T[:3, :3]

        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))
        yaw = math.atan2(R[1, 0], R[0, 0])

        return {
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }

# ============================================================
#  MAIN
# ============================================================

def main():
    if USE_GAZEBO:
        print("Gazebo jetzt aktiv")
    else:
        print("Realer Reachy jetzt aktiv")

    # ROS init
    rclpy.init()

    node = ReachyDualArmIK()

    # YAML Auswahl Loop: wenn YAML falsch -> erneut auswählen
    while True:
        yaml_file = choose_yaml_trajectory_interactive(YAML_DIR)
        if yaml_file is None:
            node.get_logger().warn("Keine Trajektorie gewählt. Beende.")
            rclpy.shutdown()
            return

        node.get_logger().info(f"Lade YAML-Trajektorie: {yaml_file}")
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
            validate_yaml_trajectory_allow_single_arm(data)
            break
        except Exception as e:
            node.get_logger().error(f"YAML-Validierung fehlgeschlagen: {e}")
            print("\n Bitte wähle eine andere YAML-Datei.\n")
            continue

    trajectory = data["trajectory"]

    # Synchronisieren der Home Pose (Gazebo: 0 Pose, Real: SDK Pose)
    node.get_logger().info("Synchronisiere Home-Pose...")
    node.move_arm_to_joints("left", node.left_home, base_steps=20, delay=DEFAULT_DELAY)
    node.move_arm_to_joints("right", node.right_home, base_steps=20, delay=DEFAULT_DELAY)

    node.get_logger().info("Starte Trajektorie...")
    node.follow_trajectory(trajectory)

    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Reachy sauber ausschalten (nur REAL)
        if not USE_GAZEBO and reachy is not None:
            try:
                reachy.turn_off_smoothly("reachy")
                print("Reachy Motoren ausgeschaltet")
            except Exception as e:
                print(f"Reachy turn_off_smoothly fehlgeschlagen: {e}")
