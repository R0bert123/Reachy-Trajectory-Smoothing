import os
import csv
import sys
import glob
import time
import math
import yaml
import numpy as np
import xml.etree.ElementTree as ET
import abc

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from scipy.interpolate import BSpline
from datetime import datetime

import ikpy
from ikpy.chain import Chain
from ikpy.urdf import URDF
from ikpy.link import OriginLink
from sensor_msgs.msg import JointState
from pathlib import Path

# ============================================================
#  MODE: GAZEBO or REAL (Physical Reachy)
# ============================================================

# Set to "True" when using Gazebo
# Set to "False" when using physical Reachy with ReachySDK
USE_GAZEBO = False

# Reachy SDK Imports only when using the physical Reachy
if not USE_GAZEBO:
    from reachy_sdk import ReachySDK
    from reachy_sdk.trajectory import goto
    from reachy_sdk.trajectory.interpolation import InterpolationMode

def command_grippers_sdk(
        l_deg=None,
        r_deg=None,
        duration=0.4,
        mode=None,
        ):
    if USE_GAZEBO:
        return
    if mode is None:
        mode = InterpolationMode.MINIMUM_JERK
        
    # Gripper positions in degrees, e.g. -35 (open) to -5 (closed) for the right gripper
    if l_deg is not None:
        goals[reachy.l_arm.l_gripper] = float(l_deg)
    if r_deg is not None:
        goals[reachy.r_arm.r_gripper] = float(r_deg)

    if goals:
        goto(goal_positions=goals, duration=float(duration), interpolation_mode=mode)


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# YAML trajectory files
YAML_DIR = PROJECT_ROOT / "reachy_scripts" / "YAML-Files"

# Arm-specific URDF files used by IKPy
URDF_DIR = PROJECT_ROOT / "reachy_description" / "urdf"

LEFT_ARM_URDF = URDF_DIR / "left_arm_real_reachy.urdf"
RIGHT_ARM_URDF = URDF_DIR / "right_arm_real_reachy.urdf"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
#  Configuration
# ============================================================

# Active joint names
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

# Gripper names
GRIPPER_NAMES = ["l_gripper", "r_gripper"]

NUM_LEFT_JOINTS = len(LEFT_ARM_JOINT_NAMES)
NUM_RIGHT_JOINTS = len(RIGHT_ARM_JOINT_NAMES)

# Base / End Links in ARM-URDFs
# Note: In the example URDF the base_link and r_tool / l_tool are defined.
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
#  Reachy SDK Setup (only physical Reachy)
# ============================================================

reachy = None
if not USE_GAZEBO:
    try:
        reachy = ReachySDK(host="localhost")
        reachy.turn_on("reachy")
        print("ReachySDK connected and motors enabled")
    except Exception as e:
        print("ReachySDK is NOT reachable. Set USE_GAZEBO=True if you want to use Gazebo.")
        raise


# ============================================================
#  Helper Functions
# ============================================================

def assert_file_exists(path: str, label: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """4x4 homogeneous transformation matrix (rotation) from roll/pitch/yaw."""
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

def choose_interpolation_method_interactive() -> str | None:
    print("\n Choose interpolation method:")
    print("[1] Minimum Jerk")
    print("[2] Polynomial Spline (Cubic)")
    print("[3] Polynomial Spline (Quintic)")
    print("[4] B-Spline")
    print("[5] Bezier Curve")
    print("[6] No Interpolation")

    mapping = {
        "1": "minimum_jerk",
        "2": "polynomial_spline_cubic",
        "3": "polynomial_spline_quintic",
        "4": "b_spline",
        "5": "bezier_curve",
        "6": "none",
    }

    while True:
        choice = input("\n Enter number (or q to quit): ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            return None
        if choice in mapping:
            return mapping[choice]
        print("Invalid selection, please try again.")


def choose_yaml_trajectory_interactive(yaml_dir: str) -> str | None:
    """Interactive YAML file selection."""
    pattern = os.path.join(yaml_dir, "*.yaml")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No YAML files found in {yaml_dir}.")
        return None

    print("\nAvailable trajectories:")
    for i, f in enumerate(files, start=1):
        print(f"[{i}] {os.path.basename(f)}")

    while True:
        try:
            choice = input("\nWhich YAML trajectory would you like to execute? (Enter a number or q to quit)").strip()
            if choice.lower() in ("q", "quit", "exit"):
                return None
            idx = int(choice)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        except ValueError:
            pass
        print("Invalid input. Please try again.")


def _validate_pose_dict(pose, label):
    if not isinstance(pose, dict):
        raise ValueError(f"{label} must be a dictionary.")

    has_cartesian = all(k in pose for k in ("x", "y", "z"))
    has_joints = "joints" in pose
    has_gripper = any(k in pose for k in ("l_gripper", "r_gripper"))

    if not (has_cartesian or has_joints or has_gripper):
        raise ValueError(
            f"{label} must contain either "
            f"(x,y,z), 'joints' or a gripper ('l_gripper'/'r_gripper')."
        )

    # Validate Cartesian coordinates
    if has_cartesian:
        for key in ("x", "y", "z"):
            if not isinstance(pose[key], (int, float)):
                raise ValueError(f"{label}['{key}'] must be a number.")

        for opt in ("roll", "pitch", "yaw"):
            if opt in pose and not isinstance(pose[opt], (int, float)):
                raise ValueError(f"{label}['{opt}'] must be a number.")

    # Validate joint space data
    if has_joints:
        if not isinstance(pose["joints"], dict):
            raise ValueError(f"{label}['joints'] must be a dictionary.")

        for j, v in pose["joints"].items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"{label}['joints']['{j}'] must be a number.")

    # Validate gripper values
    for g in ("l_gripper", "r_gripper"):
        if g in pose and not isinstance(pose[g], (int, float)):
            raise ValueError(f"{label}['{g}'] must be a number (in degrees).")

    # Optional
    if "use_degrees" in pose and not isinstance(pose["use_degrees"], bool):
        raise ValueError(f"{label}['use_degrees'] must be a boolean.")


def validate_yaml_trajectory_allow_single_arm(yaml_data: dict):
    """
    Supported format:
      trajectory:
        - left: {x,y,z,...}   # right optional
          right: {x,y,z,...}  # left optional
          steps: int optional
          delay: float optional
          speed_scale: float optional
    """
    if not isinstance(yaml_data, dict):
        raise ValueError("The YAML root must be a dictionary.")
    if "trajectory" not in yaml_data:
        raise ValueError('The YAML file must contain the "trajectory" field.')

    traj = yaml_data["trajectory"]
    # Optional global setting
    if "return_home" in yaml_data and not isinstance(yaml_data["return_home"], bool):
        raise ValueError('"return_home" must be a boolean (true/false).')

    if not isinstance(traj, list):
        raise ValueError('"trajectory" must be a list.')

    for idx, yamlCheckpoint in enumerate(traj):
        if not isinstance(yamlCheckpoint, dict):
            raise ValueError(f"trajectory[{idx}] is not a dictionary.")

        # --- Interactive wait-enter step ---
        if yamlCheckpoint.get("wait_enter", False):
            if not isinstance(yamlCheckpoint["wait_enter"], bool):
                raise ValueError(f"trajectory[{idx}]['wait_enter'] must be a boolean.")

            # Right side
            if "close_gripper_on_enter" in yamlCheckpoint and not isinstance(
                yamlCheckpoint["close_gripper_on_enter"], bool
            ):
                raise ValueError(
                    f"trajectory[{idx}]['close_gripper_on_enter'] must be a boolean."
                )

            # Left side
            if "close_left_gripper_on_enter" in yamlCheckpoint and not isinstance(
                yamlCheckpoint["close_left_gripper_on_enter"], bool
            ):
                raise ValueError(
                    f"trajectory[{idx}]['close_left_gripper_on_enter'] must be a boolean."
                )

            # Optional: Opens Gripper when pressing the button "ENTER"
            if "open_left_gripper_on_enter" in yamlCheckpoint and not isinstance(
                yamlCheckpoint["open_left_gripper_on_enter"], bool
            ):
                raise ValueError(
                    f"trajectory[{idx}]['open_left_gripper_on_enter'] must be a boolean."
                )

            # Optional: Target positions in degrees (if specified)
            if "left_gripper_on_enter_deg" in yamlCheckpoint and not isinstance(
                yamlCheckpoint["left_gripper_on_enter_deg"], (int, float)
            ):
                raise ValueError(
                    f"trajectory[{idx}]['left_gripper_on_enter_deg'] must be a number."
                )

            if "right_gripper_on_enter_deg" in yamlCheckpoint and not isinstance(
                yamlCheckpoint["right_gripper_on_enter_deg"], (int, float)
            ):
                raise ValueError(
                    f"trajectory[{idx}]['right_gripper_on_enter_deg'] must be a number."
                )

            # This is an action, not a movement. So it is valid.
            continue

        left = yamlCheckpoint.get("left")
        right = yamlCheckpoint.get("right")

        if left is None and right is None:
            raise ValueError(
                f"trajectory[{idx}] must contain at least 'left' or 'right' "
                f"(or 'wait_enter: true')."
            )

        if left is not None:
            _validate_pose_dict(left, f"trajectory[{idx}]['left']")
        if right is not None:
            _validate_pose_dict(right, f"trajectory[{idx}]['right']")

        if "steps" in yamlCheckpoint and not isinstance(yamlCheckpoint["steps"], int):
            raise ValueError(f"trajectory[{idx}]['steps'] must be an integer.")
        if "delay" in yamlCheckpoint and not isinstance(yamlCheckpoint["delay"], (int, float)):
            raise ValueError(f"trajectory[{idx}]['delay'] must be a float or integer.")
        if "speed_scale" in yamlCheckpoint and not isinstance(yamlCheckpoint["speed_scale"], (int, float)):
            raise ValueError(f"trajectory[{idx}]['speed_scale'] must be a float or integer.")


def load_joint_limits_from_urdf(urdf_path: str, joint_names: list[str]):
    """
    Load joint limits: lower/upper bounds + velocity (if available).
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
    print(f"\nJoint limits from URDF file ({label}):")
    for name, lim in limits.items():
        print(f"  {name:18s} min={lim['min']}  max={lim['max']}  vel={lim['velocity']}")


def get_current_joint_positions_from_reachy_or_zero(arm: str) -> np.ndarray:
    """
    REAL: Read joint positions from ReachySDK (degrees -> radians).
    GAZEBO: Return the zero pose.
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

###################################
# Minimum Jerk Interpolator
###################################

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
# Polynomial Spline Interpolator 
###################################

class PolynomialSplineInterpolator(TrajectoryInterpolator):
    """
    Polynomial splines (cubic / quintic)
    Drop-in replacement for MinimumJerkInterpolator.
    """

    def __init__(self, mode: str = "quintic"):
        self.mode = mode

    def generate(self, *, start, target, duration, dt):
        start = np.array(start, dtype=float)
        target = np.array(target, dtype=float)

        if duration <= 0.0:
            yield target
            return

        # Smooth start/stop (important for comparability)
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
##############################
# B-Spline Interpolator
##############################

class BSplineInterpolator(TrajectoryInterpolator):
    """
    B-Spline interpolation in joint space.
    """

    def __init__(self, degree: int = 3):
        self.degree = degree  # cubic B-spline by default

    def generate(self, *, start, target, duration, dt):
        start = np.array(start, dtype=float)
        target = np.array(target, dtype=float)

        if duration <= 0.0:
            yield target
            return

        steps = max(2, int(duration / dt))

        # Control points: start -> target 
        control_points = np.vstack([
            start,
            start,
            target,
            target
        ])

        n_ctrl = control_points.shape[0]
        degree = min(self.degree, n_ctrl - 1)

        # Knot vector (clamped)
        knots = np.concatenate([
            np.zeros(degree),
            np.linspace(0, 1, n_ctrl - degree + 1),
            np.ones(degree)
        ])

        spline = BSpline(knots, control_points, degree)

        for i in range(steps):
            u = (i + 1) / steps
            q = spline(u)
            yield q

        yield target

########################
# Bezier Interpolator
########################

class BezierInterpolator(TrajectoryInterpolator):
    """
    Cubic Bezier curve interpolation
    """

    def generate(self, *, start, target, duration, dt):
        start = np.array(start, dtype=float)
        target = np.array(target, dtype=float)

        if duration <= 0.0:
            yield target
            return

        steps = max(2, int(duration / dt))

        # Control points
        p0 = start
        p1 = start
        p2 = target
        p3 = target

        for i in range(steps):
            t = (i + 1) / steps
            q = (
                (1 - t)**3 * p0
                + 3 * (1 - t)**2 * t * p1
                + 3 * (1 - t) * t**2 * p2
                + t**3 * p3
            )
            yield q

        yield target


# ============================================================
#  IKPy Chain Builder (compatible with IKPy 3.4.2)
# ============================================================

def build_arm_chain_from_urdf(
        urdf_path: str, 
        joint_names: list, 
        chain_name: str
        ):
    """
    Build an IKPy chain from an arm-specific URDF (compatible with IKPy 3.4.2).
    """

    chain = Chain.from_urdf_file(urdf_path)

    # ---- Set active joints (only the actual arm joints) ----
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
    def __init__(self, interpolation_method: str):
        super().__init__("reachy_dual_arm_ik")

        # --- Publisher
        self.l_pub = self.create_publisher(Float64MultiArray, LEFT_CMD_TOPIC, 10)
        self.r_pub = self.create_publisher(Float64MultiArray, RIGHT_CMD_TOPIC, 10)

        # --- Gazebo joint_states tracking ---
        self._gazebo_joint_pos = {}  # name -> position(rad)

        if USE_GAZEBO:
            self.create_subscription(
                JointState,
                "/joint_states",
                self._on_joint_state,
                10,
            )
            self.get_logger().info("Subscribed to /joint_states for Gazebo measurements.")


        # --- Interpolator Selection Check 
        if interpolation_method == "minimum_jerk":
            self.interpolator = MinimumJerkInterpolator()
            self.interpolation_mode = "Minimum Jerk"

        elif interpolation_method == "polynomial_spline_cubic":
            self.interpolator = PolynomialSplineInterpolator(mode="cubic")
            self.interpolation_mode = "Polynomial Spline (Mode:Cubic)"

        elif interpolation_method == "polynomial_spline_quintic":
            self.interpolator = PolynomialSplineInterpolator(mode="quintic")
            self.interpolation_mode = "Polynomial Spline (Mode:Quintic)"

        elif interpolation_method == "b_spline":
            self.interpolator = BSplineInterpolator(degree=3)
            self.interpolation_mode = "B-Spline (degree = 3)"

        elif interpolation_method == "bezier_curve":
            self.interpolator = BezierInterpolator()
            self.interpolation_mode = "Bezier Curve"

        elif interpolation_method == "none":
            self.interpolator = None
            self.interpolation_mode = "No Interpolator"

        else:
            raise ValueError(f"Unknown INTERPOLATION_METHOD: {interpolation_method}")


        if self.interpolator is None:
            self.get_logger().info("Interpolation class: NONE")
        else:
            self.get_logger().info(f"Interpolation class: {type(self.interpolator).__name__}")
            if isinstance(self.interpolator, PolynomialSplineInterpolator):
                self.get_logger().info(f"Polynomial spline mode: {self.interpolator.mode}")

        if isinstance(self.interpolator, PolynomialSplineInterpolator):
            self.get_logger().info(
                f"Polynomial spline mode: {self.interpolator.mode}"
            )

        # --- Trajectory counters 
        self.step_counter = {
            "left": 0,
            "right": 0
        }

        # --- Maximum joint velocity
        self.max_joint_velocity = {
            "left": 0.0,
            "right": 0.0
        }

        # --- Previous joint position
        self.previous_joint_position = {
            "left": None,
            "right": None
        }

        # --- Global max jerk
        self.max_joint_jerk = {
            "left": 0.0,
            "right": 0.0
        }

        # --- Joint-wise jerk estimation 
        self.max_joint_jerk_per_joint = {
            "left": np.zeros(NUM_LEFT_JOINTS),
            "right": np.zeros(NUM_RIGHT_JOINTS),
        }

        # --- Previous joint velocity vector (for each jerk)
        self.previous_joint_velocity_vector = {
            "left": None,
            "right": None,
        }

        # --- Mean jerk estimation (joint-wise) 
        self.sum_joint_jerk_per_joint = {
            "left": np.zeros(NUM_LEFT_JOINTS),
            "right": np.zeros(NUM_RIGHT_JOINTS),
        }

        # --- Number of valid jerk samples
        self.jerk_sample_count = {
            "left": 0,
            "right": 0,
        }

        # RMSE jerk: Sum of squares (global over all joints) + count ---
        #self.sum_sq_jerk = {"left": 0.0, "right": 0.0}
        #self.jerk_sq_sample_count = {"left": 0, "right": 0}

        # Optional: RMSE pro Joint
        self.sum_sq_jerk_per_joint = {
            "left": np.zeros(NUM_LEFT_JOINTS),
            "right": np.zeros(NUM_RIGHT_JOINTS),
        }
        # Real Jerk
        self.previous_joint_acceleration_vector = {"left": None, "right": None}
        self.sum_sq_jerk = {"left": 0.0, "right": 0.0}
        self.jerk_sq_sample_count = {"left": 0, "right": 0}

        # --- Final position error metrics (across the entire YAML trajectory) ---
        self.final_position_error_max = 0.0     # max(|q_measured - q_target|)
        self.final_position_error_rmse = 0.0    # RMSE across all samples (see below)

        # Calculate final position RMSE across the entire trajectory, not just the worst waypoint
        self.sum_squared_final_position_error = 0.0
        self.final_position_error_sample_count = 0


        # --- Validate URDF files ---
        assert_file_exists(LEFT_ARM_URDF, "LEFT_ARM_URDF")
        assert_file_exists(RIGHT_ARM_URDF, "RIGHT_ARM_URDF")

        self.get_logger().info("Loading arm IK chains from separate URDF files:")
        self.get_logger().info(f"  LEFT : {LEFT_ARM_URDF}")
        self.get_logger().info(f"  RIGHT: {RIGHT_ARM_URDF}")

        # --- IK chains ---
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

        # --- Limits from URDF file ---
        self.left_limits = load_joint_limits_from_urdf(LEFT_ARM_URDF, LEFT_ARM_JOINT_NAMES)
        self.right_limits = load_joint_limits_from_urdf(RIGHT_ARM_URDF, RIGHT_ARM_JOINT_NAMES)

        print_joint_limits(self.left_limits, "Left Arm")
        print_joint_limits(self.right_limits, "Right Arm")

        # --- Home Pose (Real Robot or Gazebo) ---
        self.get_logger().info("Read initial joint positions as the home pose.")

        self.left_home = get_current_joint_positions_from_reachy_or_zero("left")
        self.right_home = get_current_joint_positions_from_reachy_or_zero("right")

        self.left_current = self.left_home.copy()
        self.right_current = self.right_home.copy()

        # Log joint positions in degrees
        self.get_logger().info("----------------------------------------------------")
        self.get_logger().info("HOME POSE (LEFT ARM)")
        for n, rad in zip(LEFT_ARM_JOINT_NAMES, self.left_home):
            self.get_logger().info(f"  {n:17s}: {rad:8.4f} rad | {math.degrees(rad):8.2f}°")

        self.get_logger().info("----------------------------------------------------")
        self.get_logger().info("HOME POSE (RIGHT ARM)")
        for n, rad in zip(RIGHT_ARM_JOINT_NAMES, self.right_home):
            self.get_logger().info(f"  {n:17s}: {rad:8.4f} rad | {math.degrees(rad):8.2f}°")

        self.get_logger().info("----------------------------------------------------")

############## Helper Functions ###############
    # Callback method
    def _on_joint_state(self, msg: JointState):
        # Save current joint positions (rad)
        for name, pos in zip(msg.name, msg.position):
            self._gazebo_joint_pos[name] = float(pos)

    # Measurement function for final_position_error
    def get_measured_joint_positions(self, arm: str) -> np.ndarray:
        """
        Return the measured joint angles in radians.
        - REAL: Read from ReachySDK (present_position in degrees -> radians)
        - GAZEBO: Read from /joint_states
        """
        if USE_GAZEBO:
            if arm == "left":
                names = LEFT_ARM_JOINT_NAMES
            else:
                names = RIGHT_ARM_JOINT_NAMES

            missing = [n for n in names if n not in self._gazebo_joint_pos]
            if missing:
                self.get_logger().warn(
                    f"Gazebo /joint_states are not yet available for: {missing}. "
                    "The final error may be 0 or invalid at startup."
                )
                return np.zeros(len(names), dtype=float)

            return np.array([self._gazebo_joint_pos[n] for n in names], dtype=float)

        # REAL robot
        return get_current_joint_positions_from_reachy_or_zero(arm)

    # Helper function to determine final_position_error for the CSV output
    # Wait until the measured joint positions have settled (stopped moving)
    # to ensure a more accurate measurement.
    # Required to evaluate the accuracy of the different interpolation methods;
    # without this, the measurements may be inaccurate.

    def wait_until_settled(
        self,
        arm: str,
        settle_threshold_rad: float = 0.01,
        stable_samples: int = 5,
        sample_dt: float = 0.02,
        timeout_sec: float = 1.0,
    ) -> np.ndarray:
        t0 = time.perf_counter()
        last = None
        stable = 0

        while (time.perf_counter() - t0) < float(timeout_sec):
            q = self.get_measured_joint_positions(arm)
            if last is not None:
                if float(np.max(np.abs(q - last))) < float(settle_threshold_rad):
                    stable += 1
                    if stable >= int(stable_samples):
                        return q
                else:
                    stable = 0
            last = q
            time.sleep(float(sample_dt))

        return self.get_measured_joint_positions(arm)

    # -------------------------------------------------------
    # Inverse Kinematics (IK) Computation
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

        # IKPy versions: inverse_kinematics(target_frame=...) or inverse_kinematics_frame(...)
        try:
            full = chain.inverse_kinematics(target_frame=T)
        except TypeError:
            full = chain.inverse_kinematics_frame(T)

        # IKPy returns a full array containing all links; extract the active joints.
        # Commonly: full[1:1+nj] (index 0 corresponds to the base).
        # This works if active_links_mask has been set correctly.
        joints = np.array(full[1:1 + nj], dtype=float)
        return joints

    # -------------------------------------------------------
    # Velocity Limits -> Step planning
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

    def send_arm(self, arm: str, joints: np.ndarray, time_step_sec: float):
        if arm == "left":
            joint_names = LEFT_ARM_JOINT_NAMES
            limits = self.left_limits
        else:
            joint_names = RIGHT_ARM_JOINT_NAMES
            limits = self.right_limits

        clipped, warnings = clip_joints_with_limits(joints, joint_names, limits)
        if warnings:
            self.get_logger().warn(
                f"Joint limit violation for {arm}:\n  " + "\n  ".join(warnings)
            )
        
        # Debug: Verify that joint positions are sent in radians
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

        # Count joint configurations sent from the terminal
        self.step_counter[arm] += 1

        # --- Velocity estimation ---
        current_joint_position = clipped
        previous_joint_position = self.previous_joint_position[arm]

        if previous_joint_position is not None:
            # Velocity for each joint (rad/s)
            joint_velocity_vector = (current_joint_position - previous_joint_position) / time_step_sec

            # Global max velocity
            self.max_joint_velocity[arm] = max(
                self.max_joint_velocity[arm],
                np.max(np.abs(joint_velocity_vector))
            )
        else:
            joint_velocity_vector = None

        # ---- Acceleration + Jerk estimation (actual jerk)
        previous_joint_velocity_vector = self.previous_joint_velocity_vector[arm]
        previous_joint_acceleration_vector = self.previous_joint_acceleration_vector[arm]

        joint_acceleration_vector = None

        # 1) Calculate acceleration: a = Δv / Δt
        if joint_velocity_vector is not None and previous_joint_velocity_vector is not None:
            joint_acceleration_vector = (joint_velocity_vector - previous_joint_velocity_vector) / time_step_sec

        # 2) Calculate jerk: j = Δa / Δt
        if joint_acceleration_vector is not None and previous_joint_acceleration_vector is not None:
            joint_jerk_vector = (joint_acceleration_vector - previous_joint_acceleration_vector) / time_step_sec

            # --- RMSE accumulation (squared jerk) ---
            jerk_sq = joint_jerk_vector ** 2

            # Accumulate globally across all joints
            self.sum_sq_jerk[arm] += float(np.sum(jerk_sq))
            self.jerk_sq_sample_count[arm] += int(len(joint_jerk_vector))

            # Optional: Accumulate per joint
            self.sum_sq_jerk_per_joint[arm] += jerk_sq

            # Accumulate mean jerk (per joint)
            self.sum_joint_jerk_per_joint[arm] += np.abs(joint_jerk_vector)
            self.jerk_sample_count[arm] += 1

            # Update maximum jerk for each joint
            self.max_joint_jerk_per_joint[arm] = np.maximum(
                self.max_joint_jerk_per_joint[arm],
                np.abs(joint_jerk_vector)
            )

            # Update global maximum jerk
            self.max_joint_jerk[arm] = max(
                self.max_joint_jerk[arm],
                np.max(np.abs(joint_jerk_vector))
            )

        # Store values for the next step (velocity and acceleration)
        self.previous_joint_velocity_vector[arm] = (
            joint_velocity_vector.copy()
            if joint_velocity_vector is not None
            else None
        )

        self.previous_joint_acceleration_vector[arm] = (
            joint_acceleration_vector.copy()
            if joint_acceleration_vector is not None
            else None
        )

        self.previous_joint_position[arm] = current_joint_position.copy()


    # -------------------------------------------------------
    # Move a single arm using an interpolator
    # -------------------------------------------------------
    def move_arm_to_joints(
        self,
        arm: str,
        target: np.ndarray,
        base_steps=DEFAULT_STEPS,
        delay=DEFAULT_DELAY,
        speed_scale=DEFAULT_SPEED_SCALE,
    ):
        target = np.array(target, dtype=float)

        # Select the current joint positions and joint limits
        if arm == "left":
            current = self.left_current
            joint_names = LEFT_ARM_JOINT_NAMES
            limits = self.left_limits
        else:
            current = self.right_current
            joint_names = RIGHT_ARM_JOINT_NAMES
            limits = self.right_limits

        # ---------- TARGET POSITION = effectively reachable target (joint limits) ----------
        target_effective, _warnings = clip_joints_with_limits(target, joint_names, limits)
        target_effective = np.array(target_effective, dtype=float)

        # Plan the number of steps based on velocity limits and the effective target
        steps = self.plan_steps_with_velocity_limits(
            current=current,
            target=target_effective,
            joint_names=joint_names,
            limits=limits,
            base_steps=base_steps,
            delay=delay,
            speed_scale=speed_scale,
        )
        if steps <= 0:
            return

        duration = steps * delay

        # Send the movement 
        if self.interpolator is None:
            for i in range(1, steps + 1):
                alpha = i / steps
                q_desired = current + alpha * (target_effective - current) # q_desired = desired joint position
                time.sleep(delay)
        else:
            for q_desired in self.interpolator.generate(
                start=current,
                target=target_effective,
                duration=duration,
                dt=delay,
            ):
                self.send_arm(arm, q_desired, delay)
                time.sleep(delay)

        # Update the internal commanded state (NOT the measured state)
        if arm == "left":
            self.left_current = target_effective.copy()
        else:
            self.right_current = target_effective.copy()

        # ---------- Measure END POSITION after the arm has settled ----------
        q_measured = self.wait_until_settled(
            arm=arm,
            settle_threshold_rad=0.01,  # ~0.57°
            stable_samples=5,
            sample_dt=max(0.01, float(delay)),
            timeout_sec=1.0,
        )

        # Error vector (MEASURED - TARGET)
        err_vec = q_measured - target_effective
        self.sum_squared_final_position_error += float(np.sum(err_vec**2))
        self.final_position_error_sample_count += int(err_vec.size)

        # Metrics for the current movement
        final_error_max = float(np.max(np.abs(err_vec)))
        final_error_rmse = float(np.sqrt(np.mean(err_vec**2)))

        # ---- Aggregate metrics across the entire trajectory ----
        # Maximum: Global maximum error
        self.final_position_error_max = max(
            getattr(self, "final_position_error_max", 0.0),
            final_error_max,
        )

        # RMSE: global worst-case RMSE OR global RMSE across all movements
        # -> Option A (simple and robust): worst-case RMSE across movements
        self.final_position_error_rmse = max(
            getattr(self, "final_position_error_rmse", 0.0),
            final_error_rmse,
        )

        # Logging
        self.get_logger().info(
            f"[{arm.upper()}] final_error_max={final_error_max:.6f} rad "
            f"({math.degrees(final_error_max):.3f} deg), "
            f"final_error_rmse={final_error_rmse:.6f} rad "
            f"({math.degrees(final_error_rmse):.3f} deg)"
        )

        if final_error_max > 0.02:
            self.get_logger().warn(
                f"[{arm.upper()}] Target deviation high: {final_error_max:.6f} rad"
            )

    # -------------------------------------------------------
    # Move both arms simultaneously using Interpolator
    # -------------------------------------------------------
    def move_both_arms(
        self,
        left_pose=None,
        right_pose=None,
        base_steps=DEFAULT_STEPS,
        delay=DEFAULT_DELAY,
        speed_scale=DEFAULT_SPEED_SCALE,
    ):
        l_grip = None
        r_grip = None

        if left_pose is not None:
            if "l_gripper" in left_pose:
                l_grip = left_pose["l_gripper"]
            if "joints" in left_pose and "l_gripper" in left_pose["joints"]:
                l_grip = left_pose["joints"]["l_gripper"]

        if right_pose is not None:
            if "r_gripper" in right_pose:
                r_grip = right_pose["r_gripper"]
            if "joints" in right_pose and "r_gripper" in right_pose["joints"]:
                r_grip = right_pose["joints"]["r_gripper"]

        # --- Execute gripper commands if specified ---
        if (l_grip is not None) or (r_grip is not None):
            command_grippers_sdk(
                l_deg=l_grip,
                r_deg=r_grip,
                duration=max(0.2, float(base_steps) * float(delay)),
                mode=None,  # Compatible with both Gazebo and the physical robot (command_grippers_sdk should handle mode=None)
            )

            # Optional: Skip arm movement if this pose contains only gripper commands
            def _pose_has_arm_cmd(p):
                if p is None:
                    return False
                if "joints" in p:
                    return any(k in p["joints"] for k in (LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES))
                return all(k in p for k in ("x", "y", "z"))

            if not _pose_has_arm_cmd(left_pose) and not _pose_has_arm_cmd(right_pose):
                return

        # --------------------
        # Determine Targets
        # --------------------
        left_target = self.left_current.copy()
        right_target = self.right_current.copy()

        if left_pose is not None:
            if "joints" in left_pose:
                jd = dict(left_pose["joints"])
                jd.pop("l_gripper", None)
                left_target = self.joints_dict_to_vector("left", jd, left_pose.get("use_degrees", False))
            else:
                left_target = self.compute_ik_6d(
                    "left",
                    left_pose["x"], left_pose["y"], left_pose["z"],
                    left_pose.get("roll", 0.0),
                    left_pose.get("pitch", 0.0),
                    left_pose.get("yaw", 0.0),
                )

        if right_pose is not None:
            if "joints" in right_pose:
                jd = dict(right_pose["joints"])
                jd.pop("r_gripper", None)
                right_target = self.joints_dict_to_vector("right", jd, right_pose.get("use_degrees", False))
            else:
                right_target = self.compute_ik_6d(
                    "right",
                    right_pose["x"], right_pose["y"], right_pose["z"],
                    right_pose.get("roll", 0.0),
                    right_pose.get("pitch", 0.0),
                    right_pose.get("yaw", 0.0),
                )

        # If no arm target is specified, keep the arm stationary
        left_start = self.left_current.copy()
        right_start = self.right_current.copy()
        if left_pose is None:
            left_target = left_start
        if right_pose is None:
            right_target = right_start

        # ---------- TARGET POSITIONS = EFFECTIVELY REACHABLE TARGETS (JOINT LIMITS) ----------
        left_target_effective, _ = clip_joints_with_limits(left_target, LEFT_ARM_JOINT_NAMES, self.left_limits)
        right_target_effective, _ = clip_joints_with_limits(right_target, RIGHT_ARM_JOINT_NAMES, self.right_limits)
        left_target_effective = np.array(left_target_effective, dtype=float)
        right_target_effective = np.array(right_target_effective, dtype=float)

        # Plan Steps per Arm Based on the Effective Targets
        left_steps = self.plan_steps_with_velocity_limits(
            current=self.left_current,
            target=left_target_effective,
            joint_names=LEFT_ARM_JOINT_NAMES,
            limits=self.left_limits,
            base_steps=base_steps,
            delay=delay,
            speed_scale=speed_scale,
        )
        right_steps = self.plan_steps_with_velocity_limits(
            current=self.right_current,
            target=right_target_effective,
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

        # Interpolator generators / linear interpolation
        if self.interpolator is None:
            for i in range(1, steps + 1):
                alpha = i / steps
                ql = left_start + alpha * (left_target_effective - left_start)
                qr = right_start + alpha * (right_target_effective - right_start)
                self.send_arm("left", ql, delay)
                self.send_arm("right", qr, delay)
                time.sleep(delay)
        else:
            left_gen = self.interpolator.generate(
                start=left_start,
                target=left_target_effective,
                duration=duration,
                dt=delay,
            )
            right_gen = self.interpolator.generate(
                start=right_start,
                target=right_target_effective,
                duration=duration,
                dt=delay,
            )

            for ql, qr in zip(left_gen, right_gen):
                self.send_arm("left", ql, delay)
                self.send_arm("right", qr, delay)
                time.sleep(delay)

        # Update the internal commanded state (NOT the measured state)
        self.left_current = left_target_effective.copy()
        self.right_current = right_target_effective.copy()

        # ---------- Accurately Measure the Final Position: Wait Until Settled ----------
        q_left_measured = self.wait_until_settled(
            arm="left",
            settle_threshold_rad=0.01,
            stable_samples=5,
            sample_dt=max(0.01, float(delay)),
            timeout_sec=1.0,
        )
        q_right_measured = self.wait_until_settled(
            arm="right",
            settle_threshold_rad=0.01,
            stable_samples=5,
            sample_dt=max(0.01, float(delay)),
            timeout_sec=1.0,
        )

        # Error Vectors
        err_left_vec = q_left_measured - left_target_effective
        err_right_vec = q_right_measured - right_target_effective
        self.sum_squared_final_position_error += float(np.sum(err_left_vec**2) + np.sum(err_right_vec**2))
        self.final_position_error_sample_count += int(err_left_vec.size + err_right_vec.size)

        # Metrics per Movement
        err_left_max = float(np.max(np.abs(err_left_vec)))
        err_right_max = float(np.max(np.abs(err_right_vec)))

        err_left_rmse = float(np.sqrt(np.mean(err_left_vec**2)))
        err_right_rmse = float(np.sqrt(np.mean(err_right_vec**2)))

        # ---- Aggregation Across the Entire Trajectory ----
        self.final_position_error_max = max(
            getattr(self, "final_position_error_max", 0.0),
            err_left_max,
            err_right_max,
        )

        # Worst-case RMSE across all movements (robust / simple)
        self.final_position_error_rmse = max(
            getattr(self, "final_position_error_rmse", 0.0),
            err_left_rmse,
            err_right_rmse,
        )

        self.get_logger().info(
            f"[BOTH] left_max={err_left_max:.6f} rad ({math.degrees(err_left_max):.3f} deg), "
            f"left_rmse={err_left_rmse:.6f} rad ({math.degrees(err_left_rmse):.3f} deg) | "
            f"right_max={err_right_max:.6f} rad ({math.degrees(err_right_max):.3f} deg), "
            f"right_rmse={err_right_rmse:.6f} rad ({math.degrees(err_right_rmse):.3f} deg) | "
            f"global_max={self.final_position_error_max:.6f} rad, "
            f"global_rmse={self.final_position_error_rmse:.6f} rad"
        )

    # -------------------------------------------------------
    # Trajectory execution
    # -------------------------------------------------------
    
    def follow_trajectory(self, trajectory: list[dict], return_home: bool = True):
        for idx, yaml_checkpoint in enumerate(trajectory):

            if yaml_checkpoint.get("wait_enter", False):
                self.get_logger().info("Waiting for ENTER key input...")

                while True:
                    enter = input()  # Press ENTER only
                    if enter.strip() == "":
                        break
                    self.get_logger().info("Please press ENTER.")

                # Default values (adjust to match your actual gripper positions)
                right_close_deg_default = -14
                right_open_deg_default  = 30
                left_close_deg_default  = 10
                left_open_deg_default   = -30

                # Optionally override defaults using the YAML configuration
                right_close_deg = yaml_checkpoint.get("right_gripper_close_deg", right_close_deg_default)
                right_open_deg  = yaml_checkpoint.get("right_gripper_open_deg",  right_open_deg_default)
                left_close_deg  = yaml_checkpoint.get("left_gripper_close_deg",  left_close_deg_default)
                left_open_deg   = yaml_checkpoint.get("left_gripper_open_deg",   left_open_deg_default)

                do_right_close = yaml_checkpoint.get("close_gripper_on_enter", False)
                do_right_open  = yaml_checkpoint.get("open_right_gripper_on_enter", False)
                do_left_close  = yaml_checkpoint.get("close_left_gripper_on_enter", False)
                do_left_open   = yaml_checkpoint.get("open_left_gripper_on_enter", False)

                # Detect conflicting commands
                if do_right_close and do_right_open:
                    self.get_logger().warn(
                    "Right gripper: both OPEN and CLOSE commands are set."
                    "Executing CLOSE first, followed by OPEN."
                    )
                if do_left_close and do_left_open:
                    self.get_logger().warn(
                        "Left gripper: both OPEN and CLOSE commands are set."
                        "Executing CLOSE first, followed by OPEN."
                    )

                duration = float(yaml_checkpoint.get("gripper_duration", 0.4))

                # Right Gripper
                if do_right_close:
                    self.get_logger().info(f"ENTER -> Right gripper CLOSE ({right_close_deg}°)")
                    command_grippers_sdk(r_deg=right_close_deg, duration=duration)
                if do_right_open:
                    self.get_logger().info(f"ENTER -> Right gripper OPEN ({right_open_deg}°)")
                    command_grippers_sdk(r_deg=right_open_deg, duration=duration)

                # Left Gripper
                if do_left_close:
                    self.get_logger().info(f"ENTER -> Left gripper CLOSE ({left_close_deg}°)")
                    command_grippers_sdk(l_deg=left_close_deg, duration=duration)
                if do_left_open:
                    self.get_logger().info(f"ENTER -> Left gripper OPEN ({left_open_deg}°)")
                    command_grippers_sdk(l_deg=left_open_deg, duration=duration)

                continue

            steps = yaml_checkpoint.get("steps", DEFAULT_STEPS)
            delay = yaml_checkpoint.get("delay", DEFAULT_DELAY)
            speed_scale = yaml_checkpoint.get("speed_scale", DEFAULT_SPEED_SCALE)

            left = yaml_checkpoint.get("left", None)
            right = yaml_checkpoint.get("right", None)

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

        # Returning to the home pose without affecting final_position_error (frozen_error)
        frozen_max  = getattr(self, "final_position_error_max", 0.0)
        frozen_rmse = getattr(self, "final_position_error_rmse", 0.0)

        if return_home:
            self.get_logger().info("Returning to home pose.")
            self.move_arm_to_joints("left", self.left_home, base_steps=DEFAULT_STEPS, delay=DEFAULT_DELAY)
            self.move_arm_to_joints("right", self.right_home, base_steps=DEFAULT_STEPS, delay=DEFAULT_DELAY)
        else:
            self.get_logger().info("return_home=false -> Remain in the current pose.")

        # Restore Error Metrics (Only the YAML Trajectory counts)
        self.final_position_error_max = frozen_max
        self.final_position_error_rmse = frozen_rmse


#############################################
# Convert Joint Values from YAML into Joint Vectors (Radians for IKPy)
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
                    f"Joint '{name}' missing in joints dictionary for Arm '{arm}'"
                )

            val = joints_dict[name]
            if use_degrees:
                val = math.radians(val)

            vec.append(val)

        return np.array(vec, dtype=float)
    

    # ==========================================================
    # Forward Kinematics: Joint Vector -> Cartesian Pose
    # ==========================================================
    def fk_to_cartesian(self, arm: str, joint_vec: np.ndarray):
        if arm == "left":
            chain = self.left_chain
        else:
            chain = self.right_chain

        joint_vec = np.array(joint_vec, dtype=float)

        # IKPy expects one value per link
        n_links = len(chain.links)

        # IKPy convention:
        # joints[0] = base (dummy joint)
        # followed by active and fixed joints
        full = np.zeros(n_links)

        # Insert active joint values starting at index 1
        full[1:1 + len(joint_vec)] = joint_vec

        # Compute forward kinematics
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
        print("Gazebo is now active.")
    else:
        print("The physical Reachy is now active.")

    # ROS init
    rclpy.init()

    # Choose interpolation method at the beginning
    interpolation_method = choose_interpolation_method_interactive()
    if interpolation_method is None:
        print("No interpolation method selected. Exiting.")
        return

    node = ReachyDualArmIK(interpolation_method)

    # YAML selection loop: if the YAML file is invalid, prompt the user to select another one
    while True:
        yaml_file = choose_yaml_trajectory_interactive(YAML_DIR)
        if yaml_file is None:
            node.get_logger().warn("No trajectory selected. Shutting down.")
            rclpy.shutdown()
            return

        node.get_logger().info(f"Loading YAML trajectory: {yaml_file}")
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
            validate_yaml_trajectory_allow_single_arm(data)
            break
        except Exception as e:
            node.get_logger().error(f"YAML validation failed: {e}")
            print("\nPlease select another YAML file. \n")
            continue

    trajectory = data["trajectory"]
    number_waypoints = len(trajectory)
    return_home = bool(data.get("return_home", True))

    # Synchronize the home pose (Gazebo: zero pose, physical robot reachy: SDK pose)
    node.get_logger().info("Synchronizing home pose...")
    node.move_arm_to_joints("left", node.left_home, base_steps=20, delay=DEFAULT_DELAY)
    node.move_arm_to_joints("right", node.right_home, base_steps=20, delay=DEFAULT_DELAY)

    # ---------- RESET counters (only count the YAML trajectory) ----------
    node.step_counter["left"] = 0
    node.step_counter["right"] = 0
    node.get_logger().info("Trajectory counters reset (starting to count the YAML trajectory)")

    # Start time for measurement
    t_start = time.perf_counter()

    # Reset maximum and RMSE of the final position error
    node.final_position_error_max = 0.0
    node.final_position_error_rmse = 0.0

    # Reset final position RMSE accumulator (only count the YAML trajectory)
    node.sum_squared_final_position_error = 0.0
    node.final_position_error_sample_count = 0

    ### Follow Trajectory ###
    node.get_logger().info("Starting trajectory...")
    node.follow_trajectory(trajectory, return_home=return_home)

    # Calculate global RMSE of the final position error
    final_position_error_rmse_global = math.sqrt(
        node.sum_squared_final_position_error / max(1, node.final_position_error_sample_count)
    )

    # Joint with the highest jerk (Right arm)
    jerk_vector = node.max_joint_jerk_per_joint["right"]
    worst_joint_idx = int(np.argmax(jerk_vector))
    worst_jerk_joint_name_right = RIGHT_ARM_JOINT_NAMES[worst_joint_idx]
    worst_jerk_joint_right_rad_per_sec3 = jerk_vector[worst_joint_idx]

    # Joint with the highest jerk (Left arm)
    jerk_vector_left = node.max_joint_jerk_per_joint["left"]
    worst_joint_idx_left = int(np.argmax(jerk_vector_left))
    worst_jerk_joint_name_left = LEFT_ARM_JOINT_NAMES[worst_joint_idx_left]
    worst_jerk_joint_left_rad_per_sec3 = jerk_vector_left[worst_joint_idx_left]

    # Stop time
    t_end = time.perf_counter()
    trajectory_time = t_end - t_start

    # Statistics for the executed trajectory
    node.get_logger().info(
        f"YAML file used: {yaml_file}"
    )

    node.get_logger().info(
        f"Trajectory steps LEFT : {node.step_counter['left']}"
    )
    node.get_logger().info(
        f"Trajectory steps RIGHT: {node.step_counter['right']}"
    )
    node.get_logger().info(
        f"Trajectory steps TOTAL: "
        f"{node.step_counter['left'] + node.step_counter['right']}"
    )

    # --- Jerk RMSE (global across all joints and time steps) ---
    rmse_jerk_right = math.sqrt(
        node.sum_sq_jerk["right"] / max(1, node.jerk_sq_sample_count["right"])
    )
    rmse_jerk_left = math.sqrt(
        node.sum_sq_jerk["left"] / max(1, node.jerk_sq_sample_count["left"])
    )

    # Optional: Combined RMSE across both arms
    rmse_jerk_total = math.sqrt(
        (node.sum_sq_jerk["left"] + node.sum_sq_jerk["right"]) /
        max(1, node.jerk_sq_sample_count["left"] + node.jerk_sq_sample_count["right"])
    )


    ####### --- Write results to CSV --- ###########
    csv_file = RESULTS_DIR / "trajectory_statistics.csv"
    file_exists = csv_file.is_file()

    # Step counts for the CSV file
    left_steps = node.step_counter['left']
    right_steps = node.step_counter['right']
    total_steps = left_steps + right_steps

    yaml_file = os.path.basename(yaml_file)

    interpolation_mode = node.interpolation_mode

    time_per_step = trajectory_time / max(1, total_steps)

    # Max Joint Velocity
    max_joint_velocity = max(
        node.max_joint_velocity["left"],
        node.max_joint_velocity["right"]
    )

    # Final Position Error
    final_position_error_max = node.final_position_error_max
    final_position_error_rmse = final_position_error_rmse_global

    # Max Joint Jerk
    max_joint_jerk = max(
        node.max_joint_jerk["left"],
        node.max_joint_jerk["right"]
    )

    # Mean jerk per joint (right arm)
    mean_jerk_per_joint_right = (
        node.sum_joint_jerk_per_joint["right"]
        / max(1, node.jerk_sample_count["right"])
    )

    # Mean jerk per joint (left arm)
    mean_jerk_per_joint_left = (
        node.sum_joint_jerk_per_joint["left"]
        / max(1, node.jerk_sample_count["left"])
    )

    # Critical joint based on mean jerk (right arm)
    idx_mean_r = int(np.argmax(mean_jerk_per_joint_right))
    critical_joint_mean_right = RIGHT_ARM_JOINT_NAMES[idx_mean_r]
    critical_joint_mean_value_right = mean_jerk_per_joint_right[idx_mean_r]

    # Critical joint based on mean jerk (left arm)
    idx_mean_l = int(np.argmax(mean_jerk_per_joint_left))
    critical_joint_mean_left = LEFT_ARM_JOINT_NAMES[idx_mean_l]
    critical_joint_mean_value_left = mean_jerk_per_joint_left[idx_mean_l]

    # Create a timestamp for the CSV file
    run_datetime = datetime.now().isoformat(timespec="seconds")

    # CSV-File
    fieldnames = [
        "run_date",
        "yaml_file",
        "interpolation_mode",
        "left_steps",
        "right_steps",
        "total_steps",
        "trajectory_time_sec",
        "time_per_step",
        "number_waypoints",
        "max_joint_velocity_rad_per_sec",
        "final_position_error_max_rad",
        "final_position_error_rmse_rad",
        "max_joint_jerk_rad_per_sec3",
        "worst_jerk_joint_right_arm_name",
        "worst_jerk_joint_right_arm_rad_per_sec3",
        "worst_jerk_joint_left_arm_name",
        "worst_jerk_joint_left_arm_rad_per_sec3",
        "critical_joint_mean_right",
        "critical_joint_mean_value_right_arm_rad_per_sec3",
        "critical_joint_mean_left",
        "critical_joint_mean_value_left_arm_rad_per_sec3",
        "rmse_jerk_right_arm_rad_per_sec3",
        "rmse_jerk_left_arm_rad_per_sec3",
        "rmse_jerk_total_rad_per_sec3",
    ]

    row = {
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "yaml_file": os.path.basename(yaml_file),
        "interpolation_mode": interpolation_mode,

        "left_steps": left_steps,
        "right_steps": right_steps,
        "total_steps": total_steps,

        "trajectory_time_sec": round(trajectory_time, 4),

        "time_per_step": round(time_per_step, 6),
        "number_waypoints": number_waypoints,

        "max_joint_velocity_rad_per_sec": max_joint_velocity,
        "final_position_error_max_rad": round(final_position_error_max, 6),
        "final_position_error_rmse_rad": round(final_position_error_rmse, 6),
        "max_joint_jerk_rad_per_sec3": round(max_joint_jerk, 6),

        "worst_jerk_joint_right_arm_name": worst_jerk_joint_name_right,
        "worst_jerk_joint_right_arm_rad_per_sec3": round(worst_jerk_joint_right_rad_per_sec3, 6),

        "worst_jerk_joint_left_arm_name": worst_jerk_joint_name_left,
        "worst_jerk_joint_left_arm_rad_per_sec3": round(worst_jerk_joint_left_rad_per_sec3, 6),

        "critical_joint_mean_right": critical_joint_mean_right,
        "critical_joint_mean_value_right_arm_rad_per_sec3": round(critical_joint_mean_value_right, 6),

        "critical_joint_mean_left": critical_joint_mean_left,
        "critical_joint_mean_value_left_arm_rad_per_sec3": round(critical_joint_mean_value_left, 6),

        "rmse_jerk_right_arm_rad_per_sec3": round(rmse_jerk_right, 6),
        "rmse_jerk_left_arm_rad_per_sec3": round(rmse_jerk_left, 6),
        "rmse_jerk_total_rad_per_sec3": round(rmse_jerk_total, 6),
    }

    # Open CSV file
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)  

    # Reset jerk statistics for next trajectory
    node.sum_joint_jerk_per_joint["right"].fill(0.0)
    node.sum_joint_jerk_per_joint["left"].fill(0.0)

    node.jerk_sample_count["right"] = 0
    node.jerk_sample_count["left"] = 0

    node.max_joint_jerk["right"] = 0.0
    node.max_joint_jerk["left"] = 0.0

    node.sum_sq_jerk["right"] = 0.0
    node.sum_sq_jerk["left"] = 0.0
    node.jerk_sq_sample_count["right"] = 0
    node.jerk_sq_sample_count["left"] = 0

    node.sum_sq_jerk_per_joint["right"].fill(0.0)
    node.sum_sq_jerk_per_joint["left"].fill(0.0)

    # Log the location of the generated CSV file
    node.get_logger().info(
        f"Results were written to the following CSV file: {csv_file}"
    )
    node.get_logger().info("Trajectory execution finished.")
    node.get_logger().info(f"Interpolation mode used: {node.interpolation_mode}")

    rclpy.shutdown()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Safely shut down Reachy (physical robot only)
        if not USE_GAZEBO and reachy is not None:
            try:
                reachy.turn_off_smoothly("reachy")
                print("Reachy motors turned off.")
            except Exception as e:
                print(f"Failed to turn off Reachy smoothly: {e}")
