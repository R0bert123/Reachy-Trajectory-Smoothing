#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from ikpy.chain import Chain
from ikpy.link import URDFLink
import time

# ----------------------------------------------------
# IK-KETTE FÜR REACHY 2023 – URDFLINK MIT DUMMY-GRIPPER
# ----------------------------------------------------
left_arm_chain = Chain(name='left_arm', links=[
    URDFLink(
        name='base_link',
        origin_translation=[0.0, 0.0, 0.0],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 0, 0],
        bounds=(0, 0.001) # Fast fixiert, da Dummy + "lower_bound < upper_bound"
    ),
    URDFLink(
        name='shoulder_pitch',
        origin_translation=[0.0, 0.13, 0.20],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 0, 1],
        bounds=(-1.5708, 1.5708) # -90° bis 90°
    ),
    URDFLink(
        name='shoulder_roll',
        origin_translation=[0.0, 0.0, 0.0],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 1, 0],
        bounds=(-0.7854, 0.7854) # -45° bis 45°
    ),
    URDFLink(
        name='arm_yaw',
        origin_translation=[0.0, 0.0, 0.07],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[1, 0, 0],
        bounds=(-1.5708, 1.5708) # -90° bis 90°
    ),
    URDFLink(
        name='elbow_pitch',
        origin_translation=[0.0, 0.0, 0.20],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 0, 1],
        bounds=(0, 2.0944) # 0° bis 120°
    ),
    URDFLink(
        name='forearm_dummy',
        origin_translation=[0.0, 0.0, 0.17],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 0, 0],
        bounds=(0, 0.001) # Fast fixiert, da Dummy + "lower_bound < upper_bound"
    ),
    URDFLink(
        name='gripper_dummy',
        origin_translation=[0.0, 0.0, 0.05],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 0, 0],
        bounds=(0, 0.001) # Fast fixiert, da Dummy + "lower_bound < upper_bound"
    )
])

CHAIN_LEN = len(left_arm_chain.links)
# Anzahl aktiver Arm-Gelenke in dieser Chain (ohne base + dummy)
NUM_ACTIVE = 4  # shoulder_pitch, shoulder_roll, arm_yaw, elbow_pitch

# ----------------------------------------------------
# ROS2 NODE
# ----------------------------------------------------
class ReachyIKNode(Node):
    def __init__(self):
        super().__init__('reachy_left_arm_ik')
        self.pub = self.create_publisher(Float64MultiArray,
                                         "/l_arm_forward_position_controller/commands",
                                         10)

        # Pre-Pose: Arm leicht gebeugt
        self.pre_pose = [0.0] * CHAIN_LEN  # 7 Gelenke
        self.current_joints = [0.0, 0.0, 0.0, 0.5] #Ellbogen leicht gebeugt

    def compute_ik(self, x, y, z):
        """IKPy-Inverse-Kinematik nur Position"""
        # Clip Zielpunkte
        x = max(0.0, min(0.35, x))
        y = max(-0.2, min(0.25, y))
        z = max(-0.25, min(0.1, z))

        # Sicherstellen, dass current_joints 4 Werte hat
        if len(self.current_joints) != 4:
            self.current_joints = [0.0, 0.0, 0.0, 0.5]  # Ellbogen leicht gebeugt

        # Initialposition = Base + 4 aktive Gelenke + 2 Dummy-Gelenke → genau 7 Werte
        initial_position = [0.0] + self.current_joints + [0.0, 0.0]

        joints = left_arm_chain.inverse_kinematics(
            target_position=[x, y, z],
            initial_position=initial_position,
            max_iter=200,
            regularization_parameter=0.001
        )

        target_joints = joints[1:5]  # nur die 4 aktiven Gelenke

        print("IK output joints:", joints, "len:", len(joints))
        print("target_joints:", target_joints, "len:", len(target_joints))

        self.current_joints = target_joints.copy()

        return target_joints



    def move_to_pose(self, x, y, z, steps=80, delay=0.03):
        target_joints = self.compute_ik(x, y, z)

        for i in range(steps):
            a = i / (steps - 1)
            cmd = [(1 - a) * c + a * t for c, t in zip(self.current_joints, target_joints)]            # Base + 4 echte Gelenke + Dummy-Gelenke
            msg = Float64MultiArray()
            msg.data = [0.0] + cmd + [0.0, 0.0]
            self.pub.publish(msg)
            time.sleep(delay)

        self.current_joints = target_joints.copy()


    def follow_trajectory(self, points, steps=80, delay=0.03):
        for (x, y, z) in points:
            self.get_logger().info(f"→ Trajektorienpunkt: {x:.3f}, {y:.3f}, {z:.3f}")
            self.move_to_pose(x, y, z, steps=steps, delay=delay)


# ----------------------------------------------------
# HAUPTPROGRAMM
# ----------------------------------------------------
def main():
    rclpy.init()
    node = ReachyIKNode()

    # Pre-Pose
    print("Prepose…")
    node.move_to_pose(0.25, 0.15, -0.10, steps=100, delay=0.02)
    time.sleep(1)

    # Trajektorie
    print("Starte Trajektorie…")
    trajectory = [
        (0.30, 0.20, -0.15),
        (0.32, 0.10, -0.18),
        (0.28, 0.15, -0.25),
        (0.26, 0.22, -0.12),
    ]
    node.follow_trajectory(trajectory)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
