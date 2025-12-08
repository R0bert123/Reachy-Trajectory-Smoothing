import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import math
import time

# Einfache inverse Kinematik für Reachy 2023 linken Arm
def ik_left_arm(x, y, z):
    """
    Berechnet die Gelenkwinkel für den linken Arm (vereinfacht)
    Annahmen:
    - 3 Gelenke: Schulter-Pitch, Schulter-Roll, Ellbogen-Pitch
    - Restgelenke neutral
    """
    l1 = 0.3  # Oberarm
    l2 = 0.25 # Unterarm

    r = math.sqrt(x**2 + y**2)
    z_offset = z

    # Ellbogenwinkel
    cos_elbow = (r**2 + z_offset**2 - l1**2 - l2**2)/(2*l1*l2)
    cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
    elbow = math.acos(cos_elbow)

    # Schulterwinkel
    k1 = l1 + l2*math.cos(elbow)
    k2 = l2*math.sin(elbow)
    shoulder_pitch = math.atan2(z_offset, r) - math.atan2(k2, k1)
    shoulder_roll = math.atan2(y, x)

    # Restgelenke neutral
    shoulder_yaw = 0.0
    wrist_pitch = 0.0
    wrist_roll = 0.0
    gripper = 0.0

    return [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_roll, gripper]

# Node für sanfte Bewegung entlang einer Trajektorie
class LeftArmIKTrajectory(Node):
    def __init__(self):
        super().__init__('left_arm_ik_trajectory')
        self.pub = self.create_publisher(Float64MultiArray, '/l_arm_forward_position_controller/commands', 10)
        self.current_joints = [0]*7
        self.sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

    def joint_state_callback(self, msg):
        self.current_joints = msg.position[:7]

    def move_to_pose(self, x, y, z, steps=50, delay=0.05):
        target_joints = ik_left_arm(x, y, z)
        start_joints = np.array(self.current_joints)
        target_joints = np.array(target_joints)

        for alpha in np.linspace(0, 1, steps):
            interp = (1-alpha)*start_joints + alpha*target_joints
            msg = Float64MultiArray()
            msg.data = interp.tolist()
            self.pub.publish(msg)
            time.sleep(delay)

    def follow_trajectory(self, points, steps_per_segment=50, delay=0.05):
        for x, y, z in points:
            self.move_to_pose(x, y, z, steps=steps_per_segment, delay=delay)

def main(args=None):
    rclpy.init(args=args)
    node = LeftArmIKTrajectory()

    # Warten, bis joint_states empfangen wurden
    time.sleep(1.0)

    # Zielpunkte: kleine Trajektorie
    trajectory_points = [
    (0.25, 0.1, -0.15),  # leicht vorne
    (0.28, 0.12, -0.18),
    (0.26, 0.08, -0.20),
    (0.25, 0.1, -0.15)    # zurück zum Start
]

    node.follow_trajectory(trajectory_points, steps_per_segment=50, delay=0.05)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
