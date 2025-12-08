import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import math
import time

# Vereinfachte IK für 3-DOF Schulter+Ellenbogen (nur Beispiel)
def simple_ik(x, y, z):
    """
    x, y, z: Zielposition der Hand relativ zum Schulter-Gelenk
    return: dict mit Gelenkwinkeln in Radiant
    """
    # Parameter des Arms (Längen)
    L1 = 0.15  # Schulter-Höhe
    L2 = 0.15  # Oberarm
    L3 = 0.15  # Unterarm

    # Planar Berechnung (vereinfachte IK)
    shoulder_pitch = math.atan2(z, x)
    shoulder_roll = math.atan2(y, math.sqrt(x**2 + z**2))
    # Ellbogenwinkel grob
    D = (x**2 + y**2 + z**2 - L2**2 - L3**2) / (2*L2*L3)
    D = max(min(D, 1), -1)  # Clamp
    elbow_pitch = math.acos(D)
    return {
        "l_shoulder_pitch": shoulder_pitch,
        "l_shoulder_roll": shoulder_roll,
        "l_elbow_pitch": -elbow_pitch  # Reachy-Konvention
    }

# ROS2 Node
class LeftArmIK(Node):
    def __init__(self):
        super().__init__('left_arm_ik')
        # Publisher für die 3 Gelenke
        self.joints = {
            "l_shoulder_pitch": self.create_publisher(Float64, '/left_arm/l_shoulder_pitch/command', 10),
            "l_shoulder_roll": self.create_publisher(Float64, '/left_arm/l_shoulder_roll/command', 10),
            "l_elbow_pitch": self.create_publisher(Float64, '/left_arm/l_elbow_pitch/command', 10),
        }

    def move_to(self, x, y, z, duration=5):
        angles = simple_ik(x, y, z)
        steps = 50
        for i in range(steps):
            factor = (i+1)/steps
            for name, pub in self.joints.items():
                msg = Float64()
                msg.data = angles[name]*factor
                pub.publish(msg)
            time.sleep(duration/steps)

def main(args=None):
    rclpy.init(args=args)
    node = LeftArmIK()

    # Zielpunkt für die Hand
    x, y, z = 0.25, 0.1, 0.2
    print(f"Moving left arm to x={x}, y={y}, z={z}")
    node.move_to(x, y, z, duration=5)

    print("Done")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
