import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import math
import time
import numpy as np

class LeftArmController(Node):
    def __init__(self):
        super().__init__('left_arm_controller')
        self.pub = self.create_publisher(Float64MultiArray, '/l_arm_forward_position_controller/commands', 10)
        self.current_joints = [0]*7  # Initialwert
        self.sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

    def joint_state_callback(self, msg):
        # Linker Arm ist die ersten 7 Gelenke (abhängig von URDF)
        self.current_joints = msg.position[:7]

    def move_arm_smooth(self, target_joints, steps=50, delay=0.05):
        start_joints = np.array(self.current_joints)
        target_joints = np.array(target_joints)

        for alpha in np.linspace(0,1,steps):
            interp = (1-alpha)*start_joints + alpha*target_joints
            msg = Float64MultiArray()
            msg.data = interp.tolist()
            self.pub.publish(msg)
            time.sleep(delay)
        self.get_logger().info(f"Reached target joints: {target_joints.tolist()}")

def main(args=None):
    rclpy.init(args=args)
    node = LeftArmController()

    # Warte kurz, bis joint_states empfangen wurden
    time.sleep(1.0)

    # Zielposition (Radiant)
    target_joints = [
        math.radians(-30),
        math.radians(15),
        math.radians(0),
        math.radians(-45),
        math.radians(0),
        math.radians(0),
        math.radians(0)
    ]

    node.move_arm_smooth(target_joints)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
