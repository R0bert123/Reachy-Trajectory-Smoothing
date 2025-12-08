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
        bounds=(0, 0.001)  # Dummy / fixiert (kleine Spanne, kein equal)
    ),
    URDFLink(
        name='shoulder_pitch',
        origin_translation=[0.0, 0.13, 0.20],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 0, 1],
        bounds=(-1.5708, 1.5708)
    ),
    URDFLink(
        name='shoulder_roll',
        origin_translation=[0.0, 0.0, 0.0],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 1, 0],
        bounds=(-0.7854, 0.7854)
    ),
    URDFLink(
        name='arm_yaw',
        origin_translation=[0.0, 0.0, 0.07],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[1, 0, 0],
        bounds=(-1.5708, 1.5708)
    ),
    URDFLink(
        name='elbow_pitch',
        origin_translation=[0.0, 0.0, 0.20],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 0, 1],
        bounds=(0, 2.0944)
    ),
    URDFLink(
        name='forearm_dummy',
        origin_translation=[0.0, 0.0, 0.17],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 0, 0],
        bounds=(0, 0.001)
    ),
    URDFLink(
        name='gripper_dummy',
        origin_translation=[0.0, 0.0, 0.05],
        origin_orientation=[0.0, 0.0, 0.0],
        rotation=[0, 0, 0],
        bounds=(0, 0.001)
    )
])

# Anzahl links (zur Sicherheit verwenden)
CHAIN_LEN = len(left_arm_chain.links)  # sollte 7 sein in deiner Definition
# Anzahl aktiver Arm-Gelenke in dieser Chain (ohne base + dummy)
NUM_ACTIVE = 4  # shoulder_pitch, shoulder_roll, arm_yaw, elbow_pitch

# ----------------------------------------------------
# ROS2 NODE
# ----------------------------------------------------
class ReachyIKNode(Node):
    def __init__(self):
        super().__init__('reachy_left_arm_ik')
        self.pub = self.create_publisher(
            Float64MultiArray,
            "/l_arm_forward_position_controller/commands",
            10
        )

        # Pre-Pose komplette Länge (falls benötigt)
        self.pre_pose = [0.0] * CHAIN_LEN

        # current_joints: ausschließlich die NUM_ACTIVE aktiven Gelenke (Python-Liste!)
        # Ellbogen (4. Wert) leicht gebeugt mit 0.5 rad
        self.current_joints = [0.0, 0.0, 0.0, 0.5]

    def compute_ik(self, x, y, z):
        """Robuste IK-Berechnung:
           - sorgt dafür, dass initial_position die korrekte Länge hat
           - konvertiert IK-Ausgabe zu Python-Listen
           - gibt die NUM_ACTIVE aktiven Gelenke als Liste zurück (nicht numpy array)
        """
        # Clip Zielpunkte auf sichere Reichweite
        x = max(0.0, min(0.35, x))
        y = max(-0.2, min(0.25, y))
        z = max(-0.25, min(0.1, z))

        # Sicherstellen, dass current_joints tatsächlich NUM_ACTIVE Werte hat
        if not isinstance(self.current_joints, list) or len(self.current_joints) != NUM_ACTIVE:
            self.get_logger().warn("current_joints inkonsistent -> setze auf sichere Startpose")
            self.current_joints = [0.0, 0.0, 0.0, 0.5]

        # Aufbau einer initial_position mit genau CHAIN_LEN Einträgen
        # Index 0 = base, Index 1..NUM_ACTIVE = aktive Gelenke, Rest = Dummy (0.0)
        initial_position = [0.0] * CHAIN_LEN
        # setze aktive Gelenke an die richtigen Positionen in initial_position
        for i in range(NUM_ACTIVE):
            initial_position[1 + i] = float(self.current_joints[i])
        # Dummy bleiben 0.0 (bereits so)

        # IK berechnen (ikpy liefert numpy array)
        try:
            joints = left_arm_chain.inverse_kinematics(
                target_position=[x, y, z],
                initial_position=initial_position,
                max_iter=200,
                regularization_parameter=0.001
            )
        except Exception as e:
            self.get_logger().error(f"IK-Fehler: {e}")
            return None  # signalisiert Fehler

        # in Python-Liste konvertieren (vermeidet list + ndarray Probleme)
        joints_list = joints.tolist()

        # Debug-Ausgabe
        self.get_logger().info(f"IK output joints: {joints_list} len: {len(joints_list)}")

        # extrahiere nur die NUM_ACTIVE aktiven Gelenke (Index 1..NUM_ACTIVE)
        target_joints = joints_list[1:1 + NUM_ACTIVE]
        # Sicherstellen, dass wir tatsächlich NUM_ACTIVE Werte haben
        if len(target_joints) != NUM_ACTIVE:
            self.get_logger().error(f"IK lieferte ungeeignete Anzahl Gelenke: {len(target_joints)}")
            return None

        self.get_logger().debug(f"target_joints: {target_joints} len: {len(target_joints)}")
        return target_joints

    def move_to_pose(self, x, y, z, steps=80, delay=0.03):
        # Berechne Zielstellung (nur aktive Gelenke)
        target_joints = self.compute_ik(x, y, z)
        if target_joints is None:
            self.get_logger().error("Keine gültige IK-Lösung — überspringe Punkt")
            return

        # Kopie der Startpose für Interpolation (wichtig: nicht self.current_joints selbst verwenden)
        start_joints = self.current_joints.copy()

        # Sicherheits-Check Längen
        if len(start_joints) != NUM_ACTIVE or len(target_joints) != NUM_ACTIVE:
            self.get_logger().error("Start/Target Gelenke haben falsche Länge — abbrechen")
            return

        # Interpolation und Publizieren
        for i in range(steps):
            a = i / (steps - 1)
            # Interpolierte aktive Gelenke
            cmd_active = [(1 - a) * float(s) + a * float(t) for s, t in zip(start_joints, target_joints)]

            # Nachricht: Base + aktive Gelenke + Dummy-Gelenke (Rest)
            # Base = joints_list[0] (meist 0.0), Dummy bleiben 0.0
            full_cmd = [0.0] + cmd_active + [0.0] * (CHAIN_LEN - 1 - NUM_ACTIVE)

            msg = Float64MultiArray()
            msg.data = full_cmd
            self.pub.publish(msg)
            time.sleep(delay)

        # nach erfolgreicher Bewegung: aktuliere current_joints
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

    # Pre-Pose: sanft zur Startpose fahren
    node.get_logger().info("Prepose…")
    node.move_to_pose(0.25, 0.15, -0.10, steps=100, delay=0.02)
    time.sleep(1)

    # Trajektorie
    node.get_logger().info("Starte Trajektorie…")
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
