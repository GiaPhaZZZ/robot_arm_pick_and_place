#!/usr/bin/env python3

import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# ─────────────────────────────────────────────────────────────────────────────
# World / robot constants
# ─────────────────────────────────────────────────────────────────────────────
SMALL_CUBE_POS   = (0.41,  0.15, 0.515)
DROP_ZONE_POS    = (-0.10, 0.00, 0.501)
CAMERA_STAND_POS = (0.15, -0.10, 0.44)
ROBOT_BASE       = (0.10,  0.00, 0.50)

GRIPPER_OPEN  = 0.0    # fingers open
GRIPPER_CLOSE = 0.5    # firm grip on 3 cm cube (slightly below hard limit)

# Arm straight up – safe neutral posture
HOME = [0.0, 0.0, 0.0, 0.0, 0.0]

PRE_PICK = [0.0,  0.0, 0.0, 0.0, 0.0]

# Descend onto cube grasp height
PICK = [0.0,  -1.75, 0.3, 0.625, 0.0]

LIFT = [0.0,  -0.7, 0.3, 0.4, 0.0]

ARC_VIA = [1.56, -0.7, 0.3, 0.4, 0.0]

PRE_PLACE = [1.56, -0.8, 0.0, 1.0, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZED SEQUENCE (Synchronized with Gazebo Physics)
# ─────────────────────────────────────────────────────────────────────────────
SEQUENCE = [
    (HOME,      GRIPPER_OPEN,  2.5, "Moving to HOME"),
    (PRE_PICK,  GRIPPER_OPEN,  3.5, "Hovering above small_cube  [EE → (0.41, 0.15, ~0.590)]"),
    (PICK,      GRIPPER_OPEN,  2.5, "Descending to grasp height  [EE → (0.41, 0.15, ~0.530)]"),
    (PICK,      GRIPPER_CLOSE, 2.0, "Closing gripper – securing cube"),
    (LIFT,      GRIPPER_CLOSE, 2.5, "Lifting cube clear of table"),
    (ARC_VIA,   GRIPPER_CLOSE, 3.0, "ARC VIA +Y  [J1=+1.55 rad, safe – away from camera_stand]"),
    
    # Step 7: Command the arm swing to the destination FIRST while maintaining a tight grip
    (PRE_PLACE, GRIPPER_CLOSE, 3.5, "Moving arm to drop zone (Holding cube firmly during transit)"),
    
    # Step 8: Only drop the object after the robot hand has completely finished turning in Gazebo
    (PRE_PLACE, GRIPPER_OPEN,  2.0, "Arrived at drop zone – Opening gripper and dropping object"),
    
    (HOME,      GRIPPER_OPEN,  3.0, "Returning to HOME"),
]


# ─────────────────────────────────────────────────────────────────────────────
class TopicPickAndPlace(Node):
    def __init__(self):
        super().__init__("topic_pick_and_place_node")

        self.arm_pub = self.create_publisher(
            Float64MultiArray,
            "/arm_controller/commands",
            10,
        )
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            "/gripper_controller/commands",
            10,
        )

        self.get_logger().info("=== Topic-based Pick-and-Place Node started ===")
        self.get_logger().info(
            f"Target: small_cube {SMALL_CUBE_POS}  →  drop_zone {DROP_ZONE_POS}"
        )
        self.get_logger().info(
            f"Camera stand at {CAMERA_STAND_POS} – arm sweeps through +Y (J1: ~26°→89°)"
        )
        self.get_logger().info(
            f"Gripper: joint6 only  [0.0=open, {GRIPPER_CLOSE}=closed]  joint7 mimics automatically"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def run_sequence(self):
        time.sleep(1.5)  # wait for subscriber connections

        total = len(SEQUENCE)
        for step, (arm_pos, gripper_pos, dwell, label) in enumerate(SEQUENCE, 1):
            self.get_logger().info(f"[{step}/{total}] {label}")
            self.get_logger().info(
                f"         arm={[round(v,3) for v in arm_pos]}  gripper={gripper_pos}"
            )

            arm_msg = Float64MultiArray()
            arm_msg.data = [float(v) for v in arm_pos]
            self.arm_pub.publish(arm_msg)

            gripper_msg = Float64MultiArray()
            gripper_msg.data = [float(gripper_pos)]
            self.gripper_pub.publish(gripper_msg)

            time.sleep(dwell)

            if step < total:
                self.get_logger().info("Phase complete. Pausing…")
                time.sleep(2.0)

        self.get_logger().info("✓ Pick-and-place sequence complete!")


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = TopicPickAndPlace()
    try:
        node.run_sequence()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()