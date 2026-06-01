#!/usr/bin/env python3
"""
6-Phase Motion Controller for Pick-and-Place
Phases: HOME → PRE_PICK → PICK (close grip) → LIFT → ARC_VIA → PRE_PLACE (open grip)

The SAC agent outputs DELTA joint angles per phase, which are added to
the phase hint positions. This allows the agent to learn fine corrections
while starting from a reasonable baseline.

ROS2 publishers: /arm_controller/commands, /gripper_controller/commands
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
from typing import Optional, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Phase definitions
# ─────────────────────────────────────────────────────────────────────────────
PHASES = ['HOME', 'PRE_PICK', 'PICK', 'LIFT', 'ARC_VIA', 'PRE_PLACE']

GRIPPER_OPEN  = 0.0
GRIPPER_CLOSE = 0.5

# Default dwell times per phase (seconds) — tuned to Gazebo Harmonic physics
DEFAULT_DWELL = {
    'HOME':       1.0,
    'PRE_PICK':   2.0,
    'PICK':       1.0,
    'CLOSE_GRIP': 0.5,
    'LIFT':       1.0,
    'ARC_VIA':    1.5,
    'PRE_PLACE':  2.0,
    'OPEN_GRIP':  0.5,
}

# Joint limits [min, max] radians — from objects_config.yaml
JOINT_LIMITS = np.array([
    [-3.14,  3.14],  # joint1
    [-2.0,   0.5 ],  # joint2
    [-1.5,   1.5 ],  # joint3
    [-0.5,   2.0 ],  # joint4
    [-1.57,  1.57],  # joint5
])


def clip_joints(joints: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Clip joint commands to limits. Returns (clipped, was_exceeded)."""
    clipped  = np.clip(joints, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    exceeded = not np.allclose(joints, clipped, atol=1e-4)
    return clipped, exceeded


class PhaseController(Node):
    """
    ROS2 node that executes the 6-phase pick-and-place motion.
    The SAC agent provides 5D action deltas that modify the hint positions.
    """
    def __init__(self, node_name: str = 'phase_controller'):
        super().__init__(node_name)

        self.arm_pub = self.create_publisher(
            Float64MultiArray, '/arm_controller/commands', 10)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)

        self.current_joints   = np.zeros(5)
        self.current_gripper  = GRIPPER_OPEN
        self.phase_idx        = 0
        self.last_cmd_time    = time.time()
        self._joint_limit_exceeded = False

        self.get_logger().info('[PhaseController] Ready.')

    # ─────────────────────────────────────────────────────────────────────────
    # Command publishing
    # ─────────────────────────────────────────────────────────────────────────

    def _pub_arm(self, joints: np.ndarray):
        msg      = Float64MultiArray()
        msg.data = [float(v) for v in joints]
        self.arm_pub.publish(msg)
        self.current_joints = joints.copy()

    def _pub_gripper(self, state: float):
        msg      = Float64MultiArray()
        msg.data = [float(state)]
        self.gripper_pub.publish(msg)
        self.current_gripper = state

    # ─────────────────────────────────────────────────────────────────────────
    # Execute a complete pick-and-place sequence
    # ─────────────────────────────────────────────────────────────────────────

    def execute_sequence(
        self,
        hints:   Dict[str, List[float]],
        deltas:  Dict[str, np.ndarray],        # SAC action deltas per phase
        dwell:   Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ) -> Tuple[bool, bool]:
        """
        Execute all 6 phases in order with SAC-provided deltas.
        
        Args:
            hints:  dict of base joint positions per phase
            deltas: dict of SAC action deltas ∈ [-1,1]^5 per phase
                    Scaled by action_delta_range before adding to hint.
            dwell:  optional per-phase dwell override
            verbose: log each phase

        Returns: (joint_limit_exceeded, execution_success)
        """
        if dwell is None:
            dwell = DEFAULT_DWELL

        self._joint_limit_exceeded = False
        scale = np.array([0.4, 0.5, 0.4, 0.5, 0.3])  # delta scale per joint

        def _run_phase(phase_name, gripper_state, delay_key):
            hint_pos = np.array(hints[phase_name], dtype=np.float64)
            delta    = deltas.get(phase_name, np.zeros(5))
            cmd      = hint_pos + delta * scale
            cmd, exceeded = clip_joints(cmd)
            if exceeded:
                self._joint_limit_exceeded = True
                if verbose:
                    self.get_logger().warn(
                        f'[{phase_name}] Joint limits hit! delta={delta}')

            if verbose:
                self.get_logger().info(
                    f'[{phase_name}] joints={np.round(cmd, 3).tolist()} '
                    f'gripper={gripper_state:.2f}')

            self._pub_arm(cmd)
            self._pub_gripper(gripper_state)
            time.sleep(dwell.get(delay_key, 2.5))

        # ── Phase execution ──────────────────────────────────────────────────
        _run_phase('HOME',      GRIPPER_OPEN,  'HOME')
        _run_phase('PRE_PICK',  GRIPPER_OPEN,  'PRE_PICK')
        _run_phase('PICK',      GRIPPER_OPEN,  'PICK')

        # Close gripper
        self._pub_gripper(GRIPPER_CLOSE)
        time.sleep(dwell.get('CLOSE_GRIP', 2.0))

        _run_phase('LIFT',      GRIPPER_CLOSE, 'LIFT')
        _run_phase('ARC_VIA',   GRIPPER_CLOSE, 'ARC_VIA')
        _run_phase('PRE_PLACE', GRIPPER_CLOSE, 'PRE_PLACE')

        # Open gripper (drop object)
        if verbose:
            self.get_logger().info('[PRE_PLACE] Opening gripper — dropping object')
        self._pub_gripper(GRIPPER_OPEN)
        time.sleep(dwell.get('OPEN_GRIP', 2.0))

        # Return home
        _run_phase('HOME', GRIPPER_OPEN, 'HOME')

        return self._joint_limit_exceeded, True

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: go home / reset
    # ─────────────────────────────────────────────────────────────────────────

    def go_home(self):
        home = np.zeros(5)
        self._pub_arm(home)
        self._pub_gripper(GRIPPER_OPEN)
        time.sleep(DEFAULT_DWELL['HOME'])

    def open_gripper(self):
        self._pub_gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self._pub_gripper(GRIPPER_CLOSE)

    @property
    def joint_limit_exceeded(self) -> bool:
        return self._joint_limit_exceeded
