#!/usr/bin/env python3
"""
Gazebo Pick-and-Place Gymnasium Environment
Wraps ROS2 + Gazebo Harmonic as a standard gym.Env for SAC training.

State:  64×64×1 float32 depth image (paper §4.1.1)
Action: 5D float32 joint deltas ∈ [-1,1] per phase (paper §4.1.2 adapted)
        Applied to each of the 6 phases via delta scaling.

Observation space: Box(0,1, shape=(1,64,64))
Action space:      Box(-1,1, shape=(30,))  ← 5 joints × 6 phases
                   Split into per-phase dicts inside step().

Paper §4.1.3 reward structure implemented in env/reward.py.
"""

import os
import time
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import String, Float64MultiArray
from cv_bridge import CvBridge

from env.phase_controller import PhaseController, PHASES, DEFAULT_DWELL
from env.reward import RewardComputer, RewardConfig
from utils.object_detector import VisionPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Action / observation constants
# ─────────────────────────────────────────────────────────────────────────────
N_JOINTS   = 5
N_PHASES   = 6    # HOME, PRE_PICK, PICK, LIFT, ARC_VIA, PRE_PLACE
ACTION_DIM = N_JOINTS  * N_PHASES      # 30 — flat action vector
OBS_H = OBS_W = 64                     # depth crop size (paper §4.1.1)

DROP_ZONE_POS = np.array([-0.10, 0.00, 0.501], dtype=np.float32)
ROBOT_BASE    = np.array([ 0.10,  0.00, 0.500], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
class GazeboPickPlaceEnv(gym.Env):
    """
    OpenAI Gymnasium environment for robotic pick-and-place in Gazebo.

    The agent receives a 64×64 depth image crop of the detected object,
    and outputs 30D joint delta commands (5 per phase × 6 phases).
    The environment executes the full 6-phase sequence and returns
    the reward based on grasp success/failure.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        node_name:    str  = 'gazebo_rl_env',
        object_name:  str  = 'small_cube',
        hints:        Optional[Dict] = None,
        use_vision:   bool = True,
        yolo_model:   str  = 'yolov8n.pt',
        ros_timeout:  float = 10.0,
        dwell_scale:  float = 1.0,    # speed up for training
        verbose:      bool  = False,
    ):
        super().__init__()

        self.object_name = object_name
        self.ros_timeout = ros_timeout
        self.verbose     = verbose
        self.dwell_scale = dwell_scale

        # Scale dwell times for faster training
        self._dwell = {k: v * dwell_scale for k, v in DEFAULT_DWELL.items()}

        # ── Spaces ───────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(1, OBS_H, OBS_W),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        # ── ROS2 init ────────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()

        self._ros_node = Node(node_name)
        self._bridge   = CvBridge()

        # ── Subscribers ──────────────────────────────────────────────────────
        self._latest_image    : Optional[np.ndarray] = None
        self._latest_obj_pose : Optional[np.ndarray] = None
        self._image_lock       = threading.Lock()
        self._pose_lock        = threading.Lock()

        self._ros_node.create_subscription(
            Image, '/camera/image_raw',
            self._image_cb, 10)
        self._ros_node.create_subscription(
            PoseArray, '/detected_objects/poses',
            self._poses_cb, 10)
        self._ros_node.create_subscription(
            Pose, f'/model/{object_name}/pose',
            self._object_pose_cb, 10)

        # Publisher for Gazebo object reset
        self._set_pose_pub = self._ros_node.create_publisher(
            Pose, f'/model/{object_name}/set_pose', 10)

        # Spin ROS in background thread
        self._ros_thread = threading.Thread(
            target=rclpy.spin, args=(self._ros_node,), daemon=True)
        self._ros_thread.start()

        # ── Phase controller (arm/gripper publishers) ────────────────────────
        self._ctrl = PhaseController(node_name + '_ctrl')

        # ── Vision pipeline ──────────────────────────────────────────────────
        self._vision: Optional[VisionPipeline] = None
        if use_vision:
            self._vision = VisionPipeline(
                yolo_model=yolo_model,
                yolo_conf=0.15,
                use_hybrid=True,
            )

        # ── Reward ───────────────────────────────────────────────────────────
        self._reward = RewardComputer(RewardConfig())

        # ── Hints (from curriculum / config) ────────────────────────────────
        self._hints = hints or self._default_hints(object_name)

        # ── State tracking ───────────────────────────────────────────────────
        self._current_obj_world: Optional[np.ndarray] = None
        self._last_obs = np.zeros((1, OBS_H, OBS_W), dtype=np.float32)
        self._episode_step = 0
        self._total_steps  = 0

        print(f'[Env] Initialized for object: {object_name}')

    # ─────────────────────────────────────────────────────────────────────────
    # ROS callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._image_lock:
                self._latest_image = bgr
        except Exception:
            pass

    def _poses_cb(self, msg: PoseArray):
        if msg.poses:
            p = msg.poses[0]
            with self._pose_lock:
                self._latest_obj_pose = np.array(
                    [p.position.x, p.position.y, p.position.z], dtype=np.float32)

    def _object_pose_cb(self, msg: Pose):
        with self._pose_lock:
            self._current_obj_world = np.array(
                [msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._reward.reset()
        self._episode_step = 0

        # Return arm to home
        self._ctrl.go_home()
        time.sleep(0.5)

        # Reset object to fixed position (stage 1) or random (stage 2)
        if options and 'object_pose' in options:
            self._reset_object_pose(options['object_pose'])

        # Wait for camera image
        obs = self._get_observation()
        info = {'object_name': self.object_name}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one episode: full 6-phase pick-and-place sequence.
        action: (30,) flat vector of joint deltas, split into 6 phases × 5 joints.
        """
        self._episode_step  += 1
        self._total_steps   += 1

        # Reshape action into per-phase deltas
        action_phases = action.reshape(N_PHASES, N_JOINTS)
        deltas = {phase: action_phases[i] for i, phase in enumerate(PHASES)}

        # ── Execute 6-phase sequence ─────────────────────────────────────────
        self._ctrl.go_home()
        joint_limit_hit, _ = self._ctrl.execute_sequence(
            hints=self._hints,
            deltas=deltas,
            dwell=self._dwell,
            verbose=self.verbose,
        )

        # ── Observe outcome ──────────────────────────────────────────────────
        time.sleep(0.5)  # let physics settle
        grasp_success = self._check_grasp_success()

        # Paper Eq. 11 sparse reward
        reward, done = self._reward.grasp_result(grasp_success)

        # Dense shaping bonus
        if self._current_obj_world is not None:
            ee_pos = self._estimate_ee_pos(self._hints['PRE_PLACE'])
            shaping = self._reward.phase_reward(
                phase='PRE_PLACE',
                ee_pos=ee_pos,
                object_pos=self._current_obj_world,
                drop_zone_pos=DROP_ZONE_POS,
                gripper_state=0.0,  # gripper opened
                object_lifted=grasp_success,
                object_in_drop_zone=self._reward.is_in_drop_zone(
                    self._current_obj_world, DROP_ZONE_POS),
                collision_detected=False,
                joint_limit_exceeded=joint_limit_hit,
            )
            reward += shaping * 0.3   # weight shaping below sparse

        if joint_limit_hit:
            reward += RewardConfig().joint_limit_penalty

        # ── Next observation ──────────────────────────────────────────────────
        obs   = self._get_observation()
        info  = {
            **self._reward.episode_summary(),
            'joint_limit_hit': joint_limit_hit,
            'object_world':    self._current_obj_world,
        }
        truncated = (self._reward.attempt_count >= self._reward.cfg.max_attempts)

        return obs, float(reward), done, truncated, info

    def close(self):
        try:
            self._ros_node.destroy_node()
        except Exception:
            pass

    def render(self, mode='human'):
        with self._image_lock:
            img = self._latest_image
        if img is not None and mode == 'rgb_array':
            import cv2
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Run vision pipeline on current frame, return 64×64×1 depth crop
        of the target object (paper §4.1.1 state definition).
        """
        with self._image_lock:
            bgr = self._latest_image

        if bgr is None or self._vision is None:
            return self._last_obs.copy()

        detections = self._vision.process_frame(bgr)
        if not detections:
            return self._last_obs.copy()

        # Pick detection closest to known object world position
        best = detections[0]
        if self._current_obj_world is not None:
            best = min(detections,
                       key=lambda d: np.linalg.norm(
                           d['world_pos'] - self._current_obj_world))

        obs = best['depth_crop']   # (1, 64, 64) float32
        self._last_obs = obs
        return obs

    def _check_grasp_success(self) -> bool:
        """
        Check if object was successfully placed in drop zone.
        Uses Gazebo pose topic for ground truth.
        """
        with self._pose_lock:
            pos = self._current_obj_world

        if pos is None:
            return False
        dist_xy = float(np.linalg.norm(pos[:2] - DROP_ZONE_POS[:2]))
        return dist_xy < 0.07   # 7 cm tolerance

    def _reset_object_pose(self, pose: np.ndarray):
        """Publish target pose to reset object in Gazebo."""
        msg = Pose()
        msg.position.x = float(pose[0])
        msg.position.y = float(pose[1])
        msg.position.z = float(pose[2])
        msg.orientation.w = 1.0
        self._set_pose_pub.publish(msg)
        time.sleep(0.3)

    def _estimate_ee_pos(self, pre_place_joints: List[float]) -> np.ndarray:
        """Rough FK estimate for end-effector position (for shaping)."""
        # Simplified: use drop zone as EE target after PRE_PLACE
        return DROP_ZONE_POS + np.array([0, 0, 0.1])

    def update_hints(self, new_hints: Dict):
        """Update grasp hints (called by curriculum)."""
        self._hints = new_hints

    @staticmethod
    def _default_hints(object_name: str) -> Dict:
        _HINT_DB = {
            'small_cube':   {
                'HOME':      [0.0,   0.0,   0.0,  0.0,  0.0],
                'PRE_PICK':  [0.725, 0.0,   0.0,  0.0,  0.0],
                'PICK':      [0.725,-1.65,  0.3,  0.4,  0.0],
                'LIFT':      [0.725,-0.7,   0.3,  0.4,  0.0],
                'ARC_VIA':   [0.725,-1.5,   0.0,  1.6,  0.0],
                'PRE_PLACE': [1.56, -1.5,   0.0,  1.6,  0.0],
            },
            'large_cube':   {
                'HOME':      [0.0,   0.0,   0.0,  0.0,  0.0],
                'PRE_PICK':  [-0.09, 0.0,   0.0,  0.0,  0.0],
                'PICK':      [-0.09,-1.5,   0.2,  0.09, 0.0],
                'LIFT':      [-0.09,-0.7,   0.3,  0.4,  0.0],
                'ARC_VIA':   [1.56, -0.7,   0.3,  0.4,  0.0],
                'PRE_PLACE': [1.56, -0.8,   0.0,  1.0,  0.0],
            },
            'peg_cylinder': {
                'HOME':      [0.0,   0.0,   0.0,  0.0,  0.0],
                'PRE_PICK':  [-0.85, 0.0,   0.0,  0.0,  0.0],
                'PICK':      [-0.85,-1.25,  0.0,  0.7,  0.0],
                'LIFT':      [-0.85,-0.7,   0.3,  0.4,  0.0],
                'ARC_VIA':   [1.56, -0.7,   0.3,  0.4,  0.0],
                'PRE_PLACE': [1.56, -0.8,   0.0,  1.0,  0.0],
            },
            'ellipsoid':    {
                'HOME':      [0.0,  0.0,   0.0,  0.0,  0.0],
                'PRE_PICK':  [0.0,  0.0,   0.0,  0.0,  0.0],
                'PICK':      [0.0, -1.75,  0.3,  0.625,0.0],
                'LIFT':      [0.0, -0.7,   0.3,  0.4,  0.0],
                'ARC_VIA':   [1.56,-0.7,   0.3,  0.4,  0.0],
                'PRE_PLACE': [1.56,-0.8,   0.0,  1.0,  0.0],
            },
        }
        return _HINT_DB.get(object_name, _HINT_DB['ellipsoid'])
