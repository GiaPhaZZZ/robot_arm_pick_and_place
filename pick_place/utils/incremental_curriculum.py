#!/usr/bin/env python3
"""
Incremental Learning Curriculum — Chen et al. (2023) §5.2

"In order to speed up the learning process, the idea of incremental learning
is exploited in this paper to set up the learning environment."

Stage 1 (fixed_pose_episodes): Objects at fixed positions.
         Robot learns basic grasp strategy.
         → Save weights as transfer starting point.

Stage 2 (random_pose_episodes): Objects randomly placed.
         Use Stage 1 weights as initialization (transfer learning).
         Every 100 episodes: randomize object colors/sizes (sim robustness).

Paper result: transfer learning uses 6443s / 1323 attempts vs
              no transfer: 15076s / 3635 attempts.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Object workspace bounds (reachable area on table)
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE_X = (0.30, 0.55)   # meters in world frame
WORKSPACE_Y = (-0.25, 0.25)
FIXED_OBJECT_POSES = {
    'small_cube':   [0.41,  0.15, 0.515],
    'large_cube':   [0.40,  0.00, 0.525],
    'peg_cylinder': [0.41, -0.17, 0.525],
    'ellipsoid':    [0.50,  0.00, 0.515],
}
OBJECT_Z = {
    'small_cube': 0.515, 'large_cube': 0.525,
    'peg_cylinder': 0.525, 'ellipsoid': 0.515,
}


class IncrementalCurriculum:
    """
    Manages the two-stage incremental training curriculum.
    Generates object poses for each episode.
    """
    def __init__(
        self,
        fixed_episodes:  int = 1000,
        random_episodes: int = 2000,
        color_change_interval: int = 100,   # every N episodes in stage 2
        seed: int = 42,
    ):
        self.fixed_episodes    = fixed_episodes
        self.random_episodes   = random_episodes
        self.color_interval    = color_change_interval
        self.total_episodes    = fixed_episodes + random_episodes

        self.episode_count     = 0
        self.stage             = 1          # 1=fixed, 2=random
        self.rng               = np.random.default_rng(seed)
        random.seed(seed)

        self._color_variant    = 0          # changes every color_interval episodes

    # ─────────────────────────────────────────────────────────────────────────
    def step(self) -> Dict:
        """
        Called at the start of each episode.
        Returns curriculum info for this episode.
        """
        self.episode_count += 1

        # Stage transition
        if self.stage == 1 and self.episode_count > self.fixed_episodes:
            self.stage = 2
            print(f'\n[Curriculum] >>> Entering Stage 2 (random poses) at episode '
                  f'{self.episode_count} <<<\n')

        # Color/size variant update every 100 episodes in stage 2
        if self.stage == 2:
            self._color_variant = (
                (self.episode_count - self.fixed_episodes) // self.color_interval
            )

        return {
            'stage':          self.stage,
            'episode':        self.episode_count,
            'is_fixed':       self.stage == 1,
            'color_variant':  self._color_variant,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def get_object_pose(self, object_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Returns (world_pos [x,y,z], hints) for this episode.
        Stage 1: fixed canonical pose.
        Stage 2: random pose within workspace.
        """
        if self.stage == 1:
            pos = np.array(FIXED_OBJECT_POSES[object_name], dtype=np.float64)
        else:
            # Random x,y within reachable workspace
            x = self.rng.uniform(*WORKSPACE_X)
            y = self.rng.uniform(*WORKSPACE_Y)
            z = OBJECT_Z.get(object_name, 0.515)
            pos = np.array([x, y, z], dtype=np.float64)

        # Compute approximate grasp hints for this pose
        hints = self._compute_hints(object_name, pos)
        return pos, hints

    def _compute_hints(self, object_name: str, pos: np.ndarray) -> Dict:
        """
        Generate approximate grasp hints for an arbitrary object pose.
        Uses geometric heuristics based on the known hint examples.
        
        J1 (base rotation): atan2(dy, dx) from robot base
        J2/J3/J4: pre-calibrated per object type
        """
        ROBOT_BASE = np.array([0.10, 0.00, 0.50])
        dx = pos[0] - ROBOT_BASE[0]
        dy = pos[1] - ROBOT_BASE[1]
        j1_target = float(np.arctan2(dy, dx))
        j1_target = float(np.clip(j1_target, -3.14, 3.14))

        # Per-object type base joint configs
        _TYPE_CONFIGS = {
            'small_cube':   {'pick_j2': -1.65, 'pick_j3': 0.3,  'pick_j4': 0.4},
            'large_cube':   {'pick_j2': -1.5,  'pick_j3': 0.2,  'pick_j4': 0.09},
            'peg_cylinder': {'pick_j2': -1.25, 'pick_j3': 0.0,  'pick_j4': 0.7},
            'ellipsoid':    {'pick_j2': -1.75, 'pick_j3': 0.3,  'pick_j4': 0.625},
        }
        cfg = _TYPE_CONFIGS.get(object_name, _TYPE_CONFIGS['ellipsoid'])

        return {
            'HOME':      [0.0,   0.0,          0.0,         0.0,  0.0],
            'PRE_PICK':  [j1_target, 0.0,      0.0,         0.0,  0.0],
            'PICK':      [j1_target, cfg['pick_j2'], cfg['pick_j3'], cfg['pick_j4'], 0.0],
            'LIFT':      [j1_target, -0.7,      0.3,         0.4,  0.0],
            'ARC_VIA':   [1.56,  -0.7,          0.3,         0.4,  0.0],
            'PRE_PLACE': [1.56,  -0.8,          0.0,         1.0,  0.0],
        }

    # ─────────────────────────────────────────────────────────────────────────
    def get_random_object(self) -> str:
        """Randomly select which object to grasp this episode."""
        objects = list(FIXED_OBJECT_POSES.keys())
        return random.choice(objects)

    def progress(self) -> float:
        """Training progress 0→1."""
        return min(self.episode_count / self.total_episodes, 1.0)

    def should_save_stage1_checkpoint(self) -> bool:
        """True exactly when transitioning from Stage 1 → Stage 2."""
        return (self.stage == 1 and
                self.episode_count == self.fixed_episodes)

    def summary(self) -> str:
        return (f'Stage {self.stage} | Ep {self.episode_count}/{self.total_episodes} '
                f'({self.progress()*100:.1f}%) | Color variant {self._color_variant}')
