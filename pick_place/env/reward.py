#!/usr/bin/env python3
"""
Reward Function — Chen et al. (2023) §4.1.3, Equation 11:

    r = +1.5  if successful AND first attempt in episode
    r = +1.0  if successful (not first attempt)
    r = -0.1  for each failed grasp attempt

Extended with dense shaping rewards for faster convergence:
  - Proximity bonus: reward for end-effector getting close to object
  - Height bonus: reward for lifting object above table
  - Drop-zone bonus: reward for releasing object near drop zone
  - Grasp quality: reward based on gripper closure on object

Termination conditions (paper §4.1.3):
  1. Successful grasp → episode ends with +reward
  2. >100 failed attempts → episode terminates (negative cumulative reward)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    # Paper Eq. 11 — core rewards
    success_reward:         float = 1.0
    first_attempt_bonus:    float = 0.5    # extra if success on attempt 1
    failure_penalty:        float = -0.1   # per failed attempt
    max_attempts:           int   = 100    # termination condition

    # Dense shaping (not in paper, added for training speed)
    use_shaping:            bool  = True
    proximity_scale:        float = 0.3    # max bonus for approach
    lift_bonus:             float = 0.2    # for raising object > 3cm
    arc_bonus:              float = 0.1    # for successful swing
    drop_zone_bonus:        float = 0.5    # for placing in zone
    drop_zone_tolerance:    float = 0.07   # meters (7cm radius)

    # Collision / unsafe motion penalties
    collision_penalty:      float = -0.5
    joint_limit_penalty:    float = -0.2


class RewardComputer:
    """
    Computes per-step and per-episode rewards for the pick-and-place task.
    Tracks attempt count and first-success within each episode.
    """
    def __init__(self, config: Optional[RewardConfig] = None):
        self.cfg = config or RewardConfig()
        self.reset()

    def reset(self):
        """Call at the start of each episode."""
        self.attempt_count         = 0
        self.episode_reward        = 0.0
        self.first_success_done    = False
        self.grasp_success_history = []

    # ─────────────────────────────────────────────────────────────────────────
    # Core sparse reward (paper §4.1.3, Eq. 11)
    # ─────────────────────────────────────────────────────────────────────────

    def grasp_result(self, success: bool) -> tuple[float, bool]:
        """
        Call after each grasp attempt.
        Returns (reward, episode_done).
        """
        self.attempt_count += 1
        self.grasp_success_history.append(success)

        if success:
            r = self.cfg.success_reward
            if not self.first_success_done:
                r += self.cfg.first_attempt_bonus   # bonus for first-attempt success
                self.first_success_done = True
            self.episode_reward += r
            return r, True   # episode terminates on success (paper §4.1.3)

        # Failed attempt
        r = self.cfg.failure_penalty
        self.episode_reward += r

        # Terminate if exceeded max attempts
        done = self.attempt_count >= self.cfg.max_attempts
        return r, done

    # ─────────────────────────────────────────────────────────────────────────
    # Dense shaping rewards (per phase)
    # ─────────────────────────────────────────────────────────────────────────

    def phase_reward(
        self,
        phase:                 str,
        ee_pos:                np.ndarray,   # end-effector world position [x,y,z]
        object_pos:            np.ndarray,   # detected object world position
        drop_zone_pos:         np.ndarray,   # drop zone world position
        gripper_state:         float,        # 0.0 = open, 0.5 = closed
        object_lifted:         bool,
        object_in_drop_zone:   bool,
        collision_detected:    bool = False,
        joint_limit_exceeded:  bool = False,
    ) -> float:
        """
        Returns a dense shaping reward for the current phase.
        These rewards are small compared to the terminal grasp reward.
        """
        if not self.cfg.use_shaping:
            return 0.0

        r = 0.0

        # ── Collision penalty ────────────────────────────────────────────────
        if collision_detected:
            r += self.cfg.collision_penalty
            return r
        if joint_limit_exceeded:
            r += self.cfg.joint_limit_penalty
            return r

        # ── Phase-specific shaping ───────────────────────────────────────────
        dist_to_obj = float(np.linalg.norm(ee_pos[:2] - object_pos[:2]))

        if phase in ('HOME', 'PRE_PICK'):
            # Reward reducing XY distance to object
            proximity_bonus = self.cfg.proximity_scale * max(0.0, 1.0 - dist_to_obj / 0.5)
            r += proximity_bonus * 0.3  # small weight, just guidance

        elif phase == 'PICK':
            # Reward being close to object both XY and Z
            dist_3d = float(np.linalg.norm(ee_pos - object_pos))
            proximity_bonus = self.cfg.proximity_scale * max(0.0, 1.0 - dist_3d / 0.3)
            r += proximity_bonus

            # Bonus for gripper starting to close (approaching object)
            if gripper_state > 0.1:
                r += 0.05

        elif phase == 'LIFT':
            # Reward for having object above table surface
            if object_lifted:
                r += self.cfg.lift_bonus
            # Penalize dropping object during lift
            if gripper_state < 0.2 and not object_lifted:
                r -= 0.1

        elif phase == 'ARC_VIA':
            # Reward for maintaining grip while swinging
            if gripper_state > 0.3 and object_lifted:
                r += self.cfg.arc_bonus

        elif phase == 'PRE_PLACE':
            # Reward for being above drop zone
            dist_to_drop = float(np.linalg.norm(
                ee_pos[:2] - drop_zone_pos[:2]))
            if dist_to_drop < self.cfg.drop_zone_tolerance * 2:
                r += self.cfg.arc_bonus

            # Big reward if object lands in drop zone
            if object_in_drop_zone:
                r += self.cfg.drop_zone_bonus

        return float(np.clip(r, -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def is_in_drop_zone(
        self,
        object_pos:    np.ndarray,
        drop_zone_pos: np.ndarray,
    ) -> bool:
        dist_xy = np.linalg.norm(object_pos[:2] - drop_zone_pos[:2])
        return bool(dist_xy < self.cfg.drop_zone_tolerance)

    def episode_summary(self) -> dict:
        return {
            'total_attempts':   self.attempt_count,
            'episode_reward':   self.episode_reward,
            'success':          any(self.grasp_success_history),
            'first_attempt_ok': self.grasp_success_history[0] if self.grasp_success_history else False,
        }
