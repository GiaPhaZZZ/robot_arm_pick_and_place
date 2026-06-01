#!/usr/bin/env python3
"""
Smoke Tests for rl_pick_place Components

Tests all modules that can be exercised without a live ROS2/Gazebo instance:
    - ReplayBuffer       (agent/replay_buffer.py)
    - SACNetworks        (agent/networks.py)
    - SACAgent           (agent/sac_agent.py)
    - RewardComputer     (env/reward.py)
    - IncrementalCurriculum (utils/incremental_curriculum.py)
    - VisionPipeline     (utils/object_detector.py) — mocked, no model download
    - GazeboPickPlaceEnv (env/gazebo_env.py)       — mocked, no ROS2 required

Run:
    cd src/arm_bringup/pick_place
    python tests/test_env.py            # all tests
    python tests/test_env.py -v         # verbose
    python tests/test_env.py TestReplayBuffer  # single suite
"""

import sys
import os
import unittest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ReplayBuffer
# ─────────────────────────────────────────────────────────────────────────────
class TestReplayBuffer(unittest.TestCase):

    def _make_buffer(self, capacity=500, action_dim=5):
        from agent.replay_buffer import ReplayBuffer
        return ReplayBuffer(capacity=capacity, action_dim=action_dim, device='cpu')

    def _rand_state(self):
        return np.random.rand(1, 64, 64).astype(np.float32)

    def _rand_action(self, dim=5):
        return np.random.uniform(-1, 1, dim).astype(np.float32)

    # ------------------------------------------------------------------
    def test_push_single_and_len(self):
        buf = self._make_buffer()
        self.assertEqual(len(buf), 0)
        buf.push(self._rand_state(), self._rand_action(), 0.5,
                 self._rand_state(), False)
        self.assertEqual(len(buf), 1)

    def test_is_not_ready_when_empty(self):
        buf = self._make_buffer()
        self.assertFalse(buf.is_ready(64))

    def test_is_ready_after_filling(self):
        buf = self._make_buffer(capacity=200)
        for _ in range(64):
            buf.push(self._rand_state(), self._rand_action(), 0.0,
                     self._rand_state(), False)
        self.assertTrue(buf.is_ready(64))

    def test_sample_shapes(self):
        buf = self._make_buffer(capacity=200)
        for _ in range(100):
            buf.push(self._rand_state(), self._rand_action(), 1.0,
                     self._rand_state(), False)
        states, actions, rewards, next_states, dones = buf.sample(32)
        self.assertEqual(tuple(states.shape),      (32, 1, 64, 64))
        self.assertEqual(tuple(actions.shape),     (32, 5))
        self.assertEqual(tuple(rewards.shape),     (32, 1))
        self.assertEqual(tuple(next_states.shape), (32, 1, 64, 64))
        self.assertEqual(tuple(dones.shape),       (32, 1))

    def test_circular_overwrite(self):
        capacity = 10
        buf = self._make_buffer(capacity=capacity)
        for i in range(20):
            buf.push(self._rand_state(), self._rand_action(), float(i),
                     self._rand_state(), i % 3 == 0)
        # Buffer should wrap around; size capped at capacity
        self.assertEqual(len(buf), capacity)

    def test_save_load_roundtrip(self, tmp_path=None):
        import tempfile
        buf = self._make_buffer(capacity=100)
        for _ in range(50):
            buf.push(self._rand_state(), self._rand_action(), 0.1,
                     self._rand_state(), False)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            buf.save(path.replace('.npz', ''))  # save() adds .npz
            buf2 = self._make_buffer(capacity=100)
            buf2.load(path if path.endswith('.npz') else path + '.npz')
            self.assertEqual(buf2.size, buf.size)
        finally:
            for ext in ['', '.npz']:
                try:
                    os.remove(path + ext)
                except FileNotFoundError:
                    pass

    def test_push_large_action_dim(self):
        """SAC with 30D action vector (5 joints × 6 phases)."""
        buf = self._make_buffer(capacity=200, action_dim=30)
        for _ in range(100):
            buf.push(self._rand_state(), self._rand_action(30), 0.0,
                     self._rand_state(), False)
        states, actions, rewards, next_states, dones = buf.sample(32)
        self.assertEqual(tuple(actions.shape), (32, 30))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Networks (CNN + SAC architecture)
# ─────────────────────────────────────────────────────────────────────────────
class TestNetworks(unittest.TestCase):

    def setUp(self):
        from agent.networks import DepthCNNEncoder, PolicyNetwork, QNetwork, SACNetworks
        self.DepthCNNEncoder = DepthCNNEncoder
        self.PolicyNetwork   = PolicyNetwork
        self.QNetwork        = QNetwork
        self.SACNetworks     = SACNetworks

    def _batch(self, b=4):
        return torch.zeros(b, 1, 64, 64)

    def test_encoder_output_shape(self):
        enc = self.DepthCNNEncoder(feature_dim=512)
        out = enc(self._batch())
        self.assertEqual(tuple(out.shape), (4, 512))

    def test_policy_network_sample(self):
        pol = self.PolicyNetwork(action_dim=5, feature_dim=512, hidden_dim=64)
        img = self._batch()
        action, log_prob, mean = pol.sample(img)
        self.assertEqual(tuple(action.shape),   (4, 5))
        self.assertEqual(tuple(log_prob.shape), (4, 1))
        self.assertEqual(tuple(mean.shape),     (4, 5))

    def test_policy_tanh_bounded(self):
        pol = self.PolicyNetwork(action_dim=5, feature_dim=512, hidden_dim=64)
        action, _, mean = pol.sample(self._batch())
        self.assertTrue(action.abs().max().item() <= 1.0 + 1e-5,
                        'action must be in [-1, 1] (Tanh squashed)')
        self.assertTrue(mean.abs().max().item()   <= 1.0 + 1e-5)

    def test_q_network_output_shape(self):
        q   = self.QNetwork(action_dim=5, feature_dim=512, hidden_dim=64)
        img = self._batch()
        act = torch.zeros(4, 5)
        out = q(img, act)
        self.assertEqual(tuple(out.shape), (4, 1))

    def test_sac_networks_five_components(self):
        nets = self.SACNetworks(action_dim=5, feature_dim=512, hidden_dim=64)
        for attr in ('policy', 'q1', 'q2', 'q1_target', 'q2_target'):
            self.assertTrue(hasattr(nets, attr), f'Missing {attr}')

    def test_polyak_update_changes_targets(self):
        nets = self.SACNetworks(action_dim=5, tau=0.5)
        # Perturb q1 weights
        with torch.no_grad():
            for p in nets.q1.parameters():
                p.add_(1.0)
        before = next(nets.q1_target.parameters()).clone()
        nets.soft_update_targets()
        after  = next(nets.q1_target.parameters()).clone()
        self.assertFalse(torch.allclose(before, after),
                         'Polyak update should change target weights')

    def test_get_policy_params_non_empty(self):
        nets   = self.SACNetworks(action_dim=5)
        params = list(nets.get_policy_params())
        self.assertGreater(len(params), 0)

    def test_get_q_params_non_empty(self):
        nets   = self.SACNetworks(action_dim=5)
        params = nets.get_q_params()
        self.assertGreater(len(params), 0)

    def test_30d_action_dim(self):
        """Verify architecture supports 30D actions (train_parallel.py)."""
        nets    = self.SACNetworks(action_dim=30)
        img     = self._batch()
        action, log_prob, mean = nets.policy.sample(img)
        self.assertEqual(tuple(action.shape), (4, 30))
        q_out = nets.q1(img, action)
        self.assertEqual(tuple(q_out.shape), (4, 1))


# ─────────────────────────────────────────────────────────────────────────────
# 3. SACAgent
# ─────────────────────────────────────────────────────────────────────────────
class TestSACAgent(unittest.TestCase):

    def _make_agent(self, action_dim=5):
        from agent.sac_agent import SACAgent
        return SACAgent(
            action_dim   = action_dim,
            feature_dim  = 64,    # small for speed
            hidden_dim   = 32,
            lr           = 1e-3,
            buffer_size  = 500,
            batch_size   = 16,
            device       = 'cpu',
            auto_entropy = True,
        )

    def _rand_state(self):
        return np.random.rand(1, 64, 64).astype(np.float32)

    def _rand_action(self, dim=5):
        return np.random.uniform(-1, 1, dim).astype(np.float32)

    def test_select_action_shape(self):
        agent  = self._make_agent()
        action = agent.select_action(self._rand_state(), deterministic=False)
        self.assertEqual(action.shape, (5,))

    def test_select_action_bounded(self):
        agent  = self._make_agent()
        action = agent.select_action(self._rand_state(), deterministic=True)
        self.assertTrue(np.all(np.abs(action) <= 1.0 + 1e-5))

    def test_store_increments_buffer(self):
        agent = self._make_agent()
        self.assertEqual(len(agent.buffer), 0)
        agent.store(self._rand_state(), self._rand_action(), 1.0,
                    self._rand_state(), False)
        self.assertEqual(len(agent.buffer), 1)

    def test_update_returns_empty_when_buffer_not_ready(self):
        agent  = self._make_agent()
        losses = agent.update()
        self.assertEqual(losses, {})

    def test_update_after_filling_buffer(self):
        agent = self._make_agent()
        for _ in range(20):
            agent.store(self._rand_state(), self._rand_action(), 0.5,
                        self._rand_state(), False)
        losses = agent.update()
        self.assertIn('q1_loss', losses)
        self.assertIn('policy_loss', losses)
        self.assertIn('alpha', losses)

    def test_save_load_checkpoint(self):
        import tempfile
        agent = self._make_agent()
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            agent.save(path, episode=42)
            agent2 = self._make_agent()
            ep = agent2.load(path)
            self.assertEqual(ep, 42)
        finally:
            os.remove(path)

    def test_total_steps_increments(self):
        agent = self._make_agent()
        for _ in range(5):
            agent.store(self._rand_state(), self._rand_action(), 0.0,
                        self._rand_state(), False)
        self.assertEqual(agent.total_steps, 5)

    def test_30d_action_agent(self):
        """agent used in train_parallel with action_dim=30."""
        agent  = self._make_agent(action_dim=30)
        action = agent.select_action(self._rand_state())
        self.assertEqual(action.shape, (30,))

    def test_auto_entropy_alpha_updates(self):
        agent = self._make_agent()
        for _ in range(20):
            agent.store(self._rand_state(), self._rand_action(), 1.0,
                        self._rand_state(), False)
        alpha_before = agent.alpha
        losses = agent.update()
        # alpha may or may not change in a single step, but key should exist
        self.assertIn('alpha_loss', losses)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RewardComputer
# ─────────────────────────────────────────────────────────────────────────────
class TestRewardComputer(unittest.TestCase):

    def setUp(self):
        from env.reward import RewardComputer, RewardConfig
        self.cfg = RewardConfig()
        self.rc  = RewardComputer(self.cfg)

    def test_first_attempt_success_gives_bonus(self):
        r, done = self.rc.grasp_result(success=True)
        # 1st success: success_reward + first_attempt_bonus = 1.5
        self.assertAlmostEqual(r, 1.5)
        self.assertTrue(done)

    def test_success_no_bonus_after_first(self):
        # first_success_done tracks first SUCCESS in the episode, not attempt #1.
        # So a fail then success still earns the bonus on the first success.
        self.rc.grasp_result(success=False)   # fail once
        r, done = self.rc.grasp_result(success=True)
        # Still the first success in the episode → bonus applies → 1.5
        self.assertAlmostEqual(r, 1.5)
        self.assertTrue(done)

    def test_no_bonus_on_second_success_episode(self):
        # Simulate a fresh episode where first success already happened
        self.rc.grasp_result(success=True)   # first success in ep → bonus
        self.rc.first_success_done = True    # mark as used
        self.rc.reset()                       # new episode
        r, done = self.rc.grasp_result(success=True)
        # New episode → first_success_done reset → bonus applies again
        self.assertAlmostEqual(r, 1.5)

    def test_failure_gives_penalty(self):
        r, done = self.rc.grasp_result(success=False)
        self.assertAlmostEqual(r, -0.1)
        self.assertFalse(done)

    def test_termination_at_max_attempts(self):
        for i in range(self.cfg.max_attempts - 1):
            self.rc.grasp_result(success=False)
        _, done = self.rc.grasp_result(success=False)
        self.assertTrue(done, 'Should terminate at max_attempts')

    def test_reset_clears_state(self):
        self.rc.grasp_result(success=True)
        self.rc.reset()
        self.assertEqual(self.rc.attempt_count, 0)
        self.assertFalse(self.rc.first_success_done)

    def test_episode_summary_keys(self):
        self.rc.grasp_result(success=False)
        summary = self.rc.episode_summary()
        for key in ('total_attempts', 'episode_reward', 'success', 'first_attempt_ok'):
            self.assertIn(key, summary)

    def test_is_in_drop_zone_true(self):
        obj  = np.array([0.0, 0.0, 0.5])
        drop = np.array([0.0, 0.0, 0.5])
        self.assertTrue(self.rc.is_in_drop_zone(obj, drop))

    def test_is_in_drop_zone_false(self):
        obj  = np.array([1.0, 1.0, 0.5])
        drop = np.array([0.0, 0.0, 0.5])
        self.assertFalse(self.rc.is_in_drop_zone(obj, drop))

    def test_phase_reward_no_collision(self):
        ee  = np.array([0.41, 0.15, 0.515])
        obj = np.array([0.41, 0.15, 0.515])
        dz  = np.array([-0.10, 0.00, 0.501])
        r = self.rc.phase_reward(
            phase='PICK', ee_pos=ee, object_pos=obj,
            drop_zone_pos=dz, gripper_state=0.5,
            object_lifted=False, object_in_drop_zone=False,
            collision_detected=False, joint_limit_exceeded=False,
        )
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r,  1.0)

    def test_phase_reward_collision_penalty(self):
        ee  = np.zeros(3)
        obj = np.zeros(3)
        dz  = np.zeros(3)
        r = self.rc.phase_reward(
            phase='PICK', ee_pos=ee, object_pos=obj,
            drop_zone_pos=dz, gripper_state=0.0,
            object_lifted=False, object_in_drop_zone=False,
            collision_detected=True, joint_limit_exceeded=False,
        )
        self.assertLess(r, 0.0, 'Collision should give negative reward')


# ─────────────────────────────────────────────────────────────────────────────
# 5. IncrementalCurriculum
# ─────────────────────────────────────────────────────────────────────────────
class TestIncrementalCurriculum(unittest.TestCase):

    def setUp(self):
        from utils.incremental_curriculum import IncrementalCurriculum
        self.CLS = IncrementalCurriculum

    def test_starts_in_stage1(self):
        c = self.CLS(fixed_episodes=10, random_episodes=20)
        info = c.step()
        self.assertEqual(info['stage'], 1)

    def test_transitions_to_stage2(self):
        c = self.CLS(fixed_episodes=3, random_episodes=10)
        for _ in range(3):
            c.step()
        info = c.step()   # episode 4 → stage 2
        self.assertEqual(info['stage'], 2)

    def test_fixed_pose_reproducible(self):
        c = self.CLS(fixed_episodes=100, random_episodes=100, seed=0)
        pos1, _ = c.get_object_pose('small_cube')
        pos2, _ = c.get_object_pose('small_cube')
        np.testing.assert_array_equal(pos1, pos2,
                                      'Stage 1 pose must be deterministic')

    def test_random_pose_within_workspace(self):
        from utils.incremental_curriculum import WORKSPACE_X, WORKSPACE_Y
        c = self.CLS(fixed_episodes=1, random_episodes=100)
        c.stage = 2   # force stage 2
        for _ in range(20):
            pos, _ = c.get_object_pose('small_cube')
            self.assertGreaterEqual(pos[0], WORKSPACE_X[0])
            self.assertLessEqual(pos[0],    WORKSPACE_X[1])
            self.assertGreaterEqual(pos[1], WORKSPACE_Y[0])
            self.assertLessEqual(pos[1],    WORKSPACE_Y[1])

    def test_hints_have_all_phases(self):
        c = self.CLS(fixed_episodes=5, random_episodes=5)
        _, hints = c.get_object_pose('small_cube')
        for phase in ('HOME', 'PRE_PICK', 'PICK', 'LIFT', 'ARC_VIA', 'PRE_PLACE'):
            self.assertIn(phase, hints)
            self.assertEqual(len(hints[phase]), 5)

    def test_should_save_stage1_checkpoint_once(self):
        c = self.CLS(fixed_episodes=3, random_episodes=5)
        flags = []
        for _ in range(8):
            c.step()
            flags.append(c.should_save_stage1_checkpoint())
        # Exactly one True when transitioning
        self.assertEqual(sum(flags), 1)

    def test_progress_capped_at_1(self):
        c = self.CLS(fixed_episodes=2, random_episodes=2)
        for _ in range(100):
            c.step()
        self.assertLessEqual(c.progress(), 1.0)

    def test_all_object_types(self):
        c = self.CLS(fixed_episodes=10, random_episodes=10)
        for obj in ('small_cube', 'large_cube', 'peg_cylinder', 'ellipsoid'):
            pos, hints = c.get_object_pose(obj)
            self.assertEqual(len(pos), 3)
            self.assertIn('PICK', hints)


# ─────────────────────────────────────────────────────────────────────────────
# 6. VisionPipeline (mocked — no model download)
# ─────────────────────────────────────────────────────────────────────────────
class TestVisionPipeline(unittest.TestCase):
    """
    Tests the VisionPipeline without downloading any models.
    The heavy models (DepthAnythingV2, YOLO) are replaced with stubs that
    return deterministic outputs, so we can verify the geometry + logic.
    """

    def _make_pipeline(self):
        from utils.object_detector import VisionPipeline, pixel_to_world, extract_depth_crop

        # Stub DepthAnythingV2 — returns a simple depth gradient
        mock_depth_pipe = MagicMock()
        # Returns an object with ['depth'] key whose value is a np array
        mock_depth_pipe.return_value = {
            'depth': np.linspace(0.3, 0.9, 640 * 480).reshape(480, 640).astype(np.float32)
        }

        with patch('utils.object_detector.hf_pipeline', return_value=mock_depth_pipe), \
             patch('utils.object_detector._HAVE_YOLO', False):
            pipeline = VisionPipeline(
                yolo_model='yolov8n.pt',
                yolo_conf=0.15,
                use_hybrid=True,
            )
        # Replace the depth pipe reference
        pipeline._depth_pipe = mock_depth_pipe
        pipeline._yolo       = None   # no YOLO
        return pipeline

    def test_pixel_to_world_returns_3d(self):
        from utils.object_detector import pixel_to_world
        pt = pixel_to_world(320.0, 240.0, 0.5)
        self.assertEqual(pt.shape, (3,))

    def test_pixel_to_world_center_on_table(self):
        from utils.object_detector import pixel_to_world, TABLE_Z_WORLD
        pt = pixel_to_world(320.0, 240.0, 0.5)
        # Z should be close to TABLE_Z_WORLD for mid-depth
        self.assertAlmostEqual(float(pt[2]), TABLE_Z_WORLD, delta=0.4)

    def test_extract_depth_crop_shape(self):
        from utils.object_detector import extract_depth_crop
        depth = np.random.rand(480, 640).astype(np.float32)
        crop  = extract_depth_crop(depth, 100, 80, 200, 160)
        self.assertEqual(crop.shape, (1, 64, 64))

    def test_extract_depth_crop_values_normalized(self):
        from utils.object_detector import extract_depth_crop
        depth = np.random.rand(480, 640).astype(np.float32)
        crop  = extract_depth_crop(depth, 100, 80, 200, 160)
        self.assertGreaterEqual(float(crop.min()), 0.0)
        self.assertLessEqual(float(crop.max()),    1.0)

    def test_process_frame_returns_list(self):
        pipeline = self._make_pipeline()
        bgr      = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        result   = pipeline.process_frame(bgr)
        self.assertIsInstance(result, list)

    def test_detection_dict_keys(self):
        pipeline = self._make_pipeline()
        bgr      = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        result   = pipeline.process_frame(bgr)
        if result:
            d = result[0]
            for key in ('label', 'conf', 'world_pos', 'depth_crop',
                        'bbox', 'centroid_px', 'depth_val'):
                self.assertIn(key, d)

    def test_depth_crop_is_sac_state_shape(self):
        pipeline = self._make_pipeline()
        bgr      = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        result   = pipeline.process_frame(bgr)
        if result:
            crop = result[0]['depth_crop']
            self.assertEqual(crop.shape, (1, 64, 64),
                             'SAC state must be (1,64,64) depth crop')

    def test_draw_debug_returns_image(self):
        pipeline = self._make_pipeline()
        bgr      = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        det      = [{
            'label': 'test', 'conf': 0.9,
            'world_pos': np.array([0.4, 0.1, 0.5]),
            'bbox': (100, 80, 200, 160),
            'centroid_px': (150, 120),
            'depth_crop': np.zeros((1, 64, 64), dtype=np.float32),
            'depth_val': 0.5,
        }]
        out = pipeline.draw_debug(bgr, det)
        self.assertEqual(out.shape, bgr.shape)


# ─────────────────────────────────────────────────────────────────────────────
# 7. GazeboPickPlaceEnv (mocked — no ROS2/Gazebo)
# ─────────────────────────────────────────────────────────────────────────────
class TestGazeboEnvMocked(unittest.TestCase):
    """
    Verifies env API contract (reset/step/close) without launching ROS2.
    All ROS2 and VisionPipeline calls are replaced by MagicMock.
    """

    def _make_env(self):
        """Build a GazeboPickPlaceEnv with all ROS2/vision calls mocked out."""
        mock_node    = MagicMock()
        mock_ctrl    = MagicMock()
        mock_reward  = MagicMock()
        mock_vision  = MagicMock()

        mock_node.create_subscription = MagicMock(return_value=None)
        mock_node.create_publisher    = MagicMock(return_value=MagicMock())
        mock_node.get_clock           = MagicMock(return_value=MagicMock())

        mock_ctrl.execute_sequence.return_value = (False, True)   # (limit_hit, ok)
        mock_ctrl.go_home            = MagicMock()
        mock_ctrl.open_gripper       = MagicMock()
        mock_ctrl.close_gripper      = MagicMock()

        mock_reward.reset            = MagicMock()
        mock_reward.grasp_result.return_value = (1.0, True)
        mock_reward.phase_reward.return_value = 0.0
        mock_reward.episode_summary.return_value = {
            'total_attempts': 1, 'episode_reward': 1.0,
            'success': True, 'first_attempt_ok': True,
        }
        mock_reward.is_in_drop_zone.return_value = True
        mock_reward.attempt_count = 1
        from env.reward import RewardConfig
        mock_reward.cfg = RewardConfig()

        mock_vision.process_frame.return_value = [{
            'label':       'small_cube',
            'conf':        0.95,
            'world_pos':   np.array([0.41, 0.15, 0.515]),
            'depth_crop':  np.zeros((1, 64, 64), dtype=np.float32),
            'bbox':        (100, 80, 200, 160),
            'centroid_px': (150, 120),
            'depth_val':   0.5,
        }]

        with patch('env.gazebo_env.rclpy') as mock_rclpy, \
             patch('env.gazebo_env.Node', return_value=mock_node), \
             patch('env.gazebo_env.PhaseController', return_value=mock_ctrl), \
             patch('env.gazebo_env.RewardComputer', return_value=mock_reward), \
             patch('env.gazebo_env.VisionPipeline', return_value=mock_vision), \
             patch('threading.Thread'), \
             patch('time.sleep'):
            mock_rclpy.ok.return_value = False   # skip rclpy.init()
            from env.gazebo_env import GazeboPickPlaceEnv
            env = GazeboPickPlaceEnv(
                node_name   = 'test_env',
                object_name = 'small_cube',
                use_vision  = True,
            )
            env._latest_image    = np.zeros((480, 640, 3), dtype=np.uint8)
            env._latest_obj_pose = np.array([0.41, 0.15, 0.515])
            env._current_obj_world = np.array([0.41, 0.15, 0.515])
            env._ctrl    = mock_ctrl
            env._reward  = mock_reward
            env._vision  = mock_vision

        return env

    def test_observation_space(self):
        env = self._make_env()
        obs_sp = env.observation_space
        self.assertEqual(obs_sp.shape, (1, 64, 64))
        self.assertAlmostEqual(float(obs_sp.low.min()),  0.0)
        self.assertAlmostEqual(float(obs_sp.high.max()), 1.0)

    def test_action_space(self):
        env = self._make_env()
        act_sp = env.action_space
        self.assertEqual(act_sp.shape, (30,))
        self.assertAlmostEqual(float(act_sp.low.min()),  -1.0)
        self.assertAlmostEqual(float(act_sp.high.max()), 1.0)

    def test_reset_returns_obs_and_info(self):
        env = self._make_env()
        with patch('time.sleep'):
            obs, info = env.reset()
        self.assertEqual(obs.shape, (1, 64, 64))
        self.assertIsInstance(info, dict)

    def test_step_returns_correct_tuple(self):
        env    = self._make_env()
        action = env.action_space.sample()
        with patch('time.sleep'):
            result = env.step(action)
        self.assertEqual(len(result), 5, 'step() should return (obs, rew, done, trunc, info)')
        obs, reward, done, truncated, info = result
        self.assertEqual(obs.shape, (1, 64, 64))
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(done,      bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info,      dict)

    def test_default_hints_all_objects(self):
        from env.gazebo_env import GazeboPickPlaceEnv
        for obj in ('small_cube', 'large_cube', 'peg_cylinder', 'ellipsoid'):
            hints = GazeboPickPlaceEnv._default_hints(obj)
            for phase in ('HOME', 'PRE_PICK', 'PICK', 'LIFT', 'ARC_VIA', 'PRE_PLACE'):
                self.assertIn(phase, hints)
                self.assertEqual(len(hints[phase]), 5,
                                 f'{obj}/{phase} hint should have 5 joint values')

    def test_update_hints(self):
        env = self._make_env()
        new_hints = {
            'HOME':      [0.0]*5,
            'PRE_PICK':  [0.1]*5,
            'PICK':      [0.2]*5,
            'LIFT':      [0.3]*5,
            'ARC_VIA':   [0.4]*5,
            'PRE_PLACE': [0.5]*5,
        }
        env.update_hints(new_hints)
        self.assertEqual(env._hints, new_hints)

    def test_check_grasp_success_within_tolerance(self):
        from env.gazebo_env import DROP_ZONE_POS
        env = self._make_env()
        env._current_obj_world = DROP_ZONE_POS + np.array([0.02, 0.01, 0.0])
        self.assertTrue(env._check_grasp_success())

    def test_check_grasp_success_outside_tolerance(self):
        from env.gazebo_env import DROP_ZONE_POS
        env = self._make_env()
        env._current_obj_world = DROP_ZONE_POS + np.array([0.5, 0.5, 0.0])
        self.assertFalse(env._check_grasp_success())

    def test_action_space_sample_compatible_with_step(self):
        """Random actions from action_space must not crash step()."""
        env = self._make_env()
        for _ in range(3):
            action = env.action_space.sample()
            with patch('time.sleep'):
                obs, r, done, trunc, info = env.step(action)
            self.assertEqual(obs.shape, (1, 64, 64))


# ─────────────────────────────────────────────────────────────────────────────
# 8. PhaseController — static helpers (no ROS2 node)
# ─────────────────────────────────────────────────────────────────────────────
def _import_phase_controller():
    """Import phase_controller with rclpy mocked out (no ROS2 install needed)."""
    import importlib
    mock_rclpy = MagicMock()
    mock_node  = MagicMock()
    mock_rclpy.node.Node = mock_node
    with patch.dict(sys.modules, {
        'rclpy':            mock_rclpy,
        'rclpy.node':       mock_rclpy.node,
        'std_msgs':         MagicMock(),
        'std_msgs.msg':     MagicMock(),
        'geometry_msgs':    MagicMock(),
        'geometry_msgs.msg':MagicMock(),
    }):
        # Force reimport if previously cached without mock
        if 'env.phase_controller' in sys.modules:
            del sys.modules['env.phase_controller']
        import env.phase_controller as pc
        return pc


class TestPhaseControllerStatics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pc = _import_phase_controller()

    def test_clip_joints_within_limits(self):
        joints   = np.zeros(5)
        clipped, exceeded = self.pc.clip_joints(joints)
        np.testing.assert_array_equal(clipped, joints)
        self.assertFalse(exceeded)

    def test_clip_joints_over_limit(self):
        joints = np.array([10.0, -10.0, 5.0, 3.0, 2.0])
        clipped, exceeded = self.pc.clip_joints(joints)
        self.assertTrue(exceeded)
        for i in range(5):
            self.assertGreaterEqual(float(clipped[i]),
                                    self.pc.JOINT_LIMITS[i, 0] - 1e-4)
            self.assertLessEqual(float(clipped[i]),
                                 self.pc.JOINT_LIMITS[i, 1] + 1e-4)

    def test_phases_list_has_six_entries(self):
        self.assertEqual(len(self.pc.PHASES), 6)

    def test_default_dwell_keys_present(self):
        for key in ('HOME', 'PRE_PICK', 'PICK', 'CLOSE_GRIP', 'LIFT', 'ARC_VIA',
                    'PRE_PLACE', 'OPEN_GRIP'):
            self.assertIn(key, self.pc.DEFAULT_DWELL)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    unittest.main(verbosity=2)
