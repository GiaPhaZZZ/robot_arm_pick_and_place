#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) Agent
Implements Chen et al. (2023) §4, Figure 4 algorithm:

  Equations implemented:
    Eq 1: J_Q(θ) = E[(Q_θ(s,a) - Q̂(s,a))²/2]          ← Q loss (MSE)
    Eq 2: Q̂(s,a) = r + γ·E[V_θ'(s')]                    ← Bellman target
    Eq 3: ∇J_Q(θ) — SGD Q update
    Eq 4: θ' ← τθ + (1-τ)θ'                              ← Polyak update
    Eq 5: J_π(φ) = E[α·log π_φ(a|s) - Q_θ(s,a)]        ← Policy loss
    Eq 7: ∇J_π(φ) — policy gradient with reparameterization

Hyperparameters from paper Table 1.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from agent.networks import SACNetworks
from agent.replay_buffer import ReplayBuffer


class SACAgent:
    """
    SAC agent for 5-DOF arm pick-and-place.
    Operates on 64×64×1 depth-image states, outputs 5D joint-delta actions.
    """
    def __init__(
        self,
        action_dim:  int   = 5,
        feature_dim: int   = 512,
        hidden_dim:  int   = 64,
        lr:          float = 1e-3,
        gamma:       float = 0.99,
        tau:         float = 0.005,
        alpha:       float = 0.01,
        buffer_size: int   = 200_000,
        batch_size:  int   = 64,
        device:      str   = 'cpu',
        auto_entropy: bool = True,   # Tune α automatically
    ):
        self.action_dim = action_dim
        self.gamma      = gamma
        self.tau        = tau
        self.alpha      = alpha
        self.batch_size = batch_size
        self.device     = torch.device(device)
        self.auto_entropy = auto_entropy

        # ── Networks (paper §4.2, Figure 7) ─────────────────────────────────
        self.nets = SACNetworks(
            action_dim=action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            tau=tau,
        ).to(self.device)

        # ── Optimizers (paper: Adam, lr=0.001) ──────────────────────────────
        self.policy_optimizer = torch.optim.Adam(
            self.nets.get_policy_params(), lr=lr)
        self.q_optimizer = torch.optim.Adam(
            self.nets.get_q_params(), lr=lr)

        # ── Automatic entropy tuning ─────────────────────────────────────────
        if auto_entropy:
            # Target entropy = -|action_dim| (heuristic from SAC paper)
            self.target_entropy  = -action_dim
            self.log_alpha       = torch.zeros(1, requires_grad=True,
                                               device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha           = self.log_alpha.exp().item()

        # ── Replay buffer (paper: 200k capacity, batch 64) ───────────────────
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            action_dim=action_dim,
            device=str(self.device),
        )

        # ── Stats ────────────────────────────────────────────────────────────
        self.update_count  = 0
        self.total_steps   = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        depth_img: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Given a 64×64 depth crop, returns a 5D action in [-1, 1].
        deterministic=True for evaluation (use mean, no noise).
        """
        state = torch.FloatTensor(depth_img).unsqueeze(0).to(self.device)
        if not isinstance(depth_img, np.ndarray) or depth_img.ndim == 2:
            # Ensure (1, 64, 64) shape
            state = state.unsqueeze(0)

        if deterministic:
            _, _, action = self.nets.policy.sample(state)
        else:
            action, _, _ = self.nets.policy.sample(state)

        return action.squeeze(0).cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # Store experience
    # ─────────────────────────────────────────────────────────────────────────

    def store(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """Add transition to replay buffer (paper Fig 4, line 8)."""
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Update step (paper Figure 4, lines 10-14)
    # ─────────────────────────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        One gradient step on sampled batch.
        Returns dict of loss values for logging.
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # ── 1. Compute Q targets (paper Eq. 2) ─────────────────────────────
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.nets.policy.sample(next_states)
            q1_next = self.nets.q1_target(next_states, next_actions)
            q2_next = self.nets.q2_target(next_states, next_actions)
            # Clipped double-Q (prevents overestimation)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + (1.0 - dones) * self.gamma * q_next

        # ── 2. Q-function update (paper Eq. 1, 3) ──────────────────────────
        q1_pred = self.nets.q1(states, actions)
        q2_pred = self.nets.q2(states, actions)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        q_loss  = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.nets.q1.parameters()) +
                                       list(self.nets.q2.parameters()), 1.0)
        self.q_optimizer.step()

        # ── 3. Policy update (paper Eq. 5-7) ───────────────────────────────
        new_actions, log_pi, _ = self.nets.policy.sample(states)
        q1_new = self.nets.q1(states, new_actions)
        q2_new = self.nets.q2(states, new_actions)
        q_new  = torch.min(q1_new, q2_new)

        # Maximize: E[Q - α·log π]  ≡  Minimize: E[α·log π - Q]
        policy_loss = (self.alpha * log_pi - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nets.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        # ── 4. Auto entropy temperature update ──────────────────────────────
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha.exp() *
                           (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ── 5. Polyak update target networks (paper Eq. 4) ──────────────────
        self.nets.soft_update_targets()
        self.update_count += 1

        return {
            'q1_loss':     q1_loss.item(),
            'q2_loss':     q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss':  alpha_loss.item() if self.auto_entropy else 0.0,
            'alpha':       self.alpha,
            'q1_mean':     q1_pred.mean().item(),
            'q2_mean':     q2_pred.mean().item(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str, episode: int = 0, extra: dict = None):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        payload = {
            'nets':            self.nets.state_dict(),
            'policy_opt':      self.policy_optimizer.state_dict(),
            'q_opt':           self.q_optimizer.state_dict(),
            'update_count':    self.update_count,
            'total_steps':     self.total_steps,
            'episode':         episode,
            'alpha':           self.alpha,
        }
        if self.auto_entropy:
            payload['log_alpha']      = self.log_alpha
            payload['alpha_opt']      = self.alpha_optimizer.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[SAC] Saved checkpoint → {path}  (ep={episode})")

    def load(self, path: str) -> int:
        payload = torch.load(path, map_location=self.device)
        self.nets.load_state_dict(payload['nets'])
        self.policy_optimizer.load_state_dict(payload['policy_opt'])
        self.q_optimizer.load_state_dict(payload['q_opt'])
        self.update_count = payload.get('update_count', 0)
        self.total_steps  = payload.get('total_steps', 0)
        self.alpha        = payload.get('alpha', self.alpha)
        if self.auto_entropy and 'log_alpha' in payload:
            self.log_alpha.data = payload['log_alpha'].data
            self.alpha_optimizer.load_state_dict(payload['alpha_opt'])
        ep = payload.get('episode', 0)
        print(f"[SAC] Loaded checkpoint ← {path}  (ep={ep})")
        return ep
