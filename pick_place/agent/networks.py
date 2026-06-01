#!/usr/bin/env python3
"""
SAC Neural Network Architecture
Implements Chen et al. (2023) Figure 7:
  - Policy Network:     3×CNN → 4×FC → Tanh output (mean, log_std)
  - Q-Value Networks:   3×CNN (shared) + action concat → 4×FC → scalar Q
  - Target Q-Networks:  frozen copies updated via Polyak averaging

State:  64×64×1 depth image crop of detected object (paper §4.1.1)
Action: 5D continuous joint deltas ∈ [-1,1] per phase (paper §4.1.2 adapted)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON     = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
class DepthCNNEncoder(nn.Module):
    """
    Shared CNN backbone — extracts features from 64×64×1 depth images.
    Architecture: paper Fig 7 — Conv(32,8,4) → Conv(64,4,2) → Conv(64,3,3) → FC(512)
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            # Conv layer 1: 32 filters, 8×8 kernel, stride 4  →  15×15
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            # Conv layer 2: 64 filters, 4×4 kernel, stride 2  →  6×6
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            # Conv layer 3: 64 filters, 3×3 kernel, stride 1  →  4×4
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # Compute flat size after conv
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            flat  = self.net(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, depth_img: torch.Tensor) -> torch.Tensor:
        """depth_img: (B, 1, 64, 64) float32 in [0, 1]"""
        return self.fc(self.net(depth_img))


# ─────────────────────────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    """
    Actor (policy network) π_φ(a|s).
    Input:  64×64×1 depth image
    Output: (mean, log_std) of 5D action — one delta per arm joint.

    Paper §4.1.2: action a ∈ [-1,1]^2 (image plane) → adapted here to
    5D joint-space deltas scaled by per-phase action_delta_range.
    Tanh squashing applied at output (paper Fig 7a).
    """
    def __init__(
        self,
        action_dim:  int   = 5,
        feature_dim: int   = 512,
        hidden_dim:  int   = 64,
    ):
        super().__init__()
        self.encoder = DepthCNNEncoder(in_channels=1, feature_dim=feature_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, depth_img: torch.Tensor):
        """Returns (mean, log_std) — unbounded."""
        feat    = self.encoder(depth_img)
        hidden  = self.fc_layers(feat)
        mean    = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, depth_img: torch.Tensor):
        """
        Reparameterized sample + log_prob (SAC paper Eq. 5-7).
        Returns: action ∈ [-1,1], log_prob, mean (for deterministic eval).
        """
        mean, log_std = self.forward(depth_img)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t  = dist.rsample()                      # reparameterized
        y_t  = torch.tanh(x_t)                    # squash to [-1,1]

        # Corrected log-prob for Tanh transform (SAC Appendix C)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1.0 - y_t.pow(2) + EPSILON)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)

        mean_out  = torch.tanh(mean)
        return y_t, log_prob, mean_out


# ─────────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Soft Action-Value Network Q_θ(s, a).
    Input:  depth image + action vector
    Output: scalar Q-value

    Architecture: paper Fig 7b/c — same CNN backbone, concat action after FC,
    then more FC layers → scalar output (ReLU activations throughout).
    """
    def __init__(
        self,
        action_dim:  int = 5,
        feature_dim: int = 512,
        hidden_dim:  int = 64,
    ):
        super().__init__()
        self.encoder = DepthCNNEncoder(in_channels=1, feature_dim=feature_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, depth_img: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(depth_img)
        x    = torch.cat([feat, action], dim=-1)
        return self.fc_layers(x)


# ─────────────────────────────────────────────────────────────────────────────
class SACNetworks(nn.Module):
    """
    Container for all 5 SAC networks (paper §4):
      Q_θ1, Q_θ2  — two soft action-value networks (reduces overestimation)
      Q_θ1', Q_θ2'— two target soft action-value networks (frozen, Polyak updated)
      π_φ         — policy network
    """
    def __init__(
        self,
        action_dim:  int   = 5,
        feature_dim: int   = 512,
        hidden_dim:  int   = 64,
        tau:         float = 0.005,
    ):
        super().__init__()
        self.tau = tau

        self.policy   = PolicyNetwork(action_dim, feature_dim, hidden_dim)
        self.q1       = QNetwork(action_dim, feature_dim, hidden_dim)
        self.q2       = QNetwork(action_dim, feature_dim, hidden_dim)
        self.q1_target = QNetwork(action_dim, feature_dim, hidden_dim)
        self.q2_target = QNetwork(action_dim, feature_dim, hidden_dim)

        # Initialize targets as hard copies
        self._hard_update_targets()

    def _hard_update_targets(self):
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def soft_update_targets(self):
        """Polyak averaging: θ' ← τθ + (1-τ)θ'  (paper Eq. 4)"""
        for tp, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        for tp, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def get_policy_params(self):
        return self.policy.parameters()

    def get_q_params(self):
        return list(self.q1.parameters()) + list(self.q2.parameters())
