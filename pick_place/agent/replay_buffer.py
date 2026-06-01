#!/usr/bin/env python3
"""
Replay Buffer — Chen et al. (2023) §4, Table 1: buffer size 200,000, batch 64.
Stores (state, action, reward, next_state, done) tuples.
State = 64×64×1 float32 depth image crop.
"""

import numpy as np
import torch
from collections import deque
import random
from typing import Tuple


class ReplayBuffer:
    """
    Uniform experience replay buffer.
    Stores depth-image states to match paper §4.1.1.
    """
    def __init__(
        self,
        capacity:   int   = 200_000,
        img_size:   int   = 64,
        action_dim: int   = 5,
        device:     str   = 'cpu',
    ):
        self.capacity   = capacity
        self.img_size   = img_size
        self.action_dim = action_dim
        self.device     = device
        self.ptr        = 0
        self.size       = 0

        # Pre-allocate numpy arrays for speed
        self.states      = np.zeros((capacity, 1, img_size, img_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, 1, img_size, img_size), dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim),             dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),                      dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),                      dtype=np.float32)

    def push(
        self,
        state:      np.ndarray,   # (1, 64, 64) float32
        action:     np.ndarray,   # (5,) float32
        reward:     float,
        next_state: np.ndarray,   # (1, 64, 64) float32
        done:       bool,
    ):
        self.states[self.ptr]      = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.dones[self.ptr]       = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 64) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)

        states      = torch.FloatTensor(self.states[idx]).to(self.device)
        actions     = torch.FloatTensor(self.actions[idx]).to(self.device)
        rewards     = torch.FloatTensor(self.rewards[idx]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idx]).to(self.device)
        dones       = torch.FloatTensor(self.dones[idx]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int = 64) -> bool:
        return self.size >= batch_size

    def save(self, path: str):
        np.savez_compressed(
            path,
            states=self.states[:self.size],
            next_states=self.next_states[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            dones=self.dones[:self.size],
            ptr=np.array([self.ptr]),
            size=np.array([self.size]),
        )
        print(f"[ReplayBuffer] Saved {self.size} transitions → {path}")

    def load(self, path: str):
        data = np.load(path)
        n = int(data['size'][0])
        self.states[:n]      = data['states']
        self.next_states[:n] = data['next_states']
        self.actions[:n]     = data['actions']
        self.rewards[:n]     = data['rewards']
        self.dones[:n]       = data['dones']
        self.ptr  = int(data['ptr'][0])
        self.size = n
        print(f"[ReplayBuffer] Loaded {n} transitions from {path}")
