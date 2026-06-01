#!/usr/bin/env python3
"""
Parallel SAC Training with Accelerate
Spawns N parallel Gazebo environments, each in its own ROS2 namespace.
Gradients are averaged across all processes automatically by Accelerate.

Usage:
    # 4 parallel environments (4 GPUs or 4 CPU processes)
    accelerate config   # first-time setup
    accelerate launch --num_processes 4 training/train_parallel.py \
        --config configs/sac_config.yaml

    # 2 GPUs
    accelerate launch --num_processes 2 --mixed_precision fp16 \
        training/train_parallel.py --config configs/sac_config.yaml

Each process handles one Gazebo instance + one object type:
    Process 0: small_cube
    Process 1: large_cube
    Process 2: peg_cylinder
    Process 3: ellipsoid

Shared replay buffer via main process redistribution.
"""

import os
import sys
import time
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.sac_agent import SACAgent
from agent.replay_buffer import ReplayBuffer
from env.gazebo_env import GazeboPickPlaceEnv
from utils.incremental_curriculum import IncrementalCurriculum

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][P%(process)d][%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train_parallel')


OBJECTS = ['small_cube', 'large_cube', 'peg_cylinder', 'ellipsoid']


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_parallel(args):
    accelerator = Accelerator(
        log_with='tensorboard',
        project_dir=args.log_dir or 'logs/',
        mixed_precision='fp16' if args.fp16 else 'no',
        gradient_accumulation_steps=args.grad_accum,
    )
    cfg     = load_cfg(args.config)
    sac_cfg = cfg['sac']
    trn_cfg = cfg['training']
    device  = accelerator.device
    is_main = accelerator.is_main_process
    rank    = accelerator.process_index
    world   = accelerator.num_processes

    set_seed(args.seed + rank)

    if is_main:
        os.makedirs(trn_cfg['checkpoint_dir'], exist_ok=True)
        writer = SummaryWriter(trn_cfg.get('log_dir', 'logs/'))
        log.info(f'Parallel training on {world} processes')

    # ── Per-process object assignment ─────────────────────────────────────────
    obj_name = OBJECTS[rank % len(OBJECTS)]
    log.info(f'[P{rank}] Assigned object: {obj_name}')

    # ── SAC Agent (each process has own copy, synced via Accelerate) ──────────
    agent = SACAgent(
        action_dim   = 30,
        feature_dim  = 512,
        hidden_dim   = sac_cfg['hidden_dim'],
        lr           = sac_cfg['learning_rate'],
        gamma        = sac_cfg['discount_factor'],
        tau          = sac_cfg['target_smoothing'],
        alpha        = sac_cfg['entropy_temperature'],
        buffer_size  = sac_cfg['replay_buffer_size'] // world,  # split buffer
        batch_size   = sac_cfg['batch_size'],
        device       = str(device),
        auto_entropy = True,
    )

    # Prepare with Accelerate (handles distributed training)
    (agent.nets.policy, agent.nets.q1, agent.nets.q2,
     agent.policy_optimizer, agent.q_optimizer) = accelerator.prepare(
        agent.nets.policy, agent.nets.q1, agent.nets.q2,
        agent.policy_optimizer, agent.q_optimizer,
    )

    # Load checkpoint if resuming
    start_ep = 0
    if args.resume:
        start_ep = agent.load(args.resume)
        log.info(f'[P{rank}] Resumed from ep {start_ep}')

    # ── Environment ───────────────────────────────────────────────────────────
    env = GazeboPickPlaceEnv(
        node_name   = f'rl_env_{rank}',
        object_name = obj_name,
        use_vision  = True,
        dwell_scale = 0.65,   # faster for training
        verbose     = False,
    )

    curriculum = IncrementalCurriculum(
        fixed_episodes  = trn_cfg['fixed_pose_episodes'],
        random_episodes = trn_cfg['random_pose_episodes'],
    )
    curriculum.episode_count = start_ep

    # ── Main loop ─────────────────────────────────────────────────────────────
    max_ep            = trn_cfg['fixed_pose_episodes'] + trn_cfg['random_pose_episodes']
    max_steps         = trn_cfg.get('max_steps_per_episode', float('inf'))
    recent_successes  = []
    total_t           = time.time()
    
    if is_main:
        pbar = tqdm(total=max_ep, initial=start_ep, desc="Training", dynamic_ncols=True)

    for episode in range(start_ep, max_ep):
        curr = curriculum.step()
        stage = curr['stage']

        # Stage-1 → Stage-2 transition checkpoint
        if curriculum.should_save_stage1_checkpoint() and is_main:
            ckpt = os.path.join(trn_cfg['checkpoint_dir'], 'stage1_transfer.pt')
            unwrapped = accelerator.unwrap_model(agent.nets)
            torch.save(unwrapped.state_dict(), ckpt)
            log.info(f'[P{rank}] Stage-1 transfer weights saved → {ckpt}')

        # Get hints for this episode
        _, hints = curriculum.get_object_pose(obj_name)
        env.update_hints(hints)
        options = {}
        if stage == 2:
            pose, _ = curriculum.get_object_pose(obj_name)
            options['object_pose'] = pose

        obs, _ = env.reset(options=options if options else None)
        ep_reward = 0.0
        done = truncated = False
        step_count = 0

        # Run episode steps with max step safety constraint
        while not (done or truncated):
            # Enforce max step limit manually if reached
            if step_count >= max_steps:
                truncated = True
                break

            # Sample action
            if len(agent.buffer) < sac_cfg['batch_size'] * 3:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1

            agent.store(obs, action, reward, next_obs, done or truncated)
            obs = next_obs

            # Gradient step
            if agent.buffer.is_ready(sac_cfg['batch_size']):
                with accelerator.accumulate(agent.nets.policy, agent.nets.q1, agent.nets.q2):
                    losses = agent.update()

        success = info.get('success', False) if 'info' in locals() else False
        recent_successes.append(int(success))

        # ── Aggregate metrics across processes ───────────────────────────────
        success_tensor = torch.tensor([float(success)], device=device)
        reward_tensor  = torch.tensor([ep_reward],      device=device)
        
        # Accelerate gather averages across all processes
        all_sr     = accelerator.gather(success_tensor).mean().item()
        all_reward = accelerator.gather(reward_tensor).mean().item()

        if is_main:
            writer.add_scalar('parallel/success_rate_avg', all_sr,     episode)
            writer.add_scalar('parallel/reward_avg',       all_reward,  episode)
            writer.add_scalar('parallel/stage',            stage,       episode)

            recent_sr = np.mean(recent_successes[-50:]) if recent_successes else 0
            
            # --- UPDATE PROGRESS BAR ---
            pbar.update(1)
            pbar.set_postfix({
                'Stage': stage,
                'AllSR': f'{all_sr:.1%}',
                'LocalSR': f'{recent_sr:.1%}',
                'AvgR': f'{all_reward:+.2f}'
            })

            # Print status to terminal for EVERY episode safely using pbar.write
            pbar.write(f'[Ep {episode}] AllSR={all_sr:.1%} | R={all_reward:+.2f} | LocalSR={recent_sr:.1%} | Steps={step_count}')

            # Save checkpoint
            if episode % trn_cfg['save_interval'] == 0 and episode > 0:
                unwrapped = accelerator.unwrap_model(agent.nets)
                ckpt_path = os.path.join(
                    trn_cfg['checkpoint_dir'], f'parallel_ep{episode:05d}.pt')
                torch.save({
                    'nets': unwrapped.state_dict(),
                    'episode': episode,
                    'success_rate': all_sr,
                }, ckpt_path)
                pbar.write(f'Checkpoint → {ckpt_path}')

        accelerator.wait_for_everyone()

    # Close the progress bar at the end
    if is_main:
        pbar.close()  
        log.info(f'Parallel training done | {time.time()-total_t:.0f}s')
        writer.close()

    env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    default='configs/sac_config.yaml')
    p.add_argument('--resume',    default=None)
    p.add_argument('--seed',      type=int,  default=42)
    p.add_argument('--fp16',      action='store_true')
    p.add_argument('--grad-accum',type=int,  default=1, dest='grad_accum')
    p.add_argument('--log-dir',   default=None, dest='log_dir')
    return p.parse_args()


if __name__ == '__main__':
    train_parallel(parse_args())