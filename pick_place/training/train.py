#!/usr/bin/env python3
"""
Single-GPU SAC Training Loop
Implements Chen et al. (2023) Fig 12 + 13 training procedure:

  Stage 1 (fixed_pose_episodes):
    Object placed at a fixed position → agent learns basic grasping fast.
    After ~100 episodes the arm consistently finds a correct grasp pose
    (paper §5.2, Fig 13a).

  Stage 2 (random_pose_episodes):
    Transfer weights from Stage 1, then randomize object placement +
    vary object colors/sizes every 100 episodes (paper §5.2, Fig 14).

Paper table of results reproduced by evaluate.py after training.

Usage:
    python training/train.py --config configs/sac_config.yaml
    python training/train.py --config configs/sac_config.yaml --resume checkpoints/ep00500.pt
"""

import os
import sys
import time
import yaml
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
# ── Added TQDM Imports ───────────────────────────────────────────────────────
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Allow running from the pick_place/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.sac_agent import SACAgent
from env.gazebo_env import GazeboPickPlaceEnv
from utils.incremental_curriculum import IncrementalCurriculum

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train')


# ─────────────────────────────────────────────────────────────────────────────
# Optional TensorBoard (graceful fallback if not installed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAVE_TB = True
except ImportError:
    _HAVE_TB = False
    log.warning('TensorBoard not available — install torch with tensorboard extras.')


class _NullWriter:
    """No-op writer so the rest of the code doesn't need to branch."""
    def add_scalar(self, *a, **kw): pass
    def add_scalars(self, *a, **kw): pass
    def close(self): pass


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
def build_agent(cfg: dict, device: str) -> SACAgent:
    s = cfg['sac']
    return SACAgent(
        action_dim   = 30,          # 5 joints × 6 phases
        feature_dim  = 512,
        hidden_dim   = s['hidden_dim'],
        lr           = s['learning_rate'],
        gamma        = s['discount_factor'],
        tau          = s['target_smoothing'],
        alpha        = s['entropy_temperature'],
        buffer_size  = s['replay_buffer_size'],
        batch_size   = s['batch_size'],
        device       = device,
        auto_entropy = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
def run_episode(
    env:        GazeboPickPlaceEnv,
    agent:      SACAgent,
    batch_size: int,
    warmup:     bool,
    options:    dict,
    max_steps:  int, # 1. Pass the constraint parameter here
) -> dict:
    obs, _ = env.reset(options=options if options else None)
    ep_reward  = 0.0
    done       = truncated = False
    steps      = 0
    losses     = {}

    while not (done or truncated):
        # 2. Break early if we hit our configuration step budget
        if steps >= max_steps:
            truncated = True
            break

        if warmup:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        steps     += 1

        agent.store(obs, action, reward, next_obs, done or truncated)
        obs = next_obs

        if agent.buffer.is_ready(batch_size):
            losses = agent.update()

    info['ep_reward'] = ep_reward
    info['ep_steps']  = steps
    info['losses']    = losses
    return info


# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    cfg     = load_config(args.config)
    sac_cfg = cfg['sac']
    trn_cfg = cfg['training']

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    log.info(f'Device: {device}')

    os.makedirs(trn_cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(trn_cfg.get('log_dir', 'logs/'), exist_ok=True)

    writer = (SummaryWriter(trn_cfg.get('log_dir', 'logs/'))
              if _HAVE_TB else _NullWriter())

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = build_agent(cfg, device)
    start_ep = 0
    if args.resume:
        start_ep = agent.load(args.resume)
        log.info(f'Resumed from episode {start_ep}')

    # ── Environment ───────────────────────────────────────────────────────────
    obj_name = args.object or 'small_cube'
    env = GazeboPickPlaceEnv(
        node_name   = 'rl_train_env',
        object_name = obj_name,
        use_vision  = True,
        dwell_scale = 0.65,     # speed up for training
        verbose     = args.verbose,
    )

    # ── Curriculum ────────────────────────────────────────────────────────────
    curriculum = IncrementalCurriculum(
        fixed_episodes  = trn_cfg['fixed_pose_episodes'],
        random_episodes = trn_cfg['random_pose_episodes'],
    )
    curriculum.episode_count = start_ep

    # ── Training loop ─────────────────────────────────────────────────────────
    max_ep           = trn_cfg['fixed_pose_episodes'] + trn_cfg['random_pose_episodes']
    batch_size       = sac_cfg['batch_size']
    eval_interval    = trn_cfg.get('eval_interval', 100)
    save_interval    = trn_cfg.get('save_interval', 200)
    warmup_eps       = max(10, batch_size // 5)   # collect some experience first
    max_steps        = trn_cfg.get('max_steps_per_episode', 500) # Safeguard default if missing in yaml

    # Running stats
    recent_rewards   = []
    recent_successes = []
    best_sr          = 0.0
    t0               = time.time()

    log.info(f'Training {obj_name} for {max_ep} episodes (start={start_ep})')
    log.info(f'Stage 1 (fixed): {trn_cfg["fixed_pose_episodes"]} eps  '
             f'Stage 2 (random): {trn_cfg["random_pose_episodes"]} eps')

    # ── TQDM Context and Loop ─────────────────────────────────────────────────
    # Redirect logging so log.info statements print cleanly above the progress bar
    with logging_redirect_tqdm(loggers=[log]):
        pbar = tqdm(
            range(start_ep, max_ep),
            desc="Training Agent",
            unit="ep",
            dynamic_ncols=True
        )
        
        for episode in pbar:
            curr  = curriculum.step()
            stage = curr['stage']

            # ── Stage 1 → Stage 2 transfer checkpoint ───────────────────────────
            if curriculum.should_save_stage1_checkpoint():
                stage1_path = os.path.join(trn_cfg['checkpoint_dir'], 'stage1_transfer.pt')
                agent.save(stage1_path, episode=episode,
                           extra={'stage': 1, 'note': 'fixed→random transfer'})
                log.info(f'Stage-1 transfer weights saved → {stage1_path}')

            # ── Build reset options ──────────────────────────────────────────────
            options = {}
            pose, hints = curriculum.get_object_pose(obj_name)
            env.update_hints(hints)
            
            # Check if environment reset configuration has active modifications
            has_reset_env = False
            if stage == 2:
                options['object_pose'] = pose
                has_reset_env = True

            # ── Run episode ──────────────────────────────────────────────────────
            warmup = episode < start_ep + warmup_eps
            info   = run_episode(env, agent, batch_size, warmup, options, max_steps)

            ep_reward = info['ep_reward']
            success   = info.get('success', False)
            attempts  = info.get('total_attempts', 1)
            losses    = info.get('losses', {})

            recent_rewards.append(ep_reward)
            recent_successes.append(int(success))

            # ── Logging ──────────────────────────────────────────────────────────
            writer.add_scalar('train/reward',        ep_reward,            episode)
            writer.add_scalar('train/success',       float(success),       episode)
            writer.add_scalar('train/stage',         float(stage),         episode)
            writer.add_scalar('train/attempts',      float(attempts),      episode)
            writer.add_scalar('train/buffer_size',   len(agent.buffer),    episode)
            if losses:
                for k, v in losses.items():
                    writer.add_scalar(f'train/{k}', v, episode)

            # Update running window calculations
            w = min(50, len(recent_rewards))
            avg_r  = np.mean(recent_rewards[-w:]) if w > 0 else 0.0
            avg_sr = np.mean(recent_successes[-w:]) if w > 0 else 0.0

            # Dynamic updates to the progress bar post-fix text string
            pbar.set_postfix({
                'Stage': stage,
                'AvgR': f'{avg_r:+.1f}',
                'SR': f'{avg_sr:.0%}',
                'BestSR': f'{best_sr:.0%}'
            })

            # PRINT EVERY SINGLE EPISODE OUT
            elapsed = time.time() - t0
            log.info(
                f'Ep {episode:4d}/{max_ep} | '
                f'Stage {stage} | '
                f'ResetEnv: {has_reset_env} | '
                f'R={ep_reward:+.2f} AvgR={avg_r:+.2f} | '
                f'SR={avg_sr:.0%} | '
                f'Attempts={attempts} | '
                f'Buf={len(agent.buffer):,} | '
                f'{elapsed/60:.1f}min'
            )

            # ── Checkpoint ───────────────────────────────────────────────────────
            if episode > 0 and episode % save_interval == 0:
                ckpt_path = os.path.join(
                    trn_cfg['checkpoint_dir'], f'ep{episode:05d}.pt')
                agent.save(ckpt_path, episode=episode,
                           extra={'stage': stage, 'obj': obj_name})
                log.info(f'Checkpoint → {ckpt_path}')

            # ── Best model ───────────────────────────────────────────────────────
            if len(recent_successes) >= 20:
                recent_sr = np.mean(recent_successes[-20:])
                if recent_sr > best_sr:
                    best_sr   = recent_sr
                    best_path = os.path.join(trn_cfg['checkpoint_dir'], 'best_model.pt')
                    agent.save(best_path, episode=episode,
                               extra={'best_sr': best_sr, 'stage': stage})
                    if episode % 50 == 0:
                        log.info(f'New best SR={best_sr:.1%} → {best_path}')

            # ── Periodic eval log ─────────────────────────────────────────────────
            if episode > 0 and episode % eval_interval == 0:
                w_eval = min(eval_interval, len(recent_successes))
                log.info(
                    f'─── Eval summary @ ep {episode} ───────────────\n'
                    f'    Best SR so far : {best_sr:.1%}\n'
                    f'    Last {w_eval} eps SR: {np.mean(recent_successes[-w_eval:]):.1%}\n'
                    f'    Last {w_eval} avg R : {np.mean(recent_rewards[-w_eval:]):+.3f}'
                )

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(trn_cfg['checkpoint_dir'], 'final_model.pt')
    agent.save(final_path, episode=max_ep - 1)
    log.info(f'Training complete. Final model → {final_path}')
    log.info(f'Best success rate: {best_sr:.1%}')
    log.info(f'Total time: {(time.time()-t0)/60:.1f} min')

    writer.close()
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Single-GPU SAC Training')
    p.add_argument('--config',  default='configs/sac_config.yaml',
                    help='Path to sac_config.yaml')
    p.add_argument('--resume',  default=None,
                    help='Checkpoint path to resume from')
    p.add_argument('--object',  default='small_cube',
                    choices=['small_cube', 'large_cube', 'peg_cylinder', 'ellipsoid'],
                    help='Object to train on (default: small_cube)')
    p.add_argument('--cpu',     action='store_true',
                    help='Force CPU even if CUDA available')
    p.add_argument('--verbose', action='store_true',
                    help='Verbose phase controller output')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())