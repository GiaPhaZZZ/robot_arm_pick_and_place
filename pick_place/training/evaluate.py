#!/usr/bin/env python3
"""
Evaluation Script — Matches Chen et al. (2023) Table 3 methodology.

Paper reports:
    Object          SR      In training set?
    Building block  19/20   yes
    Apple           6/10    no
    Banana          6/10    yes
    Orange          8/10    no
    Cup             9/10    no

Our evaluation:
    Per object: 20 episodes × deterministic policy
    Reports: success rate, mean attempts, mean episode reward

Usage:
    python training/evaluate.py --checkpoint checkpoints/best_model.pt
    python training/evaluate.py --checkpoint checkpoints/best_model.pt \
        --objects small_cube large_cube peg_cylinder ellipsoid \
        --n-episodes 20
"""

import os
import sys
import yaml
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tabulate import tabulate  # pip install tabulate

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.sac_agent import SACAgent
from env.gazebo_env import GazeboPickPlaceEnv

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('evaluate')


# ─────────────────────────────────────────────────────────────────────────────
def run_eval(
    agent:      SACAgent,
    env:        GazeboPickPlaceEnv,
    n_episodes: int,
    obj_name:   str,
    deterministic: bool = True,
) -> dict:
    """
    Evaluate agent on N episodes. Returns per-object metrics.
    Paper §5.3: up to 3 consecutive failures → task counted as failure.
    """
    results = {
        'successes':        0,
        'total_episodes':   n_episodes,
        'attempts_list':    [],
        'rewards_list':     [],
        'first_attempt_ok': 0,
    }

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = truncated = False
        consecutive_failures = 0
        attempt = 0

        while not (done or truncated):
            action = agent.select_action(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            attempt   += 1

            if not info.get('success', False):
                consecutive_failures += 1
                # Paper §5.3: task fails after 3 consecutive failures
                if consecutive_failures >= 3:
                    truncated = True
            else:
                consecutive_failures = 0

        success = info.get('success', False)
        results['successes']      += int(success)
        results['attempts_list'].append(attempt)
        results['rewards_list'].append(ep_reward)
        if info.get('first_attempt_ok', False):
            results['first_attempt_ok'] += 1

        status = '✓' if success else '✗'
        log.info(f'  [{obj_name}] Ep {ep+1:3d}/{n_episodes} {status} '
                 f'attempts={attempt} reward={ep_reward:+.2f}')

    results['success_rate']   = results['successes'] / n_episodes
    results['mean_attempts']  = float(np.mean(results['attempts_list']))
    results['mean_reward']    = float(np.mean(results['rewards_list']))
    results['std_reward']     = float(np.std(results['rewards_list']))
    results['first_sr']       = results['first_attempt_ok'] / n_episodes
    return results


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    log.info(f'Evaluating checkpoint: {args.checkpoint}')
    log.info(f'Device: {device}')

    # ── Load agent ────────────────────────────────────────────────────────────
    agent = SACAgent(
        action_dim  = 30,
        feature_dim = 512,
        hidden_dim  = 64,
        device      = device,
    )
    ep = agent.load(args.checkpoint)
    log.info(f'Loaded weights from episode {ep}')

    # ── Objects to evaluate ───────────────────────────────────────────────────
    objects = args.objects or ['small_cube', 'large_cube', 'peg_cylinder', 'ellipsoid']
    all_results = {}

    for obj_name in objects:
        log.info(f'\n{"─"*50}')
        log.info(f'Evaluating: {obj_name}  ({args.n_episodes} episodes)')
        log.info(f'{"─"*50}')

        env = GazeboPickPlaceEnv(
            node_name   = f'eval_env_{obj_name}',
            object_name = obj_name,
            use_vision  = True,
            verbose     = False,
        )

        try:
            result = run_eval(
                agent, env,
                n_episodes    = args.n_episodes,
                obj_name      = obj_name,
                deterministic = not args.stochastic,
            )
            all_results[obj_name] = result
        finally:
            env.close()

    # ── Print Table (paper Table 3 style) ────────────────────────────────────
    print(f'\n{"="*70}')
    print('EVALUATION RESULTS')
    print(f'Checkpoint: {args.checkpoint}  |  N={args.n_episodes} per object')
    print(f'{"="*70}')

    rows = []
    for obj, r in all_results.items():
        rows.append([
            obj,
            f"{r['successes']}/{r['total_episodes']}",
            f"{r['success_rate']:.1%}",
            f"{r['mean_attempts']:.1f}",
            f"{r['mean_reward']:+.3f} ± {r['std_reward']:.3f}",
            f"{r['first_sr']:.1%}",
        ])

    headers = ['Object', 'Success', 'SR', 'Avg Attempts',
               'Reward (mean±std)', 'First-Attempt SR']
    print(tabulate(rows, headers=headers, tablefmt='fancy_grid'))
    print()

    # Overall
    all_sr = np.mean([r['success_rate'] for r in all_results.values()])
    log.info(f'Overall success rate: {all_sr:.1%}')

    # Save results
    if args.output:
        import json
        # Convert numpy types for JSON serialization
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {
                kk: vv for kk, vv in v.items()
                if not isinstance(vv, list)
            }
        with open(args.output, 'w') as f:
            json.dump(serializable, f, indent=2)
        log.info(f'Results saved → {args.output}')


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Evaluate SAC Pick-and-Place Agent')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--objects', nargs='+', default=None,
                   choices=['small_cube', 'large_cube', 'peg_cylinder', 'ellipsoid'])
    p.add_argument('--n-episodes', type=int, default=20)
    p.add_argument('--stochastic', action='store_true',
                   help='Use stochastic policy (default: deterministic)')
    p.add_argument('--cpu',    action='store_true')
    p.add_argument('--output', default=None,
                   help='JSON output path for results')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
