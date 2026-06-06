# Robotic Arm Pick-and-Place — SAC Reinforcement Learning in Gazebo

A full reinforcement learning pipeline for training a 5-DOF robotic arm to autonomously pick up objects and place them in a target drop zone, running entirely inside a ROS 2 + Gazebo Harmonic simulation. The agent learns from raw depth-image observations using Soft Actor-Critic (SAC) with a two-stage incremental curriculum.

---

## Overview

The system trains a robot arm to perform pick-and-place by treating the full motion sequence — approach, grasp, lift, arc, place — as a single reinforcement learning problem. Rather than hand-scripting trajectories, the agent learns to refine joint-angle corrections on top of human-provided grasp hints, eventually generalising to arbitrary object positions across the table workspace.

The implementation follows the methodology of Chen et al. (2023), adapting their depth-image SAC approach for a custom 5-joint arm with a claw gripper.

---

## System Architecture

```
Gazebo Harmonic  ←──────────────────────────────────────────────┐
  │  /camera/image_raw          /arm_controller/commands         │
  │  /detected_objects/poses    /gripper_controller/commands      │
  ▼                                                              │
GazeboPickPlaceEnv  (gym.Env wrapper)                           │
  ├── VisionPipeline  (YOLOv8n + depth crop → 64×64 state)      │
  ├── PhaseController (6-phase motion sequencer)                 │
  ├── RewardComputer  (sparse success + dense shaping)           │
  └── IncrementalCurriculum (Stage 1 fixed → Stage 2 random)    │
        │                                                        │
        ▼                                                        │
    SACAgent                                                     │
      ├── CNN encoder  (3-layer, 32→64→64 channels)             │
      ├── Actor network (Gaussian policy, 30D action)            │
      ├── Twin Q-networks + target networks                      │
      ├── Automatic entropy tuning                               │
      └── Replay buffer (200k transitions)  ────────────────────┘
```

---

## How It Works

### State & Action

The agent observes the world through a **64×64 grayscale depth crop** centred on the detected object — a compact visual representation that captures the object's pose relative to the arm without requiring privileged simulator state.

The action is a **30-dimensional joint-delta vector** (5 joints × 6 phases). At each episode, the agent outputs corrections to human-provided grasp hints for each phase of the motion. This delta-from-hints design lets the agent start from a physically reasonable baseline and learn fine-grained corrections, rather than learning raw kinematics from scratch.

### 6-Phase Motion Sequence

Every episode executes a fixed sequence of motion phases:

| Phase | Description |
|---|---|
| `HOME` | Return to neutral pose |
| `PRE_PICK` | Rotate base joint toward object |
| `PICK` | Lower arm to object height |
| `LIFT` | Raise object off the table |
| `ARC_VIA` | Swing arm toward drop zone |
| `PRE_PLACE` | Position above drop zone and release |

The agent outputs joint deltas for all 6 phases simultaneously. The `PhaseController` applies these deltas, clips to joint limits, and executes each phase with a configurable dwell time for the physics to settle.

### Reward

The reward is primarily **sparse**: +1 for placing the object within 7 cm of the drop zone centre, −1 for failure. A small **dense shaping** term (weighted at 0.3×) provides gradient signal during early training based on the object's proximity to the drop zone, whether it was lifted, and whether joint limits were violated.

### Two-Stage Curriculum

Training proceeds in two stages:

**Stage 1 — Fixed Pose (1000 episodes):** Objects are placed at fixed canonical positions. The arm learns a reliable basic grasp strategy quickly. Weights are saved as a transfer checkpoint.

**Stage 2 — Random Pose (2000 episodes):** Objects are placed randomly anywhere in the reachable workspace (x ∈ [0.30, 0.55] m, y ∈ [−0.25, 0.25] m). The agent must generalise. Grasp hints are re-computed geometrically from the object's position using inverse kinematics approximations. Object colour and size variants change every 100 episodes to improve visual robustness.

Transfer from Stage 1 to Stage 2 yields roughly **2.3× fewer training attempts** compared to training from scratch on random poses (6,443s / 1,323 attempts vs 15,076s / 3,635 attempts per the reference paper).

---

## Project Structure

```
src/arm_bringup/
├── pick_place/
│   ├── agent/
│   │   ├── networks.py          # CNN encoder, actor, twin Q-networks
│   │   ├── sac_agent.py         # SAC update loop, replay buffer interface
│   │   └── replay_buffer.py     # Ring-buffer experience replay
│   ├── env/
│   │   ├── gazebo_env.py        # gym.Env wrapping ROS 2 + Gazebo
│   │   ├── phase_controller.py  # 6-phase motion executor
│   │   └── reward.py            # Sparse + dense reward computation
│   ├── training/
│   │   ├── train.py             # Single-process training loop
│   │   └── train_parallel.py    # Multi-environment parallel training
│   ├── utils/
│   │   ├── incremental_curriculum.py  # Two-stage curriculum manager
│   │   ├── object_detector.py         # YOLOv8 + depth pipeline
│   │   └── vision_node.py             # ROS 2 vision publisher node
│   └── configs/
│       ├── sac_config.yaml      # Hyperparameters and training settings
│       └── objects_config.yaml  # Object poses, grasp hints, joint limits
├── launch/
│   ├── bringup.launch.py        # Standard simulation launch
│   └── bringup_rl.launch.py     # RL training launch (with controllers)
├── meshes/                      # STL meshes for arm links
└── checkpoints/                 # Saved model weights
```

---

## Supported Objects

Four graspable objects are configured, each with hand-tuned grasp hints that the agent refines:

| Object | Shape | Notable challenge |
|---|---|---|
| `small_cube` | Small box | Precision approach angle |
| `large_cube` | Large box | Wider gripper aperture needed |
| `peg_cylinder` | Cylinder | Rotation-agnostic grasp |
| `ellipsoid` | Egg shape | Curved surface, higher slip risk |

---

## Dependencies

- **ROS 2** (Humble or later)
- **Gazebo Harmonic**
- **Python 3.10+**
- `torch`, `gymnasium`, `numpy`
- `ultralytics` (YOLOv8)
- `cv_bridge`, `rclpy`
- `pyyaml`, `tqdm`
- `torch.utils.tensorboard` (optional, for training plots)

---

## Training

```bash
# Stage 1 + Stage 2 from scratch
python pick_place/training/train.py --config pick_place/configs/sac_config.yaml

# Resume from checkpoint
python pick_place/training/train.py \
    --config pick_place/configs/sac_config.yaml \
    --resume checkpoints/ep00500.pt

# Train on a specific object
python pick_place/training/train.py --object peg_cylinder

# Parallel training (multiple Gazebo instances)
python pick_place/training/train_parallel.py --config pick_place/configs/sac_config.yaml
```

Checkpoints are saved to `checkpoints/` every 200 episodes by default. The best-performing model (by 20-episode rolling success rate) is saved separately as `best_model.pt`. The Stage 1 transfer weights are saved as `stage1_transfer.pt` at the moment of stage transition.

TensorBoard logs are written to `logs/` and can be viewed with:

```bash
tensorboard --logdir logs/
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Replay buffer size | 200,000 | Ring buffer, uniform sampling |
| Batch size | 64 | |
| Learning rate | 0.001 | Adam, all networks |
| Discount factor γ | 0.99 | |
| Target smoothing τ | 0.005 | Polyak averaging |
| Entropy temperature α | Auto-tuned | Target entropy = −30 |
| CNN channels | 32 → 64 → 64 | 3-layer encoder |
| Action dimension | 30 | 5 joints × 6 phases |
| State dimension | 64×64×1 | Depth image crop |

---

## Performance Notes

Training speed is dominated by the physics dwell times in each motion phase. With Gazebo running at 20× real-time, the effective simulation time per episode is ~22 seconds of simulated motion, but the wall-clock time depends on the `dwell_scale` factor set in the environment.

For faster iteration, lower `dwell_scale` (set to `0.65` by default in `train.py`) and reduce the `phase_dwell` values in `sac_config.yaml` to match your machine's simulation speed. The parallel training script (`train_parallel.py`) runs multiple Gazebo instances concurrently and is recommended for serious training runs.
