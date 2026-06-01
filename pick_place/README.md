# Vision-Based Robotic Pick-and-Place — SAC + YOLO + Depth Anything V2
## ROS 2 Jazzy | Gazebo Harmonic | Accelerate Parallel Training

Implements Chen et al. (2023) "Vision-Based Robotic Object Grasping — A Deep Reinforcement
Learning Approach" adapted to your 5-DOF arm (joints 1-5) + gripper (joint6).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Camera (50 cm above table, 50° tilt)                        │
│    ↓ /camera/image_raw                                        │
│  DepthAnythingV2 + YOLOv8 → /detected_objects/poses         │
│    ↓ world (x,y,z) per object                                 │
│  SAC Agent  ←→  GazeboPickPlaceEnv (ROS2 gym wrapper)        │
│    ↓ 5-phase action sequence                                  │
│  arm_controller / gripper_controller                          │
│    ↓ joint positions                                          │
│  Gazebo Harmonic (Bullet Featherstone physics)                │
└──────────────────────────────────────────────────────────────┘
```

## 6-Phase Motion Sequence
| Phase | Description |
|-------|-------------|
| HOME  | All joints zero — safe neutral |
| PRE_PICK | Rotate J1 toward object, arm up |
| PICK  | Descend to grasp height |
| LIFT  | Raise object clear of table |
| ARC_VIA | Swing arm toward drop zone |
| PRE_PLACE | Position above drop zone, open gripper |

## Files
```
rl_pick_place/
├── env/
│   ├── gazebo_env.py          # ROS2 Gym environment
│   ├── phase_controller.py    # 6-phase motion FSM
│   └── reward.py              # Reward shaping (paper §4.1.3)
├── agent/
│   ├── sac_agent.py           # SAC (paper §4)
│   ├── networks.py            # CNN policy + Q-networks (paper Fig 7)
│   └── replay_buffer.py       # Experience replay (200k)
├── training/
│   ├── train.py               # Main training loop (Accelerate)
│   ├── train_parallel.py      # Multi-env parallel trainer
│   └── evaluate.py            # Evaluation & metrics
├── utils/
│   ├── vision_node.py         # YOLO + DepthAnything ROS2 node
│   ├── object_detector.py     # Detection → world coords pipeline
│   └── incremental_curriculum.py  # Fixed→random pose curriculum
├── configs/
│   ├── sac_config.yaml        # Hyperparameters (paper Table 1)
│   └── objects_config.yaml    # Object poses & grasp hints
├── scripts/
│   ├── run_training.sh        # Launch training
│   └── run_evaluation.sh      # Launch evaluation
└── tests/
    └── test_env.py            # Smoke tests
```

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision accelerate gymnasium numpy transformers
pip install ultralytics opencv-python
pip install --break-system-packages torch accelerate  # if system Python
```

### 2. Launch Gazebo
```bash
ros2 launch arm_bringup bringup_launch.py
```

### 3. Train (single GPU)
```bash
cd rl_pick_place
python training/train.py --config configs/sac_config.yaml
```

### 4. Train (Accelerate multi-GPU)
```bash
accelerate launch training/train_parallel.py --config configs/sac_config.yaml
```

### 5. Evaluate
```bash
python training/evaluate.py --checkpoint checkpoints/best_model.pt
```
