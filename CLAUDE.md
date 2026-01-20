# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Reinforcement Learning (DRL) based robot path planning using ROS2 Humble and Gazebo simulation. The system trains agents (TQC, TD7, SAC, A3C algorithms) to navigate a Scout Mini robot through environments with obstacles.

## Build Commands

```bash
# Build entire workspace
cd ros2_ws && source /opt/ros/humble/setup.bash && colcon build

# Build single package
colcon build --packages-select drl_agent

# Source after build
source ros2_ws/install/setup.bash

# Clean build
rm -rf build/ install/ log/ && colcon build
```

## Running the System

### DRL Training/Testing
```bash
# Train TQC agent (requires Gazebo running separately)
ros2 run drl_agent train_tqc_agent.py

# Test TQC agent
ros2 launch drl_agent test_tqc.launch.py

# Run environment node standalone
ros2 run drl_agent environment.py --ros-args -p environment_mode:=train
```

### Gazebo Simulation
```bash
# Launch Gazebo with Scout Mini robot
ros2 launch scout_gazebo_sim scout_mini_empty_world.launch.py

# Spawn robot only
ros2 launch scout_gazebo_sim spawn_scout_mini.launch.py
```

### Navigation Stack
```bash
# Nav2 with localization
ros2 launch scout_nav2 nav2.launch.py slam:=False

# Nav2 with SLAM
ros2 launch scout_nav2 nav2.launch.py slam:=True
```

## Docker Environment

```bash
# Build image
docker build -t drl_path_planning .

# Run with GPU
docker run --gpus all -it --rm -v $(pwd):/root/drl_path_final --network host drl_path_planning

# Set CycloneDDS for container communication
export CYCLONEDDS_URI=file://$(pwd)/cyclonedds_config.xml
```

## Architecture

### Service-Based Environment Interface

The DRL system uses ROS2 services for agent-environment communication:

```
Environment Node (environment.py)          Agent Node (train_*_agent.py)
├── /reset           ←──────────────────── Calls reset at episode start
├── /step            ←──────────────────── Calls step with action
├── /get_dimensions  ←──────────────────── Gets state_dim, action_dim
├── /seed            ←──────────────────── Sets random seed
└── /action_space_sample ←──────────────── Samples random action (warmup)
```

Service definitions are in `drl_agent_interfaces/srv/`.

### Key ROS2 Packages

| Package | Purpose |
|---------|---------|
| `drl_agent` | DRL environment, policies, training scripts |
| `drl_agent_interfaces` | Custom service/action message definitions |
| `ugv_gazebo_sim` | Gazebo simulation for Scout Mini robot |
| `scout_nav2` | Nav2 configuration and launch files |

### DRL State/Action Space

- **State (80D)**: Agent pose (4D) + LiDAR observations (76D)
- **Action (2D)**: Linear velocity [-1.0, 1.0], Angular velocity [-1.4, 1.4] rad/s
- **Topics**: `/cmd_vel` (commands), `/odometry` (state), `/laser_scan` or `/points` (sensors)

### Algorithm Implementations

Located in `drl_agent/scripts/policy/`:
- `tqc_agent.py` - Truncated Quantile Critics (primary algorithm)
- `td7_agent.py` - Twin Delayed DDPG v7
- `sac_agent.py` - Soft Actor-Critic
- `a3c_agent.py` - Asynchronous Advantage Actor-Critic

Each has corresponding `train_*.py` and `test_*.py` scripts.

## Configuration Files

All in `drl_agent/config/`:

| File | Purpose |
|------|---------|
| `environment.yaml` | State/action dims, collision thresholds, zone detection params |
| `hyperparameters_tqc.yaml` | TQC network architecture, learning rates, buffer size |
| `train_tqc_config.yaml` | Training params: max steps, eval frequency, warmup |
| `test_tqc_config.yaml` | Testing params: episodes, model path |

## Key Training Parameters

From config files:
- Max timesteps: 1,000,000
- Warmup steps: 25,000 (random actions before training)
- Eval frequency: 5,000 steps
- Batch size: 256
- Buffer size: 1,000,000
- Goal threshold: 0.42m
- Collision threshold: 0.7m (zone-based with 8 zones)

## File Organization

```
ros2_ws/src/
├── drl_agent/
│   ├── scripts/
│   │   ├── environment/    # Environment node and interface
│   │   ├── policy/         # DRL algorithms and train/test scripts
│   │   └── utils/          # Replay buffer, file manager, plotting
│   ├── config/             # YAML configuration files
│   └── launch/             # ROS2 launch files
├── drl_agent_interfaces/   # Custom ROS2 messages (srv/, action/)
├── ugv_gazebo_sim/         # Robot URDF, Gazebo config, worlds
└── scout_nav2/             # Nav2 configuration
```

## Common Issues

### rosdep Installation
```bash
rosdep install --from-paths src -yi --rosdistro humble \
  --skip-keys='libgraphicsmagick++1-dev graphicsmagick-libmagick-dev-compat'
```

### DDS in Docker
CycloneDDS uses loopback interface. Edit `cyclonedds_config.xml` for inter-container communication.
