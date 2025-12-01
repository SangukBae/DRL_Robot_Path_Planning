# DRL Robot Path Planning

Deep Reinforcement Learning (DRL) based autonomous robot path planning system using ROS2 Humble, Gazebo simulation, and Stable-Baselines3.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Development Environment](#development-environment)
- [ROS2 Workspace Usage](#ros2-workspace-usage)
- [CycloneDDS Configuration](#cyclonedds-configuration)
- [Scout Robot Navigation Package](#scout-robot-navigation-package)
- [CUDA Environment](#cuda-environment)
- [Common Issues & Troubleshooting](#common-issues--troubleshooting)
- [Development Workflow](#development-workflow)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [Resources](#resources)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact & Support](#contact--support)

## Project Overview

This project implements intelligent path planning for mobile robots using deep reinforcement learning algorithms. The system integrates:

- **ROS2 Humble**: Robot Operating System 2 for robot control and communication
- **Gazebo (Ignition Fortress)**: High-fidelity physics simulation environment
- **Stable-Baselines3**: State-of-the-art reinforcement learning algorithms (PPO, SAC, TD3, etc.)
- **PyTorch 2.4.1**: Deep learning framework with CUDA 11.8 acceleration
- **Navigation2**: Advanced navigation stack for autonomous navigation
- **Scout Robot**: AgileX Scout mobile robot platform

The robot learns to navigate through complex environments (AWS Small Warehouse world) using sensor data and rewards-based training.

## Features

- GPU-accelerated DRL training with CUDA 11.8 support
- Dockerized development environment for reproducibility
- Integration of Stable-Baselines3 RL algorithms with ROS2
- Gazebo simulation with realistic warehouse environment
- Scout robot model with LiDAR sensors
- Navigation2 integration for hybrid classical/RL navigation
- CycloneDDS middleware configuration for optimized ROS2 communication

## System Requirements

### Hardware
- NVIDIA GPU with CUDA 11.8 support (recommended for training)
- At least 8GB RAM
- 10GB+ disk space

### Software
- Docker with NVIDIA Container Toolkit
- Ubuntu 20.04/22.04 host system (recommended)
- X11 for GUI visualization (optional)

## Quick Start

### 1. Build Docker Image

```bash
docker build -t drl_robot_path_planning:first .
```

### 2. Run Container

**With GPU support:**
```bash
docker run -it \
  --privileged \
  --gpus all \
  --net=host \
  --name DRL_Robot_Path_Planning \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI=file:///root/cyclonedds_config.xml \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/sangukbae/DRL_Robot_Path_Planning:/root/DRL_Robot_Path_Planning \
  -v /home/sangukbae/DRL_Robot_Path_Planning/cyclonedds_config.xml:/root/cyclonedds_config.xml \
  drl_robot_path_planning:first \
  /bin/bash

# in local terminal
xhost +local:
```

### 3. Build ROS2 Workspace

```bash
cd /root/DRL_Robot_Path_Planning/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

### 4. Launch Simulation

**Terminal 1: Launch Gazebo with Scout robot**
```bash
source /opt/ros/humble/setup.bash
source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash
ros2 launch agilex_scout simulate_control_gazebo.launch.py
```

**Terminal 2: Launch Navigation2**
```bash
source /opt/ros/humble/setup.bash
source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash
ros2 launch scout_nav2 nav2.launch.py
```

## Project Structure

```
DRL_Robot_Path_Planning/
├── Dockerfile                    # Docker environment configuration
├── CLAUDE.md                     # Development guidelines
├── cyclonedds_config.xml         # DDS middleware configuration
├── ros2_ws/                      # ROS2 workspace
│   ├── src/
│   │   └── scout_nav2/           # Main ROS2 package
│   │       ├── agilex_scout/     # Scout robot URDF and launch files
│   │       ├── scout_nav2/       # Navigation configuration
│   │       ├── nav2_bringup/     # Navigation2 launch files
│   │       └── aws-robomaker-small-warehouse-world/  # Simulation environment
│   ├── build/                    # Build artifacts
│   ├── install/                  # Installed packages
│   └── log/                      # Build logs
└── stable-baselines3/            # DRL algorithms (editable install)
    ├── stable_baselines3/        # Core RL implementations
    ├── docs/                     # Documentation
    ├── tests/                    # Unit tests
    └── scripts/                  # Training scripts
```

## Development Environment

### Docker Image Details

- **Base Image**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **ROS2**: Humble Hawksbill
- **Python**: 3.10 (system default)
- **PyTorch**: 2.4.1 with CUDA 11.8
- **Locale**: ko_KR.UTF-8 (Korean locale configured)

### Key Dependencies

**ROS2 Packages:**
- `ros-humble-ros-base`: Core ROS2 functionality
- `ros-humble-navigation2`: Navigation stack
- `ros-humble-slam-toolbox`: SLAM capabilities
- `ros-humble-ros-gz`: Gazebo integration
- `ignition-fortress`: Gazebo simulator

**Python Libraries:**
- `torch==2.4.1`: Deep learning framework
- `stable-baselines3`: RL algorithms (editable install from source)
- `gymnasium`: RL environment interface
- `tensorboard`: Training visualization
- `pandas`, `matplotlib`: Data analysis and plotting

**Additional Tools:**
- `swig`: Required for certain Gymnasium environments
- `libopenvdb-dev`: Spatial-temporal voxel layer support
- `libgraphicsmagick++-dev`: Image processing

## ROS2 Workspace Usage

### Additional install
```bash
apt update
apt install ros-humble-rmw-cyclonedds-cpp

apt-get update
apt-get install -y   ros-humble-openvdb-vendor   ros-humble-spatio-temporal-voxel-layer
```

### Building

**Full workspace build:**
```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Sourcing

Always source the workspace after building:
```bash
source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash
```

Or add to your `.bashrc`:
```bash
echo 'source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash' >> ~/.bashrc
```

## CycloneDDS Configuration

The project uses CycloneDDS as the DDS middleware with optimized settings for Docker:

```bash
export CYCLONEDDS_URI=file:///root/DRL_Robot_Path_Planning/cyclonedds_config.xml
```

**Configuration highlights:**
- Network interface: `lo` (loopback) for container-local communication
- Receive buffer: 10MB-256MB
- Max message size: 65500 bytes
- Optimized for low-latency communication

## Scout Robot Navigation Package

The `scout_nav2` package provides a complete navigation solution for the AgileX Scout robot using Navigation2 (Nav2) stack with Ignition Gazebo v6 (Fortress). The robot is a skid-steering mobile robot with a 3D LiDAR sensor.

### Package Components

The ROS2 workspace includes the following packages:

1. **`agilex_scout`**: URDF robot description, meshes, Gazebo simulation configuration, and launch files for both simulation and real robot control
2. **`scout_nav2`**: Navigation2 configuration files and launch files specifically tuned for the AgileX Scout robot
   - `launch/`: Main launch file for the navigation stack
   - `maps/`: Saved maps created by SLAM toolbox (pgm/yaml and data/posegraph formats)
   - `params/`: YAML configuration files for different navigation scenarios
   - `rviz/`: RViz configuration files for visualization
3. **`nav2_bringup_custom`**: Custom fork of `nav2_bringup` package with additional features including `nav2_collision_monitor` node
4. **`spatio-temporal-voxel-layer`**: Advanced costmap plugin for 3D obstacle detection
5. **`aws-robomaker-small-warehouse-world`**: Realistic warehouse simulation environment with cluttered and open spaces

### Dependencies

Install required dependencies:

```bash
# Navigation and SLAM packages
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-pointcloud-to-laserscan
sudo apt install ros-humble-slam-toolbox

# Gazebo simulation
sudo apt install ros-humble-ros-gzfortress
sudo apt install ignition-gazebo6

# Advanced costmap layer (may require significant build time/memory)
sudo apt install ros-humble-spatio-temporal-voxel-layer

# DDS middleware (for optimized communication)
sudo apt install ros-humble-rmw-cyclonedds-cpp

# Alternative: Install spatio-temporal-voxel-layer via apt (faster)
sudo apt install ros-humble-openvdb-vendor ros-humble-spatio-temporal-voxel-layer
```

### Navigation Configuration Files

The `scout_nav2/params/` folder contains carefully tuned navigation parameters for different scenarios:

| Configuration File | Localization | Robot Type | LiDAR Type | Description |
| ------------------ | ------------ | ---------- | ---------- | ----------- |
| `scout_amcl.yaml` | AMCL | Real Scout | 3D | Navigation with AMCL localization and SLAM toolbox mapping |
| `scout_slam_localization.yaml` | SLAM Toolbox | Real Scout | 3D | Navigation using SLAM toolbox for localization only |
| `sim_lidar2d_amcl.yaml` | AMCL | Simulated | 2D | Simulation with 2D LiDAR and AMCL localization |
| `sim_lidar3d_amcl.yaml` | AMCL | Simulated | 3D | Simulation with 3D LiDAR and AMCL localization |
| `sim_slam_localization.yaml` | SLAM Toolbox | Simulated | 3D | Simulation using SLAM toolbox for localization |

The appropriate configuration is automatically selected based on launch file parameters.

### Map Formats

Maps are saved in `scout_nav2/maps/` with two possible formats:

**1. AMCL-compatible maps (for `map_server`):**
- `*.pgm`: Grayscale image of the 2D occupancy grid
- `*.yaml`: Map metadata (resolution, origin, thresholds)

**2. SLAM Toolbox serialized maps (for efficient localization):**
- `*.data`: Binary serialized map data
- `*.posegraph`: Pose graph for optimization

### Launch Files & Usage

#### Simulation Environment

**Launch Gazebo simulation with Scout robot:**

```bash
# Terminal 1: Start simulation
ros2 launch agilex_scout simulate_control_gazebo.launch.py \
  lidar_type:=3d \
  rviz:=true
```

**Parameters:**
- `lidar_type`: Choose `3d` for PointCloud2 sensor or `2d` for LaserScan
- `rviz`: Set to `true` to launch RViz visualization, `false` for Gazebo only

**Launch Navigation2 stack:**

```bash
# Terminal 2: Start navigation
ros2 launch scout_nav2 nav2.launch.py \
  simulation:=true \
  slam:=true \
  localization:=amcl
```

**Parameters:**
- `simulation`: `true` for Gazebo simulation, `false` for real robot
- `slam`: `true` for SLAM mapping mode, `false` for localization with existing map
- `localization`: Choose between `amcl` (Adaptive Monte Carlo Localization) or `slam_toolbox`

#### Real Robot Usage

**Launch real Scout robot with LiDAR:**

```bash
# Terminal 1: Start robot and sensors
ros2 launch agilex_scout scout_robot_lidar.launch.py

# Terminal 2: Start navigation
ros2 launch scout_nav2 nav2.launch.py \
  simulation:=false \
  slam:=false \
  localization:=amcl
```

### Building the Navigation Stack

The navigation packages may require significant build time due to the `spatio-temporal-voxel-layer` dependency on `openvdb`:

```bash
cd /root/DRL_Robot_Path_Planning/ros2_ws
source /opt/ros/humble/setup.bash

# Standard build
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# If build consumes too much memory, limit parallel jobs
MAKEFLAGS="-j4" colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build only navigation packages
colcon build --packages-select scout_nav2 agilex_scout nav2_bringup_custom
```

### Navigation Workflow

**1. SLAM Mapping (create a new map):**

```bash
# Start simulation with 3D LiDAR
ros2 launch agilex_scout simulate_control_gazebo.launch.py lidar_type:=3d rviz:=true

# Start SLAM mode
ros2 launch scout_nav2 nav2.launch.py simulation:=true slam:=true localization:=slam_toolbox

# Drive the robot around using teleop or Nav2 goals
# Save the map when complete using SLAM Toolbox RViz plugin
```

**2. Localization & Navigation (use existing map):**

```bash
# Start simulation
ros2 launch agilex_scout simulate_control_gazebo.launch.py lidar_type:=3d rviz:=true

# Start navigation with AMCL localization
ros2 launch scout_nav2 nav2.launch.py simulation:=true slam:=false localization:=amcl

# Set initial pose estimate in RViz (2D Pose Estimate)
# Send navigation goals using Nav2 Goal button in RViz
```

### Advanced Features

- **Collision Monitoring**: Integrated `nav2_collision_monitor` for enhanced safety
- **3D Obstacle Detection**: Spatio-temporal voxel layer for dynamic 3D environments
- **Hybrid Approach**: Can combine classical Nav2 with RL-based path planning
- **Real/Sim Parity**: Consistent parameter sets for both simulation and real robot deployment

### Integration with DRL

The Scout Nav2 package provides a foundation for integrating deep reinforcement learning:

- **Sensor Data**: 3D LiDAR point clouds converted to 2D laser scans for RL observation space
- **Action Space**: Velocity commands (linear and angular) via `cmd_vel` topic
- **Reward Signals**: Can be derived from Nav2 costmaps, goal distance, and collision events
- **Hybrid Control**: Switch between Nav2 planners and RL policies based on situation

## CUDA Environment

The Docker container is pre-configured with CUDA 11.8:

```bash
CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
CUDNN_INCLUDE_DIR=/usr/local/cuda/include
CUDNN_LIB_DIR=/usr/local/cuda/lib64
```

Verify CUDA availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

## Common Issues & Troubleshooting

### rosdep Installation Conflicts

When installing dependencies, skip known conflicting keys:
```bash
rosdep install --from-paths src -yi --rosdistro humble \
  --skip-keys='libgraphicsmagick++1-dev graphicsmagick-libmagick-dev-compat'
```

### DDS Communication in Docker

If nodes cannot discover each other:
1. Ensure `--network host` is used when running the container
2. Check CycloneDDS configuration: `echo $CYCLONEDDS_URI`
3. Verify loopback interface: `ip addr show lo`

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker NVIDIA runtime
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Build Failures

**Check package dependencies:**
```bash
rosdep check --from-paths src --ignore-src
```

## Development Workflow

### 1. Develop Custom RL Environment

Create a Gymnasium-compatible environment that interfaces with ROS2:

```python
# Example: custom_env/scout_env.py
import gymnasium as gym
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class ScoutNavEnv(gym.Env):
    def __init__(self):
        super().__init__()
        rclpy.init()
        self.node = rclpy.create_node('scout_rl_env')
        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(...)
        self.action_space = gym.spaces.Box(...)

    def step(self, action):
        # Execute action in simulation
        # Return observation, reward, done, info
        pass

    def reset(self):
        # Reset simulation
        pass
```

### 2. Train RL Agent

```bash
# Inside container
python train_scout.py --algorithm PPO --timesteps 1000000
```

### 3. Evaluate in Simulation

```bash
# Launch Gazebo
ros2 launch agilex_scout simulate_control_gazebo.launch.py

# Run trained policy
python evaluate_policy.py --model-path ./models/scout_ppo.zip
```

### 4. Monitor Training

```bash
tensorboard --logdir ./logs/
```

## Environment Variables

```bash
# ROS2
source /opt/ros/humble/setup.bash
source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash

# CycloneDDS
export CYCLONEDDS_URI=file:///root/DRL_Robot_Path_Planning/cyclonedds_config.xml

# CUDA
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export CUDNN_INCLUDE_DIR=/usr/local/cuda/include
export CUDNN_LIB_DIR=/usr/local/cuda/lib64

# Custom
export DRL_AGENT_SRC_PATH=/root/DRL_Robot_Path_Planning/ros2_ws/src
```

## Contributing

This project follows standard ROS2 and Python development practices:

1. **Code Style**: Follow PEP8 for Python, ROS2 conventions for C++
2. **Testing**: Add unit tests for new features
3. **Documentation**: Update README and inline comments
4. **Git**: Use meaningful commit messages

For Stable-Baselines3 contributions, refer to their [contributing guide](https://github.com/DLR-RM/stable-baselines3/blob/master/CONTRIBUTING.md).

## Resources

### Documentation
- [ROS2 Humble Docs](https://docs.ros.org/en/humble/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Navigation2 Docs](https://navigation.ros.org/)
- [Gymnasium Docs](https://gymnasium.farama.org/)

### Tutorials
- [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [SB3 RL Tutorial](https://github.com/araffin/rl-tutorial-jnrr19)
- [Nav2 Tutorials](https://navigation.ros.org/tutorials/index.html)

### Related Projects
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo): Training framework and hyperparameter tuning
- [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib): Experimental RL algorithms
- [Stable Baselines Jax (SBX)](https://github.com/araffin/sbx): Faster Jax implementation

## License

This project combines multiple open-source components:
- **Stable-Baselines3**: MIT License
- **ROS2 Humble**: Apache 2.0 License
- **AWS Warehouse World**: MIT License

Please refer to individual package licenses for details.

## Citation

If you use this project in your research, please cite Stable-Baselines3:

```bibtex
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

## Acknowledgments

- **Stable-Baselines3 Team**: For providing robust RL implementations
- **ROS2 Community**: For the excellent robotics middleware
- **AgileX Robotics**: For the Scout robot platform
- **AWS RoboMaker**: For the warehouse simulation environment
- **Navigation2 Team**: For the advanced navigation stack

## Contact & Support

For issues related to:
- **This project**: Open an issue in this repository
- **Stable-Baselines3**: Visit [SB3 GitHub](https://github.com/DLR-RM/stable-baselines3)
- **ROS2**: Check [ROS Answers](https://answers.ros.org/)
- **General RL questions**: [RL Discord](https://discord.com/invite/xhfNqQv), [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)

---

**Status**: Development in Progress

**Last Updated**: 2025-11-17
