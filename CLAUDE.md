# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Reinforcement Learning (DRL) based robot path planning project using ROS2 Humble with Gazebo simulation. The project uses Docker for containerized development with CUDA 11.8 support for GPU-accelerated training.

## Development Environment

### Docker Setup

The project runs in a Docker container based on `nvidia/cuda:11.8.0-devel-ubuntu22.04` with ROS2 Humble.

**Build the Docker image:**
```bash
docker build -t drl_path_planning .
```

**Run the container with GPU support:**
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/root/drl_path_final \
  --network host \
  drl_path_planning
```

### Key Dependencies

- **ROS2:** Humble (Ubuntu 22.04 Jammy)
- **Python:** 3.10 (system default)
- **PyTorch:** 2.4.1 with CUDA 11.8
  - Installation: `torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118`
- **Additional ML libraries:** spconv-cu118, torch-scatter
- **Simulator:** Gazebo (via ros-humble-gazebo-*)
- **Build system:** colcon

## ROS2 Workspace

The ROS2 workspace is located at `ros2_ws/`.

### Building the Workspace

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build
```

### After Building

Source the workspace:
```bash
source ros2_ws/install/setup.bash
```

### Building a Single Package

```bash
colcon build --packages-select <package_name>
```

### Clean Build

```bash
rm -rf build/ install/ log/
colcon build
```

## CycloneDDS Configuration

The project uses CycloneDDS as the DDS middleware with custom configuration in `cyclonedds_config.xml`:
- Network interface: loopback (lo) - configured for Docker containers
- Receive buffer: 10MB-256MB
- Max message size: 65500B

To use this configuration:
```bash
export CYCLONEDDS_URI=file://$(pwd)/cyclonedds_config.xml
```

## Architecture Notes

### Directory Structure

```
DRL_Robot_Path_Planning/
├── Dockerfile              # Container definition with CUDA + ROS2 Humble
├── cyclonedds_config.xml   # DDS middleware configuration
└── ros2_ws/                # ROS2 workspace
    └── src/                # ROS2 packages go here
```

### Expected Package Types

Based on the project name and setup, this workspace will likely contain:
- **DRL training nodes:** Python nodes implementing reinforcement learning algorithms
- **Robot simulation packages:** URDF models, Gazebo worlds
- **Path planning nodes:** Nodes for executing learned policies
- **Sensor processing:** LiDAR, camera, or other perception packages

### CUDA Environment Variables

When working with GPU-accelerated training:
```bash
CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
CUDNN_INCLUDE_DIR=/usr/local/cuda/include
CUDNN_LIB_DIR=/usr/local/cuda/lib64
```

## Common Issues

### GraphicsMagick Dependencies
On Ubuntu 22.04 (Jammy), the package name is `libgraphicsmagick++-dev` (not `libgraphicsmagick++1-dev`). The Dockerfile already handles this.

### rosdep Installation
When running `rosdep install`, skip conflicting keys:
```bash
rosdep install --from-paths src -yi --rosdistro humble \
  --skip-keys='libgraphicsmagick++1-dev graphicsmagick-libmagick-dev-compat'
```

### DDS Communication in Docker
The CycloneDDS configuration uses loopback interface for container-local communication. Adjust `cyclonedds_config.xml` if inter-container or host-container communication is needed.
