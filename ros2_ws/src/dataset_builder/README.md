# Dataset Builder

Dataset builder package for continuous robot data collection with ROS2 Humble and Ignition Gazebo Fortress.

## Overview

This package provides tools for collecting continuous driving data from a robot in simulation. It automatically manages:

- Run metadata (run_meta.yaml)
- Segment-level sidecar files (sidecar.yaml)
- Rosbag2 recording of specified topics
- Pose tracking via odometry or TF

## Features

- **Continuous recording**: No episode concept, records data continuously
- **Automatic segmentation**: Splits data into time-based segments
- **Metadata management**: Automatically creates run and segment metadata files
- **Flexible pose tracking**: Supports both /odom topic and TF-based pose tracking
- **Configurable topics**: Easy YAML-based topic configuration

## Prerequisites

```bash
# ROS2 Humble
# Python 3.8+
# Dependencies: rclpy, tf2_ros, nav_msgs, sensor_msgs, geometry_msgs, pyyaml
```

## Installation

### 1. Install dependencies

```bash
cd /root/drl_path_final/ros2_ws
rosdep install --from-paths src/dataset_builder --ignore-src -r -y
```

### 2. Build the package

```bash
cd /root/drl_path_final/ros2_ws
colcon build --packages-select dataset_builder
```

### 3. Source the workspace

```bash
source /root/drl_path_final/ros2_ws/install/setup.bash
```

## Usage

### Basic Usage

Launch data recording with default parameters:

```bash
ros2 launch dataset_builder record_run.launch.py
```

### With Custom Parameters

```bash
ros2 launch dataset_builder record_run.launch.py \
  dataset_root:=/path/to/data \
  world_name:=my_world \
  segment_duration_sec:=300 \
  notes:="Test run with new obstacles"
```

### Available Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `dataset_root` | `/root/drl_path_final/ros2_ws/data` | Root directory for dataset |
| `run_id` | `auto` | Run identifier (auto generates timestamp) |
| `use_sim_time` | `true` | Use simulation time |
| `segment_duration_sec` | `600` | Segment duration in seconds |
| `world_name` | `unknown_world` | Name of Ignition Gazebo world |
| `notes` | `` | Additional notes for the run |
| `pose_source` | `odom` | Pose source: `odom` or `tf` |
| `base_frame` | `base_link` | Robot base frame |

## Directory Structure

After running, the following structure is created:

```
/root/drl_path_final/ros2_ws/data/
└── runs/
    └── run_2024-01-09_14-30-45/
        ├── run_meta.yaml          # Run-level metadata
        ├── topics.yaml            # Copy of recorded topics config
        └── segments/
            ├── rosbag/            # Rosbag2 files
            │   ├── metadata.yaml
            │   └── rosbag_*.db3
            ├── seg_0001/
            │   └── sidecar.yaml   # Segment metadata
            ├── seg_0002/
            │   └── sidecar.yaml
            └── ...
```

## Configuration Files

### topics.yaml

Located at `dataset_builder/configs/topics.yaml`, defines which topics to record:

```yaml
topics:
  # 3D LiDAR point cloud (Ouster OS1-64 via RGL)
  - /points
  # RGBD Camera point cloud
  - /depth_camera/points
  # RGBD Camera images
  - /depth_camera/image
  - /depth_camera/depth_image
  # Odometry (bridged from /scout/odometry)
  - /odometry
  # Transforms
  - /tf
  - /tf_static
  # Simulation clock
  - /clock
  # IMU data
  - /imu
  # Velocity commands
  - /cmd_vel
```

Add or remove topics as needed for your dataset. Topic names must match the `ros2_gz_bridge_config.yaml` configuration.

### run_defaults.yaml

Located at `dataset_builder/configs/run_defaults.yaml`, provides default configuration:

```yaml
dataset_root: /root/drl_path_final/ros2_ws/data
use_sim_time: true
segment_duration_sec: 600
split_policy: "time_based_600s"
fixed_frame: odom
base_frame: base_link
pose_source: odom
```

## Metadata Files

### run_meta.yaml

Contains run-level metadata:

```yaml
run_id: run_2024-01-09_14-30-45
created_at_iso: '2024-01-09T14:30:45.123456+09:00'
dataset_root: /root/drl_path_final/ros2_ws/data
ros_distro: humble
world_name: my_world
notes: Test run
record_topics:
  - /points
  - /depth_camera/points
  - /depth_camera/image
  - /depth_camera/depth_image
  - /odometry
  - /tf
  - /tf_static
  - /clock
  - /imu
  - /cmd_vel
split_policy: time_based_600s
use_sim_time: true
git:
  commit: null
  branch: null
  dirty: null
```

### sidecar.yaml

Contains segment-level metadata:

```yaml
run_id: run_2024-01-09_14-30-45
segment_id: seg_0001
start_time_ros: 1234567890123456789
end_time_ros: 1234567890723456789
pose_at_start:
  position:
    x: 0.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
  frame_id: odom
  child_frame_id: base_link
notes: ''
```

## Nodes

### run_manager

Creates run directory and metadata files.

**Parameters:**
- `dataset_root`: Dataset root directory
- `use_sim_time`: Use simulation time
- `run_id`: Run identifier (empty = auto-generate)
- `world_name`: World name
- `notes`: Run notes

### metadata_logger

Creates segment directories and sidecar files.

**Parameters:**
- `dataset_root`: Dataset root directory
- `run_id`: Run identifier (required)
- `segment_duration_sec`: Segment duration
- `pose_source`: Pose source (`odom` or `tf`)
- `base_frame`: Robot base frame
- `fixed_frame`: Fixed frame (default: `odom`)

## Stopping Recording

Press `Ctrl+C` to stop all nodes and rosbag recording. The last segment will be properly closed with end time.

## Troubleshooting

### No topics being recorded

Check that topics exist:
```bash
ros2 topic list
```

Verify topics.yaml contains correct topic names.

### Permission denied errors

Ensure dataset_root directory has write permissions:
```bash
sudo chown -R $USER:$USER /root/drl_path_final/ros2_ws/data
```

### Time synchronization issues

If using simulation, ensure `use_sim_time:=true` and that `/clock` topic is being published by Ignition Gazebo via the ros_gz_bridge.

## License

MIT
