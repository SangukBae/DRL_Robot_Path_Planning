# DRL Robot Path Planning

ROS2 Humble + Gazebo Ignition Fortress 기반 Scout 로봇 DRL 경로 계획 시뮬레이션 워크스페이스.

## Overview

Scout 로봇이 장애물 환경(AWS Warehouse, Hospital 등)에서 DRL 알고리즘(TQC, TD7, SAC, A3C)으로 자율 주행을 학습한다.

**Tech Stack**: ROS2 Humble, Gazebo Fortress, PyTorch 2.4.1 (CUDA 11.8), Navigation2

## Layout

```
/
├── ros2_ws/src/
│   ├── drl_agent/                 # DRL 환경/정책/학습 스크립트
│   ├── drl_agent_interfaces/      # ROS2 srv/action 정의
│   ├── scout_nav2/
│   │   ├── agilex_scout/          # Scout URDF, 메시, Gazebo launch
│   │   ├── scout_nav2/            # Nav2 config, maps, params
│   │   ├── nav2_bringup_custom/   # Nav2 bringup (collision monitor 포함)
│   │   └── aws-robomaker-small-warehouse-world/
│   ├── aws-robomaker-hospital-world/
│   ├── ouster_simulation/ouster_description/  # OS1-64 LiDAR (RGL)
│   ├── depth_d435/                # Intel RealSense D435
│   └── depth_d455/                # Intel RealSense D455
├── Dockerfile
├── cyclonedds_config.xml
└── CLAUDE.md
```

## Build

```bash
cd <repo>/ros2_ws
source /opt/ros/humble/setup.bash

# 전체 빌드
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# 소스
source install/setup.bash
```

## Quick Start

### 1. Gazebo 시뮬레이션

```bash
# Hospital world (RGL LiDAR)
ros2 launch agilex_scout simulate_control_gazebo_ignition.launch.py rviz:=true

# Warehouse world
ros2 launch agilex_scout simulate_control_gazebo.launch.py lidar_type:=3d rviz:=true
```

> **RGL LiDAR 사용 시**: world 파일에 `RGLServerPluginManager` 플러그인 필요, `RGL_PATTERNS_DIR` 환경변수 설정 필요.

### 2. Navigation2

```bash
# AMCL 로컬라이제이션 (기존 맵)
ros2 launch scout_nav2 nav2.launch.py simulation:=true slam:=false localization:=amcl

# SLAM 매핑
ros2 launch scout_nav2 nav2.launch.py simulation:=true slam:=true localization:=slam_toolbox

# LIO-SAM (Ignition Gazebo)
ros2 launch lio_sam run_scout_ignition.launch.py
```

Nav2 파라미터 상세: `ros2_ws/src/scout_nav2/scout_nav2/params/` 참고

### 3. DRL 학습/테스트

```bash
# 1) Gazebo 먼저 실행 (별도 터미널)
ros2 launch agilex_scout simulate_control_gazebo_ignition.launch.py rviz:=true

# 2) 학습
ros2 run drl_agent train_tqc_agent.py

# 3) 테스트
ros2 launch drl_agent test_tqc.launch.py
```

DRL 알고리즘 상세: `ros2_ws/src/drl_agent/` 참고

### 4. 데이터셋 수집 (Offline RL)

```bash
# 1) Gazebo 시뮬레이션 실행
ros2 launch agilex_scout simulate_control_gazebo_ignition.launch.py rviz:=true

# 2) 데이터셋 녹화 시작 (별도 터미널)
ros2 launch dataset_builder record_run.launch.py run_id:=test_run_01

# 커스텀 파라미터로 실행
ros2 launch dataset_builder record_run.launch.py \
  dataset_root:=/path/to/data \
  run_id:=my_run_01 \
  world_name:=aws_hospital \
  segment_duration_sec:=300 \
  notes:="Test run with obstacles"
```

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `dataset_root` | `ros2_ws/data` | 데이터 저장 루트 경로 |
| `run_id` | `auto` | 실행 ID (auto 시 타임스탬프 자동 생성) |
| `segment_duration_sec` | `600` | 세그먼트 분할 간격 (초) |
| `world_name` | `unknown_world` | Gazebo 월드 이름 |
| `use_sim_time` | `true` | 시뮬레이션 시간 사용 여부 |
| `notes` | `` | 실행 메모 |

녹화 중지: `Ctrl+C` (마지막 세그먼트 자동 저장)

## Environment Variables

```bash
# ROS2
source /opt/ros/humble/setup.bash
source <repo>/ros2_ws/install/setup.bash

# DDS (Docker 환경)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://<repo>/cyclonedds_config.xml

# RGL LiDAR
export RGL_PATTERNS_DIR=<path_to_rgl_patterns>
```

## Docker

```bash
# 이미지 빌드
docker build -t drl_path_planning .

# 컨테이너 실행 (GPU)
docker run --gpus all -it --rm \
  --network host \
  -e DISPLAY=$DISPLAY \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/root/DRL_Robot_Path_Planning \
  drl_path_planning

# X11 허용 (호스트)
xhost +local:
```

## Troubleshooting

| 문제 | 해결 |
|------|------|
| rosdep 충돌 | `rosdep install --from-paths src -yi --rosdistro humble --skip-keys='libgraphicsmagick++1-dev graphicsmagick-libmagick-dev-compat'` |
| Docker DDS 통신 | `--network host` + `RMW_IMPLEMENTATION` + `CYCLONEDDS_URI` 설정 |
| CUDA 확인 | `python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"` |

## Package Index

| 패키지 | 경로 | 설명 |
|--------|------|------|
| `drl_agent` | `ros2_ws/src/drl_agent/` | DRL 환경, 정책, 학습/테스트 스크립트 |
| `drl_agent_interfaces` | `ros2_ws/src/drl_agent_interfaces/` | ROS2 서비스/액션 정의 |
| `agilex_scout` | `ros2_ws/src/scout_nav2/agilex_scout/` | Scout URDF, Gazebo launch |
| `scout_nav2` | `ros2_ws/src/scout_nav2/scout_nav2/` | Nav2 설정, 맵, 파라미터 |
| `ouster_description` | `ros2_ws/src/ouster_simulation/ouster_description/` | OS1-64 LiDAR (RGL 플러그인) |
| `depth_d435` | `ros2_ws/src/depth_d435/` | Intel RealSense D435 |
| `depth_d455` | `ros2_ws/src/depth_d455/` | Intel RealSense D455 |
| `aws-robomaker-small-warehouse-world` | `ros2_ws/src/scout_nav2/aws-robomaker-small-warehouse-world/` | Warehouse 시뮬레이션 환경 |
| `aws-robomaker-hospital-world` | `ros2_ws/src/aws-robomaker-hospital-world/` | Hospital 시뮬레이션 환경 |
| `dataset_builder` | `ros2_ws/src/dataset_builder/` | Offline RL 데이터셋 수집 및 메타데이터 로깅 |

각 패키지 상세 사용법은 해당 패키지 디렉토리의 README 참고.
