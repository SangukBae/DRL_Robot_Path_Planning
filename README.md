# DRL Robot Path Planning

ROS2 Humble + Gazebo Ignition Fortress 기반 AgileX Hunter SE DRL 경로 계획 시뮬레이션 워크스페이스.

## Overview

AgileX Hunter SE Ackermann 조향 로봇이 장애물 환경(DRL Arena, Hospital 등)에서 DRL 알고리즘(TQC, TD7, SAC, A3C)으로 자율 주행을 학습한다.

**Tech Stack**: ROS2 Humble, Gazebo Ignition Fortress, PyTorch 2.4.1 (CUDA 11.8), Navigation2

## Layout

```
/
├── ros2_ws/src/
│   ├── drl_agent/                 # DRL 환경/정책/학습 스크립트
│   ├── drl_agent_interfaces/      # ROS2 srv/action 정의
│   ├── hunter_se_gazebo/          # Hunter SE URDF, Gazebo launch, worlds
│   ├── hunter_se_unity/           # Unity 시뮬레이션 에셋 (Hunter SE.fbx 등)
│   ├── aws-robomaker-hospital-world/
│   ├── aws-robomaker-bookstore-world/
│   ├── aws-robomaker-small-house-world/
│   ├── aws-robomaker-small-warehouse-world/
│   └── ouster_simulation/ouster_description/  # OS1-64 LiDAR (RGL)
├── Dockerfile
└── cyclonedds_config.xml
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

### 1. Gazebo 시뮬레이션 (Hunter SE)

```bash
# DRL Arena (기본, 15×15m 밀폐 환경)
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py

# Hospital world
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py world:=hospital

# RViz 비활성화 (학습 시 권장)
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# 키보드 teleop (빈 환경에서 검증용)
ros2 launch hunter_se_gazebo hunter_se_validation_empty.launch.py
```

> **RGL LiDAR**: `~/DRL_Robot_Path_Planning/third_party/rgl/RGLGazeboPlugin/install` 에 설치 필요.

### 2. AWS World 단독 실행

```bash
# Hospital World
ros2 launch aws_robomaker_hospital_world hospital_ignition.launch.py

# Bookstore World
ros2 launch aws_robomaker_bookstore_world bookstore_ignition.launch.py

# Small House World
ros2 launch aws_robomaker_small_house_world small_house_ignition.launch.py

# Small Warehouse World
ros2 launch aws_robomaker_small_warehouse_world small_warehouse_ignition.launch.py
```

### 3. DRL 학습/테스트

터미널을 3개 사용한다.

```bash
# [터미널 1] Gazebo 시뮬레이션 실행
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py

# [터미널 2] DRL 환경 노드 실행
ros2 launch drl_agent drl_hunter_se.launch.py            # 학습
ros2 launch drl_agent drl_hunter_se.launch.py mode:=test  # 테스트

# [터미널 3] 에이전트 실행
ros2 run drl_agent train_tqc_agent.py      # TQC (주력)
ros2 run drl_agent train_tqc_ieqn_agent.py # TQC + IEQn
ros2 run drl_agent train_td7_agent.py      # TD7
ros2 run drl_agent train_sac_agent.py      # SAC
ros2 run drl_agent train_a3c_agent.py      # A3C

# TensorBoard 모니터링
tensorboard --logdir ros2_ws/src/drl_agent/results/
```

테스트만 할 경우:

```bash
ros2 launch drl_agent test_tqc.launch.py
ros2 launch drl_agent test_td7.launch.py
```

## Architecture

### 서비스 기반 환경 인터페이스

```
Environment Node (environment.py)          Agent Node (train_*_agent.py)
├── /reset           ←──────────────────── 에피소드 시작 시 초기화
├── /step            ←──────────────────── 액션 전달 + 상태/보상 반환
├── /get_dimensions  ←──────────────────── state_dim, action_dim 조회
├── /seed            ←──────────────────── 랜덤 시드 설정
└── /action_space_sample ←──────────────── 랜덤 액션 샘플 (워밍업)
```

서비스 정의: `drl_agent_interfaces/srv/`

### DRL 상태/액션 공간

- **State (80D)**: 에이전트 포즈 (4D) + LiDAR 관측 (76D)
- **Action (2D)**: 선속도 [0.0, 1.333] m/s (전진 전용), 각속도 [-1.4, 1.4] rad/s
- **토픽**: `/cmd_vel` (명령), `/odometry` (상태), `/ouster/points` (3D LiDAR → `/scan` 변환)

**환경 인터페이스**: `environment_interface.py`가 에이전트 출력 `[-1, 1]`을 `[0, 1]`로 리매핑 후 `/step` 서비스 호출. 환경은 `[0, 1.333]` m/s로 스케일 변환.

### 두 가지 환경 구현

| 파일 | 시뮬레이터 | 리셋 방식 |
|------|-----------|---------|
| `environment.py` | Ignition Fortress | `ros_gz_interfaces/SetEntityPose` |
| `environment_360.py` | Classic Gazebo | `gazebo_msgs/SetEntityState` |

### 알고리즘 구현

`drl_agent/scripts/policy/`:
- `tqc_agent.py` — Truncated Quantile Critics (주력, LAP 리플레이 버퍼)
- `tqc_ieqn_agent.py` — TQC + IEQn 부등식 제약
- `td7_agent.py` — Twin Delayed DDPG v7
- `sac_agent.py` — Soft Actor-Critic
- `a3c_agent.py` — Asynchronous Advantage Actor-Critic

### LiDAR 파이프라인 (Hunter SE)

```
Gazebo Ouster RGL
  → /hunter_se/pointcloud/points (Gz IgnitionTopic)
  → ros_gz_bridge
  → /ouster/points (ROS2 PointCloud2)
  → /scan (ROS2 LaserScan, 360°)
  → environment.py (76D 관측)
```

## 주요 설정 파일

`drl_agent/config/`:

| 파일 | 설명 |
|------|------|
| `environment.yaml` | 상태/액션 차원, 충돌 임계값, 안전 영역 |
| `hyperparameters_*.yaml` | 알고리즘별 네트워크 구조, 학습률 |
| `train_*_config.yaml` | 학습 파라미터 (최대 스텝, 워밍업, 평가 주기) |
| `test_tqc_config.yaml` | 테스트 파라미터 (시작/목표 쌍, 에피소드 수) |

주요 학습 파라미터:
- 최대 타임스텝: 1,000,000
- 워밍업 스텝: 25,000
- 평가 주기: 5,000 스텝
- 목표 도달 임계값: 0.42 m
- 충돌 임계값: 전후 0.44 m / 좌우 0.352 m (직사각형 안전 영역)

## 환경 변수

```bash
source /opt/ros/humble/setup.bash
source <repo>/ros2_ws/install/setup.bash

# DDS (Docker 환경)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://<repo>/cyclonedds_config.xml
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

## 디버깅

```bash
# 토픽 주기 확인
ros2 topic hz /scan               # ~10 Hz
ros2 topic hz /cmd_vel_filtered   # ~50 Hz
ros2 topic hz /odometry           # ~50 Hz

# TF 트리 확인
ros2 run tf2_tools view_frames

# 환경 서비스 확인
ros2 service list | grep -E "reset|step|dimensions"

# GPU/CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

## Troubleshooting

| 문제 | 해결 |
|------|------|
| 서비스 타임아웃 | Gazebo와 DRL 환경 노드가 모두 실행 중인지 확인 |
| `/odometry` 없음 | ros_gz_bridge 실행 여부, bridge config 확인 |
| 시뮬레이션 RTF 저하 | RGL GPU 플러그인 경로 확인 (`IGN_GAZEBO_SYSTEM_PLUGIN_PATH`) |
| rosdep 충돌 | `rosdep install --from-paths src -yi --rosdistro humble --skip-keys='libgraphicsmagick++1-dev graphicsmagick-libmagick-dev-compat'` |
| Docker DDS 통신 | `--network host` + `RMW_IMPLEMENTATION` + `CYCLONEDDS_URI` 설정 |

## Package Index

| 패키지 | 경로 | 설명 |
|--------|------|------|
| `drl_agent` | `src/drl_agent/` | DRL 환경, 정책, 학습/테스트 스크립트 |
| `drl_agent_interfaces` | `src/drl_agent_interfaces/` | ROS2 서비스/액션 정의 |
| `hunter_se_gazebo` | `src/hunter_se_gazebo/` | Hunter SE URDF, Gazebo launch, worlds |
| `hunter_se_unity` | `src/hunter_se_unity/` | Unity 시뮬레이션 에셋 |
| `ouster_description` | `src/ouster_simulation/ouster_description/` | OS1-64 LiDAR (RGL 플러그인) |
| `aws-robomaker-hospital-world` | `src/aws-robomaker-hospital-world/` | Hospital 시뮬레이션 환경 |
| `aws-robomaker-bookstore-world` | `src/aws-robomaker-bookstore-world/` | Bookstore 시뮬레이션 환경 |
| `aws-robomaker-small-house-world` | `src/aws-robomaker-small-house-world/` | Small House 시뮬레이션 환경 |
| `aws-robomaker-small-warehouse-world` | `src/aws-robomaker-small-warehouse-world/` | Small Warehouse 시뮬레이션 환경 |
