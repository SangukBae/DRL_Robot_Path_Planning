# DRL Robot Path Planning

ROS2 Humble + Gazebo Ignition Fortress 기반 AgileX Hunter SE DRL 경로 계획 시뮬레이션 워크스페이스.

## Overview

AgileX Hunter SE Ackermann 조향 로봇이 장애물 환경(DRL Arena, Hospital 등)에서 커리큘럼 강화학습(TQC)으로 자율 주행을 학습한다.

정책은 2D Waypoint 명령(거리·방향각)을 출력하고, Pure Pursuit 로컬 컨트롤러가 이를 `cmd_vel`로 변환한다. 커리큘럼은 빈 환경 → 정적 장애물 → 동적 장애물 → 사람 포함 복합 환경 순서로 5단계에 걸쳐 자동 진급한다.

**Tech Stack**: ROS2 Humble, Gazebo Ignition Fortress, PyTorch 2.4.1 (CUDA 11.8)

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

### 2. DRL 학습 (커리큘럼 — 권장)

터미널을 3개 사용한다.

```bash
# [터미널 1] Gazebo 시뮬레이션
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# [터미널 2] 커리큘럼 환경 노드
ros2 run drl_agent environment_curriculum.py \
  --ros-args -p config_file:=$(ros2 pkg prefix drl_agent --share)/config/environment_curriculum.yaml

# [터미널 3] TQC 커리큘럼 학습
ros2 run drl_agent train_tqc_curriculum_agent.py
```

커리큘럼 진급 상황과 에피소드 보상은 아래 파일에 기록된다:
- `<run_dir>/logs/curriculum_episode_rewards_<run>.csv` — 에피소드별 스테이지·보상·성공 여부
- `<run_dir>/logs/curriculum_state.json` — 현재 스테이지·글로벌 스텝 (학습 재개용)

재개 시 `load_model: true`를 `train_tqc_config.yaml`에 설정하면 모델·리플레이 버퍼·커리큘럼 상태가 모두 복원된다.

```bash
# TensorBoard 모니터링 (run_dir 기본값: runtime/tqc_state_80_nstactics_5_obstacle_11)
tensorboard --logdir <run_dir>/logs
```

### 3. DRL 학습 (단일 알고리즘 — 대안)

```bash
# [터미널 2] 표준 환경 노드
ros2 launch drl_agent drl_hunter_se.launch.py

# [터미널 3] 알고리즘 선택
ros2 run drl_agent train_tqc_agent.py      # TQC
ros2 run drl_agent train_tqc_ieqn_agent.py # TQC + IEQn
ros2 run drl_agent train_td7_agent.py      # TD7
ros2 run drl_agent train_sac_agent.py      # SAC
ros2 run drl_agent train_a3c_agent.py      # A3C
```

### 4. 테스트

```bash
ros2 launch drl_agent test_tqc.launch.py
ros2 launch drl_agent test_td7.launch.py
```

## Architecture

### 서비스 기반 환경 인터페이스

```
Environment Node (environment_curriculum.py)   Agent Node (train_tqc_curriculum_agent.py)
├── /reset               ←─────────────────── 에피소드 시작 시 초기화
├── /step                ←─────────────────── 액션(Waypoint) 전달 + 상태/보상 반환
├── /get_dimensions      ←─────────────────── state_dim, action_dim 조회
├── /seed                ←─────────────────── 랜덤 시드 설정
└── /action_space_sample ←─────────────────── 랜덤 액션 샘플 (워밍업)
```

서비스 정의: `drl_agent_interfaces/srv/`

### DRL 상태/액션 공간

- **State (87D)**:
  - `[0:80]` — LiDAR 80 빈 (전방 180°, obs_state), 빈당 최근접 장애물 거리 [m]
  - `[80]` — 목표까지 거리 [m]
  - `[81]` — 목표 방향 오차 θ [rad]
  - `[82]` — 이전 액션 r (웨이포인트 거리), 정규화
  - `[83]` — 이전 액션 θ (웨이포인트 각도), 정규화
  - `[84]` — 실제 선속도 [m/s] (오도메트리)
  - `[85]` — 실제 요레이트 [rad/s] (오도메트리)
  - `[86]` — 중심 조향각 [rad] (조인트 스테이트)

- **Action (2D) — Waypoint 명령 (Pure Pursuit)**:
  - `action[0]`: 웨이포인트 거리 r ∈ [0.8, 2.0] m (전진, 로봇 프레임)
  - `action[1]`: 웨이포인트 각도 θ ∈ [-0.524, 0.524] rad (±30°, 로봇 프레임)
  - 정책 출력 `[-1, 1]`을 `environment.py`가 물리 단위로 스케일 변환 후 Pure Pursuit 컨트롤러 구동 → `cmd_vel`

- **토픽**: `/cmd_vel` (웨이포인트→트위스트), `/cmd_vel_filtered` (프리필터 출력), `/odometry`, `/ouster/points`, `/scan`

### 커맨드 파이프라인

```
RL policy → /cmd_vel (twist) → hunter_se_cmd_prefilter → /cmd_vel_filtered → Gazebo
                                  (스로틀/조향 셰이핑, 50 Hz)
```

### 환경 구현

| 파일 | 용도 | 리셋 방식 |
|------|------|---------|
| `environment.py` | Ignition Fortress 기본 학습 | `ros_gz_interfaces/SetEntityPose` |
| `environment_curriculum.py` | 커리큘럼 학습 (권장) | 동일 |

### 커리큘럼 학습

5단계 자동 진급 구조. `environment_curriculum.py`가 `/gym_node/set_parameters` 서비스로 스테이지를 수신하고, `train_tqc_curriculum_agent.py`가 평가 결과를 보고 진급 여부를 결정한다.

| 스테이지 | 이름 | 정적 | 동적 | 사람 |
|---------|------|------|------|------|
| 0 | empty | 0 | 0 | 0 |
| 1 | static_only | 3 | 0 | 0 |
| 2 | slow_dynamic | 2 | 3 | 1 |
| 3 | mixed_medium | 2 | 4 | 4 |
| 4 | full_complexity | 3 | 6 | 5 |

진급 조건: `pass_eval_success_rate` / `pass_eval_collision_rate` 임계값을 `consecutive_eval_passes`회 연속 통과.

### 알고리즘 구현

`drl_agent/scripts/policy/`:
- `train_tqc_curriculum_agent.py` — TQC + 5단계 커리큘럼 (주력)
- `tqc_agent.py` — Truncated Quantile Critics (LAP 리플레이 버퍼)
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
  → pointcloud_to_laserscan (height filter: z ∈ [-0.455, 0.250] m, 360°, 0.176°/bin)
  → /scan (ROS2 LaserScan)
  → environment.py (80 빈 LiDAR 관측)
```

## 주요 설정 파일

`drl_agent/config/`:

| 파일 | 설명 |
|------|------|
| `environment.yaml` | 상태/액션 차원, 충돌 임계값, 안전 영역 |
| `environment_curriculum.yaml` | 커리큘럼 환경 설정 (풀 크기, 5단계 정의) |
| `train_tqc_curriculum_config.yaml` | 커리큘럼 진급 규칙 (임계값, 연속 통과 횟수) |
| `hyperparameters_*.yaml` | 알고리즘별 네트워크 구조, 학습률 |
| `train_*_config.yaml` | 학습 파라미터 (최대 스텝, 워밍업, 평가 주기) |
| `test_tqc_config.yaml` | 테스트 파라미터 (시작/목표 쌍, 에피소드 수) |

주요 학습 파라미터:
- 최대 타임스텝: 1,000,000
- 워밍업 스텝: 6,000 (5 Hz 기준)
- 평가 주기: 6,000 스텝 (5 Hz 기준)
- 커리큘럼 스테이지 최소 스텝: 15,000
- 목표 도달 임계값: 0.42 m
- 충돌 임계값: 직사각형 안전 영역 (전방 0.476 m, 후방 0.410 m, 좌우 0.322 m)

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

## Package Index

| 패키지 | 경로 | 설명 |
|--------|------|------|
| `drl_agent` | `src/drl_agent/` | DRL 환경, 정책, 학습/테스트 스크립트 |
| `drl_agent_interfaces` | `src/drl_agent_interfaces/` | ROS2 서비스/액션/메시지 정의 |
| `hunter_se_gazebo` | `src/hunter_se_gazebo/` | Hunter SE URDF, Gazebo launch, worlds |
| `hunter_se_unity` | `src/hunter_se_unity/` | Unity 시뮬레이션 에셋 |
| `ouster_description` | `src/ouster_simulation/ouster_description/` | OS1-64 LiDAR (RGL 플러그인) |
| `aws-robomaker-hospital-world` | `src/aws-robomaker-hospital-world/` | Hospital 시뮬레이션 환경 |
| `aws-robomaker-bookstore-world` | `src/aws-robomaker-bookstore-world/` | Bookstore 시뮬레이션 환경 |
| `aws-robomaker-small-house-world` | `src/aws-robomaker-small-house-world/` | Small House 시뮬레이션 환경 |
| `aws-robomaker-small-warehouse-world` | `src/aws-robomaker-small-warehouse-world/` | Small Warehouse 시뮬레이션 환경 |
