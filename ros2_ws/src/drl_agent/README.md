# drl_agent

## Overview

Scout 로봇 자율 주행을 위한 DRL(Deep Reinforcement Learning) 에이전트 패키지.
TQC, TD7, SAC, A3C 알고리즘 구현체와 학습/테스트 스크립트를 포함한다.

환경 노드가 ROS2 서비스로 상태/보상을 제공하고, 에이전트 노드가 클라이언트로 동작하는 구조다.

## Quick Start

```bash
# 1) Gazebo 시뮬레이션 먼저 실행 (별도 터미널)
ros2 launch agilex_scout simulate_control_gazebo_ignition.launch.py rviz:=true

# 2) 환경 노드 단독 실행
ros2 run drl_agent environment.py --ros-args -p environment_mode:=train

# 3) TQC 학습
ros2 run drl_agent train_tqc_agent.py

# 4) TD7 학습
ros2 run drl_agent train_td7_agent.py

# 5) TQC 테스트 (launch 파일)
ros2 launch drl_agent test_tqc.launch.py

# 6) TD7 테스트 (launch 파일)
ros2 launch drl_agent test_td7.launch.py
```

## Interfaces

### Services (제공)

| 서비스명 | 타입 | 설명 |
|---------|------|------|
| `/reset` | `Reset.srv` | 에피소드 초기화, 초기 상태 반환 |
| `/step` | `Step.srv` | 액션 실행 → (상태, 보상, done, target) 반환 |
| `/get_dimensions` | `GetDimensions.srv` | state_dim, action_dim, max_action 반환 |
| `/seed` | `Seed.srv` | 랜덤 시드 설정 |
| `/action_space_sample` | `SampleActionSpace.srv` | 랜덤 액션 샘플링 (warmup용) |
| `/get_start_goal_pairs` | `GetStartGoalPairs.srv` | 시작/목표 좌표 반환 |

### Topics (구독/발행)

| 토픽명 | 방향 | 타입 | 설명 |
|--------|------|------|------|
| `/odometry` | 구독 | `Odometry` | 로봇 위치/자세 |
| `/laser_scan` | 구독 | `LaserScan` | 2D LiDAR 스캔 |
| `/points` | 구독 | `PointCloud2` | 3D 포인트클라우드 |
| `/cmd_vel` | 발행 | `Twist` | 로봇 속도 명령 |

### State/Action Space

- **State (80D)**: 에이전트 포즈(4D) + LiDAR 관측(76D)
- **Action (2D)**: 선속도 [-1.0, 1.0] m/s, 각속도 [-1.4, 1.4] rad/s

## Configuration

| 파일 | 역할 |
|------|------|
| `config/environment.yaml` | 환경 파라미터 (state_dim, collision_threshold 등) |
| `config/hyperparameters_tqc.yaml` | TQC 하이퍼파라미터 (batch_size, buffer_size 등) |
| `config/hyperparameters_td7.yaml` | TD7 하이퍼파라미터 |
| `config/train_tqc_config.yaml` | TQC 학습 설정 (max_timesteps, warmup 등) |
| `config/test_tqc_config.yaml` | TQC 테스트 설정 (시작/목표 쌍) |

**주요 파라미터:**
- `goal_threshold`: 0.42m (목표 도달 판정)
- `collision_threshold`: 0.7m (충돌 판정, 8존 기반)
- `warmup_steps`: 25,000 (랜덤 액션 구간)
- `max_timesteps`: 1,000,000

## Dependencies / Assumptions

### 의존성

- `rclpy`, `drl_agent_interfaces`
- `python3-tensorboard-pip`, `python3-squaternion-pip`
- PyTorch 2.4.1+ (CUDA 11.8)

### 전제조건

- Gazebo Ignition 시뮬레이션이 먼저 실행되어 있어야 함
- 환경 노드(`environment.py`)가 서비스 제공 상태여야 에이전트가 동작함
- `drl_agent_interfaces` 패키지가 빌드되어 있어야 함

## Troubleshooting

| 증상 | 조치 |
|------|------|
| 서비스 타임아웃 | Gazebo 실행 여부 확인, 환경 노드 실행 여부 확인 |
| `ModuleNotFoundError: torch` | `pip install torch==2.4.1+cu118` 설치 |
| `/odometry` 토픽 없음 | ros_gz_bridge 실행 여부 확인, 브릿지 설정 점검 |
| 학습 중 reward 수렴 안됨 | warmup 완료 후 학습 시작되는지 확인 (25k steps) |

## 이 README에서 다루지 않음

- 알고리즘 상세 구현: `scripts/policy/*.py` 소스 코드 참고
- Gazebo 시뮬레이션 설정: `agilex_scout` 패키지 README 참고
- 서비스 메시지 정의: `drl_agent_interfaces` 패키지 README 참고
