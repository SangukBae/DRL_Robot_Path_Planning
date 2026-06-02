# drl_agent

## Overview

Hunter SE 자율 주행을 위한 DRL(Deep Reinforcement Learning) 에이전트 패키지.
현재 주 학습 경로는 TQC 커리큘럼 학습이며, 관련 환경/정책/설정 스크립트를 포함한다.

환경 노드가 ROS2 서비스로 상태/보상을 제공하고, 에이전트 노드가 클라이언트로 동작하는 구조다.

## Quick Start

```bash
# 1) Gazebo 시뮬레이션 먼저 실행 (별도 터미널)
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# 2) 커리큘럼 환경 노드 실행
ros2 run drl_agent environment_curriculum.py

# 3) TQC 커리큘럼 학습
ros2 run drl_agent train_tqc_curriculum_agent.py

# 4) TQC 테스트 (launch 파일)
ros2 launch drl_agent test_tqc.launch.py
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
| `/scan` | 구독 | `LaserScan` | 2D LiDAR 스캔 |
| `/ouster/points` | 구독 | `PointCloud2` | 3D 포인트클라우드 |
| `/cmd_vel` | 발행 | `Twist` | 로봇 속도 명령 |

### State/Action Space

- **State (87D)**: 전방 180도 LiDAR 80빈 + agent state 7D
- **Action (2D)**: 웨이포인트 거리/각도 명령, `environment.py`에서 물리 단위로 변환

## Configuration

| 파일 | 역할 |
|------|------|
| `config/environment_curriculum.yaml` | 커리큘럼 환경 파라미터 및 스테이지 정의 |
| `config/hyperparameters_tqc.yaml` | TQC 하이퍼파라미터 (batch_size, buffer_size 등) |
| `config/train_tqc_config.yaml` | TQC 학습 설정 (max_timesteps, warmup 등) |
| `config/train_tqc_curriculum_config.yaml` | 커리큘럼 진급 규칙 |
| `config/test_tqc_config.yaml` | TQC 테스트 설정 (시작/목표 쌍) |

**주요 파라미터:**
- `goal_threshold`: 0.42m (목표 도달 판정)
- `collision_threshold`: 0.7m (충돌 판정)
- `timesteps_before_training`: 12,000 (랜덤 액션 구간)
- `max_timesteps`: 2,000,000

## Dependencies / Assumptions

### 의존성

- `rclpy`, `drl_agent_interfaces`
- `python3-tensorboard-pip`, `python3-squaternion-pip`
- PyTorch 2.4.1+ (CUDA 11.8)

### 전제조건

- Gazebo Ignition 시뮬레이션이 먼저 실행되어 있어야 함
- 환경 노드(`environment_curriculum.py`)가 서비스 제공 상태여야 에이전트가 동작함
- `drl_agent_interfaces` 패키지가 빌드되어 있어야 함

## Troubleshooting

| 증상 | 조치 |
|------|------|
| 서비스 타임아웃 | Gazebo 실행 여부 확인, 환경 노드 실행 여부 확인 |
| `ModuleNotFoundError: torch` | `pip install torch==2.4.1+cu118` 설치 |
| `/odometry` 토픽 없음 | ros_gz_bridge 실행 여부 확인, 브릿지 설정 점검 |
| 학습 중 reward 수렴 안됨 | warmup 완료 후 학습 시작되는지 확인 (12k steps) |

## 이 README에서 다루지 않음

- 알고리즘 상세 구현: `scripts/policy/*.py` 소스 코드 참고
- Gazebo 시뮬레이션 설정: `hunter_se_gazebo` 패키지 및 저장소 최상위 README 참고
- 서비스 메시지 정의: `drl_agent_interfaces` 패키지 README 참고
