# drl_agent

## Overview

Hunter SE 자율 주행을 위한 DRL(Deep Reinforcement Learning) 에이전트 패키지.
현재 주 학습 경로는 TQC 커리큘럼 학습이며, 관련 환경/정책/설정 스크립트를 포함한다.

환경 노드가 ROS2 서비스로 상태/보상을 제공하고, 에이전트 노드가 클라이언트로 동작하는 구조다.

## Quick Start

```bash
# 1) Gazebo 시뮬레이션 먼저 실행 (별도 터미널)
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# 2) profile 기반 커리큘럼 환경 노드 실행
ros2 run drl_agent environment_curriculum_node.py --ros-args \
  -p profile:=phase2/both_trajrisk_rbs_cf_st

# 3) 같은 profile로 TQC 커리큘럼 학습(fresh run)
ros2 run drl_agent train_node.py --ros-args \
  -p profile:=phase2/both_trajrisk_rbs_cf_st -p seed:=0

# 4) TQC 테스트 (launch 파일)
ros2 launch drl_agent test_tqc.launch.py
```

이 프로필의 사람 장애물은 환경 노드 내부 obstacle pool/human-motion 로직이
관리하므로 HuNav launch가 아니라 위의 일반 Ignition launch를 사용한다.
`both_trajrisk_rbs_cf_st`는 새 네트워크/replay 계약을 사용하는 fresh-run
프로필이므로 새 학습 명령에 `resume:=true`를 추가하지 않는다. 기본 config를
직접 실행하는 저수준 진입점(`environment_curriculum.py`,
`train_tqc_curriculum.py`)도 호환 목적으로 설치되어 있다.

## Interfaces

### Services (제공)

| 서비스명 | 타입 | 설명 |
|---------|------|------|
| `/reset` | `Reset.srv` | 에피소드 초기화, 초기 상태 반환 |
| `/step` | `Step.srv` | 액션 실행 → (상태, 보상, done, target) 반환 |
| `/get_dimensions` | `GetDimensions.srv` | state_dim, action_dim, max_action, environment_dim, agent_dim 반환 |
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

- **State (87D)**: 전방 180° LiDAR 80빈 + agent state 7D (옵션 시간 맥락 ON 시 프레임 스택 → 327D, 현재 87D 프레임이 맨 앞)
- **Action (3D, 하이브리드 stop/yield)**: 전진 waypoint 거리 r·각도 θ + yield(정지) 축. `environment.py`가 물리 단위로 변환 후 Pure Pursuit로 `cmd_vel` 생성 (비커리큘럼 `environment.yaml` baseline은 2D 유지)
- 정확한 인덱스/범위: [State/Action Reference](../../../docs/reference/state_action_reference.md)

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
- 환경 노드(`environment_curriculum_node.py -p profile:=...`, 또는 직접 실행
  `environment_curriculum.py`)가 서비스 제공 상태여야 에이전트가 동작함
- `drl_agent_interfaces` 패키지가 빌드되어 있어야 함

## Tests (ROS2 없이 실행)

순수 로직(각도/기하, 시드 재현성, 경로 탐색) 단위 테스트는 ROS2·Gazebo·빌드 없이
바로 돌아갑니다. 패키지 루트에서:

```bash
cd ros2_ws/src/drl_agent
pytest            # tests/ 디렉터리만 수집 (pytest.ini의 testpaths=tests)
```

- `tests/test_geometry_utils.py` — `drl_agent/common/geometry_utils.py` (wrap_to_pi, heading_error, goal_distance_and_heading 등)
- `tests/test_seed_utils.py` — `drl_agent/common/seed_utils.py` (random/numpy 재현성, resume-seed 파생)
- `tests/test_config_paths.py` — `drl_agent/config/paths.py` (config 파일 탐색)

> **`tqc_live_runner.py` / `td7_live_runner.py` 는 pytest 단위 테스트가 아니라**
> ROS2 + Gazebo + 체크포인트가 필요한 **실행/평가 스크립트**입니다
> (`ros2 run drl_agent tqc_live_runner.py ...`), 그래서 `test_` 접두어 없이
> canonical 이름으로 설치되어 있고 `pytest`는 이들을 수집하지 않습니다.
> 체크포인트 경로는 하드코딩 기본값을 제거했으므로 파라미터로 넘기세요:
> `-p checkpoint_actor_file:=<run_dir>/final_models/<prefix>_actor.pth`

### 분리된 공통 유틸 (`drl_agent/common/`)

`environment.py` 등에 흩어져 있던 순수 함수를 점진적으로 추출한 모듈 (동작 동일):

| 모듈 | 내용 |
|------|------|
| `geometry_utils.py` | 각도 wrap, heading error, 거리, goal 메트릭 (ROS/numpy 무의존) |
| `seed_utils.py` | random/numpy/torch 통합 시드, resume-seed 파생 |
| `config_paths.py` | config 파일 후보 탐색 (순수, ROS/ament 무의존) |

## Troubleshooting

| 증상 | 조치 |
|------|------|
| 서비스 타임아웃 | Gazebo 실행 여부 확인, 환경 노드 실행 여부 확인 |
| `ModuleNotFoundError: torch` | `pip install torch==2.4.1+cu118` 설치 |
| `/odometry` 토픽 없음 | ros_gz_bridge 실행 여부 확인, 브릿지 설정 점검 |
| 학습 중 reward 수렴 안됨 | warmup 완료 후 학습 시작되는지 확인 (12k steps) |

## 이 README에서 다루지 않음

- 알고리즘 상세 구현: `drl_agent/rl/algorithms/*/agent.py`, `drl_agent/training/{train_tqc_base,train_tqc_curriculum,baselines/*}.py` 소스 코드 참고
- Gazebo 시뮬레이션 설정: `hunter_se_gazebo` 패키지 및 저장소 최상위 README 참고
- 서비스 메시지 정의: `drl_agent_interfaces` 패키지 README 참고
