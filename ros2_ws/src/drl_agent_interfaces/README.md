# drl_agent_interfaces

## Overview

DRL 에이전트-환경 통신을 위한 ROS2 커스텀 인터페이스 패키지.
서비스(7개)와 액션(1개) 정의를 포함한다.

환경 노드가 서버, 에이전트 노드가 클라이언트로 동작하는 서비스 기반 아키텍처를 지원한다.

## Quick Start

```bash
# 패키지 빌드
cd ros2_ws
colcon build --packages-select drl_agent_interfaces

# 소스
source install/setup.bash

# 서비스 타입 확인
ros2 interface show drl_agent_interfaces/srv/Step

# 액션 타입 확인
ros2 interface show drl_agent_interfaces/action/StartDrlSession
```

## Interfaces

### Services (srv/)

| 서비스 | Request | Response | 용도 |
|--------|---------|----------|------|
| `Step.srv` | `float32[] action` | `state, reward, done, target` | 환경 스텝 실행 |
| `Reset.srv` | (empty) | `float32[] state` | 에피소드 리셋 |
| `GetDimensions.srv` | (empty) | `state_dim, action_dim, max_action` | 환경 차원 조회 |
| `Seed.srv` | `int32 seed` | `bool success` | 랜덤 시드 설정 |
| `SampleActionSpace.srv` | (empty) | `float32[] action` | 랜덤 액션 샘플 |
| `GetTestEpisodes.srv` | (empty) | `int32 num_of_test_episodes` | 테스트 에피소드 수 조회 |
| `GetStartGoalPairs.srv` | (empty) | `start[], goal[]` | 시작/목표 좌표 조회 |

### Actions (action/)

| 액션 | Goal | Feedback | Result | 용도 |
|------|------|----------|--------|------|
| `StartDrlSession.action` | `string mode` | `string progress` | `bool success` | 학습/테스트 세션 제어 |

### 메시지 정의 예시

```
# Step.srv
float32[] action
---
float32[] state
float32 reward
bool done
bool target
```

## Configuration

| 파일 | 역할 |
|------|------|
| `srv/*.srv` | 서비스 메시지 정의 (7개) |
| `action/*.action` | 액션 메시지 정의 (1개) |
| `CMakeLists.txt` | rosidl 인터페이스 생성 설정 |

빌드 시 `rosidl_generate_interfaces()`가 C++/Python 바인딩을 자동 생성한다.

## Dependencies / Assumptions

### 의존성

- `rosidl_default_generators` (빌드)
- `rosidl_default_runtime` (런타임)
- `rclcpp`, `std_msgs`

### 전제조건

- 이 패키지는 `drl_agent` 패키지보다 먼저 빌드되어야 함
- 메시지 타입 변경 시 의존 패키지 재빌드 필요

## Troubleshooting

| 증상 | 조치 |
|------|------|
| `ModuleNotFoundError: drl_agent_interfaces` | `source install/setup.bash` 실행 |
| 인터페이스 변경 후 타입 불일치 | `colcon build --packages-select drl_agent_interfaces` 후 의존 패키지도 재빌드 |
| `ros2 interface show` 실패 | 패키지 빌드 완료 및 소스 여부 확인 |

## 이 README에서 다루지 않음

- 서비스 호출 구현 예제: `drl_agent/scripts/environment/environment_interface.py` 참고
- 환경 노드 서비스 제공 로직: `drl_agent/scripts/environment/environment.py` 참고
