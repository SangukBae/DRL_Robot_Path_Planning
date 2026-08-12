# Environment Design

이 문서는 **환경 노드가 어떻게 동작하는지**와 **`environment.py` vs `environment_curriculum.py` 역할 분리**를 설명한다.

## What
환경 노드는 Gazebo를 제어하고, 로봇 관측을 RL이 쓰는 state로 바꾸고, action을 실행해 보상/종료를 돌려주는 **"RL 환경" 그 자체**다. ROS2 서비스(`/reset`, `/step`, `/get_dimensions`, `/seed`, `/action_space_sample`)를 노출한다.

## Why — 두 파일로 나눈 이유
- `environment.py` = **executor**: noise/보상/충돌/배치 같은 *의미 해석과 실제 주입*을 모두 담당하는 기반 클래스.
- `environment_curriculum.py` = **selector**: `environment.py`를 상속해, **stage마다 어떤 설정(장애물 수·맵·noise 프로파일)을 쓸지 고르기만** 한다. noise 수식이나 state 구성은 건드리지 않는다.

→ 학습 난이도 스케줄(커리큘럼)과 환경의 물리/관측 로직을 분리해, 한쪽을 바꿔도 다른 쪽이 안 깨진다.

| | environment.py | environment_curriculum.py |
|--|--|--|
| 역할 | 관측 생성, action 실행, 보상/충돌, noise 주입, 맵/장애물 배치 | stage 전환, stage별 프로파일 선택/merge |
| 상속 | 기반 클래스 (`Environment`) | 서브클래스 (`EnvironmentCurriculum`) |
| reset | `reset_callback` (실제 배치·관측) | `reset_callback` 오버라이드 → stage 적용 후 `super()` 호출 |

## How — 한 step의 내부 (`step_pipeline.py::StepPipelineMixin`)
1. 정규화 action → profile action mode에 따라 `waypoint_yield` 또는 `speed_steering` decode → `cmd_vel` publish
   (`waypoint_yield`의 MOVE/YIELD 계약과 `speed_steering` 2축 계약은 [state/action 표](../reference/state_action_reference.md#action-modes) 참고)
2. `gazebo_runtime.py`가 Gazebo를 `time_delta`(0.1s) 진행하고, 환경 노드의
   20Hz human-motion 타이머가 obstacle pool의 사람을 이동
3. LiDAR(`/scan`) → `obs_state`(전방 180° 80빈, 정책 입력) + 충돌용 360° 80빈 `environment_state`
4. odom/joint → 실제 속도·요레이트·조향, 목표 거리/방향 계산
5. goal 관측(state[80],[81])에 **localization noise** 주입(옵션), proprio 슬롯에 proprio noise(옵션)
6. 보상·충돌·도달·타임아웃 판정 → `(state, reward, done)` 반환
   - **보상/종료는 항상 ground-truth 좌표** 사용 (noise는 관측에만)

## 커맨드 / LiDAR 파이프라인
```
policy → /cmd_vel → hunter_se_cmd_prefilter(50Hz, 벽시계) → /cmd_vel_filtered → bridge → Gazebo
Gazebo Ouster(RGL) → /ouster/points → pointcloud_to_laserscan → /scan → environment.py
```
- 프리필터는 벽시계 50Hz라 시뮬 RTF가 낮아도 명령이 끊기지 않는다.
- LiDAR height filter(`obs_z_min/max_sensor_m`)로 바닥/천장을 잘라 장애물만 본다.

## 환경 변형
| 파일 | 시뮬레이터 | reset |
|--|--|--|
| `environment.py` | Ignition Fortress | `ros_gz_interfaces/SetEntityPose` |
| `environment_curriculum.py` | Ignition Fortress | 위 + 커리큘럼 |
| `environment_360.py` | Classic Gazebo | `gazebo_msgs/SetEntityState` |

## Where in code
- `ros2_ws/src/drl_agent/drl_agent/env/simulation/environment.py` (ROS node 구성, 공유 상태와 컴포넌트 조율)
- `ros2_ws/src/drl_agent/drl_agent/env/simulation/step_pipeline.py` (`/step`, action decode, 관측·보상 연결)
- `ros2_ws/src/drl_agent/drl_agent/env/simulation/reset_pipeline.py` (`/reset`, episode 초기화·배치)
- `ros2_ws/src/drl_agent/drl_agent/env/simulation/gazebo_runtime.py` (world service, 물리 step, sensor freshness)
- `ros2_ws/src/drl_agent/drl_agent/env/simulation/risk_targets.py` (directional/CF swept-path risk target)
- `ros2_ws/src/drl_agent/drl_agent/env/curriculum/environment_curriculum.py` (stage selector)
- 서비스/토픽 표: [../reference/ros_interface_reference.md](../reference/ros_interface_reference.md)
- state/action 표: [../reference/state_action_reference.md](../reference/state_action_reference.md)
