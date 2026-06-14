# `environment/` — DRL 환경 노드

강화학습 에이전트가 ROS2 서비스(`/reset`, `/step`, `/seed`, `/get_dimensions`,
`/action_space_sample`)로 통신하는 Gazebo 시뮬레이션 환경(`gym_node`) 구현체와,
거기서 분리해 낸 책임 단위 모듈들이 모여 있는 폴더입니다.

`environment.py`는 원래 한 파일에 5000줄 이상이 몰려 있었으나, 보상/충돌/노이즈
같은 순수 로직과 관찰·맵·샘플링·사람·장애물 같은 stateful 책임을 별도 모듈로 떼어내
지금은 **ROS 노드 오케스트레이션 중심(~2233줄)**으로 축소되었습니다. 노드 본체는
서비스 콜백(`reset_callback`/`step_callback`)과 초기화만 담당하고, 세부 계산은
아래 모듈들에 위임합니다.

## 노드 / 인터페이스

| 파일 | 줄 수 | 역할 |
|------|------:|------|
| `environment.py` | ~2233 | **메인 환경 노드(`gym_node`, Ignition Fortress).** 상태(87D) 조립·서비스 제공·초기화를 담당하는 오케스트레이터. 아래 mixin들을 다중 상속(`class Environment(ZoneMixin, ObservationMixin, MapLayoutMixin, StartSamplerMixin, GoalSamplerMixin, HumanSpawnMixin, HumanMotionMixin, GazeboEntityMixin, ObstacleMixin, Node)`)하고, 순수 모듈(reward/collision/noise/registry/catalog)에 위임 |
| `environment_curriculum.py` | ~462 | `Environment`를 상속한 **커리큘럼 학습 서브클래스.** `curriculum_stage` ROS 파라미터를 읽어 매 에피소드 전에 stage별 장애물 수·사람 모션·노이즈·맵 분포를 적용 |
| `environment_interface.py` | ~209 | **에이전트 측 클라이언트 베이스(`EnvInterface`).** timeout·retry 가진 견고한 서비스 호출 래퍼, `EnvServiceError`, AUX 라벨 분리 로직 |
| `environment_360.py` | ~774 | **레거시 Classic Gazebo 변형.** 동일 서비스를 노출하나 현재 메인 학습 경로 아님. 자체 `get_reward`/`get_environment_state`를 가진 독립 노드(분리 모듈을 쓰지 않음) |
| `__init__.py` | 0 | 패키지 마커(빈 파일) |

## 순수 로직 모듈 (ROS/torch 무의존 — `tests/`에서 단위 테스트)

| 파일 | 줄 수 | 역할 |
|------|------:|------|
| `reward_calculator.py` | ~140 | `compute_reward(...)` — 진행/헤딩/장애물/스무딩 보상 셰이핑 (순수 함수) |
| `collision_checker.py` | ~209 | `RectSafetyChecker` — 직사각형 안전영역 ray-rect 기하 + 충돌/근접 판정 (precompute 상태 보유) |
| `localization_noise.py` | ~292 | `LocalizationNoiseModel` / `ProprioNoiseModel` — 에피소드별 위치/자기수용 센서 노이즈 에뮬레이터 (OU/drift/jump/latency 상태 보유) |
| `map_catalog.py` | ~83 | static-obstacle 카탈로그 정책(맵타입별 허용/금지 키, size group) — mixin들이 공유 (순환 import 방지용 중립 모듈) |
| `map_layout_registry.py` | ~136 | `build_map_layouts(...)` — 맵타입별 wall/region 기하 데이터 생성 (순수 함수) |
| `aux_prediction_labels.py` | ~211 | AUX_PRED future-risk 라벨 직렬화/파싱. `environment.py`(생성)와 `environment_interface.py`(분리) 양쪽에서 import |

## 책임 단위 Mixin (환경 노드 상태를 `self`로 공유)

`Environment`에 다중 상속으로 합성됩니다. 보수적 리팩토링을 위해 `self.X` 참조를
그대로 유지(참조 무변환)하여 동작 동일성을 보장합니다. 각 mixin은 `__init__`을
정의하지 않으므로 `super().__init__()`이 `Node`로 정상 도달합니다.

| 파일 | 줄 수 | mixin | 역할 |
|------|------:|-------|------|
| `observation_builder.py` | ~218 | `ObservationMixin` | scan/cloud → `environment_state`(360° 충돌 bin) / `obs_state`(전방 180° 정책 입력) 조립, human-aware 마스킹 |
| `zone_tracker.py` | ~136 | `ZoneMixin` | 레거시 zone min/index 추적 (`use_zone_collision`, 기본 off) |
| `map_layout_runtime.py` | ~474 | `MapLayoutMixin` | map_type 선택·wall 활성화·배치 기하 술어. `_build_map_layouts`는 `map_layout_registry`에 위임 |
| `start_sampler.py` | ~290 | `StartSamplerMixin` | start/free-pose 샘플링 |
| `goal_sampler.py` | ~191 | `GoalSamplerMixin` | goal-pose 샘플링 (거리/맵타입/clearance 제약) |
| `human_spawn_sampler.py` | ~291 | `HumanSpawnMixin` | 보행자 spawn/mode/goal/waypoint 샘플링 |
| `human_motion_manager.py` | ~463 | `HumanMotionMixin` | 보행자 per-step kinematic 갱신 + Gazebo pose publish |
| `gazebo_entity_manager.py` | ~226 | `GazeboEntityMixin` | spawn/delete/goal-marker 서비스 호출 + SDF 빌더 |
| `obstacle_catalog_spawner.py` | ~564 | `ObstacleMixin` | obstacle/wall 풀 초기화·에피소드별 활성화·spawn (map_catalog 정책 적용) |

## 주요 관계

```
EnvInterface (environment_interface.py)   ← 트레이너/테스트가 상속 (클라이언트)
        │  /reset /step /seed /get_dimensions /action_space_sample
        ▼
Environment (environment.py)              ← 서비스 서버 ("gym_node")
   = ZoneMixin + ObservationMixin + MapLayoutMixin + StartSamplerMixin
   + GoalSamplerMixin + HumanSpawnMixin + HumanMotionMixin
   + GazeboEntityMixin + ObstacleMixin + Node
   (+ reward_calculator / collision_checker / localization_noise /
      map_layout_registry / map_catalog 에 위임)
        ▲
EnvironmentCurriculum (environment_curriculum.py)  ← stage 제어 추가 상속
```

- 실행 중인 환경 노드는 사용 중인 시뮬레이터와 일치해야 함:
  `environment.py`/`environment_curriculum.py` → Ignition Fortress,
  `environment_360.py` → Classic Gazebo.
- mixin/순수 모듈은 `environment.py`를 **역import하지 않습니다**(순환 방지). 공유
  상수는 중립 모듈 `map_catalog.py`에 둡니다.
- 새 모듈은 모두 flat bare import이므로 `CMakeLists.txt`의 `install(PROGRAMS ...)`에
  등록되어 있어야 합니다(빌드 후 같은 lib 디렉터리로 평탄화됨).

## 테스트

ROS/Gazebo 없이 도는 단위 테스트는 패키지 루트 `tests/`에 있습니다. 순수 모듈
(reward/collision/noise/catalog/registry)은 직접 테스트되고, mixin 경계는 ROS
런타임에서만 검증됩니다. pytest는 패키지 루트의 `conftest.py`(`collect_ignore`)와
`pytest.ini`(`norecursedirs`)로 `scripts/`·`launch/`의 ROS 실행 스크립트를 수집에서
제외하므로, 어느 작업 디렉터리에서 실행해도 동일하게 동작합니다:

```bash
# 둘 다 동일하게 105 passed (+ torch 미설치 시 skip)
python3 -m pytest -q ros2_ws/src/drl_agent
(cd ros2_ws/src/drl_agent && python3 -m pytest -q)
```

## 실행 예시

```bash
# 커리큘럼 환경 노드 (Ignition)
ros2 run drl_agent environment_curriculum.py \
  --ros-args -p config_file:=<path>/environment_curriculum.yaml

# 단일 환경 노드 (모드: train / test / random_test)
ros2 run drl_agent environment.py --ros-args -p environment_mode:=train
```

> 상태/액션 공간 정의, 설정 파일, 빌드/실행 전체 흐름은 패키지 루트
> [`../../README.md`](../../README.md)와 저장소 루트 `CLAUDE.md`를 참고하세요.
