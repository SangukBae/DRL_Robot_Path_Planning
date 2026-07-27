# DRL Robot Path Planning

ROS2 Humble + Gazebo Ignition Fortress 기반 **AgileX Hunter SE** 자율주행 경로계획 시뮬레이션.

- 보행자(동적) + 정적 장애물 환경에서 Ackermann 조향 로봇의 목표 지향 주행을 학습하는 **시뮬레이션 기반 DRL 비교·연구 프레임워크**
- TQC 커리큘럼 학습이 주축, **SAC · TD7 · SB3-SAC · SB3-TD3 · TQC+IEQN**을 동일 조건에서 비교할 baseline·로깅·평가·다중 seed 인프라 포함

**Tech Stack**: ROS2 Humble · Gazebo Ignition Fortress · PyTorch 2.4.1 (CUDA 11.8) · Python 3.10

---

## Prerequisites

| 항목 | 버전 |
|------|------|
| Ubuntu | 22.04 |
| ROS2 | Humble |
| Gazebo | Ignition Fortress |
| CUDA | 11.8 |
| Python | 3.10 |

> RGL LiDAR 플러그인 설치 필요 → [Installation](docs/guides/installation.md#rgl-lidar-plugin)

---

## Quick Install

```bash
# 1. 의존성 설치
cd ros2_ws
rosdep install --from-paths src -yi --rosdistro humble \
  --skip-keys='libgraphicsmagick++1-dev graphicsmagick-libmagick-dev-compat'

# 2. 빌드
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# 3. 소스
source install/setup.bash

# 4. (GUI 사용 시만) X11 디스플레이 접근 허용 — Gazebo GUI/RViz, Docker GUI에 필요.
#    headless(rviz:=false)면 생략 가능. Wayland는 XWayland/QT_QPA_PLATFORM 추가 설정 필요.
xhost +local:
```

---

## Quick Start

`environment_curriculum.yaml` 기본 활성 기능:

- **Structured map curriculum**: lobby/corridor/intersection/clutter 4종을 **10-stage**로 한 단계씩 도입(구조→사람→위치추정 노이즈→지형→proprio 노이즈→새 맵·군중→통합), stage·맵별 개수는 `*_by_map` (예: Stage 9 `static C5/I7/Cl8/L9`, `humans C3/I4/Cl4/L6`) · off: `map_layout_enabled=false` · [문서](docs/design/map_curriculum_design.md)
- **3D hybrid stop/yield 액션**: 전진 waypoint + 전용 yield 축
- **Localization noise**: Stage 4부터 ramp-up · off: `localization.enabled=false`
- **Auxiliary future-risk prediction**: 공유 인코더 + aux head · off: `hyperparameters_tqc.yaml`의 `aux_prediction.enabled=false` · [문서](docs/design/aux_prediction_design.md) · [지표](docs/reference/metrics_reference.md)

```bash
# [터미널 1] Gazebo 시뮬레이션
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# [터미널 2] 커리큘럼 환경 노드
ros2 run drl_agent environment_curriculum.py

# [터미널 3] TQC 커리큘럼 학습 — 실행 모드별 명령어
# 1) 새 학습 (다중 seed: -p seed:=N 또는 DRL_AGENT_SEED=N 환경변수로 sweep)
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p seed:=0

# 2) 자동 재개 — train_tqc_config.yaml(커리큘럼 config 아님)의 train_settings에서
#    load_model:true 설정 후 동일 명령 재실행. base_file_name+seed가 일치하는 최신
#    체크포인트 + replay buffer (+ curriculum_state.json 있으면 스테이지/글로벌 스텝도) 복원.
#    상세: docs/guides/training.md#학습-재개
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p seed:=0

# 3) 특정 체크포인트에서 지정 스테이지로 재시작 — 모델 가중치만 로드하고
#    replay buffer/optimizer/커리큘럼 진행도는 새로 시작 (A3/A4 fresh-run-only 아키텍처는 불가)
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args \
  -p resume_weight_prefix:=<prefix> -p resume_stage:=<0-9> \
  -p resume_weights_dir:=<run_dir>/pytorch_models   # 생략 시 이번 run의 pytorch_models_dir

# 4) run_dir / train config 커스터마이징 (예: A1-A4 스케일링 실험, UTD 비율 override)
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args \
  -p run_dir:=<경로> -p train_config_file:=<경로> -p updates_per_env_step:=4

# 5) PHASE2 후보(candidate1=risk_map_reward, candidate2=action_risk_head)
#    yaml 세트를 디렉터리 단위로 맞춰 실행한다.
#    train_config_file은 임의 yaml 파일명이 아니라 아래 config 디렉터리를 넘기는 것이 안전하다.
#
#    모드별 의미:
#      baseline              : candidate1 OFF, candidate2 OFF
#      reward_shaping_only   : candidate1 ON,  candidate2 OFF
#      action_risk_head_only : candidate1 OFF, candidate2 ON
#      both                  : candidate1 ON,  candidate2 ON
#
#    [터미널 2] 환경 노드 예시: MODE만 위 4개 중 하나로 선택
MODE=both
CFG=src/drl_agent/runtime/phase2_configs/${MODE}
ros2 run drl_agent environment_curriculum.py --ros-args \
  -p config_file:=${CFG}/environment_curriculum.yaml

#    [터미널 3] 학습 노드 예시: 터미널 2와 같은 MODE 사용
MODE=both
CFG=src/drl_agent/runtime/phase2_configs/${MODE}
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args \
  -p train_config_file:=${CFG} -p seed:=0

#    4조합을 명시적으로 실행하려면 아래처럼 MODE만 바꾼다.
#    candidate1 OFF + candidate2 OFF
MODE=baseline
CFG=src/drl_agent/runtime/phase2_configs/${MODE}
ros2 run drl_agent environment_curriculum.py --ros-args -p config_file:=${CFG}/environment_curriculum.yaml
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p train_config_file:=${CFG} -p seed:=0

#    candidate1 ON + candidate2 OFF
MODE=reward_shaping_only
CFG=src/drl_agent/runtime/phase2_configs/${MODE}
ros2 run drl_agent environment_curriculum.py --ros-args -p config_file:=${CFG}/environment_curriculum.yaml
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p train_config_file:=${CFG} -p seed:=0

#    candidate1 OFF + candidate2 ON
MODE=action_risk_head_only
CFG=src/drl_agent/runtime/phase2_configs/${MODE}
ros2 run drl_agent environment_curriculum.py --ros-args -p config_file:=${CFG}/environment_curriculum.yaml
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p train_config_file:=${CFG} -p seed:=0

#    candidate1 ON + candidate2 ON (both)
MODE=both
CFG=src/drl_agent/runtime/phase2_configs/${MODE}
ros2 run drl_agent environment_curriculum.py --ros-args -p config_file:=${CFG}/environment_curriculum.yaml
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p train_config_file:=${CFG} -p seed:=0

# 동일 프로토콜로 다른 baseline도 비교 가능:
#   train_sac_curriculum_agent.py / train_td7_curriculum_agent.py
#   train_sb3_sac_curriculum_agent.py / train_sb3_td3_curriculum_agent.py
#   train_tqc_ieqn_curriculum_agent.py
# 알고리즘을 파라미터 하나로 선택하려면 train_rl.py (레지스트리 방식, tqc_curriculum은 byte-identical):
ros2 run drl_agent train_rl.py --ros-args -p rl_model:=tqc_curriculum
ros2 run drl_agent train_rl.py --list-models   # 사용 가능한 알고리즘 목록 (Gazebo 불필요)

# TensorBoard 모니터링
tensorboard --logdir <run_dir>/logs
```

논문 비교 실험용 후처리/평가:

```bash
# 다중 seed 결과 집계 (eval_metrics_*.csv → mean±std 표 / 학습곡선 / sample efficiency)
python3 ros2_ws/src/drl_agent/scripts/utils/aggregate_results.py \
  --runtime-root ros2_ws/src/drl_agent/runtime

# 일반화 평가 (학습된 모델을 stage/world 별로 재학습 없이 평가) — 현재 TQC 전용
#   weights_dir 는 기본적으로 해당 run 의 pytorch_models/ 를 보지만,
#   final_models/ 등 다른 위치의 모델을 쓰려면 -p weights_dir:=<경로> 로 지정
ros2 run drl_agent generalization_eval.py --ros-args \
  -p weight_prefix:=<model_prefix> -p weights_dir:=<run_dir>/final_models \
  -p world:=aws_hospital -p eval_eps_override:=20
```

> 프로토콜·지표·CSV 스키마 상세는 [Experiment Protocol](docs/experiments/experiment_protocol.md) 참고.

---

## Documentation

> 전체 문서 인덱스·독자별 추천 경로는 **[Documentation Hub](docs/README.md)** 참고.

| 문서 | 내용 |
|------|------|
| [Documentation Hub](docs/README.md) | 문서 인덱스 + 독자별 추천 읽기 순서 |
| [System Overview](docs/overview/system_overview.md) | 시스템 전체가 무엇인지 (처음 보는 사람용) |
| [Training Pipeline](docs/overview/training_pipeline.md) | reset→state→action→step→train→eval 흐름 |
| [Paper Preparation Guide](docs/experiments/paper_preparation_guide.md) | 논문 방향, 수정 포인트, 평가 지표 다섯 축 정리 |
| [Experiment Protocol](docs/experiments/experiment_protocol.md) | baseline 비교 프로토콜, 다중 seed, 지표/CSV 스키마, 집계·일반화 |
| [Installation](docs/guides/installation.md) | 상세 빌드, Docker, RGL LiDAR 설치 |
| [Training](docs/guides/training.md) | 커리큘럼 학습, 재개 방법 |
| [Architecture](docs/design/environment_design.md) | 서비스 인터페이스, 상태/액션 공간, 파이프라인 |
| [Algorithms](docs/overview/repository_map.md) | TQC, TD7, SAC, A3C 알고리즘 설명 |
| [Configuration](docs/reference/config_reference.md) | config 파일별 파라미터 레퍼런스 |
| [Map Curriculum](docs/design/map_curriculum_design.md) | structured map(lobby/corridor/intersection/clutter) 커리큘럼 설계·구현 |
| [Aux Prediction Design](docs/design/aux_prediction_design.md) | 공유 인코더 + future-risk auxiliary head 설계 (single-step / action-conditioned) |
| [Aux Ablation Logging](docs/experiments/aux_ablation_logging.md) | aux on/off 비교용 run-identity / eval-summary / manifest 로깅 |
| [Aux Metric Schema](docs/reference/metrics_reference.md) | 학습 모니터링 vs 논문용 평가 지표, 저장 위치(CSV/TB/콘솔) |
| [Simulation Validation](docs/experiments/simulation_validation.md) | reset→step 일관성 등 시뮬레이션 검증 절차 |
| [Real Robot Deployment](docs/guides/real_robot_deployment.md) | 학습 정책의 실로봇 배포 가이드 |
| [Troubleshooting](docs/guides/troubleshooting.md) | 환경 변수, 디버깅, 흔한 오류 |

## Paper Work Note

논문 비교 실험용 **시스템 구축 완료**. 적용 범위:

- **6개 알고리즘 공통**(TQC, TQC+IEQN, SAC, TD7, SB3-SAC, SB3-TD3): 동일 학습/평가 프로토콜, episode/eval-level CSV 로깅, 다중 seed 실행·격리, 결과 집계(`aggregate_results.py`)
- **현재 TQC 전용**: 일반화 평가 하니스(`generalization_eval.py`) — 다른 알고리즘은 확장 필요

기본 설정에 포함된 TQC 확장 (config 플래그로 on/off, 추가 작업 불필요):

- **Auxiliary future-risk prediction**: 공유 인코더 + aux head로 미래 충돌 위험 예측. single-step / action-conditioned(`[a_t..a_{t+K-1}]`) / aux-only temporal context(GRU, actor·critic은 비순환 유지) 세 형태 지원, boundary-safe 샘플링, 평가지표 `aux_risk_rmse / aux_min_dist_mae_m / aux_peak_sector_acc / aux_near_event_f1`. off(`aux_prediction.enabled=false`)면 baseline TQC와 byte-단위 동일 → [설계](docs/design/aux_prediction_design.md) · [지표](docs/reference/metrics_reference.md)
- **Structured map curriculum**: lobby/corridor/intersection/clutter를 stage별 샘플링 → [문서](docs/design/map_curriculum_design.md)
- **Localization-noise emulation**: 상관(OU) 노이즈 + drift + latency + map-type별 강도(+ corridor 이방성) + 드문 jump

추가 검토 중인 TQC 내부 method 변경(risk-aware actor, adaptive truncation 등)은 [Paper Preparation Guide](docs/experiments/paper_preparation_guide.md) · [Experiment Protocol](docs/experiments/experiment_protocol.md) 절차를 따른다.

---

## Repository Layout

```
/
├── ros2_ws/src/
│   ├── drl_agent/                          # DRL 환경/정책/학습 스크립트
│   ├── drl_agent_interfaces/               # ROS2 srv/action 정의
│   ├── hunter_se_gazebo/                   # Hunter SE URDF, Gazebo launch, worlds
│   ├── drl_obstacle_assets/                # Gazebo 장애물 모델 라이브러리 (38종)
│   ├── ouster_simulation/ouster_description/  # OS1-64 LiDAR (RGL)
│   ├── aws-robomaker-hospital-world/
│   ├── aws-robomaker-bookstore-world/
│   ├── aws-robomaker-small-house-world/
│   └── aws-robomaker-small-warehouse-world/
├── docs/                                   # 상세 문서
├── Dockerfile
└── cyclonedds_config.xml
```

---

## License

MIT
