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

> RGL LiDAR 플러그인 설치 필요 → [Installation](docs/guides/installation.md#2-rgl-lidar-plugin)

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

Profile config 주요 기능:

- **Structured map curriculum**: lobby/corridor/intersection/clutter 4종을 **10-stage**로 한 단계씩 도입(구조→사람→위치추정 노이즈→지형→proprio 노이즈→새 맵·군중→통합), stage·맵별 개수는 `*_by_map` (예: Stage 9 `static C5/I7/Cl8/L9`, `humans C3/I4/Cl4/L6`) · off: `map_layout_enabled=false` · [문서](docs/design/map_curriculum_design.md)
- **Profile별 action mode**: 기본 curriculum은 3D waypoint_yield, phase3 profile은 2D speed_steering
- **Localization noise**: Stage 4부터 ramp-up · off: `localization.enabled=false`
- **Auxiliary future-risk prediction**: 공유 인코더 + aux head · off: `hyperparameters_tqc.yaml`의 `aux_prediction.enabled=false` · [문서](docs/design/aux_prediction_design.md) · [지표](docs/reference/metrics_reference.md)

사용 가능한 profile:

| Phase | Profile | 용도 | Action mode |
|------|---------|------|-------------|
| phase2 | `phase2/tqc_vanilla` | TQC 확장 플래그 전체 OFF 기준선 | `waypoint_yield` |
| phase2 | `phase2/baseline` | risk reward/head 모두 OFF | `waypoint_yield` |
| phase2 | `phase2/reward_shaping_only` | risk_map_reward만 ON | `waypoint_yield` |
| phase2 | `phase2/action_risk_head_only` | action_risk_head만 ON | `waypoint_yield` |
| phase2 | `phase2/both` | risk_map_reward + action_risk_head + trajectory risk/RBS | `waypoint_yield` |
| phase2 | `phase2/both_legacy` | 이전 `phase2/both` 의미 보존 | `waypoint_yield` |
| phase2 | `phase2/both_trajrisk_rbs` | trajectory risk/RBS variant 명시 | `waypoint_yield` |
| phase2 | `phase2/obs_norm_optim_split` | 관측 정규화 + optimizer split 실험 | `waypoint_yield` |
| phase3 | `phase3/speed_steering_risk_balanced` | 연속 speed/steering + trajectory risk/RBS | `speed_steering` |

```bash
PROFILE=phase2/both

# [터미널 1] Gazebo 시뮬레이션
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# [터미널 2] 커리큘럼 환경 노드
ros2 run drl_agent environment_curriculum_node.py --ros-args -p profile:=$PROFILE

# [터미널 3] 학습 노드
ros2 run drl_agent train_node.py --ros-args -p profile:=$PROFILE -p seed:=0

# 자동 재개
ros2 run drl_agent train_node.py --ros-args -p profile:=$PROFILE -p seed:=0 -p resume:=true

# 특정 체크포인트에서 재시작
ros2 run drl_agent train_node.py --ros-args -p profile:=$PROFILE \
  -p resume_weight_prefix:=<prefix> -p resume_stage:=<0-9> \
  -p resume_weights_dir:=<run_dir>/pytorch_models

# run_dir / 학습 파라미터 override
ros2 run drl_agent train_node.py --ros-args -p profile:=$PROFILE \
  -p seed:=0 -p run_dir:=<경로> -p updates_per_env_step:=4

# Profile 검증 / 목록 / sweep
python3 ros2_ws/src/drl_experiments/scripts/run_profile.py $PROFILE --validate-only
python3 ros2_ws/src/drl_experiments/scripts/run_profile.py --list
python3 ros2_ws/src/drl_experiments/scripts/run_profile.py \
  --sweep ros2_ws/src/drl_experiments/sweeps/phase2_seeds.yaml

# TensorBoard 모니터링
tensorboard --logdir <run_dir>/logs
```

논문 비교 실험용 후처리/평가:

```bash
PROFILE=phase2/both

# 다중 seed 결과 집계
python3 -m drl_agent.evaluation.analysis.aggregate_results \
  --runtime-root ros2_ws/src/drl_agent/runtime

# Profile 기반 일반화 평가
ros2 run drl_agent eval_node.py --ros-args -p profile:=$PROFILE \
  -p weight_prefix:=<model_prefix> -p world:=aws_hospital

# 일반화 평가
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
| [Package Structure](docs/overview/package_structure.md) | 새 패키지 구조 + profile 기반 실험 실행 |
| [Training Pipeline](docs/overview/training_pipeline.md) | reset→state→action→step→train→eval 흐름 |
| [Paper Preparation Guide](docs/experiments/paper_preparation_guide.md) | 논문 방향, 수정 포인트, 평가 지표 다섯 축 정리 |
| [Experiment Protocol](docs/experiments/experiment_protocol.md) | baseline 비교 프로토콜, 다중 seed, 지표/CSV 스키마, 집계·일반화 |
| [Installation](docs/guides/installation.md) | 상세 빌드, Docker, RGL LiDAR 설치 |
| [Training](docs/guides/training.md) | 커리큘럼 학습, 재개 방법 |
| [Architecture](docs/design/environment_design.md) | 서비스 인터페이스, 상태/액션 공간, 파이프라인 |
| [Algorithms](docs/overview/repository_map.md#알고리즘-drl_agentrlalgorithms-drl_agenttraining) | TQC, TD7, SAC, A3C 알고리즘 설명 |
| [Configuration](docs/reference/config_reference.md) | config 파일별 파라미터 레퍼런스 |
| [Map Curriculum](docs/design/map_curriculum_design.md) | structured map(lobby/corridor/intersection/clutter) 커리큘럼 설계·구현 |
| [Aux Prediction Design](docs/design/aux_prediction_design.md) | 공유 인코더 + future-risk auxiliary head 설계 (single-step / action-conditioned) |
| [Aux Ablation Logging](docs/experiments/aux_ablation_logging.md) | aux on/off 비교용 run-identity / eval-summary / manifest 로깅 |
| [Aux Metric Schema](docs/reference/metrics_reference.md) | 학습 모니터링 vs 논문용 평가 지표, 저장 위치(CSV/TB/콘솔) |
| [Simulation Validation](docs/experiments/simulation_validation.md) | reset→step 일관성 등 시뮬레이션 검증 절차 |
| [Phase2 Optimization Notes](docs/experiments/phase2_optimization_notes.md) | `phase2/both` 학습 경로 처리량 개선 요약 (무엇을 줄였는지 + 실측치) |
| [TQC Model Architecture](docs/design/tqc_curriculum_model_architecture.md) | TQC curriculum 전체 모델 block diagram |
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
│   ├── drl_agent/                          # DRL 환경/정책/학습 (canonical, scripts/ flat 레이어 없음)
│   │   ├── drl_agent/                      #   importable Python package — 구현의 유일한 위치
│   │   │                                   #   (config loader/validator, trainer registry,
│   │   │                                   #   run/checkpoint manager, node wrappers)
│   │   └── tests/                          #   ROS-free pytest 스위트
│   ├── drl_experiments/                    # 실험 정의: profiles/ sweeps/ scripts/ outputs(gitignored)
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
