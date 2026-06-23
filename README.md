# DRL Robot Path Planning

ROS2 Humble + Gazebo Ignition Fortress 기반 **AgileX Hunter SE** 자율주행 경로계획 시뮬레이션.

동적 장애물/보행자 혼합 환경에서 Ackermann 조향 로봇의 목표 지향 자율 주행을 학습하는 **시뮬레이션 기반 DRL 비교·연구 프레임워크**다.
TQC 커리큘럼 학습을 주축으로, **SAC · TD7 · SB3-SAC · SB3-TD3 · TQC+IEQN**을 동일 조건에서 비교할 수 있도록 baseline·로깅·평가·다중 seed 인프라를 갖췄다.

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

# 4. (GUI를 띄울 때만) 로컬 X11 클라이언트의 디스플레이 접근 허용.
#    Gazebo GUI / RViz를 보거나 Docker 컨테이너에서 GUI를 띄울 때 필요하다.
#    X11 전제이며, headless 학습(rviz:=false + GUI 미사용)이면 생략 가능하다.
#    Wayland 세션이면 XWayland 또는 QT_QPA_PLATFORM 설정이 추가로 필요할 수 있다.
xhost +local:
```

---

## Quick Start

> 기본 커리큘럼 설정(`environment_curriculum.yaml`)에는 **structured map curriculum**
> (lobby/corridor/intersection/clutter 4종, **7-stage**로 한 단계에 한 축씩 증가 —
> 구조→사람→지형→일반화→노이즈; stage별·**맵별** 장애물/휴먼 개수 `*_by_map`),
> **localization noise emulation**(상관 노이즈 + drift + map-type별 강도),
> **auxiliary future-risk prediction**(공유 인코더 + aux head, env 라벨)이 **기본 활성화**되어 있다.
> 현재 기본값은 corridor를 가장 가볍게 두도록 조정되어 있으며, 예를 들어 최종 Stage 6의
> 맵별 활성 개수는 `static: C5 / I7 / Cl8 / L9`, `humans: C3 / I4 / Cl4 / L6`이다.
> 단, localization noise는 base가 off이고 **Stage 3부터 per-stage로 ramp-up**된다
> (Stage 0~2는 clean; 전체 비활성화는 각 stage의 `localization_profile`을 `clean`으로 두거나
> base `localization.enabled: false`로).
> map curriculum은 `environment_curriculum.yaml`의 `map_layout_enabled`,
> aux prediction은 `hyperparameters_tqc.yaml`의 `aux_prediction.enabled`를 false로 두면 꺼진다.
> 설계·지표는 [Map Curriculum](docs/design/map_curriculum_design.md) ·
> [Aux Prediction Design](docs/design/aux_prediction_design.md) ·
> [Aux Metric Schema](docs/reference/metrics_reference.md) 참고.

```bash
# [터미널 1] Gazebo 시뮬레이션
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# [터미널 2] 커리큘럼 환경 노드
ros2 run drl_agent environment_curriculum.py

# [터미널 3] TQC 커리큘럼 학습 (다중 seed: -p seed:=N 으로 sweep)
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p seed:=0
# 동일 프로토콜로 다른 baseline도 비교 가능:
#   train_sac_curriculum_agent.py / train_td7_curriculum_agent.py
#   train_sb3_sac_curriculum_agent.py / train_sb3_td3_curriculum_agent.py
#   train_tqc_ieqn_curriculum_agent.py

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

논문 비교 실험을 위한 **시스템 구축 단계는 완료**된 상태다. 단, 적용 범위가 다르므로 구분한다:

- **6개 알고리즘 공통**(TQC, TQC+IEQN, SAC, TD7, SB3-SAC, SB3-TD3): 동일 학습/평가 프로토콜, episode/eval-level CSV 로깅, 다중 seed 실행·격리, 결과 집계(`aggregate_results.py`).
- **현재 TQC 기준**: 일반화 평가 하니스(`generalization_eval.py`)는 TQC 모델용이며, 다른 알고리즘은 해당 agent 클래스로 동일 패턴 확장이 필요하다.

**이미 구현되어 기본 설정에 포함된 TQC 확장**(별도 작업 불필요, config 플래그로 on/off):

- **Auxiliary future-risk prediction**: 공유 인코더 + aux head로 미래 충돌 위험을 예측하는 보조 과제. **single-step**, **action-conditioned**(`[a_t..a_{t+K-1}]`로 조건화), 그리고 **aux-only temporal context**(최근 `history_len` in-episode state를 GRU로 요약해 aux head 입력에만 concat — actor/critic은 비순환 유지) 세 형태 모두 구현. temporal/action context는 replay buffer에서 episode 경계를 넘지 않는 boundary-safe walk로 샘플링한다. env가 privileged future-risk 라벨을 생성하고, 평가 루프가 `aux_risk_rmse / aux_min_dist_mae_m / aux_peak_sector_acc / aux_near_event_f1`를 산출한다. 끄면(`aux_prediction.enabled=false`) baseline TQC와 byte-단위 동일하게 동작한다. → [Aux Prediction Design](docs/design/aux_prediction_design.md), [Aux Metric Schema](docs/reference/metrics_reference.md)
- **Structured map curriculum**: lobby/corridor/intersection/clutter 4종 맵을 stage별로 샘플링. → [Map Curriculum](docs/design/map_curriculum_design.md)
- **Localization-noise emulation**: 상관(OU) 노이즈 + drift + latency + map-type별 강도(+ corridor 이방성) + 드문 jump.

추가로 검토 중인 `TQC` 내부 method 변경(risk-aware actor, adaptive truncation 등)은 위 인프라 위에서 진행하며, 범위와 절차는 [Paper Preparation Guide](docs/experiments/paper_preparation_guide.md)와 [Experiment Protocol](docs/experiments/experiment_protocol.md)를 따른다.

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
