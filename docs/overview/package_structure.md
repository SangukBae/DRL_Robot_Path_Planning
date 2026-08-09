# Package Structure & Experiment Profiles

강화학습 코드의 canonical 구조(2026-07 개편, 완료)와 profile 기반 실험 실행 정리.
요점 중심 — 상세 명령은 각 절의 코드 블록 참조.

## 구조

```
ros2_ws/src/
├── drl_agent/                     # ROS 패키지 (ament_cmake)
│   ├── drl_agent/                 # importable Python package — 구현의 유일한 위치
│   │   ├── config/                #   schema/loader/validation + paths.py
│   │   ├── training/              #   registry/run_manager/run_layout + train_tqc_base.py,
│   │   │   │                      #   train_tqc_curriculum.py, train_rl.py, gym_parameter_client.py,
│   │   │   │                      #   episode_metrics.py, aux_ablation_logging.py, aux_eval_metrics.py,
│   │   │   │                      #   dynamic_avoidance_log.py
│   │   │   ├── curriculum/        #   stage_logic / metrics / state_io / eval_runner / aux_eval
│   │   │   └── baselines/         #   {sac,td7,a3c,tqc_ieqn,sb3_sac,sb3_td3,sb3_ppo}[_curriculum].py
│   │   │                          #   — 논문 비교군/ablation 모델 (PPO는 curriculum 변형 없음)
│   │   ├── rl/
│   │   │   ├── networks/          #   tqc.py / action_risk_head / aux_prediction /
│   │   │   │                      #   aux_losses / aux_temporal
│   │   │   ├── replay/            #   buffer.py (LAP) + schema.py (npz 필드 계약)
│   │   │   ├── checkpointing/     #   manager.py (탐색·검증) + tqc_io.py (실제 save/load)
│   │   │   └── algorithms/        #   tqc/ sac/ td7/ a3c/ tqc_ieqn/ (각 agent.py) + sb3/{sac,td3,ppo}.py
│   │   ├── evaluation/            #   generalization_eval / risk_map_eval / real_policy_runner /
│   │   │   │                      #   sim_validation_runner / sim_validation / risk_map_dump
│   │   │   ├── live/              #   tqc_live_runner.py / td7_live_runner.py (수동 live-sim 실행)
│   │   │   └── analysis/          #   aggregate_results / analyze_{aux_correlation,yield_freezing} /
│   │   │                          #   aux_ablation_summary / check_reproducibility /
│   │   │                          #   plot_{metrics,reward,trajectories_on_map} / sim_validation_summary
│   │   ├── env/                   #   environment_interface.py +
│   │   │   ├── simulation/        #   environment.py(본체) / environment_360.py(classic Gazebo) /
│   │   │   │                      #   gazebo_* / map_* / zone_tracker / collision_checker / localization_noise
│   │   │   ├── curriculum/        #   environment_curriculum.py(본체)
│   │   │   ├── observation/       #   observation_builder / obs_time_context / aux_prediction_labels
│   │   │   ├── rewards/           #   reward_calculator
│   │   │   ├── spawning/          #   start_sampler / goal_sampler / obstacle_catalog_spawner
│   │   │   └── humans/            #   human_spawn_sampler / human_motion_manager / dynamic_avoidance_telemetry
│   │   ├── common/                #   compat(소스 루트 탐색) / geometry_utils / seed_utils /
│   │   │                          #   file_manager / pure_pursuit / point_cloud2
│   │   └── nodes/                 #   train_node.py / environment_curriculum_node.py / eval_node.py / real_policy_node.py
│   ├── config/                    # 기본 config (보존)
│   └── runtime/                   # 학습 산출물 (gitignored, 보존)
└── drl_experiments/               # 실험 정의 패키지
    ├── profiles/
    │   └── phase2/{tqc_vanilla,baseline,reward_shaping_only,action_risk_head_only,both,
    │               both_legacy,both_trajrisk_rbs,obs_norm_optim_split}/
    │                              #   profile.yaml + config 4종 (self-contained)
    ├── sweeps/                    # phase2_seeds.yaml / paper_main.yaml
    ├── scripts/                   # run_profile.py / resume_profile.py / aggregate.py / export_tables.py
    └── outputs/                   # gitignored 결과물 공간
```

**이관 완료(2026-07)**: `scripts/{policy,environment,utils}` flat 레거시 디렉터리는
완전히 삭제됐다. `import drl_agent.<...>` 형태의 명시적 dotted import가 유일한
import 경로이며, bare-name import(`import buffer`, `from tqc_agent import Agent`
등)나 legacy alias shim은 더 이상 존재하지 않는다. `ros2 run drl_agent
train_tqc_curriculum_agent.py`처럼 옛 파일명을 직접 실행하던 경로도 공식적으로
중단됐다 — 대신 canonical 모듈이 자신의 파일명으로 `install(PROGRAMS ...)`에
등록되어 있다(예: `train_tqc_curriculum.py`, `sac_curriculum.py`). 새 모듈은
처음부터 위 트리의 canonical 위치에 작성한다.

구조/import 불변식은 `tests/test_package_migration.py`(canonical import 성공 +
retired bare name 전부 `ModuleNotFoundError` + 순환 import 없음)와
`tests/test_installed_canonical_imports.py`(설치된 환경에서 동일 계약 확인,
워크스페이스 자체를 subprocess 안에서 source하여 검증)가 고정한다.

## Profile 시스템

profile = `profile.yaml`(manifest) + 참조하는 config 4종을 담은 폴더.
manifest 예시 (`drl_experiments/profiles/phase2/both/profile.yaml`):

```yaml
profile:
  name: phase2/both
  algorithm: tqc
  trainer: curriculum
  environment: environment_curriculum.yaml
  train: train_tqc_config.yaml
  hparams: hyperparameters_tqc.yaml
  curriculum: train_tqc_curriculum_config.yaml
  output_prefix: tqc_phase2_both
  overrides:
    risk_map_reward_enabled: true
    action_risk_head_enabled: true
```

실행 전 `ConfigValidator`가 검증: 파일 존재, env/agent `action_risk_head.enabled`
일치, `risk_map_reward`는 env config 기준으로 기록, `output_prefix` ==
`base_file_name`, resume 시 checkpoint/replay/curriculum_state 실존 여부.
검증 실패 시 실행 거부. 통과 시 wrapper(`drl_agent/nodes/_node_common.py`)가
`drl_agent/training/registry.py`에서 `(algorithm, trainer)` 쌍에 대응하는
**canonical 모듈**을 `importlib.util.find_spec`으로 찾아 그 파일을
`python3 <resolved path>`로 exec한다(`train_config_file:=<profile dir>` 변환) —
학습 로직/체크포인트 호환성은 불변. profile 신원(이름, config sha256, override)은
run dir의 `configs/profile_manifest.json`으로 기록된다.

## 명령어

```bash
# 시뮬레이션 + 환경 노드 + 학습 (터미널 3개)
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false
ros2 run drl_agent environment_curriculum_node.py --ros-args -p profile:=phase2/both
ros2 run drl_agent train_node.py --ros-args -p profile:=phase2/both -p seed:=0

# 재개 (사전 검증 포함)
ros2 run drl_agent train_node.py --ros-args -p profile:=phase2/both -p seed:=0 -p resume:=true

# config-only 검증 / 목록 / sweep (ROS 불필요)
python3 ros2_ws/src/drl_experiments/scripts/run_profile.py phase2/both --validate-only
python3 ros2_ws/src/drl_experiments/scripts/run_profile.py --list
python3 ros2_ws/src/drl_experiments/scripts/run_profile.py --sweep ros2_ws/src/drl_experiments/sweeps/phase2_seeds.yaml

# 평가 / 집계 / TensorBoard
ros2 run drl_agent eval_node.py --ros-args -p profile:=phase2/both -p weight_prefix:=<prefix>
python3 ros2_ws/src/drl_experiments/scripts/aggregate.py
tensorboard --logdir <run_dir>/logs

# profile 없이 개별 baseline을 직접 실행 (전부 canonical 모듈, 자기 파일명으로 설치됨)
ros2 run drl_agent sac_curriculum.py
ros2 run drl_agent train_rl.py --ros-args -p rl_model:=td7_curriculum   # 또는 registry 경유
```

## 호환성 보존 사항

| 항목 | 상태 |
|------|------|
| `-p train_config_file:=<file 또는 dir>` / `-p config_file:=...` | 그대로 동작 |
| `runtime/phase2_configs/<MODE>/` | 보존된 legacy runtime snapshot. 현재 canonical은 `profiles/phase2/`이며 내용이 항상 같다고 가정하지 않는다. |
| `runtime/experiments/`, `runtime/tqc/seed_N/` 기존 run/checkpoint | 보존 — resume 우선순위 로직 불변 (`run_layout.py`), checkpoint/replay 포맷 무변경 |
| 논문 비교군/ablation 모델 구현 (SAC, TD7, A3C, TQC+IEQn, SB3-SAC/TD3/PPO) | 전부 `drl_agent/training/baselines/`·`drl_agent/rl/algorithms/`에 canonical 코드로 보존, 삭제된 것 없음 |

| 중단된 항목 | 대체 경로 |
|------------|-----------|
| bare-name import (`import buffer`, `from tqc_agent import Agent` 등) | `import drl_agent.rl.replay.buffer`, `from drl_agent.rl.algorithms.tqc.agent import Agent` 등 dotted import |
| `ros2 run drl_agent train_tqc_curriculum_agent.py` 등 옛 파일명 | `ros2 run drl_agent train_tqc_curriculum.py` (또는 `train_node.py -p profile:=...`, `train_rl.py -p rl_model:=...`) |
| `scripts/{policy,environment,utils}/` 디렉터리 | 삭제됨 — 전체 내용이 `drl_agent/drl_agent/` 아래 canonical 위치로 이관 |

주의: `runtime/phase2_configs/`와 `profiles/phase2/`는 별도 파일이므로 현재 실험 config를
수정할 땐 `profiles/phase2/`(canonical)를 수정할 것. `runtime/phase2_configs/`는 기존 run 호환용으로
남겨 둔 snapshot이며 새 profile split(`both_legacy`, `both_trajrisk_rbs`, `tqc_vanilla`)을 대표하지 않는다.
