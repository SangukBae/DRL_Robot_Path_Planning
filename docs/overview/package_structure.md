# Package Structure & Experiment Profiles

강화학습 코드의 새 구조(2026-07 개편)와 profile 기반 실험 실행 정리. 요점 중심 — 상세 명령은 각 절의 코드 블록 참조.

## 구조

```
ros2_ws/src/
├── drl_agent/                     # ROS 패키지 (ament_cmake)
│   ├── drl_agent/                 # importable Python package — 구현의 canonical 위치
│   │   ├── config/                #   schema/loader/validation + paths.py(←config_paths)
│   │   ├── training/              #   registry/run_manager/run_layout + train_tqc_base.py,
│   │   │   │                      #   train_tqc_curriculum.py(←train_tqc_curriculum_agent),
│   │   │   │                      #   train_rl.py, gym_parameter_client.py, episode_metrics.py,
│   │   │   │                      #   aux_ablation_logging.py / aux_eval_metrics.py / dynamic_avoidance_log.py
│   │   │   └── curriculum/        #   stage_logic / metrics / state_io / eval_runner / aux_eval (←curriculum_*)
│   │   ├── rl/
│   │   │   ├── networks/          #   tqc.py(←tqc_networks) / action_risk_head / aux_prediction /
│   │   │   │                      #   aux_losses(←aux_prediction_losses) / aux_temporal(←aux_prediction_temporal)
│   │   │   ├── replay/            #   buffer.py(←buffer, LAP) + schema.py (npz 필드 계약)
│   │   │   ├── checkpointing/     #   manager.py (탐색·검증) + tqc_io.py (실제 save/load)
│   │   │   └── algorithms/        #   tqc/ sac/ td7/ a3c/ tqc_ieqn/ (각 agent.py) + sb3/{sac,td3}.py
│   │   ├── evaluation/            #   generalization_eval / risk_map_eval / real_policy_runner /
│   │   │                          #   sim_validation / risk_map_dump
│   │   ├── env/                   #   environment_interface.py +
│   │   │   ├── simulation/        #   environment.py(본체) / gazebo_* / map_* / zone_tracker /
│   │   │   │                      #   collision_checker / localization_noise
│   │   │   ├── curriculum/        #   environment_curriculum.py(본체)
│   │   │   ├── observation/       #   observation_builder / obs_time_context / aux_prediction_labels
│   │   │   ├── rewards/           #   reward_calculator
│   │   │   ├── spawning/          #   start_sampler / goal_sampler / obstacle_catalog_spawner
│   │   │   └── humans/            #   human_spawn_sampler / human_motion_manager / dynamic_avoidance_telemetry
│   │   ├── common/                #   compat / geometry_utils / seed_utils / file_manager /
│   │   │                          #   pure_pursuit / point_cloud2
│   │   └── nodes/                 #   train_node.py / environment_curriculum_node.py / eval_node.py / real_policy_node.py
│   ├── scripts/                   # 이관된 모듈은 bare-name shim / exec wrapper만 잔존.
│   │                              # 실제 구현이 남은 것: environment_360, sb3_ppo, per-algo
│   │                              # train_* baseline 스크립트, plotting/분석 유틸, test_*_agent
│   ├── config/                    # legacy 기본 config (보존)
│   └── runtime/                   # 학습 산출물 (gitignored, 보존)
└── drl_experiments/               # 실험 정의 패키지
    ├── profiles/phase2/{baseline,reward_shaping_only,action_risk_head_only,both}/
    │                              #   profile.yaml + config 4종 (self-contained)
    ├── sweeps/                    # phase2_seeds.yaml / paper_main.yaml
    ├── scripts/                   # run_profile.py / resume_profile.py / aggregate.py / export_tables.py
    └── outputs/                   # gitignored 결과물 공간
```

**마이그레이션 규칙**: legacy flat 모듈(`scripts/*/*.py`, bare-name import)은
`drl_agent/` 패키지로 이동했고, 옛 경로에는 자기 자신을 패키지 모듈로 aliasing하는
shim(`sys.modules[__name__] = _impl`)이나 — ros2 run 엔트리포인트의 경우 —
`_impl.main()`을 호출하는 exec wrapper가 남아 있다. 따라서 bare-name import와
기존 `ros2 run drl_agent <name>.py` 실행 경로는 전부 그대로 동작한다.
핵심 RL/트레이너/환경 코드는 위 트리에 표시된 canonical 위치가 실제 구현이다
(마이그레이션 불변식 테스트: `tests/test_package_migration.py`).
새 모듈은 처음부터 패키지 위치에 작성한다 (예: `drl_agent/rl/networks/<new>.py`).
남은 legacy 구현: `environment_360.py`(classic Gazebo), `sb3_ppo_agent.py` +
`train_sb3_ppo_agent.py`, per-algo `train_{sac,td7,a3c,tqc_ieqn,sb3_*}[_curriculum]_agent.py`
baseline 트레이너, `test_{tqc,td7}_agent.py`(live-sim run 스크립트),
plotting/분석 유틸(`plot_*`, `aggregate_results`, `analyze_*`, `aux_ablation_summary`,
`sim_validation_summary`, `check_reproducibility`)과 `sim_validation_runner.py`.

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
검증 실패 시 실행 거부. 통과 시 wrapper가 **기존 legacy 실행 파일을 그대로 exec**
(`train_config_file:=<profile dir>` 변환) — 학습 로직/체크포인트 호환성은 불변.
profile 신원(이름, config sha256, override)은 run dir의
`configs/profile_manifest.json`으로 기록된다.

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
```

## Legacy 호환 (전부 유지)

| Legacy | 상태 |
|--------|------|
| `ros2 run drl_agent train_tqc_curriculum_agent.py` (+ 모든 `train_*` / `environment*.py`) | 그대로 동작 (wrapper가 이들을 exec) |
| `-p train_config_file:=<file 또는 dir>` / `-p config_file:=...` | 그대로 동작 |
| `runtime/phase2_configs/<MODE>/` | 보존 (canonical은 `profiles/phase2/` — 내용 동일 복사본) |
| `runtime/experiments/`, `runtime/tqc/seed_N/` 기존 run/checkpoint | 보존 — resume 우선순위 로직 불변 (`run_layout.py`) |
| `import config_paths` 등 bare-name import | shim으로 유지 — 설치 환경에서도 env hook(`env-hooks/flat_legacy_scripts.dsv.in`)이 `lib/drl_agent`를 PYTHONPATH에 추가하므로 `source install/setup.bash` 후 일반 python에서도 동작 (검증: `tests/test_installed_bare_imports.py`) |

주의: `runtime/phase2_configs/`와 `profiles/phase2/`는 별도 파일이므로 config를
수정할 땐 `profiles/phase2/`(canonical)를 수정할 것.
