# Experiment Protocol (Paper Comparison)

논문 비교 실험을 동일 조건에서 재현하기 위한 프로토콜이다.
**알고리즘 내부 로직은 변경하지 않은 상태**의 시스템 구축/통일 결과를 정리한다.

## 1. 공통 학습 protocol (6개 비교군 전부 동일)

| 항목 | 값 | 비고 |
|------|----|------|
| `seed` | 0 (기본) | 실행 시 override (아래 §3) |
| `max_episode_steps` | 600 | 10 Hz · 60초 episode |
| `max_timesteps` | 2,000,000 | |
| `load_model` | false | fresh start (특정 run 이어갈 때만 true) |
| `use_checkpoints` | false | step-wise off-policy update |
| `eval_freq` | 12,000 | |
| `eval_eps` | 10 | |
| `timesteps_before_training` | 12,000 | warmup |

curriculum 공통 설정: `min_stage_steps=30000`, `min_stage_episodes=20`,
`pass_eval_success_rate=[0.90,0.85,0.75,0.70]`, `pass_eval_collision_rate=[0.05,0.10,0.15,0.20]`,
`consecutive_eval_passes=2`.

비교군(curriculum trainer): **TQC, TQC+IEQN, SAC, TD7, SB3-SAC, SB3-TD3**.
필수 baseline은 SAC/TD3/TQC, 권장 추가는 TQC+IEQN/TD7.

## 2. 평가 / 로그 스키마 (6개 전부 동일)

`evaluate_and_print()` 반환 dict + 모든 CSV는 6개 비교군에서 동일 스키마다.

### 2.1 논문 핵심 지표 CSV (신규)

공유 모듈 `utils/episode_metrics.py`가 `(state, action)` 스트림에서 안전/네비/제어 지표를
계산한다 (ROS·env·TQC 변경 없음). path length는 속도 적분, CTE는 odom dead-reckoning 추정.

- **`eval_metrics_*.csv`** (19컬럼, **논문 표의 1순위 소스** — 결정론적 eval):
  `epoch, global_t, curriculum_stage, eval_eps, mean_reward, std_reward,
   success_rate, collision_rate, timeout_rate, mean_goal_dist,
   path_length_m, spl, mean_heading_error_rad, mean_cross_track_error_m,
   mean_action_jerk, mean_steering_change_rad, near_collision_count,
   travel_time_s, mean_speed_mps`
- **`episode_metrics_*.csv`** (17컬럼, 학습 곡선·sample-efficiency용 — 매 학습 episode):
  `episode, global_t, curriculum_stage, success, collision, timeout,
   total_reward, steps,` + 위 9개 지표.

`evaluate_and_print()`는 위 9개 지표(eval episodes 평균)를 반환 dict에 추가로 포함한다.

### 2.2 기존 CSV (유지)

- `curriculum_episode_rewards_*.csv` (12컬럼): `episode, global_t, steps, total_reward,
  mean_reward, goal_reached, collision, timeout, eval_cut, final_goal_dist_m,
  curriculum_stage, mean_gazebo_rtf`
- `episode_driving_*.csv` (13컬럼): 주행 품질(속도/조향/clearance). `mean_gazebo_rtf`는
  TQC만 실제 추적, 나머지 `NaN`.

## 3. 다중 seed 실행 (논문 필수: 최소 3, 권장 5)

config를 수정하지 않고 실행 시점에 seed를 override 한다. 우선순위:
**ROS2 파라미터 `seed` > 환경변수 `DRL_AGENT_SEED` > YAML 기본값**.

```bash
# 예: TQC를 seed 0,1,2로 sweep (각 실행 전 environment_curriculum.py 가동 필요)
for s in 0 1 2; do
  ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p seed:=$s
done

# 환경변수 방식도 동일
DRL_AGENT_SEED=1 ros2 run drl_agent train_sac_curriculum_agent.py
```

각 seed는 자동으로 분리된 디렉토리에 저장된다:

```
runtime/<algo>/seed_<N>/
├── pytorch_models/        # 학습 중 체크포인트
├── final_models/          # 최종 모델
├── results/               # evals.npy (학습 곡선)
└── logs/                  # *_episode_rewards / *_driving / curriculum_state.json
```

> `curriculum_state.json`은 고정 파일명이라 seed별 디렉토리 분리가 필수다
> (분리하지 않으면 seed 간 resume 상태가 덮어써진다).

`run_dir` 파라미터나 `DRL_AGENT_RUN_DIR`로 경로를 직접 지정하면 그 경로를 **그대로** 쓴다
(seed 분리 자동 적용 안 됨 → 다중 seed라면 경로에 seed를 직접 포함해야 함).

### Resume / 기존 run 마이그레이션

per-seed 디렉토리 구조는 **기본 경로(`runtime/<algo>/`)에만** 적용된다. 따라서:

- **명시적 `run_dir`로 만든 기존 run** (예: `runtime/tqc_state_80_...`)은 영향 없음 —
  이어서 학습하려면 그 경로를 그대로 지정하고 `load_model: true`로 둔다:
  ```bash
  ros2 run drl_agent train_tqc_curriculum_agent.py \
    --ros-args -p run_dir:=<기존 경로> -p seed:=0
  # (train_tqc_config.yaml 에서 load_model: true)
  ```
- **기본 경로(`runtime/tqc/`)에 있던 옛 run**을 새 구조로 옮기려면 한 번만 이동:
  ```bash
  mkdir -p runtime/tqc/seed_0 && mv runtime/tqc/{pytorch_models,final_models,results,logs} runtime/tqc/seed_0/
  ```
- 논문용 신규 run은 `load_model: false`(기본)라 항상 fresh start이므로 마이그레이션 불필요.

## 4. 후처리 (논문 표/그래프) — `aggregate_results.py`

seed별 `eval_metrics_*.csv`를 모아 mean±std 표 / 학습곡선 / sample-efficiency를 자동 생성:

```bash
python3 ros2_ws/src/drl_agent/scripts/utils/aggregate_results.py \
  --runtime-root ros2_ws/src/drl_agent/runtime \
  --algos tqc sac sb3_td3 td7 tqc_ieqn sb3_sac \
  --thresholds 0.5 0.7 0.8 0.9
# ros2 run drl_agent aggregate_results.py --ros-args 형태로도 실행 가능
```

`<runtime-root>/<algo>/seed_<N>/logs/eval_metrics_*.csv`를 자동 탐색(seed별 최신 파일)하여
`<runtime-root>/_aggregate/`에 생성:

- `final_eval_summary.csv` — algo별, 마지막 eval의 모든 지표 mean/std (논문 표 1)
- `eval_curve_<metric>.csv` — algo별·epoch별 mean/std (success/collision/timeout/spl/reward 곡선)
- `sample_efficiency.csv` — success_rate 임계값별 `n_hit/n_total`(도달 seed 수) +
  `steps_mean_hit`(도달 seed만 평균) + `steps_mean_censored`(미도달 seed를 마지막
  step으로 right-censoring한 평균). 미도달 seed를 빼고 평균내던 편향을 제거했다.
  `--censor-at <global_t>`로 censoring 기준(예: max_timesteps)을 지정할 수 있다.

## 5. Generalization 평가 — `generalization_eval.py`

학습된 모델을 **재학습 없이** 여러 조건에서 deterministic 평가한다. 조건 = curriculum
stage(장애물 density·동적 속도·perception noise를 묶음 → density/noise shift 축). world
축은 Gazebo 실행 시 선택하므로 `--world` 라벨로 기록한다.

```bash
# 1) Gazebo를 평가할 world로 실행  2) environment_curriculum.py 실행  3) 아래 실행
ros2 run drl_agent generalization_eval.py --ros-args \
  -p weight_prefix:=tqc_agent_seed_0_20260601 \
  -p weights_dir:=<run_dir>/final_models \
  -p world:=aws_hospital \
  -p eval_eps_override:=20 \
  -p conditions:="[-1]"      # [-1]=모든 stage, 또는 [3,4] 같이 특정 stage만
```

→ `<run_dir>/logs/generalization_<world>_<tag>.csv` (stage별 success/collision/timeout +
SPL/CTE/jerk 등 전체 지표). unseen world = Gazebo를 다른 `world:=`로 재실행 후 같은 모델로
다시 돌리고 `--world` 라벨만 바꾸면 된다. (현재 harness는 TQC 모델 기준이며, 다른 algo는
해당 agent 클래스로 동일 패턴 확장 가능.)

## 6. 참고: 기존 `curriculum_episode_rewards_*.csv`

이 파일도 6개 비교군 × N seeds에서 동일 스키마라 별도 concat/groupby로 학습 로그 분석이
가능하다. 다만 논문 표/곡선의 1순위 소스는 `eval_metrics_*.csv`(결정론적 eval)다.
