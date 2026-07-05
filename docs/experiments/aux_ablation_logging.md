# Auxiliary-Prediction Ablation 로깅 (AUX_ABLATION)

`aux_prediction.enabled=false` vs `=true`를 동일 조건에서 논문 수준으로 비교하는 방법.
이 로깅은 run-identity 컬럼 + 새 artifact 2개만 추가할 뿐, 학습 루프·reward·curriculum은
바꾸지 않는다.

## 무엇이 어디에 기록되나

모든 것은 run의 `logs/` 디렉터리에 저장된다.

| 파일 | 답하는 질문 | 핵심 컬럼 |
|------|-----------------|-------------|
| `tqc_metrics.json` (스텝별) | **aux loss curve** + aux loss가 떨어졌는지; 학습 동역학 | `step, seed, aux_enabled, aux_version, loss/critic, loss/actor, values/Q, values/Q_max, aux/loss, aux/risk_mse, aux/min_dist_mse?, aux/risk_quantile?` |
| `eval_summary_<run>.csv` | **동일 timestep aux on/off** + 학습곡선 (eval 한 줄당 한 행) | `seed, aux_enabled, aux_version, eval_global_t, curriculum_stage, eval_eps, success_rate, collision_rate, timeout_rate, mean_reward, mean_final_goal_dist, spl, cte, jerk` |
| `curriculum_episode_rewards_<run>.csv` | stage 포함 episode별 reward / 결과 | 기존 컬럼 `+ seed, aux_enabled, aux_version` |
| `episode_driving_<run>.csv` | episode별 주행 품질 | 기존 컬럼 `+ seed, aux_enabled, aux_version` |
| `run_manifest.json` | **run 설정** (seed 혼동 방지) | `seed, aux_enabled, aux_version, num_sectors, horizons_sec, loss_weight, min_distance_loss_weight, use_distributional_aux, temporal_enabled, train_config_file, environment_config_file, environment_config_sha1, env_aux, file_name, git_commit` |
| `generalization_<world>_<run>.csv` | **미학습 world / condition** 비교 | 기존 컬럼 `+ seed, aux_enabled, aux_version` |
| `aux_ablation_summary.csv` (직접 생성) | seed 전반 **최종 mean/std 표** | `group, n_runs, <metric>_mean, <metric>_std` |

참고:
- aux-disabled run도 동일 파일 구조를 유지한다. `aux_enabled=0`이고 `aux/*` JSON 키는 단순히
  생략된다(null-safe)라 아무것도 깨지지 않는다.
- 기존 논문 CSV(`episode_metrics_*.csv`, `eval_metrics_*.csv`)는 변경 없음. `eval_summary_*.csv`는
  eval 데이터의 aux-비교 뷰다.
- manifest의 `environment_config_file` / `environment_config_sha1` / `env_aux`는 **실행 중인 env 노드**에서
  온다(env가 `loaded_config_path`, `loaded_config_sha1`, `aux_*`를 `/gym_node`의 ROS 파라미터로 노출,
  trainer가 읽음). 따라서 manifest는 env가 **실제로 로드한** 설정과 **실제로 돌고 있는** 기하를 기록한다 —
  trainer가 추측한 경로가 아니다. 해시가 "같은 경로, 다른 내용"을 잡는다.

## 어떤 질문 → 어떤 파일

1. **최종 성능이 향상됐나?** → `aux_ablation_summary.csv`
   (`success_rate_mean`, `spl_mean`, `collision_rate_mean`, ...).
2. **같은 timestep에서 더 빠른가?** → `eval_summary_<run>.csv` 곡선, 또는 고정 timestep은
   `aux_ablation_summary.csv --eval-step <T>`.
3. **aux loss가 실제로 떨어졌나?** → `tqc_metrics.json`의 `aux/loss` / `aux/risk_mse` vs `step`
   (aux run만).
4. **일반화가 다른가?** → `generalization_<world>_<run>.csv` (`aux_enabled`로 그룹,
   `world` / `condition_stage`별 비교).

## Ablation 실행 (각 >= 3 seed)

aux off/on으로 seed를 바꿔가며 동일 학습을 돌린다. aux는 `hyperparameters_tqc.yaml`(agent)과
`environment_curriculum.yaml`(env) **양쪽**에서 `aux_prediction.enabled: true`로 켠다 —
`num_sectors` / `horizons_sec`를 동일하게 유지(아니면 trainer가 fail-fast).

```bash
# aux OFF, seed 0,1,2  (양쪽 config에서 aux_prediction.enabled=false)
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p seed:=0
# ... seed 1, 2

# aux ON, seed 0,1,2   (양쪽 config에서 aux_prediction.enabled=true)
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p seed:=0
# ... seed 1, 2
```

각 run은 위 파일들이 담긴 자체 `logs/`를 쓴다.

## 집계

`aux_ablation_summary.py`는 각 run의 `eval_summary_*.csv` 옆에 있는 `run_manifest.json`을 읽어
**전체 aux config**로 그룹핑하고(서로 다른 aux 설정이 조용히 평균되지 않도록) mean/std를 보고한다.

```bash
cd ros2_ws/src/drl_agent/scripts/utils

# 기본: 전체 aux config signature로 그룹 (run_manifest.json 기준)
python3 aux_ablation_summary.py /path/to/all_runs --out logs/aux_ablation_summary.csv

# loose: aux on/off로만 그룹, 나머지 config는 무시 (manifest 불필요)
python3 aux_ablation_summary.py /path/to/all_runs --group-by aux_enabled \
    --out logs/aux_ablation_summary_loose.csv

# strict: 읽을 수 있는 manifest가 없는 run이 있으면 중단
python3 aux_ablation_summary.py /path/to/all_runs --strict-manifest

# 동일 timestep 비교 (t=100000에 가장 가까운 eval):
python3 aux_ablation_summary.py /path/to/all_runs --eval-step 100000 \
    --out logs/aux_ablation_summary_100k.csv

# git commit으로도 그룹 분리:
python3 aux_ablation_summary.py /path/to/all_runs --include-git
```

출력 `aux_ablation_summary.csv`:

```
group,config_signature,n_runs,success_rate_mean,success_rate_std,...
aux_off,aux0|v1|K16|H[0.5,1,1.5]|lw0.1|mdlw0.0|distr0|temp0,3,0.72,0.02,...
aux_on,aux1|v1|K16|H[0.5,1,1.5]|lw0.1|mdlw0.0|distr0|temp0,3,0.8233,0.0252,...
```

동작:
- `--group-by full`(기본): 서로 다른 aux config마다 한 그룹; `num_sectors` / `horizons_sec` /
  `loss_weight` / ... 가 다른 run은 별도 행으로 들어간다(`config_signature` 컬럼이 정확한 config를 표시).
- `--group-by aux_enabled`: `aux_off` / `aux_on`으로만 축약(`config_signature` 비움), manifest 불필요.
- `--strict-manifest`: 읽을 manifest가 없는 run이 있으면 에러.
- manifest 없는 run(non-strict, full)은 `config_signature='(no-manifest)'`로 묶여 manifest run과 분리.
- 입력은 run 디렉터리(`eval_summary_*.csv`를 재귀 탐색) 또는 직접 CSV 경로 가능; 각 run은 대표 eval 행
  하나(최종, 또는 `--eval-step`에 가장 가까운 것)를 기여. 그룹의 run이 < 2이면 경고.

## 논문용

- 표: `aux_ablation_summary.csv`(aux_off vs aux_on의 success/collision/SPL/CTE/jerk mean ± std;
  sample-efficiency 표는 `--eval-step` 행 추가).
- 학습곡선: `eval_summary_<run>.csv`(`success_rate` / `spl` vs `eval_global_t`) 플롯, run당 한 선,
  `aux_enabled`로 그룹.
- Aux-loss 그림: aux run의 `tqc_metrics.json` `aux/loss` vs `step` 플롯.
- 일반화 표: `generalization_<world>_<run>.csv`를 `aux_enabled`와 `world` / `condition_stage`로 그룹.
- `run_manifest.json`은 부록용으로 각 run의 정확한 config를 기록한다.
