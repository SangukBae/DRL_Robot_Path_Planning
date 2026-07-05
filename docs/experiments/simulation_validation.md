# Simulation Validation (localization-aware RL)

**SIM_VALIDATION / VALIDATION_ONLY.** 이 문서와 여기서 참조하는 코드는 localization-aware
프레임워크를 *시뮬레이션에서* 검증(하드웨어가 아니라 로직 검증)하기 위한 용도로만 존재한다.
모두 기본 OFF이며 제거하기 쉽다(§6 참고).

## 무엇을 검증하나
1. localization noise가 의도대로 주입되는지,
2. reward/done이 ground-truth(GT) pose에 올바르게 분리되어 있는지,
3. reset → first-step 관측 점프가 없는지,
4. gt/loc/proprio 분리 → stale 처리,
5. curriculum stage가 noise를 의도대로 ramp-up하는지,
6. noise OFF일 때 GT baseline과 동일하게 동작하는지.

## 실행 방법

```bash
# 1) Gazebo
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# 2) 검증 로깅을 켠 환경 노드 (curriculum 또는 plain):
ros2 run drl_agent environment_curriculum.py --ros-args -p enable_sim_validation_logging:=true
#   (plain env: ros2 run drl_agent environment.py --ros-args -p enable_sim_validation_logging:=true)

# 3) 짧은 검증 드라이버 (학습 없음, ~5 episode). stage sweep 옵션:
ros2 run drl_agent sim_validation_runner.py --ros-args -p episodes:=5 -p max_steps:=80
ros2 run drl_agent sim_validation_runner.py --ros-args -p episodes:=3 -p stages:="[0,2,4]"

# 4) 요약 (콘솔 + JSON)
python3 ros2_ws/src/drl_agent/scripts/utils/sim_validation_summary.py \
  --log-dir <run_dir>/logs
```

### 시나리오 → 설정 방법
| # | 시나리오 | 방법 |
|---|---|---|
| 1 | noise off (clean) | curriculum stage 0–3, 또는 plain env (기본 off) |
| 2 | weak goal noise (gaussian + delay) | curriculum stage 4 (`-p stages:="[4]"`) |
| 3 | drift goal noise | stage 8 (`-p stages:="[8]"`) |
| 4 | strongest train (drift + jump) | stage 9 (`-p stages:="[9]"`, `robustness_train`) |
| 5 | gt=loc=proprio 동일 topic | 기본값 (단일 `/odometry`) |
| 6 | topic 분리 | env: `-p gt_odom_topic:=/odometry -p loc_odom_topic:=/loc_odom -p proprio_odom_topic:=/odometry` |
| 7 | stage 0/4/9 비교 | `-p stages:="[0,4,9]"` |

## 확인할 파일
- `<run_dir>/logs/loc_validation_step_<tag>.csv` — 스텝별: `obs_*` vs `gt_*`,
  `reward/done_goal_dist_used`, `loc_raw/loc_est/gt` pose, `use_gt_for_*`,
  role별 `odom_*_count`, `stale_*`, `loc_noise_enabled/delay/sigma/jump`.
- `<run_dir>/logs/loc_validation_reset_<tag>.csv` — episode별:
  `reset_obs_*`, `first_step_obs_*`, `reset_first_step_*_jump`.
- `<run_dir>/logs/validation_summary.json` — 집계 지표.

## 정상 결과의 모습
- **noise OFF**: `noise_off_regression_ok = true`; `mean_abs_goal_dist_error ≈ 0`;
  `noise_off_max_goal_dist_error < 1e-6`; reset jump `= 0`. → baseline과 동일.
- **noise ON**: `mean_abs_goal_dist_error > 0`(obs ≠ gt)이지만
  `fraction_reward_uses_gt_consistently = 1.0`,
  `fraction_done_uses_gt_consistently = 1.0`(reward/done은 여전히 GT 기준).
- **reset 일관성**: `max_reset_first_step_goal/heading_jump`가 작음(≈ 스텝당 이동량이지 bias 크기의
  점프가 아님) — clean→noisy 점프가 없음을 확인.
- **curriculum ramp**(`per_stage`): stage 0–3 `clean`(`enabled=0`); stage 4–7
  `weak_goal_noise`(`enabled=1`, gaussian + delay); stage 8 `drift_goal_noise`(drift 추가);
  stage 9 `robustness_train`(드문 `jump_prob>0` 추가). `mean_abs_goal_dist_error`가 stage에 따라 증가.
- **stale 처리**: 동일 topic → `episodes_with_stale_loc/proprio = 0`; 분리 topic이 멈추면 → 해당 카운트가
  > 0이 되고 env가 `[reset] odom source(s) did not refresh` 경고를 남긴다.

## 실패 시 — 의심할 곳
- `noise_off_regression_ok = false` → disable 상태에서 loc emulator가 새거나, `obs`/`gt` pose 캐시가
  교차됨. `_on_odom`의 role routing 확인.
- reward/done 일관성 < 1.0 → `step_callback`에서 `use_gt_for_*`가 지켜지지 않음.
- 큰 reset 점프 → `_reset_localization`이 bias를 seed하지 않거나, `reset_callback`이
  `agent_state[0:2]`를 패치하지 않음.
- `per_stage` noise가 stage에 걸쳐 평탄 → curriculum `localization` override 미적용
  (stage YAML + `_apply_curriculum_stage`의 base-reset/merge 확인).
- 단일 topic인데 예상치 못한 `stale_*` → odom QoS / topic 이름 불일치.

## §6 검증 기능 제거
모든 검증 코드는 `SIM_VALIDATION` / `VALIDATION_ONLY` 태그가 붙어 있고
`enable_sim_validation_logging`(기본 false)로 gating된다. 제거하려면:
- `scripts/utils/sim_validation.py`, `scripts/utils/sim_validation_summary.py`,
  `scripts/policy/sim_validation_runner.py`, 이 문서를 삭제;
- `grep -rn SIM_VALIDATION ros2_ws/src/drl_agent` → `environment.py`의 guard된 hook 3개
  (param declare + logger init, `step_callback`의 `log_step`, `reset_callback`의 `note_reset`)와
  CMakeLists install 라인 제거.
플래그를 false로 두면 이미 오버헤드 0으로 원래 동작이 나온다.
