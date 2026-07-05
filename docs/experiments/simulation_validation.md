# Simulation Validation (localization-aware RL)

**SIM_VALIDATION / VALIDATION_ONLY.** This document and the code it references
exist only to verify the localization-aware framework *in simulation* (logic, not
hardware). Everything is OFF by default and easy to remove (see §6).

## What it verifies
1. localization noise is injected as intended,
2. reward/done are correctly separated onto ground-truth (GT) pose,
3. no reset → first-step observation jump,
4. gt/loc/proprio separation → stale handling,
5. curriculum stages ramp the noise as intended,
6. with noise OFF, behaviour is identical to the GT baseline.

## How to run

```bash
# 1) Gazebo
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false

# 2) Environment WITH validation logging on (curriculum or plain):
ros2 run drl_agent environment_curriculum.py --ros-args -p enable_sim_validation_logging:=true
#   (plain env: ros2 run drl_agent environment.py --ros-args -p enable_sim_validation_logging:=true)

# 3) Short validation driver (no training, ~5 episodes). Optional stage sweep:
ros2 run drl_agent sim_validation_runner.py --ros-args -p episodes:=5 -p max_steps:=80
ros2 run drl_agent sim_validation_runner.py --ros-args -p episodes:=3 -p stages:="[0,2,4]"

# 4) Summary (console + JSON)
python3 ros2_ws/src/drl_agent/scripts/utils/sim_validation_summary.py \
  --log-dir <run_dir>/logs
```

### Scenario → how to configure
| # | scenario | how |
|---|---|---|
| 1 | noise off (clean) | curriculum stage 0–3, or plain env (default off) |
| 2 | weak goal noise (gaussian + delay) | curriculum stage 4 (`-p stages:="[4]"`) |
| 3 | drift goal noise | stage 8 (`-p stages:="[8]"`) |
| 4 | strongest train (drift + jump) | stage 9 (`-p stages:="[9]"`, `robustness_train`) |
| 5 | gt=loc=proprio same topic | default (single `/odometry`) |
| 6 | separate topics | env: `-p gt_odom_topic:=/odometry -p loc_odom_topic:=/loc_odom -p proprio_odom_topic:=/odometry` |
| 7 | stage 0/4/9 compare | `-p stages:="[0,4,9]"` |

## Files to look at
- `<run_dir>/logs/loc_validation_step_<tag>.csv` — per step: `obs_*` vs `gt_*`,
  `reward/done_goal_dist_used`, `loc_raw/loc_est/gt` poses, `use_gt_for_*`,
  per-role `odom_*_count`, `stale_*`, `loc_noise_enabled/delay/sigma/jump`.
- `<run_dir>/logs/loc_validation_reset_<tag>.csv` — per episode:
  `reset_obs_*`, `first_step_obs_*`, `reset_first_step_*_jump`.
- `<run_dir>/logs/validation_summary.json` — aggregated metrics.

## What a healthy result looks like
- **noise OFF**: `noise_off_regression_ok = true`; `mean_abs_goal_dist_error ≈ 0`;
  `noise_off_max_goal_dist_error < 1e-6`; reset jumps `= 0`. → identical to baseline.
- **noise ON**: `mean_abs_goal_dist_error > 0` (obs ≠ gt), but
  `fraction_reward_uses_gt_consistently = 1.0` and
  `fraction_done_uses_gt_consistently = 1.0` (reward/done still on GT).
- **reset consistency**: `max_reset_first_step_goal/heading_jump` small (≈ the
  per-step motion, NOT a bias-sized step) — confirms no clean→noisy jump.
- **curriculum ramp** (`per_stage`): stage 0–3 `clean` (`enabled=0`); stage 4–7
  `weak_goal_noise` (`enabled=1`, gaussian + delay); stage 8 `drift_goal_noise`
  (adds drift); stage 9 `robustness_train` (adds rare `jump_prob>0`).
  `mean_abs_goal_dist_error` increases with stage.
- **stale handling**: same-topic → `episodes_with_stale_loc/proprio = 0`;
  separate topic stopped → those counts > 0 and the env logs a `[reset] odom
  source(s) did not refresh` warning.

## If something fails — suspect
- `noise_off_regression_ok = false` → loc emulator leaking when disabled, or
  `obs`/`gt` pose caches crossed. Check `_on_odom` role routing.
- reward/done consistency < 1.0 → `use_gt_for_*` not honoured in `step_callback`.
- large reset jump → `_reset_localization` not seeding the bias / `reset_callback`
  not patching `agent_state[0:2]`.
- `per_stage` noise flat across stages → curriculum `localization` override not
  applied (check stage YAML + `_apply_curriculum_stage` base-reset/merge).
- unexpected `stale_*` with one topic → odom QoS / topic name mismatch.

## §6 Removing the validation feature
All validation code is tagged `SIM_VALIDATION` / `VALIDATION_ONLY` and gated by
`enable_sim_validation_logging` (default false). To remove:
- delete `scripts/utils/sim_validation.py`, `scripts/utils/sim_validation_summary.py`,
  `scripts/policy/sim_validation_runner.py`, this doc;
- `grep -rn SIM_VALIDATION ros2_ws/src/drl_agent` → remove the 3 guarded hooks in
  `environment.py` (param declare + logger init, `log_step` in `step_callback`,
  `note_reset` in `reset_callback`) and the CMakeLists install lines.
Leaving the flag false already yields the original behaviour with zero overhead.
