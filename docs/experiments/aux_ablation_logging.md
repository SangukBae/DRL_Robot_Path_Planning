# Auxiliary-Prediction Ablation Logging (AUX_ABLATION)

How to compare `aux_prediction.enabled=false` vs `=true` under identical
conditions at paper quality. The logging only adds run-identity columns + two
new artifacts; it does NOT change the training loop, reward, or curriculum.

## What gets logged where

Everything lands in the run's `logs/` directory.

| File | What it answers | Key columns |
|------|-----------------|-------------|
| `tqc_metrics.json` (step-wise) | **aux loss curve** + did aux loss drop; learning dynamics | `step, seed, aux_enabled, aux_version, loss/critic, loss/actor, values/Q, values/Q_max, aux/loss, aux/risk_mse, aux/min_dist_mse?, aux/risk_quantile?` |
| `eval_summary_<run>.csv` | **same-timestep aux on/off** + learning curve (one row per eval) | `seed, aux_enabled, aux_version, eval_global_t, curriculum_stage, eval_eps, success_rate, collision_rate, timeout_rate, mean_reward, mean_final_goal_dist, spl, cte, jerk` |
| `curriculum_episode_rewards_<run>.csv` | per-episode reward / outcome with stage | existing columns `+ seed, aux_enabled, aux_version` |
| `episode_driving_<run>.csv` | per-episode driving quality | existing columns `+ seed, aux_enabled, aux_version` |
| `run_manifest.json` | **run config** (so seeds are not mixed) | `seed, aux_enabled, aux_version, num_sectors, horizons_sec, loss_weight, min_distance_loss_weight, use_distributional_aux, temporal_enabled, train_config_file, environment_config_file, environment_config_sha1, env_aux, file_name, git_commit` |
| `generalization_<world>_<run>.csv` | **unseen world / condition** comparison | existing columns `+ seed, aux_enabled, aux_version` |
| `aux_ablation_summary.csv` (you generate it) | **final mean/std table** across seeds | `group, n_runs, <metric>_mean, <metric>_std` |

Notes:
- aux-disabled runs keep the same file structure; `aux_enabled=0` and the
  `aux/*` JSON keys are simply omitted (null-safe), so nothing breaks.
- The existing paper CSVs (`episode_metrics_*.csv`, `eval_metrics_*.csv`) are
  unchanged; `eval_summary_*.csv` is the aux-comparison view of the eval data.
- `environment_config_file` / `environment_config_sha1` / `env_aux` in the
  manifest come from the **running env node** (it exposes `loaded_config_path`,
  `loaded_config_sha1`, `aux_*` as ROS parameters on `/gym_node`, read by the
  trainer). So the manifest records the config the env ACTUALLY loaded and the
  geometry it is ACTUALLY running -- not a path the trainer guessed. The hash
  catches "same path, different content".

## Which questions -> which file

1. **Final performance improved?** -> `aux_ablation_summary.csv`
   (`success_rate_mean`, `spl_mean`, `collision_rate_mean`, ...).
2. **Faster at the same timestep?** -> `eval_summary_<run>.csv` curves, or
   `aux_ablation_summary.csv --eval-step <T>` for a fixed timestep.
3. **Auxiliary loss actually dropped?** -> `tqc_metrics.json` `aux/loss` /
   `aux/risk_mse` over `step` (aux runs only).
4. **Generalization differs?** -> `generalization_<world>_<run>.csv`
   (group by `aux_enabled`, compare per `world` / `condition_stage`).

## Running the ablation (>= 3 seeds each)

Run the same training with aux off and on, varying the seed. Enable aux by
setting `aux_prediction.enabled: true` in BOTH `hyperparameters_tqc.yaml`
(agent) and `environment_curriculum.yaml` (env) -- keep `num_sectors` /
`horizons_sec` identical (the trainer fail-fasts otherwise).

```bash
# aux OFF, seeds 0,1,2  (aux_prediction.enabled=false in both configs)
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p seed:=0
# ... seeds 1, 2

# aux ON, seeds 0,1,2   (aux_prediction.enabled=true in both configs)
ros2 run drl_agent train_tqc_curriculum_agent.py --ros-args -p seed:=0
# ... seeds 1, 2
```

Each run writes its own `logs/` with the files above.

## Aggregating

`aux_ablation_summary.py` reads each run's `run_manifest.json` next to its
`eval_summary_*.csv` and groups by the FULL aux config (so different aux
settings are never silently averaged together), then reports mean/std.

```bash
cd ros2_ws/src/drl_agent/scripts/utils

# default: group by FULL aux config signature (from run_manifest.json)
python3 aux_ablation_summary.py /path/to/all_runs --out logs/aux_ablation_summary.csv

# loose: group by aux on/off only, ignore the rest of the config (no manifest needed)
python3 aux_ablation_summary.py /path/to/all_runs --group-by aux_enabled \
    --out logs/aux_ablation_summary_loose.csv

# strict: abort if any run has no readable manifest
python3 aux_ablation_summary.py /path/to/all_runs --strict-manifest

# same-timestep comparison (nearest eval to t=100000):
python3 aux_ablation_summary.py /path/to/all_runs --eval-step 100000 \
    --out logs/aux_ablation_summary_100k.csv

# also split groups by git commit:
python3 aux_ablation_summary.py /path/to/all_runs --include-git
```

Output `aux_ablation_summary.csv`:

```
group,config_signature,n_runs,success_rate_mean,success_rate_std,...
aux_off,aux0|v1|K16|H[0.5,1,1.5]|lw0.1|mdlw0.0|distr0|temp0,3,0.72,0.02,...
aux_on,aux1|v1|K16|H[0.5,1,1.5]|lw0.1|mdlw0.0|distr0|temp0,3,0.8233,0.0252,...
```

Behaviour:
- `--group-by full` (default): one group per distinct aux config; runs with a
  different `num_sectors` / `horizons_sec` / `loss_weight` / ... land in
  separate rows (the `config_signature` column shows the exact config).
- `--group-by aux_enabled`: collapse to just `aux_off` / `aux_on`
  (`config_signature` empty), manifest not required.
- `--strict-manifest`: error if any run lacks a readable manifest.
- runs without a manifest (non-strict, full) are grouped under
  `config_signature='(no-manifest)'`, kept separate from manifest runs.
- inputs may be run directories (searched recursively for `eval_summary_*.csv`)
  or direct CSV paths; each run contributes one representative eval row (final,
  or nearest `--eval-step`). Warns when a group has < 2 runs.

## For the paper

- Tables: `aux_ablation_summary.csv` (mean +/- std of success/collision/SPL/
  CTE/jerk for aux_off vs aux_on; add `--eval-step` rows for a sample-efficiency
  table).
- Learning curves: plot `eval_summary_<run>.csv` (`success_rate` / `spl` vs
  `eval_global_t`), one line per run, grouped by `aux_enabled`.
- Aux-loss figure: plot `tqc_metrics.json` `aux/loss` vs `step` for aux runs.
- Generalization table: `generalization_<world>_<run>.csv` grouped by
  `aux_enabled` and `world` / `condition_stage`.
- `run_manifest.json` documents each run's exact config for the appendix.
