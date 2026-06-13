# Aux Experiment — Logging & Metric Schema

Where each metric is produced and stored for the auxiliary-prediction (aux)
experiments. Two tiers: **training-time monitoring** vs **formal (paper)
evaluation**. Everything is backward compatible: aux-off runs behave exactly as
before; new CSV columns are append-only; aux-only columns are blank when aux is
off.

## 1. Training-time monitoring

### A. Policy performance (per training episode, console + CSVs)
One-line episode summary (non-eval), aux on/off identical format:

```
T:{t} | Ep:{ep} | Steps:{n} | Reward:{r:.3f} | {GOAL/COLLISION/TIMEOUT/EVAL_CUT} | Stage:{s} | SPL:{..} | STL:{..} | PSC:{../n/a} | H-Coll:{0/1/n/a}
```

- `SPL/STL` from `EpisodeMetrics` (utils/episode_metrics.py).
- `PSC` (true human Personal-Space Compliance) and `H-Coll` are **label-derived**
  (privileged human-distance labels), computed by `_LabelProximity` in the
  trainer. They depend ONLY on the **env emitting labels**
  (`aux_prediction.enabled` on the env side), NOT on the agent aux head — so an
  **aux-OFF agent baseline still reports real PSC/H-Coll**. When the env emits no
  labels they are `n/a` (blank), never a misleading 0.
  - `PSC` = fraction of label-available steps with nearest-human distance ≥
    `psc_personal_space_m`.
  - `H-Coll` = collision episode whose nearest-human distance dropped below
    `h_coll_radius_m`. The post-step label is folded in, so the final colliding
    step is counted.
- `lidar_clearance_rate` (state-stream clearance PROXY, **not** human PSC; cannot
  tell a human from a wall/furniture) is computed by `EpisodeMetrics` and written
  to `episode_metrics_*.csv` — kept separate so it is never mistaken for PSC.
- Persisted to `episode_metrics_*.csv` (SPL/STL/lidar_clearance_rate) and
  `curriculum_episode_rewards_*.csv` (reward/result/stage/map_type).

### B. Aux self-learning (per gradient step, TensorBoard + tqc_metrics.json)
Produced in `tqc_agent.train()` (`aux_prediction_losses.compute_aux_loss`),
unchanged: `aux/loss`, `aux/risk_mse`, `aux/min_dist_mse` (v2),
`aux/risk_quantile` (distributional only), `aux/valid_len_mean`
(action-conditioned only). Aux-off → these keys are simply absent.

## 2. Formal evaluation (evaluate_and_print)

### A. Main policy metrics (eval console + CSVs)
`Success / Collision / Timeout / SPL / STL / PSC / H-Coll / CTE`, plus per-map
breakdown (`curriculum_eval_per_map_*.csv`). Aggregated in
`eval_metrics_*.csv` (paper) and `eval_summary_*.csv`.

### B. Formal aux metrics (eval console line, **aux on only**)
Computed in `utils/aux_eval_metrics.py` over every eval step (single-step:
`z_t`; action-conditioned: `z_t` + boundary-safe `[a_t..a_{t+K-1}]`, same
alignment as training, never crossing an episode boundary):

| metric | meaning |
|---|---|
| `aux_risk_rmse` | RMSE over all predicted-vs-GT risk-map cells |
| `aux_min_dist_mae_m` | future min-distance MAE in **metres** (norm err × D_c); pred derived from risk when no min-dist head |
| `aux_peak_sector_acc` | argmax-sector match per (sample, horizon); all-zero GT rows excluded; ties → lowest index |
| `aux_near_event_f1` | binary near-event = future min-dist < `aux_near_event_threshold_m`; zero-division-safe precision/recall/F1 |

Console (aux on):
```
Eval(aux) | AuxLossEval(RiskRMSE) {..} | MinDistMAE(m) {..} | PeakAcc {..} | EventF1 {..} (thr<{..}m, N={..})
```

## 3. Storage map

| metric group | TensorBoard | tqc_metrics.json | eval_summary_*.csv | curriculum_eval_per_map_*.csv | console |
|---|---|---|---|---|---|
| aux/loss, aux/risk_mse, aux/min_dist_mse, aux/risk_quantile, aux/valid_len_mean | ✓ | ✓ | – | – | – |
| SPL / STL | – | – | ✓ (eval mean) | ✓ (per map: SPL via base rates) | ✓ (episode line) |
| PSC, H-Coll (label-derived) | – | – | ✓ (blank if env labels off) | ✓ (blank if off) | ✓ (n/a if off) |
| lidar_clearance_rate (proxy) | – | – | ✓ (always) | – | ✓ (eval line) |
| aux_risk_rmse / aux_min_dist_mae_m / aux_peak_sector_acc / aux_near_event_f1 | – | – | ✓ (agent aux on) | ✓ (agent aux on) | ✓ (agent aux on) |

Blank = empty CSV cell / `n/a` console (the metric was not available), never 0.

## 4. Config (ROS params → run_manifest.json `aux_eval`)
- `aux_near_event_threshold_m` (default 0.5) — near-event distance for aux F1
- `h_coll_radius_m` (default 0.5) — human-collision distance (label-derived)
- `psc_personal_space_m` (default 0.5) — human personal-space radius (label-derived PSC)
- `stl_ref_speed_mps` (default 1.0) — STL optimal time = shortest_path / ref
- `lidar_clearance_radius_m` (default 0.5) — LiDAR clearance-proxy radius
- `risk_distance_scale` (D_c) taken from the **running env** node (single source).

## 5. Notes / extension points
- **PSC / H-Coll are label-derived** (env privileged human-distance labels), so
  they work for an aux-OFF agent **iff the env emits labels**; blank when the env
  emits none. `lidar_clearance_rate` is a separate state-stream clearance proxy
  (human vs static not distinguished) — not directly comparable to Falcon/DiPCAN
  PSC, hence the distinct name.
- `AuxEvalAccumulator` (utils/aux_eval_metrics.py) is the extension point for
  ESR / AD / ALV / encounter-count: collect extra per-episode arrays there and
  extend `finalize()` — no trainer-loop change needed.
- **Not run here:** ROS/Gazebo runtime + torch were unavailable for offline
  validation, so aux-head forward shapes and full eval flow are validated by
  contract/unit tests, not a live run.
