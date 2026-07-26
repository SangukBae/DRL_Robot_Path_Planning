"""Curriculum evaluation episode loop + metric aggregation / reporting.\n\nExtracted from train_tqc_curriculum_agent.py. Runs the eval episodes (eval_mode toggling, reset/step rollout), aggregates success/collision/timeout/SPL/STL/clearance/PSC/H-Coll (incl. per-map breakdown and aux metrics) and prints / logs them. Mixed into TrainTQCCurriculum; the trainer keeps only the call, stage-advance decision and logging around it."""

import os
import sys
import csv
import math
import time
import json
import pickle
import random
from datetime import datetime

import numpy as np
import torch

from environment_interface import EnvServiceError
import curriculum_stage_logic
import curriculum_state_io
import seed_utils
from episode_metrics import EpisodeMetrics, PaperMetricsCSV
from aux_prediction_labels import AUX_WIRE_VERSION
import aux_ablation_logging as aux_log
from aux_eval_metrics import AuxEvalAccumulator, split_label
from curriculum_metrics import _LabelProximity


class CurriculumEvalMixin:
    def evaluate_and_print(self, evals, epoch, start_time):
        """Run eval_eps episodes and return a metrics dict (not just mean reward)."""
        self.get_logger().info("=" * 55)
        self.get_logger().info(
            f"[Curriculum] Evaluating — Epoch {epoch} | Stage {self._curriculum_stage}"
        )
        self.get_logger().info(f"Elapsed: {time.time() - start_time:.1f}s")
        self.get_logger().info("=" * 55)

        ENV_DIM = self.environment_dim
        rewards, final_dists = [], []
        success_count = collision_count = timeout_count = 0
        per_ep_metrics = []
        # Structured map curriculum: per-map_type breakdown of this evaluation.
        per_map = {}   # map_type -> {n, success, collision, timeout, reward, goal_dist, h_coll}
        # map_type -> list of per-episode EpisodeMetrics.compute() dicts, so the
        # per-map CSV can report spl / lidar_clearance_rate via
        # PaperMetricsCSV.aggregate() (the same function the global eval_summary
        # row already uses) without a second metric-computation path.
        per_map_ep_metrics = {}
        # Formal aux-evaluation accumulator (global + per map_type). Only used
        # when the agent has an aux head; otherwise it stays empty (aux off path).
        aux_acc = None
        if self._aux_eval_on:
            aux_acc = AuxEvalAccumulator(
                num_horizons=self._aux_eval_H, num_sectors=self._aux_eval_K,
                risk_distance_scale=self._aux_eval_Dc,
                near_event_threshold_m=self._aux_near_event_threshold_m,
            )
        # Switch the env to the eval_map_types round-robin for these episodes
        # (obstacles still activate; env stays in train mode). If the toggle fails
        # we do NOT silently evaluate on the training mix — warn loudly and flag
        # the fallback so the per-map numbers are not mistaken for eval_map_types.
        eval_mode_on = self._set_eval_mode(True)
        if not eval_mode_on:
            self.get_logger().warn(
                "[Curriculum] ⚠ FAILED to enable curriculum_eval_mode "
                "(set_parameters rejected/timed out). This evaluation FALLS BACK to "
                "the TRAINING map distribution — the per-map breakdown below is NOT "
                "an eval_map_types evaluation. Check that environment_curriculum.py "
                "is running and exposes curriculum_eval_mode."
            )
        # The eval loop may fail for two independent reasons:
        #   1. the rollout itself raises (reset/step/logging/service failure)
        #   2. restoring curriculum_eval_mode=False fails afterwards
        # We must NOT let (2) silently continue, because training resets would
        # then keep cycling eval_map_types and corrupt the training map
        # distribution. Capture both and re-raise the primary rollout failure
        # first; otherwise fail-fast on the restore failure.
        eval_error = None
        restore_error = None
        try:
            for _ in range(self.eval_eps):
                state    = self.reset()
                map_type = self._fetch_current_map_type()
                done     = False
                ep_steps = 0
                ep_rew   = 0.0
                self._em.reset(state)
                # Per-step buffers for the formal aux eval (state_t, label_t paired
                # with s_t, and a_t taken from s_t). Boundary-safe: one episode.
                ev_states, ev_labels, ev_actions = [], [], []
                # Label-based human proximity (H-Coll / true PSC). Gated on the
                # ENV emitting labels — independent of the agent aux head.
                prox = _LabelProximity(self._psc_personal_space_m, self._h_coll_radius_m)

                while not done and ep_steps < self.max_episode_steps:
                    # Capture s_t + its paired aux label BEFORE stepping (aux head
                    # eval needs the encoder input s_t and its label).
                    if self._aux_eval_on and self.last_aux_label is not None:
                        ev_states.append(np.asarray(state, dtype=np.float32).ravel())
                        ev_labels.append(np.asarray(self.last_aux_label, dtype=np.float64).ravel())
                    action = self.rl_agent.select_action(
                        state, use_checkpoint=False, use_exploration=False
                    )
                    if self._aux_eval_on and len(ev_states) == len(ev_actions) + 1:
                        ev_actions.append(np.asarray(action, dtype=np.float32).ravel())
                    state, reward, done, info = self.step(action)
                    self._em.update(state, action)
                    # Fold the post-step label (paired with s_{t+1}) so the FINAL
                    # collision step's proximity is counted (it would be missed if
                    # we only read labels before stepping).
                    prox.add_dist(self._human_min_dist_m_from_label(self.last_aux_label))
                    ep_rew   += reward
                    ep_steps += 1

                s = np.asarray(state, dtype=np.float32).ravel()
                final_dist = float(s[ENV_DIM])
                final_dists.append(final_dist)
                rewards.append(ep_rew)
                _ep_metrics = self._em.compute(bool(done and info))
                per_ep_metrics.append(_ep_metrics)
                # Per-map paper-metric aggregation (spl / lidar_clearance_rate / ...)
                # -- reuses PaperMetricsCSV.aggregate() below, the SAME function the
                # global eval_summary row uses, instead of recomputing anything.
                per_map_ep_metrics.setdefault(map_type or "na", []).append(_ep_metrics)

                ep_success   = bool(done and info)
                ep_collision = bool(done and not info)
                ep_timeout   = bool(not done)
                if ep_success:
                    success_count   += 1
                elif ep_collision:
                    collision_count += 1
                else:
                    timeout_count   += 1
                # Label-derived human metrics (None when the env emits no labels).
                ep_h_coll = prox.h_coll(ep_collision)
                ep_psc    = prox.psc()

                # Formal aux metrics for this episode (boundary-safe).
                if aux_acc is not None and ev_states:
                    # Align lengths defensively (last action may be missing if the
                    # episode ended on the captured state).
                    L = min(len(ev_states), len(ev_labels), len(ev_actions)) \
                        if self._aux_eval_action_conditioned else min(len(ev_states), len(ev_labels))
                    if L > 0:
                        self._aux_eval_episode(
                            aux_acc, ev_states[:L], ev_labels[:L],
                            ev_actions[:L] if self._aux_eval_action_conditioned else None,
                            map_type,
                        )

                m = per_map.setdefault(
                    map_type or "na",
                    {"n": 0, "success": 0, "collision": 0, "timeout": 0,
                     "reward": 0.0, "goal_dist": 0.0,
                     "h_coll": 0, "h_coll_n": 0, "psc_sum": 0.0, "psc_n": 0},
                )
                m["n"]         += 1
                m["success"]   += int(ep_success)
                m["collision"] += int(ep_collision)
                m["timeout"]   += int(ep_timeout)
                m["reward"]    += ep_rew
                m["goal_dist"] += final_dist
                # Only count episodes where labels were available (avail) so the
                # rates are over the eligible denominator, not all episodes.
                if ep_h_coll is not None:
                    m["h_coll"]   += int(ep_h_coll)
                    m["h_coll_n"] += 1
                if ep_psc is not None:
                    m["psc_sum"]  += float(ep_psc)
                    m["psc_n"]    += 1
        except Exception as e:
            eval_error = e

        # Always return the env to the training map distribution so the next
        # training reset is NOT stuck on the eval round-robin — even if the
        # eval loop raised. A failed restore is a hard error, not a warning.
        try:
            if not self._set_eval_mode(False):
                restore_error = EnvServiceError(
                    "Failed to clear curriculum_eval_mode after evaluation; "
                    "refusing to continue because training resets may keep "
                    "cycling eval_map_types."
                )
        except EnvServiceError as e:
            restore_error = e

        if eval_error is not None:
            if restore_error is not None:
                self.get_logger().warn(
                    "[Curriculum] Eval loop raised and eval-mode restore also failed: "
                    f"{restore_error}"
                )
            raise eval_error
        if restore_error is not None:
            raise restore_error

        n = self.eval_eps
        metrics = {
            "mean_reward":    float(np.mean(rewards)),
            "std_reward":     float(np.std(rewards)),
            "success_rate":   success_count   / n,
            "collision_rate": collision_count / n,
            "timeout_rate":   timeout_count   / n,
            "mean_goal_dist": float(np.mean(final_dists)),
            # False → eval ran on the training map mix (eval_map_types NOT applied).
            "eval_map_applied": bool(eval_mode_on),
        }
        # Aggregate paper metrics (SPL, CTE, jerk, STL, lidar_clearance, ...).
        _agg = PaperMetricsCSV.aggregate(per_ep_metrics)
        metrics.update(_agg)
        # Label-derived human metrics, aggregated over the EPISODES THAT HAD
        # LABELS (env-side). None (→ blank) when no eval episode had labels, so an
        # aux-off-env run never reports a misleading 0. An aux-OFF agent with the
        # env labels ON still gets real H-Coll / PSC.
        _hc_sum = sum(d["h_coll"] for d in per_map.values())
        _hc_n   = sum(d["h_coll_n"] for d in per_map.values())
        _psc_sum = sum(d["psc_sum"] for d in per_map.values())
        _psc_n   = sum(d["psc_n"] for d in per_map.values())
        metrics["h_coll_rate"] = (float(_hc_sum) / _hc_n) if _hc_n > 0 else None
        # TRUE (human personal-space) PSC overrides the state-stream proxy name in
        # the summary; the LiDAR clearance proxy is kept separately as
        # lidar_clearance_rate (from _agg).
        metrics["psc"] = (float(_psc_sum) / _psc_n) if _psc_n > 0 else None
        # Formal aux-eval metrics (global). Merged into metrics so the eval
        # summary CSV + console can read them; absent/blank when aux is off.
        aux_eval_metrics = aux_acc.finalize() if aux_acc is not None else {}
        if aux_eval_metrics:
            metrics.update(aux_eval_metrics)
        aux_eval_per_map = aux_acc.finalize_per_map() if aux_acc is not None else {}
        self._paper.write_eval(
            epoch=epoch, global_t=self._last_global_t,
            stage=self._curriculum_stage, eval_eps=n,
            base=metrics, metrics_mean=_agg,
        )

        # AUX_ABLATION: one eval-summary row per evaluation -> aux on/off and
        # learning-curve comparison at the same timesteps.
        try:
            self._aux_eval_summary.append(
                eval_global_t=self._last_global_t,
                curriculum_stage=self._curriculum_stage,
                eval_eps=n, metrics=metrics,
            )
        except Exception as _e:
            self.get_logger().warn(f"[AUX_ABLATION] eval-summary append failed: {_e}")

        _psc_v = metrics.get("psc")
        _hc_v = metrics.get("h_coll_rate")
        _psc_s = f"{_psc_v:.3f}" if _psc_v is not None else "n/a"
        _hc_s = f"{_hc_v*100:.1f}%" if _hc_v is not None else "n/a"
        self.get_logger().info(
            f"Eval {n} eps | "
            f"Reward {metrics['mean_reward']:.3f}±{metrics['std_reward']:.3f} | "
            f"Success {metrics['success_rate']*100:.1f}% | "
            f"Collision {metrics['collision_rate']*100:.1f}% | "
            f"Timeout {metrics['timeout_rate']*100:.1f}% | "
            f"GoalDist {metrics['mean_goal_dist']:.3f}m | "
            f"SPL {metrics['spl']:.3f} | STL {metrics.get('stl', 0.0):.3f} | "
            f"PSC {_psc_s} | H-Coll {_hc_s} | "
            f"Clearance {metrics.get('lidar_clearance_rate', 0.0):.3f} | "
            f"CTE {metrics['mean_cross_track_error_m']:.3f}m"
        )
        # AUX_PRED: formal aux-evaluation line — printed ONLY when aux is on.
        if self._aux_eval_on and aux_eval_metrics:
            _pk = aux_eval_metrics.get("aux_peak_sector_acc")
            _pk_s = f"{_pk:.3f}" if (_pk is not None and _pk == _pk) else "n/a"
            self.get_logger().info(
                "Eval(aux) | "
                f"AuxLossEval(RiskRMSE) {aux_eval_metrics['aux_risk_rmse']:.4f} | "
                f"MinDistMAE(m) {aux_eval_metrics['aux_min_dist_mae_m']:.4f} | "
                f"PeakAcc {_pk_s} | "
                f"EventF1 {aux_eval_metrics['aux_near_event_f1']:.3f} "
                f"(thr<{self._aux_near_event_threshold_m:.2f}m, N={aux_eval_metrics.get('aux_eval_samples', 0)})"
            )

        # Structured map curriculum: write + log the per-map_type breakdown so a
        # mixed-map stage average never hides one map collapsing.
        metrics["per_map"] = {}
        if not eval_mode_on:
            self.get_logger().warn(
                "[Curriculum] per-map breakdown below is a FALLBACK (training map "
                "distribution), not eval_map_types — see warning above."
            )
        try:
            with open(self._curriculum_eval_per_map_csv, "a", newline="") as f:
                w = csv.writer(f)
                def _b(am, key=None):
                    # blank when a metric is absent / None / NaN (aux off / no labels).
                    # `key=None` -> `am` IS the raw value (h_coll_rate / psc / a lookup
                    # from a per-map paper-metric dict); `key` set -> dict-lookup style
                    # (aux_eval_per_map entries).
                    v = am.get(key) if (key is not None and isinstance(am, dict)) else am
                    if v is None:
                        return ""
                    try:
                        fv = float(v)
                        return "" if fv != fv else round(fv, 6)
                    except Exception:
                        return ""
                for mt in sorted(per_map.keys()):
                    d = per_map[mt]
                    nn = max(d["n"], 1)
                    sr, cr, tr = d["success"] / nn, d["collision"] / nn, d["timeout"] / nn
                    mr, gd = d["reward"] / nn, d["goal_dist"] / nn
                    # Label-derived rates over label-available episodes only (None
                    # → blank when this map had no labels).
                    hc = (d["h_coll"] / d["h_coll_n"]) if d["h_coll_n"] > 0 else None
                    psc = (d["psc_sum"] / d["psc_n"]) if d["psc_n"] > 0 else None
                    am = aux_eval_per_map.get(mt, {})
                    # Per-map paper metrics (spl / lidar_clearance_rate / ...) via the
                    # SAME aggregate() the global eval_summary row uses -- no map ever
                    # reaches here with an empty episode list (per_map only gets a key
                    # once >=1 episode ran on it), so this is never the all-zero
                    # EpisodeMetrics.empty() fallback in practice.
                    pm_paper = PaperMetricsCSV.aggregate(per_map_ep_metrics.get(mt, []))
                    metrics["per_map"][mt] = {
                        "eval_eps": d["n"], "success_rate": sr,
                        "collision_rate": cr, "timeout_rate": tr,
                        "mean_reward": mr, "mean_goal_dist": gd,
                        "h_coll_rate": hc, "psc": psc,
                        "spl": pm_paper.get("spl"),
                        "lidar_clearance_rate": pm_paper.get("lidar_clearance_rate"),
                        **am,
                    }
                    w.writerow([
                        epoch, self._last_global_t, self._curriculum_stage, mt, d["n"],
                        round(sr, 4), round(cr, 4), round(tr, 4),
                        round(mr, 4), round(gd, 4),
                        _b(hc), _b(psc),
                        _b(am, "aux_risk_rmse"), _b(am, "aux_min_dist_mae_m"),
                        _b(am, "aux_peak_sector_acc"), _b(am, "aux_near_event_f1"),
                        _b(pm_paper, "lidar_clearance_rate"), _b(pm_paper, "spl"),
                    ])
                    _hc_ms = f"{hc*100:.1f}%" if hc is not None else "n/a"
                    self.get_logger().info(
                        f"  [map={mt:<12}] n={d['n']:<3} "
                        f"Success {sr*100:5.1f}% | Collision {cr*100:5.1f}% | "
                        f"Timeout {tr*100:5.1f}% | GoalDist {gd:.3f}m | H-Coll {_hc_ms}"
                    )
        except Exception as _e:
            self.get_logger().warn(f"[Curriculum] per-map eval log failed: {_e}")

        evals.append(metrics["mean_reward"])
        np.save(f"{self.results_dir}/{self.file_name}", evals)
        return metrics

    # ------------------------------------------------------------------ #
    #  Override: train_online — adds stage advancement around eval          #
    # ------------------------------------------------------------------ #
