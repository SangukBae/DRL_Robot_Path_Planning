#!/usr/bin/env python3
"""Aggregate multi-seed paper results into tables and curves.

Reads the per-run ``eval_metrics_*.csv`` (and optionally
``episode_metrics_*.csv``) produced by the curriculum trainers and computes,
across seeds, the mean +/- std tables and learning curves that the paper figures
are built from. Pure stdlib + numpy (no pandas), so it runs anywhere.

Expected layout (the per-seed default of the trainers):

    <runtime-root>/<algo>/seed_<N>/logs/eval_metrics_*.csv
                                        /episode_metrics_*.csv

Usage
-----
    python3 aggregate_results.py --runtime-root ros2_ws/src/drl_agent/runtime
    python3 aggregate_results.py --runtime-root runtime --algos tqc sac sb3_td3 \
            --out runtime/_aggregate --thresholds 0.5 0.7 0.8 0.9

Outputs (written under --out, default <runtime-root>/_aggregate):
    final_eval_summary.csv     one row per algo: mean & std (across seeds) of the
                               last eval point for every metric.
    eval_curve_<metric>.csv    per algo, per epoch: mean & std across seeds
                               (success_rate, collision_rate, timeout_rate, spl,
                                mean_reward) — for learning / sample-efficiency plots.
    sample_efficiency.csv      per algo: global_t to first reach each success-rate
                               threshold (mean & std across seeds).
"""

import os
import csv
import glob
import json
import argparse
from collections import defaultdict

import numpy as np

# Metrics summarised in the final table (must exist in eval_metrics_*.csv).
SUMMARY_METRICS = [
    "mean_reward", "success_rate", "collision_rate", "timeout_rate",
    "mean_goal_dist", "path_length_m", "spl", "mean_heading_error_rad",
    "mean_cross_track_error_m", "mean_action_jerk", "mean_steering_change_rad",
    "near_collision_count", "travel_time_s", "mean_speed_mps",
]
CURVE_METRICS = ["success_rate", "collision_rate", "timeout_rate", "spl", "mean_reward"]


# --------------------------------------------------------------------------- #
#  Discovery / loading                                                          #
# --------------------------------------------------------------------------- #
def discover_runs(runtime_root, algos=None):
    """Return {algo: {seed: latest_eval_csv_path}}."""
    runs = defaultdict(dict)
    for algo_dir in sorted(glob.glob(os.path.join(runtime_root, "*"))):
        if not os.path.isdir(algo_dir):
            continue
        algo = os.path.basename(algo_dir)
        if algos and algo not in algos:
            continue
        for seed_dir in sorted(glob.glob(os.path.join(algo_dir, "seed_*"))):
            seed = os.path.basename(seed_dir)
            cands = glob.glob(os.path.join(seed_dir, "logs", "eval_metrics_*.csv"))
            if not cands:
                continue
            latest = max(cands, key=os.path.getmtime)   # newest run for this seed
            runs[algo][seed] = latest
    return runs


def load_csv(path):
    """Load a CSV into a list of dicts with floats where possible."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            try:
                rr[k] = float(v)
            except (TypeError, ValueError):
                rr[k] = v
        out.append(rr)
    return out


# --------------------------------------------------------------------------- #
#  Aggregation                                                                  #
# --------------------------------------------------------------------------- #
def final_summary(runs):
    """mean/std across seeds of the LAST eval row, per algo."""
    table = {}
    for algo, seedmap in runs.items():
        per_metric = defaultdict(list)
        n_seeds = 0
        for seed, path in seedmap.items():
            rows = load_csv(path)
            if not rows:
                continue
            n_seeds += 1
            last = rows[-1]
            for m in SUMMARY_METRICS:
                if m in last and isinstance(last[m], float):
                    per_metric[m].append(last[m])
        table[algo] = {
            "n_seeds": n_seeds,
            **{m: (float(np.mean(v)) if v else float("nan"),
                   float(np.std(v)) if v else float("nan"))
               for m, v in per_metric.items()},
        }
    return table


def eval_curves(runs):
    """Per algo, per epoch: mean/std across seeds for CURVE_METRICS + global_t."""
    curves = {}
    for algo, seedmap in runs.items():
        by_epoch = defaultdict(lambda: defaultdict(list))   # epoch -> metric -> [vals]
        for seed, path in seedmap.items():
            for r in load_csv(path):
                ep = int(r.get("epoch", 0))
                by_epoch[ep]["global_t"].append(r.get("global_t", float("nan")))
                for m in CURVE_METRICS:
                    if m in r and isinstance(r[m], float):
                        by_epoch[ep][m].append(r[m])
        curves[algo] = by_epoch
    return curves


def sample_efficiency(runs, thresholds, censor_at=None):
    """Per algo: global_t at which success_rate first reaches each threshold.

    Seeds that NEVER reach a threshold are not silently dropped (which would
    make a partly-failing algorithm look optimistic). For each threshold we
    report:
      * n_hit / n_total      — how many seeds reached it (reach rate)
      * mean/std over hits    — averaged over reaching seeds only
      * censored mean/std     — non-reaching seeds are right-censored at
                                ``censor_at`` (or, if None, that seed's final
                                eval global_t), so failing seeds still penalise
                                the average instead of vanishing from it.
    """
    out = {}
    for algo, seedmap in runs.items():
        per_thr = {thr: {"hit": [], "censored": []} for thr in thresholds}
        n_total = 0
        for seed, path in seedmap.items():
            rows = load_csv(path)
            if not rows:
                continue
            n_total += 1
            last_t = max((r["global_t"] for r in rows
                          if isinstance(r.get("global_t"), float)), default=float("nan"))
            cap = float(censor_at) if censor_at else last_t
            for thr in thresholds:
                hit = next((r["global_t"] for r in rows
                            if isinstance(r.get("success_rate"), float)
                            and r["success_rate"] >= thr), None)
                if hit is not None:
                    per_thr[thr]["hit"].append(hit)
                    per_thr[thr]["censored"].append(hit)
                else:
                    per_thr[thr]["censored"].append(cap)   # right-censored
        res = {}
        for thr in thresholds:
            hit = per_thr[thr]["hit"]
            cen = per_thr[thr]["censored"]
            res[thr] = {
                "n_hit": len(hit),
                "n_total": n_total,
                "mean_hit": float(np.mean(hit)) if hit else float("nan"),
                "std_hit": float(np.std(hit)) if hit else float("nan"),
                "mean_censored": float(np.mean(cen)) if cen else float("nan"),
                "std_censored": float(np.std(cen)) if cen else float("nan"),
            }
        out[algo] = res
    return out


# --------------------------------------------------------------------------- #
#  Writers                                                                      #
# --------------------------------------------------------------------------- #
def write_final_summary(table, out_dir):
    path = os.path.join(out_dir, "final_eval_summary.csv")
    header = ["algo", "n_seeds"]
    for m in SUMMARY_METRICS:
        header += [f"{m}_mean", f"{m}_std"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for algo, d in sorted(table.items()):
            row = [algo, d["n_seeds"]]
            for m in SUMMARY_METRICS:
                mean, std = d.get(m, (float("nan"), float("nan")))
                row += [round(mean, 4), round(std, 4)]
            w.writerow(row)
    return path


def write_curves(curves, out_dir):
    paths = []
    for m in CURVE_METRICS:
        path = os.path.join(out_dir, f"eval_curve_{m}.csv")
        algos = sorted(curves.keys())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["epoch"]
            for a in algos:
                header += [f"{a}_global_t", f"{a}_mean", f"{a}_std", f"{a}_n"]
            w.writerow(header)
            all_eps = sorted({ep for a in algos for ep in curves[a]})
            for ep in all_eps:
                row = [ep]
                for a in algos:
                    cell = curves[a].get(ep, {})
                    gt = cell.get("global_t", [])
                    vals = cell.get(m, [])
                    row += [
                        round(float(np.mean(gt)), 1) if gt else "",
                        round(float(np.mean(vals)), 4) if vals else "",
                        round(float(np.std(vals)), 4) if vals else "",
                        len(vals),
                    ]
                w.writerow(row)
        paths.append(path)
    return paths


def write_sample_efficiency(se, thresholds, out_dir):
    def _r(x):
        return round(x, 1) if isinstance(x, float) and x == x else ""   # nan -> ""

    path = os.path.join(out_dir, "sample_efficiency.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["algo"]
        for thr in thresholds:
            header += [
                f"sr{thr}_n_hit", f"sr{thr}_n_total",
                f"sr{thr}_steps_mean_hit", f"sr{thr}_steps_std_hit",
                f"sr{thr}_steps_mean_censored", f"sr{thr}_steps_std_censored",
            ]
        w.writerow(header)
        for algo, d in sorted(se.items()):
            row = [algo]
            for thr in thresholds:
                c = d[thr]
                row += [
                    c["n_hit"], c["n_total"],
                    _r(c["mean_hit"]), _r(c["std_hit"]),
                    _r(c["mean_censored"]), _r(c["std_censored"]),
                ]
            w.writerow(row)
    return path


def print_table(table):
    print("\n=== Final eval (mean ± std across seeds) ===")
    cols = ["success_rate", "collision_rate", "timeout_rate", "spl",
            "mean_cross_track_error_m", "mean_action_jerk"]
    hdr = f"{'algo':12} {'seeds':>5}  " + "  ".join(f"{c:>22}" for c in cols)
    print(hdr)
    for algo, d in sorted(table.items()):
        line = f"{algo:12} {d['n_seeds']:>5}  "
        for c in cols:
            mean, std = d.get(c, (float('nan'), float('nan')))
            line += f"  {mean:8.3f}±{std:<6.3f}      "[:24]
        print(line)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-root", required=True,
                    help="Root containing <algo>/seed_<N>/logs/eval_metrics_*.csv")
    ap.add_argument("--algos", nargs="*", default=None,
                    help="Subset of algo dir names to include (default: all)")
    ap.add_argument("--out", default=None, help="Output dir (default: <root>/_aggregate)")
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.5, 0.7, 0.8, 0.9],
                    help="Success-rate thresholds for sample efficiency")
    ap.add_argument("--censor-at", type=float, default=None,
                    help="global_t at which non-reaching seeds are right-censored "
                         "(default: each seed's final eval step, e.g. max_timesteps)")
    args = ap.parse_args()

    root = os.path.expanduser(args.runtime_root)
    out_dir = args.out or os.path.join(root, "_aggregate")
    os.makedirs(out_dir, exist_ok=True)

    runs = discover_runs(root, set(args.algos) if args.algos else None)
    if not runs:
        print(f"[aggregate] No eval_metrics_*.csv found under {root}")
        print("  expected: <root>/<algo>/seed_<N>/logs/eval_metrics_*.csv")
        return
    n = {a: len(s) for a, s in runs.items()}
    print(f"[aggregate] runs discovered: {json.dumps(n)}")

    table = final_summary(runs)
    curves = eval_curves(runs)
    se = sample_efficiency(runs, args.thresholds, censor_at=args.censor_at)

    p1 = write_final_summary(table, out_dir)
    p2 = write_curves(curves, out_dir)
    p3 = write_sample_efficiency(se, args.thresholds, out_dir)

    print_table(table)
    print("\n[aggregate] wrote:")
    for p in [p1, *p2, p3]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
