#!/usr/bin/env python3
"""Phase 1a post-processing: aux-accuracy vs avoidance-performance correlation,
from EXISTING ``eval_summary_<run_tag>.csv`` logs -- no retraining needed.

``eval_summary_<run_tag>.csv`` (written once per eval call, i.e. per
seed x checkpoint) already carries BOTH the aux-accuracy metrics
(aux_risk_rmse, aux_min_dist_mae_m, aux_peak_sector_acc, aux_near_event_f1)
and the avoidance/task metrics (h_coll_rate, psc, success_rate,
collision_rate, timeout_rate, spl) in the same row, so this script only needs
to read that one file family and correlate columns -- see
utils/aux_ablation_logging.py::EVAL_SUMMARY_HEADER for the schema.

For each (aux metric, avoidance metric) pair this computes Pearson r over all
(seed, checkpoint) rows that have BOTH values present, and writes:
  * aux_correlation_summary.csv / .json  -- one row per metric pair (r, n)
  * aux_correlation_scatter_<x>_vs_<y>.png -- scatter plots (best-effort; a
    missing/broken matplotlib is a warning, not a crash -- see module spec:
    "입력 파일이 부족하거나 컬럼이 없을 때는 fail-fast보다는 경고와 빈 값 처리")

Guards against the "fake correlation from training progress" trap the roadmap
calls out (aux getting better AND avoidance getting better over training time
produces a correlation that isn't causal) only insofar as it reports N and
lets the reader group by seed/checkpoint themselves; it does not itself
de-trend the series -- see the script's printed caveat.

Usage
-----
    python3 analyze_aux_correlation.py \\
        --runtime-root ../../runtime \\
        --out ../../runtime/_aux_correlation_analysis

    python3 analyze_aux_correlation.py \\
        --csv runtime/tqc/seed_0/logs/eval_summary_20260704_092437.csv \\
        --out /tmp/aux_corr --no-plots
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import warnings

AUX_METRICS = [
    "aux_risk_rmse", "aux_min_dist_mae_m", "aux_peak_sector_acc", "aux_near_event_f1",
]
AVOIDANCE_METRICS = [
    "h_coll_rate", "psc", "success_rate", "collision_rate", "timeout_rate", "spl",
]
REQUIRED_COLS = ["seed", "eval_global_t", "curriculum_stage"] + AUX_METRICS + AVOIDANCE_METRICS


def _to_float(v):
    if v is None:
        return float("nan")
    s = str(v).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def discover_csvs(runtime_root):
    """glob <runtime_root>/*/seed_*/logs/eval_summary_*.csv, mirroring the
    discovery convention already used by utils/aggregate_results.py."""
    pattern = os.path.join(runtime_root, "*", "seed_*", "logs", "eval_summary_*.csv")
    return sorted(glob.glob(pattern))


def load_rows(csv_paths):
    rows = []
    for path in csv_paths:
        if not os.path.isfile(path):
            warnings.warn(f"[analyze_aux_correlation] file not found, skipping: {path}")
            continue
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                missing = [c for c in REQUIRED_COLS if c not in fieldnames]
                if missing:
                    warnings.warn(
                        f"[analyze_aux_correlation] {path}: missing columns {missing}; "
                        "those fields will be NaN for every row in this file"
                    )
                for row in reader:
                    row["_source_file"] = path
                    rows.append(row)
        except Exception as exc:  # noqa: BLE001 - never crash the whole run on one bad file
            warnings.warn(f"[analyze_aux_correlation] failed to read {path}: {exc}")
    if not rows:
        warnings.warn("[analyze_aux_correlation] no usable rows found in any input file")
    return rows


def pearson_r(xs, ys):
    """Pearson correlation over paired (x, y) with any NaN dropped. Returns
    (r, n); r is NaN when n < 3 or either series is constant (no fail-fast)."""
    pairs = [
        (x, y) for x, y in zip(xs, ys)
        if not (math.isnan(x) or math.isnan(y))
    ]
    n = len(pairs)
    if n < 3:
        return float("nan"), n
    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    mx = sum(xv) / n
    my = sum(yv) / n
    sxx = sum((x - mx) ** 2 for x in xv)
    syy = sum((y - my) ** 2 for y in yv)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xv, yv))
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan"), n
    return sxy / math.sqrt(sxx * syy), n


def compute_correlations(rows):
    """Return {(aux_metric, avoidance_metric): {"r": ..., "n": ...}}."""
    out = {}
    for am in AUX_METRICS:
        xs = [_to_float(r.get(am)) for r in rows]
        for tm in AVOIDANCE_METRICS:
            ys = [_to_float(r.get(tm)) for r in rows]
            r, n = pearson_r(xs, ys)
            out[(am, tm)] = {"r": r, "n": n}
    return out


def write_summary(correlations, out_dir):
    csv_path = os.path.join(out_dir, "aux_correlation_summary.csv")
    json_path = os.path.join(out_dir, "aux_correlation_summary.json")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aux_metric", "avoidance_metric", "pearson_r", "n_samples"])
        for (am, tm), v in correlations.items():
            w.writerow([am, tm, v["r"], v["n"]])
    json_obj = {
        f"{am}__vs__{tm}": {"pearson_r": v["r"], "n_samples": v["n"]}
        for (am, tm), v in correlations.items()
    }
    with open(json_path, "w") as f:
        json.dump(json_obj, f, indent=2, sort_keys=True)
    return csv_path, json_path


def write_scatter_plots(rows, out_dir):
    """Best-effort PNG scatter plots, one per (aux_metric, avoidance_metric)
    pair. A missing/broken matplotlib is a warning, not a crash."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"[analyze_aux_correlation] matplotlib unavailable ({exc}); "
            "skipping scatter plots (CSV/JSON summaries were still written)"
        )
        return []

    written = []
    for am in AUX_METRICS:
        xs = [_to_float(r.get(am)) for r in rows]
        for tm in AVOIDANCE_METRICS:
            ys = [_to_float(r.get(tm)) for r in rows]
            pairs = [(x, y) for x, y in zip(xs, ys) if not (math.isnan(x) or math.isnan(y))]
            if len(pairs) < 2:
                continue
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=14, alpha=0.7)
            ax.set_xlabel(am)
            ax.set_ylabel(tm)
            r, n = pearson_r(xs, ys)
            r_label = "nan" if math.isnan(r) else f"{r:.3f}"
            ax.set_title(f"r={r_label} (n={n})")
            fig.tight_layout()
            out_path = os.path.join(out_dir, f"aux_correlation_scatter_{am}_vs_{tm}.png")
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            written.append(out_path)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-root", default=None,
                     help="Discover eval_summary_*.csv under <root>/*/seed_*/logs/.")
    ap.add_argument("--csv", nargs="*", default=None,
                     help="Explicit CSV path(s); combined with --runtime-root if both given.")
    ap.add_argument("--out", required=True, help="Output directory (created if missing).")
    ap.add_argument("--no-plots", action="store_true",
                     help="Skip PNG scatter plots even if matplotlib is available.")
    args = ap.parse_args(argv)

    csv_paths = list(args.csv or [])
    if args.runtime_root:
        found = discover_csvs(args.runtime_root)
        if not found:
            warnings.warn(
                f"[analyze_aux_correlation] no eval_summary_*.csv found under "
                f"{args.runtime_root}"
            )
        csv_paths.extend(found)
    if not csv_paths:
        warnings.warn("[analyze_aux_correlation] no input CSVs given "
                       "(--csv / --runtime-root both empty) -- writing empty outputs")

    rows = load_rows(csv_paths)
    correlations = compute_correlations(rows)

    os.makedirs(args.out, exist_ok=True)
    csv_path, json_path = write_summary(correlations, args.out)
    plots = [] if args.no_plots else write_scatter_plots(rows, args.out)

    print(f"[analyze_aux_correlation] rows: {len(rows)} | files: {len(csv_paths)}")
    print(f"[analyze_aux_correlation] wrote: {csv_path}")
    print(f"[analyze_aux_correlation] wrote: {json_path}")
    print(f"[analyze_aux_correlation] wrote {len(plots)} scatter plot(s) to {args.out}")
    print("[analyze_aux_correlation] CAVEAT: a same-run-over-training correlation can be "
          "a training-progress artifact (both aux accuracy and avoidance improve together) "
          "rather than causal -- see roadmap Slide 6 'trap'. Group/inspect by "
          "seed x checkpoint rather than reading a single pooled r at face value.")
    for (am, tm), v in correlations.items():
        r = v["r"]
        r_label = "nan" if math.isnan(r) else f"{r:+.3f}"
        print(f"  {am:22s} vs {tm:16s}  r={r_label}  n={v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
