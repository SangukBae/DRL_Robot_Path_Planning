#!/usr/bin/env python3
"""Phase 1a post-processing: yield precision/recall + freezing diagnostics from
EXISTING ``dynamic_avoidance_metrics_<run_tag>.csv`` logs -- no retraining needed.

Reads one or more dynamic_avoidance_metrics CSVs (see
``utils/dynamic_avoidance_log.py`` for the schema) and derives, per episode:

  * yield_precision        = yield_in_risk_steps / yield_steps
                              ("when the policy yielded, was it actually risky?")
  * yield_recall           = yield_in_risk_steps / risk_steps
                              ("of all risky steps, how many did it yield during?")
  * bad_yield_rate         = yield_no_risk_steps / yield_steps
                              ("of the steps it yielded, how many were unnecessary?")
  * yield_mean_streak_steps = yield_steps / yield_trigger_count
                              (avg. steps held per yield activation -- a duration proxy;
                              the raw CSV has no per-streak-length column, so this is
                              the best available estimate from existing counters)
  * yield_step_frac        = yield_steps / steps
  * low_obs_speed_frac     = passed through (existing freezing-adjacent column)

then aggregates (overall, and optionally grouped by map_type / aux_enabled /
curriculum_stage) into a summary CSV + JSON, plus a "timeout vs yield" cross-tab
capturing what fraction of timeout episodes show heavy yield usage.

Never fail-fasts on missing files/columns: a missing file or column is a
warning, and the affected values are NaN rather than an exception, so a partial
log set still produces partial results (see the module docstring requirement:
"입력 파일이 부족하거나 컬럼이 없을 때는 fail-fast보다는 경고와 빈 값 처리").

Usage
-----
    python3 analyze_yield_freezing.py \\
        --runtime-root ../../runtime \\
        --out ../../runtime/_yield_freezing_analysis

    # or point directly at specific CSVs:
    python3 analyze_yield_freezing.py \\
        --csv runtime/tqc/seed_0/logs/dynamic_avoidance_metrics_20260706_030957.csv \\
        --out /tmp/yield_analysis --group-by map_type
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
import warnings

REQUIRED_COLS = [
    "episode", "curriculum_stage", "map_type", "seed", "aux_enabled",
    "success", "collision", "timeout", "steps",
    "low_obs_speed_frac", "yield_available", "yield_used",
    "yield_trigger_count", "yield_steps", "yield_in_risk_steps",
    "yield_no_risk_steps", "risk_steps",
]

GROUP_CHOICES = ["none", "map_type", "aux_enabled", "curriculum_stage"]


def _to_float(v):
    """Parse a CSV cell to float; '' / None / unparsable -> NaN (never raises)."""
    if v is None:
        return float("nan")
    s = str(v).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _to_int_or_none(v):
    f = _to_float(v)
    return None if math.isnan(f) else int(f)


def _nanmean(values):
    vals = [v for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(statistics.fmean(vals)) if vals else float("nan")


def _ratio(numer, denom):
    if denom is None or math.isnan(denom) or denom <= 0:
        return float("nan")
    if numer is None or math.isnan(numer):
        return float("nan")
    return numer / denom


def discover_csvs(runtime_root):
    """glob <runtime_root>/*/seed_*/logs/dynamic_avoidance_metrics_*.csv, mirroring
    the discovery convention already used by utils/aggregate_results.py."""
    pattern = os.path.join(runtime_root, "*", "seed_*", "logs",
                            "dynamic_avoidance_metrics_*.csv")
    return sorted(glob.glob(pattern))


def load_episode_rows(csv_paths):
    """Read every row from every CSV, tagging with the source file. Missing
    files/columns/parse errors are warnings, not exceptions."""
    rows = []
    for path in csv_paths:
        if not os.path.isfile(path):
            warnings.warn(f"[analyze_yield_freezing] file not found, skipping: {path}")
            continue
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                missing = [c for c in REQUIRED_COLS if c not in fieldnames]
                if missing:
                    warnings.warn(
                        f"[analyze_yield_freezing] {path}: missing columns {missing}; "
                        "those fields will be NaN for every row in this file"
                    )
                for row in reader:
                    row["_source_file"] = path
                    rows.append(row)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: never crash the run
            warnings.warn(f"[analyze_yield_freezing] failed to read {path}: {exc}")
    if not rows:
        warnings.warn("[analyze_yield_freezing] no usable rows found in any input file")
    return rows


def derive_episode_metrics(row):
    """Compute the per-episode derived yield/freezing fields for one CSV row."""
    steps = _to_float(row.get("steps"))
    yield_steps = _to_float(row.get("yield_steps"))
    yield_in_risk = _to_float(row.get("yield_in_risk_steps"))
    yield_no_risk = _to_float(row.get("yield_no_risk_steps"))
    risk_steps = _to_float(row.get("risk_steps"))
    trigger = _to_float(row.get("yield_trigger_count"))

    out = dict(row)
    out["yield_precision"] = _ratio(yield_in_risk, yield_steps)
    out["yield_recall"] = _ratio(yield_in_risk, risk_steps)
    out["bad_yield_rate"] = _ratio(yield_no_risk, yield_steps)
    out["yield_mean_streak_steps"] = _ratio(yield_steps, trigger)
    out["yield_step_frac"] = _ratio(yield_steps, steps)
    out["_timeout"] = _to_int_or_none(row.get("timeout"))
    out["_success"] = _to_int_or_none(row.get("success"))
    out["_collision"] = _to_int_or_none(row.get("collision"))
    out["_yield_used"] = _to_int_or_none(row.get("yield_used"))
    out["_low_obs_speed_frac"] = _to_float(row.get("low_obs_speed_frac"))
    return out


PER_EPISODE_OUT_COLS = [
    "episode", "global_t", "curriculum_stage", "map_type", "seed", "aux_enabled",
    "success", "collision", "timeout", "steps",
    "yield_available", "yield_used", "yield_trigger_count",
    "yield_steps", "yield_in_risk_steps", "yield_no_risk_steps", "risk_steps",
    "low_obs_speed_frac",
    "yield_precision", "yield_recall", "bad_yield_rate",
    "yield_mean_streak_steps", "yield_step_frac",
    "_source_file",
]


def write_per_episode_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PER_EPISODE_OUT_COLS)
        for r in rows:
            w.writerow([r.get(c, "") for c in PER_EPISODE_OUT_COLS])


def _group_key(row, group_by):
    if group_by == "none":
        return "all"
    v = row.get(group_by, "")
    return "" if v is None else str(v)


def summarize(rows, group_by="none"):
    """Group episodes (default: no grouping) and compute mean yield/freezing
    metrics + a timeout-vs-yield cross-tab per group."""
    groups = {}
    for r in rows:
        groups.setdefault(_group_key(r, group_by), []).append(r)

    summary = {}
    for key, grp in groups.items():
        n = len(grp)
        timeout_eps = [r for r in grp if r["_timeout"] == 1]
        summary[key] = {
            "n_episodes": n,
            "success_rate": _nanmean([r["_success"] for r in grp]),
            "collision_rate": _nanmean([r["_collision"] for r in grp]),
            "timeout_rate": _nanmean([r["_timeout"] for r in grp]),
            "yield_precision_mean": _nanmean([r["yield_precision"] for r in grp]),
            "yield_recall_mean": _nanmean([r["yield_recall"] for r in grp]),
            "bad_yield_rate_mean": _nanmean([r["bad_yield_rate"] for r in grp]),
            "yield_mean_streak_steps_mean": _nanmean(
                [r["yield_mean_streak_steps"] for r in grp]),
            "yield_step_frac_mean": _nanmean([r["yield_step_frac"] for r in grp]),
            "low_obs_speed_frac_mean": _nanmean([r["_low_obs_speed_frac"] for r in grp]),
            # Freezing/timeout <-> yield relationship: among TIMEOUT episodes only,
            # how yield-heavy were they, and what fraction even used yield at all.
            "timeout_n": len(timeout_eps),
            "timeout_yield_step_frac_mean": _nanmean(
                [r["yield_step_frac"] for r in timeout_eps]),
            "timeout_yield_used_rate": _nanmean(
                [r["_yield_used"] for r in timeout_eps]),
            "timeout_low_obs_speed_frac_mean": _nanmean(
                [r["_low_obs_speed_frac"] for r in timeout_eps]),
        }
    return summary


def write_summary_csv(summary, group_by, out_path):
    cols = ["group"] + list(next(iter(summary.values())).keys()) if summary else ["group"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key, vals in summary.items():
            w.writerow([key] + [vals.get(c, "") for c in cols[1:]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-root", default=None,
                     help="Discover dynamic_avoidance_metrics_*.csv under "
                          "<root>/*/seed_*/logs/ (mirrors aggregate_results.py).")
    ap.add_argument("--csv", nargs="*", default=None,
                     help="Explicit CSV path(s); combined with --runtime-root if both given.")
    ap.add_argument("--group-by", choices=GROUP_CHOICES, default="none",
                     help="Aggregate the summary by this column (default: no grouping).")
    ap.add_argument("--out", required=True,
                     help="Output directory (created if missing). Writes "
                          "per_episode_yield_freezing.csv, "
                          "yield_freezing_summary.csv/.json")
    args = ap.parse_args(argv)

    csv_paths = list(args.csv or [])
    if args.runtime_root:
        found = discover_csvs(args.runtime_root)
        if not found:
            warnings.warn(
                f"[analyze_yield_freezing] no dynamic_avoidance_metrics_*.csv "
                f"found under {args.runtime_root}"
            )
        csv_paths.extend(found)
    if not csv_paths:
        warnings.warn("[analyze_yield_freezing] no input CSVs given "
                       "(--csv / --runtime-root both empty) -- writing empty outputs")

    rows = load_episode_rows(csv_paths)
    derived = [derive_episode_metrics(r) for r in rows]
    summary = summarize(derived, group_by=args.group_by)

    os.makedirs(args.out, exist_ok=True)
    per_ep_path = os.path.join(args.out, "per_episode_yield_freezing.csv")
    summary_csv_path = os.path.join(args.out, "yield_freezing_summary.csv")
    summary_json_path = os.path.join(args.out, "yield_freezing_summary.json")

    write_per_episode_csv(derived, per_ep_path)
    write_summary_csv(summary, args.group_by, summary_csv_path)
    with open(summary_json_path, "w") as f:
        json.dump({
            "group_by": args.group_by,
            "n_input_files": len(csv_paths),
            "n_episodes": len(rows),
            "groups": summary,
        }, f, indent=2, sort_keys=True)

    print(f"[analyze_yield_freezing] episodes: {len(rows)} | files: {len(csv_paths)}")
    print(f"[analyze_yield_freezing] wrote: {per_ep_path}")
    print(f"[analyze_yield_freezing] wrote: {summary_csv_path}")
    print(f"[analyze_yield_freezing] wrote: {summary_json_path}")
    for key, vals in summary.items():
        print(
            f"  [{key}] n={vals['n_episodes']} "
            f"yield_precision={vals['yield_precision_mean']:.3f} "
            f"yield_recall={vals['yield_recall_mean']:.3f} "
            f"bad_yield_rate={vals['bad_yield_rate_mean']:.3f} "
            f"timeout_rate={vals['timeout_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
