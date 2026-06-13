#!/usr/bin/env python3
# AUX_ABLATION: aggregate aux on/off ablation runs into a paper-ready summary.
#
# Reads several eval_summary_*.csv files (written by aux_ablation_logging.
# EvalSummaryCSV) -- one per run/seed -- and the run_manifest.json sitting next
# to each, then groups runs and reports mean/std across seeds for the headline
# metrics.  Pure stdlib (csv / json / statistics / argparse / glob).
#
# Grouping (so different aux configs are NOT silently averaged together):
#   --group-by full        (default) group by the FULL aux config signature from
#                          run_manifest.json (aux_enabled, aux_version,
#                          num_sectors, horizons_sec, loss_weight,
#                          min_distance_loss_weight, use_distributional_aux,
#                          temporal_enabled; + git_commit with --include-git).
#                          Runs whose config differs form SEPARATE groups.
#   --group-by aux_enabled (loose) group by aux_enabled only (manifest not
#                          required) -> two groups aux_off / aux_on.
#   --strict-manifest      error out if ANY run lacks a readable manifest (or,
#                          implicitly, mixes configs you did not intend).
#
# Each run contributes ONE representative eval row:
#   * default        -> the final eval (max eval_global_t)
#   * --eval-step N  -> the eval whose eval_global_t is nearest to N
#
# Usage:
#   python3 aux_ablation_summary.py RUN_DIR_OR_CSV [more ...] \
#       [--group-by full|aux_enabled] [--strict-manifest] [--include-git] \
#       [--eval-step 100000] [--out logs/aux_ablation_summary.csv]

import argparse
import csv
import glob
import json
import math
import os
import statistics

# Must match aux_ablation_logging.MANIFEST_CONFIG_KEYS (kept local so this CLI
# stays self-contained / runnable without the package on the path).
MANIFEST_CONFIG_KEYS = [
    "aux_enabled", "aux_version", "num_sectors", "horizons_sec",
    "loss_weight", "min_distance_loss_weight", "use_distributional_aux",
    "temporal_enabled",
]
_SIG_SHORT = {
    "aux_enabled": "aux", "aux_version": "v", "num_sectors": "K",
    "horizons_sec": "H", "loss_weight": "lw", "min_distance_loss_weight": "mdlw",
    "use_distributional_aux": "distr", "temporal_enabled": "temp",
}

# Metrics aggregated per group (eval_summary column).
METRICS = [
    "success_rate", "collision_rate", "timeout_rate",
    "mean_reward", "spl", "cte", "jerk",
]

OUTPUT_HEADER = ["group", "config_signature", "n_runs"]
for _m in METRICS:
    OUTPUT_HEADER += [f"{_m}_mean", f"{_m}_std"]


def _find_csvs(paths):
    found = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            found += sorted(glob.glob(os.path.join(p, "**", "eval_summary_*.csv"),
                                      recursive=True))
        elif os.path.isfile(p):
            found.append(p)
        else:
            print(f"[AUX_ABLATION][warn] input not found: {p}")
    seen, uniq = set(), []
    for f in found:
        rp = os.path.realpath(f)
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    return uniq


def _to_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _pick_row(rows, eval_step):
    cand = [r for r in rows if _to_float(r.get("eval_global_t")) is not None]
    if not cand:
        return None
    if eval_step is None:
        return max(cand, key=lambda r: _to_float(r["eval_global_t"]))
    return min(cand, key=lambda r: abs(_to_float(r["eval_global_t"]) - float(eval_step)))


def _load_manifest(csv_path):
    man_path = os.path.join(os.path.dirname(csv_path), "run_manifest.json")
    if not os.path.isfile(man_path):
        return None
    try:
        with open(man_path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"[AUX_ABLATION][warn] cannot read {man_path}: {exc}")
        return None


def _fmt(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, list):
        return "[" + ",".join(f"{float(x):g}" for x in v) + "]"
    if v is None:
        return "?"
    return str(v)


def _config_signature(man, include_git):
    if man is None:
        return "(no-manifest)"
    parts = [f"{_SIG_SHORT[k]}{_fmt(man.get(k))}" for k in MANIFEST_CONFIG_KEYS]
    if include_git and man.get("git_commit"):
        parts.append(f"git{man['git_commit']}")
    return "|".join(parts)


def _aux_enabled_of(row, man):
    # Prefer manifest, fall back to the eval_summary column.
    if man is not None and man.get("aux_enabled") is not None:
        return int(round(float(man["aux_enabled"])))
    v = _to_float(row.get("aux_enabled"))
    return int(round(v)) if v is not None else None


def _load_run(csv_path, eval_step):
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        print(f"[AUX_ABLATION][warn] cannot read {csv_path}: {exc}")
        return None
    if not rows:
        print(f"[AUX_ABLATION][warn] empty eval summary: {csv_path}")
        return None
    row = _pick_row(rows, eval_step)
    if row is None:
        print(f"[AUX_ABLATION][warn] no usable eval row in {csv_path}")
        return None
    return row, _load_manifest(csv_path), csv_path


def _agg_group(rows):
    out = {}
    for m in METRICS:
        vals = [v for v in (_to_float(r.get(m)) for r in rows) if v is not None]
        if not vals:
            out[m] = (float("nan"), float("nan"))
        elif len(vals) == 1:
            out[m] = (vals[0], 0.0)
        else:
            out[m] = (statistics.mean(vals), statistics.stdev(vals))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="AUX_ABLATION: aggregate aux on/off eval_summary CSVs.")
    ap.add_argument("inputs", nargs="+",
                    help="run directories and/or eval_summary_*.csv files")
    ap.add_argument("--group-by", choices=["full", "aux_enabled"], default="full",
                    help="full = group by complete aux config (manifest); "
                         "aux_enabled = group by on/off only (default: full)")
    ap.add_argument("--strict-manifest", action="store_true",
                    help="error if any run lacks a readable run_manifest.json")
    ap.add_argument("--include-git", action="store_true",
                    help="include git_commit in the full config signature")
    ap.add_argument("--eval-step", type=float, default=None,
                    help="aggregate the eval nearest this global timestep "
                         "(default: each run's final eval)")
    ap.add_argument("--out", default="logs/aux_ablation_summary.csv",
                    help="output summary CSV path")
    args = ap.parse_args()

    csvs = _find_csvs(args.inputs)
    if not csvs:
        print("[AUX_ABLATION][error] no eval_summary_*.csv found in inputs.")
        return 2
    print(f"[AUX_ABLATION] found {len(csvs)} run CSV(s); group-by={args.group_by}.")

    # group_key -> (group_label, config_signature, [rows])
    groups = {}
    n_missing_manifest = 0
    for c in csvs:
        loaded = _load_run(c, args.eval_step)
        if loaded is None:
            continue
        row, man, path = loaded
        if man is None:
            n_missing_manifest += 1
            if args.strict_manifest:
                print(f"[AUX_ABLATION][error] --strict-manifest: missing/unreadable "
                      f"run_manifest.json next to {path}")
                return 3

        aux_enabled = _aux_enabled_of(row, man)
        if aux_enabled is None:
            print(f"[AUX_ABLATION][warn] cannot determine aux_enabled for {path}; skipped.")
            continue
        label = "aux_on" if aux_enabled == 1 else "aux_off"

        if args.group_by == "aux_enabled":
            sig = ""
            key = (label,)
        else:  # full
            sig = _config_signature(man, args.include_git)
            key = (label, sig)

        if key not in groups:
            groups[key] = [label, sig, []]
        groups[key][2].append(row)

    if n_missing_manifest and not args.strict_manifest:
        print(f"[AUX_ABLATION][warn] {n_missing_manifest} run(s) had no manifest; "
              "grouped under config_signature='(no-manifest)' (use --strict-manifest "
              "to forbid, or --group-by aux_enabled to ignore config).")

    if not groups:
        print("[AUX_ABLATION][error] nothing to aggregate.")
        return 2

    out_rows = []
    for key in sorted(groups):
        label, sig, rows = groups[key]
        if len(rows) < 2:
            print(f"[AUX_ABLATION][warn] group '{label}' [{sig or 'any'}] has only "
                  f"{len(rows)} run -> std is 0 / not meaningful; add seeds (>=3).")
        agg = _agg_group(rows)
        line = [label, sig, len(rows)]
        for m in METRICS:
            mean, std = agg[m]
            line += [round(mean, 4) if math.isfinite(mean) else "nan",
                     round(std, 4) if math.isfinite(std) else "nan"]
        out_rows.append(line)

    out_path = os.path.expanduser(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUTPUT_HEADER)
        w.writerows(out_rows)

    step_note = ("final eval" if args.eval_step is None
                 else f"eval nearest t={int(args.eval_step)}")
    print(f"[AUX_ABLATION] wrote {out_path} ({step_note}, {len(out_rows)} group(s)).")
    for line in out_rows:
        print(f"  {line[0]} [{line[1] or 'any'}] n={line[2]} "
              f"success_rate={line[3]}+/-{line[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
