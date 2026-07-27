#!/usr/bin/env python3
# SIM_VALIDATION: simulation verification summary (localization-aware RL).
# VALIDATION_ONLY — reads loc_validation_*.csv and prints/saves a pass-fail
# summary. Remove with this file when validation is done.
"""Summarise a localization-validation run.

Reads the latest loc_validation_step_*.csv (+ loc_validation_reset_*.csv) under
a log dir and reports obs-vs-gt error, reward/done source consistency, reset→
first-step jump, stale counts, per-stage noise params, and a noise-OFF
regression check. Console summary + validation_summary.json.

Usage:
  python3 sim_validation_summary.py --log-dir <run_dir>/logs
  python3 sim_validation_summary.py --step-csv <...> --reset-csv <...> --out <...>.json
"""

import os
import csv
import glob
import json
import argparse
import math
from collections import defaultdict

import numpy as np


def _load(path):
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


def _latest(log_dir, prefix):
    cands = glob.glob(os.path.join(log_dir, f"{prefix}_*.csv"))
    return max(cands, key=os.path.getmtime) if cands else None


def _circular_angle_diff(a, b):
    """SIM_VALIDATION: smallest signed angle difference a-b, wrapped to [-pi, pi]."""
    return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi


def summarize(step_rows, reset_rows):
    s = {}
    if step_rows:
        og = np.array([r["obs_goal_dist"] for r in step_rows])
        gg = np.array([r["gt_goal_dist"] for r in step_rows])
        oh = np.array([r["obs_heading_err"] for r in step_rows])
        gh = np.array([r["gt_heading_err"] for r in step_rows])
        s["n_steps"] = len(step_rows)
        s["mean_abs_goal_dist_error"] = float(np.mean(np.abs(og - gg)))
        s["mean_abs_heading_error_delta"] = float(np.mean(np.abs(oh - gh)))

        # reward/done source consistency
        def consistent(rows, used_key, use_gt_key):
            ok = 0
            for r in rows:
                exp = r["gt_goal_dist"] if r[use_gt_key] >= 0.5 else r["obs_goal_dist"]
                if abs(r[used_key] - exp) < 1e-4:
                    ok += 1
            return ok / len(rows)
        s["fraction_reward_uses_gt_consistently"] = consistent(step_rows, "reward_goal_dist_used", "use_gt_for_reward")
        s["fraction_done_uses_gt_consistently"] = consistent(step_rows, "done_goal_dist_used", "use_gt_for_done")

        # stale episodes
        ep_stale_loc, ep_stale_prop = set(), set()
        for r in step_rows:
            if r.get("stale_loc", 0) >= 0.5:
                ep_stale_loc.add(int(r["episode"]))
            if r.get("stale_proprio", 0) >= 0.5:
                ep_stale_prop.add(int(r["episode"]))
        s["episodes_with_stale_loc"] = len(ep_stale_loc)
        s["episodes_with_stale_proprio"] = len(ep_stale_prop)

        # per-stage noise params (as seen in the data)
        per_stage = {}
        by_stage = defaultdict(list)
        for r in step_rows:
            by_stage[int(r["curriculum_stage"])].append(r)
        for st, rs in sorted(by_stage.items()):
            last = rs[-1]
            per_stage[st] = {
                "enabled": int(last["loc_noise_enabled"]),
                "sigma_xy_m": last["loc_sigma_xy"],
                "sigma_yaw_rad": last["loc_sigma_yaw"],
                "delay_steps": int(last["loc_delay_steps"]),
                "jump_prob": last["loc_jump_prob"],
                "mean_abs_goal_dist_error": float(np.mean([abs(r["obs_goal_dist"] - r["gt_goal_dist"]) for r in rs])),
            }
        s["per_stage"] = per_stage

        # noise-OFF regression (step part): rows with loc_noise_enabled==0 must have obs==gt
        off = [r for r in step_rows if r["loc_noise_enabled"] < 0.5]
        if off:
            s["noise_off_max_goal_dist_error"] = float(max(abs(r["obs_goal_dist"] - r["gt_goal_dist"]) for r in off))
            s["noise_off_max_heading_error"] = float(max(abs(r["obs_heading_err"] - r["gt_heading_err"]) for r in off))

    if reset_rows:
        # The jump must compare the reset obs against the first step's PRE-MOTION
        # obs (before propagate) so real robot motion is excluded; fall back to
        # the post-motion first_step_obs_* for older CSVs that lack pre_motion_*.
        # Heading is recomputed via circular diff so legacy raw abs(a-b) rows that
        # crossed the ±pi wrap also summarise correctly.
        def _jump_d(r):
            cur = r["pre_motion_obs_goal_dist"] if "pre_motion_obs_goal_dist" in r else r["first_step_obs_goal_dist"]
            return abs(float(cur) - float(r["reset_obs_goal_dist"]))

        def _jump_h(r):
            cur = r["pre_motion_obs_heading_err"] if "pre_motion_obs_heading_err" in r else r["first_step_obs_heading_err"]
            return abs(_circular_angle_diff(cur, r["reset_obs_heading_err"]))

        s["max_reset_first_step_goal_jump"] = float(max(_jump_d(r) for r in reset_rows))
        s["max_reset_first_step_heading_jump"] = float(max(_jump_h(r) for r in reset_rows))
        # noise-OFF regression (reset part): reset→pre-motion jump must be 0.
        off_reset = [r for r in reset_rows if r["loc_noise_enabled"] < 0.5]
        if off_reset:
            s["noise_off_max_reset_goal_jump"] = float(max(_jump_d(r) for r in off_reset))
            s["noise_off_max_reset_heading_jump"] = float(max(_jump_h(r) for r in off_reset))

    # noise-OFF regression (combined): obs==gt AND reset jump==0 for off data.
    # Missing data defaults to pass (does not fail the run), but the flag is only
    # emitted when at least one source of off-data is present.
    EPS = 1e-6
    if ("noise_off_max_goal_dist_error" in s) or ("noise_off_max_reset_goal_jump" in s):
        step_ok = (s.get("noise_off_max_goal_dist_error", 0.0) < EPS
                   and s.get("noise_off_max_heading_error", 0.0) < EPS)
        reset_ok = (s.get("noise_off_max_reset_goal_jump", 0.0) < EPS
                    and s.get("noise_off_max_reset_heading_jump", 0.0) < EPS)
        s["noise_off_regression_ok"] = bool(step_ok and reset_ok)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--step-csv", default=None)
    ap.add_argument("--reset-csv", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    step_csv = a.step_csv or (_latest(a.log_dir, "loc_validation_step") if a.log_dir else None)
    reset_csv = a.reset_csv or (_latest(a.log_dir, "loc_validation_reset") if a.log_dir else None)
    if not step_csv or not os.path.isfile(step_csv):
        print("[SIM_VALIDATION] no loc_validation_step_*.csv found. "
              "Run with enable_sim_validation_logging:=true and sim_validation_runner.py first.")
        return
    step_rows = _load(step_csv)
    reset_rows = _load(reset_csv) if (reset_csv and os.path.isfile(reset_csv)) else []
    s = summarize(step_rows, reset_rows)

    print("\n=== SIM_VALIDATION summary ===")
    print(f" step csv : {step_csv}")
    print(f" reset csv: {reset_csv}")
    for k in ("n_steps", "mean_abs_goal_dist_error", "mean_abs_heading_error_delta",
              "fraction_reward_uses_gt_consistently", "fraction_done_uses_gt_consistently",
              "max_reset_first_step_goal_jump", "max_reset_first_step_heading_jump",
              "episodes_with_stale_loc", "episodes_with_stale_proprio"):
        if k in s:
            print(f"  {k:38} {s[k]}")
    if "noise_off_regression_ok" in s:
        tag = "PASS" if s["noise_off_regression_ok"] else "*** FAIL ***"
        print(f"  noise_off_regression_ok               {s['noise_off_regression_ok']}  [{tag}]")
        if not s["noise_off_regression_ok"]:
            print("    !! noise-OFF regression broken (obs==gt AND reset jump==0 expected):")
            print(f"       obs_goal_err={s.get('noise_off_max_goal_dist_error')} "
                  f"obs_head_err={s.get('noise_off_max_heading_error')} "
                  f"reset_goal_jump={s.get('noise_off_max_reset_goal_jump')} "
                  f"reset_head_jump={s.get('noise_off_max_reset_heading_jump')}")
    if s.get("per_stage"):
        print("  per-stage (stage: enabled sigma_xy delay jump_prob | mean|obs-gt|):")
        for st, d in s["per_stage"].items():
            print(f"    stage {st}: en={d['enabled']} sig_xy={d['sigma_xy_m']:.3f} "
                  f"delay={d['delay_steps']} jump={d['jump_prob']:.4f} | err={d['mean_abs_goal_dist_error']:.4f}")

    out = a.out or (os.path.join(os.path.dirname(step_csv), "validation_summary.json"))
    with open(out, "w") as f:
        json.dump(s, f, indent=2)
    print(f"\n[SIM_VALIDATION] wrote {out}")


if __name__ == "__main__":
    main()
