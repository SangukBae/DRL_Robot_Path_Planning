#!/usr/bin/env python3
"""Reproducibility check / inspection entry point.

This is the single automatable place to (a) run the fast ROS-free reproducibility
contract tests, (b) inspect what a given run was produced with, and (c) read the
documented procedures for the heavier human-stage integration smoke test and the
strict-determinism check. It needs NO ROS/Gazebo for `unit` and `manifest`.

Usage:
  python3 check_reproducibility.py unit                 # run contract unit tests
  python3 check_reproducibility.py manifest <logs_dir>  # print run provenance
  python3 check_reproducibility.py procedures           # print heavy/strict steps

Exit code is non-zero if the requested check fails (so it is CI-friendly).
"""

import json
import os
import subprocess
import sys

# The ROS-free tests that encode the reproducibility contract. Kept here so the
# check has one authoritative list (human RNG isolation + manifest provenance +
# determinism flags + per-episode reseed independence).
_CONTRACT_TESTS = [
    "tests/test_seed_utils.py",
    "tests/test_human_rng.py",
    "tests/test_human_spawn_pose_rng.py",
    "tests/test_run_manifest.py",
]

# Manifest fields that summarise "what produced this run".
_PROVENANCE_KEYS = [
    "run_tag", "seed", "seed_source",
    "git_commit_short", "git_branch", "git_dirty",
    "train_config_sha1", "curriculum_config_sha1",
    "hyperparameters_sha1", "environment_config_sha1",
    "determinism", "human_rng",
]


def _pkg_root():
    # drl_agent/evaluation/analysis/<this> -> package root is three levels up.
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", ".."))


def run_unit() -> int:
    root = _pkg_root()
    tests = [t for t in _CONTRACT_TESTS if os.path.isfile(os.path.join(root, t))]
    if not tests:
        print("[repro] no contract tests found", file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "pytest", "-p", "no:anyio", "-q", *tests]
    print(f"[repro] running: {' '.join(cmd)}\n[repro] cwd: {root}")
    return subprocess.call(cmd, cwd=root)


def show_manifest(logs_dir: str) -> int:
    # Prefer a specific per-run manifest if a single one exists, else the latest.
    candidates = []
    if os.path.isdir(logs_dir):
        for f in sorted(os.listdir(logs_dir)):
            if f.startswith("run_manifest_") and f.endswith(".json"):
                candidates.append(os.path.join(logs_dir, f))
        shared = os.path.join(logs_dir, "run_manifest.json")
        if os.path.isfile(shared):
            candidates.append(shared)
    elif os.path.isfile(logs_dir):
        candidates = [logs_dir]
    if not candidates:
        print(f"[repro] no run_manifest*.json under {logs_dir}", file=sys.stderr)
        return 2
    rc = 0
    for path in candidates:
        try:
            with open(path) as f:
                man = json.load(f)
        except Exception as exc:
            print(f"[repro] cannot read {path}: {exc}", file=sys.stderr)
            rc = 1
            continue
        print(f"\n=== {os.path.basename(path)} ===")
        for k in _PROVENANCE_KEYS:
            print(f"  {k}: {man.get(k)}")
        # Flag the two things most likely to silently break a same-seed compare.
        if man.get("git_dirty"):
            print("  [warn] git_dirty=True — run was made from uncommitted edits.")
        det = man.get("determinism") or {}
        if not det.get("use_deterministic_algorithms", False):
            print("  [warn] torch deterministic algorithms NOT confirmed for this run.")
    return rc


def print_procedures() -> int:
    print(_PROCEDURES.strip())
    return 0


_PROCEDURES = """
Human-stage reproducibility — heavy integration smoke test (needs built ws + Gazebo)
------------------------------------------------------------------------------------
Goal: confirm that two same-seed runs take the SAME early trajectory now that the
human RNG is isolated and the global stream is no longer perturbed by humans.

  1. Build:           cd ros2_ws && colcon build --packages-select drl_agent
  2. Two short runs at the SAME seed, each long enough to reach a human stage:
        # terminal A (sim) + terminal B (env) + terminal C (trainer), seed 0
        ros2 run drl_agent train_node.py --ros-args -p profile:=phase2/both -p seed:=0
     Run it twice into two different runtime dirs (or copy logs/ between runs).
  3. Compare the EARLY eval metrics of the two runs:
        python3 -m drl_agent.evaluation.analysis.check_reproducibility manifest <run1>/logs
        # (same git + cfg hashes?)
        # then diff the first few rows of eval_metrics_*.csv (success_rate / spl).
     With determinism on + identical code/config, the global-stream-driven task
     (start/goal/map/static) is reproducible; per-episode human SPAWN config is
     reproducible too. WITHIN-episode human MOTION is NOT bit-exact (wall-clock
     paced ticks) — expect tiny motion differences, not task-stream divergence.

Strict determinism check (deterministic_warn_only: false)
---------------------------------------------------------
train_tqc_config.yaml sets:
    deterministic_torch: true        # cuDNN det + use_deterministic_algorithms
    deterministic_warn_only: true    # downgrade "no det kernel" errors to warns
To find any op that lacks a deterministic CUDA kernel, flip warn_only off:
    ros2 run drl_agent train_node.py --ros-args \\
        -p profile:=phase2/both -p seed:=0 -p deterministic_warn_only:=false
torch then RAISES at the first non-deterministic op (e.g. a scatter/index_add),
naming it in the traceback. Use that to either replace the op or accept warn_only.
The applied flags are recorded per run in logs/run_manifest_<tag>.json:determinism.
"""


def main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "unit":
        return run_unit()
    if cmd == "manifest":
        if len(argv) < 2:
            print("usage: check_reproducibility.py manifest <logs_dir>", file=sys.stderr)
            return 2
        return show_manifest(argv[1])
    if cmd == "procedures":
        return print_procedures()
    print(f"[repro] unknown command: {cmd}\n", file=sys.stderr)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
