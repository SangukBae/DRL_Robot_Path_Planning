#!/usr/bin/env python3
"""Validate an experiment profile and print (or exec) its launch commands.

Usage:
    # validate + print the two-terminal launch commands
    python3 run_profile.py phase2/both --seed 0

    # config-only validation (CI / pre-flight)
    python3 run_profile.py phase2/both --validate-only

    # actually start the TRAINER in this terminal (env node must already run)
    python3 run_profile.py phase2/both --seed 0 --exec-trainer

    # whole sweep: print one trainer command per (profile, seed)
    python3 run_profile.py --sweep ../sweeps/phase2_seeds.yaml

The default is deliberately print-not-launch: training needs Gazebo + the env
node in separate terminals, so this script validates strongly and tells you
exactly what to run where.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.ensure_drl_agent_importable()

from drl_agent.common import compat            # noqa: E402
from drl_agent.config import ConfigValidator, ProfileError, ProfileLoader  # noqa: E402
from drl_agent.training import registry        # noqa: E402


def _commands_for(spec, entry, seed, resume):
    env_cmd = (f"ros2 run drl_agent environment_curriculum_node.py "
               f"--ros-args -p profile:={spec.name}")
    train_cmd = (f"ros2 run drl_agent train_node.py --ros-args "
                 f"-p profile:={spec.name}")
    if seed is not None:
        train_cmd += f" -p seed:={seed}"
    if resume:
        train_cmd += " -p resume:=true"
    return env_cmd, train_cmd


def _validate(name, seed, resume):
    loader = ProfileLoader()
    try:
        spec = loader.load(name)
    except ProfileError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        avail = loader.available_profiles()
        if avail:
            print("available profiles:", ", ".join(avail), file=sys.stderr)
        sys.exit(1)
    entry = registry.lookup(spec.algorithm, spec.trainer)
    rep = ConfigValidator(spec).validate(
        resume=resume, seed=seed, package_root=compat.package_source_root())
    print(f"== profile {spec.name} ({spec.algorithm}/{spec.trainer}) ==")
    print(rep.summary())
    if not rep.ok:
        print("validation FAILED", file=sys.stderr)
        sys.exit(1)
    return spec, entry


def _load_sweep(path):
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    sweep = data.get("sweep") or {}
    return list(sweep.get("profiles") or []), list(sweep.get("seeds") or [None])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", nargs="?", help="profile name, e.g. phase2/both")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--exec-trainer", action="store_true",
                    help="exec the trainer here (env node must already be running)")
    ap.add_argument("--sweep", default="",
                    help="sweep yaml — validate + print commands for every (profile, seed)")
    ap.add_argument("--list", action="store_true", help="list available profiles")
    args = ap.parse_args()

    if args.list:
        for p in ProfileLoader().available_profiles():
            print(p)
        return

    if args.sweep:
        profiles, seeds = _load_sweep(args.sweep)
        for name in profiles:
            spec, entry = _validate(name, None, args.resume)
            for seed in seeds:
                _env, train_cmd = _commands_for(spec, entry, seed, args.resume)
                print(f"  {train_cmd}")
        print("\n(env node, once per profile change): "
              "ros2 run drl_agent environment_curriculum_node.py --ros-args -p profile:=<name>")
        return

    if not args.profile:
        ap.error("profile name (or --sweep/--list) required")

    spec, entry = _validate(args.profile, args.seed, args.resume)
    if args.validate_only:
        print("validation OK")
        return

    env_cmd, train_cmd = _commands_for(spec, entry, args.seed, args.resume)
    if args.exec_trainer:
        # Delegate to train_node (same validation path, then exec the resolved
        # canonical trainer module).
        node = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "drl_agent", "drl_agent", "nodes", "train_node.py")
        node = os.path.normpath(node)
        argv = [sys.executable, node, "--profile", spec.name]
        if args.seed is not None:
            argv += ["--seed", str(args.seed)]
        if args.resume:
            argv += ["--resume"]
        os.execv(sys.executable, argv)

    print("\nRun in separate terminals (Gazebo first):")
    print(f"  [1] ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false")
    print(f"  [2] {env_cmd}")
    print(f"  [3] {train_cmd}")


if __name__ == "__main__":
    main()
