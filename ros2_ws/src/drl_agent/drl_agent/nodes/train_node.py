#!/usr/bin/env python3
"""Profile-based training entrypoint.

Usage (new, preferred):

    ros2 run drl_agent train_node.py --ros-args \
        -p profile:=phase2/both -p seed:=0 -p resume:=true

    # config-only pre-flight (no ROS graph needed):
    python3 train_node.py --profile phase2/both --validate-only

Resolves the profile (drl_experiments/profiles/<name>/profile.yaml), runs the
strong ConfigValidator pre-flight, then ``exec``s the CANONICAL trainer module
for the profile's (algorithm, trainer) pair (``drl_agent.training.registry``)
with the translated parameters (``train_config_file:=<profile dir>``,
``load_model:=true`` on resume). Every algorithm the registry knows about is
also directly reachable without a profile via
``ros2 run drl_agent train_rl.py --ros-args -p rl_model:=<name>``.
"""

import os
import sys

# Make the drl_agent package importable when run from the source tree.
_PKG_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if (os.path.isfile(os.path.join(_PKG_ROOT, "drl_agent", "__init__.py"))
        and _PKG_ROOT not in sys.path):
    sys.path.insert(0, _PKG_ROOT)

from drl_agent.nodes import _node_common as nc  # noqa: E402


def main():
    opts, passthrough = nc.parse_wrapper_args(sys.argv[1:])
    if not opts["profile"]:
        print("usage: ros2 run drl_agent train_node.py --ros-args -p profile:=<group/variant> "
              "[-p seed:=N] [-p resume:=true]\n"
              "       python3 train_node.py --profile <group/variant> [--seed N] "
              "[--resume] [--validate-only]", file=sys.stderr)
        from drl_agent.config import ProfileLoader
        avail = ProfileLoader().available_profiles()
        if avail:
            print("available profiles: " + ", ".join(avail), file=sys.stderr)
        sys.exit(2)

    spec, entry = nc.load_and_validate(
        opts["profile"], resume=opts["resume"], seed=opts["seed"])
    if opts["validate_only"]:
        print("[profile] validation OK (validate_only — not launching).")
        return

    nc.export_profile_manifest_env(spec, resume=opts["resume"], seed=opts["seed"])

    params = {"train_config_file": spec.profile_dir}
    if opts["resume"]:
        params["load_model"] = "true"
    if opts["seed"] is not None:
        params["seed"] = opts["seed"]
    # Trainer-side mirror of the profile's declared flag (configs already carry
    # it — validated above — so this is belt-and-braces explicitness).
    if "action_risk_head_enabled" in spec.overrides:
        params["action_risk_head_enabled"] = str(spec.overrides["action_risk_head_enabled"]).lower()

    nc.exec_module(entry.trainer_module, params, passthrough)


if __name__ == "__main__":
    main()
