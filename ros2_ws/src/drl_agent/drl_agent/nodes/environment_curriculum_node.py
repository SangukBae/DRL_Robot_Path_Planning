#!/usr/bin/env python3
"""Profile-based environment entrypoint (curriculum env).

Usage:

    ros2 run drl_agent environment_curriculum_node.py --ros-args \
        -p profile:=phase2/both

Resolves the profile, validates it, then ``exec``s the canonical
``drl_agent.env.curriculum.environment_curriculum`` module (or
``drl_agent.env.simulation.environment`` for a base-trainer profile) with
``config_file:=<profile's environment yaml>`` plus the profile's declared
env-side flags (``risk_map_reward_enabled`` / ``action_risk_head_enabled``).
Without ``profile`` it is a pure passthrough to the curriculum environment
module (same behaviour as launching it directly).
"""

import os
import sys

_PKG_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if (os.path.isfile(os.path.join(_PKG_ROOT, "drl_agent", "__init__.py"))
        and _PKG_ROOT not in sys.path):
    sys.path.insert(0, _PKG_ROOT)

from drl_agent.nodes import _node_common as nc  # noqa: E402


def main():
    opts, passthrough = nc.parse_wrapper_args(sys.argv[1:])

    if not opts["profile"]:
        # No profile: pure passthrough to the canonical curriculum env module.
        nc.exec_module("drl_agent.env.curriculum.environment_curriculum", {}, passthrough)
        return

    spec, entry = nc.load_and_validate(opts["profile"], resume=False, seed=None)
    if opts["validate_only"]:
        print("[profile] validation OK (validate_only — not launching).")
        return

    params = {"config_file": spec.environment_yaml}
    for flag in ("risk_map_reward_enabled", "action_risk_head_enabled"):
        if flag in spec.overrides:
            params[flag] = str(spec.overrides[flag]).lower()

    nc.exec_module(entry.env_module, params, passthrough)


if __name__ == "__main__":
    main()
