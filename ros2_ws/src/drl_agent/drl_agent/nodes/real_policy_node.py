#!/usr/bin/env python3
"""Profile-based real-robot policy entrypoint.

Usage:

    ros2 run drl_agent real_policy_node.py --ros-args \
        -p profile:=phase2/both -p actor_path:=<..._actor.pth>

Resolves + validates the profile, then ``exec``s the canonical
``drl_agent.evaluation.real_policy_runner`` module with ``env_config`` /
``hparams_config`` pointing at the profile's yamls. Without ``profile`` it is
a pure passthrough.
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
        nc.exec_module("drl_agent.evaluation.real_policy_runner", {}, passthrough)
        return

    spec, _entry = nc.load_and_validate(opts["profile"], resume=False, seed=None)
    if opts["validate_only"]:
        print("[profile] validation OK (validate_only — not launching).")
        return

    params = {
        "env_config": spec.environment_yaml,
        "hparams_config": spec.hparams_yaml,
    }
    nc.exec_module("drl_agent.evaluation.real_policy_runner", params, passthrough)


if __name__ == "__main__":
    main()
