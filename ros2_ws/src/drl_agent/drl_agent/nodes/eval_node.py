#!/usr/bin/env python3
"""Profile-based evaluation entrypoint.

Usage:

    ros2 run drl_agent eval_node.py --ros-args \
        -p profile:=phase2/both -p seed:=0 \
        -p weight_prefix:=<checkpoint prefix> [-p conditions:="[5,6]"] ...

Resolves + validates the profile, then ``exec``s the unchanged
``generalization_eval.py`` harness (TQC eval over curriculum stages) with the
profile's configs. All generalization_eval parameters (``weight_prefix``,
``weights_dir``, ``world``, ``conditions``, …) pass through untouched.
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
        print("usage: ros2 run drl_agent eval_node.py --ros-args -p profile:=<group/variant> "
              "[-p seed:=N] [generalization_eval params...]", file=sys.stderr)
        sys.exit(2)

    spec, _entry = nc.load_and_validate(
        opts["profile"], resume=False, seed=opts["seed"])
    if opts["validate_only"]:
        print("[profile] validation OK (validate_only — not launching).")
        return

    params = {"train_config_file": spec.profile_dir}
    if opts["seed"] is not None:
        params["seed"] = opts["seed"]

    # generalization_eval.py is the (TQC-based) eval harness for every profile
    # algorithm we currently evaluate this way.
    nc.exec_legacy("generalization_eval.py", params, passthrough)


if __name__ == "__main__":
    main()
