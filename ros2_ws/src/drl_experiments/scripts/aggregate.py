#!/usr/bin/env python3
"""Aggregate multi-seed results — thin wrapper over drl_agent's canonical
``drl_agent.evaluation.analysis.aggregate_results`` module, with the runtime
root defaulted to the drl_agent package's ``runtime/`` dir.

Usage:
    python3 aggregate.py [--runtime-root PATH] [aggregate_results args...]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_MODULE = "drl_agent.evaluation.analysis.aggregate_results"


def main():
    pkg_root = _bootstrap.ensure_drl_agent_importable()
    target = _bootstrap.canonical_module_file(_MODULE)
    if not target:
        print(f"ERROR: {_MODULE} not found (source tree or installed "
              "drl_agent required)", file=sys.stderr)
        sys.exit(1)

    args = sys.argv[1:]
    if "--runtime-root" not in args:
        default_root = os.path.join(pkg_root, "runtime") if pkg_root else ""
        if default_root and os.path.isdir(default_root):
            args = ["--runtime-root", default_root] + args
    os.execv(sys.executable, [sys.executable, target] + args)


if __name__ == "__main__":
    main()
