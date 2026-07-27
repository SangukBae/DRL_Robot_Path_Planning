#!/usr/bin/env python3
"""Aggregate multi-seed results — thin wrapper over drl_agent's
``aggregate_results.py`` with the runtime root defaulted to the drl_agent
package's ``runtime/`` dir.

Usage:
    python3 aggregate.py [--runtime-root PATH] [aggregate_results args...]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402


def main():
    target = _bootstrap.legacy_utils_script("aggregate_results.py")
    if not target:
        print("ERROR: aggregate_results.py not found (source tree or installed "
              "drl_agent required)", file=sys.stderr)
        sys.exit(1)

    args = sys.argv[1:]
    if "--runtime-root" not in args:
        pkg_root = _bootstrap.ensure_drl_agent_importable()
        default_root = os.path.join(pkg_root, "runtime") if pkg_root else ""
        if default_root and os.path.isdir(default_root):
            args = ["--runtime-root", default_root] + args
    os.execv(sys.executable, [sys.executable, target] + args)


if __name__ == "__main__":
    main()
