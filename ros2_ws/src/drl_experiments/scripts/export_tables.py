#!/usr/bin/env python3
"""Export grouped result tables (paper/ablation) — thin wrapper over
drl_agent's ``aux_ablation_summary.py`` (per-run manifest-grouped eval
summaries).

Usage:
    python3 export_tables.py [aux_ablation_summary args...]
    # e.g. --group-by full --strict-manifest --out ../outputs/tables
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402


def main():
    target = _bootstrap.legacy_utils_script("aux_ablation_summary.py")
    if not target:
        print("ERROR: aux_ablation_summary.py not found (source tree or installed "
              "drl_agent required)", file=sys.stderr)
        sys.exit(1)
    os.execv(sys.executable, [sys.executable, target] + sys.argv[1:])


if __name__ == "__main__":
    main()
