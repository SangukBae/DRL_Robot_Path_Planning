#!/usr/bin/env python3
"""Export grouped result tables (paper/ablation) — thin wrapper over
drl_agent's canonical ``drl_agent.evaluation.analysis.aux_ablation_summary``
module (per-run manifest-grouped eval summaries).

Usage:
    python3 export_tables.py [aux_ablation_summary args...]
    # e.g. --group-by full --strict-manifest --out ../outputs/tables
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_MODULE = "drl_agent.evaluation.analysis.aux_ablation_summary"


def main():
    _bootstrap.ensure_drl_agent_importable()
    target = _bootstrap.canonical_module_file(_MODULE)
    if not target:
        print(f"ERROR: {_MODULE} not found (source tree or installed "
              "drl_agent required)", file=sys.stderr)
        sys.exit(1)
    os.execv(sys.executable, [sys.executable, target] + sys.argv[1:])


if __name__ == "__main__":
    main()
