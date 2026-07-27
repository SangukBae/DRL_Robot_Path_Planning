#!/usr/bin/env python3
"""Resume a profile run: ``run_profile.py <profile> --resume`` shorthand.

Usage:
    python3 resume_profile.py phase2/both --seed 0 [--exec-trainer]

Validates that a resumable checkpoint / replay buffer / curriculum state
actually exists (hard error otherwise) and prints — or execs — the resume
command.
"""

import os
import sys

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    argv = [sys.executable, os.path.join(here, "run_profile.py")] + sys.argv[1:]
    if "--resume" not in argv:
        argv.append("--resume")
    os.execv(sys.executable, argv)
