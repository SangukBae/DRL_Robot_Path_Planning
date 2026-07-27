"""Installed-environment bare-name import contract.

The rest of the suite (via conftest.py) puts the SOURCE scripts/ dirs on
sys.path, so it can never detect a broken *installed* import path. This test
closes that gap: it runs ``import <bare-name>`` in a SUBPROCESS from a neutral
cwd, relying only on the sourced workspace environment — i.e. exactly what a
user typing ``python3 -c "import tqc_agent"`` after ``source install/setup.bash``
gets. The flat shim dir reaches PYTHONPATH through the package's
``env-hooks/flat_legacy_scripts.dsv.in`` hook (prepends ``lib/drl_agent``).

Skips (rather than fails) when no sourced install is available — e.g. on a
host running the suite straight from the source tree — and runs for real under
``colcon test`` / inside the docker workspace.
"""

import os
import subprocess
import sys

import pytest


def _installed_flat_dir_on_pythonpath():
    """Return <prefix>/lib/drl_agent if a sourced install put it on PYTHONPATH."""
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        entry = entry.rstrip(os.sep)
        if (entry.endswith(os.path.join("lib", "drl_agent"))
                and os.path.isfile(os.path.join(entry, "tqc_agent.py"))):
            return entry
    return ""


_FLAT_DIR = _installed_flat_dir_on_pythonpath()

pytestmark = pytest.mark.skipif(
    not _FLAT_DIR,
    reason="no sourced drl_agent install on PYTHONPATH (env hook inactive); "
           "build + `source install/setup.bash` to run this contract",
)

# Torch-free shims — must import in ANY sourced environment.
PURE_BARE = ["run_layout", "config_paths", "geometry_utils", "seed_utils",
             "file_manager", "pure_pursuit", "episode_metrics",
             "curriculum_stage_logic", "map_catalog", "reward_calculator"]

# Heavier shims — required by the feedback contract; need torch (+ROS for the
# env/trainer stack), so they are asserted only where those deps exist.
TORCH_BARE = ["tqc_networks", "buffer", "tqc_io", "tqc_agent"]
ROS_TORCH_BARE = ["environment", "train_rl"]


def _subprocess_import(names, tmp_path):
    """Import names in a fresh python from a neutral cwd (no source tree)."""
    code = (
        "import importlib, sys\n"
        + "\n".join(f"m_{i} = importlib.import_module({n!r})"
                    for i, n in enumerate(names))
        + "\nimport drl_agent"
        + "\n" + "\n".join(
            f"assert m_{i}.__name__.startswith('drl_agent.'), "
            f"({n!r}, m_{i}.__name__)" for i, n in enumerate(names))
        + "\nprint('OK')\n"
    )
    return subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120)


def test_installed_pure_bare_imports(tmp_path):
    proc = _subprocess_import(PURE_BARE, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_installed_torch_bare_imports(tmp_path):
    pytest.importorskip("torch")
    proc = _subprocess_import(TORCH_BARE, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_installed_ros_torch_bare_imports(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("drl_agent_interfaces")
    proc = _subprocess_import(ROS_TORCH_BARE, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
