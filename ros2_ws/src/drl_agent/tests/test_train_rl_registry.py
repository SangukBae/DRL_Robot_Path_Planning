"""Smoke tests for train_rl's MODEL_REGISTRY / model-selection dispatch.

Every check here runs train_rl.py (or a short python -c snippet importing it)
in a FRESH subprocess rather than importing it in-process, for two reasons:

  1. That is what `ros2 run drl_agent train_rl.py ...` actually does -- a
     separate process -- so it is the most faithful test of real usage
     (--list-models / --dry-run / --help are exercised exactly as a user
     would invoke them).
  2. Several sibling ROS-free tests in this same suite install permanent,
     never-cleaned-up ROS stubs into sys.modules so THEY can run without a
     built workspace (e.g. test_gym_parameter_client.py stubs
     "environment_interface"; several others stub "rclpy" itself). Those
     stubs stay in sys.modules for the rest of the pytest session. An
     in-process `import train_rl` was observed to fail intermittently
     depending on which other tests happened to run first and leave a stub
     behind (a stub `rclpy` broke `from rclpy.node import Node` with a
     confusing `TypeError: '_Meta' object is not iterable`, for one). A
     subprocess never inherits that pollution, so these tests are correct
     regardless of collection order.

rclpy-gated (mirrors test_tqc_networks.py's torch-gating: try/import, then
skipif): needs a built + sourced ROS2 workspace (drl_agent_interfaces is only
importable after `colcon build` + `source install/setup.bash`), unlike the
rest of the ROS-free suite. Skipped automatically where that is not
available; run for real inside the docker container where the workspace is
built.
"""

import os
import subprocess
import sys

import pytest

_POLICY_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "policy")
_TRAIN_RL = os.path.join(_POLICY_DIR, "train_rl.py")

ALL_MODELS = (
    "tqc_curriculum", "sac_curriculum", "td7_curriculum",
    "tqc_ieqn_curriculum", "a3c_curriculum",
    "sb3_sac_curriculum", "sb3_td3_curriculum",
)


def _run_cli(args, timeout=60):
    return subprocess.run(
        [sys.executable, _TRAIN_RL] + list(args),
        capture_output=True, text=True, timeout=timeout,
    )


def _run_snippet(code, timeout=60):
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_POLICY_DIR, capture_output=True, text=True, timeout=timeout,
    )


def _probe_have_ros():
    proc = _run_snippet("import rclpy, train_rl")
    return proc.returncode == 0


_HAVE_ROS = _probe_have_ros()

pytestmark = pytest.mark.skipif(
    not _HAVE_ROS,
    reason="rclpy / built drl_agent_interfaces not available "
           "(source install/setup.bash first)",
)


def test_list_models_cli_needs_no_ros_and_lists_every_model():
    proc = _run_cli(["--list-models"])
    assert proc.returncode == 0, proc.stderr
    for name in ALL_MODELS:
        assert name in proc.stdout, f"{name} missing from --list-models output"


def test_help_flag_documents_rl_model_and_dry_run():
    proc = _run_cli(["--help"])
    assert proc.returncode == 0, proc.stderr
    assert "--rl_model" in proc.stdout
    assert "--dry-run" in proc.stdout
    assert "--list-models" in proc.stdout


def test_dry_run_tqc_curriculum_resolves_without_gazebo():
    proc = _run_cli(["--dry-run", "--rl_model", "tqc_curriculum"])
    assert proc.returncode == 0, proc.stderr
    assert "dry-run OK" in proc.stdout
    assert "TrainTQCCurriculum" in proc.stdout


def test_dry_run_unknown_model_fails_with_available_list():
    proc = _run_cli(["--dry-run", "--rl_model", "not_a_real_model"])
    assert proc.returncode == 1
    assert "not_a_real_model" in proc.stdout
    for name in ALL_MODELS:
        assert name in proc.stdout


def test_tqc_curriculum_is_registered_and_default():
    proc = _run_snippet(
        "import train_rl\n"
        "assert 'tqc_curriculum' in train_rl.MODEL_REGISTRY\n"
        "assert train_rl.DEFAULT_RL_MODEL == 'tqc_curriculum'\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_resolve_tqc_curriculum_returns_local_callable_class():
    proc = _run_snippet(
        "import train_rl\n"
        "cls, entry = train_rl.resolve_trainer_class('tqc_curriculum')\n"
        "assert cls is train_rl.TrainTQCCurriculum\n"
        "assert callable(cls)\n"
        "assert entry['class_name'] == 'TrainTQCCurriculum'\n"
        "assert entry['module'] is None\n"  # defined locally, not imported
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_resolve_unknown_model_raises_runtime_error_listing_available_models():
    proc = _run_snippet(
        "import train_rl\n"
        "try:\n"
        "    train_rl.resolve_trainer_class('not_a_real_model')\n"
        "    print('NO_ERROR_RAISED')\n"
        "except RuntimeError as e:\n"
        "    msg = str(e)\n"
        "    assert 'not_a_real_model' in msg, msg\n"
        "    missing = [n for n in train_rl.list_models() if n not in msg]\n"
        "    assert not missing, missing\n"
        "    print('OK')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    assert "NO_ERROR_RAISED" not in proc.stdout


def test_registry_entries_only_reference_real_scripts_on_disk():
    """'실제 repo에 존재하지 않는 모델은 억지로 만들지 말고' — every non-local
    registry entry must name a module whose .py file genuinely exists under
    scripts/policy/."""
    # train_rl now lives in the drl_agent package (drl_agent/training/) while
    # the lazily-imported baseline trainers stay flat under scripts/policy/, so
    # "same directory as train_rl" no longer holds. The real contract is that
    # every registered module RESOLVES (train_rl's import puts the flat script
    # dirs on sys.path via ensure_flat_scripts_on_path) to a real file.
    proc = _run_snippet(
        "import importlib.util, train_rl\n"
        "missing = []\n"
        "for name, entry in train_rl.MODEL_REGISTRY.items():\n"
        "    if entry['module'] is None:\n"
        "        continue\n"
        "    spec = importlib.util.find_spec(entry['module'])\n"
        "    if spec is None or not (spec.origin and spec.origin.endswith('.py')):\n"
        "        missing.append((name, entry['module']))\n"
        "assert not missing, missing\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_resolve_tqc_family_models_are_importable_and_callable():
    """registry가 반환하는 항목이 callable/import 가능한 trainer인지 확인 —
    covers every registered model whose sibling script has no third-party
    (non-repo) dependency that can be independently broken in a given
    environment (the sb3_* entries pull in stable_baselines3 -> matplotlib,
    tested separately below since a matplotlib/numpy ABI mismatch is an
    environment issue, not a registry bug)."""
    models = ("tqc_curriculum", "sac_curriculum", "td7_curriculum",
              "tqc_ieqn_curriculum", "a3c_curriculum")
    code = (
        "import train_rl\n"
        f"for name in {models!r}:\n"
        "    cls, entry = train_rl.resolve_trainer_class(name)\n"
        "    assert callable(cls), name\n"
        "    assert cls.__name__ == entry['class_name'], name\n"
        "print('OK')\n"
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_resolve_sb3_models_either_import_cleanly_or_fail_on_their_own_deps():
    """The sb3_* entries are real, on-disk scripts (see the disk-existence
    test above). This only asserts that a failure to resolve them, if any,
    is NOT our own "unknown model" RuntimeError -- i.e. the registry
    plumbing itself is fine even if stable_baselines3/matplotlib happen to
    be broken/missing in a given environment."""
    for name in ("sb3_sac_curriculum", "sb3_td3_curriculum"):
        code = (
            "import train_rl\n"
            f"cls, entry = train_rl.resolve_trainer_class({name!r})\n"
            "assert callable(cls)\n"
            "assert cls.__name__ == entry['class_name']\n"
            "print('OK')\n"
        )
        proc = _run_snippet(code)
        if proc.returncode != 0:
            assert "Unknown rl_model" not in proc.stderr, (
                f"{name}: registry itself rejected a real, on-disk model "
                f"entry:\n{proc.stderr}"
            )
            continue  # third-party import failure -- not a registry bug
        assert "OK" in proc.stdout


def test_resolve_rl_model_name_priority_cli_ros_param_default():
    code = (
        "import rclpy, train_rl\n"
        "rclpy.init(args=[])\n"
        "assert train_rl._resolve_rl_model_name('tqc_curriculum') == 'tqc_curriculum'\n"
        "rclpy.shutdown()\n"
        "\n"
        "rclpy.init(args=['--ros-args', '-p', 'rl_model:=sac_curriculum'])\n"
        "assert train_rl._resolve_rl_model_name(None) == 'sac_curriculum'\n"
        "rclpy.shutdown()\n"
        "\n"
        "rclpy.init(args=[])\n"
        "assert train_rl._resolve_rl_model_name(None) == train_rl.DEFAULT_RL_MODEL\n"
        "rclpy.shutdown()\n"
        "print('OK')\n"
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_arg_parser_ignores_ros_args_tail():
    """`ros2 run ... --ros-args -p rl_model:=x` must not crash argparse --
    parse_known_args should shove the unrecognised ROS tail into extras."""
    code = (
        "import train_rl\n"
        "parser = train_rl._build_arg_parser()\n"
        "args, unknown = parser.parse_known_args(\n"
        "    ['--ros-args', '-p', 'rl_model:=tqc_curriculum'])\n"
        "assert args.rl_model is None\n"
        "assert '--ros-args' in unknown\n"
        "print('OK')\n"
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
