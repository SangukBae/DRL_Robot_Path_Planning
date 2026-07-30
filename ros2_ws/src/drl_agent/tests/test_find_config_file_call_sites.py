"""Structural regression test for a discovered bug class: every
``self._find_config_file(filename)`` call in the TQC curriculum trainer must
pass a second argument (the profile's ``train_config_file`` hint,
``self._train_config_file_param`` -- cached once in TrainTQCBase.__init__),
not rely on ``_find_config_file``'s default ``user_param_path=None``.

Background: ``_init_motion_logging_contract`` (episode CSV telemetry) and the
curriculum-config / run-manifest lookups in ``TrainTQCCurriculum``/its
content-identical duplicate in ``train_rl.py`` called
``self._find_config_file("environment_curriculum.yaml")`` /
``self._find_config_file("train_tqc_curriculum_config.yaml")`` with NO hint.
Without the hint, ``_find_config_file`` skips the profile directory entirely
and falls through to drl_agent's generic package-default config (source-tree
search order) -- a DIFFERENT file than the one the profile actually names.

This was silently benign for phase2 profiles (whose action_dim=3 matches the
generic default's), but reproduced as an ACTUAL CRASH for
phase3/speed_steering_risk_balanced (action_dim=2): the trainer read the
generic action_dim=3 config for its motion-telemetry mirror, then crashed
calling pure_pursuit.hybrid_action_to_command with a 2-D action against
3-element actions_low/high arrays (ValueError: operands could not be
broadcast together with shapes (2,) (3,)), on literally the first training
step -- discovered during a real Gazebo/ROS smoke run, not by any pure-Python
unit test, since a mocked/stubbed test never exercises real config-file
discovery on disk.

This test parses the AST of the trainer source files (no ROS/torch import
needed) and asserts every ``self._find_config_file(...)`` CALL SITE (not the
method definition itself) passes at least 2 arguments (or an explicit
``user_param_path=`` keyword) -- catching a future call site that omits the
hint before it ships, rather than only being caught by a live smoke run
against a profile whose action contract happens to differ from the generic
default.
"""

import ast
import os

import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "drl_agent")
)

_FILES_REQUIRING_HINT = [
    os.path.join(_REPO_ROOT, "training", "train_tqc_curriculum.py"),
    os.path.join(_REPO_ROOT, "training", "train_rl.py"),
]


def _find_config_file_call_sites(path):
    """Return a list of (lineno, num_args, has_keyword) for every
    ``self._find_config_file(...)`` CALL in the file (not the ``def``)."""
    with open(path, "r") as f:
        tree = ast.parse(f.read(), filename=path)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "_find_config_file"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "self"):
            continue
        sites.append((node.lineno, len(node.args), bool(node.keywords)))
    return sites


@pytest.mark.parametrize("path", _FILES_REQUIRING_HINT)
def test_every_find_config_file_call_passes_a_hint(path):
    assert os.path.isfile(path), f"expected file not found: {path}"
    sites = _find_config_file_call_sites(path)
    # Sanity: this file really does call _find_config_file somewhere (else
    # the test would vacuously pass and stop meaning anything).
    assert sites, f"no self._find_config_file(...) call sites found in {path}"
    missing_hint = [
        lineno for (lineno, nargs, has_kw) in sites
        if nargs < 2 and not has_kw
    ]
    assert not missing_hint, (
        f"{path}: self._find_config_file(filename) called WITHOUT a "
        f"user_param_path hint at line(s) {missing_hint} -- this falls "
        "through to drl_agent's generic package-default config instead of "
        "the running profile's own file (see this test's module docstring "
        "for the exact crash this caused). Pass self._train_config_file_param "
        "as the second argument."
    )


def test_train_tqc_base_caches_train_config_file_param():
    """The hint every subclass call site above relies on must actually be
    cached as self._train_config_file_param inside TrainTQCBase.__init__ --
    regression for the fix itself, in case the caching line is ever removed
    while the call sites above are (correctly) left passing it."""
    path = os.path.join(_REPO_ROOT, "training", "train_tqc_base.py")
    with open(path, "r") as f:
        src = f.read()
    assert "self._train_config_file_param = user_param_path" in src, (
        "TrainTQCBase.__init__ no longer caches self._train_config_file_param "
        "-- every self._find_config_file(filename, self._train_config_file_param) "
        "call site in train_tqc_curriculum.py/train_rl.py would then raise "
        "AttributeError on first use."
    )
