"""Unit tests for drl_agent.config.loader (ProfileLoader).

Covers: name lookup against a temp profiles root, direct-path loading,
relative-path resolution, error cases (missing profile, malformed manifest,
unknown override key), and the real phase2 profiles in drl_experiments (when
the sibling package is present in the source tree).
"""

import os
import textwrap

import pytest

from drl_agent.config import ProfileError, ProfileLoader
from drl_agent.config.schema import CONFIG_FILE_KEYS


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))


def _make_profile(root, name="grp/var", **kw):
    d = os.path.join(root, name)
    body = {
        "name": name,
        "algorithm": kw.get("algorithm", "tqc"),
        "trainer": kw.get("trainer", "curriculum"),
        "environment": "environment_curriculum.yaml",
        "train": "train_tqc_config.yaml",
        "hparams": "hyperparameters_tqc.yaml",
        "curriculum": "train_tqc_curriculum_config.yaml",
        "output_prefix": kw.get("output_prefix", "tqc_x"),
    }
    lines = ["profile:"] + [f"  {k}: {v}" for k, v in body.items()]
    if "overrides" in kw:
        lines.append("  overrides:")
        lines += [f"    {k}: {str(v).lower()}" for k, v in kw["overrides"].items()]
    _write(os.path.join(d, "profile.yaml"), "\n".join(lines) + "\n")
    return d


def test_load_by_name_resolves_relative_paths(tmp_path):
    root = str(tmp_path)
    d = _make_profile(root, "phase9/demo")
    loader = ProfileLoader(profiles_root=root)
    spec = loader.load("phase9/demo")
    assert spec.name == "phase9/demo"
    assert spec.profile_dir == d
    for key in CONFIG_FILE_KEYS:
        assert spec.config_paths[key].startswith(d + os.sep)
        assert not os.path.relpath(spec.config_paths[key], d).startswith("..")


def test_load_by_direct_path_and_manifest_file(tmp_path):
    d = _make_profile(str(tmp_path), "g/v")
    loader = ProfileLoader(profiles_root="/nonexistent")
    assert loader.load(d).name == "g/v"
    assert loader.load(os.path.join(d, "profile.yaml")).name == "g/v"


def test_missing_profile_raises_with_tried_paths(tmp_path):
    loader = ProfileLoader(profiles_root=str(tmp_path))
    with pytest.raises(ProfileError, match="not found"):
        loader.load("does/not_exist")


def test_missing_required_key_raises(tmp_path):
    d = os.path.join(str(tmp_path), "g/v")
    _write(os.path.join(d, "profile.yaml"), "profile:\n  name: g/v\n  trainer: curriculum\n")
    with pytest.raises(ProfileError, match="algorithm"):
        ProfileLoader(profiles_root=str(tmp_path)).load("g/v")


def test_unknown_override_key_raises(tmp_path):
    _make_profile(str(tmp_path), "g/v", overrides={"not_a_real_flag": True})
    with pytest.raises(ProfileError, match="unknown override"):
        ProfileLoader(profiles_root=str(tmp_path)).load("g/v")


def test_available_profiles_lists_nested_names(tmp_path):
    _make_profile(str(tmp_path), "a/one")
    _make_profile(str(tmp_path), "a/two")
    _make_profile(str(tmp_path), "b/three")
    loader = ProfileLoader(profiles_root=str(tmp_path))
    assert loader.available_profiles() == ["a/one", "a/two", "b/three"]


# --------------------------------------------------------------------------- #
#  Real phase2 profiles (source-tree only)                                     #
# --------------------------------------------------------------------------- #

_PHASE2 = ("phase2/baseline", "phase2/reward_shaping_only",
           "phase2/action_risk_head_only", "phase2/both")


def _experiments_present():
    return bool(ProfileLoader().profiles_root())


@pytest.mark.skipif(not _experiments_present(),
                    reason="drl_experiments profiles root not found")
@pytest.mark.parametrize("name", _PHASE2)
def test_real_phase2_profiles_load_and_point_at_existing_files(name):
    spec = ProfileLoader().load(name)
    assert spec.algorithm == "tqc"
    assert spec.trainer == "curriculum"
    assert spec.output_prefix == "tqc_phase2_" + name.split("/", 1)[1]
    for key, path in spec.config_paths.items():
        assert os.path.isfile(path), f"{name}: {key} missing at {path}"
    # PHASE2 matrix flags are declared explicitly on every variant.
    assert set(spec.overrides) == {"risk_map_reward_enabled",
                                   "action_risk_head_enabled"}
