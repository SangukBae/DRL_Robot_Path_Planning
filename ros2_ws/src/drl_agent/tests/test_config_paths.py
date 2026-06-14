"""ROS-free unit tests for utils/config_paths.py.

These pin the exact semantics the trainers/test scripts rely on, so that wiring
the helper into _find_config_file() cannot silently change behaviour — in
particular the "file hint resolves against its DIRECTORY" rule.
"""

import os

import config_paths as cp


def test_expand_user_path_handles_empty():
    assert cp.expand_user_path("") == ""
    assert cp.expand_user_path(None) == ""
    assert cp.expand_user_path("~") == os.path.expanduser("~")


def test_location_candidate_file_hint_resolves_against_its_dir(tmp_path):
    # A FILE hint must look for `filename` NEXT TO it (its dir), not return itself.
    sibling = tmp_path / "train_tqc_config.yaml"
    sibling.write_text("a: 1\n")
    target = tmp_path / "hyperparameters_tqc.yaml"
    target.write_text("b: 2\n")
    cand = cp.location_candidate(str(sibling), "hyperparameters_tqc.yaml")
    assert cand == str(target)


def test_location_candidate_dir_hint_joins(tmp_path):
    cand = cp.location_candidate(str(tmp_path), "cfg.yaml")
    assert cand == str(tmp_path / "cfg.yaml")


def test_location_candidate_empty_or_missing_returns_blank():
    assert cp.location_candidate("", "cfg.yaml") == ""
    assert cp.location_candidate(None, "cfg.yaml") == ""
    assert cp.location_candidate("/no/such/path", "cfg.yaml") == ""


def test_first_existing_file(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    b.write_text("x: 1\n")
    assert cp.first_existing_file([str(a), str(b)]) == str(b)
    assert cp.first_existing_file([str(a)]) == ""
    assert cp.first_existing_file(["", None, str(b)]) == str(b)


def test_candidate_config_paths_is_pure_join():
    out = cp.candidate_config_paths("env.yaml", ["/x", "", "/y/z"])
    assert out == [os.path.join("/x", "env.yaml"),
                   os.path.join("/y/z", "env.yaml")]


def test_find_config_file_file_hint_then_dir_hint(tmp_path):
    # file hint → its dir
    f = tmp_path / "anchor.yaml"
    f.write_text("a: 1\n")
    target = tmp_path / "cfg.yaml"
    target.write_text("b: 2\n")
    assert cp.find_config_file("cfg.yaml", [str(f)]) == str(target)
    # dir hint
    assert cp.find_config_file("cfg.yaml", [str(tmp_path)]) == str(target)
    # first existing across multiple hints wins
    other = tmp_path / "other"
    other.mkdir()
    assert cp.find_config_file("cfg.yaml", [str(other), str(tmp_path)]) == str(target)


def test_find_config_file_returns_empty_when_missing(tmp_path):
    assert cp.find_config_file("nope.yaml", [str(tmp_path)]) == ""
    assert cp.find_config_file("nope.yaml", ["", None]) == ""
