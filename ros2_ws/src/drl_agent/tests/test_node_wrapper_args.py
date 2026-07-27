"""Unit tests for the wrapper-arg parsing in drl_agent.nodes._node_common.

The wrappers must (a) recognise both ROS-style ``-p key:=value`` and plain
``--key`` forms, (b) strip ONLY the wrapper-only tokens, (c) leave a
ROS-style ``-p seed:=N`` in the passthrough (the trainer node consumes it
natively) while converting a plain ``--seed N``, and (d) never leave a
dangling EMPTY ``--ros-args`` section behind after stripping (exec_module
would otherwise tack its own ``--ros-args`` section onto an already-empty
one).
"""

import yaml

from drl_agent.nodes._node_common import (_format_ros_param_value,
                                          _strip_empty_ros_args_sections,
                                          parse_wrapper_args)


def test_ros_style_profile_resume_are_stripped():
    argv = ["--ros-args", "-p", "profile:=phase2/both", "-p", "resume:=true",
            "-p", "seed:=3", "-p", "other:=x"]
    opts, out = parse_wrapper_args(argv)
    assert opts["profile"] == "phase2/both"
    assert opts["resume"] is True
    assert opts["seed"] == 3
    # profile/resume stripped; seed + unrelated params stay for the legacy node
    assert out == ["--ros-args", "-p", "seed:=3", "-p", "other:=x"]


def test_plain_cli_forms():
    argv = ["--profile", "phase2/baseline", "--seed", "5", "--resume",
            "--validate-only"]
    opts, out = parse_wrapper_args(argv)
    assert opts == {"profile": "phase2/baseline", "resume": True,
                    "validate_only": True, "seed": 5}
    assert out == []  # plain forms are wrapper-only, all stripped


def test_validate_only_ros_param_and_defaults():
    opts, out = parse_wrapper_args(["-p", "validate_only:=true"])
    assert opts["validate_only"] is True and out == []
    opts, out = parse_wrapper_args([])
    assert opts == {"profile": "", "resume": False, "validate_only": False,
                    "seed": None}


def test_unrelated_args_pass_through_untouched():
    argv = ["--ros-args", "-p", "weight_prefix:=tqc_x_seed_0_20260101",
            "-r", "__node:=foo"]
    opts, out = parse_wrapper_args(argv)
    assert opts["profile"] == ""
    assert out == argv


def test_ros_args_only_wrapper_params_leaves_no_dangling_marker():
    # Every token in the --ros-args section is wrapper-only -> the whole
    # section (including the "--ros-args" marker itself) must disappear.
    argv = ["--ros-args", "-p", "profile:=phase2/both"]
    opts, out = parse_wrapper_args(argv)
    assert opts["profile"] == "phase2/both"
    assert out == []


def test_ros_args_only_wrapper_params_with_leading_flag():
    argv = ["--flag", "--ros-args", "-p", "resume:=true"]
    opts, out = parse_wrapper_args(argv)
    assert opts["resume"] is True
    assert out == ["--flag"]


def test_strip_empty_ros_args_sections_directly():
    assert _strip_empty_ros_args_sections(["--ros-args"]) == []
    assert _strip_empty_ros_args_sections(
        ["--ros-args", "--ros-args", "-p", "a:=1"]) == ["--ros-args", "-p", "a:=1"]
    assert _strip_empty_ros_args_sections(
        ["-p", "a:=1", "--ros-args"]) == ["-p", "a:=1"]
    assert _strip_empty_ros_args_sections(
        ["--ros-args", "-p", "a:=1"]) == ["--ros-args", "-p", "a:=1"]


# --------------------------------------------------------------------------- #
# _format_ros_param_value — regression coverage for the reported bug:
# `-p risk_map_reward_enabled:=true` (unquoted) made ROS 2's own CLI parser
# infer BOOL and raise InvalidParameterTypeException, because the receiving
# node declares that parameter (and action_risk_head_enabled, load_model)
# with a STRING default (`declare_parameter(name, "")`) — the empty-string-
# means-"no override" contract documented in CLAUDE.md. `seed`, by contrast,
# is declared with an int default (`declare_parameter("seed", -1)`) and must
# stay unquoted so ROS still infers INT. These tests exercise the actual
# contract (via yaml.safe_load, mirroring ROS 2's own override-type
# inference) rather than just the string shape.
# --------------------------------------------------------------------------- #
def test_format_ros_param_value_quotes_boolean_flags_as_yaml_string():
    for raw in ("true", "True", "TRUE", "false", "False"):
        formatted = _format_ros_param_value(raw)
        assert formatted == f'"{raw}"'
        # This is exactly what ROS 2's CLI parameter parser does with the
        # token after `key:=`; must come back as a str, not a bool.
        parsed = yaml.safe_load(formatted)
        assert isinstance(parsed, str) and parsed == raw


def test_format_ros_param_value_leaves_int_unquoted():
    formatted = _format_ros_param_value(0)
    assert formatted == "0"
    parsed = yaml.safe_load(formatted)
    assert isinstance(parsed, int) and parsed == 0


def test_format_ros_param_value_leaves_plain_strings_unquoted():
    path = "/root/DRL_Robot_Path_Planning/ros2_ws/src/drl_experiments/profiles/phase2/both"
    assert _format_ros_param_value(path) == path
    assert yaml.safe_load(path) == path
