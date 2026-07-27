"""Unit tests for the wrapper-arg parsing in drl_agent.nodes._node_common.

The wrappers must (a) recognise both ROS-style ``-p key:=value`` and plain
``--key`` forms, (b) strip ONLY the wrapper-only tokens, (c) leave a
ROS-style ``-p seed:=N`` in the passthrough (the legacy trainers consume it
natively) while converting a plain ``--seed N``, and (d) never leave a
dangling EMPTY ``--ros-args`` section behind after stripping (exec_legacy
would otherwise tack its own ``--ros-args`` section onto an already-empty
one).
"""

from drl_agent.nodes._node_common import (_strip_empty_ros_args_sections,
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
