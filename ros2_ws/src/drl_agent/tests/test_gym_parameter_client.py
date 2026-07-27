"""ROS-free unit tests for gym_parameter_client.GymParameterClient.

``gym_parameter_client`` imports ``rcl_interfaces`` and
``drl_agent.env.environment_interface`` (both ROS). To keep this test hermetic
(runs on CI without a built workspace AND on the ROS box), we stub those
modules in ``sys.modules`` BEFORE importing the client. The stubs mirror only
the surface the client touches: request types, ``Parameter`` /
``ParameterValue`` / ``ParameterType`` and ``EnvServiceError``.

A ``FakeNode`` supplies canned ``_call_service`` responses (or raises), so every
branch — accept/reject, present/absent params, confirm path, fallback, type
parsing and EnvServiceError propagation/absorption — is exercised directly.
"""

import sys
import types

import pytest


# --------------------------------------------------------------------------- #
# Stub the ROS modules the client imports, before importing the client.
# --------------------------------------------------------------------------- #
class _EnvServiceError(Exception):
    pass


def _install_stubs():
    _CANON = "drl_agent.env.environment_interface"
    if _CANON in sys.modules:
        # Real module already imported (ROS box): reuse its EnvServiceError.
        globals()["_EnvServiceError"] = sys.modules[_CANON].EnvServiceError
    else:
        try:
            import importlib
            real = importlib.import_module(_CANON)
            globals()["_EnvServiceError"] = real.EnvServiceError
        except Exception:
            m = types.ModuleType(_CANON)
            m.EnvServiceError = _EnvServiceError
            sys.modules[_CANON] = m

    if "rcl_interfaces.srv" not in sys.modules:
        srv = types.ModuleType("rcl_interfaces.srv")

        class _Req:
            pass

        class GetParameters:
            Request = staticmethod(lambda: _Req())

        class SetParameters:
            Request = staticmethod(lambda: _Req())

        srv.GetParameters = GetParameters
        srv.SetParameters = SetParameters
        sys.modules["rcl_interfaces.srv"] = srv

    if "rcl_interfaces.msg" not in sys.modules:
        msg = types.ModuleType("rcl_interfaces.msg")

        class ParameterType:
            PARAMETER_NOT_SET = 0
            PARAMETER_BOOL = 1
            PARAMETER_INTEGER = 2
            PARAMETER_DOUBLE = 3
            PARAMETER_STRING = 4
            PARAMETER_DOUBLE_ARRAY = 8

        class ParameterValue:
            def __init__(self, type=0, **kw):
                self.type = type
                for k, v in kw.items():
                    setattr(self, k, v)

        class Parameter:
            def __init__(self, name="", value=None):
                self.name = name
                self.value = value

        msg.ParameterType = ParameterType
        msg.ParameterValue = ParameterValue
        msg.Parameter = Parameter
        sys.modules["rcl_interfaces.msg"] = msg

    # ensure parent package exists
    if "rcl_interfaces" not in sys.modules:
        pkg = types.ModuleType("rcl_interfaces")
        sys.modules["rcl_interfaces"] = pkg


_install_stubs()

from rcl_interfaces.msg import ParameterType  # noqa: E402
from drl_agent.training.gym_parameter_client import GymParameterClient  # noqa: E402

EnvServiceError = sys.modules["drl_agent.env.environment_interface"].EnvServiceError


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _Logger:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


class _PV:
    """Minimal ParameterValue-like response field."""
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Result:
    def __init__(self, successful):
        self.successful = successful


class FakeNode:
    def __init__(self, responses):
        # responses: list of either a response object or an Exception to raise
        self._responses = list(responses)
        self.calls = []
        self._logger = _Logger()

    def create_client(self, srv, name):
        return ("client", name)

    def get_logger(self):
        return self._logger

    def _call_service(self, client, request, name, **kw):
        self.calls.append((client, request, name))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _resp(**attrs):
    r = types.SimpleNamespace()
    for k, v in attrs.items():
        setattr(r, k, v)
    return r


# --------------------------------------------------------------------------- #
# set_curriculum_stage
# --------------------------------------------------------------------------- #
def test_set_curriculum_stage_accepted():
    node = FakeNode([_resp(results=[_Result(True)])])
    c = GymParameterClient(node)
    assert c.set_curriculum_stage(3) is True


def test_set_curriculum_stage_rejected():
    node = FakeNode([_resp(results=[_Result(False)])])
    c = GymParameterClient(node)
    assert c.set_curriculum_stage(3) is False


def test_set_curriculum_stage_empty_results_is_false():
    node = FakeNode([_resp(results=[])])
    c = GymParameterClient(node)
    assert c.set_curriculum_stage(1) is False


# --------------------------------------------------------------------------- #
# get_eval_mode
# --------------------------------------------------------------------------- #
def test_get_eval_mode_true_false():
    node = FakeNode([_resp(values=[_PV(ParameterType.PARAMETER_BOOL, bool_value=True)])])
    assert GymParameterClient(node).get_eval_mode() is True
    node = FakeNode([_resp(values=[_PV(ParameterType.PARAMETER_BOOL, bool_value=False)])])
    assert GymParameterClient(node).get_eval_mode() is False


def test_get_eval_mode_absent_returns_none():
    node = FakeNode([_resp(values=[])])
    assert GymParameterClient(node).get_eval_mode() is None
    # wrong type also -> None
    node = FakeNode([_resp(values=[_PV(ParameterType.PARAMETER_STRING, string_value="x")])])
    assert GymParameterClient(node).get_eval_mode() is None


def test_get_eval_mode_propagates_env_service_error():
    node = FakeNode([EnvServiceError("down")])
    with pytest.raises(EnvServiceError):
        GymParameterClient(node).get_eval_mode()


# --------------------------------------------------------------------------- #
# set_eval_mode (confirm path: set, then get to verify)
# --------------------------------------------------------------------------- #
def test_set_eval_mode_confirmed():
    node = FakeNode([
        _resp(results=[_Result(True)]),                                   # set
        _resp(values=[_PV(ParameterType.PARAMETER_BOOL, bool_value=True)]),  # confirm get
    ])
    assert GymParameterClient(node).set_eval_mode(True) is True


def test_set_eval_mode_set_ok_but_confirm_mismatch():
    node = FakeNode([
        _resp(results=[_Result(True)]),
        _resp(values=[_PV(ParameterType.PARAMETER_BOOL, bool_value=False)]),
    ])
    assert GymParameterClient(node).set_eval_mode(True) is False


def test_set_eval_mode_rejected_but_confirm_true_is_true():
    # Request looked rejected, but env actually applied it → ground truth wins.
    node = FakeNode([
        _resp(results=[_Result(False)]),
        _resp(values=[_PV(ParameterType.PARAMETER_BOOL, bool_value=True)]),
    ])
    assert GymParameterClient(node).set_eval_mode(True) is True


# --------------------------------------------------------------------------- #
# get_num_stages
# --------------------------------------------------------------------------- #
def test_get_num_stages_valid():
    node = FakeNode([_resp(values=[_PV(ParameterType.PARAMETER_INTEGER, integer_value=7)])])
    assert GymParameterClient(node).get_num_stages() == 7


def test_get_num_stages_absent_fallback_5():
    node = FakeNode([_resp(values=[])])
    assert GymParameterClient(node).get_num_stages() == 5


def test_get_num_stages_invalid_fallback_5():
    node = FakeNode([_resp(values=[_PV(ParameterType.PARAMETER_INTEGER, integer_value=0)])])
    assert GymParameterClient(node).get_num_stages() == 5


# --------------------------------------------------------------------------- #
# get_env_aux_params (type parsing)
# --------------------------------------------------------------------------- #
def test_get_env_aux_params_type_parsing():
    names = ["loaded_config_path", "loaded_config_sha1", "aux_enabled",
             "aux_num_sectors", "aux_horizons_sec", "aux_risk_distance_scale"]
    values = [
        _PV(ParameterType.PARAMETER_STRING, string_value="/cfg.yaml"),
        _PV(ParameterType.PARAMETER_STRING, string_value="abc123"),
        _PV(ParameterType.PARAMETER_BOOL, bool_value=True),
        _PV(ParameterType.PARAMETER_INTEGER, integer_value=12),
        _PV(ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=[1.0, 2.5]),
        _PV(ParameterType.PARAMETER_DOUBLE, double_value=0.75),
    ]
    node = FakeNode([_resp(values=values)])
    out = GymParameterClient(node).get_env_aux_params()
    assert out["loaded_config_path"] == "/cfg.yaml"
    assert out["aux_enabled"] is True
    assert out["aux_num_sectors"] == 12
    assert out["aux_horizons_sec"] == [1.0, 2.5]
    assert out["aux_risk_distance_scale"] == pytest.approx(0.75)


def test_get_env_aux_params_absorbs_errors():
    node = FakeNode([EnvServiceError("boom")])
    assert GymParameterClient(node).get_env_aux_params() == {}


# --------------------------------------------------------------------------- #
# get_current_map_type
# --------------------------------------------------------------------------- #
def test_get_current_map_type_value():
    node = FakeNode([_resp(values=[_PV(ParameterType.PARAMETER_STRING, string_value="corridor")])])
    assert GymParameterClient(node).get_current_map_type() == "corridor"


def test_get_current_map_type_absent_empty():
    node = FakeNode([_resp(values=[])])
    assert GymParameterClient(node).get_current_map_type() == ""


def test_get_current_map_type_absorbs_env_service_error():
    node = FakeNode([EnvServiceError("down")])
    assert GymParameterClient(node).get_current_map_type() == ""
