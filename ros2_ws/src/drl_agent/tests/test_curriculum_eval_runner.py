"""ROS-free regression tests for curriculum_eval_runner.py."""

import importlib.util
import os
import sys
import types

import numpy as np

from episode_metrics import EpisodeMetrics


def _load_curriculum_eval_runner(monkeypatch):
    """Import curriculum_eval_runner with test-local stubs only."""
    torch_stub = types.ModuleType("torch")
    env_mod = types.ModuleType("environment_interface")

    class _EnvServiceError(RuntimeError):
        pass

    env_mod.EnvServiceError = _EnvServiceError
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "environment_interface", env_mod)

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "policy",
        "curriculum_eval_runner.py",
    )
    spec = importlib.util.spec_from_file_location(
        "curriculum_eval_runner_testisolated", os.path.normpath(path)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CurriculumEvalMixin, _EnvServiceError


class _Logger:
    def __init__(self):
        self.infos = []
        self.warns = []

    def info(self, msg):
        self.infos.append(str(msg))

    def warn(self, msg):
        self.warns.append(str(msg))


class _EpisodeMetricsStub:
    def reset(self, state):
        pass

    def update(self, state, action):
        pass

    def compute(self, success):
        return EpisodeMetrics.empty()


class _PaperStub:
    def write_eval(self, **kwargs):
        pass


class _AuxSummaryStub:
    def append(self, **kwargs):
        pass


class _AgentStub:
    def select_action(self, state, use_checkpoint=False, use_exploration=False):
        return np.asarray([0.0], dtype=np.float32)


def test_evaluate_raises_when_eval_mode_restore_fails(tmp_path, monkeypatch):
    CurriculumEvalMixin, EnvServiceError = _load_curriculum_eval_runner(monkeypatch)

    class _EvalNode(CurriculumEvalMixin):
        def __init__(self, tmp_path, *, clear_ok: bool):
            self.environment_dim = 0
            self.eval_eps = 1
            self.max_episode_steps = 1
            self._aux_eval_on = False
            self._curriculum_stage = 2
            self._last_global_t = 12000
            self._psc_personal_space_m = 0.5
            self._h_coll_radius_m = 0.5
            self._aux_eval_summary = _AuxSummaryStub()
            self._paper = _PaperStub()
            self._em = _EpisodeMetricsStub()
            self.rl_agent = _AgentStub()
            self.results_dir = str(tmp_path)
            self.file_name = "eval_runner_regression"
            self._curriculum_eval_per_map_csv = str(tmp_path / "per_map.csv")
            self.last_aux_label = None
            self._clear_ok = clear_ok
            self._logger = _Logger()

        def get_logger(self):
            return self._logger

        def _set_eval_mode(self, on: bool) -> bool:
            return True if on else self._clear_ok

        def reset(self):
            self.last_aux_label = None
            return np.asarray([0.0], dtype=np.float32)

        def step(self, action):
            self.last_aux_label = None
            return np.asarray([0.0], dtype=np.float32), 1.0, True, True

        def _fetch_current_map_type(self) -> str:
            return "corridor"

        def _human_min_dist_m_from_label(self, label):
            return None

    node = _EvalNode(tmp_path, clear_ok=False)
    with open(node._curriculum_eval_per_map_csv, "w", encoding="utf-8") as f:
        f.write("")

    try:
        node.evaluate_and_print([], epoch=1, start_time=0.0)
        raised = False
    except EnvServiceError as e:
        raised = True
        assert "Failed to clear curriculum_eval_mode" in str(e)

    assert raised, "evaluate_and_print must fail-fast if eval mode cannot be cleared"
