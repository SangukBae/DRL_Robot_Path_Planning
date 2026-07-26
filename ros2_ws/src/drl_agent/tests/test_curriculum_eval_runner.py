"""ROS-free regression tests for curriculum_eval_runner.py."""

import csv
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


# --------------------------------------------------------------------------- #
#  Regression: per-map h_coll_rate / psc / spl / lidar_clearance_rate must be
#  WRITTEN, not blanked (the bug: `_b(None, hc)` / `_b(None, psc)` passed the
#  computed value as the SECOND positional arg to a helper that only reads it
#  when the FIRST arg is a dict -- so a real value was always discarded and
#  replaced with "" ("" for 0.0, not just for None), regardless of whether a
#  label was actually available).
# --------------------------------------------------------------------------- #

_PER_MAP_HEADER = [
    "epoch", "global_t", "curriculum_stage", "map_type", "eval_eps",
    "success_rate", "collision_rate", "timeout_rate",
    "mean_reward", "mean_goal_dist",
    "h_coll_rate", "psc",
    "aux_risk_rmse", "aux_min_dist_mae_m",
    "aux_peak_sector_acc", "aux_near_event_f1",
    "lidar_clearance_rate", "spl",
]


def _metrics(spl, clearance):
    m = EpisodeMetrics.empty()
    m["spl"] = spl
    m["lidar_clearance_rate"] = clearance
    return m


class _ScriptedEpisodeMetrics:
    """Returns one canned metrics dict per `compute()` call, in call order --
    lets a test control per-episode spl / lidar_clearance_rate directly rather
    than driving the full state/action arithmetic in EpisodeMetrics."""

    def __init__(self, per_episode_metrics):
        self._metrics = list(per_episode_metrics)
        self._i = 0

    def reset(self, state):
        pass

    def update(self, state, action):
        pass

    def compute(self, success):
        m = self._metrics[self._i]
        self._i += 1
        return m


def test_per_map_csv_writes_h_coll_psc_spl_clearance_not_blank(tmp_path, monkeypatch):
    CurriculumEvalMixin, _ = _load_curriculum_eval_runner(monkeypatch)

    class _MultiMapEvalNode(CurriculumEvalMixin):
        """Drives `eval_eps` 1-step episodes, one per (map_type, human_dist,
        success, metrics) tuple supplied by the test -- exercises the per-map
        aggregation/write path in evaluate_and_print with fully-scripted,
        deterministic per-episode outcomes."""

        def __init__(self, tmp_path, *, map_types, dists, successes, em_metrics):
            assert len(map_types) == len(dists) == len(successes) == len(em_metrics)
            self.environment_dim = 0
            self.eval_eps = len(map_types)
            self.max_episode_steps = 1
            self._aux_eval_on = False
            self._curriculum_stage = 4
            self._last_global_t = 800000
            self._psc_personal_space_m = 0.5
            self._h_coll_radius_m = 0.5
            self._aux_eval_summary = _AuxSummaryStub()
            self._paper = _PaperStub()
            self._em = _ScriptedEpisodeMetrics(em_metrics)
            self.rl_agent = _AgentStub()
            self.results_dir = str(tmp_path)
            self.file_name = "multi_map_regression"
            self._curriculum_eval_per_map_csv = str(tmp_path / "per_map.csv")
            self.last_aux_label = None
            self._logger = _Logger()
            self._map_types = list(map_types)
            self._dists = list(dists)
            self._successes = list(successes)
            self._ep_idx = -1

        def get_logger(self):
            return self._logger

        def _set_eval_mode(self, on: bool) -> bool:
            return True

        def reset(self):
            self._ep_idx += 1
            self.last_aux_label = None
            return np.asarray([0.0], dtype=np.float32)

        def step(self, action):
            self.last_aux_label = None
            return (np.asarray([0.0], dtype=np.float32), 1.0, True,
                    self._successes[self._ep_idx])

        def _fetch_current_map_type(self) -> str:
            return self._map_types[self._ep_idx]

        def _human_min_dist_m_from_label(self, label):
            return self._dists[self._ep_idx]

    # corridor: 2 SAFE episodes (dist=2.0m, outside both 0.5m thresholds),
    #   1 success + 1 collision -> h_coll must stay 0 (collision but NOT close
    #   to a human), psc=1.0 (no personal-space intrusions).
    # intersection: 2 UNSAFE episodes (dist=0.1m), both collisions
    #   -> h_coll_rate=1.0, psc=0.0 (personal space intruded every step).
    # lobby: 1 episode with NO label (dist=None) -> h_coll/psc must stay BLANK
    #   (label unavailable), while spl/lidar_clearance_rate are still written
    #   (paper metrics do not depend on human labels at all).
    node = _MultiMapEvalNode(
        tmp_path,
        map_types=["corridor", "corridor", "intersection", "intersection", "lobby"],
        dists=[2.0, 2.0, 0.1, 0.1, None],
        successes=[True, False, False, False, True],
        em_metrics=[
            _metrics(spl=1.0, clearance=0.9),
            _metrics(spl=0.5, clearance=0.7),
            _metrics(spl=0.8, clearance=0.6),
            _metrics(spl=0.8, clearance=0.6),
            _metrics(spl=0.6, clearance=0.4),
        ],
    )
    with open(node._curriculum_eval_per_map_csv, "w", newline="") as f:
        csv.writer(f).writerow(_PER_MAP_HEADER)

    result = node.evaluate_and_print([], epoch=7, start_time=0.0)

    with open(node._curriculum_eval_per_map_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    by_map = {r["map_type"]: r for r in rows}
    assert set(by_map) == {"corridor", "intersection", "lobby"}

    cor = by_map["corridor"]
    assert float(cor["h_coll_rate"]) == 0.0   # written, not blank -- and genuinely 0
    assert float(cor["psc"]) == 1.0
    assert float(cor["spl"]) == 0.75           # mean(1.0, 0.5)
    assert float(cor["lidar_clearance_rate"]) == 0.8  # mean(0.9, 0.7)

    isec = by_map["intersection"]
    assert float(isec["h_coll_rate"]) == 1.0
    assert float(isec["psc"]) == 0.0
    assert float(isec["spl"]) == 0.8
    assert float(isec["lidar_clearance_rate"]) == 0.6

    lobby = by_map["lobby"]
    assert lobby["h_coll_rate"] == ""          # no label this episode -> blank
    assert lobby["psc"] == ""
    assert float(lobby["spl"]) == 0.6          # paper metrics unaffected by labels
    assert float(lobby["lidar_clearance_rate"]) == 0.4

    # per_map dict returned in `metrics` must agree with the CSV (same source).
    pm = result["per_map"]
    assert pm["corridor"]["h_coll_rate"] == 0.0
    assert pm["corridor"]["psc"] == 1.0
    assert pm["intersection"]["h_coll_rate"] == 1.0
    assert pm["lobby"]["h_coll_rate"] is None
