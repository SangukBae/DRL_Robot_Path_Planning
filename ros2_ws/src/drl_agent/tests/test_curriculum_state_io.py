"""ROS-free unit tests for curriculum_state_io (resume JSON round-trip).

torch is imported lazily inside the RNG try-blocks of the module, so the JSON
progress round-trip — the resume-critical part — is testable without torch. The
RNG snapshot files are best-effort and not asserted here.
"""

import json
import os

import curriculum_state_io as cs


class _Logger:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass


class FakeTrainer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self._curriculum_stage = 0
        self._stage_start_step = 0
        self._stage_start_ep = 0
        self._consecutive_pass_count = 0
        self._total_episodes = 0
        self._resume_epoch = 1
        self._partial_ep_timesteps = 0
        self._partial_ep_reward = 0.0
        self._resume_global_t = 0
        self._last_global_t = 0
        self._resume_loaded = False
        self._logger = _Logger()

    def get_logger(self):
        return self._logger


def test_save_writes_expected_json(tmp_path):
    t = FakeTrainer(str(tmp_path))
    t._curriculum_stage = 2
    t._stage_start_step = 30000
    t._stage_start_ep = 40
    t._consecutive_pass_count = 1
    t._total_episodes = 123
    t._resume_epoch = 5
    t._partial_ep_timesteps = 7
    t._partial_ep_reward = 12.5
    cs.save_curriculum_state(t, global_t=55000)

    path = tmp_path / "curriculum_state.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["stage"] == 2
    assert data["stage_start_step"] == 30000
    assert data["stage_start_episode"] == 40
    assert data["consecutive_pass_count"] == 1
    assert data["global_t"] == 55000
    assert data["total_episodes"] == 123
    assert data["epoch"] == 5
    assert data["ep_timesteps"] == 7
    assert data["ep_total_reward"] == 12.5


def test_load_missing_returns_false(tmp_path):
    t = FakeTrainer(str(tmp_path))
    assert cs.load_curriculum_state(t) is False
    assert t._resume_loaded is False


def test_save_load_roundtrip(tmp_path):
    src = FakeTrainer(str(tmp_path))
    src._curriculum_stage = 3
    src._stage_start_step = 12345
    src._stage_start_ep = 9
    src._consecutive_pass_count = 2
    src._total_episodes = 200
    src._resume_epoch = 8
    src._partial_ep_timesteps = 11
    src._partial_ep_reward = -4.0
    cs.save_curriculum_state(src, global_t=99999)

    dst = FakeTrainer(str(tmp_path))
    assert cs.load_curriculum_state(dst) is True
    assert dst._curriculum_stage == 3
    assert dst._stage_start_step == 12345
    assert dst._stage_start_ep == 9
    assert dst._consecutive_pass_count == 2
    assert dst._resume_global_t == 99999
    assert dst._last_global_t == 99999
    assert dst._total_episodes == 200
    assert dst._resume_epoch == 8
    assert dst._partial_ep_timesteps == 11
    assert dst._partial_ep_reward == -4.0
    assert dst._resume_loaded is True


def test_load_corrupt_json_returns_false(tmp_path):
    (tmp_path / "curriculum_state.json").write_text("{not valid json")
    t = FakeTrainer(str(tmp_path))
    assert cs.load_curriculum_state(t) is False
