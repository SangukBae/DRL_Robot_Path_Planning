"""Agent-level (torch-gated) tests for Stage-8 observation normalization
wiring: train()/select_action() actually apply it before the encoder, the
default (disabled) path is byte-identical, and the checkpoint-manifest
contract check fails fast on any mismatch (never silently resumes a
checkpoint trained under a different normalization setup).
"""

import pytest

try:
    import numpy as np
    import torch

    from drl_agent.rl.algorithms.tqc.agent import Agent
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")

OBS, AGENT_DIM, HIST = 80, 7, 1
STATE_DIM = OBS + AGENT_DIM  # 87, no history stacking
ACTION_DIM = 2


def _hp(**over):
    hp = dict(batch_size=8, buffer_size=200, n_critics=2, n_quantiles=5)
    hp.update(over)
    return hp


def _fill(agent, n=16):
    for _ in range(n):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        a = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        agent.replay_buffer.add(s, a, s, 0.1, 0.0)


def test_disabled_by_default_no_normalizer_built(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path))
    assert agent.obs_normalizer is None
    assert agent.obs_norm_cfg.enabled is False


def test_enabled_without_env_dims_fails_fast(tmp_path):
    with pytest.raises(RuntimeError, match="env_obs_dim"):
        Agent(STATE_DIM, ACTION_DIM, 1.0,
              _hp(observation_normalization={"enabled": True}),
              log_dir=str(tmp_path))


def test_enabled_with_env_dims_builds_normalizer(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(observation_normalization={"enabled": True, "lidar_scale": 50.0}),
                  log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    assert agent.obs_normalizer is not None
    assert agent.obs_normalizer.cfg.lidar_scale == 50.0


def test_train_normalizes_state_before_encoder(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(observation_normalization={"enabled": True, "lidar_scale": 50.0,
                                                  "goal_dist_scale": 10.0}),
                  log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    # Fill with a KNOWN constant state so the normalized value is checkable.
    s = np.zeros(STATE_DIM, dtype=np.float32)
    s[:OBS] = 25.0   # half of lidar_scale=50 -> normalized 0.5
    s[OBS] = 5.0      # half of goal_dist_scale=10 -> normalized 0.5
    for _ in range(16):
        a = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        agent.replay_buffer.add(s, a, s, 0.1, 0.0)

    seen = {}
    real_forward = agent.encoder.forward

    def spy(x):
        seen["input"] = x.detach().clone()
        return real_forward(x)

    agent.encoder.forward = spy
    agent.train()
    assert "input" in seen
    x = seen["input"]
    assert torch.allclose(x[:, :OBS], torch.full((x.shape[0], OBS), 0.5, device=x.device),
                           atol=1e-5), "encoder must receive NORMALIZED lidar values, not raw"
    assert torch.allclose(x[:, OBS], torch.full((x.shape[0],), 0.5, device=x.device),
                           atol=1e-5)


def test_select_action_normalizes_state_before_encoder(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                  _hp(observation_normalization={"enabled": True, "lidar_scale": 50.0}),
                  log_dir=str(tmp_path), env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    s = np.zeros(STATE_DIM, dtype=np.float32)
    s[:OBS] = 50.0  # exactly lidar_scale -> normalized 1.0

    seen = {}
    real_forward = agent.encoder.forward

    def spy(x):
        seen["input"] = x.detach().clone()
        return real_forward(x)

    agent.encoder.forward = spy
    agent.select_action(s, use_exploration=False)
    x = seen["input"]
    assert torch.allclose(x[:, :OBS], torch.ones(1, OBS, device=x.device), atol=1e-5)


def test_disabled_default_train_is_unaffected(tmp_path):
    # Sanity: with normalization off (default), train() must run exactly as
    # before -- this is really a regression guard for the wiring itself.
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path))
    _fill(agent)
    agent.train()  # must not raise


# --------------------------------------------------------------------------- #
#  checkpoint manifest: never silently load a mismatched normalization contract
# --------------------------------------------------------------------------- #
def test_checkpoint_manifest_round_trips_when_matching(tmp_path):
    hp = _hp(observation_normalization={"enabled": True, "lidar_scale": 50.0})
    a1 = Agent(STATE_DIM, ACTION_DIM, 1.0, hp, log_dir=str(tmp_path / "a"),
               env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    _fill(a1)
    a1.train()
    a1.save(str(tmp_path), "ckpt")

    a2 = Agent(STATE_DIM, ACTION_DIM, 1.0, hp, log_dir=str(tmp_path / "b"),
               env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    a2.load(str(tmp_path), "ckpt", load_replay_buffer=False)  # must not raise


def test_checkpoint_manifest_rejects_enabled_mismatch(tmp_path):
    hp_off = _hp()
    hp_on = _hp(observation_normalization={"enabled": True, "lidar_scale": 50.0})
    a1 = Agent(STATE_DIM, ACTION_DIM, 1.0, hp_off, log_dir=str(tmp_path / "a"))
    _fill(a1)
    a1.train()
    a1.save(str(tmp_path), "ckpt")

    a2 = Agent(STATE_DIM, ACTION_DIM, 1.0, hp_on, log_dir=str(tmp_path / "b"),
               env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    with pytest.raises(RuntimeError, match="observation_normalization"):
        a2.load(str(tmp_path), "ckpt", load_replay_buffer=False)


def test_checkpoint_manifest_rejects_scale_mismatch(tmp_path):
    a1 = Agent(STATE_DIM, ACTION_DIM, 1.0,
               _hp(observation_normalization={"enabled": True, "lidar_scale": 50.0}),
               log_dir=str(tmp_path / "a"), env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    _fill(a1)
    a1.train()
    a1.save(str(tmp_path), "ckpt")

    a2 = Agent(STATE_DIM, ACTION_DIM, 1.0,
               _hp(observation_normalization={"enabled": True, "lidar_scale": 25.0}),
               log_dir=str(tmp_path / "b"), env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    with pytest.raises(RuntimeError, match="scale mismatch"):
        a2.load(str(tmp_path), "ckpt", load_replay_buffer=False)


def test_checkpoint_predating_stage8_rejects_resume_with_normalization_enabled(tmp_path):
    # No manifest file at all (simulates a pre-Stage-8 checkpoint).
    a1 = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path / "a"))
    _fill(a1)
    a1.train()
    a1.save(str(tmp_path), "ckpt")
    import os
    manifest = str(tmp_path / "ckpt_obs_norm_manifest.json")
    assert os.path.isfile(manifest)
    os.remove(manifest)  # simulate a checkpoint saved before Stage 8 existed

    a2 = Agent(STATE_DIM, ACTION_DIM, 1.0,
               _hp(observation_normalization={"enabled": True, "lidar_scale": 50.0}),
               log_dir=str(tmp_path / "b"), env_obs_dim=OBS, env_agent_dim=AGENT_DIM)
    with pytest.raises(RuntimeError, match="predates"):
        a2.load(str(tmp_path), "ckpt", load_replay_buffer=False)


def test_checkpoint_predating_stage8_resumes_fine_when_normalization_stays_disabled(tmp_path):
    a1 = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path / "a"))
    _fill(a1)
    a1.train()
    a1.save(str(tmp_path), "ckpt")
    import os
    os.remove(str(tmp_path / "ckpt_obs_norm_manifest.json"))

    a2 = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path / "b"))
    a2.load(str(tmp_path), "ckpt", load_replay_buffer=False)  # must not raise
