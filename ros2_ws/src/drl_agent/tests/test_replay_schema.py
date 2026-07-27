"""Replay schema contract tests.

The torch-free part checks the declarative schema module itself; the
torch-gated part round-trips a real LAP buffer save() against the schema so
the on-disk npz contract can never silently drift from the documentation.
"""

import numpy as np
import pytest

from drl_agent.rl.replay import schema

try:
    import torch
    from drl_agent.rl.replay.buffer import LAP
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


def test_expected_keys_baseline():
    keys = schema.expected_npz_keys()
    assert keys == {"state", "action", "next_state", "reward", "not_done",
                    "meta", "max_priority"}


def test_expected_keys_all_optional():
    keys = schema.expected_npz_keys(aux_dim=3, track_traj=True, action_risk_dim=2)
    assert {"aux_target", "traj_end", "action_risk_target"} <= keys


def test_validate_npz_keys_reports_missing_and_unexpected():
    missing, unexpected = schema.validate_npz_keys(
        ["state", "action", "bogus"], track_traj=True)
    assert "traj_end" in missing and "not_done" in missing
    assert unexpected == {"bogus"}


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
@pytest.mark.parametrize("aux_dim,track_traj,action_risk_dim", [
    (0, False, 0),
    (3, True, 2),
])
def test_lap_save_matches_schema(tmp_path, aux_dim, track_traj, action_risk_dim):
    buf = LAP(state_dim=5, action_dim=2, device=torch.device("cpu"), max_size=32,
              batch_size=4, prioritized=True, aux_dim=aux_dim,
              track_traj=track_traj, action_risk_dim=action_risk_dim)
    rng = np.random.default_rng(0)
    for _ in range(6):
        buf.add(rng.normal(size=5), rng.normal(size=2), rng.normal(size=5),
                0.5, 0.0,
                aux_target=(rng.normal(size=aux_dim) if aux_dim else None),
                action_risk_target=(rng.normal(size=action_risk_dim)
                                    if action_risk_dim else None))
    buf.mark_last_traj_end()
    prefix = str(tmp_path / "rb")
    buf.save(prefix)

    d = np.load(prefix + ".npz")
    missing, unexpected = schema.validate_npz_keys(
        d.files, **schema.describe_buffer(buf))
    assert not missing and not unexpected

    # Round-trip into an identically-configured buffer preserves contents.
    buf2 = LAP(state_dim=5, action_dim=2, device=torch.device("cpu"), max_size=32,
               batch_size=4, prioritized=True, aux_dim=aux_dim,
               track_traj=track_traj, action_risk_dim=action_risk_dim)
    assert buf2.load(prefix) is True
    assert buf2.size == buf.size and buf2.ptr == buf.ptr
    np.testing.assert_allclose(buf2.state[:buf2.size], buf.state[:buf.size])
    np.testing.assert_allclose(buf2.action[:buf2.size], buf.action[:buf.size])
    if track_traj:
        np.testing.assert_allclose(buf2.traj_end[:buf2.size],
                                   buf.traj_end[:buf.size])
