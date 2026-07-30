"""RISK_BALANCE: per-transition metadata + risk-balanced sampling for the LAP
replay buffer (drl_agent.rl.replay.buffer).

Covers:
  * balanced sampler approximates the configured (uniform, human_risk,
    collision) pool ratios when all three pools are populated;
  * empty-pool fallback: a missing human-risk or collision pool never raises
    / never returns a short batch -- its share is silently redistributed to
    the uniform pool;
  * disabled / empty-buffer -> sample_risk_balanced() returns None (callers
    fall back to the primary batch, byte-identical to the feature being off);
  * get_last_future_actions / get_last_state_history respect an explicit
    `indices=` batch (the risk-balanced batch) with the SAME episode-boundary
    safety as the primary sample()-driven path;
  * risk_meta save/load round-trip, and the graceful (non-fail-fast) fallback
    when loading an OLDER checkpoint that has no risk_meta at all;
  * OFF-parity: with risk_balanced_sampling disabled, sample()/get_last_aux/
    get_last_action_risk behave exactly as before this feature existed.

torch is imported defensively (buffer.py imports torch); skipped if
unavailable.
"""

import pytest

try:
    import numpy as np
    import torch
    import drl_agent.rl.replay.buffer as buffer
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")

SDIM, ADIM = 4, 2


def _mk(max_size=2000, **kw):
    kw.setdefault("normalize_actions", False)
    kw.setdefault("prioritized", False)
    kw.setdefault("batch_size", 200)
    return buffer.LAP(SDIM, ADIM, torch.device("cpu"), max_size=max_size, **kw)


def _add(buf, val=0.0, risk_meta=None, action_risk_target=None):
    s = np.full(SDIM, float(val), dtype=np.float32)
    buf.add(s, np.zeros(ADIM, dtype=np.float32), s, 0.0, 0.0,
            risk_meta=risk_meta, action_risk_target=action_risk_target)


# --------------------------------------------------------------------------- #
#  Storage: risk_meta array, add()/save()/load()
# --------------------------------------------------------------------------- #
def test_store_risk_meta_off_by_default():
    buf = _mk()
    assert buf.risk_meta is None
    assert buf.risk_balanced_enabled is False


def test_store_risk_meta_records_the_four_columns():
    buf = _mk(store_risk_meta=True)
    _add(buf, risk_meta=(3.0, 1.0, 0.0, 1.0))
    assert buf.risk_meta[0].tolist() == [3.0, 1.0, 0.0, 1.0]


def test_risk_meta_zero_pads_when_omitted():
    buf = _mk(store_risk_meta=True)
    _add(buf)   # risk_meta=None
    assert buf.risk_meta[0].tolist() == [0.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
#  sample_risk_balanced: ratio approximation + empty-pool fallback
# --------------------------------------------------------------------------- #
def _mk_populated(n_uniform=600, n_human=200, n_collision=200, **kw):
    kw.setdefault("store_risk_meta", True)
    kw.setdefault("risk_balanced_enabled", True)
    kw.setdefault("risk_balanced_ratios", (0.5, 0.25, 0.25))
    buf = _mk(**kw)
    for i in range(n_uniform):
        _add(buf, val=i, risk_meta=(0.0, 0.0, 0.0, 0.0))
    for i in range(n_human):
        _add(buf, val=1000 + i, risk_meta=(0.0, 1.0, 1.0, 0.0))
    for i in range(n_collision):
        _add(buf, val=2000 + i, risk_meta=(0.0, 0.0, 0.0, 1.0))
    return buf


def test_balanced_sampler_approximates_configured_ratios():
    buf = _mk_populated()
    ind = buf.sample_risk_balanced(batch_size=2000)
    assert ind is not None
    frac = buf.describe_risk_meta_fractions(ind)
    # Configured ratios are (uniform=0.5, human_risk=0.25, collision=0.25),
    # but the FINAL observed event fraction is NOT simply 0.25: the "uniform"
    # half of the batch is drawn via np.random.randint over the WHOLE buffer
    # ([0, size)), not from a pool that excludes already-tagged transitions --
    # so it also picks up human/collision-tagged rows at whatever rate those
    # rows already occur in the buffer, on top of the dedicated pool draws.
    # With this fixture's population (200 human / 200 collision / 1000 total
    # = 20% tagged each), the expected total per tag is:
    #   0.25 (dedicated pool share of the batch)
    #   + 0.50 (uniform share of the batch) * 0.20 (that tag's rate in the
    #     WHOLE buffer, since the uniform draw doesn't exclude tagged rows)
    #   = 0.25 + 0.10 = 0.35
    # -- confirmed empirically (mean ~0.348/0.351 for human_event_frac/
    # collision_frac over repeated trials of this exact fixture; the sampler
    # itself is unchanged, only this expectation was wrong). abs=0.03
    # comfortably covers batch_size=2000 sampling noise around 0.35 while
    # still catching a genuine stratification break (e.g. an empty-pool
    # fallback silently degrading toward the ~0.20 whole-buffer rate, or a
    # pool-selection bug skewing far off 0.35).
    assert frac["human_event_frac"] == pytest.approx(0.35, abs=0.03)
    assert frac["collision_frac"] == pytest.approx(0.35, abs=0.03)


def test_balanced_sampler_falls_back_to_uniform_when_pool_empty():
    """No human-risk / collision transitions exist at all (e.g. early
    training, human-free stage) -- the sampler must still return a FULL batch
    (no duplicate-sampling error, no short batch), just drawn entirely from
    the uniform pool."""
    buf = _mk(store_risk_meta=True, risk_balanced_enabled=True,
              risk_balanced_ratios=(0.5, 0.25, 0.25))
    for i in range(500):
        _add(buf, val=i, risk_meta=(0.0, 0.0, 0.0, 0.0))
    ind = buf.sample_risk_balanced(batch_size=128)
    assert ind is not None
    assert ind.shape[0] == 128
    frac = buf.describe_risk_meta_fractions(ind)
    assert frac["human_event_frac"] == 0.0
    assert frac["collision_frac"] == 0.0


def test_balanced_sampler_partial_pool_still_fills_batch():
    """A small-but-nonempty collision pool is sampled WITH replacement rather
    than shrinking the batch or erroring."""
    buf = _mk(store_risk_meta=True, risk_balanced_enabled=True,
              risk_balanced_ratios=(0.5, 0.25, 0.25))
    for i in range(500):
        _add(buf, val=i, risk_meta=(0.0, 0.0, 0.0, 0.0))
    _add(buf, val=999, risk_meta=(0.0, 0.0, 0.0, 1.0))   # exactly one collision row
    ind = buf.sample_risk_balanced(batch_size=100)
    assert ind.shape[0] == 100
    frac = buf.describe_risk_meta_fractions(ind)
    assert frac["collision_frac"] > 0.0   # the lone row was drawn (with replacement)


def test_balanced_sampler_returns_none_when_disabled():
    # store_risk_meta=True (metadata IS collected) but risk_balanced_sampling
    # itself explicitly off -> sample_risk_balanced() must stay a no-op, the
    # metadata-collection-only use case from the requirements.
    buf = _mk_populated(risk_balanced_enabled=False)
    assert buf.sample_risk_balanced() is None


def test_balanced_sampler_returns_none_without_metadata_storage():
    buf = _mk(risk_balanced_enabled=True)  # store_risk_meta defaults False
    assert buf.risk_balanced_enabled is False   # AND-ed with store_risk_meta at construction
    _add(buf)
    assert buf.sample_risk_balanced() is None


def test_balanced_sampler_returns_none_on_empty_buffer():
    buf = _mk(store_risk_meta=True, risk_balanced_enabled=True)
    assert buf.size == 0
    assert buf.sample_risk_balanced() is None


# --------------------------------------------------------------------------- #
#  Episode-boundary safety for the balanced batch (explicit indices= param)
# --------------------------------------------------------------------------- #
def test_future_actions_respects_explicit_indices_and_boundary():
    buf = _mk(track_traj=True, store_risk_meta=True, risk_balanced_enabled=True)
    # Episode A: 3 transitions, ends at index 2.
    for i in range(3):
        _add(buf, val=i)
    buf.mark_last_traj_end()
    # Episode B: 3 more transitions.
    for i in range(3, 6):
        _add(buf, val=i)

    ind_rb = np.array([2, 5])   # last step of episode A, mid-episode B
    fut, vlen = buf.get_last_future_actions(4, indices=ind_rb)
    assert int(vlen[0]) == 1    # index 2 is the episode-A boundary itself
    assert int(vlen[1]) == 1    # index 5 is the buffer's current write frontier


def test_state_history_respects_explicit_indices_and_boundary():
    buf = _mk(track_traj=True, store_risk_meta=True, risk_balanced_enabled=True)
    for i in range(3):
        _add(buf, val=i)
    buf.mark_last_traj_end()
    for i in range(3, 6):
        _add(buf, val=i)

    ind_rb = np.array([5])
    hist, vlen = buf.get_last_state_history(4, indices=ind_rb)
    assert int(vlen[0]) == 3            # 5,4,3 then episode-A boundary at 2
    assert hist[0, :, 0].tolist() == [5.0, 4.0, 3.0, 0.0]


def test_default_indices_still_uses_self_ind_when_omitted():
    """Backward-compat: omitting `indices=` must still default to self.ind
    (the primary sample()'s draw), exactly as before this parameter existed."""
    buf = _mk(track_traj=True)
    for i in range(3):
        _add(buf, val=i)
    buf.ind = np.array([2])
    hist, vlen = buf.get_last_state_history(2)
    assert int(vlen[0]) == 2
    assert hist[0, :, 0].tolist() == [2.0, 1.0]


# --------------------------------------------------------------------------- #
#  Save / load: risk_meta round-trip + graceful fallback for an old checkpoint
# --------------------------------------------------------------------------- #
def test_risk_meta_survives_save_load(tmp_path):
    buf = _mk(store_risk_meta=True)
    _add(buf, risk_meta=(2.0, 1.0, 1.0, 0.0))
    _add(buf, risk_meta=(2.0, 0.0, 0.0, 1.0))
    path = str(tmp_path / "buf")
    buf.save(path)

    buf2 = _mk(store_risk_meta=True)
    assert buf2.load(path) is True
    assert buf2.risk_meta[:2].tolist() == [[2.0, 1.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]]


def test_old_checkpoint_without_risk_meta_loads_without_crash(tmp_path):
    """A checkpoint saved BEFORE this feature existed has no 'risk_meta' key at
    all. Loading it into a run that now wants metadata must NOT crash and must
    NOT guess values -- restored rows stay all-zero (safe: sample_risk_balanced
    then transparently falls back to the uniform pool for those rows)."""
    old = _mk(store_risk_meta=False)
    for i in range(5):
        _add(old, val=i)
    path = str(tmp_path / "old_buf")
    old.save(path)
    assert "risk_meta" not in np.load(path + ".npz").files

    new = _mk(store_risk_meta=True, risk_balanced_enabled=True)
    assert new.load(path) is True   # must not raise
    assert new.size == 5
    assert np.all(new.risk_meta[:5] == 0.0)
    # And risk-balanced sampling on this all-zero metadata gracefully falls
    # back to 100% uniform instead of erroring or returning an empty batch.
    ind = new.sample_risk_balanced(batch_size=32)
    assert ind is not None and ind.shape[0] == 32


def test_new_checkpoint_without_metadata_request_ignores_risk_meta_key(tmp_path):
    """The reverse direction: a checkpoint WITH risk_meta loaded into a buffer
    that doesn't want it (store_risk_meta=False) must simply ignore the key,
    same graceful pattern as aux_target/action_risk_target."""
    old = _mk(store_risk_meta=True)
    _add(old, risk_meta=(1.0, 1.0, 1.0, 1.0))
    path = str(tmp_path / "buf_with_meta")
    old.save(path)

    new = _mk(store_risk_meta=False)
    assert new.load(path) is True
    assert new.risk_meta is None


# --------------------------------------------------------------------------- #
#  OFF-parity: disabled feature changes NOTHING about the primary batch path
# --------------------------------------------------------------------------- #
def test_sample_and_get_last_accessors_unaffected_when_disabled():
    buf = _mk(aux_dim=3, action_risk_dim=2)
    for i in range(50):
        _add(buf, val=i, action_risk_target=np.array([0.1, 0.2], dtype=np.float32))
    state, action, next_state, reward, not_done = buf.sample()
    assert state.shape == (buf.batch_size, SDIM)
    aux = buf.get_last_aux()
    assert aux.shape == (buf.batch_size, 3)   # zero-padded (never passed to add() here)
    ar = buf.get_last_action_risk()
    assert ar.shape == (buf.batch_size, 2)
    assert buf.sample_risk_balanced() is None
    assert buf.ind_balanced is None
