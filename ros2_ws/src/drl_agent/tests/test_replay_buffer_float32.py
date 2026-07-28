"""Unit tests for Stage-7: replay buffer continuous arrays converted to
explicit float32 (from the previous no-dtype-specified, implicit float64).

Real measured motivation (Stage 0 baseline, live Docker): the actual phase2/
both replay checkpoint (180k/1M transitions, state_dim=327) is 1032.5MB in
float64; extrapolated to the configured max_size=1,000,000 (eagerly allocated
at buffer construction regardless of fill level) that's ~5.7GB float64 ->
~2.87GB at float32.

Covers:
  * every continuous array (state/action/next_state/reward/not_done/
    aux_target/action_risk_target) is np.float32 from construction.
  * add()/sample() preserve float32 end-to-end (torch tensors are float32,
    matching torch.float exactly, not silently upcast).
  * load() explicitly casts an OLDER float64 .npz checkpoint to float32 (not
    relying on implicit numpy behavior alone), preserving shape/schema
    validation BEFORE the dtype conversion (existing fail-fast checks
    untouched), with a bounded float64->float32 precision tolerance and a
    printed log line documenting the cast.
  * save() from an already-float32 buffer writes a float32 .npz (a fresh
    save->load round trip stays float32 throughout, no upcast anywhere).
  * a checkpoint that already IS float32 loads with no cast log (no false
    "casting" claim when nothing was actually cast).
  * measured (not estimated) memory usage: float32 buffer occupies
    (byte-for-byte) half of an identically-shaped float64 buffer.
"""

import numpy as np
import pytest

try:
    import torch
    from drl_agent.rl.replay.buffer import LAP
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")

STATE_DIM, ACTION_DIM = 7, 2


def _buf(**over):
    kw = dict(state_dim=STATE_DIM, action_dim=ACTION_DIM, device=torch.device("cpu"),
              max_size=32, batch_size=4, prioritized=True)
    kw.update(over)
    return LAP(**kw)


def _fill(buf, n=10, aux_dim=0, action_risk_dim=0, rng=None):
    rng = rng or np.random.default_rng(0)
    for _ in range(n):
        buf.add(
            rng.normal(size=STATE_DIM), rng.normal(size=ACTION_DIM),
            rng.normal(size=STATE_DIM), float(rng.normal()), 0.0,
            aux_target=(rng.normal(size=aux_dim) if aux_dim else None),
            action_risk_target=(rng.normal(size=action_risk_dim) if action_risk_dim else None),
        )


# --------------------------------------------------------------------------- #
#  array dtypes at construction
# --------------------------------------------------------------------------- #
def test_core_arrays_are_float32():
    buf = _buf()
    assert buf.state.dtype == np.float32
    assert buf.action.dtype == np.float32
    assert buf.next_state.dtype == np.float32
    assert buf.reward.dtype == np.float32
    assert buf.not_done.dtype == np.float32


def test_optional_arrays_are_float32_when_enabled():
    buf = _buf(aux_dim=6, action_risk_dim=2, track_traj=True)
    assert buf.aux_target.dtype == np.float32
    assert buf.action_risk_target.dtype == np.float32
    assert buf.traj_end.dtype == np.float32


# --------------------------------------------------------------------------- #
#  add()/sample() preserve float32 end-to-end
# --------------------------------------------------------------------------- #
def test_add_stores_as_float32_even_from_float64_input():
    buf = _buf()
    rng = np.random.default_rng(1)
    s64 = rng.normal(size=STATE_DIM).astype(np.float64)
    buf.add(s64, rng.normal(size=ACTION_DIM), s64, 1.0, 0.0)
    assert buf.state.dtype == np.float32
    assert buf.state[0].dtype == np.float32


def test_sample_returns_float32_tensors():
    buf = _buf()
    _fill(buf)
    state, action, next_state, reward, not_done = buf.sample()
    for t in (state, action, next_state, reward, not_done):
        assert t.dtype == torch.float32


def test_get_last_aux_and_action_risk_return_float32():
    buf = _buf(aux_dim=6, action_risk_dim=2)
    _fill(buf, aux_dim=6, action_risk_dim=2)
    buf.sample()
    aux = buf.get_last_aux()
    ar = buf.get_last_action_risk()
    assert aux.dtype == torch.float32
    assert ar.dtype == torch.float32


# --------------------------------------------------------------------------- #
#  load(): explicit, logged cast from an older float64 checkpoint
# --------------------------------------------------------------------------- #
def _save_raw_float64_checkpoint(path, size, state_dim, action_dim):
    """Write a .npz using the buffer's OLD (pre-Stage-7) no-dtype-specified
    convention directly, bypassing LAP.save() -- simulates a real checkpoint
    saved before this change."""
    rng = np.random.default_rng(2)
    state = rng.normal(size=(size, state_dim))       # float64 (numpy default)
    action = rng.normal(size=(size, action_dim))
    next_state = rng.normal(size=(size, state_dim))
    reward = rng.normal(size=(size, 1))
    not_done = np.ones((size, 1))
    assert state.dtype == np.float64
    np.savez_compressed(
        path, state=state, action=action, next_state=next_state,
        reward=reward, not_done=not_done,
        meta=np.array([size % 32, size, 32]),
        max_priority=np.array([1.0]),
    )
    return state, action, next_state, reward, not_done


def test_load_casts_float64_checkpoint_to_float32_within_tolerance(tmp_path, capsys):
    prefix = str(tmp_path / "legacy_rb")
    state64, action64, *_ = _save_raw_float64_checkpoint(prefix + ".npz", 10, STATE_DIM, ACTION_DIM)

    buf = _buf()
    assert buf.load(prefix) is True
    assert buf.state.dtype == np.float32
    # float64 -> float32 precision loss is tiny; bounded, explicit tolerance.
    np.testing.assert_allclose(buf.state[:10], state64, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(buf.action[:10], action64, rtol=1e-6, atol=1e-6)

    out = capsys.readouterr().out
    assert "float32" in out and "float64" in out, \
        "load() must log the float64->float32 cast, not silently downcast"


def test_load_from_already_float32_checkpoint_does_not_claim_a_cast(tmp_path, capsys):
    buf1 = _buf()
    _fill(buf1)
    prefix = str(tmp_path / "fresh_rb")
    buf1.save(prefix)
    assert np.load(prefix + ".npz")["state"].dtype == np.float32  # fresh save is already float32

    buf2 = _buf()
    assert buf2.load(prefix) is True
    out = capsys.readouterr().out
    assert "float64" not in out, \
        "loading an already-float32 checkpoint must not print a cast log"


def test_load_never_mutates_the_source_npz_file(tmp_path):
    prefix = str(tmp_path / "legacy_rb2")
    npz_path = prefix + ".npz"
    _save_raw_float64_checkpoint(npz_path, 8, STATE_DIM, ACTION_DIM)
    before_bytes = open(npz_path, "rb").read()

    buf = _buf()
    assert buf.load(prefix) is True

    after_bytes = open(npz_path, "rb").read()
    assert before_bytes == after_bytes, \
        "load() (a read-only operation) must never modify the source .npz"


def test_shape_validation_still_runs_before_dtype_conversion(tmp_path):
    # A checkpoint with the WRONG state_dim must still fail-fast (existing
    # contract), regardless of dtype -- the float32 conversion must not have
    # displaced or weakened this guard.
    prefix = str(tmp_path / "bad_dim")
    rng = np.random.default_rng(3)
    np.savez_compressed(
        prefix + ".npz",
        state=rng.normal(size=(5, STATE_DIM + 1)),  # wrong width
        action=rng.normal(size=(5, ACTION_DIM)),
        next_state=rng.normal(size=(5, STATE_DIM + 1)),
        reward=rng.normal(size=(5, 1)), not_done=np.ones((5, 1)),
        meta=np.array([5, 5, 32]), max_priority=np.array([1.0]),
    )
    buf = _buf()
    with pytest.raises(RuntimeError, match="state_dim"):
        buf.load(prefix)


# --------------------------------------------------------------------------- #
#  fresh float32 save -> load round trip
# --------------------------------------------------------------------------- #
def test_fresh_float32_save_load_round_trip_stays_float32(tmp_path):
    buf1 = _buf(aux_dim=4, action_risk_dim=2, track_traj=True)
    _fill(buf1, aux_dim=4, action_risk_dim=2)
    buf1.mark_last_traj_end()
    prefix = str(tmp_path / "rt")
    buf1.save(prefix)

    on_disk = np.load(prefix + ".npz")
    assert on_disk["state"].dtype == np.float32
    assert on_disk["aux_target"].dtype == np.float32
    assert on_disk["action_risk_target"].dtype == np.float32

    buf2 = _buf(aux_dim=4, action_risk_dim=2, track_traj=True)
    assert buf2.load(prefix) is True
    assert buf2.state.dtype == np.float32
    np.testing.assert_allclose(buf2.state[:buf2.size], buf1.state[:buf1.size])
    np.testing.assert_allclose(buf2.aux_target[:buf2.size], buf1.aux_target[:buf1.size])


# --------------------------------------------------------------------------- #
#  measured (not estimated) memory usage
# --------------------------------------------------------------------------- #
def test_float32_buffer_uses_exactly_half_the_bytes_of_float64():
    max_size = 1000
    buf32 = _buf(max_size=max_size)
    nbytes32 = (buf32.state.nbytes + buf32.action.nbytes + buf32.next_state.nbytes
                + buf32.reward.nbytes + buf32.not_done.nbytes)

    # Reference: what the ORIGINAL no-dtype-specified (float64) allocation
    # would have used, for the SAME shapes.
    nbytes64 = (
        np.zeros((max_size, STATE_DIM)).nbytes
        + np.zeros((max_size, ACTION_DIM)).nbytes
        + np.zeros((max_size, STATE_DIM)).nbytes
        + np.zeros((max_size, 1)).nbytes
        + np.zeros((max_size, 1)).nbytes
    )
    assert nbytes32 * 2 == nbytes64
