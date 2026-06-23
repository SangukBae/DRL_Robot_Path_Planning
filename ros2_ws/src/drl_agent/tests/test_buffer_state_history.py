"""Boundary-safety tests for LAP.get_last_state_history (AUX_PRED v2 temporal).

The aux-only temporal context walks the replay buffer BACKWARD for the last N
in-episode states. These tests lock the contract that makes that safe for
off-policy i.i.d. replay:
  * the walk stops at an episode boundary (traj_end) — never splices across it,
  * it stops at the circular-buffer seam (oldest written slot),
  * out-of-range steps are zero-padded and valid_len reports the true count,
  * a save/load round-trip preserves the boundary flags it depends on.

torch is imported defensively (buffer.py imports torch); skipped if unavailable.
"""

import os

import pytest

try:
    import numpy as np
    import torch
    import buffer
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")

SDIM, ADIM = 3, 2


def _mk(max_size=64):
    return buffer.LAP(SDIM, ADIM, torch.device("cpu"), max_size=max_size,
                      batch_size=4, normalize_actions=False, prioritized=False,
                      track_traj=True)


def _add(buf, s_val, done=False):
    s = np.full(SDIM, float(s_val), dtype=np.float32)
    buf.add(s, np.zeros(ADIM), s, 0.0, float(done))


def test_history_stops_at_episode_boundary():
    buf = _mk()
    # Episode A: states 0,1,2  (transition 2 ends the episode)
    _add(buf, 0); _add(buf, 1); _add(buf, 2, done=True); buf.mark_last_traj_end()
    # Episode B: states 3,4,5
    _add(buf, 3); _add(buf, 4); _add(buf, 5)
    buf.ind = np.array([5])           # newest state, mid-episode-B
    hist, vlen = buf.get_last_state_history(4)
    assert int(vlen[0]) == 3                                  # 5,4,3 then boundary
    assert hist[0, :, 0].tolist() == [5.0, 4.0, 3.0, 0.0]     # k=3 zero-padded


def test_history_stops_at_oldest_seam_when_not_full():
    buf = _mk()
    _add(buf, 0); _add(buf, 1); _add(buf, 2)   # one episode, no boundary
    buf.ind = np.array([1])
    hist, vlen = buf.get_last_state_history(4)
    assert int(vlen[0]) == 2                    # 1,0 then oldest(0) seam
    assert hist[0, :, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    buf.ind = np.array([0])
    _, vlen0 = buf.get_last_state_history(4)
    assert int(vlen0[0]) == 1                   # the very first state: only itself


def test_history_current_state_always_index0():
    buf = _mk()
    for i in range(5):
        _add(buf, i)
    buf.ind = np.array([0, 2, 4])
    hist, vlen = buf.get_last_state_history(3)
    assert hist[:, 0, 0].tolist() == [0.0, 2.0, 4.0]   # index 0 == sampled state
    assert vlen.tolist() == [1, 3, 3]


def test_history_wraps_full_buffer_without_crossing_seam():
    buf = _mk(max_size=6)
    # Fill then overwrite so the buffer wraps: ptr lands at 2 (oldest == 2).
    for i in range(8):                 # writes slots 0..5,0,1 -> ptr=2, full
        _add(buf, i)
    assert buf.size == 6 and buf.ptr == 2
    buf.ind = np.array([1])            # slot 1 holds the newest state (value 7)
    # Chronological order oldest->newest is slots 2,3,4,5,0,1 (values 2,3,4,5,6,7).
    # With no episode boundary the backward walk legitimately spans all 6 states.
    hist, vlen = buf.get_last_state_history(6)
    assert int(vlen[0]) == 6
    assert hist[0, :, 0].tolist() == [7.0, 6.0, 5.0, 4.0, 3.0, 2.0]  # 1,0,5,4,3,2
    # Asking for MORE than the buffer holds must NOT wrap past the seam: it caps
    # at 6 valid and zero-pads the rest (never re-reads the newest / write head).
    hist2, vlen2 = buf.get_last_state_history(8)
    assert int(vlen2[0]) == 6
    assert hist2[0, 6, 0] == 0.0 and hist2[0, 7, 0] == 0.0


def test_history_disabled_without_track_traj():
    buf = buffer.LAP(SDIM, ADIM, torch.device("cpu"), max_size=16, batch_size=2,
                     normalize_actions=False, prioritized=False, track_traj=False)
    _add(buf, 0); _add(buf, 1)
    buf.ind = np.array([1])
    assert buf.get_last_state_history(4) is None   # graceful: no boundary tracking


def test_load_boundaryless_buffer_into_tracking_run_fail_fasts(tmp_path):
    """Low-level guard: a buffer saved WITHOUT episode boundaries cannot be
    loaded into a run that needs them (temporal / action-conditioned aux) — it
    would splice the backward/forward walks across episodes. buffer.load() raises
    here; tqc_io.load is what degrades this to a fresh buffer at resume time."""
    old = buffer.LAP(SDIM, ADIM, torch.device("cpu"), max_size=16, batch_size=2,
                     normalize_actions=False, prioritized=False, track_traj=False)
    _add(old, 0); _add(old, 1)
    path = os.path.join(str(tmp_path), "old_buf")
    old.save(path)
    new = _mk(max_size=16)               # track_traj=True -> needs boundaries
    with pytest.raises(RuntimeError):
        new.load(path)


def test_traj_end_survives_save_load(tmp_path):
    buf = _mk()
    _add(buf, 0); _add(buf, 1, done=True); buf.mark_last_traj_end(); _add(buf, 2)
    path = os.path.join(str(tmp_path), "buf")
    buf.save(path)
    buf2 = _mk()
    assert buf2.load(path) is True
    buf2.ind = np.array([2])
    _, vlen = buf2.get_last_state_history(4)
    assert int(vlen[0]) == 1            # boundary at transition 1 preserved
