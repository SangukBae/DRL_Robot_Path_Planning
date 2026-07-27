"""Replay-transition schema for the LAP buffer (documentation + introspection).

Single place that names every per-transition field the LAP replay buffer can
carry, which constructor flag enables it, and which key it uses inside the
``<prefix>_replay_buffer.npz`` checkpoint. The buffer implementation
(:mod:`drl_agent.rl.replay.buffer`) is the behavioural source of truth; this
module exists so tests / tooling can assert the on-disk contract without
re-deriving it, and so the optional AUX_PRED / PHASE2 fields are documented in
one spot.

Pure numpy/stdlib — importable without torch.

Field overview
--------------
Core (always present, enabled by the constructor's ``state_dim``/``action_dim``):

  state / next_state : (N, state_dim)  raw transported state (87-D baseline, or
                       80*N+7 when observation time-context stacking is ON)
  action             : (N, action_dim) stored NORMALIZED (divided by max_action
                       when ``normalize_actions`` is true)
  reward, not_done   : (N, 1)

Optional (each defaults OFF; the buffer stays byte-identical to baseline
when its flag is off):

  aux_target         : (N, aux_dim)  AUX_PRED future-risk supervision label,
                       aligned with ``state`` (the encoder input). Enabled by
                       ``aux_dim > 0``.
  traj_end           : (N, 1)  episode-boundary flag; needed whenever
                       future-action (action-conditioned aux) or backward
                       state-history (temporal aux) walks must not cross an
                       episode boundary. Enabled by ``track_traj=True``.
  action_risk_target : (N, action_risk_dim)  PHASE2 Action-Risk Head
                       supervision (risk_dir, min_dist_dir), aligned with
                       ``state``+``action``. Enabled by ``action_risk_dim > 0``.

Checkpoint layout (see ``LAP.save`` / ``LAP.load``):

  <prefix>.npz            core fields + optional fields present in THIS run's
                          config, plus ``meta = [ptr, size, max_size]`` and
                          ``max_priority`` (1.0 when non-prioritized).
  <prefix>_priority.pt    torch tensor sidecar, written only when prioritized.
"""

from collections import namedtuple

ReplayField = namedtuple(
    "ReplayField",
    ["name", "npz_key", "enabled_by", "width_attr", "description"],
)

#: Core per-transition fields — always saved.
CORE_FIELDS = (
    ReplayField("state", "state", None, "state_dim",
                "raw transported state (encoder/actor input)"),
    ReplayField("action", "action", None, "action_dim",
                "normalized action (divided by max_action when normalize_actions)"),
    ReplayField("next_state", "next_state", None, "state_dim",
                "raw transported next state"),
    ReplayField("reward", "reward", None, None, "scalar reward, shape (N, 1)"),
    ReplayField("not_done", "not_done", None, None, "1 - done, shape (N, 1)"),
)

#: Optional per-transition fields — saved only when their flag is on.
OPTIONAL_FIELDS = (
    ReplayField("aux_target", "aux_target", "aux_dim > 0", "aux_dim",
                "AUX_PRED future-risk label aligned with state"),
    ReplayField("traj_end", "traj_end", "track_traj", None,
                "episode-boundary flag for future-action / state-history walks"),
    ReplayField("action_risk_target", "action_risk_target",
                "action_risk_dim > 0", "action_risk_dim",
                "PHASE2 action-risk label aligned with state+action"),
)

#: Bookkeeping keys always present in the .npz checkpoint.
META_KEYS = ("meta", "max_priority")


def expected_npz_keys(aux_dim=0, track_traj=False, action_risk_dim=0):
    """Exact key set ``LAP.save`` writes for a buffer built with these flags."""
    keys = {f.npz_key for f in CORE_FIELDS} | set(META_KEYS)
    if int(aux_dim) > 0:
        keys.add("aux_target")
    if bool(track_traj):
        keys.add("traj_end")
    if int(action_risk_dim) > 0:
        keys.add("action_risk_target")
    return keys


def describe_buffer(buf):
    """Flags of a live ``LAP`` instance, as ``expected_npz_keys`` kwargs."""
    return {
        "aux_dim": int(getattr(buf, "aux_dim", 0) or 0),
        "track_traj": getattr(buf, "traj_end", None) is not None,
        "action_risk_dim": int(getattr(buf, "action_risk_dim", 0) or 0),
    }


def validate_npz_keys(npz_files, aux_dim=0, track_traj=False, action_risk_dim=0):
    """Check a saved checkpoint's key set against this schema.

    Parameters
    ----------
    npz_files : iterable of str
        ``numpy.load(path).files`` of the checkpoint.
    Returns ``(missing, unexpected)`` key sets — both empty on an exact match.
    """
    expected = expected_npz_keys(aux_dim, track_traj, action_risk_dim)
    actual = set(npz_files)
    return expected - actual, actual - expected
