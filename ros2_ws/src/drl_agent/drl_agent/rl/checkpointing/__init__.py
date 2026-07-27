"""Checkpoint management for the RL agents.

Responsibility split:

  manager.py  discovery + validation (which run dir / checkpoint prefix to
              resume from, ResumeState bookkeeping) — pure stdlib, no torch.
  tqc_io.py   the actual TQC save()/load()/load_encoder_for_inference()
              implementations (state_dicts + optimizer state + replay buffer;
              requires torch, import the submodule lazily).

Legacy bare-name shim: ``tqc_io``.
"""

from .manager import CheckpointManager, ResumeState  # noqa: F401

__all__ = ["CheckpointManager", "ResumeState", "tqc_io"]
