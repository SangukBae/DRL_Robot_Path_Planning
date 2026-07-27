#!/usr/bin/env python3
"""Centralised RNG seeding helpers (reproducibility).

Extracted so the env node, the trainers and the curriculum resume path all seed
the SAME set of RNGs in the SAME way, instead of each duplicating (and drifting
on) ``random.seed`` / ``np.random.seed`` / ``torch.manual_seed`` calls.

Design notes:
  * ``numpy`` is a hard dependency of the package, so it is imported eagerly.
  * ``torch`` is imported LAZILY inside ``seed_all`` so this module (and its unit
    tests) import fine in a torch-free environment; callers that do not need
    torch can use ``seed_basic_rngs`` and never touch it.
  * Each function returns the list of RNG names actually seeded, so callers can
    log exactly what was fixed.
"""

import os
import random

import numpy as np

# Max value that fits a signed 32-bit field (the Seed.srv int) — resume seeds
# are reduced modulo this so a derived seed never overflows the service request.
_INT32_MAX = 2 ** 31 - 1


def seed_basic_rngs(seed: int) -> list:
    """Seed the two global RNGs every process in this package draws from
    WITHOUT importing torch: Python ``random`` and NumPy. Returns the names
    seeded. Used by the environment node (no torch there)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    return ["random", "numpy"]


def seed_all(seed: int, *, seed_torch: bool = True) -> list:
    """Seed random + numpy (+ torch CPU and all CUDA devices when available).

    ``seed_torch=False`` keeps it torch-free. Returns the list of RNG names that
    were actually seeded so the caller can log them."""
    seeded = seed_basic_rngs(seed)
    if not seed_torch:
        return seeded
    try:
        import torch
    except Exception:
        return seeded
    torch.manual_seed(int(seed))
    seeded.append("torch")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        seeded.append("cuda")
    return seeded


def enable_torch_determinism(warn_only: bool = True) -> dict:
    """Make torch as deterministic as possible for reproducible TRAINING.

    Without this, two runs with the SAME seed and SAME code still diverge: cuDNN
    auto-tuning picks different conv/reduction kernels per run, and several CUDA
    ops use non-deterministic atomic accumulation. Those tiny gradient
    differences compound over 100k+ updates, so the policy reaches a given
    success/SPL at slightly different steps each run — exactly the stage-0
    drift we saw between the 20260613/14 and 20260615 curriculum runs even
    though stage 0 (empty lobby) has an identical environment stream.

    Effects (all best-effort; never raises):
      * cudnn.deterministic=True, cudnn.benchmark=False  → fixed cuDNN kernels
      * torch.use_deterministic_algorithms(True, warn_only) → deterministic ops
        (warn_only=True downgrades "no deterministic impl" errors to warnings so
        a rare op can't hard-crash a long training run)
      * CUBLAS_WORKSPACE_CONFIG=:4096:8 → required for deterministic CUDA matmul

    Returns the flags actually applied (logged into the run manifest). Call AFTER
    seeding and BEFORE the first GPU matmul (cuBLAS reads the env var on first
    handle creation).
    """
    applied = {"requested": True}
    # Must be set before the first cuBLAS handle is created; setdefault so an
    # explicit user value is respected.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    applied["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    try:
        import torch
    except Exception:
        applied["torch"] = False
        return applied
    applied["torch"] = True
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        applied["cudnn_deterministic"] = True
        applied["cudnn_benchmark"] = False
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        applied["use_deterministic_algorithms"] = True
        applied["warn_only"] = bool(warn_only)
    except Exception as e:
        applied["use_deterministic_algorithms"] = False
        applied["error"] = repr(e)
    return applied


def make_substream_rngs(base_seed: int, index: int):
    """Return ``(np.random.RandomState, random.Random)`` for a DEDICATED RNG
    sub-stream seeded from a well-mixed ``(base_seed, index)`` pair.

    Use this for any sampling that must NOT advance the global ``np.random`` /
    ``random`` streams — e.g. pedestrian spawn + motion. Because the human-motion
    ROS timer fires on wall-clock time, the number of motion draws between two
    episodes is non-deterministic; if those draws came from the global RNG they
    would shift the NEXT episode's start/goal/map sampling. A private sub-stream
    keeps the global stream reproducible regardless of how many motion ticks ran,
    and re-seeding it per episode (index = episode counter) makes each episode's
    human SPAWN config reproducible and independent of the previous episode's
    tick count.

    ``np.random.RandomState`` (legacy MT19937) is chosen over ``default_rng``
    deliberately: it is a drop-in for the ``np.random.uniform/choice/rand`` calls
    being moved here (same signatures + same per-call distribution), so the
    substitution is mechanical and behaviour-preserving. ``SeedSequence`` mixes
    the two integers into a high-quality 32-bit seed so different episodes get
    well-separated streams.
    """
    ss = np.random.SeedSequence([int(base_seed) & 0xFFFFFFFF,
                                 int(index) & 0xFFFFFFFF])
    s = int(ss.generate_state(1, dtype=np.uint32)[0])
    return np.random.RandomState(s), random.Random(s)


def derive_resume_seed(base_seed: int, offset: int, modulus: int = _INT32_MAX) -> int:
    """Deterministically derive a resume seed from a base seed and an offset
    (e.g. the global timestep at the checkpoint), kept inside int32 range.

    Pure and deterministic: the same (base_seed, offset) always yields the same
    value, so a given checkpoint always resumes the same environment stream.
    """
    return (int(base_seed) + int(offset)) % int(modulus)
