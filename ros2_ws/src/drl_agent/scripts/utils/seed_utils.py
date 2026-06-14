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


def derive_resume_seed(base_seed: int, offset: int, modulus: int = _INT32_MAX) -> int:
    """Deterministically derive a resume seed from a base seed and an offset
    (e.g. the global timestep at the checkpoint), kept inside int32 range.

    Pure and deterministic: the same (base_seed, offset) always yields the same
    value, so a given checkpoint always resumes the same environment stream.
    """
    return (int(base_seed) + int(offset)) % int(modulus)
