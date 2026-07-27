"""ROS-free unit tests for utils/seed_utils.py (reproducibility logic)."""

import random

import numpy as np
import pytest

import drl_agent.common.seed_utils as seed_utils


def test_seed_basic_rngs_makes_random_and_numpy_reproducible():
    names = seed_utils.seed_basic_rngs(123)
    assert names == ["random", "numpy"]
    seq_a = [random.random() for _ in range(5)] + list(np.random.rand(5))

    seed_utils.seed_basic_rngs(123)
    seq_b = [random.random() for _ in range(5)] + list(np.random.rand(5))
    assert seq_a == seq_b


def test_different_seeds_give_different_streams():
    seed_utils.seed_basic_rngs(1)
    a = [random.random() for _ in range(5)]
    seed_utils.seed_basic_rngs(2)
    b = [random.random() for _ in range(5)]
    assert a != b


def test_seed_all_without_torch_is_basic():
    names = seed_utils.seed_all(7, seed_torch=False)
    assert names == ["random", "numpy"]


def test_seed_all_with_torch_is_optional():
    # torch may or may not be importable in the test environment; either way the
    # function must succeed and at least seed random + numpy.
    names = seed_utils.seed_all(7, seed_torch=True)
    assert names[:2] == ["random", "numpy"]
    try:
        import torch  # noqa: F401
        assert "torch" in names
    except Exception:
        assert "torch" not in names


@pytest.mark.parametrize("base,offset,expected", [
    (0, 0, 0),
    (5, 100, 105),
    (1, 0, 1),
])
def test_derive_resume_seed_is_deterministic(base, offset, expected):
    assert seed_utils.derive_resume_seed(base, offset) == expected
    # deterministic: same inputs → same output
    assert seed_utils.derive_resume_seed(base, offset) == \
        seed_utils.derive_resume_seed(base, offset)


def test_derive_resume_seed_stays_in_int32():
    big = seed_utils.derive_resume_seed(2 ** 31 - 1, 10 ** 9)
    assert 0 <= big < 2 ** 31


def test_enable_torch_determinism_is_safe_without_torch():
    # Must never raise, always set the cuBLAS workspace env, and report whether
    # torch was actually available (torch-free env → torch False, no crash).
    info = seed_utils.enable_torch_determinism(warn_only=True)
    assert info["requested"] is True
    assert info["cublas_workspace_config"] == ":4096:8"
    import os
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "torch" in info
    try:
        import torch  # noqa: F401
        # When torch is present the deterministic flags must have been applied.
        assert info["torch"] is True
        assert info.get("cudnn_deterministic") is True
        assert info.get("cudnn_benchmark") is False
    except Exception:
        assert info["torch"] is False
