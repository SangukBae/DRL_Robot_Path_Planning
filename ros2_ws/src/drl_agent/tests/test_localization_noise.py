"""ROS-free unit tests for the localization / proprio noise emulators.

These pin the behaviour that matters for training reproducibility: exact
pass-through when disabled, determinism under a fixed seed, latency-buffer delay,
and the per-map-type multiplier lookup.
"""

import math

import numpy as np
import pytest

from drl_agent.env.simulation.localization_noise import LocalizationNoiseModel, ProprioNoiseModel


def _loc_cfg(**over):
    cfg = dict(
        enabled=True,
        noise_goal_enabled=True,
        noise_delay_enabled=False,
        noise_jump_enabled=False,
        noise_flip_enabled=False,
        bias_xy_m=0.0,
        bias_yaw_rad=0.0,
        sigma_xy_m=0.0,
        sigma_yaw_rad=0.0,
        drift_xy_mps=0.0,
        drift_yaw_radps=0.0,
        corr_time_xy_s=0.0,
        corr_time_yaw_s=0.0,
        delay_steps=0,
        big_jump_prob=0.0,
        jump_prob=0.0,
        big_jump_xy_m=0.0,
        big_jump_yaw_rad=0.0,
        jump_xy_m=0.0,
        jump_yaw_rad=0.0,
        yaw_flip_prob=0.0,
        yaw_flip_map_types=[],
        map_type_multipliers={},
    )
    cfg.update(over)
    return cfg


def _pp_cfg(**over):
    cfg = dict(
        enabled=True,
        speed_sigma_mps=0.0,
        speed_bias_mps=0.0,
        speed_scale_sigma=0.0,
        yaw_rate_sigma_radps=0.0,
        yaw_rate_bias_radps=0.0,
        steer_sigma_rad=0.0,
        delay_steps=0,
    )
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------- localization
def test_loc_disabled_is_passthrough():
    m = LocalizationNoiseModel(_loc_cfg(enabled=False), time_delta=0.1)
    assert m.reset(1.0, 2.0, 0.3) == (1.0, 2.0, 0.3)
    for _ in range(5):
        assert m.step(1.5, 2.5, 0.4) == (1.5, 2.5, 0.4)


def test_loc_enabled_zero_noise_is_passthrough():
    m = LocalizationNoiseModel(_loc_cfg(), time_delta=0.1)
    seed = m.reset(0.0, 0.0, 0.0)
    assert seed == pytest.approx((0.0, 0.0, 0.0))
    out = m.step(1.0, -1.0, 0.5)
    assert out == pytest.approx((1.0, -1.0, 0.5))


def test_loc_constant_bias_applied():
    np.random.seed(0)
    m = LocalizationNoiseModel(_loc_cfg(bias_xy_m=0.5), time_delta=0.1)
    ix, iy, iyaw = m.reset(0.0, 0.0, 0.0)
    # Bias is sampled once at reset; the seeded estimate already carries it.
    assert ix != 0.0 or iy != 0.0
    # With no sigma/drift the per-step error equals the constant bias, so the
    # returned estimate stays offset from the clean pose by exactly the bias.
    out = m.step(0.0, 0.0, 0.0)
    assert out[0] == pytest.approx(ix)
    assert out[1] == pytest.approx(iy)


def test_loc_determinism_under_seed():
    m1 = LocalizationNoiseModel(_loc_cfg(sigma_xy_m=0.1, sigma_yaw_rad=0.05), time_delta=0.1)
    m2 = LocalizationNoiseModel(_loc_cfg(sigma_xy_m=0.1, sigma_yaw_rad=0.05), time_delta=0.1)
    np.random.seed(42)
    m1.reset(0.0, 0.0, 0.0)
    seq1 = [m1.step(1.0, 1.0, 0.0) for _ in range(10)]
    np.random.seed(42)
    m2.reset(0.0, 0.0, 0.0)
    seq2 = [m2.step(1.0, 1.0, 0.0) for _ in range(10)]
    assert seq1 == seq2


def test_loc_latency_delays_output():
    # delay_steps=2 → the first two steps return the seeded (reset) pose.
    m = LocalizationNoiseModel(
        _loc_cfg(noise_delay_enabled=True, delay_steps=2, sigma_xy_m=0.2),
        time_delta=0.1,
    )
    np.random.seed(1)
    seed = m.reset(3.0, 4.0, 0.0)
    out1 = m.step(3.0, 4.0, 0.0)
    out2 = m.step(3.0, 4.0, 0.0)
    assert out1 == pytest.approx(seed)
    assert out2 == pytest.approx(seed)
    out3 = m.step(3.0, 4.0, 0.0)
    # By the 3rd step the buffer has rotated to a freshly-noised sample.
    assert out3 != pytest.approx(seed)


def test_loc_map_multiplier_lookup():
    cfg = _loc_cfg(map_type_multipliers={
        "corridor": {"sigma_xy": 2.0, "drift": 3.0, "along_axis": "x", "along_extra": 4.0},
    })
    m = LocalizationNoiseModel(cfg, time_delta=0.1)
    msxy, msyaw, mdrift, mjump, axis, extra = m.map_multiplier("corridor")
    assert (msxy, mdrift, axis, extra) == (2.0, 3.0, "x", 4.0)
    assert msyaw == 1.0 and mjump == 1.0  # unspecified → 1.0
    # Unknown / empty map types fall back to all-1.0, no anisotropy.
    assert m.map_multiplier("") == (1.0, 1.0, 1.0, 1.0, "", 1.0)


# ----------------------------------------------------------------- proprio
def test_proprio_disabled_passthrough():
    m = ProprioNoiseModel(_pp_cfg(enabled=False))
    m.reset(1.0, 2.0, 3.0)
    assert m.peek() == (1.0, 2.0, 3.0)
    assert m.step(0.5, 0.6, 0.7) == (0.5, 0.6, 0.7)


def test_proprio_zero_noise_passthrough():
    m = ProprioNoiseModel(_pp_cfg())
    m.reset(1.0, 2.0, 3.0)
    assert m.step(0.5, 0.6, 0.7) == pytest.approx((0.5, 0.6, 0.7))


def test_proprio_determinism_and_peek():
    m1 = ProprioNoiseModel(_pp_cfg(speed_sigma_mps=0.1, speed_bias_mps=0.05))
    m2 = ProprioNoiseModel(_pp_cfg(speed_sigma_mps=0.1, speed_bias_mps=0.05))
    np.random.seed(7)
    m1.reset(0.0, 0.0, 0.0)
    peek1 = m1.peek()
    seq1 = [m1.step(1.0, 0.0, 0.0) for _ in range(5)]
    np.random.seed(7)
    m2.reset(0.0, 0.0, 0.0)
    peek2 = m2.peek()
    seq2 = [m2.step(1.0, 0.0, 0.0) for _ in range(5)]
    assert peek1 == peek2
    assert seq1 == seq2
