"""ROS-free unit tests for collision_checker.RectSafetyChecker.

Covers the ray–rectangle geometry, the precomputed per-bin ranges, the paper
Algorithm 1 collision test, and the reward-proximity deficit.
"""

import math

import numpy as np
import pytest

from drl_agent.env.simulation.collision_checker import RectSafetyChecker


def _make_bins(n):
    """Full-360 bin layout matching environment.py's collision bins."""
    eps = 0.03
    width = 2 * math.pi / n
    start = -math.pi - eps
    bins = [[start + i * width, start + (i + 1) * width] for i in range(n)]
    bins[-1][-1] += eps
    return bins


def _checker(n=16, **over):
    cfg = dict(
        d_front=0.40, d_rear=0.40, d_left=0.30, d_right=0.30,
        margin_front=0.0, margin_rear=0.0, margin_left=0.0, margin_right=0.0,
        bins=_make_bins(n), environment_dim=n,
        collision_threshold=0.3, lidar_max_range=10.0,
    )
    cfg.update(over)
    return RectSafetyChecker(**cfg)


def test_safety_hit_front_face():
    c = _checker()
    dist, face = c.compute_safety_hit(0.0)  # straight ahead
    assert face == "front"
    assert dist == pytest.approx(0.40)


def test_safety_hit_rear_face():
    c = _checker()
    dist, face = c.compute_safety_hit(math.pi)  # straight back
    assert face == "rear"
    assert dist == pytest.approx(0.40)


def test_safety_hit_left_right_faces():
    c = _checker()
    dist_l, face_l = c.compute_safety_hit(math.pi / 2)
    dist_r, face_r = c.compute_safety_hit(-math.pi / 2)
    assert face_l == "left" and dist_l == pytest.approx(0.30)
    assert face_r == "right" and dist_r == pytest.approx(0.30)


def test_margins_inflate_footprint():
    c = _checker(margin_front=0.1)
    dist, face = c.compute_safety_hit(0.0)
    assert face == "front"
    assert dist == pytest.approx(0.50)


def test_warning_scale_expands_ranges():
    c = _checker(warning_scale_front=1.5, warning_scale_rear=1.5,
                 warning_scale_left=1.5, warning_scale_right=1.5)
    # Warning ranges are uniformly 1.5x the safety ranges here.
    assert np.allclose(c.warning_ranges, 1.5 * c.safety_ranges)


def test_precompute_ranges_shape_and_finiteness():
    c = _checker(n=24)
    assert c.safety_ranges.shape == (24,)
    assert np.all(np.isfinite(c.safety_ranges))
    assert np.all(c.safety_ranges > 0.0)


def test_check_collision_triggers_when_below_range():
    c = _checker(n=16)
    obs = np.full(16, 5.0)  # all clear
    done, coll, _ = c.check_collision(obs)
    assert not done and not coll
    # Make one bin read inside its safety range.
    idx = 0
    obs[idx] = c.safety_ranges[idx] * 0.5
    done, coll, min_used = c.check_collision(obs)
    assert done and coll
    assert min_used == pytest.approx(obs[idx])


def test_check_collision_ignores_max_range_returns():
    c = _checker(n=16)
    obs = np.full(16, 10.0)  # == lidar_max_range → treated as "no return"
    done, coll, min_used = c.check_collision(obs)
    assert not done and not coll
    assert math.isinf(min_used)


def test_compute_proximity_bounds():
    c = _checker(n=16, warning_scale_front=2.0, warning_scale_rear=2.0,
                 warning_scale_left=2.0, warning_scale_right=2.0)
    # Fully clear → zero deficit.
    assert c.compute_proximity(np.full(16, 9.0)) == pytest.approx(0.0)
    # At the hard safety boundary → deficit approaches 1 - safety/warning = 0.5.
    obs = c.safety_ranges.copy()
    prox = c.compute_proximity(obs)
    assert prox == pytest.approx(0.5, abs=1e-6)
    assert 0.0 <= prox <= 1.0
