"""Contract test for the SHARED free-pose sampler's RNG threading.

The shared samplers (start_sampler._sample_free_pose / _sample_free_pose_in_region
/ _sample_free_pose_layout and map_layout_runtime._sample_xy_in_regions) are used
by goal / robot-start / static-obstacle sampling (global RNG) AND by pedestrian
spawn (human sub-stream). The fix threads an optional ``rng`` through all of them
with the single idiom:

    r = rng if rng is not None else np.random
    ... r.uniform(...) / r.choice(...) ...

so a None rng keeps the global stream (static/robot/goal unchanged) while a human
caller passing ``self._human_np_rng`` never advances the global stream.

Those modules import rclpy and cannot be imported in this ROS-free suite, so this
test exercises the EXACT idiom against the REAL ``seed_utils.make_substream_rngs``
sub-stream. (The real `_sample_xy_in_regions` was additionally verified by hand
under ROS stubs: rng=substream leaves the global stream put; rng=None consumes it.)
"""

import random

import numpy as np

import drl_agent.common.seed_utils as seed_utils


def _shared_sampler(regions, radius, rng=None):
    """Faithful replica of map_layout_runtime._sample_xy_in_regions's RNG path —
    same ``r = rng if rng is not None else np.random`` contract, same draw count
    (one choice + two uniform)."""
    r = rng if rng is not None else np.random
    usable, weights = [], []
    for (x_lo, x_hi, y_lo, y_hi) in regions:
        ax_lo, ax_hi = x_lo + radius, x_hi - radius
        ay_lo, ay_hi = y_lo + radius, y_hi - radius
        if ax_hi <= ax_lo or ay_hi <= ay_lo:
            continue
        usable.append((ax_lo, ax_hi, ay_lo, ay_hi))
        weights.append((ax_hi - ax_lo) * (ay_hi - ay_lo))
    if not usable:
        return None
    w = np.asarray(weights, dtype=float)
    idx = int(r.choice(len(usable), p=w / w.sum()))
    ax_lo, ax_hi, ay_lo, ay_hi = usable[idx]
    return float(r.uniform(ax_lo, ax_hi)), float(r.uniform(ay_lo, ay_hi))


_REGIONS = [(-5.0, 5.0, -5.0, 5.0)]


def test_human_substream_does_not_advance_global_stream():
    np.random.seed(0)
    base = [np.random.rand() for _ in range(5)]
    # Re-seed global, then run MANY human-substream sampler calls in between.
    np.random.seed(0)
    h_np, _ = seed_utils.make_substream_rngs(0, 0)
    for _ in range(500):
        assert _shared_sampler(_REGIONS, 0.3, rng=h_np) is not None
    after = [np.random.rand() for _ in range(5)]
    assert after == base          # human spawn draws left the global stream put


def test_default_rng_still_uses_global_stream():
    # Static / goal / robot callers (rng=None) MUST keep drawing from global so
    # their sampling is unchanged by the human split.
    np.random.seed(0)
    base = [np.random.rand() for _ in range(5)]
    np.random.seed(0)
    _shared_sampler(_REGIONS, 0.3)                # rng=None -> global
    after = [np.random.rand() for _ in range(5)]
    assert after != base


def test_human_substream_is_reproducible_for_same_seed():
    h1, _ = seed_utils.make_substream_rngs(7, 3)
    a = [_shared_sampler(_REGIONS, 0.3, rng=h1) for _ in range(10)]
    h2, _ = seed_utils.make_substream_rngs(7, 3)
    b = [_shared_sampler(_REGIONS, 0.3, rng=h2) for _ in range(10)]
    assert a == b
