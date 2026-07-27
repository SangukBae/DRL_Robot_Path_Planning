"""ROS-free unit tests for the dedicated pedestrian (human) RNG sub-stream.

These lock the reproducibility contract behind the human RNG split: human spawn
+ motion sampling draws from a private sub-stream (seed_utils.make_substream_rngs)
so the wall-clock-paced human-motion timer can NEVER perturb the global
random/np.random streams used for the robot's start/goal/map/static sampling, and
each episode's human spawn config is a pure function of (run seed, episode index).
"""

import random

import numpy as np

import drl_agent.common.seed_utils as seed_utils


def test_make_substream_rngs_returns_randomstate_and_random():
    np_rng, py_rng = seed_utils.make_substream_rngs(7, 0)
    # RandomState chosen for drop-in API compatibility (.uniform/.choice/.rand).
    assert isinstance(np_rng, np.random.RandomState)
    assert isinstance(py_rng, random.Random)
    assert hasattr(np_rng, "rand") and hasattr(np_rng, "choice")


def test_substream_is_deterministic_for_same_base_and_index():
    a_np, a_py = seed_utils.make_substream_rngs(7, 3)
    b_np, b_py = seed_utils.make_substream_rngs(7, 3)
    assert [a_np.uniform() for _ in range(8)] == [b_np.uniform() for _ in range(8)]
    assert [a_py.random() for _ in range(8)] == [b_py.random() for _ in range(8)]


def test_substream_differs_across_index_and_base():
    base_idx = [seed_utils.make_substream_rngs(7, 1)[0].uniform() for _ in range(8)]
    other_idx = [seed_utils.make_substream_rngs(7, 2)[0].uniform() for _ in range(8)]
    other_base = [seed_utils.make_substream_rngs(8, 1)[0].uniform() for _ in range(8)]
    assert base_idx != other_idx
    assert base_idx != other_base


def test_substream_draws_do_not_advance_global_rngs():
    # The core guarantee: consuming human draws must leave the GLOBAL streams
    # (robot start/goal/map/static sampling) exactly where they were.
    np.random.seed(123)
    random.seed(123)
    want_np = [np.random.rand() for _ in range(5)]
    want_py = [random.random() for _ in range(5)]

    np.random.seed(123)
    random.seed(123)
    h_np, h_py = seed_utils.make_substream_rngs(123, 0)
    for _ in range(1000):           # stand-in for many wall-clock motion ticks
        h_np.uniform()
        h_np.rand()
        h_py.random()
        h_py.shuffle([1, 2, 3, 4])
    got_np = [np.random.rand() for _ in range(5)]
    got_py = [random.random() for _ in range(5)]

    assert got_np == want_np        # global numpy stream untouched by human draws
    assert got_py == want_py        # global python-random stream untouched


def test_per_episode_reseed_isolates_from_previous_episode_tick_count():
    # Episode N's spawn config must be identical regardless of how many motion
    # draws episode N-1 consumed (the tick count is wall-clock dependent). The env
    # achieves this by reseeding make_substream_rngs(base, episode) every reset.
    def episode_spawn_config(base, episode, prev_motion_ticks):
        # previous episode burned a VARIABLE number of motion draws ...
        prev_np, _ = seed_utils.make_substream_rngs(base, episode - 1)
        for _ in range(prev_motion_ticks):
            prev_np.uniform()
        # ... then THIS episode reseeds fresh and samples its spawn config.
        np_rng, _ = seed_utils.make_substream_rngs(base, episode)
        return [np_rng.uniform() for _ in range(6)]

    a = episode_spawn_config(42, 5, prev_motion_ticks=10)
    b = episode_spawn_config(42, 5, prev_motion_ticks=9999)
    assert a == b                   # independent of previous-episode tick count
