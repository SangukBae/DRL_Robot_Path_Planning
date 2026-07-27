"""Goal-arm distribution tests for the structured map curriculum.

Regression target: on the INTERSECTION, goal sampling used to collapse onto the
OPPOSITE arm. goal_sampler phase A required the straight start->goal segment to
stay off internal walls; only the opposite arm has a wall-free straight line, so
the two side arms (reached by routing through the free centre) were always
rejected in phase A and phase B was never reached. The fix skips the lane check
on the intersection so all three non-start arms are sampled ~uniformly, while
corridor keeps opposite-end-only goals.

The environment modules import ROS message packages unavailable here, so the ROS
deps are stubbed in sys.modules before import; the real
``_sample_goal_layout`` / ``_sample_xy_in_regions`` / ``_segment_hits_wall`` /
``_point_in_walls`` are exercised against the real layout geometry.
"""

import math
import sys
import types

import numpy as np


def _install_ros_stubs():
    class _Meta(type):
        _n = [0]

        def __getattr__(cls, name):
            _Meta._n[0] += 1
            return _Meta._n[0]

    def _dummy(name):
        return _Meta(name, (), {})

    for name in (
        "squaternion", "rclpy", "rclpy.parameter",
        "geometry_msgs", "geometry_msgs.msg",
        "nav_msgs", "nav_msgs.msg",
        "sensor_msgs", "sensor_msgs.msg",
        "visualization_msgs", "visualization_msgs.msg",
        "drl_agent_interfaces", "drl_agent_interfaces.msg",
        "ros_gz_interfaces", "ros_gz_interfaces.msg", "ros_gz_interfaces.srv",
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__getattr__ = _dummy
            sys.modules[name] = mod


_install_ros_stubs()

import drl_agent.env.spawning.goal_sampler as gs  # noqa: E402
import drl_agent.env.simulation.map_layout_runtime as mlr  # noqa: E402
import drl_agent.env.simulation.map_layout_registry as reg  # noqa: E402


_LAYOUTS = reg.build_map_layouts(
    map_inner_lower=-11.8, map_inner_upper=11.8, map_wall_thickness=0.30,
    map_corridor_width=5.2, map_corridor_passage_width=1.6,
    map_intersection_width=5.2, map_intersection_passage_width=1.6,
    map_lobby_open_half_extent=6.0, map_start_band_depth=2.5,
)
# Named arm rects (match map_layout_registry intersection construction).
_HALF = 2.6
_BAND = 2.5
_ARMS = {
    "left":   (-11.8, -11.8 + _BAND, -_HALF, _HALF),
    "right":  (11.8 - _BAND, 11.8, -_HALF, _HALF),
    "bottom": (-_HALF, _HALF, -11.8, -11.8 + _BAND),
    "top":    (-_HALF, _HALF, 11.8 - _BAND, 11.8),
}


def _classify_arm(x, y, tol=0.6):
    for name, (xl, xh, yl, yh) in _ARMS.items():
        if xl - tol <= x <= xh + tol and yl - tol <= y <= yh + tol:
            return name
    return "center/other"


class _Logger:
    def warn(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


class _GoalNode(gs.GoalSamplerMixin, mlr.MapLayoutMixin):
    def __init__(self, map_type):
        self.current_map_type = map_type
        self.current_layout_spec = _LAYOUTS[map_type]
        self.map_wall_clearance = 0.55
        self.goal_obstacle_lower = -11.5
        self.goal_obstacle_upper = 11.5
        self._current_start_region = None

    def get_logger(self):
        return _Logger()

    def check_dead_zone(self, x, y, use_cross_mask=False, **kw):
        return False

    def _make_is_valid(self, start_x, start_y, goal_radius):
        """Mirror environment.change_goal._is_valid_goal (lingering empty)."""
        def is_valid(x, y, require_clearance=True):
            if math.hypot(x - start_x, y - start_y) < 3.0:
                return False
            if self._point_in_walls(x, y, self.map_wall_clearance + goal_radius):
                return False
            return True
        return is_valid


# Representative start point at the END of each arm.
_ARM_START = {
    "left":   (-10.5, 0.0),
    "right":  (10.5, 0.0),
    "bottom": (0.0, -10.5),
    "top":    (0.0, 10.5),
}
_OPPOSITE = {"left": "right", "right": "left", "bottom": "top", "top": "bottom"}
_GOAL_RADIUS = 0.42


def _run_distribution(map_type, start_name, n=400, seed=0):
    node = _GoalNode(map_type)
    node._current_start_region = next(
        r for r in node.current_layout_spec["start_regions"] if r["name"] == start_name)
    sx, sy = _ARM_START[start_name]
    is_valid = node._make_is_valid(sx, sy, _GOAL_RADIUS)
    np.random.seed(seed)
    counts = {}
    for _ in range(n):
        gx, gy = node._sample_goal_layout(sx, sy, _GOAL_RADIUS, is_valid, lingering=[])
        # Safety: goal must be a VALID pose (off walls / corner blocks, far enough).
        assert is_valid(gx, gy), (map_type, start_name, gx, gy)
        counts[_classify_arm(gx, gy)] = counts.get(_classify_arm(gx, gy), 0) + 1
    return counts


def test_intersection_goal_spreads_over_all_three_other_arms():
    """For every start arm, each of the 3 other arms is sampled meaningfully and
    the start arm itself is never chosen."""
    for start in ("left", "right", "bottom", "top"):
        counts = _run_distribution("intersection", start, n=400)
        total = sum(counts.values())
        others = [a for a in _ARMS if a != start]
        # start arm never selected
        assert counts.get(start, 0) == 0, (start, counts)
        # no goals leaked outside the arms
        assert counts.get("center/other", 0) == 0, (start, counts)
        # each of the 3 non-start arms gets a meaningful share (>=15%): the
        # opposite-arm bias is gone.
        for arm in others:
            frac = counts.get(arm, 0) / total
            assert frac >= 0.15, (start, arm, frac, counts)
        # opposite arm no longer dominates (was ~100%); cap well below that.
        opp_frac = counts.get(_OPPOSITE[start], 0) / total
        assert opp_frac <= 0.55, (start, opp_frac, counts)


def test_intersection_side_arms_are_actually_reached():
    """Explicit check that the two SIDE arms (the previously-starved ones) appear."""
    counts = _run_distribution("intersection", "left", n=600)
    assert counts.get("top", 0) > 0 and counts.get("bottom", 0) > 0, counts


def _force_phase_d_goal(node, start_name, lingering, seed):
    """Drive _sample_goal_layout down to Phase D (the deterministic grid fallback)
    by making is_valid accept ONLY exact grid points: phases A/B/C draw continuous
    random points (never grid-aligned) and exhaust, so the grid phase decides."""
    node._current_start_region = next(
        r for r in node.current_layout_spec["start_regions"] if r["name"] == start_name)
    sx, sy = _ARM_START[start_name]
    goal_bands = node.current_layout_spec["goal_regions"][start_name]
    grid = node._layout_region_grid(goal_bands, _GOAL_RADIUS)
    gridset = {(round(x, 2), round(y, 2)) for x, y in grid}

    def is_valid(x, y, require_clearance=True):
        return (round(x, 2), round(y, 2)) in gridset

    np.random.seed(seed)
    gx, gy = node._sample_goal_layout(sx, sy, _GOAL_RADIUS, is_valid, lingering)
    assert (round(gx, 2), round(gy, 2)) in gridset      # confirms the grid path
    return gx, gy


def test_phase_d_fallback_not_opposite_biased():
    """Crowded/exhausted episodes fall to Phase D. With random obstacles scattered
    across the three candidate arms, the fallback must pick the least-crowded arm
    (~uniform), NOT keep collapsing onto the opposite arm (old lane_bonus +2.0 plus
    the start-distance term reintroduced that bias)."""
    import random as pyrandom
    node = _GoalNode("intersection")
    start, opp = "left", _OPPOSITE["left"]
    arms = node.current_layout_spec["goal_regions"][start]   # [right(opp), top, bottom]
    rng = pyrandom.Random(42)
    chosen = []
    for i in range(150):
        ling = []
        for _ in range(rng.randint(2, 5)):
            xl, xh, yl, yh = arms[rng.randrange(len(arms))]
            ling.append((rng.uniform(xl, xh), rng.uniform(yl, yh), 0.5))
        gx, gy = _force_phase_d_goal(node, start, ling, seed=i)
        chosen.append(_classify_arm(gx, gy))
    n = len(chosen)
    counts = {a: chosen.count(a) for a in ("right", "top", "bottom")}
    # every non-start arm reached, and the opposite arm does NOT dominate the
    # fallback the way it did with the lane bonus (~uniform → ~1/3 each).
    for arm in ("right", "top", "bottom"):
        assert counts[arm] > 0, counts
    assert counts[opp] / n <= 0.55, counts


def test_corridor_goal_stays_on_opposite_end():
    """Regression: corridor must still send the goal ONLY to the opposite end."""
    for start, opp in (("left", "right"), ("right", "left")):
        node = _GoalNode("corridor")
        node._current_start_region = next(
            r for r in node.current_layout_spec["start_regions"] if r["name"] == start)
        sx, sy = _ARM_START[start]
        is_valid = node._make_is_valid(sx, sy, _GOAL_RADIUS)
        np.random.seed(1)
        for _ in range(200):
            gx, gy = node._sample_goal_layout(sx, sy, _GOAL_RADIUS, is_valid, lingering=[])
            # opposite end band only (x sign flipped from start)
            assert (gx > 0) == (opp == "right"), (start, gx, gy)
