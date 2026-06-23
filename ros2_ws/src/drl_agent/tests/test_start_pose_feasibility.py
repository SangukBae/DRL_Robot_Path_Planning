"""Tests for the mode-aware obstacle handling in start/goal sampling.

Background: ``_sample_train_start_pose`` (start_sampler.py) and ``change_goal``
(environment.py) used to reject candidates against ``spawned_obstacle_records``.
In POOL mode those records hold the PREVIOUS episode's placements at sampling
time (the pool teleports obstacles AFTER start/goal are chosen), so the check was
a STALE constraint that shrank the structured start band and forced the "heading
checks exhausted" / "deterministic start-region centre" fallbacks, collapsing
start-pose diversity.

The fix makes ``lingering`` mode-aware: empty in pool mode (records are stale
about-to-be-teleported placements — obstacles are re-placed keeping
obstacle_robot_margin clear of the chosen start), and the genuinely-present
failed-delete leftovers in non-pool mode.

These tests run the REAL ``StartSamplerMixin._sample_train_start_pose`` against
the REAL layout geometry. The environment modules import ROS message packages
that are unavailable in this ROS-free suite, so the ROS deps are stubbed in
``sys.modules`` before import (the sampler itself touches none of them on the
sampling path — only the imported names matter). Two further tests guard the
config invariants that make the fix correct.
"""

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml


# ── Stub the ROS message/param packages so start_sampler + map_layout_runtime
#    import cleanly. PEP 562 module __getattr__ returns a fresh dummy class for
#    any `from X import Y`; the dummy's metaclass auto-supplies any CLASS attribute
#    (e.g. PointField.INT8 used as a dict key by point_cloud2) as a unique int.
#    The sampling path instantiates none of these.
class _AutoAttrMeta(type):
    _n = [0]

    def __getattr__(cls, name):
        _AutoAttrMeta._n[0] += 1
        return _AutoAttrMeta._n[0]


def _dummy_class(name):
    return _AutoAttrMeta(name, (), {})


def _install_ros_stubs():
    for name in (
        "squaternion", "rclpy", "rclpy.parameter",
        "geometry_msgs", "geometry_msgs.msg",
        "nav_msgs", "nav_msgs.msg",
        "sensor_msgs", "sensor_msgs.msg",
        "visualization_msgs", "visualization_msgs.msg",
        "drl_agent_interfaces", "drl_agent_interfaces.msg",
        "ros_gz_interfaces", "ros_gz_interfaces.msg", "ros_gz_interfaces.srv",
    ):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__getattr__ = _dummy_class
        sys.modules[name] = mod


_install_ros_stubs()

import map_layout_registry as reg          # noqa: E402  (pure, ROS-free)
import start_sampler                        # noqa: E402  (ROS deps now stubbed)
import map_layout_runtime                   # noqa: E402  (helper mixin)
import gazebo_entity_manager as gem         # noqa: E402  (record reconciliation)


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "config" / "environment_curriculum.yaml"


def _load_cfg():
    with ENV_CFG.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _robot_collision_radius(sr: dict) -> float:
    """Mirror map_layout_runtime._robot_collision_radius."""
    return max(
        float(sr["d_front"]) + float(sr["margin_front"]),
        float(sr["d_rear"]) + float(sr["margin_rear"]),
        float(sr["d_left"]) + float(sr["margin_left"]),
        float(sr["d_right"]) + float(sr["margin_right"]),
    )


def _navigable_extent(env: dict):
    """Mirror map_layout_runtime._build_map_layouts: the NAVIGABLE extent the
    structured regions are built within (map_inner shrunk by the outer-wall margin)."""
    outer_margin = 0.5 * float(env["map_wall_thickness"]) + float(env["map_wall_clearance"])
    return (float(env["map_inner_lower"]) + outer_margin,
            float(env["map_inner_upper"]) - outer_margin)


def _build_layouts(env: dict) -> dict:
    nav_lower, nav_upper = _navigable_extent(env)
    return reg.build_map_layouts(
        map_inner_lower=nav_lower,
        map_inner_upper=nav_upper,
        map_wall_thickness=float(env["map_wall_thickness"]),
        map_corridor_width=float(env["map_corridor_width"]),
        map_corridor_passage_width=float(env["map_corridor_passage_width"]),
        map_intersection_width=float(env["map_intersection_width"]),
        map_intersection_passage_width=float(env["map_intersection_passage_width"]),
        map_lobby_open_half_extent=float(env.get("map_lobby_open_half_extent", 4.0)),
        map_start_band_depth=float(env["map_start_band_depth"]),
    )


class _Logger:
    def __init__(self):
        self.warns = []

    def warn(self, msg):
        self.warns.append(str(msg))

    def info(self, msg):
        pass


class _SamplerNode(start_sampler.StartSamplerMixin, map_layout_runtime.MapLayoutMixin):
    """Minimal node exposing exactly what _sample_train_start_pose touches."""

    def __init__(self, layouts, map_type, records, use_pool):
        cfg = _load_cfg()
        env = cfg["environment"]
        sr = cfg["safety_region"]
        self.sr_d_front = float(sr["d_front"]); self.sr_margin_front = float(sr["margin_front"])
        self.sr_d_rear = float(sr["d_rear"]);   self.sr_margin_rear = float(sr["margin_rear"])
        self.sr_d_left = float(sr["d_left"]);   self.sr_margin_left = float(sr["margin_left"])
        self.sr_d_right = float(sr["d_right"]); self.sr_margin_right = float(sr["margin_right"])
        self.map_wall_clearance = float(env["map_wall_clearance"])
        self.map_start_side_margin = float(env["map_start_side_margin"])
        self.map_passage_safety_margin = float(env.get("map_passage_safety_margin", 0.25))
        self.map_spawn_yaw_center_bias = math.radians(float(env["map_spawn_yaw_center_bias_deg"]))
        self.map_spawn_yaw_jitter = math.radians(float(env["map_spawn_yaw_jitter_deg"]))
        self._arena_wall_lower = float(env["goal_obstacle_lower"]) - float(env["obstacle_wall_margin"])
        self._arena_wall_upper = float(env["goal_obstacle_upper"]) + float(env["obstacle_wall_margin"])
        self.start_edge_heading_margin = float(env["start_edge_heading_margin_m"])
        self.start_front_clearance = float(env["start_front_clearance_m"])
        self.start_front_fov_deg = float(env["start_front_fov_deg"])
        # Mirror the REAL environment: self.lower/upper come from the legacy start
        # box (env["lower"]/["upper"] = ±9.0), NOT map_inner. The previous test set
        # them to map_inner and stubbed check_dead_zone to False, which masked the
        # bounds-mismatch bug entirely.
        self.lower = float(env["lower"])
        self.upper = float(env["upper"])
        self.goal_obstacle_lower = float(env["goal_obstacle_lower"])
        self.goal_obstacle_upper = float(env["goal_obstacle_upper"])
        # Navigable extent = the exact box the structured bands are built within.
        # Mirrors map_layout_runtime._build_map_layouts.
        outer_margin = 0.5 * float(env["map_wall_thickness"]) + float(env["map_wall_clearance"])
        self.map_navigable_lower = float(env["map_inner_lower"]) + outer_margin
        self.map_navigable_upper = float(env["map_inner_upper"]) - outer_margin
        self.current_map_type = map_type
        self.current_layout_spec = layouts[map_type]
        self.spawned_obstacle_records = records
        self.use_obstacle_pool = use_pool
        self._logger = _Logger()
        self._dead_zone_bounds_seen = []

    def get_logger(self):
        return self._logger

    def check_dead_zone(self, x, y, use_cross_mask=False,
                        lower_bound=None, upper_bound=None):
        """Faithful mirror of Environment.check_dead_zone (use_cross_mask=False
        path): an "outside the map bounds" test. Records the bounds it was called
        with so contract tests can assert the sampler passes the navigable extent
        (not the legacy ±9.0 box)."""
        self._dead_zone_bounds_seen.append((lower_bound, upper_bound))
        lo = self.lower if lower_bound is None else lower_bound
        hi = self.upper if upper_bound is None else upper_bound
        if x < lo or x > hi or y < lo or y > hi:
            return True
        return False   # structured maps have no cross-mask dead zone inside bands


def _band_rects(spec):
    return [r["rect"] for r in spec["start_regions"]]


def _in_any_band(x, y, spec, tol=1e-6):
    for (x_lo, x_hi, y_lo, y_hi) in _band_rects(spec):
        if x_lo - tol <= x <= x_hi + tol and y_lo - tol <= y <= y_hi + tol:
            return True
    return False


def _blanket_records(spec, robot_radius, step=0.8):
    """Static records densely covering every start band so EVERY candidate in the
    band overlaps one (used to prove the stale constraint forces fallback)."""
    recs = {}
    n = 0
    for (x_lo, x_hi, y_lo, y_hi) in _band_rects(spec):
        xs = np.arange(x_lo, x_hi + step, step)
        ys = np.arange(y_lo, y_hi + step, step)
        for x in xs:
            for y in ys:
                recs[f"blanket_{n}"] = (float(x), float(y), 0.5)
                n += 1
    return recs


# ──────────────────────────────────────────────────────────────────────────
#  Real-sampler behaviour
# ──────────────────────────────────────────────────────────────────────────
def test_pool_mode_ignores_stale_records_no_fallback_and_diverse():
    """Pool mode: even with the previous episode's obstacles blanketing the whole
    start band, the sampler treats them as stale (empty `lingering`) and returns a
    valid in-band start every time — no relaxed/centre fallback, full diversity."""
    env = _load_cfg()["environment"]
    layouts = _build_layouts(env)
    spec = layouts["corridor"]
    robot_radius = _robot_collision_radius(_load_cfg()["safety_region"])
    blanket = _blanket_records(spec, robot_radius)

    node = _SamplerNode(layouts, "corridor", blanket, use_pool=True)
    np.random.seed(1234)
    xs, ys, bands_used = [], [], set()
    for _ in range(300):
        x, y, yaw = node._sample_train_start_pose()
        assert _in_any_band(x, y, spec, tol=0.05), (x, y)
        assert math.isfinite(yaw)
        xs.append(x); ys.append(y)
        bands_used.add(x < 0.0)   # left band (x<0) vs right band (x>0)

    # No fallback was taken (the whole point of ignoring stale records).
    assert getattr(node, "_start_relaxed_fallback_count", 0) == 0
    assert getattr(node, "_start_centre_fallback_count", 0) == 0
    assert node.get_logger().warns == []
    # Diversity preserved: both end bands are used and starts spread along y.
    assert bands_used == {True, False}
    assert (max(ys) - min(ys)) > 1.0


def test_nonpool_blanket_forces_fallback_and_increments_counter():
    """Mechanism + counter proof: when obstacles ARE honoured (non-pool, records =
    genuinely-present leftovers) a band fully blanketed with obstacles exhausts all
    700 tries and takes the deterministic-centre fallback, incrementing the counter
    and returning a band-centre pose. This is exactly the failure the pool fix
    avoids by not honouring stale records."""
    env = _load_cfg()["environment"]
    layouts = _build_layouts(env)
    spec = layouts["corridor"]
    robot_radius = _robot_collision_radius(_load_cfg()["safety_region"])
    blanket = _blanket_records(spec, robot_radius)

    node = _SamplerNode(layouts, "corridor", blanket, use_pool=False)
    np.random.seed(7)
    x, y, yaw = node._sample_train_start_pose()

    assert node._start_relaxed_fallback_count == 1     # relaxed pass attempted
    assert node._start_centre_fallback_count == 1      # then centre fallback
    assert any("deterministic start-region centre" in w for w in node.get_logger().warns)
    assert any("centre fallbacks so far: 1" in w for w in node.get_logger().warns)
    # Centre fallback returns a band-centre pose (on the lane centre line, y == 0).
    assert _in_any_band(x, y, spec, tol=0.05)
    assert abs(y) < 1e-6


def test_nonpool_avoids_genuinely_present_leftover():
    """Non-pool: a real failed-delete leftover IS honoured — every sampled start
    keeps clear of it (footprint margin), confirming the mode-aware branch still
    protects against starting on top of a real obstacle."""
    env = _load_cfg()["environment"]
    layouts = _build_layouts(env)
    spec = layouts["corridor"]
    robot_radius = _robot_collision_radius(_load_cfg()["safety_region"])
    # One small leftover near the left band's lane-centre.
    obs = (-7.0, 0.0, 0.4)
    node = _SamplerNode(layouts, "corridor", {"leftover": obs}, use_pool=False)

    np.random.seed(99)
    for _ in range(200):
        x, y, _yaw = node._sample_train_start_pose()
        assert math.hypot(x - obs[0], y - obs[1]) >= robot_radius + obs[2] - 1e-6


# ──────────────────────────────────────────────────────────────────────────
#  Non-pool record reconciliation (human leftover tracking)
# ──────────────────────────────────────────────────────────────────────────
class _RecNode(gem.GazeboEntityMixin):
    """Minimal node exposing only spawned_obstacle_records for reconciliation."""

    def __init__(self, records):
        self.spawned_obstacle_records = dict(records)


def _human_part_names(prefix):
    return [prefix + suf for suf in gem._HUMAN_PART_SUFFIXES]


def test_human_footprint_retained_when_any_part_delete_fails():
    """The reported gap: p_torso deletes but a proxy leg lingers. The body
    footprint record (keyed by p_torso) MUST survive so non-pool sampling avoids
    the real leftover."""
    pref = "rl_human_0007_001"
    parts = _human_part_names(pref)
    torso_key = pref + "_p_torso"
    node = _RecNode({torso_key: (5.0, 3.0, 0.30)})

    # p_torso (and most parts) deleted, but the right proxy leg delete failed.
    leftover = pref + "_p_rl"
    deleted = [p for p in parts if p != leftover]
    node._reconcile_obstacle_records(deleted_names=deleted, remaining_names=[leftover])

    assert torso_key in node.spawned_obstacle_records          # footprint kept
    assert node.spawned_obstacle_records[torso_key] == (5.0, 3.0, 0.30)


def test_human_footprint_dropped_when_all_parts_deleted():
    pref = "rl_human_0007_002"
    parts = _human_part_names(pref)
    torso_key = pref + "_p_torso"
    node = _RecNode({torso_key: (1.0, 1.0, 0.30)})
    node._reconcile_obstacle_records(deleted_names=parts, remaining_names=[])
    assert torso_key not in node.spawned_obstacle_records       # fully gone -> drop


def test_static_record_dropped_per_name():
    node = _RecNode({"rl_sta_0007_001": (2.0, 2.0, 0.5),
                     "rl_sta_0007_002": (3.0, 3.0, 0.5)})
    node._reconcile_obstacle_records(
        deleted_names=["rl_sta_0007_001"], remaining_names=["rl_sta_0007_002"])
    assert "rl_sta_0007_001" not in node.spawned_obstacle_records
    assert "rl_sta_0007_002" in node.spawned_obstacle_records    # not deleted -> kept


# ──────────────────────────────────────────────────────────────────────────
#  Config invariants that make the fix correct
# ──────────────────────────────────────────────────────────────────────────
def test_obstacle_robot_margin_subsumes_front_clearance():
    """Placed obstacles stay obstacle_robot_margin (+radius) from the start, so
    their footprint edge clears the robot footprint by more than
    start_front_clearance: the dropped front-cone-vs-obstacle check could never
    have rejected a this-episode obstacle, i.e. dropping it loses no safety."""
    cfg = _load_cfg()
    env = cfg["environment"]
    robot_radius = _robot_collision_radius(cfg["safety_region"])
    edge_clearance = float(env["obstacle_robot_margin"]) - robot_radius
    assert edge_clearance >= float(env["start_front_clearance_m"]), (
        f"obstacle_robot_margin ({env['obstacle_robot_margin']}) - robot_radius "
        f"({robot_radius:.3f}) = {edge_clearance:.3f} m must stay >= "
        f"start_front_clearance_m ({env['start_front_clearance_m']})."
    )
    assert float(env["obstacle_goal_margin"]) > robot_radius


def test_structured_start_bands_have_nonempty_sampling_box():
    """Every corridor/intersection start band leaves a usable post-margin box, so
    geometry-only sampling never starts from an already-degenerate band."""
    cfg = _load_cfg()
    env = cfg["environment"]
    robot_radius = _robot_collision_radius(cfg["safety_region"])
    side_margin = float(env["map_start_side_margin"])
    layouts = _build_layouts(env)

    for map_type in ("corridor", "intersection"):
        for region in layouts[map_type]["start_regions"]:
            x_lo, x_hi, y_lo, y_hi = region["rect"]
            mx = my = robot_radius
            if region["axis"] == "x":
                my += side_margin
            else:
                mx += side_margin
            ax_lo, ax_hi, ay_lo, ay_hi = x_lo + mx, x_hi - mx, y_lo + my, y_hi - my
            assert ax_hi > ax_lo and ay_hi > ay_lo, (
                f"{map_type} band {region['name']} empty after margins"
            )
            assert (ax_hi - ax_lo) * (ay_hi - ay_lo) > 0.25


# ──────────────────────────────────────────────────────────────────────────
#  Structured start dead-zone bounds contract (the ±9.0 legacy-box bug)
# ──────────────────────────────────────────────────────────────────────────
def test_structured_start_passes_navigable_bounds_not_legacy_box():
    """Contract: on a structured map the sampler must hand check_dead_zone the
    NAVIGABLE extent — never the legacy ±lower/upper start box (which sits inside
    the end/arm bands). Every recorded dead-zone call uses the navigable bounds, and
    sampled starts reach beyond ±lower/upper (so the legacy box would have rejected
    them), with NO fallback taken."""
    env = _load_cfg()["environment"]
    layouts = _build_layouts(env)
    spec = layouts["corridor"]
    node = _SamplerNode(layouts, "corridor", {}, use_pool=True)

    # Sanity: this config is exactly the case where the legacy box is inside the
    # band (band starts at |x| ~ 9.3 > upper 9.0) so the bug would 100%-clip.
    assert node.map_navigable_upper > node.upper
    band_inner = min(r["rect"][0] for r in spec["start_regions"] if r["name"] == "right")
    assert band_inner > node.upper, (band_inner, node.upper)

    np.random.seed(2024)
    max_abs_x = 0.0
    for _ in range(300):
        x, y, yaw = node._sample_train_start_pose()
        assert _in_any_band(x, y, spec, tol=0.05)
        assert math.isfinite(yaw)
        max_abs_x = max(max_abs_x, abs(x))

    # No fallback path was taken.
    assert getattr(node, "_start_relaxed_fallback_count", 0) == 0
    assert getattr(node, "_start_centre_fallback_count", 0) == 0
    assert node.get_logger().warns == []
    # Starts genuinely live beyond the legacy ±9.0 box (the band the old bound clipped).
    assert max_abs_x > node.upper
    # Every dead-zone call used the navigable extent, not the legacy box / None / goal.
    assert node._dead_zone_bounds_seen
    for lo, hi in node._dead_zone_bounds_seen:
        assert lo == node.map_navigable_lower and hi == node.map_navigable_upper


def test_legacy_box_clips_band_navigable_does_not():
    """Mechanism proof, bound-by-bound, for a far-edge corridor band point: the
    legacy ±lower/upper box AND a default-bound call reject it; the navigable extent
    accepts it. This is the exact rejection that fired every episode pre-fix."""
    env = _load_cfg()["environment"]
    layouts = _build_layouts(env)
    node = _SamplerNode(layouts, "corridor", {}, use_pool=True)

    # A point inside the right band but outside the legacy ±9.0 box.
    px = 0.5 * (node.upper + node.map_navigable_upper)   # e.g. ~10.4, between 9.0 and 11.8
    assert px > node.upper and px < node.map_navigable_upper

    # Legacy default bound (self.lower/upper) → rejected (the bug).
    assert node.check_dead_zone(px, 0.0, use_cross_mask=False) is True
    # Explicit legacy bounds → rejected.
    assert node.check_dead_zone(px, 0.0, use_cross_mask=False,
                                lower_bound=node.lower, upper_bound=node.upper) is True
    # Navigable extent → accepted (the fix).
    assert node.check_dead_zone(px, 0.0, use_cross_mask=False,
                                lower_bound=node.map_navigable_lower,
                                upper_bound=node.map_navigable_upper) is False


def test_navigable_extent_is_band_construction_box():
    """Why navigable, not goal_obstacle: the band rects are built from the navigable
    extent, which here exceeds goal_obstacle_upper — so navigable is the only bound
    guaranteed to contain every band rect. (goal_obstacle only avoids clipping when
    the footprint shrink keeps samples inside it, which is config-contingent.)"""
    env = _load_cfg()["environment"]
    layouts = _build_layouts(env)
    node = _SamplerNode(layouts, "corridor", {}, use_pool=True)

    assert node.map_navigable_upper > node.goal_obstacle_upper
    assert node.map_navigable_lower < node.goal_obstacle_lower
    # Every structured band rect lies within [navigable_lower, navigable_upper].
    for mt in ("corridor", "intersection"):
        for region in layouts[mt]["start_regions"]:
            x_lo, x_hi, y_lo, y_hi = region["rect"]
            assert node.map_navigable_lower - 1e-6 <= x_lo
            assert x_hi <= node.map_navigable_upper + 1e-6
            assert node.map_navigable_lower - 1e-6 <= y_lo
            assert y_hi <= node.map_navigable_upper + 1e-6
