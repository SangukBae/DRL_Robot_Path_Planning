"""Static map-layout geometry registry (pure, ROS-free).

Extracted from ``Environment._build_map_layouts``. Given the per-map-type
geometry scalars (arena bounds, wall thickness, lane widths, …) it returns the
precomputed wall boxes + sampling regions for each structured map_type. This is
pure data construction with no ROS / node state, so it is unit-testable and
shared by ``map_layout_runtime.MapLayoutMixin`` (which calls it with the node's
``map_*`` config) — the runtime activation of these layouts lives there.
"""


def structured_lane_footprint_fits(
    *,
    radius: float,
    lane_width: float,
    passage_width: float,
    wall_clearance: float,
    passage_safety_margin: float,
) -> bool:
    """Return whether a circular footprint can exist OFF the reserved passage.

    Corridor / intersection maps keep a central reserved passage clear while
    also requiring every sampled pose to stay off the side walls by
    ``wall_clearance``. When the lane is too narrow for a given obstacle radius,
    no legal centre point exists and the obstacle would only generate
    ``no free pose`` warnings at runtime.
    """
    half = lane_width / 2.0
    passage_half = passage_width / 2.0
    max_cross = half - (wall_clearance + radius)
    min_cross = passage_half + radius + passage_safety_margin
    return max_cross > min_cross


def build_map_layouts(
    *,
    map_inner_lower,
    map_inner_upper,
    map_wall_thickness,
    map_corridor_width,
    map_corridor_passage_width,
    map_intersection_width,
    map_intersection_passage_width,
    map_lobby_open_half_extent,
    map_start_band_depth,
) -> dict:
    """Precompute the wall boxes + sampling regions for each map_type.

    Walls are axis-aligned boxes {cx, cy, sx, sy}. free_regions /
    human_regions are (x_lo, x_hi, y_lo, y_hi) rectangles. open_area (lobby)
    is a central rectangle kept clear of LARGE obstacles. Wall entity names
    are assigned here and spawned once by _initialize_obstacle_pool().
    """
    lo, hi = map_inner_lower, map_inner_upper
    span = hi - lo
    t = map_wall_thickness
    layouts = {}

    # lobby — open space, no internal walls; large items avoid the centre.
    ohe = map_lobby_open_half_extent
    layouts["lobby"] = {
        "walls": [],
        "free_regions": [(lo, hi, lo, hi)],
        "human_regions": [(lo, hi, lo, hi)],
        "open_area": (-ohe, ohe, -ohe, ohe),
    }

    # corridor — one horizontal travel lane bounded by two long walls.
    half = map_corridor_width / 2.0
    band = min(map_start_band_depth, 0.45 * span)  # end-band depth
    c_ph = map_corridor_passage_width / 2.0        # reserved-strip half width
    left_band  = (lo, lo + band, -half, half)
    right_band = (hi - band, hi, -half, half)
    layouts["corridor"] = {
        "walls": [
            {"cx": 0.0, "cy":  (half + t / 2.0), "sx": span, "sy": t},
            {"cx": 0.0, "cy": -(half + t / 2.0), "sx": span, "sy": t},
        ],
        "free_regions": [(lo, hi, -half, half)],
        "human_regions": [(lo, hi, -half, half)],
        "open_area": None,
        # Structured spawn metadata: start at either end band, head inward.
        "start_regions": [
            {"name": "left",  "rect": left_band,  "axis": "x",
             "dir": 1.0,  "lane_half": half},
            {"name": "right", "rect": right_band, "axis": "x",
             "dir": -1.0, "lane_half": half},
        ],
        "goal_regions": {"left": [right_band], "right": [left_band]},
        # Central along-lane strip kept clear of static obstacles.
        "reserved_passages": [
            {"axis": "x", "y_center": 0.0, "half_width": c_ph},
        ],
    }

    # intersection — plus-shaped lanes; four corner blocks wall off the rest.
    half = map_intersection_width / 2.0
    blk = hi - half          # corner block side length
    cmid = (half + hi) / 2.0  # corner block centre offset
    band = min(map_start_band_depth, 0.9 * (hi - half))
    i_ph = map_intersection_passage_width / 2.0
    arm_l = (lo, lo + band, -half, half)
    arm_r = (hi - band, hi, -half, half)
    arm_b = (-half, half, lo, lo + band)
    arm_t = (-half, half, hi - band, hi)
    layouts["intersection"] = {
        "walls": [
            {"cx":  cmid, "cy":  cmid, "sx": blk, "sy": blk},
            {"cx": -cmid, "cy":  cmid, "sx": blk, "sy": blk},
            {"cx":  cmid, "cy": -cmid, "sx": blk, "sy": blk},
            {"cx": -cmid, "cy": -cmid, "sx": blk, "sy": blk},
        ],
        "free_regions": [(lo, hi, -half, half), (-half, half, lo, hi)],
        "human_regions": [
            (half, hi, -half, half), (lo, -half, -half, half),
            (-half, half, half, hi), (-half, half, lo, -half),
        ],
        "open_area": None,
        # Start at any of the 4 arm-end bands, head toward the centre.
        "start_regions": [
            {"name": "left",   "rect": arm_l, "axis": "x",
             "dir": 1.0,  "lane_half": half},
            {"name": "right",  "rect": arm_r, "axis": "x",
             "dir": -1.0, "lane_half": half},
            {"name": "bottom", "rect": arm_b, "axis": "y",
             "dir": 1.0,  "lane_half": half},
            {"name": "top",    "rect": arm_t, "axis": "y",
             "dir": -1.0, "lane_half": half},
        ],
        # Goal lives in a DIFFERENT arm than the start arm.
        "goal_regions": {
            "left":   [arm_r, arm_t, arm_b],
            "right":  [arm_l, arm_t, arm_b],
            "bottom": [arm_t, arm_l, arm_r],
            "top":    [arm_b, arm_l, arm_r],
        },
        # Cross-shaped central lanes kept clear of static obstacles.
        "reserved_passages": [
            {"axis": "x", "y_center": 0.0, "half_width": i_ph},
            {"axis": "y", "x_center": 0.0, "half_width": i_ph},
        ],
    }

    # clutter — a few short internal walls (wide gaps keep connectivity).
    layouts["clutter"] = {
        "walls": [
            {"cx": -0.28 * span, "cy":  0.18 * span, "sx": t,            "sy": 0.30 * span},
            {"cx":  0.26 * span, "cy": -0.10 * span, "sx": 0.30 * span,  "sy": t},
            {"cx":  0.10 * span, "cy":  0.30 * span, "sx": 0.26 * span,  "sy": t},
            {"cx": -0.10 * span, "cy": -0.30 * span, "sx": 0.26 * span,  "sy": t},
        ],
        "free_regions": [(lo, hi, lo, hi)],
        "human_regions": [(lo, hi, lo, hi)],
        "open_area": None,
    }

    for mt, spec in layouts.items():
        spec["wall_names"] = [f"rl_wall_{mt}_{i:02d}" for i in range(len(spec["walls"]))]
    return layouts
