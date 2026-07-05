"""Tests for map-type-aware curriculum active counts.

Two layers:
  1. The pure (ROS-free) helpers in map_catalog — clamp_active_by_map /
     resolve_active_count — which the curriculum subclass and the base env use to
     turn a stage's active_static_by_map / active_humans_by_map into THIS
     episode's num_of_static_obstacles / num_of_humans.
  2. The redesigned 7-stage curriculum config — structure, backward compatibility,
     and that the per-map counts respect pool / candidate ceilings (esp. the
     narrow corridor being given fewer obstacles than the open maps).
"""

from pathlib import Path

import yaml

import map_catalog as mc


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "config" / "environment_curriculum.yaml"
TRAIN_CFG = ROOT / "config" / "train_tqc_curriculum_config.yaml"


def _load(path):
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Pure helpers ───────────────────────────────────────────────────────────
def test_clamp_active_by_map_valid():
    clean, warns = mc.clamp_active_by_map(
        {"corridor": 5, "intersection": 7, "clutter": 8, "lobby": 4}, cap=10)
    assert clean == {"corridor": 5, "intersection": 7, "clutter": 8, "lobby": 4}
    assert warns == []


def test_clamp_active_by_map_clamps_to_cap_and_floor():
    clean, warns = mc.clamp_active_by_map({"corridor": 99, "lobby": -3}, cap=8)
    assert clean == {"corridor": 8, "lobby": 0}
    assert warns == []


def test_clamp_active_by_map_drops_unknown_key_with_warning():
    clean, warns = mc.clamp_active_by_map({"corridor": 5, "atrium": 9}, cap=10)
    assert clean == {"corridor": 5}
    assert any("atrium" in w for w in warns)


def test_clamp_active_by_map_drops_non_int_with_warning():
    clean, warns = mc.clamp_active_by_map({"corridor": "lots"}, cap=10)
    assert clean == {}
    assert any("not an int" in w for w in warns)


def test_clamp_active_by_map_none_and_non_dict():
    assert mc.clamp_active_by_map(None, cap=10) == ({}, [])
    clean, warns = mc.clamp_active_by_map([1, 2, 3], cap=10)
    assert clean == {} and warns


def test_resolve_active_count_prefers_by_map():
    by = {"corridor": 5, "intersection": 7}
    assert mc.resolve_active_count(by, single=6, cap=10, map_type="corridor") == 5
    assert mc.resolve_active_count(by, single=6, cap=10, map_type="intersection") == 7


def test_resolve_active_count_falls_back_to_single():
    by = {"corridor": 5}
    # map not in by_map → single value
    assert mc.resolve_active_count(by, single=6, cap=10, map_type="lobby") == 6
    # empty by_map → single value (backward compatible, single-value stages)
    assert mc.resolve_active_count({}, single=4, cap=10, map_type="corridor") == 4


def test_resolve_active_count_clamps():
    assert mc.resolve_active_count({}, single=99, cap=8, map_type="lobby") == 8
    assert mc.resolve_active_count({"corridor": -2}, single=5, cap=8, map_type="corridor") == 0


def test_resolve_active_count_corridor_lighter_than_intersection():
    """The headline use case: same stage, corridor fewer than intersection."""
    by = {"corridor": 5, "intersection": 7}
    c = mc.resolve_active_count(by, single=6, cap=24, map_type="corridor")
    i = mc.resolve_active_count(by, single=6, cap=24, map_type="intersection")
    assert c < i


# ── Redesigned curriculum config ───────────────────────────────────────────
def test_curriculum_has_ten_stages_and_initial_zero():
    # 10-stage layout: the human (dynamic-avoidance) and observation-noise axes
    # are introduced on SEPARATE stages (3/4 humans→loc-noise, 5/6 clutter→proprio,
    # 7 new-map/scan-noise, 8/9 integration).
    cfg = _load(ENV_CFG)["curriculum"]
    assert cfg["enabled"] is True
    assert len(cfg["stages"]) == 10
    assert cfg["initial_stage"] == 0


def test_promotion_threshold_lists_match_stage_count():
    n = len(_load(ENV_CFG)["curriculum"]["stages"])
    cur = _load(TRAIN_CFG)["curriculum_settings"]
    for key in ("pass_eval_success_rate", "pass_eval_collision_rate", "pass_eval_spl"):
        assert len(cur[key]) == n - 1, (key, len(cur[key]), n - 1)


def test_stage0_and_1_are_single_value_backward_compatible():
    stages = _load(ENV_CFG)["curriculum"]["stages"]
    for i in (0, 1):
        assert "active_static_by_map" not in stages[i]
        assert "active_humans_by_map" not in stages[i]
        assert "active_static" in stages[i] and "active_humans" in stages[i]


def test_by_map_counts_within_pool_and_corridor_is_lightest():
    cfg = _load(ENV_CFG)
    env = cfg["environment"]
    stages = cfg["curriculum"]["stages"]
    static_cap = int(env["obstacle_pool_static_size"])  # fallback; coverage usually grows it
    human_pool = int(env["obstacle_pool_human_size"])
    # corridor activatable static candidate ceiling (only small + medium r<0.5 fit
    # the 5.2 m lane) — keep corridor static <= this.
    corridor_static_ceiling = 10

    for s in stages:
        bm_s = s.get("active_static_by_map", {}) or {}
        bm_h = s.get("active_humans_by_map", {}) or {}
        # every value a valid int within bounds
        for mt, v in bm_h.items():
            assert mt in mc.MAP_TYPES
            assert 0 <= v <= human_pool, (s["name"], "human", mt, v)
        for mt, v in bm_s.items():
            assert mt in mc.MAP_TYPES
            assert 0 <= v, (s["name"], "static", mt, v)
        # corridor must be the lightest map for static whenever it shares a stage
        # with a roomier map (the whole point of per-map counts).
        if "corridor" in bm_s and len(bm_s) > 1:
            others = [v for mt, v in bm_s.items() if mt != "corridor"]
            assert bm_s["corridor"] <= min(others), (s["name"], bm_s)
            assert bm_s["corridor"] <= corridor_static_ceiling, (s["name"], bm_s)


def test_max_active_humans_fits_human_pool():
    cfg = _load(ENV_CFG)
    human_pool = int(cfg["environment"]["obstacle_pool_human_size"])
    mx = 0
    for s in cfg["curriculum"]["stages"]:
        mx = max([mx, int(s.get("active_humans", 0))]
                 + list((s.get("active_humans_by_map", {}) or {}).values()))
    assert mx <= human_pool, (mx, human_pool)


def test_stage_noise_profiles_exist():
    cfg = _load(ENV_CFG)
    loc_profiles = set((cfg.get("localization_profiles") or {}).keys())
    pp_profiles = set((cfg.get("proprio_noise_profiles") or {}).keys())
    for s in cfg["curriculum"]["stages"]:
        if "localization_profile" in s:
            assert s["localization_profile"] in loc_profiles, s["name"]
        if "proprio_noise_profile" in s:
            assert s["proprio_noise_profile"] in pp_profiles, s["name"]
