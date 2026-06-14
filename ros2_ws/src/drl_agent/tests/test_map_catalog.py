"""ROS-free unit tests for map_catalog (static-obstacle catalog policy)."""

import map_catalog as mc


def test_globally_banned_key_present():
    assert "house_cooking_bench" in mc.STATIC_GLOBALLY_BANNED_KEYS


def test_map_types_tuple():
    assert set(mc.MAP_TYPES) == {"lobby", "corridor", "intersection", "clutter"}


def test_allowed_keys_cover_all_map_types():
    for mt in mc.MAP_TYPES:
        assert mt in mc.MAP_TYPE_ALLOWED_STATIC_KEYS
        assert isinstance(mc.MAP_TYPE_ALLOWED_STATIC_KEYS[mt], frozenset)
        assert len(mc.MAP_TYPE_ALLOWED_STATIC_KEYS[mt]) > 0


def test_intersection_superset_of_corridor():
    corridor = mc.MAP_TYPE_ALLOWED_STATIC_KEYS["corridor"]
    intersection = mc.MAP_TYPE_ALLOWED_STATIC_KEYS["intersection"]
    assert corridor <= intersection


def test_static_size_group_thresholds():
    assert mc.static_size_group(0.40) == "small"
    assert mc.static_size_group(0.41) == "medium"
    assert mc.static_size_group(0.65) == "medium"
    assert mc.static_size_group(0.66) == "large"
    assert mc.static_size_group(2.0) == "large"
