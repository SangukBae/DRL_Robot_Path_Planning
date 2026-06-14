"""Structured map curriculum — static-obstacle catalog policy (pure data).

Extracted from ``environment.py`` so the per-map-type static-obstacle key sets,
the global ban list, the map-type tuple and the radius→size-group helper live in
one neutral, ROS-free module. Both ``environment.py`` and the extracted
environment mixins (obstacle spawner, map-layout runtime) import these names from
here, which keeps the mixins free of any back-import to ``environment.py`` (no
import cycle).

Source of truth: docs/map_curriculum_plan.md §4.

Two-stage filter, applied per episode:
  1. globally banned keys are never spawned/activated anywhere
  2. per-map-type allowed keys decide what a given map_type may activate
The stage's ``allowed_static_groups`` is an ORTHOGONAL size filter (small/medium/
large) that further narrows the per-episode candidates.
"""

# §4.3 — sensor-unfriendly / shape-mismatched obstacles excluded everywhere.
STATIC_GLOBALLY_BANNED_KEYS = frozenset({
    "house_cooking_bench",   # ~0.10 m tall → disappears under the LiDAR height filter
})

# §4.2 corridor — small/medium that do not block a single travel lane.
_CORRIDOR_ALLOWED_KEYS = frozenset({
    "warehouse_bucket", "warehouse_trash_can", "hospital_chair", "bookstore_chair",
    "house_chair_a", "bookstore_column_a", "hospital_bedside_table", "hospital_drawer",
    "hospital_instrument_cart1", "hospital_mop_cart", "hospital_surgical_trolley",
    "warehouse_cluttering_c", "warehouse_cluttering_d", "house_kitchen_table",
    "house_refrigerator",
})

# §4.2 intersection — corridor set + a few larger items (edge-anchor in spirit;
# v1 places them in the free regions, large-first, so they settle on arm edges).
_INTERSECTION_ALLOWED_KEYS = _CORRIDOR_ALLOWED_KEYS | frozenset({
    "hospital_metal_cabinet", "hospital_vending_machine", "hospital_parking_trolley_max",
    "hospital_wheelchair", "warehouse_cluttering_a", "house_fitness_equipment",
    "bookstore_desk_a", "bookstore_shelf_a", "bookstore_shelf_b", "bookstore_shelf_c",
    "house_kitchen_cabinet", "hospital_xray_machine", "warehouse_shelf", "warehouse_shelf_e",
})

# §4.2 clutter — complexity is the point; everything except desk_b and the banned.
_CLUTTER_ALLOWED_KEYS = frozenset({
    "warehouse_bucket", "warehouse_cluttering_a", "warehouse_cluttering_c",
    "warehouse_cluttering_d", "warehouse_trash_can", "warehouse_shelf", "warehouse_shelf_e",
    "hospital_instrument_cart1", "hospital_drawer", "hospital_bedside_table",
    "hospital_metal_cabinet", "hospital_vending_machine", "hospital_chair",
    "hospital_mop_cart", "hospital_parking_trolley_max", "hospital_surgical_trolley",
    "hospital_wheelchair", "hospital_trolley_bed", "hospital_xray_machine",
    "bookstore_desk_a", "bookstore_shelf_a", "bookstore_shelf_b", "bookstore_shelf_c",
    "bookstore_info_desk", "bookstore_chair", "bookstore_column_a", "house_kitchen_table",
    "house_bed", "house_kitchen_cabinet", "house_refrigerator", "house_wardrobe",
    "house_chair_a", "house_fitness_equipment", "house_sofa",
})

# §4.2 lobby — corridor set + perimeter-friendly large; desk_b only here (lobby-
# only per §4.3). Open-space character preserved by keeping large items out of the
# central open area at placement time.
_LOBBY_ALLOWED_KEYS = _CORRIDOR_ALLOWED_KEYS | frozenset({
    "hospital_metal_cabinet", "hospital_vending_machine", "hospital_parking_trolley_max",
    "hospital_xray_machine", "bookstore_desk_a", "bookstore_shelf_a", "bookstore_shelf_b",
    "bookstore_shelf_c", "bookstore_info_desk", "house_kitchen_cabinet", "house_wardrobe",
    "house_sofa", "warehouse_shelf", "warehouse_shelf_e",
    "hospital_trolley_bed", "house_bed", "bookstore_desk_b",
})

MAP_TYPE_ALLOWED_STATIC_KEYS = {
    "corridor":     _CORRIDOR_ALLOWED_KEYS,
    "intersection": _INTERSECTION_ALLOWED_KEYS,
    "clutter":      _CLUTTER_ALLOWED_KEYS,
    "lobby":        _LOBBY_ALLOWED_KEYS,
}

MAP_TYPES = ("lobby", "corridor", "intersection", "clutter")


def static_size_group(radius: float) -> str:
    """Map a catalog radius to a size group (docs/map_curriculum_plan.md §4.1)."""
    if radius <= 0.40:
        return "small"
    if radius <= 0.65:
        return "medium"
    return "large"
