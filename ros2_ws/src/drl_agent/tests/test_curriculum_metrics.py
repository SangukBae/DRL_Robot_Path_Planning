"""ROS-free unit tests for curriculum_metrics._LabelProximity."""

import math

from drl_agent.training.curriculum.metrics import _LabelProximity, LabelProximity


def test_alias_is_same_class():
    assert LabelProximity is _LabelProximity


def test_no_labels_reports_none():
    p = _LabelProximity(personal_space_m=0.5, h_coll_radius_m=0.3)
    assert p.available is False
    assert p.psc() is None
    assert p.h_coll(collision=True) is None


def test_non_finite_distances_ignored():
    p = _LabelProximity(0.5, 0.3)
    p.add_dist(None)
    p.add_dist(float("inf"))
    p.add_dist(float("nan"))
    assert p.available is False


def test_psc_counts_personal_space_compliance():
    p = _LabelProximity(personal_space_m=0.5, h_coll_radius_m=0.3)
    # 4 steps: two intrude (<0.5), two respect.
    for d in [0.6, 0.4, 0.7, 0.2]:
        p.add_dist(d)
    assert p.available is True
    assert p.psc() == 0.5  # 1 - 2/4


def test_h_coll_requires_collision_and_proximity():
    p = _LabelProximity(personal_space_m=0.5, h_coll_radius_m=0.3)
    for d in [0.6, 0.25]:  # min 0.25 < h_coll_radius
        p.add_dist(d)
    assert p.h_coll(collision=True) == 1
    assert p.h_coll(collision=False) == 0
    # Far-only episode → not a human collision even on collision.
    p2 = _LabelProximity(0.5, 0.3)
    p2.add_dist(0.9)
    assert p2.h_coll(collision=True) == 0


def test_reset_clears_state():
    p = _LabelProximity(0.5, 0.3)
    p.add_dist(0.1)
    p.reset()
    assert p.available is False
    assert math.isinf(p.min_m)
