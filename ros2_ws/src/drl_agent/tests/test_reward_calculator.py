"""ROS-free unit + regression tests for reward_calculator.compute_reward.

These pin the reward shaping extracted from Environment.get_reward so future
edits cannot silently change the training signal.
"""

import math

import numpy as np
import pytest

import reward_calculator as rc


def test_terminal_goal_short_circuits():
    r, terms = rc.compute_reward(
        target=True, collision=False, v=1.0, w=0.0,
        prev_goal_dist=5.0, curr_goal_dist=4.0, return_terms=True,
    )
    assert r == 20.0
    assert terms["terminal"] == 20.0
    # Nothing else should have been accumulated on a terminal step.
    assert terms["progress"] == 0.0 and terms["obstacle"] == 0.0


def test_terminal_collision_short_circuits():
    r = rc.compute_reward(
        target=False, collision=True, v=0.5, w=0.1,
        prev_goal_dist=2.0, curr_goal_dist=2.0,
    )
    assert r == -30.0


def test_collision_takes_precedence_only_after_goal():
    # target wins when both flags set (goal checked first).
    r = rc.compute_reward(
        target=True, collision=True, v=0.0, w=0.0,
        prev_goal_dist=1.0, curr_goal_dist=1.0,
    )
    assert r == 20.0


def test_progress_term_clipped_and_scaled():
    # delta_d = prev - curr = 1.0, clipped to progress_clip=0.25, * k_p=0.5
    r, terms = rc.compute_reward(
        target=False, collision=False, v=1.0, w=0.0,
        prev_goal_dist=5.0, curr_goal_dist=4.0,
        k_p=0.5, progress_clip=0.25, step_pen=0.0, return_terms=True,
    )
    assert terms["delta_d"] == pytest.approx(0.25)
    assert terms["progress"] == pytest.approx(0.125)


def test_heading_bonus_only_when_progressing():
    # Moving away from the goal (delta_d <= 0) → no heading bonus.
    _, terms_away = rc.compute_reward(
        target=False, collision=False, v=1.0, w=0.0,
        prev_goal_dist=4.0, curr_goal_dist=5.0,
        theta_err=0.0, k_h=0.03, step_pen=0.0, return_terms=True,
    )
    assert terms_away["heading"] == 0.0
    # Progressing with perfect heading → full k_h.
    _, terms_to = rc.compute_reward(
        target=False, collision=False, v=1.0, w=0.0,
        prev_goal_dist=5.0, curr_goal_dist=4.0,
        theta_err=0.0, k_h=0.03, step_pen=0.0, return_terms=True,
    )
    assert terms_to["heading"] == pytest.approx(0.03)


def test_rect_proximity_obstacle_penalty():
    _, terms = rc.compute_reward(
        target=False, collision=False, v=0.0, w=0.0,
        prev_goal_dist=1.0, curr_goal_dist=1.0,
        rect_proximity=0.5, w_obs=1.5, step_pen=0.0, return_terms=True,
    )
    assert terms["obstacle"] == pytest.approx(0.75)


def test_fallback_min_laser_penalty_when_no_rect():
    # No rect_proximity / zones → speed-dependent safe distance fallback.
    v = 0.5
    min_laser = 0.4
    d_safe = 0.55 + 0.30 * abs(v)  # = 0.70
    expected_obs = 1.5 * (1.0 - min_laser / d_safe)
    _, terms = rc.compute_reward(
        target=False, collision=False, v=v, w=0.0,
        prev_goal_dist=1.0, curr_goal_dist=1.0,
        min_laser=min_laser, w_obs=1.5, step_pen=0.0, return_terms=True,
    )
    assert terms["obstacle"] == pytest.approx(expected_obs)


def test_full_sum_regression():
    """Hand-computed reference for a representative non-terminal step."""
    r, terms = rc.compute_reward(
        target=False, collision=False,
        v=1.0, w=0.0,
        prev_goal_dist=5.0, curr_goal_dist=4.9,    # delta_d = 0.1
        theta_err=0.0,
        rect_proximity=0.2,
        v_max=1.5, w_max=6.0,
        k_p=0.5, progress_clip=0.25,
        w_obs=1.5, k_h=0.03, step_pen=0.05,
        waypoint_theta=0.0, prev_waypoint_theta=0.0, k_smooth_wp=0.05,
        return_terms=True,
    )
    progress = 0.5 * 0.1
    heading = 0.03 * math.cos(0.0)
    obstacle = 1.5 * 0.2
    expected = progress + heading - 0.0 - obstacle - 0.05 - 0.0 - 0.0
    assert r == pytest.approx(expected)
    # The returned reward must equal the sum of its decomposed terms.
    recomputed = (terms["progress"] + terms["heading"] - terms["curv_pen"]
                  - terms["obstacle"] - terms["step_pen"] - terms["smooth"]
                  - terms["wp_smooth"])
    assert r == pytest.approx(recomputed)


def test_waypoint_smoothing_wraps_angle():
    # A near-±pi swing should wrap to a small effective dtheta.
    _, terms = rc.compute_reward(
        target=False, collision=False, v=1.0, w=0.0,
        prev_goal_dist=1.0, curr_goal_dist=1.0,
        waypoint_theta=math.pi - 0.1, prev_waypoint_theta=-math.pi + 0.1,
        k_smooth_wp=0.05, step_pen=0.0, return_terms=True,
    )
    # wrapped dtheta = 0.2, penalty = 0.05 * 0.2/pi
    assert terms["wp_smooth"] == pytest.approx(0.05 * 0.2 / math.pi)
