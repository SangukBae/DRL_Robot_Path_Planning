"""ROS-free unit tests for PHASE2 candidate 1: Directional Risk-map Reward
Shaping (reward_calculator.compute_reward's risk_map_reward_* params) and the
shared sector_index_for_theta convention it depends on.

Covers:
  * sector_index_for_theta: theta=0 -> middle bin, wraps correctly, matches the
    convention documented in risk_map_dump.py (kept as the canonical owner).
  * risk_map_reward disabled (default) -> reward byte-identical to before this
    feature (existing baseline behaviour untouched).
  * the anti-reward-hacking progress-positive gate: BOTH the penalty and the
    bonus are exactly 0 unless delta_d > progress_positive_gate_eps.
  * penalty / bonus math when the gate is open, including the bonus_max clamp
    and the prev_risk_dir=None (episode-start) case.
"""

import math

import aux_prediction_labels as al
import reward_calculator as rc


# --------------------------------------------------------------------------- #
#  sector_index_for_theta
# --------------------------------------------------------------------------- #
def test_sector_index_theta_zero_is_middle_bin():
    assert al.sector_index_for_theta(0.0, 16) == 8


def test_sector_index_wraps_and_clamps():
    K = 16
    # +-pi should land at (or adjacent to, due to float wrap) bin 0 / K-1.
    idx_pos = al.sector_index_for_theta(math.pi - 1e-6, K)
    idx_neg = al.sector_index_for_theta(-math.pi, K)
    assert 0 <= idx_pos < K
    assert 0 <= idx_neg < K
    # Well outside [-pi, pi] still wraps into range instead of raising / going OOB.
    assert 0 <= al.sector_index_for_theta(3 * math.pi, K) < K


def test_sector_index_zero_sectors_is_safe():
    assert al.sector_index_for_theta(0.3, 0) == 0


# --------------------------------------------------------------------------- #
#  compute_reward: disabled -> byte-identical baseline
# --------------------------------------------------------------------------- #
def _base_kwargs(**over):
    kw = dict(
        target=False, collision=False,
        v=0.5, w=0.0,
        prev_goal_dist=5.0, curr_goal_dist=4.8,   # delta_d = 0.2 (positive progress)
        theta_err=0.0,
        return_terms=True,
    )
    kw.update(over)
    return kw


def test_risk_map_reward_disabled_matches_baseline():
    reward_off, terms_off = rc.compute_reward(**_base_kwargs())
    reward_on_but_disabled, terms_on_but_disabled = rc.compute_reward(
        **_base_kwargs(risk_map_reward_enabled=False, risk_dir=0.9,
                       prev_risk_dir=0.5))
    assert reward_off == reward_on_but_disabled
    assert terms_off["risk_map_penalty"] == 0.0
    assert terms_off["risk_map_bonus"] == 0.0
    assert terms_on_but_disabled["risk_map_penalty"] == 0.0
    assert terms_on_but_disabled["risk_map_bonus"] == 0.0


# --------------------------------------------------------------------------- #
#  progress_positive_gate: the anti-reward-hacking guard
# --------------------------------------------------------------------------- #
def test_gate_closed_zeroes_both_penalty_and_bonus():
    # prev_goal_dist == curr_goal_dist -> delta_d == 0 -> gate closed (0 > 0 is False).
    _, terms = rc.compute_reward(**_base_kwargs(
        prev_goal_dist=5.0, curr_goal_dist=5.0,
        risk_map_reward_enabled=True, risk_dir=0.9, prev_risk_dir=0.5,
        risk_penalty_weight=0.3, risk_bonus_weight=0.15,
    ))
    assert terms["risk_map_penalty"] == 0.0
    assert terms["risk_map_bonus"] == 0.0


def test_gate_closed_when_regressing():
    # Negative progress (moving away from goal) must not trigger shaping either.
    _, terms = rc.compute_reward(**_base_kwargs(
        prev_goal_dist=5.0, curr_goal_dist=5.2,
        risk_map_reward_enabled=True, risk_dir=0.9, prev_risk_dir=0.5,
    ))
    assert terms["risk_map_penalty"] == 0.0
    assert terms["risk_map_bonus"] == 0.0


def test_gate_open_applies_penalty_and_bonus():
    _, terms = rc.compute_reward(**_base_kwargs(
        risk_map_reward_enabled=True, risk_dir=0.4, prev_risk_dir=0.6,
        risk_penalty_weight=0.3, risk_bonus_weight=0.15, risk_bonus_max=0.1,
    ))
    assert abs(terms["risk_map_penalty"] - 0.3 * 0.4) < 1e-9
    # bonus = min(bonus_max, bonus_weight * max(0, prev - cur)) = min(0.1, 0.15*0.2)=0.03
    assert abs(terms["risk_map_bonus"] - 0.15 * 0.2) < 1e-9


def test_bonus_is_clamped_to_bonus_max():
    _, terms = rc.compute_reward(**_base_kwargs(
        risk_map_reward_enabled=True, risk_dir=0.0, prev_risk_dir=1.0,
        risk_penalty_weight=0.0, risk_bonus_weight=1.0, risk_bonus_max=0.1,
    ))
    assert terms["risk_map_bonus"] == 0.1  # would be 1.0*1.0 uncapped


def test_no_bonus_on_episode_start_prev_risk_dir_none():
    _, terms = rc.compute_reward(**_base_kwargs(
        risk_map_reward_enabled=True, risk_dir=0.2, prev_risk_dir=None,
        risk_penalty_weight=0.3, risk_bonus_weight=0.15,
    ))
    assert terms["risk_map_bonus"] == 0.0
    assert abs(terms["risk_map_penalty"] - 0.3 * 0.2) < 1e-9


def test_no_shaping_when_risk_dir_unknown():
    # risk_dir=None (feature computed nothing, e.g. env-side switch mismatch)
    # must not crash and must not apply any shaping.
    _, terms = rc.compute_reward(**_base_kwargs(
        risk_map_reward_enabled=True, risk_dir=None, prev_risk_dir=0.5,
    ))
    assert terms["risk_map_penalty"] == 0.0
    assert terms["risk_map_bonus"] == 0.0


def test_terminal_rewards_short_circuit_risk_shaping():
    reward, terms = rc.compute_reward(**_base_kwargs(
        target=True, risk_map_reward_enabled=True, risk_dir=0.9,
    ))
    assert reward == 20.0
    assert terms["risk_map_penalty"] == 0.0
    assert terms["risk_map_bonus"] == 0.0
