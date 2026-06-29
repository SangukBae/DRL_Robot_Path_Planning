"""ROS-free consistency checks between the curriculum stage definitions
(environment_curriculum.yaml) and the promotion-gate thresholds
(train_tqc_curriculum_config.yaml).

After the 7→9 stage split these two files must stay in lockstep: every per-stage
threshold list that is *configured* (non-empty) must have exactly one entry per
PROMOTABLE stage (num_stages - 1). curriculum_stage_logic clamps an out-of-range
index to the last entry, so a mismatch would not crash — it would silently reuse
the wrong stage's gate. This test pins the intended length so that never happens
unnoticed.
"""

from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parent.parent / "config"


def _load(name):
    with open(_CONFIG / name) as f:
        return yaml.safe_load(f)


def _stages():
    return _load("environment_curriculum.yaml")["curriculum"]["stages"]


def test_curriculum_has_nine_uniquely_named_stages():
    stages = _stages()
    assert len(stages) == 9
    names = [s["name"] for s in stages]
    assert names == [
        "empty", "corridor_static", "add_intersection",
        "first_human_clean", "first_human_noisy",
        "add_clutter_clean", "add_clutter_noisy",
        "generalize", "full_complexity",
    ]
    assert len(set(names)) == len(names)   # no duplicates


def test_threshold_lists_match_promotable_stage_count():
    num_stages = len(_stages())
    expected = num_stages - 1   # one gate entry per promotable transition
    cur = _load("train_tqc_curriculum_config.yaml")["curriculum_settings"]
    gate_keys = [
        "pass_eval_success_rate", "pass_eval_collision_rate", "pass_eval_spl",
        "pass_eval_psc", "pass_eval_h_coll_rate",
        "pass_eval_per_map_success_rate", "pass_eval_per_map_collision_rate",
    ]
    for key in gate_keys:
        lst = cur.get(key, [])
        # Empty list = gate disabled; only enforce length when it is configured.
        if lst:
            assert len(lst) == expected, (
                f"{key} has {len(lst)} entries, expected {expected} "
                f"(num_stages={num_stages})")


def test_axis_separation_human_vs_observation_noise():
    """Stages 3-6 must each change EXACTLY ONE axis vs. the prior stage, so the
    credit-assignment signal for every robustness skill stays clean."""
    by_name = {s["name"]: s for s in _stages()}
    s3 = by_name["first_human_clean"]
    s4 = by_name["first_human_noisy"]
    s5 = by_name["add_clutter_clean"]
    s6 = by_name["add_clutter_noisy"]

    # Stage 3 introduces humans on a CLEAN observation channel.
    assert s3["active_humans"] >= 1
    assert s3.get("localization_profile") == "clean"
    assert "proprio_noise_profile" not in s3

    # Stage 4 = same human distribution/modes as Stage 3, ONLY adds loc noise.
    for k in ("active_humans", "active_static", "active_humans_by_map",
              "active_static_by_map", "human_mode_weights",
              "human_scan_noise_std", "human_scan_dropout_prob"):
        assert s4.get(k) == s3.get(k), k
    assert s4.get("localization_profile") == "weak_goal_noise"
    assert "proprio_noise_profile" not in s4

    # Stage 5 adds clutter ONLY: human count/modes and ALL observation noise are
    # held at the Stage-4 level (terrain is the single new variable). The crowd is
    # NOT scaled here (that is the integration stage 7's job).
    assert "clutter" in s5["allowed_map_types"]
    assert s5["active_humans"] == s4["active_humans"]
    assert s5["human_mode_weights"] == s4["human_mode_weights"]
    assert s5["human_scan_noise_std"] == s4["human_scan_noise_std"]
    assert s5["human_scan_dropout_prob"] == s4["human_scan_dropout_prob"]
    assert s5.get("localization_profile") == s4.get("localization_profile")
    assert "proprio_noise_profile" not in s5

    # Stage 6 = same map/humans/scan/loc as Stage 5, ONLY adds proprio noise
    # (human-scan perception noise stays clean — it is deferred to stage 7).
    for k in ("active_humans", "active_static", "active_humans_by_map",
              "active_static_by_map", "allowed_map_types", "human_mode_weights",
              "human_scan_noise_std", "human_scan_dropout_prob",
              "human_social_avoid_strength", "human_goal_span_multiplier",
              "localization_profile"):
        assert s6.get(k) == s5.get(k), k
    assert s6.get("proprio_noise_profile") == "light"
    assert s6["human_scan_noise_std"] == 0.0   # scan noise NOT turned on yet


# ── TEMPORAL_ACTOR: env transport (obs_frame_stack) vs agent temporal_actor_context
def test_temporal_actor_context_matches_env_transport():
    """The agent's compressed temporal path splits the env-transported stacked
    state, so observation_time_context (env) and temporal_actor_context (agent)
    MUST agree on history_len and stack_agent_state, and the derived stacked
    state_dim must be self-consistent. ROS-free: just the two YAMLs."""
    env = _load("environment_curriculum.yaml")["environment"]
    hp = _load("hyperparameters_tqc.yaml")["hyperparameters"]
    otc = env.get("observation_time_context", {}) or {}
    tac = hp.get("temporal_actor_context", {}) or {}
    if not tac.get("enabled", False):
        return  # nothing to check when the compressed temporal path is off

    # The env MUST be transporting history for the agent to split it.
    assert otc.get("enabled", False), \
        "temporal_actor_context.enabled requires observation_time_context.enabled"
    assert int(otc["obs_frame_stack"]) == int(tac["history_len"])
    assert bool(otc.get("stack_agent_state", False)) == bool(tac.get("stack_agent_state", False))

    # Derived stacked state_dim is self-consistent (current + appended history).
    obs_dim = int(env["environment_state_dim"])
    agent_dim = int(env["agent_state_dim"])
    N = int(otc["obs_frame_stack"])
    frame_len = (obs_dim + agent_dim) if otc.get("stack_agent_state") else obs_dim
    expected = (obs_dim + agent_dim) + (N - 1) * frame_len
    assert expected == 87 + (N - 1) * obs_dim   # sanity for the obs-only default


def test_aux_stagewise_schedule_present_and_sane():
    hp = _load("hyperparameters_tqc.yaml")["hyperparameters"]["aux_prediction"]
    sched = hp.get("stagewise_loss_schedule", [])
    if sched:
        assert all(float(x) >= 0.0 for x in sched)
        assert sched[0] == 0.0   # easy stage 0 starts with no aux pressure
