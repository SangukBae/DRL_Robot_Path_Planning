#!/usr/bin/env python3
"""Curriculum-learning subclass of Environment.

No existing file is modified. This file only adds:
  - curriculum_stage  ROS parameter (integer, writable at runtime by the trainer)
  - _load_curriculum_config()        parses the curriculum: block from self.config
  - _apply_curriculum_stage(idx)     overrides active counts + motion params
  - reset_callback override          reads the parameter and applies the stage
                                     before every episode's obstacle placement

Launch with:
  ros2 run drl_agent environment_curriculum.py \\
    --ros-args -p config_file:=<path>/environment_curriculum.yaml

The trainer (train_tqc_curriculum_agent.py) advances the stage by writing to
the /gym_node/set_parameters service after each successful evaluation.
The trainer also reads /gym_node/get_parameters::curriculum_num_stages to stay
in sync with the actual number of stages in the config that was passed here.
"""

import os
import sys
import math
import copy

# Make sure the environment directory is on the path so we can import Environment.

import rclpy
import rclpy.executors
from rcl_interfaces.msg import SetParametersResult

from drl_agent.env.simulation.environment import Environment   # base class — not modified
from drl_agent.env.simulation.map_catalog import MAP_TYPES, clamp_active_by_map
import drl_agent.config.paths as config_paths


def _preferred_curriculum_config_path() -> str:
    """Prefer the editable source-tree curriculum config over install/share.

    This avoids the common failure mode where a user updates
    ``ros2_ws/src/drl_agent/config/environment_curriculum.yaml`` but the launched
    env still reads an older installed copy from ``install/share``.
    """
    share_dir = ""
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("drl_agent")
    except Exception:
        pass

    src_candidates = config_paths.source_package_candidates(
        "drl_agent",
        src_hint=os.environ.get("DRL_AGENT_SRC_PATH", "").strip(),
        installed_share_dir=share_dir,
        script_path=__file__,
    )
    hit = config_paths.find_config_file("environment_curriculum.yaml", src_candidates)
    if hit:
        return hit

    if share_dir:
        cand = os.path.join(share_dir, "config", "environment_curriculum.yaml")
        if os.path.isfile(cand):
            return cand
    return ""


def _with_default_curriculum_config(args=None):
    """Inject `config_file:=environment_curriculum.yaml` when the caller did
    not explicitly provide one.

    This must happen before rclpy.init(), otherwise the base Environment class
    will fall back to the installed default `environment.yaml`.
    """
    arg_list = list(sys.argv[1:] if args is None else args)
    joined = " ".join(arg_list)
    if "config_file:=" in joined:
        return arg_list

    default_cfg = _preferred_curriculum_config_path()
    if default_cfg:
        arg_list.extend(["--ros-args", "-p", f"config_file:={default_cfg}"])
    return arg_list


class EnvironmentCurriculum(Environment):
    """Environment with runtime curriculum stage control.

    The pool is always built to the maximum size (from yaml).
    _apply_curriculum_stage() just changes how many pool slots are activated
    and what motion parameters are used — no pool rebuild ever happens.
    """

    # ------------------------------------------------------------------ #
    #  Auxiliary-Prediction-friendly pedestrian motion                      #
    # ------------------------------------------------------------------ #
    # Goal: humans must be PREDICTABLE over short horizons (≈ constant
    # velocity) yet not fully static, with uncertainty injected only at the
    # *mode* level — i.e. a new waypoint every few seconds — and they must
    # NOT react to the robot (reactive pedestrians are for final evaluation
    # only). The predictable profile (low accel / low yaw rate / bounded
    # waypoint distance / segment timeout / retarget-only jitter / weak
    # stop-go) lives in the YAML `environment:` block so it applies to BOTH
    # the curriculum AND the plain environment.py (non-curriculum) path.
    #
    # The keys below are the human-motion knobs a stage MAY override; if a
    # stage omits a key it simply inherits the base YAML value. So a stage can
    # e.g. lengthen segments or add pauses without redefining the whole profile.
    HUMAN_MOTION_OVERRIDE_KEYS = (
        "human_max_accel",
        "human_max_yaw_rate",
        "human_max_yaw_accel",
        "human_desired_yaw_rate_limit",
        "human_min_retarget_interval",
        "human_retarget_prob_per_sec",
        "human_stop_prob_per_sec",
        "human_pause_duration",
        "human_waypoint_dist_min",
        "human_waypoint_dist_max",
        "human_max_segment_sec",
        # Falcon-lite goal-driven + weak social avoidance (scalar knobs only; the
        # on/off toggles human_goal_driven_enabled / human_social_avoid_enabled
        # stay global in the YAML). A harder stage can e.g. widen the goal span or
        # strengthen avoidance; an omitted key inherits the base value.
        "human_goal_reach_threshold",
        "human_goal_pause_duration",
        "human_local_step_m",
        "human_goal_span_multiplier",
        "human_goal_min_span_m",
        "human_static_obstacle_clearance",
        "human_social_avoid_radius",
        "human_social_avoid_strength",
    )

    def __init__(self):
        # Prefer the curriculum config by default. The base Environment loader
        # checks the explicit ROS parameter first, so a user-provided
        # `-p config_file:=...` still overrides this fallback cleanly.
        if "DRL_AGENT_CONFIG" not in os.environ:
            default_cfg = _preferred_curriculum_config_path()
            if default_cfg:
                os.environ["DRL_AGENT_CONFIG"] = default_cfg

        super().__init__()                      # full base init (config, pool, timers …)
        self._load_curriculum_config()
        # Declare AFTER super() so we don't conflict with base parameter declarations.
        self.declare_parameter("curriculum_stage", self.curriculum_initial_stage)
        # curriculum_num_stages is read-only: lets the trainer query the actual
        # stage count from the config file the environment was launched with,
        # avoiding any mismatch between the two processes.
        self.declare_parameter("curriculum_num_stages", len(self.curriculum_stages))
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._current_stage = self.curriculum_initial_stage
        self._last_stage_apply_logged = None
        # Snapshot the base human-motion knobs (loaded from the YAML by the base
        # Environment) BEFORE any stage is applied. Every stage is then computed
        # as base + its own overrides, so omitting a key inherits the base value
        # instead of leaking the previous stage's value.
        self._snapshot_human_base()
        self._apply_curriculum_stage(self._current_stage)
        self.get_logger().info(
            f"[Curriculum] Ready — stage {self._current_stage} "
            f"('{self._stage_name(self._current_stage)}') | "
            f"total stages: {len(self.curriculum_stages)}"
        )

    # ------------------------------------------------------------------ #
    #  Config loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_curriculum_config(self):
        """Parse the curriculum: block from self.config (loaded by super().__init__)."""
        cfg = self.config.get("curriculum", {})
        self.curriculum_enabled       = bool(cfg.get("enabled", False))
        self.curriculum_initial_stage = int(cfg.get("initial_stage", 0))
        self.curriculum_stages        = cfg.get("stages", [])
        if not self.curriculum_stages:
            self.curriculum_enabled = False
            self.get_logger().warn(
                "[Curriculum] No stages found in config — curriculum disabled. "
                "Make sure you are using environment_curriculum.yaml."
            )

    def _on_set_parameters(self, params):
        """Reject writes to curriculum_num_stages — it is read-only."""
        for p in params:
            if p.name == "curriculum_num_stages":
                return SetParametersResult(
                    successful=False,
                    reason="curriculum_num_stages is read-only",
                )
        return SetParametersResult(successful=True)

    def _stage_name(self, idx: int) -> str:
        if 0 <= idx < len(self.curriculum_stages):
            return self.curriculum_stages[idx].get("name", f"stage_{idx}")
        return f"stage_{idx}"

    # ------------------------------------------------------------------ #
    #  Stage application                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deep_merge(dst: dict, src: dict) -> dict:
        """Recursively merge src INTO a copy of dst and return it.

        Dict values are merged key-by-key (recursing into nested dicts); every
        other type (scalars, lists) is replaced wholesale. Used for per-stage
        `localization:` overrides so a stage can change a single nested value
        (e.g. map_type_multipliers.corridor.sigma_xy) without clobbering the
        sibling map-type entries. dst is not mutated."""
        out = copy.deepcopy(dst) if isinstance(dst, dict) else {}
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = EnvironmentCurriculum._deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out

    def _resolve_noise_override(self, stage: dict, *, profile_key: str,
                                inline_key: str, catalog_key: str) -> dict:
        """Resolve a stage's noise override as (named profile) + (inline block).

        Pure SELECTOR — no noise maths. Returns the merged override dict to be
        deep-merged onto the (already restored) base:
          • stage[profile_key] = <name>  → looks <name> up in self.config[catalog_key]
          • stage[inline_key]  = {...}    → inline override, applied ON TOP of the
                                            named profile (legacy path; still works)
        Returns {} when the stage selects neither (→ inherit base). An unknown
        profile name is warned about and ignored (falls back to inline/base)."""
        out = {}
        name = stage.get(profile_key)
        if name:
            catalog = self.config.get(catalog_key, {}) or {}
            prof = catalog.get(name)
            if isinstance(prof, dict):
                out = self._deep_merge(out, prof)
            else:
                self.get_logger().warn(
                    f"[Curriculum] unknown {profile_key}='{name}' "
                    f"(not in {catalog_key}); using inline/base only."
                )
        inline = stage.get(inline_key)
        if isinstance(inline, dict):
            out = self._deep_merge(out, inline)
        return out

    def _snapshot_human_base(self):
        """Capture the YAML-loaded base values of every stage-overridable human
        knob, so each stage is applied as (base + that stage's overrides) and
        omitting a key inherits the base value (no cross-stage leakage)."""
        self._human_base = {k: getattr(self, k) for k in self.HUMAN_MOTION_OVERRIDE_KEYS}
        self._human_base.update({
            "human_heading_jitter":    self.human_heading_jitter,
            # Falcon-lite avoidance cap is stored in radians; a stage overrides it
            # via the _deg key (handled specially below), so snapshot the radians.
            "human_social_avoid_max_heading_offset": self.human_social_avoid_max_heading_offset,
            "human_scan_noise_std":    self.human_scan_noise_std,
            "human_scan_dropout_prob": self.human_scan_dropout_prob,
            "human_placement_mode":    getattr(self, "human_placement_mode", "quadrants"),
            "human_mode_weights":      dict(self.human_mode_weights),
            "human_mode_params":       copy.deepcopy(self.human_mode_params),
            # Structured map curriculum (docs/map_curriculum_plan.md §8): base
            # map-sampling config so a stage that omits a field inherits the YAML
            # base instead of leaking the previous stage's map setting.
            "allowed_map_types":       list(getattr(self, "allowed_map_types", [])),
            "map_type_probs":          list(getattr(self, "map_type_probs", [])),
            "allowed_static_groups":   list(getattr(self, "allowed_static_groups", [])),
            "eval_map_types":          list(getattr(self, "eval_map_types", [])),
            # Base active counts (YAML maximums) so a stage omitting active_static /
            # active_humans inherits the base instead of the previous stage's count.
            "num_of_static_obstacles": self.num_of_static_obstacles,
            "num_of_humans":           self.num_of_humans,
            # Base goal-obs (localization-uncertainty) + proprio noise configs.
            # The curriculum only SELECTS/merges a noise profile per stage; the
            # noise maths lives entirely in environment.py. Deep-restored each
            # stage so a partial override never leaks across stages.
            "loc_noise":               copy.deepcopy(self.loc_noise),
            "proprio_noise":           copy.deepcopy(self.proprio_noise),
            # Stop/yield capability (global env: block = stop-ON default). A stage
            # may override these to run stop-OFF (basic driving); omitting a key
            # inherits the base so stage 3+ keeps the stop-ON default.
            "actions_low":                     list(self.actions_low),
            "controller_min_speed_mps":        float(self.controller_min_speed_mps),
            "controller_low_speed_distance_m": float(self.controller_low_speed_distance_m),
            "yield_enabled":                   bool(self.yield_enabled),
            # 3D-action yield channel gate (controller-side). Sealed in early
            # stages so avoidance is learned before yielding is permitted.
            "yield_action_enabled":            bool(self.yield_action_enabled),
        })

    def _parse_active_by_map(self, raw, cap: int, label: str) -> dict:
        """Validate + clamp a stage's active_*_by_map dict via the ROS-free
        map_catalog.clamp_active_by_map helper (unknown map keys / non-int values
        dropped, counts clamped to [0, cap]); any issues are surfaced as warnings.
        An empty / missing / malformed input → {} so the base resolver falls back
        to the stage single value. Never raises (a bad stage entry can't crash a
        reset)."""
        clean, warns = clamp_active_by_map(raw, cap, allowed=MAP_TYPES)
        for w in warns:
            self.get_logger().warn(f"[Curriculum] {label}: {w}")
        return clean

    def _apply_curriculum_stage(self, idx: int):
        """Apply stage idx as (base human knobs) + (this stage's overrides).

        Pool sizes are fixed at startup (from yaml obstacle_pool_*_size).
        This method only changes the *active* subset; inactive pool slots
        are returned to their parking positions by _activate_random_obstacles.
        """
        if not self.curriculum_enabled:
            return
        idx = max(0, min(idx, len(self.curriculum_stages) - 1))
        stage = self.curriculum_stages[idx]

        # Restore every stage-overridable human knob to its base (YAML) value
        # first, so a stage that omits a key inherits the base — not whatever the
        # previously-applied stage left behind.
        b = self._human_base
        for key in self.HUMAN_MOTION_OVERRIDE_KEYS:
            setattr(self, key, b[key])
        self.human_heading_jitter    = b["human_heading_jitter"]
        self.human_social_avoid_max_heading_offset = b["human_social_avoid_max_heading_offset"]
        self.human_scan_noise_std    = b["human_scan_noise_std"]
        self.human_scan_dropout_prob = b["human_scan_dropout_prob"]
        self.human_placement_mode    = b["human_placement_mode"]
        self.human_mode_weights      = dict(b["human_mode_weights"])
        self.human_mode_params       = copy.deepcopy(b["human_mode_params"])
        self.num_of_static_obstacles = b["num_of_static_obstacles"]
        self.num_of_humans           = b["num_of_humans"]
        self.loc_noise               = copy.deepcopy(b["loc_noise"])
        self.proprio_noise           = copy.deepcopy(b["proprio_noise"])
        # Stop/yield capability — restore base (global stop-ON) before any override.
        self.actions_low                     = list(b["actions_low"])
        self.controller_min_speed_mps        = b["controller_min_speed_mps"]
        self.controller_low_speed_distance_m = b["controller_low_speed_distance_m"]
        self.yield_enabled                   = b["yield_enabled"]
        self.yield_action_enabled            = b["yield_action_enabled"]
        # Structured map curriculum: restore base map-sampling config before
        # applying this stage's overrides (no cross-stage leakage).
        self.allowed_map_types       = list(b["allowed_map_types"])
        self.map_type_probs          = list(b["map_type_probs"])
        self.allowed_static_groups   = list(b["allowed_static_groups"])
        self.eval_map_types          = list(b["eval_map_types"])

        # Active counts — never exceed pre-allocated pool sizes. Base values were
        # just restored above, so a stage that omits a key inherits the base.
        # Dynamic obstacles are removed: only static obstacles and humans remain.
        self.num_of_static_obstacles = min(
            int(stage.get("active_static",  self.num_of_static_obstacles)),
            self.obstacle_pool_static_size,
        )
        self.num_of_humans = min(
            int(stage.get("active_humans",  self.num_of_humans)),
            self.obstacle_pool_human_size,
        )

        # ── Per-MAP active counts (same stage, different map geometry) ──────────
        # A stage may set active_static_by_map / active_humans_by_map so e.g. the
        # narrow corridor gets fewer obstacles than the intersection/clutter/lobby
        # WITHIN the same stage. Resolution happens per episode in the base
        # _apply_episode_active_counts() (after map_type is chosen): it prefers the
        # by-map value, else the stage single value just computed above. We store
        # both here; the single values double as the fallback for any map_type a
        # by-map dict omits, and as the count for the non-structured path.
        self._stage_active_static = self.num_of_static_obstacles
        self._stage_active_humans = self.num_of_humans
        self._stage_active_static_by_map = self._parse_active_by_map(
            stage.get("active_static_by_map"), self.obstacle_pool_static_size,
            "active_static_by_map")
        self._stage_active_humans_by_map = self._parse_active_by_map(
            stage.get("active_humans_by_map"), self.obstacle_pool_human_size,
            "active_humans_by_map")

        # Human sensor noise / dropout (domain randomisation intensity)
        if "human_scan_noise_std" in stage:
            self.human_scan_noise_std = float(stage["human_scan_noise_std"])
        if "human_scan_dropout_prob" in stage:
            self.human_scan_dropout_prob = float(stage["human_scan_dropout_prob"])
        if "human_placement_mode" in stage:
            self.human_placement_mode = str(stage["human_placement_mode"])

        # Per-stage overrides of the scalar human-motion knobs (base already
        # restored above; a stage only lists the keys it wants to change).
        for key in self.HUMAN_MOTION_OVERRIDE_KEYS:
            if key in stage:
                setattr(self, key, float(stage[key]))
        if "human_heading_jitter_deg" in stage:
            self.human_heading_jitter = math.radians(float(stage["human_heading_jitter_deg"]))
        # Falcon-lite avoidance cap (degrees -> radians), stage-overridable so a
        # harder stage can let humans bend a bit more to dodge each other.
        if "human_social_avoid_max_heading_offset_deg" in stage:
            self.human_social_avoid_max_heading_offset = math.radians(
                float(stage["human_social_avoid_max_heading_offset_deg"]))
        # Keep heading changes confined to retarget instants so motion between
        # waypoints stays (near-)constant heading — ideal for prediction.
        self.human_heading_jitter_on_retarget_only = True

        # Per-stage pedestrian behaviour-mode controls (crossing / along_path /
        # waiting / slow_turn):
        #   • human_mode_weights → replaces the mix (a mix is atomic)
        #   • human_mode_params  → deep-merged per mode, so a stage can retune a
        #     few modes while keeping the base tuning of the others.
        if "human_mode_weights" in stage:
            self.human_mode_weights = dict(stage["human_mode_weights"])
        if "human_mode_params" in stage:
            for mode, params in (stage["human_mode_params"] or {}).items():
                self.human_mode_params.setdefault(mode, {}).update(params or {})

        # ── Per-stage NOISE PROFILE selection (curriculum's only noise role) ──
        # The stage picks a goal-obs (localization-uncertainty) noise profile and
        # an optional proprio-noise profile; environment.py owns the maths. Both
        # are DEEP-merged onto the deep-restored base, so a partial override never
        # drops sibling/other keys and never leaks across stages.
        #
        # Two equivalent ways to set the goal-obs profile (selector only):
        #   1. localization_profile: <name>   → looks up `localization_profiles`
        #   2. localization: {...}            → inline override (legacy; still works)
        # If both are given, the named profile is applied first, then the inline
        # block on top. Resolution order: base → named profile → inline.
        loc_override = self._resolve_noise_override(
            stage, profile_key="localization_profile",
            inline_key="localization", catalog_key="localization_profiles")
        if loc_override:
            self.loc_noise = self._deep_merge(self.loc_noise, loc_override)

        # Proprioception observation noise — a SEPARATE axis (speed / yaw-rate /
        # steering corruption). Same profile-or-inline selection.
        pp_override = self._resolve_noise_override(
            stage, profile_key="proprio_noise_profile",
            inline_key="proprio_noise", catalog_key="proprio_noise_profiles")
        if pp_override:
            self.proprio_noise = self._deep_merge(self.proprio_noise, pp_override)

        # Structured map curriculum per-stage overrides (base restored above).
        # A stage that omits these inherits the YAML base. map_type_probs is kept
        # only when it matches allowed_map_types length (else uniform downstream).
        if "allowed_map_types" in stage:
            self.allowed_map_types = [
                m for m in (stage["allowed_map_types"] or []) if isinstance(m, str)
            ] or self.allowed_map_types
        if "map_type_probs" in stage:
            self.map_type_probs = [float(p) for p in (stage["map_type_probs"] or [])]
        if "allowed_static_groups" in stage:
            self.allowed_static_groups = [
                str(g) for g in (stage["allowed_static_groups"] or [])
            ]
        if "eval_map_types" in stage:
            self.eval_map_types = [
                m for m in (stage["eval_map_types"] or []) if isinstance(m, str)
            ]

        # ── Per-stage STOP/YIELD capability override (base stop-ON restored above) ──
        # Early stages run stop-OFF (basic-driving) by overriding these; a stage
        # that omits them inherits the global stop-ON default.
        if "actions_low" in stage:
            self.actions_low = [float(x) for x in stage["actions_low"]]
        if "controller_min_speed_mps" in stage:
            self.controller_min_speed_mps = float(stage["controller_min_speed_mps"])
        if "controller_low_speed_distance_m" in stage:
            self.controller_low_speed_distance_m = float(stage["controller_low_speed_distance_m"])
        # Nested to mirror the global yield_reward.enabled key.
        _yr = stage.get("yield_reward")
        if isinstance(_yr, dict):
            if "enabled" in _yr:
                self.yield_enabled = bool(_yr["enabled"])
            # Controller-side seal: action_enabled:false makes the yield channel
            # (action[2]) a no-op so early stages cannot stop at all.
            if "action_enabled" in _yr:
                self.yield_action_enabled = bool(_yr["action_enabled"])

        if self._last_stage_apply_logged != idx:
            self.get_logger().info(
                f"[Curriculum] Stage {idx} '{self._stage_name(idx)}' applied — "
                f"static={self.num_of_static_obstacles} "
                f"humans={self.num_of_humans} "
                f"human_place={self.human_placement_mode} "
                f"noise={self.human_scan_noise_std:.3f} "
                f"dropout={self.human_scan_dropout_prob:.3f} | "
                f"motion(acc={self.human_max_accel:.2f} "
                f"yaw_rate={self.human_max_yaw_rate:.2f} "
                f"retarget_int={self.human_min_retarget_interval:.1f}s "
                f"retarget_p={self.human_retarget_prob_per_sec:.3f} "
                f"stop_p={self.human_stop_prob_per_sec:.3f}) "
                f"mode_mix={self.human_mode_weights or 'uniform'} | "
                f"maps={self.allowed_map_types}"
                f"{('p=' + str(self.map_type_probs)) if self.map_type_probs else ''} "
                f"groups={self.allowed_static_groups or 'all'} | "
                f"loc(on={self.loc_noise['enabled']} "
                f"sig_xy={self.loc_noise['sigma_xy_m']:.2f} "
                f"delay={self.loc_noise['delay_steps']} "
                f"gt_rew={self.loc_noise['use_gt_for_reward']}) | "
                f"by_map(static={self._stage_active_static_by_map or '-'} "
                f"humans={self._stage_active_humans_by_map or '-'}) | "
                f"stop(act_low={[round(float(x), 3) for x in self.actions_low]} "
                f"min_v={self.controller_min_speed_mps:.2f} "
                f"low_dist={self.controller_low_speed_distance_m:.2f} "
                f"yield={self.yield_enabled} "
                f"yield_act={self.yield_action_enabled})"
            )
            self._last_stage_apply_logged = idx

    # ------------------------------------------------------------------ #
    #  reset_callback override                                              #
    # ------------------------------------------------------------------ #

    def reset_callback(self, request, response):
        """Read curriculum_stage parameter and apply it before each episode.

        The trainer writes /gym_node/set_parameters between episodes.
        This override detects the change and applies the new stage config
        before _activate_random_obstacles() is called in the base class.
        """
        stage = int(
            self.get_parameter("curriculum_stage")
            .get_parameter_value()
            .integer_value
        )
        if stage != self._current_stage:
            self.get_logger().info(
                f"[Curriculum] Stage transition: "
                f"{self._current_stage} ('{self._stage_name(self._current_stage)}') → "
                f"{stage} ('{self._stage_name(stage)}')"
            )
            self._current_stage = stage
        # Always re-apply so per-stage counts and noise settings take effect
        # even on the very first call or after a parameter write.
        self._apply_curriculum_stage(self._current_stage)

        return super().reset_callback(request, response)


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=_with_default_curriculum_config(args))
    node = None
    try:
        node = EnvironmentCurriculum()
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
