"""ConfigValidator — strong pre-flight checks for a resolved profile.

Run BEFORE any trainer/env process starts (train_node does this, and
``drl_experiments/scripts/run_profile.py --validate`` exposes it standalone).

Checks (per the PHASE2 experiment-safety requirements):
  1. every config file the profile names exists and parses as YAML;
  2. env-side and agent-side ``action_risk_head.enabled`` agree (candidate2
     needs the flag on BOTH processes), and both agree with the profile's
     declared override, if any;
  3. ``risk_map_reward.enabled`` is read from the ENV config (the single
     source of truth) and recorded in the report info — and cross-checked
     against the profile's declared override, if any;
  4. ``output_prefix`` matches the train config's ``base_file_name`` (run
     dirs / checkpoints / aggregation all key off base_file_name);
  5. when resuming (``resume=True``), the checkpoint / replay buffer /
     curriculum_state the run would load are located and reported explicitly
     (an error if nothing resumable exists).
"""

import json
import math
import os

import yaml

from .schema import ProfileSpec, ValidationReport


def _load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


class ConfigValidator:
    def __init__(self, spec: ProfileSpec):
        self.spec = spec

    # ------------------------------------------------------------------ #

    def validate(self, *, resume: bool = False, seed: int = None,
                 package_root: str = "") -> ValidationReport:
        rep = ValidationReport()
        docs = self._check_files_exist(rep)
        if rep.errors:
            return rep  # unreadable configs — nothing further is meaningful
        self._check_action_risk_head_consistency(rep, docs)
        self._check_counterfactual_and_spatiotemporal(rep, docs)
        self._check_risk_map_reward(rep, docs)
        self._check_continuous_control_reward(rep, docs)
        self._check_directional_risk_rollout_config(rep, docs)
        self._check_output_prefix(rep, docs)
        if resume:
            self._check_resume_state(rep, docs, seed=seed, package_root=package_root)
        self._check_action_mode(rep, docs, resume=resume, seed=seed, package_root=package_root)
        self._check_risk_feature_contract(
            rep, docs, resume=resume, seed=seed, package_root=package_root)
        return rep

    # ------------------------------------------------------------------ #

    def _check_files_exist(self, rep: ValidationReport) -> dict:
        docs = {}
        required = ("environment", "train", "hparams")
        keys = list(self.spec.config_paths)
        for key in required:
            if key not in keys:
                rep.errors.append(f"profile '{self.spec.name}' names no '{key}' config")
        if self.spec.trainer == "curriculum" and "curriculum" not in keys:
            rep.errors.append(
                f"profile '{self.spec.name}' uses trainer=curriculum but names no "
                f"'curriculum' config")
        for key, path in self.spec.config_paths.items():
            if not os.path.isfile(path):
                rep.errors.append(f"{key} config does not exist: {path}")
                continue
            try:
                docs[key] = _load_yaml(path)
            except yaml.YAMLError as e:
                rep.errors.append(f"{key} config is not valid YAML ({path}): {e}")
        return docs

    def _check_action_risk_head_consistency(self, rep, docs):
        env_cfg = (docs.get("environment") or {}).get("environment", docs.get("environment") or {})
        hp = (docs.get("hparams") or {}).get("hyperparameters", {}) or {}

        env_arh = bool(((env_cfg.get("action_risk_head") or {}).get("enabled", False)))
        agent_arh = bool(((hp.get("action_risk_head") or {}).get("enabled", False)))
        rep.info["action_risk_head.env_enabled"] = env_arh
        rep.info["action_risk_head.agent_enabled"] = agent_arh
        if env_arh != agent_arh:
            rep.errors.append(
                f"action_risk_head.enabled mismatch: env={env_arh} vs agent={agent_arh} "
                f"(candidate2 requires the flag on BOTH the env node and the trainer)")

        declared = self.spec.overrides.get("action_risk_head_enabled")
        if declared is not None and env_arh == agent_arh and declared != env_arh:
            rep.errors.append(
                f"profile declares action_risk_head_enabled={declared} but the config "
                f"files say {env_arh} — fix the profile or the yamls")

    def _check_counterfactual_and_spatiotemporal(self, rep, docs):
        env_cfg = (docs.get("environment") or {}).get(
            "environment", docs.get("environment") or {})
        hp = (docs.get("hparams") or {}).get("hyperparameters", {}) or {}
        env_cf = dict(env_cfg.get("counterfactual_multi_horizon_risk", {}) or {})
        agent_cf = dict(hp.get("counterfactual_multi_horizon_risk", {}) or {})
        env_on = bool(env_cf.get("enabled", False))
        agent_on = bool(agent_cf.get("enabled", False))
        rep.info["counterfactual_multi_horizon_risk.env_enabled"] = env_on
        rep.info["counterfactual_multi_horizon_risk.agent_enabled"] = agent_on
        if env_on != agent_on:
            rep.errors.append(
                "counterfactual_multi_horizon_risk.enabled mismatch: "
                f"env={env_on} vs agent={agent_on}")

        # horizon_weights is agent-only (no env-side counterpart -- it only
        # feeds the actor-penalty aggregation, never an env computation), so
        # this is validated unconditionally here rather than under the
        # `env_on and agent_on` block below: a profile that keeps the CF
        # block SYNCED with these keys but enabled=false (e.g. the baseline
        # toggle profile) must still catch a malformed horizon_weights at
        # --validate-only time instead of only failing much later at Agent
        # construction. Mirrors CounterfactualRiskConfig's own "always-valid
        # config regardless of enabled" policy.
        agent_horizons = [float(x) for x in agent_cf.get("horizons_sec", [])]
        if agent_cf.get("actor_risk_aggregation", "max") == "weighted_mean":
            hw = agent_cf.get("horizon_weights", None)
            if not hw:
                rep.errors.append(
                    "counterfactual_multi_horizon_risk.actor_risk_aggregation="
                    "'weighted_mean' requires horizon_weights")
            elif len(hw) != len(agent_horizons):
                rep.errors.append(
                    "counterfactual_multi_horizon_risk.horizon_weights length "
                    f"({len(hw)}) must match horizons_sec length "
                    f"({len(agent_horizons)})")
            elif any(float(w) < 0.0 for w in hw):
                rep.errors.append(
                    "counterfactual_multi_horizon_risk.horizon_weights "
                    "entries must all be >= 0")
            elif sum(float(w) for w in hw) <= 0.0:
                rep.errors.append(
                    "counterfactual_multi_horizon_risk.horizon_weights must "
                    "sum to > 0")

        if env_on and agent_on:
            env_h = [float(x) for x in env_cf.get("horizons_sec", [])]
            agent_h = [float(x) for x in agent_cf.get("horizons_sec", [])]
            env_a = env_cf.get("candidate_actions", [])
            agent_a = agent_cf.get("candidate_actions", [])
            if env_h != agent_h or env_a != agent_a:
                rep.errors.append(
                    "counterfactual env/agent horizons_sec and "
                    "candidate_actions must match exactly")
            action_dim = int(env_cfg.get("action_dim", 0) or 0)
            if not env_h or any(h <= 0.0 for h in env_h):
                rep.errors.append(
                    "counterfactual horizons_sec must be non-empty/positive")
            if not env_a or any(len(a) != action_dim for a in env_a):
                rep.errors.append(
                    "counterfactual candidate_actions must be non-empty and "
                    f"each have action_dim={action_dim} values")

        st = dict(hp.get("spatiotemporal_lidar", {}) or {})
        st_on = bool(st.get("enabled", False))
        temporal_on = bool(dict(hp.get("temporal_actor_context", {}) or {}).get(
            "enabled", False))
        rep.info["spatiotemporal_lidar.enabled"] = st_on
        if st_on and not temporal_on:
            rep.errors.append(
                "spatiotemporal_lidar.enabled=true requires "
                "temporal_actor_context.enabled=true")

    def _check_risk_map_reward(self, rep, docs):
        env_cfg = (docs.get("environment") or {}).get("environment", docs.get("environment") or {})
        rmr = bool(((env_cfg.get("risk_map_reward") or {}).get("enabled", False)))
        # The env config is the single source of truth for risk_map_reward —
        # record it so run manifests can pin the effective value.
        rep.info["risk_map_reward.enabled(env)"] = rmr

        hp = (docs.get("hparams") or {}).get("hyperparameters", {}) or {}
        if "risk_map_reward" in hp:
            rep.warnings.append(
                "hyperparameters yaml contains a 'risk_map_reward' block — it is "
                "IGNORED (env-side setting is the source of truth)")

        declared = self.spec.overrides.get("risk_map_reward_enabled")
        if declared is not None and declared != rmr:
            rep.errors.append(
                f"profile declares risk_map_reward_enabled={declared} but the env "
                f"config says {rmr} — fix the profile or environment yaml")

    def _check_continuous_control_reward(self, rep, docs):
        """speed_steering action mode: fail-fast checks for continuous_control_reward.

        reward_calculator.compute_reward normalises its penalty terms by
        vehicle_steering_limit_rad / controller_cruise_speed_mps, falling
        back to a near-zero divisor (1e-6) whenever either is missing/<=0 —
        that silently blows the corresponding penalty up to a huge value
        instead of erroring. Only meaningful (and only checked) when the
        block is actually enabled: a disabled block cannot affect the reward
        regardless of what garbage values it holds.
        """
        env_cfg = (docs.get("environment") or {}).get("environment", docs.get("environment") or {})
        ccr = dict(env_cfg.get("continuous_control_reward") or {})
        enabled = bool(ccr.get("enabled", False))
        rep.info["continuous_control_reward.enabled"] = enabled
        if not enabled:
            return

        action_dim = env_cfg.get("action_dim")
        action_mode = str(env_cfg.get(
            "action_mode",
            "waypoint_yield" if (action_dim or 0) >= 3 else "waypoint",
        )).strip().lower()
        if action_mode != "speed_steering":
            rep.warnings.append(
                f"continuous_control_reward.enabled=true but environment."
                f"action_mode={action_mode!r} (not 'speed_steering') — "
                "reward_calculator gates this feature off itself (harmless "
                "no-op), but the config is misleading; disable the block or "
                "fix action_mode")

        for key in ("heading_delta_weight", "steering_magnitude_weight",
                    "steering_change_weight", "speed_change_weight"):
            raw = ccr.get(key, 0.0)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                rep.errors.append(
                    f"continuous_control_reward.{key}={raw!r} is not a number")
                continue
            if not math.isfinite(val) or val < 0.0:
                rep.errors.append(
                    f"continuous_control_reward.{key}={val} must be finite and >= 0")

        steering_limit_deg = env_cfg.get("vehicle_steering_limit_deg", 21.58)
        try:
            steering_limit_deg = float(steering_limit_deg)
        except (TypeError, ValueError):
            steering_limit_deg = float("nan")
        if not (math.isfinite(steering_limit_deg) and steering_limit_deg > 0.0):
            rep.errors.append(
                "continuous_control_reward.enabled=true requires a positive "
                f"environment.vehicle_steering_limit_deg (got {steering_limit_deg!r}) "
                "— the steering-penalty normalizer would silently fall back to "
                "a near-zero divisor and blow up the penalty")

        cruise_speed = env_cfg.get("controller_cruise_speed_mps", 1.0)
        try:
            cruise_speed = float(cruise_speed)
        except (TypeError, ValueError):
            cruise_speed = float("nan")
        if not (math.isfinite(cruise_speed) and cruise_speed > 0.0):
            rep.errors.append(
                "continuous_control_reward.enabled=true requires a positive "
                f"environment.controller_cruise_speed_mps (got {cruise_speed!r}) "
                "— the speed-change penalty normalizer (and v_max fallback) "
                "would silently collapse to a near-zero divisor")

    def _check_directional_risk_rollout_config(self, rep, docs):
        """LOW: fail-fast sanity checks for the Ackermann swept-path rollout
        dynamics (directional_risk.rollout_*) and the risk-balanced sampling
        ratios -- neither was previously validated.

        A typo'd/degenerate rollout config fails SILENTLY rather than loudly:
        pure_pursuit.ackermann_swept_path returns an EMPTY path at
        horizon_sec<=0 or num_samples<=0, which collapses the action-
        conditioned risk target back to effectively a current-position-only
        read (compute_action_conditioned_risk treats an empty robot_path as
        "robot never moves") -- no crash, just a silently degraded target.
        Only checked when the rollout is actually reachable: action_mode==
        speed_steering, or waypoint_trajectory_risk_enabled=true
        (phase2/both) -- irrelevant/inert for every other profile.
        """
        env_cfg = (docs.get("environment") or {}).get("environment", docs.get("environment") or {})
        hp = (docs.get("hparams") or {}).get("hyperparameters", {}) or {}

        action_dim = env_cfg.get("action_dim")
        action_mode = str(env_cfg.get(
            "action_mode",
            "waypoint_yield" if (action_dim or 0) >= 3 else "waypoint",
        )).strip().lower()
        dr = dict(env_cfg.get("directional_risk", {}) or {})
        traj_risk_enabled = bool(dr.get("waypoint_trajectory_risk_enabled", False))

        if action_mode == "speed_steering" or traj_risk_enabled:
            def _finite_positive(key, default):
                raw = dr.get(key, default)
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    rep.errors.append(
                        f"directional_risk.{key}={raw!r} is not a number")
                    return
                if not math.isfinite(val) or val <= 0.0:
                    rep.errors.append(
                        f"directional_risk.{key}={val} must be finite and > 0 "
                        "-- required for the Ackermann swept-path rollout "
                        f"(action_mode={action_mode!r}, "
                        f"waypoint_trajectory_risk_enabled={traj_risk_enabled})"
                    )

            _finite_positive("horizon_sec", 1.0)
            _finite_positive("rollout_accel_limit_mps2", 6.0)
            _finite_positive("rollout_brake_decel_mps2", 6.0)
            _finite_positive("rollout_steering_rate_deg_s", 200.0)

            raw_samples = dr.get("rollout_path_samples", 15)
            # Strict integer check: int(15.5) == 15 would otherwise silently
            # accept a fractional sample count (no error, no effect on THIS
            # value, but a config typo like "15.5" should be rejected rather
            # than quietly truncated).
            samples = None
            if isinstance(raw_samples, bool):
                rep.errors.append(
                    f"directional_risk.rollout_path_samples={raw_samples!r} "
                    "is not an integer")
            elif isinstance(raw_samples, int):
                samples = raw_samples
            elif isinstance(raw_samples, float):
                if raw_samples.is_integer():
                    samples = int(raw_samples)
                else:
                    rep.errors.append(
                        f"directional_risk.rollout_path_samples={raw_samples!r} "
                        "is not an integer (has a fractional part)")
            else:
                try:
                    samples = int(str(raw_samples))
                except (TypeError, ValueError):
                    rep.errors.append(
                        f"directional_risk.rollout_path_samples={raw_samples!r} "
                        "is not an integer")
            if samples is not None and samples <= 0:
                rep.errors.append(
                    f"directional_risk.rollout_path_samples={samples} must be "
                    "> 0 -- 0 collapses the swept-path rollout to an EMPTY "
                    "path, silently degrading the action-conditioned risk "
                    "target to a current-position-only read"
                )

        rbs = dict((hp.get("replay_buffer", {}) or {}).get("risk_balanced_sampling", {}) or {})
        if bool(rbs.get("enabled", False)):
            # Effective (explicit-or-default) value per ratio, used for the
            # all-zero sum check below; per-key validity errors are only
            # raised for keys the profile actually SET (an absent key uses
            # LAP's own default and is never itself invalid).
            _ratio_defaults = {
                "ratio_uniform": 0.5, "ratio_human_risk": 0.25, "ratio_collision": 0.25,
            }
            effective_ratios = {}
            for key, default in _ratio_defaults.items():
                if key not in rbs:
                    effective_ratios[key] = default
                    continue
                raw = rbs.get(key)
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    rep.errors.append(
                        f"replay_buffer.risk_balanced_sampling.{key}={raw!r} "
                        "is not a number")
                    continue
                if not math.isfinite(val) or val < 0.0:
                    rep.errors.append(
                        f"replay_buffer.risk_balanced_sampling.{key}={val} "
                        "must be finite and >= 0")
                    continue
                effective_ratios[key] = val
            # Only meaningful once every ratio resolved to a valid value above
            # (a per-key error already covers the invalid-value case).
            if len(effective_ratios) == 3 and sum(effective_ratios.values()) <= 0.0:
                rep.errors.append(
                    "replay_buffer.risk_balanced_sampling: ratio_uniform + "
                    "ratio_human_risk + ratio_collision must sum to > 0 (all "
                    "currently 0) -- with all-zero ratios, sample_risk_balanced() "
                    "draws n_human=n_collision=0 and the WHOLE batch comes from "
                    "the uniform pool (see buffer.py's sample_risk_balanced), "
                    "functionally identical to plain uniform sampling despite "
                    "risk_balanced_sampling.enabled=true -- almost certainly an "
                    "activation mistake, not the intended config"
                )

    def _check_output_prefix(self, rep, docs):
        train = (docs.get("train") or {}).get("train_settings", {}) or {}
        base = str(train.get("base_file_name", "")).strip()
        rep.info["train.base_file_name"] = base
        if self.spec.output_prefix and base and self.spec.output_prefix != base:
            rep.errors.append(
                f"profile output_prefix='{self.spec.output_prefix}' != train config "
                f"base_file_name='{base}' — run dirs/checkpoints key off base_file_name")
        elif not self.spec.output_prefix:
            rep.warnings.append("profile declares no output_prefix (base_file_name "
                                f"'{base}' will be used unchecked)")

    def _check_action_mode(self, rep, docs, *, resume: bool, seed=None, package_root: str = ""):
        """environment.action_mode="speed_steering" changes the action
        contract (action_dim 2, no waypoint/yield geometry) and therefore the
        actor/critic/action-risk-head input width vs. any waypoint_yield /
        legacy-waypoint checkpoint — NOT just an architecture-flag change like
        A3/A4, an outright incompatible contract.

        Resume is only ever allowed when the checkpoint being resumed was
        ITSELF trained under the exact same (action_mode, action_dim) — verified
        via the resumed run's ``configs/profile_manifest.json`` (written by
        TrainTQCCurriculum._init_motion_logging_contract's manifest-augmentation
        call; see train_tqc_curriculum.py). Any other case — no manifest found
        (legacy layout / a run that predates this contract field), or a
        manifest recording a DIFFERENT action_mode/action_dim — is rejected
        outright, rather than relying on a shape-mismatch crash/silent-
        fresh-init deep inside tqc_io.load() to catch it (see that module:
        an actor/critic shape mismatch there is caught and silently degrades
        to a freshly-initialised network, which is NOT a safe way to detect
        this for a long unattended run).
        """
        env_cfg = (docs.get("environment") or {}).get("environment", docs.get("environment") or {})
        action_dim = env_cfg.get("action_dim")
        action_mode = str(env_cfg.get(
            "action_mode",
            "waypoint_yield" if (action_dim or 0) >= 3 else "waypoint",
        )).strip().lower()
        rep.info["environment.action_mode"] = action_mode
        if action_mode != "speed_steering" or not resume:
            return

        run_dir = rep.info.get("resume.run_dir", "(none)")
        if not run_dir or run_dir == "(none)":
            # _check_resume_state already reported "no checkpoint found" (or
            # resume wasn't otherwise resolvable) — nothing further to check.
            return

        contract = self._read_action_contract(run_dir)
        if contract is None:
            rep.errors.append(
                "profile uses environment.action_mode=speed_steering and "
                f"resume=true, but no action-contract record (action_mode/"
                f"action_dim) was found for the checkpoint in {run_dir} "
                "(missing/legacy-layout profile_manifest.json — it predates "
                "this contract check). Refusing to risk silently loading an "
                "incompatible checkpoint; start a fresh run instead."
            )
            return
        saved_mode = str(contract.get("action_mode", "")).strip().lower()
        saved_dim = contract.get("action_dim")
        if saved_mode != action_mode or int(saved_dim or -1) != int(action_dim or 0):
            rep.errors.append(
                f"profile uses environment.action_mode={action_mode!r} "
                f"(action_dim={action_dim}) but the checkpoint in {run_dir} "
                f"was trained under action_mode={saved_mode!r} "
                f"(action_dim={saved_dim}) — incompatible action contracts. "
                "Start a fresh run (no -p resume:=true / --resume)."
            )

    @staticmethod
    def _read_manifest(run_dir: str):
        """Read a prior run's configs/profile_manifest.json verbatim, or None
        if the file is absent/unreadable (legacy layout has no configs/ dir at
        all). Does NOT require any particular key to be present -- callers
        that need specific keys (e.g. _read_action_contract) check for those
        themselves."""
        path = os.path.join(run_dir, "configs", "profile_manifest.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _read_action_contract(run_dir: str):
        """Read {action_mode, action_dim} back from a prior run's
        configs/profile_manifest.json, or None if absent/unreadable/missing
        those keys (legacy layout has no configs/ dir at all)."""
        manifest = ConfigValidator._read_manifest(run_dir)
        if manifest is None or "action_mode" not in manifest or "action_dim" not in manifest:
            return None
        return manifest

    def _check_risk_feature_contract(self, rep, docs, *, resume: bool, seed=None,
                                      package_root: str = ""):
        """RISK_BALANCE / TRAJ_RISK: resume-safety hard-reject when either
        feature's activation state differs from (or is unrecorded in) the
        checkpoint that would be resumed.

        Both features change what a stored transition's replay_meta /
        action_risk_target MEAN (not just their array shape): risk_balanced_
        sampling changes which transitions the aux/action-risk supervised loss
        trains on, and waypoint_trajectory_risk_enabled changes the actual
        VALUE of the action_risk_target/risk_map_reward signal for
        waypoint_yield/waypoint. A shape-only checkpoint-load check (tqc_io.
        load()'s graceful degrade-to-fresh-init on a dimension mismatch) would
        not catch a same-shape-but-different-meaning drift, so this mirrors
        _check_action_mode's verify-or-reject pattern instead.

        Enforcement rule (only applies when resume=True):
          1. the checkpoint's manifest carries BOTH feature-contract keys
             (i.e. it was written by a run that already had this contract
             check) -> ALWAYS compare against the current config, regardless
             of the current config's own on/off state. A prior review found
             this previously short-circuited (returned early) whenever the
             CURRENT config had both features off, silently allowing a
             resume into a checkpoint that was trained WITH them on — fixed
             here by checking the manifest unconditionally instead of only
             when the current config enables something.
          2. no contract recorded (legacy manifest / no manifest at all) AND
             the current config enables NEITHER feature -> allow (nothing
             this check protects against; an old checkpoint predates the
             feature entirely and the current run doesn't use it either).
          3. no contract recorded AND the current config enables at least
             one feature -> reject (can't verify what the checkpoint was
             trained with).
        """
        env_cfg = (docs.get("environment") or {}).get("environment", docs.get("environment") or {})
        hp = (docs.get("hparams") or {}).get("hyperparameters", {}) or {}

        dr = dict(env_cfg.get("directional_risk", {}) or {})
        traj_risk_enabled = bool(dr.get("waypoint_trajectory_risk_enabled", False))
        rbs = dict((hp.get("replay_buffer", {}) or {}).get("risk_balanced_sampling", {}) or {})
        risk_balanced_enabled = bool(rbs.get("enabled", False))
        counterfactual_enabled = bool(dict(hp.get(
            "counterfactual_multi_horizon_risk", {}) or {}).get(
                "enabled", False))
        spatiotemporal_enabled = bool(dict(hp.get(
            "spatiotemporal_lidar", {}) or {}).get("enabled", False))

        rep.info["directional_risk.waypoint_trajectory_risk_enabled"] = traj_risk_enabled
        rep.info["replay_buffer.risk_balanced_sampling.enabled"] = risk_balanced_enabled
        rep.info["counterfactual_multi_horizon_risk.enabled"] = counterfactual_enabled
        rep.info["spatiotemporal_lidar.enabled"] = spatiotemporal_enabled

        if not resume:
            return

        run_dir = rep.info.get("resume.run_dir", "(none)")
        if not run_dir or run_dir == "(none)":
            # _check_resume_state already reported "no checkpoint found" (or
            # resume wasn't otherwise resolvable) — nothing further to check.
            return

        manifest = self._read_manifest(run_dir)
        has_contract = bool(
            manifest is not None
            and "waypoint_trajectory_risk_enabled" in manifest
            and "risk_balanced_sampling_enabled" in manifest
        )
        if not has_contract:
            if traj_risk_enabled or risk_balanced_enabled:
                rep.errors.append(
                    "profile enables replay_buffer.risk_balanced_sampling.enabled and/or "
                    "directional_risk.waypoint_trajectory_risk_enabled with resume=true, "
                    f"but no profile_manifest.json feature contract was found for the "
                    f"checkpoint in {run_dir} (missing/legacy-layout manifest — it "
                    "predates this contract check). Refusing to risk silently resuming "
                    "a checkpoint whose replay/target semantics may not match; start a "
                    "fresh run instead."
                )
            return  # no contract + both currently off -> nothing to protect against

        saved_traj = bool(manifest.get("waypoint_trajectory_risk_enabled", False))
        saved_rbs = bool(manifest.get("risk_balanced_sampling_enabled", False))
        saved_cf = bool(manifest.get(
            "counterfactual_multi_horizon_risk_enabled", False))
        saved_st = bool(manifest.get("spatiotemporal_lidar_enabled", False))
        if ((counterfactual_enabled or spatiotemporal_enabled)
                and ("counterfactual_multi_horizon_risk_enabled" not in manifest
                     or "spatiotemporal_lidar_enabled" not in manifest)):
            rep.errors.append(
                "checkpoint manifest predates the counterfactual/spatiotemporal "
                "architecture contract; start a fresh run when enabling either "
                "feature")
            return
        if (saved_traj != traj_risk_enabled or saved_rbs != risk_balanced_enabled
                or saved_cf != counterfactual_enabled
                or saved_st != spatiotemporal_enabled):
            rep.errors.append(
                "risk feature contract mismatch: profile requests "
                f"waypoint_trajectory_risk_enabled={traj_risk_enabled}, "
                f"risk_balanced_sampling_enabled={risk_balanced_enabled}, "
                f"counterfactual={counterfactual_enabled}, "
                f"spatiotemporal_lidar={spatiotemporal_enabled} but the "
                f"checkpoint in {run_dir} was trained with "
                f"waypoint_trajectory_risk_enabled={saved_traj}, "
                f"risk_balanced_sampling_enabled={saved_rbs}, "
                f"counterfactual={saved_cf}, spatiotemporal_lidar={saved_st} — "
                "the replay/target "
                "semantics differ. Start a fresh run (no -p resume:=true / --resume)."
            )

    def _check_resume_state(self, rep, docs, *, seed, package_root):
        """Verify what ``resume=true`` would ACTUALLY restore, not just that a
        checkpoint exists.

        A checkpoint (actor .pth) alone only guarantees a MODEL-WEIGHTS
        resume — the replay buffer, curriculum stage/step progress, and RNG
        snapshot are separate files the legacy loaders tolerate missing (see
        ``ResumeState``'s docstring). For a ``trainer: curriculum`` profile
        that silent degradation is never what "resume" should mean here: it
        would restart curriculum progress from stage 0 and/or the replay
        buffer from empty without the operator noticing, so both are HARD
        requirements. Non-curriculum (``trainer: base``) profiles have no
        curriculum state to lose, so a missing replay buffer only WARNS
        (explicit model-only resume is a legitimate use case there).
        """
        from ..rl.checkpointing.manager import CheckpointManager

        train = (docs.get("train") or {}).get("train_settings", {}) or {}
        base = str(train.get("base_file_name", "")).strip() or self.spec.output_prefix
        eff_seed = int(seed if seed is not None else train.get("seed", 0))
        rep.info["resume.seed"] = eff_seed

        mgr = CheckpointManager(package_root=package_root)
        state = mgr.describe_resume_state(base, eff_seed)
        for k, v in state.as_info().items():
            rep.info[f"resume.{k}"] = v
        if not state.resumable:
            rep.errors.append(
                f"resume requested but no checkpoint found for "
                f"(base_file_name='{base}', seed={eff_seed}) under "
                f"{state.searched_roots}")
            return

        if self.spec.trainer == "curriculum":
            if not state.has_replay_buffer:
                rep.errors.append(
                    f"resume requested for a curriculum profile but no replay "
                    f"buffer found next to checkpoint '{state.checkpoint_prefix}' "
                    f"in {state.run_dir} — off-policy resume would silently "
                    f"start with an EMPTY replay buffer")
            if not state.has_curriculum_state:
                rep.errors.append(
                    f"resume requested for a curriculum profile but no "
                    f"curriculum_state.json found in {state.run_dir} — resume "
                    f"would silently RESTART curriculum progress from stage 0")
        elif not state.has_replay_buffer:
            rep.warnings.append(
                f"no replay buffer found next to checkpoint "
                f"'{state.checkpoint_prefix}' — this will be a MODEL-ONLY "
                f"resume (fresh/empty replay buffer)")
