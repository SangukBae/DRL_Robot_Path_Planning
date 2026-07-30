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
        self._check_risk_map_reward(rep, docs)
        self._check_output_prefix(rep, docs)
        if resume:
            self._check_resume_state(rep, docs, seed=seed, package_root=package_root)
        self._check_action_mode(rep, docs, resume=resume, seed=seed, package_root=package_root)
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
    def _read_action_contract(run_dir: str):
        """Read {action_mode, action_dim} back from a prior run's
        configs/profile_manifest.json, or None if absent/unreadable/missing
        those keys (legacy layout has no configs/ dir at all)."""
        path = os.path.join(run_dir, "configs", "profile_manifest.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            return None
        if "action_mode" not in manifest or "action_dim" not in manifest:
            return None
        return manifest

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
