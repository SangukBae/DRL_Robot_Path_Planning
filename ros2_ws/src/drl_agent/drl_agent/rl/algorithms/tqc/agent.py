import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os


import drl_agent.rl.replay.buffer as buffer

# AUX_PRED: auxiliary prediction network (shared encoder + future-risk head).
# All auxiliary logic lives in these modules; with aux_prediction.enabled=false
# the SharedEncoder is a parameter-free identity passthrough and the baseline
# TQC behaviour is reproduced exactly.
from drl_agent.rl.networks.aux_prediction import (
    AuxPredConfig, SharedEncoder, AuxiliaryHead, ActionConditionedAuxHead,
)
# AUX_PRED (v2): aux-only temporal context (GRU over recent in-episode states).
# Disabled -> out_dim 0, so the head is built exactly as the single-step v1 head.
from drl_agent.rl.networks.aux_temporal import TemporalContextEncoder
# TEMPORAL_ACTOR: compressed temporal feature on the ACTOR/CRITIC path. Splits the
# transported stacked state into current(87) + scan-history and fuses a small
# temporal feature into the shared latent (actor/critic stay non-recurrent).
from drl_agent.rl.networks.aux_temporal import TemporalFusionEncoder

# PHASE2: Critic-connected Action-Risk Head (default OFF). See
# action_risk_head.py's module docstring for the gradient rule.
from drl_agent.rl.networks.action_risk_head import (
    ActionRiskConfig, ActionRiskHead,
    CounterfactualRiskConfig, CounterfactualMultiHorizonRiskHead,
)

# STAGE 8: fixed physical-range observation normalization (ISOLATED
# experimental feature, default OFF -- see obs_normalization.py docstring).
from drl_agent.rl.networks.obs_normalization import ObsNormalizationConfig, ObsNormalizer


# Network definitions + TQC loss live in drl_agent.rl.networks.tqc; checkpoint
# I/O in drl_agent.rl.checkpointing.tqc_io. Actor/Critic are used directly
# below (network construction); quantile_huber_loss has no caller left in
# this file (its only use, inside train(), moved to update.py) but stays
# re-imported here because rl/algorithms/tqc/__init__.py's lazy __getattr__
# exposes Agent/Actor/Critic/quantile_huber_loss as attributes of THIS
# module, not update.py.
from drl_agent.rl.networks.tqc import Actor, Critic, quantile_huber_loss
import drl_agent.rl.checkpointing.tqc_io as tqc_io

# Actor/critic update step + CF penalty schedule, metric logging, and
# network/optimizer construction — split out of this file for cohesion; see
# each module's docstring.
# _frozen_params / _polyak_update_foreach re-exported for existing callers
# that import them directly from this module (e.g. tests exercising the
# freeze-not-detach mechanism) — the actual implementations now live in
# update.py alongside train(), their only caller.
from drl_agent.rl.algorithms.tqc.update import (
    UpdateMixin, _frozen_params, _polyak_update_foreach,
)
from drl_agent.rl.algorithms.tqc.metrics import MetricsMixin
from drl_agent.rl.algorithms.tqc.networks import NetworksMixin


class Agent(UpdateMixin, MetricsMixin, NetworksMixin, object):
    def __init__(self, state_dim, action_dim, max_action, hyperparameters, log_dir=None,
                 env_obs_dim=None, env_agent_dim=None):
        # ----------------------------
        # Hyperparameters (preprocess)
        # ----------------------------
        self.hyperparameters = self.prep_hyperparameters(hyperparameters)

        # Common
        self.discount = self.hyperparameters["discount"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.buffer_size = self.hyperparameters["buffer_size"]
        self.target_update_interval = self.hyperparameters.get("target_update_interval", 1)
        self.tau = self.hyperparameters.get("tau", 0.005)

        # TQC specific
        self.n_quantiles = self.hyperparameters.get("n_quantiles", 25)
        self.n_critics = self.hyperparameters.get("n_critics", 2)
        self.top_quantiles_to_drop_per_net = self.hyperparameters.get("top_quantiles_to_drop_per_net", 2)

        # Entropy / Temperature
        self.ent_coef = self.hyperparameters.get("ent_coef", "auto")
        # dtype 혼선 방지를 위해 float로 고정
        self.target_entropy = float(self.hyperparameters.get("target_entropy", -float(action_dim)))
        self.ent_coef_lr = float(self.hyperparameters.get("ent_coef_lr", 3e-4))

        # Model hparams
        self.actor_hdim = self.hyperparameters.get("actor_hdim", 256)
        self.actor_activ = self.hyperparameters.get("actor_activ", F.relu)
        self.actor_lr = float(self.hyperparameters.get("actor_lr", 3e-4))
        self.critic_hdim = self.hyperparameters.get("critic_hdim", 256)
        self.critic_activ = self.hyperparameters.get("critic_activ", F.elu)
        self.critic_lr = float(self.hyperparameters.get("critic_lr", 3e-4))
        # A3 (critic scaling): opt-in residual critic body + LayerNorm. Both
        # default OFF -> Critic is byte-for-byte the original plain MLP, so the
        # baseline (and any plain-critic checkpoint) is unaffected. Enabling
        # residual changes the critic state_dict -> FRESH RUN only.
        self.critic_residual = bool(self.hyperparameters.get("critic_residual", False))
        self.critic_layernorm = bool(self.hyperparameters.get("critic_layernorm", False))
        self.critic_residual_blocks = int(self.hyperparameters.get("critic_residual_blocks", 2))

        # Checkpointing (trainer 호환)
        self.reset_weight = self.hyperparameters.get("reset_weight", 0.9)
        self.steps_before_checkpointing = self.hyperparameters.get("steps_before_checkpointing", 40000)
        self.max_eps_when_checkpointing = self.hyperparameters.get("max_eps_when_checkpointing", 50)

        # Prioritized replay 옵션
        self.prioritized = bool(self.hyperparameters.get("prioritized", False))
        if self.prioritized:
            self.alpha = float(self.hyperparameters.get("alpha", 0.4))
            self.min_priority = float(self.hyperparameters.get("min_priority", 1))

        # Device & scales
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = float(max_action)

        # ----------------------------
        # RISK_BALANCE: read replay_buffer.risk_balanced_sampling.enabled EARLY
        # (before AuxPredConfig/ActionRiskConfig below) -- it is the single
        # master activation flag for the WHOLE risk-balance feature, including
        # the sparse-positive loss weighting knobs
        # (aux_prediction.risk_map_positive_weight/risk_map_loss_type/
        # hazard_pos_weight, action_risk_head.pos_weight/loss_type), not just
        # the balanced batch draw. Those weight/loss_type keys otherwise parse
        # independently of this flag (reviewed HIGH finding: setting
        # risk_balanced_sampling.enabled=false alone did NOT restore byte-
        # identical unweighted-MSE loss when a profile still set e.g.
        # risk_map_positive_weight=3.0/loss_type=smooth_l1) -- see the
        # neutralisation below, just before each config is constructed.
        # ----------------------------
        _rb_cfg = dict(self.hyperparameters.get("replay_buffer", {}) or {})
        _rm_cfg = dict(_rb_cfg.get("risk_meta", {}) or {})
        _rbs_cfg = dict(_rb_cfg.get("risk_balanced_sampling", {}) or {})
        self.risk_balanced_enabled = bool(_rbs_cfg.get("enabled", False))
        self.store_risk_meta = bool(_rm_cfg.get("enabled", False)) or self.risk_balanced_enabled
        self.risk_balanced_ratios = (
            float(_rbs_cfg.get("ratio_uniform", 0.5)),
            float(_rbs_cfg.get("ratio_human_risk", 0.25)),
            float(_rbs_cfg.get("ratio_collision", 0.25)),
        )

        # ----------------------------
        # AUX_PRED: Shared encoder (E_psi)
        # ----------------------------
        # The encoder sits between the raw 87-D state and the actor/critic.
        # When disabled it is an identity passthrough (out_dim == state_dim), so
        # actor/critic see the raw state exactly as in baseline TQC.  When
        # enabled it maps 87 -> hidden -> latent (ELU) and the actor consumes a
        # DETACHED latent so the actor loss never updates the encoder.
        _aux_pred_dict = dict(self.hyperparameters.get("aux_prediction", {}) or {})
        if not self.risk_balanced_enabled:
            # RISK_BALANCE: neutralise the sparse-positive weighting knobs so
            # risk_balanced_sampling.enabled=false is byte-identical to this
            # feature never having existed, REGARDLESS of what these keys are
            # set to in the YAML (weight=1.0 makes the weighted branch in
            # compute_aux_loss inert, but loss_type must ALSO be forced back
            # to "mse" -- smooth_l1 vs mse differ numerically even at
            # weight==1.0).
            _aux_pred_dict["risk_map_positive_weight"] = 1.0
            _aux_pred_dict["risk_map_loss_type"] = "mse"
            _aux_pred_dict["hazard_pos_weight"] = None
        self.aux_cfg = AuxPredConfig(_aux_pred_dict)
        self.aux_enabled = self.aux_cfg.enabled
        self.aux_beta = self.aux_cfg.loss_weight

        # STAGE 8: fixed physical-range observation normalization (ISOLATED
        # experimental feature -- default OFF, NOT enabled for phase2/both;
        # see obs_normalization.py's module docstring). Applied to `state`/
        # `next_state` right after sampling in train() and to `state_t` in
        # select_action() -- NOT wired into the aux branch's separate
        # get_last_state_history() temporal-context path, an explicit,
        # documented scope exclusion for this pass (see the Stage-8 report).
        onc = dict(self.hyperparameters.get("observation_normalization", {}) or {})
        self.obs_norm_cfg = ObsNormalizationConfig.from_dict(onc)
        self.obs_normalizer = None
        if self.obs_norm_cfg.enabled:
            if not env_obs_dim or not env_agent_dim:
                raise RuntimeError(
                    "observation_normalization.enabled=true requires env_obs_dim "
                    "and env_agent_dim (the trainer must pass them, same as "
                    "temporal_actor_context) so the per-frame LiDAR/agent scale "
                    "layout can be built correctly."
                )
            self.obs_normalizer = ObsNormalizer(
                self.obs_norm_cfg, state_dim=state_dim,
                obs_dim=int(env_obs_dim), agent_dim=int(env_agent_dim),
                history_len=int(onc.get("history_len", 1)),
                stack_agent_state=bool(onc.get("stack_agent_state", False)),
            )
        # AUX_PRED: action-conditioned variant (predict the same future-risk
        # target from z_t AND the upcoming action sequence).  Only valid when aux
        # is enabled; fail-fast on a contradictory config.
        self.aux_action_conditioned = bool(
            self.aux_enabled and self.aux_cfg.action_conditioned_aux
        )
        if self.aux_cfg.action_conditioned_aux and not self.aux_enabled:
            raise RuntimeError(
                "aux_prediction.action_conditioned_aux=true requires "
                "aux_prediction.enabled=true."
            )
        if self.aux_action_conditioned and self.aux_cfg.action_conditioned_steps < 1:
            raise RuntimeError(
                "aux_prediction.action_conditioned_steps must be >= 1 "
                f"(got {self.aux_cfg.action_conditioned_steps})."
            )
        # AUX_PRED (v2): aux-only temporal context. Same fail-fast contract as the
        # action-conditioned variant: it requires the shared encoder (enabled).
        self.aux_temporal_enabled = bool(
            self.aux_enabled and self.aux_cfg.temporal_enabled
        )
        if self.aux_cfg.temporal_enabled and not self.aux_enabled:
            raise RuntimeError(
                "aux_prediction.temporal_enabled=true requires "
                "aux_prediction.enabled=true."
            )
        if self.aux_temporal_enabled and self.aux_cfg.history_len < 1:
            raise RuntimeError(
                "aux_prediction.history_len must be >= 1 "
                f"(got {self.aux_cfg.history_len})."
            )

        # ----------------------------
        # TEMPORAL_ACTOR: compressed temporal feature on the actor/critic path.
        # ----------------------------
        # The env transports the stacked state (current 87 + appended scan
        # history). Instead of feeding that raw 327-D vector to one big shared
        # encoder, split it: a light MLP on the CURRENT 87-D state + a small
        # ScanTemporalEncoder on the scan history, fused back to a latent. This
        # keeps the heavy raw stack off the actor/critic main path and lets the
        # temporal strength be gated per curriculum stage (set_curriculum_stage).
        #
        # INDEPENDENT of aux supervision (deliberately decoupled so "add time
        # context to the actor" and "strengthen the aux head" can be ablated
        # separately): with aux_prediction.enabled=false the main SharedEncoder is
        # an identity, so the fused latent is [current(87) + temporal] -> 87 and
        # the temporal/fusion params still train from the CRITIC loss alone (no aux
        # head). With aux enabled the main encoder is a real 87->latent_dim MLP.
        # Requires only the env to be stacking (state_dim == current + history).
        # Default OFF.
        tcfg = dict(self.hyperparameters.get("temporal_actor_context", {}) or {})
        self.temporal_actor_enabled = bool(tcfg.get("enabled", False))
        self.temporal_stage_enable_from = int(tcfg.get("stage_enable_from", 0))
        self.current_stage = 0

        # Independent architecture switch: replace only the scan-history
        # compressor while keeping TemporalFusionEncoder's public/output
        # contract unchanged.
        _st_cfg = dict(self.hyperparameters.get("spatiotemporal_lidar", {}) or {})
        self.spatiotemporal_lidar_enabled = bool(_st_cfg.get("enabled", False))
        if self.spatiotemporal_lidar_enabled and not self.temporal_actor_enabled:
            raise RuntimeError(
                "spatiotemporal_lidar.enabled=true requires "
                "temporal_actor_context.enabled=true.")

        # ----------------------------
        # PHASE2: Critic-connected Action-Risk Head (default OFF, independent of
        # aux_prediction). action_risk_head.enabled builds the head + its
        # supervised loss; critic_risk_input.enabled additionally feeds the
        # head's (DETACHED) prediction into the critic's input -- a separate
        # switch so "train the head" and "let the critic use it" can be ablated
        # independently. critic_risk_input requires action_risk_head (nothing to
        # feed the critic otherwise) and, when on, changes the critic's input
        # width -> FRESH-RUN ONLY (an old checkpoint's critic will not strict-load).
        # ----------------------------
        _action_risk_dict = dict(self.hyperparameters.get("action_risk_head", {}) or {})
        if not self.risk_balanced_enabled:
            # RISK_BALANCE: same neutralisation as aux_cfg above, applied to
            # the Action-Risk Head's own (risk_dir, min_dist_dir) supervised
            # loss weighting.
            _action_risk_dict["pos_weight"] = 1.0
            _action_risk_dict["loss_type"] = "mse"
        self.action_risk_cfg = ActionRiskConfig(_action_risk_dict)
        self.action_risk_enabled = self.action_risk_cfg.enabled
        self.critic_risk_input_enabled = bool(
            dict(self.hyperparameters.get("critic_risk_input", {}) or {}).get(
                "enabled", False))
        if self.critic_risk_input_enabled and not self.action_risk_enabled:
            raise RuntimeError(
                "critic_risk_input.enabled=true requires "
                "action_risk_head.enabled=true."
            )
        # STAGE 3: gate derived from (current_stage, enable_from_stage) -- set
        # here from the stage-0 default so a trainer that never calls
        # set_curriculum_stage() (e.g. non-curriculum) still gets a defined,
        # correct gate instead of an undefined attribute. Recomputed on every
        # set_curriculum_stage() call below.
        self._action_risk_active = bool(
            self.action_risk_enabled and self.current_stage >= self.action_risk_cfg.enable_from_stage
        )
        if self.critic_risk_input_enabled:
            print(
                "[Agent] critic_risk_input.enabled=true: the critic's input "
                "width changes (+2 for the Action-Risk Head prediction) -- this "
                "is a FRESH-RUN-ONLY architecture change, an old checkpoint's "
                "critic will NOT strict-load."
            )

        # PHASE2: optional temporal context on the Action-Risk Head, reusing the
        # SAME compressed feature temporal_actor_context already computes for
        # the actor/critic (TemporalFusionEncoder.temporal_feature()) -- no new
        # feature extractor, no replay-buffer schema change. Config-validation
        # fail-fast only (no silent fallback): a wrong/incompatible config must
        # error immediately, not quietly train on the un-augmented head.
        if self.action_risk_cfg.use_temporal_context and not self.action_risk_enabled:
            raise RuntimeError(
                "action_risk_head.use_temporal_context=true requires "
                "action_risk_head.enabled=true."
            )
        self.action_risk_temporal_enabled = bool(
            self.action_risk_enabled and self.action_risk_cfg.use_temporal_context
        )
        if self.action_risk_temporal_enabled:
            if not self.temporal_actor_enabled:
                raise RuntimeError(
                    "action_risk_head.use_temporal_context=true requires "
                    "temporal_actor_context.enabled=true -- the Action-Risk Head "
                    "reuses the actor's compressed temporal feature and has no "
                    "fallback source. Enable temporal_actor_context or turn "
                    "use_temporal_context back off."
                )
            if self.action_risk_cfg.temporal_context_source != "actor":
                raise RuntimeError(
                    "action_risk_head.temporal_context_source="
                    f"'{self.action_risk_cfg.temporal_context_source}' is not "
                    "supported -- only 'actor' (reusing temporal_actor_context's "
                    "ScanTemporalEncoder feature) is implemented."
                )
            print(
                "[Agent] action_risk_head.use_temporal_context=true: the "
                "Action-Risk Head's input width changes (+temporal_feature_dim) "
                "-- an old action-risk-head checkpoint will NOT strict-load; "
                "tqc_io.load() falls back to a freshly-initialised head (and "
                "fresh critic-optimizer moments) while actor/critic/encoder/"
                "replay resume normally."
            )

        # Counterfactual multi-horizon head is independent of the legacy
        # selected-action head and critic_risk_input. It reuses the actor's
        # temporal feature when requested and adds a direct actor penalty.
        _cf_dict = dict(self.hyperparameters.get(
            "counterfactual_multi_horizon_risk", {}) or {})
        if not self.risk_balanced_enabled:
            _cf_dict["pos_weight"] = 1.0
            _cf_dict["loss_type"] = "mse"
        self.counterfactual_risk_cfg = CounterfactualRiskConfig(_cf_dict)
        self.counterfactual_risk_enabled = self.counterfactual_risk_cfg.enabled
        self._counterfactual_risk_active = bool(
            self.counterfactual_risk_enabled
            and self.current_stage >= self.counterfactual_risk_cfg.enable_from_stage)
        if self.counterfactual_risk_enabled:
            if self.counterfactual_risk_cfg.num_candidates < 1:
                raise RuntimeError(
                    "counterfactual_multi_horizon_risk.enabled=true requires "
                    "at least one candidate_actions entry.")
            if any(len(a) != action_dim
                   for a in self.counterfactual_risk_cfg.candidate_actions):
                raise RuntimeError(
                    "Every counterfactual candidate action must have action_dim="
                    f"{action_dim} values.")
            if any(abs(v) > 1.0
                   for a in self.counterfactual_risk_cfg.candidate_actions
                   for v in a):
                raise RuntimeError(
                    "Counterfactual candidate actions must be normalized to "
                    "[-1, 1].")
            if (self.counterfactual_risk_cfg.use_temporal_context
                    and not self.temporal_actor_enabled):
                raise RuntimeError(
                    "counterfactual_multi_horizon_risk.use_temporal_context="
                    "true requires temporal_actor_context.enabled=true.")
        # Actor-penalty warm-up/ramp: counts CF SUPERVISED updates that have
        # actually run (never a train() call where the head had no target),
        # persisted across checkpoint save/load (see tqc_io.py) so a resumed
        # run continues its ramp instead of restarting it. effective weight
        # is recomputed from this counter every train() call it applies in.
        self._cf_supervised_updates = 0
        self._cf_effective_actor_penalty_weight = 0.0
        # weighted_mean horizon aggregation: precompute the normalized weight
        # tensor once (CounterfactualRiskConfig already validated length/
        # non-negativity/sum>0 and normalized it to sum to 1).
        self._cf_horizon_weights = (
            torch.tensor(self.counterfactual_risk_cfg.horizon_weights,
                        dtype=torch.float32, device=self.device)
            if (self.counterfactual_risk_enabled
                and self.counterfactual_risk_cfg.actor_risk_aggregation
                == "weighted_mean")
            else None)

        # NOTE: replay_buffer.risk_meta / risk_balanced_sampling (self.
        # risk_balanced_enabled / self.store_risk_meta / self.
        # risk_balanced_ratios) are parsed EARLY, right after self.max_action
        # above -- required there so the weighted-loss neutralisation for
        # aux_cfg/action_risk_cfg can see risk_balanced_enabled before those
        # configs are constructed. Kept as a single read, not duplicated here.

        def _make_encoder():
            if not self.temporal_actor_enabled:
                return SharedEncoder(state_dim, self.aux_cfg)
            obs_dim = int(env_obs_dim if env_obs_dim is not None
                          else tcfg.get("obs_dim", 0))
            agent_dim = int(env_agent_dim if env_agent_dim is not None
                            else tcfg.get("agent_dim", 0))
            if obs_dim <= 0 or agent_dim <= 0:
                raise RuntimeError(
                    "temporal_actor_context.enabled=true needs the env obs/agent "
                    "dims; the trainer must pass env_obs_dim/env_agent_dim (or set "
                    "temporal_actor_context.obs_dim/agent_dim).")
            enc = TemporalFusionEncoder(
                state_dim, obs_dim, agent_dim,
                history_len=int(tcfg.get("history_len", 4)),
                stack_agent_state=bool(tcfg.get("stack_agent_state", False)),
                aux_cfg=self.aux_cfg,
                feature_dim=int(tcfg.get("temporal_feature_dim", 32)),
                encoder_type=(
                    "spatiotemporal" if self.spatiotemporal_lidar_enabled
                    else str(tcfg.get("encoder_type", "conv1d"))),
                angular_tokens=int(_st_cfg.get("angular_tokens", 10)),
                use_range_rate=bool(_st_cfg.get("use_range_rate", True)),
            )
            exp = enc.expected_state_dim()
            if exp != state_dim:
                raise RuntimeError(
                    f"temporal_actor_context expects state_dim={exp} "
                    f"(obs_dim={obs_dim}, agent_dim={agent_dim}, "
                    f"history_len={enc.history_len}, "
                    f"stack_agent_state={enc.stack_agent_state}) but the env "
                    f"reports state_dim={state_dim}. Make observation_time_context."
                    f"obs_frame_stack match temporal_actor_context.history_len.")
            return enc

        self.encoder = _make_encoder().to(self.device)
        self.encoder_target = _make_encoder().to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.checkpoint_encoder = _make_encoder().to(self.device)
        self.checkpoint_encoder.load_state_dict(self.encoder.state_dict())
        # STAGE 5: both are only ever Polyak-copied (encoder_target) or used
        # under torch.no_grad() for eval-time inference (checkpoint_encoder,
        # select_action) -- never backprop'd into directly. Permanently
        # freezing is defensive (a future forward call outside no_grad() can
        # never accidentally build a wasted graph) and matches encoder_target's
        # existing Polyak-only role; .data.copy_() is unaffected by
        # requires_grad either way.
        for p in self.encoder_target.parameters():
            p.requires_grad_(False)
        for p in self.checkpoint_encoder.parameters():
            p.requires_grad_(False)
        latent_dim = self.encoder.out_dim

        # AUX_PRED (v2): aux-only temporal context encoder (GRU over the shared
        # latents of the last history_len in-episode states). out_dim == 0 when
        # disabled, so the aux head trunk width is unchanged from v1. The
        # actor/critic NEVER consume this context -> the policy stays non-recurrent.
        self.temporal_encoder = (
            TemporalContextEncoder(latent_dim, self.aux_cfg).to(self.device)
            if self.aux_temporal_enabled else None
        )
        temporal_dim = self.temporal_encoder.out_dim if self.temporal_encoder else 0

        # AUX_PRED: training-only auxiliary head (dropped at inference).  The
        # action-conditioned head adds an action-sequence GRU; both emit the same
        # output dict so the loss (compute_aux_loss) is identical. temporal_dim>0
        # appends the temporal context to the head input only.
        if not self.aux_enabled:
            self.aux_head = None
        elif self.aux_action_conditioned:
            self.aux_head = ActionConditionedAuxHead(
                latent_dim, action_dim, self.aux_cfg, temporal_dim=temporal_dim
            ).to(self.device)
        else:
            self.aux_head = AuxiliaryHead(
                latent_dim, self.aux_cfg, temporal_dim=temporal_dim
            ).to(self.device)

        # PHASE2: Action-Risk Head (+ polyak-averaged target copy, used ONLY to
        # feed the TD-target critic call -- mirrors why encoder_target exists:
        # keep whatever extra signal the TARGET Q sees on a slow-moving copy for
        # stability). None when disabled.
        if self.action_risk_enabled:
            # PHASE2: temporal_dim>0 only when use_temporal_context=true (which
            # the fail-fast above already guaranteed implies self.encoder IS a
            # TemporalFusionEncoder, i.e. has .temporal.out_dim).
            _ar_temporal_dim = (
                self.encoder.temporal.out_dim
                if self.action_risk_temporal_enabled else 0
            )
            self.action_risk_head = ActionRiskHead(
                latent_dim, action_dim, self.action_risk_cfg,
                temporal_dim=_ar_temporal_dim).to(self.device)
            self.action_risk_head_target = ActionRiskHead(
                latent_dim, action_dim, self.action_risk_cfg,
                temporal_dim=_ar_temporal_dim).to(self.device)
            self.action_risk_head_target.load_state_dict(
                self.action_risk_head.state_dict())
            # STAGE 5: Polyak-only, same rationale as encoder_target above.
            for p in self.action_risk_head_target.parameters():
                p.requires_grad_(False)
        else:
            self.action_risk_head = None
            self.action_risk_head_target = None

        if self.counterfactual_risk_enabled:
            cf_temporal_dim = (
                self.encoder.temporal.out_dim
                if self.counterfactual_risk_cfg.use_temporal_context else 0)
            self.counterfactual_risk_head = CounterfactualMultiHorizonRiskHead(
                latent_dim, action_dim, self.counterfactual_risk_cfg,
                temporal_dim=cf_temporal_dim).to(self.device)
            self._counterfactual_candidates = torch.tensor(
                self.counterfactual_risk_cfg.candidate_actions,
                dtype=torch.float32, device=self.device)
        else:
            self.counterfactual_risk_head = None
            self._counterfactual_candidates = None

        # ----------------------------
        # Networks & Optimizers
        # ----------------------------
        # NOTE: actor/critic are now sized by the encoder latent dim.  With aux
        # disabled latent_dim == state_dim, so these are identical to baseline.
        self.actor = Actor(latent_dim, action_dim, self.actor_hdim, self.actor_activ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # PHASE2 (critic_risk_input): extra_dim=2 (risk_dir, min_dist_dir) when
        # enabled, else 0 -> byte-identical Critic input width to baseline.
        _critic_extra_dim = 2 if self.critic_risk_input_enabled else 0
        self.critic = Critic(
            latent_dim, action_dim, self.critic_hdim, self.critic_activ,
            self.n_quantiles, self.n_critics,
            residual=self.critic_residual, layernorm=self.critic_layernorm,
            residual_blocks=self.critic_residual_blocks,
            extra_dim=_critic_extra_dim,
        ).to(self.device)

        # AUX_PRED: a single optimizer updates critic + encoder + aux head from
        # (critic_loss + beta_aux * aux_loss).  With aux disabled it reduces to
        # the critic's parameters only (encoder has none), matching baseline.
        self.critic_optimizer = self._make_critic_optimizer()

        self.critic_target = Critic(
            latent_dim, action_dim, self.critic_hdim, self.critic_activ,
            self.n_quantiles, self.n_critics,
            residual=self.critic_residual, layernorm=self.critic_layernorm,
            residual_blocks=self.critic_residual_blocks,
            extra_dim=_critic_extra_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # STAGE 5: Polyak-only, same rationale as encoder_target above.
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Checkpoint actor (평가 시 use_checkpoint=True 경로)
        self.checkpoint_actor = Actor(latent_dim, action_dim, self.actor_hdim, self.actor_activ).to(self.device)
        # STAGE 5: only ever used under torch.no_grad() (select_action's
        # use_checkpoint branch) -- same defensive rationale as
        # checkpoint_encoder above.
        for p in self.checkpoint_actor.parameters():
            p.requires_grad_(False)

        # Temperature (α): auto / fixed
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
            self.log_ent_coef = torch.log(torch.tensor([init_value], device=self.device)).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.ent_coef_lr)
            self.ent_coef_auto = True
        else:
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef), device=self.device)
            self.ent_coef_auto = False

        # ----------------------------
        # Replay Buffer (TD7 호환 LAP)
        # ----------------------------
        self.replay_buffer = buffer.LAP(
            state_dim,
            action_dim,
            self.device,
            self.buffer_size,
            self.batch_size,
            self.max_action,
            # TQC actor and environment interface both operate in normalized
            # action space [-1, 1]. Keep replay-buffer actions unchanged.
            normalize_actions=False,
            prioritized=self.prioritized,
            # AUX_PRED: store the future-risk label alongside each transition.
            aux_dim=(self.aux_cfg.label_dim if self.aux_enabled else 0),
            # PHASE2: store the Action-Risk Head's (risk_dir, min_dist_dir) target.
            action_risk_dim=(2 if self.action_risk_enabled else 0),
            counterfactual_risk_dim=(
                self.counterfactual_risk_cfg.target_dim
                if self.counterfactual_risk_enabled else 0),
            executed_action_risk_dim=(
                self.counterfactual_risk_cfg.num_horizons
                if self.counterfactual_risk_enabled else 0),
            # AUX_PRED: track episode boundaries when EITHER the action-conditioned
            # aux (future-action lookup) OR the temporal context (backward
            # state-history lookup) is on, so neither walk crosses an episode.
            track_traj=(self.aux_action_conditioned or self.aux_temporal_enabled),
            # RISK_BALANCE: optional per-transition metadata + stratified
            # aux/action-risk supervision sampling (both default OFF).
            store_risk_meta=self.store_risk_meta,
            risk_balanced_enabled=self.risk_balanced_enabled,
            risk_balanced_ratios=self.risk_balanced_ratios,
        )

        # ----------------------------
        # Book-keeping
        # ----------------------------
        self.training_steps = 0
        # AUX_ABLATION: run seed for JSON metric stamping (set by the trainer).
        self.run_seed = None
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # TensorBoard Writer (항상 생성; log_dir 없으면 temp)
        self.log_dir = log_dir or ""
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir else SummaryWriter()
        except Exception:
            self.writer = SummaryWriter()

        # --- JSONL 로그 경로 설정 (여기가 추가) ---
        writer_dir = getattr(self.writer, "log_dir", None)
        base_dir = self.log_dir or writer_dir or os.getcwd()
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass

        # 기본 경로: <log_dir>/tqc_metrics.jsonl
        self.json_log_path = os.path.join(base_dir, "tqc_metrics.json")

        # STAGE 6: interval-configurable logging (default 1 -> log every
        # step, byte-identical to before). scalar_log_interval gates
        # TensorBoard add_scalar calls; json_log_interval gates whether a
        # JSON record is built for this step at all; json_flush_interval
        # (independent of json_log_interval) controls how many BUFFERED
        # records accumulate before a single physical open/write-all/close,
        # instead of one open/close per record. All three preserve the real
        # self.training_steps as the logged step number regardless of
        # interval -- sparse logging skips steps, it never renumbers them.
        self.scalar_log_interval = max(1, int(hyperparameters.get("scalar_log_interval", 1)))
        self.json_log_interval = max(1, int(hyperparameters.get("json_log_interval", 1)))
        self.json_flush_interval = max(1, int(hyperparameters.get("json_flush_interval", 1)))
        self._json_buffer = []

    def set_curriculum_stage(self, stage: int):
        """TEMPORAL_ACTOR / AUX_PRED / STAGE 3: notify the agent of the current
        curriculum stage so the temporal feature strength, aux loss weight and
        action-risk-head activation can ramp in with the curriculum WITHOUT
        changing any tensor shapes.

        - temporal_gain = 1.0 once stage >= stage_enable_from, else 0.0 (the
          temporal feature contributes nothing and gets no gradient at the easy
          early stages, and its forward pass is skipped entirely -- see
          TemporalFusionEncoder.forward). Applied to the online / target /
          checkpoint encoders so rollout, TD target and checkpoint-eval all use
          the same gain.
        - the aux loss weight follows aux_prediction.stagewise_loss_schedule
          (read in _current_aux_beta); train() skips aux_head's forward pass
          entirely when that weight is exactly 0 for the current step.
        - _action_risk_active = stage >= action_risk_head.enable_from_stage;
          train() skips action_risk_head's forward pass entirely (both call
          sites: its own supervised loss AND critic_risk_input's `extra`) when
          this is False, substituting a fixed all-zero (batch, 2) tensor for
          `extra` when critic_risk_input is enabled so the critic's input
          width never changes with stage. Re-derived from the CURRENT stage on
          every call (not latched), so demoting the stage re-disables it too --
          resume calls this from the restored curriculum_stage parameter, so
          the gate is always consistent with checkpoint state.
        Safe no-op for the legacy (non-fusion) encoder / disabled temporal /
        disabled action-risk head."""
        self.current_stage = int(stage)
        gain = 1.0 if int(stage) >= self.temporal_stage_enable_from else 0.0
        for enc in (self.encoder, self.encoder_target, self.checkpoint_encoder):
            if hasattr(enc, "set_temporal_gain"):
                enc.set_temporal_gain(gain)
        self._action_risk_active = bool(
            self.action_risk_enabled and self.current_stage >= self.action_risk_cfg.enable_from_stage
        )
        self._counterfactual_risk_active = bool(
            self.counterfactual_risk_enabled
            and self.current_stage >= self.counterfactual_risk_cfg.enable_from_stage
        )


    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        """상태로부터 정규화 action [-1, 1] 선택"""
        with torch.no_grad():

            state_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
            state_t  = torch.from_numpy(state_np).to(self.device)
            # STAGE 8 (isolated experimental, default OFF): same fixed-scale
            # normalization as train(), applied before either encoder.
            if self.obs_normalizer is not None:
                state_t = self.obs_normalizer.normalize(state_t)

            # AUX_PRED: encode the raw state first, then run the policy on the
            # latent.  At inference NO privileged info / aux head is used.
            if use_checkpoint:
                z = self.checkpoint_encoder(state_t)
                action = self.checkpoint_actor(z, deterministic=not use_exploration)
            else:
                z = self.encoder(state_t)
                action = self.actor(z, deterministic=not use_exploration)

            # Keep the policy output in normalized action space [-1, 1].
            # environment.py._map_action_to_twist() converts this into
            # physical [v, w] commands using actions_low/high.
            action = action.clamp(-1, 1)
            return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------ #
    #  AUX_PRED: read-only auxiliary-head inference for FORMAL evaluation   #
    #  (used by the trainer's eval loop; never touches the training path). #
    # ------------------------------------------------------------------ #
    @property
    def aux_eval_enabled(self) -> bool:
        """True when an auxiliary head exists and can be queried for eval."""
        return bool(self.aux_enabled and self.aux_head is not None)

    def aux_predict_eval(self, states, future_actions=None, valid_len=None,
                         state_history=None, hist_valid_len=None):
        """Run encoder + aux head on a batch of states (no grad) and return the
        prediction as NumPy, for the formal aux eval metrics.

        Parameters
        ----------
        states        : array-like [N, state_dim]
        future_actions: array-like [N, K, action_dim] or None.  Required (and
                        only used) when the head is action-conditioned; the same
                        boundary-safe alignment used in training applies — pass
                        only in-episode actions, zero-padded past the episode end.
        valid_len     : array-like [N] (long) in [1, K], number of leading
                        in-episode actions.  Required for action-conditioned.
        state_history : array-like [N, H, state_dim] or None.  REVERSE-time state
                        window (index 0 == current state) for the temporal
                        context; only used when the head is temporal.  When
                        omitted on a temporal head, a length-1 history (current
                        state only) is used — a valid but minimal context.
        hist_valid_len: array-like [N] (long) in [1, H], leading-valid history
                        length; defaults to all-1 when state_history is omitted.

        Returns
        -------
        dict with keys "risk_map" [N, H*K] and (if the head emits it) "min_dist"
        [N, H], all float32 NumPy; or None when aux is disabled / no head.
        """
        if not self.aux_eval_enabled:
            return None
        with torch.no_grad():
            s_np = np.asarray(states, dtype=np.float32)
            if s_np.ndim == 1:
                s_np = s_np[None, :]
            s_np = s_np.reshape(s_np.shape[0], -1)
            s_t = torch.from_numpy(s_np).to(self.device)
            z = self.encoder(s_t)

            # AUX_PRED (v2): build the eval temporal context (None on a v1 head).
            temporal_ctx = self._temporal_ctx_for_eval(z, state_history, hist_valid_len)

            if self.aux_action_conditioned:
                if future_actions is None or valid_len is None:
                    return None
                fa = torch.from_numpy(
                    np.asarray(future_actions, dtype=np.float32)
                ).to(self.device)
                vl = torch.from_numpy(
                    np.asarray(valid_len, dtype=np.int64)
                ).to(self.device)
                out = self.aux_head(z, fa, vl, temporal_ctx=temporal_ctx)
            else:
                out = self.aux_head(z, temporal_ctx=temporal_ctx)
            res = {"risk_map": out["risk_map"].cpu().numpy().astype(np.float32)}
            if "min_dist" in out:
                res["min_dist"] = out["min_dist"].cpu().numpy().astype(np.float32)
            return res

    def _temporal_ctx_for_eval(self, z, state_history, hist_valid_len):
        """AUX_PRED (v2): temporal context for an eval batch, or None on a v1
        head. Mirrors training: encode the (reverse-time) history window through
        the shared encoder and summarise it with the temporal GRU. Falls back to
        a length-1 (current-state-only) window when no history is supplied."""
        if self.temporal_encoder is None:
            return None
        b = z.shape[0]
        n = self.aux_cfg.history_len
        if state_history is None:
            z_seq = z.new_zeros((b, n, z.shape[1]))
            z_seq[:, 0, :] = z
            vl = torch.ones(b, dtype=torch.long, device=self.device)
        else:
            h_np = np.asarray(state_history, dtype=np.float32)
            h_t = torch.from_numpy(h_np).to(self.device)        # (b, n, S)
            z_seq = self.encoder(h_t.reshape(b * n, -1)).reshape(b, n, -1)
            vl = (torch.ones(b, dtype=torch.long, device=self.device)
                  if hist_valid_len is None else
                  torch.from_numpy(np.asarray(hist_valid_len, dtype=np.int64)).to(self.device))
        return self.temporal_encoder(z_seq, vl)

    def save(self, directory, filename):
        """Save model parameters (delegates to tqc_io.save)."""
        tqc_io.save(self, directory, filename)

    def load(
        self,
        directory,
        filename,
        *,
        load_optimizer_state=True,
        load_replay_buffer=True,
    ):
        """Restore model parameters (delegates to tqc_io.load)."""
        tqc_io.load(
            self,
            directory,
            filename,
            load_optimizer_state=load_optimizer_state,
            load_replay_buffer=load_replay_buffer,
        )

    def load_encoder_for_inference(self, actor_path):
        """AUX_PRED: restore the shared encoder for actor-only inference.

        Delegates to tqc_io.load_encoder_for_inference. Returns True when the
        encoder is ready (loaded, or not needed for a baseline identity encoder),
        False when a required aux encoder file is missing."""
        return tqc_io.load_encoder_for_inference(self, actor_path)
