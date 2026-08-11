import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, time, json


import drl_agent.rl.replay.buffer as buffer

# AUX_PRED: auxiliary prediction network (shared encoder + future-risk head).
# All auxiliary logic lives in these modules; with aux_prediction.enabled=false
# the SharedEncoder is a parameter-free identity passthrough and the baseline
# TQC behaviour is reproduced exactly.
from drl_agent.rl.networks.aux_prediction import (
    AuxPredConfig, SharedEncoder, AuxiliaryHead, ActionConditionedAuxHead,
)
from drl_agent.rl.networks.aux_losses import compute_aux_loss
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
    ActionRiskConfig, ActionRiskHead, weighted_action_risk_loss,
    CounterfactualRiskConfig, CounterfactualMultiHorizonRiskHead,
    weighted_counterfactual_risk_loss,
)

# STAGE 8: fixed physical-range observation normalization (ISOLATED
# experimental feature, default OFF -- see obs_normalization.py docstring).
from drl_agent.rl.networks.obs_normalization import ObsNormalizationConfig, ObsNormalizer


# Network definitions + TQC loss live in drl_agent.rl.networks.tqc; checkpoint
# I/O in drl_agent.rl.checkpointing.tqc_io. Re-imported here so callers of
# this module can keep using `Actor`, `Critic`, `quantile_huber_loss` directly.
from drl_agent.rl.networks.tqc import Actor, Critic, quantile_huber_loss
import drl_agent.rl.checkpointing.tqc_io as tqc_io


@contextlib.contextmanager
def _frozen_params(*modules):
    """STAGE 5: temporarily set requires_grad_(False) on every parameter of the
    given module(s) for a single forward call, guaranteed to restore True even
    if the wrapped block raises (exception-safe, unlike a bare set-False/call/
    set-True sequence). Used so a "frozen submodule, live input" forward pass
    (e.g. the critic during the actor update) never picks up a gradient into
    its OWN parameters from that pass's backward(), while gradient still flows
    through to whatever tensor was fed as input (the action, in the critic's
    case) -- exactly mirroring why action_risk_head is frozen the same way for
    the actor-update's extra_pi computation."""
    params = [p for m in modules for p in m.parameters()]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in params:
            p.requires_grad_(True)


def _polyak_update_foreach(src_params, tgt_params, tau: float):
    """STAGE 5: Polyak (exponential moving average) target update, batched via
    torch._foreach_* instead of a Python for-loop over individual parameter
    tensors -- one CUDA kernel launch per op across ALL parameters instead of
    one per parameter, mathematically identical to the original per-parameter
    ``target.data.copy_(tau * src.data + (1 - tau) * target.data)`` loop (see
    test_foreach_polyak_matches_per_parameter_loop_numerically). Wrapped in
    torch.no_grad() for clarity/defensiveness -- the .data-based ops already
    bypass autograd either way, so this changes no numerics, only makes the
    "never build a graph here" intent explicit."""
    src_list = list(src_params)
    tgt_list = list(tgt_params)
    if not src_list:
        return
    with torch.no_grad():
        src_data = [p.data for p in src_list]
        tgt_data = [p.data for p in tgt_list]
        scaled_src = torch._foreach_mul(src_data, tau)
        torch._foreach_mul_(tgt_data, 1.0 - tau)
        torch._foreach_add_(tgt_data, scaled_src)


class Agent(object):
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

    # JSON 라인 기록 헬퍼
    def _json_log(self, step: int, **metrics):
        path = getattr(self, "json_log_path", None)
        if not path:
            return

        rec = {"step": int(step), "time": float(time.time())}
        # AUX_ABLATION: stamp run identity on every record so tqc_metrics.json can
        # be grouped by seed / aux on-off without a separate join.  aux_enabled /
        # aux_version are always present (null-safe); seed only when known.
        if getattr(self, "run_seed", None) is not None:
            rec["seed"] = int(self.run_seed)
        rec["aux_enabled"] = int(bool(getattr(self, "aux_enabled", False)))
        rec["aux_version"] = int(getattr(self.aux_cfg, "version", 0)) if getattr(self, "aux_cfg", None) else 0
        for k, v in metrics.items():
            try:
                val = float(v)
                if np.isfinite(val):
                    rec[k] = val
            except Exception:
                continue

        self._json_buffer.append(rec)
        if len(self._json_buffer) >= self.json_flush_interval:
            self._flush_json_buffer()

    def _flush_json_buffer(self):
        """STAGE 6: physically write every buffered JSON record in ONE
        open/write-all/close instead of one open/close per record."""
        if not self._json_buffer:
            return
        path = self.json_log_path
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for rec in self._json_buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._json_buffer.clear()

    def flush_logs(self):
        """STAGE 6: flush any buffered JSON records + the TensorBoard writer
        on demand. Call on ALL exit paths (normal completion, KeyboardInterrupt,
        environment-service failure, any other exception, checkpoint-on-
        failure) so a run that stops between flush intervals never silently
        loses its most recent metrics."""
        self._flush_json_buffer()
        if self.writer:
            self.writer.flush()

    @staticmethod
    def prep_hyperparameters(hyperparameters):
        """Pre-process hyperparameters: 문자열 활성화 → 함수, 기본값 주입"""
        hp = dict(hyperparameters or {})
        activation_functions = {
            "elu": F.elu,
            "relu": F.relu,
        }

        # 활성화 함수 문자열이면 함수로 매핑
        if "actor_activ" in hp and isinstance(hp["actor_activ"], str):
            hp["actor_activ"] = activation_functions.get(hp["actor_activ"].lower(), F.relu)
        if "critic_activ" in hp and isinstance(hp["critic_activ"], str):
            hp["critic_activ"] = activation_functions.get(hp["critic_activ"].lower(), F.elu)

        # 자주 쓰는 기본값(누락 방지)
        hp.setdefault("discount", 0.99)
        hp.setdefault("batch_size", 256)
        hp.setdefault("buffer_size", 1_000_000)
        hp.setdefault("actor_lr", 3e-4)
        hp.setdefault("critic_lr", 3e-4)
        hp.setdefault("n_quantiles", 25)
        hp.setdefault("n_critics", 2)
        hp.setdefault("top_quantiles_to_drop_per_net", 2)
        hp.setdefault("tau", 0.005)
        hp.setdefault("target_update_interval", 1)
        hp.setdefault("ent_coef", "auto")
        hp.setdefault("ent_coef_lr", 3e-4)
        hp.setdefault("actor_hdim", 256)
        hp.setdefault("critic_hdim", 256)
        hp.setdefault("reset_weight", 0.9)
        hp.setdefault("steps_before_checkpointing", 40000)
        hp.setdefault("max_eps_when_checkpointing", 50)
        # prioritized 관련 키는 사용 시에만 읽음

        return hp

    def _make_critic_optimizer(self):
        """AUX_PRED: build the Adam optimizer over the shared trunk (critic +
        encoder + aux head).  Used at construction AND to reset the moments when
        an aux-head architecture change on resume makes the saved moments stale.

        STAGE 8 (isolated experimental feature, default OFF): when
        optimizer_groups.enabled is set, builds ONE Adam instance with
        SEPARATE param_groups per component (critic/encoder/aux_head/
        action_risk_head) instead of one flat parameter list -- each group
        can have its own LR (e.g. a smaller encoder_lr than the head LRs),
        while still being a single optimizer (one .zero_grad()/.step() pair,
        unchanged call sites in train()). Any component whose LR isn't
        explicitly configured falls back to critic_lr, so enabling the
        feature WITHOUT setting different LRs is behaviorally a no-op
        (identical effective LR everywhere, just split across groups)."""
        og = dict(self.hyperparameters.get("optimizer_groups", {}) or {})
        use_groups = bool(og.get("enabled", False))

        components = [("critic", list(self.critic.parameters()))]
        if self.encoder.has_params():
            components.append(("encoder", list(self.encoder.parameters())))
        if self.aux_head is not None:
            components.append(("aux_head", list(self.aux_head.parameters())))
        # AUX_PRED (v2): the temporal encoder is part of the aux trunk; its grads
        # flow with critic + aux loss (never the actor) -> same optimizer group.
        if getattr(self, "temporal_encoder", None) is not None:
            components.append(("aux_head", list(self.temporal_encoder.parameters())))
        # PHASE2: the Action-Risk Head trains from its own supervised loss added
        # into the SAME trunk loss (critic + beta_aux*aux + risk_beta*action_risk)
        # -> same optimizer group. The TARGET copy is never optimized directly
        # (polyak-updated only, like critic_target/encoder_target).
        if getattr(self, "action_risk_head", None) is not None:
            components.append(("action_risk_head", list(self.action_risk_head.parameters())))
        if getattr(self, "counterfactual_risk_head", None) is not None:
            components.append((
                "counterfactual_risk_head",
                list(self.counterfactual_risk_head.parameters())))

        if not use_groups:
            trunk_params = [p for _, params in components for p in params]
            return torch.optim.Adam(trunk_params, lr=self.critic_lr)

        lr_key = {"critic": "critic_lr", "encoder": "encoder_lr",
                  "aux_head": "aux_head_lr", "action_risk_head": "action_risk_head_lr",
                  "counterfactual_risk_head": "counterfactual_risk_head_lr"}
        param_groups = []
        merged = {}
        for name, params in components:
            merged.setdefault(name, []).extend(params)
        for name, params in merged.items():
            lr = float(og.get(lr_key[name], self.critic_lr))
            param_groups.append({"params": params, "lr": lr})
        return torch.optim.Adam(param_groups, lr=self.critic_lr)

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

    def reset_counterfactual_penalty_schedule(self):
        """Reset actor-penalty warm-up after an action-contract change.

        A curriculum promotion that also resets the replay buffer (e.g.
        stage 5 unsealing the yield channel -- see train_tqc_curriculum.py's
        reset_buffer_on_promote_to) throws away every transition the CF head
        was supervised on. Without this call, self._cf_supervised_updates
        keeps counting from before the reset, so a head that has ALREADY
        cleared warmup+ramp under the OLD contract would apply a full-weight
        actor penalty from its very first update on the NEW (off-contract-
        free) data -- exactly the "untrained head steers the actor" failure
        this schedule exists to avoid.

        Deliberately does NOT touch counterfactual_risk_head's parameters:
        the head's learned dynamic-risk representation (avoid nearby humans,
        walls, etc.) is still largely valid across the contract change -- only
        the ACTOR PENALTY needs to re-earn trust via a fresh warm-up/ramp
        against post-reset data, not the head's whole knowledge.

        No-op when the feature is disabled (nothing to reset)."""
        if not self.counterfactual_risk_enabled:
            return
        self._cf_supervised_updates = 0
        self._cf_effective_actor_penalty_weight = 0.0

    def reset_replay_for_action_contract_change(self):
        """Clear off-contract replay and re-arm the optional CF penalty."""
        self.replay_buffer.reset()
        self.reset_counterfactual_penalty_schedule()

    def _current_aux_beta(self):
        """AUX_PRED: effective trunk-level aux weight at this step.

        If a per-stage schedule is configured (aux_prediction.stagewise_loss_
        schedule) it takes precedence: beta = schedule[clamp(current_stage)], so
        aux supervision ramps in with the curriculum. Otherwise it linearly ramps
        0 -> loss_weight over aux_beta_warmup_steps (a noisy, freshly-initialised
        aux head does not perturb early critic learning), or the constant
        loss_weight when warmup is disabled (== 0).

        ``train()`` increments ``training_steps`` BEFORE the aux block runs, so the
        first update has ``training_steps == 1``; the ``-1`` makes that first
        update's beta exactly 0 (a true 0 -> loss_weight ramp) and reaches the
        full weight after ``w`` updates (at ``training_steps == w + 1``)."""
        sched = self.aux_cfg.stagewise_loss_schedule
        if sched:
            return float(sched[min(self.current_stage, len(sched) - 1)])
        w = self.aux_cfg.aux_beta_warmup_steps
        if w <= 0:
            return self.aux_beta
        return self.aux_beta * min(1.0, max(0, self.training_steps - 1) / float(w))

    def _compute_temporal_ctx(self, indices=None):
        """AUX_PRED (v2): temporal context for the given batch (default: the
        last sample()'s indices), or None.

        Walks the replay buffer backward (boundary-safe) for the last
        history_len in-episode states, encodes them through the SHARED encoder
        (so the temporal aux loss also shapes the encoder) and summarises the
        latent window with the temporal GRU. Returns (context, hist_valid_len) or
        None when temporal is off / history is unavailable. RISK_BALANCE: pass
        ``indices=ind_rb`` to align this with the risk-balanced batch instead."""
        if self.temporal_encoder is None:
            return None
        hist = self.replay_buffer.get_last_state_history(self.aux_cfg.history_len, indices=indices)
        if hist is None:
            return None
        hist_states, hist_valid = hist                 # (B, N, S), (B,)
        b, n, s = hist_states.shape
        z_seq = self.encoder(hist_states.reshape(b * n, s)).reshape(b, n, -1)
        return self.temporal_encoder(z_seq, hist_valid), hist_valid

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

    def train(self):
        """Train the agent for one step"""
        self.training_steps += 1

        # Sample batch from replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample()
        # STAGE 8 (isolated experimental, default OFF): fixed physical-range
        # normalization applied identically to state and next_state, right
        # before either reaches an encoder.
        if self.obs_normalizer is not None:
            state = self.obs_normalizer.normalize(state)
            next_state = self.obs_normalizer.normalize(next_state)
        # AUX_PRED: matching auxiliary targets for this batch (None if disabled).
        aux_target = self.replay_buffer.get_last_aux()

        # AUX_PRED: encode once.  `z` keeps the graph (critic + aux back-prop
        # into the encoder); `z_actor` is detached so the actor / temperature
        # updates never flow gradients into the encoder.
        z = self.encoder(state)
        z_actor = z.detach()

        # RISK_BALANCE: a SEPARATE, stratified batch (uniform / human-risk /
        # collision pools) used ONLY for the aux / action-risk SUPERVISED loss
        # below -- the critic loss above/below always uses the primary
        # uniform/prioritized `state`/`action` batch, untouched. None (feature
        # off, or the buffer is still empty) -> both use_rb_* flags below are
        # False and every block falls back to its ORIGINAL primary-batch
        # computation, byte-identical to before this feature existed.
        # STAGE 3/AUX_PRED: compute the effective aux beta BEFORE deciding
        # whether to draw the balanced batch below -- beta==0 (aux loss
        # disabled this stage / still in its warmup ramp) already skips the
        # aux loss entirely further down (see the beta != 0.0 check further
        # below), so drawing+encoding state_rb here would be pure waste: an
        # extra forward pass that no loss ever reads. _current_aux_beta() is
        # side-effect-free (reads only stage/step/config), so computing it
        # here changes no numerics.
        #
        # need_rb_for_{aux,risk} mirror the use_rb_* conditions below EXACTLY,
        # but computed BEFORE calling sample_risk_balanced() -- that call
        # itself rescans the ENTIRE risk_meta array to rebuild the human-risk
        # / collision pools every time it's invoked (see buffer.py's
        # sample_risk_balanced), so calling it on a stage/step where NEITHER
        # consumer would use the result (e.g. stage 0-2, before aux_beta or
        # action_risk_head.enable_from_stage kick in) wastes a full-buffer
        # scan every train() call for nothing.
        beta = self._current_aux_beta()
        need_rb_for_aux = (
            self.risk_balanced_enabled and self.aux_enabled and self.aux_head is not None
            and beta != 0.0
        )
        need_rb_for_risk = (
            self.risk_balanced_enabled and self.action_risk_enabled and self._action_risk_active
        )
        need_rb_for_cf = (
            self.risk_balanced_enabled and self.counterfactual_risk_enabled
            and self._counterfactual_risk_active
        )
        ind_rb = (
            self.replay_buffer.sample_risk_balanced()
            if (need_rb_for_aux or need_rb_for_risk or need_rb_for_cf) else None
        )
        use_rb_for_aux = ind_rb is not None and need_rb_for_aux
        use_rb_for_risk = ind_rb is not None and need_rb_for_risk
        use_rb_for_cf = ind_rb is not None and need_rb_for_cf
        z_rb = action_rb = state_rb = None
        risk_balance_logs = {}
        if use_rb_for_aux or use_rb_for_risk or use_rb_for_cf:
            state_rb, action_rb, _ns_rb, _r_rb, _nd_rb = self.replay_buffer.get_batch_by_indices(ind_rb)
            if self.obs_normalizer is not None:
                state_rb = self.obs_normalizer.normalize(state_rb)
            z_rb = self.encoder(state_rb)
            risk_balance_logs.update(
                {f"risk_balance/sampled_{k}": v
                 for k, v in self.replay_buffer.describe_risk_meta_fractions(ind_rb).items()}
            )

        """******************************************
        ** Entropy Coefficient Update (if auto)
        ******************************************"""
        if self.ent_coef_auto:
            with torch.no_grad():
                _, log_prob = self.actor.action_log_prob(z_actor)

            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
        else:
            ent_coef = self.ent_coef_tensor

        """******************************************
        ** Critic Update
        ******************************************"""
        with torch.no_grad():
            # AUX_PRED: target path runs through the target encoder.
            z_next = self.encoder_target(next_state)
            # Sample actions for next states
            next_actions, next_log_prob = self.actor.action_log_prob(z_next)

            # PHASE2 (critic_risk_input): action-risk prediction for the TD
            # target's (z_next, next_actions) pair, via the polyak-averaged
            # TARGET head (mirrors encoder_target -- keeps whatever extra signal
            # feeds the target Q on a slow-moving copy). Already under
            # torch.no_grad() so no explicit .detach() is needed.
            # STAGE 3: below action_risk_head.enable_from_stage, skip the
            # forward pass entirely and feed the critic a fixed all-zero
            # (batch, 2) tensor instead -- extra_dim never changes with stage.
            extra_next = None
            if self.critic_risk_input_enabled:
                if self._action_risk_active:
                    # PHASE2 (temporal): paired with z_next -> from the TARGET
                    # encoder on next_state, mirroring z_next's own origin.
                    ar_temporal_next = (
                        self.encoder_target.temporal_feature(next_state)
                        if self.action_risk_temporal_enabled else None
                    )
                    extra_next = self.action_risk_head_target(
                        z_next, next_actions, temporal_feature=ar_temporal_next)
                else:
                    extra_next = torch.zeros(
                        z_next.shape[0], self.action_risk_head_target.out.out_features,
                        device=self.device)

            # Get target quantiles
            next_quantiles = self.critic_target(z_next, next_actions, extra=extra_next)  # [B, n_critics, n_quantiles]

            # Sort and truncate quantiles
            batch_size = state.shape[0]
            next_quantiles_flat = next_quantiles.reshape(batch_size, -1)
            next_quantiles_sorted, _ = torch.sort(next_quantiles_flat, dim=1)

            # Drop top quantiles
            n_target_quantiles = self.n_critics * self.n_quantiles - self.top_quantiles_to_drop_per_net * self.n_critics
            next_quantiles_truncated = next_quantiles_sorted[:, :n_target_quantiles]

            # Compute target with entropy term
            target_quantiles = reward + not_done * self.discount * (
                next_quantiles_truncated - ent_coef * next_log_prob.view(-1, 1)
            )
            target_quantiles = target_quantiles.unsqueeze(1)  # [B, 1, n_target_quantiles]

        # PHASE2: Action-Risk Head prediction for the STORED (z, action) pair.
        # Computed once (with graph) and reused: `.detach()` for the critic's
        # `extra` input (critic loss must never backprop into this head -- see
        # action_risk_head.py's gradient-rule docstring), the SAME tensor
        # (still attached) for the supervised loss further below.
        # PHASE2 (temporal): ar_temporal_cur also gets REUSED (detached) for the
        # actor-update call further below -- same state batch, same encoder ->
        # identical feature, no need to recompute it a third time.
        # STAGE 3: below action_risk_head.enable_from_stage, skip this forward
        # pass entirely (ar_pred_cur stays None -> the supervised loss below is
        # naturally skipped too) -- extra_cur is filled with a fixed all-zero
        # (batch, 2) tensor when critic_risk_input is enabled, same as extra_next.
        ar_pred_cur = None
        ar_temporal_cur = None
        if self.action_risk_enabled and self.action_risk_head is not None and self._action_risk_active:
            if self.action_risk_temporal_enabled:
                ar_temporal_cur = self.encoder.temporal_feature(state)
            ar_pred_cur = self.action_risk_head(
                z, action, temporal_feature=ar_temporal_cur)
        if self.critic_risk_input_enabled:
            extra_cur = (
                ar_pred_cur.detach() if ar_pred_cur is not None
                else torch.zeros(z.shape[0], self.action_risk_head.out.out_features,
                                  device=self.device)
            )
        else:
            extra_cur = None

        # Get current quantiles (encoder latent keeps its graph here)
        current_quantiles = self.critic(z, action, extra=extra_cur)  # [B, n_critics, n_quantiles]

        # Compute critic loss
        critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)

        # AUX_PRED: future-risk prediction loss; gradients flow into the shared
        # encoder together with the critic loss (encoder = critic + beta*aux).
        # STAGE 3: beta computed FIRST and the WHOLE block (aux head forward,
        # aux-temporal-context encoder forward, action-conditioned future-
        # action lookup) is skipped when it's exactly 0 -- not just the loss
        # contribution zeroed after computing it. beta==0 covers BOTH an
        # explicit aux_prediction.stagewise_loss_schedule entry of 0.0 for the
        # current stage AND the pre-warmup window (aux_beta_warmup_steps not
        # yet elapsed) -- both cases already contribute nothing to the loss,
        # so skipping the compute changes no numerics, only wasted work.
        aux_logs = dict(risk_balance_logs)
        total_trunk_loss = critic_loss
        # beta already computed above (before the balanced-batch draw) so that
        # use_rb_for_aux can skip the wasted encoder forward when beta == 0.
        # RISK_BALANCE: when a balanced batch is available, the aux SUPERVISED
        # loss trains on it (z_rb/action_rb/its own aux target) INSTEAD OF the
        # primary uniform batch -- the encoder still gets exactly one aux
        # gradient contribution per train() call, just sourced from the
        # stratified batch. Falls back to the primary z/aux_target below
        # whenever use_rb_for_aux is False (feature off, or no balanced draw
        # this step), which is byte-identical to the pre-existing behaviour.
        _aux_z = z_rb if use_rb_for_aux else z
        _aux_target_src = (
            self.replay_buffer.get_last_aux(ind_rb) if use_rb_for_aux else aux_target
        )
        _aux_indices = ind_rb if use_rb_for_aux else None
        if (self.aux_enabled and self.aux_head is not None and _aux_target_src is not None
                and beta != 0.0):
            # AUX_PRED (v2): aux-only temporal context (recent in-episode state
            # history). None when temporal is off -> the head ignores it and the
            # v1 path is unchanged. Shares the encoder graph so it shapes E_psi.
            temporal_ctx = None
            tctx = self._compute_temporal_ctx(indices=_aux_indices)
            if tctx is not None:
                temporal_ctx, hist_valid = tctx
                aux_logs["aux/hist_len_mean"] = float(hist_valid.float().mean().item())

            if self.aux_action_conditioned:
                # Same target L_i (future risk from s_i), but conditioned on the
                # upcoming in-episode action sequence [a_i, .., a_{i+K-1}].
                fa = self.replay_buffer.get_last_future_actions(
                    self.aux_cfg.action_conditioned_steps, indices=_aux_indices)
                aux_pred = None
                if fa is not None:
                    future_actions, valid_len = fa
                    aux_pred = self.aux_head(
                        _aux_z, future_actions, valid_len, temporal_ctx=temporal_ctx)
                    aux_logs["aux/valid_len_mean"] = float(valid_len.float().mean().item())
            else:
                aux_pred = self.aux_head(_aux_z, temporal_ctx=temporal_ctx)

            if aux_pred is not None:
                aux_loss, _logs = compute_aux_loss(
                    aux_pred, _aux_target_src, self.aux_cfg, self.device
                )
                aux_logs.update(_logs)
                total_trunk_loss = critic_loss + beta * aux_loss
                aux_logs["aux/beta"] = float(beta)
                if use_rb_for_aux:
                    aux_logs["risk_balance/aux_uses_balanced_batch"] = 1.0

        # PHASE2: Action-Risk Head supervised loss (weighted MSE/SmoothL1
        # against the PRIVILEGED GT target stored in the buffer). This is the
        # head's ONLY gradient path -- added into the SAME trunk loss as the
        # aux loss above, independent of whether critic_risk_input is on (the
        # head can be trained standalone, condition 3 of the 4-way experiment
        # matrix). RISK_BALANCE: when a balanced batch is available, this
        # SUPERVISED loss trains on a FRESH forward pass over (z_rb, action_rb)
        # instead of reusing ar_pred_cur -- ar_pred_cur itself is untouched and
        # still exclusively feeds critic_risk_input's extra_cur / the actor
        # update's extra_pi below, so switching the loss source here changes
        # NOTHING about the critic-feed / actor-gradient channel.
        action_risk_loss_val = 0.0
        action_risk_logs = dict(risk_balance_logs) if use_rb_for_risk else {}
        if self.action_risk_enabled and ar_pred_cur is not None:
            if use_rb_for_risk:
                ar_temporal_rb = (
                    self.encoder.temporal_feature(state_rb)
                    if self.action_risk_temporal_enabled else None
                )
                ar_pred_src = self.action_risk_head(
                    z_rb, action_rb, temporal_feature=ar_temporal_rb)
                ar_target = self.replay_buffer.get_last_action_risk(ind_rb)
                action_risk_logs["risk_balance/action_risk_uses_balanced_batch"] = 1.0
            else:
                ar_pred_src = ar_pred_cur
                ar_target = self.replay_buffer.get_last_action_risk()
            if ar_target is not None:
                action_risk_loss, _ar_logs = weighted_action_risk_loss(
                    ar_pred_src, ar_target, self.action_risk_cfg)
                total_trunk_loss = total_trunk_loss + self.action_risk_cfg.loss_weight * action_risk_loss
                action_risk_loss_val = float(action_risk_loss.detach().item())
                action_risk_logs["action_risk/loss"] = action_risk_loss_val
                action_risk_logs.update(_ar_logs)

        # Counterfactual supervision: TWO terms trained from the SAME head/
        # encoder call contract [z, action(, temporal)] -> (B, H):
        #   1. fixed_candidate_loss: every configured candidate action at once
        #      for each replay state, matching the env's (candidate,horizon)
        #      target matrix -- the fixed candidates themselves are never
        #      executed or inserted into the core TQC batch.
        #   2. executed_action_loss: the REPLAY-STORED action (the batch's own
        #      `action`/`action_rb`, i.e. what was actually executed) against
        #      its own multi-horizon target (computed pre-motion by the env
        #      for that exact transition) -- this is what lets the head score
        #      continuous actions it was never shown as a fixed candidate,
        #      closing the fixed-candidate-only generalization gap.
        # cf_supervised_ran gates BOTH the warm-up counter increment AND the
        # actor penalty below: a step with no target (buffer not yet filled /
        # stage-gated off) must neither advance the ramp nor apply a penalty
        # from an untrained head.
        counterfactual_logs = {}
        cf_temporal_cur = None
        cf_supervised_ran = False
        if (self.counterfactual_risk_enabled
                and self.counterfactual_risk_head is not None
                and self._counterfactual_risk_active):
            cf_z = z_rb if use_rb_for_cf else z
            cf_action = action_rb if use_rb_for_cf else action
            cf_state = state_rb if use_rb_for_cf else state
            cf_indices = ind_rb if use_rb_for_cf else None
            cf_target = self.replay_buffer.get_last_counterfactual_risk(
                cf_indices)
            cf_executed_target = self.replay_buffer.get_last_executed_action_risk(
                cf_indices)
            if cf_target is not None and cf_executed_target is not None:
                batch_n = cf_z.shape[0]
                cand_n = self.counterfactual_risk_cfg.num_candidates
                horizon_n = self.counterfactual_risk_cfg.num_horizons
                candidates = self._counterfactual_candidates.unsqueeze(0).expand(
                    batch_n, -1, -1)
                z_candidates = cf_z.unsqueeze(1).expand(
                    -1, cand_n, -1).reshape(batch_n * cand_n, -1)
                temporal_candidates = None
                cf_temporal = None
                if self.counterfactual_risk_cfg.use_temporal_context:
                    cf_temporal = self.encoder.temporal_feature(cf_state)
                    temporal_candidates = cf_temporal.unsqueeze(1).expand(
                        -1, cand_n, -1).reshape(batch_n * cand_n, -1)
                    # Cache the PRIMARY-batch feature before the trunk update
                    # so the later actor call pairs z_actor with a feature from
                    # the same encoder weights. The balanced supervised batch
                    # has a different state batch and therefore needs a second
                    # primary forward here.
                    cf_temporal_cur = (
                        self.encoder.temporal_feature(state)
                        if use_rb_for_cf else cf_temporal)
                cf_pred = self.counterfactual_risk_head(
                    z_candidates, candidates.reshape(batch_n * cand_n, -1),
                    temporal_feature=temporal_candidates).reshape(
                        batch_n, cand_n, horizon_n)
                cf_target = cf_target.reshape(batch_n, cand_n, horizon_n)
                fixed_loss, fixed_loss_logs = weighted_counterfactual_risk_loss(
                    cf_pred, cf_target, self.counterfactual_risk_cfg)

                # Executed-action term: same head, REPLAY action as input
                # (not a fixed candidate), against the executed-action target.
                cf_pred_executed = self.counterfactual_risk_head(
                    cf_z, cf_action, temporal_feature=cf_temporal)
                executed_loss, executed_loss_logs = weighted_counterfactual_risk_loss(
                    cf_pred_executed, cf_executed_target, self.counterfactual_risk_cfg)

                cf_loss = (
                    fixed_loss
                    + self.counterfactual_risk_cfg.executed_action_loss_weight
                    * executed_loss)
                total_trunk_loss = (
                    total_trunk_loss
                    + self.counterfactual_risk_cfg.loss_weight * cf_loss)
                counterfactual_logs.update(fixed_loss_logs)
                counterfactual_logs.update({
                    f"counterfactual_risk/executed_{k.split('/', 1)[1]}": v
                    for k, v in executed_loss_logs.items()
                })
                counterfactual_logs["counterfactual_risk/loss"] = float(
                    cf_loss.detach().item())
                counterfactual_logs["counterfactual_risk/fixed_candidate_loss"] = float(
                    fixed_loss.detach().item())
                counterfactual_logs["counterfactual_risk/executed_action_loss"] = float(
                    executed_loss.detach().item())
                with torch.no_grad():
                    executed_err = (cf_pred_executed - cf_executed_target).abs()
                    counterfactual_logs["counterfactual_risk/executed_action_mae"] = float(
                        executed_err.mean().item())
                    per_horizon_mae = executed_err.mean(dim=0)
                    for _hi, _h_sec in enumerate(self.counterfactual_risk_cfg.horizons_sec):
                        counterfactual_logs[
                            f"counterfactual_risk/executed_mae_h{_h_sec:g}s"] = float(
                                per_horizon_mae[_hi].item())
                if use_rb_for_cf:
                    counterfactual_logs[
                        "risk_balance/counterfactual_uses_balanced_batch"] = 1.0

                cf_supervised_ran = True

        self.critic_optimizer.zero_grad()
        total_trunk_loss.backward()
        self.critic_optimizer.step()

        # Only count an update as "supervised" once the trunk optimizer has
        # actually applied it -- an exception between building the loss and
        # this step() call (e.g. a NaN in backward()) would otherwise have
        # already advanced the warm-up/ramp counter for a step that never
        # updated the head's weights.
        if cf_supervised_ran:
            self._cf_supervised_updates += 1

        """******************************************
        ** Actor Update
        ******************************************"""
        # Sample new actions (on the DETACHED latent: actor never updates encoder)
        actions_pi, log_prob = self.actor.action_log_prob(z_actor)

        # PHASE2 (critic_risk_input): action-risk prediction for the actor's OWN
        # candidate action. Do NOT detach the whole tensor here -- that would
        # sever d(extra_pi)/d(actions_pi), which is exactly the pathway that
        # lets "this action raises predicted risk" propagate into the actor's
        # gradient via qf_pi (the actor never sees extra_pi directly; it only
        # feels it through the critic's learned Q, which is the intended
        # "indirect" channel). What must NOT happen is action_risk_head's OWN
        # parameters picking up a gradient from actor_loss (only its dedicated
        # supervised loss may update it) -- so freeze its params for this one
        # forward call instead of detaching the output. requires_grad=False at
        # forward-time means autograd never records those weights as needing
        # grad for this graph, so restoring True right after (before backward())
        # is safe and standard practice for a "frozen submodule, live input"
        # forward pass.
        # STAGE 3: below action_risk_head.enable_from_stage, skip this forward
        # pass too and feed the critic a fixed all-zero (batch, 2) tensor
        # (same neutral value as extra_cur/extra_next) -- constant w.r.t.
        # actions_pi, so it contributes no gradient signal to the actor via
        # this channel, matching "no risk information available yet."
        extra_pi = None
        if self.critic_risk_input_enabled:
            if self._action_risk_active:
                # PHASE2 (temporal): reuse ar_temporal_cur (same state batch,
                # same online encoder as z_actor's origin) -- .detach() so
                # actor_loss cannot flow into self.encoder.temporal's
                # parameters, mirroring why z_actor itself is detached from z.
                # This does NOT touch the actions_pi gradient path (a separate
                # cat() branch below), so it does not affect the
                # critic_risk_input "gradient into the actor" channel at all.
                ar_temporal_pi = (
                    ar_temporal_cur.detach() if ar_temporal_cur is not None else None
                )
                # STAGE 5: exception-safe freeze (was a bare set-False/call/
                # set-True sequence that could leave the head permanently
                # frozen if the forward call raised).
                with _frozen_params(self.action_risk_head):
                    extra_pi = self.action_risk_head(
                        z_actor, actions_pi, temporal_feature=ar_temporal_pi)
            else:
                extra_pi = torch.zeros(
                    z_actor.shape[0], self.action_risk_head.out.out_features,
                    device=self.device)

        # Get Q-values for policy actions.
        # STAGE 5: freeze the critic's OWN parameters for this forward call
        # (exception-safe context manager) -- actor_loss.backward() must
        # reach actions_pi (the intended "does this action raise Q" signal)
        # but must NOT waste a backward pass building gradients into the
        # critic's parameters, which the critic-trunk update above already
        # owns exclusively and which get overwritten by critic_optimizer.step()
        # on the NEXT train() call regardless. Same freeze-not-detach pattern
        # already used for action_risk_head just above (detaching the whole
        # output would sever d(qf_pi)/d(actions_pi) too).
        with _frozen_params(self.critic):
            qf_pi = self.critic(z_actor, actions_pi, extra=extra_pi)
        # Average over quantiles and critics
        qf_pi = qf_pi.mean(dim=2).mean(dim=1, keepdim=True)

        # Direct actor risk penalty. Freeze the head's weights but keep the
        # input-side gradient d(risk)/d(action), exactly as for the indirect
        # action-risk path above. The state/temporal representation is detached,
        # so this update remains actor-only.
        #
        # Gated on cf_supervised_ran (not just _counterfactual_risk_active):
        # a step where the head had no supervised target to train on must not
        # apply a penalty from an untrained/stale head, and must not advance
        # the warm-up/ramp counter either (see CounterfactualRiskConfig.
        # effective_actor_penalty_weight's docstring).
        actor_risk_penalty = actions_pi.new_zeros(())
        effective_penalty_weight = 0.0
        if (self.counterfactual_risk_enabled
                and self.counterfactual_risk_head is not None
                and self._counterfactual_risk_active
                and cf_supervised_ran):
            temporal_pi = None
            if self.counterfactual_risk_cfg.use_temporal_context:
                if cf_temporal_cur is None:
                    # Defensive fallback (normally populated by the supervised
                    # block above). Inference-only: do not build an encoder graph.
                    with torch.no_grad():
                        cf_temporal_cur = self.encoder.temporal_feature(state)
                temporal_pi = cf_temporal_cur.detach()
            with _frozen_params(self.counterfactual_risk_head):
                predicted_horizon_risk = self.counterfactual_risk_head(
                    z_actor, actions_pi, temporal_feature=temporal_pi)
            if self.counterfactual_risk_cfg.actor_risk_aggregation == "max":
                actor_risk_penalty = predicted_horizon_risk.max(dim=1).values.mean()
            elif self.counterfactual_risk_cfg.actor_risk_aggregation == "weighted_mean":
                actor_risk_penalty = (
                    predicted_horizon_risk * self._cf_horizon_weights
                ).sum(dim=1).mean()
            else:
                actor_risk_penalty = predicted_horizon_risk.mean()
            effective_penalty_weight = (
                self.counterfactual_risk_cfg.effective_actor_penalty_weight(
                    self._cf_supervised_updates))
            self._cf_effective_actor_penalty_weight = effective_penalty_weight

        actor_loss = (
            (ent_coef * log_prob - qf_pi).mean()
            + effective_penalty_weight * actor_risk_penalty)
        if self.counterfactual_risk_enabled:
            counterfactual_logs["counterfactual_risk/actor_penalty"] = float(
                actor_risk_penalty.detach().item())
            counterfactual_logs[
                "counterfactual_risk/actor_penalty_weight"] = float(
                    self.counterfactual_risk_cfg.actor_penalty_weight)
            counterfactual_logs[
                "counterfactual_risk/effective_actor_penalty_weight"] = float(
                    effective_penalty_weight)
            counterfactual_logs[
                "counterfactual_risk/actor_penalty_contribution"] = float(
                    effective_penalty_weight * actor_risk_penalty.detach().item())
            counterfactual_logs["counterfactual_risk/supervised_updates"] = float(
                self._cf_supervised_updates)
            # OOD monitoring: normalized-action L2 distance from the actor's
            # own continuous action to its NEAREST fixed candidate -- computed
            # whenever the head is stage-active, independent of whether a
            # supervised update ran THIS step, so drift is visible through
            # warm-up too.
            if self._counterfactual_risk_active:
                with torch.no_grad():
                    _diffs = (actions_pi.unsqueeze(1)
                             - self._counterfactual_candidates.unsqueeze(0))
                    _nearest = _diffs.norm(dim=-1).min(dim=1).values
                    counterfactual_logs[
                        "counterfactual_risk/actor_candidate_distance_mean"] = float(
                            _nearest.mean().item())
                    counterfactual_logs[
                        "counterfactual_risk/actor_candidate_distance_max"] = float(
                            _nearest.max().item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        """******************************************
        ** Update Priority (if using prioritized replay)
        ******************************************"""
        if self.prioritized:
            with torch.no_grad():
                # Use mean absolute TD error for priority
                td_errors = (current_quantiles.mean(dim=[1, 2], keepdim=True) -
                            target_quantiles.mean(dim=[1, 2], keepdim=True)).abs()
                priority = td_errors.squeeze().clamp(min=self.min_priority).pow(self.alpha)
                self.replay_buffer.update_priority(priority)

        """******************************************
        ** Target Network Update
        ******************************************"""
        if self.training_steps % self.target_update_interval == 0:
            # STAGE 5: Polyak averaging, batched via torch._foreach_* instead
            # of a per-parameter Python loop -- numerically identical to the
            # original ``target.data.copy_(tau*src.data + (1-tau)*target.data)``
            # loop (see test_foreach_polyak_matches_per_parameter_loop_
            # numerically), fewer CUDA kernel launches for networks with many
            # parameter tensors (5 critics here).
            _polyak_update_foreach(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            # AUX_PRED: keep the target encoder in lock-step with the encoder.
            if self.encoder.has_params():
                _polyak_update_foreach(self.encoder.parameters(), self.encoder_target.parameters(), self.tau)

            # PHASE2: keep the target Action-Risk Head in lock-step (mirrors the
            # encoder_target block above).
            if self.action_risk_head is not None:
                _polyak_update_foreach(
                    self.action_risk_head.parameters(),
                    self.action_risk_head_target.parameters(), self.tau)

            if self.prioritized:
                self.replay_buffer.reset_max_priority()

        """******************************************
        ** Logging
        ******************************************"""
        # STAGE 6: .item() forces a GPU->CPU sync. Compute each scalar AT MOST
        # ONCE per train() call (the original code called critic_loss.item()
        # etc. TWICE -- once for TensorBoard, once for JSON) and skip the sync
        # entirely on a step where NEITHER sink is due (scalar_log_interval /
        # json_log_interval, both default 1 -> log every step, byte-identical
        # to before).
        log_scalar_now = (self.training_steps % self.scalar_log_interval == 0) and self.writer
        log_json_now = (self.training_steps % self.json_log_interval == 0)
        if log_scalar_now or log_json_now:
            scalars = {
                "loss/critic":  float(critic_loss.item()),
                "loss/actor":   float(actor_loss.item()),
                "values/Q":     float(qf_pi.mean().item()),
                "values/Q_max": float(current_quantiles.max().item()),
            }
            if self.ent_coef_auto:
                scalars["values/ent_coef"] = float(ent_coef.item())
                scalars["loss/ent_coef"] = float(ent_coef_loss.item())
            # AUX_PRED / PHASE2: aux_logs / action_risk_logs are already
            # plain floats (materialised where computed, gated by Stage 3's
            # forward-skip -- not re-synced here).
            if self.aux_enabled and aux_logs:
                scalars.update(aux_logs)
            if self.action_risk_enabled and action_risk_logs:
                scalars.update(action_risk_logs)
            if self.counterfactual_risk_enabled and counterfactual_logs:
                scalars.update(counterfactual_logs)
            # RISK_BALANCE: sampled-pool fractions logged regardless of which
            # consumer (aux / action-risk / both) used the balanced batch this
            # step, so they're visible even if only one of the two is enabled.
            if risk_balance_logs:
                scalars.update(risk_balance_logs)
            # RISK_BALANCE: RAW (whole-buffer) pool fractions alongside the
            # sampled_* ones above -- lets a run's actual positive-class share
            # (the imbalance risk-balanced sampling / pos_weight are
            # correcting for) be read directly off TensorBoard/JSON instead of
            # inferred only from the balanced batch's own composition. Only
            # computed when metadata storage is on (store_risk_meta implies
            # replay_buffer.risk_meta.enabled) and only at the log interval
            # (never every train() call) -- an O(buffer_size) reduction is
            # cheap at this cadence but wasteful every step.
            if self.store_risk_meta:
                scalars.update({
                    f"risk_balance/raw_{k}": v
                    for k, v in self.replay_buffer.describe_risk_meta_fractions_raw().items()
                })

            if log_scalar_now:
                for _k, _v in scalars.items():
                    self.writer.add_scalar(_k, _v, self.training_steps)
            if log_json_now:
                self._json_log(self.training_steps, **scalars)

    def train_and_checkpoint(self, ep_timesteps, ep_return):
        """Train and potentially update checkpoint"""
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.train_and_reset()
            # Keep the checkpoint policy aligned with the freshly trained actor.
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            # AUX_PRED: the checkpoint policy reads through its own encoder copy.
            self.checkpoint_encoder.load_state_dict(self.encoder.state_dict())

    def train_and_reset(self):
        """Batch training and reset counters"""
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.steps_before_checkpointing:
                self.best_min_return *= self.reset_weight
                self.max_eps_before_update = self.max_eps_when_checkpointing

            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

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
