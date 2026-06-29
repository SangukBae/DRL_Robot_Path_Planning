import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, time, json


import buffer

# AUX_PRED: auxiliary prediction network (shared encoder + future-risk head).
# All auxiliary logic lives in these modules; with aux_prediction.enabled=false
# the SharedEncoder is a parameter-free identity passthrough and the baseline
# TQC behaviour is reproduced exactly.
from aux_prediction import (
    AuxPredConfig, SharedEncoder, AuxiliaryHead, ActionConditionedAuxHead,
)
from aux_prediction_losses import compute_aux_loss
# AUX_PRED (v2): aux-only temporal context (GRU over recent in-episode states).
# Disabled -> out_dim 0, so the head is built exactly as the single-step v1 head.
from aux_prediction_temporal import TemporalContextEncoder
# TEMPORAL_ACTOR: compressed temporal feature on the ACTOR/CRITIC path. Splits the
# transported stacked state into current(87) + scan-history and fuses a small
# temporal feature into the shared latent (actor/critic stay non-recurrent).
from aux_prediction_temporal import TemporalFusionEncoder


# Network definitions + TQC loss now live in tqc_networks.py; checkpoint I/O in
# tqc_io.py.  Re-imported here so `from tqc_agent import Actor` etc. still works.
from tqc_networks import Actor, Critic, quantile_huber_loss
import tqc_io


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
        # AUX_PRED: Shared encoder (E_psi)
        # ----------------------------
        # The encoder sits between the raw 87-D state and the actor/critic.
        # When disabled it is an identity passthrough (out_dim == state_dim), so
        # actor/critic see the raw state exactly as in baseline TQC.  When
        # enabled it maps 87 -> hidden -> latent (ELU) and the actor consumes a
        # DETACHED latent so the actor loss never updates the encoder.
        self.aux_cfg = AuxPredConfig(self.hyperparameters.get("aux_prediction", {}))
        self.aux_enabled = self.aux_cfg.enabled
        self.aux_beta = self.aux_cfg.loss_weight
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
                encoder_type=str(tcfg.get("encoder_type", "conv1d")),
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

        # ----------------------------
        # Networks & Optimizers
        # ----------------------------
        # NOTE: actor/critic are now sized by the encoder latent dim.  With aux
        # disabled latent_dim == state_dim, so these are identical to baseline.
        self.actor = Actor(latent_dim, action_dim, self.actor_hdim, self.actor_activ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(
            latent_dim, action_dim, self.critic_hdim, self.critic_activ,
            self.n_quantiles, self.n_critics
        ).to(self.device)

        # AUX_PRED: a single optimizer updates critic + encoder + aux head from
        # (critic_loss + beta_aux * aux_loss).  With aux disabled it reduces to
        # the critic's parameters only (encoder has none), matching baseline.
        self.critic_optimizer = self._make_critic_optimizer()

        self.critic_target = Critic(
            latent_dim, action_dim, self.critic_hdim, self.critic_activ,
            self.n_quantiles, self.n_critics
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Checkpoint actor (평가 시 use_checkpoint=True 경로)
        self.checkpoint_actor = Actor(latent_dim, action_dim, self.actor_hdim, self.actor_activ).to(self.device)

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
            # AUX_PRED: track episode boundaries when EITHER the action-conditioned
            # aux (future-action lookup) OR the temporal context (backward
            # state-history lookup) is on, so neither walk crosses an episode.
            track_traj=(self.aux_action_conditioned or self.aux_temporal_enabled),
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
            
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
    
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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
        """
        trunk_params = list(self.critic.parameters())
        if self.encoder.has_params():
            trunk_params += list(self.encoder.parameters())
        if self.aux_head is not None:
            trunk_params += list(self.aux_head.parameters())
        # AUX_PRED (v2): the temporal encoder is part of the aux trunk; its grads
        # flow with critic + aux loss (never the actor) -> same optimizer group.
        if getattr(self, "temporal_encoder", None) is not None:
            trunk_params += list(self.temporal_encoder.parameters())
        return torch.optim.Adam(trunk_params, lr=self.critic_lr)

    def set_curriculum_stage(self, stage: int):
        """TEMPORAL_ACTOR / AUX_PRED: notify the agent of the current curriculum
        stage so the temporal feature strength and aux loss weight can ramp in
        with the curriculum WITHOUT changing any tensor shapes.

        - temporal_gain = 1.0 once stage >= stage_enable_from, else 0.0 (the
          temporal feature contributes nothing and gets no gradient at the easy
          early stages). Applied to the online / target / checkpoint encoders so
          rollout, TD target and checkpoint-eval all use the same gain.
        - the aux loss weight follows aux_prediction.stagewise_loss_schedule
          (read in _current_aux_beta).
        Safe no-op for the legacy (non-fusion) encoder / disabled temporal."""
        self.current_stage = int(stage)
        gain = 1.0 if int(stage) >= self.temporal_stage_enable_from else 0.0
        for enc in (self.encoder, self.encoder_target, self.checkpoint_encoder):
            if hasattr(enc, "set_temporal_gain"):
                enc.set_temporal_gain(gain)

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

    def _compute_temporal_ctx(self):
        """AUX_PRED (v2): temporal context for the just-sampled batch, or None.

        Walks the replay buffer backward (boundary-safe) for the last
        history_len in-episode states, encodes them through the SHARED encoder
        (so the temporal aux loss also shapes the encoder) and summarises the
        latent window with the temporal GRU. Returns (context, hist_valid_len) or
        None when temporal is off / history is unavailable."""
        if self.temporal_encoder is None:
            return None
        hist = self.replay_buffer.get_last_state_history(self.aux_cfg.history_len)
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
        # AUX_PRED: matching auxiliary targets for this batch (None if disabled).
        aux_target = self.replay_buffer.get_last_aux()

        # AUX_PRED: encode once.  `z` keeps the graph (critic + aux back-prop
        # into the encoder); `z_actor` is detached so the actor / temperature
        # updates never flow gradients into the encoder.
        z = self.encoder(state)
        z_actor = z.detach()

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

            # Get target quantiles
            next_quantiles = self.critic_target(z_next, next_actions)  # [B, n_critics, n_quantiles]

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
        
        # Get current quantiles (encoder latent keeps its graph here)
        current_quantiles = self.critic(z, action)  # [B, n_critics, n_quantiles]

        # Compute critic loss
        critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)

        # AUX_PRED: future-risk prediction loss; gradients flow into the shared
        # encoder together with the critic loss (encoder = critic + beta*aux).
        aux_loss_val = 0.0
        aux_logs = {}
        total_trunk_loss = critic_loss
        if self.aux_enabled and self.aux_head is not None and aux_target is not None:
            # AUX_PRED (v2): aux-only temporal context (recent in-episode state
            # history). None when temporal is off -> the head ignores it and the
            # v1 path is unchanged. Shares the encoder graph so it shapes E_psi.
            temporal_ctx = None
            tctx = self._compute_temporal_ctx()
            if tctx is not None:
                temporal_ctx, hist_valid = tctx
                aux_logs["aux/hist_len_mean"] = float(hist_valid.float().mean().item())

            if self.aux_action_conditioned:
                # Same target L_i (future risk from s_i), but conditioned on the
                # upcoming in-episode action sequence [a_i, .., a_{i+K-1}].
                fa = self.replay_buffer.get_last_future_actions(
                    self.aux_cfg.action_conditioned_steps)
                aux_pred = None
                if fa is not None:
                    future_actions, valid_len = fa
                    aux_pred = self.aux_head(
                        z, future_actions, valid_len, temporal_ctx=temporal_ctx)
                    aux_logs["aux/valid_len_mean"] = float(valid_len.float().mean().item())
            else:
                aux_pred = self.aux_head(z, temporal_ctx=temporal_ctx)

            if aux_pred is not None:
                aux_loss, _logs = compute_aux_loss(
                    aux_pred, aux_target, self.aux_cfg, self.device
                )
                aux_logs.update(_logs)
                beta = self._current_aux_beta()
                total_trunk_loss = critic_loss + beta * aux_loss
                aux_loss_val = float(aux_loss.detach().item())
                aux_logs["aux/beta"] = float(beta)

        self.critic_optimizer.zero_grad()
        total_trunk_loss.backward()
        self.critic_optimizer.step()

        """******************************************
        ** Actor Update
        ******************************************"""
        # Sample new actions (on the DETACHED latent: actor never updates encoder)
        actions_pi, log_prob = self.actor.action_log_prob(z_actor)

        # Get Q-values for policy actions
        qf_pi = self.critic(z_actor, actions_pi)
        # Average over quantiles and critics
        qf_pi = qf_pi.mean(dim=2).mean(dim=1, keepdim=True)
        
        # Actor loss
        actor_loss = (ent_coef * log_prob - qf_pi).mean()
        
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
            # Polyak averaging
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # AUX_PRED: keep the target encoder in lock-step with the encoder.
            if self.encoder.has_params():
                for param, target_param in zip(
                    self.encoder.parameters(), self.encoder_target.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.prioritized:
                self.replay_buffer.reset_max_priority()
        
        """******************************************
        ** Logging
        ******************************************"""
        if self.writer:
            self.writer.add_scalar("loss/critic", float(critic_loss.item()), self.training_steps)
            self.writer.add_scalar("loss/actor",  float(actor_loss.item()),  self.training_steps)
            self.writer.add_scalar("values/Q",    float(qf_pi.mean().item()), self.training_steps)
            self.writer.add_scalar("values/Q_max", float(current_quantiles.max().item()), self.training_steps)
            if self.ent_coef_auto:
                self.writer.add_scalar("values/ent_coef", float(ent_coef.item()), self.training_steps)
                self.writer.add_scalar("loss/ent_coef", float(ent_coef_loss.item()), self.training_steps)
            # AUX_PRED: log auxiliary loss terms when active.
            if self.aux_enabled and aux_logs:
                for _k, _v in aux_logs.items():
                    self.writer.add_scalar(_k, float(_v), self.training_steps)

        # === JSON 동시 기록 ===
        self._json_log(
            self.training_steps,
            **{
                "loss/critic":  float(critic_loss.item()),
                "loss/actor":   float(actor_loss.item()),
                "values/Q":     float(qf_pi.mean().item()),
                "values/Q_max": float(current_quantiles.max().item()),
                **({"values/ent_coef": float(ent_coef.item()),
                    "loss/ent_coef":   float(ent_coef_loss.item())} if self.ent_coef_auto else {}),
                # AUX_PRED: persist auxiliary metrics to the JSONL log too.
                **(aux_logs if (self.aux_enabled and aux_logs) else {})
            }
        )
    
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
