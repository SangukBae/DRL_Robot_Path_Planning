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


def quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False, kappa=1.0):
    """
    Quantile Huber loss for TQC
    current_quantiles: (batch_size, n_critics, n_quantiles)
    target_quantiles: (batch_size, 1, n_target_quantiles)
    """
    batch_size, n_critics, n_quantiles = current_quantiles.shape
    n_target_quantiles = target_quantiles.shape[-1]
    
    # Expand quantiles for pairwise TD errors
    current_quantiles = current_quantiles.unsqueeze(-1)  # (batch, n_critics, n_quantiles, 1)
    target_quantiles = target_quantiles.unsqueeze(2)  # (batch, 1, 1, n_target_quantiles)
    
    # Compute TD errors
    td_errors = target_quantiles - current_quantiles  # (batch, n_critics, n_quantiles, n_target_quantiles)
    
    # Compute quantile weights (tau)
    tau = torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float32)
    tau = (tau + 0.5) / n_quantiles
    tau = tau.view(1, 1, n_quantiles, 1)
    
    # Huber loss
    huber_loss = torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa)
    )
    
    # Quantile regression loss
    quantile_loss = (tau - (td_errors < 0).float()).abs() * huber_loss
    
    if sum_over_quantiles:
        return quantile_loss.sum(dim=2).mean()
    else:
        return quantile_loss.mean()


class Actor(nn.Module):
    """Actor network with Gaussian policy for TQC"""
    def __init__(self, state_dim, action_dim, hdim=256, activ=F.relu, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.activ = activ
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Network layers
        self.l1 = nn.Linear(state_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, hdim)
        
        # Mean and log_std heads
        self.mean = nn.Linear(hdim, action_dim)
        self.log_std = nn.Linear(hdim, action_dim)
    
    def forward(self, state, deterministic=False):
        a = self.activ(self.l1(state))
        a = self.activ(self.l2(a))
        a = self.activ(self.l3(a))
        
        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            # Sample from Gaussian
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()
            action = torch.tanh(x)
            return action
    
    def action_log_prob(self, state):
        """Get action and log probability"""
        a = self.activ(self.l1(state))
        a = self.activ(self.l2(a))
        a = self.activ(self.l3(a))
        
        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x).sum(1, keepdim=True)
        log_prob -= (2 * (np.log(2) - x - F.softplus(-2 * x))).sum(1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hdim=256,
                 activ=F.elu, n_quantiles=25, n_critics=2):
        super().__init__()
        self.activ = activ
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics

        # activ 모듈 선택
        ActivMod = nn.ELU if activ is F.elu else nn.ReLU

        self.critics = nn.ModuleList()
        for _ in range(n_critics):
            self.critics.append(nn.Sequential(
                nn.Linear(state_dim + action_dim, hdim),
                ActivMod(),
                nn.Linear(hdim, hdim),
                ActivMod(),
                nn.Linear(hdim, hdim),
                ActivMod(),
                nn.Linear(hdim, n_quantiles),
            ))
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        quantiles_list = []
        for critic in self.critics:
            q = critic(sa)  # (batch_size, n_quantiles)
            quantiles_list.append(q)
        
        # Stack to (batch_size, n_critics, n_quantiles)
        quantiles = torch.stack(quantiles_list, dim=1)
        return quantiles


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, hyperparameters, log_dir=None):
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

        self.encoder = SharedEncoder(state_dim, self.aux_cfg).to(self.device)
        self.encoder_target = SharedEncoder(state_dim, self.aux_cfg).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.checkpoint_encoder = SharedEncoder(state_dim, self.aux_cfg).to(self.device)
        self.checkpoint_encoder.load_state_dict(self.encoder.state_dict())
        latent_dim = self.encoder.out_dim

        # AUX_PRED: training-only auxiliary head (dropped at inference).  The
        # action-conditioned head adds an action-sequence GRU; both emit the same
        # output dict so the loss (compute_aux_loss) is identical.
        if not self.aux_enabled:
            self.aux_head = None
        elif self.aux_action_conditioned:
            self.aux_head = ActionConditionedAuxHead(
                latent_dim, action_dim, self.aux_cfg).to(self.device)
        else:
            self.aux_head = AuxiliaryHead(latent_dim, self.aux_cfg).to(self.device)

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
            # AUX_PRED: track episode boundaries only for the action-conditioned
            # aux (so future-action lookups never cross episodes).
            track_traj=self.aux_action_conditioned,
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
        return torch.optim.Adam(trunk_params, lr=self.critic_lr)

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

    def aux_predict_eval(self, states, future_actions=None, valid_len=None):
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
            if self.aux_action_conditioned:
                if future_actions is None or valid_len is None:
                    return None
                fa = torch.from_numpy(
                    np.asarray(future_actions, dtype=np.float32)
                ).to(self.device)
                vl = torch.from_numpy(
                    np.asarray(valid_len, dtype=np.int64)
                ).to(self.device)
                out = self.aux_head(z, fa, vl)
            else:
                out = self.aux_head(z)
            res = {"risk_map": out["risk_map"].cpu().numpy().astype(np.float32)}
            if "min_dist" in out:
                res["min_dist"] = out["min_dist"].cpu().numpy().astype(np.float32)
            return res

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
            if self.aux_action_conditioned:
                # Same target L_i (future risk from s_i), but conditioned on the
                # upcoming in-episode action sequence [a_i, .., a_{i+K-1}].
                fa = self.replay_buffer.get_last_future_actions(
                    self.aux_cfg.action_conditioned_steps)
                aux_pred = None
                if fa is not None:
                    future_actions, valid_len = fa
                    aux_pred = self.aux_head(z, future_actions, valid_len)
                    aux_logs["aux/valid_len_mean"] = float(valid_len.float().mean().item())
            else:
                aux_pred = self.aux_head(z)

            if aux_pred is not None:
                aux_loss, _logs = compute_aux_loss(
                    aux_pred, aux_target, self.aux_cfg, self.device
                )
                aux_logs.update(_logs)
                total_trunk_loss = critic_loss + self.aux_beta * aux_loss
                aux_loss_val = float(aux_loss.detach().item())

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
        """Save model parameters"""
        import os
        os.makedirs(directory, exist_ok=True)
        # Actor
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), f"{directory}/{filename}_actor_optimizer.pth")
        
        # Critic
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        torch.save(self.critic_optimizer.state_dict(), f"{directory}/{filename}_critic_optimizer.pth")
        
        # Checkpoint
        torch.save(self.checkpoint_actor.state_dict(), f"{directory}/{filename}_checkpoint_actor.pth")

        # AUX_PRED: shared encoder + auxiliary head (training-only).  Saved only
        # when the encoder actually has parameters (aux enabled); inference does
        # not require the aux head.
        if self.encoder.has_params():
            torch.save(self.encoder.state_dict(), f"{directory}/{filename}_encoder.pth")
            torch.save(self.encoder_target.state_dict(), f"{directory}/{filename}_encoder_target.pth")
            torch.save(self.checkpoint_encoder.state_dict(), f"{directory}/{filename}_checkpoint_encoder.pth")
            if self.aux_head is not None:
                torch.save(self.aux_head.state_dict(), f"{directory}/{filename}_aux_head.pth")

        # Entropy coefficient
        if self.ent_coef_auto:
            torch.save(self.log_ent_coef, f"{directory}/{filename}_log_ent_coef.pth")
            torch.save(self.ent_coef_optimizer.state_dict(), f"{directory}/{filename}_ent_coef_optimizer.pth")
        else:
            torch.save(self.ent_coef_tensor, f"{directory}/{filename}_ent_coef_tensor.pth")

        # Replay buffer — enables full off-policy resume
        self.replay_buffer.save(f"{directory}/{filename}_replay_buffer")
    
    def load(
        self,
        directory,
        filename,
        *,
        load_optimizer_state=True,
        load_replay_buffer=True,
    ):
        import os, torch

        maploc = self.device

        def _torch_load(path):
            try:
                return torch.load(path, map_location=maploc, weights_only=True)
            except TypeError:
                # Older PyTorch versions do not support weights_only.
                return torch.load(path, map_location=maploc)

        # Actor
        p = f"{directory}/{filename}_actor.pth"
        if os.path.exists(p):
            self.actor.load_state_dict(_torch_load(p))
        if load_optimizer_state:
            p = f"{directory}/{filename}_actor_optimizer.pth"
            if os.path.exists(p):
                self.actor_optimizer.load_state_dict(_torch_load(p))

        # Critic
        p = f"{directory}/{filename}_critic.pth"
        if os.path.exists(p):
            self.critic.load_state_dict(_torch_load(p))
        p = f"{directory}/{filename}_critic_target.pth"
        if os.path.exists(p):
            self.critic_target.load_state_dict(_torch_load(p))
        if load_optimizer_state:
            p = f"{directory}/{filename}_critic_optimizer.pth"
            if os.path.exists(p):
                # AUX_PRED: critic_optimizer also owns the encoder + aux-head
                # params, so its param-group size changes when the aux head
                # architecture changes (single-step <-> action-conditioned).  On
                # such a mismatch keep fresh optimizer moments (the critic /
                # encoder WEIGHTS already loaded above) instead of aborting.
                # Even when this load "succeeds", a later aux-head state-dict
                # mismatch can still invalidate the loaded moments if only the
                # param COUNT matched while shapes / semantics changed; that
                # later path rebuilds this optimizer unconditionally.
                try:
                    self.critic_optimizer.load_state_dict(_torch_load(p))
                except (ValueError, RuntimeError, KeyError) as e:
                    print(
                        "[AUX_PRED] critic optimizer state is incompatible with "
                        "the current aux config (the aux head changed the trunk "
                        "param group); keeping fresh optimizer moments. "
                        f"Details: {e}"
                    )

        # Checkpoint actor
        p = f"{directory}/{filename}_checkpoint_actor.pth"
        if os.path.exists(p):
            self.checkpoint_actor.load_state_dict(_torch_load(p))

        # AUX_PRED: shared encoder + auxiliary head (only when this run uses aux
        # AND the checkpoint carries them; baseline checkpoints lack these files
        # and are loaded unchanged).
        if self.encoder.has_params():
            p = f"{directory}/{filename}_encoder.pth"
            if os.path.exists(p):
                self.encoder.load_state_dict(_torch_load(p))
            p = f"{directory}/{filename}_encoder_target.pth"
            if os.path.exists(p):
                self.encoder_target.load_state_dict(_torch_load(p))
            p = f"{directory}/{filename}_checkpoint_encoder.pth"
            if os.path.exists(p):
                self.checkpoint_encoder.load_state_dict(_torch_load(p))
            p = f"{directory}/{filename}_aux_head.pth"
            if self.aux_head is not None and os.path.exists(p):
                # AUX_PRED: the aux head is TRAINING-ONLY.  Its architecture
                # changes between the single-step AuxiliaryHead and the
                # ActionConditionedAuxHead (and with min-dist / distributional
                # options), so a strict load fails when upgrading / switching an
                # ablation.  Try a strict load; on a state-dict mismatch keep the
                # freshly-initialised head (it retrains) and warn, rather than
                # aborting the whole resume -- the encoder/actor/critic, which
                # actually drive the policy, still load above.
                try:
                    self.aux_head.load_state_dict(_torch_load(p))
                except (RuntimeError, KeyError) as e:
                    # The aux head changed architecture.  The critic_optimizer
                    # (loaded earlier) owns the aux-head params in the SAME param
                    # group, so its moments are now stale -- and the load above
                    # may have "succeeded" if only the param COUNT matched while
                    # shapes/semantics differ, leaving wrong moments that would
                    # crash or silently corrupt the next step().  Rebuild the
                    # optimizer fresh to guarantee no stale moment survives.
                    self.critic_optimizer = self._make_critic_optimizer()
                    print(
                        "[AUX_PRED] aux-head checkpoint is incompatible with the "
                        "current aux config (e.g. single-step <-> action-"
                        "conditioned, or changed heads); keeping a freshly-"
                        "initialised aux head AND fresh critic-optimizer moments "
                        f"(both will retrain). Details: {e}"
                    )

        # Entropy coefficient
        if self.ent_coef_auto:
            p = f"{directory}/{filename}_log_ent_coef.pth"
            if os.path.exists(p):
                loaded = _torch_load(p)
                # <<< 핵심: 텐서 객체를 교체하지 말고 data만 복사 >>>
                self.log_ent_coef.data.copy_(loaded.to(maploc).data)
            if load_optimizer_state:
                p = f"{directory}/{filename}_ent_coef_optimizer.pth"
                if os.path.exists(p):
                    self.ent_coef_optimizer.load_state_dict(_torch_load(p))
        else:
            p = f"{directory}/{filename}_ent_coef_tensor.pth"
            if os.path.exists(p):
                loaded = _torch_load(p)
                self.ent_coef_tensor = loaded.to(maploc).detach()

        # Replay buffer
        if load_replay_buffer:
            buf_path = f"{directory}/{filename}_replay_buffer"
            if os.path.isfile(buf_path + ".npz"):
                self.replay_buffer.load(buf_path)

    def load_encoder_for_inference(self, actor_path):
        """AUX_PRED: restore the shared encoder for an actor-only inference path.

        Inference runs state -> encoder -> actor, so an aux-enabled checkpoint
        MUST load the encoder alongside the actor; otherwise the actor receives
        a randomly-initialised latent and the policy is broken.  The matching
        encoder file is derived from the actor checkpoint path
        (``<prefix>_actor.pth`` -> ``<prefix>_encoder.pth``).

        Returns
        -------
        bool
            True  -> encoder ready (loaded, or not needed for a baseline /
                     identity encoder, i.e. aux disabled).
            False -> an aux encoder is REQUIRED but its file is missing; the
                     caller should treat this as a fatal inference error.
        """
        import os, torch

        # Baseline (aux disabled): encoder is a parameter-free identity, so the
        # actor consumes the raw state and there is nothing to restore.
        if not self.encoder.has_params():
            return True

        if actor_path.endswith("_actor.pth"):
            enc_path = actor_path[: -len("_actor.pth")] + "_encoder.pth"
        else:
            enc_path = os.path.join(os.path.dirname(actor_path), "encoder.pth")
        if not os.path.isfile(enc_path):
            return False

        try:
            sd = torch.load(enc_path, map_location=self.device, weights_only=True)
        except TypeError:
            sd = torch.load(enc_path, map_location=self.device)
        self.encoder.load_state_dict(sd)
        self.encoder.eval()
        if getattr(self, "checkpoint_encoder", None) is not None:
            self.checkpoint_encoder.load_state_dict(sd)
            self.checkpoint_encoder.eval()
        return True
