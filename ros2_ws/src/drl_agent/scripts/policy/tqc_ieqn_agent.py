import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, time, json


import buffer


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

class Mapper(nn.Module):
    def __init__(self, hdim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hdim), nn.LayerNorm(hdim), nn.ReLU(), nn.Linear(hdim, 1)
        )
    def forward(self, tau):              # tau: [B, N]
        x = tau.reshape(-1, 1)
        delta = torch.tanh(self.mlp(x))
        alpha = x + delta
        return alpha.reshape_as(tau).clamp(0.0+1e-6, 1.0-1e-6)
    
class CriticIEQN(nn.Module):
    def __init__(self, state_dim, action_dim, hdim=256, activ=F.elu, n_critics=2, n_basis=64):
        super().__init__()
        self.n_critics, self.activ, self.n_basis = n_critics, activ, n_basis
        # SA feature
        Activ = nn.ELU if activ is F.elu else nn.ReLU
        self.sa = nn.Sequential(nn.Linear(state_dim+action_dim, hdim), Activ(),
                                nn.Linear(hdim, hdim), Activ())
        # τ 임베딩
        self.tau_fc = nn.Sequential(nn.Linear(n_basis, hdim), nn.ReLU())
        # 헤드(critic별)
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(hdim, hdim), Activ(), nn.Linear(hdim, 1)) for _ in range(n_critics)])
    def tau_embed(self, tau):            # tau: [B,N]
        i = torch.arange(1, self.n_basis+1, device=tau.device, dtype=tau.dtype).view(1,1,-1)
        cos = torch.cos(np.pi * tau.unsqueeze(-1) * i)     # [B,N,n_basis]
        e = self.tau_fc(cos.reshape(-1, self.n_basis))     # [B*N, hdim]
        return e.view(tau.size(0), tau.size(1), -1)        # [B,N,hdim]
    def forward(self, state, action, tau):                 # → [B, n_critics, N]
        B, N = tau.size(0), tau.size(1)
        h_sa = self.sa(torch.cat([state, action], 1))      # [B, hdim]
        h_sa = h_sa.unsqueeze(1).expand(B, N, -1)          # [B,N,hdim]
        h_tau = self.tau_embed(tau)                        # [B,N,hdim]
        h = h_sa * h_tau                                   # Hadamard
        outs=[]
        for head in self.heads:
            q = head(h.reshape(B*N, -1)).view(B, N)        # [B,N]
            outs.append(q)
        return torch.stack(outs, dim=1)                    # [B, n_critics, N]

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

        # tqc_ieqn parameters
        self.use_ieqn = bool(self.hyperparameters.get("use_ieqn", True))
        self.n_statistics = int(self.hyperparameters.get("n_statistics", 64))
        self.kappa_quantiles = float(self.hyperparameters.get("kappa_quantiles", 0.0))
        self.mapper_lr = float(self.hyperparameters.get("mapper_lr", 7e-5))
        self.tau_mapper = float(self.hyperparameters.get("tau_mapper", 0.5))

        # ----------------------------
        # Networks & Optimizers
        # ----------------------------
        self.actor = Actor(state_dim, action_dim, self.actor_hdim, self.actor_activ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        if self.use_ieqn:
            # τ-조건부 크리틱 + 매퍼
            self.critic = CriticIEQN(state_dim, action_dim, self.critic_hdim, self.critic_activ, self.n_critics).to(self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

            self.critic_target = CriticIEQN(state_dim, action_dim, self.critic_hdim, self.critic_activ, self.n_critics).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.mapper = Mapper().to(self.device)
            self.mapper_target = Mapper().to(self.device)          # ★ 여기!
            self.mapper_target.load_state_dict(self.mapper.state_dict())
            self.mapper_optimizer = torch.optim.Adam(self.mapper.parameters(), lr=self.mapper_lr)
        else:
            # 기존 TQC 경로
            self.critic = Critic(state_dim, action_dim, self.critic_hdim, self.critic_activ, self.n_quantiles, self.n_critics).to(self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

            self.critic_target = Critic(state_dim, action_dim, self.critic_hdim, self.critic_activ, self.n_quantiles, self.n_critics).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        # Checkpoint actor (평가 시 use_checkpoint=True 경로)
        self.checkpoint_actor = Actor(state_dim, action_dim, self.actor_hdim, self.actor_activ).to(self.device)

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
            normalize_actions=True,
            prioritized=self.prioritized,
        )

        # ----------------------------
        # Book-keeping
        # ----------------------------
        self.training_steps = 0
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

        # === JSON 로그 파일 준비 ===
        self.json_log_path = None
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self.json_log_path = os.path.join(self.log_dir, "metrics.jsonl")
                # 파일 생성 보장
                with open(self.json_log_path, "a", encoding="utf-8") as f:
                    pass
            except Exception:
                self.json_log_path = None

    # JSON 라인 기록 헬퍼
    def _json_log(self, step: int, **metrics):
        if not self.json_log_path:
            return
        # 숫자형만 기록 + NaN/Inf 방지
        rec = {"step": int(step), "time": time.time()}
        for k, v in metrics.items():
            try:
                val = float(v)
                if np.isfinite(val):
                    rec[k] = val
            except Exception:
                continue
        with open(self.json_log_path, "a", encoding="utf-8") as f:
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

        # tqc_ieqn parameters
        hp.setdefault("use_ieqn", True)         # IEQN on/off 토글
        hp.setdefault("n_statistics", 64)       # τ 샘플 수
        hp.setdefault("mapper_lr", 7e-5)
        hp.setdefault("tau_mapper", 0.5)
        hp.setdefault("kappa_quantiles", 0.0)   # Huber κ (0이면 L1)

        return hp

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        """상태로부터 행동 선택 (tanh 스쿼시 → max_action 스케일)"""
        with torch.no_grad():
            
            state_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
            state_t  = torch.from_numpy(state_np).to(self.device)

            actor_to_use = self.checkpoint_actor if use_checkpoint else self.actor
            action = actor_to_use(state_t, deterministic=not use_exploration)

            # [-1,1] 클램프 후 환경 스케일로 변환
            action = action.clamp(-1, 1)
            return (action.cpu().numpy().flatten() * self.max_action)
    
    def train(self):
        """Train the agent for one step"""
        self.training_steps += 1

        # Sample batch from replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample()

        # ==========================================
        # Entropy Coefficient Update (if auto)
        # ==========================================
        if self.ent_coef_auto:
            with torch.no_grad():
                _, log_prob_tmp = self.actor.action_log_prob(state)
            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob_tmp + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
        else:
            ent_coef = self.ent_coef_tensor

        # ------------------------------------------------------------------
        # ============ IEQN PATH (expectile critic + mapper) ===============
        # ------------------------------------------------------------------
        if getattr(self, "use_ieqn", False):
            B = state.size(0)
            N = int(self.n_statistics)

            with torch.no_grad():
                # Target distribution using mapper_target(τ_t)
                tau_t = torch.rand(B, N, device=self.device)
                alpha_t = self.mapper_target(tau_t)
                next_actions, next_log_prob = self.actor.action_log_prob(next_state)

                # critic_target returns [B, n_critics, N]
                Z_tgt = self.critic_target(next_state, next_actions, alpha_t)   # [B,C,N]
                Z_tgt = Z_tgt.reshape(B, -1)                                   # [B, C*N]

                # TQC truncation (drop top quantiles)
                n_total = Z_tgt.size(1)  # C*N
                drop = self.top_quantiles_to_drop_per_net * self.n_critics
                keep = n_total - drop
                Z_sorted, _ = torch.sort(Z_tgt, dim=1)
                Z_trunc = Z_sorted[:, :keep]                                   # [B, M], M=keep

                # Bellman target with entropy
                target_quantiles = reward + not_done * self.discount * (
                    Z_trunc - ent_coef * next_log_prob.view(-1, 1)
                )                                                               # [B, M]

            # -------- Critic update (Expectile-L2) --------
            tau_e = torch.rand(B, N, device=self.device)                        # [B,N]
            # critic(state, action, tau) -> [B, C, N] -> 평균(critics)로 [B,N]
            E_pred = self.critic(state, action, tau_e).mean(dim=1)              # [B,N]

            # pairwise losses: (target M) x (expectiles N)
            M = target_quantiles.size(1)
            z_rep = target_quantiles.repeat(1, N)                                # [B, M*N]
            e_tile = E_pred.unsqueeze(1).repeat(1, M, 1).reshape(B, -1)          # [B, M*N]
            te    = tau_e.unsqueeze(1).repeat(1, M, 1).reshape(B, -1)            # [B, M*N]
            diff  = z_rep - e_tile
            loss_e = torch.where(diff > 0, te * diff.pow(2), (1 - te) * diff.pow(2)).mean()

            self.critic_optimizer.zero_grad()
            loss_e.backward()
            self.critic_optimizer.step()

            # -------- Mapper update (Quantile-Huber) --------
            tau_q = torch.rand(B, N, device=self.device)
            alpha = self.mapper(tau_q)                                           # [B,N]

            # critic는 일시적으로 freeze → gradient가 mapper로만 흐르도록
            for p in self.critic.parameters():
                p.requires_grad_(False)

            # ⬇️ detach 제거: mapper(→alpha)로 미분 경로 유지
            Q_pred = self.critic(state, action, alpha).mean(dim=1)               # [B,N]

            q_tile = Q_pred.unsqueeze(1).repeat(1, M, 1).reshape(B, -1)          # [B, M*N]
            tq    = tau_q.unsqueeze(1).repeat(1, M, 1).reshape(B, -1)            # [B, M*N]
            u = z_rep - q_tile                                                   # z_rep은 위에서 정의된 target 타일

            kappa = float(self.kappa_quantiles)
            huber = (
                u.abs()
                if kappa == 0.0
                else torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
            )

            loss_q = torch.where(u > 0, tq * huber, (1 - tq) * huber).mean()

            self.mapper_optimizer.zero_grad()
            loss_q.backward()
            self.mapper_optimizer.step()

            # critic 다시 활성화
            for p in self.critic.parameters():
                p.requires_grad_(True)

            # -------- Actor update (unchanged; IEQN critic 평균 사용) --------
            actions_pi, log_prob = self.actor.action_log_prob(state)
            tau_pi = torch.rand(B, N, device=self.device)
            qf_pi = self.critic(state, actions_pi, tau_pi).mean(dim=2).mean(dim=1, keepdim=True)  # mean over N and critics
            actor_loss = (ent_coef * log_prob - qf_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # -------- Prioritized replay (optional) --------
            if self.prioritized:
                with torch.no_grad():
                    # priority by mean absolute TD error of expectations
                    curr_mean = self.critic(state, action, torch.rand(B, N, device=self.device)).mean(dim=[1, 2], keepdim=True)
                    tgt_mean  = target_quantiles.mean(dim=1, keepdim=True)
                    td_errors = (curr_mean - tgt_mean).abs()
                    priority = td_errors.squeeze().clamp(min=self.min_priority).pow(self.alpha)
                    self.replay_buffer.update_priority(priority)

            # -------- Target updates (critic + mapper) --------
            if self.training_steps % self.target_update_interval == 0:
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
                for p, tp in zip(self.mapper.parameters(), self.mapper_target.parameters()):
                    tp.data.copy_(self.tau_mapper * p.data + (1 - self.tau_mapper) * tp.data)
                if self.prioritized:
                    self.replay_buffer.reset_max_priority()

            # -------- Logging --------
            if self.writer:
                self.writer.add_scalar("loss/critic_expectile", float(loss_e.item()), self.training_steps)
                self.writer.add_scalar("loss/mapper_quantile",  float(loss_q.item()), self.training_steps)
                self.writer.add_scalar("values/Q_pi",           float(qf_pi.mean().item()), self.training_steps)
                if self.ent_coef_auto:
                    self.writer.add_scalar("values/ent_coef", float(ent_coef.item()), self.training_steps)
                    self.writer.add_scalar("loss/ent_coef",  float(ent_coef_loss.item()), self.training_steps)

            # === JSON 동시 기록 ===
            self._json_log(
                self.training_steps,
                **{
                    "loss/critic_expectile": float(loss_e.item()),
                    "loss/mapper_quantile":  float(loss_q.item()),
                    "values/Q_pi":           float(qf_pi.mean().item()),
                    **({"values/ent_coef": float(ent_coef.item()),
                        "loss/ent_coef":   float(ent_coef_loss.item())} if self.ent_coef_auto else {})
                }
            )
            return

        # ------------------------------------------------------------------
        # ============== ORIGINAL TQC PATH (unchanged) =====================
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Sample actions for next states
            next_actions, next_log_prob = self.actor.action_log_prob(next_state)

            # Get target quantiles
            next_quantiles = self.critic_target(next_state, next_actions)  # [B, n_critics, n_quantiles]

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

        # Get current quantiles
        current_quantiles = self.critic(state, action)  # [B, n_critics, n_quantiles]

        # Compute critic loss
        critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------- Actor Update --------
        actions_pi, log_prob = self.actor.action_log_prob(state)

        # Get Q-values for policy actions
        qf_pi = self.critic(state, actions_pi)
        # Average over quantiles and critics
        qf_pi = qf_pi.mean(dim=2).mean(dim=1, keepdim=True)

        # Actor loss
        actor_loss = (ent_coef * log_prob - qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -------- Update Priority (if prioritized replay) --------
        if self.prioritized:
            with torch.no_grad():
                td_errors = (current_quantiles.mean(dim=[1, 2], keepdim=True) -
                             target_quantiles.mean(dim=[1, 2], keepdim=True)).abs()
                priority = td_errors.squeeze().clamp(min=self.min_priority).pow(self.alpha)
                self.replay_buffer.update_priority(priority)

        # -------- Target Network Update --------
        if self.training_steps % self.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if self.prioritized:
                self.replay_buffer.reset_max_priority()

        # -------- Logging --------
        if self.writer:
            self.writer.add_scalar("loss/critic", float(critic_loss.item()), self.training_steps)
            self.writer.add_scalar("loss/actor",  float(actor_loss.item()),  self.training_steps)
            self.writer.add_scalar("values/Q",    float(qf_pi.mean().item()), self.training_steps)
            self.writer.add_scalar("values/Q_max", float(current_quantiles.max().item()), self.training_steps)
            if self.ent_coef_auto:
                self.writer.add_scalar("values/ent_coef", float(ent_coef.item()), self.training_steps)
                self.writer.add_scalar("loss/ent_coef", float(ent_coef_loss.item()), self.training_steps)
    
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
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            
            self.train_and_reset()
    
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

        # ----- Actor -----
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), f"{directory}/{filename}_actor_optimizer.pth")

        # ----- Critic -----
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        torch.save(self.critic_optimizer.state_dict(), f"{directory}/{filename}_critic_optimizer.pth")

        # ----- Mapper (IEQN 전용) -----
        if getattr(self, "use_ieqn", False):
            torch.save(self.mapper.state_dict(), f"{directory}/{filename}_mapper.pth")
            torch.save(self.mapper_target.state_dict(), f"{directory}/{filename}_mapper_target.pth")
            torch.save(self.mapper_optimizer.state_dict(), f"{directory}/{filename}_mapper_optimizer.pth")

        # ----- Checkpoint actor -----
        torch.save(self.checkpoint_actor.state_dict(), f"{directory}/{filename}_checkpoint_actor.pth")

        # ----- Entropy coefficient -----
        if self.ent_coef_auto:
            torch.save(self.log_ent_coef, f"{directory}/{filename}_log_ent_coef.pth")
            torch.save(self.ent_coef_optimizer.state_dict(), f"{directory}/{filename}_ent_coef_optimizer.pth")
        else:
            torch.save(self.ent_coef_tensor, f"{directory}/{filename}_ent_coef_tensor.pth")


    def load(self, directory, filename):
        import os

        maploc = self.device

        # ----- Actor -----
        p = f"{directory}/{filename}_actor.pth"
        if os.path.exists(p):
            self.actor.load_state_dict(torch.load(p, map_location=maploc))
        p = f"{directory}/{filename}_actor_optimizer.pth"
        if os.path.exists(p):
            self.actor_optimizer.load_state_dict(torch.load(p, map_location=maploc))

        # ----- Critic -----
        p = f"{directory}/{filename}_critic.pth"
        if os.path.exists(p):
            self.critic.load_state_dict(torch.load(p, map_location=maploc))
        p = f"{directory}/{filename}_critic_target.pth"
        if os.path.exists(p):
            self.critic_target.load_state_dict(torch.load(p, map_location=maploc))
        p = f"{directory}/{filename}_critic_optimizer.pth"
        if os.path.exists(p):
            self.critic_optimizer.load_state_dict(torch.load(p, map_location=maploc))

        # ----- Mapper (IEQN 전용) -----
        if getattr(self, "use_ieqn", False):
            p = f"{directory}/{filename}_mapper.pth"
            if os.path.exists(p):
                self.mapper.load_state_dict(torch.load(p, map_location=maploc))
            p = f"{directory}/{filename}_mapper_target.pth"
            if os.path.exists(p):
                self.mapper_target.load_state_dict(torch.load(p, map_location=maploc))
            p = f"{directory}/{filename}_mapper_optimizer.pth"
            if os.path.exists(p):
                self.mapper_optimizer.load_state_dict(torch.load(p, map_location=maploc))

        # ----- Checkpoint actor -----
        p = f"{directory}/{filename}_checkpoint_actor.pth"
        if os.path.exists(p):
            self.checkpoint_actor.load_state_dict(torch.load(p, map_location=maploc))

        # ----- Entropy coefficient -----
        if self.ent_coef_auto:
            p = f"{directory}/{filename}_log_ent_coef.pth"
            if os.path.exists(p):
                loaded = torch.load(p, map_location=maploc)
                # 핵심: 텐서 객체 교체 X, data만 복사
                self.log_ent_coef.data.copy_(loaded.to(maploc).data)
            p = f"{directory}/{filename}_ent_coef_optimizer.pth"
            if os.path.exists(p):
                self.ent_coef_optimizer.load_state_dict(torch.load(p, map_location=maploc))
        else:
            p = f"{directory}/{filename}_ent_coef_tensor.pth"
            if os.path.exists(p):
                loaded = torch.load(p, map_location=maploc)
                self.ent_coef_tensor = loaded.to(maploc).detach()