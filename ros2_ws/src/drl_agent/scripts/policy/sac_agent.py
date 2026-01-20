import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import buffer

# ===== SAC constants / numerics =====
LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-6


def AvgL1Norm(x, eps=1e-8):
    """TD7와 동일한 평균 L1 정규화(필요 시 재사용)"""
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class Actor(nn.Module):
    """SAC 확률 정책: N(μ, σ) → tanh-squash with log-prob correction"""
    def __init__(self, state_dim, action_dim, hdim=256, activ=F.relu):
        super().__init__()
        self.activ = activ
        self.l1 = nn.Linear(state_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.mean_layer = nn.Linear(hdim, action_dim)
        self.log_std_layer = nn.Linear(hdim, action_dim)

    def forward(self, state):
        a = self.activ(self.l1(state))
        a = self.activ(self.l2(a))
        mean = self.mean_layer(a)
        log_std = self.log_std_layer(a)
        # squash to [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def sample(self, state):
        """Reparameterization trick + tanh + log-prob correction"""
        mean, log_std = self.forward(state)
        std = log_std.exp().clamp_min(1e-6)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()                             # reparameterized sample
        y_t = torch.tanh(x_t)
        y_t = y_t.clamp(-0.999999, 0.999999)              # numerical safety
        action = y_t

        # log π(a|s) with tanh correction
        log_prob = normal.log_prob(x_t)                    # per-dim
        log_prob -= torch.log(1 - y_t.pow(2) + EPS)        # tanh correction
        log_prob = log_prob.sum(1, keepdim=True)           # sum over dims

        mean = torch.tanh(mean)                            # deterministic action (eval)
        return action, log_prob, mean


class Critic(nn.Module):
    """Twin Q networks: return [Q1, Q2] concatenated on dim=1"""
    def __init__(self, state_dim, action_dim, hdim=256, activ=F.elu):
        super().__init__()
        self.activ = activ
        # Q1
        self.q1_l1 = nn.Linear(state_dim + action_dim, hdim)
        self.q1_l2 = nn.Linear(hdim, hdim)
        self.q1_l3 = nn.Linear(hdim, 1)
        # Q2
        self.q2_l1 = nn.Linear(state_dim + action_dim, hdim)
        self.q2_l2 = nn.Linear(hdim, hdim)
        self.q2_l3 = nn.Linear(hdim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # Q1
        q1 = self.activ(self.q1_l1(sa))
        q1 = self.activ(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        # Q2
        q2 = self.activ(self.q2_l1(sa))
        q2 = self.activ(self.q2_l2(q2))
        q2 = self.q2_l3(q2)
        return torch.cat([q1, q2], 1)

    def q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.activ(self.q1_l1(sa))
        q1 = self.activ(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        return q1


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, hp, log_dir=None):
        # -----------------
        # Hyperparameters
        # -----------------
        self.hyperparameters = self.prep_hyperparameters(hp)

        # Generic
        self.discount = self.hyperparameters["discount"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.buffer_size = self.hyperparameters["buffer_size"]
        # SAC는 소프트 업데이트(Polyak)가 표준
        self.tau = self.hyperparameters.get("tau", 0.005)
        # (TD7 호환을 위해 남겨두되 기본적으로 사용하지 않음)
        self.target_update_rate = self.hyperparameters.get("target_update_rate", None)

        # SAC specific
        # 기본: 매 스텝 actor/alpha/critic 업데이트(딜레이 원하면 2 등으로 조절)
        self.policy_freq = self.hyperparameters.get("policy_freq", 1)
        self.autotune_entropy = self.hyperparameters.get("autotune_entropy", True)
        self.target_entropy_scale = self.hyperparameters.get("target_entropy_scale", 1.0)
        self.initial_alpha = self.hyperparameters.get("initial_alpha", 0.2)
        self.alpha_lr = self.hyperparameters.get("alpha_lr", 3e-4)

        # Actor
        self.actor_hdim = self.hyperparameters.get("actor_hdim", 256)
        self.actor_activ = self.hyperparameters.get("actor_activ", F.relu)
        self.actor_lr = self.hyperparameters.get("actor_lr", 3e-4)

        # Critic
        self.critic_hdim = self.hyperparameters.get("critic_hdim", 256)
        self.critic_activ = self.hyperparameters.get("critic_activ", F.elu)
        self.critic_lr = self.hyperparameters.get("critic_lr", 3e-4)

        # Checkpointing (TD7 스타일 유지)
        self.reset_weight = self.hyperparameters.get("reset_weight", 0.9)
        self.steps_before_checkpointing = self.hyperparameters.get("steps_before_checkpointing", 40000)
        self.max_eps_when_checkpointing = self.hyperparameters.get("max_eps_when_checkpointing", 10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------
        # Networks & Opts
        # -----------------
        self.actor = Actor(state_dim, action_dim, self.actor_hdim, self.actor_activ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(state_dim, action_dim, self.critic_hdim, self.critic_activ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.critic_target = Critic(state_dim, action_dim, self.critic_hdim, self.critic_activ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # checkpoint actor (평가/스냅샷 용)
        self.checkpoint_actor = Actor(state_dim, action_dim, self.actor_hdim, self.actor_activ).to(self.device)
        self.checkpoint_actor.load_state_dict(self.actor.state_dict())

        # Entropy temperature α
        if self.autotune_entropy:
            self.target_entropy = -action_dim * self.target_entropy_scale
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = float(self.initial_alpha)

        # Replay buffer (TD7의 LAP 사용: normalize_actions/prioritized 활성)
        self.replay_buffer = buffer.LAP(
            state_dim,
            action_dim,
            self.device,
            self.buffer_size,
            self.batch_size,
            max_action,
            normalize_actions=True,
            prioritized=True,
        )

        self.max_action = max_action
        self.training_steps = 0

        # Checkpointing trackers
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # (TD7 유산) 값 트래킹: 필요 시 로깅에 사용
        self.max = -1e8
        self.min = 1e8

        # Logger
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)

    @staticmethod
    def prep_hyperparameters(hyperparameters):
        """Map string activations to functions, keep others intact."""
        activation_functions = {
            "elu": F.elu,
            "relu": F.relu,
        }
        if "actor_activ" in hyperparameters and isinstance(hyperparameters["actor_activ"], str):
            hyperparameters["actor_activ"] = activation_functions[hyperparameters["actor_activ"]]
        if "critic_activ" in hyperparameters and isinstance(hyperparameters["critic_activ"], str):
            hyperparameters["critic_activ"] = activation_functions[hyperparameters["critic_activ"]]
        return hyperparameters

    @staticmethod
    def soft_update(target: nn.Module, source: nn.Module, tau: float):
        """Polyak averaging: target ← (1-τ) target + τ source"""
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    def train(self):
        """SAC training step"""
        self.training_steps += 1
    
        state, action, next_state, reward, not_done = self.replay_buffer.sample()
    
        # -------- Critic update --------
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q_all = self.critic_target(next_state, next_action)      # [B,2]
            target_q_min = torch.min(target_q_all[:, 0:1], target_q_all[:, 1:2])
            target_q = reward + not_done * self.discount * (target_q_min - self.alpha * next_log_prob)
    
            # tracking only
            self.max = max(self.max, float(target_q.max()))
            self.min = min(self.min, float(target_q.min()))
    
        current_q_all = self.critic(state, action)                          # [B,2]
        critic_loss = (
            F.mse_loss(current_q_all[:, 0:1], target_q)
            + F.mse_loss(current_q_all[:, 1:2], target_q)
        )
    
        # TD abs for PER
        td_abs = (current_q_all - target_q).abs().detach()
    
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
        # -------- PER priority update --------
        min_pri = self.hyperparameters.get("min_priority", 1.0)
        pri_alpha = self.hyperparameters.get("priority_alpha", 0.6)
        priority = td_abs.max(1)[0].clamp(min=min_pri).pow(pri_alpha)
        self.replay_buffer.update_priority(priority)
    
        # -------- Actor + Alpha update --------
        actor_loss = None
        if self.training_steps % self.policy_freq == 0:
            new_action, log_prob, _ = self.actor.sample(state)
            q_new_all = self.critic(state, new_action)
            q_new = torch.min(q_new_all[:, 0:1], q_new_all[:, 1:2])         # min(Q1,Q2)
    
            actor_loss = (self.alpha * log_prob - q_new).mean()
    
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
    
            if self.autotune_entropy:
                # 표준 SAC: log_alpha에 대한 손실
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().item())
    
        # -------- Target soft-update (Polyak) --------
        # SAC 기본: 매 스텝 소프트업데이트. (원하면 target_update_rate로 간헐적 실행 가능)
        # if (self.target_update_rate is None) or (self.training_steps % int(self.target_update_rate) == 0):
        #     self.soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
    
        # PER max priority 주기 리셋(옵션)
        if self.training_steps % 1000 == 0:
            self.replay_buffer.reset_max_priority()
    
        # -------- Logging --------
        if self.training_steps % 100 == 0:
            self.writer.add_scalar("loss/critic", critic_loss.item(), self.training_steps)
            if actor_loss is not None:
                self.writer.add_scalar("loss/actor", actor_loss.item(), self.training_steps)
            if self.autotune_entropy:
                self.writer.add_scalar("alpha/value", self.alpha, self.training_steps)
            self.writer.add_scalar("Q/mean", current_q_all.mean().item(), self.training_steps)
            self.writer.add_scalar("Q/max", self.max, self.training_steps)
            self.writer.add_scalar("Q/min", self.min, self.training_steps)

    def select_action(self, state, use_checkpoint: bool = False, use_exploration: bool = True):
        """TD7トレーナ互換のAPI。SACでは追加ノイズ不要:
        - use_exploration=True: stochastic (sample)
        - use_exploration=False: deterministic (mean)
        """
        with torch.no_grad():
            s = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
            actor = self.checkpoint_actor if use_checkpoint else self.actor
            if use_exploration:
                a, _, _ = actor.sample(s)   # stochastic
                out = a
            else:
                _, _, m = actor.sample(s)   # deterministic (mean)
                out = m
            out = out.clamp(-1, 1).cpu().numpy().flatten()
            return out * self.max_action

    def train_and_checkpoint(self, ep_timesteps, ep_return):
        """エピソード終了時のチェックポイント処理（TD7スタイル）"""
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # 現在のポリシーの評価を早期終了
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # チェックポイントの更新
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.train_and_reset()

    def train_and_reset(self):
        """バッチ訓練とリセット"""
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.steps_before_checkpointing:
                self.best_min_return *= self.reset_weight
                self.max_eps_before_update = self.max_eps_when_checkpointing
            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

    def save(self, directory, filename):
        """モデルパラメータの保存（αは値のみ保存し，optimizer stateは別途保存）"""
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), f"{directory}/{filename}_actor_optimizer.pth")

        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        torch.save(self.critic_optimizer.state_dict(), f"{directory}/{filename}_critic_optimizer.pth")

        torch.save(self.checkpoint_actor.state_dict(), f"{directory}/{filename}_checkpoint_actor.pth")

        if self.autotune_entropy:
            torch.save({
                "log_alpha_value": float(self.log_alpha.detach().cpu().item()),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            }, f"{directory}/{filename}_alpha.pth")

    def load(self, directory, filename):
        """モデルパラメータの読み込み（log_alphaはパラメタオブジェクトを再作成せず値だけ注入）"""
        map_loc = self.device

        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", map_location=map_loc))
        self.actor_optimizer.load_state_dict(torch.load(f"{directory}/{filename}_actor_optimizer.pth", map_location=map_loc))

        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", map_location=map_loc))
        self.critic_target.load_state_dict(torch.load(f"{directory}/{filename}_critic_target.pth", map_location=map_loc))
        self.critic_optimizer.load_state_dict(torch.load(f"{directory}/{filename}_critic_optimizer.pth", map_location=map_loc))

        self.checkpoint_actor.load_state_dict(torch.load(f"{directory}/{filename}_checkpoint_actor.pth", map_location=map_loc))

        if self.autotune_entropy:
            alpha_dict = torch.load(f"{directory}/{filename}_alpha.pth", map_location=map_loc)
            # 既存のself.log_alpha(leaf parameter)に値だけをコピーしてoptimizerのひも付けを維持
            self.log_alpha.data.copy_(torch.tensor(alpha_dict["log_alpha_value"], device=self.device))
            self.alpha_optimizer.load_state_dict(alpha_dict["alpha_optimizer"])
            self.alpha = float(self.log_alpha.exp())