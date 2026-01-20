import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import buffer


class ActorCriticNetwork(nn.Module):
    """A3C Actor-Critic Network with LSTM for navigation (stochastic policy + scaling)"""
    def __init__(self, state_dim, action_dim, hidden_dim=512, lstm_dim=512):
        super(ActorCriticNetwork, self).__init__()

        # Shared feature extraction layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # LSTM for long-term memory (Actor side)
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, batch_first=True)

        # Actor head (stochastic policy)
        self.actor_fc = nn.Linear(lstm_dim, 256)
        self.actor_mu = nn.Linear(256, action_dim)                 # mean head
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))  # learnable log-std

        # Critic head (value network)
        self.critic_fc = nn.Linear(hidden_dim, 256)
        self.critic_out = nn.Linear(256, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    # ---------- Actor utilities (변경 1) ----------
    def _actor_dist(self, lstm_out):
        """Build Normal(μ, σ) in pre-tanh space from LSTM features."""
        h = F.relu(self.actor_fc(lstm_out))
        mu = torch.tanh(self.actor_mu(h))                 # keep mean in [-1, 1]
        log_std = self.actor_log_std.clamp(-5, 2)
        std = log_std.exp()
        return D.Normal(mu, std)

    def _sample_action_and_stats(self, dist):
        """Sample tanh-squashed action and compute corrected log_prob & entropy."""
        z = dist.rsample()                                # reparameterization trick
        a = torch.tanh(z)                                 # squash to [-1, 1]
        # Tanh correction for log_prob: log π(a) = log N(z; μ, σ) - Σ log(1 - tanh(z)^2)
        log_prob = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)  # pre-tanh Normal entropy
        return a, log_prob, entropy

    # ---------- Action scaling to robot's physical limits (변경 2) ----------
    def scale_action(self, a_raw):
        """
        Scale tanh-squashed action in [-1,1]^2 to physical (v, ω).
        v ∈ [0, 0.26] m/s, ω ∈ [-2.7, 2.7] rad/s.
        """
        v = 0.26 * (a_raw[..., 0] + 1.0) / 2.0   # [-1,1]→[0,1]→[0,0.26]
        w = 2.7  *  a_raw[..., 1]                # [-1,1]→[-2.7,2.7]
        return torch.stack([v, w], dim=-1)

    def unscale_action(self, a_scaled):
        """
        Inverse of scale_action: physical (v, ω) → a_raw in [-1,1]^2.
        """
        v_raw = (a_scaled[..., 0] / 0.26) * 2.0 - 1.0
        w_raw =  a_scaled[..., 1] / 2.7
        return torch.stack([v_raw, w_raw], dim=-1).clamp(-1.0, 1.0)

    # ---------- Forward / Evaluate (변경 2) ----------
    def forward(self, state, hidden_state=None, return_stats=False):
        """
        If return_stats:
            returns (a_scaled, value, hidden_state, log_prob, entropy)
        else:
            returns (a_scaled, value, hidden_state)
        """
        # Shared backbone
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        features = F.relu(self.fc3(x))

        # Critic from backbone features
        v = F.relu(self.critic_fc(features))
        value = self.critic_out(v)

        # Actor through LSTM
        lstm_in = features.unsqueeze(1) if features.dim() == 2 else features  # (B,1,H) or (B,T,H)
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(lstm_in)
        else:
            lstm_out, hidden_state = self.lstm(lstm_in, hidden_state)

        # Use last step if sequence
        if lstm_out.dim() == 3 and lstm_out.size(1) == 1:
            lstm_out = lstm_out.squeeze(1)                   # (B,H)
        elif lstm_out.dim() == 3:
            lstm_out = lstm_out[:, -1]                       # (B,H)

        dist = self._actor_dist(lstm_out)

        if return_stats:
            a_raw, log_prob, entropy = self._sample_action_and_stats(dist)
            a_scaled = self.scale_action(a_raw)
            return a_scaled, value, hidden_state, log_prob, entropy
        else:
            # Inference: use mean action
            a_raw = torch.tanh(dist.mean)
            a_scaled = self.scale_action(a_raw)
            return a_scaled, value, hidden_state

    def evaluate_actions(self, state, actions, hidden_state=None):
        """
        Compute log_prob, entropy, value for given (state, actions in physical scale).
        Returns: (log_prob, entropy, value)
        """
        # Physical (v, ω) → [-1,1]^2 → atanh for pre-tanh Normal
        a_raw = self.unscale_action(actions).clamp(-0.999, 0.999)
        z = torch.atanh(a_raw)

        # Shared backbone
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        features = F.relu(self.fc3(x))

        # Critic
        v = F.relu(self.critic_fc(features))
        value = self.critic_out(v)

        # Actor LSTM
        lstm_in = features.unsqueeze(1) if features.dim() == 2 else features
        if hidden_state is None:
            lstm_out, _ = self.lstm(lstm_in)
        else:
            lstm_out, _ = self.lstm(lstm_in, hidden_state)

        if lstm_out.dim() == 3 and lstm_out.size(1) == 1:
            lstm_out = lstm_out.squeeze(1)
        elif lstm_out.dim() == 3:
            lstm_out = lstm_out[:, -1]

        dist = self._actor_dist(lstm_out)

        # log_prob with tanh correction
        log_prob = (dist.log_prob(z) - torch.log(1 - a_raw.pow(2) + 1e-6)).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy, value

    # ---------- Convenience (기존 함수와 호환) ----------
    def get_action(self, state, hidden_state=None):
        """Get scaled (v, ω) action for inference (mean action)."""
        action, _, hidden_state = self.forward(state, hidden_state, return_stats=False)
        return action, hidden_state

    def get_value(self, state):
        """Get value estimate only."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        features = F.relu(self.fc3(x))
        v = F.relu(self.critic_fc(features))
        value = self.critic_out(v)
        return value

class Agent(object):
    """A3C-LSTM based navigation agent (TD7 compatible interface)"""
    
    def __init__(self, state_dim, action_dim, max_action, hp, log_dir=None):
        # Store dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Process hyperparameters
        self.hyperparameters = self.prep_hyperparameters(hp)

        # ==== 변경 3: 온폴리시 전환 플래그/하이퍼 ====
        # 기본값을 논문 사양에 맞춤: γ=0.9, GAE λ=0.95, rollout_horizon=20
        self.use_on_policy   = self.hyperparameters.get("use_on_policy", True)
        self.discount        = self.hyperparameters.get("discount", 0.9)      # 논문 γ=0.9
        self.gae_lambda      = self.hyperparameters.get("gae_lambda", 0.95)   # GAE(선택)
        self.rollout_horizon = self.hyperparameters.get("rollout_horizon", 20)

        # 공통 하이퍼
        self.batch_size      = self.hyperparameters["batch_size"]
        self.buffer_size     = self.hyperparameters["buffer_size"]
        self.learning_rate   = self.hyperparameters.get("learning_rate", 1e-4)
        self.entropy_coef    = self.hyperparameters.get("entropy_coef", 0.01)
        self.value_loss_coef = self.hyperparameters.get("value_loss_coef", 0.5)
        self.max_grad_norm   = self.hyperparameters.get("max_grad_norm", 0.5)

        # (오프폴리시용 값은 유지하되 온폴리시에서 사용하지 않음)
        self.target_update_rate = self.hyperparameters.get("target_update_rate", 1000)

        # Exploration parameters (TD7 compatible; 온폴리시에서는 비활성)
        if self.use_on_policy:
            self.exploration_noise = 0.0
            self.exploration_noise_min = 0.0
        else:
            self.exploration_noise = self.hyperparameters.get("exploration_noise", 0.1)
            self.exploration_noise_min = self.hyperparameters.get("exploration_noise_min", 0.01)
        self.exploration_noise_decay_steps = self.hyperparameters.get("exploration_noise_decay_steps", 100000)

        # Checkpointing parameters (TD7 compatible; 온폴리시에서는 비활성 권장)
        self.use_checkpoints = False if self.use_on_policy else self.hyperparameters.get("use_checkpoints", True)
        self.reset_weight = self.hyperparameters.get("reset_weight", 0.9)
        self.steps_before_checkpointing = self.hyperparameters.get("steps_before_checkpointing", 75000)
        self.max_eps_when_checkpointing = self.hyperparameters.get("max_eps_when_checkpointing", 100)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim).to(self.device)

        # 온폴리시: 타깃넷/체크포인트 넷 사용 안 함
        if self.use_on_policy:
            self.actor_critic_target = None
            self.checkpoint_network  = None
        else:
            self.actor_critic_target = ActorCriticNetwork(state_dim, action_dim).to(self.device)
            self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())
            self.checkpoint_network  = ActorCriticNetwork(state_dim, action_dim).to(self.device)
            self.checkpoint_network.load_state_dict(self.actor_critic.state_dict())

        # Optimizer (논문: Adam eps=0.003)
        # === 학습률 분리 적용 ===
        # 1. 하이퍼파라미터에서 Actor/Critic 학습률 가져오기
        self.policy_lr = self.hyperparameters.get("policy_lr", 1e-4) # 논문의 β
        self.value_lr = self.hyperparameters.get("value_lr", 1e-4)   # 논문의 α

        # 2. 네트워크 파라미터를 actor와 critic 그룹으로 분리
        #    (Shared backbone 파라미터는 actor 쪽에서 업데이트 담당)
        actor_params = list(self.actor_critic.fc1.parameters()) + \
                       list(self.actor_critic.fc2.parameters()) + \
                       list(self.actor_critic.fc3.parameters()) + \
                       list(self.actor_critic.lstm.parameters()) + \
                       list(self.actor_critic.actor_fc.parameters()) + \
                       list(self.actor_critic.actor_mu.parameters()) + \
                       [self.actor_critic.actor_log_std]
                       
        critic_params = list(self.actor_critic.critic_fc.parameters()) + \
                        list(self.actor_critic.critic_out.parameters())

        # 3. 별도의 옵티마이저 생성
        self.policy_optimizer = torch.optim.Adam(
            actor_params, lr=self.policy_lr, eps=0.003
        )
        self.value_optimizer = torch.optim.Adam(
            critic_params, lr=self.value_lr, eps=0.003
        )
        # =======================

        # Replay buffer
        if self.use_on_policy:
            self.replay_buffer = None    # 온폴리시: 롤아웃 리스트 사용
            self.rollout = []            # 온폴리시 임시 저장소
        else:
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
            self.rollout = None

        # Hidden states for LSTM
        self.hidden_state = None
        self.target_hidden_state = None
        self.checkpoint_hidden_state = None

        # Training tracking (TD7 compatible fields는 유지)
        self.training_steps = 0
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Logging
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir) if log_dir else None
        
    @staticmethod
    def prep_hyperparameters(hyperparameters):
        """Process hyperparameters with defaults and required casts"""
        defaults = {
            # ---- 논문/온폴리시 기본 ----
            "use_on_policy": True,
            "discount": 0.9,          # ← 0.99 → 0.9
            "gae_lambda": 0.95,
            "rollout_horizon": 20,

            # ---- 공통 ----
            "batch_size": 256,
            "buffer_size": 1_000_000,
            "learning_rate": 1e-4,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_update_rate": 1000,

            # ---- 오프폴리시(TD7)용(온폴리시에선 미사용) ----
            "exploration_noise": 0.1,
            "exploration_noise_min": 0.01,
            "exploration_noise_decay_steps": 100000,
            "use_checkpoints": True,
            "reset_weight": 0.9,
            "steps_before_checkpointing": 75000,
            "max_eps_when_checkpointing": 100,
            "alpha": 0.6,
            "min_priority": 0.01,
        }

        for k, v in defaults.items():
            if k not in hyperparameters:
                hyperparameters[k] = v

        # 정수 캐스팅
        hyperparameters["buffer_size"] = int(hyperparameters["buffer_size"])
        hyperparameters["batch_size"] = int(hyperparameters["batch_size"])
        hyperparameters["target_update_rate"] = int(hyperparameters["target_update_rate"])
        return hyperparameters

    
    def reset_hidden_states(self, batch_size=1):
        """Reset LSTM hidden states"""
        self.hidden_state = (
            torch.zeros(1, batch_size, 512).to(self.device),
            torch.zeros(1, batch_size, 512).to(self.device)
        )
        self.target_hidden_state = (
            torch.zeros(1, batch_size, 512).to(self.device),
            torch.zeros(1, batch_size, 512).to(self.device)
        )
        self.checkpoint_hidden_state = (
            torch.zeros(1, batch_size, 512).to(self.device),
            torch.zeros(1, batch_size, 512).to(self.device)
        )

    # ✅ 전이 저장 훅 (온폴리시 전환)
    def store_transition(self, s, a, s2, r, done):
        """온폴리시 롤아웃 저장 (논문 부합). 리플레이버퍼는 사용하지 않음."""
        if not getattr(self, "use_on_policy", False):
            raise RuntimeError("Set use_on_policy=True for A3C-like on-policy training.")
        # 롤아웃: (state, action(물리단위), reward, done, next_state)
        self.rollout.append((s, a, r, done, s2))


    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        """확률정책에서 샘플링된 액션을 물리 단위(v, ω)로 반환."""

        # 온폴리시 모드인 경우
        if getattr(self, "use_on_policy", False):
            with torch.no_grad():
                s = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

                if use_exploration:
                    # 훈련 모드: 확률 분포에서 샘플링
                    action, value, self.hidden_state, logp, ent = self.actor_critic(
                        s, self.hidden_state, return_stats=True
                    )
                else:
                    # 평가 모드: 평균 액션 사용 (deterministic)
                    action, value, self.hidden_state = self.actor_critic(
                        s, self.hidden_state, return_stats=False
                    )

                return action.cpu().numpy().flatten()

        # 오프폴리시 모드인 경우 (TD7 호환성 유지)
        else:
            with torch.no_grad():
                s = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

                # 체크포인트 네트워크 사용 여부
                if use_checkpoint and self.checkpoint_network is not None:
                    action, _, self.checkpoint_hidden_state = self.checkpoint_network(
                        s, self.checkpoint_hidden_state, return_stats=False
                    )
                else:
                    action, _, self.hidden_state = self.actor_critic(
                        s, self.hidden_state, return_stats=False
                    )

                # 오프폴리시에서 탐색 노이즈 추가
                if use_exploration and self.exploration_noise > 0:
                    noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
                    action = action.cpu().numpy().flatten() + noise
                    action = np.clip(action, 0, self.max_action)  # 물리적 한계 내로 클리핑
                else:
                    action = action.cpu().numpy().flatten()

                return action

    def _batchify(self, traj):
        # traj: list of (s, a, r, done, s2)
        states      = torch.tensor(np.stack([t[0] for t in traj]), dtype=torch.float, device=self.device)
        actions     = torch.tensor(np.stack([t[1] for t in traj]), dtype=torch.float, device=self.device)
        rewards     = torch.tensor(np.array([t[2] for t in traj], dtype=np.float32), device=self.device).unsqueeze(-1)
        dones       = torch.tensor(np.array([t[3] for t in traj], dtype=np.float32), device=self.device).unsqueeze(-1)
        next_states = torch.tensor(np.stack([t[4] for t in traj]), dtype=torch.float, device=self.device)
        return states, actions, rewards, dones, next_states

    def _compute_returns_advantages(self, rewards, dones, values, next_value):
        # GAE(λ)
        gamma, lam = self.discount, self.gae_lambda
        T = rewards.size(0)
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(T)):
            v_next = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * (1.0 - dones[t]) * v_next - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values
        return returns, adv

    def update_on_policy(self):
        """rollout_horizon 만큼 쌓인 온폴리시 데이터를 사용해 A3C/A2C 업데이트"""
        if len(self.rollout) == 0:
            return

        states, actions, rewards, dones, next_states = self._batchify(self.rollout)

        # 값/로그우도/엔트로피 계산 (ActorCriticNetwork.evaluate_actions 필요)
        logp, entropy, values = self.actor_critic.evaluate_actions(states, actions, self.hidden_state)

        with torch.no_grad():
            # bootstrap V(s_T+1)
            _, next_v, _ = self.actor_critic(next_states[-1:].detach(), self.hidden_state)
            returns, advantages = self._compute_returns_advantages(rewards, dones, values, next_v)

        # 안정화: 어드밴티지 표준화(선택)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === 학습 로직 분리 적용 ===
        # 1. 손실 계산 (분리)
        policy_loss  = -(logp * advantages.detach()).mean()
        value_loss   = F.mse_loss(values, returns)
        entropy_loss = -self.entropy_coef * entropy.mean()

        # 2. Critic 네트워크 업데이트 (Value Loss 사용)
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True) # Actor 업데이트를 위해 그래프 유지
        self.value_optimizer.step()

        # 3. Actor 네트워크 업데이트 (Policy Loss + Entropy Loss 사용)
        self.policy_optimizer.zero_grad()
        actor_total_loss = policy_loss + entropy_loss
        actor_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        # ==========================
        
        # 로깅 (total_loss 변수만 맞춰줌)
        total_loss = actor_total_loss + self.value_loss_coef * value_loss

        # 로깅
        if self.writer:
            self.writer.add_scalar("loss/total",   total_loss.item(),  self.training_steps)
            self.writer.add_scalar("loss/policy",  policy_loss.item(), self.training_steps)
            self.writer.add_scalar("loss/value",   value_loss.item(),  self.training_steps)
            self.writer.add_scalar("loss/entropy", entropy_loss.item(),self.training_steps)
            self.writer.add_scalar("adv/mean",     advantages.mean().item(), self.training_steps)

        # 클리어 & 카운터
        self.rollout.clear()
        self.training_steps += 1
    
    def mask_hidden(self, hidden, done):
        """done==1인 배치의 hidden을 0으로 마스킹"""
        if hidden is None:
            return None
        h, c = hidden
        m = (1.0 - done.float()).view(1, -1, 1)  # shape (1,B,1)
        return (h * m, c * m)
    
    def train(self):
        """외부에 노출되는 공용 train(). 모드에 따라 분기."""
        if getattr(self, "use_on_policy", False):
            # 온폴리시: rollout_horizon에 도달했을 때만 업데이트 실행
            if len(self.rollout) >= self.rollout_horizon:
                self.update_on_policy()
            return
        else:
            # 오프폴리시(TD7 호환) 경로
            return self._train_offpolicy()


    def _train_offpolicy(self):
        """(기존) 오프폴리시 학습 경로: 리플레이버퍼/타깃넷 사용."""
        self.training_steps += 1

        if self.replay_buffer is None or self.actor_critic_target is None:
            raise RuntimeError(
                "Off-policy training requires replay_buffer and actor_critic_target. "
                "Set use_on_policy=True for on-policy A3C/A2C, or re-enable TD7 components."
            )

        # Sample from replay buffer
        states, actions, next_states, rewards, not_dones = self.replay_buffer.sample()

        # Reset hidden states for batch processing
        batch_size = states.size(0)
        hidden = (
            torch.zeros(1, batch_size, 512).to(self.device),
            torch.zeros(1, batch_size, 512).to(self.device)
        )
        target_hidden = (
            torch.zeros(1, batch_size, 512).to(self.device),
            torch.zeros(1, batch_size, 512).to(self.device)
        )

        # Forward pass through current network
        predicted_actions, values, _ = self.actor_critic(states, hidden)

        # Compute target values
        with torch.no_grad():
            _, next_values, _ = self.actor_critic_target(next_states, target_hidden)
            target_values = rewards + not_dones * self.discount * next_values

        # Losses
        # Value loss (TD error)
        value_loss = F.mse_loss(values, target_values)

        # Policy loss (advantage-based weighting of behavior cloning MSE)
        advantages = (target_values - values).detach()
        per_sample_mse = F.mse_loss(
            predicted_actions, actions, reduction='none'
        ).mean(dim=1, keepdim=True)
        policy_loss = -(advantages * per_sample_mse).mean()

        # Entropy bonus (approximate for continuous actions)
        action_std = predicted_actions.std(dim=1, keepdim=True).clamp(min=1e-6)
        entropy = action_std.log() + 0.5 * np.log(2 * np.pi * np.e)
        entropy_loss = -self.entropy_coef * entropy.mean()

        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Update priorities for prioritized replay
        with torch.no_grad():
            td_errors = (values - target_values).abs().squeeze()
            priorities = td_errors.clamp(min=self.hyperparameters["min_priority"]).pow(
                self.hyperparameters["alpha"]
            )
            self.replay_buffer.update_priority(priorities)

        # Update target network
        if self.training_steps % self.target_update_rate == 0:
            self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())
            self.replay_buffer.reset_max_priority()

        # Logging
        if self.writer:
            self.writer.add_scalar("loss/total",   total_loss.item(),  self.training_steps)
            self.writer.add_scalar("loss/value",   value_loss.item(),  self.training_steps)
            self.writer.add_scalar("loss/policy",  policy_loss.item(), self.training_steps)
            self.writer.add_scalar("loss/entropy", entropy_loss.item(),self.training_steps)
            self.writer.add_scalar("values/mean",  values.mean().item(), self.training_steps)
            self.writer.add_scalar("values/std",   values.std().item(),  self.training_steps)
            
    def train_and_checkpoint(self, ep_timesteps, ep_return):
        """Checkpointing logic (TD7 compatible)"""
        # 온폴리시 또는 체크포인트 비활성: 조용히 빠져나가기
        if getattr(self, "use_on_policy", False) or not getattr(self, "use_checkpoints", False):
            # 온폴리시에서는 에피소드 통계만 갱신(선택)
            self.timesteps_since_update += ep_timesteps
            self.min_return = min(self.min_return, ep_return)
            return

        # ---- 이하 기존 TD7 로직 유지 ----
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps
        self.min_return = min(self.min_return, ep_return)

        if self.min_return < self.best_min_return:
            self.train_and_reset()
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            if self.checkpoint_network is not None:
                self.checkpoint_network.load_state_dict(self.actor_critic.state_dict())
                self.checkpoint_hidden_state = self.hidden_state
            self.train_and_reset()
    
    def train_and_reset(self):
        """Batch training and reset"""
        # 온폴리시: timesteps_since_update 길이에 비례해 update 호출(대충 1회 이상)
        if getattr(self, "use_on_policy", False):
            updates = max(1, self.timesteps_since_update // max(1, self.rollout_horizon))
            for _ in range(updates):
                self.train()
        else:
            for _ in range(self.timesteps_since_update):
                self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

        # Reset LSTM hidden states
        self.reset_hidden_states(1)
    
    def save(self, directory, filename):
        """Save model (온/오프폴리시 모두 지원)"""
        # 메인 네트워크는 항상 저장
        torch.save(self.actor_critic.state_dict(), 
                  f"{directory}/{filename}_actor_critic.pth")
        
        # 온폴리시 모드가 아닐 때만 타겟/체크포인트 네트워크 저장
        if not getattr(self, "use_on_policy", False):
            if self.actor_critic_target is not None:
                torch.save(self.actor_critic_target.state_dict(), 
                          f"{directory}/{filename}_actor_critic_target.pth")
            if self.checkpoint_network is not None:
                torch.save(self.checkpoint_network.state_dict(), 
                          f"{directory}/{filename}_checkpoint.pth")
        
        # 옵티마이저 저장 (분리된 경우와 통합된 경우 모두 처리)
        if hasattr(self, 'policy_optimizer') and hasattr(self, 'value_optimizer'):
            # 온폴리시: Actor/Critic 분리 옵티마이저
            torch.save({
                'policy': self.policy_optimizer.state_dict(),
                'value': self.value_optimizer.state_dict()
            }, f"{directory}/{filename}_optimizers.pth")
        elif hasattr(self, 'optimizer'):
            # 오프폴리시: 단일 옵티마이저
            torch.save(self.optimizer.state_dict(), 
                      f"{directory}/{filename}_optimizer.pth")
        
        # Hidden states 저장
        hidden_states_dict = {'hidden': self.hidden_state}
        if hasattr(self, 'target_hidden_state'):
            hidden_states_dict['target_hidden'] = self.target_hidden_state
        if hasattr(self, 'checkpoint_hidden_state'):
            hidden_states_dict['checkpoint_hidden'] = self.checkpoint_hidden_state
        torch.save(hidden_states_dict, f"{directory}/{filename}_hidden_states.pth")
        
        # Training state 저장
        training_state_dict = {
            'training_steps': self.training_steps,
            'eps_since_update': self.eps_since_update,
            'timesteps_since_update': self.timesteps_since_update,
            'min_return': self.min_return,
            'best_min_return': self.best_min_return,
        }
        # exploration_noise는 온폴리시에서 0이지만 저장은 함
        if hasattr(self, 'exploration_noise'):
            training_state_dict['exploration_noise'] = self.exploration_noise
        
        torch.save(training_state_dict, f"{directory}/{filename}_training_state.pth")
    
    def load(self, directory, filename):
        """Load model (온/오프폴리시 모두 지원)"""
        # 메인 네트워크는 항상 로드
        self.actor_critic.load_state_dict(
            torch.load(f"{directory}/{filename}_actor_critic.pth", map_location=self.device)
        )
        
        # 온폴리시가 아닐 때만 타겟/체크포인트 네트워크 로드 시도
        if not getattr(self, "use_on_policy", False):
            # 타겟 네트워크 로드
            if self.actor_critic_target is not None:
                try:
                    self.actor_critic_target.load_state_dict(
                        torch.load(f"{directory}/{filename}_actor_critic_target.pth", 
                                  map_location=self.device)
                    )
                except FileNotFoundError:
                    print(f"Warning: Target network file not found, using main network weights")
                    self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())
            
            # 체크포인트 네트워크 로드
            if self.checkpoint_network is not None:
                try:
                    self.checkpoint_network.load_state_dict(
                        torch.load(f"{directory}/{filename}_checkpoint.pth", 
                                  map_location=self.device)
                    )
                except FileNotFoundError:
                    print(f"Warning: Checkpoint network file not found, using main network weights")
                    self.checkpoint_network.load_state_dict(self.actor_critic.state_dict())
        
        # 옵티마이저 로드 (분리/통합 모두 시도)
        try:
            # 먼저 분리 옵티마이저 시도 (온폴리시)
            opt_dict = torch.load(f"{directory}/{filename}_optimizers.pth", 
                                 map_location=self.device)
            if hasattr(self, 'policy_optimizer') and hasattr(self, 'value_optimizer'):
                self.policy_optimizer.load_state_dict(opt_dict['policy'])
                self.value_optimizer.load_state_dict(opt_dict['value'])
        except FileNotFoundError:
            # 단일 옵티마이저 시도 (오프폴리시 또는 이전 버전)
            try:
                if hasattr(self, 'optimizer'):
                    self.optimizer.load_state_dict(
                        torch.load(f"{directory}/{filename}_optimizer.pth", 
                                  map_location=self.device)
                    )
            except FileNotFoundError:
                print("Warning: Optimizer file not found, using fresh optimizer")
        
        # Hidden states 로드
        try:
            hidden_states = torch.load(f"{directory}/{filename}_hidden_states.pth", 
                                       map_location=self.device)
            self.hidden_state = hidden_states.get('hidden', None)
            if hasattr(self, 'target_hidden_state'):
                self.target_hidden_state = hidden_states.get('target_hidden', None)
            if hasattr(self, 'checkpoint_hidden_state'):
                self.checkpoint_hidden_state = hidden_states.get('checkpoint_hidden', None)
            
            # None이면 리셋
            if self.hidden_state is None:
                self.reset_hidden_states(1)
        except FileNotFoundError:
            print("Warning: Hidden states file not found, initializing fresh")
            self.reset_hidden_states(1)
        
        # Training state 로드
        try:
            training_state = torch.load(f"{directory}/{filename}_training_state.pth", 
                                        map_location=self.device)
            self.training_steps = training_state.get('training_steps', 0)
            self.eps_since_update = training_state.get('eps_since_update', 0)
            self.timesteps_since_update = training_state.get('timesteps_since_update', 0)
            self.min_return = training_state.get('min_return', 1e8)
            self.best_min_return = training_state.get('best_min_return', -1e8)
            if hasattr(self, 'exploration_noise'):
                self.exploration_noise = training_state.get('exploration_noise', 
                                                            self.exploration_noise)
        except FileNotFoundError:
            print("Warning: Training state file not found, using default values")
    