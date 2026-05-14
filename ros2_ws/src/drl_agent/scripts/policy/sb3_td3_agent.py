import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time


# ---------------------------------------------------------------------------
# Simple circular replay buffer (same interface as in sb3_sac_agent.py)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple circular replay buffer matching the .add() interface of LAP."""

    def __init__(self, state_dim, action_dim, device, max_size=1_000_000, batch_size=100):
        self.max_size   = int(max_size)
        self.ptr        = 0
        self.size       = 0
        self.batch_size = batch_size
        self.device     = device

        self.state      = np.zeros((self.max_size, state_dim),  dtype=np.float32)
        self.action     = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim),  dtype=np.float32)
        self.reward     = np.zeros((self.max_size, 1),          dtype=np.float32)
        self.not_done   = np.zeros((self.max_size, 1),          dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr]      = np.asarray(state,      dtype=np.float32).ravel()
        self.action[self.ptr]     = np.asarray(action,     dtype=np.float32).ravel()
        self.next_state[self.ptr] = np.asarray(next_state, dtype=np.float32).ravel()
        self.reward[self.ptr]     = float(reward)
        self.not_done[self.ptr]   = 1.0 - float(done)

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idx = np.random.randint(0, self.size, size=self.batch_size)
        return (
            torch.tensor(self.state[idx],      dtype=torch.float32, device=self.device),
            torch.tensor(self.action[idx],     dtype=torch.float32, device=self.device),
            torch.tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.reward[idx],     dtype=torch.float32, device=self.device),
            torch.tensor(self.not_done[idx],   dtype=torch.float32, device=self.device),
        )


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Deterministic MLP actor with tanh output."""

    def __init__(self, state_dim, action_dim, net_arch=(256, 256)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for hidden in net_arch:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return torch.tanh(self.net(state))


class Critic(nn.Module):
    """Twin MLP critic with a q1_forward helper (SB3 TD3 style)."""

    def __init__(self, state_dim, action_dim, net_arch=(256, 256)):
        super().__init__()
        self.qf1 = self._build_mlp(state_dim + action_dim, net_arch)
        self.qf2 = self._build_mlp(state_dim + action_dim, net_arch)

    @staticmethod
    def _build_mlp(in_dim, net_arch):
        layers = []
        for hidden in net_arch:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.qf1(sa), self.qf2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.qf1(sa)


# ---------------------------------------------------------------------------
# SB3 TD3 Agent
# ---------------------------------------------------------------------------

class SB3TD3Agent:
    """
    Standalone SB3-style TD3 agent (no gymnasium dependency).

    Interface matches tqc_agent.py (Agent):
      select_action(state, use_exploration=True, use_checkpoint=False) -> np.ndarray
      train()
      save(directory, file_name)
      load(directory, file_name)
      replay_buffer.add(state, action, next_state, reward, done)
    """

    def __init__(self, state_dim, action_dim, max_action, hyperparameters, log_dir=None):
        hp = dict(hyperparameters or {})

        # Hyperparameters
        self.learning_rate       = float(hp.get("learning_rate", 1e-3))
        self.buffer_size         = int(hp.get("buffer_size", 1_000_000))
        self.batch_size          = int(hp.get("batch_size", 100))
        self.tau                 = float(hp.get("tau", 0.005))
        self.discount            = float(hp.get("discount", 0.99))
        self.net_arch            = list(hp.get("net_arch", [256, 256]))
        self.policy_delay        = int(hp.get("policy_delay", 2))
        self.target_policy_noise = float(hp.get("target_policy_noise", 0.2))
        self.target_noise_clip   = float(hp.get("target_noise_clip", 0.5))
        self.exploration_noise   = float(hp.get("exploration_noise", 0.1))

        self.max_action = float(max_action)
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor         = Actor(state_dim, action_dim, self.net_arch).to(self.device)
        self.actor_target  = Actor(state_dim, action_dim, self.net_arch).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = Critic(state_dim, action_dim, self.net_arch).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.net_arch).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Freeze target parameters
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            state_dim, action_dim, self.device,
            max_size=self.buffer_size,
            batch_size=self.batch_size,
        )

        # Book-keeping
        self.training_steps = 0
        self._n_updates      = 0

        # TensorBoard
        self.log_dir = log_dir or ""
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir else SummaryWriter()
        except Exception:
            self.writer = SummaryWriter()

        # JSONL log
        base_dir = self.log_dir or getattr(self.writer, "log_dir", None) or os.getcwd()
        os.makedirs(base_dir, exist_ok=True)
        self.json_log_path = os.path.join(base_dir, "sb3_td3_metrics.jsonl")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _json_log(self, step, **metrics):
        rec = {"step": int(step), "time": float(time.time())}
        for k, v in metrics.items():
            try:
                val = float(v)
                if np.isfinite(val):
                    rec[k] = val
            except Exception:
                continue
        with open(self.json_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    @staticmethod
    def _polyak_update(source, target, tau):
        with torch.no_grad():
            for sp, tp in zip(source.parameters(), target.parameters()):
                tp.data.mul_(1.0 - tau)
                tp.data.add_(tau * sp.data)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, state, use_exploration=True, use_checkpoint=False):
        """
        Select action in normalized action space [-1, 1].
        use_exploration=True  → deterministic + Gaussian noise (std=exploration_noise)
        use_exploration=False → deterministic (no noise)
        """
        with torch.no_grad():
            state_t = torch.tensor(
                np.asarray(state, dtype=np.float32).reshape(1, -1),
                device=self.device
            )
            action = self.actor(state_t).cpu().numpy().flatten()

        if use_exploration:
            noise  = np.random.normal(0.0, self.exploration_noise, size=action.shape).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)
        else:
            action = np.clip(action, -1.0, 1.0)

        return action

    def train(self):
        """One gradient step of SB3 TD3."""
        if self.replay_buffer.size < self.batch_size:
            return

        self.training_steps += 1
        self._n_updates      += 1

        state, action, next_state, reward, not_done = self.replay_buffer.sample()

        # ---- Critic update ----
        with torch.no_grad():
            noise = torch.randn_like(action) * self.target_policy_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            next_q1, next_q2 = self.critic_target(next_state, next_action)
            next_q    = torch.min(next_q1, next_q2)
            target_q  = reward + not_done * self.discount * next_q

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = 0.0

        # ---- Delayed actor update ----
        if self._n_updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            actor_loss_val = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Polyak update both actor and critic targets
            self._polyak_update(self.actor,  self.actor_target,  self.tau)
            self._polyak_update(self.critic, self.critic_target, self.tau)

        # ---- Logging ----
        if self.writer:
            self.writer.add_scalar("loss/critic", critic_loss.item(), self.training_steps)
            if actor_loss_val != 0.0:
                self.writer.add_scalar("loss/actor", actor_loss_val, self.training_steps)
        self._json_log(
            self.training_steps,
            critic_loss=critic_loss.item(),
            actor_loss=actor_loss_val,
        )

    # train_and_checkpoint kept as alias for compatibility
    def train_and_checkpoint(self, ep_timesteps=None, ep_return=None):
        self.train()

    def save(self, directory, file_name):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(),
                   os.path.join(directory, f"{file_name}_actor.pth"))
        torch.save(self.actor_optimizer.state_dict(),
                   os.path.join(directory, f"{file_name}_actor_optimizer.pth"))
        torch.save(self.actor_target.state_dict(),
                   os.path.join(directory, f"{file_name}_actor_target.pth"))
        torch.save(self.critic.state_dict(),
                   os.path.join(directory, f"{file_name}_critic.pth"))
        torch.save(self.critic_target.state_dict(),
                   os.path.join(directory, f"{file_name}_critic_target.pth"))
        torch.save(self.critic_optimizer.state_dict(),
                   os.path.join(directory, f"{file_name}_critic_optimizer.pth"))

    def load(self, directory, file_name):
        def _load(fname, obj):
            p = os.path.join(directory, fname)
            if os.path.exists(p):
                obj.load_state_dict(torch.load(p, map_location=self.device))

        _load(f"{file_name}_actor.pth",            self.actor)
        _load(f"{file_name}_actor_optimizer.pth",  self.actor_optimizer)
        _load(f"{file_name}_actor_target.pth",     self.actor_target)
        _load(f"{file_name}_critic.pth",           self.critic)
        _load(f"{file_name}_critic_target.pth",    self.critic_target)
        _load(f"{file_name}_critic_optimizer.pth", self.critic_optimizer)
