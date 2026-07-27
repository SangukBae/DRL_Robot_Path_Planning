import torch
import torch.nn as nn
import torch.nn.functional as F  # used in PPO value loss (mse_loss)
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time


# ---------------------------------------------------------------------------
# Rollout Buffer (on-policy, no experience replay)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-length on-policy rollout buffer for PPO."""

    def __init__(self, n_steps, state_dim, action_dim):
        self.n_steps    = n_steps
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.ptr      = 0
        self.full     = False
        self.states    = np.zeros((self.n_steps, self.state_dim),  dtype=np.float32)
        self.actions   = np.zeros((self.n_steps, self.action_dim), dtype=np.float32)
        self.rewards   = np.zeros((self.n_steps,),                dtype=np.float32)
        self.dones     = np.zeros((self.n_steps,),                dtype=np.float32)
        self.values    = np.zeros((self.n_steps,),                dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps,),                dtype=np.float32)

    def add(self, state, action, reward, done, value, log_prob):
        self.states[self.ptr]    = np.asarray(state,    dtype=np.float32).ravel()
        self.actions[self.ptr]   = np.asarray(action,   dtype=np.float32).ravel()
        self.rewards[self.ptr]   = float(reward)
        self.dones[self.ptr]     = float(done)
        self.values[self.ptr]    = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def is_full(self):
        return self.full

    def compute_returns_and_advantages(self, last_value, last_done, gamma, gae_lambda):
        """GAE advantage estimation."""
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value        = float(last_value)
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]
            delta      = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae   = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def get_minibatches(self, advantages, returns, batch_size, device):
        """Yield random mini-batches over the full rollout."""
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)
        start = 0
        while start < self.n_steps:
            idx = indices[start:start + batch_size]
            start += batch_size
            if len(idx) == 0:
                break
            yield (
                torch.tensor(self.states[idx],    dtype=torch.float32, device=device),
                torch.tensor(self.actions[idx],   dtype=torch.float32, device=device),
                torch.tensor(advantages[idx],     dtype=torch.float32, device=device),
                torch.tensor(returns[idx],        dtype=torch.float32, device=device),
                torch.tensor(self.log_probs[idx], dtype=torch.float32, device=device),
            )


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared MLP feature extractor → policy head (mu + log_std param) + value head.
    SB3 PPO default: [64, 64] tanh, separate policy and value networks.
    """

    def __init__(self, state_dim, action_dim, net_arch=(64, 64)):
        super().__init__()

        # Policy network
        policy_layers = []
        in_dim = state_dim
        for hidden in net_arch:
            policy_layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        self.policy_net = nn.Sequential(*policy_layers)
        self.mu_head    = nn.Linear(in_dim, action_dim)

        # Value network (separate, same arch)
        value_layers = []
        in_dim = state_dim
        for hidden in net_arch:
            value_layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        self.value_net  = nn.Sequential(*value_layers)
        self.value_head = nn.Linear(in_dim, 1)

        # Log std as a learnable parameter (not state-dependent, SB3 default)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward_actor(self, state):
        latent = self.policy_net(state)
        mean   = self.mu_head(latent)
        return mean

    def forward_critic(self, state):
        latent = self.value_net(state)
        return self.value_head(latent)

    def forward(self, state):
        return self.forward_actor(state), self.forward_critic(state)

    def evaluate_actions(self, state, actions):
        """
        Returns (values, log_prob, entropy) for the given (state, action) pairs.
        SB3 PPO style: DiagGaussian, no tanh squashing.
        actions must be the raw Normal samples stored in the rollout buffer.
        """
        mean   = self.forward_actor(state)
        std    = self.log_std.exp().expand_as(mean)
        dist   = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=1)
        entropy  = dist.entropy().sum(dim=1)
        values   = self.forward_critic(state).squeeze(-1)
        return values, log_prob, entropy

    def get_value(self, state):
        return self.forward_critic(state)

    def get_action_and_value(self, state, deterministic=False):
        """
        Returns (action_raw, log_prob_scalar, value_scalar).
        SB3 PPO style: DiagGaussian, no tanh squashing.
        action_raw is the unclipped Normal sample. The caller must clip to
        [-1, 1] before passing to the environment, but should store action_raw
        in the rollout buffer so evaluate_actions can recompute log_prob exactly.
        """
        mean   = self.forward_actor(state)
        std    = self.log_std.exp().expand_as(mean)
        dist   = torch.distributions.Normal(mean, std)

        x = mean if deterministic else dist.rsample()

        log_prob = dist.log_prob(x).sum(dim=1)
        value    = self.forward_critic(state).squeeze(-1)
        return x, log_prob, value


# ---------------------------------------------------------------------------
# SB3 PPO Agent
# ---------------------------------------------------------------------------

class SB3PPOAgent:
    """
    Standalone SB3-style PPO agent (on-policy, no gymnasium dependency).

    Interface:
      select_action(state, deterministic=False) -> (action, log_prob, value)
      add_rollout(state, action, reward, done, value, log_prob)
      rollout_full() -> bool
      get_value(state) -> float
      train(last_value, last_done)
      clear_rollout()
      save(directory, file_name)
      load(directory, file_name)
      replay_buffer = None  (on-policy: no replay buffer)
    """

    def __init__(self, state_dim, action_dim, max_action, hyperparameters, log_dir=None):
        hp = dict(hyperparameters or {})

        # Hyperparameters
        self.learning_rate  = float(hp.get("learning_rate", 3e-4))
        self.n_steps        = int(hp.get("n_steps", 2048))
        self.batch_size     = int(hp.get("batch_size", 64))
        self.n_epochs       = int(hp.get("n_epochs", 10))
        self.discount       = float(hp.get("discount", 0.99))
        self.gae_lambda     = float(hp.get("gae_lambda", 0.95))
        self.clip_range     = float(hp.get("clip_range", 0.2))
        self.ent_coef       = float(hp.get("ent_coef", 0.0))
        self.vf_coef        = float(hp.get("vf_coef", 0.5))
        self.max_grad_norm  = float(hp.get("max_grad_norm", 0.5))
        self.net_arch       = list(hp.get("net_arch", [64, 64]))

        self.max_action     = float(max_action)
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # No replay buffer (on-policy)
        self.replay_buffer = None

        # Network
        self.policy    = ActorCritic(state_dim, action_dim, self.net_arch).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(self.n_steps, state_dim, action_dim)

        # Book-keeping
        self.training_steps = 0

        # TensorBoard
        self.log_dir = log_dir or ""
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir else SummaryWriter()
        except Exception:
            self.writer = SummaryWriter()

        # JSONL log
        base_dir = self.log_dir or getattr(self.writer, "log_dir", None) or os.getcwd()
        os.makedirs(base_dir, exist_ok=True)
        self.json_log_path = os.path.join(base_dir, "sb3_ppo_metrics.jsonl")

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, state, deterministic=False):
        """
        Returns (action_np, log_prob_float, value_float).
        deterministic=True for evaluation (mean action, no sampling).
        """
        with torch.no_grad():
            state_t = torch.tensor(
                np.asarray(state, dtype=np.float32).reshape(1, -1),
                device=self.device
            )
            action_t, log_prob_t, value_t = self.policy.get_action_and_value(
                state_t, deterministic=deterministic
            )
        action   = action_t.cpu().numpy().flatten()
        log_prob = float(log_prob_t.cpu().item())
        value    = float(value_t.cpu().item())
        return action, log_prob, value

    def add_rollout(self, state, action, reward, done, value, log_prob):
        """Add one transition to the rollout buffer."""
        self.rollout_buffer.add(state, action, reward, done, value, log_prob)

    def rollout_full(self):
        """Returns True when the rollout buffer is full."""
        return self.rollout_buffer.is_full()

    def get_value(self, state):
        """Bootstrap value for the last state."""
        with torch.no_grad():
            state_t = torch.tensor(
                np.asarray(state, dtype=np.float32).reshape(1, -1),
                device=self.device
            )
            v = self.policy.get_value(state_t)
        return float(v.cpu().item())

    def train(self, last_value, last_done):
        """
        Compute GAE advantages and run n_epochs of PPO updates.

        Args:
            last_value: bootstrap value (float) for the state after rollout ends
            last_done:  whether the last transition was terminal (bool/float)
        """
        if not self.rollout_buffer.is_full():
            return

        advantages, returns = self.rollout_buffer.compute_returns_and_advantages(
            last_value, last_done, self.discount, self.gae_lambda
        )

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        total_policy_loss  = 0.0
        total_value_loss   = 0.0
        total_entropy_loss = 0.0
        n_batches          = 0

        for _ in range(self.n_epochs):
            for states_b, actions_b, adv_b, ret_b, old_lp_b in self.rollout_buffer.get_minibatches(
                advantages, returns, self.batch_size, self.device
            ):
                values_b, log_prob_b, entropy_b = self.policy.evaluate_actions(states_b, actions_b)

                ratio = torch.exp(log_prob_b - old_lp_b)

                # PPO clipped objective
                policy_loss_1 = adv_b * ratio
                policy_loss_2 = adv_b * ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss   = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                value_loss    = F.mse_loss(ret_b, values_b)

                # Entropy loss
                entropy_loss  = -entropy_b.mean()

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss  += policy_loss.item()
                total_value_loss   += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_batches          += 1

        self.training_steps += 1

        if n_batches > 0:
            avg_policy_loss  = total_policy_loss  / n_batches
            avg_value_loss   = total_value_loss   / n_batches
            avg_entropy_loss = total_entropy_loss / n_batches

            if self.writer:
                self.writer.add_scalar("loss/policy",  avg_policy_loss,  self.training_steps)
                self.writer.add_scalar("loss/value",   avg_value_loss,   self.training_steps)
                self.writer.add_scalar("loss/entropy", avg_entropy_loss, self.training_steps)
            self._json_log(
                self.training_steps,
                policy_loss=avg_policy_loss,
                value_loss=avg_value_loss,
                entropy_loss=avg_entropy_loss,
            )

    def clear_rollout(self):
        """Reset the rollout buffer for the next collection phase."""
        self.rollout_buffer.reset()

    def save(self, directory, file_name):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.policy.state_dict(),
                   os.path.join(directory, f"{file_name}_actor.pth"))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(directory, f"{file_name}_optimizer.pth"))

    def load(self, directory, file_name):
        p = os.path.join(directory, f"{file_name}_actor.pth")
        if os.path.exists(p):
            self.policy.load_state_dict(torch.load(p, map_location=self.device))
        p = os.path.join(directory, f"{file_name}_optimizer.pth")
        if os.path.exists(p):
            self.optimizer.load_state_dict(torch.load(p, map_location=self.device))
