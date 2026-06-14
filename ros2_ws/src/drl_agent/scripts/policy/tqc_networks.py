"""Neural-network definitions and loss for the TQC agent.

Extracted verbatim from ``tqc_agent.py`` so the network architecture (Gaussian
``Actor``, distributional ``Critic``) and the TQC ``quantile_huber_loss`` live in
one place, separate from the agent's training / optimization / checkpoint logic.
``tqc_agent.py`` re-imports these names, so ``from tqc_agent import Actor`` etc.
still works and behaviour is unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
