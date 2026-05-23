import os

import numpy as np
import torch


class LAP(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_size=1e6,
        batch_size=256,
        max_action=1,
        normalize_actions=True,
        prioritized=True,
    ):

        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, device=device)
            self.max_priority = 1

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        if self.prioritized:
            csum = torch.cumsum(self.priority[: self.size], 0)
            val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
            self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
        else:
            self.ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(
                self.next_state[self.ind], dtype=torch.float, device=self.device
            ),
            torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(
                self.not_done[self.ind], dtype=torch.float, device=self.device
            ),
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())

    def save(self, path: str):
        """Save buffer to <path>.npz (+ <path>_priority.pt when prioritized).

        Only the filled portion of the arrays is written, so early-training
        checkpoints are compact even when max_size is 1 M.
        """
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(
            path,
            state=self.state[: self.size],
            action=self.action[: self.size],
            next_state=self.next_state[: self.size],
            reward=self.reward[: self.size],
            not_done=self.not_done[: self.size],
            meta=np.array([self.ptr, self.size, self.max_size]),
            max_priority=np.array([self.max_priority if self.prioritized else 1.0]),
        )
        if self.prioritized:
            torch.save(
                self.priority[: self.size].cpu(), path + "_priority.pt"
            )

    def load(self, path: str) -> bool:
        """Restore buffer from <path>.npz.  Returns True on success."""
        npz = path if path.endswith(".npz") else path + ".npz"
        if not os.path.isfile(npz):
            return False
        d    = np.load(npz)
        meta = d["meta"].tolist()
        ptr, size = int(meta[0]), int(meta[1])
        self.state[: size]      = d["state"]
        self.action[: size]     = d["action"]
        self.next_state[: size] = d["next_state"]
        self.reward[: size]     = d["reward"]
        self.not_done[: size]   = d["not_done"]
        self.ptr  = ptr
        self.size = size
        if self.prioritized:
            self.max_priority = float(d["max_priority"][0])
            ppt = path + "_priority.pt"
            if os.path.isfile(ppt):
                self.priority[: size] = torch.load(
                    ppt, map_location=self.device
                ).to(self.device)
            else:
                self.priority[: size] = self.max_priority
        return True

    def load_D4RL(self, dataset):
        self.state = dataset["observations"]
        self.action = dataset["actions"]
        self.next_state = dataset["next_observations"]
        self.reward = dataset["rewards"].reshape(-1, 1)
        self.not_done = 1.0 - dataset["terminals"].reshape(-1, 1)
        self.size = self.state.shape[0]

        if self.prioritized:
            self.priority = torch.ones(self.size).to(self.device)
