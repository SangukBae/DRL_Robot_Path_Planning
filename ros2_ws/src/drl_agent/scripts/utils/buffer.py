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
        aux_dim=0,
        track_traj=False,
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

        # AUX_PRED: optional per-transition auxiliary target (future risk map +
        # min-distance label).  aux_dim == 0 keeps the buffer identical to the
        # baseline so non-aux algorithms / configs are unaffected.
        self.aux_dim = int(aux_dim)
        if self.aux_dim > 0:
            self.aux_target = np.zeros((max_size, self.aux_dim))
        else:
            self.aux_target = None

        # AUX_PRED: optional episode-boundary flag per transition.  Needed only by
        # the action-conditioned auxiliary, so future-action lookups never cross
        # an episode boundary.  None keeps the buffer identical to baseline.
        self.track_traj = bool(track_traj)
        if self.track_traj:
            self.traj_end = np.zeros((max_size, 1))
        else:
            self.traj_end = None

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, device=device)
            self.max_priority = 1

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, next_state, reward, done, aux_target=None,
            traj_end=0.0):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        # AUX_PRED: store the auxiliary target aligned with `state` (the encoder
        # input).  Missing / wrong-length labels are zero-padded so a momentary
        # absence of privileged info never corrupts the buffer.
        if self.aux_target is not None:
            row = np.zeros(self.aux_dim, dtype=self.aux_target.dtype)
            if aux_target is not None:
                a = np.asarray(aux_target, dtype=self.aux_target.dtype).reshape(-1)
                n = min(self.aux_dim, a.shape[0])
                row[:n] = a[:n]
            self.aux_target[self.ptr] = row

        # AUX_PRED: reset this slot's boundary flag (clears a stale boundary left
        # by a previous wrap); mark_last_traj_end() sets it after an episode ends.
        if self.traj_end is not None:
            self.traj_end[self.ptr] = float(traj_end)

        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def mark_last_traj_end(self):
        """AUX_PRED: flag the most-recently-added transition as an episode end.

        Called by the trainer when an episode terminates for ANY reason
        (goal / collision / timeout / eval-cut), so the action-conditioned
        future-action lookup stops at the true boundary.  No-op when boundary
        tracking is disabled or the buffer is empty.
        """
        if self.traj_end is None or self.size == 0:
            return
        last = (self.ptr - 1) % self.max_size
        self.traj_end[last] = 1.0

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

    def get_last_aux(self):
        """AUX_PRED: auxiliary targets for the indices from the last sample().

        Returns a (batch, aux_dim) float tensor, or None when auxiliary targets
        are disabled.  sample() leaves self.ind set, so this mirrors that batch
        without changing sample()'s return signature (baseline compatibility).
        """
        if self.aux_target is None:
            return None
        ind = getattr(self, "ind", None)
        if ind is None:
            return None
        return torch.tensor(
            self.aux_target[ind], dtype=torch.float, device=self.device
        )

    def get_last_future_actions(self, k_steps):
        """AUX_PRED: future action sequence for the last sample()'s indices.

        For each sampled index i returns the actions [a_i, a_{i+1}, ..., a_{i+K-1}]
        (a_j is the action stored in transition j), stopping at an episode
        boundary or the buffer write head.  Steps past the stop are zero-padded.

        Returns (future_actions, valid_len) or None when boundary tracking is off:
          future_actions : (B, K, action_dim) float tensor
          valid_len      : (B,) long tensor in [1, K]  (number of leading
                           in-episode actions; index 0 is always valid)

        Circular-buffer safe: a step is valid only if the previous index is not
        a trajectory end AND the next index is not the write head ``ptr`` (which
        is either unwritten or the oldest, stale transition).
        """
        if self.traj_end is None or k_steps < 1:
            return None
        ind = getattr(self, "ind", None)
        if ind is None:
            return None
        ind = np.asarray(ind)
        B = ind.shape[0]
        A = self.action.shape[1]
        M = self.max_size

        fut = np.zeros((B, k_steps, A), dtype=np.float32)
        vlen = np.ones(B, dtype=np.int64)
        valid = np.ones(B, dtype=bool)
        fut[:, 0, :] = self.action[ind]   # k = 0 is always in-episode
        for k in range(1, k_steps):
            prev = (ind + k - 1) % M
            cur = (ind + k) % M
            step_ok = valid & (self.traj_end[prev, 0] == 0.0) & (cur != self.ptr)
            valid = valid & step_ok
            if valid.any():
                fut[valid, k, :] = self.action[cur[valid]]
                vlen[valid] += 1
        return (
            torch.tensor(fut, dtype=torch.float, device=self.device),
            torch.tensor(vlen, dtype=torch.long, device=self.device),
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
        save_kwargs = dict(
            state=self.state[: self.size],
            action=self.action[: self.size],
            next_state=self.next_state[: self.size],
            reward=self.reward[: self.size],
            not_done=self.not_done[: self.size],
            meta=np.array([self.ptr, self.size, self.max_size]),
            max_priority=np.array([self.max_priority if self.prioritized else 1.0]),
        )
        # AUX_PRED: persist auxiliary targets / boundary flags (optional keys).
        if self.aux_target is not None:
            save_kwargs["aux_target"] = self.aux_target[: self.size]
        if self.traj_end is not None:
            save_kwargs["traj_end"] = self.traj_end[: self.size]
        np.savez_compressed(path, **save_kwargs)
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
        # AUX_PRED: action-conditioned aux needs per-transition episode boundaries.
        # Refuse (fail-fast) an old checkpoint that lacks them, instead of silently
        # leaving every boundary at 0 and corrupting the future-action supervision
        # (get_last_future_actions would splice across past episode boundaries).
        if self.track_traj and "traj_end" not in d.files:
            raise RuntimeError(
                "[buffer] replay-buffer checkpoint has no 'traj_end' (it was saved "
                "before action_conditioned_aux), but this run needs episode "
                "boundaries for the action-conditioned auxiliary. Resume with "
                "load_replay_buffer=False (fresh buffer) or disable "
                "action_conditioned_aux. Loading it as-is would silently corrupt "
                "the auxiliary supervision."
            )
        meta = d["meta"].tolist()
        ptr, size = int(meta[0]), int(meta[1])
        self.state[: size]      = d["state"]
        self.action[: size]     = d["action"]
        self.next_state[: size] = d["next_state"]
        self.reward[: size]     = d["reward"]
        self.not_done[: size]   = d["not_done"]
        # AUX_PRED: restore auxiliary targets when both this buffer and the
        # checkpoint carry them; otherwise leave zeros (graceful fallback).
        if self.aux_target is not None and "aux_target" in d.files:
            saved = d["aux_target"]
            cols = min(self.aux_dim, saved.shape[1])
            self.aux_target[: size, :cols] = saved[: size, :cols]
        # AUX_PRED: restore episode-boundary flags when both sides carry them.
        if self.traj_end is not None and "traj_end" in d.files:
            self.traj_end[: size] = d["traj_end"][: size]
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
