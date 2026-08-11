import os

import numpy as np
import torch


# RISK_BALANCE: fixed per-transition metadata layout (see LAP.risk_meta).
# Columns: [curriculum_stage, human_event, risk_positive, collision_or_near].
RISK_META_DIM = 4
RISK_META_COLUMNS = ("stage", "human_event", "risk_positive", "collision_or_near")


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
        action_risk_dim=0,
        counterfactual_risk_dim=0,
        executed_action_risk_dim=0,
        store_risk_meta=False,
        risk_balanced_enabled=False,
        risk_balanced_ratios=(0.5, 0.25, 0.25),
    ):

        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size

        # STAGE 7: explicit float32 (was no dtype specified -> numpy's
        # implicit float64 default). Real measured impact (Stage 0 baseline):
        # the actual phase2/both replay checkpoint (180k/1M transitions,
        # state_dim=327) was 1032.5MB at float64; extrapolated to the
        # configured max_size=1e6 (eagerly allocated here regardless of fill
        # level) that's ~5.7GB float64 -> ~2.87GB at float32. sample()
        # already requested dtype=torch.float (float32) on every call
        # regardless of the source array's dtype, so this changes no
        # downstream numerics -- only removes a redundant float64->float32
        # cast that used to happen on every sample() call.
        _dt = np.float32
        self.state = np.zeros((max_size, state_dim), dtype=_dt)
        self.action = np.zeros((max_size, action_dim), dtype=_dt)
        self.next_state = np.zeros((max_size, state_dim), dtype=_dt)
        self.reward = np.zeros((max_size, 1), dtype=_dt)
        self.not_done = np.zeros((max_size, 1), dtype=_dt)

        # AUX_PRED: optional per-transition auxiliary target (future risk map +
        # min-distance label).  aux_dim == 0 keeps the buffer identical to the
        # baseline so non-aux algorithms / configs are unaffected.
        self.aux_dim = int(aux_dim)
        if self.aux_dim > 0:
            self.aux_target = np.zeros((max_size, self.aux_dim), dtype=_dt)
        else:
            self.aux_target = None

        # AUX_PRED: optional episode-boundary flag per transition.  Needed only by
        # the action-conditioned auxiliary, so future-action lookups never cross
        # an episode boundary.  None keeps the buffer identical to baseline.
        self.track_traj = bool(track_traj)
        if self.track_traj:
            self.traj_end = np.zeros((max_size, 1), dtype=_dt)
        else:
            self.traj_end = None

        # PHASE2: optional per-transition Action-Risk Head supervision target
        # (risk_dir, min_dist_dir), aligned with `state`+`action` exactly like
        # aux_target. action_risk_dim == 0 keeps the buffer identical to
        # baseline for every config that doesn't opt in.
        self.action_risk_dim = int(action_risk_dim)
        if self.action_risk_dim > 0:
            self.action_risk_target = np.zeros((max_size, self.action_risk_dim), dtype=_dt)
        else:
            self.action_risk_target = None

        # Optional fixed candidate x horizon counterfactual-risk labels.
        self.counterfactual_risk_dim = int(counterfactual_risk_dim)
        if self.counterfactual_risk_dim > 0:
            self.counterfactual_risk_target = np.zeros(
                (max_size, self.counterfactual_risk_dim), dtype=_dt)
        else:
            self.counterfactual_risk_target = None

        # Optional per-transition multi-horizon target for the action ACTUALLY
        # executed this step (as opposed to the fixed candidates above),
        # aligned with `state`+`action` exactly like action_risk_target.
        # executed_action_risk_dim == 0 keeps the buffer identical to
        # baseline; it is only > 0 alongside counterfactual_risk_dim > 0
        # (the two are enabled by the same config toggle).
        self.executed_action_risk_dim = int(executed_action_risk_dim)
        if self.executed_action_risk_dim > 0:
            self.executed_action_risk_target = np.zeros(
                (max_size, self.executed_action_risk_dim), dtype=_dt)
        else:
            self.executed_action_risk_target = None

        # RISK_BALANCE: optional fixed-width per-transition metadata (stage /
        # human-event / risk-positive / collision-or-near-collision flags), used
        # ONLY by risk-balanced sampling (get_last_aux / get_last_action_risk /
        # the core sample() batch are completely unaffected). store_risk_meta
        # =False (default) keeps the buffer identical to baseline -- no new
        # array, no new npz key.
        self.store_risk_meta = bool(store_risk_meta)
        if self.store_risk_meta:
            self.risk_meta = np.zeros((max_size, RISK_META_DIM), dtype=_dt)
        else:
            self.risk_meta = None

        # RISK_BALANCE: stratified sampling for the risk-balanced AUX/action-risk
        # supervision batch (sample_risk_balanced()). Default OFF -> that method
        # always returns None and every caller falls back to the primary
        # uniform/prioritized batch from sample(), byte-identical to baseline.
        # Independent of store_risk_meta (balanced sampling REQUIRES metadata,
        # enforced below) so a caller can store metadata for later analysis
        # without turning sampling bias on.
        self.risk_balanced_enabled = bool(risk_balanced_enabled) and self.store_risk_meta
        r = [max(0.0, float(x)) for x in risk_balanced_ratios]
        rsum = sum(r) or 1.0
        self.risk_balanced_ratios = tuple(x / rsum for x in r)  # (uniform, human_risk, collision)
        self.ind_balanced = None

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, device=device)
            self.max_priority = 1

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, next_state, reward, done, aux_target=None,
            traj_end=0.0, action_risk_target=None,
            counterfactual_risk_target=None, executed_action_risk_target=None,
            risk_meta=None):
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

        # PHASE2: store the Action-Risk Head target aligned with `state`+`action`
        # (the transition that just happened). Missing / wrong-length targets
        # zero-pad, same graceful contract as aux_target.
        if self.action_risk_target is not None:
            row = np.zeros(self.action_risk_dim, dtype=self.action_risk_target.dtype)
            if action_risk_target is not None:
                a = np.asarray(action_risk_target, dtype=self.action_risk_target.dtype).reshape(-1)
                n = min(self.action_risk_dim, a.shape[0])
                row[:n] = a[:n]
            self.action_risk_target[self.ptr] = row

        if self.counterfactual_risk_target is not None:
            row = np.zeros(
                self.counterfactual_risk_dim,
                dtype=self.counterfactual_risk_target.dtype)
            if counterfactual_risk_target is not None:
                values = np.asarray(
                    counterfactual_risk_target,
                    dtype=self.counterfactual_risk_target.dtype).reshape(-1)
                n = min(self.counterfactual_risk_dim, values.shape[0])
                row[:n] = values[:n]
            self.counterfactual_risk_target[self.ptr] = row

        if self.executed_action_risk_target is not None:
            row = np.zeros(
                self.executed_action_risk_dim,
                dtype=self.executed_action_risk_target.dtype)
            if executed_action_risk_target is not None:
                values = np.asarray(
                    executed_action_risk_target,
                    dtype=self.executed_action_risk_target.dtype).reshape(-1)
                n = min(self.executed_action_risk_dim, values.shape[0])
                row[:n] = values[:n]
            self.executed_action_risk_target[self.ptr] = row

        # RISK_BALANCE: store the per-transition metadata aligned with `state`
        # (same zero-pad-on-missing contract as aux_target/action_risk_target).
        if self.risk_meta is not None:
            row = np.zeros(RISK_META_DIM, dtype=self.risk_meta.dtype)
            if risk_meta is not None:
                m = np.asarray(risk_meta, dtype=self.risk_meta.dtype).reshape(-1)
                n = min(RISK_META_DIM, m.shape[0])
                row[:n] = m[:n]
            self.risk_meta[self.ptr] = row

        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def reset(self):
        """Empty the buffer in place (allocated arrays are kept).

        Used at a curriculum stage boundary where the action/control contract
        changes, so off-contract transitions don't leak into later training.
        Future add() calls overwrite the arrays; only the pointers, count and
        priorities are cleared."""
        self.ptr = 0
        self.size = 0
        self.ind = None
        self.ind_balanced = None
        if self.prioritized:
            self.priority.zero_()
            self.max_priority = 1

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
        return self.get_batch_by_indices(self.ind)

    def get_batch_by_indices(self, indices):
        """Gather the (state, action, next_state, reward, not_done) 5-tuple for
        an explicit index array -- the same tensor-conversion logic sample()
        uses, factored out so RISK_BALANCE's sample_risk_balanced() batch can
        reuse it without duplicating sample()'s body or touching self.ind
        (which update_priority() and the get_last_* accessors rely on staying
        exactly what the last core sample() call drew)."""
        # STAGE 7: self.state[indices] (fancy/advanced indexing) already
        # allocates a fresh, C-contiguous NumPy copy -- unavoidable for random-
        # index sampling. torch.from_numpy() SHARES that copy's memory instead
        # of torch.tensor()'s own additional copy on top of it; now that the
        # source arrays are float32 (see __init__), this needs no dtype
        # conversion either (from_numpy preserves whatever dtype the array
        # already has). .to(self.device) is a no-op view on CPU or the one
        # unavoidable host->device copy on CUDA, same as before.
        return (
            torch.from_numpy(self.state[indices]).to(self.device),
            torch.from_numpy(self.action[indices]).to(self.device),
            torch.from_numpy(self.next_state[indices]).to(self.device),
            torch.from_numpy(self.reward[indices]).to(self.device),
            torch.from_numpy(self.not_done[indices]).to(self.device),
        )

    def sample_risk_balanced(self, batch_size=None):
        """RISK_BALANCE: draw a SEPARATE batch of indices stratified over three
        pools -- uniform / human-risk-positive / collision-or-near-collision --
        for aux / action-risk supervision ONLY (the core critic batch from
        sample() is never touched by this method).

        Returns an int64 index array of length ``batch_size`` (default
        self.batch_size), or None when disabled / metadata isn't stored / the
        buffer is still empty (caller should fall back to the primary sample()
        batch's own aux/action-risk loss for that step, i.e. behave exactly
        like the feature being off).

        Ratios come from risk_balanced_ratios (uniform, human_risk, collision),
        normalised to sum to 1 at construction time. If the human-risk or
        collision pool is empty (or the whole buffer has no metadata yet, e.g.
        early warmup), that pool's share is silently redistributed to the
        uniform pool instead of erroring or shrinking the batch -- this can
        never produce a duplicate-sampling error or an empty batch as long as
        self.size > 0.
        """
        if not self.risk_balanced_enabled or self.risk_meta is None or self.size == 0:
            return None
        bs = int(batch_size) if batch_size else self.batch_size
        meta = self.risk_meta[: self.size]
        human_pool = np.nonzero(
            (meta[:, 1] > 0.5) | (meta[:, 2] > 0.5)
        )[0]
        collision_pool = np.nonzero(meta[:, 3] > 0.5)[0]

        _, ratio_human, ratio_collision = self.risk_balanced_ratios
        n_col = int(round(bs * ratio_collision))
        n_hum = int(round(bs * ratio_human))

        def _draw(pool, n):
            if n <= 0:
                return np.empty(0, dtype=np.int64)
            if pool.size == 0:
                return None  # signals "redistribute to uniform"
            replace = pool.size < n
            return np.random.choice(pool, size=n, replace=replace).astype(np.int64)

        # An empty pool draws None; its share is simply left out of col_idx/
        # hum_idx, so the uniform draw below (bs - len(col_idx) - len(hum_idx))
        # automatically absorbs the shortfall -- no explicit redistribution
        # bookkeeping needed.
        col_idx = _draw(collision_pool, n_col)
        col_idx = np.empty(0, dtype=np.int64) if col_idx is None else col_idx
        hum_idx = _draw(human_pool, n_hum)
        hum_idx = np.empty(0, dtype=np.int64) if hum_idx is None else hum_idx
        n_uni = bs - col_idx.shape[0] - hum_idx.shape[0]
        uni_idx = np.random.randint(0, self.size, size=max(0, n_uni)).astype(np.int64)

        ind = np.concatenate([uni_idx, hum_idx, col_idx])
        np.random.shuffle(ind)
        self.ind_balanced = ind
        return ind

    @staticmethod
    def _risk_meta_fraction_dict(m):
        """RISK_BALANCE: shared fraction computation for a risk_meta slice
        `m` (rows x RISK_META_DIM), used by both describe_risk_meta_fractions
        and its _raw whole-buffer variant.

        collision_{human_event,risk_positive}_frac: of the rows tagged
        collision_or_near, what fraction are ALSO human_event / risk_positive
        -- collision_or_near itself is sourced from the env's full-360
        collision/near-collision verdict, which fires on STATIC obstacles
        too, not just humans. If a run's collision pool turns out to be
        mostly static-obstacle collisions, the "collision" pool feeding the
        (human-risk-focused) aux/action-risk supervised loss is diluted with
        safe-for-humans rows -- these two fractions make that observable
        without changing the pool definition itself. Omitted (not just 0.0)
        when the collision pool is empty, so a 0/0 doesn't silently read as
        "no overlap"."""
        out = {
            "human_event_frac": float((m[:, 1] > 0.5).mean()),
            "risk_positive_frac": float((m[:, 2] > 0.5).mean()),
            "collision_frac": float((m[:, 3] > 0.5).mean()),
        }
        collision_mask = m[:, 3] > 0.5
        n_collision = int(collision_mask.sum())
        if n_collision > 0:
            out["collision_human_event_frac"] = float(
                (m[collision_mask, 1] > 0.5).mean())
            out["collision_risk_positive_frac"] = float(
                (m[collision_mask, 2] > 0.5).mean())
        return out

    def describe_risk_meta_fractions(self, indices):
        """RISK_BALANCE: fraction of `indices` carrying each metadata flag --
        logging-only helper (sampled risk-positive / human-event / collision
        fraction), never used in the loss/sampling path itself."""
        if self.risk_meta is None or indices is None or len(indices) == 0:
            return {}
        m = self.risk_meta[np.asarray(indices)]
        return self._risk_meta_fraction_dict(m)

    def describe_risk_meta_fractions_raw(self):
        """RISK_BALANCE: same fractions as describe_risk_meta_fractions, but
        over the ENTIRE current buffer contents (self.risk_meta[:self.size])
        instead of one sampled batch's indices -- the raw replay distribution
        that risk-balanced sampling is correcting for. Logging-only; never
        used in the loss/sampling path."""
        if self.risk_meta is None or self.size == 0:
            return {}
        m = self.risk_meta[: self.size]
        return self._risk_meta_fraction_dict(m)

    def get_last_aux(self, indices=None):
        """AUX_PRED: auxiliary targets for the given indices (default: the
        indices from the last sample()).

        Returns a (batch, aux_dim) float tensor, or None when auxiliary targets
        are disabled.  sample() leaves self.ind set, so the default mirrors
        that batch without changing sample()'s return signature (baseline
        compatibility). RISK_BALANCE: pass ``indices=ind_balanced`` explicitly
        to fetch the aux target for the risk-balanced batch instead.
        """
        if self.aux_target is None:
            return None
        ind = self.ind if indices is None else indices
        if ind is None:
            return None
        return torch.from_numpy(self.aux_target[ind]).to(self.device)

    def get_last_action_risk(self, indices=None):
        """PHASE2: Action-Risk Head supervision targets for the given indices
        (default: the indices from the last sample()). Returns a (batch,
        action_risk_dim) float tensor, or None when the feature is disabled.
        Mirrors get_last_aux()."""
        if self.action_risk_target is None:
            return None
        ind = self.ind if indices is None else indices
        if ind is None:
            return None
        return torch.from_numpy(self.action_risk_target[ind]).to(self.device)

    def get_last_counterfactual_risk(self, indices=None):
        if self.counterfactual_risk_target is None:
            return None
        ind = self.ind if indices is None else indices
        if ind is None:
            return None
        return torch.from_numpy(
            self.counterfactual_risk_target[ind]).to(self.device)

    def get_last_executed_action_risk(self, indices=None):
        """Executed-action multi-horizon supervision targets for the given
        indices (default: the indices from the last sample()). Returns a
        (batch, executed_action_risk_dim) float tensor, or None when the
        feature is disabled. Mirrors get_last_counterfactual_risk()."""
        if self.executed_action_risk_target is None:
            return None
        ind = self.ind if indices is None else indices
        if ind is None:
            return None
        return torch.from_numpy(
            self.executed_action_risk_target[ind]).to(self.device)

    def get_last_future_actions(self, k_steps, indices=None):
        """AUX_PRED: future action sequence for the given indices (default: the
        last sample()'s indices).

        For each sampled index i returns the actions [a_i, a_{i+1}, ..., a_{i+K-1}]
        (a_j is the action stored in transition j), stopping at an episode
        boundary or the buffer write head.  Steps past the stop are zero-padded.

        Returns (future_actions, valid_len) or None when boundary tracking is off:
          future_actions : (B, K, action_dim) float tensor
          valid_len      : (B,) long tensor in [1, K]  (number of leading
                           in-episode actions; index 0 is always valid)

        Circular-buffer safe: a step is valid only if the previous index is not
        a trajectory end AND the next index is not the write head ``ptr`` (which
        is either unwritten or the oldest, stale transition). RISK_BALANCE: pass
        ``indices=ind_balanced`` to align this lookup with the risk-balanced
        batch instead -- the same boundary rule applies unchanged.
        """
        if self.traj_end is None or k_steps < 1:
            return None
        ind = self.ind if indices is None else indices
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

    def get_last_state_history(self, n_steps, indices=None):
        """AUX_PRED: REVERSE-time state window for the given indices (default:
        the last sample()'s indices).

        For each sampled index i returns the states [s_i, s_{i-1}, ..., s_{i-N+1}]
        (index 0 == the current state, always valid), walking BACKWARD and
        stopping at an episode boundary or the buffer seam.  Steps past the stop
        are zero-padded.  Used ONLY by the aux-branch temporal context — the
        actor/critic never see it.

        Returns (history_states, valid_len) or None when boundary tracking is off:
          history_states : (B, N, state_dim) float tensor
          valid_len      : (B,) long tensor in [1, N]  (number of leading valid
                           in-episode states; index 0 is always valid)

        Boundary rule (symmetric to get_last_future_actions): stepping from the
        newer neighbour ``prev=(i-k+1)`` to the older slot ``cur=(i-k)`` is valid
        only while we have not already reached the oldest written slot
        (``prev == oldest`` stops, where oldest == ``ptr`` for a full buffer else
        0 — this is the circular seam) AND transition ``cur`` is not an episode
        end (``traj_end[cur]==0``; a 1 there means s_prev is a fresh reset state,
        so s_cur is the PREVIOUS episode and must not be spliced in). RISK_BALANCE:
        pass ``indices=ind_balanced`` to align this lookup with the risk-balanced
        batch instead -- the same boundary rule applies unchanged.
        """
        if self.traj_end is None or n_steps < 1:
            return None
        ind = self.ind if indices is None else indices
        if ind is None:
            return None
        ind = np.asarray(ind)
        B = ind.shape[0]
        S = self.state.shape[1]
        M = self.max_size
        oldest = self.ptr if self.size == self.max_size else 0

        hist = np.zeros((B, n_steps, S), dtype=np.float32)
        vlen = np.ones(B, dtype=np.int64)
        valid = np.ones(B, dtype=bool)
        hist[:, 0, :] = self.state[ind]   # k = 0 is the current state, always valid
        for k in range(1, n_steps):
            prev = (ind - (k - 1)) % M
            cur = (ind - k) % M
            step_ok = valid & (prev != oldest) & (self.traj_end[cur, 0] == 0.0)
            valid = valid & step_ok
            if valid.any():
                hist[valid, k, :] = self.state[cur[valid]]
                vlen[valid] += 1
        return (
            torch.tensor(hist, dtype=torch.float, device=self.device),
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
        # PHASE2: persist the Action-Risk Head target (optional key).
        if self.action_risk_target is not None:
            save_kwargs["action_risk_target"] = self.action_risk_target[: self.size]
        if self.counterfactual_risk_target is not None:
            save_kwargs["counterfactual_risk_target"] = (
                self.counterfactual_risk_target[: self.size])
        if self.executed_action_risk_target is not None:
            save_kwargs["executed_action_risk_target"] = (
                self.executed_action_risk_target[: self.size])
        # RISK_BALANCE: persist per-transition metadata (optional key).
        if self.risk_meta is not None:
            save_kwargs["risk_meta"] = self.risk_meta[: self.size]
        np.savez_compressed(path, **save_kwargs)
        if self.prioritized:
            torch.save(
                self.priority[: self.size].cpu(), path + "_priority.pt"
            )

    def _cast_loaded(self, name: str, arr):
        """STAGE 7: explicit float64 (or any other dtype) -> float32 cast for
        a loaded checkpoint array, logging the FIRST time it actually changes
        something for this load() call (a checkpoint saved since Stage 7 is
        already float32 -> true no-op, no log). Never mutates ``arr`` (the
        caller's view into the just-``np.load``ed .npz) in place."""
        arr = np.asarray(arr)
        if arr.dtype != self.state.dtype:
            print(f"[buffer] load(): casting '{name}' from {arr.dtype} to "
                  f"{self.state.dtype} (checkpoint predates the float32 replay "
                  f"buffer change).")
            arr = arr.astype(self.state.dtype)
        return arr

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
        # State-width guard: a checkpoint saved with a different state_dim (e.g.
        # observation time-context / frame stacking was toggled) cannot be loaded
        # into this buffer. Fail-fast with RuntimeError so the resume orchestrator
        # (tqc_io.load) degrades to a FRESH buffer instead of a numpy broadcast
        # error mid-load. aux_target is checked the same way below.
        saved_state_dim = int(d["state"].shape[1]) if d["state"].ndim == 2 else -1
        if saved_state_dim != self.state.shape[1]:
            raise RuntimeError(
                f"[buffer] replay-buffer checkpoint state_dim={saved_state_dim} != "
                f"current state_dim={self.state.shape[1]} (observation time-context "
                "/ frame stacking likely toggled). Resume with load_replay_buffer="
                "False or keep the stacking config unchanged."
            )
        if self.aux_target is not None and "aux_target" in d.files:
            saved_aux_dim = int(d["aux_target"].shape[1]) if d["aux_target"].ndim == 2 else -1
            if saved_aux_dim != self.aux_target.shape[1]:
                raise RuntimeError(
                    f"[buffer] replay-buffer checkpoint aux_dim={saved_aux_dim} != "
                    f"current aux_dim={self.aux_target.shape[1]} (aux label layout "
                    "changed, e.g. TTC / hazard heads toggled). Resume with "
                    "load_replay_buffer=False."
                )
        # PHASE2: same fail-fast contract for the Action-Risk Head target width.
        if self.action_risk_target is not None and "action_risk_target" in d.files:
            saved_ar_dim = (int(d["action_risk_target"].shape[1])
                             if d["action_risk_target"].ndim == 2 else -1)
            if saved_ar_dim != self.action_risk_target.shape[1]:
                raise RuntimeError(
                    f"[buffer] replay-buffer checkpoint action_risk_dim={saved_ar_dim} "
                    f"!= current action_risk_dim={self.action_risk_target.shape[1]} "
                    "(action_risk_head toggled/changed). Resume with "
                    "load_replay_buffer=False."
                )
        if self.counterfactual_risk_target is not None:
            if "counterfactual_risk_target" not in d.files:
                raise RuntimeError(
                    "[buffer] replay checkpoint has no counterfactual-risk "
                    "targets. Resume with load_replay_buffer=False.")
            saved_cf_dim = int(d["counterfactual_risk_target"].shape[1])
            if saved_cf_dim != self.counterfactual_risk_dim:
                raise RuntimeError(
                    f"[buffer] replay counterfactual_risk_dim={saved_cf_dim} "
                    f"!= current dim={self.counterfactual_risk_dim}. Resume "
                    "with load_replay_buffer=False.")
        if self.executed_action_risk_target is not None:
            if "executed_action_risk_target" not in d.files:
                raise RuntimeError(
                    "[buffer] replay checkpoint has no executed-action-risk "
                    "targets (predates the executed-action multi-horizon "
                    "target). Resume with load_replay_buffer=False.")
            saved_ex_dim = int(d["executed_action_risk_target"].shape[1])
            if saved_ex_dim != self.executed_action_risk_dim:
                raise RuntimeError(
                    f"[buffer] replay executed_action_risk_dim={saved_ex_dim} "
                    f"!= current dim={self.executed_action_risk_dim}. Resume "
                    "with load_replay_buffer=False.")
        meta = d["meta"].tolist()
        ptr, size = int(meta[0]), int(meta[1])
        # STAGE 7: explicit, logged float64 (or any other dtype) -> float32
        # cast on load -- runs AFTER the shape/schema validation above (so a
        # width mismatch still fails fast before any conversion), not relying
        # on numpy's implicit downcast-on-assignment alone. A checkpoint saved
        # since this change is already float32 (save() writes whatever dtype
        # the live arrays have), so this is a true no-op cast for those and
        # only actually converts pre-Stage-7 checkpoints.
        self.state[: size]      = self._cast_loaded("state", d["state"])
        self.action[: size]     = self._cast_loaded("action", d["action"])
        self.next_state[: size] = self._cast_loaded("next_state", d["next_state"])
        self.reward[: size]     = self._cast_loaded("reward", d["reward"])
        self.not_done[: size]   = self._cast_loaded("not_done", d["not_done"])
        # AUX_PRED: restore auxiliary targets when both this buffer and the
        # checkpoint carry them; otherwise leave zeros (graceful fallback).
        if self.aux_target is not None and "aux_target" in d.files:
            saved = self._cast_loaded("aux_target", d["aux_target"])
            cols = min(self.aux_dim, saved.shape[1])
            self.aux_target[: size, :cols] = saved[: size, :cols]
        # AUX_PRED: restore episode-boundary flags when both sides carry them.
        if self.traj_end is not None and "traj_end" in d.files:
            self.traj_end[: size] = self._cast_loaded("traj_end", d["traj_end"])[: size]
        # PHASE2: restore the Action-Risk Head target when both sides carry it.
        if self.action_risk_target is not None and "action_risk_target" in d.files:
            saved_ar = self._cast_loaded("action_risk_target", d["action_risk_target"])
            cols = min(self.action_risk_dim, saved_ar.shape[1])
            self.action_risk_target[: size, :cols] = saved_ar[: size, :cols]
        if (self.counterfactual_risk_target is not None
                and "counterfactual_risk_target" in d.files):
            saved_cf = self._cast_loaded(
                "counterfactual_risk_target", d["counterfactual_risk_target"])
            self.counterfactual_risk_target[:size] = saved_cf[:size]
        if (self.executed_action_risk_target is not None
                and "executed_action_risk_target" in d.files):
            saved_ex = self._cast_loaded(
                "executed_action_risk_target", d["executed_action_risk_target"])
            self.executed_action_risk_target[:size] = saved_ex[:size]
        # RISK_BALANCE: restore per-transition metadata when both sides carry
        # it. DELIBERATELY graceful (unlike traj_end/aux_dim/action_risk_dim
        # above, which fail-fast on a mismatch): an old checkpoint saved before
        # this feature existed has no 'risk_meta' key, and the safe response is
        # NOT to crash the resume or guess values -- restored rows simply stay
        # all-zero, which sample_risk_balanced() already treats as "no event"
        # (both the human-risk and collision pools are empty for those rows),
        # so it transparently redistributes their share to the uniform pool
        # until enough freshly-labelled transitions accumulate. A dim mismatch
        # (shouldn't happen -- RISK_META_DIM is fixed -- but defensive) is
        # handled the same way: restore whatever columns overlap, zero-pad rest.
        if self.risk_meta is not None:
            if "risk_meta" in d.files:
                saved_rm = self._cast_loaded("risk_meta", d["risk_meta"])
                cols = min(RISK_META_DIM, saved_rm.shape[1] if saved_rm.ndim == 2 else 0)
                if cols > 0:
                    self.risk_meta[: size, :cols] = saved_rm[: size, :cols]
            else:
                print(
                    "[buffer] load(): checkpoint has no 'risk_meta' (saved before "
                    "risk-balanced replay) -- restored transitions default to "
                    "all-zero metadata; risk-balanced sampling falls back to the "
                    "uniform pool until fresh labelled transitions accumulate."
                )
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
