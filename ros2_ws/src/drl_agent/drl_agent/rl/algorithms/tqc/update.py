#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TQC actor/critic update step, auxiliary/action-risk/counterfactual (CF)
loss inclusion, and the CF actor-penalty warm-up schedule.

Extracted unchanged from rl/algorithms/tqc/agent.py: train() computes the
critic loss, folds the auxiliary-prediction / action-risk-head /
counterfactual-risk supervised losses into ONE trunk loss (single combined
backward() + optimizer.step() — the aux/action-risk/CF losses are NOT split
into their own optimizer steps, so their computation is deliberately kept
inline in train() here rather than fragmented into separate call-and-return
functions across files, which would risk silently changing which
computational graph the final backward() actually covers), then runs the
actor update (with the critic and both risk heads frozen via
_frozen_params — see that function's docstring for why this is
requires_grad toggling and NOT .detach(), which would sever the actor's
intended gradient channel through the risk heads' predictions), and finally
the target-network Polyak update. Every formula, loss weight, optimizer
application order, the supervised-update counter's meaning, and the actor
penalty warm-up/ramp behaviour are byte-identical to the pre-split code —
this is pure code motion (see docs/design and the TQC agent's own
docstrings for the algorithm itself).

train_and_checkpoint / train_and_reset (LAP checkpoint-ensembling wrappers
around train()) and the small CF-penalty-schedule reset helpers
(reset_counterfactual_penalty_schedule / reset_replay_for_action_contract_
change, called externally on a curriculum promotion that resets the replay
buffer) live here too, alongside the two module-level helpers train() uses
(_frozen_params, _polyak_update_foreach).
"""

import contextlib

import torch

from drl_agent.rl.networks.aux_losses import compute_aux_loss
from drl_agent.rl.networks.action_risk_head import (
    weighted_action_risk_loss, weighted_counterfactual_risk_loss,
)
from drl_agent.rl.networks.tqc import quantile_huber_loss


@contextlib.contextmanager
def _frozen_params(*modules):
    """STAGE 5: temporarily set requires_grad_(False) on every parameter of the
    given module(s) for a single forward call, guaranteed to restore True even
    if the wrapped block raises (exception-safe, unlike a bare set-False/call/
    set-True sequence). Used so a "frozen submodule, live input" forward pass
    (e.g. the critic during the actor update) never picks up a gradient into
    its OWN parameters from that pass's backward(), while gradient still flows
    through to whatever tensor was fed as input (the action, in the critic's
    case) -- exactly mirroring why action_risk_head is frozen the same way for
    the actor-update's extra_pi computation."""
    params = [p for m in modules for p in m.parameters()]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in params:
            p.requires_grad_(True)


def _polyak_update_foreach(src_params, tgt_params, tau: float):
    """STAGE 5: Polyak (exponential moving average) target update, batched via
    torch._foreach_* instead of a Python for-loop over individual parameter
    tensors -- one CUDA kernel launch per op across ALL parameters instead of
    one per parameter, mathematically identical to the original per-parameter
    ``target.data.copy_(tau * src.data + (1 - tau) * target.data)`` loop (see
    test_foreach_polyak_matches_per_parameter_loop_numerically). Wrapped in
    torch.no_grad() for clarity/defensiveness -- the .data-based ops already
    bypass autograd either way, so this changes no numerics, only makes the
    "never build a graph here" intent explicit."""
    src_list = list(src_params)
    tgt_list = list(tgt_params)
    if not src_list:
        return
    with torch.no_grad():
        src_data = [p.data for p in src_list]
        tgt_data = [p.data for p in tgt_list]
        scaled_src = torch._foreach_mul(src_data, tau)
        torch._foreach_mul_(tgt_data, 1.0 - tau)
        torch._foreach_add_(tgt_data, scaled_src)


class UpdateMixin:
    """Actor/critic update step + CF penalty schedule.

    Mixed into Agent (rl/algorithms/tqc/agent.py); every method here reads/
    writes Agent instance state via ``self`` exactly as it did before
    extraction.
    """

    def reset_counterfactual_penalty_schedule(self):
        """Reset actor-penalty warm-up after an action-contract change.

        A curriculum promotion that also resets the replay buffer (e.g.
        stage 5 unsealing the yield channel -- see train_tqc_curriculum.py's
        reset_buffer_on_promote_to) throws away every transition the CF head
        was supervised on. Without this call, self._cf_supervised_updates
        keeps counting from before the reset, so a head that has ALREADY
        cleared warmup+ramp under the OLD contract would apply a full-weight
        actor penalty from its very first update on the NEW (off-contract-
        free) data -- exactly the "untrained head steers the actor" failure
        this schedule exists to avoid.

        Deliberately does NOT touch counterfactual_risk_head's parameters:
        the head's learned dynamic-risk representation (avoid nearby humans,
        walls, etc.) is still largely valid across the contract change -- only
        the ACTOR PENALTY needs to re-earn trust via a fresh warm-up/ramp
        against post-reset data, not the head's whole knowledge.

        No-op when the feature is disabled (nothing to reset)."""
        if not self.counterfactual_risk_enabled:
            return
        self._cf_supervised_updates = 0
        self._cf_effective_actor_penalty_weight = 0.0

    def reset_replay_for_action_contract_change(self):
        """Clear off-contract replay and re-arm the optional CF penalty."""
        self.replay_buffer.reset()
        self.reset_counterfactual_penalty_schedule()

    def _current_aux_beta(self):
        """AUX_PRED: effective trunk-level aux weight at this step.

        If a per-stage schedule is configured (aux_prediction.stagewise_loss_
        schedule) it takes precedence: beta = schedule[clamp(current_stage)], so
        aux supervision ramps in with the curriculum. Otherwise it linearly ramps
        0 -> loss_weight over aux_beta_warmup_steps (a noisy, freshly-initialised
        aux head does not perturb early critic learning), or the constant
        loss_weight when warmup is disabled (== 0).

        ``train()`` increments ``training_steps`` BEFORE the aux block runs, so the
        first update has ``training_steps == 1``; the ``-1`` makes that first
        update's beta exactly 0 (a true 0 -> loss_weight ramp) and reaches the
        full weight after ``w`` updates (at ``training_steps == w + 1``)."""
        sched = self.aux_cfg.stagewise_loss_schedule
        if sched:
            return float(sched[min(self.current_stage, len(sched) - 1)])
        w = self.aux_cfg.aux_beta_warmup_steps
        if w <= 0:
            return self.aux_beta
        return self.aux_beta * min(1.0, max(0, self.training_steps - 1) / float(w))

    def _compute_temporal_ctx(self, indices=None):
        """AUX_PRED (v2): temporal context for the given batch (default: the
        last sample()'s indices), or None.

        Walks the replay buffer backward (boundary-safe) for the last
        history_len in-episode states, encodes them through the SHARED encoder
        (so the temporal aux loss also shapes the encoder) and summarises the
        latent window with the temporal GRU. Returns (context, hist_valid_len) or
        None when temporal is off / history is unavailable. RISK_BALANCE: pass
        ``indices=ind_rb`` to align this with the risk-balanced batch instead."""
        if self.temporal_encoder is None:
            return None
        hist = self.replay_buffer.get_last_state_history(self.aux_cfg.history_len, indices=indices)
        if hist is None:
            return None
        hist_states, hist_valid = hist                 # (B, N, S), (B,)
        b, n, s = hist_states.shape
        z_seq = self.encoder(hist_states.reshape(b * n, s)).reshape(b, n, -1)
        return self.temporal_encoder(z_seq, hist_valid), hist_valid

    def train(self):
        """Train the agent for one step"""
        self.training_steps += 1

        # Sample batch from replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample()
        # STAGE 8 (isolated experimental, default OFF): fixed physical-range
        # normalization applied identically to state and next_state, right
        # before either reaches an encoder.
        if self.obs_normalizer is not None:
            state = self.obs_normalizer.normalize(state)
            next_state = self.obs_normalizer.normalize(next_state)
        # AUX_PRED: matching auxiliary targets for this batch (None if disabled).
        aux_target = self.replay_buffer.get_last_aux()

        # AUX_PRED: encode once.  `z` keeps the graph (critic + aux back-prop
        # into the encoder); `z_actor` is detached so the actor / temperature
        # updates never flow gradients into the encoder.
        z = self.encoder(state)
        z_actor = z.detach()

        # RISK_BALANCE: a SEPARATE, stratified batch (uniform / human-risk /
        # collision pools) used ONLY for the aux / action-risk SUPERVISED loss
        # below -- the critic loss above/below always uses the primary
        # uniform/prioritized `state`/`action` batch, untouched. None (feature
        # off, or the buffer is still empty) -> both use_rb_* flags below are
        # False and every block falls back to its ORIGINAL primary-batch
        # computation, byte-identical to before this feature existed.
        # STAGE 3/AUX_PRED: compute the effective aux beta BEFORE deciding
        # whether to draw the balanced batch below -- beta==0 (aux loss
        # disabled this stage / still in its warmup ramp) already skips the
        # aux loss entirely further down (see the beta != 0.0 check further
        # below), so drawing+encoding state_rb here would be pure waste: an
        # extra forward pass that no loss ever reads. _current_aux_beta() is
        # side-effect-free (reads only stage/step/config), so computing it
        # here changes no numerics.
        #
        # need_rb_for_{aux,risk} mirror the use_rb_* conditions below EXACTLY,
        # but computed BEFORE calling sample_risk_balanced() -- that call
        # itself rescans the ENTIRE risk_meta array to rebuild the human-risk
        # / collision pools every time it's invoked (see buffer.py's
        # sample_risk_balanced), so calling it on a stage/step where NEITHER
        # consumer would use the result (e.g. stage 0-2, before aux_beta or
        # action_risk_head.enable_from_stage kick in) wastes a full-buffer
        # scan every train() call for nothing.
        beta = self._current_aux_beta()
        need_rb_for_aux = (
            self.risk_balanced_enabled and self.aux_enabled and self.aux_head is not None
            and beta != 0.0
        )
        need_rb_for_risk = (
            self.risk_balanced_enabled and self.action_risk_enabled and self._action_risk_active
        )
        need_rb_for_cf = (
            self.risk_balanced_enabled and self.counterfactual_risk_enabled
            and self._counterfactual_risk_active
        )
        ind_rb = (
            self.replay_buffer.sample_risk_balanced()
            if (need_rb_for_aux or need_rb_for_risk or need_rb_for_cf) else None
        )
        use_rb_for_aux = ind_rb is not None and need_rb_for_aux
        use_rb_for_risk = ind_rb is not None and need_rb_for_risk
        use_rb_for_cf = ind_rb is not None and need_rb_for_cf
        z_rb = action_rb = state_rb = None
        risk_balance_logs = {}
        if use_rb_for_aux or use_rb_for_risk or use_rb_for_cf:
            state_rb, action_rb, _ns_rb, _r_rb, _nd_rb = self.replay_buffer.get_batch_by_indices(ind_rb)
            if self.obs_normalizer is not None:
                state_rb = self.obs_normalizer.normalize(state_rb)
            z_rb = self.encoder(state_rb)
            risk_balance_logs.update(
                {f"risk_balance/sampled_{k}": v
                 for k, v in self.replay_buffer.describe_risk_meta_fractions(ind_rb).items()}
            )

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

            # PHASE2 (critic_risk_input): action-risk prediction for the TD
            # target's (z_next, next_actions) pair, via the polyak-averaged
            # TARGET head (mirrors encoder_target -- keeps whatever extra signal
            # feeds the target Q on a slow-moving copy). Already under
            # torch.no_grad() so no explicit .detach() is needed.
            # STAGE 3: below action_risk_head.enable_from_stage, skip the
            # forward pass entirely and feed the critic a fixed all-zero
            # (batch, 2) tensor instead -- extra_dim never changes with stage.
            extra_next = None
            if self.critic_risk_input_enabled:
                if self._action_risk_active:
                    # PHASE2 (temporal): paired with z_next -> from the TARGET
                    # encoder on next_state, mirroring z_next's own origin.
                    ar_temporal_next = (
                        self.encoder_target.temporal_feature(next_state)
                        if self.action_risk_temporal_enabled else None
                    )
                    extra_next = self.action_risk_head_target(
                        z_next, next_actions, temporal_feature=ar_temporal_next)
                else:
                    extra_next = torch.zeros(
                        z_next.shape[0], self.action_risk_head_target.out.out_features,
                        device=self.device)

            # Get target quantiles
            next_quantiles = self.critic_target(z_next, next_actions, extra=extra_next)  # [B, n_critics, n_quantiles]

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

        # PHASE2: Action-Risk Head prediction for the STORED (z, action) pair.
        # Computed once (with graph) and reused: `.detach()` for the critic's
        # `extra` input (critic loss must never backprop into this head -- see
        # action_risk_head.py's gradient-rule docstring), the SAME tensor
        # (still attached) for the supervised loss further below.
        # PHASE2 (temporal): ar_temporal_cur also gets REUSED (detached) for the
        # actor-update call further below -- same state batch, same encoder ->
        # identical feature, no need to recompute it a third time.
        # STAGE 3: below action_risk_head.enable_from_stage, skip this forward
        # pass entirely (ar_pred_cur stays None -> the supervised loss below is
        # naturally skipped too) -- extra_cur is filled with a fixed all-zero
        # (batch, 2) tensor when critic_risk_input is enabled, same as extra_next.
        ar_pred_cur = None
        ar_temporal_cur = None
        if self.action_risk_enabled and self.action_risk_head is not None and self._action_risk_active:
            if self.action_risk_temporal_enabled:
                ar_temporal_cur = self.encoder.temporal_feature(state)
            ar_pred_cur = self.action_risk_head(
                z, action, temporal_feature=ar_temporal_cur)
        if self.critic_risk_input_enabled:
            extra_cur = (
                ar_pred_cur.detach() if ar_pred_cur is not None
                else torch.zeros(z.shape[0], self.action_risk_head.out.out_features,
                                  device=self.device)
            )
        else:
            extra_cur = None

        # Get current quantiles (encoder latent keeps its graph here)
        current_quantiles = self.critic(z, action, extra=extra_cur)  # [B, n_critics, n_quantiles]

        # Compute critic loss
        critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)

        # AUX_PRED: future-risk prediction loss; gradients flow into the shared
        # encoder together with the critic loss (encoder = critic + beta*aux).
        # STAGE 3: beta computed FIRST and the WHOLE block (aux head forward,
        # aux-temporal-context encoder forward, action-conditioned future-
        # action lookup) is skipped when it's exactly 0 -- not just the loss
        # contribution zeroed after computing it. beta==0 covers BOTH an
        # explicit aux_prediction.stagewise_loss_schedule entry of 0.0 for the
        # current stage AND the pre-warmup window (aux_beta_warmup_steps not
        # yet elapsed) -- both cases already contribute nothing to the loss,
        # so skipping the compute changes no numerics, only wasted work.
        aux_logs = dict(risk_balance_logs)
        total_trunk_loss = critic_loss
        # beta already computed above (before the balanced-batch draw) so that
        # use_rb_for_aux can skip the wasted encoder forward when beta == 0.
        # RISK_BALANCE: when a balanced batch is available, the aux SUPERVISED
        # loss trains on it (z_rb/action_rb/its own aux target) INSTEAD OF the
        # primary uniform batch -- the encoder still gets exactly one aux
        # gradient contribution per train() call, just sourced from the
        # stratified batch. Falls back to the primary z/aux_target below
        # whenever use_rb_for_aux is False (feature off, or no balanced draw
        # this step), which is byte-identical to the pre-existing behaviour.
        _aux_z = z_rb if use_rb_for_aux else z
        _aux_target_src = (
            self.replay_buffer.get_last_aux(ind_rb) if use_rb_for_aux else aux_target
        )
        _aux_indices = ind_rb if use_rb_for_aux else None
        if (self.aux_enabled and self.aux_head is not None and _aux_target_src is not None
                and beta != 0.0):
            # AUX_PRED (v2): aux-only temporal context (recent in-episode state
            # history). None when temporal is off -> the head ignores it and the
            # v1 path is unchanged. Shares the encoder graph so it shapes E_psi.
            temporal_ctx = None
            tctx = self._compute_temporal_ctx(indices=_aux_indices)
            if tctx is not None:
                temporal_ctx, hist_valid = tctx
                aux_logs["aux/hist_len_mean"] = float(hist_valid.float().mean().item())

            if self.aux_action_conditioned:
                # Same target L_i (future risk from s_i), but conditioned on the
                # upcoming in-episode action sequence [a_i, .., a_{i+K-1}].
                fa = self.replay_buffer.get_last_future_actions(
                    self.aux_cfg.action_conditioned_steps, indices=_aux_indices)
                aux_pred = None
                if fa is not None:
                    future_actions, valid_len = fa
                    aux_pred = self.aux_head(
                        _aux_z, future_actions, valid_len, temporal_ctx=temporal_ctx)
                    aux_logs["aux/valid_len_mean"] = float(valid_len.float().mean().item())
            else:
                aux_pred = self.aux_head(_aux_z, temporal_ctx=temporal_ctx)

            if aux_pred is not None:
                aux_loss, _logs = compute_aux_loss(
                    aux_pred, _aux_target_src, self.aux_cfg, self.device
                )
                aux_logs.update(_logs)
                total_trunk_loss = critic_loss + beta * aux_loss
                aux_logs["aux/beta"] = float(beta)
                if use_rb_for_aux:
                    aux_logs["risk_balance/aux_uses_balanced_batch"] = 1.0

        # PHASE2: Action-Risk Head supervised loss (weighted MSE/SmoothL1
        # against the PRIVILEGED GT target stored in the buffer). This is the
        # head's ONLY gradient path -- added into the SAME trunk loss as the
        # aux loss above, independent of whether critic_risk_input is on (the
        # head can be trained standalone, condition 3 of the 4-way experiment
        # matrix). RISK_BALANCE: when a balanced batch is available, this
        # SUPERVISED loss trains on a FRESH forward pass over (z_rb, action_rb)
        # instead of reusing ar_pred_cur -- ar_pred_cur itself is untouched and
        # still exclusively feeds critic_risk_input's extra_cur / the actor
        # update's extra_pi below, so switching the loss source here changes
        # NOTHING about the critic-feed / actor-gradient channel.
        action_risk_loss_val = 0.0
        action_risk_logs = dict(risk_balance_logs) if use_rb_for_risk else {}
        if self.action_risk_enabled and ar_pred_cur is not None:
            if use_rb_for_risk:
                ar_temporal_rb = (
                    self.encoder.temporal_feature(state_rb)
                    if self.action_risk_temporal_enabled else None
                )
                ar_pred_src = self.action_risk_head(
                    z_rb, action_rb, temporal_feature=ar_temporal_rb)
                ar_target = self.replay_buffer.get_last_action_risk(ind_rb)
                action_risk_logs["risk_balance/action_risk_uses_balanced_batch"] = 1.0
            else:
                ar_pred_src = ar_pred_cur
                ar_target = self.replay_buffer.get_last_action_risk()
            if ar_target is not None:
                action_risk_loss, _ar_logs = weighted_action_risk_loss(
                    ar_pred_src, ar_target, self.action_risk_cfg)
                total_trunk_loss = total_trunk_loss + self.action_risk_cfg.loss_weight * action_risk_loss
                action_risk_loss_val = float(action_risk_loss.detach().item())
                action_risk_logs["action_risk/loss"] = action_risk_loss_val
                action_risk_logs.update(_ar_logs)

        # Counterfactual supervision: TWO terms trained from the SAME head/
        # encoder call contract [z, action(, temporal)] -> (B, H):
        #   1. fixed_candidate_loss: every configured candidate action at once
        #      for each replay state, matching the env's (candidate,horizon)
        #      target matrix -- the fixed candidates themselves are never
        #      executed or inserted into the core TQC batch.
        #   2. executed_action_loss: the REPLAY-STORED action (the batch's own
        #      `action`/`action_rb`, i.e. what was actually executed) against
        #      its own multi-horizon target (computed pre-motion by the env
        #      for that exact transition) -- this is what lets the head score
        #      continuous actions it was never shown as a fixed candidate,
        #      closing the fixed-candidate-only generalization gap.
        # cf_supervised_ran gates BOTH the warm-up counter increment AND the
        # actor penalty below: a step with no target (buffer not yet filled /
        # stage-gated off) must neither advance the ramp nor apply a penalty
        # from an untrained head.
        counterfactual_logs = {}
        cf_temporal_cur = None
        cf_supervised_ran = False
        if (self.counterfactual_risk_enabled
                and self.counterfactual_risk_head is not None
                and self._counterfactual_risk_active):
            cf_z = z_rb if use_rb_for_cf else z
            cf_action = action_rb if use_rb_for_cf else action
            cf_state = state_rb if use_rb_for_cf else state
            cf_indices = ind_rb if use_rb_for_cf else None
            cf_target = self.replay_buffer.get_last_counterfactual_risk(
                cf_indices)
            cf_executed_target = self.replay_buffer.get_last_executed_action_risk(
                cf_indices)
            if cf_target is not None and cf_executed_target is not None:
                batch_n = cf_z.shape[0]
                cand_n = self.counterfactual_risk_cfg.num_candidates
                horizon_n = self.counterfactual_risk_cfg.num_horizons
                candidates = self._counterfactual_candidates.unsqueeze(0).expand(
                    batch_n, -1, -1)
                z_candidates = cf_z.unsqueeze(1).expand(
                    -1, cand_n, -1).reshape(batch_n * cand_n, -1)
                temporal_candidates = None
                cf_temporal = None
                if self.counterfactual_risk_cfg.use_temporal_context:
                    cf_temporal = self.encoder.temporal_feature(cf_state)
                    temporal_candidates = cf_temporal.unsqueeze(1).expand(
                        -1, cand_n, -1).reshape(batch_n * cand_n, -1)
                    # Cache the PRIMARY-batch feature before the trunk update
                    # so the later actor call pairs z_actor with a feature from
                    # the same encoder weights. The balanced supervised batch
                    # has a different state batch and therefore needs a second
                    # primary forward here.
                    cf_temporal_cur = (
                        self.encoder.temporal_feature(state)
                        if use_rb_for_cf else cf_temporal)
                cf_pred = self.counterfactual_risk_head(
                    z_candidates, candidates.reshape(batch_n * cand_n, -1),
                    temporal_feature=temporal_candidates).reshape(
                        batch_n, cand_n, horizon_n)
                cf_target = cf_target.reshape(batch_n, cand_n, horizon_n)
                fixed_loss, fixed_loss_logs = weighted_counterfactual_risk_loss(
                    cf_pred, cf_target, self.counterfactual_risk_cfg)

                # Executed-action term: same head, REPLAY action as input
                # (not a fixed candidate), against the executed-action target.
                cf_pred_executed = self.counterfactual_risk_head(
                    cf_z, cf_action, temporal_feature=cf_temporal)
                executed_loss, executed_loss_logs = weighted_counterfactual_risk_loss(
                    cf_pred_executed, cf_executed_target, self.counterfactual_risk_cfg)

                cf_loss = (
                    fixed_loss
                    + self.counterfactual_risk_cfg.executed_action_loss_weight
                    * executed_loss)
                total_trunk_loss = (
                    total_trunk_loss
                    + self.counterfactual_risk_cfg.loss_weight * cf_loss)
                counterfactual_logs.update(fixed_loss_logs)
                counterfactual_logs.update({
                    f"counterfactual_risk/executed_{k.split('/', 1)[1]}": v
                    for k, v in executed_loss_logs.items()
                })
                counterfactual_logs["counterfactual_risk/loss"] = float(
                    cf_loss.detach().item())
                counterfactual_logs["counterfactual_risk/fixed_candidate_loss"] = float(
                    fixed_loss.detach().item())
                counterfactual_logs["counterfactual_risk/executed_action_loss"] = float(
                    executed_loss.detach().item())
                with torch.no_grad():
                    executed_err = (cf_pred_executed - cf_executed_target).abs()
                    counterfactual_logs["counterfactual_risk/executed_action_mae"] = float(
                        executed_err.mean().item())
                    per_horizon_mae = executed_err.mean(dim=0)
                    for _hi, _h_sec in enumerate(self.counterfactual_risk_cfg.horizons_sec):
                        counterfactual_logs[
                            f"counterfactual_risk/executed_mae_h{_h_sec:g}s"] = float(
                                per_horizon_mae[_hi].item())
                if use_rb_for_cf:
                    counterfactual_logs[
                        "risk_balance/counterfactual_uses_balanced_batch"] = 1.0

                cf_supervised_ran = True

        self.critic_optimizer.zero_grad()
        total_trunk_loss.backward()
        self.critic_optimizer.step()

        # Only count an update as "supervised" once the trunk optimizer has
        # actually applied it -- an exception between building the loss and
        # this step() call (e.g. a NaN in backward()) would otherwise have
        # already advanced the warm-up/ramp counter for a step that never
        # updated the head's weights.
        if cf_supervised_ran:
            self._cf_supervised_updates += 1

        """******************************************
        ** Actor Update
        ******************************************"""
        # Sample new actions (on the DETACHED latent: actor never updates encoder)
        actions_pi, log_prob = self.actor.action_log_prob(z_actor)

        # PHASE2 (critic_risk_input): action-risk prediction for the actor's OWN
        # candidate action. Do NOT detach the whole tensor here -- that would
        # sever d(extra_pi)/d(actions_pi), which is exactly the pathway that
        # lets "this action raises predicted risk" propagate into the actor's
        # gradient via qf_pi (the actor never sees extra_pi directly; it only
        # feels it through the critic's learned Q, which is the intended
        # "indirect" channel). What must NOT happen is action_risk_head's OWN
        # parameters picking up a gradient from actor_loss (only its dedicated
        # supervised loss may update it) -- so freeze its params for this one
        # forward call instead of detaching the output. requires_grad=False at
        # forward-time means autograd never records those weights as needing
        # grad for this graph, so restoring True right after (before backward())
        # is safe and standard practice for a "frozen submodule, live input"
        # forward pass.
        # STAGE 3: below action_risk_head.enable_from_stage, skip this forward
        # pass too and feed the critic a fixed all-zero (batch, 2) tensor
        # (same neutral value as extra_cur/extra_next) -- constant w.r.t.
        # actions_pi, so it contributes no gradient signal to the actor via
        # this channel, matching "no risk information available yet."
        extra_pi = None
        if self.critic_risk_input_enabled:
            if self._action_risk_active:
                # PHASE2 (temporal): reuse ar_temporal_cur (same state batch,
                # same online encoder as z_actor's origin) -- .detach() so
                # actor_loss cannot flow into self.encoder.temporal's
                # parameters, mirroring why z_actor itself is detached from z.
                # This does NOT touch the actions_pi gradient path (a separate
                # cat() branch below), so it does not affect the
                # critic_risk_input "gradient into the actor" channel at all.
                ar_temporal_pi = (
                    ar_temporal_cur.detach() if ar_temporal_cur is not None else None
                )
                # STAGE 5: exception-safe freeze (was a bare set-False/call/
                # set-True sequence that could leave the head permanently
                # frozen if the forward call raised).
                with _frozen_params(self.action_risk_head):
                    extra_pi = self.action_risk_head(
                        z_actor, actions_pi, temporal_feature=ar_temporal_pi)
            else:
                extra_pi = torch.zeros(
                    z_actor.shape[0], self.action_risk_head.out.out_features,
                    device=self.device)

        # Get Q-values for policy actions.
        # STAGE 5: freeze the critic's OWN parameters for this forward call
        # (exception-safe context manager) -- actor_loss.backward() must
        # reach actions_pi (the intended "does this action raise Q" signal)
        # but must NOT waste a backward pass building gradients into the
        # critic's parameters, which the critic-trunk update above already
        # owns exclusively and which get overwritten by critic_optimizer.step()
        # on the NEXT train() call regardless. Same freeze-not-detach pattern
        # already used for action_risk_head just above (detaching the whole
        # output would sever d(qf_pi)/d(actions_pi) too).
        with _frozen_params(self.critic):
            qf_pi = self.critic(z_actor, actions_pi, extra=extra_pi)
        # Average over quantiles and critics
        qf_pi = qf_pi.mean(dim=2).mean(dim=1, keepdim=True)

        # Direct actor risk penalty. Freeze the head's weights but keep the
        # input-side gradient d(risk)/d(action), exactly as for the indirect
        # action-risk path above. The state/temporal representation is detached,
        # so this update remains actor-only.
        #
        # Gated on cf_supervised_ran (not just _counterfactual_risk_active):
        # a step where the head had no supervised target to train on must not
        # apply a penalty from an untrained/stale head, and must not advance
        # the warm-up/ramp counter either (see CounterfactualRiskConfig.
        # effective_actor_penalty_weight's docstring).
        actor_risk_penalty = actions_pi.new_zeros(())
        effective_penalty_weight = 0.0
        if (self.counterfactual_risk_enabled
                and self.counterfactual_risk_head is not None
                and self._counterfactual_risk_active
                and cf_supervised_ran):
            temporal_pi = None
            if self.counterfactual_risk_cfg.use_temporal_context:
                if cf_temporal_cur is None:
                    # Defensive fallback (normally populated by the supervised
                    # block above). Inference-only: do not build an encoder graph.
                    with torch.no_grad():
                        cf_temporal_cur = self.encoder.temporal_feature(state)
                temporal_pi = cf_temporal_cur.detach()
            with _frozen_params(self.counterfactual_risk_head):
                predicted_horizon_risk = self.counterfactual_risk_head(
                    z_actor, actions_pi, temporal_feature=temporal_pi)
            if self.counterfactual_risk_cfg.actor_risk_aggregation == "max":
                actor_risk_penalty = predicted_horizon_risk.max(dim=1).values.mean()
            elif self.counterfactual_risk_cfg.actor_risk_aggregation == "weighted_mean":
                actor_risk_penalty = (
                    predicted_horizon_risk * self._cf_horizon_weights
                ).sum(dim=1).mean()
            else:
                actor_risk_penalty = predicted_horizon_risk.mean()
            effective_penalty_weight = (
                self.counterfactual_risk_cfg.effective_actor_penalty_weight(
                    self._cf_supervised_updates))
            self._cf_effective_actor_penalty_weight = effective_penalty_weight

        actor_loss = (
            (ent_coef * log_prob - qf_pi).mean()
            + effective_penalty_weight * actor_risk_penalty)
        if self.counterfactual_risk_enabled:
            counterfactual_logs["counterfactual_risk/actor_penalty"] = float(
                actor_risk_penalty.detach().item())
            counterfactual_logs[
                "counterfactual_risk/actor_penalty_weight"] = float(
                    self.counterfactual_risk_cfg.actor_penalty_weight)
            counterfactual_logs[
                "counterfactual_risk/effective_actor_penalty_weight"] = float(
                    effective_penalty_weight)
            counterfactual_logs[
                "counterfactual_risk/actor_penalty_contribution"] = float(
                    effective_penalty_weight * actor_risk_penalty.detach().item())
            counterfactual_logs["counterfactual_risk/supervised_updates"] = float(
                self._cf_supervised_updates)
            # OOD monitoring: normalized-action L2 distance from the actor's
            # own continuous action to its NEAREST fixed candidate -- computed
            # whenever the head is stage-active, independent of whether a
            # supervised update ran THIS step, so drift is visible through
            # warm-up too.
            if self._counterfactual_risk_active:
                with torch.no_grad():
                    _diffs = (actions_pi.unsqueeze(1)
                             - self._counterfactual_candidates.unsqueeze(0))
                    _nearest = _diffs.norm(dim=-1).min(dim=1).values
                    counterfactual_logs[
                        "counterfactual_risk/actor_candidate_distance_mean"] = float(
                            _nearest.mean().item())
                    counterfactual_logs[
                        "counterfactual_risk/actor_candidate_distance_max"] = float(
                            _nearest.max().item())

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
            # STAGE 5: Polyak averaging, batched via torch._foreach_* instead
            # of a per-parameter Python loop -- numerically identical to the
            # original ``target.data.copy_(tau*src.data + (1-tau)*target.data)``
            # loop (see test_foreach_polyak_matches_per_parameter_loop_
            # numerically), fewer CUDA kernel launches for networks with many
            # parameter tensors (5 critics here).
            _polyak_update_foreach(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            # AUX_PRED: keep the target encoder in lock-step with the encoder.
            if self.encoder.has_params():
                _polyak_update_foreach(self.encoder.parameters(), self.encoder_target.parameters(), self.tau)

            # PHASE2: keep the target Action-Risk Head in lock-step (mirrors the
            # encoder_target block above).
            if self.action_risk_head is not None:
                _polyak_update_foreach(
                    self.action_risk_head.parameters(),
                    self.action_risk_head_target.parameters(), self.tau)

            if self.prioritized:
                self.replay_buffer.reset_max_priority()

        """******************************************
        ** Logging
        ******************************************"""
        # STAGE 6: .item() forces a GPU->CPU sync. Compute each scalar AT MOST
        # ONCE per train() call (the original code called critic_loss.item()
        # etc. TWICE -- once for TensorBoard, once for JSON) and skip the sync
        # entirely on a step where NEITHER sink is due (scalar_log_interval /
        # json_log_interval, both default 1 -> log every step, byte-identical
        # to before).
        log_scalar_now = (self.training_steps % self.scalar_log_interval == 0) and self.writer
        log_json_now = (self.training_steps % self.json_log_interval == 0)
        if log_scalar_now or log_json_now:
            scalars = {
                "loss/critic":  float(critic_loss.item()),
                "loss/actor":   float(actor_loss.item()),
                "values/Q":     float(qf_pi.mean().item()),
                "values/Q_max": float(current_quantiles.max().item()),
            }
            if self.ent_coef_auto:
                scalars["values/ent_coef"] = float(ent_coef.item())
                scalars["loss/ent_coef"] = float(ent_coef_loss.item())
            # AUX_PRED / PHASE2: aux_logs / action_risk_logs are already
            # plain floats (materialised where computed, gated by Stage 3's
            # forward-skip -- not re-synced here).
            if self.aux_enabled and aux_logs:
                scalars.update(aux_logs)
            if self.action_risk_enabled and action_risk_logs:
                scalars.update(action_risk_logs)
            if self.counterfactual_risk_enabled and counterfactual_logs:
                scalars.update(counterfactual_logs)
            # RISK_BALANCE: sampled-pool fractions logged regardless of which
            # consumer (aux / action-risk / both) used the balanced batch this
            # step, so they're visible even if only one of the two is enabled.
            if risk_balance_logs:
                scalars.update(risk_balance_logs)
            # RISK_BALANCE: RAW (whole-buffer) pool fractions alongside the
            # sampled_* ones above -- lets a run's actual positive-class share
            # (the imbalance risk-balanced sampling / pos_weight are
            # correcting for) be read directly off TensorBoard/JSON instead of
            # inferred only from the balanced batch's own composition. Only
            # computed when metadata storage is on (store_risk_meta implies
            # replay_buffer.risk_meta.enabled) and only at the log interval
            # (never every train() call) -- an O(buffer_size) reduction is
            # cheap at this cadence but wasteful every step.
            if self.store_risk_meta:
                scalars.update({
                    f"risk_balance/raw_{k}": v
                    for k, v in self.replay_buffer.describe_risk_meta_fractions_raw().items()
                })

            if log_scalar_now:
                for _k, _v in scalars.items():
                    self.writer.add_scalar(_k, _v, self.training_steps)
            if log_json_now:
                self._json_log(self.training_steps, **scalars)

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

