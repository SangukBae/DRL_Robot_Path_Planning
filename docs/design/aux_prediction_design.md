# Auxiliary Prediction Network for TQC (AUX_PRED)

Future-aware auxiliary prediction added on top of the TQC curriculum agent.
A shared encoder sits between the 87-D state and the actor/critic; an auxiliary
head predicts the **future risk of dynamic obstacles** from the shared latent,
which pushes the encoder to learn motion-aware features. All auxiliary code is
tagged `AUX_PRED` for easy discovery / removal.

## 1. Which paper ideas were borrowed

| Paper (in `paper/`) | Idea reused | Where |
|---|---|---|
| **Falcon** — *From Cognition to Precognition* (`Falcon/falcon/auxiliary_tasks.py`) | Shared latent feeds a future-prediction auxiliary head; auxiliary loss back-props only into the shared representation; `L_total = L_main + beta * L_aux`; privileged sim labels used at training time only | `aux_prediction.py`, `tqc_agent.py` gradient rule |
| **Proximity-Aware** — *Exploiting Proximity-Aware Tasks* (`ProximitySocialNav/.../social_auxiliary_tasks.py`) | Predict a **fixed-size egocentric risk / compass map** instead of an ID-matched per-human trajectory: `risk = clamp(1 - d/D_c, 0, 1)` per angular sector. No masking / data association. Heads are detached at evaluation | `aux_prediction_labels.py`, `AuxiliaryHead` |
| **DiPCAN** — *Distilling Privileged Information* (`DiPCAN...pdf`) | Privileged pedestrian ground-truth is used ONLY to build training labels in simulation; at deployment the policy is encoder + actor, no privileged info / aux head needed | env label path is training-only; inference uses `encoder -> actor` |

## 2. What was simplified for this project

- **Off-policy / i.i.d. replay.** Falcon and Proximity-Aware are on-policy
  (PPO) and recurrent (LSTM/GRU over a rollout). TQC samples i.i.d.
  transitions from a replay buffer, so the v1 path is **single-step**: predict
  the future risk map directly from the encoder latent of one state. A
  recurrent encoder is intentionally avoided (it would change the actor/critic
  input path and hurt off-policy stability).
- **Fixed-size labels.** Variable pedestrian count + ID matching is replaced by
  a fixed `H x K` egocentric risk map (Proximity-Aware), so the buffer stores a
  constant-width target and the head is a plain MLP.
- **Constant-velocity rollout.** Privileged future positions use a short CV
  extrapolation per horizon, which is accurate because
  `environment_curriculum.py` deliberately makes pedestrians predictable /
  non-reactive during training.

## 3. v1 (default) vs v2 (opt-in)

| | v1 (primary path) | v2 (options) |
|---|---|---|
| Encoder | `87 -> 256 -> 128` (ELU), shared | same (**deliberately unchanged** — feeds actor/critic) |
| Aux head trunk | 1× `Linear -> ELU` | configurable depth/width + LayerNorm (`aux_trunk_layers`, `aux_trunk_hidden_dim`, `aux_head_layernorm`) |
| Aux head output | risk map `H x K` (sigmoid) | + future min-distance `H`; + distributional risk `H x K x Q` |
| Action context | action-embed + GRU last hidden | + 1-block masked self-attention over the GRU sequence (`action_condition_attention`) |
| Loss | MSE on risk map | + `min_distance_loss_weight * MSE`; + `distributional_loss_weight *` quantile/pinball on risk; + optional `aux_beta_warmup_steps` ramp |
| Temporal (state history) | none | **WIRED, opt-in**: GRU over recent in-episode states, aux-head input only (`aux_prediction_temporal.py`, Section 3d) |

**Implemented and wired today:** v1 risk-map MSE; the future min-distance head
(`min_distance_loss_weight > 0`, **on in the curriculum config**); the
distributional risk option (`use_distributional_aux`, wired, default off, now
weighted by `distributional_loss_weight`); the **action-conditioned** variant
(`action_conditioned_aux`, see Section 3b); the **configurable deeper/LayerNorm
aux trunk** and the **action-GRU self-attention block** (Section 3c); the
**aux-only temporal context** over recent state history (`temporal_enabled`,
**on in the curriculum config**, Section 3d); the **beta warmup schedule**
(`aux_beta_warmup_steps`) and an **optional shared-encoder LayerNorm**
(`encoder_layernorm`, default off, Section 3e). All are exercised by
`tests/test_aux_prediction.py`, `tests/test_buffer_state_history.py` and
`tests/test_tqc_agent_temporal.py`.

**Aux-branch-only by design.** Everything in this round adds parameters to the
**aux head**, never to the shared encoder or the actor/critic. The shared encoder
is kept at `256/128` ON PURPOSE: `latent_dim` is the actor/critic input width, so
growing it is an actor/critic change and breaks strict checkpoint resume of those
nets. Concentrating capacity in the aux branch keeps the off-policy actor/critic
path stable (the encoder grows the aux head 6.8× — ~101k → ~690k params — while
the encoder stays 55k and actor/critic are byte-for-byte unchanged).

The wired v2 options never change the v1 actor/critic path: extra heads are
additive and gated by config flags. The temporal context (Section 3d) likewise
feeds **only** the aux head — the actor/critic stay non-recurrent.

## 3b. Action-conditioned auxiliary (Proximity-Aware style)

**Why.** The v1 head predicts the future risk map from a SINGLE state `s_t`
(`z_t -> risk`). But the same `s_t` can lead to very different futures depending
on what the robot does next; a single-step observation cannot express that. The
action-conditioned variant predicts the SAME target from `z_t` AND the upcoming
action sequence, so the encoder/head learn an action-aware future representation
(this is exactly the Proximity-Aware idea: predict the proximity feature over
`[t, t+k]` conditioned on the belief `h_t` and the actions `{a_j}`; here `z_t`
plays the belief role).

**Difference from single-step aux.** Only the *prediction source* changes; the
*target and loss are identical* (`compute_aux_loss` is reused). v1:
`aux_head(z_t)`. Action-conditioned: `aux_head(z_t, [a_t..a_{t+K-1}])` via an
action embedder + GRU whose context is concatenated onto `z_t`.

**Target alignment.** For a sampled transition `i`, `z_i = E(s_i)`, the target is
the unchanged `L_i` (future risk computed from `s_i`), and the conditioning
actions are `[a_i, a_{i+1}, ..., a_{i+K-1}]` where `a_j` is the action stored in
transition `j` (the action taken AT `s_j`). So `z_i` + the actions taken from
`s_i` onward predict the future measured from `s_i` — no misalignment. `K =
action_conditioned_steps` (default 4). `a_i` (k=0) is always in-episode, so every
sample has ≥1 valid conditioning action.

**Replay buffer changes (off-policy).** TQC samples i.i.d. transitions, so the
future action sequence is reconstructed by index walking:
- The buffer optionally tracks a per-transition `traj_end` flag (allocated only
  when `track_traj=True`, i.e. action-conditioned aux is on). `add()` clears the
  slot's flag (so a circular-reuse never inherits a stale boundary); the trainer
  calls `mark_last_traj_end()` in the episode-end block (goal / collision /
  timeout / eval-cut) to flag the true last transition.
- `get_last_future_actions(K)` returns `(future_actions (B,K,A), valid_len (B,))`
  for the just-sampled indices. A step is valid only while the previous index is
  not a boundary AND the next index is not the write head `ptr` (which is
  unwritten or the stale oldest transition). Invalid steps are zero-padded; the
  GRU consumes only the first `valid_len` actions (output gathered at
  `valid_len-1`), so **out-of-episode actions never reach the loss**. Circular
  wrap is handled by `% max_size` plus the `ptr` check. `save()/load()` persist
  `traj_end` (optional npz key, backward compatible).

**actor / critic path: UNCHANGED.** The actor still consumes `z.detach()`; the
critic still consumes `(z, action)`. The action sequence only feeds the aux
head, whose params (incl. the GRU) are updated by `critic_loss + beta*aux_loss`
together with the encoder. The actor loss leaks no gradient into the encoder or
the aux head (verified by the gradient-isolation tests).

**Resume / ablation compatibility.**
- *Replay buffer:* a buffer checkpoint saved WITHOUT episode boundaries (no
  `traj_end`, i.e. it predates action-conditioned/temporal aux) cannot be loaded
  into a run that needs them — silently zero boundaries would splice the
  future-action / state-history walks across old episode boundaries. This is
  enforced at two levels: `buffer.load()` **fail-fasts** (`RuntimeError`) as the
  authoritative low-level guard, and the resume orchestrator `tqc_io.load`
  **catches that and degrades to a FRESH buffer with a loud log** so the MODEL
  still resumes (the actor/critic/encoder drive the policy; warmup refills the
  buffer). So full `load_replay_buffer=True` resume is **graceful** for ANY old
  checkpoint — but the replay buffer is only *carried over* when the old buffer
  already had boundaries (it did whenever action-conditioned or temporal aux was
  on); otherwise it restarts empty. A direct `buffer.load()` caller still gets
  the hard error.
- *Aux head / optimizer:* the aux head is training-only and its architecture
  differs between the single-step `AuxiliaryHead` and `ActionConditionedAuxHead`
  (and it lives in the same optimizer param group as the encoder/critic). When a
  checkpoint's aux head does not match the current config, `tqc_agent.load()`
  **keeps the freshly-initialised aux head and fresh critic-optimizer moments**
  (with a warning) instead of aborting — so the encoder / actor / critic, which
  drive the policy, still resume and the aux head retrains. Matching configs load
  exactly as before.

**Config (`hyperparameters_tqc.yaml`, requires `enabled: true`).**
`action_conditioned_aux` (master, default false), `action_conditioned_steps`
(K), `action_embed_dim`, `action_condition_hidden_dim` (GRU hidden), plus the
strengthening keys in Section 3c (`action_condition_attention`,
`action_condition_attention_heads`, `aux_trunk_*`). A contradictory config
(`action_conditioned_aux` true while `enabled` false, or
`action_conditioned_steps < 1`) raises at agent construction.

## 3c. Strengthened aux branch (Falcon-inspired, aux-only)

The action-conditioned head was deepened to bring in Falcon's "LSTM + attention
+ richer prediction" idea while keeping every change inside the aux branch.

**Self-attention over the action-GRU sequence.** Falcon applies a
`MultiheadAttention` over its LSTM output sequence. Here, when
`action_condition_attention: true`, ONE `nn.MultiheadAttention` block
(residual + `LayerNorm`) is applied to the GRU's per-step outputs `out_seq`
`(B, K, H_gru)` **before** the boundary-safe gather:

```
out_seq = GRU(embed(a_0..a_{K-1}))
attn    = MultiheadAttention(out_seq, key_padding_mask = [k >= valid_len])
out_seq = LayerNorm(out_seq + attn)        # residual block
ctx     = out_seq[ valid_len - 1 ]         # gather as before
```

The `key_padding_mask` hides every out-of-episode step (`k >= valid_len`), so the
context can weigh the whole **in-episode** action window yet **padded /
cross-boundary actions still never influence the prediction**. This preserves the
replay-buffer boundary contract from Section 3b: `valid_len >= 1`, so index 0 is
always unmasked and the gathered position is finite. The masking contract is
locked by `test_padded_actions_do_not_change_output` /
`test_padded_actions_have_zero_gradient` (asserted with attention ON and OFF:
perturbing or differentiating w.r.t. the padded actions changes nothing).
`action_condition_attention_heads` must divide `H_gru`; it silently falls back to
1 head otherwise.

**Deeper aux trunk.** `_build_aux_trunk` builds `aux_trunk_layers` blocks of
`Linear [-> LayerNorm] -> ELU` at width `aux_trunk_hidden_dim`, shared by both
`AuxiliaryHead` and `ActionConditionedAuxHead`. The defaults (1 layer, no
LayerNorm, `hidden = max(latent_dim, 128)`) reproduce the original single
`Linear -> ELU` trunk exactly, so configs that do not set the new keys — and the
`enabled: false` baseline — are unchanged. The curriculum config opts into
`aux_trunk_layers: 2`, `aux_trunk_hidden_dim: 256`, `aux_head_layernorm: true`.

**Capacity (curriculum config).** Encoder unchanged (~55k). Action-conditioned
aux head ~101k -> ~690k params (`action_embed_dim 32->64`,
`action_condition_hidden_dim 128->256`, +attention block, 2-layer LayerNorm
trunk, +min-distance head). Actor/critic param count and input width unchanged
(`latent_dim` still 128). New keys:
`action_condition_attention`, `action_condition_attention_heads`,
`aux_trunk_hidden_dim`, `aux_trunk_layers`, `aux_head_layernorm`.

**Resume.** The aux head lives in the same optimizer param group as the
encoder/critic and is training-only; on a checkpoint whose aux-head architecture
no longer matches, `tqc_agent.load()` keeps the fresh aux head + fresh
critic-optimizer moments (with a warning) — so this architecture change resumes
the encoder/actor/critic cleanly and just retrains the aux head (Section 3b).

## 3d. Aux-only temporal context (recent state history)

**Why.** Falcon runs an LSTM over the rollout to forecast future human motion. A
single state `s_t` cannot express where obstacles are *heading*; a short history
`[s_t, s_{t-1}, ...]` can. The temporal branch gives the encoder/aux-head that
backward-looking cue **without making the policy recurrent**.

**What is wired.** When `temporal_enabled: true` (requires `enabled: true`):
- `LAP.get_last_state_history(N)` returns, for each sampled transition `i`, the
  REVERSE-time window `[s_i, s_{i-1}, ..., s_{i-N+1}]` (`N = history_len`) plus a
  `valid_len` count. It walks BACKWARD with the **same boundary-safety contract**
  as `get_last_future_actions`: it stops at an episode boundary (`traj_end`) and
  at the circular-buffer seam (the oldest written slot), zero-padding the rest.
  Index 0 is always the current state, so `valid_len >= 1`.
- Each history state is encoded by the **shared encoder** (so the temporal loss
  also shapes `E_psi`); `TemporalContextEncoder` (a small GRU, with optional
  masked self-attention) summarises the latent window into a context vector
  gathered at `valid_len - 1` — identical leading-valid masking to the action
  path, so padded / cross-boundary states never influence the output.
- The context is **concatenated** onto the aux-head input only:
  `feat = trunk([z_t, action_ctx?, temporal_ctx])`. Concatenation (not
  cross-attention) is deliberate — it is the same stable fusion already used for
  `z_t + action_ctx`, avoiding the instability/compute of cross-attending three
  heterogeneous vectors. The actor (`z.detach()`) and critic (`z, action`) paths
  are byte-for-byte unchanged.

**Why a GRU over latents (not a temporal conv, not a recurrent encoder).** The
window is short (`N≈4`) and variable (a boundary truncates it); a GRU with a
per-row `valid_len` gather handles that exactly as the existing action GRU does,
so there is ONE masking contract in the codebase. A recurrent *encoder* was
rejected on purpose: it would change the actor/critic input path and hurt
off-policy i.i.d. replay stability.

**Replay/boundary safety.** `track_traj` is enabled whenever EITHER the
action-conditioned OR the temporal branch is on, so the buffer always carries the
`traj_end` flags both walks need. The eval path
(`curriculum_aux_eval._build_state_history`) reconstructs the same backward
window inside one finished episode, so formal aux metrics use a faithful temporal
context (a length-1 window is the safe fallback when no history is supplied).

**Resume.** The temporal encoder is **training-only** (dropped at inference) and
saved as `<prefix>_temporal_encoder.pth`. On resume `tqc_io.load` loads it
strictly when present and matching; on an architecture mismatch OR a missing file
(resuming a pre-temporal checkpoint) it keeps a freshly-initialised temporal
encoder and rebuilds the critic optimizer (its params share that group), logging
the fresh-init either way — the encoder/actor/critic still resume cleanly and only
the temporal branch retrains. The matching replay-buffer policy (carry over when
the old buffer had boundaries, else degrade to a fresh buffer) is in Section 3b.

## 3e. Shared-encoder polish + loss balancing

**Encoder LayerNorm (opt-in, default off).** `encoder_layernorm: true` inserts a
`LayerNorm` after each hidden pre-activation in `SharedEncoder`. **`out_dim`
stays `latent_dim`**, so the actor/critic input width — and their checkpoints —
never move. It is left OFF by default and NOT enabled in the shipped config
because it changes the encoder `state_dict`: a strict encoder load on resume
would then mismatch (`tqc_io` falls back to a fresh encoder with a logged
warning and rebuilds the optimizer). Flip it on only for a FRESH run.

**Why `latent_dim` itself is NOT grown.** `latent_dim` is the actor/critic input
width. Growing it is an actor/critic architectural change that breaks strict
checkpoint resume of those nets and alters the policy input contract — exactly
the off-policy-stability risk this work avoids. All added capacity therefore goes
into the **aux branch** (deeper trunk, action GRU + attention, temporal GRU),
never the shared latent.

**Loss balancing.**
- `distributional_loss_weight` (default 1.0) now multiplies the pinball term
  (previously an implicit 1.0). The risk target is in `[0,1]` and the pinball
  uses `kappa=1.0`, so `|td| <= 1` keeps it in the quadratic-Huber regime — it
  largely echoes the MSE with added quantile spread. That is why it stays **off
  by default** (low marginal value for the cost) yet is a balanced, tested option
  (`use_distributional_aux: true`, weight `0.5` in the config when enabled).
- `aux_beta_warmup_steps` (curriculum config: 5000) linearly ramps the
  trunk-level `beta_aux` from 0 to `loss_weight`, so a noisy freshly-initialised
  aux head does not perturb early critic learning. 0 disables the schedule
  (constant `beta`, the previous behaviour). Logged as `aux/beta`.

## 4. File structure

New (all `AUX_PRED`-tagged):
- `policy/aux_prediction.py` — `AuxPredConfig`, `SharedEncoder`, `_build_aux_trunk`
  (shared deeper/LayerNorm trunk), `AuxiliaryHead`, `ActionConditionedAuxHead`
  (action-embed + GRU + optional masked self-attention, Sections 3b/3c).
- `policy/aux_prediction_losses.py` — `compute_aux_loss`, `quantile_pinball_loss`.
- `policy/aux_prediction_temporal.py` — v2 `TemporalContextEncoder` (WIRED,
  opt-in): GRU over recent in-episode state latents, aux-head input only
  (Section 3d).
- `environment/aux_prediction_labels.py` — `AuxLabelConfig`,
  `compute_future_risk_labels` (privileged label generation).
- `docs/design/aux_prediction_design.md` — this file.

Minimally edited:
- `policy/tqc_agent.py` — encoder insertion + aux loss in `train()` + save/load
  + `load_encoder_for_inference()` (restores the encoder for actor-only paths).
- `utils/buffer.py` — optional `aux_dim` target storage + `track_traj` episode
  boundaries; `get_last_future_actions` (action-conditioned) and
  `get_last_state_history` (temporal) boundary-safe index walks (backward
  compatible; arrays allocated only when the matching aux feature is on).
- `policy/aux_prediction_losses.py` — `distributional_loss_weight` on the pinball
  term (Section 3e).
- `policy/tqc_io.py` — save/load the training-only temporal encoder with the same
  graceful fresh-init-on-mismatch contract as the aux head (Section 3d).
- `environment/environment_interface.py` — **common layer**: `get_dimensions()`
  caches the RL state dim; `reset()/step()` slice the appended label off for
  EVERY client and expose it as `self.last_aux_label` (no-op when aux disabled).
- `policy/train_tqc_curriculum_agent.py` — snapshot `self.last_aux_label` into
  the (cur/next) bookkeeping; pass `aux_target` to `replay_buffer.add`.
- `environment/environment.py` — parse config + `_append_aux_labels()` on the
  step/reset responses.
- `policy/test_tqc_agent.py`, `policy/real_policy_runner.py` — call
  `load_encoder_for_inference()` so aux-enabled checkpoints restore the encoder.
- `config/hyperparameters_tqc.yaml`, `config/environment_curriculum.yaml`,
  `config/environment.yaml` — `aux_prediction:` blocks.

Because the label split now lives in `EnvInterface`, all clients (the TD7/SAC
trainers, `generalization_eval.py`, the actor-only test path, ...) strip the
appended label automatically; `generalization_eval.py` already restores the
encoder via the full `Agent.load(...)`.

## 5. Training data flow

```
env step/reset (privileged human_states + GT robot pose)
  -> compute_future_risk_labels  ->  label [H*K risk | H min-dist]
  -> appended after the 87-D state in the Step/Reset service response
trainer reset()/step() override
  -> slice 87-D RL state  +  aux label  (_aux_label_cur / _aux_label_next)
  -> replay_buffer.add(s, a, s', r, done, aux_target = label_for_s)
agent.train()
  -> z = encoder(s)                       (grad)
  -> critic(z, a) -> critic_loss
  -> aux_head(z)  -> aux_loss             (vs stored label)
  -> (critic_loss + beta * aux_loss).backward()  updates encoder+critic+aux
  -> actor reads z.detach()               (actor never updates the encoder)
```

Gradient rule (enforced in `tqc_agent.py`):
- encoder is updated by `critic_loss + beta_aux * aux_loss` only.
- `z_actor = z.detach()` -> actor / temperature gradients never reach the
  encoder.
- target path uses a Polyak-synced `encoder_target`.

## 6. Inference / deployment (no privileged info)

At evaluation / test / real-robot time the policy is just
`encoder -> actor`. The auxiliary head and the privileged label path are
training-only and unused; with the env-side flag off, the service returns the
plain 87-D state and the trainer's slicer is a no-op. Baseline TQC checkpoints
(no encoder files) load unchanged.

## 7. How to enable

v1 (single-step risk map) requires BOTH switches on, with **matching**
`num_sectors` / `horizons_sec`:

- `config/hyperparameters_tqc.yaml` -> `aux_prediction.enabled: true`
- `config/environment_curriculum.yaml` -> `aux_prediction.enabled: true`

v2 add-ons (any subset):
- `min_distance_loss_weight: > 0`  (future min-distance head) -- wired (ON in the
  curriculum config; the env always emits the min-dist label block, so no env
  change is needed)
- `use_distributional_aux: true`   (quantile/pinball risk) -- wired, default off
- `action_condition_attention: true` + `aux_trunk_layers`/`aux_trunk_hidden_dim`/
  `aux_head_layernorm` -- wired aux-branch strengthening (Section 3c)
- `temporal_enabled: true` (+ `history_len`, `temporal_context_dim`,
  `temporal_attention`) -- aux-only recent-state context, WIRED (Section 3d)
- `aux_beta_warmup_steps`, `distributional_loss_weight`, `encoder_layernorm`
  -- loss balancing / encoder polish (Section 3e)

With both switches off (default) the system is byte-for-byte baseline TQC.

## 8. Safety guards (fail-fast)

Auxiliary prediction has two hard guards so a misconfiguration aborts the run
instead of training a silently-wrong model:

- **Config consistency (fail-fast, STRUCTURAL).** The label geometry lives in
  two configs (`hyperparameters_tqc.yaml` agent side, `environment_curriculum.yaml`
  env side). To make the check airtight, the env prepends a **geometry/version
  header** to every label on the wire:
  `[VERSION, num_sectors, num_horizons, h_0 .. h_{H-1}]` (see
  `aux_prediction_labels.wire_header` / `parse_aux_wire`). `EnvInterface` parses
  it into `last_aux_meta` and strips it, so the buffer still stores only the
  label. At the first reset the curriculum trainer compares this header
  field-by-field against the agent's aux config and **raises** on any
  disagreement: missing label, missing/garbled header, **wire-format `VERSION`
  != `AUX_WIRE_VERSION`**, different `num_sectors`, different number of
  horizons, different horizon **values**, or mismatched label length. The
  version check is done first, so even an env/agent pair whose length and
  geometry numbers happen to match but whose wire layout differs is rejected.
  This closes the "same total length, different structure" hole
  (e.g. `K=16,H=3` vs `K=2,H=17` both give `label_dim=51`) that a length-only
  check would miss, and is authoritative regardless of which env config file the
  env node actually loaded -- it reflects the real wire output.
- **Non-curriculum block.** Auxiliary prediction is only wired into
  `train_tqc_curriculum_agent.py`. `TrainTQCBase` carries
  `AUX_SUPPORTED = False` and raises in `__init__` if an aux-enabled agent is
  built by a non-supporting subclass; `TrainTQCCurriculum` and the aux-aware
  `GeneralizationEval` opt in with `AUX_SUPPORTED = True`. So enabling
  `aux_prediction.enabled` on any other TQC trainer fails immediately. (The
  IEQN trainers use a separate `tqc_ieqn_agent.Agent` without aux and are
  unaffected; the common `EnvInterface` still strips any env-appended label for
  them.)
