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

| | v1 (primary path) | v2 (options, off by default) |
|---|---|---|
| Encoder | `87 -> 256 -> 128` (ELU), shared | same |
| Aux head output | risk map `H x K` (sigmoid) | + future min-distance `H`; + distributional risk `H x K x Q` |
| Loss | MSE on risk map | + `min_distance_loss_weight * MSE`; + quantile/pinball on risk |
| Temporal | none | **scaffold only, NOT wired** (`aux_prediction_temporal.py`) |

**Implemented and wired today:** v1 risk-map MSE, the future min-distance head
(`min_distance_loss_weight > 0`), the distributional risk option
(`use_distributional_aux`), and the **action-conditioned** variant
(`action_conditioned_aux`, see Section 3b). These are exercised by the unit
tests.

**Not wired (scaffold only):** the temporal branch. `temporal_enabled`,
`temporal_mode`, `history_len`, `state_stack_len` are parsed into
`AuxPredConfig` and `aux_prediction_temporal.py` defines
`TemporalContextEncoder`, but nothing constructs it, samples a stacked
state/history from the replay buffer, or concatenates a temporal context into
the aux head. Setting `temporal_enabled: true` therefore has **no effect**
today; finishing it requires (a) buffer support for stacked-state sampling and
(b) concatenating the temporal context onto the aux-head input only. Listed
here as future work so the flag is not mistaken for a working feature.

The wired v2 options never change the v1 actor/critic path: extra heads are
additive and gated by config flags.

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
- *Replay buffer:* loading a pre-action-conditioned buffer checkpoint (no
  `traj_end`) into an action-conditioned run is **refused with a fail-fast**
  error in `buffer.load()` — silently zero boundaries would splice future-action
  sequences across old episode boundaries. Resume with `load_replay_buffer=False`
  (fresh buffer) or disable `action_conditioned_aux`.
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
(K), `action_embed_dim`, `action_condition_hidden_dim` (GRU hidden). A
contradictory config (`action_conditioned_aux` true while `enabled` false, or
`action_conditioned_steps < 1`) raises at agent construction.

## 4. File structure

New (all `AUX_PRED`-tagged):
- `policy/aux_prediction.py` — `AuxPredConfig`, `SharedEncoder`, `AuxiliaryHead`,
  `ActionConditionedAuxHead` (action-embed + GRU, Section 3b).
- `policy/aux_prediction_losses.py` — `compute_aux_loss`, `quantile_pinball_loss`.
- `policy/aux_prediction_temporal.py` — v2 `TemporalContextEncoder` (opt-in).
- `environment/aux_prediction_labels.py` — `AuxLabelConfig`,
  `compute_future_risk_labels` (privileged label generation).
- `docs/design/aux_prediction_design.md` — this file.

Minimally edited:
- `policy/tqc_agent.py` — encoder insertion + aux loss in `train()` + save/load
  + `load_encoder_for_inference()` (restores the encoder for actor-only paths).
- `utils/buffer.py` — optional `aux_dim` target storage (backward compatible).
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
- `min_distance_loss_weight: > 0`  (future min-distance head) -- wired
- `use_distributional_aux: true`   (quantile/pinball risk) -- wired
- `temporal_enabled: true`, `temporal_mode: state_stack` -- **scaffold only,
  has no effect yet** (see the v1/v2 section)

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
