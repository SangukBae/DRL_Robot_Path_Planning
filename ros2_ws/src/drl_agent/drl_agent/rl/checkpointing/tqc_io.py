"""Checkpoint save / load for the TQC agent.

Extracted from ``tqc_agent.py`` so all model/optimizer/replay-buffer
serialization lives in one place, separate from the agent's network construction
and training step. These are free functions that take the ``Agent`` instance as
their first argument; ``Agent.save`` / ``Agent.load`` /
``Agent.load_encoder_for_inference`` are kept as thin methods that delegate here,
so every existing call site and the on-disk checkpoint layout are unchanged.
"""

import copy
import json
import os

import torch


def _obs_norm_manifest_path(directory, filename):
    return f"{directory}/{filename}_obs_norm_manifest.json"


def _cf_state_path(directory, filename):
    return f"{directory}/{filename}_cf_state.json"


def save(agent, directory, filename):
    """Save model parameters (+ optimizers, entropy coef, replay buffer)."""
    os.makedirs(directory, exist_ok=True)

    # STAGE 8 (isolated experimental feature): record the observation-
    # normalization contract (mode + every scale value) so a checkpoint saved
    # under a DIFFERENT normalization setup can never be silently resumed
    # (see load()'s matching check below). Always written (even when
    # disabled) so an explicit "disabled" record distinguishes "predates
    # Stage 8 entirely" (no manifest file) from "Stage 8 present but off".
    obs_normalizer = getattr(agent, "obs_normalizer", None)
    manifest = {"enabled": obs_normalizer is not None}
    if obs_normalizer is not None:
        manifest.update(agent.obs_norm_cfg.manifest_dict())
    with open(_obs_norm_manifest_path(directory, filename), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    # Actor
    torch.save(agent.actor.state_dict(), f"{directory}/{filename}_actor.pth")
    torch.save(agent.actor_optimizer.state_dict(), f"{directory}/{filename}_actor_optimizer.pth")

    # Critic
    torch.save(agent.critic.state_dict(), f"{directory}/{filename}_critic.pth")
    torch.save(agent.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
    torch.save(agent.critic_optimizer.state_dict(), f"{directory}/{filename}_critic_optimizer.pth")

    # Checkpoint
    torch.save(agent.checkpoint_actor.state_dict(), f"{directory}/{filename}_checkpoint_actor.pth")

    # AUX_PRED: shared encoder + auxiliary head (training-only).  Saved only
    # when the encoder actually has parameters (aux enabled); inference does
    # not require the aux head.
    if agent.encoder.has_params():
        torch.save(agent.encoder.state_dict(), f"{directory}/{filename}_encoder.pth")
        torch.save(agent.encoder_target.state_dict(), f"{directory}/{filename}_encoder_target.pth")
        torch.save(agent.checkpoint_encoder.state_dict(), f"{directory}/{filename}_checkpoint_encoder.pth")
        if agent.aux_head is not None:
            torch.save(agent.aux_head.state_dict(), f"{directory}/{filename}_aux_head.pth")
        # AUX_PRED (v2): aux-only temporal encoder (training-only, like the head).
        if getattr(agent, "temporal_encoder", None) is not None:
            torch.save(
                agent.temporal_encoder.state_dict(),
                f"{directory}/{filename}_temporal_encoder.pth",
            )

    # PHASE2: Action-Risk Head (+ target copy), training-only, like the aux head.
    if getattr(agent, "action_risk_head", None) is not None:
        torch.save(agent.action_risk_head.state_dict(),
                   f"{directory}/{filename}_action_risk_head.pth")
        torch.save(agent.action_risk_head_target.state_dict(),
                   f"{directory}/{filename}_action_risk_head_target.pth")
    if getattr(agent, "counterfactual_risk_head", None) is not None:
        torch.save(
            agent.counterfactual_risk_head.state_dict(),
            f"{directory}/{filename}_counterfactual_risk_head.pth")
        # Actor-penalty warm-up/ramp counter: persisted so a resumed run
        # continues its ramp instead of restarting from update 0.
        with open(_cf_state_path(directory, filename), "w", encoding="utf-8") as f:
            json.dump(
                {"supervised_updates": int(agent._cf_supervised_updates)}, f)

    # Entropy coefficient
    if agent.ent_coef_auto:
        torch.save(agent.log_ent_coef, f"{directory}/{filename}_log_ent_coef.pth")
        torch.save(agent.ent_coef_optimizer.state_dict(), f"{directory}/{filename}_ent_coef_optimizer.pth")
    else:
        torch.save(agent.ent_coef_tensor, f"{directory}/{filename}_ent_coef_tensor.pth")

    # Replay buffer — enables full off-policy resume
    agent.replay_buffer.save(f"{directory}/{filename}_replay_buffer")


def load(
    agent,
    directory,
    filename,
    *,
    load_optimizer_state=True,
    load_replay_buffer=True,
):
    maploc = agent.device

    # STAGE 8: fail fast (BEFORE loading any weights) on an observation-
    # normalization contract mismatch -- never silently load a checkpoint
    # trained under a different normalization setup (different scale values
    # would make every saved weight's learned input distribution wrong).
    current_enabled = getattr(agent, "obs_normalizer", None) is not None
    manifest_path = _obs_norm_manifest_path(directory, filename)
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        saved_enabled = bool(saved.get("enabled", False))
        if saved_enabled != current_enabled:
            raise RuntimeError(
                f"[tqc_io] observation_normalization contract mismatch: "
                f"checkpoint was saved with enabled={saved_enabled}, this run "
                f"has enabled={current_enabled}. Resuming would silently feed "
                "every saved weight a differently-scaled input. Start a fresh "
                "run (load_replay_buffer/load_model=False) or match the "
                "observation_normalization config exactly."
            )
        if current_enabled:
            current = agent.obs_norm_cfg.manifest_dict()
            mismatched = {
                k: (saved.get(k), current.get(k))
                for k in current if k != "enabled" and saved.get(k) != current.get(k)
            }
            if mismatched:
                raise RuntimeError(
                    "[tqc_io] observation_normalization scale mismatch between "
                    f"checkpoint and this run: {mismatched} (saved, current). "
                    "Resuming would silently feed every saved weight a "
                    "differently-scaled input. Start a fresh run or match the "
                    "observation_normalization config exactly."
                )
    elif current_enabled:
        raise RuntimeError(
            "[tqc_io] observation_normalization.enabled=true but this "
            f"checkpoint has no '{os.path.basename(manifest_path)}' (it "
            "predates the observation-normalization feature, so it was "
            "trained WITHOUT normalization). Resuming would silently feed "
            "every saved weight a differently-scaled input. Start a fresh "
            "run or disable observation_normalization."
        )

    def _torch_load(path):
        try:
            return torch.load(path, map_location=maploc, weights_only=True)
        except TypeError:
            # Older PyTorch versions do not support weights_only.
            return torch.load(path, map_location=maploc)

    # Actor.
    # When aux is DISABLED the shared encoder is a parameter-free identity, so the
    # actor/critic consume the RAW state_dim directly. Toggling observation
    # time-context (frame stacking) then changes the actor/critic INPUT width and
    # an old checkpoint no longer matches. That is a deliberate architecture
    # change, so — like the encoder/aux-head paths — degrade gracefully: keep the
    # freshly-initialised net (its __init__ optimizer is already bound to it) and
    # SKIP loading the now-stale optimizer state, with a loud log, rather than
    # aborting the whole resume. (Aux ENABLED: actor/critic take latent_dim, which
    # stacking does not change, so this load just succeeds as before.)
    actor_fresh = False
    p = f"{directory}/{filename}_actor.pth"
    if os.path.exists(p):
        try:
            agent.actor.load_state_dict(_torch_load(p))
        except (RuntimeError, KeyError) as e:
            actor_fresh = True
            print(
                "[AUX_PRED] actor checkpoint is incompatible with the current "
                "state_dim (observation time-context / frame stacking toggled on "
                "an aux-DISABLED policy); keeping a freshly-initialised actor AND "
                f"fresh actor-optimizer moments (they retrain). Details: {e}"
            )
    if load_optimizer_state and not actor_fresh:
        p = f"{directory}/{filename}_actor_optimizer.pth"
        if os.path.exists(p):
            agent.actor_optimizer.load_state_dict(_torch_load(p))

    # Critic (same graceful contract as the actor above).
    critic_fresh = False
    p = f"{directory}/{filename}_critic.pth"
    if os.path.exists(p):
        try:
            agent.critic.load_state_dict(_torch_load(p))
            p = f"{directory}/{filename}_critic_target.pth"
            if os.path.exists(p):
                agent.critic_target.load_state_dict(_torch_load(p))
        except (RuntimeError, KeyError) as e:
            critic_fresh = True
            print(
                "[AUX_PRED] critic checkpoint is incompatible with the current "
                "state_dim (observation time-context / frame stacking toggled on "
                "an aux-DISABLED policy); keeping a freshly-initialised critic AND "
                f"fresh critic-optimizer moments (they retrain). Details: {e}"
            )
    if load_optimizer_state and not critic_fresh:
        p = f"{directory}/{filename}_critic_optimizer.pth"
        if os.path.exists(p):
            # AUX_PRED: critic_optimizer also owns the encoder + aux-head
            # params, so its param-group size changes when the aux head
            # architecture changes (single-step <-> action-conditioned).  On
            # such a mismatch keep fresh optimizer moments (the critic /
            # encoder WEIGHTS already loaded above) instead of aborting.
            # Even when this load "succeeds", a later aux-head state-dict
            # mismatch can still invalidate the loaded moments if only the
            # param COUNT matched while shapes / semantics changed; that
            # later path rebuilds this optimizer unconditionally.
            try:
                agent.critic_optimizer.load_state_dict(_torch_load(p))
            except (ValueError, RuntimeError, KeyError) as e:
                print(
                    "[AUX_PRED] critic optimizer state is incompatible with "
                    "the current aux config (the aux head changed the trunk "
                    "param group); keeping fresh optimizer moments. "
                    f"Details: {e}"
                )

    # Checkpoint actor. Its input width tracks the actor's (latent_dim, which
    # equals raw state_dim when aux is disabled and the encoder is identity), so a
    # state_dim change breaks this load too. When the actor went fresh — or the
    # saved checkpoint actor itself mismatches — MIRROR the (fresh) actor into
    # checkpoint_actor instead of leaving an independently-random net: the
    # use_checkpoint=True select/eval path reads checkpoint_actor immediately on
    # resume (before train_and_checkpoint() next syncs it), so it must hold the
    # same weights the actor uses, not a different random init.
    if actor_fresh:
        agent.checkpoint_actor.load_state_dict(agent.actor.state_dict())
    else:
        p = f"{directory}/{filename}_checkpoint_actor.pth"
        if os.path.exists(p):
            try:
                agent.checkpoint_actor.load_state_dict(_torch_load(p))
            except (RuntimeError, KeyError) as e:
                agent.checkpoint_actor.load_state_dict(agent.actor.state_dict())
                print(
                    "[AUX_PRED] checkpoint-actor is incompatible with the current "
                    "state_dim (frame stacking toggled); syncing it from the "
                    f"loaded actor instead. Details: {e}"
                )

    # AUX_PRED: shared encoder + auxiliary head (only when this run uses aux
    # AND the checkpoint carries them; baseline checkpoints lack these files
    # and are loaded unchanged).
    if agent.encoder.has_params():
        # The shared encoder's INPUT width == state_dim. Toggling observation
        # time-context (frame stacking) changes state_dim, so an old checkpoint's
        # encoder no longer matches. That is a deliberate architecture change, so
        # mirror the aux-head contract: try a strict load; on a shape/key mismatch
        # keep the freshly-initialised encoder(s) AND rebuild the critic optimizer
        # (encoder params live in that trunk group) and log it, rather than
        # aborting the whole resume. The actor/critic still load below and keep
        # driving the policy; only the encoder retrains from the new input width.
        try:
            p = f"{directory}/{filename}_encoder.pth"
            if os.path.exists(p):
                agent.encoder.load_state_dict(_torch_load(p))
            p = f"{directory}/{filename}_encoder_target.pth"
            if os.path.exists(p):
                agent.encoder_target.load_state_dict(_torch_load(p))
            p = f"{directory}/{filename}_checkpoint_encoder.pth"
            if os.path.exists(p):
                agent.checkpoint_encoder.load_state_dict(_torch_load(p))
        except (RuntimeError, KeyError) as e:
            agent.critic_optimizer = agent._make_critic_optimizer()
            print(
                "[AUX_PRED] shared-encoder checkpoint is incompatible with the "
                "current state_dim (observation time-context / frame stacking was "
                "toggled); keeping freshly-initialised encoder(s) AND fresh "
                f"critic-optimizer moments (they retrain). Details: {e}"
            )
        p = f"{directory}/{filename}_aux_head.pth"
        if agent.aux_head is not None and os.path.exists(p):
            # AUX_PRED: the aux head is TRAINING-ONLY.  Its architecture
            # changes between the single-step AuxiliaryHead and the
            # ActionConditionedAuxHead (and with min-dist / distributional
            # options), so a strict load fails when upgrading / switching an
            # ablation.  Try a strict load; on a state-dict mismatch keep the
            # freshly-initialised head (it retrains) and warn, rather than
            # aborting the whole resume -- the encoder/actor/critic, which
            # actually drive the policy, still load above.
            try:
                agent.aux_head.load_state_dict(_torch_load(p))
            except (RuntimeError, KeyError) as e:
                # The aux head changed architecture.  The critic_optimizer
                # (loaded earlier) owns the aux-head params in the SAME param
                # group, so its moments are now stale -- and the load above
                # may have "succeeded" if only the param COUNT matched while
                # shapes/semantics differ, leaving wrong moments that would
                # crash or silently corrupt the next step().  Rebuild the
                # optimizer fresh to guarantee no stale moment survives.
                agent.critic_optimizer = agent._make_critic_optimizer()
                print(
                    "[AUX_PRED] aux-head checkpoint is incompatible with the "
                    "current aux config (e.g. single-step <-> action-"
                    "conditioned, or changed heads); keeping a freshly-"
                    "initialised aux head AND fresh critic-optimizer moments "
                    f"(both will retrain). Details: {e}"
                )

        # AUX_PRED (v2): aux-only temporal encoder (training-only). Same graceful
        # contract as the aux head: load strictly when the checkpoint carries a
        # matching module; on a missing file (resuming a PRE-temporal checkpoint)
        # or an architecture mismatch keep the freshly-initialised temporal
        # encoder and rebuild the critic optimizer (its params share that group),
        # so the encoder/actor/critic still resume and only the temporal branch
        # retrains. Logged either way so the fresh-init path is never silent.
        if getattr(agent, "temporal_encoder", None) is not None:
            p = f"{directory}/{filename}_temporal_encoder.pth"
            if os.path.exists(p):
                try:
                    agent.temporal_encoder.load_state_dict(_torch_load(p))
                except (RuntimeError, KeyError) as e:
                    agent.critic_optimizer = agent._make_critic_optimizer()
                    print(
                        "[AUX_PRED] temporal-encoder checkpoint is incompatible "
                        "with the current aux config; keeping a freshly-"
                        "initialised temporal encoder AND fresh critic-optimizer "
                        f"moments (both retrain). Details: {e}"
                    )
            else:
                agent.critic_optimizer = agent._make_critic_optimizer()
                print(
                    "[AUX_PRED] no temporal-encoder file in this checkpoint "
                    "(resuming a run saved before the temporal context was "
                    "enabled); using a freshly-initialised temporal encoder and "
                    "fresh critic-optimizer moments."
                )

    # PHASE2: Action-Risk Head (+ target), training-only. Same graceful
    # mismatch contract as the aux head: a strict load on an architecture/
    # config change (e.g. hidden_dim toggled, or use_temporal_context flipped)
    # falls back to a freshly-initialised head and rebuilds the critic
    # optimizer (the head's params live in that trunk group) rather than
    # aborting the whole resume.
    if getattr(agent, "action_risk_head", None) is not None:
        p = f"{directory}/{filename}_action_risk_head.pth"
        if os.path.exists(p):
            # torch's strict load_state_dict copies EVERY matching-shape
            # tensor in place BEFORE raising for a mismatched one (e.g. only
            # l1.weight's width changed -> l1.bias/out.* silently warm-start
            # from the incompatible checkpoint even though the call raises).
            # Snapshot this run's own fresh init now and restore it verbatim
            # on failure, so "keeping a freshly-initialised head" below is
            # actually true instead of a fresh/partially-loaded hybrid.
            _ar_fresh = copy.deepcopy(agent.action_risk_head.state_dict())
            _art_fresh = copy.deepcopy(agent.action_risk_head_target.state_dict())
            try:
                agent.action_risk_head.load_state_dict(_torch_load(p))
                p_t = f"{directory}/{filename}_action_risk_head_target.pth"
                if os.path.exists(p_t):
                    agent.action_risk_head_target.load_state_dict(_torch_load(p_t))
            except (RuntimeError, KeyError) as e:
                agent.action_risk_head.load_state_dict(_ar_fresh)
                agent.action_risk_head_target.load_state_dict(_art_fresh)
                agent.critic_optimizer = agent._make_critic_optimizer()
                print(
                    "[PHASE2] action-risk-head checkpoint is incompatible with "
                    "the current config; keeping a freshly-initialised head AND "
                    f"fresh critic-optimizer moments (both retrain). Details: {e}"
                )
        else:
            print(
                "[PHASE2] no action-risk-head file in this checkpoint (resuming "
                "a run saved before action_risk_head was enabled); using a "
                "freshly-initialised head."
            )

    if getattr(agent, "counterfactual_risk_head", None) is not None:
        p = f"{directory}/{filename}_counterfactual_risk_head.pth"
        cf_state_p = _cf_state_path(directory, filename)
        if os.path.exists(p):
            fresh = copy.deepcopy(agent.counterfactual_risk_head.state_dict())
            try:
                agent.counterfactual_risk_head.load_state_dict(_torch_load(p))
                if os.path.isfile(cf_state_p):
                    with open(cf_state_p, "r", encoding="utf-8") as f:
                        saved_cf_state = json.load(f)
                    agent._cf_supervised_updates = int(
                        saved_cf_state.get("supervised_updates", 0))
                else:
                    # Head loaded but predates the warm-up/ramp counter --
                    # restart the ramp rather than guessing a count.
                    agent._cf_supervised_updates = 0
                    print(
                        "[PHASE2] no counterfactual-risk warm-up/ramp state "
                        "in this checkpoint (resuming a run saved before "
                        "actor_penalty_warmup/ramp_updates was added); "
                        "starting the counter at 0.")
            except (RuntimeError, KeyError) as e:
                agent.counterfactual_risk_head.load_state_dict(fresh)
                agent.critic_optimizer = agent._make_critic_optimizer()
                agent._cf_supervised_updates = 0
                print(
                    "[PHASE2] counterfactual-risk-head checkpoint is "
                    "incompatible; using fresh head/optimizer state. "
                    f"Details: {e}")
        else:
            agent.critic_optimizer = agent._make_critic_optimizer()
            agent._cf_supervised_updates = 0
            print(
                "[PHASE2] checkpoint has no counterfactual-risk-head; using "
                "a freshly-initialised head and optimizer state.")

    # Entropy coefficient
    if agent.ent_coef_auto:
        p = f"{directory}/{filename}_log_ent_coef.pth"
        if os.path.exists(p):
            loaded = _torch_load(p)
            # <<< 핵심: 텐서 객체를 교체하지 말고 data만 복사 >>>
            agent.log_ent_coef.data.copy_(loaded.to(maploc).data)
        if load_optimizer_state:
            p = f"{directory}/{filename}_ent_coef_optimizer.pth"
            if os.path.exists(p):
                agent.ent_coef_optimizer.load_state_dict(_torch_load(p))
    else:
        p = f"{directory}/{filename}_ent_coef_tensor.pth"
        if os.path.exists(p):
            loaded = _torch_load(p)
            agent.ent_coef_tensor = loaded.to(maploc).detach()

    # Replay buffer
    if load_replay_buffer:
        buf_path = f"{directory}/{filename}_replay_buffer"
        if os.path.isfile(buf_path + ".npz"):
            # AUX_PRED: buffer.load() FAIL-FASTS (RuntimeError) when this run needs
            # per-transition episode boundaries (track_traj — action-conditioned
            # and/or temporal aux) but the saved buffer predates them and has no
            # 'traj_end'. That low-level guard is correct: loading those zeros
            # would silently splice future-action / state-history walks across
            # episode boundaries. Here, at the RESUME orchestrator level, we don't
            # let it abort the whole model load (the actor/critic/encoder already
            # loaded above and DRIVE the policy): we degrade to a FRESH buffer and
            # log it loudly. So a pre-boundary checkpoint still resumes the model;
            # only the replay buffer restarts (warmup refills it). A direct
            # buffer.load() caller still gets the hard error.
            try:
                agent.replay_buffer.load(buf_path)
            except RuntimeError as e:
                print(
                    "[AUX_PRED] replay-buffer checkpoint is incompatible with the "
                    "current aux config (it has no episode boundaries this run "
                    "needs); resuming the MODEL with a FRESH replay buffer "
                    f"(warmup will refill it). Details: {e}"
                )


def load_encoder_for_inference(agent, actor_path):
    """AUX_PRED: restore the shared encoder for an actor-only inference path.

    Inference runs state -> encoder -> actor, so an aux-enabled checkpoint
    MUST load the encoder alongside the actor; otherwise the actor receives
    a randomly-initialised latent and the policy is broken.  The matching
    encoder file is derived from the actor checkpoint path
    (``<prefix>_actor.pth`` -> ``<prefix>_encoder.pth``).

    Returns
    -------
    bool
        True  -> encoder ready (loaded, or not needed for a baseline /
                 identity encoder, i.e. aux disabled).
        False -> an aux encoder is REQUIRED but its file is missing; the
                 caller should treat this as a fatal inference error.
    """
    # Baseline (aux disabled): encoder is a parameter-free identity, so the
    # actor consumes the raw state and there is nothing to restore.
    if not agent.encoder.has_params():
        return True

    if actor_path.endswith("_actor.pth"):
        enc_path = actor_path[: -len("_actor.pth")] + "_encoder.pth"
    else:
        enc_path = os.path.join(os.path.dirname(actor_path), "encoder.pth")
    if not os.path.isfile(enc_path):
        return False

    try:
        sd = torch.load(enc_path, map_location=agent.device, weights_only=True)
    except TypeError:
        sd = torch.load(enc_path, map_location=agent.device)
    agent.encoder.load_state_dict(sd)
    agent.encoder.eval()
    if getattr(agent, "checkpoint_encoder", None) is not None:
        agent.checkpoint_encoder.load_state_dict(sd)
        agent.checkpoint_encoder.eval()
    return True
