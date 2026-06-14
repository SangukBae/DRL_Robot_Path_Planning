"""Checkpoint save / load for the TQC agent.

Extracted from ``tqc_agent.py`` so all model/optimizer/replay-buffer
serialization lives in one place, separate from the agent's network construction
and training step. These are free functions that take the ``Agent`` instance as
their first argument; ``Agent.save`` / ``Agent.load`` /
``Agent.load_encoder_for_inference`` are kept as thin methods that delegate here,
so every existing call site and the on-disk checkpoint layout are unchanged.
"""

import os

import torch


def save(agent, directory, filename):
    """Save model parameters (+ optimizers, entropy coef, replay buffer)."""
    os.makedirs(directory, exist_ok=True)
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

    def _torch_load(path):
        try:
            return torch.load(path, map_location=maploc, weights_only=True)
        except TypeError:
            # Older PyTorch versions do not support weights_only.
            return torch.load(path, map_location=maploc)

    # Actor
    p = f"{directory}/{filename}_actor.pth"
    if os.path.exists(p):
        agent.actor.load_state_dict(_torch_load(p))
    if load_optimizer_state:
        p = f"{directory}/{filename}_actor_optimizer.pth"
        if os.path.exists(p):
            agent.actor_optimizer.load_state_dict(_torch_load(p))

    # Critic
    p = f"{directory}/{filename}_critic.pth"
    if os.path.exists(p):
        agent.critic.load_state_dict(_torch_load(p))
    p = f"{directory}/{filename}_critic_target.pth"
    if os.path.exists(p):
        agent.critic_target.load_state_dict(_torch_load(p))
    if load_optimizer_state:
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

    # Checkpoint actor
    p = f"{directory}/{filename}_checkpoint_actor.pth"
    if os.path.exists(p):
        agent.checkpoint_actor.load_state_dict(_torch_load(p))

    # AUX_PRED: shared encoder + auxiliary head (only when this run uses aux
    # AND the checkpoint carries them; baseline checkpoints lack these files
    # and are loaded unchanged).
    if agent.encoder.has_params():
        p = f"{directory}/{filename}_encoder.pth"
        if os.path.exists(p):
            agent.encoder.load_state_dict(_torch_load(p))
        p = f"{directory}/{filename}_encoder_target.pth"
        if os.path.exists(p):
            agent.encoder_target.load_state_dict(_torch_load(p))
        p = f"{directory}/{filename}_checkpoint_encoder.pth"
        if os.path.exists(p):
            agent.checkpoint_encoder.load_state_dict(_torch_load(p))
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
            agent.replay_buffer.load(buf_path)


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
