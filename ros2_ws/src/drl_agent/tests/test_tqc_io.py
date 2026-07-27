"""Unit tests for tqc_io (checkpoint save / load), torch-gated.

Uses a lightweight fake Agent whose sub-objects expose the same
``state_dict`` / ``load_state_dict`` / ``has_params`` / ``save`` / ``load``
surface that tqc_io touches, so the on-disk file set, optional-load flags, the
aux-head-mismatch optimizer rebuild and the inference-encoder paths are all
exercised without torch nn graphs or ROS.
"""

import os

import pytest

# Defensive torch import (a module-level Skipped can abort directory collection
# under this pytest + ament plugin stack).
try:
    import torch
    import numpy as np
    import drl_agent.rl.checkpointing.tqc_io as tqc_io
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")


class FakeModule:
    def __init__(self, raise_on_load=False, tag="m"):
        self._sd = {"w": torch.tensor([1.0, 2.0])}
        self.raise_on_load = raise_on_load
        self.loaded = None
        self.tag = tag

    def state_dict(self):
        return self._sd

    def load_state_dict(self, sd):
        if self.raise_on_load:
            raise RuntimeError("aux arch mismatch")
        self.loaded = sd

    def eval(self):
        pass


class FakeEncoder(FakeModule):
    def __init__(self, has_params=False, **kw):
        super().__init__(**kw)
        self._has = has_params

    def has_params(self):
        return self._has


class FakeBuffer:
    def __init__(self):
        self.loaded = False

    def save(self, path):
        np.savez(path, x=np.array([1, 2, 3]))

    def load(self, path):
        self.loaded = True


class FakeAgent:
    def __init__(self, *, aux=False, aux_head_raises=False, ent_auto=True,
                 actor_raises=False, critic_raises=False):
        self.device = "cpu"
        self.actor = FakeModule(raise_on_load=actor_raises, tag="actor")
        self.actor_optimizer = FakeModule(tag="actor_opt")
        self.critic = FakeModule(raise_on_load=critic_raises, tag="critic")
        self.critic_target = FakeModule(tag="critic_target")
        self.critic_optimizer = FakeModule(tag="critic_opt")
        self.checkpoint_actor = FakeModule(tag="ckpt_actor")
        self.encoder = FakeEncoder(has_params=aux, tag="encoder")
        self.encoder_target = FakeModule(tag="encoder_target")
        self.checkpoint_encoder = FakeModule(tag="ckpt_encoder")
        self.aux_head = FakeModule(raise_on_load=aux_head_raises, tag="aux_head") if aux else None
        self.ent_coef_auto = ent_auto
        self.log_ent_coef = torch.tensor([0.5], requires_grad=True)
        self.ent_coef_optimizer = FakeModule(tag="ent_opt")
        self.ent_coef_tensor = torch.tensor([0.3])
        self.replay_buffer = FakeBuffer()
        self._rebuilt = 0

    def _make_critic_optimizer(self):
        self._rebuilt += 1
        return FakeModule(tag="critic_opt_rebuilt")


# --------------------------------------------------------------------------- #
def test_save_creates_expected_files_baseline(tmp_path):
    agent = FakeAgent(aux=False)
    tqc_io.save(agent, str(tmp_path), "ckpt")
    for suffix in ["_actor.pth", "_actor_optimizer.pth", "_critic.pth",
                   "_critic_target.pth", "_critic_optimizer.pth",
                   "_checkpoint_actor.pth", "_log_ent_coef.pth",
                   "_ent_coef_optimizer.pth", "_replay_buffer.npz"]:
        assert (tmp_path / f"ckpt{suffix}").exists(), suffix
    # Baseline (no aux): encoder/aux files must NOT be written.
    assert not (tmp_path / "ckpt_encoder.pth").exists()
    assert not (tmp_path / "ckpt_aux_head.pth").exists()


def test_save_writes_aux_files_when_enabled(tmp_path):
    agent = FakeAgent(aux=True)
    tqc_io.save(agent, str(tmp_path), "ckpt")
    for suffix in ["_encoder.pth", "_encoder_target.pth",
                   "_checkpoint_encoder.pth", "_aux_head.pth"]:
        assert (tmp_path / f"ckpt{suffix}").exists(), suffix


def test_load_roundtrip(tmp_path):
    src = FakeAgent(aux=False)
    tqc_io.save(src, str(tmp_path), "ckpt")
    dst = FakeAgent(aux=False)
    tqc_io.load(dst, str(tmp_path), "ckpt")
    assert dst.actor.loaded is not None
    assert dst.critic.loaded is not None
    assert dst.critic_target.loaded is not None
    assert dst.checkpoint_actor.loaded is not None
    assert dst.actor_optimizer.loaded is not None
    assert dst.replay_buffer.loaded is True


def test_load_skips_optimizer_when_requested(tmp_path):
    src = FakeAgent(aux=False)
    tqc_io.save(src, str(tmp_path), "ckpt")
    dst = FakeAgent(aux=False)
    tqc_io.load(dst, str(tmp_path), "ckpt", load_optimizer_state=False)
    assert dst.actor.loaded is not None          # weights still load
    assert dst.actor_optimizer.loaded is None    # optimizer skipped
    assert dst.critic_optimizer.loaded is None


def test_load_skips_replay_buffer_when_requested(tmp_path):
    src = FakeAgent(aux=False)
    tqc_io.save(src, str(tmp_path), "ckpt")
    dst = FakeAgent(aux=False)
    tqc_io.load(dst, str(tmp_path), "ckpt", load_replay_buffer=False)
    assert dst.replay_buffer.loaded is False


def test_ent_coef_fixed_path(tmp_path):
    src = FakeAgent(aux=False, ent_auto=False)
    tqc_io.save(src, str(tmp_path), "ckpt")
    assert (tmp_path / "ckpt_ent_coef_tensor.pth").exists()
    assert not (tmp_path / "ckpt_log_ent_coef.pth").exists()
    dst = FakeAgent(aux=False, ent_auto=False)
    tqc_io.load(dst, str(tmp_path), "ckpt")
    assert float(dst.ent_coef_tensor.item()) == pytest.approx(0.3)


def test_aux_head_mismatch_rebuilds_critic_optimizer(tmp_path):
    # Save with aux enabled, then load into an agent whose aux head rejects the
    # state dict (architecture changed) → critic_optimizer must be rebuilt.
    src = FakeAgent(aux=True)
    tqc_io.save(src, str(tmp_path), "ckpt")
    dst = FakeAgent(aux=True, aux_head_raises=True)
    tqc_io.load(dst, str(tmp_path), "ckpt")
    assert dst._rebuilt == 1
    assert dst.critic_optimizer.tag == "critic_opt_rebuilt"


def test_actor_state_dim_mismatch_keeps_fresh_and_syncs_checkpoint_actor(tmp_path):
    # Simulate a state_dim change (frame stacking toggled) on an aux-DISABLED
    # policy: the actor load raises. The resume must NOT abort; the actor stays
    # fresh, its optimizer state is skipped, and checkpoint_actor is SYNCED from
    # the fresh actor (not left as an independent random net) so the
    # use_checkpoint=True path reads the same weights.
    src = FakeAgent(aux=False)
    tqc_io.save(src, str(tmp_path), "ckpt")
    dst = FakeAgent(aux=False, actor_raises=True)
    tqc_io.load(dst, str(tmp_path), "ckpt")             # must not raise
    assert dst.actor.loaded is None                     # load was rejected → fresh
    assert dst.actor_optimizer.loaded is None           # stale optimizer skipped
    # checkpoint_actor mirrored from the fresh actor's state_dict (same object
    # the fake returns each call, so identity check avoids tensor-eq ambiguity).
    assert dst.checkpoint_actor.loaded is dst.actor.state_dict()


def test_critic_state_dim_mismatch_keeps_fresh_and_skips_optimizer(tmp_path):
    src = FakeAgent(aux=False)
    tqc_io.save(src, str(tmp_path), "ckpt")
    dst = FakeAgent(aux=False, critic_raises=True)
    tqc_io.load(dst, str(tmp_path), "ckpt")             # must not raise
    assert dst.critic.loaded is None                    # fresh critic
    assert dst.critic_target.loaded is None             # not reached → fresh
    assert dst.critic_optimizer.loaded is None          # stale optimizer skipped
    # Actor side unaffected (loaded normally).
    assert dst.actor.loaded is not None


def test_load_encoder_for_inference_baseline_true():
    agent = FakeAgent(aux=False)  # identity encoder → nothing to restore
    assert tqc_io.load_encoder_for_inference(agent, "/x/ckpt_actor.pth") is True


def test_load_encoder_for_inference_missing_file_false(tmp_path):
    agent = FakeAgent(aux=True)
    actor_path = str(tmp_path / "ckpt_actor.pth")  # no matching _encoder.pth
    assert tqc_io.load_encoder_for_inference(agent, actor_path) is False


def test_load_encoder_for_inference_present_true(tmp_path):
    src = FakeAgent(aux=True)
    tqc_io.save(src, str(tmp_path), "ckpt")  # writes ckpt_encoder.pth
    dst = FakeAgent(aux=True)
    actor_path = str(tmp_path / "ckpt_actor.pth")
    assert tqc_io.load_encoder_for_inference(dst, actor_path) is True
    assert dst.encoder.loaded is not None
