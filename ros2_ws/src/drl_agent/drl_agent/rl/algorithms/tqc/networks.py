#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TQC hyperparameter pre-processing + the shared-trunk critic optimizer.

Extracted unchanged from rl/algorithms/tqc/agent.py: prep_hyperparameters
maps activation-function strings to callables and injects defaults for any
missing hyperparameter key; _make_critic_optimizer builds the single Adam
instance over the critic + encoder + aux/action-risk/CF heads (optionally
with per-component param_groups — see its own docstring), used both at
construction and to rebuild the optimizer when a resumed checkpoint's
architecture doesn't match the current one.
"""

import torch
import torch.nn.functional as F


class NetworksMixin:
    """Hyperparameter pre-processing + shared-trunk optimizer construction.

    Mixed into Agent (rl/algorithms/tqc/agent.py); _make_critic_optimizer
    reads Agent instance state via ``self`` exactly as it did before
    extraction. prep_hyperparameters is a pure staticmethod.
    """

    @staticmethod
    def prep_hyperparameters(hyperparameters):
        """Pre-process hyperparameters: 문자열 활성화 → 함수, 기본값 주입"""
        hp = dict(hyperparameters or {})
        activation_functions = {
            "elu": F.elu,
            "relu": F.relu,
        }

        # 활성화 함수 문자열이면 함수로 매핑
        if "actor_activ" in hp and isinstance(hp["actor_activ"], str):
            hp["actor_activ"] = activation_functions.get(hp["actor_activ"].lower(), F.relu)
        if "critic_activ" in hp and isinstance(hp["critic_activ"], str):
            hp["critic_activ"] = activation_functions.get(hp["critic_activ"].lower(), F.elu)

        # 자주 쓰는 기본값(누락 방지)
        hp.setdefault("discount", 0.99)
        hp.setdefault("batch_size", 256)
        hp.setdefault("buffer_size", 1_000_000)
        hp.setdefault("actor_lr", 3e-4)
        hp.setdefault("critic_lr", 3e-4)
        hp.setdefault("n_quantiles", 25)
        hp.setdefault("n_critics", 2)
        hp.setdefault("top_quantiles_to_drop_per_net", 2)
        hp.setdefault("tau", 0.005)
        hp.setdefault("target_update_interval", 1)
        hp.setdefault("ent_coef", "auto")
        hp.setdefault("ent_coef_lr", 3e-4)
        hp.setdefault("actor_hdim", 256)
        hp.setdefault("critic_hdim", 256)
        hp.setdefault("reset_weight", 0.9)
        hp.setdefault("steps_before_checkpointing", 40000)
        hp.setdefault("max_eps_when_checkpointing", 50)
        # prioritized 관련 키는 사용 시에만 읽음

        return hp

    def _make_critic_optimizer(self):
        """AUX_PRED: build the Adam optimizer over the shared trunk (critic +
        encoder + aux head).  Used at construction AND to reset the moments when
        an aux-head architecture change on resume makes the saved moments stale.

        STAGE 8 (isolated experimental feature, default OFF): when
        optimizer_groups.enabled is set, builds ONE Adam instance with
        SEPARATE param_groups per component (critic/encoder/aux_head/
        action_risk_head) instead of one flat parameter list -- each group
        can have its own LR (e.g. a smaller encoder_lr than the head LRs),
        while still being a single optimizer (one .zero_grad()/.step() pair,
        unchanged call sites in train()). Any component whose LR isn't
        explicitly configured falls back to critic_lr, so enabling the
        feature WITHOUT setting different LRs is behaviorally a no-op
        (identical effective LR everywhere, just split across groups)."""
        og = dict(self.hyperparameters.get("optimizer_groups", {}) or {})
        use_groups = bool(og.get("enabled", False))

        components = [("critic", list(self.critic.parameters()))]
        if self.encoder.has_params():
            components.append(("encoder", list(self.encoder.parameters())))
        if self.aux_head is not None:
            components.append(("aux_head", list(self.aux_head.parameters())))
        # AUX_PRED (v2): the temporal encoder is part of the aux trunk; its grads
        # flow with critic + aux loss (never the actor) -> same optimizer group.
        if getattr(self, "temporal_encoder", None) is not None:
            components.append(("aux_head", list(self.temporal_encoder.parameters())))
        # PHASE2: the Action-Risk Head trains from its own supervised loss added
        # into the SAME trunk loss (critic + beta_aux*aux + risk_beta*action_risk)
        # -> same optimizer group. The TARGET copy is never optimized directly
        # (polyak-updated only, like critic_target/encoder_target).
        if getattr(self, "action_risk_head", None) is not None:
            components.append(("action_risk_head", list(self.action_risk_head.parameters())))
        if getattr(self, "counterfactual_risk_head", None) is not None:
            components.append((
                "counterfactual_risk_head",
                list(self.counterfactual_risk_head.parameters())))

        if not use_groups:
            trunk_params = [p for _, params in components for p in params]
            return torch.optim.Adam(trunk_params, lr=self.critic_lr)

        lr_key = {"critic": "critic_lr", "encoder": "encoder_lr",
                  "aux_head": "aux_head_lr", "action_risk_head": "action_risk_head_lr",
                  "counterfactual_risk_head": "counterfactual_risk_head_lr"}
        param_groups = []
        merged = {}
        for name, params in components:
            merged.setdefault(name, []).extend(params)
        for name, params in merged.items():
            lr = float(og.get(lr_key[name], self.critic_lr))
            param_groups.append({"params": params, "lr": lr})
        return torch.optim.Adam(param_groups, lr=self.critic_lr)

