"""AUX_PRED formal-evaluation helpers for the curriculum trainer.\n\nExtracted from drl_agent.training.train_tqc_curriculum. Builds privileged nearest-human labels, action sequences and per-episode aux accumulation used by the eval loop. Mixed into TrainTQCCurriculum; references shared trainer state via self (aux on/off gating is explicit inside each method)."""

import os
import sys
import csv
import math
import time
import json
import pickle
import random
from datetime import datetime

import numpy as np
import torch

from drl_agent.env.environment_interface import EnvServiceError
import drl_agent.training.curriculum.stage_logic as curriculum_stage_logic
import drl_agent.training.curriculum.state_io as curriculum_state_io
import drl_agent.common.seed_utils as seed_utils
from drl_agent.training.episode_metrics import EpisodeMetrics, PaperMetricsCSV
from drl_agent.env.observation.aux_prediction_labels import AUX_WIRE_VERSION
import drl_agent.training.aux_ablation_logging as aux_log
from drl_agent.training.aux_eval_metrics import AuxEvalAccumulator, split_label
from drl_agent.training.curriculum.metrics import _LabelProximity


class CurriculumAuxEvalMixin:
    def _human_min_dist_m_from_label(self, label):
        """Privileged nearest-human distance [m] from an env label: the closest
        future approach over all horizons (min over the min_dist_norm block * D_c).

        Depends ONLY on the env emitting labels (self._h_coll_available), NOT on
        the agent aux head — so H-Coll / true PSC work for an aux-OFF agent
        baseline too. Returns None when no usable label is available."""
        if not self._h_coll_available or label is None or self._label_H <= 0:
            return None
        arr = np.asarray(label, dtype=np.float64).ravel()
        risk_dim = self._label_H * self._label_K
        if arr.shape[0] < risk_dim + self._label_H:
            return None
        md_norm = arr[risk_dim:risk_dim + self._label_H]
        return float(np.min(md_norm)) * self._label_Dc

    def _build_future_actions(self, actions_list):
        """Boundary-safe future-action tensor for action-conditioned aux eval.

        actions_list[i] = a_i (action taken from s_i), all within ONE episode.
        For step i returns [a_i, .., a_{i+K-1}] zero-padded past the episode end,
        and valid_len[i] = min(K, T - i) (>= 1) — NEVER reading across the
        boundary, identical to the training-time alignment.
        """
        K = max(1, int(self._aux_eval_ac_steps))
        T = len(actions_list)
        adim = len(actions_list[0]) if T > 0 else 0
        fut = np.zeros((T, K, adim), dtype=np.float32)
        vlen = np.ones((T,), dtype=np.int64)
        for i in range(T):
            n = min(K, T - i)
            vlen[i] = max(1, n)
            for j in range(n):
                fut[i, j] = np.asarray(actions_list[i + j], dtype=np.float32)
        return fut, vlen

    def _build_state_history(self, states_list):
        """Boundary-safe REVERSE-time state window for temporal aux eval.

        states_list[i] = s_i, all within ONE eval episode. For step i returns
        [s_i, s_{i-1}, .., s_{i-H+1}] zero-padded past the episode start, and
        hist_valid_len[i] = min(H, i+1) (>= 1) — the exact training-time
        backward alignment (LAP.get_last_state_history), kept inside one episode
        so it never splices across a boundary."""
        H = max(1, int(getattr(self, "_aux_eval_hist_len", 4)))
        T = len(states_list)
        sdim = len(states_list[0]) if T > 0 else 0
        hist = np.zeros((T, H, sdim), dtype=np.float32)
        vlen = np.ones((T,), dtype=np.int64)
        for i in range(T):
            n = min(H, i + 1)
            vlen[i] = max(1, n)
            for k in range(n):
                hist[i, k] = np.asarray(states_list[i - k], dtype=np.float32)
        return hist, vlen

    def _aux_eval_episode(self, acc, states_list, labels_list, actions_list, map_type):
        """Run the aux head over one finished eval episode and add the batch to
        the accumulator (single-step OR action-conditioned, boundary-safe)."""
        if not self._aux_eval_on or not states_list:
            return
        states = np.asarray(states_list, dtype=np.float32)
        labels = np.asarray(labels_list, dtype=np.float64)
        # AUX_PRED (v2): faithful backward state history for the temporal head
        # (None-safe on a non-temporal head -> aux_predict_eval ignores it).
        sh, shl = (self._build_state_history(states_list)
                   if getattr(self, "_aux_eval_temporal", False) else (None, None))
        if self._aux_eval_action_conditioned:
            fut, vlen = self._build_future_actions(actions_list)
            preds = self.rl_agent.aux_predict_eval(
                states, fut, vlen, state_history=sh, hist_valid_len=shl)
        else:
            preds = self.rl_agent.aux_predict_eval(
                states, state_history=sh, hist_valid_len=shl)
        if preds is None:
            return
        risk_gt, md_gt = split_label(labels, self._aux_eval_H, self._aux_eval_K)
        risk_pred = preds["risk_map"]
        md_pred = preds.get("min_dist")   # None when the head has no min-dist out
        acc.add_batch(
            risk_pred.reshape(len(states_list), -1),
            risk_gt.reshape(len(states_list), -1),
            md_pred,
            md_gt,
            map_type=map_type or "na",
        )

    def _check_aux_label_contract(self):
        """AUX_PRED: fail-fast on any agent/env aux mismatch (STRUCTURAL).

        The auxiliary label geometry lives in two configs (agent-side
        hyperparameters_tqc.yaml and env-side environment_curriculum.yaml).
        Rather than clipping or trusting a total-length match (different
        num_sectors / horizons_sec can yield the same length), the env sends a
        geometry header with the label; here we compare that header field-by-
        field against the agent's aux config and raise immediately on ANY
        inconsistency (missing label, wrong num_sectors, wrong number of
        horizons, different horizon values, or mismatched label length).
        """
        if not self._aux_enabled:
            return
        exp = getattr(self.rl_agent, "aux_cfg", None)
        lab = self.last_aux_label
        meta = self.last_aux_meta

        if lab is None:
            raise RuntimeError(
                "[AUX_PRED] agent aux_prediction.enabled=true but the environment "
                "appended no future-risk label. Set aux_prediction.enabled=true in "
                "environment_curriculum.yaml (and rebuild) so the env emits labels."
            )
        if meta is None:
            raise RuntimeError(
                "[AUX_PRED] env label is missing its geometry header (malformed or "
                "version-incompatible wire format). Rebuild so env and agent share "
                "the same aux_prediction_labels module."
            )
        if meta.get("version") != AUX_WIRE_VERSION:
            raise RuntimeError(
                f"[AUX_PRED] wire-format version mismatch: env="
                f"{meta.get('version')} but agent expects {AUX_WIRE_VERSION}. "
                "Rebuild so env and agent share the same aux_prediction_labels "
                "module (the wire layout changed incompatibly)."
            )
        if exp is None:
            return

        hint = ("Make num_sectors / horizons_sec IDENTICAL in "
                "hyperparameters_tqc.yaml and environment_curriculum.yaml, then rebuild.")
        if meta["num_sectors"] != exp.num_sectors:
            raise RuntimeError(
                f"[AUX_PRED] num_sectors mismatch: env={meta['num_sectors']} "
                f"agent={exp.num_sectors}. {hint}"
            )
        if meta["num_horizons"] != exp.num_horizons:
            raise RuntimeError(
                f"[AUX_PRED] horizon count mismatch: env={meta['num_horizons']} "
                f"agent={exp.num_horizons}. {hint}"
            )
        env_h = list(meta["horizons_sec"])
        agent_h = list(exp.horizons_sec)
        if any(abs(float(a) - float(b)) > 1e-3 for a, b in zip(env_h, agent_h)):
            raise RuntimeError(
                f"[AUX_PRED] horizon values mismatch: env={env_h} agent={agent_h}. {hint}"
            )
        # v2 append-only blocks: the env's ttc / hazard sizes must match the
        # agent's enabled heads exactly (a mismatch means the configs disagree on
        # which direct heads are on, or on hazard_sector_bins).
        ttc_hint = ("Set ttc_head_enabled / hazard_sector_head_enabled / "
                    "hazard_sector_bins IDENTICALLY in hyperparameters_tqc.yaml "
                    "and environment_curriculum.yaml, then rebuild.")
        if int(meta.get("ttc_len", 0)) != int(exp.ttc_len):
            raise RuntimeError(
                f"[AUX_PRED] TTC block mismatch: env ttc_len={meta.get('ttc_len', 0)} "
                f"agent ttc_len={exp.ttc_len}. {ttc_hint}"
            )
        if int(meta.get("hazard_len", 0)) != int(exp.hazard_len):
            raise RuntimeError(
                f"[AUX_PRED] hazard block mismatch: env hazard_len="
                f"{meta.get('hazard_len', 0)} agent hazard_len={exp.hazard_len}. "
                f"{ttc_hint}"
            )
        if lab.shape[0] != exp.label_dim:
            raise RuntimeError(
                f"[AUX_PRED] label length mismatch: env sent {lab.shape[0]} but the "
                f"agent expects {exp.label_dim}. {hint}"
            )
