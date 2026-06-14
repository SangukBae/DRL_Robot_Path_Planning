"""AUX_PRED formal-evaluation helpers for the curriculum trainer.\n\nExtracted from train_tqc_curriculum_agent.py. Builds privileged nearest-human labels, action sequences and per-episode aux accumulation used by the eval loop. Mixed into TrainTQCCurriculum; references shared trainer state via self (aux on/off gating is explicit inside each method)."""

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

from environment_interface import EnvServiceError
import curriculum_stage_logic
import curriculum_state_io
import seed_utils
from episode_metrics import EpisodeMetrics, PaperMetricsCSV
from aux_prediction_labels import AUX_WIRE_VERSION
import aux_ablation_logging as aux_log
from aux_eval_metrics import AuxEvalAccumulator, split_label
from curriculum_metrics import _LabelProximity


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

    def _aux_eval_episode(self, acc, states_list, labels_list, actions_list, map_type):
        """Run the aux head over one finished eval episode and add the batch to
        the accumulator (single-step OR action-conditioned, boundary-safe)."""
        if not self._aux_eval_on or not states_list:
            return
        states = np.asarray(states_list, dtype=np.float32)
        labels = np.asarray(labels_list, dtype=np.float64)
        if self._aux_eval_action_conditioned:
            fut, vlen = self._build_future_actions(actions_list)
            preds = self.rl_agent.aux_predict_eval(states, fut, vlen)
        else:
            preds = self.rl_agent.aux_predict_eval(states)
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
        if lab.shape[0] != exp.label_dim:
            raise RuntimeError(
                f"[AUX_PRED] label length mismatch: env sent {lab.shape[0]} but the "
                f"agent expects {exp.label_dim}. {hint}"
            )
