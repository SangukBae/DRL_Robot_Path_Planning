#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TensorBoard/JSON metric logging for the TQC agent.

Extracted unchanged from rl/algorithms/tqc/agent.py: buffers per-step JSON
records (flushed in one open/write-all/close batch, not one open/close per
record) and flushes both the JSON buffer and the TensorBoard SummaryWriter
on demand — called from every training-loop exit path (normal completion,
KeyboardInterrupt, environment-service failure, any other exception) so a
run that stops between flush intervals never silently loses its most
recent metrics.
"""

import os
import time
import json

import numpy as np


class MetricsMixin:
    """Buffered JSON + TensorBoard metric logging.

    Mixed into Agent (rl/algorithms/tqc/agent.py); every method here reads/
    writes Agent instance state via ``self`` exactly as it did before
    extraction.
    """

    # JSON 라인 기록 헬퍼
    def _json_log(self, step: int, **metrics):
        path = getattr(self, "json_log_path", None)
        if not path:
            return

        rec = {"step": int(step), "time": float(time.time())}
        # AUX_ABLATION: stamp run identity on every record so tqc_metrics.json can
        # be grouped by seed / aux on-off without a separate join.  aux_enabled /
        # aux_version are always present (null-safe); seed only when known.
        if getattr(self, "run_seed", None) is not None:
            rec["seed"] = int(self.run_seed)
        rec["aux_enabled"] = int(bool(getattr(self, "aux_enabled", False)))
        rec["aux_version"] = int(getattr(self.aux_cfg, "version", 0)) if getattr(self, "aux_cfg", None) else 0
        for k, v in metrics.items():
            try:
                val = float(v)
                if np.isfinite(val):
                    rec[k] = val
            except Exception:
                continue

        self._json_buffer.append(rec)
        if len(self._json_buffer) >= self.json_flush_interval:
            self._flush_json_buffer()

    def _flush_json_buffer(self):
        """STAGE 6: physically write every buffered JSON record in ONE
        open/write-all/close instead of one open/close per record."""
        if not self._json_buffer:
            return
        path = self.json_log_path
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for rec in self._json_buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._json_buffer.clear()

    def flush_logs(self):
        """STAGE 6: flush any buffered JSON records + the TensorBoard writer
        on demand. Call on ALL exit paths (normal completion, KeyboardInterrupt,
        environment-service failure, any other exception, checkpoint-on-
        failure) so a run that stops between flush intervals never silently
        loses its most recent metrics."""
        self._flush_json_buffer()
        if self.writer:
            self.writer.flush()

