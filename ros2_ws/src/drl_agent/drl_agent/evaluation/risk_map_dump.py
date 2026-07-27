"""Phase 1b: per-step risk-map dump writer for evaluation-time paper figures.

Pure Python, no ROS/torch dependency (importable and unit-testable standalone).
Writes one JSON object per line (JSONL) -- a plain CSV isn't a good fit here
since each record carries variable-length ARRAYS (the risk_map is H*K floats),
not just scalars, unlike the rest of this codebase's scalar-column CSVs.

Each record captures, for one eval step:
  * episode / step identifiers
  * robot GT pose (x, y, yaw)
  * the selected waypoint (r, theta) and its risk-map sector index
  * the GT (label-side, privileged) risk_map / min_dist -- None when aux is
    disabled at the env (so the wire carries no label at all)
  * the aux HEAD's PREDICTED risk_map / min_dist -- None when aux is disabled
    at the policy (so there is nothing to predict), kept separate from the GT
    fields above per the task's "GT risk_map과 aux predicted risk_map을 구분해서
    저장" requirement
  * a lightweight human-state summary (x, y, yaw, v, mode) list

Never raises on a missing/None field -- every field defaults to ``None`` and
is written as JSON ``null``, so a partial record (e.g. aux entirely off) is
still a valid, parseable line.
"""

import json
import math
import os


def sector_index_for_theta(theta: float, num_sectors: int) -> int:
    """Angle (radians, robot frame) -> risk_map sector index.

    Matches aux_prediction_labels.py's risk_map sector convention EXACTLY
    (NOT the front-centered hazard-sector convention): straight ahead
    (theta=0) maps to the MIDDLE bin (num_sectors // 2), not bin 0. Reusing
    this exact formula keeps the dumped sector_index directly comparable to
    the risk_map/pred_risk_map arrays' K axis.
    """
    if num_sectors <= 0:
        return 0
    ang = (theta + math.pi) % (2.0 * math.pi) - math.pi
    k = int((ang + math.pi) / (2.0 * math.pi) * num_sectors)
    return max(0, min(num_sectors - 1, k))


def human_state_summary(human_states):
    """Build the lightweight {x,y,yaw,v,mode} list this module writes, from
    either a dict-of-dicts (env.human_states shape) or an already-list-shaped
    iterable of dicts (e.g. parsed from step_debug_state JSON)."""
    if human_states is None:
        return []
    values = human_states.values() if hasattr(human_states, "values") else human_states
    return [
        {
            "x": float(s.get("x", 0.0)),
            "y": float(s.get("y", 0.0)),
            "yaw": float(s.get("yaw", 0.0)),
            "v": float(s.get("v", 0.0)),
            "mode": str(s.get("mode", "")),
        }
        for s in values
    ]


class RiskMapDumpWriter:
    """Append-only JSONL writer for per-step risk-map dump records."""

    def __init__(self, path, mode="w"):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._f = open(path, mode)

    def write_step(
        self,
        *,
        episode: int,
        step: int,
        robot_pose=None,           # (x, y, yaw) or None
        action_r=None,
        action_theta=None,
        num_sectors=None,
        gt_risk_map=None,          # list[float] (H*K) or None
        gt_min_dist=None,          # list[float] (H) or None
        pred_risk_map=None,        # list[float] (H*K) or None
        pred_min_dist=None,        # list[float] (H) or None
        humans=None,               # list[{x,y,yaw,v,mode}] or None
        extra=None,
    ):
        sector_index = None
        if action_theta is not None and num_sectors:
            sector_index = sector_index_for_theta(float(action_theta), int(num_sectors))
        record = {
            "episode": int(episode),
            "step": int(step),
            "robot_x": None if robot_pose is None else float(robot_pose[0]),
            "robot_y": None if robot_pose is None else float(robot_pose[1]),
            "robot_yaw": None if robot_pose is None else float(robot_pose[2]),
            "action_r": None if action_r is None else float(action_r),
            "action_theta": None if action_theta is None else float(action_theta),
            "num_sectors": None if num_sectors is None else int(num_sectors),
            "sector_index": sector_index,
            "gt_risk_map": None if gt_risk_map is None else [float(v) for v in gt_risk_map],
            "gt_min_dist": None if gt_min_dist is None else [float(v) for v in gt_min_dist],
            "pred_risk_map": (
                None if pred_risk_map is None else [float(v) for v in pred_risk_map]
            ),
            "pred_min_dist": (
                None if pred_min_dist is None else [float(v) for v in pred_min_dist]
            ),
            "humans": humans if humans is not None else [],
        }
        if extra:
            record.update(extra)
        self._f.write(json.dumps(record) + "\n")
        return record

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def read_dump(path):
    """Read a JSONL dump back into a list of dicts (helper for post-hoc
    analysis / tests)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
