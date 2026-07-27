"""Curriculum evaluation metric accumulators (pure, ROS/torch-free).

Extracted from ``train_tqc_curriculum_agent.py``. Holds the per-episode
human-proximity tracker used to derive H-Coll and the TRUE (human) Personal-Space
Compliance from the env's privileged human-distance labels. Pure Python (only
``math``), so it is unit-testable in isolation; the trainer imports
``_LabelProximity`` from here.
"""

import math


class _LabelProximity:
    """Per-episode human-proximity tracker from privileged aux labels.

    H-Coll and the TRUE (human) PSC are derived from the env's privileged
    human-distance labels, NOT from the agent's aux head.  So they are available
    for an aux-OFF agent baseline too, as long as the ENV emits labels
    (aux_prediction.enabled on the env side).  When the env emits no labels both
    metrics are reported BLANK (None) — never a misleading 0.
    """

    def __init__(self, personal_space_m: float, h_coll_radius_m: float):
        self.ps = float(personal_space_m)
        self.hcr = float(h_coll_radius_m)
        self.reset()

    def reset(self):
        self.min_m = float("inf")
        self.steps = 0
        self.intrusions = 0

    def add_dist(self, dist_m):
        """Fold one step's nearest-human distance [m] (ignored if not finite)."""
        if dist_m is None or not math.isfinite(dist_m):
            return
        self.steps += 1
        if dist_m < self.min_m:
            self.min_m = dist_m
        if dist_m < self.ps:
            self.intrusions += 1

    @property
    def available(self) -> bool:
        return self.steps > 0

    def psc(self):
        """Fraction of label-available steps that respected personal space.
        None when no label was seen (env labels off)."""
        return (1.0 - self.intrusions / self.steps) if self.steps > 0 else None

    def h_coll(self, collision: bool):
        """1 if a collision episode ended with a human inside h_coll_radius.
        None when no label was seen (env labels off)."""
        if self.steps == 0:
            return None
        return int(bool(collision) and self.min_m < self.hcr)


# Public alias (the underscore name is kept for the trainer's existing call sites).
LabelProximity = _LabelProximity
