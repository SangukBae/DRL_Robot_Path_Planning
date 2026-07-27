"""CheckpointManager — one place that KNOWS the on-disk checkpoint layout.

Today it only *inspects* (locate the latest checkpoint prefix, replay buffer,
curriculum/RNG state, across both the new ``runtime/experiments/<run_id>/`` and
the legacy ``runtime/tqc/seed_<seed>/`` layouts) so resume decisions can be
validated and logged BEFORE a trainer starts. Actual tensor save/load stays in
``tqc_io.py`` / ``curriculum_state_io.py`` (imported by the trainers) — those
migrate here incrementally.

File conventions (source of truth: tqc_io.save / curriculum_state_io):
  models:     <models_dir>/<base>_seed_<seed>_<YYYYMMDD>_actor.pth (+ _critic,
              optimizers, aux/temporal/action-risk heads, _replay_buffer*)
              — ``*_checkpoint_actor.pth`` files are MID-EPISODE snapshots and
              never count as a resumable checkpoint prefix.
  logs:       <logs_dir>/curriculum_state.json, rng_state.pkl

Pure Python (glob/os only) — no ROS, no torch.
"""

import glob
import os
from dataclasses import dataclass, field

from ...training import run_layout


@dataclass
class ResumeState:
    """Everything a resume would load, with explicit paths (or None).

    ``resumable`` means only "an actor checkpoint exists" — that is the SAME
    criterion ``train_tqc_base._find_latest_prefix`` uses to trigger the
    legacy trainer's automatic resume, so it is NOT sufficient to promise a
    full off-policy resume: the replay buffer, curriculum progress, and RNG
    snapshot are each optional/independent files that may be missing (the
    legacy loaders degrade gracefully — see ``tqc_io.load`` /
    ``curriculum_state_io.load_curriculum_state`` — rather than failing).
    Callers that need to know whether resume would be FULL must check the
    ``has_replay_buffer`` / ``has_curriculum_state`` properties explicitly
    (``ConfigValidator`` does this and turns them into errors for curriculum
    profiles, where silently losing progress/replay data is never desired).
    """

    resumable: bool = False
    layout_kind: str = "none"          # "new" | "legacy" | "none"
    run_dir: str = ""
    checkpoint_prefix: str = ""        # filename prefix inside models_dir
    actor_path: str = ""
    replay_buffer_paths: list = field(default_factory=list)
    curriculum_state_path: str = ""
    rng_state_path: str = ""
    searched_roots: list = field(default_factory=list)

    @property
    def has_replay_buffer(self) -> bool:
        return bool(self.replay_buffer_paths)

    @property
    def has_curriculum_state(self) -> bool:
        return bool(self.curriculum_state_path)

    @property
    def has_rng_state(self) -> bool:
        return bool(self.rng_state_path)

    def as_info(self) -> dict:
        """Flat dict for logging / ValidationReport.info."""
        return {
            "resumable": self.resumable,
            "layout": self.layout_kind,
            "run_dir": self.run_dir or "(none)",
            "checkpoint_prefix": self.checkpoint_prefix or "(none)",
            "actor": self.actor_path or "(none)",
            "replay_buffer": (self.replay_buffer_paths[0]
                              if self.replay_buffer_paths else "(none)"),
            "has_replay_buffer": self.has_replay_buffer,
            "curriculum_state": self.curriculum_state_path or "(none)",
            "has_curriculum_state": self.has_curriculum_state,
            "rng_state": self.rng_state_path or "(none)",
            "has_rng_state": self.has_rng_state,
        }


def latest_checkpoint_prefix(models_dir: str, base_file_name: str, seed: int) -> str:
    """Most-recently-modified non-mid-episode checkpoint prefix, or "".

    Mirrors train_tqc_base._find_latest_prefix so the validator reports the
    SAME checkpoint the trainer would actually load.
    """
    pat = os.path.join(models_dir, f"{base_file_name}_seed_{int(seed)}_*_actor.pth")
    cands = [p for p in glob.glob(pat)
             if not os.path.basename(p).endswith("_checkpoint_actor.pth")]
    if not cands:
        return ""
    cands.sort(key=os.path.getmtime, reverse=True)
    return os.path.basename(cands[0])[: -len("_actor.pth")]


class CheckpointManager:
    """Locate and describe resumable state for a ``(base_file_name, seed)``."""

    def __init__(self, package_root: str = ""):
        if not package_root:
            from ...common import compat
            package_root = compat.package_source_root()
        self.package_root = package_root

    # ------------------------------------------------------------------ #

    def describe_resume_state(self, base_file_name: str, seed: int) -> ResumeState:
        """Inspect (never modify) what ``load_model=true`` would resume from.

        Uses the same precedence as ``run_layout.resolve_run_layout``:
        newest new-structure run dir first, then the legacy per-seed dir.
        """
        state = ResumeState()
        root = self.package_root
        state.searched_roots = [
            run_layout.experiments_root(root),
            run_layout.legacy_run_dir(root, seed),
        ]

        new_dir = run_layout.find_latest_new_run_dir(root, base_file_name, seed)
        if new_dir:
            layout = run_layout.new_layout(new_dir, is_fresh=False)
            self._fill_from_layout(state, layout, "new", base_file_name, seed)
            return state

        if run_layout.legacy_has_checkpoint(root, base_file_name, seed):
            layout = run_layout.legacy_layout(run_layout.legacy_run_dir(root, seed))
            self._fill_from_layout(state, layout, "legacy", base_file_name, seed)
            return state

        return state  # not resumable

    def _fill_from_layout(self, state, layout, kind, base_file_name, seed):
        models_dir = layout["models_dir"]
        logs_dir = layout["logs_dir"]
        prefix = latest_checkpoint_prefix(models_dir, base_file_name, seed)
        if not prefix:
            return
        state.resumable = True
        state.layout_kind = kind
        state.run_dir = layout["run_dir"]
        state.checkpoint_prefix = prefix
        state.actor_path = os.path.join(models_dir, f"{prefix}_actor.pth")
        state.replay_buffer_paths = sorted(
            glob.glob(os.path.join(models_dir, f"{prefix}_replay_buffer*")))
        cur = os.path.join(logs_dir, "curriculum_state.json")
        if os.path.isfile(cur):
            state.curriculum_state_path = cur
        rng = os.path.join(logs_dir, "rng_state.pkl")
        if os.path.isfile(rng):
            state.rng_state_path = rng
