"""Curriculum resume state persistence (JSON progress + RNG snapshots).

Extracted from ``TrainTQCCurriculum._save_curriculum_state`` /
``_load_curriculum_state``. These are free functions that take the trainer
instance, so the on-disk format (``curriculum_state.json`` + ``rng_state.pkl`` +
``rng_torch.pt`` / ``rng_cuda.pt``) is byte-for-byte unchanged and the trainer
keeps thin ``_save_curriculum_state`` / ``_load_curriculum_state`` wrappers.

torch is imported lazily inside the RNG try-blocks (exactly the best-effort
semantics of the original code), so the JSON progress round-trip is importable
and unit-testable without torch / ROS.
"""

import os
import json
import pickle
import random

import numpy as np


def save_curriculum_state(trainer, global_t: int):
    """Write curriculum_state.json + RNG state files for full off-policy resume."""
    path = os.path.join(trainer.log_dir, "curriculum_state.json")
    with open(path, "w") as f:
        json.dump(
            {
                "stage":                  trainer._curriculum_stage,
                "stage_start_step":       trainer._stage_start_step,
                "stage_start_episode":    trainer._stage_start_ep,
                "consecutive_pass_count": trainer._consecutive_pass_count,
                "global_t":               global_t,
                "total_episodes":         trainer._total_episodes,
                "epoch":                  trainer._resume_epoch,
                "ep_timesteps":           trainer._partial_ep_timesteps,
                "ep_total_reward":        trainer._partial_ep_reward,
            },
            f,
            indent=2,
        )
    # RNG states — binary, saved alongside the JSON. Scope note: these are
    # the TRAINER process RNGs only (numpy/random/torch[/cuda]); the env node
    # runs in a separate process and is re-seeded deterministically on resume
    # via _reseed_env_for_resume() rather than snapshotted here.
    try:
        import torch
        with open(os.path.join(trainer.log_dir, "rng_state.pkl"), "wb") as f:
            pickle.dump(
                {"numpy": np.random.get_state(), "python": random.getstate()},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        torch.save(
            torch.get_rng_state(),
            os.path.join(trainer.log_dir, "rng_torch.pt"),
        )
        if torch.cuda.is_available():
            torch.save(
                torch.cuda.get_rng_state(),
                os.path.join(trainer.log_dir, "rng_cuda.pt"),
            )
    except Exception as _e:
        trainer.get_logger().warn(f"[Curriculum] RNG state save failed: {_e}")


def load_curriculum_state(trainer) -> bool:
    """Restore saved curriculum progress when resuming a run."""
    path = os.path.join(trainer.log_dir, "curriculum_state.json")
    if not os.path.isfile(path):
        trainer.get_logger().info(
            "[Curriculum] No curriculum_state.json found; resume will restart "
            "from stage 0 even though model weights were loaded."
        )
        return False
    try:
        with open(path, "r") as f:
            state = json.load(f)
        trainer._curriculum_stage = int(state.get("stage", 0))
        trainer._stage_start_step = int(state.get("stage_start_step", 0))
        trainer._stage_start_ep = int(state.get("stage_start_episode", 0))
        trainer._consecutive_pass_count = int(
            state.get("consecutive_pass_count", 0)
        )
        trainer._resume_global_t      = int(state.get("global_t", 0))
        trainer._total_episodes       = int(state.get("total_episodes", 0))
        trainer._resume_epoch         = int(state.get("epoch", 1))
        trainer._partial_ep_timesteps = int(state.get("ep_timesteps", 0))
        trainer._partial_ep_reward    = float(state.get("ep_total_reward", 0.0))
        trainer._last_global_t = trainer._resume_global_t
        trainer._resume_loaded = True
        trainer.get_logger().info(
            f"[Curriculum] Restored state from {path} | "
            f"stage={trainer._curriculum_stage} "
            f"global_t={trainer._resume_global_t} "
            f"episodes={trainer._total_episodes} "
            f"pass_streak={trainer._consecutive_pass_count}"
        )
        # Restore RNG states for reproducible off-policy resume
        try:
            import torch
            pkl = os.path.join(trainer.log_dir, "rng_state.pkl")
            if os.path.isfile(pkl):
                with open(pkl, "rb") as f:
                    rng = pickle.load(f)
                np.random.set_state(rng["numpy"])
                random.setstate(rng["python"])
            pt = os.path.join(trainer.log_dir, "rng_torch.pt")
            if os.path.isfile(pt):
                torch.set_rng_state(torch.load(pt))
            cuda_pt = os.path.join(trainer.log_dir, "rng_cuda.pt")
            if torch.cuda.is_available() and os.path.isfile(cuda_pt):
                torch.cuda.set_rng_state(torch.load(cuda_pt))
            trainer.get_logger().info("[Curriculum] RNG states restored.")
        except Exception as _e:
            trainer.get_logger().warn(
                f"[Curriculum] RNG state restore failed: {_e}"
            )
        return True
    except Exception as e:
        trainer.get_logger().warn(
            f"[Curriculum] Failed to load curriculum_state.json: {e}. "
            "Falling back to fresh curriculum progression."
        )
        return False
