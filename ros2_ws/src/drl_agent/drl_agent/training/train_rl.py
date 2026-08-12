#!/usr/bin/env python3
"""Unified DRL trainer launcher — select a registered model/trainer by name.

Pure CLI/dispatch: MODEL_REGISTRY maps a flat model name to its canonical
``drl_agent.training.<...>`` module + class (see below); every entry,
including "tqc_curriculum", is imported lazily via importlib — no trainer's
code is duplicated or defined here. This is a SEPARATE launcher from the
profile system (``drl_agent.training.registry`` + ``ros2 run drl_agent
train_node.py -p profile:=...``), which resolves trainer+env pairs by
``(algorithm, trainer)`` instead of a flat model name; see that module's
docstring for how the two registries relate.

What this file provides:
  - A model-name registry (MODEL_REGISTRY).
  - --rl_model CLI flag (also accepts --model) and a ROS `rl_model` parameter;
    CLI > ROS param > default ("tqc_curriculum").
  - --list-models / --dry-run: resolve-only smoke tests that need no Gazebo,
    no simulation, and (for --list-models) no ROS node at all.
  - run_manifest.json gets rl_model / rl_model_module / rl_model_class fields
    added after the trainer's own __init__ writes it (best-effort, generic
    for any registered model).

Usage:
  ros2 run drl_agent train_rl.py --ros-args -p rl_model:=tqc_curriculum
  python3 -m drl_agent.training.train_rl --rl_model tqc_curriculum
  python3 -m drl_agent.training.train_rl --list-models
  python3 -m drl_agent.training.train_rl --dry-run --rl_model tqc_curriculum

The environment must be running the curriculum environment node
(``ros2 run drl_agent environment_curriculum_node.py``, or its canonical
module ``drl_agent.env.curriculum.environment_curriculum`` — NOT the
non-curriculum ``drl_agent.env.simulation.environment``) so that the
curriculum_stage / curriculum_num_stages parameters exist on /gym_node.
"""

import os
import sys
import json
import argparse
import importlib

import rclpy

from drl_agent.env.environment_interface import EnvServiceError


# --------------------------------------------------------------------------- #
#  Model registry                                                              #
# --------------------------------------------------------------------------- #
# Every entry names an EXISTING, canonical module ("tqc_curriculum" ->
# drl_agent.training.train_tqc_curriculum, the primary trainer; every other
# name -> its own drl_agent.training.baselines.<algo>_curriculum module) —
# imported lazily via importlib so choosing one model never pulls in every
# other trainer's deps (e.g. stable_baselines3 for a plain TQC run). Only add
# an entry here once its class/module genuinely exists — do not invent
# placeholders for algorithms that have no curriculum trainer yet.
MODEL_REGISTRY = {
    "tqc_curriculum": {
        "module": "drl_agent.training.train_tqc_curriculum",
        "class_name": "TrainTQCCurriculum",
        "description": "TQC + 10-stage curriculum (primary training path).",
    },
    "sac_curriculum": {
        "module": "drl_agent.training.baselines.sac_curriculum",
        "class_name": "TrainSACCurriculum",
        "description": "SAC curriculum baseline.",
    },
    "td7_curriculum": {
        "module": "drl_agent.training.baselines.td7_curriculum",
        "class_name": "TrainTD7Curriculum",
        "description": "TD7 curriculum baseline.",
    },
    "tqc_ieqn_curriculum": {
        "module": "drl_agent.training.baselines.tqc_ieqn_curriculum",
        "class_name": "TrainTQCIEQNCurriculum",
        "description": "TQC + IEQn (inequality constraint) curriculum variant.",
    },
    "a3c_curriculum": {
        "module": "drl_agent.training.baselines.a3c_curriculum",
        "class_name": "TrainA3CCurriculum",
        "description": "A3C curriculum baseline.",
    },
    "sb3_sac_curriculum": {
        "module": "drl_agent.training.baselines.sb3_sac_curriculum",
        "class_name": "TrainSB3SACCurriculum",
        "description": "Stable-Baselines3 SAC curriculum baseline.",
    },
    "sb3_td3_curriculum": {
        "module": "drl_agent.training.baselines.sb3_td3_curriculum",
        "class_name": "TrainSB3TD3Curriculum",
        "description": "Stable-Baselines3 TD3 curriculum baseline.",
    },
}

DEFAULT_RL_MODEL = "tqc_curriculum"


def list_models():
    """Sorted list of registered model names."""
    return sorted(MODEL_REGISTRY)


def resolve_trainer_class(model_name):
    """Return ``(trainer_class, registry_entry)`` for ``model_name``.

    Raises RuntimeError (with the list of available model names) for an
    unregistered name — this is the one error type callers should expect and
    handle, matching every other explicit config-validation failure in this
    codebase (e.g. train_tqc_base.py's missing-config-file errors).
    """
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        available = ", ".join(list_models())
        raise RuntimeError(
            f"Unknown rl_model '{model_name}'. Available models: {available}"
        )
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class_name"])
    return cls, entry


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="train_rl.py",
        description=(
            "Unified DRL trainer launcher: selects a registered curriculum "
            "trainer by name (--rl_model / ROS param 'rl_model') and runs it "
            "exactly as that model's own "
            "drl_agent.training.baselines.<model>_curriculum module would."
        ),
    )
    parser.add_argument(
        "--rl_model", "--model", dest="rl_model", default=None,
        help=(
            f"Model to train (default: '{DEFAULT_RL_MODEL}'). "
            f"Available: {', '.join(list_models())}"
        ),
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available rl_model names (with descriptions) and exit. "
             "No ROS node / Gazebo required.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve/import the selected trainer class and exit WITHOUT "
             "creating a ROS node, connecting to Gazebo, or starting "
             "training.",
    )
    return parser


def _resolve_rl_model_name(cli_rl_model):
    """CLI --rl_model > ROS ``rl_model`` parameter > DEFAULT_RL_MODEL.

    Reading the ROS parameter needs a live node (declare_parameter is the
    only way to observe a ``-p rl_model:=...`` override); a short-lived
    helper node does that lookup and is destroyed immediately after — the
    trainer node created right after this call still owns/declares its own
    parameters exactly as before (nothing here consumes or shadows them).
    """
    if cli_rl_model:
        return cli_rl_model
    helper = rclpy.create_node("train_rl_model_selector")
    try:
        helper.declare_parameter("rl_model", DEFAULT_RL_MODEL)
        value = helper.get_parameter("rl_model").get_parameter_value().string_value
        return value or DEFAULT_RL_MODEL
    finally:
        helper.destroy_node()


def _record_rl_model_in_manifest(node, model_name, entry):
    """Best-effort: add rl_model/rl_model_module/rl_model_class fields to this
    run's run_manifest.json (+ its timestamped copy, if the run uses the
    legacy layout) after the trainer's own __init__ has already written it.

    Generic across every registered model (does not require touching each
    sibling trainer file) and never raises — manifest bookkeeping must not be
    able to break a training run.
    """
    log_dir = getattr(node, "log_dir", None)
    if not log_dir:
        return
    fields = {
        "rl_model": model_name,
        "rl_model_module": entry["module"],
        "rl_model_class": entry["class_name"],
    }
    filenames = {"run_manifest.json"}
    run_tag = getattr(node, "_csv_run_tag", "") or ""
    if run_tag:
        filenames.add(f"run_manifest_{run_tag}.json")
    for fname in filenames:
        path = os.path.join(log_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest.update(fields)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass


def main(args=None):
    parser = _build_arg_parser()
    cli_args, _unknown = parser.parse_known_args(
        sys.argv[1:] if args is None else args
    )

    if cli_args.list_models:
        print("Available rl_model values:")
        for name in list_models():
            print(f"  - {name}: {MODEL_REGISTRY[name].get('description', '')}")
        return

    if cli_args.dry_run:
        model_name = cli_args.rl_model or DEFAULT_RL_MODEL
        try:
            cls, entry = resolve_trainer_class(model_name)
        except RuntimeError as e:
            print(f"[train_rl] {e}")
            sys.exit(1)
        print(
            f"[train_rl] dry-run OK — rl_model='{model_name}' resolves to "
            f"{entry['module']}.{entry['class_name']} ({cls})."
        )
        return

    rclpy.init(args=args)
    node = None
    try:
        model_name = _resolve_rl_model_name(cli_args.rl_model)
        trainer_cls, entry = resolve_trainer_class(model_name)
        print(
            f"[train_rl] rl_model='{model_name}' -> "
            f"{entry['module']}.{entry['class_name']}"
        )
        node = trainer_cls()
        _record_rl_model_in_manifest(node, model_name, entry)
        node.train_online()
        while rclpy.ok() and not node.done_training:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\n[train_rl] Training interrupted by user.")
        if node is not None:
            try:
                node._save_curriculum_state(getattr(node, "_last_global_t", 0))
            except Exception:
                pass
    except EnvServiceError as e:
        # Environment service died (Gazebo/env node hung or crashed). Fail-fast:
        # save a checkpoint + curriculum state so the run is resumable, then stop
        # cleanly instead of hanging or losing progress.
        print(f"\n[train_rl] Environment service failure: {e}")
        if node is not None:
            try:
                node.save_models(node.pytorch_models_dir, node.file_name)
                node._save_curriculum_state(getattr(node, "_last_global_t", 0))
                print("[train_rl] Saved checkpoint after service failure — "
                      "run is resumable.")
            except Exception as se:
                print(f"[train_rl] Checkpoint save after service failure failed: {se}")
    except RuntimeError as e:
        # Unknown --rl_model / ROS `rl_model` param value.
        print(f"[train_rl] {e}")
    except Exception as e:
        print(f"[train_rl] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Flush any buffered TensorBoard/JSON metrics on EVERY exit path (normal
        # completion, KeyboardInterrupt, EnvServiceError, any other exception) --
        # mirrors train_tqc_curriculum.py's own main(); generic here (getattr)
        # since this launcher dispatches to whichever agent the resolved model
        # actually built, not just TQC.
        if node is not None and getattr(node, "rl_agent", None) is not None:
            try:
                node.rl_agent.flush_logs()
            except Exception as fe:
                print(f"[train_rl] flush_logs() failed during shutdown: {fe}")
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
