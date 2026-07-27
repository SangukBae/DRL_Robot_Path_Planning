"""Migration invariants for the scripts/ -> drl_agent/ package restructure.

Locks the two contracts every migrated module must honour:

  1. the CANONICAL package path imports on its own (no flat scripts dir
     needed on sys.path beyond what conftest provides), and
  2. the legacy bare-name shim aliases the SAME module object (so legacy
     imports, monkeypatching and isinstance checks all keep working).

Grouped by dependency weight: pure (always run), torch-gated, ROS-gated —
mirroring the suite's existing skip conventions so the whole matrix runs
inside the built docker workspace while the host still checks the pure set.
"""

import importlib

import pytest

try:
    import torch  # noqa: F401
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

try:
    import rclpy  # noqa: F401
    from drl_agent_interfaces.srv import Step  # noqa: F401
    _HAVE_ROS = True
except Exception:  # pragma: no cover
    _HAVE_ROS = False


# (bare legacy name, canonical module path)
PURE_PAIRS = [
    ("run_layout", "drl_agent.training.run_layout"),
    ("config_paths", "drl_agent.config.paths"),
    ("geometry_utils", "drl_agent.common.geometry_utils"),
    ("seed_utils", "drl_agent.common.seed_utils"),
    ("file_manager", "drl_agent.common.file_manager"),
    ("pure_pursuit", "drl_agent.common.pure_pursuit"),
    ("episode_metrics", "drl_agent.training.episode_metrics"),
    ("aux_ablation_logging", "drl_agent.training.aux_ablation_logging"),
    ("aux_eval_metrics", "drl_agent.training.aux_eval_metrics"),
    ("dynamic_avoidance_log", "drl_agent.training.dynamic_avoidance_log"),
    ("risk_map_dump", "drl_agent.evaluation.risk_map_dump"),
    ("sim_validation", "drl_agent.evaluation.sim_validation"),
    ("curriculum_stage_logic", "drl_agent.training.curriculum.stage_logic"),
    ("curriculum_metrics", "drl_agent.training.curriculum.metrics"),
    ("curriculum_state_io", "drl_agent.training.curriculum.state_io"),
    ("aux_prediction_labels", "drl_agent.env.observation.aux_prediction_labels"),
    ("obs_time_context", "drl_agent.env.observation.obs_time_context"),
    ("reward_calculator", "drl_agent.env.rewards.reward_calculator"),
    ("collision_checker", "drl_agent.env.simulation.collision_checker"),
    ("localization_noise", "drl_agent.env.simulation.localization_noise"),
    ("map_catalog", "drl_agent.env.simulation.map_catalog"),
    ("map_layout_registry", "drl_agent.env.simulation.map_layout_registry"),
]

TORCH_PAIRS = [
    ("tqc_networks", "drl_agent.rl.networks.tqc"),
    ("action_risk_head", "drl_agent.rl.networks.action_risk_head"),
    ("aux_prediction", "drl_agent.rl.networks.aux_prediction"),
    ("aux_prediction_losses", "drl_agent.rl.networks.aux_losses"),
    ("aux_prediction_temporal", "drl_agent.rl.networks.aux_temporal"),
    ("buffer", "drl_agent.rl.replay.buffer"),
    ("tqc_io", "drl_agent.rl.checkpointing.tqc_io"),
    ("tqc_agent", "drl_agent.rl.algorithms.tqc.agent"),
    ("sac_agent", "drl_agent.rl.algorithms.sac.agent"),
    ("td7_agent", "drl_agent.rl.algorithms.td7.agent"),
    ("a3c_agent", "drl_agent.rl.algorithms.a3c.agent"),
    ("tqc_ieqn_agent", "drl_agent.rl.algorithms.tqc_ieqn.agent"),
    ("sb3_sac_agent", "drl_agent.rl.algorithms.sb3.sac"),
    ("sb3_td3_agent", "drl_agent.rl.algorithms.sb3.td3"),
]

ROS_PAIRS = [
    ("point_cloud2", "drl_agent.common.point_cloud2"),
    ("environment_interface", "drl_agent.env.environment_interface"),
    ("gazebo_service_wait", "drl_agent.env.simulation.gazebo_service_wait"),
]

# Need BOTH torch and a built ROS workspace (env node / trainer stacks).
ROS_TORCH_PAIRS = [
    ("gym_parameter_client", "drl_agent.training.gym_parameter_client"),
    ("curriculum_eval_runner", "drl_agent.training.curriculum.eval_runner"),
    ("curriculum_aux_eval", "drl_agent.training.curriculum.aux_eval"),
    ("train_tqc_base", "drl_agent.training.train_tqc_base"),
    ("train_tqc_curriculum_agent", "drl_agent.training.train_tqc_curriculum"),
    ("train_rl", "drl_agent.training.train_rl"),
    ("generalization_eval", "drl_agent.evaluation.generalization_eval"),
    ("risk_map_eval", "drl_agent.evaluation.risk_map_eval"),
    ("observation_builder", "drl_agent.env.observation.observation_builder"),
    ("start_sampler", "drl_agent.env.spawning.start_sampler"),
    ("goal_sampler", "drl_agent.env.spawning.goal_sampler"),
    ("obstacle_catalog_spawner", "drl_agent.env.spawning.obstacle_catalog_spawner"),
    ("human_spawn_sampler", "drl_agent.env.humans.human_spawn_sampler"),
    ("human_motion_manager", "drl_agent.env.humans.human_motion_manager"),
    ("dynamic_avoidance_telemetry", "drl_agent.env.humans.dynamic_avoidance_telemetry"),
    ("map_layout_runtime", "drl_agent.env.simulation.map_layout_runtime"),
    ("gazebo_entity_manager", "drl_agent.env.simulation.gazebo_entity_manager"),
    ("zone_tracker", "drl_agent.env.simulation.zone_tracker"),
    ("environment", "drl_agent.env.simulation.environment"),
    ("environment_curriculum", "drl_agent.env.curriculum.environment_curriculum"),
]


def _assert_alias(bare, canonical):
    canon_mod = importlib.import_module(canonical)
    bare_mod = importlib.import_module(bare)
    assert bare_mod is canon_mod, (
        f"legacy shim '{bare}' is not aliased to '{canonical}'"
    )


@pytest.mark.parametrize("bare,canonical", PURE_PAIRS)
def test_pure_module_shim_aliases_canonical(bare, canonical):
    _assert_alias(bare, canonical)


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
@pytest.mark.parametrize("bare,canonical", TORCH_PAIRS)
def test_torch_module_shim_aliases_canonical(bare, canonical):
    _assert_alias(bare, canonical)


@pytest.mark.skipif(not _HAVE_ROS, reason="needs built+sourced ROS2 workspace")
@pytest.mark.parametrize("bare,canonical", ROS_PAIRS)
def test_ros_module_shim_aliases_canonical(bare, canonical):
    _assert_alias(bare, canonical)


@pytest.mark.skipif(not (_HAVE_ROS and _HAVE_TORCH),
                    reason="needs torch AND a built+sourced ROS2 workspace")
@pytest.mark.parametrize("bare,canonical", ROS_TORCH_PAIRS)
def test_ros_torch_module_shim_aliases_canonical(bare, canonical):
    _assert_alias(bare, canonical)


def test_no_circular_import_package_roots():
    """Importing the package roots must not require torch or ROS."""
    for mod in ("drl_agent", "drl_agent.rl", "drl_agent.rl.networks",
                "drl_agent.rl.replay", "drl_agent.rl.algorithms",
                "drl_agent.rl.checkpointing", "drl_agent.training",
                "drl_agent.training.curriculum", "drl_agent.evaluation",
                "drl_agent.env", "drl_agent.env.observation",
                "drl_agent.env.rewards", "drl_agent.env.simulation",
                "drl_agent.env.spawning", "drl_agent.env.humans",
                "drl_agent.env.curriculum", "drl_agent.common",
                "drl_agent.config"):
        importlib.import_module(mod)
