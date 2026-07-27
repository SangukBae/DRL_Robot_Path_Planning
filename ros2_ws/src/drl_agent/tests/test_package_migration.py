"""Canonical-package invariants for drl_agent.

The historical flat ``scripts/{policy,environment,utils}`` layout and its
bare-name import compatibility shims are RETIRED — every drl_agent module now
has exactly one import path: ``drl_agent.<...>``. This file locks:

  1. every canonical module listed below imports on its own (grouped by
     dependency weight: pure / torch / ROS / torch+ROS, mirroring the
     suite's existing skip conventions so the whole matrix runs inside the
     built docker workspace while the host still checks the pure set), and
  2. the old bare names are GONE — they must not resolve via any import
     mechanism, so a stray bare-name import in new code fails loudly instead
     of silently working by accident (e.g. because some other test left it
     in ``sys.modules``).
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


PURE_MODULES = [
    "drl_agent.training.run_layout",
    "drl_agent.config.paths",
    "drl_agent.common.geometry_utils",
    "drl_agent.common.seed_utils",
    "drl_agent.common.file_manager",
    "drl_agent.common.pure_pursuit",
    "drl_agent.training.episode_metrics",
    "drl_agent.training.aux_ablation_logging",
    "drl_agent.training.aux_eval_metrics",
    "drl_agent.training.dynamic_avoidance_log",
    "drl_agent.evaluation.risk_map_dump",
    "drl_agent.evaluation.sim_validation",
    "drl_agent.training.curriculum.stage_logic",
    "drl_agent.training.curriculum.metrics",
    "drl_agent.training.curriculum.state_io",
    "drl_agent.env.observation.aux_prediction_labels",
    "drl_agent.env.observation.obs_time_context",
    "drl_agent.env.rewards.reward_calculator",
    "drl_agent.env.simulation.collision_checker",
    "drl_agent.env.simulation.localization_noise",
    "drl_agent.env.simulation.map_catalog",
    "drl_agent.env.simulation.map_layout_registry",
]

TORCH_MODULES = [
    "drl_agent.rl.networks.tqc",
    "drl_agent.rl.networks.action_risk_head",
    "drl_agent.rl.networks.aux_prediction",
    "drl_agent.rl.networks.aux_losses",
    "drl_agent.rl.networks.aux_temporal",
    "drl_agent.rl.replay.buffer",
    "drl_agent.rl.replay.schema",
    "drl_agent.rl.checkpointing.tqc_io",
    # Paper comparison/ablation baselines — every algorithm must stay
    # importable from its canonical location.
    "drl_agent.rl.algorithms.tqc.agent",
    "drl_agent.rl.algorithms.sac.agent",
    "drl_agent.rl.algorithms.td7.agent",
    "drl_agent.rl.algorithms.a3c.agent",
    "drl_agent.rl.algorithms.tqc_ieqn.agent",
    "drl_agent.rl.algorithms.sb3.sac",
    "drl_agent.rl.algorithms.sb3.td3",
    "drl_agent.rl.algorithms.sb3.ppo",
]

ROS_MODULES = [
    "drl_agent.common.point_cloud2",
    "drl_agent.env.environment_interface",
    "drl_agent.env.simulation.gazebo_service_wait",
    "drl_agent.env.simulation.environment_360",
]

# Need BOTH torch and a built ROS workspace (env node / trainer stacks).
ROS_TORCH_MODULES = [
    "drl_agent.training.gym_parameter_client",
    "drl_agent.training.curriculum.eval_runner",
    "drl_agent.training.curriculum.aux_eval",
    "drl_agent.training.train_tqc_base",
    "drl_agent.training.train_tqc_curriculum",
    "drl_agent.training.train_rl",
    "drl_agent.training.baselines.sac",
    "drl_agent.training.baselines.sac_curriculum",
    "drl_agent.training.baselines.td7",
    "drl_agent.training.baselines.td7_curriculum",
    "drl_agent.training.baselines.a3c",
    "drl_agent.training.baselines.a3c_curriculum",
    "drl_agent.training.baselines.tqc_ieqn",
    "drl_agent.training.baselines.tqc_ieqn_curriculum",
    "drl_agent.training.baselines.sb3_sac",
    "drl_agent.training.baselines.sb3_sac_curriculum",
    "drl_agent.training.baselines.sb3_td3",
    "drl_agent.training.baselines.sb3_td3_curriculum",
    "drl_agent.training.baselines.sb3_ppo",
    "drl_agent.evaluation.generalization_eval",
    "drl_agent.evaluation.risk_map_eval",
    "drl_agent.evaluation.real_policy_runner",
    "drl_agent.evaluation.sim_validation_runner",
    "drl_agent.evaluation.live.tqc_live_runner",
    "drl_agent.evaluation.live.td7_live_runner",
    "drl_agent.env.observation.observation_builder",
    "drl_agent.env.spawning.start_sampler",
    "drl_agent.env.spawning.goal_sampler",
    "drl_agent.env.spawning.obstacle_catalog_spawner",
    "drl_agent.env.humans.human_spawn_sampler",
    "drl_agent.env.humans.human_motion_manager",
    "drl_agent.env.humans.dynamic_avoidance_telemetry",
    "drl_agent.env.simulation.map_layout_runtime",
    "drl_agent.env.simulation.gazebo_entity_manager",
    "drl_agent.env.simulation.zone_tracker",
    "drl_agent.env.simulation.environment",
    "drl_agent.env.curriculum.environment_curriculum",
]

# Every retired bare name from the old scripts/ layout. None of these may
# resolve any more — they are not on sys.path, not shimmed, nothing.
RETIRED_BARE_NAMES = [
    "run_layout", "config_paths", "geometry_utils", "seed_utils",
    "file_manager", "pure_pursuit", "point_cloud2", "episode_metrics",
    "aux_ablation_logging", "aux_eval_metrics", "dynamic_avoidance_log",
    "risk_map_dump", "sim_validation", "curriculum_stage_logic",
    "curriculum_metrics", "curriculum_state_io", "curriculum_eval_runner",
    "curriculum_aux_eval", "gym_parameter_client", "environment_interface",
    "aux_prediction_labels", "obs_time_context", "observation_builder",
    "reward_calculator", "collision_checker", "localization_noise",
    "map_catalog", "map_layout_registry", "map_layout_runtime",
    "gazebo_entity_manager", "gazebo_service_wait", "zone_tracker",
    "start_sampler", "goal_sampler", "obstacle_catalog_spawner",
    "human_spawn_sampler", "human_motion_manager",
    "dynamic_avoidance_telemetry", "environment", "environment_curriculum",
    "environment_360", "tqc_networks", "action_risk_head", "aux_prediction",
    "aux_prediction_losses", "aux_prediction_temporal", "buffer", "tqc_io",
    "tqc_agent", "sac_agent", "td7_agent", "a3c_agent", "tqc_ieqn_agent",
    "sb3_sac_agent", "sb3_td3_agent", "sb3_ppo_agent", "train_tqc_base",
    "train_tqc_curriculum_agent", "train_rl", "generalization_eval",
    "risk_map_eval", "real_policy_runner", "train_sac_agent",
    "train_sac_curriculum_agent", "train_td7_agent",
    "train_td7_curriculum_agent", "train_a3c_agent",
    "train_a3c_curriculum_agent", "train_tqc_ieqn_agent",
    "train_tqc_ieqn_curriculum_agent", "train_sb3_sac_agent",
    "train_sb3_sac_curriculum_agent", "train_sb3_td3_agent",
    "train_sb3_td3_curriculum_agent", "train_sb3_ppo_agent",
    "test_tqc_agent", "test_td7_agent", "sim_validation_runner",
    "aggregate_results", "analyze_aux_correlation", "analyze_yield_freezing",
    "aux_ablation_summary", "check_reproducibility", "plot_metrics",
    "plot_reward", "plot_trajectories_on_map", "sim_validation_summary",
]


@pytest.mark.parametrize("module", PURE_MODULES)
def test_pure_module_imports(module):
    importlib.import_module(module)


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
@pytest.mark.parametrize("module", TORCH_MODULES)
def test_torch_module_imports(module):
    importlib.import_module(module)


@pytest.mark.skipif(not _HAVE_ROS, reason="needs built+sourced ROS2 workspace")
@pytest.mark.parametrize("module", ROS_MODULES)
def test_ros_module_imports(module):
    importlib.import_module(module)


@pytest.mark.skipif(not (_HAVE_ROS and _HAVE_TORCH),
                    reason="needs torch AND a built+sourced ROS2 workspace")
@pytest.mark.parametrize("module", ROS_TORCH_MODULES)
def test_ros_torch_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("bare_name", RETIRED_BARE_NAMES)
def test_bare_name_import_is_retired(bare_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(bare_name)


def test_no_circular_import_package_roots():
    """Importing the package roots must not require torch or ROS."""
    for mod in ("drl_agent", "drl_agent.rl", "drl_agent.rl.networks",
                "drl_agent.rl.replay", "drl_agent.rl.algorithms",
                "drl_agent.rl.checkpointing", "drl_agent.training",
                "drl_agent.training.curriculum", "drl_agent.training.baselines",
                "drl_agent.evaluation", "drl_agent.evaluation.live",
                "drl_agent.evaluation.analysis",
                "drl_agent.env", "drl_agent.env.observation",
                "drl_agent.env.rewards", "drl_agent.env.simulation",
                "drl_agent.env.spawning", "drl_agent.env.humans",
                "drl_agent.env.curriculum", "drl_agent.common",
                "drl_agent.config"):
        importlib.import_module(mod)
