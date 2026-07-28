"""Unit tests for Stage-8 (ISOLATED experimental feature, default OFF, NOT
enabled for phase2/both) fixed physical-range observation normalization.

Covers: default-disabled passthrough, correct per-index scale-vector
construction for BOTH the 87-D (no history stacking) and 327-D
(observation_time_context, history_len=4) canonical layouts, "identical
normalization applied to encoder and temporal history inputs" (every LiDAR
slice -- current AND history frames alike -- divides by the SAME
lidar_scale), fail-fast on a mismatched state_dim, and numpy/torch dtype
handling.
"""

import numpy as np
import pytest

from drl_agent.rl.networks.obs_normalization import (
    ObsNormalizationConfig, ObsNormalizer, build_scale_vector,
)

OBS, AGENT = 80, 7
CUR = OBS + AGENT  # 87


def test_default_config_is_disabled():
    cfg = ObsNormalizationConfig.from_dict({})
    assert cfg.enabled is False


def test_disabled_normalizer_is_a_true_passthrough():
    cfg = ObsNormalizationConfig.from_dict({"enabled": False})
    norm = ObsNormalizer(cfg, state_dim=CUR)
    x = np.random.randn(4, CUR).astype(np.float32)
    out = norm.normalize(x)
    assert out is x, "disabled normalizer must return the SAME object, no copy/compute"


def test_scale_vector_87d_no_history():
    cfg = ObsNormalizationConfig.from_dict({
        "enabled": True, "lidar_scale": 50.0, "goal_dist_scale": 20.0,
        "heading_scale": 3.14159265, "speed_scale": 2.0,
        "yaw_rate_scale": 1.0, "steering_scale": 0.5,
    })
    scale = build_scale_vector(cfg, state_dim=CUR, obs_dim=OBS, agent_dim=AGENT,
                                history_len=1, stack_agent_state=False)
    assert scale.shape == (CUR,)
    assert np.all(scale[:OBS] == 50.0)          # lidar block
    assert scale[OBS] == 20.0                    # goal_dist
    assert scale[OBS + 1] == pytest.approx(3.14159265)  # heading
    assert scale[OBS + 4] == 2.0                  # speed
    assert scale[OBS + 5] == 1.0                  # yaw_rate
    assert scale[OBS + 6] == 0.5                  # steering


def test_scale_vector_327d_matches_observation_time_context_layout():
    # history_len=4, stack_agent_state=False -> current(87) + 3*obs(80) = 327
    cfg = ObsNormalizationConfig.from_dict({"enabled": True, "lidar_scale": 50.0})
    state_dim = CUR + 3 * OBS
    scale = build_scale_vector(cfg, state_dim=state_dim, obs_dim=OBS, agent_dim=AGENT,
                                history_len=4, stack_agent_state=False)
    assert scale.shape == (state_dim,)
    # Current frame's LiDAR block.
    assert np.all(scale[:OBS] == 50.0)
    # Every history frame (80-wide, obs-only) is ALSO all-lidar_scale --
    # "identical normalization applied to ... temporal history inputs".
    for k in range(3):
        base = CUR + k * OBS
        assert np.all(scale[base: base + OBS] == 50.0), f"history frame {k}"


def test_scale_vector_with_stack_agent_state_repeats_full_current_pattern():
    cfg = ObsNormalizationConfig.from_dict({
        "enabled": True, "lidar_scale": 50.0, "goal_dist_scale": 20.0,
    })
    state_dim = CUR + 3 * CUR  # history_len=4, stack_agent_state=True
    scale = build_scale_vector(cfg, state_dim=state_dim, obs_dim=OBS, agent_dim=AGENT,
                                history_len=4, stack_agent_state=True)
    assert scale.shape == (state_dim,)
    for k in range(4):
        base = k * CUR
        assert np.all(scale[base: base + OBS] == 50.0)
        assert scale[base + OBS] == 20.0  # each frame's own goal_dist slot


def test_mismatched_state_dim_fails_fast():
    cfg = ObsNormalizationConfig.from_dict({"enabled": True})
    with pytest.raises(ValueError, match="state_dim"):
        build_scale_vector(cfg, state_dim=CUR + 1, obs_dim=OBS, agent_dim=AGENT,
                            history_len=1, stack_agent_state=False)


def test_agent_dim_mismatch_fails_fast():
    cfg = ObsNormalizationConfig.from_dict({"enabled": True})
    with pytest.raises(ValueError, match="agent_dim"):
        build_scale_vector(cfg, state_dim=OBS + 5, obs_dim=OBS, agent_dim=5,
                            history_len=1, stack_agent_state=False)


def test_normalize_numpy_divides_by_scale():
    cfg = ObsNormalizationConfig.from_dict({"enabled": True, "lidar_scale": 50.0,
                                             "goal_dist_scale": 10.0})
    norm = ObsNormalizer(cfg, state_dim=CUR, obs_dim=OBS, agent_dim=AGENT)
    x = np.zeros((2, CUR), dtype=np.float32)
    x[:, :OBS] = 25.0    # half of lidar_scale
    x[:, OBS] = 5.0       # half of goal_dist_scale
    out = norm.normalize(x)
    assert np.allclose(out[:, :OBS], 0.5)
    assert np.allclose(out[:, OBS], 0.5)


def test_normalize_torch_tensor_matches_numpy():
    torch = pytest.importorskip("torch")
    cfg = ObsNormalizationConfig.from_dict({"enabled": True, "lidar_scale": 50.0})
    norm = ObsNormalizer(cfg, state_dim=CUR, obs_dim=OBS, agent_dim=AGENT)
    x_np = np.random.rand(3, CUR).astype(np.float32) * 10
    x_t = torch.from_numpy(x_np.copy())
    out_np = norm.normalize(x_np)
    out_t = norm.normalize(x_t)
    assert torch.allclose(out_t, torch.from_numpy(out_np), atol=1e-6)


def test_manifest_dict_records_every_scale_field():
    cfg = ObsNormalizationConfig.from_dict({"enabled": True, "lidar_scale": 42.0})
    m = cfg.manifest_dict()
    assert m["enabled"] is True
    assert m["lidar_scale"] == 42.0
    for key in ("goal_dist_scale", "heading_scale", "prev_action_r_scale",
                "prev_action_theta_scale", "speed_scale", "yaw_rate_scale",
                "steering_scale", "offset"):
        assert key in m
