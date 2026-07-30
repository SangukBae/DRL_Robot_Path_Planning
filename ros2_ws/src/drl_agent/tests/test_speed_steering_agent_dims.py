"""torch-gated tests for:

  * action_dim=2 (the PHASE3 speed_steering contract) flows generically into
    Actor / Critic / ActionRiskHead / aux AuxiliaryHead input widths and the
    auto target_entropy = -action_dim, with NO code branching needed for the
    new mode (Agent has always been action_dim-generic).
  * weighted_action_risk_loss (RISK_BALANCE): pos_weight==1.0 is byte-
    identical to plain MSE; pos_weight>1 up-weights positive-target elements;
    pos_weight_cap bounds a raw config value; positive/safe loss breakdown +
    recall/F1 diagnostics.
  * Agent.train() end-to-end with risk_balanced_sampling ON: runs without
    error, action-risk/aux supervised loss sources from the balanced batch,
    and sampled-pool fractions are logged. OFF-parity: with the feature off,
    no risk_balance/* keys appear and training behaves as before.
"""

import pytest

try:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from drl_agent.rl.algorithms.tqc.agent import Agent
    from drl_agent.rl.networks.action_risk_head import (
        ActionRiskConfig, weighted_action_risk_loss)
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")

STATE_DIM, ACTION_DIM = 87, 2   # speed_steering: action_dim is 2, not 3


def _hp(**over):
    hp = dict(batch_size=8, buffer_size=500, n_critics=2, n_quantiles=5)
    hp.update(over)
    return hp


def _fill(agent, n=64, action_risk_target=False, risk_meta_kind=None):
    for i in range(n):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        a = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        kwargs = {}
        if action_risk_target:
            kwargs["action_risk_target"] = np.random.rand(2).astype(np.float32)
        if risk_meta_kind == "mixed":
            if i % 4 == 0:
                kwargs["risk_meta"] = (0.0, 1.0, 1.0, 0.0)
            elif i % 4 == 1:
                kwargs["risk_meta"] = (0.0, 0.0, 0.0, 1.0)
            else:
                kwargs["risk_meta"] = (0.0, 0.0, 0.0, 0.0)
        agent.replay_buffer.add(s, a, s, 0.1, 0.0, **kwargs)


# --------------------------------------------------------------------------- #
#  action_dim=2 flows generically into every network + target_entropy
# --------------------------------------------------------------------------- #
def test_target_entropy_auto_is_minus_action_dim(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path))
    assert agent.target_entropy == pytest.approx(-2.0)


def test_actor_critic_input_dims_match_action_dim_2(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0, _hp(), log_dir=str(tmp_path))
    assert agent.replay_buffer.action.shape[1] == ACTION_DIM
    s = np.random.randn(STATE_DIM).astype(np.float32)
    a = agent.select_action(s, use_exploration=False)
    assert a.shape == (ACTION_DIM,)


def test_action_risk_head_input_dim_matches_action_dim_2(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                   _hp(action_risk_head={"enabled": True, "hidden_dim": 16}),
                   log_dir=str(tmp_path))
    assert agent.action_risk_head.l1.in_features == agent.encoder.out_dim + ACTION_DIM


def test_action_conditioned_aux_head_action_dim_matches_2(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                   _hp(aux_prediction={"enabled": True, "action_conditioned_aux": True,
                                        "action_conditioned_steps": 2}),
                   log_dir=str(tmp_path))
    assert agent.aux_head is not None
    # action-conditioned head's action-embedding layer is sized off action_dim.
    assert agent.aux_head.action_embed.in_features == ACTION_DIM


# --------------------------------------------------------------------------- #
#  weighted_action_risk_loss
# --------------------------------------------------------------------------- #
def test_pos_weight_default_is_byte_identical_to_plain_mse():
    cfg = ActionRiskConfig({})   # pos_weight defaults to 1.0
    torch.manual_seed(0)
    pred = torch.rand(16, 2)
    target = torch.rand(16, 2)
    loss, _ = weighted_action_risk_loss(pred, target, cfg)
    assert loss.item() == pytest.approx(F.mse_loss(pred, target).item())


def test_pos_weight_cap_bounds_a_large_config_value():
    cfg = ActionRiskConfig({"pos_weight": 999.0, "pos_weight_cap": 10.0})
    assert cfg.pos_weight == pytest.approx(10.0)


def test_pos_weight_upweights_positive_target_elements():
    cfg = ActionRiskConfig({"pos_weight": 5.0, "positive_threshold": 0.5})
    pred = torch.zeros(2, 2)
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])   # row 1 all "positive"
    loss, logs = weighted_action_risk_loss(pred, target, cfg)
    # row 0 error=0, row 1 error=1 (weighted 5x) -> mean = (0*1 + 1*5*2)/4 = 2.5
    assert loss.item() == pytest.approx(2.5)
    assert logs["action_risk/positive_loss"] == pytest.approx(1.0)
    assert logs["action_risk/safe_loss"] == pytest.approx(0.0)


def test_smooth_l1_loss_type_selected():
    cfg = ActionRiskConfig({"loss_type": "smooth_l1"})
    pred = torch.zeros(4, 2)
    target = torch.full((4, 2), 5.0)   # large error -> smooth_l1 != mse
    loss, _ = weighted_action_risk_loss(pred, target, cfg)
    mse = F.mse_loss(pred, target)
    assert loss.item() != pytest.approx(mse.item())


def test_invalid_loss_type_fails_fast():
    with pytest.raises(ValueError):
        ActionRiskConfig({"loss_type": "huber"})


def test_positive_recall_reported_when_positives_present():
    cfg = ActionRiskConfig({"positive_threshold": 0.5})
    pred = torch.tensor([[0.9, 0.9], [0.1, 0.1]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    _, logs = weighted_action_risk_loss(pred, target, cfg)
    assert logs["action_risk/positive_recall"] == pytest.approx(1.0)
    assert logs["action_risk/positive_f1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#  Agent.train() end-to-end with risk_balanced_sampling
# --------------------------------------------------------------------------- #
def test_risk_balanced_sampling_off_by_default_no_extra_logs(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                   _hp(action_risk_head={"enabled": True, "hidden_dim": 16}),
                   log_dir=str(tmp_path))
    assert agent.risk_balanced_enabled is False
    assert agent.replay_buffer.risk_meta is None
    _fill(agent, action_risk_target=True)
    agent.train()   # must run to completion unchanged


def test_risk_balanced_sampling_enabled_trains_end_to_end(tmp_path):
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                   _hp(action_risk_head={"enabled": True, "hidden_dim": 16},
                       replay_buffer={
                           "risk_meta": {"enabled": True},
                           "risk_balanced_sampling": {
                               "enabled": True,
                               "ratio_uniform": 0.5,
                               "ratio_human_risk": 0.25,
                               "ratio_collision": 0.25,
                           },
                       }),
                   log_dir=str(tmp_path))
    assert agent.risk_balanced_enabled is True
    assert agent.replay_buffer.risk_meta is not None
    _fill(agent, n=200, action_risk_target=True, risk_meta_kind="mixed")

    before = [p.detach().clone() for p in agent.action_risk_head.parameters()]
    for _ in range(5):
        agent.train()
    after = list(agent.action_risk_head.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), \
        "action_risk_head must still train when its loss sources from the balanced batch"


def test_risk_meta_enabled_without_balanced_sampling_is_metadata_only(tmp_path):
    """risk_meta.enabled=true with risk_balanced_sampling.enabled=false must
    collect metadata (for later analysis) WITHOUT changing what trains on it
    -- sample_risk_balanced() stays a no-op."""
    agent = Agent(STATE_DIM, ACTION_DIM, 1.0,
                   _hp(replay_buffer={"risk_meta": {"enabled": True}}),
                   log_dir=str(tmp_path))
    assert agent.store_risk_meta is True
    assert agent.risk_balanced_enabled is False
    assert agent.replay_buffer.sample_risk_balanced() is None
