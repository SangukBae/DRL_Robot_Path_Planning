"""Network building blocks for the RL agents (canonical home).

Submodules (all require torch; exports resolve lazily so ``import
drl_agent.rl.networks`` and the config-only tooling never pull in torch):

  drl_agent.rl.networks.tqc              Actor / Critic / quantile_huber_loss
  drl_agent.rl.networks.action_risk_head PHASE2 critic-connected action-risk head
  drl_agent.rl.networks.aux_prediction   AUX_PRED shared encoder + aux heads
  drl_agent.rl.networks.aux_losses       AUX_PRED loss functions
  drl_agent.rl.networks.aux_temporal     AUX_PRED temporal context / fusion encoders

Legacy bare-name shims: ``tqc_networks``, ``action_risk_head``,
``aux_prediction``, ``aux_prediction_losses``, ``aux_prediction_temporal``.
"""

# PEP 562 lazy attribute access: `from drl_agent.rl.networks import Actor`
# works without importing torch at package-import time.
_LAZY = {
    "Actor": ("tqc", "Actor"),
    "Critic": ("tqc", "Critic"),
    "quantile_huber_loss": ("tqc", "quantile_huber_loss"),
    "ActionRiskConfig": ("action_risk_head", "ActionRiskConfig"),
    "ActionRiskHead": ("action_risk_head", "ActionRiskHead"),
    "AuxPredConfig": ("aux_prediction", "AuxPredConfig"),
    "SharedEncoder": ("aux_prediction", "SharedEncoder"),
    "AuxiliaryHead": ("aux_prediction", "AuxiliaryHead"),
    "ActionConditionedAuxHead": ("aux_prediction", "ActionConditionedAuxHead"),
    "compute_aux_loss": ("aux_losses", "compute_aux_loss"),
    "TemporalContextEncoder": ("aux_temporal", "TemporalContextEncoder"),
    "TemporalFusionEncoder": ("aux_temporal", "TemporalFusionEncoder"),
    "ScanTemporalEncoder": ("aux_temporal", "ScanTemporalEncoder"),
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    try:
        submod, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(f".{submod}", __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
