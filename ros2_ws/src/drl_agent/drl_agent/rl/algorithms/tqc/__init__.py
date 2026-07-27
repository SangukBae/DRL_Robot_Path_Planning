"""TQC (Truncated Quantile Critics) — primary algorithm.

Canonical home of the flat legacy ``tqc_agent`` module. ``Agent`` is exported
lazily (torch required). Legacy bare-name shim: ``tqc_agent``.
"""


def __getattr__(name):
    if name in ("Agent", "Actor", "Critic", "quantile_huber_loss"):
        from . import agent as _agent
        value = getattr(_agent, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Agent", "Actor", "Critic", "quantile_huber_loss", "agent"]
