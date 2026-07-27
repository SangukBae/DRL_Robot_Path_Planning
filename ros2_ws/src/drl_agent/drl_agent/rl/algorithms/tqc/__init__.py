"""TQC (Truncated Quantile Critics) — primary algorithm.

``Agent`` is exported lazily (torch required).
"""


def __getattr__(name):
    if name in ("Agent", "Actor", "Critic", "quantile_huber_loss"):
        from . import agent as _agent
        value = getattr(_agent, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Agent", "Actor", "Critic", "quantile_huber_loss", "agent"]
