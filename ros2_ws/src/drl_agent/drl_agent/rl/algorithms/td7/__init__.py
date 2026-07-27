"""td7 agent.

`Agent` is exported lazily (torch required).
"""


def __getattr__(name):
    if name == "Agent":
        from .agent import Agent
        globals()["Agent"] = Agent
        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Agent", "agent"]
