"""tqc_ieqn agent (canonical home of the flat legacy `tqc_ieqn_agent` module).

`Agent` is exported lazily (torch required). Legacy bare-name shim: `tqc_ieqn_agent`.
"""


def __getattr__(name):
    if name == "Agent":
        from .agent import Agent
        globals()["Agent"] = Agent
        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Agent", "agent"]
