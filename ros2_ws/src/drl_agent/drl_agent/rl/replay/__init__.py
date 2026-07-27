"""Replay buffer.

  drl_agent.rl.replay.buffer  LAP replay buffer (requires torch; import lazily)
  drl_agent.rl.replay.schema  per-transition field / npz checkpoint contract
                              (pure numpy — importable without torch)
"""


def __getattr__(name):
    if name == "LAP":
        from .buffer import LAP
        globals()["LAP"] = LAP
        return LAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LAP", "buffer", "schema"]
