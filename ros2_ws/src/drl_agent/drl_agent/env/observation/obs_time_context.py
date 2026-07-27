"""Observation time-context (actor-visible frame stacking) — ROS-free helper.

Extracted from environment.py so the stacked-state contract is one pure,
unit-testable place. Stacks the last N ``obs_state`` frames (the front-180°,
80-D RL scan — NOT the raw point cloud and NOT the 360° collision
``environment_state``) so the ACTOR sees short-horizon temporal context through
the shared encoder, with NO recurrence.

Layout (history APPENDED after the current frame so the 87-D baseline layout is
byte-for-byte preserved — state[:obs_dim] = current obs, state[obs_dim] =
goal_dist — and every existing index-based reader keeps working):

    [ obs_t(O), agent_t(A), obs_{t-1}(O), ..., obs_{t-(N-1)}(O) ]      (obs-only history)
    [ obs_t(O), agent_t(A), frame_{t-1}(O+A), ..., frame_{t-(N-1)} ]    (stack_agent_state)

Disabled (or obs_frame_stack < 2) -> returns the exact [obs_t, agent_t] vector,
so the baseline is unchanged. The history deque is most-recent-first and bounded
by maxlen, so the stacked width is constant for every step of an episode.

Episode start uses FIRST-FRAME REPEAT (not zero-pad): obs values are LiDAR
distances, so a zero would read as 'obstacle touching the robot in every bin' — a
false imminent-hazard signal. Repeating the (stationary) first observation is the
physically-correct 'no motion yet' history.
"""

from collections import deque

import numpy as np


class ObsTimeContext:
    def __init__(self, environment_dim, agent_dim, enabled=False,
                 obs_frame_stack=1, stack_agent_state=False):
        self.environment_dim = int(environment_dim)
        self.agent_dim = int(agent_dim)
        self.obs_frame_stack = max(1, int(obs_frame_stack))
        self.stack_agent_state = bool(stack_agent_state)
        # enabled but a single frame == no stacking; normalize to disabled so the
        # state dim stays the base frame and the contract is unambiguous.
        self.enabled = bool(enabled) and self.obs_frame_stack >= 2
        self._history = deque(maxlen=max(0, self.obs_frame_stack - 1))

    def stacked_state_dim(self) -> int:
        """Full RL state width on the wire (current frame + appended history)."""
        base = self.environment_dim + self.agent_dim
        if not self.enabled:
            return base
        frame_len = (self.environment_dim + self.agent_dim
                     if self.stack_agent_state else self.environment_dim)
        return base + (self.obs_frame_stack - 1) * frame_len

    def history_frame(self, obs_state, agent_state):
        """Per-step vector pushed into history: obs-only (O) by default, or the
        full (O+A) frame when stack_agent_state is set."""
        if self.stack_agent_state:
            return np.append(obs_state, agent_state).astype(np.float32)
        return np.asarray(obs_state, dtype=np.float32).copy()

    def reset(self, obs_state, agent_state):
        """Seed the history with (N-1) copies of the first frame at episode start.

        Does NOT advance: the seeded deque already represents this episode's
        'past' (all = the first frame), so the first assemble() reads it directly.
        """
        self._history.clear()
        if not self.enabled:
            return
        frame0 = self.history_frame(obs_state, agent_state)
        for _ in range(self.obs_frame_stack - 1):
            self._history.append(frame0.copy())

    def assemble(self, obs_state, agent_state, advance=True):
        """Build the (optionally stacked) RL state and advance the history.

        Returns [obs_t, agent_t] when disabled, else [obs_t, agent_t, obs_{t-1},
        ...] from the frames seeded/pushed so far. ``advance`` pushes the current
        frame for the NEXT step (the deque already holds this step's PAST). Call
        EXACTLY once per produced state, in temporal order."""
        current = np.append(obs_state, agent_state).astype(np.float32)
        if not self.enabled:
            return current
        stacked = np.concatenate([current] + list(self._history)).astype(np.float32)
        if advance:
            self._history.appendleft(self.history_frame(obs_state, agent_state))
        return stacked
