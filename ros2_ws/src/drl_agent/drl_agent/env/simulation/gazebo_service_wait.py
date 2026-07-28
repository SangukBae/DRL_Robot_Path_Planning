"""ROS-free bounded service-availability wait for the Gazebo entity I/O.

The gym node (`environment.py`) talks to Gazebo through ros_gz service clients
(`/world/<world>/control`, `/world/<world>/set_pose`, …) from inside the `/step`
and `/reset` service callbacks.  The wait for those clients to come up used to be
an UNBOUNDED ``while not client.wait_for_service(timeout_sec=1.0): ...`` loop, so
if Gazebo (or just the world-control service) died, the callback never returned:
the gym node hung forever and the trainer only saw repeated `/step` timeouts.

This module isolates the "never block forever" guarantee in a pure helper so it
can be unit tested without ROS/Gazebo, and defines the failure type the node
raises so it propagates out of the callback (the trainer's bounded
``EnvInterface._call_service`` then sees a service failure and runs its
checkpoint-on-failure path — the two ends now share the same fail-fast policy).

``compute_physics_step_count`` supports ``environment.py``'s opt-in
``gazebo_deterministic_stepping`` (WorldControl ``multi_step``, see
``Environment._advance_physics_deterministic``'s docstring): a stricter
per-step bounded wait on ``/clock`` reaching the target sim time was
attempted and excluded — it reproducibly hung real ``/step`` calls under the
live node's MultiThreadedExecutor even though an isolated probe against the
same ``/clock`` topic worked reliably every time (see memory note
``gazebo_multi_step_clock_wait_landmine`` for the full investigation). The
shipped design instead sleeps for the requested duration and verifies sensor
freshness via the same scan/odom-count wait ``reset_callback`` already uses.
"""

import time


class GazeboServiceError(RuntimeError):
    """Raised when a Gazebo world-control / set_pose service call cannot be
    completed: the service was unavailable within the wait budget, the response
    future did not arrive within the call budget, or the call raised.

    ``environment.py`` lets this propagate out of ``step_callback`` /
    ``reset_callback`` instead of swallowing it (the old code warned and carried
    on with a possibly-unpaused/half-reset world) or calling ``sys.exit(-1)``
    from an executor worker thread.  Propagating means the gym node stops cleanly
    with a clear diagnostic, the trainer's ``/step`` // ``/reset`` call times out
    on its OWN bounded budget and raises ``EnvServiceError``, and the trainer's
    checkpoint-on-failure path runs.  Deliberately mirrors the trainer-side
    ``EnvServiceError`` so the failure semantics match on both ends."""


def bounded_wait_for_service(
    wait_once, timeout_sec, poll_sec, *, clock=time.monotonic, on_wait=None
):
    """Poll ``wait_once`` until the service is up or the deadline passes.

    Parameters
    ----------
    wait_once(step_sec) -> bool
        Blocks up to ``step_sec`` seconds and returns ``True`` iff the service is
        available — the exact contract of ``rclpy`` ``Client.wait_for_service``.
    timeout_sec : float
        Total availability-wait budget.  ``<= 0`` means "probe exactly once".
    poll_sec : float
        Per-probe block / log cadence.
    clock : callable
        Monotonic time source (injectable for tests).
    on_wait(elapsed_sec) : callable, optional
        Called after each FAILED probe so the caller can log progress.

    Returns
    -------
    (ok: bool, elapsed_sec: float)
        ``ok`` is ``True`` as soon as a probe succeeds.  Guaranteed to terminate:
        the total time is bounded by ``timeout_sec`` plus at most one poll
        quantum, regardless of the service ever coming up — there is no
        unbounded loop.
    """
    timeout_sec = max(0.0, float(timeout_sec))
    poll_sec = max(1e-3, float(poll_sec))
    start = clock()
    deadline = start + timeout_sec
    while True:
        # Always make at least one probe (so timeout_sec == 0 still checks once).
        if wait_once(poll_sec):
            return True, clock() - start
        elapsed = clock() - start
        if on_wait is not None:
            on_wait(elapsed)
        if clock() >= deadline:
            return False, clock() - start


def compute_physics_step_count(duration: float, physics_step_size: float, *, tol: float = 1e-6) -> int:
    """Return the exact integer number of ``physics_step_size``-second physics
    iterations that make up ``duration`` seconds, for a single WorldControl
    ``multi_step`` call (empirically confirmed: multi_step=N steps physics by
    exactly N * max_step_size seconds of sim time, live-measured against
    drl_arena.world's max_step_size=0.001s).

    Fails fast (ValueError) instead of silently rounding when ``duration``
    isn't (close to) an integer multiple of ``physics_step_size``, so a
    mismatched time_delta/physics_step_size pair is caught at startup rather
    than silently drifting the requested vs. actual sim-time advance.
    """
    if physics_step_size <= 0:
        raise ValueError(f"physics_step_size must be > 0, got {physics_step_size}")
    raw_n = duration / physics_step_size
    n_steps = round(raw_n)
    if n_steps < 1:
        raise ValueError(
            f"duration={duration:.6f}s is shorter than one physics step "
            f"({physics_step_size:.6f}s); deterministic Gazebo stepping needs "
            "duration >= one physics_step_size.")
    if abs(raw_n - n_steps) > tol:
        raise ValueError(
            f"duration={duration:.6f}s is not an integer multiple of "
            f"physics_step_size ({physics_step_size:.6f}s); got {raw_n:.6f} "
            "steps. Adjust time_delta or physics_step_size so this divides "
            "evenly, or disable gazebo_deterministic_stepping.")
    return n_steps
