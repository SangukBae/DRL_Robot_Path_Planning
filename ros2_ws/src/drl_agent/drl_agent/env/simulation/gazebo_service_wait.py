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
