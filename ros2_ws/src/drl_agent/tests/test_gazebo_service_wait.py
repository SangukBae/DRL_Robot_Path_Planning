"""ROS-free tests for the bounded Gazebo service-availability wait.

These guard the core of the fail-fast fix: ``environment.py::_wait_for_srv`` used
to loop forever (``while not client.wait_for_service(...)``), so a dead Gazebo
world-control / set_pose service hung the gym node's /step or /reset callback
indefinitely. The wait is now delegated to ``bounded_wait_for_service`` whose
contract is: ALWAYS terminate within the budget, succeed as soon as a probe
succeeds, and report elapsed time. ``GazeboServiceError`` is the failure type the
callbacks propagate so the trainer's bounded service call sees a clean failure.

Pure module (no ROS/Gazebo/torch); ``conftest.py`` puts scripts/environment on
sys.path. Run: ``python3 -m pytest -q -p no:anyio tests/test_gazebo_service_wait.py``
"""

import itertools

import pytest

from gazebo_service_wait import GazeboServiceError, bounded_wait_for_service


class FakeClock:
    """Deterministic monotonic clock; every probe advances it by the poll step so
    the wait reaches its deadline without real sleeping (and a non-terminating
    loop would instead show up as a hang/timeout in the test, not pass)."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _scripted_wait(clock, results):
    """Return a wait_once(step) that advances `clock` by `step` and yields the
    next scripted bool. Mirrors rclpy wait_for_service: blocks ~step, returns
    True iff up."""
    it = iter(results)

    def wait_once(step):
        clock.advance(step)
        return next(it)

    return wait_once


def test_available_immediately_probes_once():
    clock = FakeClock()
    calls = []
    ok, elapsed = bounded_wait_for_service(
        _scripted_wait(clock, [True]),
        timeout_sec=5.0, poll_sec=1.0, clock=clock,
        on_wait=lambda e: calls.append(e),
    )
    assert ok is True
    assert elapsed == pytest.approx(1.0)   # one poll quantum
    assert calls == []                      # on_wait only fires on FAILED probes


def test_available_after_a_few_polls():
    clock = FakeClock()
    waited = []
    ok, elapsed = bounded_wait_for_service(
        _scripted_wait(clock, [False, False, True]),
        timeout_sec=10.0, poll_sec=1.0, clock=clock,
        on_wait=lambda e: waited.append(e),
    )
    assert ok is True
    assert elapsed == pytest.approx(3.0)
    # on_wait fired once per failed probe, with the elapsed-so-far each time.
    assert waited == pytest.approx([1.0, 2.0])


def test_never_available_terminates_and_fails():
    """The critical regression guard: an always-down service must NOT hang."""
    clock = FakeClock()
    ok, elapsed = bounded_wait_for_service(
        _scripted_wait(clock, itertools.repeat(False)),
        timeout_sec=5.0, poll_sec=1.0, clock=clock,
    )
    assert ok is False
    # Bounded by timeout + at most one poll quantum.
    assert elapsed <= 5.0 + 1.0 + 1e-9
    assert elapsed >= 5.0


def test_zero_timeout_probes_exactly_once():
    clock = FakeClock()
    probes = {"n": 0}

    def wait_once(step):
        probes["n"] += 1
        clock.advance(step)
        return False

    ok, _ = bounded_wait_for_service(
        wait_once, timeout_sec=0.0, poll_sec=1.0, clock=clock)
    assert ok is False
    assert probes["n"] == 1


def test_poll_larger_than_timeout_still_one_probe():
    clock = FakeClock()
    ok, elapsed = bounded_wait_for_service(
        _scripted_wait(clock, itertools.repeat(False)),
        timeout_sec=2.0, poll_sec=10.0, clock=clock)
    assert ok is False
    # Single 10s probe overshoots the 2s budget by design (one quantum), then stops.
    assert elapsed == pytest.approx(10.0)


def test_terminates_under_real_monotonic_clock():
    """Belt-and-suspenders: with the REAL clock and a service that never comes up,
    the call returns quickly (no infinite loop), proving termination end-to-end."""
    ok, elapsed = bounded_wait_for_service(
        lambda step: False, timeout_sec=0.05, poll_sec=0.01)
    assert ok is False
    assert elapsed < 1.0


def test_error_is_runtime_error_subclass():
    assert issubclass(GazeboServiceError, RuntimeError)
    with pytest.raises(RuntimeError):
        raise GazeboServiceError("boom")
