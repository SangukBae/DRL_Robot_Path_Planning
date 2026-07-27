"""Package-level pytest bootstrap (collection guard).

Guarantees pytest never descends into ``launch/`` or the colcon build
artifacts, REGARDLESS of how it is invoked:

  * from this package dir:   python3 -m pytest -q
  * from the repo root:      python3 -m pytest -q ros2_ws/src/drl_agent

The ROS ``launch_testing`` pytest plugin tries to import every ``test_*.py``
it sees (e.g. ``launch/test_td7.launch.py`` / ``test_tqc.launch.py``), which
``norecursedirs`` alone does not reliably prevent under that plugin.
``collect_ignore`` is honored at the rootdir conftest before any per-file
collection hook runs, so it stops the descent cleanly. The real ROS-free units
live under ``tests/`` (scoped by ``testpaths`` in pytest.ini for the no-arg
case); the live-sim run modules (``drl_agent/evaluation/live/``) are
deliberately named without a ``test_`` prefix so they were never a collection
hazard in the first place.
"""

# Paths are relative to this conftest's directory (the package root).
# - launch/  : *.launch.py files the ROS launch_testing plugin tries to import
#              (e.g. test_td7.launch.py / test_tqc.launch.py).
# - build/install/log : colcon artifacts.
collect_ignore = ["launch", "build", "install", "log"]
