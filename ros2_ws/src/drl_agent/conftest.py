"""Package-level pytest bootstrap (collection guard).

Guarantees pytest never descends into ``scripts/`` or the colcon build artifacts,
REGARDLESS of how it is invoked:

  * from this package dir:   python3 -m pytest -q
  * from the repo root:      python3 -m pytest -q ros2_ws/src/drl_agent

``scripts/policy/test_tqc_agent.py`` and ``test_td7_agent.py`` match the
``test_*`` pattern but are ROS2/Gazebo run/eval scripts (they import rclpy/torch
and drive a live sim), so collecting them errors out — and the ROS
``launch_testing`` pytest plugin tries to import every ``test_*.py`` it sees,
which ``norecursedirs`` alone does not reliably prevent under that plugin.
``collect_ignore`` is honored at the rootdir conftest before any per-file
collection hook runs, so it stops the descent cleanly. The real ROS-free units
live under ``tests/`` (scoped by ``testpaths`` in pytest.ini for the no-arg case).
"""

# Paths are relative to this conftest's directory (the package root).
# - scripts/ : ROS run/eval scripts named test_*_agent.py (import rclpy/torch).
# - launch/  : *.launch.py files the ROS launch_testing plugin tries to import
#              (e.g. test_td7.launch.py / test_tqc.launch.py).
# - build/install/log : colcon artifacts.
collect_ignore = ["scripts", "launch", "build", "install", "log"]
