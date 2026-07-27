"""Live-sim run scripts (canonical home of the flat legacy
``test_tqc_agent.py`` / ``test_td7_agent.py`` run scripts).

These drive a trained policy against a LIVE running environment node/Gazebo
for manual inspection — not pytest units (hence no ``test_`` prefix on the
module names here, unlike their historical scripts/ filenames, so a bare
``pytest`` invocation never mistakes them for test modules).
"""
