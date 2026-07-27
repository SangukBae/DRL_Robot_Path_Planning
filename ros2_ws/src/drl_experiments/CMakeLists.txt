cmake_minimum_required(VERSION 3.8)
project(drl_experiments)

find_package(ament_cmake REQUIRED)

# Profiles and sweeps are data: installed to share/ so the profile loader can
# fall back to the installed copy when the source tree is unavailable.
# (outputs/ is deliberately NOT installed — it is a gitignored results dir.)
install(
  DIRECTORY
    profiles
    sweeps
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
    scripts/_bootstrap.py
    scripts/run_profile.py
    scripts/resume_profile.py
    scripts/aggregate.py
    scripts/export_tables.py
  DESTINATION
    lib/${PROJECT_NAME}
)

ament_package()
