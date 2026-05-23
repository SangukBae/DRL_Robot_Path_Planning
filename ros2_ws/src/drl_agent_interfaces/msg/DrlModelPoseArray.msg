# Batch model-name + pose list for DrlModelPosePlugin.
# Sent by environment.py at 20 Hz; consumed by the Gazebo plugin which
# applies all poses in one PreUpdate() pass — zero blocking service calls.
string[] names
geometry_msgs/Pose[] poses
