# Ouster Simulation

Ouster LiDAR simulation for Gazebo Ignition. Publishes `sensor_msgs/msg/PointCloud2` on `/ouster/points`.

## Packages
- **ouster_description**: URDF models (OS1-64, OS1-32), launch files, rviz config
- **ouster_gazebo_plugins**: Gazebo Classic plugins (disabled)

## Environment
- ROS2 Humble
- Gazebo Ignition (ros_gz_sim)

## Build
```bash
colcon build --packages-select ouster_description
```

## Launch
```bash
ros2 launch ouster_description os1_64_alone.launch.py
ros2 launch ouster_description os1_32_alone.launch.py

# Headless mode
ros2 launch ouster_description os1_64_alone.launch.py gui:=false
```