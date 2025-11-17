# Scout Description Package

ROS2 URDF/XACRO description package for the **AgileX Scout Mini** mobile robot platform. This package provides robot models, visual meshes, and configuration files for simulation and visualization.

## Package Information

- **Package Name**: `scout_description`
- **Version**: 0.0.0
- **License**: Apache License 2.0
- **Maintainer**: Mattia Dutto (mattia.dutto@polito.it)
- **Description**: AgileX Scout Mini robot description for ROS2 Humble

## Package Structure

```
scout_description/
├── CMakeLists.txt              # Build configuration
├── package.xml                 # Package metadata
├── README.md                   # This file
├── meshes/                     # 3D mesh files for visualization
│   ├── scout_mini_base_link.dae    # Base body mesh (14 MB)
│   └── wheel.dae                   # Wheel mesh (2.7 MB)
├── urdf/                       # Robot description files
│   └── scout_mini.xacro            # Main robot XACRO definition
└── rviz/                       # RViz configuration files
    └── scout_mini.rviz             # Default visualization config
```

## Dependencies

### Build Dependencies
- `ament_cmake`: CMake build system for ROS2
- `xacro`: XML macro language for URDF

### Runtime Dependencies
- `xacro`: Required for processing XACRO files at runtime

## Installation

### Build from Source

```bash
cd /path/to/ros2_ws
source /opt/ros/humble/setup.bash

# Build this package
colcon build --packages-select scout_description

# Source the workspace
source install/setup.bash
```

## Scout Mini Robot Specification

### Overview

The **AgileX Scout Mini** is a 4-wheel drive skid-steering mobile robot designed for outdoor and indoor navigation. This URDF model provides an accurate representation for Gazebo simulation.

### Physical Dimensions

#### **Base Link (Robot Body)**

The Scout Mini base uses a custom 3D mesh (`scout_mini_base_link.dae`) for visual representation.

**Orientation:**
- Visual/Collision mesh is rotated: `rpy="1.57 0 -1.57"` (90° pitch, -90° yaw)
- This aligns the robot model correctly in ROS coordinate frame (X-forward, Y-left, Z-up)

**Estimated Dimensions** (from mesh file):
- Length: ~0.650 m
- Width: ~0.490 m
- Height: ~0.280 m

### Inertial Properties

#### **Base Link Inertia**

Located in `inertial_link` (fixed to `base_link`):

```xml
<mass value="132.3898489950015" />  <!-- 132.39 kg -->
<origin xyz="0 0 0" />
<inertia
  ixx="0.185196122711036"
  ixy="4.30144213829512E-08"
  ixz="5.81037523686401E-08"
  iyy="0.364893736238929"
  iyz="-0.000386720198091934"
  izz="0.223868521722778"
/>
```

**Notes:**
- Total robot mass: **132.39 kg** (excluding wheels)
- Inertia tensor is calculated from CAD model
- Center of mass at origin of `base_link`

### Wheel Specifications

The Scout Mini uses **4 identical wheels** in a skid-steering configuration.

#### **Wheel Physical Properties**

**Individual Wheel Mass:**
```xml
<mass value="3" />  <!-- 3 kg per wheel -->
```

**Total Mass with Wheels:** 132.39 kg (base) + 4 × 3 kg (wheels) = **144.39 kg**

**Wheel Inertia:**
```xml
<inertia
  ixx="0.7171" ixy="0" ixz="0"
  iyy="0.7171" iyz="0"
  izz="0.1361"
/>
```

**Wheel Mesh:** `wheel.dae` (2.7 MB)

#### **Wheel Positions**

All wheel positions are relative to `base_link` origin (center of robot base).

| Wheel | X (m) | Y (m) | Z (m) | RPY (rad) | Description |
|-------|-------|-------|-------|-----------|-------------|
| **Front Right** | 0.2319755 | -0.2082515 | -0.099998 | `1.57 0 0` | Right side, front |
| **Front Left** | 0.2319755 | 0.2082515 | -0.100998 | `-1.57 0 0` | Left side, front |
| **Rear Right** | -0.2319755 | -0.2082515 | -0.099998 | `1.57 0 0` | Right side, rear |
| **Rear Left** | -0.2319755 | 0.2082515 | -0.100998 | `-1.57 0 0` | Left side, rear |

#### **Derived Wheel Geometry**

From the wheel positions, we can calculate key dimensions:

**Wheelbase** (front-to-rear distance):
```
wheelbase = |front_x - rear_x|
          = |0.2319755 - (-0.2319755)|
          = 0.463951 m
          ≈ 0.464 m
```

**Track Width** (left-to-right distance):
```
track_width = |left_y - right_y|
            = |0.2082515 - (-0.2082515)|
            = 0.416503 m
            ≈ 0.417 m
```

**Wheel Height** (ground clearance):
```
ground_clearance = |wheel_z|
                 = |-0.099998|
                 ≈ 0.100 m = 10 cm
```

**Wheel Radius** (from Gazebo plugin):
```
wheel_radius = 0.160 m = 16 cm
```

**Wheel Separation** (for differential drive):
```
wheel_separation = 0.490 m = 49 cm
```
*Note: This is the effective distance between left and right wheel centers used by the differential drive controller*

### Robot Coordinate Frame

```
         Z (up)
         │
         │
         │
         └─────── X (forward)
        ╱
       ╱
      Y (left)

Top View:
    Y (left)
    ↑
    │
    │     [FL]●────────●[FR]
    │       │          │
    │       │  BASE    │
    │       │          │
    │     [RL]●────────●[RR]
    │
    └──────────────────→ X (forward)

    Wheelbase: 0.464 m
    Track Width: 0.417 m
```

### Bounding Box Dimensions

**Estimated Robot Bounding Box** (for collision detection):

```
Minimum Bounding Box:
┌─────────────────────────────────────┐
│  Length (X):  ~0.650 m              │
│  Width (Y):   ~0.490 m              │
│  Height (Z):  ~0.280 m (body)       │
│               +0.160 m (wheel)      │
│               ─────────────────     │
│  Total Height: ~0.440 m             │
└─────────────────────────────────────┘

With Safety Margin (+10%):
  Length:  0.715 m
  Width:   0.540 m
  Height:  0.484 m
```

**Collision Geometry:**
- Visual mesh is used for collision detection
- Same DAE file for both visual and collision elements
- Precise collision detection with mesh geometry

### Joint Definitions

#### **Wheel Joints**

All wheel joints are of type `continuous` (unlimited rotation).

| Joint Name | Parent Link | Child Link | Axis | Description |
|------------|-------------|------------|------|-------------|
| `front_right_wheel` | `base_link` | `front_right_wheel_link` | `[0 0 -1]` | Right wheels rotate negatively |
| `front_left_wheel` | `base_link` | `front_left_wheel_link` | `[0 0 1]` | Left wheels rotate positively |
| `rear_right_wheel` | `base_link` | `rear_right_wheel_link` | `[0 0 -1]` | Right wheels rotate negatively |
| `rear_left_wheel` | `base_link` | `rear_left_wheel_link` | `[0 0 1]` | Left wheels rotate positively |

**Rotation Axes:**
- Left wheels: Positive Z-axis (after transformation)
- Right wheels: Negative Z-axis (after transformation)
- This configuration ensures synchronized movement for skid-steering

## Gazebo Plugins

The Scout Mini URDF includes two Ignition Gazebo plugins for realistic simulation.

### 1. Joint State Publisher Plugin

**Purpose:** Publishes the state of all wheel joints (position, velocity, effort)

**Configuration:**
```xml
<plugin filename="libignition-gazebo-joint-state-publisher-system.so"
        name="ignition::gazebo::systems::JointStatePublisher">
    <topic>joint_states</topic>
    <update_rate>50</update_rate>

    <joint_name>front_right_wheel</joint_name>
    <joint_name>front_left_wheel</joint_name>
    <joint_name>rear_right_wheel</joint_name>
    <joint_name>rear_left_wheel</joint_name>
</plugin>
```

**Published Topic:** `joint_states` (bridged to `/scout_mini/joint_states`)
- **Type:** `sensor_msgs/msg/JointState`
- **Frequency:** 50 Hz
- **Data:** Position, velocity, and effort for all 4 wheels

### 2. Differential Drive Plugin

**Purpose:** Provides skid-steering locomotion control

**Configuration:**
```xml
<plugin filename="libignition-gazebo-diff-drive-system.so"
        name="ignition::gazebo::systems::DiffDrive">

    <!-- Left side wheels -->
    <left_joint>front_left_wheel</left_joint>
    <left_joint>rear_left_wheel</left_joint>

    <!-- Right side wheels -->
    <right_joint>front_right_wheel</right_joint>
    <right_joint>rear_right_wheel</right_joint>

    <!-- Physical parameters -->
    <wheel_separation>0.490</wheel_separation>
    <wheel_radius>0.160</wheel_radius>

    <!-- Limits -->
    <max_wheel_torque>20</max_wheel_torque>
    <max_linear_acceleration>1.0</max_linear_acceleration>

    <!-- Topics -->
    <topic>cmd_vel</topic>
    <odom_topic>odom</odom_topic>
    <tf_topic>tf</tf_topic>

    <!-- Odometry settings -->
    <odom_publish_frequency>50</odom_publish_frequency>
    <frame_id>odom</frame_id>
    <child_frame_id>base_link</child_frame_id>
</plugin>
```

**Subscribed Topics:**
- `cmd_vel` (bridged to `/scout_mini/cmd_vel`)
  - **Type:** `geometry_msgs/msg/Twist`
  - **Description:** Velocity commands (linear.x, angular.z)

**Published Topics:**
- `odom` (bridged to `/scout_mini/odom`)
  - **Type:** `nav_msgs/msg/Odometry`
  - **Frequency:** 50 Hz
  - **Description:** Odometry information (pose, twist)

- `tf` (bridged to `/scout_mini/tf`)
  - **Type:** `tf2_msgs/msg/TFMessage`
  - **Description:** Transform from `odom` to `base_link`

**Performance Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Wheel Separation | 0.490 m | Distance between left and right wheels |
| Wheel Radius | 0.160 m | Radius of each wheel |
| Max Wheel Torque | 20 Nm | Maximum torque per wheel |
| Max Linear Acceleration | 1.0 m/s² | Maximum acceleration constraint |
| Odometry Frequency | 50 Hz | Update rate for odometry data |

**Velocity Limits** (derived from physical parameters):

```
Maximum Linear Velocity (estimated):
  v_max ≈ ω_max × r_wheel

  Assuming max wheel speed of ~5 rad/s:
  v_max ≈ 5 × 0.16 = 0.8 m/s

Maximum Angular Velocity (estimated):
  ω_max = v_max / (wheel_separation / 2)
        ≈ 0.8 / 0.245
        ≈ 3.27 rad/s ≈ 187°/s
```

## TF Frame Tree

```
odom (odometry frame, fixed world reference)
 └── base_link (robot base center)
      ├── inertial_link (center of mass, fixed)
      ├── front_right_wheel_link
      ├── front_left_wheel_link
      ├── rear_right_wheel_link
      └── rear_left_wheel_link
```

**Frame Relationships:**
- `odom` → `base_link`: Published by differential drive plugin (dynamic)
- `base_link` → `inertial_link`: Fixed transform (identity)
- `base_link` → `*_wheel_link`: Published by joint state publisher (rotating)

## Usage Examples

### 1. Process XACRO to URDF

Convert the XACRO file to pure URDF:

```bash
# Install xacro if not already installed
sudo apt install ros-humble-xacro

# Process XACRO file
ros2 run xacro xacro \
  /path/to/scout_description/urdf/scout_mini.xacro \
  > scout_mini.urdf

# Or use xacro command directly
xacro scout_mini.xacro > scout_mini.urdf
```

### 2. Visualize in RViz

View the robot model without simulation:

```bash
# Launch RViz with robot model
ros2 launch scout_description view_robot.launch.py
```

*Note: You may need to create a simple launch file or use joint_state_publisher_gui:*

```bash
# Terminal 1: Publish robot description
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$(xacro /path/to/scout_mini.xacro)"

# Terminal 2: GUI for joint control
ros2 run joint_state_publisher_gui joint_state_publisher_gui

# Terminal 3: Launch RViz
ros2 run rviz2 rviz2 -d /path/to/scout_description/rviz/scout_mini.rviz
```

### 3. Check URDF Validity

Validate the URDF structure:

```bash
# Install urdf tools
sudo apt install ros-humble-urdfdom ros-humble-urdf-tutorial

# Check URDF validity
check_urdf scout_mini.urdf

# View robot model info
urdf_to_graphviz scout_mini.urdf
```

### 4. Inspect Robot Model

```bash
# View TF tree
ros2 run tf2_tools view_frames

# Echo robot description
ros2 topic echo /robot_description --once

# List all links and joints
ros2 run xacro xacro scout_mini.xacro | grep -E '<link name=|<joint name='
```

## Integration with Gazebo

This package is designed to work with `scout_gazebo_sim` package for full simulation.

### Required Gazebo Packages

```bash
sudo apt install \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-bridge \
  ros-humble-ros-gz-image \
  ignition-fortress
```

### Launch Simulation

```bash
# Build workspace
cd /path/to/ros2_ws
colcon build --packages-select scout_description scout_gazebo_sim

# Source workspace
source install/setup.bash

# Launch Gazebo simulation
ros2 launch scout_gazebo_sim scout_mini_empty_world.launch.py
```

See `scout_gazebo_sim` package for detailed simulation instructions.

## Mesh Files

### scout_mini_base_link.dae

- **File Size:** 14 MB
- **Format:** COLLADA (.dae)
- **Description:** High-detail 3D mesh of Scout Mini chassis
- **Usage:** Both visual and collision representation
- **Coordinate Frame:** Requires rotation (rpy="1.57 0 -1.57") to align with ROS convention

### wheel.dae

- **File Size:** 2.7 MB
- **Format:** COLLADA (.dae)
- **Description:** 3D mesh of Scout Mini wheel
- **Usage:** Visual and collision for all 4 wheels
- **Instances:** Referenced 4 times (one per wheel)

## RViz Configuration

### scout_mini.rviz

Default RViz configuration file includes:

- **Robot Model**: Display URDF with TF frames
- **TF**: Show coordinate frame tree
- **Grid**: Ground plane reference
- **Camera**: Pre-configured viewpoint
- **Laser Scan**: If LiDAR sensor is added
- **Camera Image**: If RGB camera is added

**Display Settings:**
- Fixed Frame: `odom`
- Target Frame: `base_link`
- Grid Cell Size: 1.0 m

## Customization

### Adding Sensors

You can extend the XACRO file to add sensors:

#### Example: Adding a LiDAR

```xml
<!-- Add after wheel definitions -->
<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder radius="0.05" length="0.07"/>
    </geometry>
    <material name="black"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.05" length="0.07"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.5"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>

<joint name="lidar_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0.2 0.0 0.15" rpy="0 0 0"/>
</joint>

<!-- Gazebo plugin for LiDAR -->
<gazebo reference="lidar_link">
  <sensor name="lidar_sensor" type="gpu_lidar">
    <topic>scan</topic>
    <update_rate>10</update_rate>
    <lidar>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
      </range>
    </lidar>
  </sensor>
</gazebo>
```

#### Example: Adding an IMU

```xml
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="base_link"/>
  <child link="imu_link"/>
  <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
</joint>

<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <topic>imu</topic>
    <update_rate>100</update_rate>
  </sensor>
</gazebo>
```

### Modifying Physical Properties

To change robot mass or dimensions, edit the XACRO file:

```xml
<!-- Change base mass -->
<inertial>
  <mass value="150.0" />  <!-- Increase from 132.39 kg -->
  <!-- Recalculate inertia tensor accordingly -->
</inertial>

<!-- Change wheel mass -->
<link name="front_right_wheel_link">
  <inertial>
    <mass value="4.0" />  <!-- Increase from 3 kg -->
    <!-- Update inertia values -->
  </inertial>
</link>
```

**Important:** When changing masses, you should recalculate inertia tensors for realistic physics simulation.

## Performance Characteristics

### Mobility

**Turning Radius:**
```
Minimum turning radius (in-place turn): 0 m (skid-steering)
Forward turning radius (typical): ~0.5-1.0 m
```

**Speed Capabilities** (estimated from wheel parameters):
```
Max Linear Speed: ~0.8 m/s (2.88 km/h)
Max Angular Speed: ~3.27 rad/s (187°/s)
Max Acceleration: 1.0 m/s²
```

**Terrain Capability:**
```
Ground Clearance: 10 cm
Wheel Diameter: 32 cm (2 × radius)
Step Height (estimated): ~8 cm (80% of wheel radius)
```

### Power and Torque

```
Max Wheel Torque: 20 Nm per wheel
Total Driving Torque: 80 Nm (4 wheels)
Wheel Power (estimated): P = τ × ω_max ≈ 20 × 5 = 100 W per wheel
```

## Troubleshooting

### Common Issues

#### Issue 1: Mesh files not found

**Error:**
```
[ERROR] Could not find mesh file: package://scout_description/meshes/...
```

**Solution:**
```bash
# Ensure package is properly installed
colcon build --packages-select scout_description
source install/setup.bash

# Check if meshes exist
ls install/scout_description/share/scout_description/meshes/
```

#### Issue 2: Robot falls through ground in Gazebo

**Error:** Robot spawns and immediately falls

**Solution:** Check spawn height in launch file. Default is `z=0.245` to account for wheel radius.

```bash
# Correct spawn height
ros2 launch scout_gazebo_sim spawn_scout_mini.launch.py z_pose:=0.245
```

#### Issue 3: Wheels not rotating

**Error:** Robot doesn't move when cmd_vel is published

**Solution:**
1. Check if differential drive plugin is loaded
2. Verify joint names match between URDF and plugin
3. Ensure ROS-Gazebo bridge is running

```bash
# Check active topics
ros2 topic list | grep cmd_vel

# Test velocity command
ros2 topic pub /scout_mini/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2}, angular: {z: 0.0}}" --once
```

#### Issue 4: High CPU usage with mesh collision

**Solution:** Meshes are detailed (14 MB base, 2.7 MB wheels). For better performance, consider using primitive shapes for collision:

```xml
<collision>
  <geometry>
    <box size="0.65 0.49 0.28"/>  <!-- Use box instead of mesh -->
  </geometry>
</collision>
```

## Technical Specifications Summary

| **Category** | **Parameter** | **Value** |
|--------------|---------------|-----------|
| **Dimensions** | Length | ~0.650 m |
| | Width | ~0.490 m |
| | Height | ~0.440 m (total) |
| | Wheelbase | 0.464 m |
| | Track Width | 0.417 m |
| | Ground Clearance | 0.100 m |
| **Mass** | Base Mass | 132.39 kg |
| | Wheel Mass (each) | 3 kg |
| | Total Mass | 144.39 kg |
| **Wheels** | Radius | 0.160 m |
| | Diameter | 0.320 m |
| | Wheel Separation | 0.490 m |
| | Number of Wheels | 4 |
| **Performance** | Max Wheel Torque | 20 Nm |
| | Max Linear Acceleration | 1.0 m/s² |
| | Estimated Max Speed | ~0.8 m/s |
| | Estimated Max Angular Speed | ~3.27 rad/s |
| **Sensors** | Joint States | 50 Hz |
| | Odometry | 50 Hz |
| | LiDAR (optional) | Configurable |
| | Camera (optional) | Configurable |
| **Control** | Drive Type | Skid-steering (4WD) |
| | Input Topic | `cmd_vel` |
| | Output Topics | `odom`, `tf`, `joint_states` |

## References

- **AgileX Robotics Official**: [https://www.agilex.ai/](https://www.agilex.ai/)
- **Scout Mini Product Page**: Check manufacturer's website for official specifications
- **ROS2 URDF Tutorials**: [https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)
- **Ignition Gazebo Documentation**: [https://gazebosim.org/docs](https://gazebosim.org/docs)

## Contributing

To improve this robot description:

1. **Validate dimensions** against real Scout Mini hardware
2. **Refine inertia tensors** based on actual CAD data
3. **Add sensor models** (LiDAR, cameras, IMU, GPS)
4. **Optimize mesh files** for better performance
5. **Add material properties** for realistic visual appearance

## License

This package is released under the **Apache License 2.0**.

```
Copyright 2024 Mattia Dutto

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Contact

**Maintainer:** Mattia Dutto
**Email:** mattia.dutto@polito.it

For issues and questions regarding this package, please refer to the parent repository or contact the maintainer.

---

**Last Updated:** 2025-11-17
**ROS2 Version:** Humble Hawksbill
**Gazebo Version:** Ignition Fortress
