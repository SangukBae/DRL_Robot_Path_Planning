# Installation

## 사전 요구사항

- Ubuntu 22.04
- ROS2 Humble ([설치 가이드](https://docs.ros.org/en/humble/Installation.html))
- Gazebo Ignition Fortress
- CUDA 11.8 + PyTorch 2.4.1
- Python 3.10

---

## 1. 의존성 설치

```bash
cd ros2_ws
rosdep install --from-paths src -yi --rosdistro humble \
  --skip-keys='libgraphicsmagick++1-dev graphicsmagick-libmagick-dev-compat'
```

Python 패키지:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3 tensorboard pyyaml
```

---

## 2. RGL LiDAR Plugin

Ouster OS1-64 LiDAR는 Robotec GPU Lidar(RGL) 플러그인으로 GPU 가속 포인트 클라우드를 생성한다.

```bash
# 플러그인 설치 확인
ls ~/DRL_Robot_Path_Planning/third_party/rgl/RGLGazeboPlugin/install/

# 환경 변수 설정 (launch 파일이 자동 설정하지만 수동 설정 시)
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=~/DRL_Robot_Path_Planning/third_party/rgl/RGLGazeboPlugin/install/RGLServerPlugin
```

> RGL 플러그인이 없으면 CPU 폴백으로 동작하며 시뮬레이션 RTF가 ~0.001×로 급감한다.

---

## 3. 빌드

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash

# 전체 빌드
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# 메모리 제한 환경
MAKEFLAGS="-j4" colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# 단일 패키지
colcon build --packages-select drl_agent

# 클린 빌드
rm -rf build/ install/ log/ && colcon build

# 소스
source install/setup.bash
```

> 커스텀 메시지(`drl_agent_interfaces`) 변경 후에는 클린 빌드 필요:
> ```bash
> colcon build --packages-select drl_agent_interfaces --cmake-clean-first
> ```

---

## 4. Docker

### 이미지 빌드

```bash
docker build -t drl_path_planning .
```

### 컨테이너 실행 (GPU)

```bash
docker run --gpus all -it --rm \
  --network host \
  -e DISPLAY=$DISPLAY \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/root/DRL_Robot_Path_Planning \
  drl_path_planning
```

### X11 허용 (호스트)

```bash
xhost +local:
```

> Docker 내부 install 경로: `/root/DRL_Robot_Path_Planning/ros2_ws/install/...`

---

## 5. 환경 변수

```bash
source /opt/ros/humble/setup.bash
source <repo>/ros2_ws/install/setup.bash

# DDS (Docker 또는 멀티머신 환경)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://<repo>/cyclonedds_config.xml
```
