# Base: CUDA 11.8 + Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# 1) 기본 패키지
# - swig: Gymnasium(stable-baselines3 의존성)의 일부 환경 빌드에 필요
# - libopenvdb-dev: scout_nav2의 spatio-temporal-voxel-layer 서브모듈 빌드에 필요
# - tree: 디렉토리 구조 시각화
# - gedit: GUI 텍스트 에디터
RUN apt-get update && apt-get install -y \
    locales curl gnupg2 software-properties-common \
    build-essential cmake git \
    libomp-dev \
    libpython3.10-dev \
    python3 python3-dev python3-pip python3-distutils python3-venv \
    libgl1-mesa-glx x11-apps mesa-utils \
    swig \
    libopenvdb-dev \
    tree \
    gedit \
 && rm -rf /var/lib/apt/lists/*

RUN locale-gen ko_KR.UTF-8 && update-locale LC_ALL=ko_KR.UTF-8 LANG=ko_KR.UTF-8
ENV LANG=ko_KR.UTF-8

# 2) ROS2 APT 설정
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

# 3) ROS2 + Gazebo(Ignition Fortress) + scout_nav2 의존성
#  
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-dev-tools \
    ignition-fortress \
    ros-humble-ros-gz \
    ros-humble-navigation2 \
    ros-humble-slam-toolbox \
    ros-humble-pointcloud-to-laserscan \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-rviz2 \
    ros-humble-teleop-twist-keyboard \
 && rm -rf /var/lib/apt/lists/*

# rosdep 초기화
RUN rosdep init || true && rosdep update

# Python 편의 링크 / pip 업그레이드
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python -m pip install --upgrade pip

# 4) 호스트의 drl_path 전체 복사
#    (이 Dockerfile을 빌드하는 위치에 DRL_Robot_Path_Planning 디렉토리가 있어야 함)
WORKDIR /root
COPY . /root/DRL_Robot_Path_Planning

# 5) PyTorch + stable-baselines3 의존성 + requirements
WORKDIR /root/DRL_Robot_Path_Planning
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118

# stable-baselines3 의존성 (gymnasium 및 [common]의존성) 설치
#
RUN pip install --no-cache-dir \
    "gymnasium[all]>=0.29.1,<1.3.0" \
    pandas \
    matplotlib \
    tensorboard

# 사용자의 추가 라이브러리 (위에서 설치한 것과 중복될 수 있음)
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# (선택) 3D 포인트클라우드 관련 작업 시 필요
# RUN pip3 install spconv-cu118
# RUN pip3 install torch-scatter

# 6) rosdep 의존성 설치 + colcon 빌드 (ROS2 워크스페이스)
WORKDIR /root/DRL_Robot_Path_Planning/ros2_ws

# jammy에서 패키지명이 libgraphicsmagick++-dev 이므로 먼저 설치
RUN apt-get update && apt-get install -y \
    libgraphicsmagick++-dev \
    graphicsmagick-imagemagick-compat \
 && rm -rf /var/lib/apt/lists/*

# rosdep DB 업데이트
RUN rosdep update

# ✅ 여기서 apt-get update + rosdep install 을 같은 레이어에서 수행
RUN apt-get update && \
    /bin/bash -lc "source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src -yi --rosdistro humble \
    --skip-keys='libgraphicsmagick++1-dev graphicsmagick-libmagick-dev-compat nav2 ign gazebo'" && \
    rm -rf /var/lib/apt/lists/*

# RUN /bin/bash -lc "source /opt/ros/humble/setup.bash && \
#      colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release"

# --- 신규 단계 ---
# 6.5) Git Clone한 stable-baselines3 설치 (편집 가능 모드)
#
# /root/DRL_Robot_Path_Planning/stable-baselines3 에 위치한다고 가정하고
# 수동으로 pip -e 설치를 실행합니다.
RUN /bin/bash -lc "source /opt/ros/humble/setup.bash && \
     pip install -e /root/DRL_Robot_Path_Planning/stable-baselines3"
# --- 신규 단계 종료 ---

# 7) 런타임 환경변수 설정
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc
# 빌드된 워크스페이스 source 추가
# RUN echo 'source /root/DRL_Robot_Path_Planning/ros2_ws/install/setup.bash' >> /root/.bashrc && \
#     echo 'export DRL_AGENT_SRC_PATH=/root/DRL_Robot_Path_Planning/ros2_ws/src' >> /root/.bashrc
RUN echo 'export DRL_AGENT_SRC_PATH=/root/DRL_Robot_Path_Planning/ros2_ws/src' >> /root/.bashrc

# 8) CUDA 환경 변수 설정
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDNN_INCLUDE_DIR=/usr/local/cuda/include
ENV CUDNN_LIB_DIR=/usr/local/cuda/lib64

# 기본 작업 디렉토리 = 워크스페이스 루트
WORKDIR /root/DRL_Robot_Path_Planning