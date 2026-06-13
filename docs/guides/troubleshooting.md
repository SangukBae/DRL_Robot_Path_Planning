# Troubleshooting

## 환경 변수

```bash
source /opt/ros/humble/setup.bash
source <repo>/ros2_ws/install/setup.bash

# DDS (Docker 또는 멀티머신 환경)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://<repo>/cyclonedds_config.xml

# RGL LiDAR (launch 파일이 자동 설정. 수동 실행 시)
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=~/DRL_Robot_Path_Planning/third_party/rgl/RGLGazeboPlugin/install/RGLServerPlugin
```

---

## 디버깅 명령어

```bash
# 토픽 Hz 확인
ros2 topic hz /ouster/points    # 목표: ~10 Hz
ros2 topic hz /scan             # 목표: ~10 Hz
ros2 topic hz /odometry         # 목표: ~50 Hz

# TF 트리 확인
ros2 run tf2_tools view_frames

# 환경 서비스 확인
ros2 service list | grep -E "reset|step|dimensions"

# GPU/CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

---

## 흔한 오류

| 문제 | 원인 | 해결 |
|------|------|------|
| Service timeout | Gazebo가 먼저 실행되지 않음 | Gazebo 완전 기동 후 에이전트 실행 |
| `/odometry` 없음 | ros_gz_bridge 미실행 또는 브릿지 설정 오류 | `ros2_gz_bridge_config.yaml` 확인 |
| `filtered_cmd_v_mps=0` in CSV | 정상: step sleep 구간에서의 오독, 실제 문제 아님 | 무시 |
| 시뮬레이션 RTF ~0.001× | RGL 플러그인 미로드 → CPU 폴백 | `IGN_GAZEBO_SYSTEM_PLUGIN_PATH` 확인 |
| rosdep 충돌 | graphicsmagick 패키지 의존성 충돌 | `--skip-keys` 옵션 사용 (Quick Install 참고) |
| Docker DDS 문제 | DDS 멀티캐스트 브릿지 불가 | `--network host` + `rmw_cyclonedds_cpp` 사용 |

---

## RGL LiDAR 로드 확인

```bash
# Gazebo 실행 후 로그에서 확인
ros2 launch hunter_se_gazebo simulate_hunter_se_ignition.launch.py rviz:=false 2>&1 | grep -i rgl
```

RGL이 정상 로드되면 `/ouster/points`가 ~10 Hz로 발행된다. 로드 실패 시 CPU 폴백으로 RTF가 급감한다.

---

## 커리큘럼 학습 재개

학습이 중단된 경우:

1. `train_tqc_config.yaml`의 `train_settings` 블록에서 `load_model: true` 설정 (`train_tqc_curriculum_config.yaml` 아님)
2. `base_file_name`과 `seed`가 이전 학습과 동일한지 확인 — 체크포인트 탐색이 이 두 값으로 `pytorch_models_dir`에서 가장 최근 파일을 찾으므로, 달라지면 같은 `run_dir`라도 로드 실패
3. `<run_dir>/logs/curriculum_state.json` 존재 여부 확인
   - 존재하면: 모델 가중치 + 리플레이 버퍼 + 커리큘럼 스테이지 모두 복원
   - 없으면: 모델 가중치·버퍼만 복원, 커리큘럼 스테이지는 0부터 재시작
4. 동일 `run_dir`로 재실행

---

## LIO-SAM 설정 참고

`LIO-SAM/config/params.yaml` 주요 값:

```yaml
imuTopic: "/scout/imu"
pointCloudTopic: "/points"
lidarFrame: "os_lidar"
baselinkFrame: "base_footprint"
sensor: ouster
N_SCAN: 64
Horizon_SCAN: 1024
extrinsicTrans: [0, 0, -0.42]
```
