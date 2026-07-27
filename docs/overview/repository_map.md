# Repository Map

이 문서는 **어디에 무엇이 있는지** 빠르게 찾기 위한 지도다. 자세한 동작은 각 design/reference 문서를 본다.

## ROS2 패키지 (`ros2_ws/src/`)
| 패키지 | 설명 |
|--------|------|
| `drl_agent` | DRL 환경 · 정책 · 학습/평가 스크립트 (핵심) |
| `drl_agent_interfaces` | ROS2 서비스/메시지 정의 (`srv/`) |
| `hunter_se_gazebo` | Hunter SE URDF, Gazebo launch, worlds, cmd prefilter |
| `drl_obstacle_assets` | Gazebo 장애물 모델 라이브러리(38종) + 카탈로그 |
| `ouster_simulation/ouster_description` | OS1-64 LiDAR (RGL GPU 플러그인) |
| `aws-robomaker-*-world` | hospital / bookstore / small-house / small-warehouse 월드 |

## `drl_agent` 내부 (`drl_agent/drl_agent/` — importable 패키지, 구현의 유일한 위치)
| 폴더 | 내용 |
|------|------|
| `env/` | 환경 노드 (`simulation/environment.py`, `curriculum/environment_curriculum.py`, `environment_interface.py`, observation/rewards/spawning/humans mixin) |
| `training/` + `training/baselines/` | 알고리즘 트레이너 (TQC: `train_tqc_curriculum.py`; baseline: `baselines/<algo>_curriculum.py`, …) |
| `rl/algorithms/` | 알고리즘 네트워크/에이전트 (`tqc/agent.py`, …) |
| `evaluation/` | 리플레이 지표, 로깅 헬퍼, 플롯, live-sim 실행, 후처리 분석 |
| `config/` (drl_agent/config/, 별도) | 모든 YAML 설정 |

> **주의**: 옛 `scripts/{policy,environment,utils}/` flat 레거시 디렉터리는 삭제됐다.
> bare-name import(`import tqc_agent` 등)는 더 이상 동작하지 않는다 — `import
> drl_agent.<...>` dotted import만 유효하다. 상세: [package_structure](package_structure.md)

## 알고리즘 (`drl_agent/rl/algorithms/` + `drl_agent/training/`)
| 알고리즘 | 계열 | 파일 | 특징 |
|---------|------|------|------|
| **TQC** (주력) | off-policy AC | `rl/algorithms/tqc/agent.py` + `training/train_tqc_curriculum.py` | 분위수 분포 추정 + 상위 truncation으로 과대추정 완화. LAP 버퍼. aux/커리큘럼 결합 |
| TQC+IEQn | off-policy AC | `rl/algorithms/tqc_ieqn/agent.py` + `training/baselines/tqc_ieqn_curriculum.py` | TQC + 부등식(안전) 제약 |
| TD7 | off-policy AC | `rl/algorithms/td7/agent.py` + `training/baselines/td7_curriculum.py` | 상태-액션 표현 학습 + 체크포인트 정책 선택 |
| SAC | off-policy AC | `rl/algorithms/sac/agent.py` + `training/baselines/sac_curriculum.py` | 최대 엔트로피, 자동 온도 |
| SB3-SAC / SB3-TD3 / SB3-PPO | off-policy/on-policy AC | `rl/algorithms/sb3/{sac,td3,ppo}.py` + `training/baselines/sb3_*[_curriculum].py` | Stable-Baselines3 baseline (PPO는 curriculum 변형 없음) |
| A3C | on-policy PG | `rl/algorithms/a3c/agent.py` + `training/baselines/a3c_curriculum.py` | 비동기 업데이트(샘플 효율 낮음) |

모든 비교군은 각각의 canonical 위치에 그대로 보존되어 있으며(삭제된 모델 없음),
동일한 커리큘럼/평가 프로토콜을 공유한다. → [experiment_protocol](../experiments/experiment_protocol.md)

## 주요 유틸 (`drl_agent/rl/`, `drl_agent/training/`, `drl_agent/common/`, `drl_agent/evaluation/`)
| 파일 | 설명 |
|------|------|
| `rl/replay/buffer.py` | LAP 리플레이 버퍼 (npz 계약: `rl/replay/schema.py`) |
| `training/episode_metrics.py` | per-episode 지표(SPL/STL/CTE/…) + paper CSV |
| `training/aux_eval_metrics.py` | aux 정식 평가 지표(RMSE/MAE/F1) |
| `training/aux_ablation_logging.py` | run-identity / eval-summary / manifest 로깅 |
| `common/file_manager.py` | 체크포인트, YAML 로드 |
| `evaluation/analysis/plot_*.py` | 학습/보상/궤적 시각화 |

## 학습 산출물 (`runtime/<algo>/`)
| 경로 | 내용 |
|------|------|
| `logs/` | 에피소드/평가 CSV, TensorBoard, `curriculum_state.json` |
| `pytorch_models/` | 학습 중 체크포인트 |
| `final_models/` | 최종 모델 |

## Where in code
- 패키지 루트: `ros2_ws/src/`
- 설정: `ros2_ws/src/drl_agent/config/` → [config_reference](../reference/config_reference.md)
