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

## `drl_agent` 내부 (`scripts/`)
| 폴더 | 내용 |
|------|------|
| `environment/` | 환경 노드 (`environment.py`, `environment_curriculum.py`, `environment_interface.py`, aux 라벨 생성) |
| `policy/` | 알고리즘 + 학습 스크립트 (`tqc_agent.py`, `train_tqc_curriculum_agent.py`, …) |
| `utils/` | 리플레이 버퍼, 지표, 로깅 헬퍼, 플롯 |
| `config/` (별도) | 모든 YAML 설정 |

## 알고리즘 (`scripts/policy/`)
| 알고리즘 | 계열 | 파일 | 특징 |
|---------|------|------|------|
| **TQC** (주력) | off-policy AC | `tqc_agent.py` + `train_tqc_curriculum_agent.py` | 분위수 분포 추정 + 상위 truncation으로 과대추정 완화. LAP 버퍼. aux/커리큘럼 결합 |
| TQC+IEQn | off-policy AC | `tqc_ieqn_agent.py` | TQC + 부등식(안전) 제약 |
| TD7 | off-policy AC | `td7_agent.py` | 상태-액션 표현 학습 + 체크포인트 정책 선택 |
| SAC | off-policy AC | `sac_agent.py` | 최대 엔트로피, 자동 온도 |
| SB3-SAC / SB3-TD3 | off-policy AC | `sb3_*_agent.py` | Stable-Baselines3 baseline |
| A3C | on-policy PG | `a3c_agent.py` | 비동기 업데이트(샘플 효율 낮음) |

> 6개 비교군이 동일한 커리큘럼/평가 프로토콜을 공유한다. → [experiment_protocol](../experiments/experiment_protocol.md)

## 주요 유틸 (`scripts/utils/`)
| 파일 | 설명 |
|------|------|
| `buffer.py` | LAP 리플레이 버퍼 |
| `episode_metrics.py` | per-episode 지표(SPL/STL/CTE/…) + paper CSV |
| `aux_eval_metrics.py` | aux 정식 평가 지표(RMSE/MAE/F1) |
| `aux_ablation_logging.py` | run-identity / eval-summary / manifest 로깅 |
| `file_manager.py` | 체크포인트, YAML 로드 |
| `plot_*.py` | 학습/보상/궤적 시각화 |

## 학습 산출물 (`runtime/<algo>/`)
| 경로 | 내용 |
|------|------|
| `logs/` | 에피소드/평가 CSV, TensorBoard, `curriculum_state.json` |
| `pytorch_models/` | 학습 중 체크포인트 |
| `final_models/` | 최종 모델 |

## Where in code
- 패키지 루트: `ros2_ws/src/`
- 설정: `ros2_ws/src/drl_agent/config/` → [config_reference](../reference/config_reference.md)
