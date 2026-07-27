# Evaluation

학습된 정책을 **평가하고 결과를 모으는 방법**. (학습 실행은 [training.md](training.md), 지표 정의는 [../reference/metrics_reference.md](../reference/metrics_reference.md))

## 1. 학습 중 자동 평가
- trainer가 `eval_freq` 스텝마다 탐험 없이 N 에피소드를 평가한다.
- 성공률/충돌률/SPL/STL 등을 집계하고, 기준 통과 시 커리큘럼 stage를 올린다.
- 맵 커리큘럼이 켜져 있으면 **맵 타입별** 결과도 따로 남긴다(`curriculum_eval_per_map_*.csv`).
- aux가 켜져 있으면 **aux 평가 지표**(RMSE/MAE/peak-acc/F1)도 같이 집계.

출력(아래 경로의 `logs/`):
| 파일 | 내용 |
|--|--|
| `eval_metrics_*.csv` | 평가별 종합 지표(paper) |
| `eval_summary_*.csv` | aux on/off 비교 요약 |
| `curriculum_eval_per_map_*.csv` | 맵 타입별 분해 |

## 2. 다중 seed 결과 집계
```bash
python3 -m drl_agent.evaluation.analysis.aggregate_results \
  --runtime-root ros2_ws/src/drl_agent/runtime
# 또는: ros2 run drl_agent aggregate_results.py --runtime-root ros2_ws/src/drl_agent/runtime
```
→ seed별 `eval_metrics_*.csv`를 mean±std 표 / 학습곡선 / sample-efficiency로 정리.

## 3. 일반화 평가 (재학습 없이 stage/world 별 평가) — 현재 TQC 전용
```bash
ros2 run drl_agent generalization_eval.py --ros-args \
  -p weight_prefix:=<model_prefix> -p weights_dir:=<run_dir>/final_models \
  -p world:=aws_hospital -p eval_eps_override:=20
```
- `weights_dir` 미지정 시 해당 run의 `pytorch_models/`를 본다.
- 다른 알고리즘은 같은 패턴으로 agent 클래스만 바꿔 확장.

## 4. 테스트 launch (정성 확인)
```bash
ros2 launch drl_agent test_tqc.launch.py     # TQC
ros2 launch drl_agent test_td7.launch.py      # TD7
```

## 더 보기
- 비교 실험 프로토콜·CSV 스키마: [../experiments/experiment_protocol.md](../experiments/experiment_protocol.md)
- aux ablation 로깅: [../experiments/aux_ablation_logging.md](../experiments/aux_ablation_logging.md)
- 지표 정의(SPL/STL/PSC/H-Coll/aux): [../reference/metrics_reference.md](../reference/metrics_reference.md)
