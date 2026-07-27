"""2026-07 review regression: constructing PaperMetricsCSV / DynamicAvoidanceCSV
/ EvalSummaryCSV a SECOND time against the same path (simulating a resume into
an existing RUN_LAYOUT new-structure run dir, where the CSV filename is plain
and therefore identical across process restarts -- see run_layout.tagged_
filename) must never truncate the file's existing rows.

Each writer's constructor calls run_layout.write_csv_header_if_new instead of
unconditionally opening in "w" mode; these tests pin that contract directly on
the real writer classes (not just the underlying helper).
"""

import drl_agent.training.episode_metrics as em
import drl_agent.training.dynamic_avoidance_log as dal
import drl_agent.training.aux_ablation_logging as aux_log


class _FakeAgent:
    aux_enabled = False
    aux_cfg = None


def test_paper_metrics_csv_second_construction_does_not_truncate(tmp_path):
    log_dir = str(tmp_path)
    first = em.PaperMetricsCSV(log_dir, "")  # "" run_tag == RUN_LAYOUT new-structure
    first.write_episode(
        episode=1, global_t=100, stage=0, success=True, collision=False,
        timeout=False, total_reward=1.23, steps=50,
        metrics={c: 0.0 for c in em.METRIC_COLUMNS},
    )
    with open(first.episode_path) as f:
        before = f.read()
    assert before.count("\n") == 2  # header + 1 data row

    # Simulate a resumed process re-constructing the SAME writer at the SAME
    # (plain, run_tag="") path.
    second = em.PaperMetricsCSV(log_dir, "")
    assert second.episode_path == first.episode_path
    with open(second.episode_path) as f:
        after_construct = f.read()
    assert after_construct == before, "resume construction must not truncate prior rows"

    second.write_episode(
        episode=2, global_t=200, stage=0, success=False, collision=True,
        timeout=False, total_reward=-1.0, steps=30,
        metrics={c: 0.0 for c in em.METRIC_COLUMNS},
    )
    with open(second.episode_path) as f:
        final = f.read()
    assert final.count("\n") == 3  # header + 2 data rows, both episodes present
    assert "1.23" in final and "-1.0" in final


def test_dynamic_avoidance_csv_second_construction_does_not_truncate(tmp_path):
    log_dir = str(tmp_path)
    first = dal.DynamicAvoidanceCSV(log_dir, "")
    with open(first.path, "a", newline="") as f:
        f.write("sentinel_row_from_first_run\n")

    second = dal.DynamicAvoidanceCSV(log_dir, "")
    assert second.path == first.path
    with open(second.path) as f:
        content = f.read()
    assert "sentinel_row_from_first_run" in content


def test_eval_summary_csv_second_construction_does_not_truncate(tmp_path):
    log_dir = str(tmp_path)
    first = aux_log.EvalSummaryCSV(log_dir, "", seed=0, agent=_FakeAgent())
    first.append(
        eval_global_t=1000, curriculum_stage=0, eval_eps=10,
        metrics={"success_rate": 0.5},
    )
    with open(first.path) as f:
        before = f.read()
    assert before.count("\n") == 2  # header + 1 row

    second = aux_log.EvalSummaryCSV(log_dir, "", seed=0, agent=_FakeAgent())
    assert second.path == first.path
    with open(second.path) as f:
        assert f.read() == before, "resume construction must not truncate prior rows"
