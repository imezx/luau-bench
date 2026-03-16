from __future__ import annotations

import json
from pathlib import Path

import pytest

from luau_bench.evaluator import BenchmarkRun, TaskResult


def _make_run(
    model_id: str = "test-model",
    task_results: list[TaskResult] | None = None,
    composite_score: float | None = None,
    composite_se: dict | None = None,
) -> BenchmarkRun:
    run = BenchmarkRun(
        run_id="abc12345",
        model_id=model_id,
        provider="mock",
        task_results=task_results or [],
        finished_at=0.0,
        composite_score=composite_score,
        composite_se=composite_se,
    )
    run.started_at = 0.0
    return run


def _make_task(name: str, metrics: dict, primary: str | None = None, error: str | None = None):
    return TaskResult(
        task_name=name,
        version=1.0,
        num_docs=5,
        metrics=metrics,
        error=error,
        primary_metric=primary,
    )


class TestReporterNoSideEffects:
    def test_init_does_not_create_dir(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter

        output = tmp_path / "results"
        Reporter(output_dir=str(output))
        assert not output.exists()

    def test_save_json_creates_dir(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter

        output = tmp_path / "nested" / "results"
        r = Reporter(output_dir=str(output))
        run = _make_run(task_results=[_make_task("t1", {"exact_match": 100.0})])
        path = r.save_json(run)
        assert output.exists()
        assert path.exists()

    def test_save_markdown_creates_dir(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter

        output = tmp_path / "md_out"
        r = Reporter(output_dir=str(output))
        run = _make_run(task_results=[_make_task("t1", {"exact_match": 75.0})])
        path = r.save_markdown(run)
        assert output.exists()
        assert path.exists()

    def test_save_html_creates_dir(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter

        output = tmp_path / "html_out"
        r = Reporter(output_dir=str(output))
        run = _make_run(task_results=[_make_task("t1", {"exact_match": 50.0})])
        path = r.save_html(run)
        assert output.exists()
        assert path.exists()


class TestMarkdownReport:
    def test_contains_percent_values(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter

        r = Reporter(output_dir=str(tmp_path))
        run = _make_run(
            task_results=[
                _make_task("my_task", {"exact_match": 66.67, "luau_exec_pass_rate": 80.0})
            ]
        )
        text = r.save_markdown(run).read_text()
        assert "66.7%" in text
        assert "80.0%" in text

    def test_contains_composite_section(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter

        r = Reporter(output_dir=str(tmp_path))
        run = _make_run(
            task_results=[_make_task("t1", {"exact_match": 100.0}, primary="exact_match")],
            composite_score=100.0,
            composite_se={"ci_lower": 95.0, "ci_upper": 100.0},
        )
        text = r.save_markdown(run).read_text()
        assert "Composite Score" in text
        assert "100.00%" in text
        assert "95%" in text

    def test_no_composite_section_when_absent(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter

        r = Reporter(output_dir=str(tmp_path))
        run = _make_run(task_results=[_make_task("t1", {"exact_match": 50.0})])
        text = r.save_markdown(run).read_text()
        assert "Composite Score" not in text

    def test_error_task_shown(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter

        r = Reporter(output_dir=str(tmp_path))
        run = _make_run(task_results=[_make_task("bad_task", {}, error="timeout")])
        text = r.save_markdown(run).read_text()
        assert "bad_task" in text
        assert "ERROR" in text


class TestJsonReport:
    def test_json_structure(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter

        r = Reporter(output_dir=str(tmp_path))
        run = _make_run(
            task_results=[_make_task("t1", {"exact_match": 80.0}, primary="exact_match")],
            composite_score=80.0,
        )
        data = json.loads(r.save_json(run).read_text())
        assert data["run_id"] == "abc12345"
        assert data["model_id"] == "test-model"
        assert "composite_score" in data
        assert data["composite_score"] == pytest.approx(80.0)
        assert "t1" in data["tasks"]
        assert data["tasks"]["t1"]["exact_match"] == pytest.approx(80.0)

    def test_json_samples_included_when_present(self, tmp_path):
        from luau_bench.reporting.reporter import Reporter
        from luau_bench.evaluator import DocResult

        r = Reporter(output_dir=str(tmp_path))
        tr = _make_task("t1", {"exact_match": 100.0})
        tr.doc_results = [DocResult(doc={"q": "hi"}, prediction="hello", reference="hello")]
        run = _make_run(task_results=[tr])
        data = json.loads(r.save_json(run).read_text())
        assert "samples" in data["tasks"]["t1"]
        assert data["tasks"]["t1"]["samples"][0]["prediction"] == "hello"
