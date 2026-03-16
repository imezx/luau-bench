import asyncio

import pytest

import luau_bench.api.metrics
import luau_bench.api.filters

from luau_bench.api.task import ConfigurableTask, parse_task_config
from luau_bench.models import GenerationResult, ModelAdapter, ModelConfig


class MockAdapter(ModelAdapter):
    """A fake model adapter that returns a canned response."""

    def __init__(self, response: str = "hello"):
        config = ModelConfig(provider="mock", model_name="mock-model")
        super().__init__(config)
        self.response = response
        self.call_count = 0

    async def generate(self, system: str, user: str) -> GenerationResult:
        self.call_count += 1
        return GenerationResult(
            text=self.response,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
        )


class TestEvaluator:
    def _make_task(self, **overrides):
        raw = {
            "task": "eval_test",
            "system_prompt": "Answer concisely.",
            "doc_to_text": "{{ question }}",
            "doc_to_target": "{{ answer }}",
            "filters": [{"name": "strip_whitespace"}],
            "metric_list": [
                {"metric": "exact_match"},
                {"metric": "contains"},
            ],
            "docs": [
                {"question": "What is 1+1?", "answer": "2"},
                {"question": "What is 2+2?", "answer": "4"},
            ],
        }
        raw.update(overrides)
        return ConfigurableTask(parse_task_config(raw))

    def test_evaluate_basic(self):
        from luau_bench.evaluator import evaluate

        task = self._make_task()
        adapter = MockAdapter(response="  2  ")

        result = asyncio.run(
            evaluate(
                tasks=[task],
                adapter=adapter,
                quiet=True,
            )
        )

        assert len(result.task_results) == 1
        tr = result.task_results[0]
        assert tr.task_name == "eval_test"
        assert tr.num_docs == 2

        assert "exact_match" in tr.metrics
        assert "contains" in tr.metrics
        assert adapter.call_count == 2

    def test_evaluate_perfect_match(self):
        from luau_bench.evaluator import evaluate

        task = self._make_task(
            docs=[
                {"question": "Say hello", "answer": "hello"},
            ],
        )
        adapter = MockAdapter(response="hello")

        result = asyncio.run(
            evaluate(
                tasks=[task],
                adapter=adapter,
                quiet=True,
            )
        )

        tr = result.task_results[0]
        assert tr.metrics["exact_match"] == pytest.approx(100.0)

    def test_evaluate_with_task_filter(self):
        from luau_bench.evaluator import evaluate

        task = self._make_task()
        adapter = MockAdapter(response="2")

        result = asyncio.run(
            evaluate(
                tasks=[task],
                adapter=adapter,
                task_names=["nonexistent_task"],
                quiet=True,
            )
        )

        assert len(result.task_results) == 0

    def test_evaluate_model_id(self):
        from luau_bench.evaluator import evaluate

        task = self._make_task()
        adapter = MockAdapter(response="x")

        result = asyncio.run(
            evaluate(
                tasks=[task],
                adapter=adapter,
                quiet=True,
            )
        )

        assert result.model_id == "mock-model"

    def test_evaluate_log_samples(self):
        from luau_bench.evaluator import evaluate

        task = self._make_task(
            docs=[{"question": "hi", "answer": "hello"}],
        )
        adapter = MockAdapter(response="hello")

        result = asyncio.run(
            evaluate(
                tasks=[task],
                adapter=adapter,
                log_samples=True,
                quiet=True,
            )
        )

        tr = result.task_results[0]
        assert len(tr.doc_results) == 1
        assert tr.doc_results[0].prediction == "hello"


class TestCompositeScore:
    def _make_task(self, name: str, answer: str, **overrides):
        raw = {
            "task": name,
            "doc_to_text": "{{ question }}",
            "doc_to_target": "{{ answer }}",
            "metric_list": [{"metric": "exact_match", "primary": True}],
            "docs": [{"question": "Q", "answer": answer}],
        }
        raw.update(overrides)
        from luau_bench.api.task import ConfigurableTask, parse_task_config

        return ConfigurableTask(parse_task_config(raw))

    def test_composite_absent_for_single_task_with_no_primary(self):
        from luau_bench.evaluator import evaluate

        task = self._make_task("t1", "hello")
        adapter = MockAdapter(response="hello")
        result = asyncio.run(evaluate(tasks=[task], adapter=adapter, quiet=True))

        assert result.composite_score is not None
        assert result.composite_score == pytest.approx(100.0)

    def test_composite_weighted_mean(self):
        from luau_bench.evaluator import evaluate

        task_a = self._make_task("ta", "yes", metadata={"weight": 2.0})
        task_b = self._make_task("tb", "no", metadata={"weight": 1.0})

        result = asyncio.run(
            evaluate(
                tasks=[task_a, task_b],
                adapter=MockAdapter(response="yes"),
                quiet=True,
            )
        )

        tr_a = next(t for t in result.task_results if t.task_name == "ta")
        tr_b = next(t for t in result.task_results if t.task_name == "tb")
        assert tr_a.metrics["exact_match"] == pytest.approx(100.0)
        assert tr_b.metrics["exact_match"] == pytest.approx(0.0)

        expected = (100.0 * 2.0 + 0.0 * 1.0) / 3.0
        assert result.composite_score == pytest.approx(expected, abs=1e-9)

    def test_composite_none_when_all_tasks_error(self):
        from luau_bench.evaluator import _compute_composite, TaskResult

        results = [
            TaskResult(task_name="bad", error="boom", primary_metric="exact_match"),
        ]
        score, se = _compute_composite(results)
        assert score is None
        assert se is None

    def test_composite_se_present_for_multiple_tasks(self):
        from luau_bench.evaluator import evaluate

        tasks = [self._make_task(f"t{i}", "x") for i in range(3)]
        result = asyncio.run(evaluate(tasks=tasks, adapter=MockAdapter(response="x"), quiet=True))

        assert result.composite_score == pytest.approx(100.0)
        assert result.composite_se is not None
        assert "ci_lower" in result.composite_se
        assert "ci_upper" in result.composite_se

    def test_benchmark_run_summary_includes_composite(self):
        from luau_bench.evaluator import evaluate

        task = self._make_task("summary_task", "42")
        result = asyncio.run(evaluate(tasks=[task], adapter=MockAdapter(response="42"), quiet=True))

        summary = result.summary
        assert "composite_score" in summary
        assert summary["run_id"] == result.run_id
        assert summary["model_id"] == "mock-model"


class TestParallelIsolation:
    """Verify that concurrent tasks each see their own adapter config (deepcopy isolation)."""

    def _make_task(self, name: str, temperature: float):
        raw = {
            "task": name,
            "doc_to_text": "{{ q }}",
            "doc_to_target": "{{ a }}",
            "generation_kwargs": {"temperature": temperature},
            "metric_list": [{"metric": "exact_match", "primary": True}],
            "docs": [{"q": "Q", "a": "A"}],
        }
        return ConfigurableTask(parse_task_config(raw))

    def test_concurrent_tasks_see_own_temperature(self):
        from luau_bench.evaluator import evaluate

        observed: list[float] = []

        class RecordingAdapter(ModelAdapter):
            def __init__(self):
                super().__init__(ModelConfig(provider="mock", model_name="recording-model"))

            async def generate(self, system: str, user: str) -> GenerationResult:
                observed.append(self.config.temperature)
                await asyncio.sleep(0.01)
                return GenerationResult(text="A", finish_reason="stop")

        task_a = self._make_task("task_temp_01", temperature=0.1)
        task_b = self._make_task("task_temp_09", temperature=0.9)

        asyncio.run(
            evaluate(
                tasks=[task_a, task_b],
                adapter=RecordingAdapter(),
                task_parallel=2,
                quiet=True,
            )
        )

        assert len(observed) == 2
        assert set(round(t, 2) for t in observed) == {0.1, 0.9}

    def test_single_task_does_not_deepcopy(self):
        from luau_bench.evaluator import evaluate
        import copy as _copy

        deepcopy_calls = []
        original_deepcopy = _copy.deepcopy

        def counting_deepcopy(obj, *args, **kwargs):
            if isinstance(obj, ModelAdapter):
                deepcopy_calls.append(True)
            return original_deepcopy(obj, *args, **kwargs)

        task = self._make_task("single_task", temperature=0.5)
        adapter = MockAdapter(response="A")

        _copy.deepcopy = counting_deepcopy
        try:
            asyncio.run(evaluate(tasks=[task], adapter=adapter, task_parallel=4, quiet=True))
        finally:
            _copy.deepcopy = original_deepcopy

        assert len(deepcopy_calls) == 0
