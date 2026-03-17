from __future__ import annotations

import asyncio
import copy
import logging
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import click

from luau_bench.api import get_filter, get_metric, resolve_tasks
from luau_bench.api.task import ConfigurableTask, Task
from luau_bench.models import GenerationResult, ModelAdapter

logger = logging.getLogger(__name__)

_MULTI_SAMPLE_METRICS: frozenset[str] = frozenset({"luau_exec", "pass_at_k"})
_SIDE_CHANNEL_PREFIX = "_"
_EXTRAS_KEYS: tuple[str, ...] = ("_exec_details", "_analyze_details", "_pass_flags")
_PRIMARY_SUFFIXES: tuple[str, ...] = ("_pass_rate", "_clean_rate", "_acc")


def _resolve_primary_key(spec_name: str, metric_keys: set[str]) -> Optional[str]:
    if spec_name in metric_keys:
        return spec_name
    for suffix in _PRIMARY_SUFFIXES:
        if (candidate := spec_name + suffix) in metric_keys:
            return candidate
    for key in sorted(metric_keys):
        if key.startswith(spec_name):
            return key
    return None


@dataclass
class DocResult:
    doc: dict[str, Any]
    prediction: str = ""
    reference: str = ""
    raw_generation: str = ""
    generation_time_ms: float = 0.0
    tokens: int = 0
    all_predictions: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    task_name: str
    version: float = 0.0
    num_docs: int = 0
    doc_results: list[DocResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    std_errors: dict[str, dict[str, float]] = field(default_factory=dict)
    weight: float = 1.0
    primary_metric: Optional[str] = None
    metric_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    run_id: str
    model_id: str
    provider: str
    task_results: list[TaskResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    config: dict[str, Any] = field(default_factory=dict)
    composite_score: Optional[float] = None
    composite_se: Optional[dict[str, float]] = None

    @property
    def summary(self) -> dict[str, Any]:
        per_task: dict[str, Any] = {}
        for tr in self.task_results:
            per_task[tr.task_name] = {
                "version": tr.version,
                "num_docs": tr.num_docs,
                "weight": tr.weight,
                "primary_metric": tr.primary_metric,
                **tr.metrics,
            }
            if tr.error:
                per_task[tr.task_name]["error"] = tr.error
            if tr.std_errors:
                per_task[tr.task_name]["std_errors"] = tr.std_errors

        result: dict[str, Any] = {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "tasks": per_task,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.composite_score is not None:
            result["composite_score"] = self.composite_score
        if self.composite_se is not None:
            result["composite_se"] = self.composite_se
        return result


async def evaluate(
    tasks: list[Task],
    adapter: ModelAdapter,
    *,
    task_names: Optional[list[str]] = None,
    num_samples: int = 1,
    parallel: int = 1,
    task_parallel: int = 4,
    log_samples: bool = False,
    show_samples: bool = False,
    cache=None,
    quiet: bool = False,
) -> BenchmarkRun:
    run_id = str(uuid.uuid4())[:8]
    model_id = adapter.model_id()
    run = BenchmarkRun(
        run_id=run_id,
        model_id=model_id,
        provider=adapter.config.provider,
    )

    if task_names:
        resolved = set(resolve_tasks(task_names))
        tasks = [t for t in tasks if t.get_config().task in resolved]

    if not tasks:
        logger.error("No tasks to evaluate.")
        return run

    if not quiet:
        _print_header(model_id, len(tasks))

    task_sem = asyncio.Semaphore(max(1, task_parallel))
    gen_sem = asyncio.Semaphore(max(1, parallel))
    print_lock = asyncio.Lock()

    async def run_one_task(task: Task) -> TaskResult:
        task_adapter = copy.deepcopy(adapter) if (task_parallel > 1 and len(tasks) > 1) else adapter

        config = task.get_config()
        task_name = config.task

        try:
            task_adapter.apply_generation_config(task.get_generation_config())
            docs = task.get_docs()

            if not docs:
                logger.warning("Task '%s': no documents - skipping.", task_name)
                return TaskResult(
                    task_name=task_name,
                    version=config.version,
                    error="No documents in task",
                    weight=float(config.metadata.get("weight", 1.0)),
                )

            effective_n = max(1, config.num_samples if config.num_samples > 1 else num_samples)
            is_loglike = config.output_type == "loglikelihood"
            doc_results: list[DocResult] = []

            n_total = len(docs) * effective_n
            _desc = f"{task_name} ({n_total} gen{'s' if n_total != 1 else ''})"

            for doc_idx, doc in _progress(
                enumerate(docs), desc=_desc, total=len(docs), quiet=quiet
            ):
                prompt = task.build_prompt(doc)
                reference = task.get_target(doc) if isinstance(task, ConfigurableTask) else ""

                if is_loglike:
                    choices: list[str] = (
                        task.get_choices(doc) if isinstance(task, ConfigurableTask) else []
                    )
                    if not choices:
                        logger.warning(
                            "Task '%s' doc %d: loglikelihood task has no choices.",
                            task_name,
                            doc_idx,
                        )
                        doc_results.append(DocResult(doc=doc, prediction="", reference=reference))
                        continue

                    t0 = time.perf_counter()
                    async with gen_sem:
                        ll_results = await task_adapter.loglikelihood_batch(
                            system=prompt["system"],
                            user=prompt["user"],
                            continuations=choices,
                        )
                    gen_time = (time.perf_counter() - t0) * 1000.0
                    best_idx = max(
                        range(len(ll_results)),
                        key=lambda i: ll_results[i].normalized_logprob,
                    )
                    chosen = choices[best_idx]
                    if reference.strip().lstrip("-").isdigit():
                        ref_idx = int(reference.strip())
                        reference = choices[ref_idx] if 0 <= ref_idx < len(choices) else reference
                    doc_results.append(
                        DocResult(
                            doc=doc,
                            prediction=chosen,
                            reference=reference,
                            generation_time_ms=gen_time,
                            all_predictions=[chosen],
                        )
                    )

                else:
                    all_raw: list[GenerationResult] = []
                    total_gen_time = 0.0

                    for sample_idx in range(effective_n):
                        cached = None
                        cache_key = None
                        if cache is not None:
                            cache_key = cache.make_key(
                                model_id=model_id,
                                task_name=task_name,
                                system=prompt["system"],
                                user=prompt["user"],
                                sample_index=sample_idx,
                                temperature=task_adapter.config.temperature,
                                max_tokens=task_adapter.config.max_tokens,
                            )
                            cached = cache.get(cache_key)
                            if cached:
                                logger.debug(
                                    "Cache hit: %s doc %d sample %d",
                                    task_name,
                                    doc_idx,
                                    sample_idx,
                                )

                        if cached is not None:
                            all_raw.append(cached)
                        else:
                            t0 = time.perf_counter()
                            async with gen_sem:
                                gen: GenerationResult = await task_adapter.generate(
                                    system=prompt["system"],
                                    user=prompt["user"],
                                )
                            total_gen_time += (time.perf_counter() - t0) * 1000.0
                            all_raw.append(gen)
                            if cache is not None and cache_key is not None:
                                cache.set(cache_key, gen)

                    all_predictions: list[str] = []
                    for gen in all_raw:
                        pred = gen.text
                        for fspec in task.get_filter_specs():
                            fn = get_filter(fspec.name)
                            pred = fn(pred, **fspec.args)
                        all_predictions.append(pred)

                    total_tokens = sum(g.prompt_tokens + g.completion_tokens for g in all_raw)
                    doc_results.append(
                        DocResult(
                            doc=doc,
                            prediction=all_predictions[0],
                            reference=reference,
                            raw_generation=all_raw[0].text if log_samples else "",
                            generation_time_ms=total_gen_time,
                            tokens=total_tokens,
                            all_predictions=all_predictions,
                        )
                    )

            predictions_1 = [dr.prediction for dr in doc_results]
            references = [dr.reference for dr in doc_results]
            all_docs = [dr.doc for dr in doc_results]
            predictions_n = [
                p for dr in doc_results for p in (dr.all_predictions or [dr.prediction])
            ]
            references_n = [
                dr.reference for dr in doc_results for _ in (dr.all_predictions or [dr.prediction])
            ]
            docs_n = [dr.doc for dr in doc_results for _ in (dr.all_predictions or [dr.prediction])]

            task_metrics: dict[str, float] = {}
            metric_context: dict[str, Any] = {}

            for mspec in task.get_metric_specs():
                fn = get_metric(mspec.metric)
                is_multi = mspec.metric in _MULTI_SAMPLE_METRICS and not is_loglike

                raw_result = fn(
                    predictions_n if is_multi else predictions_1,
                    references_n if is_multi else references,
                    docs=docs_n if is_multi else all_docs,
                    num_samples=effective_n if is_multi else 1,
                    **{k: v for k, v in metric_context.items()},
                    **mspec.args,
                )
                if asyncio.iscoroutine(raw_result):
                    raw_result = await raw_result

                for k, v in raw_result.items():
                    if k.startswith(_SIDE_CHANNEL_PREFIX):
                        metric_context[k] = v
                    else:
                        task_metrics[k] = v

            std_errors: dict[str, dict[str, float]] = {}
            per_doc_scores: list[float] = metric_context.get("_per_doc_scores", [])

            spec_primary = next(
                (m.metric for m in task.get_metric_specs() if m.primary),
                next(iter(task_metrics), None),
            )
            primary_metric_name: Optional[str] = (
                _resolve_primary_key(spec_primary, set(task_metrics)) if spec_primary else None
            )

            if per_doc_scores and len(per_doc_scores) >= 2 and primary_metric_name:
                from luau_bench.stats import all_se

                cluster_field = config.metadata.get("se_cluster_field")
                cluster_ids = None
                if cluster_field:
                    cluster_ids = [
                        str(dr.doc.get(cluster_field, i)) for i, dr in enumerate(doc_results)
                    ][: len(per_doc_scores)]

                std_errors[primary_metric_name] = all_se(
                    per_doc_scores,
                    cluster_ids=cluster_ids,
                )

            metric_extras = {k: metric_context[k] for k in _EXTRAS_KEYS if k in metric_context}
            metric_extras["_effective_n"] = effective_n

            tr = TaskResult(
                task_name=task_name,
                version=config.version,
                num_docs=len(docs),
                doc_results=doc_results if log_samples else [],
                metrics=task_metrics,
                std_errors=std_errors,
                weight=float(config.metadata.get("weight", 1.0)),
                primary_metric=primary_metric_name,
                metric_extras=metric_extras,
            )

            async with print_lock:
                if not quiet:
                    _print_task_result(task_name, task_metrics, len(docs), effective_n)
                if show_samples:
                    _print_samples(
                        task_name=task_name,
                        doc_results=doc_results,
                        exec_details=metric_context.get("_exec_details"),
                        effective_n=effective_n,
                    )

            return tr

        except Exception as exc:
            logger.error("Task '%s' failed: %s", task_name, exc, exc_info=True)
            return TaskResult(
                task_name=task_name,
                version=config.version,
                error=str(exc),
                weight=float(config.metadata.get("weight", 1.0)),
            )

    async def bounded_run(task: Task) -> TaskResult:
        async with task_sem:
            return await run_one_task(task)

    task_results: list[TaskResult] = await asyncio.gather(*(bounded_run(t) for t in tasks))

    run.task_results = list(task_results)
    run.finished_at = time.time()

    run.composite_score, run.composite_se = _compute_composite(run.task_results)

    if not quiet:
        _print_footer(run)

    return run


def _compute_composite(
    task_results: list[TaskResult],
) -> tuple[Optional[float], Optional[dict[str, float]]]:
    eligible = [
        tr
        for tr in task_results
        if not tr.error and tr.primary_metric and tr.primary_metric in tr.metrics
    ]
    if not eligible:
        return None, None

    total_weight = sum(tr.weight for tr in eligible)
    if total_weight == 0:
        return None, None

    composite = sum(tr.metrics[tr.primary_metric] * tr.weight for tr in eligible) / total_weight
    composite_01 = composite / 100.0

    task_scores_01 = [tr.metrics[tr.primary_metric] / 100.0 for tr in eligible]

    if len(task_scores_01) >= 2:
        from luau_bench.stats import all_se

        se_dict = all_se(task_scores_01)
        bootstrap_pct = se_dict["bootstrap"] * 100.0
        se_dict["formula"] *= 100.0
        se_dict["bootstrap"] = bootstrap_pct
        se_dict["clustered"] *= 100.0
        se_dict["mean"] = composite_01
        se_dict["ci_lower"] = max(0.0, composite - 1.96 * bootstrap_pct)
        se_dict["ci_upper"] = min(100.0, composite + 1.96 * bootstrap_pct)
    else:
        se_dict = {
            "formula": 0.0,
            "bootstrap": 0.0,
            "clustered": 0.0,
            "n": float(len(task_scores_01)),
            "mean": composite_01,
            "ci_lower": composite,
            "ci_upper": composite,
            "ci_level": 0.95,
            "z": 1.96,
        }

    return composite, se_dict


def _progress(iterable, *, desc: str = "", total: int = 0, quiet: bool = False):
    if quiet:
        yield from iterable
        return
    try:
        from tqdm import tqdm  # type: ignore[import]

        yield from tqdm(
            iterable,
            desc=desc,
            total=total or None,
            unit="doc",
            leave=False,
            dynamic_ncols=True,
        )
    except ImportError:
        yield from iterable


def _print_header(model_id: str, total: int) -> None:
    click.echo("\nLuau Bench")
    click.echo(f"  Model : {model_id}")
    click.echo(f"  Tasks : {total}\n")


def _print_task_result(
    name: str,
    metrics: dict[str, float],
    num_docs: int,
    num_samples: int,
) -> None:
    parts = [f"{k}={v:.1f}%" for k, v in metrics.items() if isinstance(v, (int, float))]
    score_str = ", ".join(parts[:4])
    tag = f" x{num_samples}" if num_samples > 1 else ""
    status = click.style("ok", fg="green", bold=True)
    click.echo(f"  {status}  {name:30s} ({num_docs} docs{tag})  {score_str}")


def _print_footer(run: BenchmarkRun) -> None:
    elapsed = (run.finished_at or time.time()) - run.started_at
    click.echo(f"\n  Completed {len(run.task_results)} task(s) in {elapsed:.1f}s")
    for tr in run.task_results:
        if tr.error:
            status = click.style("err", fg="red", bold=True)
            metrics_str = tr.error
        else:
            status = click.style("ok ", fg="green", bold=True)
            metrics_str = ", ".join(
                f"{k}={v:.1f}%"
                for k, v in list(tr.metrics.items())[:3]
                if isinstance(v, (int, float))
            )
        click.echo(f"    {status}  {tr.task_name}: {metrics_str}")

    if run.composite_score is not None:
        se_str = ""
        if run.composite_se:
            lo = run.composite_se.get("ci_lower", 0.0)
            hi = run.composite_se.get("ci_upper", 0.0)
            se_str = f"  [{lo:.1f}%, {hi:.1f}%] 95% CI"
        score_col = click.style(f"{run.composite_score:.2f}%", fg="cyan", bold=True)
        click.echo(f"\n  Composite score: {score_col}{se_str}")
    click.echo("")


def _print_samples(
    task_name: str,
    doc_results: list[DocResult],
    exec_details: Optional[list],
    effective_n: int,
) -> None:
    click.echo(f"\n  SAMPLES: {task_name}")
    click.echo(f"  {'-' * 60}")

    for doc_idx, dr in enumerate(doc_results):
        click.echo(f"\n  doc {doc_idx + 1}/{len(doc_results)}")

        prompt_preview = textwrap.fill(
            dr.doc.get("description", "") or str(list(dr.doc.keys())[:3]),
            width=72,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        if prompt_preview:
            click.echo(f"\n  Prompt:\n{prompt_preview}\n")

        for sample_idx in range(effective_n):
            pred = (
                dr.all_predictions[sample_idx]
                if sample_idx < len(dr.all_predictions)
                else dr.prediction
            )
            raw = dr.raw_generation if sample_idx == 0 else ""

            tag = f"sample {sample_idx + 1}/{effective_n}" if effective_n > 1 else "output"
            click.echo(f"\n  [{tag}]")

            if raw:
                click.echo(
                    f"\n  Raw ({len(raw)} chars):\n"
                    + textwrap.indent(raw[:600] + ("..." if len(raw) > 600 else ""), "    ")
                )

            click.echo(
                f"\n  Extracted ({len(pred)} chars):\n"
                + textwrap.indent(pred[:500] + ("..." if len(pred) > 500 else ""), "    ")
            )

            flat_idx = doc_idx * effective_n + sample_idx
            detail: Optional[dict] = None
            if exec_details and flat_idx < len(exec_details):
                detail = exec_details[flat_idx]

            if detail is not None:
                total = detail.get("total", 0)
                passed = detail.get("passed", 0)
                ms = detail.get("runtime_ms", 0.0)
                timed_out = detail.get("timed_out", False)
                tag2 = (
                    click.style("TIMEOUT", fg="red", bold=True)
                    if timed_out
                    else f"{passed}/{total} tests passed"
                )
                click.echo(f"\n  Execution ({ms:.0f}ms): {tag2}")
                _status_styles = {
                    "pass": ("pass", "green"),
                    "fail": ("FAIL", "red"),
                    "error": ("ERR ", "yellow"),
                }
                for t in detail.get("details", []):
                    label, colour = _status_styles.get(t["status"], ("?   ", "white"))
                    styled = click.style(label, fg=colour, bold=True)
                    msg = f": {t['message']}" if t.get("message") else ""
                    click.echo(f"    {styled}  {t['test']}{msg}")
                stderr = (detail.get("stderr") or "").strip()
                if stderr:
                    click.echo("\n  stderr:\n" + textwrap.indent(stderr[:300], "    "))
            else:
                click.echo("\n  (no execution)")

        click.echo("")

    click.echo(f"  {'-' * 60}\n")
