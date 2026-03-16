from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Optional
from pathlib import Path

import click

import luau_bench.api.metrics  # noqa: F401
import luau_bench.api.filters  # noqa: F401


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


@click.group()
@click.version_option(version="0.2.1", prog_name="luau-bench")
def main():
    pass


# run
@main.command()
@click.option(
    "--provider",
    type=click.Choice(["vllm", "openai", "anthropic", "ollama", "tgi", "lmstudio"]),
    required=True,
    help="Model provider",
)
@click.option("--model", "model_name", required=True, help="Model name/ID")
@click.option("--url", "base_url", default="", help="API base URL")
@click.option("--api-key", default="", help="API key (or set env vars)")
@click.option(
    "--include-path",
    type=click.Path(),
    multiple=True,
    default=("./tasks",),
    help="Directory containing task YAML files (repeatable)",
)
@click.option(
    "--tasks", "task_names", default="", help="Comma-separated task/group names (default: all)"
)
@click.option("--num-samples", type=int, default=1, help="Completions per document (for pass@k)")
@click.option("--parallel", type=int, default=1, help="Max concurrent model calls")
@click.option(
    "--output", type=click.Path(), default="./results", help="Output directory for exported reports"
)
@click.option("--log-samples", is_flag=True, help="Store full predictions in result files")
@click.option(
    "--show-samples",
    is_flag=True,
    help="Print prompt, raw output, and execution details for every doc",
)
@click.option(
    "--cache-dir",
    default="~/.cache/luau-bench",
    show_default=True,
    help="Directory for the generation result cache",
)
@click.option("--no-cache", is_flag=True, help="Disable the generation result cache")
@click.option(
    "--task-parallel", type=int, default=4, show_default=True, help="Max tasks running concurrently"
)
@click.option(
    "--export",
    "export_formats",
    default=None,
    is_flag=False,
    flag_value="json,md,html",
    type=str,
    metavar="FORMATS",
    help="Export results. Omit value for all formats, or pass e.g. html,json to select specific ones.",
)
@click.option(
    "--luau-runtime", type=click.Path(), default="", help="Path to Luau/Zune runtime binary"
)
@click.option("--luau-analyzer", type=click.Path(), default="", help="Path to luau-analyze binary")
@click.option("--temperature", type=float, default=0.0)
@click.option("--max-tokens", type=int, default=4096)
@click.option("-v", "--verbose", is_flag=True)
def run(
    provider,
    model_name,
    base_url,
    api_key,
    include_path,
    task_names,
    num_samples,
    parallel,
    task_parallel,
    output,
    log_samples,
    show_samples,
    cache_dir,
    no_cache,
    export_formats,
    luau_runtime,
    luau_analyzer,
    temperature,
    max_tokens,
    verbose,
):
    _setup_logging(verbose)

    from luau_bench.models import ModelConfig, create_adapter
    from luau_bench.tasks import load_task_dirs
    from luau_bench.evaluator import evaluate
    from luau_bench.reporting.reporter import Reporter

    tasks = load_task_dirs(list(include_path))
    if not tasks:
        click.echo(
            "\nNo tasks found. Check your --include-path directories.\n"
            "See templates/ for how to write task YAML files.\n"
        )
        sys.exit(1)

    config = ModelConfig(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    adapter = create_adapter(config)

    names = [t.strip() for t in task_names.split(",") if t.strip()] if task_names else None

    generation_cache = None
    if not no_cache:
        from luau_bench.cache import GenerationCache

        generation_cache = GenerationCache(cache_dir)
        if verbose:
            click.echo(f"  Generation cache: {cache_dir}")

    if luau_runtime:
        os.environ["LUAU_RUNTIME"] = luau_runtime
    if luau_analyzer:
        os.environ["LUAU_ANALYZE"] = luau_analyzer

    result = asyncio.run(
        evaluate(
            tasks=tasks,
            adapter=adapter,
            task_names=names,
            num_samples=num_samples,
            parallel=parallel,
            task_parallel=task_parallel,
            log_samples=log_samples,
            show_samples=show_samples,
            cache=generation_cache,
        )
    )

    if generation_cache is not None:
        stats = generation_cache.stats
        if stats["hits"] or stats["misses"]:
            click.echo(
                f"  Cache: {stats['hits']} hits, {stats['misses']} misses "
                f"(saved ~{stats['hits']} API calls)\n"
            )

    reporter = Reporter(output_dir=output)
    reporter.print_summary(result)

    if export_formats:
        fmts = {f.strip().lower() for f in export_formats.split(",") if f.strip()}
        saved: list[str] = []
        if "json" in fmts:
            saved.append(str(reporter.save_json(result)))
        if "md" in fmts or "markdown" in fmts:
            saved.append(str(reporter.save_markdown(result)))
        if "html" in fmts:
            saved.append(str(reporter.save_html(result)))
        if saved:
            click.echo(f"\n  Saved: {', '.join(saved)}\n")


# ls
@main.command("ls")
@click.option("--include-path", multiple=True, help="Directory containing task YAML files")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def ls(include_path, json_output):
    _setup_logging(False)

    from luau_bench.tasks import load_task_dirs
    from luau_bench.api import list_groups

    if not include_path:
        click.echo(
            "\nNo task directories specified. Use --include-path to add one.\n"
            "The harness ships no built-in tasks — tasks are user-defined.\n"
        )
        return

    tasks = load_task_dirs(list(include_path))
    groups = list_groups()

    if json_output:
        data = {
            "tasks": [
                {
                    "name": t.config.task,
                    "group": t.config.group,
                    "version": t.config.version,
                    "num_docs": len(t.get_docs()),
                    "metrics": [m.metric for m in t.config.metric_list],
                }
                for t in tasks
            ],
            "groups": groups,
        }
        click.echo(json.dumps(data, indent=2))
    else:
        if groups:
            click.echo(f"\nGroups ({len(groups)}):\n")
            for gname, members in sorted(groups.items()):
                click.echo(
                    f"  {click.style(gname, fg='magenta', bold=True):30s} -> {', '.join(members)}"
                )
            click.echo()

        if tasks:
            click.echo(f"Tasks ({len(tasks)}):\n")
            for t in sorted(tasks, key=lambda x: x.config.task):
                c = t.config
                metrics = ", ".join(m.metric for m in c.metric_list)
                docs = len(t.get_docs())
                click.echo(
                    f"  {click.style(c.task, fg='cyan', bold=True):30s}  "
                    f"v{c.version:<6}  "
                    f"{docs} docs  "
                    f"metrics=[{metrics}]"
                )
        else:
            click.echo("\nNo tasks found in the specified directories.\n")
        click.echo()


# validate
@main.command()
@click.option(
    "--include-path", multiple=True, required=True, help="Directory containing task YAML files"
)
@click.option("-v", "--verbose", is_flag=True)
def validate(include_path, verbose):
    _setup_logging(verbose)

    from luau_bench.tasks import load_task_dirs
    from luau_bench.api import list_metrics, list_filters

    tasks = load_task_dirs(list(include_path))
    if not tasks:
        click.echo("\nNo tasks found. Check your --include-path.\n")
        return

    available_metrics = set(list_metrics())
    available_filters = set(list_filters())
    errors = 0

    for task in sorted(tasks, key=lambda t: t.config.task):
        issues: list[str] = []
        c = task.config

        if not c.doc_to_text:
            issues.append("Missing doc_to_text (no prompt template)")
        if not c.metric_list:
            issues.append("No metrics defined")
        if not task.get_docs():
            issues.append("No documents")

        for m in c.metric_list:
            if m.metric not in available_metrics:
                issues.append(f"Unknown metric: {m.metric}")
        for f in c.filters:
            if f.name not in available_filters:
                issues.append(f"Unknown filter: {f.name}")

        if issues:
            click.echo(click.style(f"  ✗ {c.task}: ", fg="red") + "; ".join(issues))
            errors += 1
        else:
            docs = len(task.get_docs())
            metrics = ", ".join(m.metric for m in c.metric_list)
            click.echo(
                click.style(f"  ✓ {c.task}", fg="green") + f"  ({docs} docs, metrics=[{metrics}])"
            )

    click.echo(f"\n{len(tasks)} task(s) checked, {errors} with issues.\n")
    if errors:
        sys.exit(1)


# selftest
@main.command()
@click.option(
    "--include-path", multiple=True, required=True, help="Directory containing task YAML files"
)
@click.option(
    "--tasks", "task_names", default="", help="Comma-separated task names to test (default: all)"
)
@click.option("--luau-runtime", type=click.Path(), help="Path to luau/lune binary")
@click.option("-v", "--verbose", is_flag=True)
def selftest(include_path, task_names, luau_runtime, verbose):
    _setup_logging(verbose)

    from luau_bench.tasks import load_task_dirs
    from luau_bench.runtime.executor import LuauExecutor

    tasks = load_task_dirs(list(include_path))
    if task_names:
        names = {t.strip() for t in task_names.split(",") if t.strip()}
        tasks = [t for t in tasks if t.config.task in names]

    if not tasks:
        click.echo("\nNo tasks found.\n")
        return

    executor = LuauExecutor(runtime_path=luau_runtime or None)
    if not executor.available:
        click.echo(
            click.style(
                "Error: No Luau runtime found. Install 'luau' or 'zune'.\n",
                fg="red",
            )
        )
        sys.exit(1)

    click.echo(f"\nRunning self-tests for {len(tasks)} task(s)...\n")
    passed_tasks = failed_tasks = skipped = 0

    async def _run():
        nonlocal passed_tasks, failed_tasks, skipped
        for task in tasks:
            c = task.config
            ref_sol = c.metadata.get("reference_solution", "")
            if not ref_sol:
                click.echo(click.style(f"  ⊘ {c.task}", fg="yellow") + "  (no reference_solution)")
                skipped += 1
                continue

            docs = task.get_docs()
            all_passed = True
            total = total_pass = 0

            for doc in docs:
                harness = doc.get("test_harness", "")
                if not harness:
                    continue
                script = harness.replace("{{CODE}}", ref_sol)
                result = await executor.run_script(script)
                total += result["total"]
                total_pass += result["passed"]
                if result["passed"] < result["total"]:
                    all_passed = False
                    if verbose:
                        for d in result["details"]:
                            if d["status"] != "pass":
                                click.echo(f"      {d['status'].upper()}: {d['test']}")

            if total == 0:
                click.echo(click.style(f"  ⊘ {c.task}", fg="yellow") + "  (no test harnesses)")
                skipped += 1
            elif all_passed:
                passed_tasks += 1
                click.echo(click.style(f"  ✓ {c.task}", fg="green") + f"  {total_pass}/{total}")
            else:
                failed_tasks += 1
                click.echo(click.style(f"  ✗ {c.task}", fg="red") + f"  {total_pass}/{total}")

    asyncio.run(_run())

    click.echo(f"\n{'─' * 50}")
    click.echo(f"  {passed_tasks} passed, {failed_tasks} failed, {skipped} skipped\n")
    if failed_tasks:
        sys.exit(1)


# compare
@main.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--baseline",
    "baseline_file",
    type=click.Path(exists=True),
    default=None,
    help="Result file to treat as baseline for delta computation. "
    "Defaults to the first positional argument.",
)
@click.option(
    "--per-task", is_flag=True, help="Show a per-task breakdown in addition to the summary."
)
def compare(files, baseline_file, per_task):
    if len(files) < 2:
        click.echo("Error: need at least 2 result files.", err=True)
        sys.exit(1)

    baseline_path = baseline_file or files[0]
    other_paths = [f for f in files if f != baseline_path]
    if not other_paths:
        other_paths = list(files[1:])

    def _load(path: str) -> dict:
        return json.loads(Path(path).read_text())

    baseline_data = _load(baseline_path)
    other_runs = [_load(p) for p in other_paths]

    _NON_METRIC_KEYS = frozenset({"version", "num_docs", "weight"})

    all_metric_names: list[str] = []
    seen_metrics: set[str] = set()
    for run_data in [baseline_data] + other_runs:
        for task_data in run_data.get("tasks", {}).values():
            for k, v in task_data.items():
                if (
                    isinstance(v, (int, float))
                    and k not in _NON_METRIC_KEYS
                    and k not in seen_metrics
                ):
                    all_metric_names.append(k)
                    seen_metrics.add(k)

    all_task_names: list[str] = sorted(
        {task for run_data in [baseline_data] + other_runs for task in run_data.get("tasks", {})}
    )

    def _avg(run_data: dict, metric: str) -> float:
        vals = [
            t[metric]
            for t in run_data.get("tasks", {}).values()
            if isinstance(t.get(metric), (int, float))
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def _task_val(run_data: dict, task: str, metric: str) -> Optional[float]:
        t = run_data.get("tasks", {}).get(task, {})
        v = t.get(metric)
        return float(v) if isinstance(v, (int, float)) else None

    click.echo()
    _model_w = 36
    _m_w = max(max((len(m) for m in all_metric_names), default=12), 14) + 2

    baseline_model = baseline_data.get("model_id", Path(baseline_path).stem)
    click.echo(click.style("  Baseline: ", bold=True) + baseline_model)
    click.echo()

    header = f"  {'Model / Run':{_model_w}s}"
    for m in all_metric_names:
        header += f"  {m:>{_m_w}s}"
    click.echo(click.style(header, bold=True))
    click.echo("  " + "─" * (_model_w + (_m_w + 2) * len(all_metric_names)))

    def _delta_str(base_val: float, new_val: float) -> str:
        d = new_val - base_val
        if abs(d) < 0.5:
            return f"{new_val:7.2f}     "
        sign = "+" if d > 0 else ""
        delta = f"{sign}{d:.2f}"
        col = "green" if d > 0 else "red"
        return f"{new_val:7.2f} " + click.style(f"({delta})", fg=col)

    row = f"  {baseline_model:{_model_w}s}"
    for m in all_metric_names:
        val = _avg(baseline_data, m)
        row += f"  {val:>{_m_w}.2f}"
    click.echo(row)

    for idx, (run_data, path) in enumerate(zip(other_runs, other_paths)):
        model_label = run_data.get("model_id", Path(path).stem)
        row = f"  {model_label:{_model_w}s}"
        for m in all_metric_names:
            base_val = _avg(baseline_data, m)
            new_val = _avg(run_data, m)
            row += "  " + _delta_str(base_val, new_val).rjust(_m_w)
        click.echo(row)

    if per_task and all_task_names:
        click.echo()
        click.echo(click.style("  Per-task breakdown:", bold=True))
        click.echo(
            "  Significance: "
            + click.style("✓ sig", fg="green")
            + " = CIs do not overlap  |  "
            + click.style("~ not sig", fg="yellow")
            + " = CIs overlap (difference may be noise)"
        )

        def _get_ci(run_data: dict, task: str, metric: str) -> Optional[tuple[float, float]]:
            se_block = run_data.get("tasks", {}).get(task, {}).get("std_errors", {}).get(metric)
            if se_block and "ci_lower" in se_block and "ci_upper" in se_block:
                return (se_block["ci_lower"], se_block["ci_upper"])
            return None

        for task in all_task_names:
            click.echo(click.style(f"  {task}", fg="cyan", bold=True))
            t_header = f"    {'Run':{_model_w}s}"
            for m in all_metric_names:
                t_header += f"  {m:>{_m_w}s}"
            click.echo(t_header)

            row = f"    {baseline_model:{_model_w}s}"
            for m in all_metric_names:
                val = _task_val(baseline_data, task, m)
                base_ci = _get_ci(baseline_data, task, m)
                if val is None:
                    row += f"  {'—':>{_m_w}s}"
                elif base_ci:
                    lo, hi = base_ci
                    cell = f"{val:.1f} [{lo:.0f},{hi:.0f}]"
                    row += f"  {cell:>{_m_w}s}"
                else:
                    row += f"  {val:>{_m_w}.2f}"
            click.echo(row)

            for run_data, path in zip(other_runs, other_paths):
                model_label = run_data.get("model_id", Path(path).stem)
                row = f"    {model_label:{_model_w}s}"
                sig_flags: list[str] = []

                for m in all_metric_names:
                    base_v = _task_val(baseline_data, task, m)
                    new_v = _task_val(run_data, task, m)
                    base_ci = _get_ci(baseline_data, task, m)
                    new_ci = _get_ci(run_data, task, m)

                    if base_v is None or new_v is None:
                        row += f"  {'—':>{_m_w}s}"
                        continue

                    cell = _delta_str(base_v, new_v)

                    if base_ci and new_ci:
                        from luau_bench.stats import ci_overlaps

                        if ci_overlaps(base_ci, new_ci):
                            sig_flags.append(click.style("~ not sig", fg="yellow"))
                        else:
                            sig_flags.append(click.style("✓ sig", fg="green"))

                    row += "  " + cell.rjust(_m_w)

                click.echo(row)

                if sig_flags:
                    pad = " " * (_model_w + 6)
                    click.echo(pad + "  ".join(sig_flags))

            click.echo()

    click.echo()


# cache
@main.group()
def cache():
    pass


@cache.command("clear")
@click.option(
    "--cache-dir", default="~/.cache/luau-bench", show_default=True, help="Cache directory to clear"
)
@click.confirmation_option(prompt="Clear all cached generations?")
def cache_clear(cache_dir):
    from luau_bench.cache import GenerationCache

    n = GenerationCache(cache_dir).clear()
    click.echo(f"  Cleared {n} cached generation(s) from {cache_dir}")


@cache.command("stats")
@click.option(
    "--cache-dir",
    default="~/.cache/luau-bench",
    show_default=True,
    help="Cache directory to inspect",
)
def cache_stats(cache_dir):
    from pathlib import Path as _Path

    cache_path = _Path(cache_dir).expanduser() / "generations"
    if not cache_path.exists():
        click.echo(f"  Cache directory does not exist: {cache_path}")
        return
    files = list(cache_path.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in files)
    click.echo(f"  Cache: {cache_path}")
    click.echo(f"  Entries: {len(files)}")
    click.echo(f"  Size:    {total_bytes / 1024:.1f} KB")


# info
@main.command()
@click.option(
    "--include-path",
    multiple=True,
    help="Task directories (for task count)",
)
def info(include_path):
    click.echo("\nLuau Bench Environment\n")
    click.echo(f"  Python:       {sys.version.split()[0]}")

    from luau_bench.runtime.executor import (
        find_luau_runtime,
        find_luau_analyzer,
        get_runtime_version,
    )

    rt = find_luau_runtime()
    if rt:
        ver = get_runtime_version(rt) or "unknown"
        click.echo(f"  Luau:         {rt}")
        click.echo(f"               {ver}")
    else:
        click.echo(click.style("  Luau:         NOT FOUND", fg="red"))
        click.echo("               Install 'luau' or 'zune' for code execution.")

    az = find_luau_analyzer()
    if az:
        ver = get_runtime_version(az) or "unknown"
        click.echo(f"  luau-analyze: {az}")
        click.echo(f"               {ver}")
    else:
        click.echo(click.style("  luau-analyze: NOT FOUND (optional)", fg="yellow"))

    from luau_bench.runtime.stylua import find_stylua, get_stylua_version

    stylua = find_stylua()
    if stylua:
        ver = get_stylua_version(stylua) or "unknown"
        click.echo(f"  StyLua:       {stylua}")
        click.echo(f"               {ver}")
    else:
        click.echo(click.style("  StyLua:       NOT FOUND (optional)", fg="yellow"))

    from luau_bench.api import list_metrics, list_filters

    click.echo(f"\n  Metrics:      {', '.join(list_metrics())}")
    click.echo(f"  Filters:      {', '.join(list_filters())}")

    if include_path:
        from luau_bench.tasks import load_task_dirs

        tasks = load_task_dirs(list(include_path))
        click.echo(f"\n  Tasks loaded: {len(tasks)}")
    else:
        click.echo("\n  No --include-path specified (harness ships no built-in tasks).")

    click.echo()


if __name__ == "__main__":
    main()
