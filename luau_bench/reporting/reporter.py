from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from luau_bench.evaluator import BenchmarkRun

logger = logging.getLogger(__name__)


class Reporter:
    def __init__(self, output_dir: str = "./results") -> None:
        self.output_dir = Path(output_dir)

    def _ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_summary(self, run: BenchmarkRun) -> None:
        try:
            from rich import box
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            self._print_rich(run, Console(), box, Panel, Table)
        except ImportError:
            self._print_plain(run)

    def _print_rich(self, run, console, box, Panel, Table) -> None:
        console.print()
        console.print(
            Panel(
                f"[bold cyan]Benchmark Results[/bold cyan]\n"
                f"Model: [bold]{run.model_id}[/bold]  |  "
                f"Tasks: {len(run.task_results)}",
                box=box.DOUBLE,
            )
        )

        table = Table(title="Task Results", box=box.ROUNDED, show_lines=True)
        table.add_column("Task", style="cyan", min_width=20)
        table.add_column("Version", justify="right", min_width=8)
        table.add_column("Docs", justify="right", min_width=6)
        table.add_column("Metrics", min_width=40)

        for tr in sorted(run.task_results, key=lambda t: t.task_name):
            metrics_str = ", ".join(
                f"{k}={v:.1f}%" for k, v in tr.metrics.items() if isinstance(v, (int, float))
            )
            if tr.error:
                metrics_str = f"[red]ERROR: {tr.error}[/red]"
            table.add_row(
                tr.task_name,
                f"v{tr.version}",
                str(tr.num_docs),
                metrics_str,
            )

        console.print(table)

        if run.composite_score is not None:
            se_str = ""
            if run.composite_se:
                lo = run.composite_se.get("ci_lower", 0.0)
                hi = run.composite_se.get("ci_upper", 0.0)
                se_str = f"  [dim]\\[{lo:.1f}%, {hi:.1f}%] 95% CI[/dim]"
            console.print(
                f"  [bold]Composite score:[/bold] [green]{run.composite_score:.2f}%[/green]{se_str}"
            )

        console.print()

    def _print_plain(self, run: BenchmarkRun) -> None:
        print(f"\n{'=' * 60}")
        print(f"Benchmark Results — {run.model_id}")
        print(f"{'=' * 60}")
        print(f"Tasks: {len(run.task_results)}\n")
        for tr in sorted(run.task_results, key=lambda t: t.task_name):
            metrics_str = ", ".join(
                f"{k}={v:.1f}%" for k, v in tr.metrics.items() if isinstance(v, (int, float))
            )
            if tr.error:
                metrics_str = f"ERROR: {tr.error}"
            print(f"  {tr.task_name:30s}  v{tr.version}  {tr.num_docs} docs  {metrics_str}")
        if run.composite_score is not None:
            print(f"\n  Composite score: {run.composite_score:.2f}%")
        print(f"{'=' * 60}\n")

    def save_json(self, run: BenchmarkRun, filename: Optional[str] = None) -> Path:
        self._ensure_output_dir()
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe = run.model_id.replace("/", "_").replace(":", "_")
            filename = f"luau_bench_{safe}_{ts}.json"

        path = self.output_dir / filename
        data = run.summary

        for tr in run.task_results:
            if tr.doc_results:
                data["tasks"][tr.task_name]["samples"] = [
                    {
                        "prediction": dr.prediction,
                        "reference": dr.reference,
                        "generation_time_ms": round(dr.generation_time_ms, 2),
                        "tokens": dr.tokens,
                    }
                    for dr in tr.doc_results
                ]

        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("JSON report: %s", path)
        return path

    def save_html(self, run: BenchmarkRun, filename: Optional[str] = None) -> Path:
        self._ensure_output_dir()
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe = run.model_id.replace("/", "_").replace(":", "_")
            filename = f"luau_bench_{safe}_{ts}.html"
        path = self.output_dir / filename
        from luau_bench.reporting.html_report import save_html

        save_html(run, path)
        logger.info("HTML report: %s", path)
        return path

    def save_markdown(self, run: BenchmarkRun, filename: Optional[str] = None) -> Path:
        self._ensure_output_dir()
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe = run.model_id.replace("/", "_").replace(":", "_")
            filename = f"luau_bench_{safe}_{ts}.md"

        path = self.output_dir / filename

        all_metrics: list[str] = []
        seen: set[str] = set()
        for tr in run.task_results:
            for k in tr.metrics:
                if k not in seen:
                    all_metrics.append(k)
                    seen.add(k)

        all_metrics = [
            k
            for k in all_metrics
            if all(
                isinstance(tr.metrics.get(k, 0.0), (int, float))
                for tr in run.task_results
                if not tr.error
            )
        ]

        lines = [
            "# Benchmark Report\n",
            f"**Model:** {run.model_id}  ",
            f"**Date:** {datetime.now(timezone.utc).isoformat()}  ",
            f"**Tasks:** {len(run.task_results)}  \n",
            "## Results\n",
            "| Task | Version | Docs | " + " | ".join(all_metrics) + " |",
            "|------|---------|------|" + "|".join("------" for _ in all_metrics) + "|",
        ]
        for tr in sorted(run.task_results, key=lambda t: t.task_name):
            vals = " | ".join(
                f"{tr.metrics.get(m, 0.0):.1f}%"
                if isinstance(tr.metrics.get(m, 0.0), (int, float))
                else "—"
                for m in all_metrics
            )
            err = f" (ERROR: {tr.error})" if tr.error else ""
            lines.append(f"| {tr.task_name}{err} | v{tr.version} | {tr.num_docs} | {vals} |")

        if run.composite_score is not None:
            lines.append("")
            lines.append("## Composite Score\n")
            se_str = ""
            if run.composite_se:
                lo = run.composite_se.get("ci_lower", 0.0)
                hi = run.composite_se.get("ci_upper", 0.0)
                se_str = f" (95% CI: [{lo:.1f}%, {hi:.1f}%])"
            lines.append(f"**{run.composite_score:.2f}%**{se_str}  ")

        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown report: %s", path)
        return path
