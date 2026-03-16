from __future__ import annotations

import asyncio
import math
import re
from typing import Any, Optional

from luau_bench.api import register_metric


@register_metric("exact_match")
def exact_match(
    predictions: list[str],
    references: list[str],
    *,
    docs: Optional[list[dict]] = None,
    ignore_case: bool = False,
    ignore_whitespace: bool = False,
    **kwargs,
) -> dict[str, float]:
    correct = 0
    for pred, ref in zip(predictions, references):
        p, r = pred, ref
        if ignore_case:
            p, r = p.lower(), r.lower()
        if ignore_whitespace:
            p, r = p.strip(), r.strip()
        if p == r:
            correct += 1
    n = len(predictions) or 1
    per_doc = [
        1.0 if (p.lower() == r.lower() if ignore_case else p == r) else 0.0
        for p, r in zip(
            [p.strip() if ignore_whitespace else p for p in predictions],
            [r.strip() if ignore_whitespace else r for r in references],
        )
    ]
    return {
        "exact_match": correct / n * 100.0,
        "_per_doc_scores": per_doc,
    }


@register_metric("contains")
def contains(
    predictions: list[str],
    references: list[str],
    *,
    docs: Optional[list[dict]] = None,
    ignore_case: bool = False,
    **kwargs,
) -> dict[str, float]:
    correct = 0
    for pred, ref in zip(predictions, references):
        p, r = pred, ref
        if ignore_case:
            p, r = p.lower(), r.lower()
        if r in p:
            correct += 1
    n = len(predictions) or 1
    return {"contains": correct / n * 100.0}


@register_metric("regex_match")
def regex_match(
    predictions: list[str],
    references: list[str],
    *,
    docs: Optional[list[dict]] = None,
    pattern: str = "",
    flags: int = 0,
    **kwargs,
) -> dict[str, float]:
    correct = 0
    for i, pred in enumerate(predictions):
        pat = pattern or (references[i] if i < len(references) else "")
        try:
            if re.search(pat, pred, flags):
                correct += 1
        except re.error:
            pass
    n = len(predictions) or 1
    return {"regex_match": correct / n * 100.0}


@register_metric("loglikelihood_acc")
def loglikelihood_acc(
    predictions: list[str],
    references: list[str],
    *,
    docs: Optional[list[dict]] = None,
    ignore_case: bool = False,
    **kwargs,
) -> dict[str, float]:
    correct = 0
    for pred, ref in zip(predictions, references):
        p, r = (
            (pred.strip().lower(), ref.strip().lower())
            if ignore_case
            else (pred.strip(), ref.strip())
        )
        if p == r:
            correct += 1
    n = len(predictions) or 1
    per_doc = [
        1.0
        if (p.strip().lower() == r.strip().lower() if ignore_case else p.strip() == r.strip())
        else 0.0
        for p, r in zip(predictions, references)
    ]
    return {
        "loglikelihood_acc": correct / n * 100.0,
        "_per_doc_scores": per_doc,
    }


@register_metric("pass_at_k")
def pass_at_k(
    predictions: list[str],
    references: list[str],
    *,
    docs: Optional[list[dict]] = None,
    k: int = 1,
    num_samples: int = 1,
    _pass_flags: Optional[list[bool]] = None,
    **kwargs,
) -> dict[str, float]:
    if _pass_flags is None:
        _pass_flags = [p.strip() == r.strip() for p, r in zip(predictions, references)]

    total = len(_pass_flags)
    if total == 0:
        return {"pass_at_k": 0.0}

    effective_n = max(1, num_samples)
    num_docs = max(1, total // effective_n)

    per_doc: list[float] = []
    for di in range(num_docs):
        flags = _pass_flags[di * effective_n : (di + 1) * effective_n]
        n = len(flags)
        c = sum(flags)
        kk = min(k, n)

        if n == 0 or c == 0:
            per_doc.append(0.0)
        elif c >= n or n - c < kk:
            per_doc.append(1.0)
        else:
            log_ratio = sum(math.log(max(1, n - c - i)) - math.log(n - i) for i in range(kk))
            per_doc.append(1.0 - math.exp(log_ratio))

    return {"pass_at_k": sum(per_doc) / len(per_doc) * 100.0}


@register_metric("luau_exec")
async def luau_exec(
    predictions: list[str],
    references: list[str],
    *,
    docs: Optional[list[dict]] = None,
    timeout: float = 30.0,
    runtime_path: str = "",
    num_samples: int = 1,
    **kwargs,
) -> dict[str, Any]:
    from luau_bench.runtime.executor import LuauExecutor

    executor = LuauExecutor(runtime_path=runtime_path or None, timeout=timeout)
    n = len(predictions)

    if not executor.available:
        return {
            "luau_exec_pass_rate": 0.0,
            "luau_exec_non_strict_rate": 0.0,
            "luau_exec_strict_rate": 0.0,
            "luau_exec_stderr_rate": 0.0,
            "luau_exec_total_passed": 0.0,
            "luau_exec_total_tests": 0.0,
            "luau_exec_available": 0.0,
            "_pass_flags": [False] * n,
            "_exec_details": [None] * n,
        }

    resolved_docs = docs or [{} for _ in predictions]

    harnesses: list[Optional[str]] = []
    for pred, doc in zip(predictions, resolved_docs):
        template = doc.get("test_harness", "")
        harnesses.append(template.replace("{{CODE}}", pred) if template else None)

    async def _run_one(script: Optional[str]) -> Optional[dict]:
        return await executor.run_script(script) if script is not None else None

    exec_results: list[Optional[dict]] = list(
        await asyncio.gather(*(_run_one(h) for h in harnesses))
    )

    total_passed = total_tests = 0
    strict_hits = 0
    nonstrict_hits = 0
    stderr_hits = 0
    n_with_harness = 0

    pass_flags: list[bool] = []
    exec_details: list[Optional[dict]] = []
    per_doc_scores: list[float] = []

    for res in exec_results:
        if res is None:
            pass_flags.append(False)
            exec_details.append(None)
            continue

        n_with_harness += 1
        total_passed += res["passed"]
        total_tests += res["total"]

        all_passed = res["total"] > 0 and res["passed"] == res["total"]
        has_stderr = bool((res.get("stderr") or "").strip()) or res.get("timed_out", False)

        if all_passed:
            nonstrict_hits += 1
        if all_passed and not has_stderr:
            strict_hits += 1
        if has_stderr:
            stderr_hits += 1

        pass_flags.append(all_passed)
        exec_details.append(res)
        per_doc_scores.append(1.0 if all_passed else 0.0)

    denom = max(1, n_with_harness)
    norm_rate = (total_passed / total_tests * 100.0) if total_tests > 0 else 0.0
    non_strict_rate = nonstrict_hits / denom * 100.0
    strict_rate = strict_hits / denom * 100.0
    stderr_rate = stderr_hits / denom * 100.0

    return {
        "luau_exec_pass_rate": norm_rate,
        "luau_exec_non_strict_rate": non_strict_rate,
        "luau_exec_strict_rate": strict_rate,
        "luau_exec_stderr_rate": stderr_rate,
        "luau_exec_total_passed": float(total_passed),
        "luau_exec_total_tests": float(total_tests),
        "luau_exec_available": 1.0,
        "_pass_flags": pass_flags,
        "_exec_details": exec_details,
        "_per_doc_scores": per_doc_scores,
    }


@register_metric("luau_static_analysis")
def luau_static_analysis(
    predictions: list[str],
    references: list[str],
    *,
    docs: Optional[list[dict]] = None,
    dimensions: Optional[list[str]] = None,
    **kwargs,
) -> dict[str, float]:
    from luau_bench.analysis.static import analyze, check_code_validity

    all_dims = dimensions or [
        "type_coverage",
        "comment_ratio",
        "locality_ratio",
        "cyclomatic_complexity",
        "nesting_depth",
        "uses_strict_mode",
        "modern_feature_count",
        "code_validity",
    ]

    accum: dict[str, float] = {d: 0.0 for d in all_dims}
    n = len(predictions) or 1

    for pred in predictions:
        a = analyze(pred)
        v = check_code_validity(pred)

        dim_values: dict[str, float] = {
            "type_coverage": a.type_coverage,
            "comment_ratio": a.comment_ratio,
            "locality_ratio": a.locality_ratio,
            "cyclomatic_complexity": float(a.cyclomatic_complexity),
            "nesting_depth": float(a.max_nesting_depth),
            "uses_strict_mode": 100.0 if a.has_strict_mode else 0.0,
            "modern_feature_count": float(
                sum(
                    1
                    for feat in [
                        a.uses_if_expression,
                        a.uses_string_interpolation,
                        a.uses_compound_assignment,
                        a.uses_generics,
                        a.uses_table_freeze,
                        a.uses_table_clone,
                        a.uses_typeof,
                        a.uses_const,
                    ]
                    if feat
                )
            ),
            "code_validity": v["confidence"] * 100.0,
        }
        for d in all_dims:
            accum[d] += dim_values.get(d, 0.0)

    return {f"static_{d}": accum[d] / n for d in all_dims}


@register_metric("luau_analyze")
async def luau_analyze(
    predictions: list[str],
    references: list[str],
    *,
    docs: Optional[list[dict]] = None,
    analyzer_path: str = "",
    timeout: float = 15.0,
    **kwargs,
) -> dict[str, Any]:
    from luau_bench.runtime.executor import LuauAnalyzer

    analyzer = LuauAnalyzer(
        analyzer_path=analyzer_path or None,
        timeout=timeout,
    )
    n = len(predictions)

    if not analyzer.available:
        return {
            "luau_analyze_clean_rate": 0.0,
            "luau_analyze_error_rate": 0.0,
            "luau_analyze_avg_errors": 0.0,
            "luau_analyze_avg_warnings": 0.0,
            "luau_analyze_available": 0.0,
            "_per_doc_scores": [0.0] * n,
            "_analyze_details": [None] * n,
        }

    results: list[dict] = list(await asyncio.gather(*(analyzer.analyze(p) for p in predictions)))

    total_errors = 0
    total_warnings = 0
    clean_count = 0
    error_count = 0
    per_doc: list[float] = []
    details: list[Any] = []

    for res in results:
        e = res.get("errors", 0)
        w = res.get("warnings", 0)
        total_errors += e
        total_warnings += w
        is_clean = e == 0 and w == 0
        if is_clean:
            clean_count += 1
        if e > 0:
            error_count += 1
        per_doc.append(1.0 if is_clean else 0.0)
        details.append(res.get("diagnostics", []))

    denom = max(1, n)
    return {
        "luau_analyze_clean_rate": clean_count / denom * 100.0,
        "luau_analyze_error_rate": error_count / denom * 100.0,
        "luau_analyze_avg_errors": total_errors / denom,
        "luau_analyze_avg_warnings": total_warnings / denom,
        "luau_analyze_available": 1.0,
        "_per_doc_scores": per_doc,
        "_analyze_details": details,
    }
