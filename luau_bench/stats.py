from __future__ import annotations

import math
import random
from collections import defaultdict

Z_95 = 1.96


def formula_se(scores: list[float]) -> float:
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    return math.sqrt(variance / n)


def bootstrap_se(
    scores: list[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> float:
    n = len(scores)
    if n < 2:
        return 0.0

    rng = random.Random(seed)
    means = [sum(scores[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)]
    mu = sum(means) / n_resamples
    variance = sum((m - mu) ** 2 for m in means) / (n_resamples - 1)
    return math.sqrt(variance)


def clustered_se(
    scores: list[float],
    cluster_ids: list[str],
) -> float:
    if len(scores) < 2:
        return 0.0

    clusters: dict[str, list[float]] = defaultdict(list)
    for score, cid in zip(scores, cluster_ids):
        clusters[cid].append(score)

    G = len(clusters)
    if G < 2:
        return formula_se(scores)

    n_total = len(scores)
    grand_mean = sum(scores) / n_total

    cluster_contribs = [len(vs) * (sum(vs) / len(vs) - grand_mean) for vs in clusters.values()]
    ss = sum(c**2 for c in cluster_contribs)
    variance = (G / (G - 1)) * ss / (n_total**2)
    return math.sqrt(max(0.0, variance))


def confidence_interval(
    mean: float,
    se: float,
    z: float = Z_95,
) -> tuple[float, float]:
    margin = z * se
    lo = max(0.0, mean - margin)
    hi = min(100.0, mean + margin)
    return (lo, hi)


def ci_overlaps(
    ci_a: tuple[float, float],
    ci_b: tuple[float, float],
) -> bool:
    a_lo, a_hi = ci_a
    b_lo, b_hi = ci_b
    return a_lo < b_hi and b_lo < a_hi


def all_se(
    scores: list[float],
    cluster_ids: list[str] | None = None,
    n_resamples: int = 1000,
    bootstrap_seed: int = 42,
    z: float = Z_95,
) -> dict[str, float]:
    """
    ``formula`` - formula SE
    ``bootstrap`` - bootstrap SE (used for CI and display)
    ``clustered`` - cluster-robust SE
    ``n`` - sample size
    ``mean``  - sample mean (proportion, 0-1)
    ``ci_lower`` - lower bound of 95% CI (as %, 0-100)
    ``ci_upper`` - upper bound of 95% CI (as %, 0-100)
    ``ci_level`` - confidence level (0.95)
    ``z`` - z multiplier used (1.96)

    all score values should be in [0, 1]. CI bounds are returned in [0, 100]
    to match the percentage scale used throughout luau-bench.
    """
    n = len(scores)
    mean = (sum(scores) / n) if n > 0 else 0.0
    ids = cluster_ids or [str(i) for i in range(n)]

    bse = bootstrap_se(scores, n_resamples=n_resamples, seed=bootstrap_seed)

    mean_pct = mean * 100.0
    bse_pct = bse * 100.0
    ci_lo, ci_hi = confidence_interval(mean_pct, bse_pct, z=z)

    return {
        "formula": formula_se(scores),
        "bootstrap": bse,
        "clustered": clustered_se(scores, ids),
        "n": float(n),
        "mean": mean,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "ci_level": 0.95,
        "z": z,
    }
