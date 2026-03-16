from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_METRICS: dict[str, Callable] = {}


def register_metric(name: str) -> Callable:
    """Decorator that registers a metric function under *name*.

    A metric function has the signature::

        def my_metric(
            predictions: list[str],
            references: list[str],
            *,
            docs: list[dict] | None = None,
            **kwargs,
        ) -> dict[str, float]:
            ...

    It must return a dict whose keys are score names and values are floats.
    """

    def _wrap(fn: Callable) -> Callable:
        if name in _METRICS:
            logger.debug("Overwriting metric '%s'", name)
        _METRICS[name] = fn
        return fn

    return _wrap


def get_metric(name: str) -> Callable:
    if name not in _METRICS:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {sorted(_METRICS)}. "
            "Did you forget to register it?"
        )
    return _METRICS[name]


def list_metrics() -> list[str]:
    return sorted(_METRICS)



_FILTERS: dict[str, Callable] = {}


def register_filter(name: str) -> Callable:
    """Decorator that registers an output-processing filter under *name*.

    A filter function transforms a raw model output string into a cleaned
    string::

        def my_filter(text: str, **kwargs) -> str:
            ...
    """

    def _wrap(fn: Callable) -> Callable:
        if name in _FILTERS:
            logger.debug("Overwriting filter '%s'", name)
        _FILTERS[name] = fn
        return fn

    return _wrap


def get_filter(name: str) -> Callable:
    if name not in _FILTERS:
        raise KeyError(
            f"Unknown filter '{name}'. Available: {sorted(_FILTERS)}. "
            "Did you forget to register it?"
        )
    return _FILTERS[name]


def list_filters() -> list[str]:
    return sorted(_FILTERS)



_TASK_CONFIGS: dict[str, dict[str, Any]] = {}
_TASK_GROUPS: dict[str, list[str]] = {}


def register_task_config(name: str, config: dict[str, Any]) -> None:
    if name in _TASK_CONFIGS:
        logger.debug("Overwriting task config '%s'", name)
    _TASK_CONFIGS[name] = config


def get_task_config(name: str) -> dict[str, Any]:
    if name not in _TASK_CONFIGS:
        raise KeyError(
            f"Unknown task '{name}'. Available: {sorted(_TASK_CONFIGS)}. "
            "Pass --include-path to add task directories."
        )
    return _TASK_CONFIGS[name]


def register_group(name: str, task_names: list[str]) -> None:
    _TASK_GROUPS[name] = task_names


def resolve_tasks(names: list[str]) -> list[str]:
    """Expand group names into individual task names, preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in _TASK_GROUPS:
            for t in _TASK_GROUPS[name]:
                if t not in seen:
                    result.append(t)
                    seen.add(t)
        else:
            if name not in seen:
                result.append(name)
                seen.add(name)
    return result


def list_tasks() -> list[str]:
    return sorted(_TASK_CONFIGS)


def list_groups() -> dict[str, list[str]]:
    return dict(_TASK_GROUPS)
