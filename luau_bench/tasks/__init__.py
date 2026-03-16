from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from luau_bench.api import register_group, register_task_config
from luau_bench.api.task import ConfigurableTask, TaskConfig, parse_task_config

logger = logging.getLogger(__name__)


def load_tasks_from_path(path: Path) -> list[ConfigurableTask]:
    if not path.exists():
        logger.warning("Task path does not exist: %s", path)
        return []

    yaml_files = sorted(path.rglob("*.yaml")) + sorted(path.rglob("*.yml"))
    logger.info("Found %d YAML file(s) in %s", len(yaml_files), path)

    raw_configs: dict[str, dict[str, Any]] = {}
    group_defs: list[dict[str, Any]] = []

    for yf in yaml_files:
        try:
            raws = _load_yaml_file(yf, search_dir=yf.parent)
        except Exception as exc:
            logger.error("Failed to load %s: %s", yf, exc)
            continue
        for raw in raws:
            raw.setdefault("_base_dir", str(yf.parent))
            if _is_group_def(raw):
                group_defs.append(raw)
            elif "task" in raw:
                name = raw["task"]
                if name in raw_configs:
                    logger.warning("Duplicate task '%s' — skipping %s", name, yf)
                else:
                    raw_configs[name] = raw
            else:
                logger.debug("Skipping entry without 'task' key in %s", yf)

    for gd in group_defs:
        group_name = gd.get("group", "")
        subtasks = gd.get("task", [])
        if isinstance(subtasks, list) and group_name:
            flat: list[str] = []
            for st in subtasks:
                if isinstance(st, str):
                    flat.append(st)
                elif isinstance(st, dict) and "task" in st:
                    flat.append(st["task"])
            register_group(group_name, flat)
            logger.info("Registered group '%s' with %d subtask(s)", group_name, len(flat))

    tasks: list[ConfigurableTask] = []
    for name, raw in raw_configs.items():
        try:
            config = parse_task_config(raw)
            register_task_config(name, raw)
            tasks.append(ConfigurableTask(config))
        except Exception as exc:
            logger.error("Failed to parse task '%s': %s", name, exc)

    logger.info("Loaded %d task(s)", len(tasks))
    return tasks


def load_task_dirs(dirs: list[str | Path]) -> list[ConfigurableTask]:
    all_tasks: list[ConfigurableTask] = []
    seen: set[str] = set()
    for d in dirs:
        for task in load_tasks_from_path(Path(d)):
            name = task.config.task
            if name not in seen:
                all_tasks.append(task)
                seen.add(name)
    return all_tasks


def _load_yaml_file(path: Path, search_dir: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return []

    # normalize
    if isinstance(data, dict):
        entries = data.get("tasks", [data]) if "tasks" in data else [data]
    elif isinstance(data, list):
        entries = data
    else:
        return []

    resolved: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry = _resolve_include(entry, search_dir)
        resolved.append(entry)
    return resolved


def _resolve_include(raw: dict[str, Any], search_dir: Path) -> dict[str, Any]:
    include_name = raw.pop("include", None)
    if not include_name:
        return raw

    include_path = search_dir / include_name
    if not include_path.exists():
        logger.warning("Include file not found: %s", include_path)
        return raw

    base_text = include_path.read_text(encoding="utf-8")
    base = yaml.safe_load(base_text) or {}
    if not isinstance(base, dict):
        return raw

    merged = _deep_merge(base, raw)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _is_group_def(raw: dict[str, Any]) -> bool:
    return (
        "group" in raw
        and "task" in raw
        and isinstance(raw["task"], list)
        and "doc_to_text" not in raw
    )
