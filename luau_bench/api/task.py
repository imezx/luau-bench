from __future__ import annotations

import abc
import copy
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jinja2

logger = logging.getLogger(__name__)

_JINJA_ENV = jinja2.Environment(undefined=jinja2.Undefined)

_FEWSHOT_SEP = "\n\n---\n\n"


@dataclass
class MetricSpec:
    metric: str
    args: dict[str, Any] = field(default_factory=dict)
    higher_is_better: bool = True
    primary: bool = False


@dataclass
class FilterSpec:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class TaskConfig:
    task: str
    group: Optional[str] = None
    version: float = 0.0

    # "local" -> docs are inline in the YAML.
    # "jsonl" -> dataset_name is a path to a .jsonl file.
    # "huggingface" -> dataset_name is a HuggingFace Hub repo.
    dataset_path: str = "local"
    dataset_name: Optional[str] = None
    dataset_base_dir: str = ""
    docs: list[dict[str, Any]] = field(default_factory=list)

    system_prompt: str = ""
    doc_to_text: str = ""
    doc_to_target: str = ""
    doc_to_choices: str = ""  # Jinja2 template -> JSON list of choices (loglikelihood tasks)

    output_type: str = "generate_until"

    generation_kwargs: GenerationConfig = field(default_factory=GenerationConfig)

    num_fewshot: int = 0
    fewshot_seed: int = 42

    filters: list[FilterSpec] = field(default_factory=list)

    metric_list: list[MetricSpec] = field(default_factory=list)

    num_samples: int = 1

    metadata: dict[str, Any] = field(default_factory=dict)


class Task(abc.ABC):
    @abc.abstractmethod
    def get_docs(self) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def build_prompt(self, doc: dict[str, Any]) -> dict[str, str]: ...

    @abc.abstractmethod
    def get_metric_specs(self) -> list[MetricSpec]: ...

    @abc.abstractmethod
    def get_filter_specs(self) -> list[FilterSpec]: ...

    @abc.abstractmethod
    def get_generation_config(self) -> GenerationConfig: ...

    @abc.abstractmethod
    def get_config(self) -> TaskConfig: ...


class ConfigurableTask(Task):
    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self._docs: Optional[list[dict[str, Any]]] = None

    def get_docs(self) -> list[dict[str, Any]]:
        if self._docs is None:
            self._docs = self._load_docs()
        return self._docs

    def build_prompt(self, doc: dict[str, Any]) -> dict[str, str]:
        system = self._render(self.config.system_prompt, doc)
        user_body = self._render(self.config.doc_to_text, doc)

        if self.config.num_fewshot > 0:
            prefix = self._build_fewshot_prefix(doc)
            user = prefix + user_body
        else:
            user = user_body

        return {"system": system, "user": user}

    def get_target(self, doc: dict[str, Any]) -> str:
        return self._render(self.config.doc_to_target, doc)

    def get_choices(self, doc: dict[str, Any]) -> list[str]:
        """
        render ``doc_to_choices`` and parse the resulting JSON list.
        it returns an empty list if ``doc_to_choices`` is not set.
        the template should render to a JSON array, e.g. ``["choice A", "choice B"]``.
        """
        import json as _json

        template = self.config.doc_to_choices
        if not template:
            return []
        rendered = self._render(template, doc)
        try:
            choices = _json.loads(rendered)
            if isinstance(choices, list):
                return [str(c) for c in choices]
        except (_json.JSONDecodeError, ValueError):
            pass
        # fallback
        return [c.strip() for c in rendered.split("|") if c.strip()]

    def get_metric_specs(self) -> list[MetricSpec]:
        return self.config.metric_list

    def get_filter_specs(self) -> list[FilterSpec]:
        return self.config.filters

    def get_generation_config(self) -> GenerationConfig:
        return self.config.generation_kwargs

    def get_config(self) -> TaskConfig:
        return self.config

    def _load_docs(self) -> list[dict[str, Any]]:
        kind = self.config.dataset_path

        if kind == "local":
            return copy.deepcopy(self.config.docs)

        if kind == "jsonl":
            return self._load_jsonl(self.config.dataset_name or "")

        if kind == "huggingface":
            return self._load_huggingface()

        raise NotImplementedError(
            f"Dataset loader '{kind}' is not implemented. "
            "Supported: 'local', 'jsonl', 'huggingface'. "
            "Use dataset_path: local with inline docs for the simplest case."
        )

    def _load_jsonl(self, path_str: str) -> list[dict[str, Any]]:
        path = Path(path_str)
        if not path.is_absolute() and self.config.dataset_base_dir:
            path = Path(self.config.dataset_base_dir) / path

        if not path.exists():
            raise FileNotFoundError(
                f"JSONL dataset not found: {path}  (base dir: {self.config.dataset_base_dir!r})"
            )

        docs: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON on line %d of %s: %s",
                        lineno,
                        path,
                        exc,
                    )

        logger.info("Loaded %d docs from JSONL: %s", len(docs), path)
        return docs

    def _load_huggingface(self) -> list[dict[str, Any]]:
        """
        ``pip install luau-bench[datasets]``
        """
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "The HuggingFace loader requires the 'datasets' package. "
                "Install it with:  pip install luau-bench[datasets]"
            )

        repo = self.config.dataset_name or ""
        config_name = self.config.metadata.get("dataset_config") or None
        split = str(self.config.metadata.get("dataset_split", "test"))

        logger.info(
            "Loading HuggingFace dataset '%s' (config=%s, split=%s)",
            repo,
            config_name,
            split,
        )
        dataset = load_dataset(repo, config_name, split=split, trust_remote_code=True)
        docs = [dict(row) for row in dataset]
        logger.info("Loaded %d docs from HuggingFace '%s'", len(docs), repo)
        return docs

    def _build_fewshot_prefix(self, current_doc: dict[str, Any]) -> str:
        pool = [d for d in self.get_docs() if d is not current_doc]
        if not pool:
            return ""

        rng = random.Random(self.config.fewshot_seed)
        n = min(self.config.num_fewshot, len(pool))
        shots = rng.sample(pool, n)

        parts: list[str] = []
        for shot in shots:
            question = self._render(self.config.doc_to_text, shot)
            answer = self._render(self.config.doc_to_target, shot)
            parts.append(f"{question}\n{answer}")
        return _FEWSHOT_SEP.join(parts) + _FEWSHOT_SEP

    @staticmethod
    def _render(template_str: str, doc: dict[str, Any]) -> str:
        if not template_str:
            return ""
        try:
            tmpl = _JINJA_ENV.from_string(template_str)
            return tmpl.render(**doc).strip()
        except jinja2.TemplateError as exc:
            logger.warning("Jinja2 render error: %s (template: %s)", exc, template_str[:80])
            return template_str


def parse_task_config(raw: dict[str, Any]) -> TaskConfig:
    gen_raw = raw.get("generation_kwargs", {})
    gen = GenerationConfig(
        max_tokens=int(gen_raw.get("max_tokens", 4096)),
        temperature=float(gen_raw.get("temperature", 0.0)),
        top_p=float(gen_raw.get("top_p", 1.0)),
        stop_sequences=gen_raw.get("stop_sequences", []),
    )

    filters = [FilterSpec(name=f["name"], args=f.get("args", {})) for f in raw.get("filters", [])]

    metrics = []
    for m in raw.get("metric_list", []):
        metrics.append(
            MetricSpec(
                metric=m["metric"],
                args=m.get("args", {}),
                higher_is_better=m.get("higher_is_better", True),
                primary=m.get("primary", False),
            )
        )

    return TaskConfig(
        task=raw.get("task", ""),
        group=raw.get("group"),
        version=float(raw.get("version", raw.get("metadata", {}).get("version", 0))),
        dataset_path=raw.get("dataset_path", "local"),
        dataset_name=raw.get("dataset_name"),
        dataset_base_dir=raw.get("_base_dir", ""),
        docs=raw.get("docs", []),
        system_prompt=raw.get("system_prompt", ""),
        doc_to_text=raw.get("doc_to_text", ""),
        doc_to_target=raw.get("doc_to_target", ""),
        doc_to_choices=raw.get("doc_to_choices", ""),
        output_type=raw.get("output_type", "generate_until"),
        generation_kwargs=gen,
        num_fewshot=int(raw.get("num_fewshot", 0)),
        fewshot_seed=int(raw.get("fewshot_seed", 42)),
        filters=filters,
        metric_list=metrics,
        num_samples=int(raw.get("num_samples", 1)),
        metadata=raw.get("metadata", {}),
    )
