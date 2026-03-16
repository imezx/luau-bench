from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from luau_bench.models import GenerationResult

logger = logging.getLogger(__name__)

_SUBDIR = "generations"


class GenerationCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self._dir = Path(cache_dir).expanduser() / _SUBDIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(
        model_id: str,
        task_name: str,
        system: str,
        user: str,
        sample_index: int,
        temperature: float,
        max_tokens: int,
    ) -> str:
        parts = {
            "model": model_id,
            "task": task_name,
            "system": system,
            "user": user,
            "sample": sample_index,
            "temperature": round(temperature, 6),
            "max_tokens": max_tokens,
        }
        raw = json.dumps(parts, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, key: str) -> Optional[GenerationResult]:
        path = self._dir / f"{key}.json"
        if not path.exists():
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            return GenerationResult(**data)
        except Exception as exc:
            logger.warning("Cache read error for key %s: %s", key, exc)
            self._misses += 1
            return None

    def set(self, key: str, result: GenerationResult) -> None:
        path = self._dir / f"{key}.json"
        try:
            data = dataclasses.asdict(result)
            path.write_text(json.dumps(data, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Cache write error for key %s: %s", key, exc)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def clear(self) -> int:
        removed = 0
        for p in self._dir.glob("*.json"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        return removed
