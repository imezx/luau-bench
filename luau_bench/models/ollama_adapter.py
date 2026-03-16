from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from luau_bench.models import GenerationResult, ModelAdapter, ModelConfig

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:11434"


class OllamaAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        base = (config.base_url or _DEFAULT_URL).rstrip("/")
        self._chat_url = f"{base}/api/chat"
        self._tags_url = f"{base}/api/tags"
        self._model = config.model_name or "codellama"

    async def generate(self, system: str, user: str) -> GenerationResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            },
        }
        if self.config.stop_sequences:
            payload["options"]["stop"] = self.config.stop_sequences

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                resp = await client.post(self._chat_url, json=payload)
                resp.raise_for_status()
            except httpx.ConnectError:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self._chat_url}. Is 'ollama serve' running?"
                )

        latency = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        raw_text = data.get("message", {}).get("content", "")

        return GenerationResult(
            text=raw_text,
            finish_reason="stop" if data.get("done") else "length",
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency,
            model_id=self._model,
            raw_response=data,
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self._tags_url)
                return resp.status_code == 200
        except Exception:
            return False
