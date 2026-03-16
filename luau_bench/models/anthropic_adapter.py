# https://platform.claude.com/docs/en/api/beta-headers
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from luau_bench.models import GenerationResult, ModelAdapter, ModelConfig

logger = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class AnthropicAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._url = config.base_url or _API_URL
        self._model = config.model_name or "claude-sonnet-4-6"
        if not self._api_key:
            logger.warning("No Anthropic API key. Set ANTHROPIC_API_KEY or pass api_key.")

    async def generate(self, system: str, user: str) -> GenerationResult:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            payload["system"] = system
        if self.config.stop_sequences:
            payload["stop_sequences"] = self.config.stop_sequences

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(self._url, json=payload, headers=headers)
            resp.raise_for_status()

        latency = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        raw_text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})

        return GenerationResult(
            text=raw_text,
            finish_reason=data.get("stop_reason", ""),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_ms=latency,
            model_id=data.get("model", self._model),
            raw_response=data,
        )

    async def health_check(self) -> bool:
        return bool(self._api_key)
