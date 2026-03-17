from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any
import httpx

from luau_bench.models import GenerationResult, ModelAdapter, ModelConfig

logger = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"

_THINKING_MODELS = frozenset(
    {
        "claude-opus-4-5",
        "claude-opus-4-20250514",
        "claude-sonnet-4-5",
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-6",
    }
)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 529})
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 60.0


class AnthropicAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._url = config.base_url or _API_URL
        self._model = config.model_name or "claude-sonnet-4-6"
        self._thinking_budget: int = int(config.extra_params.get("thinking_budget", 0))
        if not self._api_key:
            logger.warning("No Anthropic API key. Set ANTHROPIC_API_KEY or pass api_key.")

    async def generate(self, system: str, user: str) -> GenerationResult:
        payload = self._build_payload(user, system)
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    t0 = time.perf_counter()
                    resp = await client.post(self._url, json=payload, headers=self._headers())
                    latency = (time.perf_counter() - t0) * 1000.0

                if resp.status_code in _RETRYABLE_STATUS:
                    delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
                    logger.warning(
                        "Anthropic API %d on attempt %d/%d - retrying in %.1fs",
                        resp.status_code,
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                    )
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                return self._parse_response(resp.json(), latency)

            except httpx.TimeoutException:
                delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
                logger.warning(
                    "Anthropic API timeout on attempt %d/%d - retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(delay)
                else:
                    raise

        raise RuntimeError("Anthropic adapter exhausted retries.")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
        }
        if self._thinking_budget > 0:
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
        return headers

    def _build_payload(self, user: str, system: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            payload["system"] = system
        if self.config.stop_sequences:
            payload["stop_sequences"] = self.config.stop_sequences

        temp = self.config.temperature
        top_p = self.config.top_p
        if temp != 1.0:
            payload["temperature"] = temp
        elif top_p != 1.0:
            payload["top_p"] = top_p

        if self._thinking_budget > 0 and self._model in _THINKING_MODELS:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}
            payload["temperature"] = 1
            payload.pop("top_p", None)
            logger.debug("Extended thinking enabled (budget=%d tokens)", self._thinking_budget)

        return payload

    @staticmethod
    def _parse_response(data: dict[str, Any], latency_ms: float) -> GenerationResult:
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
            latency_ms=latency_ms,
            model_id=data.get("model", ""),
            raw_response=data,
        )

    async def health_check(self) -> bool:
        return bool(self._api_key)
