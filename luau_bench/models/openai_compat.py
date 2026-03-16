from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from luau_bench.models import (
    GenerationResult,
    LoglikelihoodResult,
    ModelAdapter,
    ModelConfig,
)

logger = logging.getLogger(__name__)

_SENTINEL = "\x00SPLIT\x00"


class OpenAICompatAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        base = config.base_url.rstrip("/") or "http://localhost:8000/v1"
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        self._chat_url = f"{base}/chat/completions"
        self._completions_url = f"{base}/completions"  # logprobs
        self._models_url = f"{base}/models"
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    async def generate(self, system: str, user: str) -> GenerationResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences
        payload.update(self.config.extra_params)

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    self._chat_url,
                    json=payload,
                    headers=self._headers,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "HTTP %s from %s: %s",
                    exc.response.status_code,
                    self._chat_url,
                    exc.response.text[:500],
                )
                raise
            except httpx.ConnectError:
                raise ConnectionError(
                    f"Cannot connect to {self._chat_url}. Is the inference server running?"
                )

        latency = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        choice = data["choices"][0]
        raw = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})

        return GenerationResult(
            text=raw,
            finish_reason=choice.get("finish_reason", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency,
            model_id=data.get("model", self.config.model_name),
            raw_response=data,
        )

    async def loglikelihood_batch(
        self,
        system: str,
        user: str,
        continuations: list[str],
    ) -> list[LoglikelihoodResult]:
        if not continuations:
            return []

        prefix = (system.strip() + "\n\n" if system else "") + user.strip()

        async def _score(cont: str) -> LoglikelihoodResult:
            full_prompt = prefix + " " + _SENTINEL + " " + cont.strip()
            payload: dict[str, Any] = {
                "model": self.config.model_name,
                "prompt": full_prompt,
                "max_tokens": 1,
                "echo": True,
                "logprobs": 1,
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    self._completions_url,
                    json=payload,
                    headers=self._headers,
                )
                resp.raise_for_status()

            lp = resp.json()["choices"][0]["logprobs"]
            tokens = lp.get("tokens") or []
            lprobs = lp.get("token_logprobs") or []

            split_idx = None
            for i, tok in enumerate(tokens):
                if _SENTINEL in str(tok):
                    split_idx = i
            if split_idx is None:
                logger.warning(
                    "loglikelihood sentinel not found in token list — "
                    "falling back to full sequence. Scores may be unreliable."
                )
                split_idx = 0

            cont_lps: list[float] = [
                v for v in lprobs[split_idx + 1 : -1] if isinstance(v, (int, float))
            ]

            if not cont_lps:
                last = lprobs[-1] if lprobs else None
                cont_lps = [float(last)] if isinstance(last, (int, float)) else [-100.0]
                logger.debug(
                    "Empty continuation logprobs for %r — using generated token fallback",
                    cont[:40],
                )

            log_sum = sum(cont_lps)
            n_tokens = len(cont_lps)
            normalised = log_sum / max(1, n_tokens)

            return LoglikelihoodResult(
                log_sum=log_sum,
                tokens=n_tokens,
                normalized_logprob=normalised,
            )

        results = await asyncio.gather(*(_score(c) for c in continuations))
        return list(results)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self._models_url, headers=self._headers)
                return resp.status_code == 200
        except Exception:
            return False
