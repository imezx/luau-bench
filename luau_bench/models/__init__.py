from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    """configuration for connecting to an LLM inference endpoint."""

    provider: str = "openai_compat"
    model_name: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """raw output from a single model generation call."""

    text: str
    code: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model_id: str = ""
    raw_response: Optional[dict] = None


@dataclass
class LoglikelihoodResult:
    """per-continuation scoring result for a loglikelihood request."""

    log_sum: float = 0.0
    tokens: int = 0
    normalized_logprob: float = 0.0


class ModelAdapter(abc.ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abc.abstractmethod
    async def generate(self, system: str, user: str) -> GenerationResult: ...

    async def loglikelihood_batch(
        self,
        system: str,
        user: str,
        continuations: list[str],
    ) -> list[LoglikelihoodResult]:
        """
        score a list of candidate continuations given a prompt.

        returns one ``LoglikelihoodResult`` per continuation in the same order.

        the default implementation raises ``NotImplementedError``.
        subclasses that support logprob access (e.g. OpenAI-compatible backends) should
        override this method.
        """
        raise NotImplementedError(
            f"Provider '{self.config.provider}' does not support loglikelihood scoring. "
            "Use an OpenAI-compatible backend (vllm, tgi, llama.cpp) which exposes "
            "the /v1/completions endpoint with echo=True and logprobs=1."
        )

    async def health_check(self) -> bool:
        """verify that the model endpoint is reachable."""
        try:
            result = await self.generate(
                system="You are a helpful assistant.",
                user="Reply with exactly: OK",
            )
            return len(result.text) > 0
        except Exception:
            return False

    def model_id(self) -> str:
        return self.config.model_name or "unknown"

    def apply_generation_config(self, gen_config) -> None:
        """temporarily override generation params from a task config."""
        if gen_config.max_tokens:
            self.config.max_tokens = gen_config.max_tokens
        if gen_config.temperature is not None:
            self.config.temperature = gen_config.temperature
        if gen_config.top_p is not None:
            self.config.top_p = gen_config.top_p
        if gen_config.stop_sequences:
            self.config.stop_sequences = gen_config.stop_sequences


PROVIDER_MAP = {
    "openai_compat": "openai_compat",
    "openai": "openai_compat",
    "vllm": "openai_compat",
    "tgi": "openai_compat",
    "lmstudio": "openai_compat",
    "anthropic": "anthropic",
    "ollama": "ollama",
}


def create_adapter(config: ModelConfig) -> ModelAdapter:
    """instantiate the right adapter for the configured provider."""
    provider = PROVIDER_MAP.get(config.provider, config.provider)

    if provider == "openai_compat":
        from luau_bench.models.openai_compat import OpenAICompatAdapter

        return OpenAICompatAdapter(config)

    if provider == "anthropic":
        from luau_bench.models.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(config)

    if provider == "ollama":
        from luau_bench.models.ollama_adapter import OllamaAdapter

        return OllamaAdapter(config)

    raise ValueError(f"Unknown provider '{config.provider}'. Supported: {sorted(PROVIDER_MAP)}")
