"""
LLM Client Wrapper with exponential backoff for rate limits.
Wraps graphiti-core's OpenAI client to add retry logic for 429 errors.
"""
import asyncio
import logging
import random
from typing import Any

from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)


class _RetryingRateLimitMixin:
    rate_limit_max_retries: int
    base_delay: float
    max_delay: float

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate response with exponential backoff on rate limits."""
        last_error = None

        for attempt in range(self.rate_limit_max_retries + 1):
            try:
                return await super().generate_response(
                    messages=messages,
                    response_model=response_model,
                    max_tokens=max_tokens,
                    model_size=model_size,
                    group_id=group_id,
                    prompt_name=prompt_name,
                    **kwargs,
                )
            except RateLimitError as e:
                last_error = e

                if attempt >= self.rate_limit_max_retries:
                    logger.error(
                        f"Rate limit: max retries ({self.rate_limit_max_retries}) exceeded"
                    )
                    raise

                delay = min(
                    self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.max_delay
                )

                logger.warning(
                    f"Rate limited (attempt {attempt + 1}/{self.rate_limit_max_retries + 1}). "
                    f"Waiting {delay:.1f}s before retry..."
                )

                await asyncio.sleep(delay)

        raise last_error or RateLimitError("Max retries exceeded")


class RetryingOpenAIClient(_RetryingRateLimitMixin, OpenAIClient):
    """
    OpenAI client with exponential backoff for rate limit errors.

    Default settings:
    - max_retries: 5
    - base_delay: 2 seconds
    - max_delay: 60 seconds
    - exponential backoff with jitter
    - reasoning/verbosity only set for reasoning models (gpt-5, o1, o3, etc.)
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = 16384,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        # reasoning/verbosity only supported by reasoning models (gpt-5*, o1*, o3*)
        # For standard models like gpt-4o-mini, these must be None
        super().__init__(config=config, cache=cache, max_tokens=max_tokens, reasoning=reasoning, verbosity=verbosity)
        self.rate_limit_max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

class RetryingOpenAIGenericClient(_RetryingRateLimitMixin, OpenAIGenericClient):
    """OpenAI-compatible client with exponential backoff for rate limit errors."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = 16384,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
    ):
        try:
            super().__init__(config=config, cache=cache, max_tokens=max_tokens)
        except TypeError as exc:
            if "max_tokens" not in str(exc):
                raise
            super().__init__(config=config, cache=cache)
            self.max_tokens = max_tokens
        self.rate_limit_max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay


def create_retrying_llm_client(
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> RetryingOpenAIClient | RetryingOpenAIGenericClient:
    """
    Factory function to create a RetryingOpenAIClient.

    Args:
        api_key: OpenAI API key
        base_url: Optional custom base URL (for Azure, vLLM, etc.)
        model: Model name (default: gpt-4o-mini)
        max_retries: Max retry attempts for rate limits (default: 5)
        base_delay: Initial delay in seconds (default: 2.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)

    Returns:
        RetryingOpenAIClient for OpenAI direct, otherwise an OpenAI-compatible
        generic client for OpenRouter, vLLM, llama.cpp, and similar endpoints.
    """
    model_name = model or "gpt-4o-mini"
    config = LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )

    normalized_base_url = (base_url or "").lower().rstrip("/")

    if normalized_base_url and "api.openai.com" not in normalized_base_url:
        return RetryingOpenAIGenericClient(
            config=config,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )

    # Only set reasoning/verbosity for reasoning models (gpt-5*, o1*, o3*)
    is_reasoning_model = any(
        model_name.startswith(prefix) for prefix in ("gpt-5", "o1", "o3")
    )
    reasoning = "minimal" if is_reasoning_model else None
    verbosity = "low" if is_reasoning_model else None

    return RetryingOpenAIClient(
        config=config,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        reasoning=reasoning,
        verbosity=verbosity,
    )
