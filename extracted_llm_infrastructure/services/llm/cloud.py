"""
Cloud LLM Backend with Fallback.

Uses Groq as primary (fastest latency) with Together.ai fallback.
Automatically retries with fallback on failure.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ..base import BaseModelService
from ..protocols import Message, ModelInfo
from ..registry import register_llm

logger = logging.getLogger("atlas.llm.cloud")


@register_llm("cloud")
class CloudLLM(BaseModelService):
    """Cloud LLM service with Groq primary and Together fallback."""

    CAPABILITIES = ["text", "chat", "reasoning", "tool_calling"]

    def __init__(
        self,
        groq_model: str = "llama-3.3-70b-versatile",
        together_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        groq_api_key: Optional[str] = None,
        together_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Cloud LLM with Groq primary and Together fallback.

        Args:
            groq_model: Groq model name
            together_model: Together model name
            groq_api_key: Groq API key (defaults to GROQ_API_KEY env var)
            together_api_key: Together API key (defaults to TOGETHER_API_KEY env var)
            **kwargs: Additional options
        """
        super().__init__(name="cloud", model_id=groq_model)
        self._groq_model = groq_model
        self._together_model = together_model
        self._groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._together_api_key = together_api_key or os.environ.get("TOGETHER_API_KEY", "")
        self._primary: Any = None
        self._fallback: Any = None
        self._loaded = False

    @property
    def model_info(self) -> ModelInfo:
        """Return model information."""
        return ModelInfo(
            name=self.name,
            model_id=f"groq:{self._groq_model}|together:{self._together_model}",
            is_loaded=self.is_loaded,
            device="cloud",
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        """Initialize both cloud providers."""
        from .groq import GroqLLM
        from .together import TogetherLLM

        if self._groq_api_key:
            self._primary = GroqLLM(
                model=self._groq_model,
                api_key=self._groq_api_key,
            )
            self._primary.load()
            logger.info("Cloud LLM primary: Groq (%s)", self._groq_model)
        else:
            logger.warning("Groq API key not set, skipping primary")

        if self._together_api_key:
            self._fallback = TogetherLLM(
                model=self._together_model,
                api_key=self._together_api_key,
            )
            self._fallback.load()
            logger.info("Cloud LLM fallback: Together (%s)", self._together_model)
        else:
            logger.warning("Together API key not set, skipping fallback")

        if not self._primary and not self._fallback:
            raise ValueError("At least one cloud API key must be set")

        self._loaded = True
        logger.info("Cloud LLM initialized with fallback support")

    def unload(self) -> None:
        """Close both providers."""
        if self._primary:
            self._primary.unload()
            self._primary = None
        if self._fallback:
            self._fallback.unload()
            self._fallback = None
        self._loaded = False
        logger.info("Cloud LLM unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if service is loaded."""
        return self._loaded

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Chat with automatic fallback.

        Tries Groq first, falls back to Together on failure.
        """
        if self._primary:
            try:
                result = self._primary.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return result
            except Exception as e:
                logger.warning("Groq failed, trying Together fallback: %s", e)

        if self._fallback:
            try:
                result = self._fallback.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                logger.info("Together fallback succeeded")
                return result
            except Exception as e:
                logger.error("Together fallback also failed: %s", e)
                raise

        raise RuntimeError("No cloud provider available")

    async def chat_async(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Async chat with automatic fallback.

        Tries Groq first, falls back to Together on failure.
        """
        if self._primary:
            try:
                result = await self._primary.chat_async(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return result
            except Exception as e:
                logger.warning("Groq async failed, trying fallback: %s", e)

        if self._fallback:
            try:
                result = await self._fallback.chat_async(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                logger.info("Together async fallback succeeded")
                return result
            except Exception as e:
                logger.error("Together async fallback also failed: %s", e)
                raise

        raise RuntimeError("No cloud provider available")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate text from a prompt with fallback."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Chat with tools and automatic fallback.

        Tries Groq first, falls back to Together on failure.
        """
        if self._primary:
            try:
                result = self._primary.chat_with_tools(
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return result
            except Exception as e:
                logger.warning("Groq chat_with_tools failed, trying fallback: %s", e)

        if self._fallback:
            try:
                result = self._fallback.chat_with_tools(
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                logger.info("Together fallback succeeded for chat_with_tools")
                return result
            except Exception as e:
                logger.error("Together fallback also failed: %s", e)
                raise

        raise RuntimeError("No cloud provider available")
