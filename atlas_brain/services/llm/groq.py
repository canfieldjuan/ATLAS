"""
Groq LLM Backend.

Uses Groq's OpenAI-compatible API for fast cloud inference.
Primary cloud provider for Atlas due to low latency.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

from ..base import BaseModelService
from ..protocols import Message, ModelInfo
from ..registry import register_llm

logger = logging.getLogger("atlas.llm.groq")


@register_llm("groq")
class GroqLLM(BaseModelService):
    """LLM service using Groq's API."""

    CAPABILITIES = ["text", "chat", "reasoning", "tool_calling"]

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Groq LLM.

        Args:
            model: Model name (e.g., "llama-3.3-70b-versatile")
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            base_url: API base URL
            **kwargs: Additional options
        """
        super().__init__(name="groq", model_id=model)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._loaded = False

    @property
    def model_info(self) -> ModelInfo:
        """Return model information."""
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self.is_loaded,
            device="cloud",
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        """Initialize HTTP clients."""
        if not self.api_key:
            raise ValueError("Groq API key not set. Set GROQ_API_KEY env var.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._sync_client = httpx.Client(timeout=60.0, headers=headers)
        self._client = httpx.AsyncClient(timeout=60.0, headers=headers)
        self._loaded = True
        logger.info("Groq LLM initialized: model=%s", self.model)

    def unload(self) -> None:
        """Close HTTP clients."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        if self._client:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._client.aclose())
            except RuntimeError:
                asyncio.run(self._client.aclose())
            self._client = None
        self._loaded = False
        logger.info("Groq LLM unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if service is loaded."""
        return self._loaded

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to OpenAI format."""
        result = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            result.append(m)
        return result

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Synchronous chat completion.

        Args:
            messages: List of Message objects
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional options

        Returns:
            Dict with response text and metadata
        """
        if not self._sync_client:
            raise RuntimeError("Groq LLM not loaded")

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = self._sync_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "").strip()

            logger.info("Groq chat: tokens=%s, content_len=%d",
                       data.get("usage", {}), len(content))

            return {
                "response": content,
                "message": {"role": "assistant", "content": content},
            }
        except httpx.HTTPError as e:
            logger.error("Groq chat error: %s", e)
            raise

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Chat with tool calling support.

        Args:
            messages: List of Message objects
            tools: List of tool schemas in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with response, tool_calls, and message
        """
        if not self._sync_client:
            raise RuntimeError("Groq LLM not loaded")

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            logger.info("Sending %d tools to Groq", len(tools))

        try:
            response = self._sync_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls", [])

            normalized_calls = []
            for tc in tool_calls:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                normalized_calls.append({
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args,
                    }
                })

            logger.info("Groq response: content_len=%d, tool_calls=%d",
                       len(content), len(normalized_calls))

            if normalized_calls:
                logger.info("Tool calls: %s",
                           [tc["function"]["name"] for tc in normalized_calls])

            return {
                "response": content.strip(),
                "tool_calls": normalized_calls,
                "message": message,
            }
        except httpx.HTTPError as e:
            logger.error("Groq chat_with_tools error: %s", e)
            raise

    async def chat_async(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Async chat completion.

        Args:
            messages: List of Message objects
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional options

        Returns:
            Generated response text
        """
        if not self._client:
            raise RuntimeError("Groq LLM not loaded")

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            return message.get("content", "").strip()
        except httpx.HTTPError as e:
            logger.error("Groq async chat error: %s", e)
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate text from a prompt."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)
