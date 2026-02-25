"""
OpenRouter LLM Backend.

Uses OpenRouter's OpenAI-compatible API for access to many cloud models.
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

logger = logging.getLogger("atlas.llm.openrouter")


@register_llm("openrouter")
class OpenRouterLLM(BaseModelService):
    """LLM service using OpenRouter's API."""

    CAPABILITIES = ["text", "chat", "reasoning", "tool_calling"]

    def __init__(
        self,
        model: str = "anthropic/claude-haiku",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name="openrouter", model_id=model)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._loaded = False

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self.is_loaded,
            device="cloud",
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        if not self.api_key:
            raise ValueError("OpenRouter API key not set. Set OPENROUTER_API_KEY env var.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", ""),
            "X-Title": "Atlas Brain",
        }
        self._sync_client = httpx.Client(timeout=120.0, headers=headers)
        self._client = httpx.AsyncClient(timeout=120.0, headers=headers)
        self._loaded = True
        logger.info("OpenRouter LLM initialized: model=%s", self.model)

    def unload(self) -> None:
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
        logger.info("OpenRouter LLM unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
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
        if not self._sync_client:
            raise RuntimeError("OpenRouter LLM not loaded")

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

            usage = data.get("usage", {})
            logger.info("OpenRouter chat: model=%s tokens=%s content_len=%d",
                       self.model, usage, len(content))

            return {
                "response": content,
                "message": {"role": "assistant", "content": content},
            }
        except httpx.HTTPError as e:
            logger.error("OpenRouter chat error: %s", e)
            raise

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._sync_client:
            raise RuntimeError("OpenRouter LLM not loaded")

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

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

            return {
                "response": content.strip(),
                "tool_calls": normalized_calls,
                "message": message,
            }
        except httpx.HTTPError as e:
            logger.error("OpenRouter chat_with_tools error: %s", e)
            raise

    async def chat_async(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        if not self._client:
            raise RuntimeError("OpenRouter LLM not loaded")

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
            logger.error("OpenRouter async chat error: %s", e)
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)
