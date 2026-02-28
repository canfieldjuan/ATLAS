"""
vLLM Backend.

Uses vLLM's OpenAI-compatible API for high-throughput local inference.
vLLM uses continuous batching to process multiple requests simultaneously,
making it ideal for batch workloads like deep enrichment.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from ..base import BaseModelService
from ..protocols import Message, ModelInfo
from ..registry import register_llm

logger = logging.getLogger("atlas.llm.vllm")


@register_llm("vllm")
class VLLMLLM(BaseModelService):
    """LLM service using a local vLLM server."""

    CAPABILITIES = ["text", "chat"]

    def __init__(
        self,
        model: str = "Qwen/Qwen3-14B",
        base_url: str = "http://localhost:8000",
        timeout: float = 300,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="vllm", model_id=model)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._loaded = False

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self.is_loaded,
            device="cuda",
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        headers = {"Content-Type": "application/json"}
        self._sync_client = httpx.Client(timeout=self.timeout, headers=headers)
        self._client = httpx.AsyncClient(timeout=self.timeout, headers=headers)
        self._loaded = True
        logger.info("vLLM initialized: model=%s, base_url=%s", self.model, self.base_url)

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
        logger.info("vLLM unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._sync_client:
            raise RuntimeError("vLLM not loaded")

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:
            response = self._sync_client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "").strip()

            logger.info(
                "vLLM chat: tokens=%s, content_len=%d",
                data.get("usage", {}),
                len(content),
            )

            return {
                "response": content,
                "message": {"role": "assistant", "content": content},
            }
        except httpx.HTTPError as e:
            logger.error("vLLM chat error: %s", e)
            raise

    async def chat_async(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        if not self._client:
            raise RuntimeError("vLLM not loaded")

        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            return message.get("content", "").strip()
        except httpx.HTTPError as e:
            logger.error("vLLM async chat error: %s", e)
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
