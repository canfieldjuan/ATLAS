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
        model: str = "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ",
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
        limits = httpx.Limits(max_connections=500, max_keepalive_connections=200)
        self._sync_client = httpx.Client(timeout=self.timeout, headers=headers)
        self._client = httpx.AsyncClient(timeout=self.timeout, headers=headers, limits=limits)
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

    def _build_payload(
        self, messages: list[Message], max_tokens: int, temperature: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from atlas_brain.config import settings

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._supports_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        guided_json = kwargs.get("guided_json")
        if guided_json is not None:
            if settings.llm.vllm_guided_json_enabled:
                payload["structured_outputs"] = {"json": guided_json}
            else:
                logger.warning(
                    "guided_json requested but disabled by ATLAS_LLM__VLLM_GUIDED_JSON_ENABLED",
                )

        response_format = kwargs.get("response_format")
        if response_format is not None:
            fmt_type = str(response_format.get("type", "")).lower() if isinstance(response_format, dict) else ""
            if guided_json is not None and settings.llm.vllm_guided_json_enabled:
                logger.debug(
                    "Skipping response_format because guided_json structured outputs are enabled for vLLM",
                )
                return payload
            # json_object is lightweight (just ensures valid JSON) -- always allow.
            # json_schema requires guided decoding -- gate behind vllm_guided_json_enabled.
            if fmt_type == "json_schema" and not settings.llm.vllm_guided_json_enabled:
                logger.warning(
                    "json_schema response_format requested but disabled by ATLAS_LLM__VLLM_GUIDED_JSON_ENABLED",
                )
            else:
                payload["response_format"] = response_format
        return payload

    @property
    def _supports_thinking(self) -> bool:
        # All Qwen3 variants (including Instruct) default to thinking mode
        return "qwen3" in self.model.lower()

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._sync_client:
            raise RuntimeError("vLLM not loaded")

        payload = self._build_payload(messages, max_tokens, temperature, **kwargs)

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
                "vLLM chat: tokens=%s, content_len=%d, finish_reason=%s",
                data.get("usage", {}),
                len(content),
                choice.get("finish_reason"),
            )

            return {
                "response": content,
                "message": {"role": "assistant", "content": content},
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage", {}),
            }
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if hasattr(e, 'response') else ""
            logger.error("vLLM chat error: %s | body=%s", e, body)
            raise
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

        payload = self._build_payload(messages, max_tokens, temperature, **kwargs)

        try:
            response = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "").strip()

            logger.info(
                "vLLM async chat: tokens=%s, content_len=%d",
                data.get("usage", {}),
                len(content),
            )

            return content
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
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
