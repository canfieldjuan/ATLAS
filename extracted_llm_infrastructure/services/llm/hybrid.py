"""
Hybrid LLM Backend -- per-query routing between local and cloud.

Routes each method to the best backend automatically:
- chat(), chat_async(), chat_stream_async(), prefill_async(), generate() -> local (Ollama)
- chat_with_tools() -> cloud (Groq/Together)

All methods accept a ``backend="local"|"cloud"`` kwarg to override default routing.
Falls back gracefully if one backend is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

from ..base import BaseModelService
from ..protocols import Message, ModelInfo
from ..registry import register_llm

logger = logging.getLogger("atlas.llm.hybrid")


@register_llm("hybrid")
class HybridLLM(BaseModelService):
    """Hybrid LLM that routes queries between local (Ollama) and cloud (Groq/Together)."""

    CAPABILITIES = ["text", "chat", "reasoning", "tool_calling", "streaming"]

    def __init__(
        self,
        local_kwargs: dict[str, Any] | None = None,
        cloud_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="hybrid", model_id="hybrid")
        self._local_kwargs = local_kwargs or {}
        self._cloud_kwargs = cloud_kwargs or {}
        self._local: Any = None
        self._cloud: Any = None
        self._loaded = False

    @property
    def model_info(self) -> ModelInfo:
        parts = []
        if self._local:
            parts.append(f"local:{self._local.model_info.model_id}")
        if self._cloud:
            parts.append(f"cloud:{self._cloud.model_info.model_id}")
        return ModelInfo(
            name=self.name,
            model_id="|".join(parts) or "hybrid:not-loaded",
            is_loaded=self.is_loaded,
            device="hybrid",
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        from .ollama import OllamaLLM
        from .cloud import CloudLLM

        # Load local (Ollama)
        try:
            self._local = OllamaLLM(**self._local_kwargs)
            self._local.load()
            logger.info("Hybrid: local backend loaded (Ollama %s)", self._local_kwargs.get("model", "default"))
        except Exception as e:
            logger.warning("Hybrid: local backend unavailable: %s", e)
            self._local = None

        # Load cloud (Groq/Together)
        try:
            self._cloud = CloudLLM(**self._cloud_kwargs)
            self._cloud.load()
            logger.info("Hybrid: cloud backend loaded")
        except Exception as e:
            logger.warning("Hybrid: cloud backend unavailable: %s", e)
            self._cloud = None

        if not self._local and not self._cloud:
            raise RuntimeError("Hybrid LLM: both local and cloud backends failed to load")

        self._loaded = True
        logger.info(
            "Hybrid LLM ready: local=%s, cloud=%s",
            "yes" if self._local else "no",
            "yes" if self._cloud else "no",
        )

    def unload(self) -> None:
        if self._local:
            self._local.unload()
            self._local = None
        if self._cloud:
            self._cloud.unload()
            self._cloud = None
        self._loaded = False
        logger.info("Hybrid LLM unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _pick(self, default: str, override: str | None = None) -> tuple[Any, str]:
        """Pick backend based on explicit override or default routing."""
        target = override or default

        if target == "local" and self._local:
            return self._local, "local"
        if target == "cloud" and self._cloud:
            return self._cloud, "cloud"

        # Fallback: try the other one
        if target == "cloud" and self._local:
            logger.warning("HybridLLM: cloud unavailable, falling back to local")
            return self._local, "local"
        if target == "local" and self._cloud:
            logger.warning("HybridLLM: local unavailable, falling back to cloud")
            return self._cloud, "cloud"

        raise RuntimeError("HybridLLM: no backend available")

    # -- chat() -> local -------------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        override = kwargs.pop("backend", None)
        backend, name = self._pick("local", override)
        logger.info("HybridLLM routed chat() to %s", name)
        try:
            result = backend.chat(messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
        except Exception:
            if name == "local" and self._cloud:
                logger.warning("HybridLLM: local chat() failed, retrying on cloud")
                result = self._cloud.chat(messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
                name = "cloud"
            else:
                raise
        result["routed_to"] = name
        return result

    # -- chat_with_tools() -> cloud --------------------------------------------

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        override = kwargs.pop("backend", None)
        backend, name = self._pick("cloud", override)
        logger.info("HybridLLM routed chat_with_tools() to %s", name)
        try:
            result = backend.chat_with_tools(
                messages=messages, tools=tools, max_tokens=max_tokens, temperature=temperature, **kwargs,
            )
        except Exception:
            if name == "cloud" and self._local:
                logger.warning("HybridLLM: cloud chat_with_tools() failed, retrying on local")
                result = self._local.chat_with_tools(
                    messages=messages, tools=tools, max_tokens=max_tokens, temperature=temperature, **kwargs,
                )
                name = "local"
            else:
                raise
        result["routed_to"] = name
        return result

    # -- chat_async() -> local -------------------------------------------------

    async def chat_async(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        override = kwargs.pop("backend", None)
        backend, name = self._pick("local", override)
        logger.info("HybridLLM routed chat_async() to %s", name)
        try:
            return await backend.chat_async(
                messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs,
            )
        except Exception:
            if name == "local" and self._cloud:
                logger.warning("HybridLLM: local chat_async() failed, retrying on cloud")
                return await self._cloud.chat_async(
                    messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs,
                )
            raise

    # -- chat_stream_async() -> local (Ollama-only) ----------------------------

    async def chat_stream_async(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        kwargs.pop("backend", None)
        if not self._local:
            raise RuntimeError("HybridLLM: streaming requires local (Ollama) backend")
        logger.info("HybridLLM routed chat_stream_async() to local")
        async for token in self._local.chat_stream_async(
            messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs,
        ):
            yield token

    # -- prefill_async() -> local (Ollama-only) --------------------------------

    async def prefill_async(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.pop("backend", None)
        if not self._local:
            raise RuntimeError("HybridLLM: prefill requires local (Ollama) backend")
        logger.info("HybridLLM routed prefill_async() to local")
        return await self._local.prefill_async(messages=messages, **kwargs)

    # -- generate() -> local ---------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        override = kwargs.pop("backend", None)
        backend, name = self._pick("local", override)
        logger.info("HybridLLM routed generate() to %s", name)
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        result = backend.chat(messages, max_tokens=max_tokens, temperature=temperature)
        result["routed_to"] = name
        return result
