"""
Ollama LLM Backend.

Uses Ollama's HTTP API for inference, supporting any model Ollama can run.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

from ..base import BaseModelService
from ..protocols import Message, ModelInfo
from ..registry import register_llm

logger = logging.getLogger("atlas.llm.ollama")


@register_llm("ollama")
class OllamaLLM(BaseModelService):
    """LLM service using Ollama's HTTP API."""

    CAPABILITIES = ["text", "chat", "reasoning"]

    def __init__(
        self,
        model: str = "qwen3-coder:30b",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Ollama LLM.

        Args:
            model: Ollama model name (e.g., "qwen3-coder:30b", "hermes3:8b")
            base_url: Ollama API base URL
            **kwargs: Additional options
        """
        super().__init__(name="ollama", model_id=model)
        self.model = model
        self.base_url = base_url.rstrip("/")
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
            device="api",  # Ollama runs on its own process
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        """Initialize HTTP clients."""
        self._sync_client = httpx.Client(timeout=120.0)
        self._client = httpx.AsyncClient(timeout=120.0)
        self._loaded = True
        logger.info("Ollama LLM initialized: model=%s, url=%s", self.model, self.base_url)

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
        logger.info("Ollama LLM unloaded")

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
            raise RuntimeError("Ollama LLM not loaded")

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "keep_alive": "30m",  # Keep model in VRAM
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = self._sync_client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                response_text = data.get("message", {}).get("content", "").strip()
                done_reason = data.get("done_reason", "unknown")

                # Log timing details to understand cache behavior
                prompt_eval_count = data.get("prompt_eval_count", 0)
                prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1_000_000  # ns to ms
                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 0) / 1_000_000  # ns to ms
                total_duration = data.get("total_duration", 0) / 1_000_000  # ns to ms

                logger.info(
                    "Ollama chat: prompt_tokens=%d (%.1fms), gen_tokens=%d (%.1fms), total=%.1fms",
                    prompt_eval_count, prompt_eval_duration,
                    eval_count, eval_duration,
                    total_duration
                )
                logger.info("Ollama chat: done_reason=%s, content_len=%d, response='%s'",
                           done_reason, len(response_text), response_text)

                # Retry if model hit token limit before producing output
                if done_reason == "length" and not response_text and attempt < max_retries:
                    logger.warning("Empty response with done_reason=length, retrying (%d/%d)",
                                 attempt + 1, max_retries)
                    continue

                return {
                    "response": response_text,
                    "message": {"role": "assistant", "content": response_text},
                    "prompt_eval_count": prompt_eval_count,
                    "eval_count": eval_count,
                    "done_reason": done_reason,
                    "prompt_eval_duration_ms": prompt_eval_duration,
                    "eval_duration_ms": eval_duration,
                    "total_duration_ms": total_duration,
                }
            except httpx.HTTPError as e:
                logger.error("Ollama chat error: %s", e)
                if attempt < max_retries:
                    continue
                raise

        return {"response": "", "message": {"role": "assistant", "content": ""},
                "prompt_eval_count": 0, "eval_count": 0}

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Chat with optional tool calling support.

        Args:
            messages: List of Message objects
            tools: List of tool schemas in Ollama format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with response, tool_calls, and message
        """
        if not self._sync_client:
            raise RuntimeError("Ollama LLM not loaded")

        ollama_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            # Include tool_calls for assistant messages
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            ollama_messages.append(m)

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "keep_alive": "30m",  # Keep model in VRAM
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if tools:
            payload["tools"] = tools
            logger.info("Sending %d tools to Ollama", len(tools))
            # Debug: log first tool schema
            if tools:
                import json
                logger.debug("First tool schema: %s", json.dumps(tools[0], indent=2)[:500])

        try:
            import json
            logger.debug("Ollama request payload keys: %s", list(payload.keys()))
            # Log messages with tool_calls info
            msg_summary = []
            for m in ollama_messages:
                summary = {"role": m["role"], "content": m["content"][:100] if m["content"] else ""}
                if m.get("tool_calls"):
                    summary["tool_calls"] = [tc.get("function", {}).get("name") for tc in m["tool_calls"]]
                msg_summary.append(summary)
            logger.info("Ollama messages: %s", msg_summary)
            response = self._sync_client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Debug: log full response message
            logger.debug("Full Ollama response data keys: %s", list(data.keys()))

            msg = data.get("message", {})
            logger.debug("Message keys: %s", list(msg.keys()))
            tool_calls = msg.get("tool_calls", [])
            raw_content = msg.get("content", "")
            logger.info("Ollama response: content_len=%d, tool_calls=%d, done_reason=%s",
                       len(raw_content), len(tool_calls), data.get("done_reason", "unknown"))
            logger.info("Ollama raw content: '%s'", raw_content)
            if tool_calls:
                logger.info("Tool calls received: %s", [tc.get("function", {}).get("name") for tc in tool_calls])
            return {
                "response": msg.get("content", "").strip(),
                "tool_calls": tool_calls,
                "message": msg,
            }
        except httpx.HTTPError as e:
            logger.error("Ollama chat_with_tools error: %s", e)
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
            raise RuntimeError("Ollama LLM not loaded")

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "keep_alive": "30m",  # Keep model in VRAM
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except httpx.HTTPError as e:
            logger.error("Ollama chat error: %s", e)
            raise

    async def chat_stream_async(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Async streaming chat completion - yields tokens as generated.

        Args:
            messages: List of Message objects
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Token strings as they are generated
        """
        import json as json_module

        if not self._client:
            raise RuntimeError("Ollama LLM not loaded")

        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json_module.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        break
        except httpx.HTTPError as e:
            logger.error("Ollama streaming chat error: %s", e)
            raise

    def process_text(self, query: str, **kwargs: Any) -> str:
        """
        Process a text query.

        Args:
            query: Text query
            **kwargs: Additional options

        Returns:
            Generated response
        """
        messages = [Message(role="user", content=query)]
        return self.chat(messages, **kwargs)

    async def process_text_async(self, query: str, **kwargs: Any) -> str:
        """
        Async process a text query.

        Args:
            query: Text query
            **kwargs: Additional options

        Returns:
            Generated response
        """
        messages = [Message(role="user", content=query)]
        return await self.chat_async(messages, **kwargs)

    async def prefill_async(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Prefill the KV cache without generating tokens.

        Sends the prompt to the model with num_predict=0 to populate the KV cache.
        This allows subsequent requests with the same prefix to skip the prefill phase,
        reducing time-to-first-token (TTFT) for progressive prompting.

        Args:
            messages: List of Message objects to prefill

        Returns:
            Dict with prefill timing info
        """
        if not self._client:
            raise RuntimeError("Ollama LLM not loaded")

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "keep_alive": "30m",  # Keep model in VRAM
            "options": {
                "num_predict": 0,  # Only prefill, no generation
            },
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract timing info
            prompt_eval_duration = data.get("prompt_eval_duration", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            
            logger.debug(
                "Prefill complete: %d tokens in %.2f ms",
                prompt_eval_count,
                prompt_eval_duration / 1_000_000,  # ns to ms
            )
            
            return {
                "prompt_tokens": prompt_eval_count,
                "prefill_time_ms": prompt_eval_duration / 1_000_000,
            }
        except httpx.HTTPError as e:
            logger.warning("Prefill request failed: %s", e)
            return {"prompt_tokens": 0, "prefill_time_ms": 0}
