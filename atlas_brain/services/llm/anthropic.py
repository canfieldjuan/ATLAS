"""
Anthropic LLM Backend.

Uses the Anthropic Python SDK for Claude model inference.
Primary provider for email draft generation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Protocol, runtime_checkable

from ..base import BaseModelService
from ..protocols import Message, ModelInfo
from ..registry import register_llm

logger = logging.getLogger("atlas.llm.anthropic")


@runtime_checkable
class AnthropicBatchableLLM(Protocol):
    """Structural contract for an LLM that can drive the Anthropic batch path.

    The Anthropic Message Batches API is vendor-specific, so consumers
    in ``atlas_brain.services.b2b.anthropic_batch`` and the dispatch
    gates in the autonomous tasks need a type that says "this LLM
    exposes the Anthropic SDK async client surface". Today only
    ``AnthropicLLM`` satisfies this Protocol; a future adapter
    (e.g. an Anthropic-via-Vertex client) that exposes the same
    attribute surface would also satisfy it without subclassing.

    The Protocol is ``runtime_checkable`` so existing
    ``isinstance(llm, AnthropicLLM)`` patterns can switch to
    ``isinstance(llm, AnthropicBatchableLLM)`` and keep the same
    runtime-narrowing semantics. The check is structural (it walks
    attribute presence at the moment of the call), so subclasses
    and duck-typed adapters work without nominal inheritance.

    Attribute surface:

      * ``name`` -- short provider id (used by the FTL tracer's
        pricing lookup; ``AnthropicLLM`` returns ``"anthropic"``).
      * ``model`` -- the resolved model id passed into the Anthropic
        batch request body.
      * ``_async_client`` -- the Anthropic SDK ``AsyncAnthropic``
        client. Underscore-prefixed because ``AnthropicLLM`` exposes
        it that way today; renaming would be a separate
        breaking-change PR. Including this attribute in the Protocol
        is what makes it specific to Anthropic-shaped LLMs -- without
        it, the other providers (Ollama / OpenRouter / Together / Groq)
        would also pass the structural check on ``name`` + ``model``
        alone and incorrectly route into the Anthropic batch API.

        Note: the attribute is ``None`` until ``load()`` is called.
        Protocol satisfaction only requires the attribute to exist;
        the loaded-ness gate is a SEPARATE runtime concern that all
        call sites preserve via the companion check:

            isinstance(llm, AnthropicBatchableLLM)
                and getattr(llm, "_async_client", None) is not None
    """

    name: str
    model: str
    _async_client: Any | None

_ANTHROPIC_MODEL_ALIASES: dict[str, str] = {
    "claude-3-5-haiku-latest": "claude-haiku-4-5",
}

# Default character threshold above which the system prompt is wrapped
# in a cache_control content block. Below the threshold, the system
# prompt is passed as a plain string. The 1024-char cutoff matches
# Anthropic's recommended minimum for cacheable prefixes.
_DEFAULT_CACHE_THRESHOLD_CHARS: int = 1024


def convert_messages(
    messages: list[Message],
    *,
    cache_threshold_chars: int = _DEFAULT_CACHE_THRESHOLD_CHARS,
) -> tuple[str | list[dict], list[dict]]:
    """Convert ``Message`` objects to Anthropic API format.

    Anthropic requires system content as a separate parameter, not in
    the messages array. Returns ``(system_prompt_or_blocks,
    messages_list)`` where:

      * ``system_prompt_or_blocks`` is the system text as a plain
        string when its length is at or below ``cache_threshold_chars``,
        otherwise a list of content blocks with ``cache_control`` set
        so Anthropic caches the prefix and avoids re-tokenizing large
        skill prompts on every call.
      * ``messages_list`` is the user/assistant/tool message array
        formatted for the Anthropic Messages API. Assistant tool
        calls become ``tool_use`` content blocks; ``tool`` role
        messages become ``user`` messages with ``tool_result`` blocks
        and consecutive tool results coalesce into a single user
        message (Anthropic requires this).

    Pure: no I/O, no clock, no SDK dependency. Safe to call from
    any context that has ``Message`` objects.
    """
    system_parts: list[str] = []
    api_messages: list[dict] = []

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "assistant" and getattr(msg, "tool_calls", None):
            # Anthropic: assistant tool calls are content blocks.
            content_blocks: list[dict] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": call["function"]["name"],
                    "input": call["function"].get("arguments", {}),
                })
            api_messages.append({"role": "assistant", "content": content_blocks})
        elif msg.role == "tool":
            # Anthropic: tool results are user messages with
            # tool_result blocks. Consecutive tool results are
            # coalesced into one user message.
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": getattr(msg, "tool_call_id", "") or "",
                "content": msg.content,
            }
            if (
                api_messages
                and api_messages[-1]["role"] == "user"
                and isinstance(api_messages[-1]["content"], list)
            ):
                api_messages[-1]["content"].append(tool_result_block)
            else:
                api_messages.append({
                    "role": "user",
                    "content": [tool_result_block],
                })
        else:
            api_messages.append({"role": msg.role, "content": msg.content})

    system_text = "\n\n".join(system_parts)

    # Wrap large system prompts in a cache_control block so Anthropic
    # caches the prefix (saves ~90% on repeated system prompt tokens
    # when skill prompts are 5-20 KB and reused across calls).
    if len(system_text) > cache_threshold_chars:
        return [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ], api_messages

    return system_text, api_messages


def _normalize_anthropic_model(model: str) -> str:
    """Map deprecated Anthropic model aliases to currently supported ids."""
    normalized = str(model or "").strip()
    if not normalized:
        return "claude-haiku-4-5"
    return _ANTHROPIC_MODEL_ALIASES.get(normalized, normalized)


@register_llm("anthropic")
class AnthropicLLM(BaseModelService):
    """LLM service using Anthropic's API."""

    CAPABILITIES = ["text", "chat", "reasoning", "tool_calling"]

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        normalized_model = _normalize_anthropic_model(model)
        super().__init__(name="anthropic", model_id=normalized_model)
        self.model = normalized_model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._sync_client = None
        self._async_client = None
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
        """Initialize Anthropic clients."""
        if not self.api_key:
            raise ValueError("Anthropic API key not set. Set ANTHROPIC_API_KEY env var.")

        import anthropic

        self._sync_client = anthropic.Anthropic(api_key=self.api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self._loaded = True
        logger.info("Anthropic LLM initialized: model=%s", self.model)

    def unload(self) -> None:
        """Close clients and release HTTP connections."""
        if self._sync_client:
            try:
                self._sync_client.close()
            except Exception:
                pass
            self._sync_client = None
        if self._async_client:
            # AsyncAnthropic.close() is a coroutine -- best-effort from sync ctx
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_client.close())
                else:
                    loop.run_until_complete(self._async_client.close())
            except Exception:
                pass
            self._async_client = None
        self._loaded = False
        logger.info("Anthropic LLM unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | list[dict], list[dict]]:
        """Backwards-compatible alias for the module-level
        :func:`convert_messages` helper. New callers should import the
        public function directly.
        """
        return convert_messages(messages)

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronous chat completion."""
        if not self._sync_client:
            raise RuntimeError("Anthropic LLM not loaded")

        system_prompt, api_messages = self._convert_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        request_timeout = kwargs.get("timeout")

        try:
            client = self._sync_client.with_options(timeout=request_timeout) if request_timeout else self._sync_client
            response = client.messages.create(**create_kwargs)

            # Extract text from content blocks
            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
            content = "\n".join(text_parts).strip()

            logger.info(
                "Anthropic chat: input_tokens=%d, output_tokens=%d, content_len=%d",
                response.usage.input_tokens,
                response.usage.output_tokens,
                len(content),
            )

            request_id = getattr(response, "id", None) or ""
            cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", None)
            cache_creation_tokens = getattr(response.usage, "cache_creation_input_tokens", None)

            return {
                "response": content,
                "message": {"role": "assistant", "content": content},
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "billable_input_tokens": response.usage.input_tokens,
                    "cached_tokens": int(cache_read_tokens or 0),
                    "cache_write_tokens": int(cache_creation_tokens or 0),
                },
                "_trace_meta": {
                    "api_endpoint": "https://api.anthropic.com/v1/messages",
                    "provider_request_id": request_id,
                    "billable_input_tokens": response.usage.input_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                    "cached_tokens": int(cache_read_tokens or 0),
                    "cache_write_tokens": int(cache_creation_tokens or 0),
                },
            }
        except Exception as e:
            logger.error("Anthropic chat error: %s", e)
            raise

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat with tool calling support."""
        if not self._sync_client:
            raise RuntimeError("Anthropic LLM not loaded")

        system_prompt, api_messages = self._convert_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        request_timeout = kwargs.get("timeout")

        # Convert OpenAI tool format to Anthropic format
        if tools:
            anthropic_tools = []
            for tool in tools:
                func = tool.get("function", tool)
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            create_kwargs["tools"] = anthropic_tools

        try:
            client = self._sync_client.with_options(timeout=request_timeout) if request_timeout else self._sync_client
            response = client.messages.create(**create_kwargs)

            text_parts = []
            normalized_calls = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    normalized_calls.append({
                        "id": block.id,
                        "function": {
                            "name": block.name,
                            "arguments": block.input if isinstance(block.input, dict) else {},
                        }
                    })

            content = "\n".join(text_parts).strip()

            logger.info(
                "Anthropic response: content_len=%d, tool_calls=%d",
                len(content), len(normalized_calls),
            )
            request_id = getattr(response, "id", None) or ""
            cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", None)
            cache_creation_tokens = getattr(response.usage, "cache_creation_input_tokens", None)

            return {
                "response": content,
                "tool_calls": normalized_calls,
                "message": {"role": "assistant", "content": content},
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "billable_input_tokens": response.usage.input_tokens,
                    "cached_tokens": int(cache_read_tokens or 0),
                    "cache_write_tokens": int(cache_creation_tokens or 0),
                },
                "_trace_meta": {
                    "api_endpoint": "https://api.anthropic.com/v1/messages",
                    "provider_request_id": request_id,
                    "billable_input_tokens": response.usage.input_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                    "cached_tokens": int(cache_read_tokens or 0),
                    "cache_write_tokens": int(cache_creation_tokens or 0),
                },
            }
        except Exception as e:
            logger.error("Anthropic chat_with_tools error: %s", e)
            raise

    async def chat_async(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Async chat completion."""
        if not self._async_client:
            raise RuntimeError("Anthropic LLM not loaded")

        system_prompt, api_messages = self._convert_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        request_timeout = kwargs.get("timeout")

        try:
            client = self._async_client.with_options(timeout=request_timeout) if request_timeout else self._async_client
            response = await client.messages.create(**create_kwargs)

            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)

            return "\n".join(text_parts).strip()
        except Exception as e:
            logger.error("Anthropic async chat error: %s", e)
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate text from a prompt."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)
