"""
Anthropic LLM Backend.

Uses the Anthropic Python SDK for Claude model inference.
Primary provider for email draft generation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from ..base import BaseModelService
from ..protocols import Message, ModelInfo
from ..registry import register_llm

logger = logging.getLogger("atlas.llm.anthropic")


@register_llm("anthropic")
class AnthropicLLM(BaseModelService):
    """LLM service using Anthropic's API."""

    CAPABILITIES = ["text", "chat", "reasoning", "tool_calling"]

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="anthropic", model_id=model)
        self.model = model
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
        """Convert Message objects to Anthropic format.

        Anthropic requires system content as a separate param, not in messages.
        Returns (system_prompt_or_blocks, messages_list).

        When the system prompt exceeds 1024 chars, it is returned as a list of
        content blocks with ``cache_control`` set so Anthropic can cache the
        prefix and avoid re-tokenizing large skill prompts on every call.
        """
        system_parts: list[str] = []
        api_messages: list[dict] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "assistant" and getattr(msg, "tool_calls", None):
                # Anthropic: assistant tool calls are content blocks
                content_blocks = []
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
                # Anthropic: tool results are user messages with tool_result blocks
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": getattr(msg, "tool_call_id", "") or "",
                    "content": msg.content,
                }
                # Coalesce consecutive tool results into one user message
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

        # Enable prompt caching for large system prompts (skill prompts are
        # 5-20 KB and reused across calls).  Anthropic caches the prefix when
        # cache_control is set, saving ~90% on repeated system prompt tokens.
        if len(system_text) > 1024:
            return [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ], api_messages

        return system_text, api_messages

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

        try:
            response = self._sync_client.messages.create(**create_kwargs)

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

            return {
                "response": content,
                "message": {"role": "assistant", "content": content},
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "_trace_meta": {
                    "api_endpoint": "https://api.anthropic.com/v1/messages",
                    "provider_request_id": request_id,
                    "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", None),
                    "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", None),
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
            response = self._sync_client.messages.create(**create_kwargs)

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

            return {
                "response": content,
                "tool_calls": normalized_calls,
                "message": {"role": "assistant", "content": content},
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
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

        try:
            response = await self._async_client.messages.create(**create_kwargs)

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
