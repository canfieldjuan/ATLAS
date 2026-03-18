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

    def _structured_response_format(self, **kwargs: Any) -> dict[str, Any] | None:
        """Prefer json_schema when guided_json is supplied."""
        guided_json = kwargs.get("guided_json")
        if isinstance(guided_json, dict) and guided_json:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": str(guided_json.get("title") or "atlas_response"),
                    "strict": True,
                    "schema": guided_json,
                },
            }
        response_format = kwargs.get("response_format")
        return response_format if isinstance(response_format, dict) else None

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        stripped = text.lstrip()
        return stripped.startswith("{") or stripped.startswith("[")

    @staticmethod
    def _structured_plugins(
        structured_response_format: dict[str, Any] | None,
        **kwargs: Any,
    ) -> list[dict[str, Any]] | None:
        """Add response-healing for structured non-streaming requests."""
        plugins = kwargs.get("plugins")
        normalized: list[dict[str, Any]] = []
        if isinstance(plugins, list):
            for plugin in plugins:
                if isinstance(plugin, dict) and plugin.get("id"):
                    normalized.append(plugin)
        if structured_response_format and not any(p.get("id") == "response-healing" for p in normalized):
            normalized.append({"id": "response-healing"})
        return normalized or None

    # Models that use reasoning tokens and require max_completion_tokens
    # instead of max_tokens, and ignore temperature.
    _REASONING_MODELS = frozenset({
        "openai/o4-mini", "openai/o4-mini-high", "openai/o3", "openai/o3-mini",
        "openai/o3-mini-high", "openai/o1", "openai/o1-mini", "openai/o1-preview",
    })

    def _is_reasoning_model(self) -> bool:
        return self.model in self._REASONING_MODELS or "/o4" in self.model or "/o3" in self.model or "/o1" in self.model

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._sync_client:
            raise RuntimeError("OpenRouter LLM not loaded")

        is_reasoning = self._is_reasoning_model()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
        }

        response_format = self._structured_response_format(**kwargs)
        plugins = self._structured_plugins(response_format, **kwargs)
        reasoning = kwargs.get("reasoning")

        # Reasoning models (o1/o3/o4) use max_completion_tokens, no temperature
        if is_reasoning:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature

        if response_format:
            payload["response_format"] = response_format
        if plugins:
            payload["plugins"] = plugins
        if isinstance(reasoning, dict) and reasoning:
            payload["reasoning"] = reasoning
        elif response_format:
            payload["reasoning"] = {"exclude": True}

        try:
            response = self._sync_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = (message.get("content") or "").strip()
            structured_request = bool(response_format)

            # Some models (e.g. Kimi K2.5) return output in reasoning/reasoning_content
            # when content is empty -- fall back to those fields
            if not content:
                for _rc_field in ("reasoning_content", "reasoning"):
                    _rc_val = message.get(_rc_field)
                    if isinstance(_rc_val, str) and _rc_val.strip():
                        if structured_request and not self._looks_like_json(_rc_val):
                            continue
                        content = _rc_val.strip()
                        logger.info(
                            "OpenRouter: content was null, using %s (%d chars)",
                            _rc_field, len(content),
                        )
                        break

            if not content:
                # Log the full message object so we can diagnose null-content responses
                logger.warning(
                    "OpenRouter returned empty content: model=%s message_keys=%s finish_reason=%s",
                    self.model,
                    list(message.keys()),
                    choice.get("finish_reason"),
                )
                # Dump full response for diagnosis
                import os
                _diag_path = "/tmp/openrouter_null_content.json"
                try:
                    with open(_diag_path, "w") as _f:
                        import json as _json
                        _json.dump({"choice": choice, "usage": data.get("usage"), "model": self.model}, _f, indent=2, default=str)
                    logger.warning("Diagnostic response written to %s", _diag_path)
                except Exception:
                    pass

            usage = data.get("usage", {})
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            logger.info(
                "OpenRouter chat: model=%s tokens=%s reasoning_tokens=%d content_len=%d",
                self.model, usage, reasoning_tokens, len(content),
            )

            return {
                "response": content,
                "message": {"role": "assistant", "content": content},
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": reasoning_tokens,
                },
                "_trace_meta": {
                    "api_endpoint": f"{self.base_url}/chat/completions",
                    "provider_request_id": response.headers.get("x-request-id") or data.get("id", ""),
                    "finish_reason": choice.get("finish_reason"),
                    "native_finish_reason": choice.get("native_finish_reason"),
                },
            }
        except httpx.HTTPStatusError as e:
            body = e.response.text[:1000] if e.response else ""
            logger.error(
                "OpenRouter chat error: %d %s | %s",
                e.response.status_code if e.response else 0,
                e,
                body,
            )
            raise
        except httpx.HTTPError as e:
            logger.error("OpenRouter chat error: %s %s", type(e).__name__, e)
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

            usage = data.get("usage", {})

            return {
                "response": content.strip(),
                "tool_calls": normalized_calls,
                "message": message,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
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
        }
        is_reasoning = self._is_reasoning_model()
        response_format = self._structured_response_format(**kwargs)
        plugins = self._structured_plugins(response_format, **kwargs)
        reasoning = kwargs.get("reasoning")
        if is_reasoning:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        if response_format:
            payload["response_format"] = response_format
        if plugins:
            payload["plugins"] = plugins
        if isinstance(reasoning, dict) and reasoning:
            payload["reasoning"] = reasoning
        elif response_format:
            payload["reasoning"] = {"exclude": True}

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = (message.get("content") or "").strip()
            if not content:
                for _rc_field in ("reasoning_content", "reasoning"):
                    _rc_val = message.get(_rc_field)
                    if isinstance(_rc_val, str) and _rc_val.strip():
                        if response_format and not self._looks_like_json(_rc_val):
                            continue
                        content = _rc_val.strip()
                        break
            return content
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response else ""
            logger.error(
                "OpenRouter async chat error: %d %s | %s",
                e.response.status_code if e.response else 0, e, body,
            )
            raise
        except httpx.HTTPError as e:
            logger.error("OpenRouter async chat error: %s %s", type(e).__name__, e)
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
