"""LLMClient adapter for the standalone campaign product."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .campaign_ports import LLMMessage, LLMResponse


class LLMUnavailableError(RuntimeError):
    """Raised when the host has not configured an LLM route."""


LLMResolver = Callable[..., Any]


def _default_resolver(**kwargs: Any) -> Any:
    from .pipelines.llm import get_pipeline_llm

    return get_pipeline_llm(**kwargs)


@dataclass(frozen=True)
class PipelineLLMClient:
    """Adapt extracted LLM infrastructure services to the product LLMClient port."""

    workload: str | None = "draft"
    prefer_cloud: bool = True
    try_openrouter: bool = True
    auto_activate_ollama: bool = True
    openrouter_model: str | None = None
    resolver: LLMResolver = _default_resolver

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        llm = self.resolver(
            workload=self.workload,
            prefer_cloud=self.prefer_cloud,
            try_openrouter=self.try_openrouter,
            auto_activate_ollama=self.auto_activate_ollama,
            openrouter_model=self.openrouter_model,
        )
        if llm is None:
            raise LLMUnavailableError("No LLM route configured for campaign generation")

        result = self._call_llm(
            llm,
            list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if inspect.isawaitable(result):
            result = await result
        return _to_response(result, llm=llm)

    def _call_llm(
        self,
        llm: Any,
        messages: list[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        if hasattr(llm, "chat"):
            return llm.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        if hasattr(llm, "generate"):
            system_prompt, prompt = _messages_to_prompt(messages)
            return llm.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        raise LLMUnavailableError("Resolved LLM does not expose chat() or generate()")


def _messages_to_prompt(messages: Sequence[LLMMessage]) -> tuple[str | None, str]:
    system_parts: list[str] = []
    prompt_parts: list[str] = []
    for message in messages:
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        if getattr(message, "role", "") == "system":
            system_parts.append(content)
        else:
            prompt_parts.append(content)
    return "\n\n".join(system_parts) or None, "\n\n".join(prompt_parts)


def _to_response(result: Any, *, llm: Any) -> LLMResponse:
    if isinstance(result, str):
        return LLMResponse(content=result, model=_model_name(llm), raw=result)

    if not isinstance(result, Mapping):
        return LLMResponse(content=str(result or ""), model=_model_name(llm), raw=result)

    message = result.get("message")
    content = (
        result.get("response")
        or result.get("content")
        or result.get("text")
        or (message.get("content") if isinstance(message, Mapping) else None)
        or ""
    )
    usage = result.get("usage") if isinstance(result.get("usage"), Mapping) else {}
    return LLMResponse(
        content=str(content),
        model=str(result.get("model") or _model_name(llm) or "") or None,
        usage=dict(usage),
        raw=result,
    )


def _model_name(llm: Any) -> str | None:
    for attr in ("model", "model_id", "name"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)
    model_info = getattr(llm, "model_info", None)
    value = getattr(model_info, "model_id", None) if model_info is not None else None
    return str(value) if value else None
