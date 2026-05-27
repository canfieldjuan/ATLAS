"""LLMClient adapter for the standalone campaign product."""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from .campaign_ports import LLMMessage, LLMResponse
from .content_ops_cache_policy import (
    ContentOpsCacheDecision,
    ContentOpsExactCachePolicy,
)


class LLMUnavailableError(RuntimeError):
    """Raised when the host has not configured an LLM route."""


LLMResolver = Callable[..., Any]
LLMTraceCallable = Callable[..., None]
_TRACE_CONTEXT: ContextVar[Mapping[str, Any]] = ContextVar(
    "content_ops_llm_trace_context",
    default={},
)


def _default_resolver(**kwargs: Any) -> Any:
    from .pipelines.llm import get_pipeline_llm

    return get_pipeline_llm(**kwargs)


def _default_trace_llm_call(*args: Any, **kwargs: Any) -> None:
    from .pipelines.llm import trace_llm_call

    trace_llm_call(*args, **kwargs)


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def set_content_ops_llm_trace_context(
    metadata: Mapping[str, Any] | None,
) -> Token[Mapping[str, Any]]:
    """Set request/tenant metadata that all Content Ops LLM traces should carry."""

    return _TRACE_CONTEXT.set(_clean_trace_context(metadata))


def reset_content_ops_llm_trace_context(token: Token[Mapping[str, Any]]) -> None:
    _TRACE_CONTEXT.reset(token)


def current_content_ops_llm_trace_context() -> dict[str, Any]:
    return dict(_TRACE_CONTEXT.get({}))


def _clean_trace_context(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        text = _optional_text(value)
        if text:
            cleaned[str(key)] = text
    return cleaned


@dataclass(frozen=True)
class PipelineLLMClientConfig:
    """Provider routing config for the product LLM client."""

    workload: str | None = "draft"
    prefer_cloud: bool = True
    try_openrouter: bool = True
    auto_activate_ollama: bool = True
    openrouter_model: str | None = None
    exact_cache_enabled: bool = False
    customer_data_exact_cache_enabled: bool = False
    exact_cache_namespace_prefix: str = "content_ops"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PipelineLLMClientConfig":
        return cls(
            workload=_optional_text(values.get("workload")) or "draft",
            prefer_cloud=_to_bool(values.get("prefer_cloud"), True),
            try_openrouter=_to_bool(values.get("try_openrouter"), True),
            auto_activate_ollama=_to_bool(values.get("auto_activate_ollama"), True),
            openrouter_model=_optional_text(values.get("openrouter_model")),
            exact_cache_enabled=_to_bool(values.get("exact_cache_enabled"), False),
            customer_data_exact_cache_enabled=_to_bool(
                values.get("customer_data_exact_cache_enabled"),
                False,
            ),
            exact_cache_namespace_prefix=(
                _optional_text(values.get("exact_cache_namespace_prefix"))
                or "content_ops"
            ),
        )

    @classmethod
    def from_settings(cls, settings_obj: Any) -> "PipelineLLMClientConfig":
        return cls(
            workload=_optional_text(getattr(settings_obj, "workload", None)) or "draft",
            prefer_cloud=_to_bool(getattr(settings_obj, "prefer_cloud", None), True),
            try_openrouter=_to_bool(getattr(settings_obj, "try_openrouter", None), True),
            auto_activate_ollama=_to_bool(
                getattr(settings_obj, "auto_activate_ollama", None),
                True,
            ),
            openrouter_model=_optional_text(
                getattr(settings_obj, "openrouter_model", None),
            ),
            exact_cache_enabled=_to_bool(
                getattr(settings_obj, "exact_cache_enabled", None),
                False,
            ),
            customer_data_exact_cache_enabled=_to_bool(
                getattr(settings_obj, "customer_data_exact_cache_enabled", None),
                False,
            ),
            exact_cache_namespace_prefix=(
                _optional_text(
                    getattr(settings_obj, "exact_cache_namespace_prefix", None),
                )
                or "content_ops"
            ),
        )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        prefix: str = "EXTRACTED_CAMPAIGN_LLM_",
    ) -> "PipelineLLMClientConfig":
        env = os.environ if environ is None else environ
        return cls(
            workload=_optional_text(env.get(f"{prefix}WORKLOAD")) or "draft",
            prefer_cloud=_to_bool(env.get(f"{prefix}PREFER_CLOUD"), True),
            try_openrouter=_to_bool(env.get(f"{prefix}TRY_OPENROUTER"), True),
            auto_activate_ollama=_to_bool(
                env.get(f"{prefix}AUTO_ACTIVATE_OLLAMA"),
                True,
            ),
            openrouter_model=_optional_text(env.get(f"{prefix}OPENROUTER_MODEL")),
            exact_cache_enabled=_to_bool(env.get(f"{prefix}EXACT_CACHE_ENABLED"), False),
            customer_data_exact_cache_enabled=_to_bool(
                env.get(f"{prefix}CUSTOMER_DATA_EXACT_CACHE_ENABLED"),
                False,
            ),
            exact_cache_namespace_prefix=(
                _optional_text(env.get(f"{prefix}EXACT_CACHE_NAMESPACE_PREFIX"))
                or "content_ops"
            ),
        )


@dataclass(frozen=True)
class ContentOpsExactCacheAdapter:
    """Thin adapter over the shared exact-cache helpers.

    Content Ops owns the cache eligibility policy. Once that policy returns an
    account-scoped exact decision, this adapter reuses the shared request
    envelope and Postgres cache helpers without applying the legacy B2B/Gateway
    namespace feature flag a second time.
    """

    def build_request_envelope(
        self,
        *,
        provider: str,
        model: str,
        messages: Sequence[LLMMessage],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        from extracted_llm_infrastructure.services.b2b import llm_exact_cache

        return llm_exact_cache.build_request_envelope(
            provider=provider,
            model=model,
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            extra={"product": "content_ops"},
        )

    async def lookup(
        self,
        decision: ContentOpsCacheDecision,
        request_envelope: dict[str, Any],
    ) -> Mapping[str, Any] | None:
        from extracted_llm_infrastructure.services.b2b import llm_exact_cache

        if not decision.namespace or not decision.account_id:
            return None
        return await llm_exact_cache.lookup_cached_text(
            decision.namespace,
            request_envelope,
            account_id=decision.account_id,
            require_namespace_enabled=False,
        )

    async def store(
        self,
        decision: ContentOpsCacheDecision,
        request_envelope: dict[str, Any],
        *,
        provider: str,
        model: str,
        response: LLMResponse,
    ) -> bool:
        from extracted_llm_infrastructure.services.b2b import llm_exact_cache

        if not decision.namespace or not decision.account_id:
            return False
        return bool(await llm_exact_cache.store_cached_text(
            decision.namespace,
            request_envelope,
            provider=provider,
            model=model,
            response_text=response.content,
            usage=dict(response.usage or {}),
            metadata={"product": "content_ops"},
            account_id=decision.account_id,
            require_namespace_enabled=False,
        ))


@dataclass(frozen=True)
class PipelineLLMClient:
    """Adapt extracted LLM infrastructure services to the product LLMClient port."""

    workload: str | None = "draft"
    prefer_cloud: bool = True
    try_openrouter: bool = True
    auto_activate_ollama: bool = True
    openrouter_model: str | None = None
    resolver: LLMResolver = _default_resolver
    tracer: LLMTraceCallable | None = _default_trace_llm_call
    cache_policy: ContentOpsExactCachePolicy = ContentOpsExactCachePolicy()
    exact_cache: ContentOpsExactCacheAdapter | None = field(
        default_factory=ContentOpsExactCacheAdapter,
    )

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

        cache_decision = self.cache_policy.decide(
            _cache_policy_metadata(metadata),
            messages=messages,
        )
        started = time.monotonic()
        cache_request: dict[str, Any] | None = None
        cache_metadata: dict[str, Any] = {}
        provider = _provider_name(llm)
        model = _model_name(llm) or ""
        if cache_decision.cacheable and self.exact_cache is not None:
            try:
                cache_request = self.exact_cache.build_request_envelope(
                    provider=provider,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                cache_hit = await self.exact_cache.lookup(
                    cache_decision,
                    cache_request,
                )
                if cache_hit is not None:
                    response = _response_from_cache_hit(cache_hit)
                    self._trace_call(
                        llm=llm,
                        response=response,
                        duration_ms=_duration_ms(started),
                        metadata=metadata,
                        cache_decision=cache_decision,
                        cache_metadata={"cache_result": "hit"},
                    )
                    return response
                cache_metadata["cache_result"] = "miss"
            except Exception as exc:
                cache_request = None
                cache_metadata.update(_cache_error_metadata("lookup_error", exc))
        try:
            call_args = {
                "llm": llm,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if _llm_call_is_async(llm):
                result = self._call_llm(**call_args)
            else:
                result = await asyncio.to_thread(self._call_llm, **call_args)
            if inspect.isawaitable(result):
                result = await result
            response = _to_response(result, llm=llm)
        except Exception as exc:
            self._trace_call(
                llm=llm,
                response=None,
                duration_ms=_duration_ms(started),
                metadata=metadata,
                cache_decision=cache_decision,
                cache_metadata=cache_metadata,
                status="failed",
                error_message=str(exc)[:500],
                error_type=type(exc).__name__,
            )
            raise
        if (
            cache_decision.cacheable
            and self.exact_cache is not None
            and cache_request is not None
        ):
            try:
                stored = await self.exact_cache.store(
                    cache_decision,
                    cache_request,
                    provider=provider,
                    model=model,
                    response=response,
                )
                cache_metadata["cache_store_result"] = (
                    "stored" if stored else "skipped"
                )
            except Exception as exc:
                cache_metadata.update(_cache_error_metadata("store_error", exc))
        self._trace_call(
            llm=llm,
            response=response,
            duration_ms=_duration_ms(started),
            metadata=metadata,
            cache_decision=cache_decision,
            cache_metadata=cache_metadata,
        )
        return response

    def _trace_call(
        self,
        *,
        llm: Any,
        response: LLMResponse | None,
        duration_ms: float,
        metadata: Mapping[str, Any] | None,
        cache_decision: ContentOpsCacheDecision | None,
        cache_metadata: Mapping[str, Any] | None = None,
        status: str = "completed",
        error_message: str | None = None,
        error_type: str | None = None,
    ) -> None:
        if self.tracer is None:
            return
        usage = dict(getattr(response, "usage", {}) or {}) if response else {}
        trace_meta = _trace_meta_from_response(response)
        cached_tokens, cache_write_tokens, billable_input_tokens = _trace_cache_metrics(
            usage,
            trace_meta,
        )
        trace_metadata = {
            "product": "content_ops",
            "workload": self.workload or "default",
            "llm_adapter": "pipeline",
        }
        scoped_metadata = current_content_ops_llm_trace_context()
        call_metadata = dict(metadata or {})
        call_metadata.pop("account_id", None)
        call_metadata.pop("user_id", None)
        for key in (
            "cache_mode",
            "cache_reason",
            "cache_namespace",
            "cache_account_id",
            "cache_result",
            "cache_store_result",
            "cache_error_type",
            "cache_error_message",
        ):
            call_metadata.pop(key, None)
        trace_metadata.update(scoped_metadata)
        trace_metadata.update(call_metadata)
        if cache_decision is not None:
            trace_metadata.update(cache_decision.trace_metadata())
        for key in (
            "cached_input_tokens",
            "cached_output_tokens",
            "billable_output_tokens",
        ):
            if key in trace_meta:
                trace_metadata[key] = str(_usage_int(trace_meta.get(key)))
        if cache_metadata:
            trace_metadata.update({
                str(key): str(value)
                for key, value in cache_metadata.items()
                if value not in (None, "")
            })
        try:
            self.tracer(
                "content_ops.llm.complete",
                input_tokens=_usage_int(usage.get("input_tokens")),
                output_tokens=_usage_int(usage.get("output_tokens")),
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
                billable_input_tokens=billable_input_tokens,
                model=_trace_model_name(llm=llm, response=response),
                provider=_provider_name(llm),
                duration_ms=duration_ms,
                status=status,
                metadata=trace_metadata,
                api_endpoint=_optional_trace_text(trace_meta.get("api_endpoint")),
                provider_request_id=_optional_trace_text(
                    trace_meta.get("provider_request_id"),
                ),
                ttft_ms=_optional_trace_float(trace_meta.get("ttft_ms")),
                inference_time_ms=_optional_trace_float(
                    trace_meta.get("inference_time_ms"),
                ),
                queue_time_ms=_optional_trace_float(trace_meta.get("queue_time_ms")),
                error_message=error_message,
                error_type=error_type,
            )
        except Exception:
            return

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
                _to_chat_messages(messages),
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


def create_pipeline_llm_client(
    config: PipelineLLMClientConfig | Mapping[str, Any] | None = None,
    *,
    resolver: LLMResolver = _default_resolver,
) -> PipelineLLMClient:
    """Create a campaign LLM client from product-owned provider config."""

    if config is None:
        resolved = PipelineLLMClientConfig.from_env()
    elif isinstance(config, PipelineLLMClientConfig):
        resolved = config
    else:
        resolved = PipelineLLMClientConfig.from_mapping(config)
    return PipelineLLMClient(
        workload=resolved.workload,
        prefer_cloud=resolved.prefer_cloud,
        try_openrouter=resolved.try_openrouter,
        auto_activate_ollama=resolved.auto_activate_ollama,
        openrouter_model=resolved.openrouter_model,
        resolver=resolver,
        cache_policy=ContentOpsExactCachePolicy(
            exact_cache_enabled=resolved.exact_cache_enabled,
            customer_data_exact_cache_enabled=resolved.customer_data_exact_cache_enabled,
            namespace_prefix=resolved.exact_cache_namespace_prefix,
        ),
    )


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


def _to_chat_messages(messages: Sequence[LLMMessage]) -> list[Any]:
    return [
        SimpleNamespace(
            role=str(getattr(message, "role", "") or ""),
            content=str(getattr(message, "content", "") or ""),
            tool_calls=getattr(message, "tool_calls", None),
            tool_call_id=getattr(message, "tool_call_id", None),
        )
        for message in messages
    ]


def _cache_policy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    scoped_metadata = current_content_ops_llm_trace_context()
    call_metadata = dict(metadata or {})
    call_metadata.pop("account_id", None)
    call_metadata.pop("user_id", None)
    merged = dict(scoped_metadata)
    merged.update(call_metadata)
    return merged


def _llm_call_is_async(llm: Any) -> bool:
    if hasattr(llm, "chat"):
        return inspect.iscoroutinefunction(getattr(llm, "chat"))
    if hasattr(llm, "generate"):
        return inspect.iscoroutinefunction(getattr(llm, "generate"))
    return False


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


def _response_from_cache_hit(hit: Mapping[str, Any]) -> LLMResponse:
    usage = hit.get("usage") if isinstance(hit.get("usage"), Mapping) else {}
    input_tokens = _usage_int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("total_tokens")
    )
    output_tokens = _usage_int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
    )
    raw_hit = dict(hit)
    raw_hit["_trace_meta"] = {
        "cached_input_tokens": input_tokens,
        "cached_output_tokens": output_tokens,
        "cached_tokens": input_tokens,
        "billable_input_tokens": 0,
        "billable_output_tokens": 0,
    }
    return LLMResponse(
        content=str(hit.get("response_text") or ""),
        model=str(hit.get("model") or "") or None,
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
        },
        raw=raw_hit,
    )


def _cache_error_metadata(result: str, exc: Exception) -> dict[str, str]:
    return {
        "cache_result": result,
        "cache_error_type": type(exc).__name__,
        "cache_error_message": str(exc)[:200],
    }


def _duration_ms(started: float) -> float:
    return max((time.monotonic() - started) * 1000, 0.0)


def _usage_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _trace_cache_metrics(
    usage: Mapping[str, Any],
    trace_meta: Mapping[str, Any],
) -> tuple[int, int, int | None]:
    cached_tokens = _usage_int(
        usage.get("cached_tokens")
        or usage.get("cache_read_tokens")
        or usage.get("cache_read_input_tokens")
        or trace_meta.get("cached_tokens")
        or trace_meta.get("cache_read_tokens")
    )
    cache_write_tokens = _usage_int(
        usage.get("cache_write_tokens")
        or usage.get("cache_creation_tokens")
        or usage.get("cache_creation_input_tokens")
        or trace_meta.get("cache_write_tokens")
        or trace_meta.get("cache_creation_tokens")
    )
    raw_billable = usage.get("billable_input_tokens")
    if raw_billable is None:
        raw_billable = trace_meta.get("billable_input_tokens")
    billable_input_tokens = _usage_int(raw_billable) if raw_billable is not None else None
    return cached_tokens, cache_write_tokens, billable_input_tokens


def _trace_meta_from_response(response: LLMResponse | None) -> Mapping[str, Any]:
    raw = getattr(response, "raw", None) if response is not None else None
    if isinstance(raw, Mapping) and isinstance(raw.get("_trace_meta"), Mapping):
        return raw["_trace_meta"]
    return {}


def _optional_trace_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_trace_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _provider_name(llm: Any) -> str:
    return str(getattr(llm, "name", "") or llm.__class__.__name__)


def _trace_model_name(*, llm: Any, response: LLMResponse | None) -> str:
    if response is not None and response.model:
        return str(response.model)
    return _model_name(llm) or ""


def _model_name(llm: Any) -> str | None:
    for attr in ("model", "model_id", "name"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)
    model_info = getattr(llm, "model_info", None)
    value = getattr(model_info, "model_id", None) if model_info is not None else None
    return str(value) if value else None
