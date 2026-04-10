"""Shared cache-aware helpers for B2B exact-match LLM stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from .cache_strategy import require_b2b_cache_strategy
from . import llm_exact_cache

ParsedT = TypeVar("ParsedT")


@dataclass(frozen=True)
class ExactStageRequest:
    stage_id: str
    namespace: str
    provider: str
    model: str
    request_envelope: dict[str, Any]


@dataclass
class ExactStageRunResult(Generic[ParsedT]):
    request: ExactStageRequest
    text: str | None
    parsed: ParsedT | None
    cached: bool
    usage: dict[str, Any]


def _resolve_provider_model(
    *,
    llm: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    resolved_provider = str(provider or "")
    resolved_model = str(model or "")
    if llm is not None and (not resolved_provider or not resolved_model):
        llm_provider, llm_model = llm_exact_cache.llm_identity(llm)
        if not resolved_provider:
            resolved_provider = llm_provider
        if not resolved_model:
            resolved_model = llm_model
    return resolved_provider, resolved_model


def prepare_b2b_exact_stage_request(
    stage_id: str,
    *,
    llm: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
    messages: Any,
    max_tokens: int | None,
    temperature: float | None,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> ExactStageRequest:
    """Build the normalized exact-cache request for a declared B2B stage."""
    strategy = require_b2b_cache_strategy(stage_id, expected_mode="exact")
    if not strategy.namespace:
        raise ValueError(f"Stage '{stage_id}' is missing an exact-cache namespace")
    resolved_provider, resolved_model = _resolve_provider_model(
        llm=llm,
        provider=provider,
        model=model,
    )
    request_envelope = llm_exact_cache.build_request_envelope(
        provider=resolved_provider,
        model=resolved_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        guided_json=guided_json,
        extra=extra,
    )
    return ExactStageRequest(
        stage_id=stage_id,
        namespace=str(strategy.namespace),
        provider=resolved_provider,
        model=resolved_model,
        request_envelope=request_envelope,
    )


def bind_b2b_exact_stage_request(
    stage_id: str,
    *,
    provider: str,
    model: str,
    request_envelope: dict[str, Any],
) -> ExactStageRequest:
    """Bind an existing request envelope to a declared exact-cache stage."""
    strategy = require_b2b_cache_strategy(stage_id, expected_mode="exact")
    if not strategy.namespace:
        raise ValueError(f"Stage '{stage_id}' is missing an exact-cache namespace")
    return ExactStageRequest(
        stage_id=stage_id,
        namespace=str(strategy.namespace),
        provider=str(provider or ""),
        model=str(model or ""),
        request_envelope=request_envelope,
    )


def prepare_b2b_exact_skill_stage_request(
    stage_id: str,
    *,
    skill_name: str,
    payload: Any,
    llm: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None,
    temperature: float | None,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[ExactStageRequest, list[dict[str, str]]]:
    """Build the exact-cache request for a skill-driven B2B stage."""
    strategy = require_b2b_cache_strategy(stage_id, expected_mode="exact")
    if not strategy.namespace:
        raise ValueError(f"Stage '{stage_id}' is missing an exact-cache namespace")
    resolved_provider, resolved_model = _resolve_provider_model(
        llm=llm,
        provider=provider,
        model=model,
    )
    _, request_envelope, messages = llm_exact_cache.build_skill_request_envelope(
        namespace=str(strategy.namespace),
        skill_name=skill_name,
        payload=payload,
        provider=resolved_provider,
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        guided_json=guided_json,
        extra=extra,
    )
    return (
        ExactStageRequest(
            stage_id=stage_id,
            namespace=str(strategy.namespace),
            provider=resolved_provider,
            model=resolved_model,
            request_envelope=request_envelope,
        ),
        messages,
    )


async def lookup_b2b_exact_stage_text(
    request: ExactStageRequest,
    *,
    pool: Any | None = None,
) -> llm_exact_cache.B2BLLMExactCacheHit | None:
    """Lookup a declared B2B exact-cache stage."""
    return await llm_exact_cache.lookup_cached_text(
        request.namespace,
        request.request_envelope,
        pool=pool,
    )


async def store_b2b_exact_stage_text(
    request: ExactStageRequest,
    *,
    response_text: str,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    pool: Any | None = None,
) -> bool:
    """Store a validated response for a declared B2B exact-cache stage."""
    return await llm_exact_cache.store_cached_text(
        request.namespace,
        request.request_envelope,
        provider=request.provider,
        model=request.model,
        response_text=response_text,
        usage=usage,
        metadata=metadata,
        pool=pool,
    )


def _default_extract_text_and_usage(result: Any) -> tuple[str | None, dict[str, Any]]:
    if result is None:
        return None, {}
    if isinstance(result, str):
        return result, {}
    if isinstance(result, dict):
        text = str(result.get("response") or result.get("content") or "").strip()
        usage = result.get("usage", {})
        return text or None, usage if isinstance(usage, dict) else {}
    return str(result).strip() or None, {}


async def run_b2b_exact_stage(
    stage_id: str,
    *,
    messages: Any,
    invoke: Callable[[], Awaitable[Any]],
    llm: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None,
    temperature: float | None,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    parse_response: Callable[[str], ParsedT] | None = None,
    normalize_parsed: Callable[[ParsedT], ParsedT] | None = None,
    serialize_parsed: Callable[[ParsedT], str] | None = None,
    should_store_parsed: Callable[[ParsedT], bool] | None = None,
    extract_text_and_usage: Callable[[Any], tuple[str | None, dict[str, Any]]] | None = None,
) -> ExactStageRunResult[ParsedT]:
    """Run a declared B2B exact-cache stage, storing on successful miss."""
    request = prepare_b2b_exact_stage_request(
        stage_id,
        llm=llm,
        provider=provider,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        guided_json=guided_json,
        extra=extra,
    )
    cached = await lookup_b2b_exact_stage_text(request)
    if cached is not None:
        text = str(cached["response_text"] or "")
        parsed: ParsedT | None = parse_response(text) if parse_response is not None else None
        if parsed is not None and normalize_parsed is not None:
            parsed = normalize_parsed(parsed)
            if serialize_parsed is not None:
                text = serialize_parsed(parsed)
        return ExactStageRunResult(
            request=request,
            text=text,
            parsed=parsed,
            cached=True,
            usage=dict(cached.get("usage") or {}),
        )

    result = await invoke()
    text, usage = (extract_text_and_usage or _default_extract_text_and_usage)(result)
    parsed: ParsedT | None = None
    text_to_store = text
    if text and parse_response is not None:
        parsed = parse_response(text)
        if parsed is not None and normalize_parsed is not None:
            parsed = normalize_parsed(parsed)
        if parsed is not None and serialize_parsed is not None:
            text_to_store = serialize_parsed(parsed)

    should_store = bool(text_to_store and str(text_to_store).strip())
    if parsed is not None and should_store_parsed is not None:
        should_store = bool(should_store_parsed(parsed))
    if should_store and text_to_store is not None:
        await store_b2b_exact_stage_text(
            request,
            response_text=str(text_to_store),
            usage=usage,
            metadata=metadata,
        )
    return ExactStageRunResult(
        request=request,
        text=text_to_store,
        parsed=parsed,
        cached=False,
        usage=usage,
    )
