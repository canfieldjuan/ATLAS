from __future__ import annotations

import asyncio
from typing import Any


ANTHROPIC_CACHE_MIN_CHARS = 1024

_tier1_client = None
_tier1_client_signature = None


def resolve_tier_routing(cfg: Any, *, local_only_override: bool | None = None) -> tuple[bool, bool]:
    local_only = (
        local_only_override
        if local_only_override is not None
        else bool(getattr(cfg, "enrichment_local_only", False))
    )
    has_openrouter = bool(getattr(cfg, "enrichment_openrouter_model", "")) and bool(
        getattr(cfg, "openrouter_api_key", "")
    )
    use_openrouter_tier1 = (not local_only) and has_openrouter
    use_openrouter_tier2 = use_openrouter_tier1 or (
        bool(getattr(cfg, "enrichment_tier2_force_openrouter", False))
        and has_openrouter
    )
    return use_openrouter_tier1, use_openrouter_tier2


def maybe_anthropic_cache(
    model_id: str,
    messages: list[dict],
    *,
    min_chars: int = ANTHROPIC_CACHE_MIN_CHARS,
) -> list[dict]:
    if not str(model_id or "").startswith("anthropic/"):
        return messages
    converted: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if (
            msg.get("role") == "system"
            and isinstance(content, str)
            and len(content) >= min_chars
        ):
            converted.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            converted.append(msg)
    return converted


def get_tier1_client(cfg: Any) -> Any:
    global _tier1_client, _tier1_client_signature
    client_signature = (
        str(cfg.enrichment_tier1_vllm_url).strip(),
        float(cfg.enrichment_tier1_timeout_seconds),
        float(cfg.enrichment_tier1_connect_timeout_seconds),
    )
    if (
        _tier1_client is not None
        and not _tier1_client.is_closed
        and _tier1_client_signature == client_signature
    ):
        return _tier1_client

    import httpx

    if _tier1_client is not None and not _tier1_client.is_closed:
        try:
            asyncio.get_running_loop().create_task(_tier1_client.aclose())
        except RuntimeError:
            pass

    _tier1_client = httpx.AsyncClient(
        base_url=cfg.enrichment_tier1_vllm_url,
        timeout=httpx.Timeout(
            cfg.enrichment_tier1_timeout_seconds,
            connect=cfg.enrichment_tier1_connect_timeout_seconds,
        ),
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
    )
    _tier1_client_signature = client_signature
    return _tier1_client


def get_tier2_client(
    cfg: Any,
    *,
    get_tier1_client: Any,
    coerce_float_value: Any,
) -> Any:
    raw_tier2_url = getattr(cfg, "enrichment_tier2_vllm_url", "")
    raw_tier1_url = getattr(cfg, "enrichment_tier1_vllm_url", "")
    tier2_url = raw_tier2_url if isinstance(raw_tier2_url, str) and raw_tier2_url.strip() else raw_tier1_url
    timeout = coerce_float_value(getattr(cfg, "enrichment_tier2_timeout_seconds", 120.0), 120.0)
    if not isinstance(tier2_url, str) or not tier2_url.strip() or tier2_url == raw_tier1_url:
        return get_tier1_client(cfg)
    import httpx
    return httpx.AsyncClient(base_url=tier2_url, timeout=timeout)


def trace_enrichment_llm_call(
    span_name: str,
    *,
    provider: str,
    model: str | None,
    messages: list[dict[str, str]],
    usage: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    duration_ms: float,
    api_endpoint: str | None = None,
    provider_request_id: str | None = None,
) -> None:
    from ...pipelines.llm import _trace_cache_metrics, trace_llm_call

    usage_dict = usage if isinstance(usage, dict) else {}
    prompt_details = usage_dict.get("prompt_tokens_details") or {}
    cached_from_provider = (
        usage_dict.get("cached_tokens")
        or prompt_details.get("cached_tokens")
        or usage_dict.get("cache_read_input_tokens")
        or 0
    )
    cache_write_from_provider = (
        usage_dict.get("cache_write_tokens")
        or prompt_details.get("cache_write_tokens")
        or usage_dict.get("cache_creation_input_tokens")
        or 0
    )
    normalized_usage = {
        "input_tokens": usage_dict.get("input_tokens", usage_dict.get("prompt_tokens", 0)),
        "output_tokens": usage_dict.get("output_tokens", usage_dict.get("completion_tokens", 0)),
        "billable_input_tokens": usage_dict.get("billable_input_tokens", usage_dict.get("prompt_tokens")),
        "cached_tokens": cached_from_provider,
        "cache_write_tokens": cache_write_from_provider,
    }
    cached_tokens, cache_write_tokens, billable_input_tokens = _trace_cache_metrics(normalized_usage, {})
    trace_llm_call(
        span_name,
        input_tokens=int(normalized_usage.get("input_tokens") or 0),
        output_tokens=int(normalized_usage.get("output_tokens") or 0),
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        billable_input_tokens=billable_input_tokens,
        model=str(model or ""),
        provider=provider,
        duration_ms=duration_ms,
        metadata=metadata or {},
        input_data={"messages": [{"role": msg["role"], "content": (msg["content"] or "")[:500]} for msg in messages]},
        api_endpoint=api_endpoint,
        provider_request_id=provider_request_id,
    )
