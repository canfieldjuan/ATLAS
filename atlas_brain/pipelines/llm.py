"""
Shared LLM utilities for pipeline tasks.

Extracts duplicated LLM resolution, output cleaning, and JSON parsing
from article_enrichment, daily_intelligence, complaint_analysis,
complaint_enrichment, and complaint_content_generation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger("atlas.pipelines.llm")


def _trace_cache_metrics(
    usage: dict[str, Any],
    trace_meta: dict[str, Any],
) -> tuple[int, int, int | None]:
    """Normalize cache usage metrics across providers."""
    cached_tokens = int(
        usage.get("cached_tokens")
        or trace_meta.get("cached_tokens")
        or trace_meta.get("cache_read_tokens")
        or 0
    )
    cache_write_tokens = int(
        usage.get("cache_write_tokens")
        or trace_meta.get("cache_write_tokens")
        or trace_meta.get("cache_creation_tokens")
        or 0
    )
    raw_billable = (
        usage.get("billable_input_tokens")
        if usage.get("billable_input_tokens") is not None
        else trace_meta.get("billable_input_tokens")
    )
    billable_input_tokens = int(raw_billable) if raw_billable is not None else None
    return cached_tokens, cache_write_tokens, billable_input_tokens


# ------------------------------------------------------------------
# LLM resolution
# ------------------------------------------------------------------


def _resolve_openrouter_api_key() -> str:
    """Resolve OpenRouter API key from env first, then configured settings."""
    from ..config import settings as _settings

    return (
        os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("ATLAS_B2B_CHURN_OPENROUTER_API_KEY", "")
        or str(getattr(_settings.b2b_churn, "openrouter_api_key", "") or "").strip()
    )


def _strict_openrouter_reasoning() -> bool:
    """Whether reasoning/synthesis should fail closed to OpenRouter."""
    from ..config import settings as _settings

    return bool(getattr(_settings.llm, "openrouter_reasoning_strict", False))


def _resolve_workload(workload: str):
    """Try to resolve an LLM for a specific workload type.

    Returns the LLM instance or None if no suitable model is initialized.
    """
    from ..services.llm_router import (
        get_triage_llm, get_draft_llm, get_reasoning_llm,
    )

    if workload == "triage":
        return get_triage_llm()

    if workload == "draft":
        return get_draft_llm()

    if workload in ("synthesis", "reasoning"):
        # Primary: configured OpenRouter reasoning model
        from ..config import settings as _settings
        llm = _try_openrouter(_settings.llm.openrouter_reasoning_model)
        if llm is not None:
            logger.debug(
                "Using OpenRouter %s for workload '%s'",
                _settings.llm.openrouter_reasoning_model, workload,
            )
            return llm
        if _strict_openrouter_reasoning() and str(_settings.llm.openrouter_reasoning_model or "").strip():
            logger.warning(
                "OpenRouter reasoning strict mode enabled; not falling back for workload '%s'",
                workload,
            )
            return None
        # Fallback: Anthropic Sonnet singletons
        llm = get_reasoning_llm()
        if llm is not None:
            logger.debug("Using reasoning LLM for workload '%s'", workload)
            return llm
        llm = get_draft_llm()
        if llm is not None:
            logger.debug("Using draft LLM for workload '%s'", workload)
            return llm
        return None

    if workload == "local_fast":
        from ..services import llm_registry
        return llm_registry.get_active()

    if workload == "openrouter":
        from ..config import settings as _settings
        return _try_openrouter(_settings.llm.openrouter_reasoning_model)

    if workload == "vllm":
        return _activate_vllm()

    if workload == "anthropic":
        return _anthropic_fallback()

    logger.warning("Unknown workload: '%s'", workload)
    return None


def _try_openrouter(openrouter_model: str | None = None):
    """Try to activate OpenRouter. Reuses existing instance if model matches."""
    from ..services import llm_registry
    from ..config import settings as _settings

    target_model = (
        str(openrouter_model or "").strip()
        or str(_settings.llm.openrouter_reasoning_model or "").strip()
    )
    or_key = _resolve_openrouter_api_key()
    if not or_key or not target_model:
        return None

    # Reuse existing OpenRouter instance if already active with the right model
    llm = llm_registry.get_active()
    if llm is not None and getattr(llm, "name", "") == "openrouter":
        if getattr(llm, "model", "") == target_model:
            return llm

    try:
        llm_registry.activate(
            "openrouter",
            model=target_model,
            api_key=or_key,
        )
        llm = llm_registry.get_active()
        if llm is not None:
            logger.info("Using OpenRouter LLM (%s)", target_model)
            return llm
    except Exception as e:
        logger.debug("OpenRouter fallback failed: %s", e)
    return None


def _activate_local_llm():
    """Try to activate vLLM or Ollama as a local fallback. Returns LLM or None."""
    from ..config import settings
    from ..services import llm_registry

    if settings.llm.default_model == "vllm":
        try:
            llm_registry.activate(
                "vllm",
                model=settings.llm.vllm_model,
                base_url=settings.llm.vllm_url,
            )
            llm = llm_registry.get_active()
            if llm is not None:
                logger.info("Auto-activated vLLM (%s)", settings.llm.vllm_model)
                return llm
        except Exception as e:
            logger.debug("Could not auto-activate vLLM: %s", e)

    try:
        llm_registry.activate(
            "ollama",
            model=settings.llm.ollama_model,
            base_url=settings.llm.ollama_url,
            timeout=settings.llm.ollama_timeout,
        )
        llm = llm_registry.get_active()
        if llm is not None:
            logger.info("Auto-activated Ollama LLM")
            return llm
    except Exception as e:
        logger.warning("Could not auto-activate Ollama LLM: %s", e)
    return None


def _activate_vllm():
    """Activate vLLM only (no Ollama fallback). Returns LLM or None."""
    from ..config import settings
    from ..services import llm_registry

    try:
        llm_registry.activate(
            "vllm",
            model=settings.llm.vllm_model,
            base_url=settings.llm.vllm_url,
        )
        llm = llm_registry.get_active()
        if llm is not None:
            logger.info("Activated vLLM (%s)", settings.llm.vllm_model)
            return llm
    except Exception as e:
        logger.error("Could not activate vLLM: %s", e)
    return None


def _anthropic_fallback():
    """Activate Anthropic Sonnet as cloud fallback. Returns LLM or None."""
    from ..config import settings

    api_key = (
        settings.llm.anthropic_api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ATLAS_LLM_ANTHROPIC_API_KEY", "")
    )
    if not api_key:
        logger.warning("Anthropic fallback skipped: no API key configured")
        return None
    from ..services import llm_registry

    model = settings.llm.anthropic_model
    try:
        llm_registry.activate(
            "anthropic",
            model=model,
            api_key=api_key,
        )
        llm = llm_registry.get_active()
        if llm is not None:
            logger.warning("vLLM unavailable, falling back to Anthropic Sonnet")
            return llm
    except Exception as e:
        logger.error("Anthropic fallback failed: %s", e)
    return None


def get_pipeline_llm(
    *,
    workload: str | None = None,
    prefer_cloud: bool = True,
    try_openrouter: bool = True,
    auto_activate_ollama: bool = True,
    openrouter_model: str | None = None,
):
    """Resolve an LLM instance for pipeline tasks.

    When *workload* is set, routes to the appropriate model singleton:
        triage     -> Haiku (cheap classification)
        draft      -> Sonnet (email drafts)
        synthesis  -> configured OpenRouter reasoning model primary, fallback optional
        reasoning  -> configured OpenRouter reasoning model primary, fallback optional
        openrouter -> OpenRouter only
        local_fast -> local vLLM/Ollama only
        vllm       -> vLLM primary, Anthropic fallback (no Ollama)
        anthropic  -> Anthropic primary, vLLM fallback (no Ollama)

    When workload is None, falls back to legacy prefer_cloud chain.
    Returns the LLM instance or None.
    """
    from ..services import llm_registry

    # --- Workload-based routing (preferred) ---
    if workload is not None:
        strict_reasoning = workload in ("synthesis", "reasoning") and _strict_openrouter_reasoning()
        # Explicit model override: create a standalone instance (don't swap the singleton)
        if openrouter_model and try_openrouter:
            or_key = _resolve_openrouter_api_key()
            if or_key:
                try:
                    from ..services.llm.openrouter import OpenRouterLLM
                    _or = OpenRouterLLM(model=openrouter_model, api_key=or_key)
                    _or.load()
                    logger.info("Using explicit OpenRouter model override: %s", openrouter_model)
                    return _or
                except Exception as e:
                    logger.warning("Explicit OpenRouter model %s failed: %s", openrouter_model, e)
                    if strict_reasoning:
                        return None
        llm = _resolve_workload(workload)
        if llm is not None:
            return llm
        if strict_reasoning:
            logger.warning(
                "OpenRouter reasoning strict mode enabled; '%s' workload returning no LLM",
                workload,
            )
            return None
        # local_fast: only local models
        if workload == "local_fast":
            if auto_activate_ollama:
                return _activate_local_llm()
            return None
        if workload == "openrouter":
            return None
        # vllm: fall back to Anthropic (no Ollama, no OpenRouter)
        if workload == "vllm":
            llm = _anthropic_fallback()
            if llm is not None:
                return llm
            return None
        # anthropic: fall back to vLLM (no Ollama, no OpenRouter)
        if workload == "anthropic":
            llm = _activate_vllm()
            if llm is not None:
                return llm
            return None
        # Workload model unavailable; try OpenRouter fallback
        if try_openrouter:
            llm = _try_openrouter(openrouter_model)
            if llm is not None:
                return llm
        # For synthesis/reasoning, triage (Haiku) is a last resort with warning
        if workload in ("synthesis", "reasoning"):
            from ..services.llm_router import get_triage_llm
            llm = get_triage_llm()
            if llm is not None:
                logger.warning(
                    "Falling back to triage LLM (Haiku) for '%s' workload "
                    "-- enable reasoning or draft LLM for better results",
                    workload,
                )
                return llm
        if auto_activate_ollama and workload not in ("triage", "draft"):
            return _activate_local_llm()
        return None

    # --- Legacy prefer_cloud routing (no workload specified) ---
    if prefer_cloud:
        from ..services.llm_router import get_triage_llm
        llm = get_triage_llm()
        if llm is not None:
            logger.debug("Using triage LLM (Anthropic)")
            return llm

    if try_openrouter:
        llm = _try_openrouter(openrouter_model)
        if llm is not None:
            return llm

    llm = llm_registry.get_active()
    if llm is not None:
        return llm

    if auto_activate_ollama:
        return _activate_local_llm()

    return None


# ------------------------------------------------------------------
# Output cleaning
# ------------------------------------------------------------------


def clean_llm_output(text: str) -> str:
    """Strip ``<think>`` tags and markdown fences from LLM output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    json_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    return text


# ------------------------------------------------------------------
# JSON parsing
# ------------------------------------------------------------------


def parse_json_response(
    text: str,
    *,
    recover_truncated: bool = False,
) -> dict[str, Any]:
    """Progressive JSON extraction from LLM response.

    Tries in order:
    1. ```json``` fenced block
    2. Entire response as JSON
    3. First ``{...}`` match
    4. Truncation recovery (if ``recover_truncated``)
    5. Fallback ``{"analysis_text": text}``
    """
    if not text:
        logger.warning("parse_json_response received empty/None input")
        return {"analysis_text": "", "_parse_fallback": True}

    # 1. Fenced JSON block
    json_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Entire response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. First {..} object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    # 4. Truncation recovery
    if recover_truncated:
        recovered = _recover_truncated_json(text)
        if recovered:
            return recovered

    # 5. Fallback -- all structured fields lost, only raw text preserved
    logger.warning(
        "parse_json_response fell back to raw text (%d chars). "
        "LLM output was not valid JSON.",
        len(text),
    )
    return {"analysis_text": text, "_parse_fallback": True}


def _recover_truncated_json(raw_text: str) -> dict[str, Any] | None:
    """Attempt to recover a JSON object from truncated LLM output.

    When max_tokens cuts off output mid-JSON, we try closing open
    structures progressively to salvage whatever fields were complete.
    """
    start = raw_text.find("{")
    if start < 0:
        return None

    text = raw_text[start:]

    for trim in range(0, min(len(text), 2000), 1):
        candidate = text if trim == 0 else text[:-trim]
        opens = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if opens <= 0 and open_brackets <= 0:
            continue
        suffix = "]" * max(open_brackets, 0) + "}" * max(opens, 0)
        try:
            result = json.loads(candidate + suffix)
            if isinstance(result, dict) and result:
                logger.info(
                    "Recovered truncated JSON (trimmed %d chars, closed %d braces)",
                    trim, opens + open_brackets,
                )
                return result
        except json.JSONDecodeError:
            continue

    return None


# ------------------------------------------------------------------
# Full LLM call pattern
# ------------------------------------------------------------------


def call_llm_with_skill(
    skill_name: str,
    payload: dict[str, Any],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.4,
    workload: str | None = None,
    prefer_cloud: bool = True,
    try_openrouter: bool = True,
    auto_activate_ollama: bool = True,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
    openrouter_model: str | None = None,
    usage_out: dict[str, Any] | None = None,
    span_name: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> str | None:
    """Load a skill, resolve an LLM, call it, clean the output.

    Returns the raw cleaned text, or None on failure.
    Caller is responsible for further parsing (JSON, etc.).

    If *usage_out* is provided (a mutable dict), it will be populated with:
        input_tokens (int), output_tokens (int), model (str), provider (str)
    """
    from ..skills import get_skill_registry
    from ..services.protocols import Message

    skill = get_skill_registry().get(skill_name)
    if skill is None:
        logger.warning("Skill '%s' not found", skill_name)
        return None

    llm = get_pipeline_llm(
        workload=workload,
        prefer_cloud=prefer_cloud,
        try_openrouter=try_openrouter,
        auto_activate_ollama=auto_activate_ollama,
        openrouter_model=openrouter_model,
    )
    if llm is None:
        logger.warning("No LLM available for skill '%s'", skill_name)
        return None

    messages = [
        Message(role="system", content=skill.content),
        Message(
            role="user",
            content=json.dumps(payload, separators=(",", ":"), default=str),
        ),
    ]

    kwargs: dict[str, Any] = {}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if guided_json is not None:
        kwargs["guided_json"] = guided_json

    call_span_name = span_name or f"pipeline.{skill_name}"
    metadata = {"skill": skill_name, "workload": workload or "default"}
    if trace_metadata:
        metadata.update(trace_metadata)

    t0 = time.monotonic()
    try:
        result = llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        text = result.get("response", "").strip()
        usage = result.get("usage", {})
        model_name = getattr(llm, "model", getattr(llm, "model_id", ""))
        provider_name = getattr(llm, "name", "")
        trace_meta = result.get("_trace_meta", {})
        cached_tokens, cache_write_tokens, billable_input_tokens = _trace_cache_metrics(usage, trace_meta)

        # Populate usage_out for callers that want token tracking
        if usage_out is not None:
            usage_out["input_tokens"] = usage.get("input_tokens", 0)
            usage_out["output_tokens"] = usage.get("output_tokens", 0)
            usage_out["cached_tokens"] = cached_tokens
            usage_out["cache_write_tokens"] = cache_write_tokens
            if billable_input_tokens is not None:
                usage_out["billable_input_tokens"] = billable_input_tokens
            usage_out["model"] = model_name
            usage_out["provider"] = provider_name

        # Emit FTL trace span with full I/O and provider metadata
        trace_llm_call(
            span_name=call_span_name,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            billable_input_tokens=billable_input_tokens,
            model=model_name,
            provider=provider_name,
            duration_ms=(time.monotonic() - t0) * 1000,
            metadata=metadata,
            input_data={"messages": [{"role": m.role, "content": m.content[:500]} for m in messages]},
            output_data={"response": text[:2000]} if text else None,
            api_endpoint=trace_meta.get("api_endpoint"),
            provider_request_id=trace_meta.get("provider_request_id"),
            ttft_ms=trace_meta.get("ttft_ms"),
            inference_time_ms=trace_meta.get("inference_time_ms"),
            queue_time_ms=trace_meta.get("queue_time_ms"),
        )

        if not text:
            logger.warning("LLM returned empty response for skill '%s'", skill_name)
            return None

        # Clean think tags (Qwen3 models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    except Exception as exc:
        logger.exception("LLM call failed for skill '%s'", skill_name)
        trace_llm_call(
            span_name=call_span_name,
            model=getattr(llm, "model", ""),
            provider=getattr(llm, "name", ""),
            duration_ms=(time.monotonic() - t0) * 1000,
            status="failed",
            metadata=metadata,
            error_message=str(exc)[:500],
            error_type=type(exc).__name__,
            input_data={"messages": [{"role": m.role, "content": m.content[:500]} for m in messages]},
        )
        return None


# ------------------------------------------------------------------
# FTL trace helper for direct llm.chat() callers
# ------------------------------------------------------------------


def trace_llm_call(
    span_name: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    billable_input_tokens: int | None = None,
    model: str = "",
    provider: str = "",
    duration_ms: float = 0,
    status: str = "completed",
    metadata: dict[str, Any] | None = None,
    # I/O capture
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    # Timing breakdown
    ttft_ms: float | None = None,
    inference_time_ms: float | None = None,
    queue_time_ms: float | None = None,
    # RAG / retrieval
    context_tokens: int | None = None,
    retrieval_latency_ms: float | None = None,
    rag_graph_used: bool | None = None,
    rag_nodes_retrieved: int | None = None,
    rag_chunks_used: int | None = None,
    # Provider debugging
    api_endpoint: str | None = None,
    provider_request_id: str | None = None,
    # Reasoning
    reasoning: str | None = None,
    # Pricing override (for discounted batch lanes)
    cost_usd_override: float | None = None,
    # Error details
    error_message: str | None = None,
    error_type: str | None = None,
) -> None:
    """Emit an FTL trace span for an LLM call (fire-and-forget).

    Use this for direct ``llm.chat()`` callers that don't go through
    ``call_llm_with_skill()``.  Lightweight -- no-ops when FTL is disabled.

    Accepts all fields supported by ``tracer.end_span()`` so callers can
    pass through provider metadata, I/O data, timing breakdowns, RAG
    metrics, and business context.
    """
    from ..services.tracing import tracer

    if not tracer.enabled:
        return

    span = tracer.start_span(
        span_name=span_name,
        operation_type="llm_call",
        model_name=model,
        model_provider=provider,
        metadata=metadata or {},
    )
    tracer.end_span(
        span,
        status=status,
        input_tokens=input_tokens if input_tokens else None,
        output_tokens=output_tokens if output_tokens else None,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        billable_input_tokens=billable_input_tokens,
        input_data=input_data,
        output_data=output_data,
        ttft_ms=ttft_ms,
        inference_time_ms=inference_time_ms,
        queue_time_ms=queue_time_ms,
        context_tokens=context_tokens,
        retrieval_latency_ms=retrieval_latency_ms,
        rag_graph_used=rag_graph_used,
        rag_nodes_retrieved=rag_nodes_retrieved,
        rag_chunks_used=rag_chunks_used,
        api_endpoint=api_endpoint,
        provider_request_id=provider_request_id,
        reasoning=reasoning,
        cost_usd_override=cost_usd_override,
        error_message=error_message,
        error_type=error_type,
        duration_ms_override=duration_ms if duration_ms > 0 else None,
    )
