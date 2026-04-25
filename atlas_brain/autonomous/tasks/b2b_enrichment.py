"""
B2B review enrichment: extract churn signals from pending reviews via the
current two-tier pipeline.

Flow:
  1. Tier 1 extraction for base factual fields
  2. Conditional Tier 2 classification when Tier 1 leaves extraction gaps
  3. Deterministic finalize/validation before persistence

Polls b2b_reviews WHERE enrichment_status = 'pending', stores the finalized
enrichment JSONB payload, and sets status to `enriched`, `no_signal`, or
`quarantined`.

Runs on an interval (default 5 min). Returns _skip_synthesis so the
runner does not double-synthesize.
"""

import asyncio
import inspect
import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from ...config import B2BChurnConfig, settings
from ...services.b2b.reviewer_identity import sanitize_reviewer_title
from ...services.company_normalization import normalize_company_name
from ...services.scraping.sources import filter_deprecated_sources, parse_source_allowlist
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._b2b_shared import _fetch_review_funnel_audit
from ._b2b_witnesses import (
    derive_evidence_spans,
    derive_operating_model_shift,
    derive_org_pressure_type,
    derive_productivity_delta_claim,
    derive_replacement_mode,
    derive_salience_flags,
)
from ._execution_progress import task_run_id as _task_run_id

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")

_TIER1_JSON_SCHEMA: dict[str, Any] = {
    "title": "b2b_churn_extraction",
    "type": "object",
    "additionalProperties": True,
}
_TIER2_INSIDER_SECTION_HEADER = "### insider_signals -- CLASSIFY + EXTRACT (only for insider_account)"
_TIER2_OUTPUT_SECTION_HEADER = "## Output"


def _coerce_int_value(raw_value: Any, fallback: int) -> int:
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        if raw_value != raw_value:
            return fallback
        return int(raw_value)
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return fallback
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return fallback
    return fallback


def _coerce_float_value(raw_value: Any, fallback: float) -> float:
    if isinstance(raw_value, bool):
        return float(raw_value)
    if isinstance(raw_value, (int, float)):
        numeric = float(raw_value)
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return fallback
        try:
            numeric = float(text)
        except ValueError:
            return fallback
    else:
        return fallback
    if numeric != numeric:
        return fallback
    return numeric


def _config_allowlist(raw_value: Any, fallback: str | list[str] | tuple[str, ...] | set[str] | frozenset[str] = "") -> list[str]:
    candidate = raw_value if isinstance(raw_value, (str, list, tuple, set, frozenset)) else fallback
    return list(parse_source_allowlist(candidate))


def _enrichment_batch_custom_id(stage: str, review_id: Any) -> str:
    normalized_stage = re.sub(r"[^A-Za-z0-9_-]+", "_", str(stage or "").strip()).strip("_") or "stage"
    normalized_review = re.sub(r"[^A-Za-z0-9_-]+", "_", str(review_id or "").strip()).strip("_") or "review"
    return f"{normalized_stage}_{normalized_review}"[:64]

def _get_base_enrichment_llm(local_only: bool):
    """Resolve the deterministic local enrichment model from vLLM only."""
    from ...pipelines.llm import get_pipeline_llm

    return get_pipeline_llm(
        workload="vllm",
        try_openrouter=False,
        auto_activate_ollama=False,
    )


def _tier2_system_prompt_for_content_type(prompt: str, content_type: str | None) -> str:
    """Skip insider-account instructions for non-insider rows to save tokens."""
    if str(content_type or "").strip().lower() == "insider_account":
        return prompt
    before, marker, after = prompt.partition(_TIER2_INSIDER_SECTION_HEADER)
    if not marker:
        return prompt
    _insider_body, output_marker, output_tail = after.partition(_TIER2_OUTPUT_SECTION_HEADER)
    if not output_marker:
        return prompt
    return f"{before.rstrip()}\n\n{_TIER2_OUTPUT_SECTION_HEADER}{output_tail}"


_ANTHROPIC_CACHE_MIN_CHARS = 1024


def _maybe_anthropic_cache(
    model_id: str,
    messages: list[dict],
) -> list[dict]:
    """Add cache_control: ephemeral to the system message for Anthropic models.

    The direct httpx callers in this module bypass the OpenRouterLLM wrapper,
    so they were sending plain string `content` to Anthropic models -- which
    means OpenRouter never marked the system block as cacheable. Result: 0
    cached tokens across all enrichment calls even with a static ~12k-token
    Tier 1 prompt.

    This helper rewrites the system message in-place to the Anthropic
    content-block array form when:
      - the model is an Anthropic model on OpenRouter ("anthropic/" prefix)
      - the system content is large enough to be worth caching (>= 1024 chars,
        matching the OpenRouterLLM heuristic and Anthropic's cache minimum)

    For non-Anthropic models, returns messages unchanged so OpenAI/Mistral/etc
    callers aren't broken.
    """
    if not str(model_id or "").startswith("anthropic/"):
        return messages
    converted: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if (
            msg.get("role") == "system"
            and isinstance(content, str)
            and len(content) >= _ANTHROPIC_CACHE_MIN_CHARS
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


_tier1_client = None
_tier1_client_signature = None


def _get_tier1_client(cfg):
    """Get or create a pooled httpx.AsyncClient for Tier 1 vLLM calls."""
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


async def _lookup_cached_json_response(
    namespace: str,
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    """Lookup a cached structured response and return the envelope either way."""
    from ...pipelines.llm import clean_llm_output, parse_json_response
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_stage_request,
    )

    request = prepare_b2b_exact_stage_request(
        namespace,
        provider=provider,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        guided_json=guided_json,
    )
    cached = await lookup_b2b_exact_stage_text(request)
    if cached is None:
        return None, request.request_envelope, False

    text = clean_llm_output(cached["response_text"])
    parsed = parse_json_response(text, recover_truncated=True)
    if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
        return parsed, request.request_envelope, True
    return None, request.request_envelope, False


def _unpack_cached_lookup_result(
    result: tuple[Any, ...],
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    if len(result) == 3:
        cached, request_envelope, cache_hit = result
        return cached, request_envelope, bool(cache_hit)
    if len(result) == 2:
        cached, request_envelope = result
        return cached, request_envelope, cached is not None
    raise ValueError(f"Unexpected cached lookup result shape: {len(result)}")


def _unpack_stage_result(
    result: tuple[Any, ...],
) -> tuple[dict[str, Any] | None, str | None, bool]:
    if len(result) == 3:
        parsed, model_id, cache_hit = result
        return parsed, model_id, bool(cache_hit)
    if len(result) == 2:
        parsed, model_id = result
        return parsed, model_id, False
    raise ValueError(f"Unexpected stage result shape: {len(result)}")


def _pack_stage_result(
    parsed: dict[str, Any] | None,
    model_id: str | None,
    cache_hit: bool,
    *,
    include_cache_hit: bool,
) -> tuple[dict[str, Any] | None, str | None] | tuple[dict[str, Any] | None, str | None, bool]:
    if include_cache_hit:
        return parsed, model_id, cache_hit
    return parsed, model_id


def _trace_enrichment_llm_call(
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
    normalized_usage = {
        "input_tokens": usage_dict.get("input_tokens", usage_dict.get("prompt_tokens", 0)),
        "output_tokens": usage_dict.get("output_tokens", usage_dict.get("completion_tokens", 0)),
        "billable_input_tokens": usage_dict.get("billable_input_tokens", usage_dict.get("prompt_tokens")),
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


async def _store_cached_json_response(
    namespace: str,
    request_envelope: dict[str, Any],
    *,
    provider: str,
    model: str,
    response_text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Store a validated structured response for later exact reuse."""
    from ...services.b2b.cache_runner import (
        bind_b2b_exact_stage_request,
        store_b2b_exact_stage_text,
    )

    request = bind_b2b_exact_stage_request(
        namespace,
        provider=provider,
        model=model,
        request_envelope=request_envelope,
    )
    await store_b2b_exact_stage_text(
        request,
        response_text=response_text,
        metadata=metadata,
    )


async def _call_vllm_tier1(
    payload_json: str,
    cfg,
    client,
    *,
    include_cache_hit: bool = False,
    trace_metadata: dict[str, Any] | None = None,
) -> tuple[dict | None, str | None] | tuple[dict | None, str | None, bool]:
    """Tier 1 extraction: deterministic fields via local vLLM.

    Returns (result_dict, model_id, cache_hit) or (None, None, False) on failure.
    """
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier1")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier1' not found")
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    model_id = cfg.enrichment_tier1_model
    request_envelope: dict[str, Any] | None = None
    messages = [
        {"role": "system", "content": skill.content},
        {"role": "user", "content": payload_json},
    ]
    try:
        cached, request_envelope, cache_hit = _unpack_cached_lookup_result(
            await _lookup_cached_json_response(
            "b2b_enrichment.tier1",
            provider="vllm",
            model=model_id,
            system_prompt=skill.content,
            user_content=payload_json,
            max_tokens=cfg.enrichment_tier1_max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            guided_json=_TIER1_JSON_SCHEMA,
            )
        )
        if cached is not None:
            return _pack_stage_result(cached, model_id, cache_hit, include_cache_hit=include_cache_hit)

        call_started = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": cfg.enrichment_tier1_model,
                "messages": messages,
                "max_tokens": cfg.enrichment_tier1_max_tokens,
                "temperature": 0.0,
                "guided_json": _TIER1_JSON_SCHEMA,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        body = resp.json()
        _trace_enrichment_llm_call(
            "task.b2b_enrichment.tier1",
            provider="vllm",
            model=model_id,
            messages=messages,
            usage=body.get("usage", {}),
            metadata=trace_metadata,
            duration_ms=(time.monotonic() - call_started) * 1000,
            api_endpoint=f"{str(getattr(client, 'base_url', '')).rstrip('/')}/v1/chat/completions",
        )
        text = body["choices"][0]["message"]["content"].strip()
        if not text:
            return _pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)

        from ...pipelines.llm import clean_llm_output, parse_json_response
        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            if request_envelope is not None:
                await _store_cached_json_response(
                    "b2b_enrichment.tier1",
                    request_envelope,
                    provider="vllm",
                    model=model_id,
                    response_text=text,
                    metadata={"tier": 1, "backend": "vllm"},
                )
            return _pack_stage_result(parsed, model_id, False, include_cache_hit=include_cache_hit)
        return _pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Tier 1 vLLM returned invalid JSON")
        return _pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
    except Exception:
        logger.exception("Tier 1 vLLM call failed")
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)


async def _call_openrouter_tier1(
    payload_json: str,
    cfg,
    *,
    include_cache_hit: bool = False,
    trace_metadata: dict[str, Any] | None = None,
) -> tuple[dict | None, str | None] | tuple[dict | None, str | None, bool]:
    """Tier 1 extraction via OpenRouter (cloud model, no guided_json)."""
    import httpx
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier1")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier1' not found")
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    api_key = cfg.openrouter_api_key
    if not api_key:
        logger.warning("OpenRouter API key not configured for enrichment")
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    model_id = cfg.enrichment_openrouter_model or "anthropic/claude-haiku-4-5"
    request_envelope: dict[str, Any] | None = None
    messages = [
        {"role": "system", "content": skill.content},
        {"role": "user", "content": payload_json},
    ]
    try:
        cached, request_envelope, cache_hit = _unpack_cached_lookup_result(
            await _lookup_cached_json_response(
            "b2b_enrichment.tier1",
            provider="openrouter",
            model=model_id,
            system_prompt=skill.content,
            user_content=payload_json,
            max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096),
            temperature=0.0,
            response_format={"type": "json_object"},
            )
        )
        if cached is not None:
            return _pack_stage_result(cached, model_id, cache_hit, include_cache_hit=include_cache_hit)

        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
            call_started = time.monotonic()
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": _maybe_anthropic_cache(model_id, messages),
                    "max_tokens": max(cfg.enrichment_tier1_max_tokens, 4096),
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            body = resp.json()
            _trace_enrichment_llm_call(
                "task.b2b_enrichment.tier1",
                provider="openrouter",
                model=model_id,
                messages=messages,
                usage=body.get("usage", {}),
                metadata=trace_metadata,
                duration_ms=(time.monotonic() - call_started) * 1000,
                api_endpoint="https://openrouter.ai/api/v1/chat/completions",
                provider_request_id=resp.headers.get("x-request-id") or body.get("id"),
            )
            choices = body.get("choices") or []
            if not choices:
                logger.warning("OpenRouter returned no choices")
                return _pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            # Reasoning models (o1/o3/gpt-oss) may put output in reasoning field
            if not text and msg.get("reasoning"):
                # Try to extract JSON from the reasoning
                reasoning = msg["reasoning"]
                import re as _re
                json_match = _re.search(r"\{[\s\S]*\}", reasoning)
                if json_match:
                    text = json_match.group(0)
            text = text.strip()
            if not text:
                logger.warning("OpenRouter returned empty content")
                return _pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)

            from ...pipelines.llm import clean_llm_output, parse_json_response
            text = clean_llm_output(text)
            parsed = parse_json_response(text, recover_truncated=True)
            if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                if request_envelope is not None:
                    await _store_cached_json_response(
                        "b2b_enrichment.tier1",
                        request_envelope,
                        provider="openrouter",
                        model=model_id,
                        response_text=text,
                        metadata={"tier": 1, "backend": "openrouter"},
                    )
                return _pack_stage_result(parsed, model_id, False, include_cache_hit=include_cache_hit)
            logger.warning("OpenRouter tier 1 returned unparseable JSON")
            return _pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
    except Exception:
        logger.exception("OpenRouter tier 1 call failed")
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)


def _tier1_has_extraction_gaps(tier1: dict, *, source: str | None = None) -> bool:
    """Check if tier 1 left gaps that tier 2 should fill.

    Tier 2 adds: pain_categories, competitor classification, buyer_authority,
    sentiment_trajectory. These are ALWAYS missing from tier 1 (by design).
    So we only trigger tier 2 when tier 1 missed verbatim extractions that
    indicate the review has substance worth classifying.
    """
    complaints = tier1.get("specific_complaints") or []
    quotes = tier1.get("quotable_phrases") or []
    competitors = tier1.get("competitors_mentioned") or []
    pricing = tier1.get("pricing_phrases") or []
    rec_lang = tier1.get("recommendation_language") or []
    churn = tier1.get("churn_signals") or {}
    has_churn = any(bool(v) for v in churn.values())
    has_evidence = bool(complaints or quotes or competitors or pricing or rec_lang)
    source_norm = str(source or "").strip().lower()
    strict_sources = set(
        _config_allowlist(
            getattr(settings.b2b_churn, "enrichment_tier2_strict_sources", ""),
            "",
        )
    )
    if source_norm in strict_sources:
        complaint_count = len(complaints) if isinstance(complaints, list) else 0
        quote_count = len(quotes) if isinstance(quotes, list) else 0
        evidence_groups = sum(
            1
            for present in (
                bool(complaints),
                bool(quotes),
                bool(competitors),
                bool(pricing),
                bool(rec_lang),
            )
            if present
        )
        has_strong_structured_evidence = (
            bool(competitors)
            or bool(pricing)
            or complaint_count >= _coerce_int_value(
                getattr(settings.b2b_churn, "enrichment_tier2_strict_min_complaints", 2),
                2,
            )
            or (
                quote_count >= _coerce_int_value(
                    getattr(settings.b2b_churn, "enrichment_tier2_strict_min_quotes", 2),
                    2,
                )
                and evidence_groups >= 2
            )
        )
        return has_churn or has_strong_structured_evidence
    # Tier 2 fires when the review has substance worth classifying:
    # 1. Any churn signal or negative evidence -> need pain classification
    # 2. Competitors mentioned -> need evidence_type + displacement scoring
    # Skip tier 2 ONLY for purely positive reviews with zero signals
    return has_churn or has_evidence


async def _call_vllm_tier2(
    tier1_result: dict,
    row: dict,
    cfg: Any,
    client: Any,
    truncate_length: int,
    *,
    include_cache_hit: bool = False,
    trace_metadata: dict[str, Any] | None = None,
) -> tuple[dict | None, str | None] | tuple[dict | None, str | None, bool]:
    """Tier 2 extraction: classify + extract via local vLLM.

    Receives Tier 1 output as context so it can reference extracted complaints
    and quotes when classifying pain and detecting indicators.
    Returns (result_dict, model_id, cache_hit) or (None, None, False) on failure.
    """
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier2' not found")
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    tier2_model = cfg.enrichment_tier2_model or cfg.enrichment_tier1_model
    payload = _build_classify_payload(row, truncate_length)
    # Inject Tier 1 extractions for Tier 2 to reference
    payload["tier1_specific_complaints"] = tier1_result.get("specific_complaints", [])
    payload["tier1_quotable_phrases"] = tier1_result.get("quotable_phrases", [])
    payload_json = json.dumps(payload)
    system_prompt = _tier2_system_prompt_for_content_type(
        skill.content,
        payload.get("content_type"),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload_json},
    ]

    try:
        request_envelope: dict[str, Any] | None
        cached, request_envelope, cache_hit = _unpack_cached_lookup_result(
            await _lookup_cached_json_response(
            "b2b_enrichment.tier2",
            provider="vllm",
            model=tier2_model,
            system_prompt=system_prompt,
            user_content=payload_json,
            max_tokens=cfg.enrichment_tier2_max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            )
        )
        if cached is not None:
            return _pack_stage_result(cached, tier2_model, cache_hit, include_cache_hit=include_cache_hit)

        call_started = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": tier2_model,
                "messages": messages,
                "max_tokens": cfg.enrichment_tier2_max_tokens,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        body = resp.json()
        _trace_enrichment_llm_call(
            "task.b2b_enrichment.tier2",
            provider="vllm",
            model=tier2_model,
            messages=messages,
            usage=body.get("usage", {}),
            metadata=trace_metadata,
            duration_ms=(time.monotonic() - call_started) * 1000,
            api_endpoint=f"{str(getattr(client, 'base_url', '')).rstrip('/')}/v1/chat/completions",
        )
        text = body["choices"][0]["message"]["content"].strip()
        if not text:
            return _pack_stage_result(None, tier2_model, False, include_cache_hit=include_cache_hit)

        from ...pipelines.llm import clean_llm_output, parse_json_response
        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            if request_envelope is not None:
                await _store_cached_json_response(
                    "b2b_enrichment.tier2",
                    request_envelope,
                    provider="vllm",
                    model=tier2_model,
                    response_text=text,
                    metadata={"tier": 2, "backend": "vllm"},
                )
            return _pack_stage_result(parsed, tier2_model, False, include_cache_hit=include_cache_hit)
        return _pack_stage_result(None, tier2_model, False, include_cache_hit=include_cache_hit)
    except Exception:
        logger.warning("Tier 2 vLLM call failed", exc_info=True)
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)


def _get_tier2_client(cfg: Any) -> Any:
    """Get or create the httpx client for Tier 2 vLLM."""
    raw_tier2_url = getattr(cfg, "enrichment_tier2_vllm_url", "")
    raw_tier1_url = getattr(cfg, "enrichment_tier1_vllm_url", "")
    tier2_url = raw_tier2_url if isinstance(raw_tier2_url, str) and raw_tier2_url.strip() else raw_tier1_url
    timeout = _coerce_float_value(getattr(cfg, "enrichment_tier2_timeout_seconds", 120.0), 120.0)
    # Reuse the Tier 1 client if same URL
    if not isinstance(tier2_url, str) or not tier2_url.strip() or tier2_url == raw_tier1_url:
        return _get_tier1_client(cfg)
    import httpx
    return httpx.AsyncClient(base_url=tier2_url, timeout=timeout)


async def _call_openrouter_tier2(
    tier1_result: dict,
    row: dict,
    cfg: Any,
    truncate_length: int,
    *,
    include_cache_hit: bool = False,
    trace_metadata: dict[str, Any] | None = None,
) -> tuple[dict | None, str | None] | tuple[dict | None, str | None, bool]:
    """Tier 2 extraction via OpenRouter (cloud model).

    Mirrors _call_openrouter_tier1 but uses the tier2 skill and injects
    Tier 1 extractions for context so the model can classify pain categories
    and evidence types against already-extracted complaints and quotes.
    """
    import httpx
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier2' not found for OpenRouter tier 2")
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    api_key = cfg.openrouter_api_key
    if not api_key:
        logger.warning("OpenRouter API key not configured for tier 2 enrichment")
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    model_id = (
        cfg.enrichment_tier2_openrouter_model
        or cfg.enrichment_openrouter_model
        or "anthropic/claude-haiku-4-5"
    )
    payload = _build_classify_payload(row, truncate_length)
    payload["tier1_specific_complaints"] = tier1_result.get("specific_complaints", [])
    payload["tier1_quotable_phrases"] = tier1_result.get("quotable_phrases", [])
    payload_json = json.dumps(payload)
    system_prompt = _tier2_system_prompt_for_content_type(
        skill.content,
        payload.get("content_type"),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload_json},
    ]

    try:
        request_envelope: dict[str, Any] | None
        cached, request_envelope, cache_hit = _unpack_cached_lookup_result(
            await _lookup_cached_json_response(
            "b2b_enrichment.tier2",
            provider="openrouter",
            model=model_id,
            system_prompt=system_prompt,
            user_content=payload_json,
            max_tokens=cfg.enrichment_tier2_max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            )
        )
        if cached is not None:
            return _pack_stage_result(cached, model_id, cache_hit, include_cache_hit=include_cache_hit)

        async with httpx.AsyncClient(timeout=httpx.Timeout(cfg.enrichment_tier2_timeout_seconds, connect=10.0)) as client:
            call_started = time.monotonic()
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": _maybe_anthropic_cache(model_id, messages),
                    "max_tokens": cfg.enrichment_tier2_max_tokens,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            body = resp.json()
            _trace_enrichment_llm_call(
                "task.b2b_enrichment.tier2",
                provider="openrouter",
                model=model_id,
                messages=messages,
                usage=body.get("usage", {}),
                metadata=trace_metadata,
                duration_ms=(time.monotonic() - call_started) * 1000,
                api_endpoint="https://openrouter.ai/api/v1/chat/completions",
                provider_request_id=resp.headers.get("x-request-id") or body.get("id"),
            )
            choices = body.get("choices") or []
            if not choices:
                logger.warning("OpenRouter tier 2 returned no choices")
                return _pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
            text = (choices[0].get("message") or {}).get("content") or ""
            text = text.strip()
            if not text:
                logger.warning("OpenRouter tier 2 returned empty content")
                return _pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
            from ...pipelines.llm import clean_llm_output, parse_json_response
            text = clean_llm_output(text)
            parsed = parse_json_response(text, recover_truncated=True)
            if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                if request_envelope is not None:
                    await _store_cached_json_response(
                        "b2b_enrichment.tier2",
                        request_envelope,
                        provider="openrouter",
                        model=model_id,
                        response_text=text,
                        metadata={"tier": 2, "backend": "openrouter"},
                    )
                return _pack_stage_result(parsed, model_id, False, include_cache_hit=include_cache_hit)
            logger.warning("OpenRouter tier 2 returned unparseable JSON")
            return _pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
    except Exception:
        logger.warning("OpenRouter tier 2 call failed", exc_info=True)
        return _pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)


def _merge_tier1_tier2(tier1: dict, tier2: dict | None) -> dict:
    """Merge Tier 1 + Tier 2 deterministic extraction into a single 47-field JSONB.

    Tier 1 provides the base. Tier 2 keys are overlaid on top.
    competitors_mentioned is merged by name (case-insensitive).
    If tier2 is None (failed), apply safe defaults for Tier 2 fields.
    """
    result = dict(tier1)

    if tier2 is None:
        # Tier 2 failed -- apply minimal defaults for CLASSIFY fields only.
        # INFER-derived fields (urgency_score, would_recommend, pain_category,
        # has_budget_authority, price_complaint, price_context, sentiment direction)
        # will be computed by _compute_derived_fields() downstream.
        result.setdefault("pain_categories", [])
        result.setdefault("sentiment_trajectory", {})
        result.setdefault("buyer_authority", {"role_type": "unknown", "buying_stage": "unknown",
                                              "executive_sponsor_mentioned": False})
        result.setdefault("timeline", {"decision_timeline": "unknown"})
        result.setdefault("contract_context", {"contract_value_signal": "unknown"})
        result.setdefault("insider_signals", None)
        result.setdefault("positive_aspects", [])
        result.setdefault("feature_gaps", [])
        result.setdefault("recommendation_language", [])
        result.setdefault("pricing_phrases", [])
        result.setdefault("event_mentions", [])
        result.setdefault("urgency_indicators", {})
        # Leave competitors_mentioned as-is from Tier 1 (partial data)
        for comp in result.get("competitors_mentioned", []):
            comp.setdefault("evidence_type", "neutral_mention")
            comp.setdefault("displacement_confidence", "low")
            comp.setdefault("reason_category", None)
        return result

    # --- Tier 2 succeeded: overlay CLASSIFY + EXTRACT fields ---
    _TIER2_TOP_LEVEL_KEYS = {
        "pain_categories",
        "sentiment_trajectory", "buyer_authority", "timeline",
        "contract_context", "insider_signals",
        "positive_aspects", "feature_gaps",
        # New v2 EXTRACT fields
        "recommendation_language", "pricing_phrases",
        "event_mentions", "urgency_indicators",
    }
    # Also accept legacy INFER keys if a v1 Tier 2 model returns them
    _LEGACY_TIER2_KEYS = {"urgency_score", "pain_category", "would_recommend"}
    for key in _TIER2_TOP_LEVEL_KEYS | _LEGACY_TIER2_KEYS:
        if key in tier2:
            result[key] = tier2[key]

    # Merge competitors_mentioned by name (case-insensitive)
    tier1_comps = {c["name"].lower(): c for c in result.get("competitors_mentioned", []) if isinstance(c, dict) and "name" in c}
    tier2_comps = tier2.get("competitors_mentioned", []) or []

    merged_comps = []
    seen = set()
    for t2_comp in tier2_comps:
        if not isinstance(t2_comp, dict) or "name" not in t2_comp:
            continue
        key = t2_comp["name"].lower()
        seen.add(key)
        base = dict(tier1_comps.get(key, {"name": t2_comp["name"]}))
        # Overlay Tier 2 fields
        for field in ("evidence_type", "displacement_confidence", "reason_category"):
            if field in t2_comp:
                base[field] = t2_comp[field]
        # Ensure name comes from Tier 1 if available (preserves original casing)
        if key in tier1_comps:
            base["name"] = tier1_comps[key]["name"]
        merged_comps.append(base)

    # Append Tier 1 competitors not in Tier 2 (with defaults)
    for key, t1_comp in tier1_comps.items():
        if key not in seen:
            t1_comp.setdefault("evidence_type", "neutral_mention")
            t1_comp.setdefault("displacement_confidence", "low")
            t1_comp.setdefault("reason_category", None)
            merged_comps.append(t1_comp)

    result["competitors_mentioned"] = merged_comps
    return result


def _build_pain_patterns(
    keywords: dict[str, tuple[str, ...]],
) -> dict[str, re.Pattern[str]]:
    """Compile keyword tuples into word-boundary regexes per category."""
    compiled: dict[str, re.Pattern[str]] = {}
    for category, needles in keywords.items():
        parts = [r"\b" + re.escape(n) + r"\b" for n in needles]
        compiled[category] = re.compile("|".join(parts), re.IGNORECASE)
    return compiled


_PAIN_KEYWORDS_RAW = {
    "pricing": (
        "price", "pricing", "cost", "costly", "expensive", "overpriced", "renewal",
        "invoice", "invoiced", "billing", "billed", "charged", "charge", "overcharge",
        "fee", "fees", "refund", "cost increase", "price increase",
    ),
    "support": ("support", "ticket", "response", "customer service", "escalation", "escalated", "escalate"),
    "features": ("feature", "functionality", "capability", "missing"),
    "ux": ("ui", "ux", "interface", "clunky", "usability", "navigation"),
    "reliability": ("outage", "downtime", "crash", "bug", "unstable", "reliable"),
    "performance": ("slow", "latency", "lag", "performance", "speed"),
    "integration": ("integration", "integrate", "sync", "connector", "api"),
    "security": ("security", "permission", "access control", "compliance", "sso", "mfa"),
    "onboarding": ("onboarding", "setup", "implementation", "training", "adoption"),
    "technical_debt": ("technical debt", "legacy", "outdated", "deprecated", "workaround"),
    "contract_lock_in": (
        "lock-in", "locked in", "vendor lock", "multi-year", "exit fee", "cancel",
        "cancellation", "terminate", "termination", "auto renew", "automatic renewal",
        "renewed without notice", "notice period", "contract term", "contract trap",
        "billing dispute", "runaround",
    ),
    "data_migration": ("migration", "migrate", "import", "export", "data transfer"),
    "api_limitations": ("api", "webhook", "sdk", "rate limit", "endpoint"),
    "privacy": ("spam", "unsubscribe", "unsolicited", "data breach", "privacy violation"),
}
# Keep the dict name for any external references; remove "value" (matches
# valuable/evaluate/values too broadly in review text).
_PAIN_KEYWORDS = _PAIN_KEYWORDS_RAW
_PAIN_PATTERNS = _build_pain_patterns(_PAIN_KEYWORDS_RAW)


def _normalize_text_list(values: Any) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        if value:
            normalized.append(str(value))
    return normalized


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    haystack = text.lower()
    return any(needle in haystack for needle in needles)


def _pain_scores(texts: list[str]) -> dict[str, int]:
    scores = {category: 0 for category in _PAIN_PATTERNS}
    for text in texts:
        for category, pattern in _PAIN_PATTERNS.items():
            if pattern.search(text):
                scores[category] += 1
    return scores


def _primary_reason_category(*texts: str) -> str | None:
    normalized = [text for text in texts if text]
    if not normalized:
        return None
    scored = _pain_scores(normalized)
    ranked = sorted(
        ((score, category) for category, score in scored.items() if score > 0),
        reverse=True,
    )
    return ranked[0][1] if ranked else None


_PAIN_DERIVATION_FIELDS: tuple[str, ...] = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "quotable_phrases",
)


def _subject_vendor_phrase_texts(result: dict, field: str) -> list[str]:
    """Return phrase texts allowed to describe the subject vendor's pain.

    v1 rows have no subject metadata, so this preserves legacy behavior.
    v2 rows are conservative: only subject_vendor phrases survive.
    """
    raw = result.get(field) or []
    if not raw:
        return []

    from ._b2b_phrase_metadata import is_v2_tagged, phrase_metadata_map

    if not is_v2_tagged(result):
        return _normalize_text_list(raw)

    meta = phrase_metadata_map(result)
    texts: list[str] = []
    for index, value in enumerate(raw):
        phrase = str(value or "").strip()
        if not phrase:
            continue
        row = meta.get((field, index)) or {}
        if row.get("subject") == "subject_vendor":
            texts.append(phrase)
    return texts


def _derive_pain_categories(result: dict) -> list[dict[str, str]]:
    """Derive pain_categories list from extracted phrases.

    Phase 2 (Layer 1 -- subject attribution gate): v2 phrase metadata
    filters phrases to subject='subject_vendor' before scoring so self /
    alternative / third_party references cannot inflate the subject
    vendor's pain.

    Phase 3 (Layer 2 -- polarity gate): v2 phrase metadata further
    filters phrases by polarity. `positive` and `unclear` are dropped
    entirely. `negative` phrases count at full weight (1.0). `mixed`
    phrases count at half weight (0.5) because the phrase carries some
    negative content but the LLM flagged ambiguity.

    v1 data falls through to keyword-only scoring across all phrases at
    uniform weight.
    """
    from ._b2b_phrase_metadata import is_v2_tagged, phrase_metadata_map

    if is_v2_tagged(result):
        meta = phrase_metadata_map(result)
        weighted_items: list[tuple[str, float]] = []
        for field in _PAIN_DERIVATION_FIELDS:
            raw = result.get(field) or []
            for index, value in enumerate(raw):
                phrase = str(value or "").strip()
                if not phrase:
                    continue
                row = meta.get((field, index)) or {}
                if row.get("subject") != "subject_vendor":
                    continue
                polarity = row.get("polarity")
                if polarity == "negative":
                    weighted_items.append((phrase, 1.0))
                elif polarity == "mixed":
                    weighted_items.append((phrase, 0.5))
                # positive / unclear -> dropped (Layer 2)
        if not weighted_items:
            return []
        # Weighted scoring: each matched category accumulates the phrase's
        # polarity weight. Mirrors _pain_scores semantics but with weights.
        scores: dict[str, float] = {category: 0.0 for category in _PAIN_PATTERNS}
        for text, weight in weighted_items:
            for category, pattern in _PAIN_PATTERNS.items():
                if pattern.search(text):
                    scores[category] += weight
        ranked: list[tuple[float, str]] = [
            (score, category)
            for category, score in scores.items()
            if score > 0
        ]
    else:
        texts = (
            _normalize_text_list(result.get("specific_complaints"))
            + _normalize_text_list(result.get("pricing_phrases"))
            + _normalize_text_list(result.get("feature_gaps"))
            + _normalize_text_list(result.get("quotable_phrases"))
        )
        if not texts:
            return []
        scored = _pain_scores(texts)
        ranked = [
            (float(score), category)
            for category, score in scored.items()
            if score > 0
        ]

    ranked.sort(reverse=True)
    if not ranked:
        return [{"category": "overall_dissatisfaction", "severity": "primary"}]
    categories = [{"category": ranked[0][1], "severity": "primary"}]
    for _score, category in ranked[1:3]:
        if category != categories[0]["category"]:
            categories.append({"category": category, "severity": "secondary"})
    return categories


def _count_corroborating_signals(result: dict) -> int:
    """Count universal churn / sentiment signals that support a pain claim.

    These are independent of any specific pain category -- they indicate
    the reviewer is genuinely unhappy / leaving / dissatisfied, which
    corroborates that a single pain phrase is more than a passing mention.

    Signals considered:
      - churn_signals.intent_to_leave is True
      - churn_signals.actively_evaluating is True
      - churn_signals.migration_in_progress is True
      - would_recommend is False
      - sentiment_trajectory.direction in (consistently_negative, declining)
    """
    count = 0
    churn = result.get("churn_signals")
    if isinstance(churn, dict):
        for key in ("intent_to_leave", "actively_evaluating", "migration_in_progress"):
            if churn.get(key) is True:
                count += 1
    if result.get("would_recommend") is False:
        count += 1
    sentiment = result.get("sentiment_trajectory")
    if isinstance(sentiment, dict):
        direction = str(sentiment.get("direction") or "").strip().lower()
        if direction in ("consistently_negative", "declining"):
            count += 1
    return count


def _count_pain_phrase_matches(result: dict, pain_category: str) -> int:
    """Count vendor-subject negative/mixed phrases that match a pain category.

    For v2 results, applies the subject + polarity gates so only phrases
    that actually describe the subject vendor's negative experience count.
    For v1 results, falls back to keyword scan across raw phrases.
    """
    pattern = _PAIN_PATTERNS.get(pain_category)
    if pattern is None:
        return 0

    from ._b2b_phrase_metadata import is_v2_tagged, phrase_metadata_map

    count = 0
    if is_v2_tagged(result):
        meta = phrase_metadata_map(result)
        for field in _PAIN_DERIVATION_FIELDS:
            for index, value in enumerate(result.get(field) or []):
                phrase = str(value or "").strip()
                if not phrase:
                    continue
                row = meta.get((field, index)) or {}
                if row.get("subject") != "subject_vendor":
                    continue
                if row.get("polarity") not in ("negative", "mixed"):
                    continue
                if pattern.search(phrase):
                    count += 1
    else:
        texts: list[str] = []
        for field in _PAIN_DERIVATION_FIELDS:
            texts.extend(_normalize_text_list(result.get(field)))
        for text in texts:
            if pattern.search(text):
                count += 1
    return count


def _compute_pain_confidence(result: dict, pain_category: str) -> str:
    """Layer 3 (causality gate): grade evidence for the primary pain.

    Returns one of:
      "strong" -- 2+ matching phrases (multiple independent textual evidence)
                  OR (overall_dissatisfaction with 2+ universal signals)
      "weak"   -- 1 matching phrase + at least 1 universal signal
                  OR (overall_dissatisfaction with 1 universal signal)
      "none"   -- everything weaker; caller should demote a non-fallback
                  pain_category to overall_dissatisfaction
    """
    normalized = _normalize_pain_category(pain_category)
    signal_count = _count_corroborating_signals(result)

    if normalized == "overall_dissatisfaction":
        if signal_count >= 2:
            return "strong"
        if signal_count >= 1:
            return "weak"
        return "none"

    phrase_count = _count_pain_phrase_matches(result, normalized)
    if phrase_count >= 2:
        return "strong"
    if phrase_count >= 1 and signal_count >= 1:
        return "weak"
    return "none"


def _demote_primary_pain(result: dict, demoted_category: str) -> None:
    """Demote a Layer-3-rejected primary pain to a secondary entry.

    Replaces the primary in pain_categories with overall_dissatisfaction and
    re-attaches the original primary as secondary so the keyword evidence is
    not lost (downstream reports may still want to mention it as context).
    """
    if demoted_category == "overall_dissatisfaction":
        return
    existing = result.get("pain_categories")
    if not isinstance(existing, list):
        existing = []
    new_list: list[dict[str, str]] = [
        {"category": "overall_dissatisfaction", "severity": "primary"}
    ]
    appended_demoted = False
    for entry in existing:
        if not isinstance(entry, dict):
            continue
        category = str(entry.get("category") or "").strip().lower()
        if not category:
            continue
        if category == "overall_dissatisfaction":
            continue
        if category == demoted_category and not appended_demoted:
            new_list.append({"category": demoted_category, "severity": "secondary"})
            appended_demoted = True
        elif category != demoted_category:
            new_list.append(
                {"category": category, "severity": entry.get("severity", "secondary")}
            )
    if not appended_demoted:
        new_list.append({"category": demoted_category, "severity": "secondary"})
    result["pain_categories"] = new_list


_COMPETITOR_RECOVERY_PATTERNS = (
    r"\b(?:switched to|moved to|replaced with|migrating to|migration to)\s+([A-Z][A-Za-z0-9.&+/\-]*(?:\s+[A-Z][A-Za-z0-9.&+/\-]*){0,3})",
    r"\b(?:evaluating|looking at|considering|shortlisting|shortlisted|poc with|proof of concept with)\s+([A-Z][A-Za-z0-9.&+/\-]*(?:\s+[A-Z][A-Za-z0-9.&+/\-]*){0,3})",
)

_COMPETITOR_RECOVERY_BLOCKLIST = {
    "a", "an", "the", "another tool", "another vendor", "other tool", "other vendor",
    "new tool", "new vendor", "options", "alternative", "alternatives",
    "alternative platform", "alternative platforms", "crm",
    "provider", "providers", "competing provider", "competing providers",
}

_GENERIC_COMPETITOR_TOKENS = {
    "alternative", "alternatives", "platform", "platforms", "tool", "tools",
    "vendor", "vendors", "software", "solutions", "solution", "service",
    "services", "system", "systems", "crm", "suite", "app", "apps",
    "provider", "providers", "competing",
}

_COMPETITOR_CONTEXT_PATTERNS = (
    "switched to", "moved to", "replaced with", "migrating to", "migration to",
    "evaluating", "looking at", "considering", "shortlisting", "shortlisted",
    "poc with", "proof of concept with", "instead of", "compared to", "versus", " vs ",
)


def _is_generic_competitor_name(name: str) -> bool:
    normalized = normalize_company_name(name) or str(name or "").strip().lower()
    if not normalized:
        return True
    if normalized in _COMPETITOR_RECOVERY_BLOCKLIST:
        return True
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", str(name or ""))
        if token
    ]
    return bool(tokens) and all(token in _GENERIC_COMPETITOR_TOKENS for token in tokens)


def _has_named_competitor_context(name: str, source_row: dict[str, Any]) -> bool:
    candidate = str(name or "").strip()
    if not candidate:
        return False
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    name_lower = candidate.lower()
    for match in re.finditer(re.escape(name_lower), review_blob):
        start = max(0, match.start() - 96)
        end = min(len(review_blob), match.end() + 96)
        window = review_blob[start:end]
        if any(pattern in window for pattern in _COMPETITOR_CONTEXT_PATTERNS):
            return True
    return False


def _recover_competitor_mentions(result: dict, source_row: dict[str, Any]) -> list[dict[str, Any]]:
    existing = [
        dict(comp) for comp in (result.get("competitors_mentioned") or [])
        if isinstance(comp, dict) and str(comp.get("name") or "").strip()
    ]
    if not existing and not any(source_row.get(field) for field in ("summary", "review_text", "pros", "cons")):
        return existing

    incumbent_norm = normalize_company_name(str(source_row.get("vendor_name") or "")) or ""
    seen = {
        (normalize_company_name(str(comp.get("name") or "")) or str(comp.get("name") or "").strip().lower()): comp
        for comp in existing
    }

    recovery_blob = " ".join(
        [str(source_row.get(field) or "") for field in ("summary", "review_text", "pros", "cons")]
        + _normalize_text_list(result.get("quotable_phrases"))
    )

    for pattern in _COMPETITOR_RECOVERY_PATTERNS:
        for match in re.finditer(pattern, recovery_blob):
            candidate = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9.]+$", "", match.group(1).strip())
            if not candidate:
                continue
            normalized = normalize_company_name(candidate) or candidate.lower()
            if not normalized or normalized == incumbent_norm:
                continue
            if normalized in _COMPETITOR_RECOVERY_BLOCKLIST:
                continue
            generic_tokens = [
                token.lower()
                for token in re.findall(r"[A-Za-z0-9]+", candidate)
                if token
            ]
            if generic_tokens and all(token in _GENERIC_COMPETITOR_TOKENS for token in generic_tokens):
                continue
            if normalized in seen:
                continue
            seen[normalized] = {"name": candidate}

    return list(seen.values())


def _derive_competitor_annotations(result: dict, source_row: dict[str, Any]) -> list[dict[str, Any]]:
    comps = []
    churn = result.get("churn_signals") or {}
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    for comp in result.get("competitors_mentioned", []) or []:
        if not isinstance(comp, dict):
            continue
        merged = dict(comp)
        name = str(comp.get("name") or "").strip()
        if _is_generic_competitor_name(name):
            continue
        comp_blob = " ".join(
            [name]
            + _normalize_text_list(comp.get("features"))
            + [str(comp.get("reason_detail") or "")]
        ).lower()
        named_context = _has_named_competitor_context(name, source_row)
        switch_patterns = (
            f"switched to {name.lower()}",
            f"moved to {name.lower()}",
            f"replaced with {name.lower()}",
            f"migrating to {name.lower()}",
        )
        reverse_patterns = (
            f"moved from {name.lower()}",
            f"switched from {name.lower()}",
        )
        evaluation_patterns = (
            f"evaluating {name.lower()}",
            f"looking at {name.lower()}",
            f"considering {name.lower()}",
            f"shortlist {name.lower()}",
            f"poc with {name.lower()}",
        )
        if any(pattern in review_blob for pattern in reverse_patterns):
            evidence_type = "reverse_flow"
        elif any(pattern in review_blob for pattern in switch_patterns):
            evidence_type = "explicit_switch"
        elif any(pattern in review_blob for pattern in evaluation_patterns) or churn.get("actively_evaluating"):
            evidence_type = "active_evaluation"
        elif merged.get("reason_detail") or merged.get("features"):
            evidence_type = "implied_preference"
        elif named_context:
            evidence_type = "implied_preference"
        else:
            evidence_type = "neutral_mention"
        confidence = "low"
        if evidence_type == "explicit_switch":
            confidence = "high" if churn.get("migration_in_progress") or churn.get("renewal_timing") else "medium"
        elif evidence_type == "active_evaluation":
            confidence = "medium" if merged.get("reason_detail") else "low"
        elif evidence_type == "implied_preference" and merged.get("reason_detail"):
            confidence = "medium"
        merged["evidence_type"] = evidence_type
        merged["displacement_confidence"] = confidence
        merged["reason_category"] = _primary_reason_category(
            str(merged.get("reason_detail") or ""),
            comp_blob,
        )
        if (
            merged["evidence_type"] == "neutral_mention"
            and merged["displacement_confidence"] == "low"
            and not str(merged.get("reason_detail") or "").strip()
            and not str(merged.get("reason_category") or "").strip()
            and not _normalize_text_list(merged.get("features"))
            and not named_context
        ):
            continue
        comps.append(merged)
    return comps


_TIMELINE_IMMEDIATE_PATTERNS = ("asap", "immediately", "right away", "this week", "today", "urgent")
_TIMELINE_QUARTER_PATTERNS = ("next quarter", "this quarter", "q1", "q2", "q3", "q4", "30 days", "60 days", "90 days")
_TIMELINE_YEAR_PATTERNS = ("this year", "next year", "12 months", "end of year", "2026", "2027")
_TIMELINE_DECISION_PATTERNS = (
    "decide", "decision", "renewal", "contract", "evaluate", "evaluation",
    "considering", "switch", "switching", "migration", "migrate",
    "deadline", "cutover", "go live", "go-live",
)
_TIMELINE_EXPLICIT_ANCHOR_PHRASES = (
    "end of quarter", "quarter end", "end of month", "month end",
    "end of year", "next quarter", "this quarter", "next month", "this month",
    "this week", "next week", "a few weeks", "few weeks", "a few days", "few days",
    "30 days", "60 days", "90 days", "12 months", "next year", "this year",
    "asap", "immediately", "right away", "today", "tomorrow",
)
_TIMELINE_RELATIVE_ANCHOR_RE = re.compile(
    r"\b(?:\d+\s*-\s*\d+|\d+|one|two|three|four|five|six|seven|eight|nine|ten|a few|few)"
    r"(?:\s+to\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))?"
    r"\s+(?:business\s+days?|days?|weeks?|months?)\b",
    re.IGNORECASE,
)
_TIMELINE_MONTH_DAY_RE = re.compile(
    r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|"
    r"aug|august|sep|sept|september|oct|october|nov|november|dec|december)\.?"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
_TIMELINE_SLASH_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_TIMELINE_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TIMELINE_CONTRACT_END_PATTERNS = (
    "contract end", "contract ends", "contract expires", "expiration date",
    "expiry date", "renewal date", "renewal window", "term ends", "term expires",
    "auto renew", "auto-renew", "automatic renewal", "at renewal", "upon renewal",
    "final month of", "current contract",
)
_TIMELINE_DECISION_DEADLINE_PATTERNS = (
    "notice", "notice period", "before renewal", "before the contract ends",
    "before the contract expires", "deadline", "decide", "decision", "evaluating",
    "evaluation", "considering", "switch", "switching", "migrate", "migration",
    "cutover", "go live", "go-live", "cancel by",
)
_TIMELINE_CONTRACT_EVENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:at|upon)\s+(?:the\s+)?renewal\b", re.I),
    re.compile(r"\b(?:auto[- ]?renew(?:al)?|annual renewal|next renewal|renewal date|renewal window)\b", re.I),
    re.compile(r"\bfinal month of (?:my|our|the) current contract\b", re.I),
    re.compile(r"\b(?:current|existing)\s+contract\b", re.I),
)
_TIMELINE_AMBIGUOUS_VENDOR_TOKENS = {"copper", "close"}
_TIMELINE_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT_PATTERNS = (
    "crm", "sales", "pipeline", "lead", "leads", "deal", "deals", "account",
    "contact", "contacts", "prospect", "prospects", "software", "saas",
)
_BUDGET_CURRENCY_TOKEN_RE = re.compile(
    r"(?P<raw>(?:\$|usd\s*)\s?\d[\d,]*(?:\.\d+)?\s*(?:[km])?)",
    re.IGNORECASE,
)
_BUDGET_ANY_AMOUNT_TOKEN_RE = re.compile(
    r"(?:\$|usd\s*|\u20ac|eur\s*|\u00a3|gbp\s*)\s?\d[\d,]*(?:\.\d+)?\s*(?:[km])?",
    re.IGNORECASE,
)
_BUDGET_ANNUAL_AMOUNT_RE = re.compile(
    r"(?P<raw>(?:\$|usd\s*)\s?\d[\d,]*(?:\.\d+)?\s*(?:[km])?)"
    r"\s*(?P<period>(?:/\s*|\bper\b\s*|\ba\b\s*)?(?:yr|year)\b|annually\b|annual\b|yearly\b)",
    re.IGNORECASE,
)
_BUDGET_PRICE_PER_SEAT_RE = re.compile(
    r"(?P<raw>(?:\$|usd\s*)\s?\d[\d,]*(?:\.\d+)?\s*(?:[km])?)"
    r"\s*(?:/|\bper\b)\s*(?:seat|user|license|licence)\b"
    r"(?:\s*(?:/|\bper\b)\s*(?:monthly|month|mo|annually|annual|year|yr))?",
    re.IGNORECASE,
)
_BUDGET_SEAT_COUNT_RE = re.compile(
    r"\b(?P<count>\d[\d,]{0,6})\s+(?P<unit>seats?|users?|licenses?|licences?)\b",
    re.IGNORECASE,
)
_BUDGET_PRICE_INCREASE_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?%\s+(?:price\s+)?(?:increase|higher|more|jump|hike)"
    r"|(?:price|pricing|renewal)\s+(?:increase|jump|hike)"
    r"|(?:raised|increased)\s+(?:our\s+)?(?:price|pricing|renewal|invoice))\b",
    re.IGNORECASE,
)
_BUDGET_PRICE_INCREASE_DETAIL_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?%\s+(?:price\s+)?(?:increase|higher|more|jump|hike)"
    r"|(?:price|pricing|renewal)\s+(?:increase|jump|hike)[^.!,;]{0,80}"
    r"|(?:raised|increased)[^.!,;]{0,80})",
    re.IGNORECASE,
)
_BUDGET_COMMERCIAL_CONTEXT_PATTERNS = (
    "pricing", "price", "priced", "cost", "costs", "costly", "expensive",
    "budget", "billing", "invoice", "overcharg", "renewal", "quote", "quoted",
    "contract", "subscription", "license", "licence", "plan", "seat", "user",
)
_BUDGET_ANNUAL_CONTEXT_PATTERNS = (
    "renewal", "quote", "quoted", "contract", "subscription", "license",
    "licence", "annual", "annually", "yearly", "per year", "/year", "/yr",
)
_BUDGET_MONTHLY_PERIOD_PATTERNS = (
    "monthly", "per month", "/month", "/mo", "a month",
)
_BUDGET_ANNUAL_PERIOD_PATTERNS = (
    "annual", "annually", "yearly", "per year", "/year", "/yr", "a year", "a yr",
)
_BUDGET_PER_UNIT_PATTERNS = (
    "per seat", "/seat", "per user", "/user", "per license", "/license",
    "per licence", "/licence", "per agent", "/agent", "per person", "/person",
    "per employee", "/employee", "per endpoint", "/endpoint", "per device", "/device",
    "per member", "/member", "per contact", "/contact",
)
_BUDGET_NOISE_PATTERNS = (
    "salary", "salaries", "compensation", "bonus", "payroll", "hourly",
    "per hour", "an hour", "wage", "wages", "job offer", "interview", "intern",
    "income", "revenue", "profit", "arr", "mrr", "valuation", "mortgage",
    "rent", "tuition", "commission",
)


def _normalize_timeline_anchor(anchor: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(anchor or "")).strip(" \t\r\n'\".,;:()[]{}")
    return text.lower() if text else None


def _extract_concrete_timeline_anchor(text: Any) -> str | None:
    raw_text = str(text or "")
    if not raw_text.strip():
        return None
    for pattern in (_TIMELINE_MONTH_DAY_RE, _TIMELINE_SLASH_DATE_RE, _TIMELINE_ISO_DATE_RE):
        match = pattern.search(raw_text)
        if match:
            return _normalize_timeline_anchor(match.group(0))
    lowered = raw_text.lower()
    for phrase in _TIMELINE_EXPLICIT_ANCHOR_PHRASES:
        index = lowered.find(phrase)
        if index >= 0:
            return _normalize_timeline_anchor(raw_text[index:index + len(phrase)])
    match = _TIMELINE_RELATIVE_ANCHOR_RE.search(raw_text)
    if match:
        return _normalize_timeline_anchor(match.group(0))
    return None


def _extract_contract_end_event_anchor(text: Any) -> str | None:
    raw_text = str(text or "")
    if not raw_text.strip():
        return None
    for pattern in _TIMELINE_CONTRACT_EVENT_PATTERNS:
        match = pattern.search(raw_text)
        if not match:
            continue
        anchor = _normalize_timeline_anchor(match.group(0))
        if not anchor:
            continue
        if "renew" in anchor:
            return "renewal"
        if "current contract" in anchor:
            return "current contract end"
        return anchor
    return None


def _has_timeline_commercial_signal(
    result: dict,
    source_row: dict[str, Any] | None = None,
) -> bool:
    churn = result.get("churn_signals") or {}
    review_norm = ""
    review_blob = ""
    source = ""
    if source_row is not None:
        review_blob = " ".join(
            str(source_row.get(field) or "")
            for field in ("summary", "review_text", "pros", "cons")
        )
        source = str(source_row.get("source") or "").strip().lower()
        review_norm = _normalize_compare_text(review_blob)

    structured_churn = any((
        bool(churn.get("intent_to_leave")),
        bool(churn.get("actively_evaluating")),
        bool(churn.get("migration_in_progress")),
        bool(churn.get("contract_renewal_mentioned")),
    ))
    strong_signal = any((
        structured_churn,
        bool(result.get("competitors_mentioned")),
        bool(result.get("pricing_phrases")),
        _has_strong_commercial_context(review_norm),
    ))
    soft_signal = any((
        bool(result.get("specific_complaints")),
        bool(result.get("event_mentions")),
    ))
    if source_row is not None:
        noisy_sources = _normalized_low_fidelity_noisy_sources()
        if source in noisy_sources:
            vendor_norm = _normalize_compare_text(source_row.get("vendor_name"))
            product_norm = _normalize_compare_text(source_row.get("product_name"))
            product_hit = (
                bool(source_row.get("product_name"))
                and product_norm != vendor_norm
                and _text_mentions_name(review_norm, source_row.get("product_name"))
            )
            vendor_hit = (
                bool(source_row.get("vendor_name"))
                and _text_mentions_name(review_norm, source_row.get("vendor_name"))
            )
            if vendor_norm in _TIMELINE_AMBIGUOUS_VENDOR_TOKENS and vendor_hit:
                vendor_hit = _contains_any(review_blob, _TIMELINE_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT_PATTERNS)
            vendor_reference = product_hit or vendor_hit
            if not vendor_reference and not structured_churn:
                return False

    return any((
        strong_signal,
        soft_signal and _has_commercial_context(review_norm),
    ))


def _derive_concrete_timeline_fields(
    result: dict,
    source_row: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    churn = result.get("churn_signals") or {}
    timeline = result.get("timeline") or {}
    contract_end = _normalize_timeline_anchor(timeline.get("contract_end"))
    evaluation_deadline = _normalize_timeline_anchor(timeline.get("evaluation_deadline"))
    if contract_end and evaluation_deadline:
        return contract_end, evaluation_deadline

    candidates: list[tuple[str, str]] = []
    for event in result.get("event_mentions") or []:
        if not isinstance(event, dict):
            continue
        anchor = _extract_concrete_timeline_anchor(event.get("timeframe"))
        if not anchor:
            continue
        context = " ".join(
            str(event.get(key) or "")
            for key in ("event", "detail", "timeframe")
        )
        candidates.append((anchor, context.lower()))

    if source_row is not None and _has_timeline_commercial_signal(result, source_row):
        review_blob = " ".join(
            str(source_row.get(field) or "")
            for field in ("summary", "review_text", "pros", "cons")
        )
        anchor = _extract_concrete_timeline_anchor(review_blob)
        if anchor:
            candidates.append((anchor, review_blob.lower()))

    for anchor, context in candidates:
        if not evaluation_deadline and (
            _contains_any(context, _TIMELINE_DECISION_DEADLINE_PATTERNS)
            or " before " in context
        ):
            evaluation_deadline = anchor
            continue
        if not contract_end and (
            _contains_any(context, _TIMELINE_CONTRACT_END_PATTERNS)
            or bool(churn.get("contract_renewal_mentioned"))
        ):
            contract_end = anchor
            continue
        if not evaluation_deadline and (
            bool(churn.get("actively_evaluating"))
            or bool(churn.get("migration_in_progress"))
            or bool(churn.get("intent_to_leave"))
        ):
            evaluation_deadline = anchor
            continue

    if not contract_end and source_row is not None and _has_timeline_commercial_signal(result, source_row):
        review_blob = " ".join(
            str(source_row.get(field) or "")
            for field in ("summary", "review_text", "pros", "cons")
        )
        contract_event_anchor = _extract_contract_end_event_anchor(review_blob)
        if contract_event_anchor:
            contract_end = contract_event_anchor

    return contract_end, evaluation_deadline


def _derive_decision_timeline(
    result: dict,
    source_row: dict[str, Any] | None = None,
) -> str:
    churn = result.get("churn_signals") or {}
    timeline = result.get("timeline") or {}
    event_mentions = result.get("event_mentions") or []
    parts = [
        str(churn.get("renewal_timing") or ""),
        str(timeline.get("contract_end") or ""),
        str(timeline.get("evaluation_deadline") or ""),
    ]
    for event in event_mentions:
        if isinstance(event, dict):
            parts.append(str(event.get("timeframe") or ""))
    text = " ".join(parts).lower()
    if _contains_any(text, _TIMELINE_IMMEDIATE_PATTERNS):
        return "immediate"
    if _contains_any(text, _TIMELINE_QUARTER_PATTERNS):
        return "within_quarter"
    if _contains_any(text, _TIMELINE_YEAR_PATTERNS):
        return "within_year"

    if source_row is not None:
        review_blob = " ".join(
            str(source_row.get(field) or "")
            for field in ("summary", "review_text", "pros", "cons")
        ).lower()
        has_commercial_signal = _has_timeline_commercial_signal(result, source_row)
        if has_commercial_signal and _contains_any(review_blob, _TIMELINE_DECISION_PATTERNS):
            if _contains_any(review_blob, _TIMELINE_IMMEDIATE_PATTERNS):
                return "immediate"
            if _contains_any(review_blob, _TIMELINE_QUARTER_PATTERNS):
                return "within_quarter"
            if _contains_any(review_blob, _TIMELINE_YEAR_PATTERNS):
                return "within_year"
    return "unknown"


def _budget_match_window(text: str, match: re.Match[str], radius: int = 56) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end].lower()


def _normalize_budget_value_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.lower()
    text = re.sub(r"\busd\b\s*", "$", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\$\s+", "$", text)
    text = re.sub(r"(?<=[0-9km])a(year|yr)\b", r" a \1", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\bper\s+", "per ", text)
    text = re.sub(r"\ba\s+(year|yr)\b", r"a \1", text)
    text = text.strip()
    return text or None


def _normalize_budget_detail_text(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n'\".,;:()[]{}")
    return text or None


def _extract_budget_currency_marker(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("usd") or "$" in text:
        return "$"
    if lowered.startswith("eur") or "\u20ac" in text:
        return "\u20ac"
    if lowered.startswith("gbp") or "\u00a3" in text:
        return "\u00a3"
    return None


def _extract_single_budget_amount(value: Any) -> tuple[str | None, float | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    matches = list(_BUDGET_ANY_AMOUNT_TOKEN_RE.finditer(text))
    if len(matches) != 1:
        return None, None
    raw_amount = matches[0].group(0)
    currency = _extract_budget_currency_marker(raw_amount)
    amount = _extract_numeric_amount(raw_amount)
    if currency is None or amount is None:
        return None, None
    return currency, amount


def _extract_budget_period_multiplier(value: Any) -> int | None:
    text = str(value or "").lower()
    if not text:
        return None
    if _contains_any(text, _BUDGET_ANNUAL_PERIOD_PATTERNS):
        return 1
    if _contains_any(text, _BUDGET_MONTHLY_PERIOD_PATTERNS):
        return 12
    return None


def _format_annual_budget_amount(currency: str, amount: float) -> str | None:
    if amount <= 0 or amount > 1_000_000_000_000:
        return None
    if amount >= 1_000_000:
        scaled = amount / 1_000_000
        suffix = "m"
    elif amount >= 1_000:
        scaled = amount / 1_000
        suffix = "k"
    else:
        scaled = amount
        suffix = ""

    if abs(scaled - round(scaled)) < 1e-9:
        value_text = str(int(round(scaled)))
    elif scaled >= 100:
        value_text = f"{scaled:.0f}"
    elif scaled >= 10:
        value_text = f"{scaled:.1f}".rstrip("0").rstrip(".")
    else:
        value_text = f"{scaled:.2f}".rstrip("0").rstrip(".")
    return f"{currency}{value_text}{suffix}/year"


def _derive_annual_spend_from_unit_price(budget: dict[str, Any]) -> str | None:
    try:
        seat_count = int(budget.get("seat_count"))
    except (TypeError, ValueError):
        return None
    if not (1 <= seat_count <= 1_000_000):
        return None

    currency, unit_amount = _extract_single_budget_amount(budget.get("price_per_seat"))
    if currency is None or unit_amount is None:
        return None

    period_multiplier = _extract_budget_period_multiplier(budget.get("price_per_seat"))
    if period_multiplier is None:
        return None

    return _format_annual_budget_amount(currency, unit_amount * seat_count * period_multiplier)


def _has_budget_noise_context(text: str) -> bool:
    return _contains_any(str(text or "").lower(), _BUDGET_NOISE_PATTERNS)


def _has_budget_commercial_signal(
    result: dict,
    source_row: dict[str, Any] | None = None,
) -> bool:
    churn = result.get("churn_signals") or {}
    pricing_phrases = _normalize_text_list(result.get("pricing_phrases"))
    summary_text = str((source_row or {}).get("summary") or "").strip().lower()
    review_blob = _combined_source_text(source_row)
    review_norm = _normalize_compare_text(review_blob)
    structured_churn = any((
        bool(churn.get("intent_to_leave")),
        bool(churn.get("actively_evaluating")),
        bool(churn.get("migration_in_progress")),
        bool(churn.get("contract_renewal_mentioned")),
    ))
    if not (pricing_phrases or structured_churn or _has_commercial_context(review_norm)):
        return False
    if source_row is None:
        return True

    noisy_sources = _normalized_low_fidelity_noisy_sources()
    source = str(source_row.get("source") or "").strip().lower()
    if source not in noisy_sources:
        return True

    vendor_norm = _normalize_compare_text(source_row.get("vendor_name"))
    product_norm = _normalize_compare_text(source_row.get("product_name"))
    product_hit = (
        bool(source_row.get("product_name"))
        and product_norm != vendor_norm
        and _text_mentions_name(review_norm, source_row.get("product_name"))
    )
    vendor_hit = (
        bool(source_row.get("vendor_name"))
        and _text_mentions_name(review_norm, source_row.get("vendor_name"))
    )
    if vendor_norm in _TIMELINE_AMBIGUOUS_VENDOR_TOKENS and vendor_hit:
        vendor_hit = _contains_any(review_blob, _TIMELINE_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT_PATTERNS)
    if _has_consumer_context(review_norm) and not (product_hit or vendor_hit or structured_churn):
        return False
    if _has_technical_context(summary_text, review_norm) and not structured_churn:
        return False
    return any((
        product_hit,
        vendor_hit,
        structured_churn,
        _has_strong_commercial_context(review_norm) and not _has_budget_noise_context(review_blob),
    ))


def _derive_budget_signals(result: dict, source_row: dict[str, Any]) -> dict[str, Any]:
    budget = result.get("budget_signals")
    if not isinstance(budget, dict):
        budget = {}
        result["budget_signals"] = budget

    if not _has_budget_commercial_signal(result, source_row):
        return budget

    candidates: list[str] = []
    seen_candidates: set[str] = set()
    for phrase in _normalize_text_list(result.get("pricing_phrases")):
        lowered = phrase.lower()
        if lowered not in seen_candidates:
            seen_candidates.add(lowered)
            candidates.append(phrase)
    review_blob = _combined_source_text(source_row)
    if review_blob.strip():
        candidates.append(review_blob)

    if not budget.get("price_per_seat"):
        for text in candidates:
            match = _BUDGET_PRICE_PER_SEAT_RE.search(text)
            if not match:
                continue
            window = _budget_match_window(text, match)
            if _has_budget_noise_context(window):
                continue
            normalized = _normalize_budget_value_text(match.group(0))
            if normalized:
                budget["price_per_seat"] = normalized
                break

    if not budget.get("annual_spend_estimate"):
        for text in candidates:
            match = _BUDGET_ANNUAL_AMOUNT_RE.search(text)
            if not match:
                continue
            window = _budget_match_window(text, match)
            if _has_budget_noise_context(window):
                continue
            normalized = _normalize_budget_value_text(match.group(0))
            if normalized:
                budget["annual_spend_estimate"] = normalized
                break
        if not budget.get("annual_spend_estimate"):
            for text in candidates:
                for match in _BUDGET_CURRENCY_TOKEN_RE.finditer(text):
                    window = _budget_match_window(text, match)
                    if _has_budget_noise_context(window):
                        continue
                    if _contains_any(window, _BUDGET_PER_UNIT_PATTERNS):
                        continue
                    if _contains_any(window, _BUDGET_MONTHLY_PERIOD_PATTERNS):
                        continue
                    if not _contains_any(window, _BUDGET_ANNUAL_CONTEXT_PATTERNS):
                        continue
                    normalized = _normalize_budget_value_text(match.group("raw"))
                    if normalized:
                        budget["annual_spend_estimate"] = normalized
                        break
                if budget.get("annual_spend_estimate"):
                    break

    if not budget.get("seat_count"):
        for text in candidates:
            for match in _BUDGET_SEAT_COUNT_RE.finditer(text):
                window = _budget_match_window(text, match)
                if _has_budget_noise_context(window):
                    continue
                if not _contains_any(window, _BUDGET_COMMERCIAL_CONTEXT_PATTERNS):
                    continue
                try:
                    count = int(match.group("count").replace(",", ""))
                except ValueError:
                    continue
                if 1 <= count <= 1_000_000:
                    budget["seat_count"] = count
                    break
            if budget.get("seat_count"):
                break

    if not budget.get("annual_spend_estimate"):
        derived_annual_spend = _derive_annual_spend_from_unit_price(budget)
        if derived_annual_spend:
            budget["annual_spend_estimate"] = derived_annual_spend

    if not _coerce_bool(budget.get("price_increase_mentioned")):
        for text in candidates:
            match = _BUDGET_PRICE_INCREASE_RE.search(text)
            if not match:
                continue
            window = _budget_match_window(text, match)
            if _has_budget_noise_context(window):
                continue
            if not _contains_any(window, _BUDGET_COMMERCIAL_CONTEXT_PATTERNS):
                continue
            budget["price_increase_mentioned"] = True
            if not budget.get("price_increase_detail"):
                detail_match = _BUDGET_PRICE_INCREASE_DETAIL_RE.search(text)
                detail = _normalize_budget_detail_text(
                    detail_match.group(0) if detail_match else match.group(0)
                )
                if detail:
                    budget["price_increase_detail"] = detail
            break
    elif not budget.get("price_increase_detail"):
        for text in candidates:
            detail_match = _BUDGET_PRICE_INCREASE_DETAIL_RE.search(text)
            if detail_match:
                detail = _normalize_budget_detail_text(detail_match.group(0))
                if detail:
                    budget["price_increase_detail"] = detail
                    break

    return budget


def _extract_numeric_amount(value: Any) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)(?:\s*([km]))?", str(value).lower())
    if not match:
        return None
    amount = float(match.group(1).replace(",", ""))
    suffix = match.group(2)
    if suffix == "k":
        amount *= 1_000
    elif suffix == "m":
        amount *= 1_000_000
    return amount


def _derive_contract_value_signal(result: dict) -> str:
    budget = result.get("budget_signals") or {}
    reviewer_context = result.get("reviewer_context") or {}
    spend = _extract_numeric_amount(budget.get("annual_spend_estimate"))
    seats = budget.get("seat_count")
    try:
        seat_count = int(seats) if seats is not None else 0
    except (TypeError, ValueError):
        seat_count = 0
    segment = str(reviewer_context.get("company_size_segment") or "unknown")
    if spend is not None and spend >= 100000:
        return "enterprise_high"
    if seat_count >= 500 or segment == "enterprise":
        return "enterprise_high"
    if spend is not None and spend >= 25000:
        return "enterprise_mid"
    if seat_count >= 200 or segment == "mid_market":
        return "enterprise_mid"
    if spend is not None and spend >= 5000:
        return "mid_market"
    if seat_count >= 25:
        return "mid_market"
    if segment in {"smb", "startup"}:
        return "smb"
    return "unknown"


_POST_PURCHASE_REVIEW_SOURCES = frozenset({
    "g2",
    "gartner",
    "trustradius",
    "capterra",
    "software_advice",
    "peerspot",
    "sourceforge",
    "trustpilot",
})
_POST_PURCHASE_USAGE_PATTERNS = (
    "we use",
    "we've used",
    "we have used",
    "been using",
    "using this",
    "using it",
    "in production",
    "implemented",
    "deployed",
    "rolled out",
    "adopted",
    "renewed",
    "customer since",
    "our team uses",
    "our company uses",
)


def _has_post_purchase_signal(source_row: dict[str, Any], review_blob: str) -> bool:
    source = str(source_row.get("source") or "").strip().lower()
    if source in _POST_PURCHASE_REVIEW_SOURCES:
        return True
    return _contains_any(review_blob, _POST_PURCHASE_USAGE_PATTERNS)


def _derive_buyer_authority_fields(result: dict, source_row: dict[str, Any]) -> tuple[str, bool, str]:
    reviewer_context = result.get("reviewer_context") or {}
    churn = result.get("churn_signals") or {}
    role_level = str(reviewer_context.get("role_level") or "unknown")
    decision_maker = bool(reviewer_context.get("decision_maker"))
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    if decision_maker or role_level in {"executive", "director"}:
        role_type = "economic_buyer"
    elif churn.get("actively_evaluating"):
        role_type = "evaluator"
    elif role_level == "manager":
        role_type = "champion"
    elif role_level == "ic":
        role_type = "end_user"
    else:
        role_type = "unknown"
    executive_sponsor_mentioned = _contains_any(
        review_blob,
        ("ceo", "cfo", "cto", "coo", "leadership", "executive team", "vp approved", "signed off"),
    )
    if churn.get("contract_renewal_mentioned") or churn.get("renewal_timing"):
        buying_stage = "renewal_decision"
    elif churn.get("actively_evaluating") or churn.get("migration_in_progress"):
        buying_stage = "evaluation"
    elif decision_maker and _contains_any(review_blob, ("approved", "signed off", "purchased", "bought")):
        buying_stage = "active_purchase"
    elif _has_post_purchase_signal(source_row, review_blob):
        buying_stage = "post_purchase"
    else:
        buying_stage = "unknown"
    return role_type, executive_sponsor_mentioned, buying_stage


def _derive_urgency_indicators(
    result: dict,
    source_row: dict[str, Any],
    *,
    price_complaint: bool = False,
) -> dict[str, bool]:
    churn = result.get("churn_signals") or {}
    budget = result.get("budget_signals") or {}
    timeline = result.get("timeline") or {}
    competitors = result.get("competitors_mentioned") or []
    complaints = result.get("specific_complaints") or []
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    price_text = " ".join(_normalize_text_list(result.get("pricing_phrases"))).lower()
    recommendation_text = " ".join(_normalize_text_list(result.get("recommendation_language"))).lower()
    named_alt_with_reason = any(
        isinstance(comp, dict) and comp.get("name") and comp.get("reason_detail")
        for comp in competitors
    )
    return {
        "intent_to_leave_signal": bool(churn.get("intent_to_leave")),
        "actively_evaluating_signal": bool(churn.get("actively_evaluating")),
        "migration_in_progress_signal": bool(churn.get("migration_in_progress")),
        "explicit_cancel_language": bool(churn.get("intent_to_leave")) and _contains_any(
            review_blob, ("cancel", "not renewing", "terminate", "ending our contract")
        ),
        "active_migration_language": bool(churn.get("migration_in_progress")) or "migrat" in review_blob,
        "active_evaluation_language": bool(churn.get("actively_evaluating")) or _contains_any(
            review_blob, ("evaluating", "shortlist", "poc", "comparing options")
        ),
        "completed_switch_language": _contains_any(review_blob, ("switched to", "moved to", "replaced with")),
        "comparison_shopping_language": _contains_any(review_blob, ("vs ", "alternative", "which should", "looking for options")),
        "named_alternative_with_reason": named_alt_with_reason,
        "frustration_without_alternative": bool(complaints) and not competitors,
        "price_pressure_language": bool(price_complaint) or _contains_any(
            review_blob + " " + price_text,
            (
                "price increase",
                "pricing policy",
                "too expensive",
                "costs will constantly increase",
                "forced to change provider",
                "unjustified expenses",
            ),
        ),
        "reconsideration_language": _contains_any(
            review_blob,
            (
                "reconsidering",
                "considering changing",
                "considering switching",
                "considering swtiching",
                "forced to change provider",
                "considering another tool",
            ),
        ),
        "dollar_amount_mentioned": bool(budget.get("annual_spend_estimate") or budget.get("price_per_seat")) or "$" in price_text,
        "timeline_mentioned": bool(
            churn.get("renewal_timing")
            or timeline.get("contract_end")
            or timeline.get("evaluation_deadline")
        ),
        "decision_maker_language": bool((result.get("reviewer_context") or {}).get("decision_maker")) or _contains_any(
            review_blob + " " + recommendation_text,
            ("i decided", "we approved", "signed off", "our team approved"),
        ),
    }


def _is_no_signal_result(result: dict, source_row: dict[str, Any]) -> bool:
    churn = result.get("churn_signals") or {}
    if any(bool(value) for value in churn.values()):
        return False
    if result.get("competitors_mentioned"):
        return False
    if result.get("specific_complaints") or result.get("quotable_phrases"):
        return False
    if result.get("pricing_phrases") or result.get("recommendation_language"):
        return False
    if result.get("event_mentions") or result.get("feature_gaps"):
        return False
    content_type = str(source_row.get("content_type") or "").strip().lower()
    if content_type in {"community_discussion", "comment"}:
        return True
    rating = source_row.get("rating")
    try:
        return float(rating or 0) >= 3.0
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Phrase metadata v2 schema (parallel to legacy list[str] phrase arrays).
# See atlas_brain/autonomous/tasks/_b2b_phrase_metadata.py for the reader API.
# ---------------------------------------------------------------------------

_PHRASE_METADATA_FIELDS: tuple[str, ...] = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "quotable_phrases",
    "recommendation_language",
    "positive_aspects",
)
_PHRASE_SUBJECT_VALUES: tuple[str, ...] = (
    "subject_vendor", "alternative", "self", "third_party", "unclear",
)
_PHRASE_POLARITY_VALUES: tuple[str, ...] = (
    "negative", "positive", "mixed", "unclear",
)
_PHRASE_ROLE_VALUES: tuple[str, ...] = (
    "primary_driver", "supporting_context", "passing_mention", "unclear",
)
_PHRASE_UNCLEAR = "unclear"


def _coerce_legacy_phrase_arrays(result: dict) -> None:
    """Force the six legacy phrase arrays to list[str]. In-place mutation.

    Invariant: legacy arrays stay list[str] regardless of what the LLM
    returned. If the LLM returned a dict with a 'text' field (v2-style), we
    pull the text out; if it returned something unusable, we drop the entry.
    This defends downstream scoring/witness code that assumes str elements.
    """
    for field in _PHRASE_METADATA_FIELDS:
        value = result.get(field)
        if not isinstance(value, list):
            result[field] = []
            continue
        coerced: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                text = entry.strip()
                if text:
                    coerced.append(text)
            elif isinstance(entry, dict):
                text = str(entry.get("text") or "").strip()
                if text:
                    coerced.append(text)
        result[field] = coerced


def _normalize_tag_value(value: Any, allowed: tuple[str, ...]) -> tuple[str, bool]:
    """Coerce a tag to the allowed enum. Returns (normalized, was_coerced).

    Coercion counts only unknown-value flattening ("WEIRD" -> "unclear");
    silent lowercasing or None->"unclear" are not counted as coercions.
    """
    raw = value if isinstance(value, str) else None
    if raw is None:
        return _PHRASE_UNCLEAR, False
    normalized = raw.strip().lower()
    if not normalized:
        return _PHRASE_UNCLEAR, False
    if normalized in allowed:
        return normalized, False
    return _PHRASE_UNCLEAR, True


def _normalize_phrase_metadata(
    result: dict,
    source_row: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build canonical phrase_metadata for v2 enrichments.

    Produces one metadata row per non-empty legacy-array element. Fills
    defaults for phrases the LLM didn't tag. Enforces the text-invariant:
    each row's `text` must equal the legacy array entry at (field, index);
    if it doesn't, we overwrite with the legacy value and flag verbatim=False
    so grounding downstream cannot be fooled.

    When `source_row` is provided (with `summary` + `review_text`), an LLM
    self-reported `verbatim=True` is additionally validated against the
    source text via `check_phrase_grounded`. If grounding rejects it, the
    flag is coerced to False and counted in `verbatim_grounding_failures`.
    This is Phase 1b's write-time grounding gate. Phase 1a tests pass
    `source_row=None` and skip the gate.

    Returns (canonical_list, telemetry_counters). Caller is expected to emit
    the counters via logger.info for visibility into LLM tag quality.
    """
    from ._b2b_grounding import check_phrase_grounded

    source_metadata = result.get("phrase_metadata")
    if not isinstance(source_metadata, list):
        source_metadata = []

    source_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in source_metadata:
        if not isinstance(entry, dict):
            continue
        field = str(entry.get("field") or "")
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if field and idx >= 0:
            source_by_key[(field, idx)] = entry

    counters = {
        "llm_provided_rows": 0,
        "llm_missing_rows": 0,
        "text_mismatch_rows": 0,
        "unknown_tag_coercions": 0,
        "verbatim_grounding_failures": 0,
        "verbatim_grounding_wins": 0,
    }
    canonical: list[dict[str, Any]] = []

    summary_text = (source_row or {}).get("summary") if source_row else None
    review_text = (source_row or {}).get("review_text") if source_row else None
    grounding_enabled = source_row is not None

    for field in _PHRASE_METADATA_FIELDS:
        legacy_array = result.get(field) or []
        for index, legacy_text in enumerate(legacy_array):
            llm_row = source_by_key.get((field, index))
            if llm_row is None:
                counters["llm_missing_rows"] += 1
                canonical.append({
                    "field": field,
                    "index": index,
                    "text": str(legacy_text),
                    "subject": _PHRASE_UNCLEAR,
                    "polarity": _PHRASE_UNCLEAR,
                    "role": _PHRASE_UNCLEAR,
                    "verbatim": False,
                    "category_hint": None,
                })
                continue

            counters["llm_provided_rows"] += 1
            subject, sub_coerced = _normalize_tag_value(
                llm_row.get("subject"), _PHRASE_SUBJECT_VALUES,
            )
            polarity, pol_coerced = _normalize_tag_value(
                llm_row.get("polarity"), _PHRASE_POLARITY_VALUES,
            )
            role, role_coerced = _normalize_tag_value(
                llm_row.get("role"), _PHRASE_ROLE_VALUES,
            )
            counters["unknown_tag_coercions"] += int(sub_coerced) + int(pol_coerced) + int(role_coerced)

            # Strict boolean: "false", "no", 0 must not become True after
            # bool() coercion. Only the literal Python True is accepted as
            # an LLM affirmative; everything else is False.
            verbatim = llm_row.get("verbatim") is True
            hint = llm_row.get("category_hint")
            category_hint = (
                str(hint).strip()
                if isinstance(hint, str) and str(hint).strip()
                else None
            )

            # Text invariant: metadata text must equal the legacy array entry
            llm_text = str(llm_row.get("text") or "").strip()
            if llm_text != str(legacy_text).strip():
                counters["text_mismatch_rows"] += 1
                verbatim = False

            # Phase 1b grounding gate: only retain verbatim=True when the
            # phrase actually appears (after normalization) in the source
            # text. Skipped when source_row is unavailable (Phase 1a unit
            # tests pass source_row=None).
            if verbatim and grounding_enabled:
                if check_phrase_grounded(
                    str(legacy_text),
                    summary=summary_text,
                    review_text=review_text,
                ):
                    counters["verbatim_grounding_wins"] += 1
                else:
                    counters["verbatim_grounding_failures"] += 1
                    verbatim = False

            canonical.append({
                "field": field,
                "index": index,
                "text": str(legacy_text),
                "subject": subject,
                "polarity": polarity,
                "role": role,
                "verbatim": verbatim,
                "category_hint": category_hint,
            })

    return canonical, counters


def _apply_phrase_metadata_contract(
    result: dict,
    source_row: dict[str, Any],
) -> None:
    """Normalize phrase_metadata before deterministic derived fields run.

    Phase 2 gates read phrase_metadata through the helper API, which only
    activates when the enrichment is already schema version 4. The version
    marker therefore must be set before pain, price, and witness derivation,
    not after them.
    """
    if isinstance(result.get("phrase_metadata"), list):
        canonical_metadata, metadata_counters = _normalize_phrase_metadata(
            result, source_row,
        )
        total_legacy_phrases = sum(
            len(result.get(f) or []) for f in _PHRASE_METADATA_FIELDS
        )
        llm_genuinely_tried = (
            metadata_counters["llm_provided_rows"] > 0
            or total_legacy_phrases == 0
        )
        if llm_genuinely_tried:
            result["phrase_metadata"] = canonical_metadata
            result["enrichment_schema_version"] = 4
            logger.info(
                "phrase_metadata normalized for %s: %s",
                source_row.get("id"),
                metadata_counters,
            )
        else:
            result.pop("phrase_metadata", None)
            result["enrichment_schema_version"] = 3
            logger.warning(
                "phrase_metadata present but unusable for %s "
                "(legacy_phrases=%d, provided_rows=0); falling back to v3",
                source_row.get("id"),
                total_legacy_phrases,
            )
    else:
        result.pop("phrase_metadata", None)
        result["enrichment_schema_version"] = 3


def _compute_derived_fields(result: dict, source_row: dict[str, Any]) -> dict:
    """Layer 3: compute deterministic fields from Layer 1 + Layer 2 extractions.

    Replaces 7 former LLM INFER fields with pipeline-computed values using
    the declarative Evidence Map. All downstream consumers see the same
    JSONB paths -- zero breakage.
    """
    from ...reasoning.evidence_engine import get_evidence_engine

    engine = get_evidence_engine()

    raw_meta = source_row.get("raw_metadata") or {}
    if isinstance(raw_meta, str):
        raw_meta = json.loads(raw_meta)
    source_weight = float(raw_meta.get("source_weight", 0.7))
    content_type = source_row.get("content_type") or result.get("content_classification") or "review"
    rating = float(source_row["rating"]) if source_row.get("rating") is not None else None
    rating_max = float(source_row.get("rating_max") or 5)

    # Sanitize legacy phrase arrays to list[str] before any downstream scoring
    # sees them. Defends against dict-form entries the LLM may emit under the
    # new v2 prompt; downstream code (pain scoring, witnesses) assumes strings.
    # Done BEFORE reading the locals below so pain derivation never sees dicts.
    _coerce_legacy_phrase_arrays(result)
    _apply_phrase_metadata_contract(result, source_row)

    complaints = result.get("specific_complaints", [])
    quotable = result.get("quotable_phrases", [])
    pricing_phrases = result.get("pricing_phrases", [])
    rec_lang = result.get("recommendation_language", [])
    events = result.get("event_mentions", [])
    reviewer = result.get("reviewer_context", {})

    # 0. deterministic replacements for deprecated Tier 2 classify path
    result["pain_categories"] = _derive_pain_categories(result)
    result["competitors_mentioned"] = _recover_competitor_mentions(result, source_row)
    result["competitors_mentioned"] = _derive_competitor_annotations(result, source_row)
    _derive_budget_signals(result, source_row)

    ba = result.get("buyer_authority")
    if not isinstance(ba, dict):
        ba = {}
        result["buyer_authority"] = ba
    role_type, executive_sponsor_mentioned, buying_stage = _derive_buyer_authority_fields(
        result, source_row
    )
    ba["role_type"] = role_type
    ba["executive_sponsor_mentioned"] = executive_sponsor_mentioned
    ba["buying_stage"] = buying_stage

    timeline = result.get("timeline")
    if not isinstance(timeline, dict):
        timeline = {}
        result["timeline"] = timeline
    contract_end, evaluation_deadline = _derive_concrete_timeline_fields(result, source_row)
    if contract_end and not str(timeline.get("contract_end") or "").strip():
        timeline["contract_end"] = contract_end
    if evaluation_deadline and not str(timeline.get("evaluation_deadline") or "").strip():
        timeline["evaluation_deadline"] = evaluation_deadline
    timeline["decision_timeline"] = _derive_decision_timeline(result, source_row)

    cc = result.get("contract_context")
    if not isinstance(cc, dict):
        cc = {}
        result["contract_context"] = cc
    cc["contract_value_signal"] = _derive_contract_value_signal(result)
    price_complaint = engine.derive_price_complaint(result)
    result["urgency_indicators"] = _derive_urgency_indicators(
        result,
        source_row,
        price_complaint=price_complaint,
    )

    indicators = result.get("urgency_indicators", {})
    pain_cats = result.get("pain_categories", [])

    # 1. urgency_score
    result["urgency_score"] = engine.compute_urgency(
        indicators, rating, rating_max, content_type, source_weight,
    )

    # 2. pain_category (backward compat top-level field)
    primary_pain = "overall_dissatisfaction"
    if pain_cats:
        primary_list = [p for p in pain_cats if isinstance(p, dict) and p.get("severity") == "primary"]
        if primary_list:
            primary_pain = primary_list[0].get("category", "overall_dissatisfaction")
        elif isinstance(pain_cats[0], dict):
            primary_pain = pain_cats[0].get("category", "overall_dissatisfaction")
    result["pain_category"] = engine.override_pain(
        _normalize_pain_category(primary_pain),
        _subject_vendor_phrase_texts(result, "specific_complaints"),
        _subject_vendor_phrase_texts(result, "quotable_phrases"),
        _subject_vendor_phrase_texts(result, "pricing_phrases"),
        _subject_vendor_phrase_texts(result, "feature_gaps"),
        _subject_vendor_phrase_texts(result, "recommendation_language"),
    )

    # 2b. Recommendation + sentiment are corroborating inputs to the Phase 4
    # causality gate below, so compute them before pain_confidence.
    result["would_recommend"] = engine.derive_recommend(rec_lang, rating, rating_max)

    st = result.get("sentiment_trajectory")
    if not isinstance(st, dict):
        st = {}
        result["sentiment_trajectory"] = st
    rating_norm = (rating / rating_max) if rating is not None and rating_max else None
    churn_signals_raw = result.get("churn_signals") or {}
    intent_to_leave = bool(churn_signals_raw.get("intent_to_leave")) if isinstance(churn_signals_raw, dict) else False
    would_rec = result.get("would_recommend")
    if rating_norm is not None:
        if rating_norm <= 0.4 or (rating_norm <= 0.6 and intent_to_leave):
            st["direction"] = "consistently_negative"
        elif rating_norm >= 0.8 and would_rec is True:
            st["direction"] = "stable_positive"
        elif rating_norm >= 0.7 and would_rec is not False:
            st["direction"] = "stable_positive"
        else:
            st["direction"] = "unknown"
    else:
        st["direction"] = "unknown"

    # 2c. Phase 4 (Layer 3 -- causality gate): grade the primary pain. A
    # pain category backed by a single keyword match with no churn /
    # sentiment corroboration is an unreliable classification (passing
    # mention, not a real pain), so we demote it to overall_dissatisfaction
    # and keep the original as a secondary entry for visibility.
    final_pain = _normalize_pain_category(result.get("pain_category"))
    confidence = _compute_pain_confidence(result, final_pain)
    if confidence == "none" and final_pain != "overall_dissatisfaction":
        _demote_primary_pain(result, final_pain)
        result["pain_category"] = "overall_dissatisfaction"
        # Re-grade against the new (fallback) primary so pain_confidence
        # reflects the surviving classification, not the demoted one.
        confidence = _compute_pain_confidence(result, "overall_dissatisfaction")
    result["pain_confidence"] = confidence

    # 3. sentiment_trajectory.turning_point from event_mentions
    if events and isinstance(events, list) and len(events) > 0:
        first = events[0] if isinstance(events[0], dict) else {}
        event_text = str(first.get("event", "")).strip()
        timeframe = str(first.get("timeframe", "")).strip()
        if event_text and timeframe and timeframe.lower() != "null":
            st["turning_point"] = f"{event_text} ({timeframe})"
        elif event_text:
            st["turning_point"] = event_text
        else:
            st.setdefault("turning_point", None)
    else:
        st.setdefault("turning_point", None)

    # 6. buyer_authority.has_budget_authority
    ba["has_budget_authority"] = engine.derive_budget_authority(result)

    # 7. contract_context.price_complaint + price_context
    cc["price_complaint"] = price_complaint
    cc["price_context"] = pricing_phrases[0] if pricing_phrases else None

    # 8. witness-oriented deterministic evidence primitives
    result["replacement_mode"] = derive_replacement_mode(result, source_row)
    result["operating_model_shift"] = derive_operating_model_shift(result, source_row)
    result["productivity_delta_claim"] = derive_productivity_delta_claim(source_row)
    result["org_pressure_type"] = derive_org_pressure_type(source_row)
    result["salience_flags"] = derive_salience_flags(result, source_row)
    result["evidence_spans"] = derive_evidence_spans(result, source_row)

    result["evidence_map_hash"] = engine.map_hash

    return result


def _missing_witness_primitives(result: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    if str(result.get("replacement_mode") or "").strip() not in _KNOWN_REPLACEMENT_MODES:
        missing.append("replacement_mode")
    if str(result.get("operating_model_shift") or "").strip() not in _KNOWN_OPERATING_MODEL_SHIFTS:
        missing.append("operating_model_shift")
    if str(result.get("productivity_delta_claim") or "").strip() not in _KNOWN_PRODUCTIVITY_DELTA_CLAIMS:
        missing.append("productivity_delta_claim")
    if str(result.get("org_pressure_type") or "").strip() not in _KNOWN_ORG_PRESSURE_TYPES:
        missing.append("org_pressure_type")

    salience_flags = result.get("salience_flags")
    if not isinstance(salience_flags, list):
        missing.append("salience_flags")

    evidence_spans = result.get("evidence_spans")
    if not isinstance(evidence_spans, list):
        missing.append("evidence_spans")

    if not str(result.get("evidence_map_hash") or "").strip():
        missing.append("evidence_map_hash")

    return missing


def _schema_version(result: dict[str, Any]) -> int:
    try:
        return int(result.get("enrichment_schema_version") or 0)
    except (TypeError, ValueError):
        return 0


def _finalize_enrichment_for_persist(
    result: dict[str, Any],
    source_row: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(result, dict):
        return None, "invalid_payload"

    payload = json.loads(json.dumps(result))
    try:
        payload = _compute_derived_fields(payload, source_row)
    except Exception:
        logger.warning(
            "Evidence engine compute failed while finalizing enrichment for %s",
            source_row.get("id"),
            exc_info=True,
        )
        return None, "compute_failed"

    if not _validate_enrichment(payload, source_row):
        return None, "validation_failed"

    return payload, None


def _trusted_reviewer_company_name(source_row: dict[str, Any] | None) -> str | None:
    """Return a safe reviewer company candidate from trusted raw fields."""
    row = source_row if isinstance(source_row, dict) else {}
    company = str(row.get("reviewer_company") or "").strip()
    if not company:
        return None
    company_norm = normalize_company_name(company) or company.lower()
    vendor_norm = normalize_company_name(str(row.get("vendor_name") or "")) or ""
    if vendor_norm and company_norm == vendor_norm:
        return None
    return company


async def _notify_high_urgency(
    vendor_name: str,
    reviewer_company: str,
    urgency: float,
    pain_category: str,
    intent_to_leave: bool,
) -> None:
    """Send ntfy push when a newly enriched review exceeds the urgency threshold."""
    if not settings.alerts.ntfy_enabled:
        return

    import httpx

    url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    company_part = f" at {reviewer_company}" if reviewer_company else ""
    intent_part = " | Intent to leave" if intent_to_leave else ""
    pain_part = f" | Pain: {pain_category}" if pain_category else ""

    message = (
        f"Urgency {urgency:.0f}/10{company_part}\n"
        f"Vendor: {vendor_name}{pain_part}{intent_part}"
    )

    headers: dict[str, str] = {
        "Title": f"High-Urgency Signal: {vendor_name}",
        "Priority": "high",
        "Tags": "rotating_light,b2b,churn",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("ntfy high-urgency alert sent for %s (urgency=%s)", vendor_name, urgency)
    except Exception as exc:
        logger.warning("ntfy high-urgency alert failed for %s: %s", vendor_name, exc)


async def enrich_batch(batch_id: str) -> dict[str, Any]:
    """Enrich all pending reviews from a specific import batch immediately.

    Called inline after scrape insertion so reviews are enriched on arrival
    rather than waiting for the scheduler.
    """
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"skipped": "B2B churn pipeline disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"skipped": "DB not ready"}

    max_attempts = cfg.enrichment_max_attempts

    rows = await pool.fetch(
        """
        WITH batch AS (
            SELECT id
            FROM b2b_reviews
            WHERE import_batch_id = $1
              AND enrichment_status = 'pending'
              AND enrichment_attempts < $2
            FOR UPDATE SKIP LOCKED
        )
        UPDATE b2b_reviews r
        SET enrichment_status = 'enriching'
        FROM batch
        WHERE r.id = batch.id
        RETURNING r.id, r.vendor_name, r.product_name, r.product_category,
                  r.source, r.raw_metadata,
                  r.rating, r.rating_max, r.summary, r.review_text, r.pros, r.cons,
                  r.reviewer_title, r.reviewer_company, r.company_size_raw,
                  r.reviewer_industry, r.enrichment_attempts, r.content_type
        """,
        batch_id,
        max_attempts,
    )

    if not rows:
        return {"total": 0, "enriched": 0, "failed": 0}

    return await _enrich_rows(rows, cfg, pool)


def _coerce_int_override(
    raw_value: Any,
    default_value: int,
    *,
    min_value: int,
    max_value: int,
) -> int:
    """Return clamped integer override value, or default on parse failure."""
    default_coerced = _coerce_int_value(default_value, min_value)
    coerced = _coerce_int_value(raw_value, default_coerced)
    return max(min_value, min(max_value, coerced))


def _empty_exact_cache_usage() -> dict[str, int]:
    return {
        "exact_cache_hits": 0,
        "tier1_exact_cache_hits": 0,
        "tier2_exact_cache_hits": 0,
        "generated": 0,
        "tier1_generated_calls": 0,
        "tier2_generated_calls": 0,
        "witness_rows": 0,
        "witness_count": 0,
        "secondary_write_hits": 0,
    }


def _accumulate_exact_cache_usage(
    totals: dict[str, int],
    usage: dict[str, Any] | None,
) -> None:
    if not usage:
        return
    for key in (
        "exact_cache_hits",
        "tier1_exact_cache_hits",
        "tier2_exact_cache_hits",
        "generated",
        "tier1_generated_calls",
        "tier2_generated_calls",
        "witness_rows",
        "witness_count",
        "secondary_write_hits",
    ):
        totals[key] = int(totals.get(key, 0) or 0) + int(usage.get(key, 0) or 0)


def _witness_metrics(result: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(result, dict):
        return 0, 0
    spans = result.get("evidence_spans")
    if not isinstance(spans, list):
        return 0, 0
    witness_count = 0
    for span in spans:
        if not isinstance(span, dict):
            continue
        if not str(span.get("text") or "").strip():
            continue
        witness_count += 1
    return (1 if witness_count > 0 else 0), witness_count


def _row_usage_result(status: Any, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = {"status": status}
    usage_dict = usage or {}
    for key in _empty_exact_cache_usage():
        normalized[key] = int(usage_dict.get(key, 0) or 0)
    return normalized


async def _defer_batch_row(
    pool,
    row: dict[str, Any],
    *,
    tier: str,
    usage: dict[str, Any] | None = None,
    custom_id: str | None = None,
) -> dict[str, Any]:
    await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_status = 'pending'
        WHERE id = $1
        """,
        row["id"],
    )
    logger.info(
        "Deferring B2B enrichment %s for %s; reset row to pending while existing batch artifact %s remains pending",
        tier,
        row["id"],
        custom_id or "unknown",
    )
    return _row_usage_result("deferred", usage)


async def _persist_enrichment_result(
    pool,
    row: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    model_id: str,
    max_attempts: int,
    run_id: str | None,
    cache_usage: dict[str, int],
) -> bool | str:
    review_id = row["id"]
    cfg = settings.b2b_churn

    if result:
        result, finalize_error = _finalize_enrichment_for_persist(result, row)
        if finalize_error == "compute_failed":
            logger.warning(
                "Evidence engine compute failed for %s -- quarantining to prevent model-dependent output",
                review_id, exc_info=True,
            )
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment_status = 'quarantined',
                    enrichment_attempts = enrichment_attempts + 1,
                    low_fidelity = true,
                    low_fidelity_reasons = $2::jsonb
                WHERE id = $1
                """,
                review_id,
                json.dumps(["evidence_engine_compute_failure"]),
            )
            from ..visibility import record_quarantine

            await record_quarantine(
                pool,
                review_id=str(review_id),
                vendor_name=row.get("vendor_name"),
                source=row.get("source"),
                reason_code="evidence_engine_compute_failure",
                severity="error",
                actionable=True,
                summary=f"Evidence engine failed for {row.get('vendor_name')} review",
                run_id=run_id,
            )
            return "quarantined"
        if finalize_error == "validation_failed":
            logger.warning(
                "Enrichment validation failed for %s -- quarantining",
                review_id,
            )
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment_status = 'quarantined',
                    enrichment_attempts = enrichment_attempts + 1,
                    low_fidelity = true,
                    low_fidelity_reasons = $2::jsonb
                WHERE id = $1
                """,
                review_id,
                json.dumps(["enrichment_validation_failed"]),
            )
            from ..visibility import record_quarantine

            await record_quarantine(
                pool,
                review_id=str(review_id),
                vendor_name=row.get("vendor_name"),
                source=row.get("source"),
                reason_code="enrichment_validation_failed",
                severity="warning",
                actionable=True,
                summary=f"Validation failed for {row.get('vendor_name')} review",
                run_id=run_id,
            )
            return "quarantined"

    if result:
        st = result.get("sentiment_trajectory") or {}
        st_direction = st.get("direction") if isinstance(st, dict) else None
        st_tenure = st.get("tenure") if isinstance(st, dict) else None
        st_turning = st.get("turning_point") if isinstance(st, dict) else None
        witness_rows, witness_count = _witness_metrics(result)
        cache_usage["witness_rows"] += witness_rows
        cache_usage["witness_count"] += witness_count
        low_fidelity_reasons = (
            _detect_low_fidelity_reasons(row, result)
            if cfg.enrichment_low_fidelity_enabled
            else []
        )
        detected_at = datetime.now(timezone.utc)
        if not low_fidelity_reasons and _is_no_signal_result(result, row):
            target_status = "no_signal"
        else:
            target_status = "quarantined" if low_fidelity_reasons else "enriched"

        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment = $1,
                enrichment_status = $8,
                enrichment_attempts = enrichment_attempts + 1,
                enriched_at = $2,
                enrichment_model = $4,
                sentiment_direction = $5,
                sentiment_tenure = $6,
                sentiment_turning_point = $7,
                low_fidelity = $9,
                low_fidelity_reasons = $10::jsonb,
                low_fidelity_detected_at = $11
            WHERE id = $3
            """,
            json.dumps(result),
            detected_at,
            review_id,
            model_id,
            st_direction,
            st_tenure,
            st_turning if st_turning and st_turning != "null" else None,
            target_status,
            bool(low_fidelity_reasons),
            json.dumps(low_fidelity_reasons),
            detected_at if low_fidelity_reasons else None,
        )

        if low_fidelity_reasons:
            from ..visibility import record_quarantine

            await record_quarantine(
                pool,
                review_id=str(review_id),
                vendor_name=row.get("vendor_name"),
                source=row.get("source"),
                reason_code=low_fidelity_reasons[0],
                severity="warning",
                summary=f"Low-fidelity: {', '.join(low_fidelity_reasons[:3])}",
                evidence={"reasons": low_fidelity_reasons, "source": row.get("source")},
                run_id=run_id,
            )

        try:
            urgency = result.get("urgency_score", 0)
            threshold = settings.b2b_churn.high_churn_urgency_threshold
            if urgency >= threshold:
                signals = result.get("churn_signals", {})
                await _notify_high_urgency(
                    vendor_name=row["vendor_name"],
                    reviewer_company=row.get("reviewer_company") or "",
                    urgency=urgency,
                    pain_category=result.get("pain_category", ""),
                    intent_to_leave=bool(signals.get("intent_to_leave")),
                )
        except Exception:
            logger.warning("ntfy notification failed for review %s, enrichment preserved", review_id)

        try:
            _ctx = result.get("reviewer_context") or {}
            _extracted_company = (_ctx.get("company_name") or "").strip()
            if _extracted_company and not (row.get("reviewer_company") or "").strip():
                _extracted_company_norm = normalize_company_name(_extracted_company) or None
                await pool.execute(
                    "UPDATE b2b_reviews SET reviewer_company = $1, reviewer_company_norm = $2 WHERE id = $3",
                    _extracted_company,
                    _extracted_company_norm,
                    review_id,
                )
                cache_usage["secondary_write_hits"] += 1
        except Exception:
            logger.debug("Company name backfill failed for %s (non-fatal)", review_id)

        return "quarantined" if target_status == "quarantined" else True

    await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
    return False


async def _enrich_rows(
    rows,
    cfg,
    pool,
    *,
    concurrency_override: int | None = None,
    run_id: str | None = None,
    task: ScheduledTask | Any | None = None,
) -> dict[str, Any]:
    """Enrich a list of claimed rows concurrently."""
    max_attempts = _coerce_int_value(getattr(cfg, "enrichment_max_attempts", 3), 3)

    effective_concurrency = max(
        1,
        _coerce_int_value(
            concurrency_override if concurrency_override is not None else getattr(cfg, "enrichment_concurrency", 10),
            10,
        ),
    )
    sem = asyncio.Semaphore(effective_concurrency)
    enrich_single_params = inspect.signature(_enrich_single).parameters
    supports_usage_out = "usage_out" in enrich_single_params
    supports_run_id = "run_id" in enrich_single_params

    async def _bounded_enrich(row):
        async with sem:
            usage = _empty_exact_cache_usage()
            kwargs: dict[str, Any] = {}
            if supports_run_id:
                kwargs["run_id"] = run_id
            if supports_usage_out:
                status = await _enrich_single(
                    pool,
                    row,
                    max_attempts,
                    cfg.enrichment_local_only,
                    cfg.enrichment_max_tokens,
                    cfg.review_truncate_length,
                    usage_out=usage,
                    **kwargs,
                )
            else:
                status = await _enrich_single(
                    pool,
                    row,
                    max_attempts,
                    cfg.enrichment_local_only,
                    cfg.enrichment_max_tokens,
                    cfg.review_truncate_length,
                    **kwargs,
                )
            return _row_usage_result(status, usage)

    async def _run_single_rows(target_rows: list[dict[str, Any]]) -> list[dict[str, Any] | Exception]:
        if not target_rows:
            return []
        return await asyncio.gather(
            *[_bounded_enrich(row) for row in target_rows],
            return_exceptions=True,
        )

    results: list[dict[str, Any] | Exception] = []
    batch_metrics = {
        "jobs": 0,
        "submitted_items": 0,
        "cache_prefiltered_items": 0,
        "fallback_single_call_items": 0,
        "completed_items": 0,
        "failed_items": 0,
        "reused_completed_items": 0,
        "reused_pending_items": 0,
        "rows_deferred": 0,
        "tier2_single_fallback_rows": 0,
    }

    from ...services.b2b.anthropic_batch import (
        AnthropicBatchItem,
        mark_batch_fallback_result,
        run_anthropic_message_batch,
    )
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_stage_request,
        store_b2b_exact_stage_text,
    )
    from ...services.llm.anthropic import AnthropicLLM
    from ...services.protocols import Message
    from ...pipelines.llm import clean_llm_output, parse_json_response
    from ...skills import get_skill_registry
    from ._b2b_batch_utils import (
        anthropic_batch_min_items,
        anthropic_batch_requested,
        reconcile_existing_batch_artifacts,
        resolve_anthropic_batch_llm,
    )

    def _eligible_for_batch(row: dict[str, Any]) -> bool:
        if _combined_review_text_length(row) < _effective_min_review_text_length(row):
            return False
        source = str(row.get("source") or "").strip().lower()
        return source not in _effective_enrichment_skip_sources()

    def _parse_batch_text(text: str | None) -> dict[str, Any] | None:
        if not text:
            return None
        cleaned = clean_llm_output(text)
        parsed = parse_json_response(cleaned, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            return parsed
        return None

    use_openrouter = (
        not cfg.enrichment_local_only
        and bool(getattr(cfg, "enrichment_openrouter_model", ""))
        and bool(getattr(cfg, "openrouter_api_key", ""))
    )
    batch_requested = anthropic_batch_requested(
        task,
        global_default=bool(getattr(settings.b2b_churn, "anthropic_batch_enabled", False)),
        task_default=True,
        task_keys=("enrichment_anthropic_batch_enabled",),
    )
    tier1_batch_llm = (
        resolve_anthropic_batch_llm(
            current_llm=SimpleNamespace(
                name="openrouter",
                model=str(getattr(cfg, "enrichment_openrouter_model", "") or "anthropic/claude-haiku-4-5"),
            ),
            target_model_candidates=(getattr(cfg, "enrichment_openrouter_model", ""),),
        )
        if use_openrouter and batch_requested
        else None
    )
    tier2_model_id = (
        getattr(cfg, "enrichment_tier2_openrouter_model", "")
        or getattr(cfg, "enrichment_openrouter_model", "")
        or "anthropic/claude-haiku-4-5"
    )
    tier2_batch_llm = (
        resolve_anthropic_batch_llm(
            current_llm=SimpleNamespace(name="openrouter", model=str(tier2_model_id)),
            target_model_candidates=(tier2_model_id,),
        )
        if use_openrouter and batch_requested
        else None
    )

    if not isinstance(tier1_batch_llm, AnthropicLLM):
        tier1_batch_llm = None
    if not isinstance(tier2_batch_llm, AnthropicLLM):
        tier2_batch_llm = None

    tier1_batch_model_id = str(
        getattr(cfg, "enrichment_openrouter_model", "") or "anthropic/claude-haiku-4-5"
    )
    full_extraction_timeout = max(
        0.0,
        _coerce_float_value(
            getattr(cfg, "enrichment_full_extraction_timeout_seconds", 120.0),
            120.0,
        ),
    )
    tier2_client = None

    async def _persist_wrapped(
        row: dict[str, Any],
        result: dict[str, Any] | None,
        *,
        model_id: str,
        usage: dict[str, int],
    ) -> dict[str, Any]:
        status = await _persist_enrichment_result(
            pool,
            row,
            result,
            model_id=model_id,
            max_attempts=max_attempts,
            run_id=run_id,
            cache_usage=usage,
        )
        return _row_usage_result(status, usage)

    async def _run_single_tier2_fallback(
        row: dict[str, Any],
        tier1: dict[str, Any],
        usage: dict[str, int],
    ) -> dict[str, Any]:
        nonlocal tier2_client

        logger.info(
            "B2B enrichment: Tier 2 batch unavailable for %s; falling back to single-call Tier 2",
            row["id"],
        )
        batch_metrics["tier2_single_fallback_rows"] += 1

        trace_metadata = {
            "run_id": run_id,
            "vendor_name": str(row.get("vendor_name") or ""),
            "review_id": str(row["id"]),
            "source": str(row.get("source") or ""),
            "tier": "tier2",
            "workload": "single_call_fallback",
            "batch_fallback_reason": "tier2_batch_unavailable",
        }

        if use_openrouter:
            tier2, tier2_model, tier2_cache_hit = _unpack_stage_result(await asyncio.wait_for(
                _call_openrouter_tier2(
                    tier1,
                    row,
                    cfg,
                    cfg.review_truncate_length,
                    include_cache_hit=True,
                    trace_metadata=trace_metadata,
                ),
                timeout=full_extraction_timeout,
            ))
        else:
            if tier2_client is None:
                tier2_client = _get_tier2_client(cfg)
            tier2, tier2_model, tier2_cache_hit = _unpack_stage_result(await asyncio.wait_for(
                _call_vllm_tier2(
                    tier1,
                    row,
                    cfg,
                    tier2_client,
                    cfg.review_truncate_length,
                    include_cache_hit=True,
                    trace_metadata=trace_metadata,
                ),
                timeout=full_extraction_timeout,
            ))

        if tier2_cache_hit:
            usage["tier2_exact_cache_hits"] += 1
            usage["exact_cache_hits"] += 1
        elif tier2_model is not None:
            usage["tier2_generated_calls"] += 1
            usage["generated"] += 1

        if tier2 is not None:
            model_id = f"hybrid:{tier1_batch_model_id}+{tier2_model}"
        else:
            model_id = tier1_batch_model_id

        return await _persist_wrapped(
            row,
            _merge_tier1_tier2(tier1, tier2),
            model_id=model_id,
            usage=usage,
        )

    if tier1_batch_llm is None:
        results = await _run_single_rows(rows)
    else:
        tier1_skill = get_skill_registry().get("digest/b2b_churn_extraction_tier1")
        tier2_skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
        if not tier1_skill or not tier2_skill:
            results = await _run_single_rows(rows)
        else:
            direct_rows = [row for row in rows if not _eligible_for_batch(row)]
            batched_rows = [row for row in rows if _eligible_for_batch(row)]
            row_results: dict[Any, dict[str, Any] | Exception] = {}

            if direct_rows:
                direct_results = await _run_single_rows(direct_rows)
                for row, result in zip(direct_rows, direct_results):
                    row_results[row["id"]] = result

            tier1_entries: list[dict[str, Any]] = []
            for row in batched_rows:
                payload_json = json.dumps(_build_classify_payload(row, cfg.review_truncate_length))
                messages = [
                    {"role": "system", "content": tier1_skill.content},
                    {"role": "user", "content": payload_json},
                ]
                request = prepare_b2b_exact_stage_request(
                    "b2b_enrichment.tier1",
                    provider="openrouter",
                    model=str(cfg.enrichment_openrouter_model or "anthropic/claude-haiku-4-5"),
                    messages=messages,
                    max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096),
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                cached = await lookup_b2b_exact_stage_text(request)
                tier1_entries.append(
                    {
                        "row": row,
                        "payload_json": payload_json,
                        "messages": messages,
                        "request": request,
                        "cached_response_text": str(cached["response_text"] or "") if cached is not None else None,
                        "cached_usage": dict(cached.get("usage") or {}) if cached is not None else {},
                    }
                )

            existing_tier1_results = await reconcile_existing_batch_artifacts(
                pool=pool,
                llm=tier1_batch_llm,
                task_name="b2b_enrichment",
                artifact_type="review_enrichment_tier1",
                artifact_ids=[str(entry["row"]["id"]) for entry in tier1_entries],
            )
            tier1_ready_entries: list[dict[str, Any]] = []
            remaining_tier1_entries: list[dict[str, Any]] = []
            for entry in tier1_entries:
                row = entry["row"]
                existing = existing_tier1_results.get(str(row["id"]))
                if existing and existing.get("state") == "succeeded":
                    tier1 = _parse_batch_text(existing.get("response_text"))
                    if tier1 is not None:
                        tier1_ready_entries.append(
                            {
                                "row": row,
                                "tier1": tier1,
                                "cached": bool(existing.get("cached")),
                                "request": entry["request"],
                            }
                        )
                        batch_metrics["reused_completed_items"] += 1
                        continue
                if existing and existing.get("state") == "pending":
                    row_results[row["id"]] = await _defer_batch_row(
                        pool,
                        row,
                        tier="tier1",
                        custom_id=str(existing.get("custom_id") or ""),
                    )
                    batch_metrics["reused_pending_items"] += 1
                    batch_metrics["rows_deferred"] += 1
                    continue
                remaining_tier1_entries.append(entry)
            tier1_entries = remaining_tier1_entries

            if tier1_entries:
                tier1_execution = await run_anthropic_message_batch(
                    llm=tier1_batch_llm,
                    stage_id="b2b_enrichment.tier1",
                    task_name="b2b_enrichment",
                    items=[
                        AnthropicBatchItem(
                            custom_id=_enrichment_batch_custom_id("tier1", entry["row"]["id"]),
                            artifact_type="review_enrichment_tier1",
                            artifact_id=str(entry["row"]["id"]),
                            vendor_name=str(entry["row"].get("vendor_name") or "") or None,
                            messages=[
                                Message(role=str(message["role"]), content=str(message["content"]))
                                for message in entry["messages"]
                            ],
                            max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096),
                            temperature=0.0,
                            trace_span_name="task.b2b_enrichment.tier1",
                            trace_metadata={
                                "run_id": run_id,
                                "vendor_name": str(entry["row"].get("vendor_name") or ""),
                                "review_id": str(entry["row"]["id"]),
                                "source": str(entry["row"].get("source") or ""),
                                "tier": "tier1",
                                "workload": "anthropic_batch",
                            },
                            request_metadata={"review_id": str(entry["row"]["id"]), "tier": 1},
                            cached_response_text=entry["cached_response_text"],
                            cached_usage=entry["cached_usage"],
                        )
                        for entry in tier1_entries
                    ],
                    run_id=run_id,
                    min_batch_size=anthropic_batch_min_items(
                        task,
                        default=2,
                        keys=("enrichment_anthropic_batch_min_items",),
                    ),
                    batch_metadata={"stage": "tier1"},
                    pool=pool,
                )
                batch_metrics["jobs"] += 1 if tier1_execution.provider_batch_id else 0
                batch_metrics["submitted_items"] += int(tier1_execution.submitted_items or 0)
                batch_metrics["cache_prefiltered_items"] += int(tier1_execution.cache_prefiltered_items or 0)
                batch_metrics["fallback_single_call_items"] += int(tier1_execution.fallback_single_call_items or 0)
                batch_metrics["completed_items"] += int(tier1_execution.completed_items or 0)
                batch_metrics["failed_items"] += int(tier1_execution.failed_items or 0)
            else:
                tier1_execution = SimpleNamespace(results_by_custom_id={})

            tier2_entries: list[dict[str, Any]] = []
            fallback_rows: list[dict[str, Any]] = []
            per_row_batch_usage: dict[Any, dict[str, int]] = {}

            for ready_entry in tier1_ready_entries:
                row = ready_entry["row"]
                usage = _empty_exact_cache_usage()
                if ready_entry["cached"]:
                    usage["tier1_exact_cache_hits"] += 1
                    usage["exact_cache_hits"] += 1
                else:
                    usage["tier1_generated_calls"] += 1
                    usage["generated"] += 1
                per_row_batch_usage[row["id"]] = usage
                needs_tier2 = _tier1_has_extraction_gaps(ready_entry["tier1"], source=row.get("source"))
                if needs_tier2 and tier2_batch_llm is not None:
                    payload = _build_classify_payload(row, cfg.review_truncate_length)
                    payload["tier1_specific_complaints"] = ready_entry["tier1"].get("specific_complaints", [])
                    payload["tier1_quotable_phrases"] = ready_entry["tier1"].get("quotable_phrases", [])
                    payload_json = json.dumps(payload)
                    system_prompt = _tier2_system_prompt_for_content_type(
                        tier2_skill.content,
                        payload.get("content_type"),
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": payload_json},
                    ]
                    request = prepare_b2b_exact_stage_request(
                        "b2b_enrichment.tier2",
                        provider="openrouter",
                        model=str(tier2_model_id),
                        messages=messages,
                        max_tokens=cfg.enrichment_tier2_max_tokens,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    cached = await lookup_b2b_exact_stage_text(request)
                    tier2_entries.append(
                        {
                            "row": row,
                            "tier1": ready_entry["tier1"],
                            "messages": messages,
                            "request": request,
                            "cached_response_text": str(cached["response_text"] or "") if cached is not None else None,
                            "cached_usage": dict(cached.get("usage") or {}) if cached is not None else {},
                        }
                    )
                elif needs_tier2:
                    row_results[row["id"]] = await _run_single_tier2_fallback(
                        row,
                        ready_entry["tier1"],
                        usage,
                    )
                else:
                    row_results[row["id"]] = await _persist_wrapped(
                        row,
                        _merge_tier1_tier2(ready_entry["tier1"], None),
                        model_id=tier1_batch_model_id,
                        usage=usage,
                    )

            for entry in tier1_entries:
                row = entry["row"]
                usage = _empty_exact_cache_usage()
                tier1_custom_id = _enrichment_batch_custom_id("tier1", row["id"])
                outcome = tier1_execution.results_by_custom_id.get(tier1_custom_id)
                tier1 = _parse_batch_text(outcome.response_text if outcome is not None else None)
                if tier1 is None:
                    fallback_rows.append(row)
                    if outcome is not None:
                        await mark_batch_fallback_result(
                            batch_id=tier1_execution.local_batch_id,
                            custom_id=tier1_custom_id,
                            succeeded=False,
                            error_text=outcome.error_text or "tier1_batch_parse_failed",
                            pool=pool,
                        )
                    continue
                if outcome is not None and outcome.cached:
                    usage["tier1_exact_cache_hits"] += 1
                    usage["exact_cache_hits"] += 1
                else:
                    usage["tier1_generated_calls"] += 1
                    usage["generated"] += 1
                    await store_b2b_exact_stage_text(
                        entry["request"],
                        response_text=clean_llm_output(outcome.response_text or ""),
                        metadata={"tier": 1, "backend": "anthropic_batch"},
                    )

                per_row_batch_usage[row["id"]] = usage
                needs_tier2 = _tier1_has_extraction_gaps(tier1, source=row.get("source"))
                if needs_tier2 and tier2_batch_llm is not None:
                    payload = _build_classify_payload(row, cfg.review_truncate_length)
                    payload["tier1_specific_complaints"] = tier1.get("specific_complaints", [])
                    payload["tier1_quotable_phrases"] = tier1.get("quotable_phrases", [])
                    payload_json = json.dumps(payload)
                    system_prompt = _tier2_system_prompt_for_content_type(
                        tier2_skill.content,
                        payload.get("content_type"),
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": payload_json},
                    ]
                    request = prepare_b2b_exact_stage_request(
                        "b2b_enrichment.tier2",
                        provider="openrouter",
                        model=str(tier2_model_id),
                        messages=messages,
                        max_tokens=cfg.enrichment_tier2_max_tokens,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    cached = await lookup_b2b_exact_stage_text(request)
                    tier2_entries.append(
                        {
                            "row": row,
                            "tier1": tier1,
                            "messages": messages,
                            "request": request,
                            "cached_response_text": str(cached["response_text"] or "") if cached is not None else None,
                            "cached_usage": dict(cached.get("usage") or {}) if cached is not None else {},
                        }
                    )
                elif needs_tier2:
                    row_results[row["id"]] = await _run_single_tier2_fallback(row, tier1, usage)
                else:
                    row_results[row["id"]] = await _persist_wrapped(
                        row,
                        _merge_tier1_tier2(tier1, None),
                        model_id=tier1_batch_model_id,
                        usage=usage,
                    )

            if tier2_entries:
                existing_tier2_results = await reconcile_existing_batch_artifacts(
                    pool=pool,
                    llm=tier2_batch_llm,
                    task_name="b2b_enrichment",
                    artifact_type="review_enrichment_tier2",
                    artifact_ids=[str(entry["row"]["id"]) for entry in tier2_entries],
                )
                remaining_tier2_entries: list[dict[str, Any]] = []
                for entry in tier2_entries:
                    row = entry["row"]
                    existing = existing_tier2_results.get(str(row["id"]))
                    usage = per_row_batch_usage[row["id"]]
                    if existing and existing.get("state") == "succeeded":
                        tier2 = _parse_batch_text(existing.get("response_text"))
                        if tier2 is not None:
                            if existing.get("cached"):
                                usage["tier2_exact_cache_hits"] += 1
                                usage["exact_cache_hits"] += 1
                            else:
                                usage["tier2_generated_calls"] += 1
                                usage["generated"] += 1
                            row_results[row["id"]] = await _persist_wrapped(
                                row,
                                _merge_tier1_tier2(entry["tier1"], tier2),
                                model_id=f"hybrid:{tier1_batch_model_id}+{tier2_model_id}",
                                usage=usage,
                            )
                            batch_metrics["reused_completed_items"] += 1
                            continue
                    if existing and existing.get("state") == "pending":
                        row_results[row["id"]] = await _defer_batch_row(
                            pool,
                            row,
                            tier="tier2",
                            usage=usage,
                            custom_id=str(existing.get("custom_id") or ""),
                        )
                        batch_metrics["reused_pending_items"] += 1
                        batch_metrics["rows_deferred"] += 1
                        continue
                    remaining_tier2_entries.append(entry)
                tier2_entries = remaining_tier2_entries

                if tier2_entries:
                    tier2_execution = await run_anthropic_message_batch(
                        llm=tier2_batch_llm,
                        stage_id="b2b_enrichment.tier2",
                        task_name="b2b_enrichment",
                        items=[
                            AnthropicBatchItem(
                                custom_id=_enrichment_batch_custom_id("tier2", entry["row"]["id"]),
                                artifact_type="review_enrichment_tier2",
                                artifact_id=str(entry["row"]["id"]),
                                vendor_name=str(entry["row"].get("vendor_name") or "") or None,
                                messages=[
                                    Message(role=str(message["role"]), content=str(message["content"]))
                                    for message in entry["messages"]
                                ],
                                max_tokens=cfg.enrichment_tier2_max_tokens,
                                temperature=0.0,
                                trace_span_name="task.b2b_enrichment.tier2",
                                trace_metadata={
                                    "run_id": run_id,
                                    "vendor_name": str(entry["row"].get("vendor_name") or ""),
                                    "review_id": str(entry["row"]["id"]),
                                    "source": str(entry["row"].get("source") or ""),
                                    "tier": "tier2",
                                    "workload": "anthropic_batch",
                                },
                                request_metadata={"review_id": str(entry["row"]["id"]), "tier": 2},
                                cached_response_text=entry["cached_response_text"],
                                cached_usage=entry["cached_usage"],
                            )
                            for entry in tier2_entries
                        ],
                        run_id=run_id,
                        min_batch_size=anthropic_batch_min_items(
                            task,
                            default=2,
                            keys=("enrichment_anthropic_batch_min_items",),
                        ),
                        batch_metadata={"stage": "tier2"},
                        pool=pool,
                    )
                    batch_metrics["jobs"] += 1 if tier2_execution.provider_batch_id else 0
                    batch_metrics["submitted_items"] += int(tier2_execution.submitted_items or 0)
                    batch_metrics["cache_prefiltered_items"] += int(tier2_execution.cache_prefiltered_items or 0)
                    batch_metrics["fallback_single_call_items"] += int(tier2_execution.fallback_single_call_items or 0)
                    batch_metrics["completed_items"] += int(tier2_execution.completed_items or 0)
                    batch_metrics["failed_items"] += int(tier2_execution.failed_items or 0)
                else:
                    tier2_execution = SimpleNamespace(results_by_custom_id={})

                for entry in tier2_entries:
                    row = entry["row"]
                    usage = per_row_batch_usage[row["id"]]
                    tier2_custom_id = _enrichment_batch_custom_id("tier2", row["id"])
                    outcome = tier2_execution.results_by_custom_id.get(tier2_custom_id)
                    tier2 = _parse_batch_text(outcome.response_text if outcome is not None else None)
                    if tier2 is None:
                        fallback_rows.append(row)
                        if outcome is not None:
                            await mark_batch_fallback_result(
                                batch_id=tier2_execution.local_batch_id,
                                custom_id=tier2_custom_id,
                                succeeded=False,
                                error_text=outcome.error_text or "tier2_batch_parse_failed",
                                pool=pool,
                            )
                        continue
                    if outcome is not None and outcome.cached:
                        usage["tier2_exact_cache_hits"] += 1
                        usage["exact_cache_hits"] += 1
                    else:
                        usage["tier2_generated_calls"] += 1
                        usage["generated"] += 1
                        await store_b2b_exact_stage_text(
                            entry["request"],
                            response_text=clean_llm_output(outcome.response_text or ""),
                            metadata={"tier": 2, "backend": "anthropic_batch"},
                        )
                    row_results[row["id"]] = await _persist_wrapped(
                        row,
                        _merge_tier1_tier2(entry["tier1"], tier2),
                        model_id=f"hybrid:{tier1_batch_model_id}+{tier2_model_id}",
                        usage=usage,
                    )

            fallback_results = await _run_single_rows(fallback_rows)
            for row, result in zip(fallback_rows, fallback_results):
                row_results[row["id"]] = result

            results = [row_results[row["id"]] for row in rows]

    for row, result in zip(rows, results):
        if isinstance(result, Exception):
            logger.error("Unexpected enrichment error for %s: %s", row["id"], result, exc_info=result)

    cache_usage = _empty_exact_cache_usage()
    for result in results:
        if isinstance(result, Exception):
            continue
        if isinstance(result, dict):
            _accumulate_exact_cache_usage(cache_usage, result)

    batch_ids = [row["id"] for row in rows]
    status_rows = await pool.fetch(
        """
        SELECT enrichment_status, count(*) AS ct
        FROM b2b_reviews
        WHERE id = ANY($1::uuid[])
        GROUP BY enrichment_status
        """,
        batch_ids,
    )
    status_counts = {r["enrichment_status"]: int(r["ct"]) for r in status_rows}
    enriched = status_counts.get("enriched", 0)
    quarantined = status_counts.get("quarantined", 0)
    no_signal = status_counts.get("no_signal", 0)
    failed = status_counts.get("failed", 0)

    logger.info(
        "B2B enrichment: %d enriched, %d quarantined, %d no_signal, %d failed (of %d)",
        enriched, quarantined, no_signal or 0, failed, len(rows),
    )

    return {
        "total": len(rows),
        "enriched": enriched,
        "quarantined": quarantined,
        "no_signal": no_signal or 0,
        "failed": failed,
        "anthropic_batch_jobs": batch_metrics["jobs"],
        "anthropic_batch_items_submitted": batch_metrics["submitted_items"],
        "anthropic_batch_cache_prefiltered_items": batch_metrics["cache_prefiltered_items"],
        "anthropic_batch_fallback_single_call_items": batch_metrics["fallback_single_call_items"],
        "anthropic_batch_completed_items": batch_metrics["completed_items"],
        "anthropic_batch_failed_items": batch_metrics["failed_items"],
        "anthropic_batch_reused_completed_items": batch_metrics["reused_completed_items"],
        "anthropic_batch_reused_pending_items": batch_metrics["reused_pending_items"],
        "anthropic_batch_rows_deferred": batch_metrics["rows_deferred"],
        "anthropic_batch_tier2_single_fallback_rows": batch_metrics["tier2_single_fallback_rows"],
        **cache_usage,
    }


async def _recover_orphaned_enriching(pool, max_attempts: int) -> int:
    """Reset rows stranded in enriching after an interrupted prior run."""
    result = await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_attempts = enrichment_attempts + 1,
            enrichment_status = CASE
                WHEN enrichment_attempts + 1 >= $1 THEN 'failed'
                ELSE 'pending'
            END
        WHERE enrichment_status = 'enriching'
        """,
        max_attempts,
    )
    try:
        count = int(str(result).split()[-1])
    except (TypeError, ValueError, IndexError):
        count = 0
    if count:
        logger.warning("Recovered %d orphaned B2B enrichment rows", count)
    return count


async def _mark_exhausted_pending_failed(pool, max_attempts: int) -> int:
    """Mark pending rows as failed when attempts already reached max."""
    result = await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_status = 'failed'
        WHERE enrichment_status = 'pending'
          AND enrichment_attempts >= $1
        """,
        max_attempts,
    )
    try:
        count = int(str(result).split()[-1])
    except (TypeError, ValueError, IndexError):
        count = 0
    if count:
        logger.warning("Marked %d exhausted pending rows as failed", count)
    return count


async def _queue_version_upgrades(pool) -> int:
    """Reset enrichment_status to 'pending' for reviews scraped with outdated parser versions.

    Compares each review's parser_version against the currently registered
    parser version.  Reviews with older versions are re-queued for enrichment.
    Returns the number of reviews re-queued.
    """
    if not settings.b2b_churn.enrichment_auto_requeue_parser_upgrades:
        logger.info("Parser-version auto requeue disabled; skipping version-upgrade scan")
        return 0
    try:
        from ...services.scraping.parsers import get_all_parsers

        parsers = get_all_parsers()
        if not parsers:
            return 0

        total_requeued = 0
        for source_name, parser in parsers.items():
            current_version = getattr(parser, "version", None)
            if not current_version:
                continue

            # Find enriched reviews with an older parser version
            count = await pool.fetchval(
                """
                WITH updated AS (
                    UPDATE b2b_reviews
                    SET enrichment_status = 'pending',
                        enrichment_attempts = 0,
                        requeue_reason = 'parser_upgrade',
                        low_fidelity = false,
                        low_fidelity_reasons = '[]'::jsonb,
                        low_fidelity_detected_at = NULL,
                        enrichment_repair = NULL,
                        enrichment_repair_status = NULL,
                        enrichment_repair_attempts = 0,
                        enrichment_repair_model = NULL,
                        enrichment_repaired_at = NULL,
                        enrichment_repair_applied_fields = '[]'::jsonb
                    WHERE source = $1
                      AND parser_version IS NOT NULL
                      AND parser_version != $2
                      AND enrichment_status IN ('enriched', 'no_signal', 'quarantined')
                    RETURNING 1
                )
                SELECT count(*) FROM updated
                """,
                source_name,
                current_version,
            )
            if count and count > 0:
                logger.info(
                    "Re-queued %d %s reviews for re-enrichment (parser %s -> %s)",
                    count, source_name, "old", current_version,
                )
                total_requeued += count

        return total_requeued
    except Exception:
        logger.debug("Version upgrade check skipped", exc_info=True)
        return 0


async def _queue_model_upgrades(pool, cfg) -> int:
    """Reset enrichment_status to 'pending' for reviews enriched with outdated model versions.

    Compares the review's enrichment_model signature against the currently
    active model configuration.
    Returns the number of reviews re-queued.
    """
    if not cfg.enrichment_auto_requeue_model_upgrades:
        return 0

    current_sig = str(cfg.enrichment_tier1_model or "").strip()
    if not current_sig:
        return 0

    try:
        count = await pool.fetchval(
            """
            WITH updated AS (
                UPDATE b2b_reviews
                SET enrichment_status = 'pending',
                    enrichment_attempts = 0,
                    requeue_reason = 'enrichment_model_outdated',
                    low_fidelity = false,
                    low_fidelity_reasons = '[]'::jsonb,
                    low_fidelity_detected_at = NULL,
                    enrichment_repair = NULL,
                    enrichment_repair_status = NULL,
                    enrichment_repair_attempts = 0,
                    enrichment_repair_model = NULL,
                    enrichment_repaired_at = NULL,
                    enrichment_repair_applied_fields = '[]'::jsonb
                WHERE enrichment_status IN ('enriched', 'no_signal', 'quarantined')
                  AND (enrichment_model IS NULL OR enrichment_model != $1)
                RETURNING 1
            )
            SELECT count(*) FROM updated
            """,
            current_sig,
        )
        if count and count > 0:
            logger.info(
                "Re-queued %d reviews for re-enrichment (model drift -> %s)",
                count, current_sig,
            )
        return count
    except Exception:
        logger.debug("Model upgrade check skipped", exc_info=True)
        return 0


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: enrich pending B2B reviews (fallback for anything missed)."""
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B churn pipeline disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    orphaned = await _recover_orphaned_enriching(pool, cfg.enrichment_max_attempts)
    exhausted = await _mark_exhausted_pending_failed(pool, cfg.enrichment_max_attempts)

    # Auto re-process reviews scraped with outdated parser versions
    requeued_parser = await _queue_version_upgrades(pool)
    requeued_model = await _queue_model_upgrades(pool, cfg)
    requeued = requeued_parser + requeued_model

    task_metadata = task.metadata if isinstance(task.metadata, dict) else {}
    default_max_batch = min(
        _coerce_int_value(getattr(cfg, "enrichment_max_per_batch", 10), 10),
        500,
    )
    max_batch = _coerce_int_override(
        task_metadata.get("enrichment_max_per_batch"),
        default_max_batch,
        min_value=1,
        max_value=500,
    )
    max_attempts = _coerce_int_value(getattr(cfg, "enrichment_max_attempts", 3), 3)
    default_max_rounds = max(
        1,
        _coerce_int_value(getattr(cfg, "enrichment_max_rounds_per_run", 1), 1),
    )
    max_rounds = _coerce_int_override(
        task_metadata.get("enrichment_max_rounds_per_run"),
        default_max_rounds,
        min_value=1,
        max_value=100,
    )
    effective_concurrency = _coerce_int_override(
        task_metadata.get("enrichment_concurrency"),
        max(1, _coerce_int_value(getattr(cfg, "enrichment_concurrency", 10), 10)),
        min_value=1,
        max_value=100,
    )
    inter_batch_delay = max(
        0.0,
        _coerce_float_value(
            task_metadata.get(
                "enrichment_inter_batch_delay_seconds",
                getattr(cfg, "enrichment_inter_batch_delay_seconds", 2.0),
            ),
            _coerce_float_value(getattr(cfg, "enrichment_inter_batch_delay_seconds", 2.0), 2.0),
        ),
    )
    priority_sources = [
        source.strip().lower()
        for source in str(cfg.enrichment_priority_sources or "").split(",")
        if source.strip()
    ]
    run_id = _task_run_id(task)

    total_enriched = 0
    total_failed = 0
    total_no_signal = 0
    total_quarantined = 0
    cache_usage = _empty_exact_cache_usage()
    batch_metrics = {
        "anthropic_batch_jobs": 0,
        "anthropic_batch_items_submitted": 0,
        "anthropic_batch_cache_prefiltered_items": 0,
        "anthropic_batch_fallback_single_call_items": 0,
        "anthropic_batch_completed_items": 0,
        "anthropic_batch_failed_items": 0,
        "anthropic_batch_rows_deferred": 0,
        "anthropic_batch_tier2_single_fallback_rows": 0,
    }
    rounds = 0

    while rounds < max_rounds:
        rows = await pool.fetch(
            """
            WITH batch AS (
                SELECT id
                FROM b2b_reviews
                WHERE enrichment_status = 'pending'
                  AND enrichment_attempts < $1
                ORDER BY CASE
                    WHEN source = ANY($3::text[]) THEN 0
                    ELSE 1
                END,
                imported_at DESC
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            UPDATE b2b_reviews r
            SET enrichment_status = 'enriching'
            FROM batch
            WHERE r.id = batch.id
            RETURNING r.id, r.vendor_name, r.product_name, r.product_category,
                      r.source, r.raw_metadata,
                      r.rating, r.rating_max, r.summary, r.review_text, r.pros, r.cons,
                      r.reviewer_title, r.reviewer_company, r.company_size_raw,
                      r.reviewer_industry, r.enrichment_attempts, r.content_type
            """,
            max_attempts,
            max_batch,
            priority_sources,
        )

        if not rows:
            break

        result = await _enrich_rows(
            rows,
            cfg,
            pool,
            concurrency_override=effective_concurrency,
            run_id=run_id,
            task=task,
        )
        total_enriched += result.get("enriched", 0)
        batch_failed = result.get("failed", 0)
        total_failed += batch_failed
        total_no_signal += result.get("no_signal", 0)
        total_quarantined += result.get("quarantined", 0)
        _accumulate_exact_cache_usage(cache_usage, result)
        for key in batch_metrics:
            batch_metrics[key] += int(result.get(key, 0) or 0)
        rounds += 1

        # If most of the batch failed, vLLM is likely overwhelmed -- stop the loop
        if batch_failed > len(rows) * 0.5:
            logger.warning("B2B enrichment: >50%% failures in batch (%d/%d), stopping loop",
                           batch_failed, len(rows))
            break

        if inter_batch_delay > 0:
            await asyncio.sleep(inter_batch_delay)

    if rounds == 0:
        return {"_skip_synthesis": "No B2B reviews to enrich"}

    secondary_write_breakdown = {
        "company_backfills": int(cache_usage.get("secondary_write_hits", 0) or 0),
        "orphaned_requeued": int(orphaned or 0),
        "exhausted_marked_failed": int(exhausted or 0),
        "version_upgrade_requeued": int(requeued or 0),
    }
    secondary_write_hits = sum(secondary_write_breakdown.values())
    result = {
        "enriched": total_enriched,
        "quarantined": total_quarantined,
        "failed": total_failed,
        "no_signal": total_no_signal,
        **cache_usage,
        "rounds": rounds,
        "orphaned_requeued": orphaned,
        "exhausted_marked_failed": exhausted,
        "witness_rows": int(cache_usage.get("witness_rows", 0) or 0),
        "witness_count": int(cache_usage.get("witness_count", 0) or 0),
        "reviews_processed": total_enriched + total_quarantined + total_failed + total_no_signal,
        "secondary_write_hits": secondary_write_hits,
        "secondary_write_breakdown": secondary_write_breakdown,
        **batch_metrics,
        "_skip_synthesis": "B2B enrichment complete",
    }
    result["funnel_audit"] = await _fetch_review_funnel_audit(
        pool,
        int(getattr(cfg, "intelligence_window_days", 30) or 30),
    )
    if requeued:
        result["version_upgrade_requeued"] = requeued

    # Record enrichment run summary
    from ..visibility import record_attempt, emit_event
    total_processed = total_enriched + total_quarantined + total_failed + total_no_signal
    await record_attempt(
        pool, artifact_type="enrichment", artifact_id="batch",
        run_id=run_id, stage="enrichment",
        status="succeeded" if total_failed == 0 else "failed",
        score=total_enriched,
        blocker_count=total_failed,
        warning_count=total_quarantined,
        error_message=f"{total_failed} failed, {total_quarantined} quarantined" if total_failed else None,
    )
    if total_failed > 0 or total_quarantined > 0 or secondary_write_hits > 0:
        if total_failed > 0:
            reason_code = "enrichment_failures"
        elif total_quarantined > 0:
            reason_code = "enrichment_quarantines"
        else:
            reason_code = "enrichment_secondary_writes"
        await emit_event(
            pool, stage="extraction", event_type="enrichment_run_summary",
            entity_type="pipeline", entity_id="enrichment",
            summary=f"Enrichment: {total_enriched} enriched, {total_failed} failed, {total_quarantined} quarantined",
            severity="warning" if total_failed > 0 else "info",
            actionable=total_failed > 5,
            run_id=run_id,
            reason_code=reason_code,
            detail={
                "enriched": total_enriched,
                "failed": total_failed,
                "quarantined": total_quarantined,
                "no_signal": total_no_signal,
                "processed": total_processed,
                "witness_rows": int(cache_usage.get("witness_rows", 0) or 0),
                "witness_count": int(cache_usage.get("witness_count", 0) or 0),
                "exact_cache_hits": int(cache_usage.get("exact_cache_hits", 0) or 0),
                "generated": int(cache_usage.get("generated", 0) or 0),
                "secondary_write_hits": secondary_write_hits,
                "secondary_write_breakdown": secondary_write_breakdown,
            },
        )

    return result


_MIN_REVIEW_TEXT_LENGTH = 80  # Skip LLM calls for reviews shorter than this

# Verified review platforms -- every review gets full extraction (skip triage).


def _combined_review_text_length(row: dict[str, Any] | None) -> int:
    row = row or {}
    return sum(
        len(str(row.get(field) or ""))
        for field in ("review_text", "pros", "cons")
    )


def _effective_min_review_text_length(row: dict[str, Any] | None) -> int:
    source = str((row or {}).get("source") or "").strip().lower()
    if source == "capterra":
        try:
            return min(
                _MIN_REVIEW_TEXT_LENGTH,
                max(
                    20,
                    int(
                        getattr(
                            settings.b2b_scrape,
                            "capterra_min_enrichable_text_len",
                            40,
                        ) or 40
                    ),
                ),
            )
        except (TypeError, ValueError):
            return 40
    return _MIN_REVIEW_TEXT_LENGTH


async def _enrich_single(pool, row, max_attempts: int, local_only: bool,
                         max_tokens: int, truncate_length: int = 3000,
                         run_id: str | None = None,
                         usage_out: dict[str, int] | None = None) -> bool | str:
    """Enrich a single B2B review and optionally report exact-cache usage."""
    review_id = row["id"]
    cache_usage = _empty_exact_cache_usage()

    def _finish(status: bool | str) -> bool | str:
        if usage_out is not None:
            usage_out.clear()
            usage_out.update(cache_usage)
        return status

    # Skip reviews with insufficient text -- title-only scrapes can't yield 47 fields
    combined_text_len = _combined_review_text_length(row)
    if combined_text_len < _effective_min_review_text_length(row):
        await pool.execute(
            "UPDATE b2b_reviews SET enrichment_status = 'not_applicable' WHERE id = $1",
            review_id,
        )
        return _finish(False)

    source = str(row.get("source") or "").strip().lower()
    skip_sources = _effective_enrichment_skip_sources()
    if source in skip_sources:
        await pool.execute(
            """
            UPDATE b2b_reviews
            SET enrichment_status = 'not_applicable',
                low_fidelity = false,
                low_fidelity_reasons = '[]'::jsonb,
                low_fidelity_detected_at = NULL
            WHERE id = $1
            """,
            review_id,
        )
        logger.debug(
            "Skipping unsupported churn-enrichment source %s for review %s",
            source,
            review_id,
        )
        return _finish(False)

    try:
        cfg = settings.b2b_churn
        full_extraction_timeout = max(
            0.0,
            _coerce_float_value(
                getattr(cfg, "enrichment_full_extraction_timeout_seconds", 120.0),
                120.0,
            ),
        )
        payload = _build_classify_payload(row, truncate_length)
        payload_json = json.dumps(payload)
        trace_metadata = {
            "run_id": run_id,
            "vendor_name": str(row.get("vendor_name") or ""),
            "review_id": str(review_id),
            "source": str(row.get("source") or ""),
        }
        client = _get_tier1_client(cfg)

        # Tier 1: deterministic extraction (base fields)
        # Use OpenRouter if configured and not forced local-only, otherwise local vLLM
        use_openrouter = (
            not local_only
            and bool(cfg.enrichment_openrouter_model)
            and bool(cfg.openrouter_api_key)
        )
        if use_openrouter:
            tier1, tier1_model, tier1_cache_hit = _unpack_stage_result(await asyncio.wait_for(
                _call_openrouter_tier1(
                    payload_json,
                    cfg,
                    include_cache_hit=True,
                    trace_metadata=trace_metadata | {"tier": "tier1"},
                ),
                timeout=full_extraction_timeout,
            ))
        else:
            tier1, tier1_model, tier1_cache_hit = _unpack_stage_result(await asyncio.wait_for(
                _call_vllm_tier1(
                    payload_json,
                    cfg,
                    client,
                    include_cache_hit=True,
                    trace_metadata=trace_metadata | {"tier": "tier1"},
                ),
                timeout=full_extraction_timeout,
            ))
        if tier1_cache_hit:
            cache_usage["tier1_exact_cache_hits"] += 1
            cache_usage["exact_cache_hits"] += 1
        elif tier1_model is not None:
            cache_usage["tier1_generated_calls"] += 1
            cache_usage["generated"] += 1
        if tier1 is None:
            logger.debug("Tier 1 returned None for %s, deferring to next cycle", review_id)
            await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
            return _finish(False)

        # Tier 2: conditional -- only fire when tier 1 left extraction gaps
        tier2 = None
        tier2_model = None
        tier2_cache_hit = False
        if _tier1_has_extraction_gaps(tier1, source=row.get("source")):
            try:
                if use_openrouter:
                    tier2, tier2_model, tier2_cache_hit = _unpack_stage_result(await asyncio.wait_for(
                        _call_openrouter_tier2(
                            tier1,
                            row,
                            cfg,
                            truncate_length,
                            include_cache_hit=True,
                            trace_metadata=trace_metadata | {"tier": "tier2"},
                        ),
                        timeout=full_extraction_timeout,
                    ))
                else:
                    tier2_client = _get_tier2_client(cfg)
                    tier2, tier2_model, tier2_cache_hit = _unpack_stage_result(await asyncio.wait_for(
                        _call_vllm_tier2(
                            tier1,
                            row,
                            cfg,
                            tier2_client,
                            truncate_length,
                            include_cache_hit=True,
                            trace_metadata=trace_metadata | {"tier": "tier2"},
                        ),
                        timeout=full_extraction_timeout,
                    ))
            except Exception:
                logger.warning(
                    "Tier 2 enrichment failed for review %s; persisting tier 1 result only",
                    review_id,
                    exc_info=True,
                )
        if tier2_cache_hit:
            cache_usage["tier2_exact_cache_hits"] += 1
            cache_usage["exact_cache_hits"] += 1
        elif tier2_model is not None:
            cache_usage["tier2_generated_calls"] += 1
            cache_usage["generated"] += 1
        if tier2 is not None:
            model_id = f"hybrid:{tier1_model}+{tier2_model}"
        else:
            model_id = tier1_model or ""

        result = _merge_tier1_tier2(tier1, tier2)
        return _finish(
            await _persist_enrichment_result(
                pool,
                row,
                result,
                model_id=model_id,
                max_attempts=max_attempts,
                run_id=run_id,
                cache_usage=cache_usage,
            )
        )

    except Exception:
        logger.exception("Failed to enrich B2B review %s", review_id)
        try:
            # Reset from 'enriching' back to 'pending' (or 'failed' if exhausted)
            new_status = "failed" if (row["enrichment_attempts"] + 1) >= max_attempts else "pending"
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment_attempts = enrichment_attempts + 1,
                    enrichment_status = $1
                WHERE id = $2
                """,
                new_status, review_id,
            )
        except Exception:
            pass
        return _finish(False)


def _smart_truncate(text: str, max_len: int = 3000) -> str:
    """Truncate preserving both beginning and end of review text.

    Churn signals often appear at the end ("I'm switching to X next quarter"),
    so naive head-only truncation loses them.
    """
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 15
    return text[:half] + "\n[...truncated...]\n" + text[-half:]


def _build_classify_payload(row, truncate_length: int = 3000) -> dict[str, Any]:
    """Build the JSON payload for the churn extraction skill."""
    review_text = _smart_truncate(row["review_text"] or "", max_len=truncate_length)

    raw_meta = row.get("raw_metadata") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            raw_meta = {}

    payload: dict[str, Any] = {
        "vendor_name": row["vendor_name"],
        "product_name": row["product_name"] or "",
        "product_category": row["product_category"] or "",
        "source_name": row.get("source") or "",
        "source_weight": raw_meta.get("source_weight", 0.7),
        "source_type": raw_meta.get("source_type", "unknown"),
        "content_type": row.get("content_type") or "review",
        "rating": float(row["rating"]) if row["rating"] is not None else None,
        "rating_max": int(row["rating_max"]),
        "summary": row["summary"] or "",
        "review_text": review_text,
    }
    for key, value in (
        ("pros", row["pros"]),
        ("cons", row["cons"]),
        ("reviewer_title", row["reviewer_title"]),
        ("reviewer_company", row["reviewer_company"]),
        ("company_size_raw", row["company_size_raw"]),
        ("reviewer_industry", row["reviewer_industry"]),
    ):
        if isinstance(value, str) and value.strip():
            payload[key] = value
    if not payload["product_name"]:
        payload.pop("product_name", None)
    if not payload["product_category"]:
        payload.pop("product_category", None)
    if payload.get("rating") is None:
        payload.pop("rating", None)
    if not payload["summary"]:
        payload.pop("summary", None)
    if not payload["review_text"]:
        payload.pop("review_text", None)
    return payload


_LOW_FIDELITY_TOKEN_STOPWORDS = {
    "and", "for", "the", "with", "cloud", "software", "platform",
}

_LOW_FIDELITY_COMMERCIAL_MARKERS = {
    "alternative", "alternatives", "budget", "contract", "cost", "expensive",
    "migrate", "migration", "pricing", "renewal", "replace", "replaced",
    "seat", "seats", "support", "switch", "switching",
}

_LOW_FIDELITY_STRONG_COMMERCIAL_MARKERS = {
    "alternative", "alternatives", "budget", "contract", "cost", "expensive",
    "migrate", "migration", "pricing", "renewal", "replace", "replaced",
    "seat", "seats", "switch", "switching",
}

_LOW_FIDELITY_TECHNICAL_PATTERNS = (
    r"\bhow (?:can|do|to)\b",
    r"\bbest practice\b",
    r"\bsetting up\b",
    r"\banswer to question\b",
    r"\bapi token\b",
    r"\bbuild pipeline\b",
    r"\bconnect(?:ing)?\b",
    r"\bcosmos db\b",
    r"\bdocker\b",
    r"\berror\b",
    r"\bfailed\b",
    r"\bintegrat(?:e|ion)\b",
    r"\bjenkins\b",
    r"\bkey vault\b",
    r"\blogin\b",
    r"\bplugin\b",
    r"\breact frontend\b",
    r"\bssl verification failed\b",
    r"\bsubscription form\b",
    r"\bvagrant\b",
    r"\bxamarin\b",
)

_LOW_FIDELITY_CONSUMER_PATTERNS = (
    r"\b2fa\b",
    r"\bapp support\b",
    r"\bdownloaded\b",
    r"\bghosting email\b",
    r"\bgoogle play\b",
    r"\bhacked\b",
    r"\bminecraft\b",
    r"\bmy son\b",
    r"\boutlook app\b",
    r"\btaskbar\b",
    r"\bwindows 11\b",
    r"\bworkspace account\b",
)


def _normalize_compare_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalized_name_tokens(value: Any) -> list[str]:
    normalized = _normalize_compare_text(value)
    if not normalized:
        return []
    return [
        token for token in normalized.split()
        if len(token) >= 3 and token not in _LOW_FIDELITY_TOKEN_STOPWORDS
    ]


def _text_mentions_name(haystack: str, needle: Any) -> bool:
    normalized = _normalize_compare_text(needle)
    if not normalized:
        return False
    wrapped = f" {haystack} "
    if f" {normalized} " in wrapped:
        return True
    compact_haystack = haystack.replace(" ", "")
    compact_needle = normalized.replace(" ", "")
    if compact_needle and compact_needle in compact_haystack:
        return True
    return any(
        re.search(rf"\b{re.escape(token)}\b", haystack)
        for token in _normalized_name_tokens(needle)
    )


def _dedupe_reason_codes(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def _has_commercial_context(text: str) -> bool:
    return any(marker in text for marker in _LOW_FIDELITY_COMMERCIAL_MARKERS)


def _has_strong_commercial_context(text: str) -> bool:
    return any(marker in text for marker in _LOW_FIDELITY_STRONG_COMMERCIAL_MARKERS)


def _has_technical_context(summary_text: str, combined_text: str) -> bool:
    if summary_text.endswith("?"):
        return True
    haystack = f"{summary_text} {combined_text}".strip()
    return any(re.search(pattern, haystack) for pattern in _LOW_FIDELITY_TECHNICAL_PATTERNS)


def _has_consumer_context(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _LOW_FIDELITY_CONSUMER_PATTERNS)


def _normalized_low_fidelity_noisy_sources() -> set[str]:
    configured = {
        item.strip().lower()
        for item in str(settings.b2b_churn.enrichment_low_fidelity_noisy_sources or "").split(",")
        if item.strip()
    }
    default_raw = B2BChurnConfig.model_fields["enrichment_low_fidelity_noisy_sources"].default
    default_values = {
        item.strip().lower()
        for item in str(default_raw or "").split(",")
        if item.strip()
    }
    return configured | default_values


def _detect_low_fidelity_reasons(row: dict[str, Any], result: dict[str, Any]) -> list[str]:
    source = str(row.get("source") or "").strip().lower()
    noisy_sources = _normalized_low_fidelity_noisy_sources()
    if source not in noisy_sources and source != "trustpilot":
        return []

    combined_text = " ".join(
        str(row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    )
    combined_norm = _normalize_compare_text(combined_text)
    if not combined_norm:
        return ["empty_noisy_context"]

    summary_norm = _normalize_compare_text(row.get("summary"))
    vendor_hit = any(
        _text_mentions_name(combined_norm, row.get(field))
        for field in ("vendor_name", "product_name")
        if row.get(field)
    )
    competitor_hit = any(
        _text_mentions_name(combined_norm, comp.get("name"))
        for comp in (result.get("competitors_mentioned") or [])
        if isinstance(comp, dict) and comp.get("name")
    )
    summary_tokens = _normalized_name_tokens(row.get("summary"))
    urgency = float(result.get("urgency_score") or 0)
    reasons: list[str] = []
    if source in noisy_sources:
        if not vendor_hit:
            reasons.append("vendor_absent_noisy_source")
        if not vendor_hit and competitor_hit:
            reasons.append("competitor_only_context")
        if (
            source in {"twitter", "quora", "reddit", "hackernews"}
            and len(combined_norm) < 160
            and not (vendor_hit and _has_commercial_context(combined_norm))
        ):
            reasons.append("thin_social_context")
        if source == "software_advice" and len(combined_norm) < 140 and not _has_strong_commercial_context(combined_norm):
            reasons.append("thin_review_platform_context")
        if source == "quora" and summary_tokens and len(summary_tokens) <= 3 and not vendor_hit:
            reasons.append("author_style_summary")
    if source in {"stackoverflow", "github"}:
        if (
            urgency <= 5
            and _has_technical_context(summary_norm, combined_norm)
            and not _has_commercial_context(combined_norm)
        ):
            reasons.append("technical_question_context")
    if source == "trustpilot":
        if _has_consumer_context(combined_norm) and not _has_strong_commercial_context(combined_norm):
            reasons.append("consumer_support_context")
    return _dedupe_reason_codes(reasons)


_KNOWN_PAIN_CATEGORIES = {
    "pricing", "features", "reliability", "support", "integration",
    "performance", "security", "ux", "onboarding", "overall_dissatisfaction",
    "technical_debt", "contract_lock_in", "data_migration", "api_limitations",
    "outcome_gap", "admin_burden", "ai_hallucination", "integration_debt",
    "privacy",
}

_LEGACY_GENERIC_PAIN_CATEGORIES = {"other", "general_dissatisfaction"}


def _normalize_pain_category(category: Any) -> str:
    raw = str(category or "").strip().lower()
    if not raw:
        return "overall_dissatisfaction"
    if raw in _LEGACY_GENERIC_PAIN_CATEGORIES:
        return "overall_dissatisfaction"
    if raw in _KNOWN_PAIN_CATEGORIES:
        return raw
    return "overall_dissatisfaction"


def _coerce_bool(value: Any) -> bool | None:
    """Coerce a value to bool. Returns None if unrecognizable.
    None/null is treated as False (absence of signal).
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0", "null", "none"):
            return False
    return None


_CHURN_SIGNAL_BOOL_FIELDS = (
    "intent_to_leave",
    "actively_evaluating",
    "migration_in_progress",
    "support_escalation",
    "contract_renewal_mentioned",
)

_KNOWN_SEVERITY_LEVELS = {"primary", "secondary", "minor"}
_KNOWN_LOCK_IN_LEVELS = {"high", "medium", "low", "unknown"}
_KNOWN_SENTIMENT_DIRECTIONS = {"declining", "consistently_negative", "improving", "stable_positive", "unknown"}
_KNOWN_ROLE_TYPES = {"economic_buyer", "champion", "evaluator", "end_user", "unknown"}
_KNOWN_ROLE_LEVELS = {"executive", "director", "manager", "ic", "unknown"}
_KNOWN_BUYING_STAGES = {"active_purchase", "evaluation", "renewal_decision", "post_purchase", "unknown"}
_KNOWN_DECISION_TIMELINES = {"immediate", "within_quarter", "within_year", "unknown"}
_KNOWN_CONTRACT_VALUE_SIGNALS = {"enterprise_high", "enterprise_mid", "mid_market", "smb", "unknown"}
_KNOWN_REPLACEMENT_MODES = {
    "competitor_switch", "bundled_suite_consolidation", "workflow_substitution",
    "internal_tool", "none",
}
_KNOWN_OPERATING_MODEL_SHIFTS = {
    "sync_to_async", "chat_to_docs", "chat_to_ticketing", "consolidation", "none",
}
_KNOWN_PRODUCTIVITY_DELTA_CLAIMS = {"more_productive", "less_productive", "no_change", "unknown"}
_KNOWN_ORG_PRESSURE_TYPES = {
    "procurement_mandate", "standardization_mandate", "bundle_pressure",
    "budget_freeze", "none",
}

# Insider signal validation sets (migration 133)
_KNOWN_CONTENT_TYPES = {"review", "community_discussion", "comment", "insider_account"}
_KNOWN_ORG_HEALTH_LEVELS = {"high", "medium", "low", "unknown"}
_KNOWN_LEADERSHIP_QUALITIES = {"poor", "mixed", "good", "unknown"}
_KNOWN_INNOVATION_CLIMATES = {"stagnant", "declining", "healthy", "unknown"}
_KNOWN_MORALE_LEVELS = {"high", "medium", "low", "unknown"}
_KNOWN_DEPARTURE_TYPES = {"voluntary", "involuntary", "still_employed", "unknown"}

_ROLE_TYPE_ALIASES = {
    "economicbuyer": "economic_buyer",
    "decisionmaker": "economic_buyer",
    "buyer": "economic_buyer",
    "budgetowner": "economic_buyer",
    "executive": "economic_buyer",
    "director": "economic_buyer",
    "champion": "champion",
    "manager": "champion",
    "teamlead": "champion",
    "lead": "champion",
    "evaluator": "evaluator",
    "admin": "evaluator",
    "administrator": "evaluator",
    "analyst": "evaluator",
    "architect": "evaluator",
    "enduser": "end_user",
    "user": "end_user",
    "ic": "end_user",
    "individualcontributor": "end_user",
    "unknown": "unknown",
}

_ROLE_LEVEL_ALIASES = {
    "executive": "executive",
    "exec": "executive",
    "csuite": "executive",
    "cxo": "executive",
    "ceo": "executive",
    "cto": "executive",
    "cfo": "executive",
    "cio": "executive",
    "cmo": "executive",
    "coo": "executive",
    "cro": "executive",
    "president": "executive",
    "founder": "executive",
    "owner": "executive",
    "executivedirector": "executive",
    "presidentfounder": "executive",
    "ownermanagingmember": "executive",
    "ed": "executive",
    "director": "director",
    "vp": "director",
    "vicepresident": "director",
    "head": "director",
    "directeur": "director",
    "managingdirector": "director",
    "headofcustomerexperience": "director",
    "manager": "manager",
    "lead": "manager",
    "teamlead": "manager",
    "supervisor": "manager",
    "coordinator": "manager",
    "projectmanager": "manager",
    "programmanager": "manager",
    "productmanager": "manager",
    "marketingmanager": "manager",
    "digitalmarketingmanager": "manager",
    "salesmanager": "manager",
    "operationsmanager": "manager",
    "itmanager": "manager",
    "businessdevelopmentmanager": "manager",
    "clientservicemanager": "manager",
    "customersuccessmanager": "manager",
    "pmo": "manager",
    "bdm": "manager",
    "leadconsultant": "manager",
    "projectmanagement": "manager",
    "ic": "ic",
    "individualcontributor": "ic",
    "individual": "ic",
    "user": "ic",
    "product": "ic",
    "marketing": "ic",
    "digitalmarketing": "ic",
    "consultant": "ic",
    "customersupport": "ic",
    "customersuccess": "ic",
    "humanresources": "ic",
    "softwaredevelopment": "ic",
    "it": "ic",
    "devops": "ic",
    "swe": "ic",
    "fse": "ic",
    "cybersecurityanalyst": "ic",
    "chemicalengineer": "ic",
    "industrialengineer": "ic",
    "customersatisfactionandqa": "ic",
    "marketingteam": "ic",
}

_EXEC_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(vp\b|vice president|director|head of|chief|cfo|ceo|coo|cio|cto|cro|cmo|founder|owner|president|executive director|managing member)\b",
    re.I,
)
_CHAMPION_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(manager|lead|team lead|supervisor|coordinator|pmo|project management|bdm)\b",
    re.I,
)
_EVALUATOR_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(analyst|architect|engineer|developer|administrator|admin|consultant|specialist|devops|qa|customer support|customer success|human resources|marketing|product|software development|cybersecurity|it\b|swe\b|fse\b)\b",
    re.I,
)
_EXEC_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(ceo|cto|cfo|cio|coo|cmo|cro|chief|founder|owner|president)\b",
    re.I,
)
_DIRECTOR_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(vp|vice president|svp|evp|director|head of)\b",
    re.I,
)
_MANAGER_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(manager|team lead|lead|supervisor|coordinator)\b",
    re.I,
)
_IC_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(engineer|developer|administrator|admin|analyst|specialist|"
    r"consultant|marketer|designer|architect)\b",
    re.I,
)
_ECONOMIC_BUYER_TEXT_PATTERNS = (
    re.compile(
        r"\b(we|i) decided to (switch|move|migrate|leave|replace|renew|buy|adopt|go with)\b",
        re.I,
    ),
    re.compile(r"\bapproved (the )?(purchase|renewal|budget)\b", re.I),
    re.compile(r"\bsigned off on (the )?(purchase|renewal|budget|migration)\b", re.I),
    re.compile(r"\bfinal decision (was|is) to\b", re.I),
)
_CHAMPION_TEXT_PATTERNS = (
    re.compile(r"\b(i|we) recommended\b", re.I),
    re.compile(r"\bchampioned\b", re.I),
    re.compile(r"\bpushed for\b", re.I),
    re.compile(r"\badvocated for\b", re.I),
)
_EVALUATOR_TEXT_PATTERNS = (
    re.compile(r"\bevaluating alternatives\b", re.I),
    re.compile(r"\bcomparing options\b", re.I),
    re.compile(r"\bproof of concept\b", re.I),
    re.compile(r"\bpoc\b", re.I),
    re.compile(r"\bshortlist\b", re.I),
    re.compile(r"\btrialing\b", re.I),
    re.compile(r"\bpiloting\b", re.I),
    re.compile(r"\btasked with evaluating\b", re.I),
)
_END_USER_TEXT_PATTERNS = (
    re.compile(r"\bi use\b", re.I),
    re.compile(r"\bwe use\b", re.I),
    re.compile(r"\bday-to-day\b", re.I),
    re.compile(r"\bdaily use\b", re.I),
    re.compile(r"\buse it for\b", re.I),
)
_MANAGER_DECISION_TITLE_PATTERN = re.compile(
    r"\b(operations manager|it manager|project manager|program manager|product manager|marketing manager|sales manager|business development manager|client service manager|customer success manager|team lead|lead consultant|pmo|bdm|security manager|risk management)\b",
    re.I,
)
_COMMERCIAL_DECISION_TEXT_PATTERN = re.compile(
    r"\b(renewal|quote|quoted|pricing|price increase|budget|contract|procurement|vendor selection|selected|chose|approved|sign(?:ed)? off|purchase|buying committee|rfp|rfq|evaluate|evaluation|migration)\b",
    re.I,
)


def _canonical_role_type(value: Any) -> str:
    raw = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    if not raw:
        return "unknown"
    return _ROLE_TYPE_ALIASES.get(raw, "unknown")


def _normalize_role_title_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.strip().lower())


def _clean_reviewer_title_for_role_inference(value: Any) -> str:
    title = sanitize_reviewer_title(value) or ""
    if not title or len(title) > 120:
        return ""
    return title


def _canonical_role_level(value: Any) -> str:
    raw = _normalize_role_title_key(value)
    return _ROLE_LEVEL_ALIASES.get(raw, "unknown")


def _combined_source_text(source_row: dict[str, Any] | None) -> str:
    if not isinstance(source_row, dict):
        return ""
    parts = [
        str(source_row.get("summary") or ""),
        str(source_row.get("review_text") or ""),
        str(source_row.get("pros") or ""),
        str(source_row.get("cons") or ""),
    ]
    return "\n".join(part for part in parts if part).strip()


def _infer_role_level_from_text(reviewer_title: Any, source_row: dict[str, Any] | None) -> str:
    title = _clean_reviewer_title_for_role_inference(reviewer_title)
    if title:
        canonical = _canonical_role_level(title)
        if canonical != "unknown":
            return canonical
        if re.search(r"\b(cfo|ceo|coo|cio|cto|cro|cmo|chief|founder|owner|president)\b", title, re.I):
            return "executive"
        if re.search(r"\b(vp\b|vice president|svp|evp|director|head of)\b", title, re.I):
            return "director"
        if _CHAMPION_REVIEWER_TITLE_PATTERN.search(title):
            return "manager"
        if _EVALUATOR_REVIEWER_TITLE_PATTERN.search(title):
            return "ic"
    source_text = _combined_source_text(source_row)
    if not source_text:
        return "unknown"
    if _EXEC_ROLE_TEXT_PATTERN.search(source_text):
        return "executive"
    if _DIRECTOR_ROLE_TEXT_PATTERN.search(source_text):
        return "director"
    if _MANAGER_ROLE_TEXT_PATTERN.search(source_text):
        return "manager"
    if _IC_ROLE_TEXT_PATTERN.search(source_text):
        return "ic"
    return "unknown"


def _has_manager_level_decision_context(result: dict[str, Any], source_row: dict[str, Any] | None) -> bool:
    buyer_authority = _coerce_json_dict(result.get("buyer_authority"))
    if _coerce_bool(buyer_authority.get("has_budget_authority")) is True:
        return True

    budget = _coerce_json_dict(result.get("budget_signals"))
    if any(
        budget.get(field)
        for field in ("annual_spend_estimate", "price_per_seat", "price_increase_detail")
    ):
        return True
    if _coerce_bool(budget.get("price_increase_mentioned")) is True:
        return True

    timeline = _coerce_json_dict(result.get("timeline"))
    if timeline.get("contract_end") or timeline.get("evaluation_deadline"):
        return True

    churn = _coerce_json_dict(result.get("churn_signals"))
    if any(
        _coerce_bool(churn.get(field)) is True
        for field in ("actively_evaluating", "migration_in_progress", "contract_renewal_mentioned")
    ):
        return True

    return bool(_COMMERCIAL_DECISION_TEXT_PATTERN.search(_combined_source_text(source_row)))


def _infer_decision_maker(result: dict[str, Any], source_row: dict[str, Any] | None) -> bool:
    reviewer_context = _coerce_json_dict(result.get("reviewer_context"))
    buyer_authority = _coerce_json_dict(result.get("buyer_authority"))
    role_level = _canonical_role_level(reviewer_context.get("role_level"))
    if role_level in {"executive", "director"}:
        return True
    if _coerce_bool(buyer_authority.get("has_budget_authority")) is True:
        return True
    if _canonical_role_type(buyer_authority.get("role_type")) == "economic_buyer":
        return True

    title = _clean_reviewer_title_for_role_inference((source_row or {}).get("reviewer_title"))
    if title and _EXEC_REVIEWER_TITLE_PATTERN.search(title):
        return True
    if title and _MANAGER_DECISION_TITLE_PATTERN.search(title):
        return _has_manager_level_decision_context(result, source_row)
    return False


def _infer_buyer_role_type_from_text(
    buyer_authority: dict[str, Any],
    source_row: dict[str, Any] | None,
) -> str:
    if not isinstance(source_row, dict):
        return "unknown"
    if str(source_row.get("content_type") or "").strip().lower() == "insider_account":
        return "unknown"
    source_text = _combined_source_text(source_row)
    if not source_text:
        return "unknown"
    for pattern in _ECONOMIC_BUYER_TEXT_PATTERNS:
        if pattern.search(source_text):
            return "economic_buyer"
    for pattern in _CHAMPION_TEXT_PATTERNS:
        if pattern.search(source_text):
            return "champion"
    for pattern in _EVALUATOR_TEXT_PATTERNS:
        if pattern.search(source_text):
            return "evaluator"
    buying_stage = str(buyer_authority.get("buying_stage") or "").strip().lower()
    for pattern in _END_USER_TEXT_PATTERNS:
        if pattern.search(source_text):
            return "evaluator" if buying_stage in {"evaluation", "active_purchase"} else "end_user"
    return "unknown"


def _infer_buyer_role_type(
    buyer_authority: dict[str, Any],
    reviewer_context: dict[str, Any] | None,
    reviewer_title: Any,
    source_row: dict[str, Any] | None = None,
) -> str:
    ctx = reviewer_context if isinstance(reviewer_context, dict) else {}
    role_level = str(ctx.get("role_level") or "").strip().lower()
    buying_stage = str(buyer_authority.get("buying_stage") or "").strip().lower()
    if _coerce_bool(buyer_authority.get("has_budget_authority")) is True:
        return "economic_buyer"
    if _coerce_bool(ctx.get("decision_maker")) is True:
        return "economic_buyer"
    if role_level in {"executive", "director"}:
        return "economic_buyer"
    title = _clean_reviewer_title_for_role_inference(reviewer_title)
    if title and _EXEC_REVIEWER_TITLE_PATTERN.search(title):
        return "economic_buyer"
    if role_level == "manager":
        return "champion"
    if title and _CHAMPION_REVIEWER_TITLE_PATTERN.search(title):
        return "champion"
    if role_level == "ic" and buying_stage in {"evaluation", "active_purchase"}:
        return "evaluator"
    if title and _EVALUATOR_REVIEWER_TITLE_PATTERN.search(title):
        return "evaluator" if buying_stage in {"evaluation", "active_purchase"} else "end_user"
    if role_level == "ic":
        return "end_user"
    return _infer_buyer_role_type_from_text(buyer_authority, source_row)


def _is_unknownish(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "unknown", "none", "null", "n/a", "na"}


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


_REPAIR_NEGATIVE_PATTERNS = (
    "cancel", "cancellation", "billing dispute", "refund denied", "runaround",
    "automatic renewal", "auto renew", "renewed without notice", "charged",
    "invoiced", "price increase", "overcharged", "switching cost",
)
_REPAIR_COMPETITOR_PATTERNS = (
    "switched to", "moved to", "replaced with", "evaluating", "looking at",
    "considering", "shortlisting", "shortlisted", "poc with", "proof of concept with",
)
_REPAIR_PRICING_PATTERNS = (
    "billing", "invoice", "invoiced", "charged", "refund", "renewal",
    "price increase", "cost increase", "automatic renewal", "auto renew",
    "overcharged",
)
_REPAIR_RECOMMEND_PATTERNS = (
    "would not recommend", "wouldn't recommend", "do not recommend",
    "don't recommend", "stay away", "avoid", "not worth", "cannot recommend",
)
_REPAIR_FEATURE_GAP_PATTERNS = (
    "missing", "lacks", "lacking", "wish it had", "wish they had",
    "need better", "needs better", "needs more", "could use", "limited",
)
_REPAIR_TIMELINE_PATTERNS = (
    "renewal", "contract end", "contract expires", "deadline", "next quarter",
    "q1", "q2", "q3", "q4", "30 days", "60 days", "90 days",
)
_REPAIR_CATEGORY_SHIFT_PATTERNS = (
    "async", "docs", "documentation", "notion", "confluence", "bundle",
    "workspace", "microsoft 365", "google workspace", "internal tool",
    "homegrown", "home-grown", "custom tool",
)
_REPAIR_CURRENCY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?", re.I)


def _trusted_repair_sources() -> set[str]:
    return set(
        filter_deprecated_sources(
            _config_allowlist(getattr(settings.b2b_churn, "enrichment_priority_sources", ""), ""),
            getattr(
                settings.b2b_churn,
                "deprecated_review_sources",
                "capterra,software_advice,trustpilot,trustradius",
            )
            if isinstance(getattr(settings.b2b_churn, "deprecated_review_sources", ""), str)
            else "capterra,software_advice,trustpilot,trustradius",
        )
    )


def _effective_enrichment_skip_sources() -> set[str]:
    configured = _config_allowlist(getattr(settings.b2b_churn, "enrichment_skip_sources", ""), "")
    deprecated = _config_allowlist(
        getattr(settings.b2b_churn, "deprecated_review_sources", ""),
        "capterra,software_advice,trustpilot,trustradius",
    )
    return {
        source
        for source in [*configured, *deprecated]
        if source
    }


def _repair_text_blob(source_row: dict[str, Any]) -> str:
    return " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()


def _repair_target_fields(result: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    def _add_target(field: str) -> None:
        if field not in targets:
            targets.append(field)

    review_blob = _repair_text_blob(source_row)
    source = str(source_row.get("source") or "").strip().lower()
    status = str(source_row.get("enrichment_status") or "").strip().lower()

    complaints = _normalize_text_list(result.get("specific_complaints"))
    pricing_phrases = _normalize_text_list(result.get("pricing_phrases"))
    recommendation_language = _normalize_text_list(result.get("recommendation_language"))
    feature_gaps = _normalize_text_list(result.get("feature_gaps"))
    event_mentions = result.get("event_mentions") or []
    competitors = result.get("competitors_mentioned") or []
    salience_flags = {
        str(flag or "").strip().lower()
        for flag in result.get("salience_flags") or []
        if str(flag or "").strip()
    }
    timeline = _coerce_json_dict(result.get("timeline"))

    if _normalize_pain_category(result.get("pain_category")) == "overall_dissatisfaction" and _contains_any(review_blob, _REPAIR_NEGATIVE_PATTERNS):
        for field in ("specific_complaints", "pricing_phrases", "recommendation_language"):
            _add_target(field)
    if not competitors and _contains_any(review_blob, _REPAIR_COMPETITOR_PATTERNS):
        _add_target("competitors_mentioned")
    if not pricing_phrases and _contains_any(review_blob, _REPAIR_PRICING_PATTERNS):
        _add_target("pricing_phrases")
    if (
        str(result.get("pain_category") or "").strip().lower() not in {"pricing", "contract_lock_in"}
        and (_REPAIR_CURRENCY_RE.search(review_blob) or "explicit_dollar" in salience_flags)
    ):
        for field in ("specific_complaints", "pricing_phrases"):
            _add_target(field)
    if not complaints and _contains_any(review_blob, _REPAIR_NEGATIVE_PATTERNS):
        _add_target("specific_complaints")
    if not recommendation_language and _contains_any(review_blob, _REPAIR_RECOMMEND_PATTERNS):
        _add_target("recommendation_language")
    if not feature_gaps and _contains_any(review_blob, _REPAIR_FEATURE_GAP_PATTERNS):
        _add_target("feature_gaps")
    if not event_mentions and _contains_any(review_blob, ("renewal", "migration", "switched", "price increase", "invoice")):
        _add_target("event_mentions")
    if (
        _contains_any(review_blob, _REPAIR_TIMELINE_PATTERNS)
        and _is_unknownish(timeline.get("decision_timeline"))
        and not event_mentions
    ):
        _add_target("event_mentions")
    if competitors and all(
        not str(comp.get("reason_category") or "").strip()
        for comp in competitors if isinstance(comp, dict)
    ):
        _add_target("specific_complaints")
    if _contains_any(review_blob, _REPAIR_CATEGORY_SHIFT_PATTERNS) and not feature_gaps and not complaints:
        _add_target("specific_complaints")
    if status == "no_signal" and source in _trusted_repair_sources() and _contains_any(
        review_blob, _REPAIR_NEGATIVE_PATTERNS + _REPAIR_COMPETITOR_PATTERNS
    ):
        for field in ("specific_complaints", "pricing_phrases", "competitors_mentioned", "recommendation_language"):
            _add_target(field)
    return targets


def _needs_field_repair(result: dict[str, Any], source_row: dict[str, Any]) -> bool:
    return bool(_repair_target_fields(result, source_row))


def _has_structural_gap(result: dict[str, Any]) -> bool:
    buyer_authority = _coerce_json_dict(result.get("buyer_authority"))
    timeline = _coerce_json_dict(result.get("timeline"))
    contract = _coerce_json_dict(result.get("contract_context"))
    return any((
        _is_unknownish(buyer_authority.get("role_type")),
        _is_unknownish(timeline.get("decision_timeline")),
        _is_unknownish(contract.get("contract_value_signal")),
    ))


def _apply_structural_repair(
    baseline: dict[str, Any],
    repair: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    merged = json.loads(json.dumps(baseline))
    applied: list[str] = []

    buyer_authority = _coerce_json_dict(merged.get("buyer_authority"))
    repair_authority = _coerce_json_dict(repair.get("buyer_authority"))
    if _is_unknownish(buyer_authority.get("role_type")) and not _is_unknownish(repair_authority.get("role_type")):
        buyer_authority["role_type"] = repair_authority.get("role_type")
        applied.append("buyer_authority.role_type")
    if _is_unknownish(buyer_authority.get("buying_stage")) and not _is_unknownish(repair_authority.get("buying_stage")):
        buyer_authority["buying_stage"] = repair_authority.get("buying_stage")
        applied.append("buyer_authority.buying_stage")
    if applied:
        merged["buyer_authority"] = buyer_authority

    timeline = _coerce_json_dict(merged.get("timeline"))
    repair_timeline = _coerce_json_dict(repair.get("timeline"))
    for field in ("decision_timeline", "contract_end", "evaluation_deadline"):
        if _is_unknownish(timeline.get(field)) and not _is_unknownish(repair_timeline.get(field)):
            timeline[field] = repair_timeline.get(field)
            applied.append(f"timeline.{field}")
    if any(field.startswith("timeline.") for field in applied):
        merged["timeline"] = timeline

    contract = _coerce_json_dict(merged.get("contract_context"))
    repair_contract = _coerce_json_dict(repair.get("contract_context"))
    for field in ("contract_value_signal", "usage_duration", "price_context"):
        if _is_unknownish(contract.get(field)) and not _is_unknownish(repair_contract.get(field)):
            contract[field] = repair_contract.get(field)
            applied.append(f"contract_context.{field}")
    if any(field.startswith("contract_context.") for field in applied):
        merged["contract_context"] = contract

    return merged, applied


def _apply_field_repair(
    baseline: dict[str, Any],
    repair: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    merged = json.loads(json.dumps(baseline))
    applied: list[str] = []

    for field in ("specific_complaints", "pricing_phrases", "recommendation_language", "feature_gaps"):
        existing_items = _normalize_text_list(merged.get(field))
        repair_items = _normalize_text_list(repair.get(field))
        seen = {item.strip().lower() for item in existing_items if item.strip()}
        appended = False
        for item in repair_items:
            key = item.strip().lower()
            if key and key not in seen:
                existing_items.append(item)
                seen.add(key)
                appended = True
        if appended:
            merged[field] = existing_items
            applied.append(field)

    existing_events = []
    seen_events: set[tuple[str, str]] = set()
    for event in merged.get("event_mentions") or []:
        if not isinstance(event, dict):
            continue
        key = (
            str(event.get("event") or "").strip().lower(),
            str(event.get("timeframe") or "").strip().lower(),
        )
        if key not in seen_events:
            existing_events.append(dict(event))
            seen_events.add(key)
    event_added = False
    for event in repair.get("event_mentions") or []:
        if not isinstance(event, dict):
            continue
        key = (
            str(event.get("event") or "").strip().lower(),
            str(event.get("timeframe") or "").strip().lower(),
        )
        if key[0] and key not in seen_events:
            existing_events.append(dict(event))
            seen_events.add(key)
            event_added = True
    if event_added:
        merged["event_mentions"] = existing_events
        applied.append("event_mentions")

    existing_competitors = []
    seen_competitors: dict[str, dict[str, Any]] = {}
    for comp in merged.get("competitors_mentioned") or []:
        if not isinstance(comp, dict):
            continue
        name = str(comp.get("name") or "").strip()
        if not name:
            continue
        key = normalize_company_name(name) or name.lower()
        clone = dict(comp)
        existing_competitors.append(clone)
        seen_competitors[key] = clone
    competitor_changed = False
    for comp in repair.get("competitors_mentioned") or []:
        if not isinstance(comp, dict):
            continue
        name = str(comp.get("name") or "").strip()
        if not name:
            continue
        key = normalize_company_name(name) or name.lower()
        if key in seen_competitors:
            target = seen_competitors[key]
            for field in ("reason_detail",):
                if not str(target.get(field) or "").strip() and str(comp.get(field) or "").strip():
                    target[field] = comp.get(field)
                    competitor_changed = True
            existing_features = _normalize_text_list(target.get("features"))
            feature_seen = {item.strip().lower() for item in existing_features if item.strip()}
            for item in _normalize_text_list(comp.get("features")):
                key_feature = item.strip().lower()
                if key_feature and key_feature not in feature_seen:
                    existing_features.append(item)
                    feature_seen.add(key_feature)
                    competitor_changed = True
            if existing_features:
                target["features"] = existing_features
            continue
        clone = dict(comp)
        existing_competitors.append(clone)
        seen_competitors[key] = clone
        competitor_changed = True
    if competitor_changed:
        merged["competitors_mentioned"] = existing_competitors
        applied.append("competitors_mentioned")

    return merged, applied


def _validate_enrichment(result: dict, source_row: dict[str, Any] | None = None) -> bool:
    """Validate enrichment output structure and data consistency."""
    if "churn_signals" not in result:
        return False
    if "urgency_score" not in result:
        return False
    if not isinstance(result.get("churn_signals"), dict):
        return False

    # Type check: urgency_score must be numeric
    urgency = result.get("urgency_score")
    if isinstance(urgency, str):
        try:
            urgency = float(urgency)
            result["urgency_score"] = urgency
        except (ValueError, TypeError):
            logger.warning("urgency_score is non-numeric string: %r", urgency)
            return False

    if not isinstance(urgency, (int, float)):
        logger.warning("urgency_score has unexpected type: %s", type(urgency).__name__)
        return False

    # Range check: 0-10
    if urgency < 0 or urgency > 10:
        logger.warning("urgency_score out of range [0,10]: %s", urgency)
        return False

    # Boolean coercion: churn_signals fields used in ::boolean casts
    signals = result["churn_signals"]
    for field in _CHURN_SIGNAL_BOOL_FIELDS:
        if field in signals:
            coerced = _coerce_bool(signals[field])
            if coerced is None:
                logger.warning("churn_signals.%s unrecognizable bool: %r -- rejecting", field, signals[field])
                return False
            signals[field] = coerced

    # Consistency warning: high urgency with no intent_to_leave
    intent = signals.get("intent_to_leave")
    if urgency >= 9 and intent is False:
        logger.warning(
            "Contradictory: urgency=%s but intent_to_leave=false -- accepting with warning",
            urgency,
        )

    # Boolean coercion: reviewer_context.decision_maker (used in ::boolean cast)
    reviewer_ctx = result.get("reviewer_context")
    if isinstance(reviewer_ctx, dict) and "decision_maker" in reviewer_ctx:
        coerced = _coerce_bool(reviewer_ctx["decision_maker"])
        if coerced is None:
            logger.warning("reviewer_context.decision_maker unrecognizable bool: %r -- rejecting", reviewer_ctx["decision_maker"])
            return False
        reviewer_ctx["decision_maker"] = coerced

    # Boolean coercion: would_recommend (used in ::boolean cast in vendor_churn_scores)
    if "would_recommend" in result:
        coerced = _coerce_bool(result["would_recommend"])
        if coerced is None:
            # null/None is valid (reviewer didn't express preference) -- keep as null
            result["would_recommend"] = None
        else:
            result["would_recommend"] = coerced

    # Type check: competitors_mentioned must be list; items must be dicts with "name"
    competitors = result.get("competitors_mentioned")
    if competitors is not None and not isinstance(competitors, list):
        logger.warning("competitors_mentioned is not a list: %s", type(competitors).__name__)
        result["competitors_mentioned"] = []
    elif isinstance(competitors, list):
        result["competitors_mentioned"] = [
            c for c in competitors
            if isinstance(c, dict) and "name" in c
        ]

    # Validate evidence_type, displacement_confidence, and reason_category on each competitor entry
    _VALID_EVIDENCE_TYPES = {"explicit_switch", "active_evaluation", "implied_preference", "reverse_flow", "neutral_mention"}
    _VALID_DISP_CONFIDENCE = {"high", "medium", "low", "none"}
    _VALID_REASON_CATEGORIES = {"pricing", "features", "reliability", "ux", "support", "integration"}
    _ET_TO_CONTEXT = {
        "explicit_switch": "switched_to",
        "active_evaluation": "considering",
        "implied_preference": "compared",
        "reverse_flow": "switched_from",
        "neutral_mention": "compared",
    }
    for comp in result.get("competitors_mentioned", []):
        # Coerce unknown evidence_type; fall back from legacy context field
        et = comp.get("evidence_type")
        if et not in _VALID_EVIDENCE_TYPES:
            # Map legacy context -> evidence_type
            legacy = comp.get("context", "")
            _CONTEXT_TO_ET = {
                "switched_to": "explicit_switch",
                "considering": "active_evaluation",
                "compared": "implied_preference",
                "switched_from": "reverse_flow",
            }
            comp["evidence_type"] = _CONTEXT_TO_ET.get(legacy, "neutral_mention")

        # Coerce unknown displacement_confidence
        dc = comp.get("displacement_confidence")
        if dc not in _VALID_DISP_CONFIDENCE:
            comp["displacement_confidence"] = "low"

        # Consistency: reverse_flow -> confidence "none"
        if comp["evidence_type"] == "reverse_flow":
            comp["displacement_confidence"] = "none"
        # Consistency: neutral_mention -> confidence at most "low"
        if comp["evidence_type"] == "neutral_mention" and comp.get("displacement_confidence") in ("high", "medium"):
            comp["displacement_confidence"] = "low"

        # Coerce reason_category to taxonomy
        rc = comp.get("reason_category")
        if rc and rc not in _VALID_REASON_CATEGORIES:
            comp["reason_category"] = None

        # Backward compat: populate context from evidence_type
        comp["context"] = _ET_TO_CONTEXT.get(comp["evidence_type"], "compared")

        # Backward compat: populate reason from reason_category + reason_detail
        rc = comp.get("reason_category")
        rd = comp.get("reason_detail")
        if rc and rd:
            comp["reason"] = f"{rc}: {rd}"
        elif rc:
            comp["reason"] = rc
        elif rd:
            comp["reason"] = rd
        # else: keep existing reason if any (legacy data)

    # Type check: quotable_phrases must be list if present
    qp = result.get("quotable_phrases")
    if qp is not None and not isinstance(qp, list):
        logger.warning("quotable_phrases is not a list: %s", type(qp).__name__)
        result["quotable_phrases"] = []

    # Type check: feature_gaps must be list if present
    fg = result.get("feature_gaps")
    if fg is not None and not isinstance(fg, list):
        logger.warning("feature_gaps is not a list: %s", type(fg).__name__)
        result["feature_gaps"] = []

    # Coerce unknown / legacy generic pain_category to the canonical fallback
    pain = result.get("pain_category")
    if pain is not None:
        normalized_pain = _normalize_pain_category(pain)
        if normalized_pain != str(pain).strip().lower():
            logger.warning("Normalizing pain_category %r -> %r", pain, normalized_pain)
        result["pain_category"] = normalized_pain

    # --- New expanded field validation (permissive: coerce, never reject) ---

    # pain_categories: list of {category, severity}
    pc = result.get("pain_categories")
    if pc is not None:
        if not isinstance(pc, list):
            result["pain_categories"] = []
        else:
            cleaned = []
            for item in pc:
                if not isinstance(item, dict):
                    continue
                cat = _normalize_pain_category(item.get("category", "overall_dissatisfaction"))
                sev = item.get("severity", "minor")
                if sev not in _KNOWN_SEVERITY_LEVELS:
                    sev = "minor"
                cleaned.append({"category": cat, "severity": sev})
            result["pain_categories"] = cleaned

    # budget_signals: dict with known keys
    bs = result.get("budget_signals")
    if bs is not None:
        if not isinstance(bs, dict):
            result["budget_signals"] = {}
        else:
            for field in ("annual_spend_estimate", "price_per_seat"):
                if field in bs and bs[field] is not None and not isinstance(bs[field], (int, float)):
                    text = _normalize_budget_value_text(bs[field])
                    bs[field] = text if text else None
            if "seat_count" in bs and bs["seat_count"] is not None:
                try:
                    seat = int(bs["seat_count"])
                    bs["seat_count"] = seat if 1 <= seat <= 1_000_000 else None
                except (ValueError, TypeError):
                    bs["seat_count"] = None
            if "price_increase_mentioned" in bs:
                coerced = _coerce_bool(bs["price_increase_mentioned"])
                bs["price_increase_mentioned"] = coerced if coerced is not None else False
            if "price_increase_detail" in bs and bs["price_increase_detail"] is not None:
                detail = _normalize_budget_detail_text(bs["price_increase_detail"])
                bs["price_increase_detail"] = detail if detail else None
                if bs["price_increase_detail"] and not _coerce_bool(bs.get("price_increase_mentioned")):
                    bs["price_increase_mentioned"] = True

    # use_case: dict with lists and lock_in_level
    uc = result.get("use_case")
    if uc is not None:
        if not isinstance(uc, dict):
            result["use_case"] = {}
        else:
            if "modules_mentioned" in uc and not isinstance(uc["modules_mentioned"], list):
                uc["modules_mentioned"] = []
            if "integration_stack" in uc and not isinstance(uc["integration_stack"], list):
                uc["integration_stack"] = []
            lil = uc.get("lock_in_level")
            if lil and lil not in _KNOWN_LOCK_IN_LEVELS:
                uc["lock_in_level"] = "unknown"

    # reviewer_context: normalize role level and backfill from title/text
    reviewer_ctx = result.get("reviewer_context")
    if reviewer_ctx is None or not isinstance(reviewer_ctx, dict):
        result["reviewer_context"] = {}
        reviewer_ctx = result["reviewer_context"]
    role_level = _canonical_role_level(reviewer_ctx.get("role_level"))
    if role_level == "unknown":
        role_level = _infer_role_level_from_text(
            (source_row or {}).get("reviewer_title"),
            source_row,
        )
    reviewer_ctx["role_level"] = role_level
    decision_maker = _coerce_bool(reviewer_ctx.get("decision_maker"))
    derived_decision_maker = _infer_decision_maker(result, source_row)
    if decision_maker is None:
        decision_maker = derived_decision_maker
    else:
        decision_maker = bool(decision_maker or derived_decision_maker)
    reviewer_ctx["decision_maker"] = decision_maker
    company_name = str(reviewer_ctx.get("company_name") or "").strip()
    if company_name:
        reviewer_ctx["company_name"] = company_name
    else:
        trusted_company = _trusted_reviewer_company_name(source_row)
        if trusted_company:
            reviewer_ctx["company_name"] = trusted_company

    # sentiment_trajectory: dict with direction
    st = result.get("sentiment_trajectory")
    if st is not None:
        if not isinstance(st, dict):
            result["sentiment_trajectory"] = {}
        else:
            d = st.get("direction")
            if d and d not in _KNOWN_SENTIMENT_DIRECTIONS:
                st["direction"] = "unknown"

    # buyer_authority: dict with role_type, booleans, buying_stage
    ba = result.get("buyer_authority")
    if ba is not None:
        if not isinstance(ba, dict):
            result["buyer_authority"] = {}
            ba = result["buyer_authority"]
        reviewer_ctx = (
            result.get("reviewer_context")
            if isinstance(result.get("reviewer_context"), dict)
            else {}
        )
        for bool_field in ("has_budget_authority", "executive_sponsor_mentioned"):
            if bool_field in ba:
                coerced = _coerce_bool(ba[bool_field])
                ba[bool_field] = coerced if coerced is not None else False
        bstage = ba.get("buying_stage")
        if bstage and bstage not in _KNOWN_BUYING_STAGES:
            ba["buying_stage"] = "unknown"
        canonical_rt = _canonical_role_type(ba.get("role_type"))
        derived_role_type = _infer_buyer_role_type(
            ba,
            reviewer_ctx,
            (source_row or {}).get("reviewer_title"),
            source_row,
        )
        if canonical_rt == "unknown":
            ba["role_type"] = derived_role_type
        else:
            ba["role_type"] = canonical_rt
        if derived_role_type == "economic_buyer":
            ba["role_type"] = "economic_buyer"
        if ba["role_type"] == "economic_buyer":
            reviewer_ctx["decision_maker"] = True

    # timeline: dict with decision_timeline
    tl = result.get("timeline")
    if tl is not None:
        if not isinstance(tl, dict):
            result["timeline"] = {}
        else:
            dt = tl.get("decision_timeline")
            if dt and dt not in _KNOWN_DECISION_TIMELINES:
                tl["decision_timeline"] = "unknown"

    # contract_context: dict with contract_value_signal
    cc = result.get("contract_context")
    if cc is not None:
        if not isinstance(cc, dict):
            result["contract_context"] = {}
        else:
            cvs = cc.get("contract_value_signal")
            if cvs and cvs not in _KNOWN_CONTRACT_VALUE_SIGNALS:
                cc["contract_value_signal"] = "unknown"

    # content_classification: pass-through string, coerce to known values
    cc_val = result.get("content_classification")
    if cc_val and cc_val not in _KNOWN_CONTENT_TYPES:
        result["content_classification"] = "review"

    if _schema_version(result) >= 3:
        missing_fields = _missing_witness_primitives(result)
        if missing_fields:
            if source_row is None:
                logger.warning(
                    "schema v3 enrichment missing witness primitives without source row: %s",
                    ", ".join(missing_fields),
                )
                return False
            try:
                recomputed = _compute_derived_fields(
                    json.loads(json.dumps(result)),
                    source_row,
                )
            except Exception:
                logger.warning(
                    "schema v3 witness primitive recompute failed for %s",
                    source_row.get("id"),
                    exc_info=True,
                )
                return False
            result.clear()
            result.update(recomputed)

    # witness-oriented deterministic evidence fields
    replacement_mode = str(result.get("replacement_mode") or "").strip()
    if replacement_mode not in _KNOWN_REPLACEMENT_MODES:
        result["replacement_mode"] = "none"
    operating_model_shift = str(result.get("operating_model_shift") or "").strip()
    if operating_model_shift not in _KNOWN_OPERATING_MODEL_SHIFTS:
        result["operating_model_shift"] = "none"
    productivity_delta_claim = str(result.get("productivity_delta_claim") or "").strip()
    if productivity_delta_claim not in _KNOWN_PRODUCTIVITY_DELTA_CLAIMS:
        result["productivity_delta_claim"] = "unknown"
    org_pressure_type = str(result.get("org_pressure_type") or "").strip()
    if org_pressure_type not in _KNOWN_ORG_PRESSURE_TYPES:
        result["org_pressure_type"] = "none"

    salience_flags = result.get("salience_flags")
    if salience_flags is not None:
        if not isinstance(salience_flags, list):
            result["salience_flags"] = []
        else:
            result["salience_flags"] = [
                str(flag).strip() for flag in salience_flags if str(flag or "").strip()
            ]

    evidence_spans = result.get("evidence_spans")
    if evidence_spans is not None:
        if not isinstance(evidence_spans, list):
            result["evidence_spans"] = []
        else:
            cleaned_spans: list[dict[str, Any]] = []
            for idx, span in enumerate(evidence_spans):
                if not isinstance(span, dict):
                    continue
                text = str(span.get("text") or "").strip()
                if not text:
                    continue
                pain_category = str(span.get("pain_category") or "").strip()
                replacement = str(span.get("replacement_mode") or "").strip()
                operating_shift = str(span.get("operating_model_shift") or "").strip()
                productivity = str(span.get("productivity_delta_claim") or "").strip()
                cleaned_spans.append({
                    "span_id": str(span.get("span_id") or f"derived:{idx}"),
                    "_sid": str(span.get("_sid") or span.get("span_id") or f"derived:{idx}"),
                    "text": text,
                    "start_char": span.get("start_char"),
                    "end_char": span.get("end_char"),
                    "signal_type": str(span.get("signal_type") or "review_context"),
                    "pain_category": pain_category if pain_category in _KNOWN_PAIN_CATEGORIES else None,
                    "competitor": str(span.get("competitor") or "").strip() or None,
                    "company_name": str(span.get("company_name") or "").strip() or None,
                    "reviewer_title": str(span.get("reviewer_title") or "").strip() or None,
                    "time_anchor": str(span.get("time_anchor") or "").strip() or None,
                    "numeric_literals": span.get("numeric_literals") if isinstance(span.get("numeric_literals"), dict) else {},
                    "flags": [
                        str(flag).strip() for flag in (span.get("flags") or [])
                        if str(flag or "").strip()
                    ],
                    "replacement_mode": replacement if replacement in _KNOWN_REPLACEMENT_MODES else result.get("replacement_mode"),
                    "operating_model_shift": operating_shift if operating_shift in _KNOWN_OPERATING_MODEL_SHIFTS else result.get("operating_model_shift"),
                    "productivity_delta_claim": productivity if productivity in _KNOWN_PRODUCTIVITY_DELTA_CLAIMS else result.get("productivity_delta_claim"),
                })
            result["evidence_spans"] = cleaned_spans

    if _schema_version(result) >= 3:
        remaining_missing = _missing_witness_primitives(result)
        if remaining_missing:
            logger.warning(
                "schema v3 enrichment still missing witness primitives after normalization: %s",
                ", ".join(remaining_missing),
            )
            return False

    # insider_signals: validate structure if present
    insider = result.get("insider_signals")
    if insider is not None:
        if not isinstance(insider, dict):
            result["insider_signals"] = None
        else:
            # org_health: must be dict
            oh = insider.get("org_health")
            if oh is not None and not isinstance(oh, dict):
                insider["org_health"] = {}
            elif isinstance(oh, dict):
                # culture_indicators must be list
                ci = oh.get("culture_indicators")
                if ci is not None and not isinstance(ci, list):
                    oh["culture_indicators"] = []
                # Enum fields: coerce unknowns
                for field, allowed in (
                    ("bureaucracy_level", _KNOWN_ORG_HEALTH_LEVELS),
                    ("leadership_quality", _KNOWN_LEADERSHIP_QUALITIES),
                    ("innovation_climate", _KNOWN_INNOVATION_CLIMATES),
                ):
                    val = oh.get(field)
                    if val and val not in allowed:
                        oh[field] = "unknown"

            # talent_drain: must be dict
            td = insider.get("talent_drain")
            if td is not None and not isinstance(td, dict):
                insider["talent_drain"] = {}
            elif isinstance(td, dict):
                for bool_field in ("departures_mentioned", "layoff_fear"):
                    if bool_field in td:
                        coerced = _coerce_bool(td[bool_field])
                        td[bool_field] = coerced if coerced is not None else False
                morale = td.get("morale")
                if morale and morale not in _KNOWN_MORALE_LEVELS:
                    td["morale"] = "unknown"

            # departure_type: enum
            dt = insider.get("departure_type")
            if dt and dt not in _KNOWN_DEPARTURE_TYPES:
                insider["departure_type"] = "unknown"

    return True


async def _increment_attempts(pool, review_id, current_attempts: int, max_attempts: int) -> None:
    """Bump attempts atomically; reset to pending or mark failed if exhausted."""
    new_status = "failed" if (current_attempts + 1) >= max_attempts else "pending"
    await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_attempts = enrichment_attempts + 1,
            enrichment_status = $1
        WHERE id = $2
        """,
        new_status, review_id,
    )
