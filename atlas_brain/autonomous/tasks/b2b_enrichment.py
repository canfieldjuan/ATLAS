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
import hashlib
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
from ...services.b2b.enrichment_stage_controller import (
    apply_stage_decision,
    apply_review_stage_transition,
    defer_review_transition,
    finalize_stage_batch,
    prepare_stage_execution,
    persist_review_transition,
    submit_stage_batch,
)
from ...services.b2b.enrichment_domain import (
    build_classify_payload as _domain_build_classify_payload,
    combined_review_text_length as _domain_combined_review_text_length,
    effective_enrichment_skip_sources as _domain_effective_enrichment_skip_sources,
    effective_min_review_text_length as _domain_effective_min_review_text_length,
    tier1_has_extraction_gaps as _domain_tier1_has_extraction_gaps,
    tier2_system_prompt_for_content_type as _domain_tier2_system_prompt_for_content_type,
)
from ...services.b2b.enrichment_persistence import (
    EnrichmentFinalizationDeps,
    EnrichmentPersistenceDeps,
    finalize_enrichment_for_persist as _service_finalize_enrichment_for_persist,
    persist_enrichment_result as _service_persist_enrichment_result,
)
from ...services.b2b.enrichment_outcome_policy import (
    EnrichmentOutcomePolicyDeps,
    detect_low_fidelity_reasons as _service_detect_low_fidelity_reasons,
    is_no_signal_result as _service_is_no_signal_result,
    trusted_reviewer_company_name as _service_trusted_reviewer_company_name,
    witness_metrics as _service_witness_metrics,
)
from ...services.b2b.enrichment_validation import (
    EnrichmentValidationDeps,
    validate_enrichment as _service_validate_enrichment,
)
from ...services.b2b.enrichment_derivation import (
    EnrichmentDerivationDeps,
    compute_derived_fields as _service_compute_derived_fields,
)
from ...services.b2b.enrichment_phrase_metadata import (
    EnrichmentPhraseMetadataDeps,
    apply_phrase_metadata_contract as _service_apply_phrase_metadata_contract,
    coerce_legacy_phrase_arrays as _service_coerce_legacy_phrase_arrays,
    normalize_phrase_metadata as _service_normalize_phrase_metadata,
    normalize_tag_value as _service_normalize_tag_value,
)
from ...services.b2b.enrichment_repair import (
    EnrichmentRepairDeps,
    apply_field_repair as _service_apply_field_repair,
    apply_structural_repair as _service_apply_structural_repair,
    has_structural_gap as _service_has_structural_gap,
    needs_field_repair as _service_needs_field_repair,
    repair_target_fields as _service_repair_target_fields,
)
from ...services.b2b.enrichment_buyer_authority import (
    EnrichmentBuyerAuthorityDeps,
    canonical_role_level as _service_canonical_role_level,
    canonical_role_type as _service_canonical_role_type,
    derive_buyer_authority_fields as _service_derive_buyer_authority_fields,
    infer_buyer_role_type as _service_infer_buyer_role_type,
    infer_decision_maker as _service_infer_decision_maker,
    infer_role_level_from_text as _service_infer_role_level_from_text,
)
from ...services.b2b.enrichment_timeline import (
    EnrichmentTimelineDeps,
    derive_concrete_timeline_fields as _service_derive_concrete_timeline_fields,
    derive_decision_timeline as _service_derive_decision_timeline,
)
from ...services.b2b.enrichment_budget import (
    EnrichmentBudgetDeps,
    derive_budget_signals as _service_derive_budget_signals,
    derive_contract_value_signal as _service_derive_contract_value_signal,
    normalize_budget_detail_text as _service_normalize_budget_detail_text,
    normalize_budget_value_text as _service_normalize_budget_value_text,
)
from ...services.b2b.enrichment_pain_competition import (
    EnrichmentPainCompetitionDeps,
    compute_pain_confidence as _service_compute_pain_confidence,
    demote_primary_pain as _service_demote_primary_pain,
    derive_competitor_annotations as _service_derive_competitor_annotations,
    derive_pain_categories as _service_derive_pain_categories,
    recover_competitor_mentions as _service_recover_competitor_mentions,
    subject_vendor_phrase_texts as _service_subject_vendor_phrase_texts,
)
from ...services.b2b.enrichment_urgency import (
    EnrichmentUrgencyDeps,
    derive_urgency_indicators as _service_derive_urgency_indicators,
)
from ...services.b2b.enrichment_support import (
    coerce_bool as _service_coerce_bool,
    coerce_json_dict as _service_coerce_json_dict,
    combined_source_text as _service_combined_source_text,
    contains_any as _service_contains_any,
    dedupe_reason_codes as _service_dedupe_reason_codes,
    has_commercial_context as _service_has_commercial_context,
    has_consumer_context as _service_has_consumer_context,
    has_strong_commercial_context as _service_has_strong_commercial_context,
    has_technical_context as _service_has_technical_context,
    is_unknownish as _service_is_unknownish,
    normalize_compare_text as _service_normalize_compare_text,
    normalize_text_list as _service_normalize_text_list,
    normalized_low_fidelity_noisy_sources as _service_normalized_low_fidelity_noisy_sources,
    normalized_name_tokens as _service_normalized_name_tokens,
    text_mentions_name as _service_text_mentions_name,
)
from ...services.b2b.enrichment_stage_planner import (
    build_tier1_stage_plan,
    build_tier2_stage_plan,
    stage_backend_name as _planner_stage_backend_name,
)
from ...services.b2b.enrichment_row_runner import EnrichmentRunnerDeps, run_enrichment_rows
from ...services.b2b.enrichment_task_runner import EnrichmentTaskRunnerDeps, run_enrichment_task
from ...services.b2b.enrichment_stage_runs import (
    ensure_stage_run,
    mark_stage_run,
)
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


def _stage_backend_name(*, batch_enabled: bool, provider: str) -> str:
    return _planner_stage_backend_name(batch_enabled=batch_enabled, provider=provider)


def _get_base_enrichment_llm(local_only: bool):
    """Resolve the deterministic local enrichment model from vLLM only."""
    from ...pipelines.llm import get_pipeline_llm

    return get_pipeline_llm(
        workload="vllm",
        try_openrouter=False,
        auto_activate_ollama=False,
    )


def _tier2_system_prompt_for_content_type(prompt: str, content_type: str | None) -> str:
    return _domain_tier2_system_prompt_for_content_type(prompt, content_type)


_ANTHROPIC_CACHE_MIN_CHARS = 1024


def _resolve_tier_routing(cfg, *, local_only_override: bool | None = None) -> tuple[bool, bool]:
    """Return ``(use_openrouter_tier1, use_openrouter_tier2)`` for routing.

    Tier 1 honors ``enrichment_local_only`` directly -- when True, Tier 1
    extraction (verbatim phrase pulls + phrase_metadata) stays on local
    vLLM. Tier 2 (pain_categories, competitor classification,
    buyer_authority, sentiment_trajectory) is a nuance-classification step
    that typically benefits from a frontier model, so it can escape
    local_only when ``enrichment_tier2_force_openrouter`` is set. Both
    tiers also require the OpenRouter credentials/model to actually be
    configured before they can route there.

    ``local_only_override`` lets the single-row code path pass the
    already-computed local_only flag (which may include task-level
    overrides) instead of re-reading cfg.
    """
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


def _prepare_stage_request(
    stage_id: str,
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
):
    from ...services.b2b.cache_runner import prepare_b2b_exact_stage_request
    from ._b2b_batch_utils import exact_stage_request_fingerprint

    work_fingerprint_payload = {
        "stage_id": str(stage_id or ""),
        "system_prompt": str(system_prompt or ""),
        "user_content": str(user_content or ""),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    request = prepare_b2b_exact_stage_request(
        stage_id,
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
    work_fingerprint = hashlib.sha256(
        json.dumps(
            work_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return request, exact_stage_request_fingerprint(request), work_fingerprint


def _stage_result_text(parsed: dict[str, Any] | None) -> str | None:
    if parsed is None:
        return None
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stage_usage_snapshot(*, tier: int, cache_hit: bool, generated: bool) -> dict[str, int]:
    usage = _empty_exact_cache_usage()
    if cache_hit:
        usage[f"tier{tier}_exact_cache_hits"] += 1
        usage["exact_cache_hits"] += 1
    elif generated:
        usage[f"tier{tier}_generated_calls"] += 1
        usage["generated"] += 1
    return usage


def _stage_usage_from_row(row: dict[str, Any] | None, *, tier: int) -> dict[str, int]:
    usage = _empty_exact_cache_usage()
    if not isinstance(row, dict):
        return usage
    stored = row.get("usage_json")
    if isinstance(stored, dict):
        for key in usage:
            usage[key] = int(stored.get(key) or 0)
        if any(usage.values()):
            return usage
    result_source = str(row.get("result_source") or "").strip().lower()
    if result_source == "exact_cache":
        return _stage_usage_snapshot(tier=tier, cache_hit=True, generated=False)
    if result_source in {"generated", "batch_reuse"}:
        return _stage_usage_snapshot(tier=tier, cache_hit=False, generated=True)
    return usage


def _parse_stage_row_result(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    text = str(row.get("response_text") or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


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
    # OpenRouter responses nest cache hits under prompt_tokens_details and put
    # cache writes at the top level. Anthropic-direct responses use
    # cache_read_input_tokens / cache_creation_input_tokens. Pull both shapes
    # so _trace_cache_metrics actually sees a value -- the previous version
    # only forwarded input/output/billable, so cached_tokens was always 0
    # in llm_usage even when caching genuinely fired.
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
    return _domain_tier1_has_extraction_gaps(tier1, source=source)


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
    return _service_normalize_text_list(values)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return _service_contains_any(text, needles)


_PAIN_DERIVATION_FIELDS: tuple[str, ...] = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "quotable_phrases",
)


def _pain_competition_deps() -> EnrichmentPainCompetitionDeps:
    return EnrichmentPainCompetitionDeps(
        normalize_text_list=_normalize_text_list,
        normalize_pain_category=_normalize_pain_category,
        normalize_company_name=normalize_company_name,
        pain_patterns=_PAIN_PATTERNS,
        pain_derivation_fields=_PAIN_DERIVATION_FIELDS,
        competitor_recovery_patterns=_COMPETITOR_RECOVERY_PATTERNS,
        competitor_recovery_blocklist=_COMPETITOR_RECOVERY_BLOCKLIST,
        generic_competitor_tokens=_GENERIC_COMPETITOR_TOKENS,
        competitor_context_patterns=_COMPETITOR_CONTEXT_PATTERNS,
    )


def _subject_vendor_phrase_texts(result: dict, field: str) -> list[str]:
    return _service_subject_vendor_phrase_texts(
        result,
        field,
        deps=_pain_competition_deps(),
    )


def _derive_pain_categories(result: dict) -> list[dict[str, str]]:
    return _service_derive_pain_categories(
        result,
        deps=_pain_competition_deps(),
    )


def _compute_pain_confidence(result: dict, pain_category: str) -> str:
    return _service_compute_pain_confidence(
        result,
        pain_category,
        deps=_pain_competition_deps(),
    )


def _demote_primary_pain(result: dict, demoted_category: str) -> None:
    _service_demote_primary_pain(result, demoted_category)


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


def _recover_competitor_mentions(result: dict, source_row: dict[str, Any]) -> list[dict[str, Any]]:
    return _service_recover_competitor_mentions(
        result,
        source_row,
        deps=_pain_competition_deps(),
    )


def _derive_competitor_annotations(result: dict, source_row: dict[str, Any]) -> list[dict[str, Any]]:
    return _service_derive_competitor_annotations(
        result,
        source_row,
        deps=_pain_competition_deps(),
    )


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
    return _service_derive_concrete_timeline_fields(
        result,
        source_row,
        deps=_timeline_deps(),
    )


def _derive_decision_timeline(
    result: dict,
    source_row: dict[str, Any] | None = None,
) -> str:
    return _service_derive_decision_timeline(
        result,
        source_row,
        deps=_timeline_deps(),
    )


def _budget_match_window(text: str, match: re.Match[str], radius: int = 56) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end].lower()


def _normalize_budget_value_text(value: Any) -> str | None:
    return _service_normalize_budget_value_text(value)


def _normalize_budget_detail_text(value: Any) -> str | None:
    return _service_normalize_budget_detail_text(value)


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
    return _service_derive_budget_signals(
        result,
        source_row,
        deps=_budget_deps(),
    )


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
    return _service_derive_contract_value_signal(result)


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
    return _service_derive_buyer_authority_fields(
        result,
        source_row,
        deps=_buyer_authority_deps(),
    )


def _derive_urgency_indicators(
    result: dict,
    source_row: dict[str, Any],
    *,
    price_complaint: bool = False,
) -> dict[str, bool]:
    return _service_derive_urgency_indicators(
        result,
        source_row,
        deps=EnrichmentUrgencyDeps(
            contains_any=_contains_any,
            normalize_text_list=_normalize_text_list,
        ),
        price_complaint=price_complaint,
    )


def _is_no_signal_result(result: dict, source_row: dict[str, Any]) -> bool:
    return _service_is_no_signal_result(result, source_row)


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
    _service_coerce_legacy_phrase_arrays(result)


def _normalize_tag_value(value: Any, allowed: tuple[str, ...]) -> tuple[str, bool]:
    return _service_normalize_tag_value(value, allowed)


def _normalize_phrase_metadata(
    result: dict,
    source_row: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from ._b2b_grounding import check_phrase_grounded
    return _service_normalize_phrase_metadata(
        result,
        source_row,
        deps=EnrichmentPhraseMetadataDeps(
            check_phrase_grounded=check_phrase_grounded,
        ),
    )


def _apply_phrase_metadata_contract(
    result: dict,
    source_row: dict[str, Any],
) -> None:
    from ._b2b_grounding import check_phrase_grounded

    _service_apply_phrase_metadata_contract(
        result,
        source_row,
        deps=EnrichmentPhraseMetadataDeps(
            check_phrase_grounded=check_phrase_grounded,
        ),
    )


def _compute_derived_fields(result: dict, source_row: dict[str, Any]) -> dict:
    from ...reasoning.evidence_engine import get_evidence_engine

    return _service_compute_derived_fields(
        result,
        source_row,
        deps=EnrichmentDerivationDeps(
            get_evidence_engine=get_evidence_engine,
            coerce_legacy_phrase_arrays=_coerce_legacy_phrase_arrays,
            apply_phrase_metadata_contract=_apply_phrase_metadata_contract,
            derive_pain_categories=_derive_pain_categories,
            recover_competitor_mentions=_recover_competitor_mentions,
            derive_competitor_annotations=_derive_competitor_annotations,
            derive_budget_signals=_derive_budget_signals,
            derive_buyer_authority_fields=_derive_buyer_authority_fields,
            derive_concrete_timeline_fields=_derive_concrete_timeline_fields,
            derive_decision_timeline=_derive_decision_timeline,
            derive_contract_value_signal=_derive_contract_value_signal,
            derive_urgency_indicators=_derive_urgency_indicators,
            normalize_pain_category=_normalize_pain_category,
            subject_vendor_phrase_texts=_subject_vendor_phrase_texts,
            compute_pain_confidence=_compute_pain_confidence,
            demote_primary_pain=_demote_primary_pain,
            derive_replacement_mode=derive_replacement_mode,
            derive_operating_model_shift=derive_operating_model_shift,
            derive_productivity_delta_claim=derive_productivity_delta_claim,
            derive_org_pressure_type=derive_org_pressure_type,
            derive_salience_flags=derive_salience_flags,
            derive_evidence_spans=derive_evidence_spans,
        ),
    )


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
    return _service_finalize_enrichment_for_persist(
        result,
        source_row,
        deps=EnrichmentFinalizationDeps(
            compute_derived_fields=_compute_derived_fields,
            validate_enrichment=_validate_enrichment,
        ),
    )


def _trusted_reviewer_company_name(source_row: dict[str, Any] | None) -> str | None:
    return _service_trusted_reviewer_company_name(
        source_row,
        deps=EnrichmentOutcomePolicyDeps(
            normalized_low_fidelity_noisy_sources=_normalized_low_fidelity_noisy_sources,
            normalize_compare_text=_normalize_compare_text,
            text_mentions_name=_text_mentions_name,
            normalized_name_tokens=_normalized_name_tokens,
            has_commercial_context=_has_commercial_context,
            has_strong_commercial_context=_has_strong_commercial_context,
            has_technical_context=_has_technical_context,
            has_consumer_context=_has_consumer_context,
            dedupe_reason_codes=_dedupe_reason_codes,
            normalize_company_name=normalize_company_name,
        ),
    )


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
    return _service_witness_metrics(result)


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
    return await _service_persist_enrichment_result(
        pool,
        row,
        result,
        model_id=model_id,
        max_attempts=max_attempts,
        run_id=run_id,
        cache_usage=cache_usage,
        deps=EnrichmentPersistenceDeps(
            finalize_enrichment_for_persist=_finalize_enrichment_for_persist,
            witness_metrics=_witness_metrics,
            detect_low_fidelity_reasons=_detect_low_fidelity_reasons,
            is_no_signal_result=_is_no_signal_result,
            notify_high_urgency=_notify_high_urgency,
            increment_attempts=_increment_attempts,
            normalize_company_name=normalize_company_name,
        ),
    )


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
    return await run_enrichment_rows(
        rows,
        cfg,
        pool,
        max_attempts=max_attempts,
        effective_concurrency=effective_concurrency,
        run_id=run_id,
        task=task,
        deps=EnrichmentRunnerDeps(
            apply_review_stage_transition=apply_review_stage_transition,
            apply_stage_decision=apply_stage_decision,
            build_tier1_stage_plan=build_tier1_stage_plan,
            build_tier2_stage_plan=build_tier2_stage_plan,
            defer_review_transition=defer_review_transition,
            enrich_single=_enrich_single,
            finalize_stage_batch=finalize_stage_batch,
            parse_stage_row_result=_parse_stage_row_result,
            empty_exact_cache_usage=_empty_exact_cache_usage,
            persist_review_transition=persist_review_transition,
            prepare_stage_execution=prepare_stage_execution,
            row_usage_result=_row_usage_result,
            resolve_tier_routing=_resolve_tier_routing,
            combined_review_text_length=_combined_review_text_length,
            effective_min_review_text_length=_effective_min_review_text_length,
            effective_enrichment_skip_sources=_effective_enrichment_skip_sources,
            build_classify_payload=_build_classify_payload,
            prepare_stage_request=_prepare_stage_request,
            tier2_system_prompt_for_content_type=_tier2_system_prompt_for_content_type,
            stage_usage_from_row=_stage_usage_from_row,
            stage_usage_snapshot=_stage_usage_snapshot,
            accumulate_exact_cache_usage=_accumulate_exact_cache_usage,
            tier1_has_extraction_gaps=_tier1_has_extraction_gaps,
            merge_tier1_tier2=_merge_tier1_tier2,
            persist_enrichment_result=_persist_enrichment_result,
            defer_batch_row=_defer_batch_row,
            submit_stage_batch=submit_stage_batch,
            unpack_stage_result=_unpack_stage_result,
            call_openrouter_tier2=_call_openrouter_tier2,
            call_vllm_tier2=_call_vllm_tier2,
            get_tier2_client=_get_tier2_client,
        ),
    )


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
    return await run_enrichment_task(
        task=task,
        cfg=cfg,
        pool=pool,
        deps=EnrichmentTaskRunnerDeps(
            recover_orphaned_enriching=_recover_orphaned_enriching,
            mark_exhausted_pending_failed=_mark_exhausted_pending_failed,
            queue_version_upgrades=_queue_version_upgrades,
            queue_model_upgrades=_queue_model_upgrades,
            task_run_id=_task_run_id,
            enrich_rows=_enrich_rows,
            fetch_review_funnel_audit=_fetch_review_funnel_audit,
            empty_exact_cache_usage=_empty_exact_cache_usage,
            accumulate_exact_cache_usage=_accumulate_exact_cache_usage,
            coerce_int_value=_coerce_int_value,
            coerce_int_override=_coerce_int_override,
            coerce_float_value=_coerce_float_value,
        ),
    )


_MIN_REVIEW_TEXT_LENGTH = 80  # Skip LLM calls for reviews shorter than this

# Verified review platforms -- every review gets full extraction (skip triage).


def _combined_review_text_length(row: dict[str, Any] | None) -> int:
    return _domain_combined_review_text_length(row)


def _effective_min_review_text_length(row: dict[str, Any] | None) -> int:
    return _domain_effective_min_review_text_length(row)


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
        from ...skills import get_skill_registry

        registry = get_skill_registry()
        tier1_skill = registry.get("digest/b2b_churn_extraction_tier1")
        tier1_request = None
        tier1_request_fingerprint = None
        tier1_work_fingerprint = None
        existing_tier1_stage = None
        tier1 = None
        tier1_model = None
        tier1_cache_hit = False
        tier1_reused_from_stage = False

        # Tier 1: deterministic extraction (base fields). Tier 2: nuance
        # classification. Each tier has its own routing flag so a deployment
        # can keep Tier 1 on local vLLM (cheap, large pool) while routing
        # Tier 2 to a frontier OpenRouter model (better at the nuance work).
        use_openrouter_tier1, use_openrouter_tier2 = _resolve_tier_routing(
            cfg, local_only_override=local_only,
        )
        if tier1_skill is not None:
            tier1_provider = "openrouter" if use_openrouter_tier1 else "vllm"
            tier1_model_planned = (
                cfg.enrichment_openrouter_model or "anthropic/claude-haiku-4-5"
                if use_openrouter_tier1
                else cfg.enrichment_tier1_model
            )
            tier1_plan = build_tier1_stage_plan(
                row=row,
                payload_json=payload_json,
                system_prompt=str(tier1_skill.content or ""),
                model=str(tier1_model_planned or ""),
                provider=tier1_provider,
                batch_enabled=False,
                run_id=run_id,
                prepare_stage_request=_prepare_stage_request,
                max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096)
                if use_openrouter_tier1 else cfg.enrichment_tier1_max_tokens,
                guided_json=None if use_openrouter_tier1 else _TIER1_JSON_SCHEMA,
            )
            tier1_request = tier1_plan.request
            tier1_request_fingerprint = tier1_plan.request_fingerprint
            tier1_work_fingerprint = tier1_plan.work_fingerprint
            await ensure_stage_run(
                pool,
                review_id=review_id,
                stage_id="b2b_enrichment.tier1",
                work_fingerprint=str(tier1_plan.work_fingerprint),
                request_fingerprint=str(tier1_plan.request_fingerprint),
                provider=tier1_plan.provider,
                model=tier1_plan.model,
                backend=tier1_plan.backend,
                run_id=run_id,
                metadata=tier1_plan.metadata,
            )
            stage_decision = await prepare_stage_execution(
                pool=pool,
                llm=None,
                task_name=None,
                artifact_type=None,
                artifact_id=None,
                review_id=review_id,
                stage_id="b2b_enrichment.tier1",
                work_fingerprint=str(tier1_work_fingerprint),
                request_fingerprint=str(tier1_request_fingerprint),
                parse_response_text=_parse_stage_row_result,
                defer_on_submitted=True,
                reconcile_batch=False,
            )
            existing_tier1_stage = stage_decision.stage_row
            if stage_decision.action == "reuse_stage":
                applied = await apply_stage_decision(
                    pool=pool,
                    decision=stage_decision,
                    review_id=review_id,
                    stage_id="b2b_enrichment.tier1",
                    work_fingerprint=str(tier1_work_fingerprint),
                    tier=1,
                    usage_from_stage_row=_stage_usage_from_row,
                    pending_metadata={"tier": 1, "workload": "direct"},
                    success_metadata={"tier": 1, "workload": "direct"},
                    stage_usage_snapshot=_stage_usage_snapshot,
                )
                tier1 = applied.parsed_result if applied is not None else stage_decision.parsed_result
                tier1_model = applied.model if applied is not None else None
                tier1_cache_hit = bool(applied.cache_hit) if applied is not None else False
                tier1_reused_from_stage = True
            elif stage_decision.action == "defer_submitted_stage":
                applied = await apply_stage_decision(
                    pool=pool,
                    decision=stage_decision,
                    review_id=review_id,
                    stage_id="b2b_enrichment.tier1",
                    work_fingerprint=str(tier1_work_fingerprint),
                    tier=1,
                    usage_from_stage_row=_stage_usage_from_row,
                    pending_metadata={"tier": 1, "workload": "direct"},
                    success_metadata={"tier": 1, "workload": "direct"},
                    stage_usage_snapshot=_stage_usage_snapshot,
                )
                await defer_review_transition(
                    row=row,
                    tier="tier1",
                    custom_id=str((applied.custom_id if applied is not None else "") or ""),
                    usage=None,
                    defer_review=lambda target_row, **kwargs: _defer_batch_row(pool, target_row, **kwargs),
                )
                return _finish("deferred")
        if tier1_skill is not None and tier1 is not None:
            stage_reuse_usage = _stage_usage_from_row(existing_tier1_stage, tier=1)
            _accumulate_exact_cache_usage(cache_usage, stage_reuse_usage)
        elif use_openrouter_tier1:
            tier1, tier1_model, tier1_cache_hit = _unpack_stage_result(await asyncio.wait_for(
                _call_openrouter_tier1(
                    payload_json,
                    cfg,
                    include_cache_hit=True,
                    trace_metadata=trace_metadata | {"tier": "tier1"},
                ),
                timeout=full_extraction_timeout,
            ))
        elif tier1_skill is not None:
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
        if tier1_skill is not None and tier1 is not None and existing_tier1_stage is not None and str(existing_tier1_stage.get("state") or "") == "succeeded":
            pass
        elif tier1_cache_hit:
            cache_usage["tier1_exact_cache_hits"] += 1
            cache_usage["exact_cache_hits"] += 1
        elif tier1_model is not None:
            cache_usage["tier1_generated_calls"] += 1
            cache_usage["generated"] += 1
        tier1_stage_usage = _stage_usage_snapshot(
            tier=1,
            cache_hit=bool(tier1_cache_hit),
            generated=tier1 is not None and not bool(tier1_cache_hit),
        )
        if tier1 is None:
            if tier1_request_fingerprint is not None:
                await mark_stage_run(
                    pool,
                    review_id=review_id,
                    stage_id="b2b_enrichment.tier1",
                    work_fingerprint=str(tier1_work_fingerprint),
                    state="failed",
                    backend=_stage_backend_name(
                        batch_enabled=False,
                        provider="openrouter" if use_openrouter_tier1 else "vllm",
                    ),
                    error_code="tier1_empty_result",
                    metadata={"tier": 1, "workload": "direct"},
                    completed=True,
                )
            logger.debug("Tier 1 returned None for %s, deferring to next cycle", review_id)
            await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
            return _finish(False)
        if tier1_request_fingerprint is not None and not tier1_reused_from_stage:
            await mark_stage_run(
                pool,
                review_id=review_id,
                stage_id="b2b_enrichment.tier1",
                work_fingerprint=str(tier1_work_fingerprint),
                state="succeeded",
                result_source="exact_cache" if tier1_cache_hit else "generated",
                backend=_stage_backend_name(
                    batch_enabled=False,
                    provider="openrouter" if use_openrouter_tier1 else "vllm",
                ),
                usage=tier1_stage_usage,
                response_text=_stage_result_text(tier1),
                metadata={"tier": 1, "workload": "direct"},
                completed=True,
            )

        # Tier 2: conditional -- only fire when tier 1 left extraction gaps
        tier2 = None
        tier2_model = None
        tier2_cache_hit = False
        needs_tier2 = _tier1_has_extraction_gaps(tier1, source=row.get("source"))
        tier2_request_fingerprint = None
        tier2_work_fingerprint = None
        existing_tier2_stage = None
        tier2_reused_from_stage = False
        if needs_tier2:
            tier2_skill = registry.get("digest/b2b_churn_extraction_tier2")
            if tier2_skill is not None:
                tier2_provider = "openrouter" if use_openrouter_tier2 else "vllm"
                tier2_model_planned = (
                    cfg.enrichment_tier2_openrouter_model
                    or cfg.enrichment_openrouter_model
                    or "anthropic/claude-haiku-4-5"
                    if use_openrouter_tier2
                    else (cfg.enrichment_tier2_model or cfg.enrichment_tier1_model)
                )
                tier2_payload = dict(payload)
                tier2_plan = build_tier2_stage_plan(
                    row=row,
                    base_payload=tier2_payload,
                    tier1_result=tier1,
                    system_prompt=str(tier2_skill.content or ""),
                    model=str(tier2_model_planned or ""),
                    provider=tier2_provider,
                    batch_enabled=False,
                    run_id=run_id,
                    prepare_stage_request=_prepare_stage_request,
                    prompt_for_content_type=_tier2_system_prompt_for_content_type,
                    max_tokens=cfg.enrichment_tier2_max_tokens,
                    workload="direct",
                )
                tier2_request_fingerprint = tier2_plan.request_fingerprint
                tier2_work_fingerprint = tier2_plan.work_fingerprint
                await ensure_stage_run(
                    pool,
                    review_id=review_id,
                    stage_id="b2b_enrichment.tier2",
                    work_fingerprint=str(tier2_plan.work_fingerprint),
                    request_fingerprint=str(tier2_plan.request_fingerprint),
                    provider=tier2_plan.provider,
                    model=tier2_plan.model,
                    backend=tier2_plan.backend,
                    run_id=run_id,
                    metadata=tier2_plan.metadata,
                )
                stage_decision = await prepare_stage_execution(
                    pool=pool,
                    llm=None,
                    task_name=None,
                    artifact_type=None,
                    artifact_id=None,
                    review_id=review_id,
                    stage_id="b2b_enrichment.tier2",
                    work_fingerprint=str(tier2_work_fingerprint),
                    request_fingerprint=str(tier2_request_fingerprint),
                    parse_response_text=_parse_stage_row_result,
                    defer_on_submitted=True,
                    reconcile_batch=False,
                )
                existing_tier2_stage = stage_decision.stage_row
                if stage_decision.action == "reuse_stage":
                    applied = await apply_stage_decision(
                        pool=pool,
                        decision=stage_decision,
                        review_id=review_id,
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=str(tier2_work_fingerprint),
                        tier=2,
                        usage_from_stage_row=_stage_usage_from_row,
                        pending_metadata={"tier": 2, "workload": "direct"},
                        success_metadata={"tier": 2, "workload": "direct"},
                        stage_usage_snapshot=_stage_usage_snapshot,
                    )
                    tier2 = applied.parsed_result if applied is not None else stage_decision.parsed_result
                    tier2_model = applied.model if applied is not None else None
                    tier2_cache_hit = bool(applied.cache_hit) if applied is not None else False
                    tier2_reused_from_stage = True
                elif stage_decision.action == "defer_submitted_stage":
                    applied = await apply_stage_decision(
                        pool=pool,
                        decision=stage_decision,
                        review_id=review_id,
                        stage_id="b2b_enrichment.tier2",
                        work_fingerprint=str(tier2_work_fingerprint),
                        tier=2,
                        usage_from_stage_row=_stage_usage_from_row,
                        pending_metadata={"tier": 2, "workload": "direct"},
                        success_metadata={"tier": 2, "workload": "direct"},
                        stage_usage_snapshot=_stage_usage_snapshot,
                    )
                    await defer_review_transition(
                        row=row,
                        tier="tier2",
                        custom_id=str((applied.custom_id if applied is not None else "") or ""),
                        usage=cache_usage,
                        defer_review=lambda target_row, **kwargs: _defer_batch_row(pool, target_row, **kwargs),
                    )
                    return _finish("deferred")
        if needs_tier2:
            try:
                if tier2 is not None:
                    stage_reuse_usage = _stage_usage_from_row(existing_tier2_stage, tier=2)
                    _accumulate_exact_cache_usage(cache_usage, stage_reuse_usage)
                elif use_openrouter_tier2:
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
        if needs_tier2 and tier2 is not None and existing_tier2_stage is not None and str(existing_tier2_stage.get("state") or "") == "succeeded":
            pass
        elif tier2_cache_hit:
            cache_usage["tier2_exact_cache_hits"] += 1
            cache_usage["exact_cache_hits"] += 1
        elif tier2_model is not None:
            cache_usage["tier2_generated_calls"] += 1
            cache_usage["generated"] += 1
        if needs_tier2 and tier2_request_fingerprint is not None and not tier2_reused_from_stage:
            tier2_stage_usage = _stage_usage_snapshot(
                tier=2,
                cache_hit=bool(tier2_cache_hit),
                generated=tier2 is not None and not bool(tier2_cache_hit),
            )
            await mark_stage_run(
                pool,
                review_id=review_id,
                stage_id="b2b_enrichment.tier2",
                work_fingerprint=str(tier2_work_fingerprint),
                state="succeeded" if tier2 is not None else "failed",
                result_source="exact_cache" if tier2_cache_hit else ("generated" if tier2 is not None else None),
                backend=_stage_backend_name(
                    batch_enabled=False,
                    provider="openrouter" if use_openrouter_tier2 else "vllm",
                ),
                usage=tier2_stage_usage if tier2 is not None else None,
                response_text=_stage_result_text(tier2),
                error_code=None if tier2 is not None else "tier2_empty_result",
                metadata={"tier": 2, "workload": "direct"},
                completed=True,
            )
        if tier2 is not None:
            model_id = f"hybrid:{tier1_model}+{tier2_model}"
        else:
            model_id = tier1_model or ""

        return _finish(
            await persist_review_transition(
                row=row,
                tier1_result=tier1,
                tier2_result=tier2,
                model_id=model_id,
                usage=cache_usage,
                merge_results=_merge_tier1_tier2,
                persist_review=lambda target_row, result, *, model_id, usage: _persist_enrichment_result(
                    pool,
                    target_row,
                    result,
                    model_id=model_id,
                    max_attempts=max_attempts,
                    run_id=run_id,
                    cache_usage=usage,
                ),
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
    return _domain_build_classify_payload(
        row,
        truncate_length=truncate_length,
        smart_truncate=_smart_truncate,
    )


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
    return _service_normalize_compare_text(value)


def _normalized_name_tokens(value: Any) -> list[str]:
    return _service_normalized_name_tokens(
        value,
        low_fidelity_token_stopwords=_LOW_FIDELITY_TOKEN_STOPWORDS,
    )


def _text_mentions_name(haystack: str, needle: Any) -> bool:
    return _service_text_mentions_name(
        haystack,
        needle,
        low_fidelity_token_stopwords=_LOW_FIDELITY_TOKEN_STOPWORDS,
    )


def _dedupe_reason_codes(codes: list[str]) -> list[str]:
    return _service_dedupe_reason_codes(codes)


def _has_commercial_context(text: str) -> bool:
    return _service_has_commercial_context(
        text,
        low_fidelity_commercial_markers=_LOW_FIDELITY_COMMERCIAL_MARKERS,
    )


def _has_strong_commercial_context(text: str) -> bool:
    return _service_has_strong_commercial_context(
        text,
        low_fidelity_strong_commercial_markers=_LOW_FIDELITY_STRONG_COMMERCIAL_MARKERS,
    )


def _has_technical_context(summary_text: str, combined_text: str) -> bool:
    return _service_has_technical_context(
        summary_text,
        combined_text,
        low_fidelity_technical_patterns=_LOW_FIDELITY_TECHNICAL_PATTERNS,
    )


def _has_consumer_context(text: str) -> bool:
    return _service_has_consumer_context(
        text,
        low_fidelity_consumer_patterns=_LOW_FIDELITY_CONSUMER_PATTERNS,
    )


def _normalized_low_fidelity_noisy_sources() -> set[str]:
    default_raw = B2BChurnConfig.model_fields["enrichment_low_fidelity_noisy_sources"].default
    return _service_normalized_low_fidelity_noisy_sources(
        settings.b2b_churn.enrichment_low_fidelity_noisy_sources,
        default_raw,
    )


def _detect_low_fidelity_reasons(row: dict[str, Any], result: dict[str, Any]) -> list[str]:
    return _service_detect_low_fidelity_reasons(
        row,
        result,
        deps=EnrichmentOutcomePolicyDeps(
            normalized_low_fidelity_noisy_sources=_normalized_low_fidelity_noisy_sources,
            normalize_compare_text=_normalize_compare_text,
            text_mentions_name=_text_mentions_name,
            normalized_name_tokens=_normalized_name_tokens,
            has_commercial_context=_has_commercial_context,
            has_strong_commercial_context=_has_strong_commercial_context,
            has_technical_context=_has_technical_context,
            has_consumer_context=_has_consumer_context,
            dedupe_reason_codes=_dedupe_reason_codes,
            normalize_company_name=normalize_company_name,
        ),
    )


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
    return _service_coerce_bool(value)


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
    return _service_canonical_role_type(value, deps=_buyer_authority_deps())


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
    return _service_canonical_role_level(value, deps=_buyer_authority_deps())


def _combined_source_text(source_row: dict[str, Any] | None) -> str:
    return _service_combined_source_text(source_row)


def _buyer_authority_deps() -> EnrichmentBuyerAuthorityDeps:
    return EnrichmentBuyerAuthorityDeps(
        sanitize_reviewer_title=sanitize_reviewer_title,
        coerce_bool=_coerce_bool,
        coerce_json_dict=_coerce_json_dict,
        contains_any=_contains_any,
        role_type_aliases=_ROLE_TYPE_ALIASES,
        role_level_aliases=_ROLE_LEVEL_ALIASES,
        champion_reviewer_title_pattern=_CHAMPION_REVIEWER_TITLE_PATTERN,
        evaluator_reviewer_title_pattern=_EVALUATOR_REVIEWER_TITLE_PATTERN,
        exec_role_text_pattern=_EXEC_ROLE_TEXT_PATTERN,
        director_role_text_pattern=_DIRECTOR_ROLE_TEXT_PATTERN,
        manager_role_text_pattern=_MANAGER_ROLE_TEXT_PATTERN,
        ic_role_text_pattern=_IC_ROLE_TEXT_PATTERN,
        commercial_decision_text_pattern=_COMMERCIAL_DECISION_TEXT_PATTERN,
        exec_reviewer_title_pattern=_EXEC_REVIEWER_TITLE_PATTERN,
        manager_decision_title_pattern=_MANAGER_DECISION_TITLE_PATTERN,
        economic_buyer_text_patterns=_ECONOMIC_BUYER_TEXT_PATTERNS,
        champion_text_patterns=_CHAMPION_TEXT_PATTERNS,
        evaluator_text_patterns=_EVALUATOR_TEXT_PATTERNS,
        end_user_text_patterns=_END_USER_TEXT_PATTERNS,
        post_purchase_review_sources=set(_POST_PURCHASE_REVIEW_SOURCES),
        post_purchase_usage_patterns=_POST_PURCHASE_USAGE_PATTERNS,
    )


def _timeline_deps() -> EnrichmentTimelineDeps:
    return EnrichmentTimelineDeps(
        contains_any=_contains_any,
        normalize_compare_text=_normalize_compare_text,
        has_commercial_context=_has_commercial_context,
        has_strong_commercial_context=_has_strong_commercial_context,
        has_technical_context=_has_technical_context,
        has_consumer_context=_has_consumer_context,
        normalized_low_fidelity_noisy_sources=_normalized_low_fidelity_noisy_sources,
        text_mentions_name=_text_mentions_name,
        timeline_month_day_re=_TIMELINE_MONTH_DAY_RE,
        timeline_slash_date_re=_TIMELINE_SLASH_DATE_RE,
        timeline_iso_date_re=_TIMELINE_ISO_DATE_RE,
        timeline_explicit_anchor_phrases=_TIMELINE_EXPLICIT_ANCHOR_PHRASES,
        timeline_relative_anchor_re=_TIMELINE_RELATIVE_ANCHOR_RE,
        timeline_contract_event_patterns=_TIMELINE_CONTRACT_EVENT_PATTERNS,
        timeline_decision_deadline_patterns=_TIMELINE_DECISION_DEADLINE_PATTERNS,
        timeline_contract_end_patterns=_TIMELINE_CONTRACT_END_PATTERNS,
        timeline_immediate_patterns=_TIMELINE_IMMEDIATE_PATTERNS,
        timeline_quarter_patterns=_TIMELINE_QUARTER_PATTERNS,
        timeline_year_patterns=_TIMELINE_YEAR_PATTERNS,
        timeline_decision_patterns=_TIMELINE_DECISION_PATTERNS,
        timeline_ambiguous_vendor_tokens=_TIMELINE_AMBIGUOUS_VENDOR_TOKENS,
        timeline_ambiguous_vendor_product_context_patterns=_TIMELINE_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT_PATTERNS,
    )


def _budget_deps() -> EnrichmentBudgetDeps:
    return EnrichmentBudgetDeps(
        contains_any=_contains_any,
        coerce_bool=_coerce_bool,
        normalize_compare_text=_normalize_compare_text,
        normalize_text_list=_normalize_text_list,
        combined_source_text=_combined_source_text,
        normalized_low_fidelity_noisy_sources=_normalized_low_fidelity_noisy_sources,
        text_mentions_name=_text_mentions_name,
        has_commercial_context=_has_commercial_context,
        has_strong_commercial_context=_has_strong_commercial_context,
        has_technical_context=_has_technical_context,
        has_consumer_context=_has_consumer_context,
        timeline_ambiguous_vendor_tokens=_TIMELINE_AMBIGUOUS_VENDOR_TOKENS,
        timeline_ambiguous_vendor_product_context_patterns=_TIMELINE_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT_PATTERNS,
        budget_any_amount_token_re=_BUDGET_ANY_AMOUNT_TOKEN_RE,
        budget_price_per_seat_re=_BUDGET_PRICE_PER_SEAT_RE,
        budget_annual_amount_re=_BUDGET_ANNUAL_AMOUNT_RE,
        budget_currency_token_re=_BUDGET_CURRENCY_TOKEN_RE,
        budget_seat_count_re=_BUDGET_SEAT_COUNT_RE,
        budget_price_increase_re=_BUDGET_PRICE_INCREASE_RE,
        budget_price_increase_detail_re=_BUDGET_PRICE_INCREASE_DETAIL_RE,
        budget_annual_period_patterns=_BUDGET_ANNUAL_PERIOD_PATTERNS,
        budget_monthly_period_patterns=_BUDGET_MONTHLY_PERIOD_PATTERNS,
        budget_noise_patterns=_BUDGET_NOISE_PATTERNS,
        budget_per_unit_patterns=_BUDGET_PER_UNIT_PATTERNS,
        budget_annual_context_patterns=_BUDGET_ANNUAL_CONTEXT_PATTERNS,
        budget_commercial_context_patterns=_BUDGET_COMMERCIAL_CONTEXT_PATTERNS,
    )


def _infer_role_level_from_text(reviewer_title: Any, source_row: dict[str, Any] | None) -> str:
    return _service_infer_role_level_from_text(
        reviewer_title,
        source_row,
        deps=_buyer_authority_deps(),
    )


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
    return _service_infer_decision_maker(
        result,
        source_row,
        deps=_buyer_authority_deps(),
    )


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
    return _service_infer_buyer_role_type(
        buyer_authority,
        reviewer_context,
        reviewer_title,
        source_row,
        deps=_buyer_authority_deps(),
    )


def _is_unknownish(value: Any) -> bool:
    return _service_is_unknownish(value)


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    return _service_coerce_json_dict(value)


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
    return _domain_effective_enrichment_skip_sources()


def _repair_text_blob(source_row: dict[str, Any]) -> str:
    return " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()


def _repair_target_fields(result: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    return _service_repair_target_fields(
        result,
        source_row,
        deps=EnrichmentRepairDeps(
            normalize_text_list=_normalize_text_list,
            normalize_pain_category=_normalize_pain_category,
            contains_any=_contains_any,
            coerce_json_dict=_coerce_json_dict,
            is_unknownish=_is_unknownish,
            trusted_repair_sources=_trusted_repair_sources,
            normalize_company_name=normalize_company_name,
            repair_negative_patterns=_REPAIR_NEGATIVE_PATTERNS,
            repair_competitor_patterns=_REPAIR_COMPETITOR_PATTERNS,
            repair_pricing_patterns=_REPAIR_PRICING_PATTERNS,
            repair_recommend_patterns=_REPAIR_RECOMMEND_PATTERNS,
            repair_feature_gap_patterns=_REPAIR_FEATURE_GAP_PATTERNS,
            repair_timeline_patterns=_REPAIR_TIMELINE_PATTERNS,
            repair_category_shift_patterns=_REPAIR_CATEGORY_SHIFT_PATTERNS,
            repair_currency_re=_REPAIR_CURRENCY_RE,
        ),
    )


def _needs_field_repair(result: dict[str, Any], source_row: dict[str, Any]) -> bool:
    return _service_needs_field_repair(
        result,
        source_row,
        deps=EnrichmentRepairDeps(
            normalize_text_list=_normalize_text_list,
            normalize_pain_category=_normalize_pain_category,
            contains_any=_contains_any,
            coerce_json_dict=_coerce_json_dict,
            is_unknownish=_is_unknownish,
            trusted_repair_sources=_trusted_repair_sources,
            normalize_company_name=normalize_company_name,
            repair_negative_patterns=_REPAIR_NEGATIVE_PATTERNS,
            repair_competitor_patterns=_REPAIR_COMPETITOR_PATTERNS,
            repair_pricing_patterns=_REPAIR_PRICING_PATTERNS,
            repair_recommend_patterns=_REPAIR_RECOMMEND_PATTERNS,
            repair_feature_gap_patterns=_REPAIR_FEATURE_GAP_PATTERNS,
            repair_timeline_patterns=_REPAIR_TIMELINE_PATTERNS,
            repair_category_shift_patterns=_REPAIR_CATEGORY_SHIFT_PATTERNS,
            repair_currency_re=_REPAIR_CURRENCY_RE,
        ),
    )


def _has_structural_gap(result: dict[str, Any]) -> bool:
    return _service_has_structural_gap(
        result,
        deps=EnrichmentRepairDeps(
            normalize_text_list=_normalize_text_list,
            normalize_pain_category=_normalize_pain_category,
            contains_any=_contains_any,
            coerce_json_dict=_coerce_json_dict,
            is_unknownish=_is_unknownish,
            trusted_repair_sources=_trusted_repair_sources,
            normalize_company_name=normalize_company_name,
            repair_negative_patterns=_REPAIR_NEGATIVE_PATTERNS,
            repair_competitor_patterns=_REPAIR_COMPETITOR_PATTERNS,
            repair_pricing_patterns=_REPAIR_PRICING_PATTERNS,
            repair_recommend_patterns=_REPAIR_RECOMMEND_PATTERNS,
            repair_feature_gap_patterns=_REPAIR_FEATURE_GAP_PATTERNS,
            repair_timeline_patterns=_REPAIR_TIMELINE_PATTERNS,
            repair_category_shift_patterns=_REPAIR_CATEGORY_SHIFT_PATTERNS,
            repair_currency_re=_REPAIR_CURRENCY_RE,
        ),
    )


def _apply_structural_repair(
    baseline: dict[str, Any],
    repair: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    return _service_apply_structural_repair(
        baseline,
        repair,
        deps=EnrichmentRepairDeps(
            normalize_text_list=_normalize_text_list,
            normalize_pain_category=_normalize_pain_category,
            contains_any=_contains_any,
            coerce_json_dict=_coerce_json_dict,
            is_unknownish=_is_unknownish,
            trusted_repair_sources=_trusted_repair_sources,
            normalize_company_name=normalize_company_name,
            repair_negative_patterns=_REPAIR_NEGATIVE_PATTERNS,
            repair_competitor_patterns=_REPAIR_COMPETITOR_PATTERNS,
            repair_pricing_patterns=_REPAIR_PRICING_PATTERNS,
            repair_recommend_patterns=_REPAIR_RECOMMEND_PATTERNS,
            repair_feature_gap_patterns=_REPAIR_FEATURE_GAP_PATTERNS,
            repair_timeline_patterns=_REPAIR_TIMELINE_PATTERNS,
            repair_category_shift_patterns=_REPAIR_CATEGORY_SHIFT_PATTERNS,
            repair_currency_re=_REPAIR_CURRENCY_RE,
        ),
    )


def _apply_field_repair(
    baseline: dict[str, Any],
    repair: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    return _service_apply_field_repair(
        baseline,
        repair,
        deps=EnrichmentRepairDeps(
            normalize_text_list=_normalize_text_list,
            normalize_pain_category=_normalize_pain_category,
            contains_any=_contains_any,
            coerce_json_dict=_coerce_json_dict,
            is_unknownish=_is_unknownish,
            trusted_repair_sources=_trusted_repair_sources,
            normalize_company_name=normalize_company_name,
            repair_negative_patterns=_REPAIR_NEGATIVE_PATTERNS,
            repair_competitor_patterns=_REPAIR_COMPETITOR_PATTERNS,
            repair_pricing_patterns=_REPAIR_PRICING_PATTERNS,
            repair_recommend_patterns=_REPAIR_RECOMMEND_PATTERNS,
            repair_feature_gap_patterns=_REPAIR_FEATURE_GAP_PATTERNS,
            repair_timeline_patterns=_REPAIR_TIMELINE_PATTERNS,
            repair_category_shift_patterns=_REPAIR_CATEGORY_SHIFT_PATTERNS,
            repair_currency_re=_REPAIR_CURRENCY_RE,
        ),
    )


def _validate_enrichment(result: dict, source_row: dict[str, Any] | None = None) -> bool:
    return _service_validate_enrichment(
        result,
        source_row,
        deps=EnrichmentValidationDeps(
            coerce_bool=_coerce_bool,
            normalize_pain_category=_normalize_pain_category,
            normalize_budget_value_text=_normalize_budget_value_text,
            normalize_budget_detail_text=_normalize_budget_detail_text,
            canonical_role_type=_canonical_role_type,
            canonical_role_level=_canonical_role_level,
            infer_role_level_from_text=_infer_role_level_from_text,
            infer_decision_maker=_infer_decision_maker,
            infer_buyer_role_type=_infer_buyer_role_type,
            coerce_json_dict=_coerce_json_dict,
            schema_version=_schema_version,
            missing_witness_primitives=_missing_witness_primitives,
            compute_derived_fields=_compute_derived_fields,
            trusted_reviewer_company_name=_trusted_reviewer_company_name,
            churn_signal_bool_fields=_CHURN_SIGNAL_BOOL_FIELDS,
            known_severity_levels=_KNOWN_SEVERITY_LEVELS,
            known_lock_in_levels=_KNOWN_LOCK_IN_LEVELS,
            known_sentiment_directions=_KNOWN_SENTIMENT_DIRECTIONS,
            known_buying_stages=_KNOWN_BUYING_STAGES,
            known_decision_timelines=_KNOWN_DECISION_TIMELINES,
            known_contract_value_signals=_KNOWN_CONTRACT_VALUE_SIGNALS,
            known_replacement_modes=_KNOWN_REPLACEMENT_MODES,
            known_operating_model_shifts=_KNOWN_OPERATING_MODEL_SHIFTS,
            known_productivity_delta_claims=_KNOWN_PRODUCTIVITY_DELTA_CLAIMS,
            known_org_pressure_types=_KNOWN_ORG_PRESSURE_TYPES,
            known_content_types=_KNOWN_CONTENT_TYPES,
            known_org_health_levels=_KNOWN_ORG_HEALTH_LEVELS,
            known_leadership_qualities=_KNOWN_LEADERSHIP_QUALITIES,
            known_innovation_climates=_KNOWN_INNOVATION_CLIMATES,
            known_morale_levels=_KNOWN_MORALE_LEVELS,
            known_departure_types=_KNOWN_DEPARTURE_TYPES,
            known_pain_categories=_KNOWN_PAIN_CATEGORIES,
        ),
    )


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
