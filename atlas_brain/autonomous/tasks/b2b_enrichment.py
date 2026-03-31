"""
B2B review enrichment: extract churn signals from pending reviews via LLM
using the b2b_churn_extraction skill.

Single-pass enrichment (one LLM call per review). Polls b2b_reviews WHERE
enrichment_status = 'pending', calls LLM, stores result in enrichment JSONB
column, sets status to 'enriched'.

Runs on an interval (default 5 min). Returns _skip_synthesis so the
runner does not double-synthesize.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...services.company_normalization import normalize_company_name
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._b2b_witnesses import (
    derive_evidence_spans,
    derive_operating_model_shift,
    derive_org_pressure_type,
    derive_productivity_delta_claim,
    derive_replacement_mode,
    derive_salience_flags,
)

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")

_TIER1_JSON_SCHEMA: dict[str, Any] = {
    "title": "b2b_churn_extraction",
    "type": "object",
    "additionalProperties": True,
}

def _get_base_enrichment_llm(local_only: bool):
    """Resolve the deterministic local enrichment model from vLLM only."""
    from ...pipelines.llm import get_pipeline_llm

    return get_pipeline_llm(
        workload="vllm",
        try_openrouter=False,
        auto_activate_ollama=False,
    )


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


async def _call_vllm_tier1(payload_json: str, cfg, client) -> tuple[dict | None, str | None]:
    """Tier 1 extraction: deterministic fields via local vLLM.

    Returns (result_dict, model_id) or (None, None) on failure.
    """
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier1")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier1' not found")
        return None, None

    model_id = cfg.enrichment_tier1_model
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": cfg.enrichment_tier1_model,
                "messages": [
                    {"role": "system", "content": skill.content},
                    {"role": "user", "content": payload_json},
                ],
                "max_tokens": cfg.enrichment_tier1_max_tokens,
                "temperature": 0.0,
                "guided_json": _TIER1_JSON_SCHEMA,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if not text:
            return None, model_id

        from ...pipelines.llm import clean_llm_output, parse_json_response
        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            return parsed, model_id
        return None, model_id
    except (json.JSONDecodeError, ValueError):
        logger.warning("Tier 1 vLLM returned invalid JSON")
        return None, model_id
    except Exception:
        logger.exception("Tier 1 vLLM call failed")
        return None, None


async def _call_openrouter_tier1(payload_json: str, cfg) -> tuple[dict | None, str | None]:
    """Tier 1 extraction via OpenRouter (cloud model, no guided_json)."""
    import httpx
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier1")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier1' not found")
        return None, None

    api_key = cfg.openrouter_api_key
    if not api_key:
        logger.warning("OpenRouter API key not configured for enrichment")
        return None, None

    model_id = cfg.enrichment_openrouter_model or "openai/gpt-oss-120b"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": skill.content},
                        {"role": "user", "content": payload_json},
                    ],
                    "max_tokens": max(cfg.enrichment_tier1_max_tokens, 4096),
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            body = resp.json()
            choices = body.get("choices") or []
            if not choices:
                logger.warning("OpenRouter returned no choices")
                return None, model_id
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
                return None, model_id

            from ...pipelines.llm import clean_llm_output, parse_json_response
            text = clean_llm_output(text)
            parsed = parse_json_response(text, recover_truncated=True)
            if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                return parsed, model_id
            logger.warning("OpenRouter tier 1 returned unparseable JSON")
            return None, model_id
    except Exception:
        logger.exception("OpenRouter tier 1 call failed")
        return None, None


def _tier1_has_extraction_gaps(tier1: dict) -> bool:
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
) -> tuple[dict | None, str | None]:
    """Tier 2 extraction: classify + extract via local vLLM.

    Receives Tier 1 output as context so it can reference extracted complaints
    and quotes when classifying pain and detecting indicators.
    Returns (result_dict, model_id) or (None, None) on failure.
    """
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier2' not found")
        return None, None

    tier2_model = cfg.enrichment_tier2_model or cfg.enrichment_tier1_model
    payload = _build_classify_payload(row, truncate_length)
    # Inject Tier 1 extractions for Tier 2 to reference
    payload["tier1_specific_complaints"] = tier1_result.get("specific_complaints", [])
    payload["tier1_quotable_phrases"] = tier1_result.get("quotable_phrases", [])
    payload_json = json.dumps(payload)

    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": tier2_model,
                "messages": [
                    {"role": "system", "content": skill.content},
                    {"role": "user", "content": payload_json},
                ],
                "max_tokens": cfg.enrichment_tier2_max_tokens,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if not text:
            return None, tier2_model

        from ...pipelines.llm import clean_llm_output, parse_json_response
        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            return parsed, tier2_model
        return None, tier2_model
    except Exception:
        logger.warning("Tier 2 vLLM call failed", exc_info=True)
        return None, None


def _get_tier2_client(cfg: Any) -> Any:
    """Get or create the httpx client for Tier 2 vLLM."""
    tier2_url = cfg.enrichment_tier2_vllm_url or cfg.enrichment_tier1_vllm_url
    timeout = cfg.enrichment_tier2_timeout_seconds
    # Reuse the Tier 1 client if same URL
    if tier2_url == cfg.enrichment_tier1_vllm_url:
        return _get_tier1_client(cfg)
    import httpx
    return httpx.AsyncClient(base_url=tier2_url, timeout=timeout)


async def _call_openrouter_tier2(
    tier1_result: dict,
    row: dict,
    cfg: Any,
    truncate_length: int,
) -> tuple[dict | None, str | None]:
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
        return None, None

    api_key = cfg.openrouter_api_key
    if not api_key:
        logger.warning("OpenRouter API key not configured for tier 2 enrichment")
        return None, None

    model_id = (
        cfg.enrichment_tier2_openrouter_model
        or cfg.enrichment_openrouter_model
        or "anthropic/claude-haiku-4-5"
    )
    payload = _build_classify_payload(row, truncate_length)
    payload["tier1_specific_complaints"] = tier1_result.get("specific_complaints", [])
    payload["tier1_quotable_phrases"] = tier1_result.get("quotable_phrases", [])
    payload_json = json.dumps(payload)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(cfg.enrichment_tier2_timeout_seconds, connect=10.0)) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": skill.content},
                        {"role": "user", "content": payload_json},
                    ],
                    "max_tokens": cfg.enrichment_tier2_max_tokens,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            body = resp.json()
            choices = body.get("choices") or []
            if not choices:
                logger.warning("OpenRouter tier 2 returned no choices")
                return None, model_id
            text = (choices[0].get("message") or {}).get("content") or ""
            text = text.strip()
            if not text:
                logger.warning("OpenRouter tier 2 returned empty content")
                return None, model_id
            from ...pipelines.llm import clean_llm_output, parse_json_response
            text = clean_llm_output(text)
            parsed = parse_json_response(text, recover_truncated=True)
            if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                return parsed, model_id
            logger.warning("OpenRouter tier 2 returned unparseable JSON")
            return None, model_id
    except Exception:
        logger.warning("OpenRouter tier 2 call failed", exc_info=True)
        return None, None


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


_PAIN_KEYWORDS = {
    "pricing": (
        "price", "pricing", "cost", "expensive", "overpriced", "value", "renewal",
        "invoice", "invoiced", "billing", "billed", "charged", "charge", "overcharge",
        "fee", "fees", "refund", "cost increase", "price increase",
    ),
    "support": ("support", "ticket", "response", "customer service", "escalat"),
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
}


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
    scores = {category: 0 for category in _PAIN_KEYWORDS}
    for text in texts:
        lowered = text.lower()
        for category, needles in _PAIN_KEYWORDS.items():
            if any(needle in lowered for needle in needles):
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


def _derive_pain_categories(result: dict) -> list[dict[str, str]]:
    texts = (
        _normalize_text_list(result.get("specific_complaints"))
        + _normalize_text_list(result.get("pricing_phrases"))
        + _normalize_text_list(result.get("feature_gaps"))
        + _normalize_text_list(result.get("quotable_phrases"))
    )
    if not texts:
        return []
    scored = _pain_scores(texts)
    ranked = [(score, category) for category, score in scored.items() if score > 0]
    ranked.sort(reverse=True)
    if not ranked:
        return [{"category": "overall_dissatisfaction", "severity": "primary"}]
    categories = [{"category": ranked[0][1], "severity": "primary"}]
    for _score, category in ranked[1:3]:
        if category != categories[0]["category"]:
            categories.append({"category": category, "severity": "secondary"})
    return categories


_COMPETITOR_RECOVERY_PATTERNS = (
    r"\b(?:switched to|moved to|replaced with|migrating to|migration to)\s+([A-Z][A-Za-z0-9.&+/\-]*(?:\s+[A-Z][A-Za-z0-9.&+/\-]*){0,3})",
    r"\b(?:evaluating|looking at|considering|shortlisting|shortlisted|poc with|proof of concept with)\s+([A-Z][A-Za-z0-9.&+/\-]*(?:\s+[A-Z][A-Za-z0-9.&+/\-]*){0,3})",
)

_COMPETITOR_RECOVERY_BLOCKLIST = {
    "a", "an", "the", "another tool", "another vendor", "other tool", "other vendor",
    "new tool", "new vendor", "options", "alternative", "alternatives",
    "alternative platform", "alternative platforms", "crm",
}

_GENERIC_COMPETITOR_TOKENS = {
    "alternative", "alternatives", "platform", "platforms", "tool", "tools",
    "vendor", "vendors", "software", "solutions", "solution", "service",
    "services", "system", "systems", "crm", "suite", "app", "apps",
}


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
        comp_blob = " ".join(
            [name]
            + _normalize_text_list(comp.get("features"))
            + [str(comp.get("reason_detail") or "")]
        ).lower()
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
        comps.append(merged)
    return comps


def _derive_decision_timeline(result: dict) -> str:
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
    if _contains_any(text, ("asap", "immediately", "right away", "this week", "today", "urgent")):
        return "immediate"
    if _contains_any(text, ("next quarter", "this quarter", "q1", "q2", "q3", "q4", "30 days", "60 days", "90 days")):
        return "within_quarter"
    if _contains_any(text, ("this year", "next year", "12 months", "end of year", "2026", "2027")):
        return "within_year"
    return "unknown"


def _extract_numeric_amount(value: Any) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


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
    else:
        buying_stage = "post_purchase"
    return role_type, executive_sponsor_mentioned, buying_stage


def _derive_urgency_indicators(result: dict, source_row: dict[str, Any]) -> dict[str, bool]:
    churn = result.get("churn_signals") or {}
    budget = result.get("budget_signals") or {}
    timeline = result.get("timeline") or {}
    competitors = result.get("competitors_mentioned") or []
    complaints = result.get("specific_complaints") or []
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    pricing_phrases = result.get("pricing_phrases") or []
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
        "price_pressure_language": bool(result.get("pricing_phrases")) or _contains_any(
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

    complaints = result.get("specific_complaints", [])
    quotable = result.get("quotable_phrases", [])
    pricing_phrases = result.get("pricing_phrases", [])
    rec_lang = result.get("recommendation_language", [])
    events = result.get("event_mentions", [])
    budget = result.get("budget_signals", {})
    reviewer = result.get("reviewer_context", {})

    # 0. deterministic replacements for deprecated Tier 2 classify path
    result["pain_categories"] = _derive_pain_categories(result)
    result["competitors_mentioned"] = _recover_competitor_mentions(result, source_row)
    result["competitors_mentioned"] = _derive_competitor_annotations(result, source_row)

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
    timeline["decision_timeline"] = _derive_decision_timeline(result)

    cc = result.get("contract_context")
    if not isinstance(cc, dict):
        cc = {}
        result["contract_context"] = cc
    cc["contract_value_signal"] = _derive_contract_value_signal(result)
    result["urgency_indicators"] = _derive_urgency_indicators(result, source_row)

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
        complaints,
        quotable,
        _normalize_text_list(result.get("pricing_phrases")),
        _normalize_text_list(result.get("feature_gaps")),
        _normalize_text_list(result.get("recommendation_language")),
    )

    # 3. would_recommend
    result["would_recommend"] = engine.derive_recommend(rec_lang, rating, rating_max)

    # 4. sentiment_trajectory.direction -- derived deterministically from rating,
    #    churn signals, and would_recommend. "declining" / "improving" require
    #    multi-review time context and are left for future cross-review jobs;
    #    per-review we classify as positive, negative, or unknown.
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

    # 5. sentiment_trajectory.turning_point from event_mentions
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
    cc["price_complaint"] = engine.derive_price_complaint(result)
    cc["price_context"] = pricing_phrases[0] if pricing_phrases else None

    # 8. witness-oriented deterministic evidence primitives
    result["replacement_mode"] = derive_replacement_mode(result, source_row)
    result["operating_model_shift"] = derive_operating_model_shift(result, source_row)
    result["productivity_delta_claim"] = derive_productivity_delta_claim(source_row)
    result["org_pressure_type"] = derive_org_pressure_type(source_row)
    result["salience_flags"] = derive_salience_flags(result, source_row)
    result["evidence_spans"] = derive_evidence_spans(result, source_row)

    # Mark schema version + evidence map hash for recomputation tracking
    result["enrichment_schema_version"] = 3
    result["evidence_map_hash"] = engine.map_hash

    return result


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
    try:
        coerced = int(raw_value)
    except (TypeError, ValueError):
        return default_value
    return max(min_value, min(max_value, coerced))


async def _enrich_rows(
    rows,
    cfg,
    pool,
    *,
    concurrency_override: int | None = None,
) -> dict[str, Any]:
    """Enrich a list of claimed rows concurrently."""
    max_attempts = cfg.enrichment_max_attempts

    effective_concurrency = max(1, int(concurrency_override or cfg.enrichment_concurrency))
    sem = asyncio.Semaphore(effective_concurrency)

    async def _bounded_enrich(row):
        async with sem:
            return await _enrich_single(pool, row, max_attempts, cfg.enrichment_local_only,
                                        cfg.enrichment_max_tokens, cfg.review_truncate_length)

    results = await asyncio.gather(
        *[_bounded_enrich(row) for row in rows],
        return_exceptions=True,
    )

    for row, result in zip(rows, results):
        if isinstance(result, Exception):
            logger.error("Unexpected enrichment error for %s: %s", row["id"], result, exc_info=result)

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
    default_max_batch = min(cfg.enrichment_max_per_batch, 500)
    max_batch = _coerce_int_override(
        task_metadata.get("enrichment_max_per_batch"),
        default_max_batch,
        min_value=1,
        max_value=500,
    )
    max_attempts = cfg.enrichment_max_attempts
    default_max_rounds = max(1, cfg.enrichment_max_rounds_per_run)
    max_rounds = _coerce_int_override(
        task_metadata.get("enrichment_max_rounds_per_run"),
        default_max_rounds,
        min_value=1,
        max_value=100,
    )
    effective_concurrency = _coerce_int_override(
        task_metadata.get("enrichment_concurrency"),
        max(1, cfg.enrichment_concurrency),
        min_value=1,
        max_value=100,
    )
    inter_batch_delay = max(
        0.0,
        float(
            task_metadata.get(
                "enrichment_inter_batch_delay_seconds",
                cfg.enrichment_inter_batch_delay_seconds,
            )
        ),
    )
    priority_sources = [
        source.strip().lower()
        for source in str(cfg.enrichment_priority_sources or "").split(",")
        if source.strip()
    ]

    total_enriched = 0
    total_failed = 0
    total_no_signal = 0
    total_quarantined = 0
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
        )
        total_enriched += result.get("enriched", 0)
        batch_failed = result.get("failed", 0)
        total_failed += batch_failed
        total_no_signal += result.get("no_signal", 0)
        total_quarantined += result.get("quarantined", 0)
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

    result = {
        "enriched": total_enriched,
        "quarantined": total_quarantined,
        "failed": total_failed,
        "no_signal": total_no_signal,
        "rounds": rounds,
        "orphaned_requeued": orphaned,
        "exhausted_marked_failed": exhausted,
        "_skip_synthesis": "B2B enrichment complete",
    }
    if requeued:
        result["version_upgrade_requeued"] = requeued

    # Record enrichment run summary
    from ..visibility import record_attempt, emit_event
    total_processed = total_enriched + total_quarantined + total_failed + total_no_signal
    await record_attempt(
        pool, artifact_type="enrichment", artifact_id="batch",
        run_id=str(task.id), stage="enrichment",
        status="succeeded" if total_failed == 0 else "failed",
        score=total_enriched,
        blocker_count=total_failed,
        warning_count=total_quarantined,
        error_message=f"{total_failed} failed, {total_quarantined} quarantined" if total_failed else None,
    )
    if total_failed > 0 or total_quarantined > 0:
        await emit_event(
            pool, stage="extraction", event_type="enrichment_run_summary",
            entity_type="pipeline", entity_id="enrichment",
            summary=f"Enrichment: {total_enriched} enriched, {total_failed} failed, {total_quarantined} quarantined",
            severity="warning" if total_failed > 0 else "info",
            actionable=total_failed > 5,
            run_id=str(task.id),
            reason_code="enrichment_failures" if total_failed > 0 else "enrichment_quarantines",
            detail={"enriched": total_enriched, "failed": total_failed,
                    "quarantined": total_quarantined, "no_signal": total_no_signal},
        )

    return result


_MIN_REVIEW_TEXT_LENGTH = 80  # Skip LLM calls for reviews shorter than this

# Verified review platforms -- every review gets full extraction (skip triage).


async def _enrich_single(pool, row, max_attempts: int, local_only: bool,
                         max_tokens: int, truncate_length: int = 3000) -> bool:
    """Enrich a single B2B review with churn signals. Returns True on success."""
    review_id = row["id"]

    # Skip reviews with insufficient text -- title-only scrapes can't yield 47 fields
    review_text = row.get("review_text") or ""
    if len(review_text) < _MIN_REVIEW_TEXT_LENGTH:
        await pool.execute(
            "UPDATE b2b_reviews SET enrichment_status = 'not_applicable' WHERE id = $1",
            review_id,
        )
        return False

    source = str(row.get("source") or "").strip().lower()
    skip_sources = {
        item.strip().lower()
        for item in str(settings.b2b_churn.enrichment_skip_sources or "").split(",")
        if item.strip()
    }
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
        return False

    try:
        cfg = settings.b2b_churn
        full_extraction_timeout = cfg.enrichment_full_extraction_timeout_seconds
        payload = _build_classify_payload(row, truncate_length)
        payload_json = json.dumps(payload)
        client = _get_tier1_client(cfg)

        # Tier 1: deterministic extraction (base fields)
        # Use OpenRouter if configured and not forced local-only, otherwise local vLLM
        use_openrouter = (
            not local_only
            and bool(cfg.enrichment_openrouter_model)
            and bool(cfg.openrouter_api_key)
        )
        if use_openrouter:
            tier1, tier1_model = await asyncio.wait_for(
                _call_openrouter_tier1(payload_json, cfg),
                timeout=full_extraction_timeout,
            )
        else:
            tier1, tier1_model = await asyncio.wait_for(
                _call_vllm_tier1(payload_json, cfg, client),
                timeout=full_extraction_timeout,
            )
        if tier1 is None:
            logger.debug("Tier 1 returned None for %s, deferring to next cycle", review_id)
            await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
            return False

        # Tier 2: conditional -- only fire when tier 1 left extraction gaps
        tier2 = None
        tier2_model = None
        if _tier1_has_extraction_gaps(tier1):
            if use_openrouter:
                tier2, tier2_model = await asyncio.wait_for(
                    _call_openrouter_tier2(tier1, row, cfg, truncate_length),
                    timeout=full_extraction_timeout,
                )
            else:
                tier2_client = _get_tier2_client(cfg)
                tier2, tier2_model = await asyncio.wait_for(
                    _call_vllm_tier2(tier1, row, cfg, tier2_client, truncate_length),
                    timeout=full_extraction_timeout,
                )
        if tier2 is not None:
            model_id = f"hybrid:{tier1_model}+{tier2_model}"
        else:
            model_id = tier1_model or ""

        result = _merge_tier1_tier2(tier1, tier2)

        # Layer 3: compute derived fields from indicators (urgency, pain, recommend, etc.)
        # Hard-fail if compute breaks -- do NOT fall back to model-dependent output.
        if result:
            try:
                result = _compute_derived_fields(result, row)
            except Exception:
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
                )
                return "quarantined"

        if result and _validate_enrichment(result, row):
            # Extract sentiment_trajectory subfields for indexed columns
            st = result.get("sentiment_trajectory") or {}
            st_direction = st.get("direction") if isinstance(st, dict) else None
            st_tenure = st.get("tenure") if isinstance(st, dict) else None
            st_turning = st.get("turning_point") if isinstance(st, dict) else None
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
                )

            # Fire ntfy notification for high-urgency signals (must never
            # break enrichment -- wrapped in its own try/except)
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

            # Backfill reviewer_company from LLM extraction when parser left it empty
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
            except Exception:
                logger.debug("Company name backfill failed for %s (non-fatal)", review_id)

            if target_status == "quarantined":
                return "quarantined"
            return True
        else:
            await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
            return False

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
        return False


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

    return {
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
        "pros": row["pros"] or "",
        "cons": row["cons"] or "",
        "reviewer_title": row["reviewer_title"] or "",
        "reviewer_company": row["reviewer_company"] or "",
        "company_size_raw": row["company_size_raw"] or "",
        "reviewer_industry": row["reviewer_industry"] or "",
    }


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


def _detect_low_fidelity_reasons(row: dict[str, Any], result: dict[str, Any]) -> list[str]:
    source = str(row.get("source") or "").strip().lower()
    noisy_sources = {
        item.strip().lower()
        for item in str(settings.b2b_churn.enrichment_low_fidelity_noisy_sources or "").split(",")
        if item.strip()
    }
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
        if source in {"twitter", "quora"} and len(combined_norm) < 120:
            reasons.append("thin_social_context")
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

_NOISY_REVIEWER_TITLE_PATTERNS = (
    re.compile(r"^repeat churn signal", re.I),
    re.compile(r"score:\s*\d", re.I),
)
_EXEC_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(vp\b|vice president|director|head of|chief|cfo|ceo|coo|cio|cto|cro|cmo|founder|owner|president)\b",
    re.I,
)
_CHAMPION_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(manager|lead|supervisor|coordinator)\b",
    re.I,
)
_EVALUATOR_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(analyst|architect|engineer|developer|administrator|admin|consultant|specialist)\b",
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


def _canonical_role_type(value: Any) -> str:
    raw = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    if not raw:
        return "unknown"
    return _ROLE_TYPE_ALIASES.get(raw, "unknown")


def _clean_reviewer_title_for_role_inference(value: Any) -> str:
    title = str(value or "").strip()
    if not title or len(title) > 120:
        return ""
    lowered = title.lower()
    if any(pattern.search(lowered) for pattern in _NOISY_REVIEWER_TITLE_PATTERNS):
        return ""
    return title


def _canonical_role_level(value: Any) -> str:
    raw = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    if raw in {"executive", "exec", "csuite", "cxo"}:
        return "executive"
    if raw in {"director", "vp", "vicepresident", "head"}:
        return "director"
    if raw in {"manager", "lead", "teamlead", "supervisor", "coordinator"}:
        return "manager"
    if raw in {"ic", "individualcontributor", "individual", "user"}:
        return "ic"
    return "unknown"


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
    "considering", "alternative to", " vs ",
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
    return {
        source.strip().lower()
        for source in str(settings.b2b_churn.enrichment_priority_sources or "").split(",")
        if source.strip()
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
            if "seat_count" in bs and bs["seat_count"] is not None:
                try:
                    seat = int(bs["seat_count"])
                    bs["seat_count"] = seat if 1 <= seat <= 1_000_000 else None
                except (ValueError, TypeError):
                    bs["seat_count"] = None
            if "price_increase_mentioned" in bs:
                coerced = _coerce_bool(bs["price_increase_mentioned"])
                bs["price_increase_mentioned"] = coerced if coerced is not None else False

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
    if decision_maker is None:
        decision_maker = role_level in {"executive", "director"}
    reviewer_ctx["decision_maker"] = decision_maker

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
        if canonical_rt == "unknown":
            ba["role_type"] = _infer_buyer_role_type(
                ba,
                reviewer_ctx,
                (source_row or {}).get("reviewer_title"),
                source_row,
            )
        else:
            ba["role_type"] = canonical_rt
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
