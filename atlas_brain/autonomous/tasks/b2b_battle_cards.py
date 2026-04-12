"""Follow-up task: build and LLM-enrich battle cards per vendor.

Runs after b2b_churn_core. Reads persisted artifacts from
b2b_churn_signals, b2b_reviews, and b2b_product_profiles. Builds
deterministic battle cards, runs LLM sales copy generation in parallel,
and persists to b2b_intelligence.
"""

import asyncio
import json
import logging
import re
from datetime import date, datetime
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._execution_progress import _update_execution_progress

logger = logging.getLogger("atlas.tasks.b2b_battle_cards")


def _approx_token_count(text: str) -> int:
    """Approximate token count without relying on a model-specific tokenizer."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _approx_message_input_tokens(messages: list[dict[str, Any]]) -> int:
    """Approximate total input tokens for a prompt message list."""
    total = 0
    for message in messages:
        total += _approx_token_count(str(message.get("role") or ""))
        total += _approx_token_count(str(message.get("content") or ""))
        total += 8
    return total

_STAGE_LOADING_INPUTS = "loading_inputs"
_STAGE_BUILDING = "building_deterministic_cards"
_STAGE_PERSISTING_DETERMINISTIC = "persisting_deterministic"
_STAGE_LLM_OVERLAY = "llm_overlay"
_STAGE_FINALIZING = "finalizing"

_BATTLE_CARD_LLM_FIELDS = (
    "executive_summary",
    "discovery_questions",
    "landmine_questions",
    "objection_handlers",
    "talk_track",
    "recommended_plays",
    "why_they_stay",
)

_BATTLE_CARD_RENDER_INPUT_KEYS = (
    "vendor",
    "category",
    "churn_pressure_score",
    "risk_level",
    "total_reviews",
    "confidence",
    "vendor_weaknesses",
    "customer_pain_quotes",
    "competitor_differentiators",
    "weakness_analysis",
    "competitive_landscape",
    "archetype",
    "synthesis_wedge",
    "synthesis_wedge_label",
    "archetype_risk_level",
    "archetype_key_signals",
    "evidence_depth_warning",
    "objection_data",
    "cross_vendor_battles",
    "category_council",
    "resource_asymmetry",
    "ecosystem_context",
    "high_intent_companies",
    "integration_stack",
    "buyer_authority",
    "account_reasoning",
    "account_pressure_summary",
    "account_pressure_metrics",
    "priority_account_names",
    "keyword_spikes",
    "retention_signals",
    "incumbent_strengths",
    "active_evaluation_deadlines",
    "falsification_conditions",
    "uncertainty_sources",
    "evidence_window",
    "evidence_window_days",
    "reasoning_source",
    "synthesis_schema_version",
    "low_confidence_sections",
    "section_disclaimers",
    "evidence_conclusions",
    "anchor_examples",
    "witness_highlights",
    "reference_ids",
)

_BATTLE_CARD_SALES_COPY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "executive_summary": {"type": "string"},
        "discovery_questions": {"type": "array", "items": {"type": "string"}},
        "landmine_questions": {"type": "array", "items": {"type": "string"}},
        "objection_handlers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "objection": {"type": "string"},
                    "acknowledge": {"type": "string"},
                    "pivot": {"type": "string"},
                    "proof_point": {"type": "string"},
                },
                "required": ["objection", "acknowledge", "pivot", "proof_point"],
            },
        },
        "talk_track": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "opening": {"type": "string"},
                "mid_call_pivot": {"type": "string"},
                "closing": {"type": "string"},
            },
            "required": ["opening", "mid_call_pivot", "closing"],
        },
        "recommended_plays": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "play": {"type": "string"},
                    "target_segment": {"type": "string"},
                    "key_message": {"type": "string"},
                    "timing": {"type": "string"},
                },
                "required": ["play", "target_segment", "key_message", "timing"],
            },
        },
        "why_they_stay": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "strengths": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "area": {"type": "string"},
                            "evidence": {"type": "string"},
                            "how_to_neutralize": {"type": "string"},
                        },
                        "required": ["area", "evidence", "how_to_neutralize"],
                    },
                },
            },
            "required": ["summary", "strengths"],
        },
    },
    "required": [
        "executive_summary",
        "discovery_questions",
        "landmine_questions",
        "objection_handlers",
        "talk_track",
        "recommended_plays",
        "why_they_stay",
    ],
}

_QUALITY_STATUS_SALES_READY = "sales_ready"
_QUALITY_STATUS_NEEDS_REVIEW = "needs_review"
_QUALITY_STATUS_THIN_EVIDENCE = "thin_evidence"
_QUALITY_STATUS_FALLBACK = "deterministic_fallback"
_QUALITY_PHASE_DETERMINISTIC = "deterministic"
_QUALITY_PHASE_FINAL = "final"
_QUALITY_SCHEMA_VERSION = "v1"
_QUALITY_ROLE_HINT_TERMS = (
    "vp", "director", "manager", "head of", "lead", "admin", "ops", "security", "it",
    "evaluator", "champion", "buyer", "cfo", "cio", "cto", "procurement", "finance",
)
_QUALITY_CTA_TERMS = (
    "audit", "workshop", "benchmark", "assessment", "review", "pilot", "session", "discovery",
    "engage", "target", "outreach", "reach out", "position", "approach", "contact",
    "identify", "prioritize", "focus", "deploy", "execute", "run",
)
_QUALITY_GENERIC_TARGET_SEGMENTS = {"all", "all accounts", "any", "general", "unknown"}
_QUALITY_DEFAULT_ACCOUNT_STAGE_TERMS = {"evaluation", "renewal_decision"}
_QUALITY_EVAL_FAMILY_ACCOUNT_COUNT = "account_count"
_QUALITY_EVAL_FAMILY_SIGNAL_VOLUME = "signal_volume"
_QUALITY_EVAL_FAMILY_UNKNOWN = "unknown"
_QUALITY_EVAL_PATH_FAMILY_MAP = {
    "account_pressure_metrics.active_eval_count": _QUALITY_EVAL_FAMILY_ACCOUNT_COUNT,
    "account_reasoning.active_eval_count": _QUALITY_EVAL_FAMILY_ACCOUNT_COUNT,
    "account_reasoning.active_evaluation_count": _QUALITY_EVAL_FAMILY_ACCOUNT_COUNT,
    "account_reasoning.supporting_evidence.active_eval_count": _QUALITY_EVAL_FAMILY_ACCOUNT_COUNT,
    "account_reasoning.supporting_evidence.active_evaluation_count": _QUALITY_EVAL_FAMILY_ACCOUNT_COUNT,
    "active_evaluation_deadlines.count": _QUALITY_EVAL_FAMILY_ACCOUNT_COUNT,
    "timing_metrics.active_eval_signals": _QUALITY_EVAL_FAMILY_SIGNAL_VOLUME,
    "vendor_core_reasoning.segment_playbook.active_eval_signals": _QUALITY_EVAL_FAMILY_SIGNAL_VOLUME,
    "vendor_core_reasoning.segment_playbook.supporting_evidence.active_eval_signals": _QUALITY_EVAL_FAMILY_SIGNAL_VOLUME,
    "vendor_core_reasoning.timing_intelligence.active_eval_signals": _QUALITY_EVAL_FAMILY_SIGNAL_VOLUME,
    "vendor_core_reasoning.timing_intelligence.supporting_evidence.active_eval_signals": _QUALITY_EVAL_FAMILY_SIGNAL_VOLUME,
    "displacement_reasoning.migration_proof.active_evaluation_volume": _QUALITY_EVAL_FAMILY_SIGNAL_VOLUME,
    "displacement_reasoning.migration_proof.supporting_evidence.active_evaluation_volume": _QUALITY_EVAL_FAMILY_SIGNAL_VOLUME,
}
_ANCHOR_SPECIFIC_TIME_TERMS = (
    "renewal",
    "review",
    "planning",
    "quarter",
    "month",
    "week",
    "deadline",
    "window",
    "cycle",
    "decision",
)

def _parse_battle_card_sales_copy(text: str | None) -> dict[str, Any]:
    """Parse battle-card LLM output with truncation recovery enabled."""
    from ...pipelines.llm import parse_json_response

    parsed = parse_json_response(text or "", recover_truncated=True)
    if parsed.get("_parse_fallback"):
        return parsed
    if any(field in parsed for field in _BATTLE_CARD_LLM_FIELDS):
        return parsed
    return {"analysis_text": text or "", "_parse_fallback": True}


def _battle_card_prior_attempt(parsed_copy: dict[str, Any]) -> Any:
    """Convert invalid parse fallbacks into a raw draft for retry prompts."""
    if not isinstance(parsed_copy, dict):
        return parsed_copy
    if not parsed_copy.get("_parse_fallback"):
        return parsed_copy
    raw_text = str(parsed_copy.get("analysis_text") or "").strip()
    return raw_text or {}


def _battle_card_llm_options(cfg: Any) -> dict[str, Any]:
    """Resolve backend-specific call_llm_with_skill options for battle cards."""
    from ...pipelines.llm import normalize_openrouter_model

    backend = str(getattr(cfg, "battle_card_llm_backend", "auto") or "auto").strip().lower()
    model = (
        normalize_openrouter_model(
            getattr(cfg, "battle_card_openrouter_model", ""),
            context="battle card OpenRouter model",
        )
        or normalize_openrouter_model(
            getattr(settings.llm, "openrouter_reasoning_model", ""),
            context="battle card inherited reasoning model",
        )
        or None
    )
    if backend == "anthropic":
        return {
            "workload": "anthropic",
            "try_openrouter": False,
            "openrouter_model": None,
        }
    if backend == "openrouter":
        return {
            "workload": "synthesis",
            "try_openrouter": True,
            "openrouter_model": model,
        }
    return {
        "workload": "synthesis",
        "try_openrouter": True,
        "openrouter_model": model,
    }


def _battle_card_contract(card: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve a reasoning contract from canonical contract storage."""
    contracts = card.get("reasoning_contracts")
    if isinstance(contracts, dict):
        contract = contracts.get(name)
        if isinstance(contract, dict) and contract:
            return contract
    if contracts:
        return {}
    return {}


def _reasoning_int(value: Any) -> int | None:
    """Unwrap a traced numeric contract field into an integer."""
    raw = value.get("value") if isinstance(value, dict) else value
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None


def _battle_card_account_reasoning(card: dict[str, Any]) -> dict[str, Any]:
    """Resolve account reasoning from contracts or raw compatibility fields."""
    contract = _battle_card_contract(card, "account_reasoning")
    if contract:
        return contract
    raw = card.get("account_reasoning")
    return raw if isinstance(raw, dict) else {}


def _battle_card_account_summary_payload(
    account_reasoning: dict[str, Any] | None,
) -> tuple[str, dict[str, int], list[str]]:
    """Derive readable account-pressure fields from account reasoning."""
    if not isinstance(account_reasoning, dict):
        return "", {}, []
    summary = str(account_reasoning.get("market_summary") or "").strip()
    metrics: dict[str, int] = {}
    for key in ("total_accounts", "high_intent_count", "active_eval_count"):
        value = _reasoning_int(account_reasoning.get(key))
        if value is not None:
            metrics[key] = value
    priority_names: list[str] = []
    for item in account_reasoning.get("top_accounts") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name and name not in priority_names:
            priority_names.append(name)
    return summary, metrics, priority_names[:3]


def _promote_account_reasoning_to_battle_card(card: dict[str, Any]) -> None:
    """Surface account reasoning into battle-card fields used by renderers and UI."""
    account_reasoning = _battle_card_account_reasoning(card)
    if not account_reasoning:
        return
    card["account_reasoning"] = account_reasoning
    summary, metrics, priority_names = _battle_card_account_summary_payload(
        account_reasoning,
    )
    if summary and not card.get("account_pressure_summary"):
        card["account_pressure_summary"] = summary
    if metrics and not card.get("account_pressure_metrics"):
        card["account_pressure_metrics"] = metrics
    if priority_names and not card.get("priority_account_names"):
        card["priority_account_names"] = priority_names


def _normalize_stage(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return text.replace("-", "_").replace(" ", "_")


def _quality_required_stages(cfg: Any) -> set[str]:
    raw = getattr(cfg, "battle_card_quality_required_stages", None)
    if not isinstance(raw, list):
        return set(_QUALITY_DEFAULT_ACCOUNT_STAGE_TERMS)
    parsed = {_normalize_stage(item) for item in raw if str(item or "").strip()}
    return parsed or set(_QUALITY_DEFAULT_ACCOUNT_STAGE_TERMS)


def _parse_dateish(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    candidate = text
    if "T" in candidate:
        candidate = candidate.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(candidate).date()
        except ValueError:
            pass
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


def _battle_card_data_as_of_date(card: dict[str, Any]) -> date | None:
    data_as_of = _parse_dateish(card.get("data_as_of_date"))
    if data_as_of:
        return data_as_of
    return _parse_dateish(card.get("review_window_end"))


def _battle_card_eval_signal_values(card: dict[str, Any]) -> list[tuple[str, int]]:
    """Collect active-evaluation signal counts surfaced across contracts and mirrors."""
    values: list[tuple[str, int]] = []

    def _add(path: str, raw: Any) -> None:
        value = _reasoning_int(raw)
        if value is None:
            return
        values.append((path, value))

    metrics = card.get("account_pressure_metrics") if isinstance(card.get("account_pressure_metrics"), dict) else {}
    _add("account_pressure_metrics.active_eval_count", metrics.get("active_eval_count"))
    timing_metrics = card.get("timing_metrics") if isinstance(card.get("timing_metrics"), dict) else {}
    _add("timing_metrics.active_eval_signals", timing_metrics.get("active_eval_signals"))

    deadlines = card.get("active_evaluation_deadlines")
    if isinstance(deadlines, list):
        _add("active_evaluation_deadlines.count", len(deadlines))

    account = _battle_card_account_reasoning(card)
    if account:
        _add("account_reasoning.active_eval_count", account.get("active_eval_count"))
        _add("account_reasoning.active_evaluation_count", account.get("active_evaluation_count"))
        supporting = account.get("supporting_evidence") if isinstance(account.get("supporting_evidence"), dict) else {}
        _add("account_reasoning.supporting_evidence.active_eval_count", supporting.get("active_eval_count"))
        _add("account_reasoning.supporting_evidence.active_evaluation_count", supporting.get("active_evaluation_count"))

    vendor_core = _battle_card_contract(card, "vendor_core_reasoning")
    if vendor_core:
        segment = vendor_core.get("segment_playbook") if isinstance(vendor_core.get("segment_playbook"), dict) else {}
        timing = vendor_core.get("timing_intelligence") if isinstance(vendor_core.get("timing_intelligence"), dict) else {}
        seg_support = segment.get("supporting_evidence") if isinstance(segment.get("supporting_evidence"), dict) else {}
        time_support = timing.get("supporting_evidence") if isinstance(timing.get("supporting_evidence"), dict) else {}
        _add("vendor_core_reasoning.segment_playbook.active_eval_signals", segment.get("active_eval_signals"))
        _add("vendor_core_reasoning.segment_playbook.supporting_evidence.active_eval_signals", seg_support.get("active_eval_signals"))
        _add("vendor_core_reasoning.timing_intelligence.active_eval_signals", timing.get("active_eval_signals"))
        _add("vendor_core_reasoning.timing_intelligence.supporting_evidence.active_eval_signals", time_support.get("active_eval_signals"))

    displacement = _battle_card_contract(card, "displacement_reasoning")
    if displacement:
        migration = displacement.get("migration_proof") if isinstance(displacement.get("migration_proof"), dict) else {}
        migration_support = migration.get("supporting_evidence") if isinstance(migration.get("supporting_evidence"), dict) else {}
        _add("displacement_reasoning.migration_proof.active_evaluation_volume", migration.get("active_evaluation_volume"))
        _add(
            "displacement_reasoning.migration_proof.supporting_evidence.active_evaluation_volume",
            migration_support.get("active_evaluation_volume"),
        )
    return values


def _battle_card_eval_signal_family(path: str) -> str:
    return _QUALITY_EVAL_PATH_FAMILY_MAP.get(path, _QUALITY_EVAL_FAMILY_UNKNOWN)


def _battle_card_eval_signal_families(
    values: list[tuple[str, int]],
) -> dict[str, list[tuple[str, int]]]:
    grouped: dict[str, list[tuple[str, int]]] = {}
    for path, value in values:
        family = _battle_card_eval_signal_family(path)
        grouped.setdefault(family, []).append((path, value))
    return grouped


def _battle_card_primary_weakness_label(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    label = str(row.get("weakness") or row.get("area") or "").strip().lower()
    return label


def _is_generic_other_weakness(label: str) -> bool:
    text = str(label or "").strip().lower()
    if not text:
        return False
    return (
        text == "other"
        or text.startswith("other ")
        or text.startswith("other_")
        or text == "general_dissatisfaction"
        or text.startswith("general_dissatisfaction ")
        or text == "overall_dissatisfaction"
        or text.startswith("overall_dissatisfaction ")
    )


def _prioritize_seller_usable_primary_weakness(card: dict[str, Any]) -> None:
    rows = card.get("weakness_analysis")
    if not isinstance(rows, list) or len(rows) < 2:
        return
    first_label = _battle_card_primary_weakness_label(rows[0])
    if not _is_generic_other_weakness(first_label):
        return
    for idx in range(1, len(rows)):
        candidate = rows[idx]
        candidate_label = _battle_card_primary_weakness_label(candidate)
        if _is_generic_other_weakness(candidate_label):
            continue
        rows[0], rows[idx] = rows[idx], rows[0]
        card["weakness_analysis"] = rows
        return


def _battle_card_has_role_or_account_targeting(play: dict[str, Any], account_names: list[str]) -> bool:
    target_segment = str(play.get("target_segment") or "").strip().lower()
    if target_segment and target_segment not in _QUALITY_GENERIC_TARGET_SEGMENTS:
        return True
    text = " ".join(
        str(play.get(field) or "")
        for field in ("play", "target_segment", "key_message")
    ).strip().lower()
    if not text:
        return False
    if any(term in text for term in _QUALITY_ROLE_HINT_TERMS):
        return True
    for name in account_names:
        normalized = str(name or "").strip().lower()
        if normalized and normalized in text:
            return True
    return False


def _battle_card_has_cta(play: dict[str, Any]) -> bool:
    text = " ".join(
        str(play.get(field) or "")
        for field in ("play", "timing", "key_message")
    ).strip().lower()
    if not text:
        return False
    return any(term in text for term in _QUALITY_CTA_TERMS)


def _battle_card_play_actionability(
    plays: list[Any],
    *,
    account_names: list[str],
) -> tuple[int, list[str]]:
    actionable_play_count = 0
    target_segments: list[str] = []
    for play in plays:
        if not isinstance(play, dict):
            continue
        target_segment = str(play.get("target_segment") or "").strip()
        timing = str(play.get("timing") or "").strip()
        if target_segment:
            target_segments.append(target_segment.lower())
        if not target_segment or not timing:
            continue
        if _battle_card_has_cta(play) and _battle_card_has_role_or_account_targeting(play, account_names):
            actionable_play_count += 1
    return actionable_play_count, target_segments


def _battle_card_quality_status(
    *,
    phase: str,
    hard_blockers: list[str],
    warnings: list[str],
    has_canonical_accounts: bool = False,
) -> str:
    if hard_blockers:
        return _QUALITY_STATUS_FALLBACK
    if phase == _QUALITY_PHASE_FINAL and not warnings:
        return _QUALITY_STATUS_SALES_READY
    # Separate thin-evidence (no accounts, no eval signals) from true
    # needs_review (has data but something needs attention).
    if not has_canonical_accounts and any(
        "no high-intent account" in w
        for w in warnings
    ):
        return _QUALITY_STATUS_THIN_EVIDENCE
    return _QUALITY_STATUS_NEEDS_REVIEW


def _drop_llm_sales_copy(card: dict[str, Any]) -> None:
    """Drop model-generated copy when a card fails hard quality checks."""
    for field in _BATTLE_CARD_LLM_FIELDS:
        card.pop(field, None)


def _battle_card_anchor_examples(card: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw = card.get("anchor_examples")
    if not isinstance(raw, dict):
        return {}
    resolved: dict[str, list[dict[str, Any]]] = {}
    for label, rows in raw.items():
        if not isinstance(rows, list):
            continue
        clean_rows = [dict(row) for row in rows if isinstance(row, dict)]
        if clean_rows:
            resolved[str(label)] = clean_rows
    return resolved


def _battle_card_witness_highlights(card: dict[str, Any]) -> list[dict[str, Any]]:
    raw = card.get("witness_highlights")
    return [dict(row) for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []


def _battle_card_limited_rows(
    rows: Any,
    *,
    limit: int,
    allowed_keys: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Return a bounded list of shallow-copied rows for prompt payloads."""
    if limit <= 0 or not isinstance(rows, list):
        return []
    resolved: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if allowed_keys is not None:
            item = {key: item[key] for key in allowed_keys if key in item}
        if item:
            resolved.append(item)
        if len(resolved) >= limit:
            break
    return resolved


def _battle_card_limited_strings(values: Any, *, limit: int) -> list[str]:
    """Return a bounded list of non-empty strings."""
    if limit <= 0 or not isinstance(values, list):
        return []
    resolved: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        resolved.append(text)
        if len(resolved) >= limit:
            break
    return resolved


def _battle_card_limited_reference_ids(reference_ids: Any) -> dict[str, list[str]]:
    """Return bounded metric/witness reference ids for prompt payloads."""
    if not isinstance(reference_ids, dict):
        return {}
    cfg = settings.b2b_churn
    limit = int(getattr(cfg, "battle_card_render_reference_ids_limit", 12) or 12)
    if limit <= 0:
        return {}
    limited: dict[str, list[str]] = {}
    for key in ("metric_ids", "witness_ids"):
        values = _battle_card_limited_strings(reference_ids.get(key), limit=limit)
        if values:
            limited[key] = values
    return limited


def _battle_card_limited_anchor_examples(card: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return bounded anchor examples for prompt payloads."""
    cfg = settings.b2b_churn
    limit = int(getattr(cfg, "battle_card_render_anchor_examples_per_bucket", 1) or 1)
    if limit <= 0:
        return {}
    allowed_keys = (
        "witness_id",
        "excerpt_text",
        "reviewer_company",
        "time_anchor",
        "numeric_literals",
        "competitor",
        "title",
        "company_size",
        "industry",
    )
    anchors = _battle_card_anchor_examples(card)
    resolved: dict[str, list[dict[str, Any]]] = {}
    for label, rows in anchors.items():
        limited = _battle_card_limited_rows(rows, limit=limit, allowed_keys=allowed_keys)
        if limited:
            resolved[label] = limited
    return resolved


def _battle_card_limited_witness_highlights(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Return bounded witness highlights for prompt payloads."""
    cfg = settings.b2b_churn
    limit = int(getattr(cfg, "battle_card_render_witness_highlights_limit", 4) or 4)
    allowed_keys = (
        "witness_id",
        "excerpt_text",
        "reviewer_company",
        "time_anchor",
        "numeric_literals",
        "competitor",
        "title",
        "company_size",
        "industry",
    )
    return _battle_card_limited_rows(
        _battle_card_witness_highlights(card),
        limit=limit,
        allowed_keys=allowed_keys,
    )


def _battle_card_trim_data_gaps(values: Any) -> list[str]:
    """Return bounded data-gap text for compact reasoning packets."""
    cfg = settings.b2b_churn
    limit = int(getattr(cfg, "battle_card_render_data_gaps_limit", 4) or 4)
    return _battle_card_limited_strings(values, limit=limit)


def _battle_card_compact_vendor_core_reasoning(contract: dict[str, Any]) -> dict[str, Any]:
    """Compact vendor core reasoning to the fields battle-card copy actually uses."""
    if not isinstance(contract, dict):
        return {}
    cfg = settings.b2b_churn
    strengths_limit = int(getattr(cfg, "battle_card_render_strengths_limit", 3) or 3)
    segments_limit = int(getattr(cfg, "battle_card_render_priority_segments_limit", 3) or 3)
    compact: dict[str, Any] = {}
    causal = contract.get("causal_narrative") if isinstance(contract.get("causal_narrative"), dict) else {}
    if causal:
        causal_payload = {
            key: causal.get(key)
            for key in (
                "primary_wedge",
                "trigger",
                "why_now",
                "who_most_affected",
                "causal_chain",
                "confidence",
            )
            if causal.get(key) not in (None, "", [], {})
        }
        data_gaps = _battle_card_trim_data_gaps(causal.get("data_gaps"))
        if data_gaps:
            causal_payload["data_gaps"] = data_gaps
        weakeners = _battle_card_limited_rows(
            causal.get("what_would_weaken_thesis"),
            limit=3,
            allowed_keys=("condition", "monitorable", "signal_source"),
        )
        if weakeners:
            causal_payload["what_would_weaken_thesis"] = weakeners
        if causal_payload:
            compact["causal_narrative"] = causal_payload
    segment = contract.get("segment_playbook") if isinstance(contract.get("segment_playbook"), dict) else {}
    if segment:
        segment_payload = {
            key: segment.get(key)
            for key in ("confidence",)
            if segment.get(key) not in (None, "", [], {})
        }
        data_gaps = _battle_card_trim_data_gaps(segment.get("data_gaps"))
        if data_gaps:
            segment_payload["data_gaps"] = data_gaps
        priority_segments = _battle_card_limited_rows(
            segment.get("priority_segments"),
            limit=segments_limit,
            allowed_keys=(
                "segment",
                "why_now",
                "play",
                "pain",
                "sample_size",
                "buyer_role",
                "company_size",
                "industry",
                "active_eval_signals",
            ),
        )
        if priority_segments:
            segment_payload["priority_segments"] = priority_segments
        supporting = segment.get("supporting_evidence") if isinstance(segment.get("supporting_evidence"), dict) else {}
        support_payload: dict[str, Any] = {}
        top_roles = _battle_card_limited_rows(
            supporting.get("top_roles"),
            limit=segments_limit,
            allowed_keys=("role_type", "top_pain", "top_buying_stage", "review_count", "priority_score"),
        )
        if top_roles:
            support_payload["top_roles"] = top_roles
        top_use_cases = _battle_card_limited_rows(
            supporting.get("top_use_cases"),
            limit=segments_limit,
            allowed_keys=("use_case", "mention_count", "lock_in_level"),
        )
        if top_use_cases:
            support_payload["top_use_cases"] = top_use_cases
        active_eval = supporting.get("active_eval_signals")
        if active_eval not in (None, "", [], {}):
            support_payload["active_eval_signals"] = active_eval
        if support_payload:
            segment_payload["supporting_evidence"] = support_payload
        if segment_payload:
            compact["segment_playbook"] = segment_payload
    timing = contract.get("timing_intelligence") if isinstance(contract.get("timing_intelligence"), dict) else {}
    if timing:
        timing_payload = {
            key: timing.get(key)
            for key in (
                "confidence",
                "best_timing_window",
                "seasonal_pattern",
                "sentiment_direction",
                "active_eval_signals",
            )
            if timing.get(key) not in (None, "", [], {})
        }
        immediate = _battle_card_limited_strings(timing.get("immediate_triggers"), limit=4)
        if immediate:
            timing_payload["immediate_triggers"] = immediate
        data_gaps = _battle_card_trim_data_gaps(timing.get("data_gaps"))
        if data_gaps:
            timing_payload["data_gaps"] = data_gaps
        if timing_payload:
            compact["timing_intelligence"] = timing_payload
    stay = contract.get("why_they_stay") if isinstance(contract.get("why_they_stay"), dict) else {}
    if stay:
        stay_payload = {}
        summary = str(stay.get("summary") or "").strip()
        if summary:
            stay_payload["summary"] = summary
        strengths = _battle_card_limited_rows(
            stay.get("strengths"),
            limit=strengths_limit,
            allowed_keys=("area", "evidence", "neutralization", "how_to_neutralize"),
        )
        if strengths:
            stay_payload["strengths"] = strengths
        if stay_payload:
            compact["why_they_stay"] = stay_payload
    posture = contract.get("confidence_posture") if isinstance(contract.get("confidence_posture"), dict) else {}
    if posture:
        posture_payload = {}
        overall = posture.get("overall")
        if overall not in (None, "", [], {}):
            posture_payload["overall"] = overall
        limits = _battle_card_trim_data_gaps(posture.get("limits"))
        if limits:
            posture_payload["limits"] = limits
        if posture_payload:
            compact["confidence_posture"] = posture_payload
    return compact


def _battle_card_compact_displacement_reasoning(contract: dict[str, Any]) -> dict[str, Any]:
    """Compact displacement reasoning for battle-card prompts."""
    if not isinstance(contract, dict):
        return {}
    cfg = settings.b2b_churn
    reframes_limit = int(getattr(cfg, "battle_card_render_reframes_limit", 3) or 3)
    compact: dict[str, Any] = {}
    for key in (
        "confirmed_switch_count",
        "active_evaluation_count",
        "displacement_mention_volume",
    ):
        value = contract.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    migration = contract.get("migration_proof") if isinstance(contract.get("migration_proof"), dict) else {}
    if migration:
        migration_payload = {
            key: migration.get(key)
            for key in (
                "confidence",
                "evidence_type",
                "switching_is_real",
                "primary_switch_driver",
                "top_destination",
                "switch_volume",
                "active_evaluation_volume",
                "displacement_mention_volume",
                "evaluation_vs_switching",
            )
            if migration.get(key) not in (None, "", [], {})
        }
        data_gaps = _battle_card_trim_data_gaps(migration.get("data_gaps"))
        if data_gaps:
            migration_payload["data_gaps"] = data_gaps
        named_examples = _battle_card_limited_rows(
            migration.get("named_examples"),
            limit=2,
            allowed_keys=("company", "destination", "driver", "proof"),
        )
        if named_examples:
            migration_payload["named_examples"] = named_examples
        if migration_payload:
            compact["migration_proof"] = migration_payload
    reframes = contract.get("competitive_reframes") if isinstance(contract.get("competitive_reframes"), dict) else {}
    if reframes:
        reframe_payload = {
            key: reframes.get(key)
            for key in ("confidence",)
            if reframes.get(key) not in (None, "", [], {})
        }
        data_gaps = _battle_card_trim_data_gaps(reframes.get("data_gaps"))
        if data_gaps:
            reframe_payload["data_gaps"] = data_gaps
        rows = _battle_card_limited_rows(
            reframes.get("reframes"),
            limit=reframes_limit,
            allowed_keys=("opponent", "segment", "message", "proof_point", "when_to_use"),
        )
        if rows:
            reframe_payload["reframes"] = rows
        if reframe_payload:
            compact["competitive_reframes"] = reframe_payload
    switch_triggers = _battle_card_limited_strings(contract.get("switch_triggers"), limit=4)
    if switch_triggers:
        compact["switch_triggers"] = switch_triggers
    top_flows = _battle_card_limited_rows(
        contract.get("top_flows"),
        limit=3,
        allowed_keys=("from_vendor", "to_vendor", "direction", "mention_count"),
    )
    if top_flows:
        compact["top_flows"] = top_flows
    return compact


def _battle_card_compact_category_reasoning(contract: dict[str, Any]) -> dict[str, Any]:
    """Compact category reasoning for battle-card prompts."""
    if not isinstance(contract, dict):
        return {}
    compact = {
        key: contract.get(key)
        for key in (
            "market_regime",
            "narrative",
            "confidence",
            "confidence_score",
            "durability",
            "winner",
            "loser",
            "vendor_count",
            "displacement_flow_count",
            "top_differentiator",
            "top_vulnerability",
        )
        if contract.get(key) not in (None, "", [], {})
    }
    data_gaps = _battle_card_trim_data_gaps(contract.get("data_gaps"))
    if data_gaps:
        compact["data_gaps"] = data_gaps
    key_insights = _battle_card_limited_rows(
        contract.get("key_insights"),
        limit=3,
    )
    if key_insights:
        compact["key_insights"] = key_insights
    outliers = _battle_card_limited_strings(contract.get("outlier_vendors"), limit=3)
    if outliers:
        compact["outlier_vendors"] = outliers
    return compact


def _battle_card_compact_account_reasoning(contract: dict[str, Any]) -> dict[str, Any]:
    """Compact account reasoning for battle-card prompts."""
    if not isinstance(contract, dict):
        return {}
    compact = {
        key: contract.get(key)
        for key in (
            "confidence",
            "market_summary",
            "total_accounts",
            "high_intent_count",
            "active_eval_count",
            "active_evaluation_count",
        )
        if contract.get(key) not in (None, "", [], {})
    }
    data_gaps = _battle_card_trim_data_gaps(contract.get("data_gaps"))
    if data_gaps:
        compact["data_gaps"] = data_gaps
    cfg = settings.b2b_churn
    account_limit = int(getattr(cfg, "battle_card_render_high_intent_companies_limit", 3) or 3)
    top_accounts = _battle_card_limited_rows(
        contract.get("top_accounts"),
        limit=account_limit,
        allowed_keys=("name", "role", "pain", "stage", "urgency", "decision_maker", "contract_end"),
    )
    if top_accounts:
        compact["top_accounts"] = top_accounts
    supporting = contract.get("supporting_evidence") if isinstance(contract.get("supporting_evidence"), dict) else {}
    support_payload = {
        key: supporting.get(key)
        for key in ("active_eval_count", "active_evaluation_count")
        if supporting.get(key) not in (None, "", [], {})
    }
    if support_payload:
        compact["supporting_evidence"] = support_payload
    return compact


def _battle_card_trace_metadata(
    task: ScheduledTask,
    card: dict[str, Any],
    *,
    attempt: int,
) -> dict[str, Any]:
    """Build trace metadata for battle-card sales-copy calls."""
    vendor = str(card.get("vendor") or "").strip()
    return {
        "vendor_name": vendor,
        "run_id": str(task.id),
        "source_name": "b2b_battle_cards",
        "event_type": "llm_overlay",
        "entity_type": "battle_card",
        "entity_id": vendor,
        "attempt_no": attempt,
    }


def _battle_card_render_text(card: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in _BATTLE_CARD_LLM_FIELDS:
        value = card.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(value)
        else:
            parts.append(json.dumps(value, default=str))
    return " ".join(part for part in parts if part).lower()


def _witness_numeric_tokens(witness: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    numeric_literals = witness.get("numeric_literals")
    if isinstance(numeric_literals, dict):
        for values in numeric_literals.values():
            if isinstance(values, list):
                for value in values:
                    token = str(value or "").strip().lower()
                    if token:
                        tokens.add(token)
            else:
                token = str(values or "").strip().lower()
                if token:
                    tokens.add(token)
    if tokens:
        return tokens
    excerpt = str(witness.get("excerpt_text") or "")
    for token in re.findall(r"\$\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?%", excerpt):
        normalized = token.strip().lower()
        if normalized:
            tokens.add(normalized)
    return tokens


def _battle_card_specific_time_anchor(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if re.search(r"\d", lower):
        return text
    if any(term in lower for term in _ANCHOR_SPECIFIC_TIME_TERMS):
        return text
    return ""


def _battle_card_anchor_signal_terms(card: dict[str, Any]) -> dict[str, set[str]]:
    companies: set[str] = set()
    timing_terms: set[str] = set()
    numeric_terms: set[str] = set()
    competitor_terms: set[str] = set()
    for rows in _battle_card_anchor_examples(card).values():
        for witness in rows:
            company = str(witness.get("reviewer_company") or "").strip().lower()
            if company:
                companies.add(company)
            competitor = str(witness.get("competitor") or "").strip().lower()
            if competitor:
                competitor_terms.add(competitor)
            time_anchor = _battle_card_specific_time_anchor(witness.get("time_anchor")).lower()
            if time_anchor:
                timing_terms.add(time_anchor)
            numeric_terms.update(_witness_numeric_tokens(witness))
    return {
        "companies": companies,
        "timing_terms": timing_terms,
        "numeric_terms": numeric_terms,
        "competitor_terms": competitor_terms,
    }


def _battle_card_anchor_phrase(card: dict[str, Any]) -> str:
    """Build a seller-safe anchor phrase from surfaced witness examples."""
    anchors = _battle_card_anchor_examples(card)
    for rows in anchors.values():
        for witness in rows:
            if not isinstance(witness, dict):
                continue
            company = str(witness.get("reviewer_company") or "").strip()
            competitor = str(witness.get("competitor") or "").strip()
            time_anchor = _battle_card_specific_time_anchor(witness.get("time_anchor"))
            parts: list[str] = []
            if company:
                parts.append(f"accounts like {company}")
            if competitor:
                parts.append(f"while evaluating {competitor}")
            if time_anchor:
                parts.append(f"during {time_anchor}")
            if parts:
                return " ".join(parts)
    return ""


def _populate_battle_card_fallback_sales_copy(card: dict[str, Any]) -> None:
    """Fill empty battle-card seller-copy fields from deterministic evidence."""
    from ._b2b_shared import (
        _battle_card_best_supported_quote,
        _battle_card_fallback_recommended_plays,
        _battle_card_quote_terms,
        _battle_card_safe_play_text,
        _battle_card_safe_summary,
        _battle_card_structured_proof_text,
        _battle_card_winning_position,
        _sanitize_battle_card_sales_copy,
    )

    weak_rows = card.get("weakness_analysis") if isinstance(card.get("weakness_analysis"), list) else []
    primary = weak_rows[0] if weak_rows and isinstance(weak_rows[0], dict) else {}
    weakness_headline = str(primary.get("weakness") or "").strip()
    weakness_evidence = str(primary.get("evidence") or "").strip()
    weakness_quote = str(primary.get("customer_quote") or "").strip()
    weakness_area = str(primary.get("area") or primary.get("weakness") or "").strip()
    anchor_phrase = _battle_card_anchor_phrase(card)
    proof_point = _battle_card_structured_proof_text(card)
    if anchor_phrase:
        proof_point = f"{proof_point} This is already visible in {anchor_phrase}."

    if not weakness_quote and weakness_area:
        weakness_quote = _battle_card_best_supported_quote(
            card,
            f"{weakness_area} {weakness_evidence}",
            preferred_terms=_battle_card_quote_terms(weakness_area),
        )

    summary = str(card.get("executive_summary") or "").strip()
    if not summary:
        summary = _battle_card_safe_summary(card)
        if weakness_headline:
            summary = f"{summary} {weakness_headline}."
        if anchor_phrase:
            summary = f"{summary} The strongest current signal is coming from {anchor_phrase}."
        card["executive_summary"] = summary.strip()

    plays = card.get("recommended_plays") if isinstance(card.get("recommended_plays"), list) else []
    cfg = settings.b2b_churn
    min_total_plays = int(getattr(cfg, "battle_card_quality_min_total_plays", 2))
    if len(plays) < min_total_plays:
        needed = max(min_total_plays - len(plays), 1)
        fallback_plays = _battle_card_fallback_recommended_plays(card, limit=needed)
        if fallback_plays:
            card["recommended_plays"] = plays + fallback_plays

    talk_track = card.get("talk_track") if isinstance(card.get("talk_track"), dict) else {}
    if not talk_track or not all(str(talk_track.get(key) or "").strip() for key in ("opening", "mid_call_pivot", "closing")):
        opening = (
            f"Buyers are re-evaluating {card.get('vendor') or 'the incumbent'} "
            f"because {weakness_area or 'current friction'} keeps showing up."
        )
        if anchor_phrase:
            opening = f"{opening} The clearest signal is coming from {anchor_phrase}."
        mid_call = _battle_card_safe_play_text(card, "talk_track.mid_call_pivot")
        if weakness_evidence:
            mid_call = f"{mid_call} The evidence already shows {weakness_evidence.lower()}."
        closing = _battle_card_safe_play_text(card, "talk_track.closing")
        card["talk_track"] = {
            "opening": opening.strip(),
            "mid_call_pivot": mid_call.strip(),
            "closing": closing.strip(),
        }

    if not isinstance(card.get("discovery_questions"), list) or not card.get("discovery_questions"):
        discovery_questions = [
            f"Where is {weakness_area or 'buyer friction'} creating the most pain in the current workflow?",
            "What would trigger a formal evaluation or renewal benchmark in the next planning cycle?",
        ]
        if anchor_phrase:
            discovery_questions[1] = f"What changed for {anchor_phrase} that moved this into an active evaluation?"
        card["discovery_questions"] = discovery_questions

    if not isinstance(card.get("landmine_questions"), list) or not card.get("landmine_questions"):
        landmine_questions = [
            f"What happens if {weakness_area or 'this friction'} is still unresolved at renewal?",
            "How much manual work is the team absorbing today to keep the incumbent setup operating?",
        ]
        if anchor_phrase:
            landmine_questions[0] = f"What happens if the issues visible in {anchor_phrase} are still unresolved at renewal?"
        card["landmine_questions"] = landmine_questions

    handlers = card.get("objection_handlers") if isinstance(card.get("objection_handlers"), list) else []
    if not handlers:
        strengths = card.get("incumbent_strengths") if isinstance(card.get("incumbent_strengths"), list) else []
        top_strength: dict[str, Any] = {}
        for item in strengths:
            if not isinstance(item, dict):
                continue
            area = str(item.get("area") or "").strip()
            if area and not _is_generic_other_weakness(area):
                top_strength = item
                break
        strength_area = str(top_strength.get("area") or "operational familiarity").strip()
        strength_quote = str(top_strength.get("customer_quote") or "").strip()
        acknowledge = f"It makes sense that teams stay for {strength_area}."
        if strength_quote:
            acknowledge = f"{acknowledge} Buyers still call out that strength directly."
        card["objection_handlers"] = [{
            "objection": f"The incumbent is still good enough on {strength_area}.",
            "acknowledge": acknowledge,
            "pivot": (
                f"The better question is whether {weakness_area or 'buyer friction'} is creating enough drag "
                "to justify a cleaner alternative before renewal."
            ),
            "proof_point": proof_point,
        }]

    why_they_stay = card.get("why_they_stay") if isinstance(card.get("why_they_stay"), dict) else {}
    why_strengths = why_they_stay.get("strengths") if isinstance(why_they_stay.get("strengths"), list) else []
    if not why_they_stay or not why_strengths:
        strengths = card.get("incumbent_strengths") if isinstance(card.get("incumbent_strengths"), list) else []
        normalized_strengths: list[dict[str, str]] = []
        for item in strengths:
            if not isinstance(item, dict):
                continue
            area = str(item.get("area") or "").strip()
            if not area or _is_generic_other_weakness(area):
                continue
            evidence = str(item.get("customer_quote") or "").strip()
            if not evidence:
                evidence = f"Customers still cite {area.lower()} as a reason to stay."
            normalized_strengths.append({
                "area": area,
                "evidence": evidence,
                "how_to_neutralize": _battle_card_winning_position(weakness_area or area).rstrip(".") + ".",
            })
            if len(normalized_strengths) >= 2:
                break
        if not normalized_strengths:
            normalized_strengths.append({
                "area": "Operational familiarity",
                "evidence": "Teams often stay when the incumbent still feels familiar to day-to-day operators.",
                "how_to_neutralize": _battle_card_winning_position(weakness_area or "operational familiarity").rstrip(".") + ".",
            })
        if normalized_strengths:
            card["why_they_stay"] = {
                "summary": "The incumbent still holds on where teams feel the current setup is familiar and good enough.",
                "strengths": normalized_strengths,
            }

    plays = card.get("recommended_plays") if isinstance(card.get("recommended_plays"), list) else []
    if len(plays) > 1:
        normalized_timings: set[str] = set()
        distinct_defaults = [
            "During renewal planning this quarter.",
            "Immediately after an evaluation checkpoint or pricing objection.",
            "Before the next stakeholder review on fit, cost, or migration timing.",
        ]
        next_default = 0
        for item in plays:
            if not isinstance(item, dict):
                continue
            timing = str(item.get("timing") or "").strip()
            norm = timing.lower()
            if timing and norm not in normalized_timings:
                normalized_timings.add(norm)
                continue
            while next_default < len(distinct_defaults) and distinct_defaults[next_default].lower() in normalized_timings:
                next_default += 1
            if next_default >= len(distinct_defaults):
                break
            item["timing"] = distinct_defaults[next_default]
            normalized_timings.add(item["timing"].lower())
            next_default += 1

    render_status = str(card.get("llm_render_status") or "").strip().lower()
    if render_status not in {"succeeded", "cached"}:
        generated_fields = {
            field: card[field]
            for field in _BATTLE_CARD_LLM_FIELDS
            if field in card
        }
        sanitized = _sanitize_battle_card_sales_copy(card, generated_fields)
        if isinstance(sanitized, dict):
            for field in _BATTLE_CARD_LLM_FIELDS:
                if field in sanitized:
                    card[field] = sanitized[field]


def _evaluate_battle_card_quality(
    card: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Score battle-card readiness and return a strict quality contract."""
    from ._b2b_shared import (
        _battle_card_allowed_quotes,
        _battle_card_fallback_recommended_plays,
        _battle_card_has_duplicate_recommended_play_segments,
        _validate_battle_card_sales_copy,
    )

    hard_blockers: list[str] = []
    warnings: list[str] = []
    cfg = settings.b2b_churn
    max_stale_days = int(getattr(cfg, "battle_card_quality_max_stale_days", 2))
    eval_divergence_warn_delta = int(getattr(cfg, "battle_card_quality_eval_divergence_warn_delta", 25))
    min_high_intent_urgency = float(getattr(cfg, "battle_card_quality_min_high_intent_urgency", 7.0))
    required_stages = _quality_required_stages(cfg)
    allow_global_eval_fallback = bool(getattr(cfg, "battle_card_quality_allow_global_eval_fallback", True))
    min_total_plays = int(getattr(cfg, "battle_card_quality_min_total_plays", 2))
    min_actionable_plays = int(getattr(cfg, "battle_card_quality_min_actionable_plays", 1))

    stale_days: int | None = None
    reported_data_stale = bool(card.get("data_stale"))
    as_of = _battle_card_data_as_of_date(card)
    if as_of:
        stale_days = max(0, (date.today() - as_of).days)
        if stale_days > max_stale_days:
            card["data_stale"] = True
            hard_blockers.append(
                f"source data is stale for requested report date ({stale_days}d > {max_stale_days}d)"
            )
        else:
            card["data_stale"] = False
            if reported_data_stale and stale_days > 0:
                warnings.append(
                    f"source data lag detected ({stale_days}d), below strict cutoff ({max_stale_days}d)"
                )
    elif reported_data_stale:
        card["data_stale"] = True
        hard_blockers.append("source data is stale for requested report date")
    if bool(card.get("evidence_window_is_thin")):
        warnings.append("evidence window is thin; confidence may improve with more data")

    anchors = _battle_card_anchor_examples(card)
    witness_highlights = _battle_card_witness_highlights(card)
    reference_ids = card.get("reference_ids") if isinstance(card.get("reference_ids"), dict) else {}
    witness_refs = [
        str(value).strip()
        for value in (reference_ids.get("witness_ids") or [])
        if str(value or "").strip()
    ]
    if witness_refs and not anchors:
        warnings.append("witness-backed references exist but no anchor examples were surfaced")
    if anchors and not witness_highlights:
        warnings.append("anchor examples are present but witness highlights are missing from the render packet")

    evidence_window_days = _reasoning_int(card.get("evidence_window_days"))
    if evidence_window_days is None:
        warnings.append("missing evidence_window_days metadata")

    eval_values = _battle_card_eval_signal_values(card)
    eval_families = _battle_card_eval_signal_families(eval_values)
    for family, items in eval_families.items():
        positives = [(path, value) for path, value in items if value > 0]
        zeros = [(path, value) for path, value in items if value == 0]
        if positives and zeros:
            hard_blockers.append(
                f"active-evaluation signal conflict in {family}: "
                + ", ".join(f"{path}={value}" for path, value in items[:6])
            )
            continue
        non_zero_values = sorted({value for _, value in positives})
        if len(non_zero_values) > 1 and (max(non_zero_values) - min(non_zero_values) >= eval_divergence_warn_delta):
            warnings.append(
                f"active-evaluation counts diverge in {family}: "
                + ", ".join(f"{path}={value}" for path, value in positives[:6])
            )

    weak_rows = card.get("weakness_analysis") if isinstance(card.get("weakness_analysis"), list) else []
    if not weak_rows and isinstance(card.get("vendor_weaknesses"), list):
        weak_rows = card.get("vendor_weaknesses") or []

    allowed_quotes = set(_battle_card_allowed_quotes(card))
    for idx, row in enumerate(weak_rows):
        if not isinstance(row, dict):
            continue
        quote = str(row.get("customer_quote") or "").strip()
        if quote and allowed_quotes and quote not in allowed_quotes:
            hard_blockers.append(f"weakness_analysis[{idx}] quote is not an exact source quote")

    if weak_rows:
        primary = weak_rows[0] if isinstance(weak_rows[0], dict) else {}
        primary_label = _battle_card_primary_weakness_label(primary)
        if _is_generic_other_weakness(primary_label):
            hard_blockers.append("primary weakness is generic fallback dissatisfaction instead of seller-usable wedge")

    hi_accounts = card.get("high_intent_companies") if isinstance(card.get("high_intent_companies"), list) else []
    qualified_accounts = []
    high_urgency_accounts = 0
    for item in hi_accounts:
        if not isinstance(item, dict):
            continue
        try:
            urgency = float(item.get("urgency") or 0)
        except (TypeError, ValueError):
            urgency = 0.0
        if urgency >= min_high_intent_urgency:
            high_urgency_accounts += 1
        stage = _normalize_stage(item.get("buying_stage") or item.get("stage"))
        if urgency >= min_high_intent_urgency and stage in required_stages:
            qualified_accounts.append(item)
    global_eval_present = any(value > 0 for _, value in eval_families.get(_QUALITY_EVAL_FAMILY_SIGNAL_VOLUME, []))
    if not global_eval_present:
        global_eval_present = any(value > 0 for _, value in eval_families.get(_QUALITY_EVAL_FAMILY_ACCOUNT_COUNT, []))
    if not qualified_accounts:
        stage_list = ", ".join(sorted(required_stages))
        if not hi_accounts:
            # No account data at all -- data gap, not a quality failure
            warnings.append(
                f"no high-intent account data available for required stages ({stage_list})"
            )
        elif allow_global_eval_fallback and high_urgency_accounts > 0 and global_eval_present:
            warnings.append(
                "high-intent account stage missing; using global active-evaluation evidence fallback"
            )
        else:
            hard_blockers.append(
                f"no high-intent account with urgency >= {min_high_intent_urgency:g} in required stages ({stage_list})"
            )

    plays = card.get("recommended_plays") if isinstance(card.get("recommended_plays"), list) else []
    account_names = [
        str(item.get("company") or "").strip()
        for item in qualified_accounts
        if isinstance(item, dict)
    ]
    actionable_play_count, target_segments = _battle_card_play_actionability(
        plays,
        account_names=account_names,
    )

    if phase == _QUALITY_PHASE_FINAL:
        duplicate_target_segments = (
            len(set(target_segments)) < 2 and len([seg for seg in target_segments if seg]) >= 2
        )
        duplicate_segment_labels = (
            isinstance(card.get("recommended_plays"), list)
            and _battle_card_has_duplicate_recommended_play_segments(
                {"recommended_plays": card.get("recommended_plays")}
            )
        )
        if (
            len(plays) < min_total_plays
            or actionable_play_count < min_actionable_plays
            or duplicate_target_segments
            or duplicate_segment_labels
        ):
            fallback_plays = _battle_card_fallback_recommended_plays(
                card,
                limit=max(min_total_plays, 1),
            )
            if fallback_plays:
                card["recommended_plays"] = fallback_plays
                plays = fallback_plays
                actionable_play_count, target_segments = _battle_card_play_actionability(
                    plays,
                    account_names=account_names,
                )
                duplicate_target_segments = (
                    len(set(target_segments)) < 2 and len([seg for seg in target_segments if seg]) >= 2
                )
                duplicate_segment_labels = _battle_card_has_duplicate_recommended_play_segments(
                    {"recommended_plays": plays}
                )
        if len(plays) < min_total_plays:
            hard_blockers.append(
                f"recommended_plays must contain at least {min_total_plays} distinct motions"
            )
        if actionable_play_count < min_actionable_plays:
            hard_blockers.append(
                "recommended plays are missing role/account targeting + timing + CTA"
            )
        if duplicate_target_segments:
            hard_blockers.append("recommended plays repeat the same target segment")
        if duplicate_segment_labels:
            hard_blockers.append("recommended plays contain duplicate target segments")

        generated = {
            field: card[field]
            for field in _BATTLE_CARD_LLM_FIELDS
            if field in card
        }
        if anchors and generated:
            render_text = _battle_card_render_text(card)
            signal_terms = _battle_card_anchor_signal_terms(card)
            anchor_hits = any(
                term in render_text
                for term in (
                    signal_terms["companies"]
                    | signal_terms["timing_terms"]
                    | signal_terms["numeric_terms"]
                    | signal_terms["competitor_terms"]
                )
            )
            if not anchor_hits:
                warnings.append(
                    "seller copy does not reference any witness-backed anchor despite anchors being available"
                )
            if signal_terms["companies"] and not any(term in render_text for term in signal_terms["companies"]):
                warnings.append("named-account anchor exists but seller copy does not mention a named example")
            if signal_terms["numeric_terms"] and not any(term in render_text for term in signal_terms["numeric_terms"]):
                warnings.append("money-backed outlier anchor exists but seller copy does not mention the concrete spend signal")
            if signal_terms["timing_terms"] and not any(term in render_text for term in signal_terms["timing_terms"]):
                warnings.append("timing anchor exists but seller copy does not mention the live trigger window")
        if generated:
            copy_issues = _validate_battle_card_sales_copy(card, generated)
            if copy_issues:
                hard_blockers.extend([f"sales_copy: {issue}" for issue in copy_issues])
        elif str(card.get("llm_render_status") or "").strip().lower() in {"failed", "failed_quality_gate"}:
            hard_blockers.append("model sales copy missing after render failure")

    # Phase 8: governance-aware quality checks
    contracts = card.get("reasoning_contracts")
    if isinstance(contracts, dict):
        eg = contracts.get("evidence_governance")
        if isinstance(eg, dict):
            # Contradiction overreach suppression
            contradictions = eg.get("contradictions") or []
            if contradictions:
                contradiction_dims = [
                    c.get("dimension", "")
                    for c in contradictions if isinstance(c, dict)
                ]
                dim_text = ", ".join(d for d in contradiction_dims if d)
                # Scan all LLM-generated fields for absolute language
                absolute_phrases = ("clearly", "undeniably", "without question", "definitively", "unequivocally")
                _copy_fields_to_scan = (
                    "executive_summary", "talk_track",
                    "objection_handlers", "recommended_plays",
                )
                absolute_found = False
                for field_name in _copy_fields_to_scan:
                    field_val = card.get(field_name)
                    if field_val is None:
                        continue
                    text = json.dumps(field_val) if isinstance(field_val, (list, dict)) else str(field_val)
                    text_lower = text.lower()
                    for phrase in absolute_phrases:
                        if phrase in text_lower:
                            hard_blockers.append(
                                f"{field_name} uses absolute language ('{phrase}') "
                                f"despite contradictions on [{dim_text}]"
                            )
                            absolute_found = True
                            break
                    if absolute_found:
                        break
                if not absolute_found and dim_text:
                    warnings.append(
                        f"contradictory evidence on [{dim_text}]; "
                        "verify sales copy hedges appropriately"
                    )

            # Coverage gap enforcement across all generated fields
            coverage_gaps = eg.get("coverage_gaps") or []
            if coverage_gaps:
                gap_areas = [
                    g.get("area", "").replace("_", " ")
                    for g in coverage_gaps if isinstance(g, dict) and g.get("area")
                ]
                if gap_areas:
                    warnings.append(
                        f"thin evidence in [{', '.join(gap_areas)}]; "
                        "claims about these areas should be hedged"
                    )
                    # Check if LLM copy makes strong claims about gap areas
                    for field_name in ("talk_track", "objection_handlers", "recommended_plays"):
                        field_val = card.get(field_name)
                        if field_val is None:
                            continue
                        text = json.dumps(field_val) if isinstance(field_val, (list, dict)) else str(field_val)
                        text_lower = text.lower()
                        for area in gap_areas:
                            area_lower = area.lower()
                            if area_lower in text_lower and any(
                                strong in text_lower
                                for strong in ("proven", "guaranteed", "always", "every")
                            ):
                                warnings.append(
                                    f"{field_name} makes strong claims about "
                                    f"thin-evidence area '{area}'"
                                )
                                break

        # Retention strength check: reps need both churn pressure and retention context
        vc = contracts.get("vendor_core_reasoning")
        if isinstance(vc, dict):
            wts = vc.get("why_they_stay")
            card_wts = card.get("why_they_stay")
            if isinstance(wts, dict) and wts.get("strengths"):
                if not isinstance(card_wts, dict) or not card_wts.get("strengths"):
                    warnings.append(
                        "synthesis has retention strengths but battle card "
                        "why_they_stay is missing; reps need incumbent inertia context"
                    )

    has_canonical_accounts = bool(card.get("account_pressure_metrics"))
    score = max(0, 100 - (25 * len(hard_blockers)) - (5 * len(warnings)))
    status = _battle_card_quality_status(
        phase=phase,
        hard_blockers=hard_blockers,
        warnings=warnings,
        has_canonical_accounts=has_canonical_accounts,
    )
    return {
        "schema_version": _QUALITY_SCHEMA_VERSION,
        "phase": phase,
        "status": status,
        "score": score,
        "failed_checks": hard_blockers,
        "warnings": warnings,
        "required_signals": {
            "active_eval_signals": [{"path": path, "value": value} for path, value in eval_values],
            "high_intent_accounts_total": len(hi_accounts),
            "high_intent_accounts_qualified": len(qualified_accounts),
            "recommended_plays_count": len(plays),
            "actionable_play_count": actionable_play_count,
            "evidence_window_days": evidence_window_days,
            "data_stale": bool(card.get("data_stale")),
            "stale_days": stale_days,
            "required_stages": sorted(required_stages),
        },
        "evaluated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def _apply_battle_card_quality(card: dict[str, Any], *, phase: str) -> dict[str, Any]:
    if phase == _QUALITY_PHASE_FINAL:
        _populate_battle_card_fallback_sales_copy(card)
    quality = _evaluate_battle_card_quality(card, phase=phase)
    card["battle_card_quality"] = quality
    card["quality_status"] = quality.get("status") or _QUALITY_STATUS_NEEDS_REVIEW
    return quality


def _battle_card_preflight_quality_gate(card: dict[str, Any]) -> dict[str, Any] | None:
    """Return final quality when deterministic blockers make an LLM call wasteful."""
    quality = _apply_battle_card_quality(card, phase=_QUALITY_PHASE_FINAL)
    failed_checks = quality.get("failed_checks") if isinstance(quality.get("failed_checks"), list) else []
    if quality.get("status") != _QUALITY_STATUS_FALLBACK or not failed_checks:
        return None
    return quality


def _battle_card_seller_usable_battles(
    vendor: str,
    battles: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep only pairwise battles that support a seller-facing vulnerability story."""
    vendor_norm = str(vendor or "").strip().lower()
    usable: list[dict[str, Any]] = []
    for battle in battles or []:
        if not isinstance(battle, dict):
            continue
        winner = str(battle.get("winner") or "").strip().lower()
        loser = str(battle.get("loser") or "").strip().lower()
        opponent = str(battle.get("opponent") or "").strip().lower()
        if winner and loser:
            if loser == vendor_norm or winner == opponent:
                usable.append(battle)
            continue
        usable.append(battle)
    return usable


def _build_battle_card_render_payload(
    card: dict[str, Any],
    *,
    prior_attempt: Any | None = None,
    validation_feedback: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact contract-first LLM render packet for battle cards."""
    from ._b2b_shared import _build_battle_card_locked_facts, _build_metric_ledger

    payload = {
        key: card[key]
        for key in _BATTLE_CARD_RENDER_INPUT_KEYS
        if key in card
    }
    if "cross_vendor_battles" in payload:
        filtered_battles = _battle_card_seller_usable_battles(
            str(card.get("vendor") or ""),
            payload.get("cross_vendor_battles"),
        )
        if filtered_battles:
            payload["cross_vendor_battles"] = filtered_battles
        else:
            payload.pop("cross_vendor_battles", None)

    vendor_core_reasoning = _battle_card_contract(card, "vendor_core_reasoning")
    displacement_reasoning = _battle_card_contract(card, "displacement_reasoning")
    category_reasoning = _battle_card_category_reasoning(card)
    account_reasoning = _battle_card_account_reasoning(card)
    reasoning_contracts = card.get("reasoning_contracts")
    section_contracts_present = any(
        (
            vendor_core_reasoning,
            displacement_reasoning,
            category_reasoning,
            account_reasoning,
        ),
    )

    if (
        isinstance(reasoning_contracts, dict)
        and reasoning_contracts
        and not section_contracts_present
    ):
        compact_contracts = {}
        vendor_bundle = _battle_card_compact_vendor_core_reasoning(
            reasoning_contracts.get("vendor_core_reasoning"),
        )
        if vendor_bundle:
            compact_contracts["vendor_core_reasoning"] = vendor_bundle
        displacement_bundle = _battle_card_compact_displacement_reasoning(
            reasoning_contracts.get("displacement_reasoning"),
        )
        if displacement_bundle:
            compact_contracts["displacement_reasoning"] = displacement_bundle
        category_bundle = _battle_card_compact_category_reasoning(
            reasoning_contracts.get("category_reasoning"),
        )
        if category_bundle:
            compact_contracts["category_reasoning"] = category_bundle
        account_bundle = _battle_card_compact_account_reasoning(
            reasoning_contracts.get("account_reasoning"),
        )
        if account_bundle:
            compact_contracts["account_reasoning"] = account_bundle
        if compact_contracts:
            payload["reasoning_contracts"] = compact_contracts
    if vendor_core_reasoning:
        payload["vendor_core_reasoning"] = _battle_card_compact_vendor_core_reasoning(
            vendor_core_reasoning,
        )
    if displacement_reasoning:
        payload["displacement_reasoning"] = _battle_card_compact_displacement_reasoning(
            displacement_reasoning,
        )
    if category_reasoning:
        payload["category_reasoning"] = _battle_card_compact_category_reasoning(
            category_reasoning,
        )
    if account_reasoning:
        payload["account_reasoning"] = _battle_card_compact_account_reasoning(
            account_reasoning,
        )

    payload["locked_facts"] = _build_battle_card_locked_facts(card)
    payload["render_packet_version"] = "contract_first_v1"
    metric_ledger = _build_metric_ledger(card)
    if metric_ledger:
        payload["metric_ledger"] = metric_ledger
    anchor_examples = _battle_card_limited_anchor_examples(card)
    if anchor_examples:
        payload["anchor_examples"] = anchor_examples
    witness_highlights = _battle_card_limited_witness_highlights(card)
    if witness_highlights:
        payload["witness_highlights"] = witness_highlights
    reference_ids = _battle_card_limited_reference_ids(card.get("reference_ids"))
    if reference_ids:
        payload["reference_ids"] = reference_ids

    cfg = settings.b2b_churn
    battle_limit = int(getattr(cfg, "battle_card_render_cross_vendor_battles_limit", 2) or 2)
    if battle_limit <= 0:
        payload.pop("cross_vendor_battles", None)
    else:
        battles = _battle_card_limited_rows(
            payload.get("cross_vendor_battles"),
            limit=battle_limit,
            allowed_keys=(
                "opponent",
                "conclusion",
                "durability",
                "confidence",
                "winner",
                "loser",
                "key_insights",
            ),
        )
        if battles:
            payload["cross_vendor_battles"] = battles
        else:
            payload.pop("cross_vendor_battles", None)

    account_limit = int(getattr(cfg, "battle_card_render_high_intent_companies_limit", 3) or 3)
    companies = _battle_card_limited_rows(
        payload.get("high_intent_companies"),
        limit=account_limit,
        allowed_keys=(
            "company",
            "urgency",
            "role",
            "pain",
            "company_size",
            "buying_stage",
            "decision_maker",
            "confidence_score",
            "contract_end",
        ),
    )
    if companies:
        payload["high_intent_companies"] = companies
    else:
        payload.pop("high_intent_companies", None)

    # Phase 8: inject governance context so LLM can calibrate
    contracts = card.get("reasoning_contracts")
    if isinstance(contracts, dict):
        # why_they_stay as input context (distinct from LLM output field)
        vc = contracts.get("vendor_core_reasoning")
        if isinstance(vc, dict):
            wts = vc.get("why_they_stay")
            if isinstance(wts, dict) and wts:
                payload["retention_context"] = wts
            cp = vc.get("confidence_posture")
            if isinstance(cp, dict) and cp:
                payload["confidence_posture"] = cp
        eg = contracts.get("evidence_governance")
        if isinstance(eg, dict):
            contradictions = eg.get("contradictions")
            if isinstance(contradictions, list) and contradictions:
                payload["contradictions"] = _battle_card_limited_rows(
                    contradictions,
                    limit=3,
                )
            cg = eg.get("coverage_gaps")
            if isinstance(cg, list) and cg:
                payload["coverage_gaps"] = _battle_card_limited_rows(
                    cg,
                    limit=int(getattr(cfg, "battle_card_render_data_gaps_limit", 4) or 4),
                )

    if prior_attempt is not None:
        payload["prior_attempt"] = prior_attempt
    if validation_feedback:
        payload["validation_feedback"] = list(validation_feedback)
    # Strip internal _sid provenance keys before LLM sees the payload.
    # These are source-tracing markers from pool compression that leak
    # into rendered output when the LLM echoes them.
    _strip_internal_keys(payload)
    return payload


def _strip_internal_keys(obj: Any) -> None:
    """Recursively remove _sid and other internal keys from nested dicts/lists."""
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key.startswith("_sid") or key == "_source_id":
                del obj[key]
            else:
                _strip_internal_keys(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            _strip_internal_keys(item)


def _attach_battle_card_render_provenance(
    card: dict[str, Any],
    *,
    render_packet_hash: str | None = None,
) -> None:
    """Persist render-packet provenance on the battle card for audits."""
    card["render_packet_version"] = "contract_first_v1"
    card["render_contracts_used"] = bool(
        card.get("reasoning_contracts")
        or card.get("vendor_core_reasoning")
        or card.get("displacement_reasoning")
        or card.get("category_reasoning")
    )
    if render_packet_hash:
        card["render_packet_hash"] = str(render_packet_hash)


def _battle_card_category_reasoning(card: dict[str, Any]) -> dict[str, Any]:
    """Resolve category reasoning from contracts or raw compatibility fields."""
    contract = _battle_card_contract(card, "category_reasoning")
    if contract:
        return contract
    raw = card.get("category_reasoning")
    return raw if isinstance(raw, dict) else {}


def _build_category_council_from_reasoning(
    card: dict[str, Any],
) -> dict[str, Any] | None:
    """Build category council context from category reasoning contract."""
    reasoning = _battle_card_category_reasoning(card)
    regime = str(reasoning.get("market_regime") or "").strip()
    narrative = str(reasoning.get("narrative") or "").strip()
    if not regime and not narrative:
        return None
    category_label = card.get("category") or "This category"
    return {
        "conclusion": narrative or f"{category_label} currently shows {regime} dynamics.",
        "market_regime": regime,
        "durability": reasoning.get("durability") or "uncertain",
        "confidence": reasoning.get("confidence_score") or 0,
        "winner": reasoning.get("winner"),
        "loser": reasoning.get("loser"),
        "key_insights": reasoning.get("key_insights") or [],
    }


def _build_category_council_from_cross_vendor(
    council: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert reconstructed cross-vendor council rows into battle-card shape."""
    if not isinstance(council, dict):
        return None
    cc = council.get("conclusion", {})
    if not isinstance(cc, dict):
        cc = {}
    conclusion = str(cc.get("conclusion") or "").strip()
    regime = str(cc.get("market_regime") or "").strip()
    if not conclusion and not regime:
        return None
    return {
        "conclusion": conclusion,
        "market_regime": regime,
        "durability": cc.get("durability_assessment", ""),
        "confidence": council.get("confidence", 0),
        "winner": cc.get("winner", ""),
        "loser": cc.get("loser", ""),
        "key_insights": cc.get("key_insights", []),
    }


def _category_council_score(council: dict[str, Any] | None) -> float:
    """Score battle-card council usefulness so richer sources can override generic ones."""
    if not isinstance(council, dict):
        return -1.0
    score = 0.0
    if str(council.get("conclusion") or "").strip():
        score += 2.0
    if str(council.get("market_regime") or "").strip():
        score += 1.5
    if str(council.get("winner") or "").strip():
        score += 2.0
    if str(council.get("loser") or "").strip():
        score += 2.0
    score += min(len(council.get("key_insights") or []), 3) * 0.5
    try:
        score += min(float(council.get("confidence") or 0.0), 1.0)
    except (TypeError, ValueError):
        pass
    return score


def _prefer_richer_category_council(
    preferred: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Choose the richer council between two compatible shapes."""
    preferred_score = _category_council_score(preferred)
    fallback_score = _category_council_score(fallback)
    if fallback_score > preferred_score:
        return fallback
    return preferred if preferred_score >= 0 else fallback


def _build_category_council_fallback(card: dict[str, Any]) -> dict[str, Any] | None:
    """Create deterministic category context when no council conclusion exists."""
    eco = card.get("ecosystem_context") or {}
    regime = str(eco.get("market_structure") or "").strip()
    if not eco or not regime:
        return None
    insights: list[dict[str, str]] = [
        {
            "insight": f"Category market structure is {regime}.",
            "evidence": f"market_structure: {regime}",
        },
    ]
    if eco.get("hhi") is not None:
        insights.append({"insight": "Category concentration is visible in the current HHI.", "evidence": f"hhi: {eco['hhi']}"})
    if eco.get("displacement_intensity") is not None:
        insights.append({"insight": "Competitive displacement is active in this category.", "evidence": f"displacement_intensity: {eco['displacement_intensity']}"})
    if eco.get("dominant_archetype"):
        insights.append({"insight": f"{eco['dominant_archetype']} is the dominant churn archetype in this category.", "evidence": f"dominant_archetype: {eco['dominant_archetype']}"})
    return {
        "conclusion": f"{card.get('category') or 'This category'} currently shows {regime} dynamics, so reps should anchor positioning to category-wide pressure instead of a single isolated complaint stream.",
        "market_regime": regime,
        "durability": "uncertain",
        "confidence": 0.0,
        "winner": None,
        "loser": None,
        "key_insights": insights[:5],
    }


def _pair_opponent(pair_key: tuple[str, ...] | list[str] | Any, vendor: str) -> str:
    """Return the non-self vendor from a pair key, or empty string."""
    if not isinstance(pair_key, (tuple, list)):
        return ""
    vendor_text = str(vendor or "").strip()
    for value in pair_key:
        candidate = str(value or "").strip()
        if candidate and candidate != vendor_text:
            return candidate
    return ""


def _ecosystem_context_from_analysis(eco_data: Any) -> dict[str, Any] | None:
    """Normalize EcosystemEvidence or dict payloads into battle-card context."""
    if not eco_data:
        return None
    health = eco_data.get("health") if isinstance(eco_data, dict) else getattr(eco_data, "health", eco_data)
    hhi = health.get("hhi") if isinstance(health, dict) else getattr(health, "hhi", None)
    market_structure = health.get("market_structure") if isinstance(health, dict) else getattr(health, "market_structure", None)
    displacement = health.get("displacement_intensity") if isinstance(health, dict) else getattr(health, "displacement_intensity", None)
    archetype = health.get("dominant_archetype") if isinstance(health, dict) else getattr(health, "dominant_archetype", None)
    if hhi is None and displacement is None and not market_structure and not archetype:
        return None
    return {
        "hhi": hhi,
        "market_structure": market_structure,
        "displacement_intensity": displacement,
        "dominant_archetype": archetype,
    }


def _iso_dateish(value: Any) -> str | None:
    """Serialize date/datetime values for persisted card metadata."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _attach_battle_card_provenance(card: dict[str, Any], provenance: dict[str, Any]) -> None:
    """Attach vendor-specific source and review-window metadata to a card."""
    source_dist = provenance.get("source_distribution") or {}
    if source_dist:
        card["source_distribution"] = source_dist
    sample_ids = provenance.get("sample_review_ids") or []
    if sample_ids:
        card["sample_review_ids"] = [str(item) for item in sample_ids[:20]]
    window_start = _iso_dateish(provenance.get("review_window_start"))
    window_end = _iso_dateish(provenance.get("review_window_end"))
    if window_start:
        card["review_window_start"] = window_start
    if window_end:
        card["review_window_end"] = window_end


def _merge_battle_card_provenance(
    primary: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge vendor provenance with vault provenance fallback."""
    merged: dict[str, Any] = {}
    for source in (fallback or {}, primary or {}):
        if not isinstance(source, dict):
            continue
        for key in ("source_distribution", "sample_review_ids", "review_window_start", "review_window_end"):
            value = source.get(key)
            if value in (None, "", [], {}):
                continue
            merged[key] = value
    return merged


def _battle_card_persist_summary(card: dict[str, Any]) -> str:
    """Build the persisted summary for deterministic or LLM-enriched cards."""
    vendor = str(card.get("vendor", "") or "")
    return str(card.get("executive_summary") or (
        f"Battle card for {vendor}: "
        f"score {card.get('churn_pressure_score', 0):.0f}, "
        f"{len(card.get('vendor_weaknesses', []))} weaknesses, "
        f"{len(card.get('competitor_differentiators', []))} competitors."
    ))


def _battle_card_source_metadata(
    card: dict[str, Any],
    report_source_review_count: int | None,
    report_source_dist: dict[str, int],
) -> tuple[int | None, dict[str, int]]:
    """Resolve row-level source metadata with vendor provenance fallback."""
    card_source_dist = card.get("source_distribution") or report_source_dist
    if not card_source_dist:
        return report_source_review_count, {}
    card_source_count = sum(int(v or 0) for v in card_source_dist.values())
    return card_source_count, card_source_dist


def _battle_card_llm_model_label(card: dict[str, Any], llm_options: dict[str, Any]) -> str:
    """Choose the persisted llm_model label for the current render state."""
    render_status = str(card.get("llm_render_status", "") or "").strip().lower()
    if render_status == "cached":
        return "pipeline_cached"
    if render_status == "succeeded":
        if llm_options.get("try_openrouter"):
            return str(llm_options.get("openrouter_model") or "openrouter")
        return str(llm_options.get("workload") or "anthropic")
    return "pipeline_deterministic"


def _battle_card_row_status(card: dict[str, Any]) -> str:
    """Persist quality-aware row status for battle-card report listings."""
    status = str(card.get("quality_status") or "").strip().lower()
    if status in {
        _QUALITY_STATUS_SALES_READY,
        _QUALITY_STATUS_NEEDS_REVIEW,
        _QUALITY_STATUS_THIN_EVIDENCE,
        _QUALITY_STATUS_FALLBACK,
    }:
        return status
    return "published"


async def _persist_battle_card(
    pool: Any,
    *,
    today: date,
    card: dict[str, Any],
    data_density: str,
    report_source_review_count: int | None,
    report_source_dist: dict[str, int],
    llm_model: str,
    status: str = "published",
) -> bool:
    """Persist a battle card row without dropping deterministic sections."""
    vendor = str(card.get("vendor", "") or "")
    if not vendor:
        return False
    persisted_summary = _battle_card_persist_summary(card)
    card["executive_summary"] = persisted_summary
    card_source_count, card_source_dist = _battle_card_source_metadata(
        card,
        report_source_review_count,
        report_source_dist,
    )
    sql = """
        INSERT INTO b2b_intelligence (
            report_date, report_type, vendor_filter,
            intelligence_data, executive_summary, data_density, status, llm_model,
            source_review_count, source_distribution
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')),
                     LOWER(COALESCE(category_filter,'')),
                     COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
        DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                      executive_summary = EXCLUDED.executive_summary,
                      data_density = EXCLUDED.data_density,
                      status = EXCLUDED.status,
                      llm_model = EXCLUDED.llm_model,
                      source_review_count = EXCLUDED.source_review_count,
                      source_distribution = EXCLUDED.source_distribution,
                      created_at = now()
        RETURNING id
    """
    sql_args = (
        today,
        "battle_card",
        vendor,
        json.dumps(card, default=str),
        persisted_summary,
        data_density,
        status,
        llm_model,
        card_source_count,
        json.dumps(card_source_dist),
    )
    fetchrow = getattr(pool, "fetchrow", None)
    report_row = await fetchrow(sql, *sql_args) if callable(fetchrow) else None
    if report_row is None:
        await pool.execute(sql.replace(" RETURNING id", ""), *sql_args)
    elif report_row.get("id"):
        try:
            from ...services.b2b.webhook_dispatcher import dispatch_report_generated_webhook

            await dispatch_report_generated_webhook(
                pool,
                report_id=report_row["id"],
                report_type="battle_card",
                vendor_name=vendor,
                status=status,
                report_date=str(today),
                llm_model=llm_model,
            )
        except Exception:
            logger.debug("Webhook dispatch skipped for report_generated battle card %s", vendor)
    return True


async def _retire_gated_out_battle_cards(
    pool: Any,
    vendors: list[str] | set[str] | tuple[str, ...],
) -> int:
    """Delete stale battle-card rows for vendors that no longer qualify."""
    normalized = sorted({
        str(vendor or "").strip().lower()
        for vendor in (vendors or [])
        if str(vendor or "").strip()
    })
    if not normalized:
        return 0
    deleted = await pool.fetchval(
        """
        WITH deleted AS (
            DELETE FROM b2b_intelligence
            WHERE report_type = 'battle_card'
              AND LOWER(COALESCE(vendor_filter, '')) = ANY($1::text[])
            RETURNING 1
        )
        SELECT COUNT(*) FROM deleted
        """,
        normalized,
    )
    return int(deleted or 0)


async def _check_freshness(pool) -> date | None:
    """Return today's date if the core run completed canonically, else None."""
    from ._b2b_shared import has_complete_core_run_marker

    today = date.today()
    if not await has_complete_core_run_marker(pool, today):
        logger.info("Core run not complete for %s, skipping", today)
        return None
    return today


async def _latest_core_report_date(pool) -> date | None:
    """Return the latest complete persisted core-run date, if any."""
    from ._b2b_shared import latest_complete_core_report_date

    return await latest_complete_core_report_date(pool)


async def _resolve_core_report_date(pool, *, maintenance_run: bool) -> date | None:
    """Return the report date battle cards should read from."""
    today = await _check_freshness(pool)
    if today is not None:
        return today
    if not maintenance_run:
        return None
    latest = await _latest_core_report_date(pool)
    if latest is not None:
        logger.info("Using latest core run date %s for maintenance battle-card run", latest)
    return latest


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Build battle cards + LLM sales copy from persisted artifacts."""
    cfg = settings.b2b_churn
    maintenance_run = bool((task.metadata or {}).get("maintenance_run"))
    if (not cfg.enabled or not cfg.intelligence_enabled) and not maintenance_run:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    today = await _resolve_core_report_date(pool, maintenance_run=maintenance_run)
    if today is None:
        from ._b2b_shared import describe_core_run_gap

        return {
            "_skip_synthesis": (
                await describe_core_run_gap(pool, date.today())
                or "Core signals not fresh for today"
            )
        }

    from ._b2b_shared import (
        _aggregate_competitive_disp,
        _build_deterministic_battle_cards,
        _build_pain_lookup,
        _build_competitor_lookup,
        _build_feature_gap_lookup,
        _build_use_case_lookup,
        _build_sentiment_lookup,
        _build_buyer_auth_lookup,
        _build_keyword_spike_lookup,
        _battle_card_provenance_from_evidence_vault,
        _build_deterministic_battle_card_competitive_landscape,
        _build_deterministic_battle_card_weakness_analysis,
        _build_positive_lookup,
        _build_department_lookup,
        _build_usage_duration_lookup,
        _build_timeline_lookup,
        _build_battle_card_locked_facts,
        _canonicalize_vendor,
        _align_vendor_intelligence_records_to_scorecards,
        _sanitize_battle_card_sales_copy,
        _validate_battle_card_sales_copy,
        read_vendor_scorecards,
        _fetch_pain_distribution,
        _fetch_feature_gaps,
        _fetch_price_complaint_rates,
        _fetch_dm_churn_rates,
        _fetch_churning_companies,
        _fetch_quotable_evidence,
        _fetch_budget_signals,
        _fetch_use_case_distribution,
        _fetch_sentiment_trajectory,
        _fetch_buyer_authority_summary,
        _fetch_timeline_signals,
        _fetch_keyword_spikes,
        _fetch_product_profiles,
        _fetch_competitor_reasons,
        _fetch_data_context,
        _fetch_vendor_provenance,
        read_vendor_intelligence_records,
        _fetch_latest_account_intelligence,
        _fetch_review_text_aggregates,
        _fetch_department_distribution,
        _fetch_contract_context_distribution,
        _fetch_competitive_displacement_source_of_truth,
    )
    from .b2b_churn_intelligence import (
        _apply_vendor_scope_to_churn_inputs,
        _normalize_test_vendors,
    )

    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews

    # --- Phase 1: Parallel data fetch ---
    await _update_execution_progress(
        task,
        stage=_STAGE_LOADING_INPUTS,
        progress_message="Loading battle-card source artifacts.",
    )
    try:
        (
            vendor_scores,
            competitive_disp, pain_dist, feature_gaps,
            price_rates, dm_rates,
            churning_companies, quotable_evidence,
            budget_signals, use_case_dist,
            sentiment_traj, buyer_auth, timeline_signals,
            competitor_reasons, keyword_spikes,
            data_context, vendor_provenance,
            evidence_vault_records,
            account_intel_lookup,
            product_profiles_raw,
            review_text_agg, department_dist, contract_ctx,
        ) = await asyncio.gather(
            read_vendor_scorecards(pool, window_days=window_days, min_reviews=min_reviews),
            _fetch_competitive_displacement_source_of_truth(
                pool,
                as_of=today,
                analysis_window_days=window_days,
            ),
            _fetch_pain_distribution(pool, window_days),
            _fetch_feature_gaps(pool, window_days, min_mentions=cfg.feature_gap_min_mentions),
            _fetch_price_complaint_rates(pool, window_days),
            _fetch_dm_churn_rates(pool, window_days),
            _fetch_churning_companies(pool, window_days),
            _fetch_quotable_evidence(pool, window_days, min_urgency=cfg.quotable_phrase_min_urgency),
            _fetch_budget_signals(pool, window_days),
            _fetch_use_case_distribution(pool, window_days),
            _fetch_sentiment_trajectory(pool, window_days),
            _fetch_buyer_authority_summary(pool, window_days),
            _fetch_timeline_signals(pool, window_days),
            _fetch_competitor_reasons(pool, window_days),
            _fetch_keyword_spikes(pool),
            _fetch_data_context(pool, window_days),
            _fetch_vendor_provenance(pool, window_days),
            read_vendor_intelligence_records(
                pool,
                as_of=today,
                analysis_window_days=window_days,
            ),
            _fetch_latest_account_intelligence(pool, as_of=today, analysis_window_days=window_days),
            _fetch_product_profiles(pool),
            _fetch_review_text_aggregates(pool, window_days),
            _fetch_department_distribution(pool, window_days),
            _fetch_contract_context_distribution(pool, window_days),
        )
    except Exception:
        logger.exception("Battle card data fetch failed")
        return {"_skip_synthesis": "Data fetch failed"}

    if not vendor_scores:
        return {"_skip_synthesis": "No vendor scores"}
    competitive_disp = _aggregate_competitive_disp(competitive_disp)

    scoped_vendors = _normalize_test_vendors((task.metadata or {}).get("test_vendors"))
    if scoped_vendors:
        raw_vendor_count = len(vendor_scores)
        scoped_data, scoped_vendors = _apply_vendor_scope_to_churn_inputs(
            {
                "vendor_scores": vendor_scores,
                "competitive_disp": competitive_disp,
                "pain_dist": pain_dist,
                "feature_gaps": feature_gaps,
                "price_rates": price_rates,
                "dm_rates": dm_rates,
                "churning_companies": churning_companies,
                "quotable_evidence": quotable_evidence,
                "budget_signals": budget_signals,
                "use_case_dist": use_case_dist,
                "sentiment_traj": sentiment_traj,
                "buyer_auth": buyer_auth,
                "timeline_signals": timeline_signals,
                "competitor_reasons": competitor_reasons,
                "keyword_spikes": keyword_spikes,
                "vendor_provenance": vendor_provenance,
                "product_profiles_raw": product_profiles_raw,
                "review_text_aggs": review_text_agg,
                "department_dist": department_dist,
                "contract_ctx_aggs": contract_ctx,
            },
            scoped_vendors,
        )
        vendor_scores = scoped_data["vendor_scores"]
        competitive_disp = scoped_data["competitive_disp"]
        pain_dist = scoped_data["pain_dist"]
        feature_gaps = scoped_data["feature_gaps"]
        price_rates = scoped_data["price_rates"]
        dm_rates = scoped_data["dm_rates"]
        churning_companies = scoped_data["churning_companies"]
        quotable_evidence = scoped_data["quotable_evidence"]
        budget_signals = scoped_data["budget_signals"]
        use_case_dist = scoped_data["use_case_dist"]
        sentiment_traj = scoped_data["sentiment_traj"]
        buyer_auth = scoped_data["buyer_auth"]
        timeline_signals = scoped_data["timeline_signals"]
        competitor_reasons = scoped_data["competitor_reasons"]
        keyword_spikes = scoped_data["keyword_spikes"]
        vendor_provenance = scoped_data["vendor_provenance"]
        product_profiles_raw = scoped_data["product_profiles_raw"]
        review_text_agg = scoped_data["review_text_aggs"]
        department_dist = scoped_data["department_dist"]
        contract_ctx = scoped_data["contract_ctx_aggs"]
        logger.info(
            "Scoped battle cards to %d/%d vendors for test run: %s",
            len(vendor_scores),
            raw_vendor_count,
            sorted(scoped_vendors),
        )

    if not vendor_scores:
        return {"_skip_synthesis": "No vendor scores after vendor scope filter"}
    evidence_vault_lookup, vault_alignment = (
        _align_vendor_intelligence_records_to_scorecards(
            vendor_scores,
            evidence_vault_records,
        )
    )
    if vault_alignment["mismatched_vendor_count"]:
        logger.info(
            "Battle cards suppressed %d mismatched evidence-vault overlays: %s",
            vault_alignment["mismatched_vendor_count"],
            ", ".join(vault_alignment["mismatched_vendors"][:10]),
        )
    vendor_total = len({
        _canonicalize_vendor(row.get("vendor_name") or row.get("vendor") or "")
        for row in vendor_scores
        if _canonicalize_vendor(row.get("vendor_name") or row.get("vendor") or "")
    })

    # --- Load reasoning views (synthesis-first, legacy fallback) ---
    from ._b2b_synthesis_reader import (
        build_reasoning_lookup_from_views,
        load_best_reasoning_views,
    )

    synthesis_views: dict[str, Any] = {}
    try:
        vendor_names_for_views = [
            _canonicalize_vendor(row.get("vendor_name") or row.get("vendor") or "")
            for row in vendor_scores
            if _canonicalize_vendor(row.get("vendor_name") or row.get("vendor") or "")
        ]
        if scoped_vendors:
            vendor_names_for_views = [v for v in vendor_names_for_views if v.lower() in {s.lower() for s in scoped_vendors}]
        synthesis_views = await load_best_reasoning_views(
            pool, vendor_names_for_views,
            as_of=today,
            analysis_window_days=window_days,
        )
        logger.info(
            "Loaded reasoning views for %d vendors (%d synthesis, %d legacy)",
            len(synthesis_views),
            sum(1 for v in synthesis_views.values() if v.schema_version != "legacy"),
            sum(1 for v in synthesis_views.values() if v.schema_version == "legacy"),
        )
    except Exception:
        logger.debug("Reasoning views unavailable", exc_info=True)

    # Build reasoning_lookup from synthesis views only.
    reasoning_lookup = build_reasoning_lookup_from_views(synthesis_views)
    if scoped_vendors:
        vendor_scope = {v.lower() for v in scoped_vendors}
        reasoning_lookup = {
            k: v for k, v in reasoning_lookup.items()
            if str(k or "").strip().lower() in vendor_scope
        }
    # Prefer cross-vendor synthesis; fall back to legacy conclusions
    from ._b2b_cross_vendor_synthesis import load_best_cross_vendor_lookup
    xv_lookup = await load_best_cross_vendor_lookup(
        pool,
        as_of=today,
        analysis_window_days=window_days,
    )
    synth_count = sum(
        1 for bucket in ("battles", "councils", "asymmetries")
        for v in xv_lookup[bucket].values()
        if isinstance(v, dict) and v.get("source") == "synthesis"
    )
    logger.info(
        "Cross-vendor enrichment: %d battles, %d councils, %d asymmetries (%d from synthesis)",
        len(xv_lookup.get("battles", {})),
        len(xv_lookup.get("councils", {})),
        len(xv_lookup.get("asymmetries", {})),
        synth_count,
    )

    # Load category dynamics pool for council fallback
    category_dynamics_lookup: dict[str, dict] = {}
    try:
        cat_dyn_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (category)
                   category, dynamics
            FROM b2b_category_dynamics
            WHERE as_of_date <= $1
              AND analysis_window_days = $2
            ORDER BY category, as_of_date DESC, created_at DESC
            """,
            today,
            window_days,
        )
        for row in cat_dyn_rows:
            cat = row.get("category") or ""
            dyn = row.get("dynamics")
            if isinstance(dyn, str):
                try:
                    dyn = json.loads(dyn)
                except (json.JSONDecodeError, TypeError):
                    dyn = {}
            if cat and isinstance(dyn, dict):
                category_dynamics_lookup[cat] = dyn
        logger.info("Category dynamics pool: %d categories loaded", len(category_dynamics_lookup))
    except Exception:
        logger.debug("Category dynamics load skipped", exc_info=True)

    # --- Phase 3: Build lookups ---
    pain_lookup = _build_pain_lookup(pain_dist)
    competitor_lookup = _build_competitor_lookup(competitive_disp)
    feature_gap_lookup = _build_feature_gap_lookup(feature_gaps)
    price_lookup = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
    dm_lookup = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
    company_lookup = {r["vendor"]: r["companies"] for r in churning_companies}
    quote_lookup = {r["vendor"]: r["quotes"] for r in quotable_evidence}
    budget_lookup = {r["vendor"]: {k: v for k, v in r.items() if k != "vendor"} for r in budget_signals}
    sentiment_lookup = _build_sentiment_lookup(sentiment_traj)
    buyer_auth_lookup = _build_buyer_auth_lookup(buyer_auth)
    timeline_lookup = _build_timeline_lookup(timeline_signals)
    keyword_spike_lookup = _build_keyword_spike_lookup(keyword_spikes)
    _complaints_raw, _positives_raw = review_text_agg
    positive_lookup = _build_positive_lookup(_positives_raw)
    department_lookup = _build_department_lookup(department_dist)
    _contract_values_raw, _durations_raw = contract_ctx
    usage_duration_lookup = _build_usage_duration_lookup(_durations_raw)

    product_profile_lookup: dict[str, dict] = {}
    for pp in product_profiles_raw:
        vn = _canonicalize_vendor(pp.get("vendor_name", ""))
        if vn and vn not in product_profile_lookup:
            product_profile_lookup[vn] = pp

    # --- Phase 4: Build deterministic battle cards ---
    await _update_execution_progress(
        task,
        stage=_STAGE_BUILDING,
        progress_message="Building deterministic battle cards.",
        vendors_total=vendor_total,
    )
    deterministic_battle_cards = _build_deterministic_battle_cards(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitor_lookup=competitor_lookup,
        feature_gap_lookup=feature_gap_lookup,
        quote_lookup=quote_lookup,
        price_lookup=price_lookup,
        budget_lookup=budget_lookup,
        sentiment_lookup=sentiment_lookup,
        dm_lookup=dm_lookup,
        company_lookup=company_lookup,
        product_profile_lookup=product_profile_lookup,
        competitive_disp=competitive_disp,
        competitor_reasons=competitor_reasons,
        synthesis_views=synthesis_views,
        reasoning_lookup=reasoning_lookup,
        timeline_lookup=timeline_lookup,
        use_case_lookup=_build_use_case_lookup(use_case_dist),
        positive_lookup=positive_lookup,
        department_lookup=department_lookup,
        usage_duration_lookup=usage_duration_lookup,
        buyer_auth_lookup=buyer_auth_lookup,
        keyword_spike_lookup=keyword_spike_lookup,
        evidence_vault_lookup=evidence_vault_lookup,
        account_intel_lookup=account_intel_lookup,
        synthesis_requested_as_of=today,
        category_dynamics_lookup=category_dynamics_lookup,
    )
    scored_vendors = {
        _canonicalize_vendor(row.get("vendor_name") or row.get("vendor") or "")
        for row in vendor_scores
        if _canonicalize_vendor(row.get("vendor_name") or row.get("vendor") or "")
    }
    built_vendors = {
        _canonicalize_vendor(card.get("vendor") or "")
        for card in deterministic_battle_cards
        if _canonicalize_vendor(card.get("vendor") or "")
    }
    gated_out_vendors = sorted(scored_vendors - built_vendors)
    cards_retired = 0
    if gated_out_vendors:
        try:
            cards_retired = await _retire_gated_out_battle_cards(pool, gated_out_vendors)
        except Exception:
            logger.exception("Failed to retire stale battle cards for gated-out vendors")
        else:
            logger.info(
                "Retired %d stale battle cards for gated-out vendors",
                cards_retired,
            )

    for card in deterministic_battle_cards:
        vendor = card.get("vendor", "")
        if not vendor:
            continue
        _attach_battle_card_provenance(
            card,
            _merge_battle_card_provenance(
                vendor_provenance.get(vendor, {}),
                _battle_card_provenance_from_evidence_vault(evidence_vault_lookup.get(vendor)),
            ),
        )
        _promote_account_reasoning_to_battle_card(card)

    # Enrich with ecosystem context
    try:
        from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer
        eco = EcosystemAnalyzer(pool)
        ecosystem_evidence = await eco.analyze_all_categories()
        for card in deterministic_battle_cards:
            cat = card.get("category", "")
            eco_context = _ecosystem_context_from_analysis(ecosystem_evidence.get(cat))
            if eco_context:
                card["ecosystem_context"] = eco_context
    except Exception:
        logger.debug("Ecosystem enrichment skipped", exc_info=True)

    # Enrich with cross-vendor battle conclusions + resource asymmetry
    for card in deterministic_battle_cards:
        vendor = card.get("vendor", "")
        category = card.get("category", "")

        # 1. Primary: Category reasoning from synthesis contract
        reasoning_council = _build_category_council_from_reasoning(card)
        if reasoning_council:
            card["category_council"] = reasoning_council

        # 2. Secondary: Cross-vendor council conclusion from latest run
        council = xv_lookup.get("councils", {}).get(category, {})
        xv_council = _build_category_council_from_cross_vendor(council)
        card["category_council"] = _prefer_richer_category_council(
            card.get("category_council"),
            xv_council,
        )

        # 3. Tertiary: Category dynamics pool fallback
        if not card.get("category_council"):
            cat_dyn = card.get("category_dynamics")
            if isinstance(cat_dyn, dict):
                cs = cat_dyn.get("council_summary")
                if isinstance(cs, dict) and (cs.get("conclusion") or cs.get("market_regime")):
                    card["category_council"] = {
                        "conclusion": cs.get("conclusion") or "",
                        "market_regime": cs.get("market_regime") or "",
                        "durability": cs.get("durability_assessment") or "",
                        "confidence": cs.get("confidence") or 0,
                        "winner": cs.get("winner") or "",
                        "loser": cs.get("loser") or "",
                        "key_insights": cs.get("key_insights") or [],
                    }
        # Last resort: ecosystem_context deterministic analysis
        if not card.get("category_council") and card.get("ecosystem_context"):
            fallback_council = _build_category_council_fallback(card)
            if fallback_council:
                card["category_council"] = fallback_council
        # Battle conclusions involving this vendor
        battles = []
        for pair_key, battle in xv_lookup.get("battles", {}).items():
            if vendor in pair_key:
                bc = battle.get("conclusion", {})
                opponent = _pair_opponent(pair_key, vendor)
                if not opponent:
                    continue
                battles.append({
                    "opponent": opponent,
                    "conclusion": bc.get("conclusion", ""),
                    "durability": bc.get("durability_assessment", ""),
                    "confidence": battle.get("confidence", 0),
                    "winner": bc.get("winner", ""),
                    "loser": bc.get("loser", ""),
                    "key_insights": bc.get("key_insights", []),
                })
        seller_battles = _battle_card_seller_usable_battles(vendor, battles)
        if seller_battles:
            card["cross_vendor_battles"] = seller_battles
        else:
            card.pop("cross_vendor_battles", None)
        # Resource asymmetry involving this vendor
        for pair_key, asym in xv_lookup.get("asymmetries", {}).items():
            if vendor in pair_key:
                opponent = _pair_opponent(pair_key, vendor)
                if not opponent:
                    continue
                card["resource_asymmetry"] = {
                    "opponent": opponent,
                    "conclusion": asym.get("conclusion", {}).get("conclusion", ""),
                    "resource_advantage": asym.get("conclusion", {}).get("resource_advantage", ""),
                    "confidence": asym.get("confidence", 0),
                }
                break  # first match is highest confidence (query ordered by confidence DESC)
        card["weakness_analysis"] = _build_deterministic_battle_card_weakness_analysis(card)
        _prioritize_seller_usable_primary_weakness(card)
        card["competitive_landscape"] = _build_deterministic_battle_card_competitive_landscape(card)
        _apply_battle_card_quality(card, phase=_QUALITY_PHASE_DETERMINISTIC)

    logger.info("Built %d deterministic battle cards", len(deterministic_battle_cards))
    total_cards = len(deterministic_battle_cards)
    await _update_execution_progress(
        task,
        stage=_STAGE_PERSISTING_DETERMINISTIC,
        progress_current=0,
        progress_total=total_cards,
        progress_message="Persisting deterministic battle cards.",
        cards_built=total_cards,
        cards_persisted=0,
    )

    data_density = json.dumps({"vendors_analyzed": vendor_total})
    report_source_review_count = data_context.get("reviews_in_analysis_window")
    report_source_dist = {
        src: info["reviews"] for src, info in data_context.get("source_distribution", {}).items()
    }

    cards_persisted = 0
    for card in deterministic_battle_cards:
        from ...reasoning.semantic_cache import compute_evidence_hash as _compute_render_hash

        initial_render_hash = _compute_render_hash(_build_battle_card_render_payload(card))
        _attach_battle_card_render_provenance(card, render_packet_hash=initial_render_hash)
        card["llm_render_status"] = "pending"
        card.pop("llm_render_error", None)
        try:
            persisted = await _persist_battle_card(
                pool,
                today=today,
                card=card,
                data_density=data_density,
                report_source_review_count=report_source_review_count,
                report_source_dist=report_source_dist,
                llm_model="pipeline_deterministic",
                status=_battle_card_row_status(card),
            )
        except Exception:
            logger.exception("Failed to persist deterministic battle card for %s", card.get("vendor"))
        else:
            cards_persisted += int(bool(persisted))
            await _update_execution_progress(
                task,
                stage=_STAGE_PERSISTING_DETERMINISTIC,
                progress_current=cards_persisted,
                progress_total=total_cards,
                progress_message="Persisting deterministic battle cards.",
                cards_built=total_cards,
                cards_persisted=cards_persisted,
            )

    logger.info(
        "Persisted %d/%d deterministic battle cards before LLM rendering",
        cards_persisted,
        len(deterministic_battle_cards),
    )
    await _update_execution_progress(
        task,
        stage=_STAGE_LLM_OVERLAY,
        progress_current=0,
        progress_total=total_cards,
        progress_message="Applying LLM overlay to battle cards.",
        cards_built=total_cards,
        cards_persisted=cards_persisted,
        cards_llm_updated=0,
        llm_failures=0,
        cache_hits=0,
    )

    # --- Phase 5: LLM sales copy (semantic cache first, Anthropic batch optional) ---
    from ...pipelines.llm import call_llm_with_skill, get_pipeline_llm
    from ...reasoning.semantic_cache import SemanticCache, CacheEntry, compute_evidence_hash
    from ...services.b2b.anthropic_batch import (
        AnthropicBatchItem,
        mark_batch_fallback_result,
        run_anthropic_message_batch,
    )
    from ...services.b2b.llm_exact_cache import build_skill_messages
    from ...services.protocols import Message
    from ..visibility import record_attempt, emit_event
    from ._b2b_batch_utils import (
        anthropic_batch_min_items,
        anthropic_batch_requested,
        is_anthropic_llm,
    )

    _bc_cache = SemanticCache(pool)
    bc_llm_failures = 0
    bc_cache_hits = 0
    bc_llm_updates = 0
    bc_overlay_completed = 0
    progress_lock = asyncio.Lock()
    bc_sem = asyncio.Semaphore(cfg.battle_card_llm_concurrency)
    max_attempts = cfg.battle_card_llm_attempts
    retry_delay = cfg.battle_card_llm_retry_delay_seconds
    feedback_limit = cfg.battle_card_llm_feedback_limit
    llm_max_tokens = cfg.battle_card_llm_max_tokens
    llm_max_input_tokens = max(
        512,
        int(getattr(cfg, "battle_card_llm_max_input_tokens", 25000)),
    )
    llm_temperature = cfg.battle_card_llm_temperature
    llm_timeout = cfg.battle_card_llm_timeout_seconds
    cache_confidence = cfg.battle_card_cache_confidence
    llm_options = _battle_card_llm_options(cfg)
    batch_requested = (
        anthropic_batch_requested(
            task,
            global_default=bool(getattr(settings.b2b_churn, "anthropic_batch_enabled", False)),
            task_default=bool(getattr(cfg, "battle_card_anthropic_batch_enabled", True)),
            task_keys=("battle_card_anthropic_batch_enabled",),
        )
        and str(llm_options.get("workload") or "").strip().lower() == "anthropic"
        and not bool(llm_options.get("try_openrouter"))
    )
    batch_llm = get_pipeline_llm(workload="anthropic") if batch_requested else None
    battle_card_batch_enabled = is_anthropic_llm(batch_llm)
    battle_card_batch_metrics = {
        "jobs": 0,
        "submitted_items": 0,
        "cache_prefiltered_items": 0,
        "fallback_single_call_items": 0,
        "completed_items": 0,
        "failed_items": 0,
    }

    async def _request_sales_copy(
        card: dict[str, Any],
        payload_input: str,
        *,
        attempt: int,
        usage_out: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sales_copy = await asyncio.wait_for(
            asyncio.to_thread(
                call_llm_with_skill,
                "digest/battle_card_sales_copy",
                payload_input,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                guided_json=_BATTLE_CARD_SALES_COPY_JSON_SCHEMA,
                response_format={"type": "json_object"},
                workload=llm_options["workload"],
                try_openrouter=llm_options["try_openrouter"],
                openrouter_model=llm_options["openrouter_model"],
                span_name="b2b.churn_intelligence.battle_card_sales_copy",
                trace_metadata=_battle_card_trace_metadata(
                    task,
                    card,
                    attempt=attempt,
                ),
                usage_out=usage_out,
            ),
            timeout=llm_timeout,
        )
        return _parse_battle_card_sales_copy(sales_copy)

    async def _persist_overlay_card(
        card: dict[str, Any],
        *,
        log_context: str,
        failure: bool = False,
        cache_hit: bool = False,
    ) -> bool:
        nonlocal bc_llm_failures, bc_cache_hits, bc_llm_updates, bc_overlay_completed

        persisted_ok = False
        try:
            persisted = await _persist_battle_card(
                pool,
                today=today,
                card=card,
                data_density=data_density,
                report_source_review_count=report_source_review_count,
                report_source_dist=report_source_dist,
                llm_model=_battle_card_llm_model_label(card, llm_options),
                status=_battle_card_row_status(card),
            )
        except Exception:
            logger.exception(
                "Failed to persist %s battle card for %s",
                log_context,
                card.get("vendor"),
            )
        else:
            persisted_ok = bool(persisted)

        async with progress_lock:
            bc_llm_failures += int(failure)
            bc_cache_hits += int(cache_hit)
            bc_llm_updates += int(persisted_ok)
            bc_overlay_completed += 1
            await _update_execution_progress(
                task,
                stage=_STAGE_LLM_OVERLAY,
                progress_current=bc_overlay_completed,
                progress_total=total_cards,
                progress_message="Applying LLM overlay to battle cards.",
                cards_built=total_cards,
                cards_persisted=cards_persisted,
                cards_llm_updated=bc_llm_updates,
                llm_failures=bc_llm_failures,
                cache_hits=bc_cache_hits,
            )
        return persisted_ok

    def _overlay_failure_step(
        failure_reasons: list[str],
        *,
        quality_gate: bool = False,
    ) -> str:
        if quality_gate:
            return "quality_gate"
        normalized = [str(reason or "").strip().lower() for reason in failure_reasons]
        if any(reason.startswith("transport failure:") for reason in normalized):
            return "transport"
        if any(reason == "llm did not return valid json" for reason in normalized):
            return "parse"
        return "response_validation"

    async def _record_overlay_attempt(
        card: dict[str, Any],
        *,
        attempt_no: int,
        status: str,
        failure_reasons: list[str] | None = None,
        quality: dict[str, Any] | None = None,
        failure_step: str | None = None,
    ) -> None:
        vendor = str(card.get("vendor") or "").strip()
        if attempt_no < 1 or not vendor:
            return
        quality_obj = quality if isinstance(quality, dict) else {}
        warning_items = quality_obj.get("warnings")
        warnings = [
            str(item).strip()
            for item in (warning_items if isinstance(warning_items, list) else [])
            if str(item).strip()
        ]
        blockers = [
            str(item).strip()
            for item in (failure_reasons or [])
            if str(item).strip()
        ]
        score = quality_obj.get("score")
        threshold = quality_obj.get("threshold")
        await record_attempt(
            pool,
            artifact_type="battle_card",
            artifact_id=vendor,
            run_id=str(task.id),
            attempt_no=attempt_no,
            stage="llm_overlay",
            status=status,
            score=score if isinstance(score, int) else None,
            threshold=threshold if isinstance(threshold, int) else None,
            blocker_count=len(blockers),
            warning_count=len(warnings),
            blocking_issues=blockers,
            warnings=warnings,
            feedback_summary="; ".join(blockers[:3]) or None,
            failure_step=failure_step,
            error_message="; ".join(blockers[:3]) or None,
        )

    async def _finalize_overlay_failure(
        card: dict[str, Any],
        *,
        attempt_no: int,
        failure_reasons: list[str],
        used_fallback_single_call: bool,
        fallback_usage: dict[str, Any],
        failure_step: str | None = None,
    ) -> dict[str, Any]:
        result_error = "; ".join(failure_reasons[:3]) or "Battle card generation failed"
        card["llm_render_status"] = "failed"
        card["llm_render_error"] = result_error
        logger.warning(
            "Battle card rejected for %s: %s",
            card.get("vendor"),
            result_error,
        )
        quality = _apply_battle_card_quality(card, phase=_QUALITY_PHASE_FINAL)
        await _record_overlay_attempt(
            card,
            attempt_no=attempt_no,
            status="failed",
            failure_reasons=failure_reasons,
            quality=quality,
            failure_step=failure_step or _overlay_failure_step(failure_reasons),
        )
        await _persist_overlay_card(
            card,
            log_context="rejected",
            failure=True,
        )
        return {
            "succeeded": False,
            "used_fallback_single_call": used_fallback_single_call,
            "error_text": result_error,
            "fallback_usage": fallback_usage,
        }

    async def _prepare_overlay_entry(
        index: int,
        card: dict[str, Any],
    ) -> dict[str, Any] | None:
        preflight_quality = _battle_card_preflight_quality_gate(card)
        if preflight_quality is not None:
            card["llm_render_status"] = "skipped_preflight_quality_gate"
            failed_checks = (
                preflight_quality.get("failed_checks")
                if isinstance(preflight_quality.get("failed_checks"), list)
                else []
            )
            if failed_checks:
                card["llm_render_error"] = "; ".join(str(item) for item in failed_checks[:3])
            await _persist_overlay_card(
                card,
                log_context="preflight-gated",
                failure=True,
            )
            return None

        payload = _build_battle_card_render_payload(card)
        card_hash = compute_evidence_hash(payload)
        _attach_battle_card_render_provenance(card, render_packet_hash=card_hash)
        pattern_sig = f"battle_card:{card.get('vendor')}:{card_hash}"

        cached = await _bc_cache.lookup(pattern_sig)
        if cached:
            cached_errors = _validate_battle_card_sales_copy(card, cached.conclusion)
            if cached_errors:
                await _bc_cache.invalidate(pattern_sig, reason="invalid")
            else:
                for _cf in cached.conclusion:
                    card[_cf] = cached.conclusion[_cf]
                card["llm_render_status"] = "cached"
                card.pop("llm_render_error", None)
                quality = _apply_battle_card_quality(card, phase=_QUALITY_PHASE_FINAL)
                if quality.get("status") == _QUALITY_STATUS_FALLBACK:
                    await _bc_cache.invalidate(pattern_sig, reason="quality_gate")
                    _drop_llm_sales_copy(card)
                    _populate_battle_card_fallback_sales_copy(card)
                    card["llm_render_status"] = "failed_quality_gate"
                    failed_checks = quality.get("failed_checks") if isinstance(quality.get("failed_checks"), list) else []
                    if failed_checks:
                        card["llm_render_error"] = "; ".join(str(item) for item in failed_checks[:3])
                else:
                    await _bc_cache.validate(pattern_sig)
                await _persist_overlay_card(
                    card,
                    log_context="cached",
                    failure=quality.get("status") == _QUALITY_STATUS_FALLBACK,
                    cache_hit=quality.get("status") != _QUALITY_STATUS_FALLBACK,
                )
                return None

        payload_input = json.dumps(payload, default=str)
        request_messages = build_skill_messages(
            "digest/battle_card_sales_copy",
            payload_input,
        )
        estimated_input_tokens = _approx_message_input_tokens(request_messages)
        if estimated_input_tokens > llm_max_input_tokens:
            logger.warning(
                "Skipping battle card LLM for %s: estimated input %d exceeds cap %d",
                card.get("vendor"),
                estimated_input_tokens,
                llm_max_input_tokens,
            )
            _populate_battle_card_fallback_sales_copy(card)
            card["llm_render_status"] = "skipped_input_cap"
            card["llm_render_error"] = (
                "LLM input exceeded configured token budget "
                f"({estimated_input_tokens}>{llm_max_input_tokens})"
            )
            _apply_battle_card_quality(card, phase=_QUALITY_PHASE_FINAL)
            await _persist_overlay_card(
                card,
                log_context="input-capped",
                failure=True,
            )
            return None

        return {
            "index": index,
            "card": card,
            "payload": payload,
            "payload_input": payload_input,
            "pattern_sig": pattern_sig,
            "card_hash": card_hash,
            "request_messages": request_messages,
            "custom_id": (
                f"battle_card:{index}:"
                f"{str(card.get('vendor') or '').strip().lower()}"
            ),
        }

    async def _run_overlay_entry(
        entry: dict[str, Any],
        *,
        initial_response_text: str | None = None,
        initial_usage: dict[str, Any] | None = None,
        batch_origin: bool = False,
        initial_attempt_consumed: bool = False,
    ) -> dict[str, Any]:
        card = entry["card"]
        payload = entry["payload"]
        payload_input = str(entry["payload_input"])
        pattern_sig = str(entry["pattern_sig"])
        card_hash = str(entry["card_hash"])

        failure_reasons: list[str] = []
        parsed_copy: dict[str, Any] = {}
        candidate_for_retry: Any = {}
        render_succeeded = False
        used_fallback_single_call = False
        result_error: str | None = None
        fallback_usage: dict[str, Any] = {}
        successful_attempt_no = 0
        last_attempt_no = 0

        def _accumulate_usage(sample: dict[str, Any] | None) -> None:
            if not sample:
                return
            fallback_usage["input_tokens"] = int(fallback_usage.get("input_tokens") or 0) + int(sample.get("input_tokens") or 0)
            fallback_usage["billable_input_tokens"] = int(fallback_usage.get("billable_input_tokens") or 0) + int(
                sample.get("billable_input_tokens")
                if sample.get("billable_input_tokens") is not None
                else sample.get("input_tokens") or 0
            )
            fallback_usage["cached_tokens"] = int(fallback_usage.get("cached_tokens") or 0) + int(sample.get("cached_tokens") or 0)
            fallback_usage["cache_write_tokens"] = int(fallback_usage.get("cache_write_tokens") or 0) + int(sample.get("cache_write_tokens") or 0)
            fallback_usage["output_tokens"] = int(fallback_usage.get("output_tokens") or 0) + int(sample.get("output_tokens") or 0)
            if sample.get("provider"):
                fallback_usage["provider"] = str(sample.get("provider") or "")
            if sample.get("model"):
                fallback_usage["model"] = str(sample.get("model") or "")
            if sample.get("provider_request_id"):
                fallback_usage["provider_request_id"] = str(sample.get("provider_request_id") or "")

        def _consume_parsed_copy(candidate: dict[str, Any]) -> bool:
            nonlocal parsed_copy, candidate_for_retry, failure_reasons, render_succeeded

            parsed_copy = candidate if isinstance(candidate, dict) else {}
            candidate_for_retry = parsed_copy if isinstance(parsed_copy, dict) else {}
            if parsed_copy.get("_parse_fallback"):
                failure_reasons = ["LLM did not return valid JSON"]
                candidate_for_retry = {}
                return False

            copy_errors = _validate_battle_card_sales_copy(card, parsed_copy)
            if copy_errors:
                sanitized_copy = _sanitize_battle_card_sales_copy(card, parsed_copy)
                sanitized_errors = _validate_battle_card_sales_copy(card, sanitized_copy)
                candidate_for_retry = (
                    sanitized_copy if isinstance(sanitized_copy, dict) else candidate_for_retry
                )
                if not sanitized_errors:
                    parsed_copy = sanitized_copy
                    copy_errors = []
                else:
                    failure_reasons = sanitized_errors
            if copy_errors:
                if not failure_reasons:
                    failure_reasons = copy_errors
                return False

            for field in _BATTLE_CARD_LLM_FIELDS:
                if field in parsed_copy:
                    card[field] = parsed_copy[field]
            card["llm_render_status"] = "succeeded"
            card.pop("llm_render_error", None)
            render_succeeded = True
            return True

        if initial_response_text is not None:
            last_attempt_no = 1
            if _consume_parsed_copy(_parse_battle_card_sales_copy(initial_response_text)):
                successful_attempt_no = 1

        if not render_succeeded:
            if batch_origin:
                used_fallback_single_call = True
            loop_start = 1 if initial_attempt_consumed else 0
            if initial_attempt_consumed and initial_response_text is not None:
                if max_attempts <= 1:
                    return await _finalize_overlay_failure(
                        card,
                        attempt_no=1,
                        failure_reasons=failure_reasons or ["LLM did not return valid JSON"],
                        used_fallback_single_call=used_fallback_single_call,
                        fallback_usage=fallback_usage,
                    )
                await _record_overlay_attempt(
                    card,
                    attempt_no=1,
                    status="retry_requested",
                    failure_reasons=failure_reasons or ["LLM did not return valid JSON"],
                    failure_step=_overlay_failure_step(
                        failure_reasons or ["LLM did not return valid JSON"]
                    ),
                )
            async with bc_sem:
                for attempt in range(loop_start, max_attempts):
                    last_attempt_no = attempt + 1
                    try:
                        attempt_usage: dict[str, Any] = {}
                        parsed_copy = await _request_sales_copy(
                            card,
                            payload_input,
                            attempt=attempt + 1,
                            usage_out=attempt_usage,
                        )
                        _accumulate_usage(attempt_usage)
                    except Exception as exc:
                        parsed_copy = {}
                        candidate_for_retry = {}
                        failure_reasons = [f"transport failure: {type(exc).__name__}"]
                    else:
                        if _consume_parsed_copy(parsed_copy):
                            successful_attempt_no = attempt + 1
                            break

                    if attempt + 1 >= max_attempts:
                        return await _finalize_overlay_failure(
                            card,
                            attempt_no=attempt + 1,
                            failure_reasons=failure_reasons,
                            used_fallback_single_call=used_fallback_single_call,
                            fallback_usage=fallback_usage,
                            failure_step=_overlay_failure_step(failure_reasons),
                        )

                    await _record_overlay_attempt(
                        card,
                        attempt_no=attempt + 1,
                        status="retry_requested",
                        failure_reasons=failure_reasons,
                        failure_step=_overlay_failure_step(failure_reasons),
                    )

                    payload = _build_battle_card_render_payload(
                        card,
                        prior_attempt=_battle_card_prior_attempt(
                            candidate_for_retry or parsed_copy,
                        ),
                        validation_feedback=failure_reasons[:feedback_limit],
                    )
                    payload_input = json.dumps(payload, default=str)
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)

            if not render_succeeded:
                return await _finalize_overlay_failure(
                    card,
                    attempt_no=max(last_attempt_no, 1),
                    failure_reasons=failure_reasons or ["Battle card generation failed"],
                    used_fallback_single_call=used_fallback_single_call,
                    fallback_usage=fallback_usage,
                )

        quality = _apply_battle_card_quality(card, phase=_QUALITY_PHASE_FINAL)
        if quality.get("status") == _QUALITY_STATUS_FALLBACK:
            _drop_llm_sales_copy(card)
            _populate_battle_card_fallback_sales_copy(card)
            card["llm_render_status"] = "failed_quality_gate"
            failed_checks = (
                quality.get("failed_checks")
                if isinstance(quality.get("failed_checks"), list)
                else []
            )
            if failed_checks:
                result_error = "; ".join(str(item) for item in failed_checks[:3])
                card["llm_render_error"] = result_error
            await _record_overlay_attempt(
                card,
                attempt_no=max(successful_attempt_no, 1),
                status="failed",
                failure_reasons=failed_checks,
                quality=quality,
                failure_step=_overlay_failure_step(failed_checks, quality_gate=True),
            )
            await _persist_overlay_card(
                card,
                log_context="quality-gated",
                failure=True,
            )
            return {
                "succeeded": False,
                "used_fallback_single_call": used_fallback_single_call,
                "error_text": result_error,
                    "fallback_usage": fallback_usage,
            }

        await _record_overlay_attempt(
            card,
            attempt_no=max(successful_attempt_no, 1),
            status="succeeded",
            quality=quality,
        )

        try:
            await _bc_cache.store(
                CacheEntry(
                    pattern_sig=pattern_sig,
                    pattern_class="battle_card_sales_copy",
                    conclusion={
                        field: card[field]
                        for field in _BATTLE_CARD_LLM_FIELDS
                        if field in card
                    },
                    confidence=cache_confidence,
                    evidence_hash=card_hash,
                    vendor_name=card.get("vendor"),
                    conclusion_type="sales_copy",
                )
            )
        except Exception:
            logger.warning("Failed to cache battle card for %s", card.get("vendor"))

        await _persist_overlay_card(
            card,
            log_context="enriched",
        )
        return {
            "succeeded": True,
            "used_fallback_single_call": used_fallback_single_call,
            "error_text": None,
            "fallback_usage": fallback_usage,
        }

    overlay_entries: list[dict[str, Any]] = []
    for index, card in enumerate(deterministic_battle_cards):
        entry = await _prepare_overlay_entry(index, card)
        if entry is not None:
            overlay_entries.append(entry)

    if battle_card_batch_enabled and overlay_entries:
        execution = await run_anthropic_message_batch(
            llm=batch_llm,
            stage_id="b2b_battle_cards.sales_copy",
            task_name="b2b_battle_cards",
            items=[
                AnthropicBatchItem(
                    custom_id=str(entry["custom_id"]),
                    artifact_type="battle_card_sales_copy",
                    artifact_id=str(entry["card"].get("vendor") or f"battle-card-{entry['index']}"),
                    vendor_name=str(entry["card"].get("vendor") or "") or None,
                    messages=[
                        Message(
                            role=str(message.get("role") or ""),
                            content=str(message.get("content") or ""),
                        )
                        for message in entry["request_messages"]
                    ],
                    max_tokens=llm_max_tokens,
                    temperature=llm_temperature,
                    trace_span_name="b2b.churn_intelligence.battle_card_sales_copy",
                    trace_metadata={
                        **_battle_card_trace_metadata(
                            task,
                            entry["card"],
                            attempt=1,
                        ),
                        "workload": "anthropic_batch",
                    },
                    request_metadata={
                        "report_type": "battle_card",
                    },
                )
                for entry in overlay_entries
            ],
            run_id=str(task.id),
            min_batch_size=anthropic_batch_min_items(
                task,
                default=int(getattr(cfg, "battle_card_anthropic_batch_min_items", 2)),
                keys=("battle_card_anthropic_batch_min_items",),
            ),
            batch_metadata={
                "report_type": "battle_card",
            },
            pool=pool,
        )
        battle_card_batch_metrics["jobs"] += 1 if execution.provider_batch_id else 0
        battle_card_batch_metrics["submitted_items"] += execution.submitted_items
        battle_card_batch_metrics["cache_prefiltered_items"] += execution.cache_prefiltered_items
        battle_card_batch_metrics["fallback_single_call_items"] += execution.fallback_single_call_items
        battle_card_batch_metrics["completed_items"] += execution.completed_items
        battle_card_batch_metrics["failed_items"] += execution.failed_items

        async def _run_batched_entry(entry: dict[str, Any]) -> None:
            outcome = execution.results_by_custom_id.get(str(entry["custom_id"]))
            result = await _run_overlay_entry(
                entry,
                initial_response_text=(
                    outcome.response_text
                    if outcome is not None and outcome.response_text
                    else None
                ),
                initial_usage=outcome.usage if outcome is not None else None,
                batch_origin=True,
                initial_attempt_consumed=bool(
                    outcome is not None and outcome.response_text
                ),
            )
            if result["used_fallback_single_call"]:
                await mark_batch_fallback_result(
                    batch_id=execution.local_batch_id,
                    custom_id=str(entry["custom_id"]),
                    succeeded=bool(result["succeeded"]),
                    error_text=(
                        outcome.error_text
                        if outcome is not None
                        and outcome.error_text
                        and not result["succeeded"]
                        else result["error_text"]
                    ),
                    usage=result.get("fallback_usage"),
                    provider=str((result.get("fallback_usage") or {}).get("provider") or "") or None,
                    model=str((result.get("fallback_usage") or {}).get("model") or "") or None,
                    provider_request_id=(
                        str((result.get("fallback_usage") or {}).get("provider_request_id") or "") or None
                    ),
                    pool=pool,
                )

        await asyncio.gather(*[
            _run_batched_entry(entry)
            for entry in overlay_entries
        ])
    else:
        await asyncio.gather(*[
            _run_overlay_entry(entry)
            for entry in overlay_entries
        ])

    logger.info(
        "Battle card LLM: %d cache hits, %d generated, %d failed (of %d)",
        bc_cache_hits,
        len(deterministic_battle_cards) - bc_cache_hits - bc_llm_failures,
        bc_llm_failures,
        len(deterministic_battle_cards),
    )
    logger.info(
        "Battle card overlay writes: %d/%d cards updated after LLM phase",
        bc_llm_updates,
        len(deterministic_battle_cards),
    )
    await _update_execution_progress(
        task,
        stage=_STAGE_FINALIZING,
        progress_current=total_cards,
        progress_total=total_cards,
        progress_message="Finalizing battle-card execution status.",
        cards_built=total_cards,
        cards_persisted=cards_persisted,
        cards_llm_updated=bc_llm_updates,
        llm_failures=bc_llm_failures,
        cache_hits=bc_cache_hits,
        cards_gated_out=len(gated_out_vendors),
    )

    # Record battle card run summary to visibility system
    await record_attempt(
        pool, artifact_type="battle_card", artifact_id="batch",
        run_id=str(task.id), stage="generation",
        status="succeeded" if bc_llm_failures == 0 else "failed",
        score=cards_persisted,
        blocker_count=bc_llm_failures,
        warning_count=len(gated_out_vendors),
        error_message=f"{bc_llm_failures} LLM failures" if bc_llm_failures else None,
    )
    if bc_llm_failures > 0:
        await emit_event(
            pool, stage="battle_cards", event_type="llm_failures",
            entity_type="battle_card", entity_id="batch",
            summary=f"{bc_llm_failures}/{len(deterministic_battle_cards)} battle cards failed LLM render",
            severity="warning", actionable=bc_llm_failures > 3,
            run_id=str(task.id),
            reason_code="llm_render_failure",
            detail={"failures": bc_llm_failures, "total": len(deterministic_battle_cards),
                    "gated_out": gated_out_vendors},
        )

    return {
        "_skip_synthesis": "B2B battle cards complete",
        "cards_built": len(deterministic_battle_cards),
        "cards_persisted": cards_persisted,
        "cards_retired": cards_retired,
        "cards_llm_updated": bc_llm_updates,
        "cache_hits": bc_cache_hits,
        "llm_failures": bc_llm_failures,
        "battle_card_batch_jobs": battle_card_batch_metrics["jobs"],
        "battle_card_batch_items_submitted": battle_card_batch_metrics["submitted_items"],
        "battle_card_batch_cache_prefiltered": battle_card_batch_metrics["cache_prefiltered_items"],
        "battle_card_batch_fallback_single_call": battle_card_batch_metrics["fallback_single_call_items"],
        "battle_card_batch_completed_items": battle_card_batch_metrics["completed_items"],
        "battle_card_batch_failed_items": battle_card_batch_metrics["failed_items"],
        "reasoning_vendors": len(reasoning_lookup),
        "cards_gated_out": len(gated_out_vendors),
        "gated_out_vendors": gated_out_vendors,
    }
