"""Follow-up task: build and LLM-enrich battle cards per vendor.

Runs after b2b_churn_core. Reads persisted artifacts from
b2b_churn_signals, b2b_reviews, and b2b_product_profiles. Builds
deterministic battle cards, runs LLM sales copy generation in parallel,
and persists to b2b_intelligence.
"""

import asyncio
import json
import logging
from datetime import date, datetime
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._execution_progress import _update_execution_progress

logger = logging.getLogger("atlas.tasks.b2b_battle_cards")

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
    backend = str(getattr(cfg, "battle_card_llm_backend", "auto") or "auto").strip().lower()
    model = (
        str(getattr(cfg, "battle_card_openrouter_model", "") or "").strip()
        or str(settings.llm.openrouter_reasoning_model or "").strip()
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
    if not isinstance(numeric_literals, dict):
        return tokens
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
    return tokens


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
            time_anchor = str(witness.get("time_anchor") or "").strip().lower()
            if time_anchor:
                timing_terms.add(time_anchor)
            numeric_terms.update(_witness_numeric_tokens(witness))
    return {
        "companies": companies,
        "timing_terms": timing_terms,
        "numeric_terms": numeric_terms,
        "competitor_terms": competitor_terms,
    }


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
    max_stale_days = int(getattr(cfg, "battle_card_quality_max_stale_days", 1))
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
        if len(plays) < min_total_plays or actionable_play_count < min_actionable_plays:
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
        if len(plays) < min_total_plays:
            hard_blockers.append(
                f"recommended_plays must contain at least {min_total_plays} distinct motions"
            )
        if actionable_play_count < min_actionable_plays:
            hard_blockers.append(
                "recommended plays are missing role/account targeting + timing + CTA"
            )
        if len(set(target_segments)) < 2 and len([seg for seg in target_segments if seg]) >= 2:
            hard_blockers.append("recommended plays repeat the same target segment")
        if isinstance(card.get("recommended_plays"), list) and _battle_card_has_duplicate_recommended_play_segments({"recommended_plays": card.get("recommended_plays")}):
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
                hard_blockers.append(
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
    quality = _evaluate_battle_card_quality(card, phase=phase)
    card["battle_card_quality"] = quality
    card["quality_status"] = quality.get("status") or _QUALITY_STATUS_NEEDS_REVIEW
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

    if isinstance(reasoning_contracts, dict) and reasoning_contracts:
        payload["reasoning_contracts"] = reasoning_contracts
    if vendor_core_reasoning:
        payload["vendor_core_reasoning"] = vendor_core_reasoning
    if displacement_reasoning:
        payload["displacement_reasoning"] = displacement_reasoning
    if category_reasoning:
        payload["category_reasoning"] = category_reasoning
    if account_reasoning:
        payload["account_reasoning"] = account_reasoning

    payload["locked_facts"] = _build_battle_card_locked_facts(card)
    payload["render_packet_version"] = "contract_first_v1"
    metric_ledger = _build_metric_ledger(card)
    if metric_ledger:
        payload["metric_ledger"] = metric_ledger
    anchor_examples = _battle_card_anchor_examples(card)
    if anchor_examples:
        payload["anchor_examples"] = anchor_examples
    witness_highlights = _battle_card_witness_highlights(card)
    if witness_highlights:
        payload["witness_highlights"] = witness_highlights
    reference_ids = card.get("reference_ids")
    if isinstance(reference_ids, dict) and reference_ids:
        payload["reference_ids"] = reference_ids

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
                payload["contradictions"] = contradictions
            cg = eg.get("coverage_gaps")
            if isinstance(cg, list) and cg:
                payload["coverage_gaps"] = cg

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
    await pool.execute(
        """
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
        """,
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
    """Return today's date if core task wrote a completion marker, else None."""
    today = date.today()
    marker = await pool.fetchval(
        "SELECT 1 FROM b2b_intelligence "
        "WHERE report_type = 'core_run_complete' AND report_date = $1",
        today,
    )
    if not marker:
        logger.info("Core run not complete for %s, skipping", today)
        return None
    return today


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Build battle cards + LLM sales copy from persisted artifacts."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    today = await _check_freshness(pool)
    if today is None:
        return {"_skip_synthesis": "Core signals not fresh for today"}

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
        _sanitize_battle_card_sales_copy,
        _validate_battle_card_sales_copy,
        _fetch_vendor_churn_scores_from_signals,
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
        _fetch_latest_evidence_vault,
        _fetch_latest_account_intelligence,
        _fetch_review_text_aggregates,
        _fetch_department_distribution,
        _fetch_contract_context_distribution,
        _fetch_competitive_displacement_source_of_truth,
    )
    from .b2b_churn_intelligence import (
        _apply_vendor_scope_to_churn_inputs,
        _normalize_test_vendors,
        reconstruct_reasoning_lookup,
        reconstruct_cross_vendor_lookup,
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
            evidence_vault_lookup,
            account_intel_lookup,
            product_profiles_raw,
            review_text_agg, department_dist, contract_ctx,
        ) = await asyncio.gather(
            _fetch_vendor_churn_scores_from_signals(pool, window_days, min_reviews),
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
            _fetch_latest_evidence_vault(pool, as_of=today, analysis_window_days=window_days),
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
        vendor_scope = {v.lower() for v in scoped_vendors}
        evidence_vault_lookup = {
            k: v for k, v in (evidence_vault_lookup or {}).items()
            if str(k or "").strip().lower() in vendor_scope
        }
        logger.info(
            "Scoped battle cards to %d/%d vendors for test run: %s",
            len(vendor_scores),
            raw_vendor_count,
            sorted(scoped_vendors),
        )

    if not vendor_scores:
        return {"_skip_synthesis": "No vendor scores after vendor scope filter"}
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

    # Build reasoning_lookup synthesis-first, legacy fills gaps
    legacy_lookup = await reconstruct_reasoning_lookup(pool, as_of=today)
    synth_lookup = build_reasoning_lookup_from_views(synthesis_views)
    reasoning_lookup = {**legacy_lookup, **synth_lookup}
    if scoped_vendors:
        vendor_scope = {v.lower() for v in scoped_vendors}
        reasoning_lookup = {
            k: v for k, v in reasoning_lookup.items()
            if str(k or "").strip().lower() in vendor_scope
        }
    # Prefer cross-vendor synthesis; fall back to legacy conclusions
    from ._b2b_cross_vendor_synthesis import load_cross_vendor_synthesis_lookup
    try:
        xv_synth = await load_cross_vendor_synthesis_lookup(
            pool, as_of=today, analysis_window_days=window_days,
        )
    except Exception:
        logger.debug("Cross-vendor synthesis lookup failed, using legacy", exc_info=True)
        xv_synth = {"battles": {}, "councils": {}, "asymmetries": {}}
    xv_legacy = await reconstruct_cross_vendor_lookup(pool, as_of=today)
    # Merge: synthesis wins per key, legacy fills gaps
    xv_lookup: dict[str, dict] = {"battles": {}, "councils": {}, "asymmetries": {}}
    dedup_overrides = 0
    for bucket in ("battles", "councils", "asymmetries"):
        merged = dict(xv_legacy.get(bucket, {}))
        synth_bucket = xv_synth.get(bucket, {})
        for k, v in synth_bucket.items():
            if k in merged:
                dedup_overrides += 1
            merged[k] = v
        xv_lookup[bucket] = merged
    if dedup_overrides > 0:
        from ..visibility import emit_event
        await emit_event(
            pool, stage="battle_cards", event_type="xv_dedup_override",
            entity_type="cross_vendor", entity_id="batch",
            summary=f"Synthesis replaced {dedup_overrides} legacy cross-vendor entries",
            severity="info",
            run_id=str(task.id),
            reason_code="synthesis_preferred",
            detail={"overrides": dedup_overrides},
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

    # --- Phase 5: LLM sales copy (parallel with semantic cache) ---
    from ...pipelines.llm import call_llm_with_skill
    from ...reasoning.semantic_cache import SemanticCache, CacheEntry, compute_evidence_hash

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
    llm_temperature = cfg.battle_card_llm_temperature
    llm_timeout = cfg.battle_card_llm_timeout_seconds
    cache_confidence = cfg.battle_card_cache_confidence
    llm_options = _battle_card_llm_options(cfg)

    async def _request_sales_copy(payload: dict[str, Any]) -> dict[str, Any]:
        sales_copy = await asyncio.wait_for(
            asyncio.to_thread(
                call_llm_with_skill,
                "digest/battle_card_sales_copy",
                json.dumps(payload, default=str),
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                guided_json=_BATTLE_CARD_SALES_COPY_JSON_SCHEMA,
                response_format={"type": "json_object"},
                workload=llm_options["workload"],
                try_openrouter=llm_options["try_openrouter"],
                openrouter_model=llm_options["openrouter_model"],
            ),
            timeout=llm_timeout,
        )
        return _parse_battle_card_sales_copy(sales_copy)

    async def _enrich_one(card: dict[str, Any]) -> None:
        nonlocal bc_llm_failures, bc_cache_hits, bc_llm_updates, bc_overlay_completed

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
                    card["llm_render_status"] = "failed_quality_gate"
                    failed_checks = quality.get("failed_checks") if isinstance(quality.get("failed_checks"), list) else []
                    if failed_checks:
                        card["llm_render_error"] = "; ".join(str(item) for item in failed_checks[:3])
                else:
                    await _bc_cache.validate(pattern_sig)
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
                    logger.exception("Failed to persist cached battle card for %s", card.get("vendor"))
                else:
                    persisted_ok = bool(persisted)
                async with progress_lock:
                    if quality.get("status") == _QUALITY_STATUS_FALLBACK:
                        bc_llm_failures += 1
                    else:
                        bc_cache_hits += 1
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
                return

        async with bc_sem:
            failure_reasons: list[str] = []
            render_succeeded = False
            for attempt in range(max_attempts):
                try:
                    parsed_copy = await _request_sales_copy(payload)
                except Exception as exc:
                    parsed_copy = {}
                    failure_reasons = [f"transport failure: {type(exc).__name__}"]
                else:
                    if parsed_copy.get("_parse_fallback"):
                        failure_reasons = ["LLM did not return valid JSON"]
                    else:
                        copy_errors = _validate_battle_card_sales_copy(card, parsed_copy)
                        if copy_errors:
                            sanitized_copy = _sanitize_battle_card_sales_copy(card, parsed_copy)
                            sanitized_errors = _validate_battle_card_sales_copy(card, sanitized_copy)
                            if not sanitized_errors:
                                parsed_copy = sanitized_copy
                                copy_errors = []
                        if not copy_errors:
                            for _f in _BATTLE_CARD_LLM_FIELDS:
                                if _f in parsed_copy:
                                    card[_f] = parsed_copy[_f]
                            card["llm_render_status"] = "succeeded"
                            card.pop("llm_render_error", None)
                            render_succeeded = True
                            break
                        failure_reasons = copy_errors
                if attempt + 1 >= max_attempts:
                    card["llm_render_status"] = "failed"
                    if failure_reasons:
                        card["llm_render_error"] = "; ".join(failure_reasons[:3])
                    logger.warning("Battle card rejected for %s: %s",
                                   card.get("vendor"), "; ".join(failure_reasons[:3]))
                    _apply_battle_card_quality(card, phase=_QUALITY_PHASE_FINAL)
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
                        logger.exception("Failed to persist rejected battle card for %s", card.get("vendor"))
                    else:
                        persisted_ok = bool(persisted)
                    async with progress_lock:
                        bc_llm_failures += 1
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
                    return
                payload = _build_battle_card_render_payload(
                    card,
                    prior_attempt=_battle_card_prior_attempt(parsed_copy),
                    validation_feedback=failure_reasons[:feedback_limit],
                )
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)

            if not render_succeeded:
                return

            quality = _apply_battle_card_quality(card, phase=_QUALITY_PHASE_FINAL)
            if quality.get("status") == _QUALITY_STATUS_FALLBACK:
                _drop_llm_sales_copy(card)
                card["llm_render_status"] = "failed_quality_gate"
                failed_checks = quality.get("failed_checks") if isinstance(quality.get("failed_checks"), list) else []
                if failed_checks:
                    card["llm_render_error"] = "; ".join(str(item) for item in failed_checks[:3])
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
                    logger.exception("Failed to persist quality-gated battle card for %s", card.get("vendor"))
                    persisted_ok = False
                else:
                    persisted_ok = bool(persisted)
                async with progress_lock:
                    bc_llm_failures += 1
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
                return

            try:
                await _bc_cache.store(CacheEntry(
                    pattern_sig=pattern_sig,
                    pattern_class="battle_card_sales_copy",
                    conclusion={_f: card[_f] for _f in _BATTLE_CARD_LLM_FIELDS if _f in card},
                    confidence=cache_confidence,
                    evidence_hash=card_hash,
                    vendor_name=card.get("vendor"),
                    conclusion_type="sales_copy",
                ))
            except Exception:
                logger.warning("Failed to cache battle card for %s", card.get("vendor"))
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
                logger.exception("Failed to persist enriched battle card for %s", card.get("vendor"))
                persisted_ok = False
            else:
                persisted_ok = bool(persisted)
            async with progress_lock:
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

    await asyncio.gather(*[_enrich_one(c) for c in deterministic_battle_cards])

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
    from ..visibility import record_attempt, emit_event
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
        "reasoning_vendors": len(reasoning_lookup),
        "cards_gated_out": len(gated_out_vendors),
        "gated_out_vendors": gated_out_vendors,
    }
