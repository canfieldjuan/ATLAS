"""Deterministic review evidence spans and vendor witness-pack assembly.

This module is the bridge between per-review extraction and vendor-level
reasoning synthesis. It keeps witness selection deterministic and inspectable:
reviews emit stable evidence spans, then Stage 4 builds a compact witness pack
plus section packets from those spans.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("atlas.witnesses")

_CURRENCY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s*/\s*(?:mo|month|yr|year|seat))?", re.I)
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_TIMING_PATTERNS = (
    "renewal", "deadline", "quarter", "q1", "q2", "q3", "q4",
    "30 days", "60 days", "90 days", "this month", "this week", "next quarter",
)
_WORKFLOW_SUBSTITUTION_PATTERNS = (
    "async", "documentation", "docs", "wiki", "notion", "confluence",
    "spreadsheet", "ticketing", "tickets", "email", "shared doc", "shared docs",
)
_BUNDLE_PATTERNS = (
    "bundle", "bundled", "suite", "workspace", "microsoft 365", "google workspace",
    "already included", "included with", "all-in-one", "one platform",
    "single system", "one place", "everything in one place", "source of truth",
)
_INTERNAL_TOOL_PATTERNS = (
    "internal tool", "in-house", "homegrown", "home-grown", "custom tool",
    "built our own",
)
_PRODUCTIVITY_POSITIVE_PATTERNS = (
    "more productive", "faster without", "better without", "easier without",
    "more efficient", "save time", "saves time", "time savings", "sped up",
    "streamlined workflow", "reduced manual work", "less manual work",
    "less labor-intensive", "less labour-intensive", "eliminating manual data entry",
    "faster incident response", "speeds incident response",
)
_PRODUCTIVITY_NEGATIVE_PATTERNS = (
    "less productive", "slower now", "harder to work", "lost productivity",
    "slowed us down", "time consuming", "takes too long", "waste of time",
    "more manual work", "too many manual steps",
)
_PRODUCTIVITY_POSITIVE_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bstreamlin(?:e|ed|es|ing)\b", re.I),
    re.compile(r"\bsimplif(?:y|ies|ied|ying)\b.{0,40}\b(process(?:es)?|workflow(?:s)?|management|tasks?|work)\b", re.I),
    re.compile(r"\breduc(?:e|ed|es|ing)\b.{0,40}\b(administrative complexity|manual data entry|response time)\b", re.I),
    re.compile(r"\bfre(?:e|ed|es|ing)\b.{0,40}\b(team|soc team|staff)\b.{0,40}\bfocus\b", re.I),
    re.compile(r"\bboost(?:ed|ing)?\b.{0,40}\b(productivity|roi)\b", re.I),
)
_PRODUCTIVITY_NEGATIVE_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmanual\b.{0,24}\bnightmare\b", re.I),
    re.compile(r"\bslower\b.{0,24}\bresponse\b", re.I),
)
_PROCUREMENT_PATTERNS = (
    "procurement", "vendor standardization", "approved vendor", "approved software",
    "approved list", "security review", "legal review", "vendor review",
)
_STANDARDIZATION_PATTERNS = (
    "standardized on", "standardisation", "standardization", "preferred vendor",
    "company standard", "it standard", "standard tool",
)
_BUDGET_FREEZE_PATTERNS = (
    "budget freeze", "spend freeze", "cost cutting", "cost-cutting",
    "budget cuts", "spend reduction",
)
_PRICING_CONTEXT_PATTERNS = (
    "pricing", "price", "priced", "cost", "costs", "costly", "expensive",
    "cheaper", "budget", "billing", "invoice", "refund", "overcharg",
    "renewal", "quoted", "quote", "per seat", "per user", "subscription",
    "license", "licensed", "plan", "monthly", "annually", "/mo", "/month",
    "/yr", "/year", "seat", "user",
)
_QUOTA_ORDER: tuple[tuple[str, int], ...] = (
    ("common_pattern", 2),
    ("named_account", 2),
    ("displacement", 2),
    ("timing", 1),
    ("outlier", 1),
    ("category_shift", 1),
    ("counterevidence", 1),
)
_GENERIC_WITNESS_PATTERNS = (
    "great tool",
    "great platform",
    "good tool",
    "good platform",
    "easy to use",
    "easy-to-use",
    "works well",
    "working well",
    "very helpful",
    "super helpful",
)
_CONCRETE_PAIN_PATTERNS = (
    "pricing",
    "renewal",
    "seat",
    "integration",
    "workflow",
    "contract",
    "budget",
    "support",
    "security",
    "migration",
    "implementation",
    "downtime",
    "latency",
)
_POSITIVE_SENTIMENT_DIRECTIONS = {
    "stable_positive",
    "consistently_positive",
    "improving",
}
_NEGATIVE_SENTIMENT_DIRECTIONS = {
    "consistently_negative",
    "declining",
    "negative",
}
_NEGATIVE_RECOMMENDATION_PATTERNS = (
    "never use",
    "stay away",
    "would not recommend",
    "wouldn't recommend",
    "do not recommend",
    "don't recommend",
    "avoid this company",
    "avoid this service",
    "steer clear",
)
_MIN_GENERIC_ANCHOR_SCORE = 2.0
_SHORT_GENERIC_EXCERPT_CHARS = 55
_LONG_EXCERPT_CHARS = 80
_DEFAULT_SPECIFICITY_WEIGHTS: dict[str, float] = {
    "currency": 2.0,
    "number": 1.0,
    "timing": 1.0,
    "competitor": 1.0,
    "reviewer_company": 1.0,
    "pain_category": 0.75,
    "replacement_mode": 0.75,
    "operating_model_shift": 0.75,
    "productivity_delta_claim": 0.75,
    "signal_type": 0.5,
    "long_excerpt": 0.5,
    "concrete_pattern": 0.5,
    "generic_phrase_penalty": 2.5,
    "short_excerpt_penalty": 1.5,
}


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


def _normalize_list(values: Any) -> list[str]:
    items: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text:
            items.append(text)
    return items


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _matches_any_regex(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    lowered = text.lower()
    return any(pattern.search(lowered) for pattern in patterns)


def _has_pricing_currency_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    for match in _CURRENCY_RE.finditer(lowered):
        start = max(0, match.start() - 48)
        end = min(len(lowered), match.end() + 48)
        window = lowered[start:end]
        if _contains_any(window, _PRICING_CONTEXT_PATTERNS):
            return True
    return False


_PAIN_KEYWORDS_BY_CATEGORY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pricing", (
        "price", "pricing", "priced", "expensive", "costly", "cost",
        "billing", "invoice", "refund", "overcharge", "overcharged", "subscription",
        "license", "per seat", "per user", "renewal", "quote", "budget",
    )),
    ("support", (
        "support", "customer service", "help desk", "helpdesk",
        "robot", "chatbot", "chat bot", "response time", "ticket",
        "unresponsive", "no response", "talk to a human",
        "talk to a person", "talk to someone",
        "hold time", "waiting on hold",
    )),
    ("features", (
        "feature", "missing", "lack of", "lacking",
        "no way to", "limitation", "limited", "wish it had", "need more",
    )),
    ("reliability", (
        "bug", "crash", "slow", "laggy", "downtime", "outage",
        "broken", "glitch", "error", "unreliable", "unstable",
    )),
    ("onboarding", (
        "onboarding", "learning curve", "hard to learn", "steep curve",
        "setup", "complicated to set up", "documentation",
    )),
    ("privacy", (
        "spam", "unsubscribe", "sales email", "unsolicited",
        "privacy", "data breach", "tracking",
    )),
)
_PAIN_CATEGORY_PRIORITY = {
    "privacy": 4,
    "support": 3,
    "reliability": 2,
    "features": 1,
    "onboarding": 1,
    "pricing": 0,
}
_PAIN_KEYWORD_MAP: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (
        category,
        tuple(re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE) for keyword in keywords),
    )
    for category, keywords in _PAIN_KEYWORDS_BY_CATEGORY
)


def _classify_complaint_pain(phrase: str, default: str | None) -> str | None:
    """Classify a complaint phrase into a pain category by scored keyword match."""
    best_category = default
    best_score = 0
    for category, patterns in _PAIN_KEYWORD_MAP:
        hit_count = sum(1 for pattern in patterns if pattern.search(phrase))
        if hit_count <= 0:
            continue
        score = (hit_count * 10) + int(_PAIN_CATEGORY_PRIORITY.get(category, 0))
        if score > best_score:
            best_category = category
            best_score = score
    if best_score > 0:
        return best_category
    return default


def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _excerpt_text(
    summary: Any,
    review_text: Any,
    phrase: str,
    *,
    exact_phrase: bool = False,
) -> tuple[str, int | None, int | None]:
    full_text = " ".join(
        part.strip() for part in (str(summary or ""), str(review_text or ""))
        if str(part or "").strip()
    )
    if not full_text:
        return phrase, None, None
    haystack = full_text.lower()
    needle = phrase.strip().lower()
    if not needle:
        excerpt = full_text[:180].strip()
        return excerpt, None, None
    idx = haystack.find(needle)
    if idx < 0:
        excerpt = phrase[:220].strip() or full_text[:180].strip()
        return excerpt, None, None
    if exact_phrase:
        excerpt = full_text[idx:idx + len(phrase)].strip()
        return excerpt, idx, idx + len(phrase)
    start = max(0, idx - 40)
    end = min(len(full_text), idx + len(phrase) + 40)
    excerpt = full_text[start:end].strip()
    return excerpt, idx, idx + len(phrase)


def _extract_numeric_literals(text: str) -> dict[str, Any]:
    literals: dict[str, Any] = {}
    amounts = _CURRENCY_RE.findall(text)
    if amounts:
        literals["currency_mentions"] = amounts[:3]
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        literals["numbers"] = numbers[:5]
    return literals


def _review_supports_counterevidence(review: dict[str, Any], enrichment: dict[str, Any]) -> bool:
    churn = _coerce_json_dict(enrichment.get("churn_signals"))
    sentiment = _coerce_json_dict(enrichment.get("sentiment_trajectory"))
    recommendation_language = _normalize_list(enrichment.get("recommendation_language"))
    if enrichment.get("would_recommend") is False:
        return False
    if any(churn.get(flag) for flag in ("intent_to_leave", "actively_evaluating", "migration_in_progress")):
        return False
    direction = str(sentiment.get("direction") or "").strip().lower()
    if direction in _NEGATIVE_SENTIMENT_DIRECTIONS:
        return False
    if any(_contains_any(text, _NEGATIVE_RECOMMENDATION_PATTERNS) for text in recommendation_language):
        return False
    if enrichment.get("would_recommend") is True:
        return True
    return direction in _POSITIVE_SENTIMENT_DIRECTIONS


def _has_derivable_evidence_inputs(result: dict[str, Any]) -> bool:
    scalar_fields = (
        "specific_complaints",
        "pricing_phrases",
        "feature_gaps",
        "recommendation_language",
        "positive_aspects",
        "event_mentions",
    )
    if any(_normalize_list(result.get(field)) for field in scalar_fields[:-1]):
        return True
    if any(isinstance(item, dict) and str(item.get("event") or "").strip() for item in (result.get("event_mentions") or [])):
        return True
    if any(isinstance(item, dict) and str(item.get("name") or "").strip() for item in (result.get("competitors_mentioned") or [])):
        return True
    return False


def resolve_evidence_spans(
    result: dict[str, Any],
    source_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    persisted = result.get("evidence_spans")
    persisted_spans = [item for item in (persisted if isinstance(persisted, list) else []) if isinstance(item, dict)]
    if _has_derivable_evidence_inputs(result):
        derived_spans = derive_evidence_spans(result, source_row)
        if derived_spans:
            if persisted_spans:
                return derived_spans, "refreshed"
            return derived_spans, "derived"
    if persisted_spans:
        return persisted_spans, "persisted"
    return [], "missing"


def derive_replacement_mode(result: dict[str, Any], source_row: dict[str, Any]) -> str:
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    competitors = result.get("competitors_mentioned") or []
    churn = _coerce_json_dict(result.get("churn_signals"))
    if competitors and any(
        str(comp.get("evidence_type") or "").strip().lower() in {"explicit_switch", "active_evaluation"}
        for comp in competitors if isinstance(comp, dict)
    ):
        return "competitor_switch"
    if _contains_any(review_blob, _BUNDLE_PATTERNS):
        return "bundled_suite_consolidation"
    if _contains_any(review_blob, _WORKFLOW_SUBSTITUTION_PATTERNS):
        return "workflow_substitution"
    if _contains_any(review_blob, _INTERNAL_TOOL_PATTERNS):
        return "internal_tool"
    if bool(churn.get("migration_in_progress")) and competitors:
        return "competitor_switch"
    return "none"


def derive_operating_model_shift(result: dict[str, Any], source_row: dict[str, Any]) -> str:
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    if "async" in review_blob and _contains_any(review_blob, ("docs", "documentation", "wiki", "notion", "confluence")):
        return "sync_to_async"
    if _contains_any(review_blob, ("chat", "slack")) and _contains_any(review_blob, ("docs", "notion", "confluence", "wiki")):
        return "chat_to_docs"
    if _contains_any(review_blob, ("chat", "slack")) and _contains_any(review_blob, ("ticket", "ticketing", "help desk", "helpdesk")):
        return "chat_to_ticketing"
    if _contains_any(review_blob, _BUNDLE_PATTERNS):
        return "consolidation"
    return "none"


def derive_productivity_delta_claim(source_row: dict[str, Any]) -> str:
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    if (
        _contains_any(review_blob, _PRODUCTIVITY_POSITIVE_PATTERNS)
        or _matches_any_regex(review_blob, _PRODUCTIVITY_POSITIVE_REGEXES)
    ):
        return "more_productive"
    if (
        _contains_any(review_blob, _PRODUCTIVITY_NEGATIVE_PATTERNS)
        or _matches_any_regex(review_blob, _PRODUCTIVITY_NEGATIVE_REGEXES)
    ):
        return "less_productive"
    if "no change" in review_blob:
        return "no_change"
    return "unknown"


def derive_org_pressure_type(source_row: dict[str, Any]) -> str:
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    ).lower()
    if _contains_any(review_blob, _PROCUREMENT_PATTERNS):
        return "procurement_mandate"
    if _contains_any(review_blob, _STANDARDIZATION_PATTERNS):
        return "standardization_mandate"
    if _contains_any(review_blob, _BUNDLE_PATTERNS):
        return "bundle_pressure"
    if _contains_any(review_blob, _BUDGET_FREEZE_PATTERNS):
        return "budget_freeze"
    return "none"


def derive_salience_flags(result: dict[str, Any], source_row: dict[str, Any]) -> list[str]:
    churn = _coerce_json_dict(result.get("churn_signals"))
    budget = _coerce_json_dict(result.get("budget_signals"))
    reviewer = _coerce_json_dict(result.get("reviewer_context"))
    timeline = _coerce_json_dict(result.get("timeline"))
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    )
    price_text = " ".join(_normalize_list(result.get("pricing_phrases")))
    flags: list[str] = []

    def _add(flag: str, condition: bool) -> None:
        if condition and flag not in flags:
            flags.append(flag)

    _add("explicit_cancel", bool(churn.get("intent_to_leave")))
    _add("active_evaluation", bool(churn.get("actively_evaluating")))
    _add("migration_in_progress", bool(churn.get("migration_in_progress")))
    _add(
        "renewal_window",
        bool(churn.get("contract_renewal_mentioned"))
        or (
            str(timeline.get("decision_timeline") or "").strip().lower()
            not in {"", "unknown", "none"}
        )
        or _contains_any(review_blob, _TIMING_PATTERNS),
    )
    _add(
        "explicit_dollar",
        bool(budget.get("annual_spend_estimate") or budget.get("price_per_seat"))
        or _has_pricing_currency_signal(price_text)
        or _has_pricing_currency_signal(review_blob),
    )
    _add("named_account", bool(source_row.get("reviewer_company") or reviewer.get("company_name")))
    _add("decision_maker", bool(reviewer.get("decision_maker")))
    _add("named_competitor", bool(result.get("competitors_mentioned")))
    _add("productivity_claim", derive_productivity_delta_claim(source_row) != "unknown")
    _add("workflow_substitution", derive_replacement_mode(result, source_row) in {"workflow_substitution", "bundled_suite_consolidation", "internal_tool"})
    return flags


def derive_evidence_spans(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    max_spans: int = 8,
) -> list[dict[str, Any]]:
    """Build stable evidence units from extracted phrase arrays plus raw review text."""
    review_id = str(source_row.get("id") or "unknown")
    reviewer_company = str(source_row.get("reviewer_company") or "").strip() or None
    reviewer_title = str(source_row.get("reviewer_title") or "").strip() or None
    timeline = _coerce_json_dict(result.get("timeline"))
    budget = _coerce_json_dict(result.get("budget_signals"))
    replacement_mode = derive_replacement_mode(result, source_row)
    operating_model_shift = derive_operating_model_shift(result, source_row)
    productivity_delta = derive_productivity_delta_claim(source_row)
    review_blob = " ".join(
        str(source_row.get(field) or "")
        for field in ("summary", "review_text", "pros", "cons")
    )
    contract_context = _coerce_json_dict(result.get("contract_context"))
    company_name = reviewer_company or _coerce_json_dict(result.get("reviewer_context")).get("company_name")
    default_pain = str(result.get("pain_category") or "").strip() or None
    default_time_anchor = (
        str(timeline.get("evaluation_deadline") or "").strip()
        or str(timeline.get("contract_end") or "").strip()
        or (
            str(timeline.get("decision_timeline") or "").strip()
            if str(timeline.get("decision_timeline") or "").strip().lower() not in {"", "unknown", "none"}
            else ""
        )
        or str(_coerce_json_dict(result.get("churn_signals")).get("renewal_timing") or "").strip()
        or None
    )
    competitors = [
        comp for comp in (result.get("competitors_mentioned") or [])
        if isinstance(comp, dict) and str(comp.get("name") or "").strip()
    ]
    primary_competitor = str(competitors[0].get("name") or "").strip() if competitors else None
    item_sources: list[tuple[str, list[Any], str | None, str | None]] = [
        (
            "pricing_backlash",
            (result.get("pricing_phrases") or []) if bool(contract_context.get("price_complaint")) else [],
            "pricing",
            None,
        ),
        ("complaint", result.get("specific_complaints") or [], default_pain, None),
        ("feature_gap", result.get("feature_gaps") or [], "features", None),
        ("recommendation", result.get("recommendation_language") or [], default_pain, None),
        ("positive_anchor", result.get("positive_aspects") or [], None, None),
    ]
    for comp in competitors:
        reasons = [
            comp.get("reason_detail"),
            comp.get("reason"),
        ]
        texts = [text for text in reasons if str(text or "").strip()]
        if not texts and primary_competitor:
            texts = [f"considering {primary_competitor}"]
        item_sources.append((
            "competitor_pressure",
            texts,
            str(comp.get("reason_category") or "").strip() or default_pain,
            str(comp.get("name") or "").strip() or None,
        ))
    for event in result.get("event_mentions") or []:
        if not isinstance(event, dict):
            continue
        item_sources.append((
            "event",
            [event.get("event")],
            default_pain,
            None,
        ))

    spans: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append_span(
        *,
        signal_type: str,
        raw_text: Any,
        pain_category: str | None,
        competitor: str | None,
        time_anchor: str | None,
    ) -> None:
        phrase = str(raw_text or "").strip()
        if not phrase:
            return
        key = _normalize_key(phrase)
        if key in seen:
            return
        excerpt, start_char, end_char = _excerpt_text(
            source_row.get("summary"),
            source_row.get("review_text"),
            phrase,
            exact_phrase=signal_type == "positive_anchor",
        )
        flags: list[str] = []
        if _has_pricing_currency_signal(excerpt):
            flags.append("explicit_dollar")
        if company_name:
            flags.append("named_org")
        if time_anchor or _contains_any(excerpt, _TIMING_PATTERNS):
            flags.append("deadline")
        if primary_competitor and primary_competitor.lower() in excerpt.lower():
            flags.append("named_competitor")
        if replacement_mode != "none":
            flags.append(replacement_mode)
        if operating_model_shift != "none":
            flags.append(operating_model_shift)
        span_id = (
            f"review:{review_id}:span:{start_char}-{end_char}"
            if start_char is not None and end_char is not None
            else f"review:{review_id}:span:{len(spans)}"
        )
        spans.append({
            "span_id": span_id,
            "_sid": span_id,
            "text": excerpt,
            "start_char": start_char,
            "end_char": end_char,
            "signal_type": signal_type,
            "pain_category": pain_category or None,
            "competitor": competitor or None,
            "company_name": company_name,
            "reviewer_title": reviewer_title,
            "time_anchor": time_anchor or default_time_anchor,
            "numeric_literals": _extract_numeric_literals(excerpt),
            "flags": flags,
            "replacement_mode": replacement_mode,
            "operating_model_shift": operating_model_shift,
            "productivity_delta_claim": productivity_delta,
            "_review_id": review_id,
        })
        seen.add(key)

    for signal_type, raw_values, pain_category, competitor in item_sources:
        for raw_value in raw_values:
            time_anchor = None
            if signal_type == "event":
                time_anchor = default_time_anchor
            span_pain = pain_category
            if signal_type in ("complaint", "recommendation"):
                span_pain = _classify_complaint_pain(str(raw_value or ""), pain_category)
            _append_span(
                signal_type=signal_type,
                raw_text=raw_value,
                pain_category=span_pain,
                competitor=competitor,
                time_anchor=time_anchor,
            )
            if len(spans) >= max_spans:
                return spans

    if not spans and review_blob.strip():
        excerpt, start_char, end_char = _excerpt_text("", source_row.get("review_text"), review_blob[:160])
        spans.append({
            "span_id": f"review:{review_id}:span:fallback",
            "_sid": f"review:{review_id}:span:fallback",
            "text": excerpt,
            "start_char": start_char,
            "end_char": end_char,
            "signal_type": "review_context",
            "pain_category": default_pain,
            "competitor": primary_competitor,
            "company_name": company_name,
            "reviewer_title": reviewer_title,
            "time_anchor": default_time_anchor,
            "numeric_literals": _extract_numeric_literals(excerpt),
            "flags": derive_salience_flags(result, source_row),
            "replacement_mode": replacement_mode,
            "operating_model_shift": operating_model_shift,
            "productivity_delta_claim": productivity_delta,
            "_review_id": review_id,
        })

    if budget.get("annual_spend_estimate") and spans:
        spans[0]["numeric_literals"].setdefault(
            "annual_spend_estimate", budget.get("annual_spend_estimate"),
        )
    if budget.get("price_per_seat") and spans:
        spans[0]["numeric_literals"].setdefault(
            "price_per_seat", budget.get("price_per_seat"),
        )
    return spans[:max_spans]


def _recency_bonus(reviewed_at: Any) -> float:
    if not reviewed_at:
        return 0.4
    if isinstance(reviewed_at, str):
        try:
            reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError:
            return 0.4
    if not isinstance(reviewed_at, datetime):
        return 0.4
    if reviewed_at.tzinfo is None:
        reviewed_at = reviewed_at.replace(tzinfo=timezone.utc)
    days_old = max((datetime.now(timezone.utc) - reviewed_at).days, 0)
    if days_old <= 30:
        return 1.0
    if days_old <= 90:
        return 0.7
    if days_old <= 180:
        return 0.5
    return 0.2


def _witness_salience(review: dict[str, Any], enrichment: dict[str, Any], span: dict[str, Any]) -> float:
    churn = _coerce_json_dict(enrichment.get("churn_signals"))
    reviewer = _coerce_json_dict(enrichment.get("reviewer_context"))
    source_weight = _coerce_json_dict(review.get("raw_metadata")).get("source_weight")
    try:
        source_weight = float(source_weight)
    except (TypeError, ValueError):
        source_weight = 0.7
    score = 0.0
    if churn.get("intent_to_leave"):
        score += 3.0
    if churn.get("migration_in_progress"):
        score += 3.0
    if churn.get("actively_evaluating"):
        score += 2.0
    if churn.get("contract_renewal_mentioned"):
        score += 2.0
    if reviewer.get("decision_maker"):
        score += 1.5
    if review.get("reviewer_company"):
        score += 1.5
    if span.get("competitor"):
        score += 1.5
    flags = {str(flag).strip().lower() for flag in span.get("flags") or []}
    if "explicit_dollar" in flags:
        score += 2.0
    if span.get("operating_model_shift") not in {None, "", "none"}:
        score += 1.5
    if span.get("productivity_delta_claim") in {"more_productive", "less_productive"}:
        score += 1.5
    try:
        rating = float(review.get("rating") or 0)
        rating_max = float(review.get("rating_max") or 5)
    except (TypeError, ValueError):
        rating = 0.0
        rating_max = 5.0
    if rating_max > 0 and rating and rating / rating_max <= 0.5:
        score += 1.0
    score += _recency_bonus(review.get("reviewed_at")) * 1.5
    score += max(min(source_weight, 1.5), 0.0)
    return round(score, 2)


def _candidate_types(review: dict[str, Any], enrichment: dict[str, Any], span: dict[str, Any], primary_pains: set[str]) -> set[str]:
    types = {"flex"}
    reviewer = _coerce_json_dict(enrichment.get("reviewer_context"))
    churn = _coerce_json_dict(enrichment.get("churn_signals"))
    flags = {str(flag).strip().lower() for flag in span.get("flags") or []}
    if span.get("pain_category") and str(span.get("pain_category")) in primary_pains:
        types.add("common_pattern")
    if review.get("reviewer_company") or reviewer.get("decision_maker"):
        types.add("named_account")
    if span.get("competitor") or churn.get("migration_in_progress") or churn.get("actively_evaluating"):
        types.add("displacement")
    if span.get("time_anchor") or "deadline" in flags or churn.get("contract_renewal_mentioned"):
        types.add("timing")
    if "explicit_dollar" in flags or len(flags.intersection({"named_org", "named_competitor"})) == 2:
        types.add("outlier")
    if span.get("replacement_mode") not in {None, "", "none"} or span.get("operating_model_shift") not in {None, "", "none"}:
        types.add("category_shift")
    if span.get("signal_type") == "positive_anchor" and _review_supports_counterevidence(review, enrichment):
        types.add("counterevidence")
    return types


def _top_primary_pains(reviews: list[dict[str, Any]]) -> set[str]:
    counts: dict[str, int] = {}
    for review in reviews:
        enrichment = _coerce_json_dict(review.get("enrichment"))
        for item in enrichment.get("pain_categories") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("severity") or "").strip().lower() != "primary":
                continue
            cat = str(item.get("category") or "").strip()
            if not cat:
                continue
            counts[cat] = counts.get(cat, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {cat for cat, _ in ranked[:2]}


def _normalized_excerpt_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _witness_specificity_signals(
    candidate: dict[str, Any],
    *,
    generic_patterns: tuple[str, ...],
    concrete_patterns: tuple[str, ...],
    short_excerpt_chars: int,
    long_excerpt_chars: int,
    weights: dict[str, float],
) -> tuple[float, list[str]]:
    excerpt = str(candidate.get("excerpt_text") or "").strip()
    lowered = excerpt.lower()
    score = 0.0
    reasons: list[str] = []
    anchor_count = 0

    if _CURRENCY_RE.search(excerpt):
        score += float(weights.get("currency", 2.0))
        anchor_count += 1
    if _NUMBER_RE.search(excerpt):
        score += float(weights.get("number", 1.0))
        anchor_count += 1
    if candidate.get("time_anchor") or _contains_any(lowered, _TIMING_PATTERNS):
        score += float(weights.get("timing", 1.0))
        anchor_count += 1
    if candidate.get("competitor"):
        score += float(weights.get("competitor", 1.0))
        anchor_count += 1
    if candidate.get("reviewer_company"):
        score += float(weights.get("reviewer_company", 1.0))
        anchor_count += 1
    if candidate.get("pain_category"):
        score += float(weights.get("pain_category", 0.75))
    if candidate.get("replacement_mode") not in {None, "", "none"}:
        score += float(weights.get("replacement_mode", 0.75))
        anchor_count += 1
    if candidate.get("operating_model_shift") not in {None, "", "none"}:
        score += float(weights.get("operating_model_shift", 0.75))
        anchor_count += 1
    if candidate.get("productivity_delta_claim") in {"more_productive", "less_productive"}:
        score += float(weights.get("productivity_delta_claim", 0.75))
        anchor_count += 1
    if candidate.get("signal_type"):
        score += float(weights.get("signal_type", 0.5))
    if len(excerpt) >= long_excerpt_chars:
        score += float(weights.get("long_excerpt", 0.5))
    if _contains_any(lowered, concrete_patterns):
        score += float(weights.get("concrete_pattern", 0.5))
        anchor_count += 1

    if _contains_any(lowered, generic_patterns):
        score -= float(weights.get("generic_phrase_penalty", 2.5))
        reasons.append("boilerplate_phrase")
    if len(excerpt) < short_excerpt_chars and anchor_count == 0:
        score -= float(weights.get("short_excerpt_penalty", 1.5))
        reasons.append("too_short_without_anchors")
    if anchor_count == 0 and not reasons:
        reasons.append("no_specific_anchor")

    return round(score, 2), reasons


def compute_witness_hash(witness: dict[str, Any]) -> str:
    payload = {
        "witness_id": str(witness.get("witness_id") or witness.get("_sid") or "").strip(),
        "review_id": str(witness.get("review_id") or "").strip(),
        "excerpt_text": _normalized_excerpt_key(witness.get("excerpt_text")),
        "source": str(witness.get("source") or "").strip().lower(),
        "signal_type": str(witness.get("signal_type") or "").strip().lower(),
        "pain_category": str(witness.get("pain_category") or "").strip().lower(),
        "competitor": str(witness.get("competitor") or "").strip().lower(),
        "time_anchor": str(witness.get("time_anchor") or "").strip().lower(),
        "replacement_mode": str(witness.get("replacement_mode") or "").strip().lower(),
        "operating_model_shift": str(witness.get("operating_model_shift") or "").strip().lower(),
        "productivity_delta_claim": str(witness.get("productivity_delta_claim") or "").strip().lower(),
        "signal_tags": sorted(
            str(tag or "").strip().lower()
            for tag in (witness.get("signal_tags") or [])
            if str(tag or "").strip()
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def build_vendor_witness_artifacts(
    vendor_name: str,
    reviews: list[dict[str, Any]] | None,
    *,
    max_witnesses: int = 12,
    min_specificity_score: float = _MIN_GENERIC_ANCHOR_SCORE,
    fallback_min_witnesses: int = 4,
    generic_patterns: tuple[str, ...] | list[str] | None = None,
    concrete_patterns: tuple[str, ...] | list[str] | None = None,
    short_excerpt_chars: int = _SHORT_GENERIC_EXCERPT_CHARS,
    long_excerpt_chars: int = _LONG_EXCERPT_CHARS,
    specificity_weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic witness pack and compact section packets."""
    reviews = list(reviews or [])
    primary_pains = _top_primary_pains(reviews)
    generic_patterns = tuple(generic_patterns or _GENERIC_WITNESS_PATTERNS)
    concrete_patterns = tuple(concrete_patterns or _CONCRETE_PAIN_PATTERNS)
    weights = {
        **_DEFAULT_SPECIFICITY_WEIGHTS,
        **(specificity_weights or {}),
    }
    candidates: list[dict[str, Any]] = []
    _spans_persisted = 0
    _spans_fallback = 0
    _spans_refreshed = 0

    for review in reviews:
        enrichment = _coerce_json_dict(review.get("enrichment"))
        if not enrichment:
            continue
        spans, span_source = resolve_evidence_spans(enrichment, review)
        if span_source == "derived":
            _spans_fallback += 1
        elif span_source in {"persisted", "refreshed"}:
            _spans_persisted += 1
        if span_source == "refreshed":
            _spans_refreshed += 1
        for idx, raw_span in enumerate(spans):
            if not isinstance(raw_span, dict):
                continue
            span = dict(raw_span)
            if span.get("signal_type") == "positive_anchor" and not _review_supports_counterevidence(review, enrichment):
                continue
            excerpt_text = str(span.get("text") or "").strip()
            if not excerpt_text:
                continue
            review_id = str(review.get("id") or span.get("_review_id") or "unknown")
            types = _candidate_types(review, enrichment, span, primary_pains)
            witness_id = f"witness:{review_id}:{idx}"
            candidate = {
                "witness_id": witness_id,
                "_sid": witness_id,
                "vendor_name": vendor_name,
                "review_id": review_id,
                "excerpt_text": excerpt_text[:320],
                "source": str(review.get("source") or "").strip(),
                "reviewed_at": review.get("reviewed_at"),
                "reviewer_company": review.get("reviewer_company"),
                "reviewer_title": review.get("reviewer_title"),
                "pain_category": span.get("pain_category"),
                "competitor": span.get("competitor"),
                "signal_type": span.get("signal_type"),
                "signal_tags": list(span.get("flags") or []),
                "replacement_mode": span.get("replacement_mode"),
                "operating_model_shift": span.get("operating_model_shift"),
                "productivity_delta_claim": span.get("productivity_delta_claim"),
                "time_anchor": span.get("time_anchor"),
                "salience_score": _witness_salience(review, enrichment, span),
                "candidate_types": sorted(types),
                "selection_reason": "",
                "source_span_id": span.get("span_id"),
            }
            specificity_score, generic_reasons = _witness_specificity_signals(
                candidate,
                generic_patterns=generic_patterns,
                concrete_patterns=concrete_patterns,
                short_excerpt_chars=short_excerpt_chars,
                long_excerpt_chars=long_excerpt_chars,
                weights=weights,
            )
            candidate["specificity_score"] = specificity_score
            candidate["generic_reason"] = (
                ",".join(generic_reasons)
                if specificity_score < min_specificity_score and generic_reasons
                else (
                    "low_specificity_score"
                    if specificity_score < min_specificity_score
                    else None
                )
            )
            candidates.append(candidate)

    ranked = sorted(
        candidates,
        key=lambda item: (-float(item.get("salience_score") or 0.0), item["witness_id"]),
    )
    specific_ranked = [
        item for item in ranked
        if not str(item.get("generic_reason") or "").strip()
    ]
    generic_ranked = [
        item for item in ranked
        if str(item.get("generic_reason") or "").strip()
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_excerpts: set[str] = set()

    def _select_for(
        kind: str,
        pool: list[dict[str, Any]],
        *,
        generic_fallback: bool = False,
    ) -> bool:
        for candidate in pool:
            if candidate["witness_id"] in selected_ids:
                continue
            if kind not in set(candidate.get("candidate_types") or []):
                continue
            excerpt_key = _normalized_excerpt_key(candidate.get("excerpt_text"))
            if excerpt_key in seen_excerpts:
                continue
            clone = dict(candidate)
            clone["witness_type"] = kind
            clone["selection_reason"] = (
                f"selected_for_{kind}_generic_fallback"
                if generic_fallback
                else f"selected_for_{kind}"
            )
            clone["witness_hash"] = compute_witness_hash(clone)
            selected.append(clone)
            selected_ids.add(clone["witness_id"])
            seen_excerpts.add(excerpt_key)
            return True
        return False

    for kind, quota in _QUOTA_ORDER:
        for _ in range(quota):
            if len(selected) >= max_witnesses:
                break
            if not _select_for(kind, specific_ranked):
                break

    for candidate in specific_ranked:
        if len(selected) >= max_witnesses:
            break
        if candidate["witness_id"] in selected_ids:
            continue
        excerpt_key = _normalized_excerpt_key(candidate.get("excerpt_text"))
        if excerpt_key in seen_excerpts:
            continue
        clone = dict(candidate)
        clone["witness_type"] = "flex"
        clone["selection_reason"] = "selected_for_flex"
        clone["witness_hash"] = compute_witness_hash(clone)
        selected.append(clone)
        selected_ids.add(clone["witness_id"])
        seen_excerpts.add(excerpt_key)

    fallback_target = min(max_witnesses, max(0, int(fallback_min_witnesses)))
    if len(selected) < fallback_target:
        for kind, quota in _QUOTA_ORDER:
            for _ in range(quota):
                if len(selected) >= fallback_target:
                    break
                if not _select_for(kind, generic_ranked, generic_fallback=True):
                    break
        for candidate in generic_ranked:
            if len(selected) >= fallback_target:
                break
            if candidate["witness_id"] in selected_ids:
                continue
            excerpt_key = _normalized_excerpt_key(candidate.get("excerpt_text"))
            if excerpt_key in seen_excerpts:
                continue
            clone = dict(candidate)
            clone["witness_type"] = "flex"
            clone["selection_reason"] = "selected_for_generic_fallback"
            clone["witness_hash"] = compute_witness_hash(clone)
            selected.append(clone)
            selected_ids.add(clone["witness_id"])
            seen_excerpts.add(excerpt_key)

    selected.sort(
        key=lambda item: (
            str(item.get("witness_type") or ""),
            -float(item.get("salience_score") or 0.0),
            str(item.get("witness_id") or ""),
        ),
    )

    def _ids_for(kind: str) -> list[str]:
        return [item["witness_id"] for item in selected if item.get("witness_type") == kind]

    section_packets = {
        "causal_packet": {
            "witness_ids": [item["witness_id"] for item in selected if item.get("witness_type") != "counterevidence"],
            "primary_pain_categories": sorted(primary_pains),
        },
        "timing_packet": {"witness_ids": _ids_for("timing")},
        "displacement_packet": {"witness_ids": _ids_for("displacement")},
        "segment_packet": {"witness_ids": _ids_for("named_account")},
        "retention_packet": {"witness_ids": _ids_for("counterevidence")},
        "anchor_examples": {
            "common_pattern": _ids_for("common_pattern")[:1],
            "outlier_or_named_account": (_ids_for("outlier") + _ids_for("named_account"))[:1],
            "counterevidence": _ids_for("counterevidence")[:1],
        },
        "_witness_governance": {
            "filtered_generic_candidates": len(generic_ranked),
            "selected_specific_witnesses": sum(
                1 for item in selected if not str(item.get("generic_reason") or "").strip()
            ),
            "selected_generic_fallback_witnesses": sum(
                1 for item in selected if str(item.get("generic_reason") or "").strip()
            ),
            "thin_specific_witness_pool": any(
                str(item.get("generic_reason") or "").strip() for item in selected
            ),
            "spans_persisted": _spans_persisted,
            "spans_fallback": _spans_fallback,
            "spans_refreshed": _spans_refreshed,
        },
    }
    if _spans_fallback > 0:
        logger.info(
            "Witness fallback: %s -- %d/%d reviews missing persisted evidence_spans, recomputed from raw enrichment",
            vendor_name, _spans_fallback, _spans_persisted + _spans_fallback,
        )
    return selected, section_packets
