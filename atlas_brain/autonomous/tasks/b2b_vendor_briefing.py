"""
Vendor Intelligence Briefing -- build, render, and send.

Assembles a deterministic vendor churn briefing from existing DB tables
(no LLM calls) and sends it via Resend.

Data sources (in priority order):
1. b2b_intelligence (weekly_churn_feed) -- richest, pre-aggregated
2. b2b_churn_signals -- fallback aggregated metrics
3. b2b_evidence_vault -- canonical weakness, quote, and company-signal evidence
4. b2b_product_profiles -- enrichment (profile summary, competitive context)
5. b2b_reasoning_synthesis -- reasoning contracts fallback when feed absent (sets reasoning_synthesis flag)
6. b2b_segment_intelligence -- buyer segment breakdown (role, stage, pain)
7. b2b_temporal_intelligence -- renewal windows, budget cycles, keyword spikes
8. b2b_cross_vendor_reasoning_synthesis -- pairwise battle + category council outcomes
9. b2b_vendor_buyer_profiles -- role/stage distribution with urgency signals
10. b2b_reviews -- high-urgency quotes if evidence is still insufficient
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import httpx
import jwt as pyjwt

from ...config import settings
from ...services.campaign_sender import get_campaign_sender
from ...services.vendor_target_selection import dedupe_vendor_target_rows
from ...services.vendor_registry import resolve_vendor_name
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...templates.email.vendor_briefing import render_vendor_briefing_html
from ._b2b_shared import _timing_summary_payload, _reasoning_int
from .campaign_suppression import is_suppressed

logger = logging.getLogger("atlas.b2b.vendor_briefing")

_ACCOUNT_CARD_SYSTEM_PROMPT = (
    "You are a B2B sales intelligence analyst. "
    "Generate concise, data-grounded intelligence cards. "
    "Every claim must be supported by the provided data. "
    "Return only valid JSON."
)


def _apply_synthesis_view_to_briefing(
    briefing: dict[str, Any],
    view: Any,
    *,
    requested_as_of: date | None = None,
) -> bool:
    """Overlay normalized synthesis-view fields onto a briefing payload."""
    if view is None:
        return False

    from ._b2b_synthesis_reader import inject_synthesis_freshness

    context = view.filtered_consumer_context("vendor_briefing")
    contracts = context.get("reasoning_contracts") or {}
    vendor_core = context.get("vendor_core_reasoning") or {}
    displacement = context.get("displacement_reasoning") or {}
    account_reasoning = context.get("account_reasoning") or {}
    used = False

    if isinstance(vendor_core, dict) and vendor_core:
        briefing["vendor_core_reasoning"] = vendor_core
        timing_intelligence = vendor_core.get("timing_intelligence")
        if isinstance(timing_intelligence, dict) and timing_intelligence:
            briefing["timing_intelligence"] = timing_intelligence
            timing_summary, timing_metrics, priority_triggers = _timing_summary_payload(
                timing_intelligence,
            )
            if timing_summary:
                briefing["timing_summary"] = timing_summary
            if timing_metrics:
                briefing["timing_metrics"] = timing_metrics
            if priority_triggers:
                briefing["priority_timing_triggers"] = priority_triggers
        wts = view.why_they_stay
        if wts:
            briefing["why_they_stay"] = wts
        cp = view.confidence_posture
        if cp:
            briefing["confidence_posture"] = cp
            limits = cp.get("limits")
            if limits:
                briefing["confidence_limits"] = limits
        used = True

    if isinstance(displacement, dict) and displacement:
        briefing["displacement_reasoning"] = displacement
        switch_triggers = view.switch_triggers
        if switch_triggers:
            briefing["switch_triggers"] = switch_triggers
        used = True

    if isinstance(account_reasoning, dict) and account_reasoning:
        briefing["account_reasoning"] = account_reasoning
        summary_payload = _account_reasoning_summary_payload(account_reasoning)
        for key, value in summary_payload.items():
            if value not in ("", [], {}, None):
                briefing[key] = value
        merged_accounts = _merge_named_accounts_with_account_reasoning(
            briefing.get("named_accounts") or [],
            account_reasoning,
        )
        if merged_accounts:
            briefing["named_accounts"] = merged_accounts
        used = True

    if contracts:
        briefing["reasoning_contracts"] = contracts
        briefing["reasoning_source"] = "b2b_reasoning_synthesis"
        used = True

    evidence_window = view.meta
    if evidence_window:
        briefing["evidence_window"] = evidence_window
        ew_start = evidence_window.get("evidence_window_start")
        ew_end = evidence_window.get("evidence_window_end")
        if ew_start and ew_end:
            try:
                briefing["evidence_window_days"] = (
                    date.fromisoformat(str(ew_end)[:10])
                    - date.fromisoformat(str(ew_start)[:10])
                ).days
            except (TypeError, ValueError):
                pass
        used = True

    if view.primary_wedge:
        briefing["synthesis_wedge"] = view.primary_wedge.value
        briefing["synthesis_wedge_label"] = view.wedge_label
        used = True
    if getattr(view, "schema_version", ""):
        briefing["synthesis_schema_version"] = view.schema_version
        used = True

    coverage_gaps = view.coverage_gaps
    if coverage_gaps:
        briefing["coverage_gaps"] = coverage_gaps
        used = True

    anchors = context.get("anchor_examples")
    if isinstance(anchors, dict) and anchors:
        briefing["reasoning_anchor_examples"] = anchors
        used = True
    highlights = context.get("witness_highlights")
    if isinstance(highlights, list) and highlights:
        briefing["reasoning_witness_highlights"] = highlights
        used = True
    reference_ids = context.get("reference_ids")
    if isinstance(reference_ids, dict) and reference_ids:
        briefing["reasoning_reference_ids"] = reference_ids
        used = True

    contract_gaps = context.get("reasoning_contract_gaps") or []
    if contract_gaps:
        briefing["reasoning_contract_gaps"] = contract_gaps
        used = True
    section_disclaimers = context.get("reasoning_section_disclaimers")
    if isinstance(section_disclaimers, dict) and section_disclaimers:
        briefing["reasoning_section_disclaimers"] = section_disclaimers
        used = True

    inject_synthesis_freshness(
        briefing,
        view,
        requested_as_of=requested_as_of,
    )
    return used


def _apply_reasoning_synthesis_to_briefing(
    briefing: dict[str, Any],
    feed_entry: dict[str, Any] | None,
) -> bool:
    """Overlay contract-backed reasoning fields from weekly feed entries."""
    if not isinstance(feed_entry, dict):
        return False

    from ._b2b_synthesis_reader import load_synthesis_view

    vendor_name = str(feed_entry.get("vendor") or briefing.get("vendor") or "").strip()
    view = load_synthesis_view(
        feed_entry,
        vendor_name,
        schema_version=str(
            feed_entry.get("synthesis_schema_version")
            or feed_entry.get("schema_version")
            or ""
        ),
        as_of_date=feed_entry.get("data_as_of_date"),
    )
    used = _apply_synthesis_view_to_briefing(briefing, view)

    for field in (
        "synthesis_wedge",
        "synthesis_wedge_label",
        "synthesis_schema_version",
        "evidence_window",
        "evidence_window_days",
        "data_as_of_date",
        "data_stale",
        "account_pressure_summary",
        "account_pressure_metrics",
        "priority_account_names",
        "timing_summary",
        "timing_metrics",
        "priority_timing_triggers",
        "reasoning_anchor_examples",
        "reasoning_witness_highlights",
        "reasoning_reference_ids",
        "reasoning_contract_gaps",
        "reasoning_section_disclaimers",
        "reasoning_source",
        "category_council",
    ):
        value = feed_entry.get(field)
        if value is not None:
            briefing[field] = value
            used = True

    return used


def _account_reasoning_named_accounts(
    account_reasoning: dict[str, Any] | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Convert account reasoning top_accounts into briefing account rows."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(account_reasoning, dict):
        return rows
    for item in account_reasoning.get("top_accounts") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("company") or "").strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        try:
            urgency = round(max(0.0, min(float(item.get("intent_score") or 0), 1.0)) * 10, 1)
        except (TypeError, ValueError):
            urgency = 0.0
        rows.append({
            "company": name,
            "urgency": urgency,
            "source": "account_reasoning",
            "confidence_score": item.get("intent_score"),
            "reasoning_backed": True,
            "source_id": item.get("source_id"),
        })
        if len(rows) >= limit:
            break
    return rows


def _merge_named_accounts_with_account_reasoning(
    named_accounts: list[dict[str, Any]] | Any,
    account_reasoning: dict[str, Any] | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Preserve richer named-account rows and supplement from account reasoning."""
    merged = [item for item in (named_accounts or []) if isinstance(item, dict)]
    seen = {
        str(item.get("company") or "").strip().casefold()
        for item in merged
        if str(item.get("company") or "").strip()
    }
    for row in _account_reasoning_named_accounts(account_reasoning, limit=limit):
        key = str(row.get("company") or "").strip().casefold()
        if key and key not in seen:
            merged.append(row)
            seen.add(key)
        if len(merged) >= limit:
            break
    return merged[:limit]


def _account_reasoning_summary_payload(
    account_reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build visible account-pressure fields from account reasoning."""
    if not isinstance(account_reasoning, dict):
        return {}
    metrics: dict[str, int] = {}
    for key in ("total_accounts", "high_intent_count", "active_eval_count"):
        value = _reasoning_int(account_reasoning.get(key))
        if value is not None:
            metrics[key] = value
    summary = str(account_reasoning.get("market_summary") or "").strip()
    if not summary:
        active_eval = metrics.get("active_eval_count")
        high_intent = metrics.get("high_intent_count")
        total_accounts = metrics.get("total_accounts")
        if active_eval is not None and high_intent is not None:
            summary = (
                f"{active_eval} accounts are in active evaluation while "
                f"{high_intent} accounts show high-intent churn signals."
            )
        elif high_intent is not None:
            summary = f"{high_intent} accounts show high-intent churn signals."
        elif total_accounts is not None:
            summary = f"{total_accounts} accounts are currently in scope."
    priority_names = [
        row["company"] for row in _account_reasoning_named_accounts(account_reasoning)
    ]
    payload: dict[str, Any] = {}
    if summary:
        payload["account_pressure_summary"] = summary
    if metrics:
        payload["account_pressure_metrics"] = metrics
    if priority_names:
        payload["priority_account_names"] = priority_names
    return payload


# ---------------------------------------------------------------------------
# Gate token helpers
# ---------------------------------------------------------------------------

def create_gate_token(vendor_name: str) -> str:
    """Create a signed JWT for the briefing email gate."""
    jwt_cfg = settings.saas_auth
    now = datetime.now(timezone.utc)
    expiry_days = settings.b2b_churn.vendor_briefing_gate_expiry_days
    payload = {
        "vendor_name": vendor_name,
        "type": "briefing_gate",
        "iat": now,
        "exp": now + timedelta(days=expiry_days),
    }
    return pyjwt.encode(payload, jwt_cfg.jwt_secret, algorithm=jwt_cfg.jwt_algorithm)


def build_gate_url(vendor_name: str) -> str:
    """Build the full gate URL for a vendor briefing."""
    base = settings.b2b_churn.vendor_briefing_gate_base_url.rstrip("/")
    token = create_gate_token(vendor_name)
    return f"{base}?vendor={quote(vendor_name)}&ref={token}"


async def _is_first_briefing(pool: Any, vendor_name: str) -> bool:
    """Return True if no successful briefing has ever been sent for this vendor."""
    count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM b2b_vendor_briefings
        WHERE LOWER(vendor_name) = LOWER($1)
          AND status NOT IN ('failed', 'suppressed', 'rejected')
        """,
        vendor_name,
    )
    return (count or 0) == 0


# ---------------------------------------------------------------------------
# Analyst enrichment (Kimi K2.5 via OpenRouter)
# ---------------------------------------------------------------------------

_ANALYST_SYSTEM_PROMPT = """\
You are a B2B churn intelligence analyst writing for VP-level readers. Given \
raw churn data about a software vendor, produce a JSON object with:

1. executive_summary: 2-3 sentences for a VP of Customer Success.
2. pain_labels: Object mapping raw category codes to professional labels \
(e.g. "ux" -> "User Experience Complexity").
3. headline: Under 10 words. Bloomberg-style, not clickbait.
4. cta_hook: One sentence tying the CTA to the specific risk found \
(e.g. "Review the pricing-driven churn cluster behind this alert").
5. displacement_qualifier: If quotes mention competitors not in the \
displacement table, return a short qualifier (e.g. "Other alternatives \
appear in qualitative evidence but at lower measured frequency"). \
Return empty string if no mismatch.
6. account_persona_context: If named_account_personas are provided, note \
the seniority distribution and industries represented. This informs \
urgency calibration -- VP-level signals from enterprise accounts carry \
more weight than end-user signals from SMBs.

CRITICAL RULES:

TONE MUST MATCH THE METRICS. The input includes a `tone_band` field:
- "watchlist": Score 0-29, urgency <4. Use language like "meaningful \
retention risk", "monitor closely", "targeted intervention recommended". \
Do NOT say "immediate", "urgent", "systemic", or "acute".
- "active_risk": Score 30-59 OR urgency 4-7. Use "material risk", \
"active intervention warranted", "accelerating concern".
- "critical": Score 60+ OR urgency 8+. Use "immediate action required", \
"acute churn pressure", "executive escalation needed".

If the tone_band says "watchlist" but the data has high-urgency quotes, \
acknowledge the tension: "While aggregate urgency remains moderate, \
individual high-risk accounts warrant targeted attention."

STRENGTH CALIBRATION:
- If `retention_strengths` is present, acknowledge what the vendor does well \
before stating the risk in the executive_summary. Example: "Despite strong \
customer satisfaction in [area], [vendor] faces material churn pressure \
driven by [weakness]." This prevents overstatement and builds credibility \
with prospects who value parts of the incumbent. Do not fabricate strengths \
not in the list.

ATTRIBUTION RULES:
- NEVER make direct attribution claims about named accounts. Do NOT write \
"Meridian Technologies citing $180K as unsustainable."
- Instead write: "High-risk accounts including Meridian Technologies show \
pricing-related churn signals, including references to $180K+ annual \
contract fatigue."
- Named accounts show signals. They did not make statements to us.
- Quotes are market intelligence observations, not verified direct \
attribution from named accounts.

Return ONLY valid JSON. No markdown fences, no explanation."""

_PAIN_LABEL_FALLBACKS = {
    "pricing": "Pricing and Contract Value Fatigue",
    "support": "Support Experience Issues",
    "reliability": "Reliability Concerns",
    "usability": "User Experience Complexity",
    "ux": "User Experience Complexity",
    "features": "Feature Gap Concerns",
    "performance": "Performance Limitations",
    "integration": "Integration Friction",
    "security": "Security and Compliance Concerns",
    "onboarding": "Onboarding Friction",
    "migration": "Migration Complexity",
    "other": "Overall Dissatisfaction",
    "general_dissatisfaction": "Overall Dissatisfaction",
    "overall_dissatisfaction": "Overall Dissatisfaction",
}


def _tone_band(score: float, urgency: float) -> str:
    """Determine tone band from churn pressure score and urgency."""
    if score >= 60 or urgency >= 8:
        return "critical"
    if score >= 30 or urgency >= 4:
        return "active_risk"
    return "watchlist"


def _default_pain_label(category: Any) -> str:
    """Return a readable pain label without relying on LLM enrichment."""
    raw = str(category or "overall_dissatisfaction").strip().lower()
    return _PAIN_LABEL_FALLBACKS.get(raw, raw.replace("_", " ").title())


def _build_default_cta_hook(briefing: dict[str, Any]) -> str:
    """Build a specific CTA hook from the strongest measured signal."""
    challenger_mode = briefing.get("challenger_mode", False)
    vendor = briefing.get("vendor_name", "the vendor")
    pains = briefing.get("pain_breakdown") or []
    top_pain = pains[0].get("category") if pains and isinstance(pains[0], dict) else ""
    pain_label = _default_pain_label(top_pain).lower()

    targets = briefing.get("top_displacement_targets") or []
    top_target = ""
    if targets and isinstance(targets[0], dict):
        top_target = str(targets[0].get("competitor") or "").strip()

    if challenger_mode:
        if top_target and top_pain:
            return (
                f"See which accounts are leaving {top_target} due to "
                f"{pain_label} and how to position your outreach."
            )
        if top_pain:
            return f"Review the {pain_label} signals driving accounts to evaluate alternatives like {vendor}."
        if top_target:
            return f"See the accounts in motion from {top_target} that your team can engage now."
        return f"Review this week's account movement signals relevant to {vendor}."

    if top_target and top_pain:
        return (
            f"Review the {pain_label} signals behind the shift toward "
            f"{top_target} before the next renewal cycle."
        )
    if top_pain:
        return f"Review the {pain_label} cluster behind this alert to prioritize retention plays."
    if top_target:
        return f"Review the accounts trending toward {top_target} to focus retention outreach early."
    return f"Review this week's measured churn signals for {vendor}."


def _finalize_briefing_presentation(briefing: dict[str, Any]) -> None:
    """Fill presentation fields deterministically after enrichment."""
    pain_labels = briefing.get("pain_labels") or {}
    for pain in briefing.get("pain_breakdown") or []:
        if not isinstance(pain, dict):
            continue
        raw_category = str(pain.get("category") or "").strip()
        if raw_category and raw_category not in pain_labels:
            pain_labels[raw_category] = _default_pain_label(raw_category)
    briefing["pain_labels"] = pain_labels

    if not briefing.get("executive_summary"):
        account_pressure_summary = str(
            briefing.get("account_pressure_summary") or ""
        ).strip()
        if account_pressure_summary:
            briefing["executive_summary"] = account_pressure_summary
        else:
            timing_summary = str(briefing.get("timing_summary") or "").strip()
            if timing_summary:
                briefing["executive_summary"] = timing_summary

    if not briefing.get("cta_hook"):
        briefing["cta_hook"] = _build_default_cta_hook(briefing)

    if briefing.get("cta_hook") and not briefing.get("cta_description"):
        briefing["cta_description"] = ""


def _briefing_pain_breakdown_from_evidence_vault(
    vault: dict[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Map canonical vault pain rows to vendor-briefing pain breakdown."""
    rows: list[dict[str, Any]] = []
    for item in (vault.get("weakness_evidence") or []):
        if str(item.get("evidence_type") or "") != "pain_category":
            continue
        key = str(item.get("key") or "").strip().lower()
        if not key:
            continue
        metrics = item.get("supporting_metrics") or {}
        rows.append({
            "category": key,
            "count": int(item.get("mention_count_total") or 0),
            "avg_urgency": metrics.get("avg_urgency_when_mentioned") or metrics.get("avg_urgency"),
        })
    rows.sort(key=lambda r: int(r.get("count") or 0), reverse=True)
    return rows[:limit]


def _briefing_feature_gaps_from_evidence_vault(
    vault: dict[str, Any],
    *,
    limit: int = 5,
) -> list[str]:
    """Map canonical vault feature-gap rows to briefing strings."""
    gaps: list[tuple[int, str]] = []
    for item in (vault.get("weakness_evidence") or []):
        if str(item.get("evidence_type") or "") != "feature_gap":
            continue
        label = str(item.get("label") or item.get("key") or "").strip()
        if not label:
            continue
        gaps.append((int(item.get("mention_count_total") or 0), label))
    gaps.sort(key=lambda row: row[0], reverse=True)
    return [label for _, label in gaps[:limit]]


def _briefing_quotes_from_evidence_vault(
    vault: dict[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Build quotable briefing evidence rows from canonical weakness evidence."""
    quotes: list[dict[str, Any]] = []
    for item in (vault.get("weakness_evidence") or []):
        text = str(item.get("best_quote") or "").strip()
        if not text:
            continue
        source = item.get("quote_source") or {}
        metrics = item.get("supporting_metrics") or {}
        quotes.append({
            "quote": text,
            "text": text,
            "company": source.get("company"),
            "title": source.get("reviewer_title") or source.get("title"),
            "company_size": source.get("company_size"),
            "industry": source.get("industry"),
            "source": source.get("source"),
            "review_id": source.get("review_id"),
            "rating": source.get("rating"),
            "reviewed_at": source.get("reviewed_at"),
            "urgency": metrics.get("avg_urgency_when_mentioned") or metrics.get("avg_urgency"),
            "pain_category": item.get("key"),
            "mention_count": int(item.get("mention_count_total") or 0),
        })
    quotes.sort(
        key=lambda q: (
            float(q.get("urgency") or 0),
            int(q.get("mention_count") or 0),
        ),
        reverse=True,
    )
    return quotes[:limit]


def _briefing_named_accounts_from_evidence_vault(
    vault: dict[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Map canonical company signals into briefing named-account rows."""
    accounts: list[dict[str, Any]] = []
    for item in (vault.get("company_signals") or []):
        company = str(item.get("company_name") or "").strip()
        if not company:
            continue
        pain = str(item.get("pain_category") or "").strip().lower()
        acct: dict[str, Any] = {
            "company": company,
            "title": item.get("buyer_role"),
            "company_size": item.get("seat_count"),
            "industry": item.get("industry"),
            "urgency": item.get("urgency_score"),
            "pain_breakdown": [{"category": pain, "count": 1}] if pain else [],
            "evidence": [],
            "buying_stage": item.get("buying_stage"),
            "source": item.get("source"),
            "contract_end": item.get("contract_end"),
            "confidence_score": item.get("confidence_score"),
        }
        if item.get("decision_maker") is not None:
            acct["decision_maker"] = item["decision_maker"]
        accounts.append(acct)
    accounts.sort(
        key=lambda a: (
            float(a.get("urgency") or 0),
            float(a.get("confidence_score") or 0),
        ),
        reverse=True,
    )
    return accounts[:limit]


def _briefing_strengths_from_evidence_vault(
    vault: dict[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Map canonical vault strength rows to vendor-briefing retention strengths."""
    rows: list[dict[str, Any]] = []
    for item in (vault.get("strength_evidence") or []):
        if not isinstance(item, dict):
            continue
        area = str(item.get("label") or item.get("key") or "").strip()
        if not area:
            continue
        total = int(item.get("mention_count_total") or 0)
        trend = item.get("trend") if isinstance(item.get("trend"), dict) else {}
        entry: dict[str, Any] = {
            "area": area,
            "mention_count": total,
        }
        direction = str(trend.get("direction") or "").strip()
        if direction:
            entry["trend"] = direction
        quote = str(item.get("best_quote") or "").strip()
        if quote:
            entry["customer_quote"] = quote
        rows.append(entry)
    rows.sort(key=lambda r: int(r.get("mention_count") or 0), reverse=True)
    return rows[:limit]


def _apply_evidence_vault_to_briefing(
    briefing: dict[str, Any],
    vault: dict[str, Any] | None,
) -> bool:
    """Overlay canonical evidence-vault fields without replacing richer sources."""
    if not vault:
        return False

    used = False
    snapshot = vault.get("metric_snapshot") or {}

    if not briefing.get("review_count"):
        review_count = snapshot.get("reviews_in_analysis_window") or snapshot.get("total_reviews")
        if review_count is not None:
            briefing["review_count"] = review_count
            used = True
    if not briefing.get("avg_urgency") and snapshot.get("avg_urgency") is not None:
        briefing["avg_urgency"] = snapshot.get("avg_urgency")
        used = True
    if not briefing.get("churn_signal_density") and snapshot.get("churn_density") is not None:
        briefing["churn_signal_density"] = snapshot.get("churn_density")
        used = True
    if not briefing.get("dm_churn_rate") and snapshot.get("dm_churn_rate") is not None:
        briefing["dm_churn_rate"] = float(snapshot.get("dm_churn_rate") or 0) * 100.0
        used = True
    if not briefing.get("recommend_ratio") and snapshot.get("recommend_ratio") is not None:
        try:
            briefing["recommend_ratio"] = round(float(snapshot["recommend_ratio"]), 2)
            used = True
        except (TypeError, ValueError):
            pass

    if not briefing.get("pain_breakdown"):
        pain_breakdown = _briefing_pain_breakdown_from_evidence_vault(vault)
        if pain_breakdown:
            briefing["pain_breakdown"] = pain_breakdown
            used = True
    if not briefing.get("top_feature_gaps"):
        feature_gaps = _briefing_feature_gaps_from_evidence_vault(vault)
        if feature_gaps:
            briefing["top_feature_gaps"] = feature_gaps
            used = True
    if not briefing.get("named_accounts"):
        named_accounts = _briefing_named_accounts_from_evidence_vault(vault)
        if named_accounts:
            briefing["named_accounts"] = named_accounts
            used = True
    if not briefing.get("retention_strengths"):
        retention_strengths = _briefing_strengths_from_evidence_vault(vault)
        if retention_strengths:
            briefing["retention_strengths"] = retention_strengths
            used = True

    existing_evidence = list(briefing.get("evidence") or [])
    if len(existing_evidence) < 2:
        existing_text = {
            str(e.get("quote") or e.get("text") or "").strip().lower()
            for e in existing_evidence
            if isinstance(e, dict)
        }
        for item in _briefing_quotes_from_evidence_vault(vault):
            text = str(item.get("quote") or item.get("text") or "").strip().lower()
            if not text or text in existing_text:
                continue
            existing_evidence.append(item)
            existing_text.add(text)
            used = True
            if len(existing_evidence) >= 5:
                break
        briefing["evidence"] = existing_evidence

    return used


async def _enrich_with_analyst_summary(briefing: dict[str, Any]) -> None:
    """Call Kimi K2.5 to generate analyst summary, headline, and pain labels.

    Mutates *briefing* in place. Fails silently -- the briefing renders fine
    without enrichment.
    """
    api_key = settings.b2b_churn.openrouter_api_key
    if not api_key:
        return

    # Build a compact payload (only what the LLM needs)
    score = float(briefing.get("churn_pressure_score") or 0)
    urgency = float(briefing.get("avg_urgency") or 0)
    trend = briefing.get("trend")

    payload = {
        "vendor_name": briefing.get("vendor_name"),
        "category": briefing.get("category"),
        "churn_pressure_score": score,
        "churn_signal_density": briefing.get("churn_signal_density"),
        "avg_urgency": urgency,
        "trend": trend or "stable",
        "review_count": briefing.get("review_count"),
        "dm_churn_rate": briefing.get("dm_churn_rate"),
        "recommend_ratio": briefing.get("recommend_ratio"),
        "tone_band": _tone_band(score, urgency),
        "pain_breakdown": briefing.get("pain_breakdown", [])[:5],
        "top_displacement_targets": briefing.get("top_displacement_targets", [])[:5],
        "evidence": [
            (e.get("quote", e) if isinstance(e, dict) else e)
            for e in (briefing.get("evidence") or [])[:3]
        ],
        "named_accounts": briefing.get("named_accounts", [])[:5],
        "account_pressure_summary": briefing.get("account_pressure_summary"),
        "account_pressure_metrics": briefing.get("account_pressure_metrics"),
        "priority_account_names": briefing.get("priority_account_names", [])[:5],
        "top_feature_gaps": briefing.get("top_feature_gaps", [])[:3],
        "retention_strengths": briefing.get("retention_strengths", [])[:3],
        "named_account_personas": [
            {"company": a.get("company"), "title": a.get("title"),
             "company_size": a.get("company_size"), "industry": a.get("industry")}
            for a in (briefing.get("named_accounts") or [])[:5]
            if isinstance(a, dict) and a.get("title")
        ],
    }
    payload_json = json.dumps(payload)
    cache_namespace = "b2b_vendor_briefing.analyst_summary"

    try:
        from ...pipelines.llm import clean_llm_output
        from ...services.b2b.cache_runner import (
            lookup_b2b_exact_stage_text,
            prepare_b2b_exact_stage_request,
            store_b2b_exact_stage_text,
        )

        request = prepare_b2b_exact_stage_request(
            "b2b_vendor_briefing.analyst_summary",
            provider="openrouter",
            model=settings.b2b_churn.briefing_analyst_model,
            messages=[
                {"role": "system", "content": _ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": payload_json},
            ],
            max_tokens=4000,
            temperature=0.3,
            extra={"reasoning": {"effort": "low"}},
        )
        content: str | None = None
        cached = await lookup_b2b_exact_stage_text(request)
        if cached is not None:
            try:
                cleaned_cached = clean_llm_output(cached["response_text"])
                json.loads(cleaned_cached)
                content = cleaned_cached
            except json.JSONDecodeError:
                content = None

        if content is None:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.b2b_churn.briefing_analyst_model,
                        "messages": [
                            {"role": "system", "content": _ANALYST_SYSTEM_PROMPT},
                            {"role": "user", "content": payload_json},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4000,
                        "reasoning": {"effort": "low"},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            content = data["choices"][0]["message"].get("content")
            if not content:
                logger.warning("Analyst enrichment returned empty content")
                return
            content = clean_llm_output(content)

        result = json.loads(content)

        if result.get("executive_summary"):
            briefing["executive_summary"] = result["executive_summary"]
        if result.get("headline"):
            briefing["headline"] = result["headline"]
        if result.get("pain_labels") and isinstance(result["pain_labels"], dict):
            briefing["pain_labels"] = result["pain_labels"]
        if result.get("cta_hook"):
            briefing["cta_hook"] = result["cta_hook"]
        if result.get("displacement_qualifier"):
            briefing["displacement_qualifier"] = result["displacement_qualifier"]
        if result.get("account_persona_context"):
            briefing["account_persona_context"] = result["account_persona_context"]

        await store_b2b_exact_stage_text(
            request,
            response_text=content,
            metadata={
                "vendor_name": briefing.get("vendor_name"),
                "category": briefing.get("category"),
            },
        )

        logger.info(
            "Analyst enrichment applied for %s (tone_band=%s)",
            briefing.get("vendor_name"),
            payload["tone_band"],
        )

    except Exception:
        logger.exception("Analyst enrichment failed (non-fatal)")


# ---------------------------------------------------------------------------
# Account cards -- tiered reasoning (depth 0/1/2)
# ---------------------------------------------------------------------------

_PERSONA_KEYWORDS: dict[str, list[str]] = {
    "retention_leader": ["vp cs", "head of cx", "cs director", "vp customer success",
                         "director of customer success", "chief customer officer"],
    "growth_leader": ["vp sales", "cro", "bd director", "chief revenue officer",
                      "head of sales", "director of sales", "vp business development"],
    "product_leader": ["vp product", "cpo", "chief product officer", "head of product",
                       "director of product"],
    "technical_leader": ["cto", "vp eng", "vp engineering", "chief technology officer",
                         "director of engineering", "head of engineering"],
    "executive": ["ceo", "coo", "chief executive", "chief operating", "founder",
                  "managing director", "general manager"],
}


def _classify_persona(title: str | None) -> str:
    """Classify a reviewer title into a persona bucket via keyword match."""
    if not title:
        return "unknown"
    t = title.strip().lower()
    for persona, keywords in _PERSONA_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return persona
    return "unknown"


def _classify_company_tier(company_size: Any) -> str:
    """Classify company size into a tier label."""
    try:
        size = int(company_size)
    except (TypeError, ValueError):
        return "unknown"
    if size <= 200:
        return "smb"
    if size <= 1000:
        return "mid_market"
    if size <= 10000:
        return "enterprise"
    return "large_enterprise"


def _urgency_label(urgency: float) -> str:
    """Map urgency score to a human label."""
    if urgency >= 8:
        return "critical"
    if urgency >= 6:
        return "high"
    if urgency >= 4:
        return "moderate"
    return "watch"


def _compute_data_richness(account: dict[str, Any], briefing_data: dict[str, Any]) -> int:
    """Score 0-10 counting non-empty optional fields for depth selection."""
    score = 0
    if account.get("title"):
        score += 1
    if account.get("company_size"):
        score += 1
    if account.get("industry"):
        score += 1
    pains = account.get("pain_breakdown") or briefing_data.get("pain_breakdown") or []
    if len(pains) >= 2:
        score += 1
    evidence = account.get("evidence") or briefing_data.get("evidence") or []
    if len(evidence) >= 2:
        score += 1
    displacement = account.get("top_displacement_targets") or briefing_data.get("top_displacement_targets") or []
    if len(displacement) >= 1:
        score += 1
    gaps = account.get("top_feature_gaps") or briefing_data.get("top_feature_gaps") or []
    if len(gaps) >= 1:
        score += 1
    if account.get("budget_context") or briefing_data.get("budget_context"):
        score += 1
    trend = account.get("trend") or briefing_data.get("trend")
    if trend and trend != "stable":
        score += 1
    churn_score = float(account.get("churn_pressure_score") or briefing_data.get("churn_pressure_score") or 0)
    if churn_score > 0:
        score += 1
    return score


def _select_reasoning_depth(
    account: dict[str, Any],
    briefing_data: dict[str, Any],
    max_depth: int,
) -> int:
    """Adaptively select reasoning depth based on urgency and data richness."""
    urgency = float(account.get("urgency", 0))
    richness = _compute_data_richness(account, briefing_data)

    if urgency >= 8 and richness >= 5 and max_depth >= 2:
        return 2
    if (urgency >= 6 or richness >= 6) and max_depth >= 1:
        return 1
    if richness < 3:
        return 0
    return min(1, max_depth)


def _build_enriched_baseline(
    account: dict[str, Any],
    briefing_data: dict[str, Any],
    vendor_name: str,
    target_mode: str,
) -> dict[str, Any]:
    """Build depth-0 baseline with all raw + computed fields (no LLM)."""
    company = account.get("company", "Unknown") if isinstance(account, dict) else str(account)
    urgency = float(account.get("urgency", 0)) if isinstance(account, dict) else 0.0

    # Raw fields from account, falling back to briefing-level data
    def _get(field: str) -> Any:
        val = account.get(field) if isinstance(account, dict) else None
        if val is None:
            val = briefing_data.get(field)
        return val

    title = _get("title")
    company_size = _get("company_size")
    industry = _get("industry")
    pain_breakdown = _get("pain_breakdown") or []
    evidence = _get("evidence") or []
    displacement = _get("top_displacement_targets") or []
    feature_gaps = _get("top_feature_gaps") or []
    budget_context = _get("budget_context")
    trend = _get("trend")

    # Computed fields
    persona = _classify_persona(title)
    company_tier = _classify_company_tier(company_size)
    urg_label = _urgency_label(urgency)
    data_richness = _compute_data_richness(account, briefing_data)

    top_pain = ""
    if pain_breakdown and isinstance(pain_breakdown[0], dict):
        top_pain = pain_breakdown[0].get("category", "")
    elif pain_breakdown:
        top_pain = str(pain_breakdown[0])

    top_competitor = ""
    if displacement and isinstance(displacement[0], dict):
        top_competitor = displacement[0].get("competitor", "")
    elif displacement:
        top_competitor = str(displacement[0])

    baseline: dict[str, Any] = {
        "company": company,
        "urgency": urgency,
        "vendor_name": vendor_name,
        "target_mode": target_mode,
        # Raw fields
        "title": title,
        "company_size": company_size,
        "industry": industry,
        "pain_breakdown": pain_breakdown,
        "evidence": evidence,
        "top_displacement_targets": displacement,
        "top_feature_gaps": feature_gaps,
        "budget_context": budget_context,
        "trend": trend,
        # Computed fields
        "persona": persona,
        "company_tier": company_tier,
        "urgency_label": urg_label,
        "data_richness": data_richness,
        "top_pain": top_pain,
        "top_competitor": top_competitor,
    }
    return baseline


def _fill_template(prompt_text: str, data: dict[str, Any]) -> str:
    """Substitute {field} placeholders in a template string."""
    for key, val in data.items():
        placeholder = "{" + key + "}"
        if placeholder in prompt_text:
            if isinstance(val, (list, dict)):
                replacement = json.dumps(val, default=str)
            elif val is None:
                replacement = "N/A"
            else:
                replacement = str(val)
            prompt_text = prompt_text.replace(placeholder, replacement)
    # Strip unreplaced placeholders
    prompt_text = re.sub(r"\{[a-z_]+\}", "N/A", prompt_text)
    return prompt_text


def _mode_instruction(target_mode: str) -> str:
    """Return mode-specific instruction for STEP 4 of CoT prompt."""
    if target_mode == "challenger_intel":
        return (
            "Frame around capturing accounts in motion for a VP Sales / CRO. "
            "Position your solution against the incumbent's weaknesses."
        )
    return (
        "Frame around preventing further customer loss for a VP CS / Head of Product. "
        "Focus on retention levers and risk mitigation."
    )


def _get_llm() -> Any:
    """Get LLM instance for account card enrichment."""
    from ...services.llm_router import get_llm
    llm = get_llm("campaign")
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    return llm


async def _llm_call(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    *,
    cache_namespace: str = "b2b_vendor_briefing.account_card",
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Single LLM call returning parsed JSON or None."""
    from ...services.protocols import Message
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_stage_request,
        store_b2b_exact_stage_text,
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    request = prepare_b2b_exact_stage_request(
        cache_namespace,
        llm=llm,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    model_name = request.model

    try:
        cached = await lookup_b2b_exact_stage_text(request)
        text: str | None = None
        usage: dict[str, int] = {}
        if cached is not None:
            text = cached["response_text"]

        if hasattr(llm, "chat_async"):
            if text is None:
                text = (await llm.chat_async(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )).strip()
        elif text is None:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    llm.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                ),
                timeout=60,
            )
            text = result.get("response", "").strip() if isinstance(result, dict) else str(result).strip()
            usage = result.get("usage", {}) if isinstance(result, dict) else {}

        if not text:
            return None

        # Clean markdown fences
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if cached is None:
                raise
            logger.warning("Cached account card response was invalid JSON; retrying live")
            text = None
            usage = {}
            if hasattr(llm, "chat_async"):
                text = (await llm.chat_async(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )).strip()
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        llm.chat,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.3,
                    ),
                    timeout=60,
                )
                text = result.get("response", "").strip() if isinstance(result, dict) else str(result).strip()
                usage = result.get("usage", {}) if isinstance(result, dict) else {}
            if not text:
                return None
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()
            data = json.loads(text)
        await store_b2b_exact_stage_text(
            request,
            response_text=text,
            usage=usage,
            metadata=cache_metadata,
        )
        return {
            "data": data,
            "model": model_name,
            "token_usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        }
    except json.JSONDecodeError:
        logger.warning(
            "Account card LLM returned invalid JSON (text=%s)",
            text[:200] if text else "<empty>",
        )
        return None
    except Exception:
        logger.exception("LLM call failed")
        return None


def _merge_token_usage(usages: list[dict[str, int]]) -> dict[str, int]:
    """Sum token usage dicts."""
    return {
        "input_tokens": sum(u.get("input_tokens", 0) for u in usages),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usages),
    }


async def _enrich_account_card_cot(
    template: dict[str, Any],
    baseline: dict[str, Any],
    target_mode: str,
) -> dict[str, Any] | None:
    """Depth 1: Chain-of-thought single-pass enrichment (1 LLM call)."""
    llm = _get_llm()
    if llm is None:
        logger.warning("No LLM available for depth-1 account card enrichment")
        return None

    # Inject mode-specific instruction into the baseline for template filling
    enriched_baseline = {**baseline, "target_mode": _mode_instruction(target_mode)}
    prompt_text = _fill_template(template.get("prompt_template") or "", enriched_baseline)
    system_prompt = template.get("system_prompt") or _ACCOUNT_CARD_SYSTEM_PROMPT

    result = await _llm_call(llm, system_prompt, prompt_text)
    if not result:
        logger.warning("Depth-1 enrichment returned nothing for %s", baseline.get("company"))
        return None

    return result


async def _deep_enrich_account_card(
    templates: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    target_mode: str,
    briefing_data: dict[str, Any],
) -> dict[str, Any] | None:
    """Depth 2: Multi-pass recursive reasoning (2-4 LLM calls)."""
    llm = _get_llm()
    if llm is None:
        logger.warning("No LLM available for depth-2 account card enrichment")
        return None

    company = baseline.get("company", "Unknown")
    urgency = float(baseline.get("urgency", 0))
    enriched_baseline = {**baseline, "target_mode": _mode_instruction(target_mode)}
    all_usages: list[dict[str, int]] = []
    calls_made = 0
    conditional_calls: list[str] = []

    # --- Call 1: Decompose and Analyze (always runs) ---
    decompose_tpl = templates.get("sales_action_decompose")
    if not decompose_tpl:
        logger.warning("Missing sales_action_decompose template, falling back to depth 1")
        cot_tpl = templates.get("sales_action_cot")
        if cot_tpl:
            return await _enrich_account_card_cot(cot_tpl, baseline, target_mode)
        return None

    decompose_prompt = _fill_template(
        decompose_tpl.get("prompt_template") or "", enriched_baseline
    )
    decompose_system = decompose_tpl.get("system_prompt") or _ACCOUNT_CARD_SYSTEM_PROMPT

    decompose_result = await _llm_call(llm, decompose_system, decompose_prompt, max_tokens=2048)
    if not decompose_result:
        logger.warning("Depth-2 decompose failed for %s, falling back to depth 1", company)
        cot_tpl = templates.get("sales_action_cot")
        if cot_tpl:
            return await _enrich_account_card_cot(cot_tpl, baseline, target_mode)
        return None

    calls_made += 1
    all_usages.append(decompose_result["token_usage"])
    decomposition = decompose_result["data"]

    # --- Call 2: Synthesize + Self-Correct (always runs) ---
    synthesize_tpl = templates.get("sales_action_synthesize")
    if not synthesize_tpl:
        # Return raw decomposition if no synthesis template
        decompose_result["token_usage"] = _merge_token_usage(all_usages)
        decompose_result["data"]["reasoning_meta"] = {
            "calls_made": calls_made,
            "conditional_calls": conditional_calls,
        }
        return decompose_result

    synth_data = {
        **enriched_baseline,
        "decomposition": json.dumps(decomposition, default=str),
        "baseline_json": json.dumps(baseline, default=str),
    }
    synth_prompt = _fill_template(
        synthesize_tpl.get("prompt_template") or "", synth_data
    )
    synth_system = synthesize_tpl.get("system_prompt") or _ACCOUNT_CARD_SYSTEM_PROMPT

    synth_result = await _llm_call(llm, synth_system, synth_prompt, max_tokens=2048)
    if synth_result:
        calls_made += 1
        all_usages.append(synth_result["token_usage"])
        final_data = synth_result["data"]
    else:
        logger.warning("Depth-2 synthesis failed for %s, using decomposition", company)
        final_data = decomposition

    # --- Calls 3+4: Conditional, run in parallel ---
    displacement = baseline.get("top_displacement_targets") or []
    parallel_tasks: list[tuple[str, Any]] = []

    # Call 3: Competitive Deep-Dive (>= 2 displacement targets AND urgency >= 7)
    if len(displacement) >= 2 and urgency >= 7:
        comp_prompt = (
            f"Competitive deep-dive for {company} (urgency {urgency}/10).\n\n"
            f"Displacement targets: {json.dumps(displacement[:5], default=str)}\n"
            f"Pain drivers: {json.dumps(baseline.get('pain_breakdown', [])[:5], default=str)}\n"
            f"Persona: {baseline.get('persona', 'unknown')}\n\n"
            "Return JSON with:\n"
            '- "primary_threat": The competitor most likely to win this account and why.\n'
            '- "differentiators_to_emphasize": Array of 2-3 key differentiators to lead with.\n'
            '- "landmine_questions": Array of 2-3 questions that expose competitor weaknesses.\n\n'
            "Return ONLY valid JSON."
        )
        parallel_tasks.append(("competitive_deep_dive", _llm_call(
            llm, _ACCOUNT_CARD_SYSTEM_PROMPT, comp_prompt
        )))

    # Call 4: Objection Pre-handling (urgency >= 8, critical accounts)
    if urgency >= 8:
        obj_prompt = (
            f"Objection pre-handling for {company} (urgency {urgency}/10, CRITICAL).\n\n"
            f"Pain drivers: {json.dumps(baseline.get('pain_breakdown', [])[:5], default=str)}\n"
            f"Evidence: {json.dumps(baseline.get('evidence', [])[:3], default=str)}\n"
            f"Persona: {baseline.get('persona', 'unknown')}\n"
            f"Industry: {baseline.get('industry', 'N/A')}\n\n"
            "Return JSON with:\n"
            '- "objections": Array of 2-3 objects with "objection", "response", "data_reference".\n\n'
            "Return ONLY valid JSON."
        )
        parallel_tasks.append(("objection_prehandling", _llm_call(
            llm, _ACCOUNT_CARD_SYSTEM_PROMPT, obj_prompt
        )))

    if parallel_tasks:
        results = await asyncio.gather(
            *(task[1] for task in parallel_tasks),
            return_exceptions=True,
        )
        for (call_name, _), result in zip(parallel_tasks, results):
            if isinstance(result, Exception):
                logger.warning("Depth-2 %s failed for %s: %s", call_name, company, result)
                continue
            if result is None:
                continue
            calls_made += 1
            all_usages.append(result["token_usage"])
            conditional_calls.append(call_name)
            # Merge conditional call data into final output
            final_data[call_name] = result["data"]

    final_data["reasoning_meta"] = {
        "calls_made": calls_made,
        "conditional_calls": conditional_calls,
    }

    return {
        "data": final_data,
        "model": decompose_result.get("model", "unknown"),
        "token_usage": _merge_token_usage(all_usages),
    }


async def generate_account_cards(
    briefing_data: dict[str, Any],
    reasoning_depth: int | None = None,
    target_mode: str = "vendor_retention",
) -> list[dict[str, Any]]:
    """Generate intelligence cards for top named accounts.

    Tiered reasoning system:
    - depth 0: enriched baseline (deterministic, no LLM)
    - depth 1: chain-of-thought single pass (1 LLM call)
    - depth 2: multi-pass recursive reasoning (2-4 LLM calls)

    Mutates briefing_data["account_cards"] in place. Returns the card list.
    """
    cfg = settings.b2b_churn
    if not cfg.vendor_briefing_account_cards_enabled:
        briefing_data["account_cards"] = []
        return []

    pool = get_db_pool()
    if not pool.is_initialized:
        briefing_data["account_cards"] = []
        return []

    max_cards = cfg.vendor_briefing_account_cards_max
    if reasoning_depth is None:
        reasoning_depth = cfg.vendor_briefing_account_cards_reasoning_depth
    adaptive = cfg.vendor_briefing_account_cards_adaptive_depth

    # Fetch enabled card templates, index by name
    template_rows = await pool.fetch(
        "SELECT * FROM card_templates WHERE enabled = true ORDER BY name"
    )
    if not template_rows:
        briefing_data["account_cards"] = []
        return []

    templates_by_name: dict[str, dict[str, Any]] = {
        dict(r)["name"]: dict(r) for r in template_rows
    }

    # Select top accounts by urgency
    named_accounts = briefing_data.get("named_accounts") or []
    sorted_accounts = sorted(
        named_accounts,
        key=lambda a: float(a.get("urgency", 0)) if isinstance(a, dict) else 0,
        reverse=True,
    )[:max_cards]

    if not sorted_accounts:
        briefing_data["account_cards"] = []
        return []

    vendor_name = briefing_data.get("vendor_name", "Unknown")
    cards: list[dict[str, Any]] = []

    for account in sorted_accounts:
        if not isinstance(account, dict):
            continue

        # Step 1: always compute enriched baseline (depth 0)
        baseline = _build_enriched_baseline(account, briefing_data, vendor_name, target_mode)

        # Step 2: select depth
        if adaptive:
            depth = _select_reasoning_depth(account, briefing_data, reasoning_depth)
        else:
            depth = reasoning_depth

        card: dict[str, Any] = {
            "template_name": "sales_action",
            "template_label": "Sales Action Card",
            "company": baseline["company"],
            "urgency": baseline["urgency"],
            "reasoning_depth": depth,
            "baseline": baseline,
            "enriched": None,
            "token_usage": None,
        }

        # Step 3: route to enrichment function by depth
        if depth >= 2:
            card["template_name"] = "sales_action_decompose"
            card["template_label"] = "Sales Action Card (Deep Analysis)"
            enriched = await _deep_enrich_account_card(
                templates_by_name, baseline, target_mode, briefing_data,
            )
            if enriched:
                card["enriched"] = enriched.get("data")
                card["token_usage"] = enriched.get("token_usage")
                card["model"] = enriched.get("model")
            elif depth >= 1:
                # Depth 2 failed entirely, try depth 1 fallback
                cot_tpl = templates_by_name.get("sales_action_cot")
                if cot_tpl:
                    card["template_name"] = "sales_action_cot"
                    card["template_label"] = "Sales Action Card (Analyzed)"
                    card["reasoning_depth"] = 1
                    enriched = await _enrich_account_card_cot(
                        cot_tpl, baseline, target_mode,
                    )
                    if enriched:
                        card["enriched"] = enriched.get("data")
                        card["token_usage"] = enriched.get("token_usage")
                        card["model"] = enriched.get("model")

        elif depth == 1:
            cot_tpl = templates_by_name.get("sales_action_cot")
            if cot_tpl:
                card["template_name"] = "sales_action_cot"
                card["template_label"] = "Sales Action Card (Analyzed)"
                enriched = await _enrich_account_card_cot(
                    cot_tpl, baseline, target_mode,
                )
                if enriched:
                    card["enriched"] = enriched.get("data")
                    card["token_usage"] = enriched.get("token_usage")
                    card["model"] = enriched.get("model")

        # depth 0: baseline only, no LLM -- card already has enriched=None

        cards.append(card)

    briefing_data["account_cards"] = cards

    total_in = sum(c["token_usage"]["input_tokens"] for c in cards if c.get("token_usage"))
    total_out = sum(c["token_usage"]["output_tokens"] for c in cards if c.get("token_usage"))
    if total_in:
        logger.info(
            "Account cards: %d cards, tokens in=%d out=%d vendor=%s",
            len(cards), total_in, total_out, vendor_name,
        )
        from ...pipelines.llm import trace_llm_call
        from ...services.llm_router import get_llm as _get_llm_router
        _llm = _get_llm_router("campaign")
        trace_llm_call("task.vendor_briefing.account_cards", input_tokens=total_in,
                       output_tokens=total_out,
                       model=getattr(_llm, "model", "") if _llm else "",
                       provider=getattr(_llm, "name", "") if _llm else "",
                       metadata={"vendor": vendor_name, "card_count": len(cards)})

    return cards


# ---------------------------------------------------------------------------
# Briefing apply helpers (Sources 10-12)
# ---------------------------------------------------------------------------

def _apply_pain_points_to_briefing(
    briefing: dict[str, Any],
    pain_pts: list[dict[str, Any]],
) -> bool:
    """Backfill or enrich pain_breakdown from b2b_vendor_pain_points rows.

    Backfills when pain_breakdown is absent or sparse (< 2 entries).
    Enriches existing entries with avg_urgency, avg_rating, and
    confidence_score when the category matches.
    """
    if not pain_pts:
        return False
    current_pain = briefing.get("pain_breakdown") or []
    if len(current_pain) < 2:
        briefing["pain_breakdown"] = [
            {
                "category": p["pain_category"],
                "count": p["mention_count"],
                "avg_urgency": float(p.get("avg_urgency") or 0),
                "avg_rating": (
                    float(p["avg_rating"]) if p.get("avg_rating") is not None else None
                ),
                "confidence_score": float(p.get("confidence_score") or 0),
            }
            for p in pain_pts[:8]
        ]
        return True
    pp_by_cat = {p["pain_category"]: p for p in pain_pts}
    enriched = False
    for item in current_pain:
        if not isinstance(item, dict):
            continue
        cat = item.get("category") or item.get("pain_category") or ""
        pp = pp_by_cat.get(cat)
        if not pp:
            continue
        if pp.get("avg_urgency") is not None and "avg_urgency" not in item:
            item["avg_urgency"] = float(pp["avg_urgency"])
            enriched = True
        if pp.get("avg_rating") is not None and "avg_rating" not in item:
            item["avg_rating"] = float(pp["avg_rating"])
            enriched = True
        if pp.get("confidence_score") is not None and "confidence_score" not in item:
            item["confidence_score"] = float(pp["confidence_score"])
            enriched = True
    if enriched:
        briefing["pain_breakdown"] = current_pain
    return enriched


def _apply_account_intelligence_to_briefing(
    briefing: dict[str, Any],
    acct_data: dict[str, Any],
) -> bool:
    """Populate account-pressure fields from b2b_account_intelligence payload.

    Always writes account_pressure_metrics.
    Only backfills account_pressure_summary, named_account_count, and
    priority_account_names when the briefing does not already have them.
    """
    summary = acct_data.get("summary") or {}
    if not summary:
        return False
    briefing["account_pressure_metrics"] = {
        k: v for k, v in summary.items() if isinstance(v, (int, float))
    }
    if not briefing.get("account_pressure_summary"):
        high_intent = int(summary.get("high_intent_count") or 0)
        active_eval = int(summary.get("active_eval_signal_count") or 0)
        dm_count = int(summary.get("decision_maker_count") or 0)
        parts = []
        if high_intent:
            parts.append(
                f"{high_intent} high-intent account{'s' if high_intent != 1 else ''}"
            )
        if active_eval:
            parts.append(
                f"{active_eval} active evaluation signal{'s' if active_eval != 1 else ''}"
            )
        if dm_count:
            parts.append(
                f"{dm_count} decision-maker signal{'s' if dm_count != 1 else ''}"
            )
        if parts:
            briefing["account_pressure_summary"] = (
                "Detected: " + ", ".join(parts) + "."
            )
    if not briefing.get("named_account_count"):
        total = int(summary.get("total_accounts") or 0)
        if total:
            briefing["named_account_count"] = total
    accounts_raw = acct_data.get("accounts") or []
    # Materialize named_accounts when the briefing doesn't already have them.
    # Normalizes canonical account_intelligence shape to the named_accounts
    # format consumed by generate_account_cards() and the email template.
    if not briefing.get("named_accounts") and accounts_raw:
        normalized: list[dict[str, Any]] = []
        for a in accounts_raw:
            if not isinstance(a, dict):
                continue
            company = (
                a.get("company_name")
                or a.get("account_name")
                or a.get("name")
                or ""
            )
            if not company:
                continue
            urgency = float(a.get("urgency_score") or a.get("urgency") or 0)
            raw_title = a.get("buyer_role") or a.get("title") or ""
            title = raw_title if raw_title not in ("unknown", "") else None
            normalized.append({
                "company": company,
                "urgency": urgency,
                "title": title,
                "company_size": a.get("company_size"),
                "buying_stage": a.get("buying_stage"),
                "pain_category": a.get("pain_category"),
                "source": a.get("source"),
                "confidence_score": float(a.get("confidence_score") or 0),
                "reasoning_backed": False,
            })
        normalized.sort(key=lambda x: x["urgency"], reverse=True)
        if normalized:
            briefing["named_accounts"] = normalized[:10]
    if not briefing.get("priority_account_names"):
        priority: list[str] = []
        for a in accounts_raw:
            if not isinstance(a, dict):
                continue
            name = (
                a.get("company_name")
                or a.get("account_name")
                or a.get("name")
                or ""
            )
            if name and (
                a.get("high_intent")
                or float(a.get("urgency_score") or 0) >= 6.0
            ):
                priority.append(name)
        if priority:
            briefing["priority_account_names"] = priority[:5]
    return True


# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------

async def build_vendor_briefing(
    vendor_name: str,
    target_mode: str = "vendor_retention",
    analyst_summary_enabled: bool = True,
    account_cards_reasoning_depth: int | None = None,
) -> dict[str, Any] | None:
    """
    Build a briefing data dict for *vendor_name* from existing DB tables.

    When *target_mode* is ``'challenger_intel'``, the displacement data is
    flipped: instead of showing who the vendor is losing customers to, we
    show which incumbents are losing customers to this challenger.

    Returns None if the vendor is not found in any source.
    """
    pool = get_db_pool()
    if not pool.is_initialized:
        return None

    challenger_mode = target_mode == "challenger_intel"

    briefing: dict[str, Any] = {
        "vendor_name": vendor_name,
        "report_date": date.today().isoformat(),
        "booking_url": build_gate_url(vendor_name),
        "challenger_mode": challenger_mode,
        "data_sources": {
            "weekly_churn_feed": False,
            "reasoning_synthesis": False,
            "account_reasoning": False,
            "churn_signals": False,
            "evidence_vault": False,
            "product_profile": False,
            "segment_intelligence": False,
            "temporal_intelligence": False,
            "cross_vendor_conclusions": False,
            "buyer_profiles": False,
            "pain_points": False,
            "account_intelligence": False,
            "displacement_dynamics": False,
            "raw_review_quotes": False,
        },
    }

    found = False

    # ------------------------------------------------------------------
    # Source 1: weekly_churn_feed from b2b_intelligence
    # ------------------------------------------------------------------
    feed_entry = await _extract_feed_entry(pool, vendor_name)
    if feed_entry:
        found = True
        briefing["data_sources"]["weekly_churn_feed"] = True
        briefing.update({
            "churn_pressure_score": feed_entry.get("churn_pressure_score", 0),
            "churn_signal_density": feed_entry.get("churn_signal_density", 0),
            "avg_urgency": feed_entry.get("avg_urgency", 0),
            "review_count": feed_entry.get("total_reviews", 0),
            "dm_churn_rate": feed_entry.get("dm_churn_rate", 0),
            "pain_breakdown": feed_entry.get("pain_breakdown", []),
            "top_displacement_targets": _normalize_displacement(
                feed_entry.get("top_displacement_targets", [])
            ),
            "evidence": feed_entry.get("evidence", []),
            "trend": feed_entry.get("trend"),
            "named_accounts": feed_entry.get("named_accounts", []),
            "top_feature_gaps": feed_entry.get("top_feature_gaps", []),
            "category": feed_entry.get("category", "Software"),
            "budget_context": feed_entry.get("budget_context"),
        })
        if _apply_reasoning_synthesis_to_briefing(briefing, feed_entry):
            briefing["data_sources"]["reasoning_synthesis"] = True
        if briefing.get("account_reasoning"):
            briefing["data_sources"]["account_reasoning"] = True

    # ------------------------------------------------------------------
    # Source 2: b2b_churn_signals (fallback if no feed entry)
    # ------------------------------------------------------------------
    if not found:
        signals = await _fetch_churn_signals(pool, vendor_name)
        if signals:
            found = True
            briefing["data_sources"]["churn_signals"] = True
            # top_feature_gaps from signals is [{feature, count}] -- extract names
            raw_gaps = _jsonb_list(signals.get("top_feature_gaps"))
            feature_gap_strings = [
                g.get("feature", str(g)) if isinstance(g, dict) else str(g)
                for g in raw_gaps[:5]
            ]
            briefing.update({
                "churn_pressure_score": _compute_pressure(signals),
                "churn_signal_density": _safe_density(signals),
                "avg_urgency": float(signals.get("avg_urgency_score", 0)),
                "review_count": signals.get("total_reviews", 0),
                "dm_churn_rate": float(signals.get("decision_maker_churn_rate", 0) or 0) * 100,
                "pain_breakdown": _jsonb_list(signals.get("top_pain_categories")),
                "top_displacement_targets": _normalize_displacement(
                    _jsonb_list(signals.get("top_competitors"))
                ),
                "evidence": _jsonb_list(signals.get("quotable_evidence")),
                "trend": None,
                "named_accounts": _jsonb_list(signals.get("company_churn_list")),
                "top_feature_gaps": feature_gap_strings,
                "category": signals.get("product_category") or "Software",
            })

    if not found:
        return None

    # ------------------------------------------------------------------
    # Source 3: b2b_evidence_vault (canonical overlay for sparse fields)
    # ------------------------------------------------------------------
    vault = await _fetch_vendor_evidence_vault(pool, vendor_name)
    if _apply_evidence_vault_to_briefing(briefing, vault):
        briefing["data_sources"]["evidence_vault"] = True

    # ------------------------------------------------------------------
    # Source 4: b2b_product_profiles (enrichment)
    # ------------------------------------------------------------------
    profile = await _fetch_product_profile(pool, vendor_name)
    if profile:
        briefing["data_sources"]["product_profile"] = True
        if not briefing.get("category") or briefing["category"] == "Software":
            briefing["category"] = profile.get("product_category") or "Software"
        if profile.get("profile_summary"):
            briefing["profile_summary"] = profile["profile_summary"]
        # Feature gaps from weaknesses if not already populated
        if not briefing.get("top_feature_gaps"):
            weaknesses = _jsonb_list(profile.get("weaknesses"))
            briefing["top_feature_gaps"] = [
                w.get("area", str(w)) if isinstance(w, dict) else str(w)
                for w in weaknesses[:3]
            ]

    # ------------------------------------------------------------------
    # Source 5: reasoning view (synthesis-first, legacy fallback)
    # ------------------------------------------------------------------
    reasoning_view = None
    if (
        not briefing.get("reasoning_contracts")
        or not briefing.get("reasoning_anchor_examples")
        or not briefing.get("reasoning_reference_ids")
    ):
        from ._b2b_synthesis_reader import load_best_reasoning_view

        reasoning_view = await load_best_reasoning_view(
            pool,
            vendor_name,
            allow_legacy_fallback=False,
        )
        if reasoning_view is not None:
            if _apply_synthesis_view_to_briefing(briefing, reasoning_view):
                briefing["data_sources"]["reasoning_synthesis"] = True

    # ------------------------------------------------------------------
    # Source 6: b2b_segment_intelligence
    # ------------------------------------------------------------------
    seg_data = await _fetch_segment_intelligence(pool, vendor_name)
    if seg_data:
        briefing["segment_intelligence"] = seg_data
        briefing["data_sources"]["segment_intelligence"] = True

    # ------------------------------------------------------------------
    # Source 7: b2b_temporal_intelligence (fallback when feed is absent)
    # ------------------------------------------------------------------
    if not briefing.get("timing_intelligence"):
        temporal = await _fetch_temporal_intelligence(pool, vendor_name)
        if temporal:
            briefing["timing_intelligence"] = temporal
            timing_summary, timing_metrics, priority_triggers = (
                _timing_summary_payload(temporal)
            )
            if timing_summary:
                briefing["timing_summary"] = timing_summary
            if timing_metrics:
                briefing["timing_metrics"] = timing_metrics
            if priority_triggers:
                briefing["priority_timing_triggers"] = priority_triggers
            briefing["data_sources"]["temporal_intelligence"] = True

    # ------------------------------------------------------------------
    # Source 8: cross-vendor reasoning (synthesis-first, legacy fallback)
    # ------------------------------------------------------------------
    conclusions = await _fetch_cross_vendor_conclusions(
        pool,
        vendor_name,
        category=str(briefing.get("category") or "").strip() or None,
    )
    if conclusions:
        briefing["cross_vendor_conclusions"] = conclusions
        briefing["data_sources"]["cross_vendor_conclusions"] = True

    # ------------------------------------------------------------------
    # Source 9: b2b_vendor_buyer_profiles
    # ------------------------------------------------------------------
    buyer_profiles = await _fetch_buyer_profiles(pool, vendor_name)
    if buyer_profiles:
        briefing["buyer_profiles"] = buyer_profiles
        briefing["data_sources"]["buyer_profiles"] = True

    # ------------------------------------------------------------------
    # Source 10: b2b_vendor_pain_points (fallback + urgency enrichment)
    # ------------------------------------------------------------------
    pain_pts = await _fetch_pain_points(pool, vendor_name)
    if pain_pts:
        if _apply_pain_points_to_briefing(briefing, pain_pts):
            briefing["data_sources"]["pain_points"] = True

    # ------------------------------------------------------------------
    # Source 11: b2b_account_intelligence
    # ------------------------------------------------------------------
    acct_data = await _fetch_account_intelligence(pool, vendor_name)
    if acct_data:
        if _apply_account_intelligence_to_briefing(briefing, acct_data):
            briefing["data_sources"]["account_intelligence"] = True

    # ------------------------------------------------------------------
    # Source 12: b2b_displacement_dynamics (competitive augmentation)
    # ------------------------------------------------------------------
    disp_dynamics = await _fetch_displacement_dynamics(
        pool, vendor_name, challenger_mode=challenger_mode
    )
    if disp_dynamics:
        briefing["competitive_dynamics"] = disp_dynamics
        briefing["data_sources"]["displacement_dynamics"] = True
        # Augment cross_vendor_conclusions when fewer than 2 pairwise entries
        existing = briefing.get("cross_vendor_conclusions") or []
        pairwise_count = sum(
            1 for c in existing
            if isinstance(c, dict) and c.get("analysis_type") == "pairwise_battle"
        )
        if pairwise_count < 2:
            for pair in disp_dynamics.get("pairs") or []:
                summary = pair.get("battle_summary") or ""
                if not summary:
                    continue
                existing.append({
                    "analysis_type": "pairwise_battle",
                    "vendors": [vendor_name, pair.get("challenger") or ""],
                    "summary": summary,
                    "confidence": 0.6,
                    "source": "displacement_dynamics",
                })
                new_count = sum(
                    1 for c in existing
                    if isinstance(c, dict) and c.get("analysis_type") == "pairwise_battle"
                )
                if new_count >= 2:
                    break
            briefing["cross_vendor_conclusions"] = existing

    # ------------------------------------------------------------------
    # Challenger mode: flip displacement to show incumbents losing TO us
    # ------------------------------------------------------------------
    if challenger_mode:
        challenger_edges = await pool.fetch(
            """
            SELECT from_vendor AS competitor, SUM(mention_count) AS count
            FROM b2b_displacement_edges
            WHERE LOWER(to_vendor) = LOWER($1)
              AND computed_date > NOW() - INTERVAL '30 days'
            GROUP BY from_vendor
            ORDER BY count DESC
            LIMIT 5
            """,
            vendor_name,
        )
        if challenger_edges:
            briefing["top_displacement_targets"] = [
                {"competitor": r["competitor"], "count": int(r["count"] or 0)}
                for r in challenger_edges
            ]

    # ------------------------------------------------------------------
    # Source 5: b2b_reviews (high-urgency quotes if evidence thin)
    # ------------------------------------------------------------------
    evidence = briefing.get("evidence") or []
    if len(evidence) < 2:
        extra_quotes = await _fetch_high_urgency_quotes(pool, vendor_name, limit=5)
        # Deduplicate against existing evidence (evidence items may be dicts or strings)
        existing: set[str] = set()
        for e in evidence:
            text = e.get("quote", e.get("text", e)) if isinstance(e, dict) else e
            if isinstance(text, str):
                existing.add(text)
        for q in extra_quotes:
            q_text = q.get("text", "") if isinstance(q, dict) else q
            if q_text and q_text not in existing:
                evidence.append(q)
                existing.add(q_text)
                briefing["data_sources"]["raw_review_quotes"] = True
        briefing["evidence"] = evidence

    # ------------------------------------------------------------------
    # Archetype / wedge context: "why now" + falsification
    # ------------------------------------------------------------------
    try:
        if reasoning_view is None:
            from ._b2b_synthesis_reader import load_best_reasoning_view
            reasoning_view = await load_best_reasoning_view(
                pool,
                vendor_name,
                allow_legacy_fallback=False,
            )
        if reasoning_view is not None:
            wedge = reasoning_view.primary_wedge
            cn = reasoning_view.section("causal_narrative")
            # Prefer validated wedge, fall back to raw primary_wedge from contracts
            briefing["archetype"] = (
                wedge.value if wedge
                else cn.get("primary_wedge", "")
            )
            briefing["archetype_confidence"] = reasoning_view.confidence("causal_narrative")
            briefing["falsification_conditions"] = [
                fc.get("condition", fc) if isinstance(fc, dict) else fc
                for fc in reasoning_view.falsification_conditions()
            ]

            # "What changed" context via prior reasoning snapshot
            from ._b2b_synthesis_reader import load_prior_reasoning_snapshots

            prior = await load_prior_reasoning_snapshots(pool, [vendor_name])
            prior_data = prior.get(vendor_name, {})
            if prior_data.get("archetype"):
                briefing["archetype_was"] = prior_data["archetype"]
                # Compare using raw legacy archetype when available
                current_legacy = cn.get("_legacy_archetype") or briefing["archetype"]
                briefing["archetype_changed"] = (
                    prior_data["archetype"] != current_legacy
                )
    except Exception:
        logger.debug("Archetype enrichment skipped for %s", vendor_name, exc_info=True)

    # ------------------------------------------------------------------
    # Correlated articles: news articles aligned with archetype
    # ------------------------------------------------------------------
    try:
        art_rows = await pool.fetch(
            """
            SELECT na.title, na.url, na.source_name,
                   bac.correlation_type, bac.relevance_score,
                   bac.archetype AS corr_archetype
            FROM b2b_article_correlations bac
            JOIN news_articles na ON na.id = bac.article_id
            WHERE bac.vendor_name = $1
            ORDER BY bac.relevance_score DESC, bac.created_at DESC
            LIMIT 3
            """,
            vendor_name,
        )
        if art_rows:
            briefing["correlated_articles"] = [
                {
                    "title": r["title"],
                    "url": r["url"],
                    "source": r["source_name"],
                    "correlation_type": r["correlation_type"],
                    "relevance": float(r["relevance_score"] or 0),
                }
                for r in art_rows
            ]
    except Exception:
        logger.debug("Correlated articles skipped for %s", vendor_name, exc_info=True)

    # Analyst enrichment -- optional for scheduled deterministic batching.
    if analyst_summary_enabled:
        await _enrich_with_analyst_summary(briefing)

    # Account cards -- baseline data + optional LLM enrichment
    await generate_account_cards(
        briefing,
        reasoning_depth=account_cards_reasoning_depth,
        target_mode=target_mode,
    )

    _finalize_briefing_presentation(briefing)

    return briefing


async def _extract_feed_entry(pool: Any, vendor_name: str) -> dict | None:
    """Find vendor in the latest weekly_churn_feed."""
    row = await pool.fetchrow(
        """
        SELECT intelligence_data
        FROM b2b_intelligence
        WHERE report_type = 'weekly_churn_feed'
        ORDER BY report_date DESC, created_at DESC
        LIMIT 1
        """,
    )
    if not row:
        return None

    data = row["intelligence_data"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None

    # intelligence_data is the full report; the feed is a list inside it
    feed = data if isinstance(data, list) else data.get("weekly_churn_feed", data)
    if not isinstance(feed, list):
        return None

    vn_lower = vendor_name.lower()
    for entry in feed:
        if not isinstance(entry, dict):
            continue
        entry_vendor = entry.get("vendor", "") or entry.get("vendor_name", "")
        if entry_vendor.lower() == vn_lower:
            return entry

    return None


async def _fetch_reasoning_synthesis(pool: Any, vendor_name: str) -> dict[str, Any] | None:
    """Fetch best reasoning for one vendor (synthesis-first, legacy fallback).

    Returns the raw dict suitable for _apply_reasoning_synthesis_to_briefing.
    """
    from ._b2b_synthesis_reader import load_best_reasoning_view

    view = await load_best_reasoning_view(
        pool,
        vendor_name,
        allow_legacy_fallback=False,
    )
    if view is None:
        return None
    # Return materialized contracts as the raw dict for the existing applier
    contracts = view.materialized_contracts()
    if not contracts:
        return None
    result: dict[str, Any] = {"reasoning_contracts": contracts}
    if view.primary_wedge:
        result["synthesis_wedge"] = view.primary_wedge.value
        result["synthesis_wedge_label"] = view.wedge_label
    result["reasoning_source"] = "b2b_reasoning_synthesis"
    result["synthesis_schema_version"] = view.schema_version
    if view.as_of_date:
        result["data_as_of_date"] = view.as_of_date.isoformat()
    meta = view.meta
    if meta:
        result["evidence_window"] = meta
    return result


async def _fetch_segment_intelligence(pool: Any, vendor_name: str) -> dict[str, Any] | None:
    """Fetch top priority segments from b2b_segment_intelligence."""
    row = await pool.fetchrow(
        """
        SELECT segments
        FROM b2b_segment_intelligence
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        vendor_name,
    )
    if not row:
        return None
    data = row["segments"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(data, dict):
        return None
    roles = data.get("affected_roles") or []
    if not isinstance(roles, list):
        return None
    top_segments = sorted(
        [r for r in roles if isinstance(r, dict)],
        key=lambda r: float(r.get("priority_score") or 0),
        reverse=True,
    )[:4]
    if not top_segments:
        return None
    return {
        "top_segments": top_segments,
        "as_of_date": data.get("as_of_date"),
    }


async def _fetch_temporal_intelligence(pool: Any, vendor_name: str) -> dict[str, Any] | None:
    """Fetch temporal intelligence signals for one vendor."""
    row = await pool.fetchrow(
        """
        SELECT temporal
        FROM b2b_temporal_intelligence
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        vendor_name,
    )
    if not row:
        return None
    data = row["temporal"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    return data if isinstance(data, dict) else None


async def _fetch_cross_vendor_conclusions(
    pool: Any,
    vendor_name: str,
    *,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch pairwise battle and category council conclusions for one vendor."""
    window_days = max(settings.b2b_churn.intelligence_window_days, 60)
    from ._b2b_cross_vendor_synthesis import (
        build_cross_vendor_conclusions_for_vendor,
        load_best_cross_vendor_lookup,
    )

    xv_lookup = await load_best_cross_vendor_lookup(
        pool,
        as_of=date.today(),
        analysis_window_days=window_days,
    )
    return build_cross_vendor_conclusions_for_vendor(
        vendor_name,
        category=category,
        xv_lookup=xv_lookup,
        limit=5,
    )


async def _fetch_buyer_profiles(
    pool: Any, vendor_name: str
) -> list[dict[str, Any]]:
    """Fetch buyer role/stage profiles ranked by urgency-weighted intent signal.

    Rows at renewal_decision / evaluation stages with high urgency surface
    first even when they have lower review volume than post-purchase rows.
    """
    rows = await pool.fetch(
        """
        SELECT role_type, buying_stage, review_count, dm_count,
               avg_urgency, confidence_score
        FROM b2b_vendor_buyer_profiles
        WHERE LOWER(vendor_name) = LOWER($1)
          AND confidence_score >= 0.4
          AND role_type != 'unknown'
        ORDER BY
            (COALESCE(avg_urgency, 0) * confidence_score) DESC,
            review_count DESC
        LIMIT 6
        """,
        vendor_name,
    )
    if not rows:
        return []
    return [
        {
            "role_type": r["role_type"],
            "buying_stage": r["buying_stage"],
            "review_count": r["review_count"],
            "dm_count": r["dm_count"],
            "avg_urgency": float(r["avg_urgency"] or 0),
            "confidence_score": float(r["confidence_score"] or 0),
        }
        for r in rows
    ]


async def _fetch_pain_points(
    pool: Any, vendor_name: str
) -> list[dict[str, Any]]:
    """Fetch pain-category signal counts from b2b_vendor_pain_points."""
    rows = await pool.fetch(
        """
        SELECT pain_category, mention_count, primary_count,
               avg_urgency, avg_rating, confidence_score
        FROM b2b_vendor_pain_points
        WHERE LOWER(vendor_name) = LOWER($1)
          AND confidence_score >= 0.3
        ORDER BY mention_count DESC, confidence_score DESC
        LIMIT 8
        """,
        vendor_name,
    )
    if not rows:
        return []
    return [
        {
            "pain_category": r["pain_category"],
            "mention_count": r["mention_count"],
            "primary_count": r["primary_count"],
            "avg_urgency": float(r["avg_urgency"] or 0),
            "avg_rating": (
                float(r["avg_rating"]) if r["avg_rating"] is not None else None
            ),
            "confidence_score": float(r["confidence_score"] or 0),
        }
        for r in rows
    ]


async def _fetch_account_intelligence(
    pool: Any, vendor_name: str
) -> dict[str, Any] | None:
    """Fetch latest account-intelligence payload from b2b_account_intelligence."""
    row = await pool.fetchrow(
        """
        SELECT accounts, as_of_date
        FROM b2b_account_intelligence
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        vendor_name,
    )
    if not row:
        return None
    data = row["accounts"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(data, dict):
        return None
    return data


async def _fetch_displacement_dynamics(
    pool: Any, vendor_name: str, *, challenger_mode: bool = False
) -> dict[str, Any] | None:
    """Fetch top competitive displacement dynamics for one vendor.

    In standard mode: rows where vendor is from_vendor (losing customers).
    In challenger mode: rows where vendor is to_vendor (gaining customers);
    the 'challenger' field in each pair is the incumbent from_vendor.

    battle_summary is extracted as a plain conclusion string from the
    canonical dict shape built by build_displacement_dynamics().
    Sort key is edge_metrics.mention_count (canonical field name).
    """
    if challenger_mode:
        query = """
            SELECT DISTINCT ON (from_vendor)
                from_vendor AS peer_vendor, dynamics, as_of_date
            FROM b2b_displacement_dynamics
            WHERE LOWER(to_vendor) = LOWER($1)
            ORDER BY from_vendor, as_of_date DESC
            LIMIT 5
        """
    else:
        query = """
            SELECT DISTINCT ON (to_vendor)
                to_vendor AS peer_vendor, dynamics, as_of_date
            FROM b2b_displacement_dynamics
            WHERE LOWER(from_vendor) = LOWER($1)
            ORDER BY to_vendor, as_of_date DESC
            LIMIT 5
        """
    rows = await pool.fetch(query, vendor_name)
    if not rows:
        return None
    pairs = []
    for r in rows:
        dyn = r["dynamics"]
        if isinstance(dyn, str):
            try:
                dyn = json.loads(dyn)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(dyn, dict):
            continue
        # battle_summary is a dict in the canonical shape; extract conclusion string
        raw_summary = dyn.get("battle_summary")
        if isinstance(raw_summary, dict):
            battle_text = (
                raw_summary.get("conclusion")
                or raw_summary.get("winner") or ""
            )
        else:
            battle_text = str(raw_summary) if raw_summary else ""
        pairs.append({
            "challenger": r["peer_vendor"],
            "battle_summary": battle_text,
            "switch_reasons": (dyn.get("switch_reasons") or [])[:5],
            "flow_summary": dyn.get("flow_summary") or {},
            "edge_metrics": dyn.get("edge_metrics") or {},
            "trend_acceleration": dyn.get("trend_acceleration") or {},
            "as_of_date": (
                r["as_of_date"].isoformat()
                if r["as_of_date"] and hasattr(r["as_of_date"], "isoformat")
                else str(r["as_of_date"]) if r["as_of_date"] else None
            ),
        })
    if not pairs:
        return None
    # Sort by canonical edge_metrics.mention_count (not total_mentions)
    pairs.sort(
        key=lambda p: float(
            (p.get("edge_metrics") or {}).get("mention_count") or 0
        ),
        reverse=True,
    )
    return {"pairs": pairs[:2]}


async def _fetch_vendor_evidence_vault(pool: Any, vendor_name: str) -> dict[str, Any] | None:
    """Fetch the latest canonical evidence-vault row for one vendor."""
    row = await pool.fetchrow(
        """
        SELECT vault
        FROM b2b_evidence_vault
        WHERE LOWER(vendor_name) = LOWER($1)
          AND as_of_date <= $2
          AND analysis_window_days = $3
        ORDER BY as_of_date DESC, created_at DESC
        LIMIT 1
        """,
        vendor_name,
        date.today(),
        settings.b2b_churn.intelligence_window_days,
    )
    if not row:
        canonical_vendor = await resolve_vendor_name(vendor_name)
        row = await pool.fetchrow(
            """
            SELECT vault
            FROM b2b_evidence_vault
            WHERE LOWER(vendor_name) = LOWER($1)
              AND as_of_date <= $2
              AND analysis_window_days = $3
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            canonical_vendor,
            date.today(),
            settings.b2b_churn.intelligence_window_days,
        )
    if not row:
        return None
    vault = row.get("vault")
    if isinstance(vault, str):
        try:
            vault = json.loads(vault)
        except (json.JSONDecodeError, TypeError):
            return None
    return vault if isinstance(vault, dict) else None


async def _fetch_churn_signals(pool: Any, vendor_name: str) -> dict | None:
    """Fetch latest churn signal row for vendor."""
    row = await pool.fetchrow(
        """
        SELECT total_reviews, negative_reviews, churn_intent_count,
               avg_urgency_score, top_pain_categories, top_competitors,
               top_feature_gaps, price_complaint_rate,
               decision_maker_churn_rate, company_churn_list,
               quotable_evidence, product_category,
               archetype, archetype_confidence, falsification_conditions
        FROM b2b_churn_signals
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY last_computed_at DESC
        LIMIT 1
        """,
        vendor_name,
    )
    return dict(row) if row else None


async def _fetch_product_profile(pool: Any, vendor_name: str) -> dict | None:
    """Fetch product profile for vendor."""
    row = await pool.fetchrow(
        """
        SELECT profile_summary, commonly_compared_to, commonly_switched_from,
               weaknesses, product_category
        FROM b2b_product_profiles
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY last_computed_at DESC
        LIMIT 1
        """,
        vendor_name,
    )
    return dict(row) if row else None


async def _fetch_high_urgency_quotes(
    pool: Any, vendor_name: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Fetch quotable phrases with reviewer context from high-urgency reviews."""
    from atlas_brain.autonomous.tasks._b2b_shared import read_vendor_quote_evidence

    rows = await read_vendor_quote_evidence(
        pool,
        vendor_name=vendor_name,
        min_urgency=7.0,
        limit=limit,
        require_quotes=True,
    )
    quotes: list[dict[str, Any]] = []
    for row in rows:
        phrases = row.get("quotable_phrases")
        if isinstance(phrases, list):
            for p in phrases:
                text = p.strip() if isinstance(p, str) else ""
                if text:
                    quotes.append({
                        "text": text,
                        "company": row.get("reviewer_company"),
                        "title": row.get("reviewer_title"),
                        "company_size": row.get("company_size"),
                        "industry": row.get("industry"),
                    })
    return quotes


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_displacement(targets: list) -> list[dict]:
    """Normalize displacement target dicts to {competitor, count}."""
    result = []
    for t in (targets or []):
        if isinstance(t, dict):
            raw = t.get("competitor", t.get("name", "Unknown"))
            # Handle stringified JSON like '{"name": "Pardot", "context": ...}'
            if isinstance(raw, str) and raw.startswith("{"):
                try:
                    parsed = json.loads(raw)
                    raw = parsed.get("name") or parsed.get("competitor") or raw
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(raw, dict):
                raw = raw.get("name") or raw.get("competitor") or "Unknown"
            result.append({
                "competitor": raw,
                "count": t.get("count", t.get("mentions", 0)),
            })
        elif isinstance(t, str):
            result.append({"competitor": t, "count": 0})
    return result


def _jsonb_list(val: Any) -> list:
    """Parse a JSONB value that should be a list."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _safe_density(signals: dict) -> float:
    """Compute churn signal density % from signals row."""
    total = signals.get("total_reviews", 0)
    churn = signals.get("churn_intent_count", 0)
    if total <= 0:
        return 0.0
    return round((churn / total) * 100, 1)


def _compute_pressure(signals: dict) -> float:
    """Compute a pressure score from churn signals (0-100 scale)."""
    density = _safe_density(signals)
    urgency = float(signals.get("avg_urgency_score", 0))
    dm_rate = float(signals.get("decision_maker_churn_rate", 0) or 0) * 100
    # Weighted composite: density (40%), urgency*10 (40%), DM rate (20%)
    score = (density * 0.4) + (urgency * 10 * 0.4) + (dm_rate * 0.2)
    return min(round(score, 1), 100.0)


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

async def send_vendor_briefing(
    *,
    to_email: str,
    vendor_name: str,
    briefing_html: str,
    briefing_data: dict,
) -> dict | None:
    """Send a vendor briefing email via CampaignSender and persist to DB."""
    cfg = settings.campaign_sequence
    if not cfg.resend_api_key or not cfg.resend_from_email:
        logger.warning("Resend not configured -- cannot send briefing")
        return None

    # Canonicalize vendor name for consistent DB storage
    vendor_name = await resolve_vendor_name(vendor_name)

    pool = get_db_pool()

    challenger_mode = briefing_data.get("challenger_mode", False)

    # Suppression check
    if pool.is_initialized:
        suppressed = await is_suppressed(pool, email=to_email)
        if suppressed:
            logger.info("Suppressed briefing to %s (vendor=%s)", to_email, vendor_name)
            suppressed_subject = (
                f"Sales Intelligence Briefing: {vendor_name}"
                if challenger_mode
                else f"Churn Intelligence Briefing: {vendor_name}"
            )
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_vendor_briefings
                        (vendor_name, recipient_email, subject, briefing_data, status)
                    VALUES ($1, $2, $3, $4::jsonb, 'suppressed')
                    """,
                    vendor_name,
                    to_email,
                    suppressed_subject,
                    json.dumps(briefing_data, default=str),
                )
            except Exception as exc:
                logger.warning("Failed to persist suppressed record: %s", exc)
            return None

    sender_name = settings.b2b_churn.vendor_briefing_sender_name
    from_addr = f"{sender_name} <{cfg.resend_from_email}>"

    if briefing_data.get("prospect_mode"):
        if challenger_mode:
            subject = f"{vendor_name} -- Accounts In Motion"
        else:
            subject = f"{vendor_name} -- Churn Signals Detected"
    elif briefing_data.get("is_gated_delivery"):
        if challenger_mode:
            subject = f"Your {vendor_name} Sales Intelligence Report"
        else:
            subject = f"Your {vendor_name} Churn Intelligence Report"
    else:
        if challenger_mode:
            subject = f"Sales Intelligence Briefing: {vendor_name}"
        else:
            subject = f"Churn Intelligence Briefing: {vendor_name}"

    resend_id: str | None = None
    status = "sent"

    try:
        sender = get_campaign_sender()
        result = await sender.send(
            to=to_email,
            from_email=from_addr,
            subject=subject,
            body=briefing_html,
            tags=[
                {"name": "type", "value": "vendor_briefing"},
                {"name": "vendor", "value": vendor_name},
            ],
        )
        resend_id = result.get("id")
    except Exception as exc:
        logger.warning("Failed to send briefing to %s: %s", to_email, exc)
        status = "failed"

    # Persist delivery record
    if pool.is_initialized:
        try:
            await pool.execute(
                """
                INSERT INTO b2b_vendor_briefings
                    (vendor_name, recipient_email, subject, briefing_data, resend_id, status)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6)
                """,
                vendor_name,
                to_email,
                subject,
                json.dumps(briefing_data, default=str),
                resend_id,
                status,
            )
        except Exception as exc:
            logger.warning("Failed to persist briefing record: %s", exc)

    if status == "failed":
        return None

    return {"resend_id": resend_id, "status": status, "subject": subject}


# ---------------------------------------------------------------------------
# Approve / reject pending briefings (HITL)
# ---------------------------------------------------------------------------

async def send_approved_briefing(briefing_id: str) -> dict[str, Any]:
    """Send a pending_approval briefing and update its status to sent."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"error": "Database not ready"}

    row = await pool.fetchrow(
        """
        SELECT id, vendor_name, recipient_email, subject,
               briefing_data, briefing_html
        FROM b2b_vendor_briefings
        WHERE id = $1 AND status = 'pending_approval'
        """,
        briefing_id,
    )
    if not row:
        return {"error": "Briefing not found or not pending approval"}

    vendor_name = row["vendor_name"]
    to_email = row["recipient_email"]
    subject = row["subject"]
    html = row["briefing_html"]
    bd = row["briefing_data"]
    briefing_data = json.loads(bd) if isinstance(bd, str) else (bd or {})

    if not html:
        return {"error": "No rendered HTML stored for this briefing"}

    # Send via CampaignSender
    cfg = settings.campaign_sequence
    if not cfg.resend_api_key or not cfg.resend_from_email:
        return {"error": "Resend not configured"}

    sender_name = settings.b2b_churn.vendor_briefing_sender_name
    from_addr = f"{sender_name} <{cfg.resend_from_email}>"

    resend_id: str | None = None
    status = "sent"

    try:
        sender = get_campaign_sender()
        result = await sender.send(
            to=to_email,
            from_email=from_addr,
            subject=subject,
            body=html,
            tags=[
                {"name": "type", "value": "vendor_briefing"},
                {"name": "vendor", "value": vendor_name},
            ],
        )
        resend_id = result.get("id")
    except Exception as exc:
        logger.warning("Failed to send approved briefing %s: %s", briefing_id, exc)
        status = "failed"

    if status == "sent":
        await pool.execute(
            """
            UPDATE b2b_vendor_briefings
            SET status = $1, resend_id = $2, approved_at = NOW()
            WHERE id = $3
            """,
            status,
            resend_id,
            briefing_id,
        )
    else:
        await pool.execute(
            """
            UPDATE b2b_vendor_briefings
            SET status = $1
            WHERE id = $2
            """,
            status,
            briefing_id,
        )

    return {
        "id": str(briefing_id),
        "vendor_name": vendor_name,
        "to_email": to_email,
        "status": status,
        "resend_id": resend_id,
    }


async def reject_briefing(briefing_id: str, reason: str | None = None) -> dict[str, Any]:
    """Reject a pending_approval briefing."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"error": "Database not ready"}

    result = await pool.execute(
        """
        UPDATE b2b_vendor_briefings
        SET status = 'rejected', rejected_at = NOW(), reject_reason = $1
        WHERE id = $2 AND status = 'pending_approval'
        """,
        reason,
        briefing_id,
    )
    if result == "UPDATE 0":
        return {"error": "Briefing not found or not pending approval"}

    return {"id": str(briefing_id), "status": "rejected"}


# ---------------------------------------------------------------------------
# Recipient resolution
# ---------------------------------------------------------------------------

async def resolve_briefing_recipient(
    pool: Any, vendor_name: str
) -> dict[str, str] | None:
    """Resolve the best contact for a vendor briefing.

    Lookup priority:
    1. vendor_targets (active vendor_retention with contact_email)
    2. prospects (active, verified/probabilistic email, best seniority)
    """
    # Priority 1: vendor_targets (both vendor_retention and challenger_intel)
    row = await pool.fetchrow(
        """
        SELECT contact_email AS email, contact_name AS name,
               contact_role AS role, target_mode
        FROM vendor_targets
        WHERE LOWER(company_name) = LOWER($1)
          AND target_mode IN ('vendor_retention', 'challenger_intel')
          AND status = 'active'
          AND contact_email IS NOT NULL
        ORDER BY CASE WHEN account_id IS NULL THEN 1 ELSE 0 END,
                 updated_at DESC NULLS LAST,
                 created_at DESC NULLS LAST
        LIMIT 1
        """,
        vendor_name,
    )
    if row:
        return {
            "email": row["email"],
            "name": row["name"] or "",
            "role": row["role"] or "",
            "source": "vendor_target",
            "target_mode": row["target_mode"],
        }

    # Priority 2: prospects
    row = await pool.fetchrow(
        """
        SELECT email, first_name || ' ' || last_name AS name,
               title AS role
        FROM prospects
        WHERE LOWER(company_name) = LOWER($1)
          AND status = 'active'
          AND email IS NOT NULL
          AND email_status IN ('verified', 'probabilistic')
        ORDER BY CASE seniority
            WHEN 'c_suite' THEN 1
            WHEN 'owner' THEN 2
            WHEN 'founder' THEN 2
            WHEN 'vp' THEN 3
            WHEN 'head' THEN 4
            WHEN 'director' THEN 4
            WHEN 'manager' THEN 5
            WHEN 'senior' THEN 6
            ELSE 7
        END ASC
        LIMIT 1
        """,
        vendor_name,
    )
    if row:
        return {
            "email": row["email"],
            "name": row["name"] or "",
            "role": row["role"] or "",
            "source": "prospect",
        }

    return None


# ---------------------------------------------------------------------------
# Cooldown check
# ---------------------------------------------------------------------------

async def _check_cooldown(
    pool: Any, vendor_name: str, cooldown_days: int
) -> bool:
    """Return True if a recent briefing exists (should skip)."""
    row = await pool.fetchval(
        """
        SELECT EXISTS(
            SELECT 1 FROM b2b_vendor_briefings
            WHERE LOWER(vendor_name) = LOWER($1)
              AND status NOT IN ('failed', 'suppressed', 'rejected')
              AND created_at > NOW() - make_interval(days => $2)
        )
        """,
        vendor_name,
        cooldown_days,
    )
    return bool(row)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def generate_and_send_briefing(
    *,
    vendor_name: str,
    to_email: str | None = None,
) -> dict[str, Any]:
    """Build, render, send, and return summary for a vendor briefing."""
    pool = get_db_pool()

    # Auto-resolve recipient if not provided
    target_mode = "vendor_retention"
    if to_email is None:
        if not pool.is_initialized:
            return {"error": "Database not ready -- cannot resolve recipient"}
        contact = await resolve_briefing_recipient(pool, vendor_name)
        if not contact:
            return {"error": f"No contact found for vendor: {vendor_name}"}
        to_email = contact["email"]
        target_mode = contact.get("target_mode", "vendor_retention")

    briefing_data = await build_vendor_briefing(vendor_name, target_mode=target_mode)
    if not briefing_data:
        return {"error": f"No data found for vendor: {vendor_name}"}

    # First briefing to this vendor -> redacted prospect mode with gate CTA
    if pool.is_initialized and await _is_first_briefing(pool, vendor_name):
        briefing_data["prospect_mode"] = True
        briefing_data["gate_url"] = build_gate_url(vendor_name)

    briefing_html = render_vendor_briefing_html(briefing_data)

    result = await send_vendor_briefing(
        to_email=to_email,
        vendor_name=vendor_name,
        briefing_html=briefing_html,
        briefing_data=briefing_data,
    )

    if result is None:
        return {"error": "Failed to send briefing email (check Resend config or suppression)"}

    return {
        "vendor_name": vendor_name,
        "to_email": to_email,
        "resend_id": result.get("resend_id"),
        "status": result.get("status"),
        "subject": result.get("subject"),
        "sections": {
            "has_pain_breakdown": bool(briefing_data.get("pain_breakdown")),
            "has_displacement": bool(briefing_data.get("top_displacement_targets")),
            "has_quotes": bool(briefing_data.get("evidence")),
            "has_named_accounts": bool(briefing_data.get("named_accounts")),
            "has_feature_gaps": bool(briefing_data.get("top_feature_gaps")),
        },
    }


# ---------------------------------------------------------------------------
# Batch sender
# ---------------------------------------------------------------------------

async def send_batch_briefings() -> dict[str, Any]:
    """Send briefings to all eligible vendor targets with contacts."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"error": "Database not ready"}

    cfg = settings.b2b_churn
    max_batch = cfg.vendor_briefing_max_per_batch
    cooldown_days = cfg.vendor_briefing_cooldown_days
    analyst_summary_enabled = cfg.vendor_briefing_scheduled_analyst_enrichment_enabled
    account_cards_reasoning_depth = (
        cfg.vendor_briefing_scheduled_account_cards_reasoning_depth
    )

    # Fetch eligible vendor targets (both retention and challenger)
    rows = await pool.fetch(
        """
        SELECT company_name, contact_email, contact_name, contact_role,
               target_mode, account_id, created_at, updated_at
        FROM vendor_targets
        WHERE target_mode IN ('vendor_retention', 'challenger_intel')
          AND status = 'active'
          AND contact_email IS NOT NULL
        ORDER BY company_name
        """,
    )
    rows = dedupe_vendor_target_rows(rows)[:max_batch]

    queued = 0
    skipped_cooldown = 0
    skipped_suppressed = 0
    skipped_no_data = 0
    failed = 0
    details: list[dict] = []

    for row in rows:
        vendor_name = row["company_name"]
        to_email = row["contact_email"]
        target_mode = row["target_mode"]

        # Cooldown check
        if await _check_cooldown(pool, vendor_name, cooldown_days):
            skipped_cooldown += 1
            details.append({"vendor": vendor_name, "status": "skipped_cooldown"})
            continue

        # Suppression check
        suppressed = await is_suppressed(pool, email=to_email)
        if suppressed:
            skipped_suppressed += 1
            details.append({"vendor": vendor_name, "status": "skipped_suppressed"})
            continue

        # Build briefing data (pass target_mode for challenger framing)
        briefing_data = await build_vendor_briefing(
            vendor_name,
            target_mode=target_mode,
            analyst_summary_enabled=analyst_summary_enabled,
            account_cards_reasoning_depth=account_cards_reasoning_depth,
        )
        if not briefing_data:
            skipped_no_data += 1
            details.append({"vendor": vendor_name, "status": "skipped_no_data"})
            continue

        # First briefing to this vendor -> redacted prospect mode with gate CTA
        if await _is_first_briefing(pool, vendor_name):
            briefing_data["prospect_mode"] = True
            briefing_data["gate_url"] = build_gate_url(vendor_name)

        # Render HTML and store as pending_approval (HITL gate)
        briefing_html = render_vendor_briefing_html(briefing_data)

        challenger_mode = briefing_data.get("challenger_mode", False)
        if briefing_data.get("prospect_mode"):
            subject = (
                f"{vendor_name} -- Accounts In Motion"
                if challenger_mode
                else f"{vendor_name} -- Churn Signals Detected"
            )
        else:
            subject = (
                f"Sales Intelligence Briefing: {vendor_name}"
                if challenger_mode
                else f"Churn Intelligence Briefing: {vendor_name}"
            )

        try:
            await pool.execute(
                """
                INSERT INTO b2b_vendor_briefings
                    (vendor_name, recipient_email, subject, briefing_data,
                     briefing_html, status, target_mode)
                VALUES ($1, $2, $3, $4::jsonb, $5, 'pending_approval', $6)
                """,
                vendor_name,
                to_email,
                subject,
                json.dumps(briefing_data, default=str),
                briefing_html,
                target_mode,
            )
            queued += 1
            details.append({
                "vendor": vendor_name,
                "status": "pending_approval",
                "to": to_email,
            })
        except Exception as exc:
            logger.warning("Failed to queue briefing for %s: %s", vendor_name, exc)
            failed += 1
            details.append({"vendor": vendor_name, "status": "failed"})

    return {
        "queued": queued,
        "skipped_cooldown": skipped_cooldown,
        "skipped_suppressed": skipped_suppressed,
        "skipped_no_data": skipped_no_data,
        "failed": failed,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Scheduled task entry-point
# ---------------------------------------------------------------------------

async def run(task: ScheduledTask) -> dict:
    """Run vendor briefings as a scheduled task."""
    cfg = settings.b2b_churn
    if not cfg.vendor_briefing_enabled:
        return {"_skip_synthesis": True, "skipped": "vendor_briefing_enabled=false"}

    result = await send_batch_briefings()
    result["_skip_synthesis"] = True
    return result
