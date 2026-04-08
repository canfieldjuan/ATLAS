"""
B2B ABM Campaign Generation: uses Claude (draft LLM) to produce personalized
outreach content -- cold emails, LinkedIn messages, follow-up emails -- from
the highest-scoring churn intelligence opportunities.

Runs daily after b2b_churn_intelligence. Reads enriched b2b_reviews, scores
opportunities, groups by company, and generates multi-channel campaigns.

Returns _skip_synthesis.
"""

import asyncio
import json
import logging
import re
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...services.campaign_reasoning_context import (
    campaign_reasoning_atom_context as _shared_campaign_reasoning_atom_context,
    campaign_reasoning_delta_summary as _shared_campaign_reasoning_delta_summary,
    campaign_reasoning_scope_summary as _shared_campaign_reasoning_scope_summary,
)
from ...services.campaign_quality import campaign_quality_revalidation
from ...services.vendor_target_selection import dedupe_vendor_target_rows
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ..visibility import emit_event, record_attempt
from ._execution_progress import task_run_id as _task_run_id
from ._campaign_sequence_context import prepare_sequence_storage_contexts
from ._b2b_specificity import (
    campaign_proof_terms_from_audit,
    merge_specificity_contexts,
    specificity_audit_snapshot,
    surface_specificity_context,
)
from ._b2b_shared import _battle_card_company_is_display_safe
from .b2b_vendor_briefing import build_gate_url
from .campaign_audit import log_campaign_event

logger = logging.getLogger("atlas.autonomous.tasks.b2b_campaign_generation")

# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------

_PLACEHOLDER_NAMES = frozenset({
    "john smith", "jane doe", "john doe", "jane smith",
    "test user", "test contact", "sample contact",
    "first last", "name here",
})


def _sanitize_contact_name(name: str | None) -> str | None:
    """Return None for placeholder/seed contact names."""
    if not name:
        return None
    if name.strip().lower() in _PLACEHOLDER_NAMES:
        return None
    return name.strip()


def _build_selling_context(
    *,
    sender_name: str,
    sender_title: str,
    sender_company: str,
    booking_url: str = "",
    product_name: str = "",
    affiliate_url: str = "",
    primary_blog_post: dict[str, Any] | None = None,
    blog_posts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build selling context without hard-coded fallbacks."""
    selling = {
        "sender_name": sender_name,
        "sender_title": sender_title,
        "sender_company": sender_company,
    }
    if booking_url:
        selling["booking_url"] = booking_url
    if product_name:
        selling["product_name"] = product_name
    if affiliate_url:
        selling["affiliate_url"] = affiliate_url
    if primary_blog_post:
        selling["primary_blog_post"] = primary_blog_post
    if blog_posts:
        selling["blog_posts"] = blog_posts
    return selling


# ---------------------------------------------------------------------------
# Post-generation validation
# ---------------------------------------------------------------------------

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\w)\*([^*]+)\*(?!\w)")
_MD_HEADING_RE = re.compile(r"^#{1,3}\s+", re.MULTILINE)
_MD_LIST_RE = re.compile(r"^[-*]\s+", re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"\[(?:Name|Company|Your Name|First Name|Title)\]|\{\{.+?\}\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_REPORT_TIER_BANNED = re.compile(
    r"\b(dashboard|live feed|free trial|software|platform)\b", re.IGNORECASE,
)
# Words that trigger spam filters when used in subject lines.
_SUBJECT_SPAM_TRIGGERS = re.compile(
    r"\b(urgent|urgency|high.urgency|act now|limited time|don't miss"
    r"|last chance|exclusive offer|free|risk.free|guaranteed"
    r"|congratulations|winner|alert|warning|immediate)\b",
    re.IGNORECASE,
)
_SIGNOFF_RE = re.compile(r"<p>\s*(best|thanks|regards|sincerely|cheers)\s*,?\s*(<br>|</p>)", re.IGNORECASE)


def _ensure_html(body: str) -> str:
    """Convert markdown artifacts to minimal HTML."""
    # Bold: **text** -> <strong>text</strong>
    body = _MD_BOLD_RE.sub(r"<strong>\1</strong>", body)
    # Italic: *text* -> <em>text</em>
    body = _MD_ITALIC_RE.sub(r"<em>\1</em>", body)
    # Strip markdown headings
    body = _MD_HEADING_RE.sub("", body)
    # Strip markdown list markers
    body = _MD_LIST_RE.sub("", body)

    # Wrap in <p> tags if missing
    if "<p>" not in body.lower():
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if not paragraphs:
            # Single block -- split on double newline or treat as one paragraph
            lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
            if len(lines) > 1:
                paragraphs = lines
            else:
                paragraphs = [body.strip()]
        body = "".join(f"<p>{p}</p>" for p in paragraphs)

    return body


def _truncate_to_limit(body: str, max_words: int) -> str:
    """Truncate body to max_words at the nearest sentence/paragraph boundary."""
    plain = _HTML_TAG_RE.sub(" ", body)
    words = plain.split()
    if len(words) <= max_words:
        return body

    # Strategy 1: split on </p>, accumulate whole paragraphs
    parts = re.split(r"(</p>)", body)
    result = []
    word_count = 0
    for part in parts:
        part_plain = _HTML_TAG_RE.sub(" ", part).strip()
        part_words = len(part_plain.split()) if part_plain else 0
        if word_count + part_words > max_words and result:
            break
        result.append(part)
        word_count += part_words

    truncated = "".join(result)

    # Strategy 2: if still over (single giant paragraph), hard-truncate
    truncated_plain = _HTML_TAG_RE.sub(" ", truncated)
    if len(truncated_plain.split()) > max_words:
        # Take first max_words from the plain text, find last sentence end
        kept_words = plain.split()[:max_words]
        kept_text = " ".join(kept_words)
        # Try to cut at last period
        last_period = kept_text.rfind(". ")
        if last_period > len(kept_text) // 2:
            kept_text = kept_text[: last_period + 1]
        truncated = f"<p>{kept_text}</p>"

    # Ensure we close any open <p> tag
    if truncated.count("<p>") > truncated.count("</p>"):
        truncated += "</p>"
    return truncated if truncated.strip() else body


def _campaign_word_limits(
    *,
    channel: str,
    target_mode: str | None,
) -> tuple[int, int] | None:
    raw_limits = settings.b2b_campaign.word_limits
    if not isinstance(raw_limits, dict):
        return None
    normalized_mode = str(target_mode or "").strip().lower()
    mode_limits = raw_limits.get(normalized_mode)
    if not isinstance(mode_limits, dict):
        mode_limits = raw_limits.get("default")
    if not isinstance(mode_limits, dict):
        return None
    resolved = mode_limits.get(str(channel or "").strip())
    if not isinstance(resolved, list) or len(resolved) != 2:
        return None
    try:
        min_words = int(resolved[0])
        max_words = int(resolved[1])
    except (TypeError, ValueError):
        return None
    if min_words < 0 or max_words < min_words:
        return None
    return min_words, max_words


def _append_signoff(body: str, payload: dict[str, Any]) -> str:
    """Append a deterministic sender sign-off when the model omits one."""
    if _SIGNOFF_RE.search(body):
        return body
    selling = payload.get("selling") or {}
    sender_name = str(selling.get("sender_name") or "").strip()
    sender_title = str(selling.get("sender_title") or "").strip()
    sender_company = str(selling.get("sender_company") or "").strip()
    if not sender_name and not sender_company:
        return body
    if not sender_name and sender_company:
        sender_name = f"{sender_company} team"
    lines = ["Best,", sender_name]
    if sender_title and sender_company:
        lines.append(f"{sender_title}, {sender_company}")
    elif sender_title:
        lines.append(sender_title)
    elif sender_company and sender_company.lower() not in sender_name.lower():
        lines.append(sender_company)
    return f"{body}<p>{'<br>'.join(lines)}</p>"


def _validate_campaign_content(
    parsed: dict[str, Any],
    channel: str,
    tier: str = "",
    target_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and fix campaign content. Returns (fixed_content, issues_dict)."""
    issues: dict[str, Any] = {}

    # Required fields
    for field in ("subject", "body", "cta"):
        if not parsed.get(field) or not isinstance(parsed[field], str):
            issues["missing_field"] = field
            return parsed, issues

    subject = parsed["subject"].strip()
    body = parsed["body"].strip()
    cta = parsed["cta"].strip()

    # Subject spam trigger words: flag for LLM retry
    spam_match = _SUBJECT_SPAM_TRIGGERS.search(subject)
    if spam_match:
        issues["subject_spam_trigger"] = spam_match.group()

    # Subject length: truncate at word boundary if over 60 chars
    if len(subject) > 60:
        words = subject.split()
        truncated = []
        for w in words:
            candidate = " ".join(truncated + [w])
            if len(candidate) > 57:
                break
            truncated.append(w)
        subject = " ".join(truncated)

    # HTML enforcement
    body = _ensure_html(body)

    # Placeholder detection
    if _PLACEHOLDER_RE.search(body):
        issues["placeholders"] = True
        return parsed, issues

    # Word count
    limits = _campaign_word_limits(channel=channel, target_mode=target_mode)
    if limits:
        plain = _HTML_TAG_RE.sub(" ", body)
        wc = len(plain.split())
        min_words, max_words = limits
        if wc < min_words:
            issues["word_count"] = wc
            issues["min_words"] = min_words
        elif wc > max_words:
            issues["word_count"] = wc
            issues["max_words"] = max_words
            # Apply truncation as fallback (caller may retry first)
            body = _truncate_to_limit(body, max_words)

    # Report-tier banned words: strip from body + cta
    if tier == "report":
        for text_field in (body, cta):
            if _REPORT_TIER_BANNED.search(text_field):
                issues["report_tier_violation"] = True
                break
        # Auto-fix: replace banned phrases
        body = _REPORT_TIER_BANNED.sub("intelligence brief", body)
        cta = _REPORT_TIER_BANNED.sub("brief", cta)

    parsed = {**parsed, "subject": subject, "body": body, "cta": cta}
    return parsed, issues


def _campaign_artifact_key(
    *,
    company_name: str,
    batch_id: str | None,
    channel: str,
) -> str:
    company = str(company_name or "").strip() or "unknown_company"
    batch = str(batch_id or "").strip() or "no_batch"
    ch = str(channel or "").strip() or "unknown_channel"
    return f"{batch}:{company}:{ch}"


def _campaign_specificity_context(payload: dict[str, Any]) -> dict[str, Any]:
    direct = surface_specificity_context(
        payload,
        surface="campaign",
        nested_keys=("briefing_context",),
    )
    company_context = payload.get("company_context")
    nested = surface_specificity_context(
        company_context,
        surface="campaign",
        nested_keys=("briefing_context",),
    ) if isinstance(company_context, dict) else {}
    return merge_specificity_contexts(direct, nested)


def _inject_reasoning_campaign_context(
    target: dict[str, Any],
    consumer_context: dict[str, Any] | None,
) -> None:
    if not isinstance(target, dict) or not isinstance(consumer_context, dict):
        return
    anchors = consumer_context.get("anchor_examples")
    if isinstance(anchors, dict) and anchors:
        target["reasoning_anchor_examples"] = anchors
    highlights = consumer_context.get("witness_highlights")
    if isinstance(highlights, list) and highlights:
        target["reasoning_witness_highlights"] = highlights
    reference_ids = consumer_context.get("reference_ids")
    if isinstance(reference_ids, dict) and reference_ids:
        target["reasoning_reference_ids"] = reference_ids
    scope_summary = _campaign_reasoning_scope_summary(
        consumer_context.get("scope_manifest"),
    )
    if scope_summary:
        target["reasoning_scope_summary"] = scope_summary
    atom_context = _campaign_reasoning_atom_context(consumer_context)
    if atom_context:
        target["reasoning_atom_context"] = atom_context
    delta_summary = _campaign_reasoning_delta_summary(
        consumer_context.get("reasoning_delta"),
    )
    if delta_summary:
        target["reasoning_delta_summary"] = delta_summary
    disclaimers = consumer_context.get("reasoning_section_disclaimers")
    if isinstance(disclaimers, dict) and disclaimers:
        target["reasoning_section_disclaimers"] = disclaimers


def _campaign_reasoning_scope_summary(scope_manifest: dict[str, Any] | None) -> dict[str, Any]:
    return _shared_campaign_reasoning_scope_summary(scope_manifest)


def _campaign_reasoning_atom_context(consumer_context: dict[str, Any] | None) -> dict[str, Any]:
    return _shared_campaign_reasoning_atom_context(consumer_context)


def _campaign_reasoning_delta_summary(reasoning_delta: dict[str, Any] | None) -> dict[str, Any]:
    return _shared_campaign_reasoning_delta_summary(reasoning_delta)


def _campaign_specificity_audit(
    *,
    body: str,
    channel: str,
    specificity_context: dict[str, Any] | None,
) -> dict[str, Any]:
    context = specificity_context if isinstance(specificity_context, dict) else {}
    return specificity_audit_snapshot(
        body,
        anchor_examples=context.get("anchor_examples"),
        witness_highlights=context.get("witness_highlights"),
        reference_ids=context.get("reference_ids"),
        allow_company_names=False,
        min_anchor_hits=int(settings.b2b_campaign.specificity_min_anchor_hits),
        require_anchor_support=bool(
            settings.b2b_campaign.specificity_require_anchor_support
        ),
        require_timing_or_numeric_when_available=bool(
            settings.b2b_campaign.specificity_require_timing_or_numeric_when_available
        ),
        include_competitor_terms=channel != "email_cold",
    )


def _campaign_specificity_terms(
    specificity: dict[str, Any] | None,
    *,
    channel: str,
) -> list[str]:
    return campaign_proof_terms_from_audit(
        specificity,
        channel=channel,
        limit=int(settings.b2b_campaign.specificity_revision_term_limit),
    )


def _campaign_specificity_revision(
    *,
    channel: str,
    specificity: dict[str, Any] | None,
) -> str:
    lines = [
        "REVISION REQUIRED: The previous body failed the witness-backed specificity gate.",
    ]
    exact_terms = _campaign_specificity_terms(specificity, channel=channel)
    blocking_issues = [
        str(issue).strip().lower()
        for issue in (specificity or {}).get("blocking_issues", [])
        if str(issue or "").strip()
    ]
    if any("timing or numeric anchor" in issue for issue in blocking_issues):
        if exact_terms:
            lines.append(
                "Rewrite the body so it explicitly mentions at least one of these exact numeric or timing details: "
                + "; ".join(exact_terms)
                + "."
            )
        else:
            lines.append(
                "Rewrite the body so it explicitly mentions a concrete numeric or timing anchor from the provided witness-backed context."
            )
    elif exact_terms:
        lines.append(
            "Rewrite the body so it explicitly uses at least one of these exact witness-backed proof terms: "
            + "; ".join(exact_terms)
            + "."
        )
    else:
        lines.append(
            "Rewrite the body so it explicitly uses a concrete witness-backed proof anchor instead of aggregate summary language."
        )
    if any(issue.startswith("report_tier_language:") for issue in blocking_issues):
        lines.append(
            "Remove dashboard, live feed, free trial, software, and platform language from report-tier copy."
        )
    if channel == "email_cold":
        if any(issue.startswith("incumbent_name_in_email_cold:") for issue in blocking_issues):
            lines.append("Do not name incumbents in the cold email.")
        else:
            lines.append("Do not name competitors in the cold email.")
    lines.append("Do not reveal private account names or review sources.")
    return " ".join(lines)


def _campaign_storage_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    revalidation = payload.get("_campaign_revalidation")
    if isinstance(revalidation, dict):
        merged_metadata = revalidation.get("metadata")
        if isinstance(merged_metadata, dict):
            metadata.update(merged_metadata)
    if not metadata:
        tier = str(payload.get("tier") or "").strip()
        if tier:
            metadata["tier"] = tier
        specificity_context = payload.get("_campaign_specificity_context")
        if isinstance(specificity_context, dict):
            if specificity_context.get("anchor_examples"):
                metadata["reasoning_anchor_examples"] = specificity_context["anchor_examples"]
            if specificity_context.get("witness_highlights"):
                metadata["reasoning_witness_highlights"] = specificity_context["witness_highlights"]
            if specificity_context.get("reference_ids"):
                metadata["reasoning_reference_ids"] = specificity_context["reference_ids"]
        proof_terms = payload.get("campaign_proof_terms")
        if isinstance(proof_terms, list) and proof_terms:
            metadata["campaign_proof_terms"] = [
                str(term or "").strip()
                for term in proof_terms
                if str(term or "").strip()
            ]
    generation_audit = payload.get("_generation_audit")
    if isinstance(generation_audit, dict) and generation_audit:
        metadata["generation_audit"] = generation_audit
        specificity = generation_audit.get("specificity")
        if isinstance(specificity, dict) and specificity:
            metadata["latest_specificity_audit"] = {
                **specificity,
                "boundary": "generation",
            }
    return metadata


async def _record_campaign_generation_failure(
    pool,
    *,
    artifact_id: str,
    company_name: str,
    channel: str,
    generation_audit: dict[str, Any] | None,
) -> None:
    audit = generation_audit if isinstance(generation_audit, dict) else {}
    specificity = audit.get("specificity") if isinstance(audit.get("specificity"), dict) else {}
    blocking_issues = list(specificity.get("blocking_issues") or [])
    warnings = list(specificity.get("warnings") or [])
    failure_reason = str(audit.get("failure_reason") or "generation_failed")
    await record_attempt(
        pool,
        artifact_type="campaign",
        artifact_id=artifact_id,
        attempt_no=1,
        stage="generation",
        status="failed",
        blocker_count=len(blocking_issues),
        warning_count=len(warnings),
        blocking_issues=blocking_issues,
        warnings=warnings,
        failure_step="generation",
        error_message=failure_reason,
    )
    await emit_event(
        pool,
        stage="campaign_generation",
        event_type="generation_failure",
        entity_type="campaign",
        entity_id=artifact_id,
        summary=f"Campaign generation failed for {company_name} / {channel}",
        severity="warning",
        actionable=True,
        artifact_type="campaign",
        reason_code=failure_reason[:80],
        detail=audit,
    )

# Reuse scoring constants from b2b_affiliates
_ROLE_SCORES = {
    "decision_maker": 20,
    "economic_buyer": 15,
    "champion": 15,
    "evaluator": 10,
}

_STAGE_SCORES = {
    "active_purchase": 25,
    "evaluation": 20,
    "renewal_decision": 15,
    "post_purchase": 5,
}

_CONTEXT_SCORES = {
    "considering": 10,
    "switched_to": 8,
    "compared": 6,
    "switched_from": 2,
}


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_json_object_field(val: Any) -> dict[str, Any]:
    """Safely parse a JSONB field that may be a str, dict, or None."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _dedupe_texts(values: list[str], max_items: int) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        marker = text.lower()
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(text)
        if len(deduped) >= max_items:
            break
    return deduped


_CAMPAIGN_PAIN_SEVERITY_RANK = {
    "critical": 5,
    "high": 4,
    "primary": 4,
    "medium": 3,
    "secondary": 3,
    "low": 2,
    "minor": 2,
    "mentioned": 1,
    "": 0,
}


def _campaign_stable_row_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("review_id") or ""),
            json.dumps(row, sort_keys=True, separators=(",", ":"), default=str),
        ),
    )


def _campaign_sorted_count_rows(
    counts: dict[str, int],
    *,
    field_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    return [
        {field_name: key, "count": value}
        for key, value in sorted(
            counts.items(),
            key=lambda item: (-item[1], str(item[0]).lower(), str(item[0])),
        )[:limit]
    ]


def _campaign_sorted_count_keys(counts: dict[str, int], *, limit: int) -> list[str]:
    return [
        key
        for key, _value in sorted(
            counts.items(),
            key=lambda item: (-item[1], str(item[0]).lower(), str(item[0])),
        )[:limit]
    ]


def _campaign_merge_pain_severity(existing: str, incoming: str) -> str:
    current = str(existing or "").strip().lower()
    candidate = str(incoming or "").strip().lower()
    if _CAMPAIGN_PAIN_SEVERITY_RANK.get(candidate, 0) >= _CAMPAIGN_PAIN_SEVERITY_RANK.get(current, 0):
        return candidate or current
    return current


def _briefing_text_list(
    items: Any, *, keys: tuple[str, ...], max_items: int = 5,
) -> list[str]:
    values: list[str] = []
    if not isinstance(items, list):
        return values
    for item in items:
        if isinstance(item, str):
            values.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in keys:
            text = str(item.get(key) or "").strip()
            if text:
                values.append(text)
                break
    return _dedupe_texts(values, max_items)


def _briefing_context_from_data(briefing_data: Any) -> dict[str, Any]:
    """Extract a compact campaign-safe summary from persisted vendor briefings."""
    briefing = _parse_json_object_field(briefing_data)
    if not briefing:
        return {}

    context: dict[str, Any] = {}
    for field in (
        "account_pressure_summary",
        "timing_summary",
        "segment_targeting_summary",
        "trend",
        "category",
    ):
        text = str(briefing.get(field) or "").strip()
        if text:
            context[field] = text

    summary = str(
        briefing.get("executive_summary")
        or briefing.get("headline")
        or briefing.get("profile_summary")
        or ""
    ).strip()
    if summary:
        context["executive_summary"] = summary

    account_names = _briefing_text_list(
        briefing.get("priority_account_names") or briefing.get("named_accounts"),
        keys=("account_name", "company_name", "name", "company"),
    )
    if account_names:
        context["priority_account_names"] = account_names

    displacement = _briefing_text_list(
        briefing.get("top_displacement_targets"),
        keys=("name", "vendor", "opponent"),
    )
    if displacement:
        context["top_displacement_targets"] = displacement

    feature_gaps = _briefing_text_list(
        briefing.get("top_feature_gaps"),
        keys=("feature", "name", "gap"),
    )
    if feature_gaps:
        context["top_feature_gaps"] = feature_gaps

    pain_labels = _briefing_text_list(
        briefing.get("pain_labels") or briefing.get("pain_breakdown"),
        keys=("label", "category", "pain_category"),
    )
    if pain_labels:
        context["pain_labels"] = pain_labels

    # account pressure metrics from b2b_account_intelligence
    acct_metrics = briefing.get("account_pressure_metrics")
    if isinstance(acct_metrics, dict):
        high_intent = int(acct_metrics.get("high_intent_count") or 0)
        active_eval = int(acct_metrics.get("active_eval_signal_count") or 0)
        if high_intent:
            context["account_high_intent_count"] = high_intent
        if active_eval:
            context["account_active_eval_count"] = active_eval

    # timing triggers from b2b_temporal_intelligence
    triggers = _briefing_text_list(
        briefing.get("priority_timing_triggers"),
        keys=("trigger", "label", "text"),
        max_items=3,
    )
    if triggers:
        context["priority_timing_triggers"] = triggers

    # top buyer profiles (role + stage + urgency)
    raw_profiles = [
        p for p in (briefing.get("buyer_profiles") or []) if isinstance(p, dict)
    ]
    if raw_profiles:
        context["top_buyer_profiles"] = [
            {
                "role_type": str(p.get("role_type") or ""),
                "buying_stage": str(p.get("buying_stage") or ""),
                "avg_urgency": float(p.get("avg_urgency") or 0),
            }
            for p in raw_profiles[:2]
        ]

    # competitive dynamics from b2b_displacement_dynamics
    comp_dyn = briefing.get("competitive_dynamics")
    if isinstance(comp_dyn, dict):
        raw_pairs = [
            p for p in (comp_dyn.get("pairs") or []) if isinstance(p, dict)
        ]
        if raw_pairs:
            context["competitive_dynamics"] = [
                {
                    "challenger": str(p.get("challenger") or ""),
                    "battle_summary": str(p.get("battle_summary") or "")[:300],
                    "switch_reasons": [
                        r.get("reason") or r.get("reason_category") or str(r)
                        if isinstance(r, dict)
                        else str(r)
                        for r in (p.get("switch_reasons") or [])[:3]
                        if r
                    ],
                }
                for p in raw_pairs[:2]
            ]

    # pain urgency enrichment from pain_points overlay
    pain_with_urgency = [
        p for p in (briefing.get("pain_breakdown") or [])
        if isinstance(p, dict) and p.get("avg_urgency")
    ]
    if pain_with_urgency:
        context["pain_urgency"] = [
            {
                "category": str(
                    p.get("category") or p.get("pain_category") or ""
                ),
                "avg_urgency": float(p.get("avg_urgency") or 0),
            }
            for p in pain_with_urgency[:4]
        ]

    specificity = surface_specificity_context(briefing, surface="campaign")
    if specificity.get("anchor_examples"):
        context["reasoning_anchor_examples"] = specificity["anchor_examples"]
    if specificity.get("witness_highlights"):
        context["reasoning_witness_highlights"] = specificity["witness_highlights"]
    if specificity.get("reference_ids"):
        context["reasoning_reference_ids"] = specificity["reference_ids"]

    return context


_CAMPAIGN_WORD_RE = re.compile(r"[a-z0-9]+")
_COMPARISON_TOPIC_TYPES = {
    "vendor_showdown",
    "vendor_alternative",
    "migration_guide",
    "switching_story",
    "best_fit_guide",
    "vendor_deep_dive",
}


def _campaign_tokenize_text(value: str) -> list[str]:
    return _CAMPAIGN_WORD_RE.findall((value or "").lower())


def _campaign_text_matches_term(text: str, term: str) -> bool:
    hay = _campaign_tokenize_text(text)
    needle = _campaign_tokenize_text(term)
    if not hay or not needle:
        return False
    if len(needle) == 1:
        return needle[0] in hay
    for idx in range(0, len(hay) - len(needle) + 1):
        if hay[idx:idx + len(needle)] == needle:
            return True
    return False


def _comparison_candidates_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in context.get("recommended_alternatives") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("vendor_name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        candidates.append({
            "vendor_name": name,
            "source": "recommended_alternative",
            "selection_reason": str(item.get("reasoning") or item.get("profile_summary") or "").strip(),
        })

    for item in context.get("competitors_considering") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            reason = str(item.get("reason") or "").strip()
        else:
            name = str(item or "").strip()
            reason = ""
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        candidates.append({
            "vendor_name": name,
            "source": "competitor_considering",
            "selection_reason": reason,
        })

    return candidates


def _prioritize_blog_posts_for_context(
    blog_posts: list[dict[str, Any]],
    *,
    incumbent_vendor: str,
    candidates: list[dict[str, Any]],
    pain_terms: list[str],
) -> list[dict[str, Any]]:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    primary_vendor = str((candidates[0] or {}).get("vendor_name") or "").strip() if candidates else ""

    for idx, post in enumerate(blog_posts or []):
        if not isinstance(post, dict):
            continue
        text = " ".join(
            str(post.get(field) or "").strip()
            for field in ("title", "url", "topic_type")
        )
        score = 0
        primary_vendor_match = bool(primary_vendor and _campaign_text_matches_term(text, primary_vendor))
        incumbent_vendor_match = bool(incumbent_vendor and _campaign_text_matches_term(text, incumbent_vendor))
        pain_match = any(_campaign_text_matches_term(text, term) for term in pain_terms)
        if primary_vendor_match:
            score += 6
        if incumbent_vendor_match:
            score += 4
        if (
            str(post.get("topic_type") or "").strip() in _COMPARISON_TOPIC_TYPES
            and (primary_vendor_match or incumbent_vendor_match or pain_match)
        ):
            score += 3
        if pain_match:
            score += 2
        if score <= 0:
            continue
        scored.append((score, -idx, post))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [post for _, _, post in scored]


def _build_comparison_asset(
    context: dict[str, Any],
    blog_posts: list[dict[str, Any]],
) -> dict[str, Any]:
    company_name = str(context.get("company") or "").strip()
    incumbent_vendor = str(context.get("churning_from") or "").strip()
    pain_terms = [
        str(item.get("category") or "").strip()
        for item in context.get("pain_categories") or []
        if isinstance(item, dict) and str(item.get("category") or "").strip()
    ]
    candidates = _comparison_candidates_from_context(context)
    ranked_posts = _prioritize_blog_posts_for_context(
        blog_posts,
        incumbent_vendor=incumbent_vendor,
        candidates=candidates,
        pain_terms=pain_terms,
    )
    primary_candidate = candidates[0] if candidates else {}
    primary_post = ranked_posts[0] if ranked_posts else None
    company_safe = _battle_card_company_is_display_safe(
        company_name,
        current_vendor=incumbent_vendor,
        role=context.get("role_type") or context.get("reviewer_title"),
        company_size=context.get("company_size") or context.get("seat_count"),
        buying_stage=context.get("buying_stage") or context.get("decision_timeline"),
    )
    reasons = []
    if company_safe:
        reasons.append("named_company")
    if pain_terms:
        reasons.append("pain_signal")
    if primary_candidate.get("vendor_name"):
        reasons.append("comparison_vendor")
    if primary_post:
        reasons.append("blog_asset")

    return {
        "qualified": bool(company_safe and pain_terms and primary_candidate.get("vendor_name") and primary_post),
        "qualification_reasons": reasons,
        "company_safe": company_safe,
        "incumbent_vendor": incumbent_vendor,
        "alternative_vendor": primary_candidate.get("vendor_name"),
        "selection_source": primary_candidate.get("source"),
        "selection_reason": primary_candidate.get("selection_reason"),
        "pain_categories": pain_terms[:3],
        "primary_blog_post": primary_post,
        "supporting_blog_posts": ranked_posts[:3],
    }


def _evaluate_outbound_qualification(
    comparison_asset: dict[str, Any],
    *,
    require_display_safe_company: bool,
    require_primary_blog_post: bool,
    min_pain_categories: int,
) -> dict[str, Any]:
    pain_categories = [
        str(item).strip()
        for item in comparison_asset.get("pain_categories") or []
        if str(item).strip()
    ]
    checks = [
        ("display_safe_company", (not require_display_safe_company) or bool(comparison_asset.get("company_safe"))),
        ("pain_categories", len(pain_categories) >= min_pain_categories),
        ("alternative_vendor", bool(comparison_asset.get("alternative_vendor"))),
        ("primary_blog_post", (not require_primary_blog_post) or bool(comparison_asset.get("primary_blog_post"))),
    ]
    passed_checks = [name for name, passed in checks if passed]
    missing_checks = [name for name, passed in checks if not passed]
    return {
        "qualified": not missing_checks,
        "passed_checks": passed_checks,
        "missing_checks": missing_checks,
        "pain_category_count": len(pain_categories),
    }


async def _prepare_churning_company_context(
    pool,
    *,
    best: dict[str, Any],
    opps: list[dict[str, Any]],
    partner_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = settings.b2b_campaign
    base_context = _build_company_context(best, opps)
    base_context["opportunity_source"] = best.get("opportunity_source")

    try:
        from ...services.b2b.product_matching import match_products

        matches = await match_products(
            churning_from=best["vendor_name"],
            pain_categories=base_context["pain_categories"],
            company_size=best.get("seat_count"),
            industry=best.get("industry"),
            pool=pool,
            limit=3,
        )
        if matches:
            has_explicit_alternatives = bool(base_context.get("competitors_considering"))
            if best.get("opportunity_source") == "accounts_in_motion" and has_explicit_alternatives:
                base_context["supplemental_recommended_alternatives"] = matches
            else:
                base_context["recommended_alternatives"] = matches
    except Exception:
        logger.debug("Product matching unavailable, continuing without recommendations")

    pains = [
        item["category"]
        for item in base_context.get("pain_categories", [])
        if item.get("category")
    ]
    alternatives = [
        item.get("vendor_name")
        for item in base_context.get("recommended_alternatives", [])
        if isinstance(item, dict) and item.get("vendor_name")
    ]
    if not alternatives:
        alternatives = [
            item.get("name") or item
            for item in base_context.get("competitors_considering", [])
            if (item.get("name") if isinstance(item, dict) else item)
        ]

    blog_posts = await _fetch_blog_posts(
        pool,
        pipeline="b2b",
        vendor_name=base_context.get("churning_from"),
        category=base_context.get("category"),
        pain_categories=pains or None,
        alternative_vendors=alternatives[:3] or None,
        include_drafts=True,
    )
    comparison_asset = _build_comparison_asset(base_context, blog_posts)
    qualification = _evaluate_outbound_qualification(
        comparison_asset,
        require_display_safe_company=cfg.require_display_safe_company,
        require_primary_blog_post=cfg.require_primary_blog_post,
        min_pain_categories=cfg.min_pain_categories,
    )
    comparison_asset["qualified"] = qualification["qualified"]
    comparison_asset["qualification_reasons"] = qualification["passed_checks"]
    comparison_asset["missing_requirements"] = qualification["missing_checks"]
    comparison_asset["qualification_details"] = qualification
    base_context["comparison_asset"] = comparison_asset

    ordered_blog_posts = comparison_asset.get("supporting_blog_posts") or blog_posts
    primary_blog_post = comparison_asset.get("primary_blog_post")
    partner = _match_partner(base_context, partner_index) if partner_index else None
    return {
        "base_context": base_context,
        "qualification": qualification,
        "primary_blog_post": primary_blog_post,
        "ordered_blog_posts": ordered_blog_posts,
        "partner": partner,
    }


from ._blog_matching import fetch_relevant_blog_posts as _fetch_blog_posts


# ---------------------------------------------------------------------------
# Calibration weight cache
# ---------------------------------------------------------------------------

import time as _time

_calibration_cache: dict[str, dict[str, float]] = {}
_calibration_cache_ts: float = 0.0
_CALIBRATION_CACHE_TTL = 3600  # 1 hour


def _get_calibration_adjustments() -> dict[str, dict[str, float]]:
    """Return cached calibration adjustments {dimension: {value: adjustment}}.

    Returns empty dict if cache is stale or not loaded (caller uses static defaults).
    Cache is populated by _load_calibration_weights() which must be awaited separately.
    """
    global _calibration_cache, _calibration_cache_ts
    if _time.monotonic() - _calibration_cache_ts > _CALIBRATION_CACHE_TTL:
        return {}
    return _calibration_cache


async def load_calibration_weights() -> bool:
    """Load latest calibration weights from DB into module cache.

    Called at the start of generate_campaigns(). Returns True if weights were loaded.
    """
    global _calibration_cache, _calibration_cache_ts
    try:
        pool = get_db_pool()
        if not pool.is_initialized:
            return False

        latest_version = await pool.fetchval(
            "SELECT MAX(model_version) FROM score_calibration_weights"
        )
        if latest_version is None:
            return False

        rows = await pool.fetch(
            """
            SELECT dimension, dimension_value, weight_adjustment
            FROM score_calibration_weights
            WHERE model_version = $1
            """,
            latest_version,
        )

        cache: dict[str, dict[str, float]] = {}
        for r in rows:
            dim = r["dimension"]
            if dim not in cache:
                cache[dim] = {}
            cache[dim][r["dimension_value"]] = float(r["weight_adjustment"])

        _calibration_cache = cache
        _calibration_cache_ts = _time.monotonic()
        return True
    except Exception:
        logger.debug("Could not load calibration weights", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _compute_score(row: dict) -> tuple[int, dict]:
    """Compute opportunity score (0-100) from enrichment signals.

    Blends static defaults with calibration adjustments when available.
    Calibration adjustments are additive point shifts derived from observed
    outcome conversion rates (see b2b_score_calibration.py).

    Returns (final_score, components_dict) so callers can persist the breakdown.
    """
    cal = _get_calibration_adjustments()
    score = 0.0

    # Urgency component (max 30 pts)
    urgency = _safe_float(row.get("urgency"), 0)
    urgency_pts = max(0, min(30, (urgency - 5) * 6))
    if cal.get("urgency_bucket"):
        bucket = "high" if urgency >= 8 else ("medium" if urgency >= 5 else "low")
        urgency_pts += cal["urgency_bucket"].get(bucket, 0)
    urgency_final = max(0, min(30, urgency_pts))
    score += urgency_final

    # Role component (max ~20 pts)
    role_pts = 0.0
    role_label = None
    if row.get("is_dm"):
        role_pts = 20
        role_label = "decision_maker"
        if cal.get("role_type"):
            role_pts += cal["role_type"].get("decision_maker", 0)
    elif row.get("role_type") in _ROLE_SCORES:
        role_label = row["role_type"]
        role_pts = _ROLE_SCORES[role_label]
        if cal.get("role_type"):
            role_pts += cal["role_type"].get(role_label, 0)
    role_final = max(0, role_pts)
    score += role_final

    # Buying stage component (max ~25 pts)
    buying_stage = row.get("buying_stage") or ""
    stage_pts = _STAGE_SCORES.get(buying_stage, 0)
    if cal.get("buying_stage"):
        stage_pts += cal["buying_stage"].get(buying_stage, 0)
    stage_final = max(0, stage_pts)
    score += stage_final

    # Seat count component (max ~15 pts)
    seat_count = row.get("seat_count")
    seat_pts = 0.0
    seat_bucket = None
    if seat_count is not None:
        if seat_count >= 500:
            seat_pts = 15
            seat_bucket = "500+"
        elif seat_count >= 100:
            seat_pts = 10
            seat_bucket = "100-499"
        elif seat_count >= 20:
            seat_pts = 5
            seat_bucket = "20-99"
        else:
            seat_bucket = "small"
        if cal.get("seat_bucket"):
            seat_pts += cal["seat_bucket"].get(seat_bucket, 0)
    seat_final = max(0, seat_pts)
    score += seat_final

    # Mention context component (max ~10 pts)
    mention_context = (row.get("mention_context") or "").lower()
    context_final = 0.0
    context_keyword = None
    for keyword, pts in _CONTEXT_SCORES.items():
        if keyword in mention_context:
            context_pts = pts
            if cal.get("context_keyword"):
                context_pts += cal["context_keyword"].get(keyword, 0)
            context_final = max(0, context_pts)
            context_keyword = keyword
            break
    score += context_final

    final = int(min(100, max(0, score)))
    components = {
        "urgency": {"pts": round(urgency_final, 1), "raw": round(urgency, 1), "max": 30},
        "role": {"pts": round(role_final, 1), "label": role_label, "max": 20},
        "stage": {"pts": round(stage_final, 1), "label": buying_stage or None, "max": 25},
        "seats": {"pts": round(seat_final, 1), "bucket": seat_bucket, "max": 15},
        "context": {"pts": round(context_final, 1), "keyword": context_keyword, "max": 10},
    }
    return final, components


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate ABM campaigns from churn intelligence."""
    cfg = settings.b2b_campaign
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B campaign generation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    result = await generate_campaigns(
        pool=pool,
        min_score=cfg.min_opportunity_score,
        limit=cfg.max_campaigns_per_run,
        target_mode=cfg.target_mode,
        run_id=_task_run_id(task),
    )

    # Send notification
    if result.get("generated", 0) > 0:
        from ...pipelines.notify import send_pipeline_notification

        mode_label = result.get("target_mode", cfg.target_mode).replace("_", " ").title()
        msg = (
            f"[{mode_label}] Generated {result['generated']} campaign(s) for "
            f"{result['companies']} company/companies. "
            f"Review drafts in the lead engagement pipeline."
        )
        await send_pipeline_notification(
            msg, task, title="Atlas: ABM Campaigns",
            default_tags="briefcase,campaign",
        )

    return {"_skip_synthesis": "Campaign generation complete", **result}


async def _create_sequence_for_cold_email(
    pool,
    *,
    company_name: str,
    batch_id: str,
    partner_id: str | None,
    context: dict[str, Any],
    cold_email_subject: str,
    cold_email_body: str,
) -> _uuid.UUID | None:
    """Create a campaign_sequences row and link the cold email to it.

    Returns the sequence ID if created, None on conflict (already exists).
    """
    cfg = settings.campaign_sequence
    compact_company_context, compact_selling_context = prepare_sequence_storage_contexts(
        context,
        context.get("selling", {}),
    )

    seq_id = await pool.fetchval(
        """
        INSERT INTO campaign_sequences (
            company_name, batch_id, partner_id,
            company_context, selling_context, max_steps
        ) VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT ((LOWER(company_name)), batch_id) DO NOTHING
        RETURNING id
        """,
        company_name,
        batch_id,
        _uuid.UUID(partner_id) if partner_id else None,
        json.dumps(compact_company_context, default=str),
        json.dumps(compact_selling_context, default=str),
        cfg.max_steps,
    )

    if not seq_id:
        logger.debug("Sequence already exists for %s / %s", company_name, batch_id)
        return None

    # Link the cold email campaign row to this sequence
    await pool.execute(
        """
        UPDATE b2b_campaigns
        SET sequence_id = $1, step_number = 1
        WHERE company_name = $2 AND batch_id = $3 AND channel = 'email_cold'
        """,
        seq_id,
        company_name,
        batch_id,
    )

    # Audit log
    await log_campaign_event(
        pool,
        event_type="generated",
        sequence_id=seq_id,
        step_number=1,
        source="system",
        subject=cold_email_subject,
        body=cold_email_body,
    )

    # Best-effort CRM recipient lookup (only for churning_company mode where
    # company_name is a person/company name; vendor/challenger modes set
    # recipient_email from the target contact directly after this function).
    # Skip when target_persona is set -- persona sequences rely on the
    # prospect_matching task for differentiated recipient assignment.
    has_persona = context.get("target_persona") is not None
    if partner_id and not has_persona:
        try:
            contact_email = await pool.fetchval(
                """
                SELECT email FROM contacts
                WHERE LOWER(full_name) LIKE '%' || LOWER($1) || '%'
                  AND email IS NOT NULL
                ORDER BY created_at DESC LIMIT 1
                """,
                company_name,
            )
            if contact_email:
                await pool.execute(
                    "UPDATE campaign_sequences SET recipient_email = $1 WHERE id = $2",
                    contact_email,
                    seq_id,
                )
                logger.info(
                    "Auto-populated recipient %s for sequence %s (%s)",
                    contact_email, seq_id, company_name,
                )
        except Exception:
            logger.warning("CRM recipient lookup failed for %s, skipping", company_name)

    logger.info("Created campaign sequence %s for %s (batch %s)", seq_id, company_name, batch_id)
    return seq_id


async def generate_campaigns(
    pool,
    min_score: int = 70,
    limit: int = 20,
    vendor_filter: str | None = None,
    company_filter: str | None = None,
    target_mode: str | None = None,
    force: bool = False,
    ignore_recent_dedup: bool = False,
    ignore_briefing_gate: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Core generation logic, shared by autonomous task and manual API trigger.

    Dispatches to the appropriate generation path based on target_mode:
      - churning_company: original behavior (outreach to the churning company)
      - vendor_retention: sell churn intelligence to the vendor losing customers
      - challenger_intel: sell intent leads to the challenger gaining customers
    """
    cfg = settings.b2b_campaign
    mode = target_mode or cfg.target_mode
    bypass_recent_dedup = force or ignore_recent_dedup
    bypass_briefing_gate = force or ignore_briefing_gate

    # Load calibration weights into cache (best-effort, falls back to static defaults)
    await load_calibration_weights()

    if mode == "vendor_retention":
        return await _generate_vendor_campaigns(
            pool, min_score, limit, vendor_filter, company_filter,
            bypass_briefing_gate=bypass_briefing_gate,
            bypass_recent_dedup=bypass_recent_dedup,
            run_id=run_id,
        )
    elif mode == "challenger_intel":
        return await _generate_challenger_campaigns(
            pool, min_score, limit, vendor_filter, company_filter,
            bypass_briefing_gate=bypass_briefing_gate,
            bypass_recent_dedup=bypass_recent_dedup,
            run_id=run_id,
        )

    # Default: churning_company (original behavior)
    return await _generate_churning_company_campaigns(
        pool, min_score, limit, vendor_filter, company_filter,
        bypass_recent_dedup=bypass_recent_dedup,
        run_id=run_id,
    )


async def list_churning_company_review_candidates(
    pool,
    *,
    min_score: int,
    max_score: int,
    limit: int,
    vendor_filter: str | None = None,
    company_filter: str | None = None,
    qualified_only: bool = True,
    ignore_recent_dedup: bool = False,
) -> dict[str, Any]:
    cfg = settings.b2b_campaign
    fetched = await _fetch_accounts_in_motion_opportunities(
        pool,
        min_score=min_score,
        limit=min(max(limit * 5, limit), 500),
        vendor_filter=vendor_filter,
        company_filter=company_filter,
    )
    by_company: dict[str, list[dict[str, Any]]] = {}
    for row in fetched:
        score = int(row.get("opportunity_score") or 0)
        if score > max_score:
            continue
        company = str(row.get("reviewer_company") or "").strip()
        if not company:
            continue
        by_company.setdefault(company.lower(), []).append(row)

    partner_index = await _fetch_affiliate_partners(pool)
    candidates: list[dict[str, Any]] = []
    missing_requirements: dict[str, int] = {}
    dedup_blocked = 0
    evaluated = 0

    for company_key, opps in by_company.items():
        company_name = opps[0].get("reviewer_company") or opps[0]["vendor_name"]
        recent_campaign_count = 0
        if not ignore_recent_dedup:
            recent_campaign_count = int(await pool.fetchval(
                """
                SELECT COUNT(*) FROM b2b_campaigns
                WHERE LOWER(company_name) = $1
                  AND created_at > NOW() - make_interval(days => $2)
                """,
                company_key, cfg.dedup_days,
            ) or 0)
        if recent_campaign_count > 0 and qualified_only:
            dedup_blocked += 1
            continue

        best = max(opps, key=lambda item: item["opportunity_score"])
        prepared = await _prepare_churning_company_context(
            pool,
            best=best,
            opps=opps,
            partner_index=partner_index,
        )
        evaluated += 1
        qualification = prepared["qualification"]
        for reason in qualification["missing_checks"]:
            missing_requirements[reason] = missing_requirements.get(reason, 0) + 1
        if qualified_only and not qualification["qualified"]:
            continue

        base_context = prepared["base_context"]
        partner = prepared["partner"]
        comparison_asset = base_context.get("comparison_asset") or {}
        candidate = {
            "company_name": company_name,
            "vendor_name": best["vendor_name"],
            "product_category": best.get("product_category"),
            "opportunity_score": best["opportunity_score"],
            "urgency_score": float(best.get("urgency") or 0),
            "role_type": base_context.get("role_type"),
            "reviewer_title": base_context.get("reviewer_title"),
            "industry": base_context.get("industry"),
            "company_size": base_context.get("company_size"),
            "seat_count": best.get("seat_count"),
            "buying_stage": best.get("buying_stage"),
            "pain_categories": base_context.get("pain_categories") or [],
            "key_quotes": base_context.get("key_quotes") or [],
            "competitors_considering": base_context.get("competitors_considering") or [],
            "reasoning_anchor_examples": base_context.get("reasoning_anchor_examples") or {},
            "reasoning_witness_highlights": base_context.get("reasoning_witness_highlights") or [],
            "reasoning_reference_ids": base_context.get("reasoning_reference_ids") or {},
            "comparison_asset": comparison_asset,
            "primary_blog_post": prepared["primary_blog_post"],
            "supporting_blog_posts": prepared["ordered_blog_posts"],
            "partner": (
                {
                    "id": partner["id"],
                    "product_name": partner.get("product_name"),
                    "affiliate_url": partner.get("affiliate_url"),
                }
                if partner else None
            ),
            "qualification": qualification,
            "recent_campaign_count": recent_campaign_count,
            "dedup_blocked": recent_campaign_count > 0,
            "opportunity_source": best.get("opportunity_source") or "accounts_in_motion",
            "generate_request": {
                "target_mode": "churning_company",
                "company_name": company_name,
                "vendor_name": best["vendor_name"],
                "min_score": best["opportunity_score"],
                "limit": 1,
                "force": False,
            },
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (item["opportunity_score"], item["urgency_score"]),
        reverse=True,
    )
    visible = candidates[:limit]
    return {
        "count": len(visible),
        "candidates": visible,
        "summary": {
            "min_score": min_score,
            "max_score": max_score,
            "qualified_only": qualified_only,
            "ignore_recent_dedup": ignore_recent_dedup,
            "evaluated": evaluated,
            "qualified": sum(1 for item in candidates if item["qualification"]["qualified"]),
            "unqualified": max(evaluated - sum(1 for item in candidates if item["qualification"]["qualified"]), 0),
            "dedup_blocked": dedup_blocked,
            "missing_requirements": missing_requirements,
        },
    }


async def _generate_churning_company_campaigns(
    pool,
    min_score: int,
    limit: int,
    vendor_filter: str | None,
    company_filter: str | None,
    bypass_recent_dedup: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Original generation path: outreach to the churning company."""
    cfg = settings.b2b_campaign

    # 1. Prefer named account opportunities, then fall back to review rows.
    opportunities = await _fetch_accounts_in_motion_opportunities(
        pool, min_score, limit,
        vendor_filter=vendor_filter,
        company_filter=company_filter,
    )
    opportunity_source = "accounts_in_motion"
    if not opportunities:
        opportunities = await _fetch_opportunities(
            pool, min_score, limit,
            vendor_filter=vendor_filter,
            company_filter=company_filter,
            dm_only=cfg.require_decision_maker,
        )
        opportunity_source = "reviews"

    if not opportunities:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0}

    # 2. Group by company (one campaign set per company)
    #    Skip opportunities with no reviewer_company -- we need a real target
    by_company: dict[str, list[dict]] = {}
    skipped_no_company = 0
    for opp in opportunities:
        company = opp.get("reviewer_company")
        if not company:
            skipped_no_company += 1
            continue
        key = company.lower()
        if key not in by_company:
            by_company[key] = []
        by_company[key].append(opp)
    if not by_company:
        return {
            "generated": 0,
            "skipped": skipped_no_company,
            "failed": 0,
            "companies": 0,
            "candidate_companies": 0,
            "opportunity_source": opportunity_source,
        }

    # 3. Dedup: skip companies with recent campaigns
    companies_to_process = []
    for company_key, opps in by_company.items():
        company_name = opps[0].get("reviewer_company") or opps[0]["vendor_name"]
        if bypass_recent_dedup:
            companies_to_process.append((company_name, opps))
            continue
        existing = await pool.fetchval(
            """
            SELECT COUNT(*) FROM b2b_campaigns
            WHERE LOWER(company_name) = $1
              AND created_at > NOW() - make_interval(days => $2)
            """,
            company_key, cfg.dedup_days,
        )
        if existing == 0:
            companies_to_process.append((company_name, opps))

    if not companies_to_process:
        return {"generated": 0, "skipped": len(by_company), "failed": 0, "companies": 0}

    # 4. Get LLM
    from ...services.llm_router import get_llm
    llm = get_llm("campaign")
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0, "error": "No LLM available"}

    # 5. Load skill prompt
    from ...skills import get_skill_registry
    from ...services.protocols import Message

    skill = get_skill_registry().get("digest/b2b_campaign_generation")
    if not skill:
        logger.warning("Skill 'digest/b2b_campaign_generation' not found")
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0, "error": "Skill not found"}

    llm_model_name = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
    claimer = f"reconcile:{run_id or 'adhoc'}:{_uuid.uuid4().hex[:10]}"
    batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    generated = 0
    failed = 0
    deferred = 0
    skipped_unqualified = 0
    generated_without_partner = 0
    sequences_created = 0
    qualified_companies = 0
    qualification_summary: dict[str, int] = {}

    # Fetch affiliate partners for sender identity matching
    partner_index = await _fetch_affiliate_partners(pool)

    personas_skipped = 0
    batch_metrics = {
        "jobs": 0,
        "submitted_items": 0,
        "cache_prefiltered_items": 0,
        "fallback_single_call_items": 0,
        "completed_items": 0,
        "failed_items": 0,
    }

    # --- Pre-qualification pass (sequential, fast) ---
    qualified_units: list[dict[str, Any]] = []
    for company_name, opps in companies_to_process:
        best = max(opps, key=lambda o: o["opportunity_score"])
        prepared = await _prepare_churning_company_context(
            pool, best=best, opps=opps, partner_index=partner_index,
        )
        base_context = prepared["base_context"]
        qualification = prepared["qualification"]
        primary_blog_post = prepared["primary_blog_post"]
        ordered_blog_posts = prepared["ordered_blog_posts"]
        if not qualification["qualified"]:
            skipped_unqualified += 1
            for reason in qualification["missing_checks"]:
                qualification_summary[reason] = qualification_summary.get(reason, 0) + 1
            continue
        qualified_companies += 1
        partner = prepared["partner"]
        partner_id: str | None = None
        if partner:
            base_context["selling"] = _build_selling_context(
                sender_name=cfg.default_sender_name, sender_title=cfg.default_sender_title,
                sender_company=cfg.default_sender_company,
                product_name=partner["product_name"], affiliate_url=partner["affiliate_url"],
                primary_blog_post=primary_blog_post, blog_posts=ordered_blog_posts,
            )
            partner_id = partner["id"]
        else:
            generated_without_partner += 1
            base_context["selling"] = _build_selling_context(
                sender_name=cfg.default_sender_name, sender_title=cfg.default_sender_title,
                sender_company=cfg.default_sender_company,
                primary_blog_post=primary_blog_post, blog_posts=ordered_blog_posts,
            )
        for persona in cfg.personas:
            persona_context = _build_persona_context(base_context, persona)
            if persona_context is None:
                personas_skipped += 1
                continue
            qualified_units.append({
                "company_name": company_name, "opps": opps, "best": best,
                "persona": persona, "persona_context": persona_context,
                "persona_batch_id": f"{batch_id}_{persona}", "partner_id": partner_id,
            })

    phase_one_entries: list[dict[str, Any]] = []
    phase_one_channels = [channel for channel in cfg.channels if channel != "email_followup"]
    for unit in qualified_units:
        for channel in phase_one_channels:
            payload = {
                **unit["persona_context"],
                "channel": channel,
                "target_mode": "churning_company",
            }
            artifact_id = _campaign_artifact_key(
                company_name=unit["company_name"],
                batch_id=unit["persona_batch_id"],
                channel=channel,
            )
            phase_one_entries.append(
                {
                    "custom_id": artifact_id,
                    "artifact_id": artifact_id,
                    "campaign_batch_id": unit["persona_batch_id"],
                    "phase": "cold",
                    "payload": payload,
                    "channel": channel,
                    "company_name": unit["company_name"],
                    "persona": unit["persona"],
                    "persona_batch_id": unit["persona_batch_id"],
                    "persona_context": unit["persona_context"],
                    "best": unit["best"],
                    "review_ids": [o["review_id"] for o in unit["opps"] if o.get("review_id")][:20] or None,
                    "partner_id": unit["partner_id"],
                    "max_tokens": cfg.max_tokens,
                    "temperature": cfg.temperature,
                    "trace_metadata": _campaign_trace_metadata(
                        payload,
                        run_id=run_id,
                        stage_id="b2b_campaign_generation.content",
                    ),
                }
            )

    phase_one_results, phase_one_batch = await _run_campaign_batch(
        llm,
        skill.content,
        phase_one_entries,
        run_id=run_id,
    )
    _merge_batch_metrics(batch_metrics, phase_one_batch)

    cold_email_by_unit: dict[tuple[str, str], dict[str, str]] = {}
    for entry in phase_one_entries:
        company_name = entry["company_name"]
        persona = entry["persona"]
        channel = entry["channel"]
        payload = entry["payload"]
        artifact_id = entry["artifact_id"]
        content = phase_one_results.get(entry["custom_id"])
        if _is_deferred_campaign_content(content):
            await record_attempt(
                pool,
                artifact_type="campaign",
                artifact_id=artifact_id,
                attempt_no=1,
                stage="generation",
                status="queued",
            )
            deferred += 1
            continue
        if content:
            metadata = _campaign_storage_metadata(payload)
            generation_audit = (
                payload.get("_generation_audit")
                if isinstance(payload.get("_generation_audit"), dict)
                else {}
            )
            specificity = (
                generation_audit.get("specificity")
                if isinstance(generation_audit.get("specificity"), dict)
                else {}
            )
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_campaigns (
                        company_name, vendor_name, product_category,
                        opportunity_score, urgency_score, pain_categories,
                        competitors_considering, seat_count, contract_end,
                        decision_timeline, buying_stage, role_type,
                        key_quotes, source_review_ids,
                        channel, subject, body, cta,
                        status, batch_id, llm_model,
                        partner_id, industry, target_mode, metadata,
                        score_components
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20, $21, $22, $23, $24, $25::jsonb,
                        $26::jsonb
                    )
                    """,
                    company_name,
                    entry["best"]["vendor_name"],
                    entry["best"].get("product_category"),
                    entry["best"]["opportunity_score"],
                    entry["best"].get("urgency"),
                    json.dumps(entry["persona_context"].get("pain_categories", [])),
                    json.dumps(entry["persona_context"].get("competitors_considering", [])),
                    entry["best"].get("seat_count"),
                    entry["best"].get("contract_end"),
                    entry["best"].get("decision_timeline"),
                    entry["best"].get("buying_stage"),
                    entry["persona_context"].get("role_type"),
                    json.dumps(entry["persona_context"].get("key_quotes", [])),
                    entry["review_ids"],
                    channel,
                    content.get("subject", ""),
                    content.get("body", ""),
                    content.get("cta", ""),
                    "draft",
                    entry["persona_batch_id"],
                    llm_model_name,
                    _uuid.UUID(entry["partner_id"]) if entry["partner_id"] else None,
                    entry["persona_context"].get("industry"),
                    "churning_company",
                    json.dumps(metadata, default=str),
                    json.dumps(entry["best"].get("score_components")),
                )
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="generation",
                    status="succeeded",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    warnings=list(specificity.get("warnings") or []),
                )
                generated += 1
                if channel == "email_cold":
                    cold_email_by_unit[(company_name, persona)] = {
                        "subject": content.get("subject", ""),
                        "body": content.get("body", ""),
                    }
            except Exception:
                logger.exception("Failed to store campaign for %s/%s/%s", company_name, persona, channel)
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="storage",
                    status="failed",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    blocking_issues=list(specificity.get("blocking_issues") or []),
                    warnings=list(specificity.get("warnings") or []),
                    failure_step="storage",
                    error_message="campaign_storage_failed",
                )
                failed += 1
        else:
            await _record_campaign_generation_failure(
                pool,
                artifact_id=artifact_id,
                company_name=company_name,
                channel=channel,
                generation_audit=payload.get("_generation_audit"),
            )
            failed += 1

    if settings.campaign_sequence.enabled:
        for unit in qualified_units:
            cold = cold_email_by_unit.get((unit["company_name"], unit["persona"]))
            if not cold:
                continue
            try:
                seq_id = await _create_sequence_for_cold_email(
                    pool,
                    company_name=unit["company_name"],
                    batch_id=unit["persona_batch_id"],
                    partner_id=unit["partner_id"],
                    context=unit["persona_context"],
                    cold_email_subject=cold.get("subject", ""),
                    cold_email_body=cold.get("body", ""),
                )
                if seq_id:
                    sequences_created += 1
            except Exception as exc:
                logger.warning(
                    "Failed to create sequence for %s/%s: %s",
                    unit["company_name"],
                    unit["persona"],
                    exc,
                )

    followup_entries: list[dict[str, Any]] = []
    if "email_followup" in cfg.channels:
        for unit in qualified_units:
            cold_context = cold_email_by_unit.get((unit["company_name"], unit["persona"]))
            if not cold_context:
                continue
            payload = {
                **unit["persona_context"],
                "channel": "email_followup",
                "target_mode": "churning_company",
                "cold_email_context": cold_context,
            }
            artifact_id = _campaign_artifact_key(
                company_name=unit["company_name"],
                batch_id=unit["persona_batch_id"],
                channel="email_followup",
            )
            followup_entries.append(
                {
                    "custom_id": artifact_id,
                    "artifact_id": artifact_id,
                    "campaign_batch_id": unit["persona_batch_id"],
                    "phase": "followup",
                    "payload": payload,
                    "channel": "email_followup",
                    "company_name": unit["company_name"],
                    "persona": unit["persona"],
                    "persona_batch_id": unit["persona_batch_id"],
                    "persona_context": unit["persona_context"],
                    "best": unit["best"],
                    "review_ids": [o["review_id"] for o in unit["opps"] if o.get("review_id")][:20] or None,
                    "partner_id": unit["partner_id"],
                    "max_tokens": cfg.max_tokens,
                    "temperature": cfg.temperature,
                    "trace_metadata": _campaign_trace_metadata(
                        payload,
                        run_id=run_id,
                        stage_id="b2b_campaign_generation.content",
                    ),
                }
            )

    followup_results, phase_two_batch = await _run_campaign_batch(
        llm,
        skill.content,
        followup_entries,
        run_id=run_id,
    )
    _merge_batch_metrics(batch_metrics, phase_two_batch)

    for entry in followup_entries:
        company_name = entry["company_name"]
        persona = entry["persona"]
        payload = entry["payload"]
        artifact_id = entry["artifact_id"]
        content = followup_results.get(entry["custom_id"])
        if _is_deferred_campaign_content(content):
            await record_attempt(
                pool,
                artifact_type="campaign",
                artifact_id=artifact_id,
                attempt_no=1,
                stage="generation",
                status="queued",
            )
            deferred += 1
            continue
        if content:
            metadata = _campaign_storage_metadata(payload)
            generation_audit = (
                payload.get("_generation_audit")
                if isinstance(payload.get("_generation_audit"), dict)
                else {}
            )
            specificity = (
                generation_audit.get("specificity")
                if isinstance(generation_audit.get("specificity"), dict)
                else {}
            )
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_campaigns (
                        company_name, vendor_name, product_category,
                        opportunity_score, urgency_score, pain_categories,
                        competitors_considering, seat_count, contract_end,
                        decision_timeline, buying_stage, role_type,
                        key_quotes, source_review_ids,
                        channel, subject, body, cta,
                        status, batch_id, llm_model,
                        partner_id, industry, target_mode, metadata,
                        score_components
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20, $21, $22, $23, $24, $25::jsonb,
                        $26::jsonb
                    )
                    """,
                    company_name,
                    entry["best"]["vendor_name"],
                    entry["best"].get("product_category"),
                    entry["best"]["opportunity_score"],
                    entry["best"].get("urgency"),
                    json.dumps(entry["persona_context"].get("pain_categories", [])),
                    json.dumps(entry["persona_context"].get("competitors_considering", [])),
                    entry["best"].get("seat_count"),
                    entry["best"].get("contract_end"),
                    entry["best"].get("decision_timeline"),
                    entry["best"].get("buying_stage"),
                    entry["persona_context"].get("role_type"),
                    json.dumps(entry["persona_context"].get("key_quotes", [])),
                    entry["review_ids"],
                    "email_followup",
                    content.get("subject", ""),
                    content.get("body", ""),
                    content.get("cta", ""),
                    "draft",
                    entry["persona_batch_id"],
                    llm_model_name,
                    _uuid.UUID(entry["partner_id"]) if entry["partner_id"] else None,
                    entry["persona_context"].get("industry"),
                    "churning_company",
                    json.dumps(metadata, default=str),
                    json.dumps(entry["best"].get("score_components")),
                )
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="generation",
                    status="succeeded",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    warnings=list(specificity.get("warnings") or []),
                )
                generated += 1
            except Exception:
                logger.exception(
                    "Failed to store campaign for %s/%s/%s",
                    company_name,
                    persona,
                    "email_followup",
                )
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="storage",
                    status="failed",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    blocking_issues=list(specificity.get("blocking_issues") or []),
                    warnings=list(specificity.get("warnings") or []),
                    failure_step="storage",
                    error_message="campaign_storage_failed",
                )
                failed += 1
        else:
            await _record_campaign_generation_failure(
                pool,
                artifact_id=artifact_id,
                company_name=company_name,
                channel="email_followup",
                generation_audit=payload.get("_generation_audit"),
            )
            failed += 1

    logger.info(
        "Campaign generation (churning_company): %d generated, %d failed, %d generated without partner, "
        "%d personas skipped, %d sequences from %d companies",
        generated, failed, generated_without_partner, personas_skipped, sequences_created, len(companies_to_process),
    )

    return {
        "generated": generated,
        "failed": failed,
        "skipped": (len(by_company) - len(companies_to_process)) + skipped_no_company + skipped_unqualified,
        "skipped_dedup": len(by_company) - len(companies_to_process),
        "skipped_no_company": skipped_no_company,
        "skipped_unqualified": skipped_unqualified,
        "skipped_no_partner": 0,
        "generated_without_partner": generated_without_partner,
        "deferred": deferred,
        "personas_skipped": personas_skipped,
        "sequences_created": sequences_created,
        "companies": qualified_companies,
        "candidate_companies": len(companies_to_process),
        "qualification_summary": qualification_summary,
        "batch_id": batch_id,
        "target_mode": "churning_company",
        "opportunity_source": opportunity_source,
        "anthropic_batch_jobs": batch_metrics["jobs"],
        "anthropic_batch_items_submitted": batch_metrics["submitted_items"],
        "anthropic_batch_cache_prefiltered": batch_metrics["cache_prefiltered_items"],
        "anthropic_batch_fallback_single_call": batch_metrics["fallback_single_call_items"],
        "anthropic_batch_completed_items": batch_metrics["completed_items"],
        "anthropic_batch_failed_items": batch_metrics["failed_items"],
    }


# ------------------------------------------------------------------
# Vendor retention campaign generation (P1)
# ------------------------------------------------------------------


async def _fetch_vendor_targets(pool, vendor_name: str | None = None) -> list[dict]:
    """Fetch active vendor targets, optionally filtered by vendor name."""
    if vendor_name:
        rows = await pool.fetch(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier, status,
                   notes, account_id, created_at, updated_at
            FROM vendor_targets
            WHERE status = 'active' AND target_mode = 'vendor_retention'
              AND company_name ILIKE '%' || $1 || '%'
            """,
            vendor_name,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier, status,
                   notes, account_id, created_at, updated_at
            FROM vendor_targets
            WHERE status = 'active' AND target_mode = 'vendor_retention'
            """
        )
    return dedupe_vendor_target_rows(rows)


def _build_vendor_context(vendor_name: str, signals: list[dict]) -> dict[str, Any]:
    """Aggregate churn signals into a vendor-scoped intelligence summary."""
    ordered_signals = _campaign_stable_row_order(signals)
    total = len(ordered_signals)
    high_urgency = sum(1 for s in ordered_signals if _safe_float(s.get("urgency"), 0) >= 8)
    medium_urgency = sum(1 for s in ordered_signals if 5 <= _safe_float(s.get("urgency"), 0) < 8)

    # Pain distribution
    pain_counts: dict[str, int] = {}
    for s in ordered_signals:
        pain = _parse_json_field(s.get("pain_json"))
        for p in pain:
            if isinstance(p, dict) and p.get("category"):
                pain_counts[p["category"]] = pain_counts.get(p["category"], 0) + 1

    # Competitor distribution (who they're losing to)
    comp_counts: dict[str, int] = {}
    for s in ordered_signals:
        comps = s.get("competitors", [])
        for c in comps:
            if isinstance(c, dict) and c.get("name"):
                name = c["name"]
                comp_counts[name] = comp_counts.get(name, 0) + 1

    # Feature gaps
    gap_counts: dict[str, int] = {}
    for s in ordered_signals:
        gaps = _parse_json_field(s.get("feature_gaps"))
        for g in gaps:
            label = g if isinstance(g, str) else (g.get("feature", "") if isinstance(g, dict) else "")
            if label:
                gap_counts[label] = gap_counts.get(label, 0) + 1

    # Quotable phrases from enrichment
    quote_list: list[str] = []
    for s in ordered_signals:
        phrases = _parse_json_field(s.get("quotable_phrases"))
        for phrase in phrases:
            text = phrase if isinstance(phrase, str) else (
                phrase.get("text", "") if isinstance(phrase, dict) else ""
            )
            if text and text not in quote_list:
                quote_list.append(text)

    # Timeline signals
    timeline_count = sum(1 for s in signals if s.get("contract_end"))

    return {
        "vendor_name": vendor_name,
        "key_quotes": quote_list[:8],
        "signal_summary": {
            "total_signals": total,
            "high_urgency_count": high_urgency,
            "medium_urgency_count": medium_urgency,
            "pain_distribution": _campaign_sorted_count_rows(
                pain_counts,
                field_name="category",
                limit=10,
            ),
            "competitor_distribution": _campaign_sorted_count_rows(
                comp_counts,
                field_name="name",
                limit=10,
            ),
            "feature_gaps": _campaign_sorted_count_keys(gap_counts, limit=10),
            "timeline_signals": timeline_count,
            "trend_vs_last_month": None,  # overridden by caller with _compute_vendor_trend()
        },
    }


async def _compute_vendor_trend(
    pool,
    vendor_name: str,
    products: list[str] | None = None,
) -> str | None:
    """Compare signal count in last 30d vs previous 30d for a vendor.

    Returns 'increasing', 'stable', 'decreasing', or None on error.
    """
    # APPROVED-ENRICHMENT-READ: urgency_score
    # Reason: COUNT-only aggregation with urgency threshold, not a row-level extraction
    try:
        # Build vendor name match condition
        names = [vendor_name]
        if products:
            names.extend(products)
        name_conditions = " OR ".join(
            f"r.vendor_name ILIKE '%' || ${i + 1} || '%'" for i in range(len(names))
        )
        base_idx = len(names) + 1

        current = await pool.fetchval(
            f"""
            SELECT COUNT(*) FROM b2b_reviews r
            WHERE r.enrichment_status = 'enriched'
              AND r.enriched_at > NOW() - INTERVAL '30 days'
              AND (r.enrichment->>'urgency_score')::numeric >= ${base_idx}
              AND ({name_conditions})
            """,
            *names, 3.0,
        )
        previous = await pool.fetchval(
            f"""
            SELECT COUNT(*) FROM b2b_reviews r
            WHERE r.enrichment_status = 'enriched'
              AND r.enriched_at BETWEEN NOW() - INTERVAL '60 days' AND NOW() - INTERVAL '30 days'
              AND (r.enrichment->>'urgency_score')::numeric >= ${base_idx}
              AND ({name_conditions})
            """,
            *names, 3.0,
        )
        if previous == 0 and current == 0:
            return None
        if previous == 0:
            return "increasing"
        ratio = current / previous
        if ratio >= 1.2:
            return "increasing"
        elif ratio <= 0.8:
            return "decreasing"
        return "stable"
    except Exception:
        logger.debug("Failed to compute trend for %s", vendor_name)
        return None


_CONTACT_ROLE_MAP: dict[str, str] = {
    "vp customer success": "economic_buyer",
    "head of customer success": "economic_buyer",
    "vp cs": "economic_buyer",
    "chief customer officer": "economic_buyer",
    "head of product": "decision_maker",
    "vp product": "decision_maker",
    "cpo": "decision_maker",
    "vp sales": "economic_buyer",
    "head of sales": "economic_buyer",
    "cro": "economic_buyer",
    "head of competitive intelligence": "evaluator",
    "vp marketing": "decision_maker",
}


def _map_role_type(contact_role: str | None) -> str:
    """Map a raw contact title to an enum role_type value."""
    if not contact_role:
        return "economic_buyer"
    return _CONTACT_ROLE_MAP.get(contact_role.lower(), "economic_buyer")


async def _generate_vendor_campaigns(
    pool,
    min_score: int,
    limit: int,
    vendor_filter: str | None,
    company_filter: str | None = None,
    bypass_briefing_gate: bool = False,
    bypass_recent_dedup: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Generate campaigns targeting vendor CS/Product leaders with churn intelligence."""
    cfg = settings.b2b_campaign

    # 1. Fetch vendor targets (our customers)
    targets = await _fetch_vendor_targets(pool, vendor_filter)
    if not targets:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "vendor_retention", "error": "No active vendor targets"}

    # 2. Fetch all enriched opportunities (company_filter narrows the signal pool)
    opportunities = await _fetch_opportunities(
        pool, min_score, limit * 5, company_filter=company_filter, dm_only=False,
    )

    # 3. Get LLM + skill
    from ...services.llm_router import get_llm
    llm = get_llm("campaign")
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "vendor_retention", "error": "No LLM available"}

    from ...skills import get_skill_registry
    skill = get_skill_registry().get("digest/b2b_vendor_outreach")
    if not skill:
        logger.warning("Skill 'digest/b2b_vendor_outreach' not found")
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "vendor_retention", "error": "Skill not found"}

    llm_model_name = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
    batch_id = f"batch_vr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    generated = 0
    failed = 0
    skipped = 0
    deferred = 0
    deferred = 0
    sequences_created = 0
    batch_metrics = {
        "jobs": 0,
        "submitted_items": 0,
        "cache_prefiltered_items": 0,
        "fallback_single_call_items": 0,
        "completed_items": 0,
        "failed_items": 0,
    }
    phase_one_entries: list[dict[str, Any]] = []

    for target in targets[:limit]:
        vendor_name = target["company_name"]

        # B->A sequencing: require a briefing before campaign generation
        briefing_row = await pool.fetchrow(
            """SELECT id, created_at, briefing_data FROM b2b_vendor_briefings
               WHERE LOWER(vendor_name) = LOWER($1)
                 AND target_mode = $2
                 AND status = 'sent'
               ORDER BY created_at DESC LIMIT 1""",
            vendor_name,
            "vendor_retention",
        )
        if not briefing_row and not bypass_briefing_gate:
            skipped += 1
            continue
        if briefing_row and not bypass_briefing_gate:
            days_since_briefing = (
                datetime.now(timezone.utc) - briefing_row["created_at"]
            ).days
            if days_since_briefing < 7:
                skipped += 1
                continue

        # Dedup: skip vendor if campaign sent within dedup_days
        if not bypass_recent_dedup:
            existing = await pool.fetchval(
                """
                SELECT COUNT(*) FROM b2b_campaigns
                WHERE company_name ILIKE $1
                  AND target_mode = 'vendor_retention'
                  AND created_at > NOW() - make_interval(days => $2)
                """,
                vendor_name, cfg.dedup_days,
            )
            if existing > 0:
                skipped += 1
                continue

        # Group signals: opportunities where vendor_name matches this target
        products = target.get("products_tracked") or []
        vendor_signals = [
            opp for opp in opportunities
            if opp["vendor_name"].lower() == vendor_name.lower()
            or (products and opp["vendor_name"].lower() in [p.lower() for p in products])
        ]

        if not vendor_signals:
            logger.debug("No churn signals found for vendor %s, skipping", vendor_name)
            skipped += 1
            continue

        # Build vendor-scoped context
        vendor_ctx = _build_vendor_context(vendor_name, vendor_signals)
        briefing_context = _briefing_context_from_data(
            briefing_row.get("briefing_data") if briefing_row else None,
        )
        if briefing_context:
            vendor_ctx["briefing_context"] = briefing_context
        vendor_ctx["signal_summary"]["trend_vs_last_month"] = await _compute_vendor_trend(
            pool, vendor_name, products,
        )

        # Load reasoning context (synthesis-first, legacy fallback)
        from ._b2b_synthesis_reader import load_best_reasoning_view

        vendor_reasoning = await load_best_reasoning_view(
            pool,
            vendor_name,
        )
        if vendor_reasoning is not None:
            _inject_reasoning_campaign_context(
                vendor_ctx,
                vendor_reasoning.filtered_consumer_context("campaign"),
            )
            cn = vendor_reasoning.section("causal_narrative")
            wedge = vendor_reasoning.primary_wedge
            wedge_label = wedge.value if wedge else cn.get("primary_wedge", "")
            falsification = [
                fc.get("condition", fc) if isinstance(fc, dict) else fc
                for fc in vendor_reasoning.falsification_conditions()
            ]
            # Build summary from available fields -- v2.3 schema uses
            # trigger/why_now/causal_chain instead of summary/key_signals
            summary = cn.get("summary") or cn.get("executive_summary") or ""
            if not summary:
                parts = [cn.get("trigger", ""), cn.get("why_now", "")]
                chain = cn.get("causal_chain", "")
                if chain:
                    parts.append(chain)
                summary = ". ".join(p for p in parts if p)
            key_signals = cn.get("key_signals") or []
            if not key_signals:
                # Derive from v2.3 fields
                for field in ("trigger", "who_most_affected", "why_now"):
                    val = cn.get(field, "")
                    if val:
                        key_signals.append(val)
            reasoning_ctx: dict[str, Any] = {
                "wedge": wedge_label,
                "confidence": vendor_reasoning.confidence("causal_narrative"),
                "summary": summary,
                "key_signals": key_signals,
                "falsification": falsification,
            }
            # Phase 3 sections: why_they_stay, timing, switch_triggers, accounts
            wts = vendor_reasoning.why_they_stay
            if wts:
                reasoning_ctx["why_they_stay"] = {
                    "summary": wts.get("summary", ""),
                    "strengths": [
                        {"area": s.get("area", ""), "evidence": s.get("evidence", "")}
                        for s in wts.get("strengths", [])
                        if isinstance(s, dict)
                    ][:5],
                }
            timing = vendor_reasoning.section("timing_intelligence")
            if timing:
                reasoning_ctx["timing"] = {
                    "best_window": timing.get("best_timing_window", ""),
                    "trigger_count": len(timing.get("immediate_triggers") or []),
                }
            triggers = vendor_reasoning.switch_triggers
            if triggers:
                reasoning_ctx["switch_triggers"] = [
                    {"type": t.get("type", ""), "description": t.get("description", "")}
                    for t in triggers[:3]
                ]
            cp = vendor_reasoning.confidence_posture
            if cp and cp.get("limits"):
                reasoning_ctx["confidence_limits"] = cp["limits"]
            # Account reasoning summary
            acct = vendor_reasoning.contract("account_reasoning")
            if acct and acct.get("market_summary"):
                reasoning_ctx["account_summary"] = acct["market_summary"]
            atom_context = _campaign_reasoning_atom_context(
                vendor_reasoning.filtered_consumer_context("campaign"),
            )
            if atom_context:
                reasoning_ctx["atom_context"] = atom_context
            delta_summary = _campaign_reasoning_delta_summary(
                vendor_reasoning.reasoning_delta,
            )
            if delta_summary:
                reasoning_ctx["delta_summary"] = delta_summary
            scope_summary = _campaign_reasoning_scope_summary(
                vendor_reasoning.scope_manifest,
            )
            if scope_summary:
                reasoning_ctx["scope_summary"] = scope_summary

            vendor_ctx["reasoning_context"] = reasoning_ctx
            contracts = vendor_reasoning.materialized_contracts()
            if contracts:
                vendor_ctx["reasoning_contracts"] = contracts
        best = max(vendor_signals, key=lambda o: o["opportunity_score"])
        review_ids = [o["review_id"] for o in vendor_signals if o.get("review_id")]

        # Generate for email_cold and email_followup
        _vpains = [p["category"] for p in (vendor_ctx.get("signal_summary") or {}).get("pain_distribution", []) if p.get("category")]
        _vcomps = [c["name"] for c in (vendor_ctx.get("signal_summary") or {}).get("competitor_distribution", []) if c.get("name")]
        vendor_blog_urls = await _fetch_blog_posts(
            pool, pipeline="b2b", vendor_name=vendor_name, category=vendor_ctx.get("category"),
            pain_categories=_vpains or None,
            alternative_vendors=_vcomps[:3] or None,
            contact_role=target.get("contact_role"),
            include_drafts=True,
        )
        # Gate URL for vendor-specific report (instead of generic booking page)
        vendor_gate_url = build_gate_url(vendor_name)
        selling_context = _build_selling_context(
            sender_name=cfg.default_sender_name,
            sender_title=cfg.default_sender_title,
            sender_company=cfg.default_sender_company,
            booking_url=vendor_gate_url,
            blog_posts=vendor_blog_urls,
        )
        cold_payload = {
            **vendor_ctx,
            "contact_name": _sanitize_contact_name(target.get("contact_name")),
            "contact_role": target.get("contact_role"),
            "tier": target.get("tier", "report"),
            "selling": selling_context,
            "channel": "email_cold",
            "target_mode": "vendor_retention",
        }
        cold_artifact_id = _campaign_artifact_key(
            company_name=vendor_name,
            batch_id=batch_id,
            channel="email_cold",
        )
        phase_one_entries.append(
            {
                "custom_id": cold_artifact_id,
                "artifact_id": cold_artifact_id,
                "campaign_batch_id": batch_id,
                "phase": "cold",
                "payload": cold_payload,
                "channel": "email_cold",
                "company_name": vendor_name,
                "best": best,
                "vendor_ctx": vendor_ctx,
                "review_ids": review_ids[:20] or None,
                "target": target,
                "followup_payload": {
                    **vendor_ctx,
                    "contact_name": _sanitize_contact_name(target.get("contact_name")),
                    "contact_role": target.get("contact_role"),
                    "tier": target.get("tier", "report"),
                    "selling": selling_context,
                    "channel": "email_followup",
                    "target_mode": "vendor_retention",
                },
                "sequence_context": {
                    **vendor_ctx,
                    "contact_name": _sanitize_contact_name(target.get("contact_name")),
                    "contact_role": target.get("contact_role"),
                    "tier": target.get("tier", "report"),
                    "recipient_type": "vendor_retention",
                    "selling": selling_context,
                },
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "trace_metadata": _campaign_trace_metadata(
                    cold_payload,
                    run_id=run_id,
                    stage_id="b2b_campaign_generation.content",
                ),
            }
        )

    phase_one_results, phase_one_batch = await _run_campaign_batch(
        llm,
        skill.content,
        phase_one_entries,
        run_id=run_id,
    )
    _merge_batch_metrics(batch_metrics, phase_one_batch)

    followup_entries: list[dict[str, Any]] = []
    for entry in phase_one_entries:
        vendor_name = entry["company_name"]
        payload = entry["payload"]
        artifact_id = entry["artifact_id"]
        content = phase_one_results.get(entry["custom_id"])
        if _is_deferred_campaign_content(content):
            await record_attempt(
                pool,
                artifact_type="campaign",
                artifact_id=artifact_id,
                attempt_no=1,
                stage="generation",
                status="queued",
            )
            deferred += 1
            continue
        if content:
            metadata = _campaign_storage_metadata(payload)
            generation_audit = (
                payload.get("_generation_audit")
                if isinstance(payload.get("_generation_audit"), dict)
                else {}
            )
            specificity = (
                generation_audit.get("specificity")
                if isinstance(generation_audit.get("specificity"), dict)
                else {}
            )
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_campaigns (
                        company_name, vendor_name, product_category,
                        opportunity_score, urgency_score, pain_categories,
                        competitors_considering, seat_count, contract_end,
                        decision_timeline, buying_stage, role_type,
                        key_quotes, source_review_ids,
                        channel, subject, body, cta,
                        status, batch_id, llm_model, industry, target_mode, metadata,
                        recipient_email, score_components
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20, $21, $22, $23, $24::jsonb,
                        $25, $26::jsonb
                    )
                    """,
                    vendor_name,
                    vendor_name,
                    entry["best"].get("product_category"),
                    entry["best"]["opportunity_score"],
                    entry["best"].get("urgency"),
                    json.dumps(entry["vendor_ctx"]["signal_summary"]["pain_distribution"]),
                    json.dumps(entry["vendor_ctx"]["signal_summary"]["competitor_distribution"]),
                    entry["best"].get("seat_count"),
                    entry["best"].get("contract_end"),
                    entry["best"].get("decision_timeline"),
                    entry["best"].get("buying_stage"),
                    _map_role_type(entry["target"].get("contact_role")),
                    json.dumps(entry["vendor_ctx"].get("key_quotes", [])),
                    entry["review_ids"],
                    "email_cold",
                    content.get("subject", ""),
                    content.get("body", ""),
                    content.get("cta", ""),
                    "draft",
                    batch_id,
                    llm_model_name,
                    entry["best"].get("industry"),
                    "vendor_retention",
                    json.dumps(metadata, default=str),
                    entry["target"].get("contact_email"),
                    json.dumps(entry["best"].get("score_components")),
                )
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="generation",
                    status="succeeded",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    warnings=list(specificity.get("warnings") or []),
                )
                generated += 1
                cold_email_content = {
                    "subject": content.get("subject", ""),
                    "body": content.get("body", ""),
                }
                if settings.campaign_sequence.enabled:
                    try:
                        seq_id = await _create_sequence_for_cold_email(
                            pool,
                            company_name=vendor_name,
                            batch_id=batch_id,
                            partner_id=None,
                            context=entry["sequence_context"],
                            cold_email_subject=cold_email_content.get("subject", ""),
                            cold_email_body=cold_email_content.get("body", ""),
                        )
                        if seq_id:
                            sequences_created += 1
                            contact_email = entry["target"].get("contact_email")
                            if contact_email:
                                await pool.execute(
                                    "UPDATE campaign_sequences SET recipient_email = $1 WHERE id = $2",
                                    contact_email,
                                    seq_id,
                                )
                    except Exception as exc:
                        logger.warning("Failed to create vendor sequence for %s: %s", vendor_name, exc)

                followup_payload = dict(entry["followup_payload"])
                followup_payload["cold_email_context"] = cold_email_content
                followup_artifact_id = _campaign_artifact_key(
                    company_name=vendor_name,
                    batch_id=batch_id,
                    channel="email_followup",
                )
                followup_entries.append(
                    {
                        "custom_id": followup_artifact_id,
                        "artifact_id": followup_artifact_id,
                        "campaign_batch_id": batch_id,
                        "phase": "followup",
                        "payload": followup_payload,
                        "channel": "email_followup",
                        "company_name": vendor_name,
                        "best": entry["best"],
                        "vendor_ctx": entry["vendor_ctx"],
                        "review_ids": entry["review_ids"],
                        "target": entry["target"],
                        "max_tokens": cfg.max_tokens,
                        "temperature": cfg.temperature,
                        "trace_metadata": _campaign_trace_metadata(
                            followup_payload,
                            run_id=run_id,
                            stage_id="b2b_campaign_generation.content",
                        ),
                    }
                )
            except Exception:
                logger.exception("Failed to store vendor campaign for %s/%s", vendor_name, "email_cold")
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="storage",
                    status="failed",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    blocking_issues=list(specificity.get("blocking_issues") or []),
                    warnings=list(specificity.get("warnings") or []),
                    failure_step="storage",
                    error_message="campaign_storage_failed",
                )
                failed += 1
        else:
            await _record_campaign_generation_failure(
                pool,
                artifact_id=artifact_id,
                company_name=vendor_name,
                channel="email_cold",
                generation_audit=payload.get("_generation_audit"),
            )
            failed += 1

    followup_results, phase_two_batch = await _run_campaign_batch(
        llm,
        skill.content,
        followup_entries,
        run_id=run_id,
    )
    _merge_batch_metrics(batch_metrics, phase_two_batch)

    for entry in followup_entries:
        vendor_name = entry["company_name"]
        payload = entry["payload"]
        artifact_id = entry["artifact_id"]
        content = followup_results.get(entry["custom_id"])
        if _is_deferred_campaign_content(content):
            await record_attempt(
                pool,
                artifact_type="campaign",
                artifact_id=artifact_id,
                attempt_no=1,
                stage="generation",
                status="queued",
            )
            deferred += 1
            continue
        if content:
            metadata = _campaign_storage_metadata(payload)
            generation_audit = (
                payload.get("_generation_audit")
                if isinstance(payload.get("_generation_audit"), dict)
                else {}
            )
            specificity = (
                generation_audit.get("specificity")
                if isinstance(generation_audit.get("specificity"), dict)
                else {}
            )
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_campaigns (
                        company_name, vendor_name, product_category,
                        opportunity_score, urgency_score, pain_categories,
                        competitors_considering, seat_count, contract_end,
                        decision_timeline, buying_stage, role_type,
                        key_quotes, source_review_ids,
                        channel, subject, body, cta,
                        status, batch_id, llm_model, industry, target_mode, metadata,
                        recipient_email, score_components
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20, $21, $22, $23, $24::jsonb,
                        $25, $26::jsonb
                    )
                    """,
                    vendor_name,
                    vendor_name,
                    entry["best"].get("product_category"),
                    entry["best"]["opportunity_score"],
                    entry["best"].get("urgency"),
                    json.dumps(entry["vendor_ctx"]["signal_summary"]["pain_distribution"]),
                    json.dumps(entry["vendor_ctx"]["signal_summary"]["competitor_distribution"]),
                    entry["best"].get("seat_count"),
                    entry["best"].get("contract_end"),
                    entry["best"].get("decision_timeline"),
                    entry["best"].get("buying_stage"),
                    _map_role_type(entry["target"].get("contact_role")),
                    json.dumps(entry["vendor_ctx"].get("key_quotes", [])),
                    entry["review_ids"],
                    "email_followup",
                    content.get("subject", ""),
                    content.get("body", ""),
                    content.get("cta", ""),
                    "draft",
                    batch_id,
                    llm_model_name,
                    entry["best"].get("industry"),
                    "vendor_retention",
                    json.dumps(metadata, default=str),
                    entry["target"].get("contact_email"),
                    json.dumps(entry["best"].get("score_components")),
                )
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="generation",
                    status="succeeded",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    warnings=list(specificity.get("warnings") or []),
                )
                generated += 1
            except Exception:
                logger.exception("Failed to store vendor campaign for %s/%s", vendor_name, "email_followup")
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="storage",
                    status="failed",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    blocking_issues=list(specificity.get("blocking_issues") or []),
                    warnings=list(specificity.get("warnings") or []),
                    failure_step="storage",
                    error_message="campaign_storage_failed",
                )
                failed += 1
        else:
            await _record_campaign_generation_failure(
                pool,
                artifact_id=artifact_id,
                company_name=vendor_name,
                channel="email_followup",
                generation_audit=payload.get("_generation_audit"),
            )
            failed += 1

    logger.info(
        "Campaign generation (vendor_retention): %d generated, %d failed, %d skipped, %d sequences from %d targets",
        generated, failed, skipped, sequences_created, len(targets),
    )

    return {
        "generated": generated,
        "failed": failed,
        "deferred": deferred,
        "skipped": skipped,
        "sequences_created": sequences_created,
        "companies": len(targets) - skipped,
        "batch_id": batch_id,
        "target_mode": "vendor_retention",
        "anthropic_batch_jobs": batch_metrics["jobs"],
        "anthropic_batch_items_submitted": batch_metrics["submitted_items"],
        "anthropic_batch_cache_prefiltered": batch_metrics["cache_prefiltered_items"],
        "anthropic_batch_fallback_single_call": batch_metrics["fallback_single_call_items"],
        "anthropic_batch_completed_items": batch_metrics["completed_items"],
        "anthropic_batch_failed_items": batch_metrics["failed_items"],
    }


# ------------------------------------------------------------------
# Challenger intel campaign generation (P2)
# ------------------------------------------------------------------


async def _fetch_challenger_targets(pool, vendor_filter: str | None = None) -> list[dict]:
    """Fetch active challenger targets."""
    if vendor_filter:
        rows = await pool.fetch(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier, status,
                   notes, account_id, created_at, updated_at
            FROM vendor_targets
            WHERE status = 'active' AND target_mode = 'challenger_intel'
              AND company_name ILIKE '%' || $1 || '%'
            """,
            vendor_filter,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, company_name, target_mode, contact_name, contact_email,
                   contact_role, products_tracked, competitors_tracked, tier, status,
                   notes, account_id, created_at, updated_at
            FROM vendor_targets
            WHERE status = 'active' AND target_mode = 'challenger_intel'
            """
        )
    return dedupe_vendor_target_rows(rows)


def _build_challenger_context(challenger_name: str, signals: list[dict]) -> dict[str, Any]:
    """Aggregate signals where a specific product is mentioned as the alternative."""
    total = len(signals)

    # Buying stage distribution
    by_stage: dict[str, int] = {}
    ordered_signals = _campaign_stable_row_order(signals)

    for s in ordered_signals:
        stage = s.get("buying_stage") or "unknown"
        by_stage[stage] = by_stage.get(stage, 0) + 1

    # Role distribution
    role_counts: dict[str, int] = {}
    for s in ordered_signals:
        role = s.get("role_type") or "unknown"
        role_counts[role] = role_counts.get(role, 0) + 1

    # Pain categories driving the switch (from incumbent)
    pain_counts: dict[str, int] = {}
    for s in ordered_signals:
        pain = _parse_json_field(s.get("pain_json"))
        for p in pain:
            if isinstance(p, dict) and p.get("category"):
                pain_counts[p["category"]] = pain_counts.get(p["category"], 0) + 1

    # Incumbents losing customers
    incumbent_counts: dict[str, int] = {}
    for s in ordered_signals:
        vendor = s.get("vendor_name", "")
        if vendor:
            incumbent_counts[vendor] = incumbent_counts.get(vendor, 0) + 1

    # Seat count buckets
    large = sum(1 for s in ordered_signals if (s.get("seat_count") or 0) >= 500)
    mid = sum(1 for s in ordered_signals if 100 <= (s.get("seat_count") or 0) < 500)
    small = sum(1 for s in ordered_signals if 0 < (s.get("seat_count") or 0) < 100)

    # Feature mentions (positive mentions of challenger from competitor context)
    feature_mentions: list[str] = []
    for s in ordered_signals:
        comps = s.get("competitors", [])
        for c in comps:
            if isinstance(c, dict) and c.get("name", "").lower() == challenger_name.lower():
                reason = c.get("reason", "")
                if reason and reason not in feature_mentions:
                    feature_mentions.append(reason)

    # Quotable phrases from enrichment
    quote_list: list[str] = []
    for s in ordered_signals:
        phrases = _parse_json_field(s.get("quotable_phrases"))
        for phrase in phrases:
            text = phrase if isinstance(phrase, str) else (
                phrase.get("text", "") if isinstance(phrase, dict) else ""
            )
            if text and text not in quote_list:
                quote_list.append(text)

    return {
        "challenger_name": challenger_name,
        "key_quotes": quote_list[:8],
        "signal_summary": {
            "total_leads": total,
            "by_buying_stage": {
                "active_purchase": by_stage.get("active_purchase", 0),
                "evaluation": by_stage.get("evaluation", 0),
                "renewal_decision": by_stage.get("renewal_decision", 0),
            },
            "role_distribution": _campaign_sorted_count_rows(
                role_counts,
                field_name="role",
                limit=5,
            ),
            "pain_driving_switch": _campaign_sorted_count_rows(
                pain_counts,
                field_name="category",
                limit=10,
            ),
            "incumbents_losing": _campaign_sorted_count_rows(
                incumbent_counts,
                field_name="name",
                limit=10,
            ),
            "seat_count_signals": {
                "large_500plus": large,
                "mid_100_499": mid,
                "small_under_100": small,
            },
            "feature_mentions": feature_mentions[:10],
        },
    }


async def _generate_challenger_campaigns(
    pool,
    min_score: int,
    limit: int,
    vendor_filter: str | None,
    company_filter: str | None = None,
    bypass_briefing_gate: bool = False,
    bypass_recent_dedup: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Generate campaigns targeting challenger Sales/Competitive Intel leaders."""
    cfg = settings.b2b_campaign

    # 1. Fetch challenger targets
    targets = await _fetch_challenger_targets(pool, vendor_filter)
    if not targets:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "challenger_intel", "error": "No active challenger targets"}

    # 2. Fetch all enriched opportunities (company_filter narrows the signal pool)
    opportunities = await _fetch_opportunities(
        pool, min_score, limit * 5, company_filter=company_filter, dm_only=False,
    )

    # 3. Get LLM + skill
    from ...services.llm_router import get_llm
    llm = get_llm("campaign")
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "challenger_intel", "error": "No LLM available"}

    from ...skills import get_skill_registry
    skill = get_skill_registry().get("digest/b2b_challenger_outreach")
    if not skill:
        logger.warning("Skill 'digest/b2b_challenger_outreach' not found")
        return {"generated": 0, "skipped": 0, "failed": 0, "companies": 0,
                "target_mode": "challenger_intel", "error": "Skill not found"}

    llm_model_name = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
    batch_id = f"batch_ci_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    generated = 0
    failed = 0
    skipped = 0
    deferred = 0
    sequences_created = 0
    batch_metrics = {
        "jobs": 0,
        "submitted_items": 0,
        "cache_prefiltered_items": 0,
        "fallback_single_call_items": 0,
        "completed_items": 0,
        "failed_items": 0,
    }
    phase_one_entries: list[dict[str, Any]] = []

    for target in targets[:limit]:
        challenger_name = target["company_name"]

        # B->A sequencing: require a briefing before campaign generation
        briefing_row = await pool.fetchrow(
            """SELECT id, created_at, briefing_data FROM b2b_vendor_briefings
               WHERE LOWER(vendor_name) = LOWER($1)
                 AND target_mode = $2
                 AND status = 'sent'
               ORDER BY created_at DESC LIMIT 1""",
            challenger_name,
            "challenger_intel",
        )
        if not briefing_row and not bypass_briefing_gate:
            skipped += 1
            continue
        if briefing_row and not bypass_briefing_gate:
            days_since_briefing = (
                datetime.now(timezone.utc) - briefing_row["created_at"]
            ).days
            if days_since_briefing < 7:
                skipped += 1
                continue

        # Dedup
        if not bypass_recent_dedup:
            existing = await pool.fetchval(
                """
                SELECT COUNT(*) FROM b2b_campaigns
                WHERE company_name ILIKE $1
                  AND target_mode = 'challenger_intel'
                  AND created_at > NOW() - make_interval(days => $2)
                """,
                challenger_name, cfg.dedup_days,
            )
            if existing > 0:
                skipped += 1
                continue

        # Find signals where this challenger is mentioned as a competitor being considered
        challenger_signals = []
        seen_ids: set[str] = set()
        tracked_vendors = {p.lower() for p in (target.get("competitors_tracked") or [])}
        for opp in opportunities:
            opp_id = opp.get("review_id", id(opp))
            if opp_id in seen_ids:
                continue
            matched = False
            comps = opp.get("competitors", [])
            for c in comps:
                if isinstance(c, dict) and c.get("name", "").lower() == challenger_name.lower():
                    matched = True
                    break
            # Also match against competitors_tracked (incumbents the challenger cares about)
            if not matched and opp["vendor_name"].lower() in tracked_vendors:
                matched = True
            if matched:
                challenger_signals.append(opp)
                seen_ids.add(opp_id)

        if not challenger_signals:
            logger.debug("No intent signals found for challenger %s, skipping", challenger_name)
            skipped += 1
            continue

        # Build challenger-scoped context
        challenger_ctx = _build_challenger_context(challenger_name, challenger_signals)
        briefing_context = _briefing_context_from_data(
            briefing_row.get("briefing_data") if briefing_row else None,
        )
        if briefing_context:
            challenger_ctx["briefing_context"] = briefing_context

        # Fetch incumbent reasoning (synthesis-first, legacy fallback)
        incumbent_names = [
            inc["name"]
            for inc in challenger_ctx.get("signal_summary", {}).get("incumbents_losing", [])
            if inc.get("name")
        ]
        if incumbent_names:
            from ._b2b_synthesis_reader import load_best_reasoning_views

            inc_views = await load_best_reasoning_views(
                pool,
                incumbent_names,
            )
            if inc_views:
                by_archetype: dict[str, list[str]] = {}
                incumbent_reasoning: dict[str, dict[str, Any]] = {}
                for vname, view in inc_views.items():
                    wedge = view.primary_wedge
                    label = wedge.value if wedge else (
                        view.section("causal_narrative").get("primary_wedge", "")
                    )
                    if label:
                        by_archetype.setdefault(label, []).append(vname)
                    # Extract incumbent reasoning summary for challenger context
                    cn = view.section("causal_narrative")
                    inc_summary: dict[str, Any] = {
                        "wedge": label,
                        "summary": cn.get("summary") or cn.get("executive_summary", ""),
                    }
                    wts = view.why_they_stay
                    if wts:
                        inc_summary["why_they_stay"] = wts.get("summary", "")
                    triggers = view.switch_triggers
                    if triggers:
                        inc_summary["switch_triggers"] = [
                            t.get("type", "") for t in triggers[:3]
                        ]
                    if inc_summary.get("summary"):
                        incumbent_reasoning[vname] = inc_summary
                if by_archetype:
                    challenger_ctx["incumbent_archetypes"] = by_archetype
                if incumbent_reasoning:
                    challenger_ctx["incumbent_reasoning"] = incumbent_reasoning

        best = max(challenger_signals, key=lambda o: o["opportunity_score"])
        review_ids = [o["review_id"] for o in challenger_signals if o.get("review_id")]

        _cpains = [p["category"] for p in (challenger_ctx.get("signal_summary") or {}).get("pain_driving_switch", []) if p.get("category")]
        _cincumbents = [c["name"] for c in (challenger_ctx.get("signal_summary") or {}).get("incumbents_losing", []) if c.get("name")]
        challenger_blog_urls = await _fetch_blog_posts(
            pool, pipeline="b2b", vendor_name=challenger_name, category=challenger_ctx.get("category"),
            pain_categories=_cpains or None,
            alternative_vendors=_cincumbents[:3] or None,
            contact_role=target.get("contact_role"),
            include_drafts=True,
        )
        # Gate URL for challenger-specific report
        challenger_gate_url = build_gate_url(challenger_name)
        selling_context = _build_selling_context(
            sender_name=cfg.default_sender_name,
            sender_title=cfg.default_sender_title,
            sender_company=cfg.default_sender_company,
            booking_url=challenger_gate_url,
            blog_posts=challenger_blog_urls,
        )
        cold_payload = {
            **challenger_ctx,
            "contact_name": _sanitize_contact_name(target.get("contact_name")),
            "contact_role": target.get("contact_role"),
            "tier": target.get("tier", "report"),
            "selling": selling_context,
            "channel": "email_cold",
            "target_mode": "challenger_intel",
        }
        cold_artifact_id = _campaign_artifact_key(
            company_name=challenger_name,
            batch_id=batch_id,
            channel="email_cold",
        )
        phase_one_entries.append(
            {
                "custom_id": cold_artifact_id,
                "artifact_id": cold_artifact_id,
                "campaign_batch_id": batch_id,
                "phase": "cold",
                "payload": cold_payload,
                "channel": "email_cold",
                "company_name": challenger_name,
                "best": best,
                "challenger_ctx": challenger_ctx,
                "review_ids": review_ids[:20] or None,
                "target": target,
                "followup_payload": {
                    **challenger_ctx,
                    "contact_name": _sanitize_contact_name(target.get("contact_name")),
                    "contact_role": target.get("contact_role"),
                    "tier": target.get("tier", "report"),
                    "selling": selling_context,
                    "channel": "email_followup",
                    "target_mode": "challenger_intel",
                },
                "sequence_context": {
                    **challenger_ctx,
                    "contact_name": _sanitize_contact_name(target.get("contact_name")),
                    "contact_role": target.get("contact_role"),
                    "tier": target.get("tier", "report"),
                    "recipient_type": "challenger_intel",
                    "selling": selling_context,
                },
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "trace_metadata": _campaign_trace_metadata(
                    cold_payload,
                    run_id=run_id,
                    stage_id="b2b_campaign_generation.content",
                ),
            }
        )

    phase_one_results, phase_one_batch = await _run_campaign_batch(
        llm,
        skill.content,
        phase_one_entries,
        run_id=run_id,
    )
    _merge_batch_metrics(batch_metrics, phase_one_batch)

    followup_entries: list[dict[str, Any]] = []
    for entry in phase_one_entries:
        challenger_name = entry["company_name"]
        payload = entry["payload"]
        artifact_id = entry["artifact_id"]
        content = phase_one_results.get(entry["custom_id"])
        if _is_deferred_campaign_content(content):
            await record_attempt(
                pool,
                artifact_type="campaign",
                artifact_id=artifact_id,
                attempt_no=1,
                stage="generation",
                status="queued",
            )
            deferred += 1
            continue
        if content:
            metadata = _campaign_storage_metadata(payload)
            generation_audit = (
                payload.get("_generation_audit")
                if isinstance(payload.get("_generation_audit"), dict)
                else {}
            )
            specificity = (
                generation_audit.get("specificity")
                if isinstance(generation_audit.get("specificity"), dict)
                else {}
            )
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_campaigns (
                        company_name, vendor_name, product_category,
                        opportunity_score, urgency_score, pain_categories,
                        competitors_considering, seat_count, contract_end,
                        decision_timeline, buying_stage, role_type,
                        key_quotes, source_review_ids,
                        channel, subject, body, cta,
                        status, batch_id, llm_model, industry, target_mode, metadata,
                        recipient_email, score_components
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20, $21, $22, $23, $24::jsonb,
                        $25, $26::jsonb
                    )
                    """,
                    challenger_name,
                    entry["best"]["vendor_name"],
                    entry["best"].get("product_category"),
                    entry["best"]["opportunity_score"],
                    entry["best"].get("urgency"),
                    json.dumps(entry["challenger_ctx"]["signal_summary"]["pain_driving_switch"]),
                    json.dumps(entry["challenger_ctx"]["signal_summary"]["incumbents_losing"]),
                    entry["best"].get("seat_count"),
                    entry["best"].get("contract_end"),
                    entry["best"].get("decision_timeline"),
                    entry["best"].get("buying_stage"),
                    _map_role_type(entry["target"].get("contact_role")),
                    json.dumps(entry["challenger_ctx"].get("key_quotes", [])),
                    entry["review_ids"],
                    "email_cold",
                    content.get("subject", ""),
                    content.get("body", ""),
                    content.get("cta", ""),
                    "draft",
                    batch_id,
                    llm_model_name,
                    entry["best"].get("industry"),
                    "challenger_intel",
                    json.dumps(metadata, default=str),
                    entry["target"].get("contact_email"),
                    json.dumps(entry["best"].get("score_components")),
                )
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="generation",
                    status="succeeded",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    warnings=list(specificity.get("warnings") or []),
                )
                generated += 1
                cold_email_content = {
                    "subject": content.get("subject", ""),
                    "body": content.get("body", ""),
                }
                if settings.campaign_sequence.enabled:
                    try:
                        seq_id = await _create_sequence_for_cold_email(
                            pool,
                            company_name=challenger_name,
                            batch_id=batch_id,
                            partner_id=None,
                            context=entry["sequence_context"],
                            cold_email_subject=cold_email_content.get("subject", ""),
                            cold_email_body=cold_email_content.get("body", ""),
                        )
                        if seq_id:
                            sequences_created += 1
                            contact_email = entry["target"].get("contact_email")
                            if contact_email:
                                await pool.execute(
                                    "UPDATE campaign_sequences SET recipient_email = $1 WHERE id = $2",
                                    contact_email,
                                    seq_id,
                                )
                    except Exception as exc:
                        logger.warning("Failed to create challenger sequence for %s: %s", challenger_name, exc)

                followup_payload = dict(entry["followup_payload"])
                followup_payload["cold_email_context"] = cold_email_content
                followup_artifact_id = _campaign_artifact_key(
                    company_name=challenger_name,
                    batch_id=batch_id,
                    channel="email_followup",
                )
                followup_entries.append(
                    {
                        "custom_id": followup_artifact_id,
                        "artifact_id": followup_artifact_id,
                        "campaign_batch_id": batch_id,
                        "phase": "followup",
                        "payload": followup_payload,
                        "channel": "email_followup",
                        "company_name": challenger_name,
                        "best": entry["best"],
                        "challenger_ctx": entry["challenger_ctx"],
                        "review_ids": entry["review_ids"],
                        "target": entry["target"],
                        "max_tokens": cfg.max_tokens,
                        "temperature": cfg.temperature,
                        "trace_metadata": _campaign_trace_metadata(
                            followup_payload,
                            run_id=run_id,
                            stage_id="b2b_campaign_generation.content",
                        ),
                    }
                )
            except Exception:
                logger.exception("Failed to store challenger campaign for %s/%s", challenger_name, "email_cold")
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="storage",
                    status="failed",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    blocking_issues=list(specificity.get("blocking_issues") or []),
                    warnings=list(specificity.get("warnings") or []),
                    failure_step="storage",
                    error_message="campaign_storage_failed",
                )
                failed += 1
        else:
            await _record_campaign_generation_failure(
                pool,
                artifact_id=artifact_id,
                company_name=challenger_name,
                channel="email_cold",
                generation_audit=payload.get("_generation_audit"),
            )
            failed += 1

    followup_results, phase_two_batch = await _run_campaign_batch(
        llm,
        skill.content,
        followup_entries,
        run_id=run_id,
    )
    _merge_batch_metrics(batch_metrics, phase_two_batch)

    for entry in followup_entries:
        challenger_name = entry["company_name"]
        payload = entry["payload"]
        artifact_id = entry["artifact_id"]
        content = followup_results.get(entry["custom_id"])
        if _is_deferred_campaign_content(content):
            await record_attempt(
                pool,
                artifact_type="campaign",
                artifact_id=artifact_id,
                attempt_no=1,
                stage="generation",
                status="queued",
            )
            deferred += 1
            continue
        if content:
            metadata = _campaign_storage_metadata(payload)
            generation_audit = (
                payload.get("_generation_audit")
                if isinstance(payload.get("_generation_audit"), dict)
                else {}
            )
            specificity = (
                generation_audit.get("specificity")
                if isinstance(generation_audit.get("specificity"), dict)
                else {}
            )
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_campaigns (
                        company_name, vendor_name, product_category,
                        opportunity_score, urgency_score, pain_categories,
                        competitors_considering, seat_count, contract_end,
                        decision_timeline, buying_stage, role_type,
                        key_quotes, source_review_ids,
                        channel, subject, body, cta,
                        status, batch_id, llm_model, industry, target_mode, metadata,
                        recipient_email, score_components
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20, $21, $22, $23, $24::jsonb,
                        $25, $26::jsonb
                    )
                    """,
                    challenger_name,
                    entry["best"]["vendor_name"],
                    entry["best"].get("product_category"),
                    entry["best"]["opportunity_score"],
                    entry["best"].get("urgency"),
                    json.dumps(entry["challenger_ctx"]["signal_summary"]["pain_driving_switch"]),
                    json.dumps(entry["challenger_ctx"]["signal_summary"]["incumbents_losing"]),
                    entry["best"].get("seat_count"),
                    entry["best"].get("contract_end"),
                    entry["best"].get("decision_timeline"),
                    entry["best"].get("buying_stage"),
                    _map_role_type(entry["target"].get("contact_role")),
                    json.dumps(entry["challenger_ctx"].get("key_quotes", [])),
                    entry["review_ids"],
                    "email_followup",
                    content.get("subject", ""),
                    content.get("body", ""),
                    content.get("cta", ""),
                    "draft",
                    batch_id,
                    llm_model_name,
                    entry["best"].get("industry"),
                    "challenger_intel",
                    json.dumps(metadata, default=str),
                    entry["target"].get("contact_email"),
                    json.dumps(entry["best"].get("score_components")),
                )
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="generation",
                    status="succeeded",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    warnings=list(specificity.get("warnings") or []),
                )
                generated += 1
            except Exception:
                logger.exception("Failed to store challenger campaign for %s/%s", challenger_name, "email_followup")
                await record_attempt(
                    pool,
                    artifact_type="campaign",
                    artifact_id=artifact_id,
                    attempt_no=1,
                    stage="storage",
                    status="failed",
                    blocker_count=len(specificity.get("blocking_issues") or []),
                    warning_count=len(specificity.get("warnings") or []),
                    blocking_issues=list(specificity.get("blocking_issues") or []),
                    warnings=list(specificity.get("warnings") or []),
                    failure_step="storage",
                    error_message="campaign_storage_failed",
                )
                failed += 1
        else:
            await _record_campaign_generation_failure(
                pool,
                artifact_id=artifact_id,
                company_name=challenger_name,
                channel="email_followup",
                generation_audit=payload.get("_generation_audit"),
            )
            failed += 1

    logger.info(
        "Campaign generation (challenger_intel): %d generated, %d failed, %d skipped, %d sequences from %d targets",
        generated, failed, skipped, sequences_created, len(targets),
    )

    return {
        "generated": generated,
        "failed": failed,
        "deferred": deferred,
        "skipped": skipped,
        "sequences_created": sequences_created,
        "companies": len(targets) - skipped,
        "batch_id": batch_id,
        "target_mode": "challenger_intel",
        "anthropic_batch_jobs": batch_metrics["jobs"],
        "anthropic_batch_items_submitted": batch_metrics["submitted_items"],
        "anthropic_batch_cache_prefiltered": batch_metrics["cache_prefiltered_items"],
        "anthropic_batch_fallback_single_call": batch_metrics["fallback_single_call_items"],
        "anthropic_batch_completed_items": batch_metrics["completed_items"],
        "anthropic_batch_failed_items": batch_metrics["failed_items"],
    }


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_opportunities(
    pool,
    min_score: int,
    limit: int,
    vendor_filter: str | None = None,
    company_filter: str | None = None,
    dm_only: bool = True,
) -> list[dict[str, Any]]:
    """Fetch and score top opportunities from enriched b2b_reviews."""
    from atlas_brain.autonomous.tasks._b2b_shared import read_campaign_opportunities

    rows = await read_campaign_opportunities(
        pool,
        window_days=90,
        min_urgency=5.0,
        vendor_name=vendor_filter,
        company=company_filter,
        dm_only=dm_only,
        limit=min(limit * 3, 500),
    )

    # Fetch prior engagement speed for scoring tie-breaker
    avg_open_cache: dict[str, float | None] = {}
    vendor_names = {r["vendor_name"] for r in rows if r.get("vendor_name")}
    if vendor_names:
        eng_rows = await pool.fetch(
            """
            SELECT vendor_name, AVG(hours_to_first_open) AS avg_open_hours
            FROM b2b_campaigns
            WHERE vendor_name = ANY($1::text[])
              AND hours_to_first_open IS NOT NULL
              AND sent_at > NOW() - INTERVAL '90 days'
            GROUP BY vendor_name
            """,
            list(vendor_names),
        )
        for er in eng_rows:
            avg_open_cache[er["vendor_name"]] = (
                float(er["avg_open_hours"]) if er["avg_open_hours"] is not None else None
            )

    opportunities = []
    for r in rows:
        row_dict = dict(r)
        competitors = row_dict.get("competitors", [])

        mention_context = ""
        if competitors and isinstance(competitors[0], dict):
            mention_context = competitors[0].get("context", "")

        row_dict["mention_context"] = mention_context
        row_dict["urgency"] = _safe_float(row_dict.get("urgency"), 0)
        row_dict["avg_open_hours"] = avg_open_cache.get(row_dict.get("vendor_name"))
        opp_score, score_components = _compute_score(row_dict)

        if opp_score < min_score:
            continue

        row_dict["opportunity_score"] = opp_score
        row_dict["score_components"] = score_components
        opportunities.append(row_dict)

    # Sort by score, then by prior engagement speed as tie-breaker (lower open hours = better)
    opportunities.sort(
        key=lambda o: (o["opportunity_score"], -(o.get("avg_open_hours") or 999)),
        reverse=True,
    )
    return opportunities[:limit]


async def _fetch_accounts_in_motion_opportunities(
    pool,
    min_score: int,
    limit: int,
    vendor_filter: str | None = None,
    company_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch named company targets from the latest accounts-in-motion reports."""
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (LOWER(vendor_filter))
               vendor_filter, intelligence_data
        FROM b2b_intelligence
        WHERE report_type = 'accounts_in_motion'
          AND vendor_filter IS NOT NULL
        ORDER BY LOWER(vendor_filter), report_date DESC, created_at DESC
        """
    )
    results: list[dict[str, Any]] = []
    vendor_filter_lc = str(vendor_filter or "").lower()
    company_filter_lc = str(company_filter or "").lower()

    for row in rows:
        vendor_name = str(row.get("vendor_filter") or "").strip()
        if vendor_filter_lc and vendor_filter_lc not in vendor_name.lower():
            continue
        report = row.get("intelligence_data")
        if isinstance(report, str):
            try:
                report = json.loads(report)
            except (json.JSONDecodeError, TypeError):
                report = {}
        if not isinstance(report, dict):
            continue
        feature_gaps = _parse_json_field(report.get("feature_gaps"))
        category = report.get("category")
        for account in report.get("accounts") or []:
            if not isinstance(account, dict):
                continue
            company = str(account.get("company") or "").strip()
            if not company:
                continue
            if company_filter_lc and company_filter_lc not in company.lower():
                continue
            score = int(account.get("opportunity_score") or 0)
            if score < min_score:
                continue
            quote = str(account.get("top_quote") or "").strip()
            alternatives = [
                {"name": str(name).strip(), "reason": ""}
                for name in (account.get("alternatives_considering") or [])
                if str(name).strip()
            ]
            review_ids = account.get("source_reviews") or []
            results.append({
                "review_id": review_ids[0] if review_ids else None,
                "vendor_name": vendor_name or str(report.get("vendor") or "").strip(),
                "reviewer_company": company,
                "product_category": category,
                "opportunity_score": score,
                "urgency": account.get("urgency"),
                "pain_json": (
                    [{"category": account.get("pain_category"), "severity": "primary"}]
                    if account.get("pain_category") else []
                ),
                "competitors": alternatives,
                "quotable_phrases": [{"text": quote}] if quote else [],
                "feature_gaps": feature_gaps,
                "integration_stack": [],
                "seat_count": account.get("seat_count"),
                "contract_end": account.get("contract_end"),
                "decision_timeline": account.get("decision_timeline"),
                "buying_stage": account.get("buying_stage"),
                "role_type": account.get("role_level"),
                "industry": account.get("industry"),
                "reviewer_title": account.get("title"),
                "company_size_raw": account.get("company_size"),
                "score_components": account.get("score_components") or {},
                "opportunity_source": "accounts_in_motion",
            })

    results.sort(
        key=lambda row: (
            int(row.get("opportunity_score") or 0),
            float(row.get("urgency") or 0),
        ),
        reverse=True,
    )
    return results[:limit]


def _parse_json_field(val) -> list:
    """Safely parse a JSONB field that may be a str, list, or None."""
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


def _campaign_quote_texts(value: Any) -> list[str]:
    phrases = _parse_json_field(value)
    texts: list[str] = []
    for phrase in phrases:
        if isinstance(phrase, str):
            text = phrase.strip()
        elif isinstance(phrase, dict):
            text = str(
                phrase.get("text")
                or phrase.get("phrase")
                or phrase.get("quote")
                or phrase.get("best_quote")
                or ""
            ).strip()
        else:
            text = ""
        if text and text not in texts:
            texts.append(text)
    return texts


def _campaign_primary_pain_category(opp: dict[str, Any]) -> str:
    for item in _parse_json_field(opp.get("pain_json")):
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip()
        if category:
            return category
    return ""


def _campaign_numeric_literals(opp: dict[str, Any]) -> dict[str, list[str]]:
    literals: dict[str, list[str]] = {}
    seat_count = opp.get("seat_count")
    try:
        seats = int(seat_count) if seat_count not in (None, "") else 0
    except (TypeError, ValueError):
        seats = 0
    if seats > 0:
        literals["seat_count"] = [str(seats)]
    return literals


def _build_churning_company_anchor_context(
    best: dict[str, Any],
    all_opps: list[dict[str, Any]],
) -> dict[str, Any]:
    pain_counts: dict[str, int] = {}
    for opp in all_opps:
        category = _campaign_primary_pain_category(opp)
        if category:
            pain_counts[category.lower()] = pain_counts.get(category.lower(), 0) + 1
    dominant_pain = ""
    if pain_counts:
        dominant_pain = max(
            pain_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

    candidates: list[dict[str, Any]] = []
    for opp in all_opps:
        review_id = str(opp.get("review_id") or "").strip()
        quote_texts = _campaign_quote_texts(opp.get("quotable_phrases"))
        excerpt = quote_texts[0] if quote_texts else ""
        if not excerpt:
            continue

        competitor = ""
        competitors = opp.get("competitors") or []
        if isinstance(competitors, list):
            for item in competitors:
                if isinstance(item, dict):
                    competitor = str(item.get("name") or "").strip()
                else:
                    competitor = str(item or "").strip()
                if competitor:
                    break

        feature_gaps: list[str] = []
        for gap in _parse_json_field(opp.get("feature_gaps")):
            if isinstance(gap, dict):
                label = str(
                    gap.get("feature") or gap.get("name") or gap.get("gap") or ""
                ).strip()
            else:
                label = str(gap or "").strip()
            if label and label not in feature_gaps:
                feature_gaps.append(label)

        numeric_literals = _campaign_numeric_literals(opp)
        pain_category = _campaign_primary_pain_category(opp)
        time_anchor = str(
            opp.get("contract_end") or opp.get("decision_timeline") or ""
        ).strip()
        witness_id = (
            f"campaign_witness:{review_id}:0"
            if review_id
            else f"campaign_witness:{str(best.get('vendor_name') or '').lower()}:{len(candidates)}"
        )
        candidates.append({
            "witness_id": witness_id,
            "excerpt_text": excerpt,
            "reviewer_company": str(opp.get("reviewer_company") or "").strip(),
            "time_anchor": time_anchor,
            "competitor": competitor,
            "pain_category": pain_category,
            "signal_tags": feature_gaps[:3],
            "numeric_literals": numeric_literals,
            "_urgency": _safe_float(opp.get("urgency"), 0),
            "_has_numeric": bool(numeric_literals),
            "_has_time": bool(time_anchor),
            "_has_competitor": bool(competitor),
            "_is_common_pattern": bool(
                dominant_pain and pain_category and pain_category.lower() == dominant_pain
            ),
        })

    if not candidates:
        return {}

    candidates.sort(
        key=lambda item: (
            item["_has_numeric"],
            item["_has_time"],
            item["_has_competitor"],
            item["_is_common_pattern"],
            item["_urgency"],
            len(str(item.get("excerpt_text") or "")),
            str(item.get("witness_id") or ""),
        ),
        reverse=True,
    )

    anchor_examples: dict[str, list[dict[str, Any]]] = {}
    used_ids: set[str] = set()
    for label, predicate in (
        ("outlier_or_named_account", lambda item: True),
        ("common_pattern", lambda item: bool(item.get("_is_common_pattern"))),
    ):
        for candidate in candidates:
            witness_id = str(candidate.get("witness_id") or "")
            if witness_id in used_ids or not predicate(candidate):
                continue
            anchor_examples[label] = [
                {
                    key: value
                    for key, value in candidate.items()
                    if not key.startswith("_")
                },
            ]
            used_ids.add(witness_id)
            break

    limit = max(1, int(settings.b2b_churn.reasoning_witness_highlight_limit))
    witness_highlights = [
        {
            key: value
            for key, value in candidate.items()
            if not key.startswith("_")
        }
        for candidate in candidates[:limit]
    ]
    reference_ids = {
        "witness_ids": [
            str(item.get("witness_id") or "").strip()
            for item in witness_highlights
            if str(item.get("witness_id") or "").strip()
        ],
    }

    return {
        "reasoning_anchor_examples": anchor_examples,
        "reasoning_witness_highlights": witness_highlights,
        "reasoning_reference_ids": reference_ids,
    }


def _build_company_context(best: dict, all_opps: list[dict]) -> dict[str, Any]:
    """Build rich context dict for LLM from grouped opportunities."""
    ordered_opps = _campaign_stable_row_order(all_opps)
    pain_cats: dict[str, str] = {}
    competitors_considering: list[dict] = []
    key_quotes: list[str] = []
    all_feature_gaps: list[str] = []
    all_integrations: list[str] = []

    for opp in ordered_opps:
        # Pain categories
        pain = _parse_json_field(opp.get("pain_json"))
        for p in pain:
            if isinstance(p, dict) and p.get("category"):
                category = str(p.get("category") or "").strip()
                if not category:
                    continue
                pain_cats[category] = _campaign_merge_pain_severity(
                    pain_cats.get(category, ""),
                    str(p.get("severity") or "mentioned"),
                )

        # Competitors
        comps = opp.get("competitors", [])
        for c in comps:
            if isinstance(c, dict) and c.get("name"):
                if not any(x["name"].lower() == c["name"].lower() for x in competitors_considering):
                    competitors_considering.append({
                        "name": c["name"],
                        "reason": c.get("reason", ""),
                    })

        # Curated quotes from enrichment (replaces raw review_text truncation)
        phrases = _parse_json_field(opp.get("quotable_phrases"))
        for phrase in phrases:
            text = phrase if isinstance(phrase, str) else (phrase.get("text", "") if isinstance(phrase, dict) else "")
            if text and text not in key_quotes:
                key_quotes.append(text)

        # Feature gaps
        gaps = _parse_json_field(opp.get("feature_gaps"))
        for g in gaps:
            label = g if isinstance(g, str) else (g.get("feature", "") if isinstance(g, dict) else "")
            if label and label not in all_feature_gaps:
                all_feature_gaps.append(label)

        # Integration stack
        stacks = _parse_json_field(opp.get("integration_stack"))
        for s in stacks:
            if isinstance(s, str) and s not in all_integrations:
                all_integrations.append(s)

    context = {
        "company": best.get("reviewer_company") or best["vendor_name"],
        "churning_from": best["vendor_name"],
        "category": best.get("product_category", ""),
        "pain_categories": [
            {"category": key, "severity": pain_cats[key]}
            for key in sorted(pain_cats, key=lambda item: item.lower())
        ],
        "competitors_considering": competitors_considering[:5],
        "urgency": best.get("urgency", 0),
        "seat_count": best.get("seat_count"),
        "contract_end": best.get("contract_end"),
        "decision_timeline": best.get("decision_timeline"),
        "buying_stage": best.get("buying_stage"),
        "role_type": best.get("role_type"),
        "industry": best.get("industry"),
        "reviewer_title": best.get("reviewer_title"),
        "company_size": best.get("company_size_raw"),
        "key_quotes": key_quotes[:5],
        "feature_gaps": all_feature_gaps[:5],
        "primary_workflow": best.get("primary_workflow"),
        "integration_stack": all_integrations[:5],
        "sentiment_direction": best.get("sentiment_direction"),
    }
    context.update(_build_churning_company_anchor_context(best, all_opps))
    return context


# ------------------------------------------------------------------
# Persona-specific context filtering
# ------------------------------------------------------------------

# Pain categories relevant to each persona
_PERSONA_PAIN_FILTER: dict[str, set[str]] = {
    "executive": {"pricing", "cost", "scalability", "reliability"},
    "technical": {"features", "ux", "integration", "security", "performance"},
    "operations": {"support", "reliability", "usability", "service"},
    "evaluator": {"features", "ux", "integration", "usability", "performance", "pricing"},
    "champion": {"support", "usability", "reliability", "features", "service"},
}

# Quote keywords for filtering key_quotes per persona
_PERSONA_QUOTE_KEYWORDS: dict[str, list[str]] = {
    "executive": ["cost", "price", "budget", "roi", "renewal", "money", "expensive", "contract", "spend"],
    "technical": ["feature", "bug", "api", "migration", "workaround", "integration", "missing", "broken", "limitation"],
    "operations": ["support", "ticket", "downtime", "complaint", "productivity", "team", "workflow", "response time", "sla"],
    "evaluator": ["evaluate", "compare", "alternative", "demo", "trial", "requirement", "criteria", "shortlist", "selection", "vendor"],
    "champion": ["team", "adoption", "rollout", "training", "onboarding", "user", "daily", "workflow", "productivity", "frustrat"],
}

# Persona -> role_type mapping (controls tone via existing skill Rule #3)
_PERSONA_ROLE_TYPE: dict[str, str] = {
    "executive": "economic_buyer",
    "technical": "evaluator",
    "operations": "champion",
    "evaluator": "evaluator",
    "champion": "champion",
}

# Context fields to emphasize per persona
_PERSONA_EMPHASIS: dict[str, list[str]] = {
    "executive": ["urgency", "seat_count", "contract_end", "decision_timeline"],
    "technical": ["feature_gaps", "integration_stack", "competitors_considering"],
    "operations": ["pain_categories", "key_quotes"],
    "evaluator": ["feature_gaps", "competitors_considering", "integration_stack", "pain_categories"],
    "champion": ["pain_categories", "key_quotes", "urgency"],
}


def _build_persona_context(base_context: dict[str, Any], persona: str) -> dict[str, Any] | None:
    """Filter base company context for a specific persona.

    Returns a persona-tailored copy of the context, or None if the persona has
    no relevant pain categories (skip rule).
    """
    pain_filter = _PERSONA_PAIN_FILTER.get(persona, set())
    quote_keywords = _PERSONA_QUOTE_KEYWORDS.get(persona, [])

    # Filter pain categories
    filtered_pains = []
    for p in base_context.get("pain_categories", []):
        cat = (p.get("category") or "").lower()
        if any(f in cat for f in pain_filter):
            filtered_pains.append(p)

    # Skip rule: no relevant pain categories -> skip this persona
    if not filtered_pains:
        return None

    # Filter key quotes by persona-relevant keywords
    filtered_quotes = []
    for q in base_context.get("key_quotes", []):
        q_lower = q.lower()
        if any(kw in q_lower for kw in quote_keywords):
            filtered_quotes.append(q)
    # If no quotes matched keywords, keep top 2 from base (better than empty)
    if not filtered_quotes:
        filtered_quotes = base_context.get("key_quotes", [])[:2]

    ctx = {
        **base_context,
        "pain_categories": filtered_pains,
        "key_quotes": filtered_quotes[:5],
        "role_type": _PERSONA_ROLE_TYPE.get(persona, base_context.get("role_type")),
        "target_persona": persona,
        "emphasized_fields": _PERSONA_EMPHASIS.get(persona, []),
    }

    return ctx


# ------------------------------------------------------------------
# Partner matching
# ------------------------------------------------------------------


async def _fetch_affiliate_partners(pool) -> dict[str, Any]:
    """Fetch enabled affiliate partners, indexed by product name and category."""
    rows = await pool.fetch(
        "SELECT id, name, product_name, product_aliases, category, affiliate_url "
        "FROM affiliate_partners WHERE enabled = true"
    )
    by_product: dict[str, dict] = {}
    by_category: dict[str, list[dict]] = {}

    for r in rows:
        partner = dict(r)
        partner["id"] = str(partner["id"])
        # Index by lowercase product name + aliases
        by_product[partner["product_name"].lower()] = partner
        for alias in (partner.get("product_aliases") or []):
            by_product[alias.lower()] = partner
        # Index by lowercase category
        cat = (partner.get("category") or "").lower()
        if cat:
            by_category.setdefault(cat, []).append(partner)

    return {"by_product": by_product, "by_category": by_category}


def _match_partner(
    context: dict,
    partner_index: dict[str, Any],
) -> dict | None:
    """Match a company context to the best affiliate partner.

    Priority: (1) exact product match against competitors, (2) category fallback.
    """
    by_product = partner_index["by_product"]
    by_category = partner_index["by_category"]

    comparison_asset = context.get("comparison_asset") or {}
    comparison_vendor = str(comparison_asset.get("alternative_vendor") or "").lower()
    if comparison_vendor and comparison_vendor in by_product:
        return by_product[comparison_vendor]

    for alt in context.get("recommended_alternatives") or []:
        if not isinstance(alt, dict):
            continue
        name = str(alt.get("vendor_name") or "").lower()
        if name and name in by_product:
            return by_product[name]

    # Try exact match on competitor names
    for comp in context.get("competitors_considering", []):
        name = (comp.get("name") or "").lower()
        if name and name in by_product:
            return by_product[name]

    # Fallback: match by product category
    category = (context.get("category") or "").lower()
    if category and category in by_category:
        return by_category[category][0]

    return None


# ------------------------------------------------------------------
# LLM generation
# ------------------------------------------------------------------


async def _call_llm(
    llm,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    *,
    trace_span_name: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    usage_out: dict[str, Any] | None = None,
) -> str | None:
    """Low-level LLM call. Returns raw text or None."""
    import asyncio
    import time

    from ...pipelines.llm import _trace_cache_metrics, trace_llm_call
    from ...services.protocols import Message

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    t0 = time.monotonic()

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                llm.chat,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=float(settings.b2b_campaign.llm_timeout_seconds),
        )
        text = str(result.get("response", "") or "").strip()
        usage = result.get("usage", {})
        trace_meta = result.get("_trace_meta", {})
        cached_tokens, cache_write_tokens, billable_input_tokens = _trace_cache_metrics(
            usage if isinstance(usage, dict) else {},
            trace_meta if isinstance(trace_meta, dict) else {},
        )
        trace_llm_call(
            span_name=trace_span_name or "task.b2b_campaign_generation",
            input_tokens=int((usage or {}).get("input_tokens") or 0),
            output_tokens=int((usage or {}).get("output_tokens") or 0),
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            billable_input_tokens=billable_input_tokens,
            model=str(getattr(llm, "model", "") or getattr(llm, "model_id", "") or ""),
            provider=str(getattr(llm, "name", "") or ""),
            duration_ms=(time.monotonic() - t0) * 1000,
            metadata=trace_metadata or {},
            input_data={
                "messages": [
                    {"role": message.role, "content": message.content[:500]}
                    for message in messages
                ]
            },
            output_data={"response": text[:2000]} if text else None,
            api_endpoint=(trace_meta or {}).get("api_endpoint"),
            provider_request_id=(trace_meta or {}).get("provider_request_id"),
            ttft_ms=(trace_meta or {}).get("ttft_ms"),
            inference_time_ms=(trace_meta or {}).get("inference_time_ms"),
            queue_time_ms=(trace_meta or {}).get("queue_time_ms"),
        )
        if usage_out is not None:
            usage_out.clear()
            usage_out.update(
                {
                    "input_tokens": int((usage or {}).get("input_tokens") or 0),
                    "output_tokens": int((usage or {}).get("output_tokens") or 0),
                    "cached_tokens": int(cached_tokens or 0),
                    "cache_write_tokens": int(cache_write_tokens or 0),
                    "billable_input_tokens": int(
                        billable_input_tokens
                        if billable_input_tokens is not None
                        else int((usage or {}).get("input_tokens") or 0)
                    ),
                    "model": str(getattr(llm, "model", "") or getattr(llm, "model_id", "") or ""),
                    "provider": str(getattr(llm, "name", "") or ""),
                    "provider_request_id": (
                        str((trace_meta or {}).get("provider_request_id") or "") or None
                    ),
                }
            )
        return text or None
    except Exception as exc:
        trace_llm_call(
            span_name=trace_span_name or "task.b2b_campaign_generation",
            model=str(getattr(llm, "model", "") or getattr(llm, "model_id", "") or ""),
            provider=str(getattr(llm, "name", "") or ""),
            duration_ms=(time.monotonic() - t0) * 1000,
            status="failed",
            metadata=trace_metadata or {},
            input_data={
                "messages": [
                    {"role": message.role, "content": message.content[:500]}
                    for message in messages
                ]
            },
            error_message=str(exc)[:500],
            error_type=type(exc).__name__,
        )
        logger.exception("Campaign generation LLM call failed")
        return None


def _prepare_campaign_request_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Mutate payload with specificity context and return the first-pass request body."""
    request_payload = payload
    working_payload = dict(payload)
    channel = str(payload.get("channel") or "")
    specificity_context = _campaign_specificity_context(request_payload)
    if specificity_context:
        request_payload["_campaign_specificity_context"] = specificity_context
        proof_terms = _campaign_specificity_terms(
            _campaign_specificity_audit(
                body="",
                channel=channel,
                specificity_context=specificity_context,
            ),
            channel=channel,
        )
        if proof_terms:
            request_payload["campaign_proof_terms"] = proof_terms
            working_payload["campaign_proof_terms"] = proof_terms
    return working_payload, specificity_context


def _campaign_trace_metadata(
    payload: dict[str, Any],
    *,
    run_id: str | None,
    stage_id: str,
) -> dict[str, Any]:
    vendor_name = (
        str(payload.get("vendor_name") or "").strip()
        or str(payload.get("challenger_name") or "").strip()
        or str(payload.get("company") or "").strip()
        or str(payload.get("churning_from") or "").strip()
    )
    metadata: dict[str, Any] = {
        "stage_id": stage_id,
        "workflow": "b2b_campaign_generation",
        "channel": str(payload.get("channel") or ""),
        "target_mode": str(payload.get("target_mode") or ""),
        "tier": str(payload.get("tier") or ""),
    }
    if vendor_name:
        metadata["vendor_name"] = vendor_name
    if run_id:
        metadata["run_id"] = run_id
    return metadata


def _merge_batch_metrics(
    target: dict[str, int],
    delta: dict[str, int | str],
) -> None:
    for key in (
        "jobs",
        "submitted_items",
        "cache_prefiltered_items",
        "fallback_single_call_items",
        "completed_items",
        "failed_items",
    ):
        target[key] = int(target.get(key, 0)) + int(delta.get(key, 0) or 0)


def _is_deferred_campaign_content(content: dict[str, Any] | None) -> bool:
    return isinstance(content, dict) and bool(content.get("_deferred"))


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return value


_CAMPAIGN_BATCH_REPLAY_CONTRACT_VERSION = 1


def _safe_replay_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_replay_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_campaign_batch_replay_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    payload = entry.get("payload")
    best = entry.get("best")
    review_ids = entry.get("review_ids")
    if not isinstance(payload, dict) or not isinstance(best, dict) or not isinstance(review_ids, list):
        return None

    artifact_id = str(entry.get("artifact_id") or "").strip()
    campaign_batch_id = str(entry.get("campaign_batch_id") or "").strip()
    company_name = str(entry.get("company_name") or "").strip()
    target_mode = str(payload.get("target_mode") or "").strip()
    if not artifact_id or not campaign_batch_id or not company_name or not target_mode:
        return None

    replay_entry: dict[str, Any] = {
        "artifact_id": artifact_id,
        "campaign_batch_id": campaign_batch_id,
        "company_name": company_name,
        "payload": _json_safe(payload),
        "best": _json_safe(best),
        "review_ids": _json_safe(review_ids),
    }

    if target_mode == "churning_company":
        persona_context = entry.get("persona_context")
        if not isinstance(persona_context, dict):
            return None
        replay_entry["persona_context"] = _json_safe(persona_context)
        persona = str(entry.get("persona") or "").strip()
        if persona:
            replay_entry["persona"] = persona
        partner_id = str(entry.get("partner_id") or "").strip()
        if partner_id:
            replay_entry["partner_id"] = partner_id
    elif target_mode == "vendor_retention":
        vendor_ctx = entry.get("vendor_ctx")
        target = entry.get("target")
        followup_payload = entry.get("followup_payload")
        sequence_context = entry.get("sequence_context")
        if not all(isinstance(value, dict) for value in (vendor_ctx, target, followup_payload, sequence_context)):
            return None
        replay_entry["vendor_ctx"] = _json_safe(vendor_ctx)
        replay_entry["target"] = _json_safe(target)
        replay_entry["followup_payload"] = _json_safe(followup_payload)
        replay_entry["sequence_context"] = _json_safe(sequence_context)
    elif target_mode == "challenger_intel":
        challenger_ctx = entry.get("challenger_ctx")
        target = entry.get("target")
        followup_payload = entry.get("followup_payload")
        sequence_context = entry.get("sequence_context")
        if not all(isinstance(value, dict) for value in (challenger_ctx, target, followup_payload, sequence_context)):
            return None
        replay_entry["challenger_ctx"] = _json_safe(challenger_ctx)
        replay_entry["target"] = _json_safe(target)
        replay_entry["followup_payload"] = _json_safe(followup_payload)
        replay_entry["sequence_context"] = _json_safe(sequence_context)
    else:
        return None

    return replay_entry


def _normalize_campaign_batch_replay_entry(metadata: dict[str, Any]) -> dict[str, Any] | None:
    raw = metadata.get("replay_entry")
    if not isinstance(raw, dict):
        return None

    candidate = raw
    if _safe_replay_int(raw.get("contract_version")) == _CAMPAIGN_BATCH_REPLAY_CONTRACT_VERSION:
        nested = raw.get("entry")
        if not isinstance(nested, dict):
            return None
        candidate = nested

    replay_entry = _build_campaign_batch_replay_entry(candidate)
    if replay_entry is None:
        return None

    max_tokens = _safe_replay_int(metadata.get("_max_tokens"))
    if max_tokens is None:
        max_tokens = _safe_replay_int(candidate.get("max_tokens"))
    if max_tokens is not None:
        replay_entry["max_tokens"] = max_tokens

    temperature = _safe_replay_float(metadata.get("_temperature"))
    if temperature is None:
        temperature = _safe_replay_float(candidate.get("temperature"))
    if temperature is not None:
        replay_entry["temperature"] = temperature

    trace_metadata = metadata.get("_trace_metadata")
    if not isinstance(trace_metadata, dict):
        trace_metadata = candidate.get("trace_metadata")
    if isinstance(trace_metadata, dict) and trace_metadata:
        replay_entry["trace_metadata"] = _json_safe(trace_metadata)

    return replay_entry


def _campaign_batch_request_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    payload = entry.get("payload") or {}
    replay_entry = _build_campaign_batch_replay_entry(entry)
    metadata: dict[str, Any] = {
        "channel": payload.get("channel"),
        "target_mode": payload.get("target_mode"),
        "tier": payload.get("tier"),
        "replay_handler": "campaign_generation",
    }
    if replay_entry is not None:
        metadata["replay_entry"] = {
            "contract_version": _CAMPAIGN_BATCH_REPLAY_CONTRACT_VERSION,
            "entry": replay_entry,
        }
    return metadata


async def _run_campaign_batch(
    llm,
    system_prompt: str,
    entries: list[dict[str, Any]],
    *,
    run_id: str | None,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, int | str]]:
    """Run the first campaign attempt through Anthropic batching when eligible."""
    from ...services.b2b.anthropic_batch import (
        AnthropicBatchItem,
        mark_batch_fallback_result,
        submit_anthropic_message_batch,
        run_anthropic_message_batch,
    )
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_stage_request,
    )
    from ...services.b2b.llm_exact_cache import llm_identity
    from ...services.llm.anthropic import AnthropicLLM
    from ...services.protocols import Message
    import inspect

    async def _invoke_generate_content(
        entry: dict[str, Any],
        *,
        first_attempt_text: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        params = inspect.signature(_generate_content).parameters
        supports_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in params.values()
        )
        kwargs: dict[str, Any] = {}
        generation_usage: dict[str, Any] = {}
        if first_attempt_text is not None and ("first_attempt_text" in params or supports_kwargs):
            kwargs["first_attempt_text"] = first_attempt_text
        if "trace_span_name" in params or supports_kwargs:
            kwargs["trace_span_name"] = "task.b2b_campaign_generation"
        if "trace_metadata" in params or supports_kwargs:
            kwargs["trace_metadata"] = entry["trace_metadata"]
        if "usage_out" in params or supports_kwargs:
            kwargs["usage_out"] = generation_usage
        content = await _generate_content(
            llm,
            system_prompt,
            entry["payload"],
            entry["max_tokens"],
            entry["temperature"],
            **kwargs,
        )
        return content, generation_usage

    if not entries:
        return {}, {
            "jobs": 0,
            "submitted_items": 0,
            "cache_prefiltered_items": 0,
            "fallback_single_call_items": 0,
            "completed_items": 0,
            "failed_items": 0,
        }

    if not (
        settings.b2b_churn.anthropic_batch_enabled
        and settings.b2b_campaign.anthropic_batch_enabled
        and isinstance(llm, AnthropicLLM)
    ):
        results: dict[str, dict[str, Any] | None] = {}
        for entry in entries:
            content, _ = await _invoke_generate_content(entry)
            results[entry["custom_id"]] = content
        return results, {
            "jobs": 0,
            "submitted_items": 0,
            "cache_prefiltered_items": 0,
            "fallback_single_call_items": 0,
            "completed_items": 0,
            "failed_items": 0,
        }

    provider_name, model_name = llm_identity(llm)
    batch_items: list[AnthropicBatchItem] = []
    for entry in entries:
        payload = entry["payload"]
        working_payload, _ = _prepare_campaign_request_payload(payload)
        user_content = json.dumps(working_payload, separators=(",", ":"), default=str)
        request = prepare_b2b_exact_stage_request(
            "b2b_campaign_generation.content",
            provider=provider_name,
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=entry["max_tokens"],
            temperature=entry["temperature"],
        )
        cached = await lookup_b2b_exact_stage_text(request)
        cached_response_text = str(cached["response_text"]) if cached is not None else None
        cached_usage = dict(cached.get("usage") or {}) if cached is not None else {}
        batch_items.append(
            AnthropicBatchItem(
                custom_id=entry["custom_id"],
                artifact_type="campaign",
                artifact_id=entry["artifact_id"],
                vendor_name=entry["trace_metadata"].get("vendor_name"),
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_content),
                ],
                max_tokens=entry["max_tokens"],
                temperature=entry["temperature"],
                trace_span_name="task.b2b_campaign_generation",
                trace_metadata=entry["trace_metadata"],
                request_metadata=_campaign_batch_request_metadata(entry),
                cached_response_text=cached_response_text,
                cached_usage=cached_usage,
            )
        )

    if settings.b2b_campaign.anthropic_batch_detached_enabled:
        execution = await submit_anthropic_message_batch(
            llm=llm,
            stage_id="b2b_campaign_generation.content",
            task_name="b2b_campaign_generation",
            items=batch_items,
            run_id=run_id,
            min_batch_size=int(settings.b2b_campaign.anthropic_batch_min_items),
            batch_metadata={"phase_channels": sorted({str(entry["channel"]) for entry in entries})},
        )
        if execution.provider_batch_id:
            results: dict[str, dict[str, Any] | None] = {}
            for entry in entries:
                results[entry["custom_id"]] = {"_deferred": True}
            return results, {
                "jobs": 1,
                "submitted_items": execution.submitted_items,
                "cache_prefiltered_items": execution.cache_prefiltered_items,
                "fallback_single_call_items": execution.fallback_single_call_items,
                "completed_items": execution.completed_items,
                "failed_items": execution.failed_items,
            }
        results: dict[str, dict[str, Any] | None] = {}
        for entry in entries:
            outcome = execution.results_by_custom_id.get(entry["custom_id"])
            if outcome is not None and outcome.response_text is not None:
                content, _ = await _invoke_generate_content(
                    entry,
                    first_attempt_text=outcome.response_text,
                )
                results[entry["custom_id"]] = content
                continue
            content, fallback_usage = await _invoke_generate_content(entry)
            results[entry["custom_id"]] = content
            if outcome is not None and outcome.fallback_required:
                await mark_batch_fallback_result(
                    batch_id=execution.local_batch_id,
                    custom_id=entry["custom_id"],
                    succeeded=content is not None,
                    error_text=outcome.error_text if content is None else None,
                    response_text=json.dumps(content, separators=(",", ":"), default=str) if content else None,
                    usage=fallback_usage,
                    provider=str(fallback_usage.get("provider") or "") or None,
                    model=str(fallback_usage.get("model") or "") or None,
                    provider_request_id=(
                        str(fallback_usage.get("provider_request_id") or "") or None
                    ),
                )
        return results, {
            "jobs": 0,
            "submitted_items": execution.submitted_items,
            "cache_prefiltered_items": execution.cache_prefiltered_items,
            "fallback_single_call_items": execution.fallback_single_call_items,
            "completed_items": execution.completed_items,
            "failed_items": execution.failed_items,
        }

    execution = await run_anthropic_message_batch(
        llm=llm,
        stage_id="b2b_campaign_generation.content",
        task_name="b2b_campaign_generation",
        items=batch_items,
        run_id=run_id,
        min_batch_size=int(settings.b2b_campaign.anthropic_batch_min_items),
        batch_metadata={"phase_channels": sorted({str(entry["channel"]) for entry in entries})},
    )

    results: dict[str, dict[str, Any] | None] = {}
    for entry in entries:
        outcome = execution.results_by_custom_id.get(entry["custom_id"])
        if outcome is None:
            content, fallback_usage = await _invoke_generate_content(entry)
            results[entry["custom_id"]] = content
            await mark_batch_fallback_result(
                batch_id=execution.local_batch_id,
                custom_id=entry["custom_id"],
                succeeded=content is not None,
                error_text="missing_batch_outcome" if content is None else None,
                response_text=json.dumps(content, separators=(",", ":"), default=str) if content else None,
                usage=fallback_usage,
                provider=str(fallback_usage.get("provider") or "") or None,
                model=str(fallback_usage.get("model") or "") or None,
                provider_request_id=(
                    str(fallback_usage.get("provider_request_id") or "") or None
                ),
            )
            continue

        if outcome.response_text is not None:
            content, _ = await _invoke_generate_content(
                entry,
                first_attempt_text=outcome.response_text,
            )
            results[entry["custom_id"]] = content
            continue

        content, fallback_usage = await _invoke_generate_content(entry)
        results[entry["custom_id"]] = content
        await mark_batch_fallback_result(
            batch_id=execution.local_batch_id,
            custom_id=entry["custom_id"],
            succeeded=content is not None,
            error_text=outcome.error_text if content is None else None,
            response_text=json.dumps(content, separators=(",", ":"), default=str) if content else None,
            usage=fallback_usage,
            provider=str(fallback_usage.get("provider") or "") or None,
            model=str(fallback_usage.get("model") or "") or None,
            provider_request_id=(
                str(fallback_usage.get("provider_request_id") or "") or None
            ),
        )

    return results, {
        "jobs": 1 if execution.provider_batch_id else 0,
        "submitted_items": execution.submitted_items,
        "cache_prefiltered_items": execution.cache_prefiltered_items,
        "fallback_single_call_items": execution.fallback_single_call_items,
        "completed_items": execution.completed_items,
        "failed_items": execution.failed_items,
    }


def _normalized_batch_item_metadata(row: Any) -> dict[str, Any]:
    metadata = row.get("request_metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    return metadata if isinstance(metadata, dict) else {}


async def _mark_campaign_batch_item_applied(
    pool,
    *,
    item_id: str,
    applied_status: str,
    error: str | None = None,
) -> None:
    patch = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "applied_status": applied_status,
    }
    if error:
        patch["applied_error"] = error[:500]
    await pool.execute(
        """
        UPDATE anthropic_message_batch_items
        SET request_metadata = ((COALESCE(request_metadata, '{}'::jsonb) - 'applying_at' - 'applying_by') || $2::jsonb)
        WHERE id = $1::uuid
        """,
        item_id,
        json.dumps(patch, default=str),
    )


async def _claim_campaign_batch_item_for_apply(
    pool,
    *,
    item_id: str,
    claimer: str,
    stale_after_minutes: int = 30,
) -> bool:
    patch = {
        "applying_at": datetime.now(timezone.utc).isoformat(),
        "applying_by": claimer[:120],
    }
    row = await pool.fetchrow(
        f"""
        UPDATE anthropic_message_batch_items
        SET request_metadata = ((COALESCE(request_metadata, '{{}}'::jsonb) - 'applying_at' - 'applying_by') || $2::jsonb)
        WHERE id = $1::uuid
          AND COALESCE(request_metadata->>'applied_at', '') = ''
          AND (
                COALESCE(request_metadata->>'applying_at', '') = ''
                OR NULLIF(request_metadata->>'applying_at', '')::timestamptz < NOW() - INTERVAL '{int(stale_after_minutes)} minutes'
              )
        RETURNING id
        """,
        item_id,
        json.dumps(patch, default=str),
    )
    return row is not None


def _campaign_generation_specificity(payload: dict[str, Any]) -> dict[str, Any]:
    generation_audit = (
        payload.get("_generation_audit")
        if isinstance(payload.get("_generation_audit"), dict)
        else {}
    )
    specificity = (
        generation_audit.get("specificity")
        if isinstance(generation_audit.get("specificity"), dict)
        else {}
    )
    return specificity


async def _resolve_campaign_batch_item_content(
    row: Any,
    *,
    llm,
    system_prompt: str,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    payload = entry.get("payload") or {}
    trace_metadata = entry.get("trace_metadata")
    if not isinstance(trace_metadata, dict):
        trace_metadata = _campaign_trace_metadata(
            payload,
            run_id=str((payload.get("run_id") or "")).strip() or None,
            stage_id="b2b_campaign_generation.content",
        )
    status = str(row.get("status") or "")
    response_text = str(row.get("response_text") or "") or None
    if status in {"batch_succeeded", "cache_hit"} and response_text:
        return await _generate_content(
            llm,
            system_prompt,
            payload,
            int(entry.get("max_tokens") or settings.b2b_campaign.max_tokens),
            float(entry.get("temperature") or settings.b2b_campaign.temperature),
            first_attempt_text=response_text,
            trace_span_name="task.b2b_campaign_generation",
            trace_metadata=trace_metadata,
        )
    if status == "fallback_pending":
        content = await _generate_content(
            llm,
            system_prompt,
            payload,
            int(entry.get("max_tokens") or settings.b2b_campaign.max_tokens),
            float(entry.get("temperature") or settings.b2b_campaign.temperature),
            trace_span_name="task.b2b_campaign_generation",
            trace_metadata=trace_metadata,
        )
        from ...services.b2b.anthropic_batch import mark_batch_fallback_result

        await mark_batch_fallback_result(
            batch_id=str(row.get("batch_id")),
            custom_id=str(row.get("custom_id")),
            succeeded=content is not None,
            error_text=None if content is not None else str(row.get("error_text") or "") or "single_call_failed",
            response_text=json.dumps(content, separators=(",", ":"), default=str) if content else None,
        )
        return content
    if status == "fallback_succeeded" and response_text:
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and parsed.get("body"):
                payload["_generation_audit"] = payload.get("_generation_audit") or {"status": "succeeded", "attempts": 1}
                return parsed
        except Exception:
            pass
    return None


async def _store_churning_replayed_campaign(
    pool,
    *,
    entry: dict[str, Any],
    content: dict[str, Any],
    llm_model_name: str,
) -> dict[str, Any] | None:
    payload = entry["payload"]
    company_name = entry["company_name"]
    artifact_id = entry["artifact_id"]
    metadata = _campaign_storage_metadata(payload)
    specificity = _campaign_generation_specificity(payload)
    if str(payload.get("channel") or "") == "email_followup":
        await pool.execute(
            """
            INSERT INTO b2b_campaigns (
                company_name, vendor_name, product_category,
                opportunity_score, urgency_score, pain_categories,
                competitors_considering, seat_count, contract_end,
                decision_timeline, buying_stage, role_type,
                key_quotes, source_review_ids,
                channel, subject, body, cta,
                status, batch_id, llm_model,
                partner_id, industry, target_mode, metadata,
                score_components
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18,
                $19, $20, $21, $22, $23, $24, $25::jsonb,
                $26::jsonb
            )
            """,
            company_name,
            entry["best"]["vendor_name"],
            entry["best"].get("product_category"),
            entry["best"]["opportunity_score"],
            entry["best"].get("urgency"),
            json.dumps(entry["persona_context"].get("pain_categories", [])),
            json.dumps(entry["persona_context"].get("competitors_considering", [])),
            entry["best"].get("seat_count"),
            entry["best"].get("contract_end"),
            entry["best"].get("decision_timeline"),
            entry["best"].get("buying_stage"),
            entry["persona_context"].get("role_type"),
            json.dumps(entry["persona_context"].get("key_quotes", [])),
            entry["review_ids"],
            "email_followup",
            content.get("subject", ""),
            content.get("body", ""),
            content.get("cta", ""),
            "draft",
            entry["campaign_batch_id"],
            llm_model_name,
            _uuid.UUID(entry["partner_id"]) if entry.get("partner_id") else None,
            entry["persona_context"].get("industry"),
            "churning_company",
            json.dumps(metadata, default=str),
            json.dumps(entry["best"].get("score_components")),
        )
        await record_attempt(pool, artifact_type="campaign", artifact_id=artifact_id, attempt_no=1, stage="generation", status="succeeded", blocker_count=len(specificity.get("blocking_issues") or []), warning_count=len(specificity.get("warnings") or []), warnings=list(specificity.get("warnings") or []))
        return None

    await pool.execute(
        """
        INSERT INTO b2b_campaigns (
            company_name, vendor_name, product_category,
            opportunity_score, urgency_score, pain_categories,
            competitors_considering, seat_count, contract_end,
            decision_timeline, buying_stage, role_type,
            key_quotes, source_review_ids,
            channel, subject, body, cta,
            status, batch_id, llm_model,
            partner_id, industry, target_mode, metadata,
            score_components
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18,
            $19, $20, $21, $22, $23, $24, $25::jsonb,
            $26::jsonb
        )
        """,
        company_name,
        entry["best"]["vendor_name"],
        entry["best"].get("product_category"),
        entry["best"]["opportunity_score"],
        entry["best"].get("urgency"),
        json.dumps(entry["persona_context"].get("pain_categories", [])),
        json.dumps(entry["persona_context"].get("competitors_considering", [])),
        entry["best"].get("seat_count"),
        entry["best"].get("contract_end"),
        entry["best"].get("decision_timeline"),
        entry["best"].get("buying_stage"),
        entry["persona_context"].get("role_type"),
        json.dumps(entry["persona_context"].get("key_quotes", [])),
        entry["review_ids"],
        str(payload.get("channel") or "email_cold"),
        content.get("subject", ""),
        content.get("body", ""),
        content.get("cta", ""),
        "draft",
        entry["campaign_batch_id"],
        llm_model_name,
        _uuid.UUID(entry["partner_id"]) if entry.get("partner_id") else None,
        entry["persona_context"].get("industry"),
        "churning_company",
        json.dumps(metadata, default=str),
        json.dumps(entry["best"].get("score_components")),
    )
    await record_attempt(pool, artifact_type="campaign", artifact_id=artifact_id, attempt_no=1, stage="generation", status="succeeded", blocker_count=len(specificity.get("blocking_issues") or []), warning_count=len(specificity.get("warnings") or []), warnings=list(specificity.get("warnings") or []))
    if settings.campaign_sequence.enabled:
        try:
            await _create_sequence_for_cold_email(
                pool,
                company_name=company_name,
                batch_id=entry["campaign_batch_id"],
                partner_id=entry.get("partner_id"),
                context=entry["persona_context"],
                cold_email_subject=content.get("subject", ""),
                cold_email_body=content.get("body", ""),
            )
        except Exception as exc:
            logger.warning("Failed to create sequence for replayed campaign %s/%s: %s", company_name, entry.get("persona"), exc)
    followup_payload = {
        **entry["persona_context"],
        "channel": "email_followup",
        "target_mode": "churning_company",
        "cold_email_context": {
            "subject": content.get("subject", ""),
            "body": content.get("body", ""),
        },
    }
    return {
        "custom_id": _campaign_artifact_key(company_name=company_name, batch_id=entry["campaign_batch_id"], channel="email_followup"),
        "artifact_id": _campaign_artifact_key(company_name=company_name, batch_id=entry["campaign_batch_id"], channel="email_followup"),
        "campaign_batch_id": entry["campaign_batch_id"],
        "phase": "followup",
        "payload": followup_payload,
        "channel": "email_followup",
        "company_name": company_name,
        "persona": entry.get("persona"),
        "persona_batch_id": entry["campaign_batch_id"],
        "persona_context": entry["persona_context"],
        "best": entry["best"],
        "review_ids": entry["review_ids"],
        "partner_id": entry.get("partner_id"),
        "max_tokens": entry.get("max_tokens") or settings.b2b_campaign.max_tokens,
        "temperature": entry.get("temperature") or settings.b2b_campaign.temperature,
        "trace_metadata": _campaign_trace_metadata(
            followup_payload,
            run_id=str((entry.get("trace_metadata") or {}).get("run_id") or "").strip() or None,
            stage_id="b2b_campaign_generation.content",
        ),
    }


async def _store_vendor_retention_replayed_campaign(
    pool,
    *,
    entry: dict[str, Any],
    content: dict[str, Any],
    llm_model_name: str,
) -> dict[str, Any] | None:
    payload = entry["payload"]
    vendor_name = entry["company_name"]
    artifact_id = entry["artifact_id"]
    metadata = _campaign_storage_metadata(payload)
    specificity = _campaign_generation_specificity(payload)
    channel = str(payload.get("channel") or "email_cold")
    await pool.execute(
        """
        INSERT INTO b2b_campaigns (
            company_name, vendor_name, product_category,
            opportunity_score, urgency_score, pain_categories,
            competitors_considering, seat_count, contract_end,
            decision_timeline, buying_stage, role_type,
            key_quotes, source_review_ids,
            channel, subject, body, cta,
            status, batch_id, llm_model, industry, target_mode, metadata,
            recipient_email, score_components
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18,
            $19, $20, $21, $22, $23, $24::jsonb,
            $25, $26::jsonb
        )
        """,
        vendor_name,
        vendor_name,
        entry["best"].get("product_category"),
        entry["best"]["opportunity_score"],
        entry["best"].get("urgency"),
        json.dumps(entry["vendor_ctx"]["signal_summary"]["pain_distribution"]),
        json.dumps(entry["vendor_ctx"]["signal_summary"]["competitor_distribution"]),
        entry["best"].get("seat_count"),
        entry["best"].get("contract_end"),
        entry["best"].get("decision_timeline"),
        entry["best"].get("buying_stage"),
        _map_role_type(entry["target"].get("contact_role")),
        json.dumps(entry["vendor_ctx"].get("key_quotes", [])),
        entry["review_ids"],
        channel,
        content.get("subject", ""),
        content.get("body", ""),
        content.get("cta", ""),
        "draft",
        entry["campaign_batch_id"],
        llm_model_name,
        entry["best"].get("industry"),
        "vendor_retention",
        json.dumps(metadata, default=str),
        entry["target"].get("contact_email"),
        json.dumps(entry["best"].get("score_components")),
    )
    await record_attempt(pool, artifact_type="campaign", artifact_id=artifact_id, attempt_no=1, stage="generation", status="succeeded", blocker_count=len(specificity.get("blocking_issues") or []), warning_count=len(specificity.get("warnings") or []), warnings=list(specificity.get("warnings") or []))
    if channel == "email_followup":
        return None
    if settings.campaign_sequence.enabled:
        try:
            seq_id = await _create_sequence_for_cold_email(
                pool,
                company_name=vendor_name,
                batch_id=entry["campaign_batch_id"],
                partner_id=None,
                context=entry["sequence_context"],
                cold_email_subject=content.get("subject", ""),
                cold_email_body=content.get("body", ""),
            )
            if seq_id and entry["target"].get("contact_email"):
                await pool.execute(
                    "UPDATE campaign_sequences SET recipient_email = $1 WHERE id = $2",
                    entry["target"].get("contact_email"),
                    seq_id,
                )
        except Exception as exc:
            logger.warning("Failed to create replayed vendor sequence for %s: %s", vendor_name, exc)
    followup_payload = dict(entry["followup_payload"])
    followup_payload["cold_email_context"] = {"subject": content.get("subject", ""), "body": content.get("body", "")}
    return {
        "custom_id": _campaign_artifact_key(company_name=vendor_name, batch_id=entry["campaign_batch_id"], channel="email_followup"),
        "artifact_id": _campaign_artifact_key(company_name=vendor_name, batch_id=entry["campaign_batch_id"], channel="email_followup"),
        "campaign_batch_id": entry["campaign_batch_id"],
        "phase": "followup",
        "payload": followup_payload,
        "channel": "email_followup",
        "company_name": vendor_name,
        "best": entry["best"],
        "vendor_ctx": entry["vendor_ctx"],
        "review_ids": entry["review_ids"],
        "target": entry["target"],
        "max_tokens": entry.get("max_tokens") or settings.b2b_campaign.max_tokens,
        "temperature": entry.get("temperature") or settings.b2b_campaign.temperature,
        "trace_metadata": _campaign_trace_metadata(
            followup_payload,
            run_id=str((entry.get("trace_metadata") or {}).get("run_id") or "").strip() or None,
            stage_id="b2b_campaign_generation.content",
        ),
    }


async def _store_challenger_replayed_campaign(
    pool,
    *,
    entry: dict[str, Any],
    content: dict[str, Any],
    llm_model_name: str,
) -> dict[str, Any] | None:
    payload = entry["payload"]
    challenger_name = entry["company_name"]
    artifact_id = entry["artifact_id"]
    metadata = _campaign_storage_metadata(payload)
    specificity = _campaign_generation_specificity(payload)
    channel = str(payload.get("channel") or "email_cold")
    await pool.execute(
        """
        INSERT INTO b2b_campaigns (
            company_name, vendor_name, product_category,
            opportunity_score, urgency_score, pain_categories,
            competitors_considering, seat_count, contract_end,
            decision_timeline, buying_stage, role_type,
            key_quotes, source_review_ids,
            channel, subject, body, cta,
            status, batch_id, llm_model, industry, target_mode, metadata,
            recipient_email, score_components
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18,
            $19, $20, $21, $22, $23, $24::jsonb,
            $25, $26::jsonb
        )
        """,
        challenger_name,
        entry["best"]["vendor_name"],
        entry["best"].get("product_category"),
        entry["best"]["opportunity_score"],
        entry["best"].get("urgency"),
        json.dumps(entry["challenger_ctx"]["signal_summary"]["pain_driving_switch"]),
        json.dumps(entry["challenger_ctx"]["signal_summary"]["incumbents_losing"]),
        entry["best"].get("seat_count"),
        entry["best"].get("contract_end"),
        entry["best"].get("decision_timeline"),
        entry["best"].get("buying_stage"),
        _map_role_type(entry["target"].get("contact_role")),
        json.dumps(entry["challenger_ctx"].get("key_quotes", [])),
        entry["review_ids"],
        channel,
        content.get("subject", ""),
        content.get("body", ""),
        content.get("cta", ""),
        "draft",
        entry["campaign_batch_id"],
        llm_model_name,
        entry["best"].get("industry"),
        "challenger_intel",
        json.dumps(metadata, default=str),
        entry["target"].get("contact_email"),
        json.dumps(entry["best"].get("score_components")),
    )
    await record_attempt(pool, artifact_type="campaign", artifact_id=artifact_id, attempt_no=1, stage="generation", status="succeeded", blocker_count=len(specificity.get("blocking_issues") or []), warning_count=len(specificity.get("warnings") or []), warnings=list(specificity.get("warnings") or []))
    if channel == "email_followup":
        return None
    if settings.campaign_sequence.enabled:
        try:
            seq_id = await _create_sequence_for_cold_email(
                pool,
                company_name=challenger_name,
                batch_id=entry["campaign_batch_id"],
                partner_id=None,
                context=entry["sequence_context"],
                cold_email_subject=content.get("subject", ""),
                cold_email_body=content.get("body", ""),
            )
            if seq_id and entry["target"].get("contact_email"):
                await pool.execute(
                    "UPDATE campaign_sequences SET recipient_email = $1 WHERE id = $2",
                    entry["target"].get("contact_email"),
                    seq_id,
                )
        except Exception as exc:
            logger.warning("Failed to create replayed challenger sequence for %s: %s", challenger_name, exc)
    followup_payload = dict(entry["followup_payload"])
    followup_payload["cold_email_context"] = {"subject": content.get("subject", ""), "body": content.get("body", "")}
    return {
        "custom_id": _campaign_artifact_key(company_name=challenger_name, batch_id=entry["campaign_batch_id"], channel="email_followup"),
        "artifact_id": _campaign_artifact_key(company_name=challenger_name, batch_id=entry["campaign_batch_id"], channel="email_followup"),
        "campaign_batch_id": entry["campaign_batch_id"],
        "phase": "followup",
        "payload": followup_payload,
        "channel": "email_followup",
        "company_name": challenger_name,
        "best": entry["best"],
        "challenger_ctx": entry["challenger_ctx"],
        "review_ids": entry["review_ids"],
        "target": entry["target"],
        "max_tokens": entry.get("max_tokens") or settings.b2b_campaign.max_tokens,
        "temperature": entry.get("temperature") or settings.b2b_campaign.temperature,
        "trace_metadata": _campaign_trace_metadata(
            followup_payload,
            run_id=str((entry.get("trace_metadata") or {}).get("run_id") or "").strip() or None,
            stage_id="b2b_campaign_generation.content",
        ),
    }


async def _store_replayed_campaign_entry(
    pool,
    *,
    entry: dict[str, Any],
    content: dict[str, Any],
    llm_model_name: str,
) -> dict[str, Any] | None:
    target_mode = str(((entry.get("payload") or {}).get("target_mode")) or "")
    if target_mode == "churning_company":
        return await _store_churning_replayed_campaign(pool, entry=entry, content=content, llm_model_name=llm_model_name)
    if target_mode == "vendor_retention":
        return await _store_vendor_retention_replayed_campaign(pool, entry=entry, content=content, llm_model_name=llm_model_name)
    if target_mode == "challenger_intel":
        return await _store_challenger_replayed_campaign(pool, entry=entry, content=content, llm_model_name=llm_model_name)
    raise ValueError(f"Unsupported campaign replay target mode: {target_mode}")


async def reconcile_batches(task: ScheduledTask) -> dict[str, Any]:
    cfg = settings.b2b_campaign
    if not cfg.enabled or not cfg.anthropic_batch_detached_enabled:
        return {"_skip_synthesis": "Detached campaign batch reconciliation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    from ...services.llm_router import get_llm
    from ...skills import get_skill_registry
    from ...services.b2b.anthropic_batch import reconcile_anthropic_message_batch

    llm = get_llm("campaign")
    skill = get_skill_registry().get("digest/b2b_campaign_generation")
    if llm is None or skill is None:
        return {"reconciled_batches": 0, "applied_items": 0, "submitted_followups": 0, "error": "campaign_llm_or_skill_missing"}
    llm_model_name = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
    claimer = f"reconcile:{getattr(task, 'id', None) or 'adhoc'}:{_uuid.uuid4().hex[:10]}"

    batch_rows = await pool.fetch(
        """
        SELECT DISTINCT b.id, b.run_id, b.status
        FROM anthropic_message_batches b
        JOIN anthropic_message_batch_items i ON i.batch_id = b.id
        WHERE b.task_name = 'b2b_campaign_generation'
          AND b.provider_batch_id IS NOT NULL
          AND COALESCE(i.request_metadata->>'replay_handler', '') = 'campaign_generation'
          AND COALESCE(i.request_metadata->>'applied_at', '') = ''
        ORDER BY b.created_at ASC
        LIMIT 25
        """
    )

    reconciled_batches = 0
    applied_items = 0
    submitted_followups = 0
    failed_items = 0

    for batch_row in batch_rows:
        execution = await reconcile_anthropic_message_batch(
            llm=llm,
            local_batch_id=str(batch_row["id"]),
            pool=pool,
        )
        reconciled_batches += 1

        item_rows = await pool.fetch(
            """
            SELECT *
            FROM anthropic_message_batch_items
            WHERE batch_id = $1::uuid
            ORDER BY created_at ASC
            """,
            batch_row["id"],
        )

        followup_entries: list[dict[str, Any]] = []
        for item_row in item_rows:
            metadata = _normalized_batch_item_metadata(item_row)
            if metadata.get("replay_handler") != "campaign_generation":
                continue
            if metadata.get("applied_at"):
                continue
            entry = _normalize_campaign_batch_replay_entry(metadata)
            if not isinstance(entry, dict):
                await _mark_campaign_batch_item_applied(
                    pool,
                    item_id=str(item_row["id"]),
                    applied_status="failed",
                    error="invalid_replay_entry",
                )
                failed_items += 1
                continue

            status = str(item_row.get("status") or "")
            if status == "pending":
                continue
            claimed = await _claim_campaign_batch_item_for_apply(
                pool,
                item_id=str(item_row["id"]),
                claimer=claimer,
                stale_after_minutes=int(getattr(cfg, "anthropic_batch_stale_minutes", 30) or 30),
            )
            if not claimed:
                continue

            content = await _resolve_campaign_batch_item_content(
                item_row,
                llm=llm,
                system_prompt=skill.content,
                entry=entry,
            )
            if content is None:
                if status in {"batch_errored", "batch_expired", "batch_canceled", "fallback_failed"}:
                    await _record_campaign_generation_failure(
                        pool,
                        artifact_id=str(entry.get("artifact_id") or item_row.get("artifact_id") or ""),
                        company_name=str(entry.get("company_name") or ""),
                        channel=str(((entry.get("payload") or {}).get("channel")) or ""),
                        generation_audit={
                            "status": "failed",
                            "failure_reason": status,
                        },
                    )
                await _mark_campaign_batch_item_applied(
                    pool,
                    item_id=str(item_row["id"]),
                    applied_status="failed",
                    error=status or "no_content",
                )
                failed_items += 1
                continue

            try:
                next_entry = await _store_replayed_campaign_entry(
                    pool,
                    entry=entry,
                    content=content,
                    llm_model_name=llm_model_name,
                )
                await _mark_campaign_batch_item_applied(
                    pool,
                    item_id=str(item_row["id"]),
                    applied_status="succeeded",
                )
                applied_items += 1
                if next_entry is not None:
                    followup_entries.append(next_entry)
            except Exception as exc:
                logger.exception("Failed to apply replayed campaign batch item %s", item_row.get("custom_id"))
                await _mark_campaign_batch_item_applied(
                    pool,
                    item_id=str(item_row["id"]),
                    applied_status="failed",
                    error=str(exc),
                )
                failed_items += 1

        if followup_entries:
            _, followup_metrics = await _run_campaign_batch(
                llm,
                skill.content,
                followup_entries,
                run_id=str(batch_row.get("run_id") or "") or None,
            )
            submitted_followups += int(followup_metrics.get("submitted_items") or 0)

    return {
        "_skip_synthesis": "Campaign batch reconciliation complete",
        "reconciled_batches": reconciled_batches,
        "applied_items": applied_items,
        "submitted_followups": submitted_followups,
        "failed_items": failed_items,
    }


async def _generate_content(
    llm,
    system_prompt: str,
    payload: dict[str, Any],
    max_tokens: int,
    temperature: float,
    *,
    first_attempt_text: str | None = None,
    trace_span_name: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    usage_out: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Call LLM, validate output, retry once if word count exceeded."""
    import inspect

    from ...pipelines.llm import clean_llm_output
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_stage_request,
        store_b2b_exact_stage_text,
    )
    from ...services.b2b.llm_exact_cache import llm_identity

    request_payload = payload
    channel = payload.get("channel", "")
    tier = payload.get("tier", "")
    target_mode = payload.get("target_mode")
    last_wc = 0
    last_min = 0
    last_max = 0
    retry_reasons: list[str] = []
    working_payload, specificity_context = _prepare_campaign_request_payload(request_payload)
    provider_name, model_name = llm_identity(llm)
    cache_namespace = "b2b_campaign_generation.content"
    usage_total: dict[str, Any] = {}
    call_llm_params = inspect.signature(_call_llm).parameters
    call_llm_supports_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in call_llm_params.values()
    )

    def _accumulate_usage(sample: dict[str, Any] | None) -> None:
        if not sample:
            return
        usage_total["input_tokens"] = int(usage_total.get("input_tokens") or 0) + int(sample.get("input_tokens") or 0)
        usage_total["billable_input_tokens"] = int(usage_total.get("billable_input_tokens") or 0) + int(
            sample.get("billable_input_tokens")
            if sample.get("billable_input_tokens") is not None
            else sample.get("input_tokens") or 0
        )
        usage_total["cached_tokens"] = int(usage_total.get("cached_tokens") or 0) + int(sample.get("cached_tokens") or 0)
        usage_total["cache_write_tokens"] = int(usage_total.get("cache_write_tokens") or 0) + int(sample.get("cache_write_tokens") or 0)
        usage_total["output_tokens"] = int(usage_total.get("output_tokens") or 0) + int(sample.get("output_tokens") or 0)
        if sample.get("provider"):
            usage_total["provider"] = sample.get("provider")
        if sample.get("model"):
            usage_total["model"] = sample.get("model")
        if sample.get("provider_request_id"):
            usage_total["provider_request_id"] = sample.get("provider_request_id")

    def _write_usage_out() -> None:
        if usage_out is not None:
            usage_out.clear()
            usage_out.update(usage_total)

    for attempt in range(2):
        user_content = json.dumps(working_payload, separators=(",", ":"), default=str)

        # On retry, prepend a revision instruction
        if attempt == 1:
            revision = working_payload.pop("_revision", None)
            if revision:
                user_content = f"{revision}\n\n{user_content}"
            elif last_wc:
                if last_min and last_wc < last_min:
                    user_content = (
                        f"REVISION REQUIRED: The previous body was {last_wc} words "
                        f"but MUST be at least {last_min} words. Add one or two "
                        f"concrete sentences using the provided evidence. Return "
                        f"the same JSON format.\n\n{user_content}"
                    )
                elif last_max:
                    user_content = (
                        f"REVISION REQUIRED: The previous body was {last_wc} words "
                        f"but MUST be under {last_max} words. Rewrite the body "
                        f"shorter -- cut sentences, not words. Return the same JSON "
                        f"format.\n\n{user_content}"
                    )

        request = prepare_b2b_exact_stage_request(
            cache_namespace,
            provider=provider_name,
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cached = await lookup_b2b_exact_stage_text(request)
        text = cached["response_text"] if cached is not None else None
        if cached is not None and usage_out is not None:
            _write_usage_out()
        if text is None and attempt == 0 and first_attempt_text is not None:
            text = first_attempt_text
        if text is None:
            call_kwargs: dict[str, Any] = {}
            call_usage: dict[str, Any] = {}
            if "trace_span_name" in call_llm_params or call_llm_supports_kwargs:
                call_kwargs["trace_span_name"] = trace_span_name
            if "trace_metadata" in call_llm_params or call_llm_supports_kwargs:
                call_kwargs["trace_metadata"] = trace_metadata
            if "usage_out" in call_llm_params or call_llm_supports_kwargs:
                call_kwargs["usage_out"] = call_usage
            text = await _call_llm(
                llm,
                system_prompt,
                user_content,
                max_tokens,
                temperature,
                **call_kwargs,
            )
            _accumulate_usage(call_usage)
            _write_usage_out()
        if not text:
            _write_usage_out()
            request_payload["_generation_audit"] = {
                "status": "failed",
                "attempts": attempt + 1,
                "retry_reasons": retry_reasons,
                "failure_reason": "llm_returned_empty",
            }
            return None

        try:
            text = clean_llm_output(text)
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse campaign generation JSON: %.200s", text)
            _write_usage_out()
            request_payload["_generation_audit"] = {
                "status": "failed",
                "attempts": attempt + 1,
                "retry_reasons": retry_reasons,
                "failure_reason": "invalid_json",
            }
            return None

        if not isinstance(parsed, dict) or "body" not in parsed:
            logger.warning("Campaign generation missing 'body' field")
            _write_usage_out()
            request_payload["_generation_audit"] = {
                "status": "failed",
                "attempts": attempt + 1,
                "retry_reasons": retry_reasons,
                "failure_reason": "missing_body",
            }
            return None

        # Validate and fix
        validated, issues = _validate_campaign_content(
            parsed,
            channel,
            tier=tier,
            target_mode=target_mode,
        )

        if issues.get("missing_field"):
            logger.warning("Campaign missing required field: %s", issues["missing_field"])
            _write_usage_out()
            request_payload["_generation_audit"] = {
                "status": "failed",
                "attempts": attempt + 1,
                "retry_reasons": retry_reasons,
                "failure_reason": f"missing_field:{issues['missing_field']}",
                "validation_issues": issues,
            }
            return None

        if issues.get("placeholders"):
            logger.warning("Campaign body contains placeholder brackets, rejecting")
            _write_usage_out()
            request_payload["_generation_audit"] = {
                "status": "failed",
                "attempts": attempt + 1,
                "retry_reasons": retry_reasons,
                "failure_reason": "placeholders",
                "validation_issues": issues,
            }
            return None

        # Retry on spam trigger in subject line
        if issues.get("subject_spam_trigger") and attempt == 0:
            logger.info(
                "Subject contains spam trigger %r, retrying",
                issues["subject_spam_trigger"],
            )
            retry_reasons.append("subject_spam_trigger")
            working_payload = {**working_payload, "_revision": (
                f"REVISION REQUIRED: The subject line contains the spam "
                f"trigger word '{issues['subject_spam_trigger']}'. Rewrite "
                f"the subject line without urgency/alarm language. Use "
                f"curiosity or data-driven phrasing instead."
            )}
            continue

        if issues.get("word_count") and attempt == 0:
            # Retry with correction prompt
            last_wc = issues["word_count"]
            last_min = int(issues.get("min_words") or 0)
            last_max = int(issues.get("max_words") or 0)
            if last_min and last_wc < last_min:
                logger.info(
                    "Campaign body %d words (min %d), retrying with correction",
                    last_wc, last_min,
                )
            else:
                logger.info(
                    "Campaign body %d words (max %d), retrying with correction",
                    last_wc, last_max,
                )
            retry_reasons.append("word_count")
            continue

        if issues.get("word_count"):
            if issues.get("min_words"):
                logger.warning(
                    "Campaign body still %d words after retry, below min %d",
                    issues["word_count"], issues["min_words"],
                )
                request_payload["_generation_audit"] = {
                    "status": "failed",
                    "attempts": attempt + 1,
                    "retry_reasons": retry_reasons,
                    "failure_reason": "word_count_too_short",
                    "validation_issues": issues,
                }
                return None
            logger.warning(
                "Campaign body still %d words after retry, truncated to %d",
                issues["word_count"], issues["max_words"],
            )

        revalidation = campaign_quality_revalidation(
            campaign={
                **request_payload,
                "subject": validated.get("subject") or "",
                "body": validated.get("body") or "",
                "cta": validated.get("cta") or "",
            },
            boundary="generation",
            specificity_context=specificity_context,
        )
        request_payload["_campaign_revalidation"] = revalidation
        specificity = revalidation["audit"]
        if specificity.get("campaign_proof_terms") and not request_payload.get("campaign_proof_terms"):
            request_payload["campaign_proof_terms"] = list(specificity["campaign_proof_terms"])
            working_payload["campaign_proof_terms"] = list(specificity["campaign_proof_terms"])
        if specificity.get("blocking_issues"):
            if attempt == 0:
                retry_reasons.append("specificity")
                working_payload = {
                    **working_payload,
                    "_revision": _campaign_specificity_revision(
                        channel=channel,
                        specificity=specificity,
                    ),
                }
                continue
            logger.warning(
                "Campaign specificity gate failed: %s",
                "; ".join(specificity["blocking_issues"]),
            )
            _write_usage_out()
            request_payload["_generation_audit"] = {
                "status": "failed",
                "attempts": attempt + 1,
                "retry_reasons": retry_reasons,
                "failure_reason": "specificity_gate",
                "validation_issues": issues,
                "specificity": specificity,
            }
            return None

        validated["body"] = _append_signoff(validated["body"], request_payload)
        request_payload["_generation_audit"] = {
            "status": "succeeded",
            "attempts": attempt + 1,
            "retry_reasons": retry_reasons,
            "validation_issues": issues,
            "specificity": specificity,
        }
        await store_b2b_exact_stage_text(
            request,
            response_text=json.dumps(validated, separators=(",", ":"), default=str),
            metadata={
                "channel": channel,
                "tier": tier,
                "target_mode": target_mode,
            },
        )
        _write_usage_out()
        return validated

    _write_usage_out()
    request_payload["_generation_audit"] = {
        "status": "failed",
        "attempts": 2,
        "retry_reasons": retry_reasons,
        "failure_reason": "exhausted_retries",
    }
    return None
