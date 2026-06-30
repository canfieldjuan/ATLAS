"""Support-ticket input packages for AI Content Ops generation.

This module is the adapter between host-owned ticket ingestion and the generic
Content Ops input-provider contract. It accepts already-loaded ticket rows or a
ticket bundle, keeps customer wording intact, and returns request inputs that
the existing FAQ, landing-page, and blog planners already understand.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from .campaign_source_adapters import source_material_to_source_rows
from .content_ops_input_provider import ContentOpsInputPackage
from .support_ticket_clustering import (
    assign_support_ticket_clusters_with_diagnostics,
    support_ticket_cluster_quality,
    support_ticket_cluster_summary,
    support_ticket_plain_text,
)
from .support_ticket_context_contract import (
    SUPPORT_TICKET_DEFAULT_TOPIC,
    UPLOADED_SUPPORT_TICKETS_SOURCE_PERIOD,
    support_ticket_topic_filter,
)
from .support_ticket_dates import parse_support_ticket_source_date


DEFAULT_SUPPORT_TICKET_OUTPUTS: tuple[str, ...] = (
    "faq_markdown",
    "landing_page",
    "blog_post",
)
DEFAULT_FAQ_REPORT_CAMPAIGN_NAME = "FAQ Report"
DEFAULT_FAQ_REPORT_AUDIENCE = "Small teams answering repeat support questions"
DEFAULT_FAQ_REPORT_OFFER = (
    "Turn repeat support tickets into clear FAQ answers customers can use before "
    "they email support"
)
DEFAULT_FAQ_REPORT_CTA_LABEL = "Upload Ticket CSV -- Free Analysis"
DEFAULT_FAQ_REPORT_CTA_URL = "/systems/ai-content-ops/intake"
DEFAULT_FAQ_REPORT_TARGET_KEYWORD = "support ticket FAQ report"
DEFAULT_FAQ_REPRESENTATIVE_TAXONOMY_TERMS: tuple[str, ...] = (
    "Advertising",
    "Attempts to collect debt not owed",
    "Checking or savings account",
    "Closing an account",
    "Communication tactics",
    "Credit card or prepaid card",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Customer service",
    "Debt collection",
    "Fees or interest",
    "Getting a credit card",
    "Improper use of your report",
    "Managing an account",
    "Managing the loan or lease",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Opening an account",
    "Other service problem",
    "Other transaction problem",
    "Struggling to pay mortgage",
    "Trouble during payment process",
    "Vehicle loan or lease",
    "Wire transfer problem",
)

_QUESTION_RE = re.compile(
    r"\b(can|could|do|does|how|is|should|what|when|where|why)\b[^?]*\?",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_QUESTION_STARTS = (
    "can ",
    "could ",
    "do ",
    "does ",
    "how ",
    "is ",
    "should ",
    "what ",
    "when ",
    "where ",
    "why ",
)

_SOURCE_ID_KEYS = ("source_id", "ticket_id", "id", "case_id", "conversation_id")
_SOURCE_TITLE_KEYS = (
    "source_title",
    "subject",
    "ticket_subject",
    "request_subject",
    "conversation_title",
    "title",
    "summary",
)
_TEXT_KEYS = (
    "text",
    "body",
    "description",
    "ticket_description",
    "conversation_body",
    "initial_message",
    "question",
    "message",
    "content",
    "latest_message",
    "requester_comment",
    "customer_comment",
    "customer_message",
    "initial_comment",
    "initial_message",
    "first_message",
    "latest_customer_message",
    "last_customer_message",
    "complaint",
    "notes",
    "summary",
)
_PUBLIC_COMMENT_KEYS = (
    "comments",
    "messages",
    "thread",
    "conversation",
    "comment",
    "comment_body",
    "public_comment",
    "public_comments",
    "ticket_comments",
    "ticket_history",
    "history",
    "conversation_history",
)
_RESOLUTION_TEXT_KEYS = (
    "resolution",
    "resolution_text",
    "resolution_summary",
    "resolution_notes",
    "resolved_notes",
    "solution",
    "solution_text",
    "answer",
    "answer_text",
    "support_answer",
    "support_response",
    "support_reply",
    "agent_answer",
    "agent_response",
    "agent_reply",
    "admin_reply",
    "latest_agent_reply",
    "last_agent_reply",
    "public_agent_reply",
    "staff_reply",
    "fix_summary",
    "workaround",
)
_MEASURED_OUTCOME_KEYS = (
    "measured_outcome",
    "measured_outcomes",
    "outcome_value",
    "impact_value",
    "result_value",
    "deflection_rate",
    "ticket_reduction",
    "time_saved",
    "hours_saved",
    "resolution_rate",
    "outcome_metric",
    "impact_metric",
    "result_metric",
)
_PASSTHROUGH_KEYS = (
    "faq_draft_id",
    "faq_rank",
    "faq_question",
    "faq_topic",
    "faq_summary",
    "faq_steps",
    "faq_customer_language",
    "faq_source_ticket_ids",
    "faq_answer_evidence_status",
)
_STRUCTURED_CONTEXT_KEYS = (
    ("product", ("product", "product_name")),
    ("sub_product", ("sub_product", "sub_product_name")),
    ("issue", ("issue", "issue_type")),
    ("sub_issue", ("sub_issue", "sub_issue_type")),
)
_ROUTING_CONTEXT_KEYS = (
    ("group", ("group", "ticket_group", "support_group", "zendesk_group")),
    ("assignee", ("assignee", "assigned_to", "agent", "agent_name")),
    ("tags", ("tags", "tag", "ticket_tags", "labels")),
    ("brand", ("brand", "ticket_brand", "zendesk_brand")),
    ("organization", ("organization", "organisation", "org", "requester_organization")),
    ("product_area", ("product_area", "product area", "area")),
    (
        "custom_product_area",
        ("custom_product_area", "custom product area", "custom_area"),
    ),
)
_DATE_KEYS = (
    "created_at",
    "created_time",
    "created",
    "conversation_created_at",
    "submitted_at",
    "updated_at",
    "date",
)
_URL_KEYS = ("source_url", "ticket_url", "url", "link")
_SUPPORT_PLATFORM_KEYS = (
    "support_platform",
    "support platform",
    "platform",
    "source_platform",
    "helpdesk_platform",
)
_COMPANY_KEYS = ("company_name", "account_name", "company", "account")
_VENDOR_KEYS = ("vendor_name", "product_name", "vendor", "product")
_CONTACT_EMAIL_KEYS = (
    "contact_email",
    "contact_email_address",
    "requester_email",
    "user_email",
    "email",
    "customer_email",
)
_PAIN_KEYS = ("pain_category", "category", "intent", "topic")
_STATUS_KEYS = (
    "ticket_status",
    "issue_status",
    "case_status",
    "status",
    "ticket_state",
    "state",
)
_CSAT_KEYS = (
    "csat",
    "csat_score",
    "satisfaction_score",
    "satisfaction_rating",
    "customer_satisfaction_rating",
    "customer_satisfaction_score",
    "customer_satisfaction",
    "satisfaction",
    "rating",
)
# Canonical lifecycle buckets. Values are compared after `_key()` normalization
# (lowercase, alphanumerics only), so "In Progress" -> "inprogress". Unknown
# vocabulary maps to "other" so nothing is silently relabeled resolved/open.
_REOPENED_STATUS_VALUES = frozenset({"reopened", "reopen"})
_RESOLVED_STATUS_VALUES = frozenset({
    "resolved",
    "closed",
    "done",
    "solved",
    "complete",
    "completed",
    "fixed",
    "resolvedundermonitoring",
})
_CANCELLED_STATUS_VALUES = frozenset({
    "cancelled",
    "canceled",
    "rejected",
    "withdrawn",
    "void",
})
_OPEN_STATUS_VALUES = frozenset({
    "open",
    "new",
    "todo",
    "inprogress",
    "pending",
    "pendingcustomer",
    "pendingcustomerapproval",
    "pendingcustomerresponse",
    "awaitingcustomer",
    "customerresponse",
    "waitingoncustomer",
    "pendingdeployment",
    "waiting",
    "onhold",
    "hold",
    "underreview",
    "inreview",
    "validation",
    "monitoring",
    "testingmonitoring",
    "deployment",
    "investigating",
    "active",
})


def build_support_ticket_input_package(
    source_material: Any,
    *,
    provider: str = "support_ticket_upload",
    outputs: Sequence[str] = DEFAULT_SUPPORT_TICKET_OUTPUTS,
    window_days: int = 90,
    max_rows: int = 1000,
    campaign_name: str = DEFAULT_FAQ_REPORT_CAMPAIGN_NAME,
    audience: str = DEFAULT_FAQ_REPORT_AUDIENCE,
    offer: str = DEFAULT_FAQ_REPORT_OFFER,
    target_keyword: str = DEFAULT_FAQ_REPORT_TARGET_KEYWORD,
    cta_label: str = DEFAULT_FAQ_REPORT_CTA_LABEL,
    cta_url: str = DEFAULT_FAQ_REPORT_CTA_URL,
    secondary_keywords: Sequence[str] | None = None,
    objections: Sequence[str] | None = None,
    internal_links: Sequence[str] | None = None,
) -> ContentOpsInputPackage:
    """Build a Content Ops input package from support-ticket source material."""

    if window_days < 1:
        raise ValueError("window_days must be at least 1")
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")

    raw_rows = _rows_from_source_material(source_material)
    normalized_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not raw_rows:
        warnings.append({
            "code": "source_material_empty",
            "message": "No support-ticket source rows were provided.",
        })
    source_date_signal_count = 0
    for index, row in enumerate(raw_rows[:max_rows], start=1):
        if not isinstance(row, Mapping):
            warnings.append({
                "code": "ticket_row_not_object",
                "row_index": index,
                "message": "Skipped ticket row because it was not an object.",
            })
            continue
        normalized = _normalize_ticket_row(row, row_index=index)
        if normalized:
            normalized_rows.append(normalized)
            # Carry the date-column-present signal out-of-band rather than
            # stamping a marker onto the shared row dict (which would ride
            # through clustering and every normalized_rows consumer). A row
            # has the signal when the raw upload carried a date column, even
            # if that cell was blank -- _has_any_key supersets a parseable
            # created_at, so this reproduces the old marker exactly.
            if _has_any_key(row, _DATE_KEYS):
                source_date_signal_count += 1
            continue
        warnings.append({
            "code": "ticket_row_missing_text",
            "row_index": index,
            "message": "Skipped ticket row because it did not include customer wording.",
        })
    if len(raw_rows) > max_rows:
        warnings.append({
            "code": "ticket_rows_truncated",
            "message": f"Used first {max_rows} ticket rows out of {len(raw_rows)}.",
            "row_count": len(raw_rows),
            "max_rows": max_rows,
            "truncated_row_count": len(raw_rows) - max_rows,
        })
    skipped_row_count = len(raw_rows[:max_rows]) - len(normalized_rows)
    truncated_row_count = max(0, len(raw_rows) - max_rows)
    clustered_rows, cluster_diagnostics = (
        assign_support_ticket_clusters_with_diagnostics(normalized_rows)
    )
    normalized_rows = list(clustered_rows)
    if cluster_diagnostics["cluster_preview_skipped"]:
        warnings.append({
            "code": "cluster_preview_skipped_large_upload",
            "message": (
                "Skipped the ticket-cluster preview for "
                f"{cluster_diagnostics['token_set_row_count']} uncategorized "
                "rows; clustering above "
                f"{cluster_diagnostics['max_token_set_rows']} rows without a "
                "category column is not supported. Rows are still included "
                "in the report."
            ),
            "row_count": cluster_diagnostics["token_set_row_count"],
            "max_token_set_rows": cluster_diagnostics["max_token_set_rows"],
        })
    date_diagnostics = _source_date_diagnostics(
        normalized_rows, source_date_signal_count=source_date_signal_count
    )
    has_valid_date_window = (
        bool(normalized_rows) and date_diagnostics["missing_count"] == 0
    )
    if (
        normalized_rows
        and not has_valid_date_window
        and date_diagnostics["source_date_signal_count"] > 0
    ):
        warnings.append(_date_window_disabled_warning(date_diagnostics))
    source_period = (
        f"Last {window_days} days of support tickets"
        if has_valid_date_window
        else UPLOADED_SUPPORT_TICKETS_SOURCE_PERIOD
    )
    faq_questions = _ticket_questions(normalized_rows)
    faq_source_types = _source_types(normalized_rows)
    question_like_ticket_count = _question_like_ticket_count(normalized_rows)
    top_ticket_clusters = _top_ticket_clusters(normalized_rows)
    customer_wording_examples = _customer_wording_examples(normalized_rows)
    resolution_evidence_count = _resolution_evidence_count(normalized_rows)
    resolution_evidence_examples = _resolution_evidence_examples(normalized_rows)
    evidence_tier = _support_ticket_evidence_tier(
        normalized_rows,
        resolution_evidence_count=resolution_evidence_count,
    )
    for row in normalized_rows:
        row["support_ticket_evidence_tier"] = _support_ticket_row_evidence_tier(row)
    measured_outcome_count = _measured_outcome_count(normalized_rows)
    measured_outcome_examples = _measured_outcome_examples(normalized_rows)
    ticket_status_summary = _ticket_status_summary(normalized_rows)
    ticket_status_present_count = sum(ticket_status_summary.values())
    csat_present_count = _csat_present_count(normalized_rows)
    csat_scores = _csat_scores(normalized_rows)
    resolved_secondary_keywords = tuple(secondary_keywords or (
        "customer support FAQ",
        "repeat support questions",
        "help center answers from support tickets",
    ))
    resolved_objections = tuple(objections or (
        "Will this publish automatically?",
        "Do we need a full-time docs person?",
        "Can our team review the answers first?",
    ))

    source_material_rows = [_public_ticket_row(row) for row in normalized_rows]

    inputs = {
        "source_material": source_material_rows,
        "faq_source_types": faq_source_types,
        "faq_title": campaign_name,
        "faq_representative_taxonomy_terms": list(
            DEFAULT_FAQ_REPRESENTATIVE_TAXONOMY_TERMS
        ),
        "topic": SUPPORT_TICKET_DEFAULT_TOPIC,
        "filters": support_ticket_topic_filter(),
        "campaign_name": campaign_name,
        "offer": offer,
        "audience": audience,
        "target_keyword": target_keyword,
        "secondary_keywords": list(resolved_secondary_keywords),
        "search_intent": (
            "Small teams looking for a practical way to turn repeat support "
            "questions into help-center answers."
        ),
        "primary_entity": campaign_name,
        "audience_entity": audience,
        "competitors": ["manual help-center cleanup", "chatbot setup"],
        "objections": list(resolved_objections),
        "faq_questions": faq_questions,
        "source_period": source_period,
        "has_dated_window": has_valid_date_window,
        "source_row_count": len(raw_rows),
        "included_ticket_row_count": len(normalized_rows),
        "skipped_ticket_row_count": skipped_row_count,
        "truncated_ticket_row_count": truncated_row_count,
        "question_like_ticket_count": question_like_ticket_count,
        "top_ticket_clusters": top_ticket_clusters,
        "customer_wording_examples": customer_wording_examples,
        "support_ticket_resolution_evidence_present": resolution_evidence_count > 0,
        "support_ticket_resolution_evidence_count": resolution_evidence_count,
        "support_ticket_resolution_examples": resolution_evidence_examples,
        "support_ticket_evidence_tier": evidence_tier,
        "has_measured_outcomes": measured_outcome_count > 0,
        "measured_outcome_count": measured_outcome_count,
        "measured_outcome_examples": measured_outcome_examples,
        "internal_links": list(internal_links or (DEFAULT_FAQ_REPORT_CTA_URL,)),
        "cta_label": cta_label,
        "cta_url": cta_url,
    }
    if has_valid_date_window:
        inputs["faq_window_days"] = window_days

    metadata: dict[str, Any] = {
        "source": "support_ticket_input_package",
        "source_row_count": len(raw_rows),
        "included_row_count": len(normalized_rows),
        "skipped_row_count": skipped_row_count,
        "truncated_row_count": truncated_row_count,
        "source_period": source_period,
        "has_dated_window": has_valid_date_window,
        "support_ticket_resolution_evidence_present": resolution_evidence_count > 0,
        "support_ticket_resolution_evidence_count": resolution_evidence_count,
        "support_ticket_evidence_tier": evidence_tier,
        "has_measured_outcomes": measured_outcome_count > 0,
        "measured_outcome_count": measured_outcome_count,
        "top_ticket_clusters": top_ticket_clusters,
        "cluster_quality": support_ticket_cluster_quality(normalized_rows),
        "ticket_status_present": ticket_status_present_count > 0,
        "ticket_status_present_count": ticket_status_present_count,
        "ticket_status_summary": ticket_status_summary,
        "csat_present": csat_present_count > 0,
        "csat_present_count": csat_present_count,
        "csat_score_count": len(csat_scores),
        "csat_score_average": (
            round(sum(csat_scores) / len(csat_scores), 2) if csat_scores else None
        ),
    }
    if cluster_diagnostics["cluster_preview_skipped"]:
        metadata["cluster_preview_skipped"] = True
        metadata["cluster_preview_token_set_row_count"] = (
            cluster_diagnostics["token_set_row_count"]
        )
    return ContentOpsInputPackage(
        provider=_clean(provider) or "support_ticket_upload",
        outputs=_normalize_outputs(outputs),
        target_mode="vendor_retention",
        ingestion_profile="existing_evidence",
        inputs=inputs,
        metadata=metadata,
        warnings=tuple(warnings),
    )


def _rows_from_source_material(source_material: Any) -> list[Any]:
    if isinstance(source_material, str):
        text = _clean(source_material)
        return [{"text": text, "source_type": "support_ticket"}] if text else []
    return source_material_to_source_rows(source_material)


def _normalize_ticket_row(row: Any, *, row_index: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    source_title = support_ticket_plain_text(_first_text(row, _SOURCE_TITLE_KEYS))
    text = _ticket_text(row, source_title=source_title)
    if not text:
        return {}
    source_id = _first_text(row, _SOURCE_ID_KEYS) or f"ticket-{row_index}"
    normalized: dict[str, Any] = {
        "source_id": source_id,
        "source_type": _first_text(row, ("source_type", "type")) or "support_ticket",
        "source_title": source_title or source_id,
        "text": text,
    }
    resolution_text = _first_text(row, _RESOLUTION_TEXT_KEYS)
    if resolution_text:
        normalized["resolution_text"] = _clip_text(
            support_ticket_plain_text(resolution_text),
            max_chars=500,
        )
    measured_outcome = _evidence_text(_first_value(row, _MEASURED_OUTCOME_KEYS))
    if measured_outcome:
        normalized["measured_outcome"] = _clip_text(measured_outcome, max_chars=500)
    if _has_customer_text(row):
        normalized["_customer_text_present"] = True
    for key in _PASSTHROUGH_KEYS:
        value = row.get(key)
        if value not in (None, "", [], {}):
            normalized[key] = value
    for key, keys in _STRUCTURED_CONTEXT_KEYS:
        value = _first_value(row, keys)
        if value not in (None, "", [], {}):
            normalized[key] = value
    for key, keys in _ROUTING_CONTEXT_KEYS:
        value = _routing_context_value(_first_value(row, keys))
        if value not in (None, "", [], {}):
            normalized[key] = value
    for key, keys in (
        ("created_at", _DATE_KEYS),
        ("source_url", _URL_KEYS),
        ("support_platform", _SUPPORT_PLATFORM_KEYS),
        ("company_name", _COMPANY_KEYS),
        ("vendor_name", _VENDOR_KEYS),
        ("contact_email", _CONTACT_EMAIL_KEYS),
        ("pain_category", _PAIN_KEYS),
    ):
        value = _first_value(row, keys)
        if value not in (None, "", [], {}):
            normalized[key] = value
    status_raw = _first_text(row, _STATUS_KEYS)
    if status_raw:
        normalized["ticket_status"] = status_raw
        state = _normalize_status_state(status_raw)
        if state:
            normalized["ticket_status_state"] = state
    csat_value = _first_value(row, _CSAT_KEYS)
    csat_raw = _evidence_text(csat_value)
    if csat_raw:
        normalized["csat"] = csat_raw
        csat_score = _parse_csat_score(csat_value)
        if csat_score is not None:
            normalized["csat_score"] = csat_score
    return normalized


def _support_ticket_row_evidence_tier(row: Mapping[str, Any]) -> str:
    if _clean(row.get("resolution_text")):
        return "csv_full_thread_resolution_evidence"
    if row.get("_customer_text_present"):
        return "csv_customer_text"
    return "csv_index_metadata_only"


def _support_ticket_evidence_tier(
    rows: Sequence[Mapping[str, Any]],
    *,
    resolution_evidence_count: int,
) -> str:
    if resolution_evidence_count > 0:
        return "csv_full_thread_resolution_evidence"
    if any(row.get("_customer_text_present") for row in rows):
        return "csv_customer_text"
    return "csv_index_metadata_only"


def _has_customer_text(row: Mapping[str, Any]) -> bool:
    return bool(_first_text(row, _TEXT_KEYS) or _comments_text(row))


def _routing_context_value(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, Mapping):
        label = _first_text(value, ("name", "title", "label"))
        return label or None
    if isinstance(value, str):
        return support_ticket_plain_text(value)
    if isinstance(value, (bytes, bytearray)):
        return None
    if isinstance(value, Sequence):
        values = [
            normalized
            for item in value
            if (normalized := _routing_context_value(item)) not in (None, "", [], {})
        ]
        flattened: list[str] = []
        for item in values:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(str(item))
        return flattened
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _ticket_text(row: Mapping[str, Any], *, source_title: str) -> str:
    parts: list[str] = []
    if source_title:
        parts.append(source_title)
    body = support_ticket_plain_text(_first_text(row, _TEXT_KEYS))
    if body and body.lower() != source_title.lower():
        parts.append(body)
    comments = _comments_text(row)
    if comments:
        parts.append(comments)
    return support_ticket_plain_text("\n".join(parts))


def _comments_text(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in _PUBLIC_COMMENT_KEYS:
        comments = _first_value(row, (key,))
        if isinstance(comments, Mapping):
            comments = (comments,)
        elif isinstance(comments, str):
            comments = (comments,)
        elif not isinstance(comments, Sequence) or isinstance(
            comments,
            (bytes, bytearray),
        ):
            continue
        for item in comments:
            text = _comment_text(item)
            if text:
                parts.append(support_ticket_plain_text(text))
    return support_ticket_plain_text("\n".join(parts))


def _comment_text(item: Any) -> str:
    if isinstance(item, Mapping):
        if item.get("public") is False:
            return ""
        return _first_text(item, ("body", "message", "text", "content", "description"))
    return _clean(item)


def _ticket_questions(rows: Sequence[Mapping[str, Any]], *, limit: int = 6) -> list[str]:
    questions: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for candidate in (
            _first_question(row.get("text")),
            _question_like(row.get("source_title")),
        ):
            question = _compact(candidate)
            key = question.lower()
            if question and key not in seen:
                questions.append(question)
                seen.add(key)
            if len(questions) >= limit:
                return questions
    return questions


def _source_types(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    source_types: list[str] = []
    for row in rows:
        source_type = _clean(row.get("source_type"))
        if source_type and source_type not in source_types:
            source_types.append(source_type)
    return source_types or ["support_ticket"]


def _question_like_ticket_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if _first_question(row.get("text")) or _question_like(row.get("source_title"))
    )


def _top_ticket_clusters(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    return support_ticket_cluster_summary(rows, limit=limit)


def _customer_wording_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 6,
    max_text_chars: int = 220,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        text = _compact(row.get("text"))
        if not text:
            continue
        example = {
            "source_id": _clean(row.get("source_id")),
            "text": _clip_text(text, max_chars=max_text_chars),
        }
        source_title = _clean(row.get("source_title"))
        source_id = _clean(row.get("source_id"))
        if source_title and source_title != source_id:
            example["source_title"] = source_title
        pain_category = _clean(row.get("pain_category"))
        if pain_category:
            example["pain_category"] = pain_category
        examples.append(example)
        if len(examples) >= limit:
            return examples
    return examples


def _resolution_evidence_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if _clean(row.get("resolution_text")))


def _resolution_evidence_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 4,
    max_text_chars: int = 240,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        resolution_text = _compact(row.get("resolution_text"))
        if not resolution_text:
            continue
        example = {
            "source_id": _clean(row.get("source_id")),
            "text": _clip_text(resolution_text, max_chars=max_text_chars),
        }
        source_title = _clean(row.get("source_title"))
        source_id = _clean(row.get("source_id"))
        if source_title and source_title != source_id:
            example["source_title"] = source_title
        examples.append(example)
        if len(examples) >= limit:
            return examples
    return examples


def _measured_outcome_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if _evidence_text(row.get("measured_outcome")))


def _measured_outcome_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 4,
    max_text_chars: int = 240,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        measured_outcome = _evidence_text(row.get("measured_outcome"))
        if not measured_outcome:
            continue
        example = {
            "source_id": _clean(row.get("source_id")),
            "text": _clip_text(measured_outcome, max_chars=max_text_chars),
        }
        source_title = _clean(row.get("source_title"))
        source_id = _clean(row.get("source_id"))
        if source_title and source_title != source_id:
            example["source_title"] = source_title
        examples.append(example)
        if len(examples) >= limit:
            return examples
    return examples


def _normalize_status_state(value: Any) -> str:
    key = _key(value)
    if not key:
        return ""
    if key in _REOPENED_STATUS_VALUES:
        return "reopened"
    if key in _RESOLVED_STATUS_VALUES:
        return "resolved"
    if key in _CANCELLED_STATUS_VALUES:
        return "cancelled"
    if key in _OPEN_STATUS_VALUES:
        return "open"
    return "other"


def _parse_csat_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _ticket_status_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        state = _clean(row.get("ticket_status_state"))
        if state:
            counts[state] = counts.get(state, 0) + 1
    return counts


def _csat_present_count(rows: Sequence[Mapping[str, Any]]) -> int:
    # Presence reflects any raw CSAT value (textual good/bad included), not just
    # numeric scores -- a textual-only export must not read as absent.
    return sum(1 for row in rows if _clean(row.get("csat")))


def _csat_scores(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    scores: list[float] = []
    for row in rows:
        score = row.get("csat_score")
        if isinstance(score, bool):
            continue
        if isinstance(score, (int, float)):
            scores.append(float(score))
    return scores


def _evidence_text(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    if value in (None, "", [], {}):
        return ""
    return _WHITESPACE_RE.sub(" ", str(value)).strip()


def _clip_text(value: str, *, max_chars: int) -> str:
    text = _compact(value)
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return f"{trimmed}..."


def _all_rows_have_dates(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and _source_date_diagnostics(rows)["missing_count"] == 0


def _source_date_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_date_signal_count: int | None = None,
) -> dict[str, Any]:
    # The date-column-present signal is supplied out-of-band (computed during
    # row normalization) so it never has to ride on the row dicts. When it is
    # omitted, fall back to the created_at value alone; that under-counts
    # blank-date-column rows, which only matters for source_date_signal_count
    # (the warning trigger), not for the missing_count window gate.
    dated_count = 0
    fallback_signal_count = 0
    missing_source_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        if source_date_signal_count is None and _clean(row.get("created_at")) != "":
            fallback_signal_count += 1
        if parse_support_ticket_source_date(row.get("created_at")) is not None:
            dated_count += 1
            continue
        source_id = _clean(row.get("source_id")) or f"row-{index}"
        missing_source_ids.append(source_id)
    return {
        "included_count": len(rows),
        "dated_count": dated_count,
        "missing_count": len(rows) - dated_count,
        "source_date_signal_count": (
            source_date_signal_count
            if source_date_signal_count is not None
            else fallback_signal_count
        ),
        "example_source_ids": missing_source_ids[:5],
    }


def _public_ticket_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def _date_window_disabled_warning(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    included_count = int(diagnostics.get("included_count") or 0)
    missing_count = int(diagnostics.get("missing_count") or 0)
    return {
        "code": "support_ticket_date_window_disabled",
        "message": (
            "Disabled the dated support-ticket source window because "
            f"{missing_count} of {included_count} included ticket rows did not "
            "include a parseable source date."
        ),
        "included_row_count": included_count,
        "dated_row_count": int(diagnostics.get("dated_count") or 0),
        "missing_or_unparseable_date_count": missing_count,
        "example_source_ids": list(diagnostics.get("example_source_ids") or ()),
    }


def _parse_ticket_source_date(value: Any) -> date | None:
    return parse_support_ticket_source_date(value)


def _first_question(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    match = _QUESTION_RE.search(text)
    return match.group(0).strip() if match else _question_like(text)


def _question_like(value: Any) -> str:
    text = _compact(value)
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith(_QUESTION_STARTS):
        return text if text.endswith("?") else f"{text}?"
    return ""


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    value = _first_value(row, keys)
    return _clean(value)


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        normalized_key = _key(key)
        for raw_key, value in row.items():
            if _key(raw_key) == normalized_key and value not in (None, "", [], {}):
                return value
    return None


def _has_any_key(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    normalized_keys = {_key(key) for key in keys}
    return any(_key(raw_key) in normalized_keys for raw_key in row)


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _compact(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _normalize_outputs(outputs: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(outputs, str):
        return tuple(item.strip() for item in outputs.split(",") if item.strip())
    return tuple(str(output).strip() for output in outputs if str(output).strip())


__all__ = [
    "DEFAULT_FAQ_REPORT_AUDIENCE",
    "DEFAULT_FAQ_REPORT_CAMPAIGN_NAME",
    "DEFAULT_FAQ_REPORT_CTA_LABEL",
    "DEFAULT_FAQ_REPORT_CTA_URL",
    "DEFAULT_FAQ_REPORT_OFFER",
    "DEFAULT_FAQ_REPORT_TARGET_KEYWORD",
    "DEFAULT_SUPPORT_TICKET_OUTPUTS",
    "build_support_ticket_input_package",
]
