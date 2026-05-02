"""Product-owned campaign opportunity input normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_EMPTY_VALUES = (None, "", [], {})
_TARGET_ID_KEYS = ("target_id", "id", "company_id", "vendor_id", "email")
_TARGET_NAME_KEYS = (
    "company_name",
    "company",
    "account_name",
    "reviewer_company",
    "customer_name",
    "vendor_name",
    "vendor",
    "incumbent_vendor",
    "current_vendor",
    "product_name",
    "name",
)
_COMPANY_KEYS = (
    "company_name",
    "company",
    "account_name",
    "reviewer_company",
    "customer_name",
    "name",
)
_VENDOR_KEYS = (
    "vendor_name",
    "vendor",
    "incumbent_vendor",
    "current_vendor",
    "product_name",
)
_CONTACT_NAME_KEYS = ("contact_name", "recipient_name", "person_name", "name")
_CONTACT_EMAIL_KEYS = ("contact_email", "recipient_email", "email")
_CONTACT_TITLE_KEYS = ("contact_title", "recipient_title", "job_title", "title")
_PAIN_KEYS = ("pain_points", "pain_categories", "pain_category", "primary_pain", "pain")
_COMPETITOR_KEYS = (
    "competitors",
    "alternative_vendors",
    "competitor",
    "competing_vendor",
)
_EVIDENCE_KEYS = (
    "evidence",
    "quote_evidence",
    "witness_highlights",
    "reasoning_witness_highlights",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = _clean_text(row.get(key))
        if value:
            return value
    return ""


def _clean_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _text_list(value: Any) -> list[str]:
    if value in _EMPTY_VALUES:
        return []
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = (value,)
    out: list[str] = []
    for item in values:
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _first_text_list(row: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    for key in keys:
        values = _text_list(row.get(key))
        if values:
            return values
    return []


def _first_non_empty(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in _EMPTY_VALUES:
            return value
    return None


def _drop_empty(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in row.items()
        if value not in _EMPTY_VALUES
    }


def opportunity_target_id(opportunity: Mapping[str, Any]) -> str:
    """Return the stable target identifier for a host-provided opportunity row."""
    for key in _TARGET_ID_KEYS:
        value = _clean_text(opportunity.get(key))
        if value:
            return value
    for key in _TARGET_NAME_KEYS:
        value = _clean_text(opportunity.get(key))
        if value:
            return value
    return ""


def normalize_campaign_opportunity(
    opportunity: Mapping[str, Any],
    *,
    target_mode: str | None = None,
) -> dict[str, Any]:
    """Normalize raw customer opportunity data into the campaign prompt contract.

    The product accepts loose host/customer rows, but prompts and saved drafts
    should see stable field names. Original non-empty fields are preserved so
    customer-specific columns still reach the prompt; canonical fields are added
    alongside them.
    """
    if not isinstance(opportunity, Mapping):
        return {}

    normalized = _drop_empty(opportunity)
    target_id = opportunity_target_id(normalized)
    if target_id:
        normalized["target_id"] = target_id

    company_name = _first_text(normalized, _COMPANY_KEYS)
    if company_name:
        normalized["company_name"] = company_name

    vendor_name = _first_text(normalized, _VENDOR_KEYS)
    if vendor_name:
        normalized["vendor_name"] = vendor_name

    contact_name = _first_text(normalized, _CONTACT_NAME_KEYS)
    contact_email = _first_text(normalized, _CONTACT_EMAIL_KEYS)
    contact_title = _first_text(normalized, _CONTACT_TITLE_KEYS)
    if contact_name:
        normalized["contact_name"] = contact_name
    if contact_email:
        normalized["contact_email"] = contact_email
    if contact_title:
        normalized["contact_title"] = contact_title

    if target_mode:
        normalized["target_mode"] = _clean_text(target_mode)

    opportunity_score = _clean_number(normalized.get("opportunity_score"))
    if opportunity_score is not None:
        normalized["opportunity_score"] = opportunity_score

    urgency_score = _clean_number(normalized.get("urgency_score"))
    if urgency_score is not None:
        normalized["urgency_score"] = urgency_score

    pain_points = _first_text_list(normalized, _PAIN_KEYS)
    if pain_points:
        normalized["pain_points"] = pain_points

    competitors = _first_text_list(normalized, _COMPETITOR_KEYS)
    if competitors:
        normalized["competitors"] = competitors

    evidence = _first_non_empty(normalized, _EVIDENCE_KEYS)
    if evidence not in _EMPTY_VALUES:
        normalized["evidence"] = evidence

    return normalized


__all__ = [
    "normalize_campaign_opportunity",
    "opportunity_target_id",
]
