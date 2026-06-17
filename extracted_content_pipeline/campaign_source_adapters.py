"""Convert richer host source rows into campaign opportunities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from .campaign_customer_data import (
    CampaignOpportunityLoadResult,
    CampaignOpportunityWarning,
    CsvDictRowsLoadResult,
    _load_csv_dict_rows_result,
    normalize_campaign_opportunity_rows,
)
from .campaign_opportunities import normalize_campaign_opportunity
from .faq_output_ingestion import (
    faq_output_to_source_rows,
    is_faq_output_bundle,
)


SourceDataFormat = Literal["auto", "json", "jsonl", "csv"]
_SOURCE_ADMISSION_FIELD_LIMIT = 25

_ROW_LIST_KEYS = (
    "sources",
    "opportunities",
    "documents",
    "reviews",
    "chats",
    "chat_transcripts",
    "transcripts",
    "calls",
    "call_transcripts",
    "meetings",
    "meeting_transcripts",
    "deals",
    "crm_deals",
    "crm_opportunities",
    "deal_notes",
    "opportunity_notes",
    "account_notes",
    "crm_notes",
    "activities",
    "contracts",
    "contract_notes",
    "renewals",
    "renewal_notes",
    "subscriptions",
    "subscription_notes",
    "sales_objections",
    "objections",
    "complaints",
    "search_logs",
    "site_searches",
    "search_queries",
    "zero_result_searches",
    "zero_result_queries",
    "support_tickets",
    "tickets",
    "cases",
    "conversations",
    "survey_responses",
    "surveys",
    "nps_responses",
    "csat_responses",
    "feedback",
    "rows",
    "data",
)
_SOURCE_ID_KEYS = (
    "source_id",
    "id",
    "chat_id",
    "sales_objection_id",
    "objection_id",
    "review_id",
    "transcript_id",
    "call_id",
    "meeting_id",
    "recording_id",
    "deal_id",
    "opportunity_id",
    "note_id",
    "activity_id",
    "renewal_id",
    "contract_id",
    "subscription_id",
    "document_id",
    "search_id",
    "search_log_id",
    "search_query_id",
    "query_id",
    "zero_result_query_id",
    "complaint_id",
    "ticket_id",
    "ticket_number",
    "case_id",
    "case_number",
    "conversation_id",
    "conversation_number",
    "request_id",
    "message_id",
    "survey_id",
    "response_id",
    "feedback_id",
)
_TEXT_KEYS = (
    "text",
    "chat_transcript",
    "review_text",
    "transcript",
    "content",
    "body",
    "quote",
    "search_query",
    "query",
    "query_text",
    "search_term",
    "search_terms",
    "search_phrase",
    "zero_result_query",
    "objection",
    "objection_text",
    "buyer_objection",
    "sales_objection",
    "complaint",
    "complaint_narrative",
    "consumer_complaint_narrative",
    "narrative",
    "message",
    "description",
    "issue_description",
    "ticket_description",
    "actionbody",
    "action_body",
    "requester_comment",
    "customer_comment",
    "customer_message",
    "initial_comment",
    "summary",
    "notes",
    "feedback",
    "feedback_text",
    "response_text",
    "comment_text",
    "latest_comment",
    "initial_message",
    "requester_message",
    "customer_message",
    "open_ended_response",
)
_THREAD_KEYS = (
    "messages",
    "comment",
    "comments",
    "comment_body",
    "public_comment",
    "public_comments",
    "ticket_comments",
    "ticket_history",
    "history",
    "thread",
    "conversation",
    "conversation_history",
    "entries",
    "turns",
    "segments",
    "utterances",
    "dialogue",
)
# Thread items favor message-shaped keys before generic body/content keys,
# while row-level source text keeps document/review body precedence.
_THREAD_TEXT_KEYS = (
    "text",
    "message",
    "body",
    "content",
    "comment",
    "description",
    "summary",
    "notes",
)
_THREAD_SPEAKER_KEYS = ("speaker", "author", "role", "name")
_JSON_CUSTOMER_MESSAGE_KEYS = (
    "message",
    "customer_message",
    "requester_message",
    "requester_comment",
    "customer_comment",
    "comment",
    "body",
    "text",
    "content",
    "complaint",
    "complaint_narrative",
    "consumer_complaint_narrative",
    "narrative",
    "issue_description",
)
_PRIVATE_TEXT_KEYS = (
    "internal_note",
    "internal_notes",
    "private_note",
    "private_notes",
    "internal_comment",
    "internal_comments",
    "private_comment",
    "private_comments",
)
_PRIVATE_ROW_KEYS = (
    "is_private",
    "is_internal",
)
_PUBLIC_ROW_KEYS = (
    "public",
    "is_public",
)
_SOURCE_TYPE_KEYS = ("source_type", "type", "kind")
_SOURCE_TITLE_KEYS = (
    "source_title",
    "ticket_subject",
    "ticket_title",
    "case_subject",
    "case_title",
    "subject",
    "title",
    "name",
)
_SOURCE_TITLE_COLLISION_KEYS = (
    "subject",
    "ticket_subject",
    "ticket_title",
    "case_subject",
    "case_title",
    "title",
    "name",
)
_PAIN_KEYS = ("pain_points", "pain_categories", "pain_category", "issue", "topic", "category")
_PARENT_EXCLUDE_KEYS = set(_ROW_LIST_KEYS) | set(_SOURCE_TITLE_COLLISION_KEYS)
_COMPANY_KEYS = (
    "company_name",
    "company",
    "account",
    "account_name",
    "organization",
    "organization_name",
    "requester_company",
    "requester_organization",
    "customer_company",
    "reviewer_company",
    "customer_name",
)
_VENDOR_KEYS = (
    "vendor_name",
    "vendor",
    "incumbent_vendor",
    "current_vendor",
    "product_name",
)
_CONTACT_NAME_KEYS = (
    "contact_name",
    "recipient_name",
    "person_name",
    "requester_name",
    "requester",
    "customer_contact_name",
    "user_name",
)
_CONTACT_EMAIL_KEYS = (
    "contact_email",
    "recipient_email",
    "email",
    "requester_email",
    "customer_email",
    "user_email",
)
_CONTACT_TITLE_KEYS = (
    "contact_title",
    "recipient_title",
    "job_title",
    "requester_title",
    "customer_title",
    "user_title",
)
_NPS_SCORE_KEYS = ("nps_score", "nps")
_CSAT_SCORE_KEYS = ("csat_score", "csat")
_CANONICAL_TEXT_ALIAS_KEYS = (
    ("company_name", _COMPANY_KEYS),
    ("vendor_name", _VENDOR_KEYS),
    ("contact_name", _CONTACT_NAME_KEYS),
    ("contact_email", _CONTACT_EMAIL_KEYS),
    ("contact_title", _CONTACT_TITLE_KEYS),
)
_CANONICAL_VALUE_ALIAS_KEYS = (
    ("nps_score", _NPS_SCORE_KEYS),
    ("csat_score", _CSAT_SCORE_KEYS),
)
# Ordered source-type contract for ambiguous rows. Text-bearing source fields
# win over ids, and note ids win over lifecycle ids because lifecycle ids can
# travel as context on notes without changing the evidence row type.
_SOURCE_TYPE_PRECEDENCE = (
    (("review_text",), "review"),
    ((
        "objection",
        "objection_text",
        "buyer_objection",
        "sales_objection",
        "sales_objection_id",
        "objection_id",
    ), "sales_objection"),
    (("transcript",), "transcript"),
    ((
        "search_query",
        "query",
        "query_text",
        "search_term",
        "search_terms",
        "search_phrase",
        "zero_result_query",
        "search_id",
        "search_log_id",
        "search_query_id",
        "query_id",
        "zero_result_query_id",
    ), "search_log"),
    (("chat_id",), "chat"),
    (("call_id", "recording_id"), "sales_call"),
    (("meeting_id",), "meeting"),
    (("deal_id", "opportunity_id"), "crm_deal"),
    (("note_id", "activity_id"), "crm_note"),
    (("renewal_id",), "renewal"),
    (("contract_id",), "contract"),
    (("subscription_id",), "subscription"),
    (("complaint", "complaint_id", "complaint_narrative", "consumer_complaint_narrative"), "complaint"),
    (("ticket_id", "ticket_number", "request_id"), "support_ticket"),
    (("case_id", "case_number"), "case"),
    (("conversation_id", "conversation_number"), "conversation"),
    (("nps_score", "nps"), "nps_response"),
    (("csat_score", "csat"), "csat_response"),
    (("survey_id", "response_id"), "survey_response"),
)
_FIELD_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
_MAX_BUNDLE_DEPTH = 8
_MISSING = object()


class _IndexedSourceRow(dict[str, Any]):
    """Source row dict that remembers its physical JSONL line number."""

    def __init__(self, row: Mapping[str, Any], *, source_row_index: int) -> None:
        super().__init__(row)
        self.source_row_index = source_row_index


@dataclass(frozen=True)
class SourceRowAdmissionDiagnostics:
    """Source-row admission evidence for operator-facing diagnostics."""

    input_format: str
    raw_source_row_count: int
    usable_source_row_count: int
    mapped_fields: Mapping[str, tuple[str, ...]]
    ignored_private_fields: tuple[str, ...]
    populated_unmapped_fields: tuple[str, ...]
    field_sample_limit: int = _SOURCE_ADMISSION_FIELD_LIMIT

    @property
    def usable_source_ratio(self) -> float | None:
        if self.raw_source_row_count < 1:
            return None
        return round(self.usable_source_row_count / self.raw_source_row_count, 6)

    @property
    def admission_decision(self) -> dict[str, str] | None:
        if self.input_format != "csv" or self.raw_source_row_count < 1:
            return None
        if self.usable_source_row_count < 1:
            return {
                "status": "REJECT",
                "reason": "no_usable_source_rows",
                "location": "source_row_csv",
            }
        return {"status": "ACCEPT"}

    @property
    def coverage_warnings(self) -> tuple[dict[str, Any], ...]:
        if (
            self.input_format != "csv"
            or self.raw_source_row_count < 1
            or self.usable_source_row_count < 1
            or self.usable_source_row_count >= self.raw_source_row_count
        ):
            return ()
        return ({
            "code": "partial_source_row_coverage",
            "location": "source_row_csv",
            "raw_source_row_count": self.raw_source_row_count,
            "usable_source_row_count": self.usable_source_row_count,
            "skipped_source_row_count": (
                self.raw_source_row_count - self.usable_source_row_count
            ),
            "usable_source_ratio": self.usable_source_ratio,
        },)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "input_format": self.input_format,
            "raw_source_row_count": self.raw_source_row_count,
            "usable_source_row_count": self.usable_source_row_count,
            "usable_source_ratio": self.usable_source_ratio,
            "mapped_fields": {
                key: list(values)
                for key, values in sorted(self.mapped_fields.items())
                if values
            },
            "ignored_private_fields": list(self.ignored_private_fields),
            "populated_unmapped_fields": list(self.populated_unmapped_fields),
            "field_sample_limit": self.field_sample_limit,
        }
        decision = self.admission_decision
        if decision:
            payload["admission_decision"] = decision
        warnings = self.coverage_warnings
        if warnings:
            payload["coverage_warnings"] = [dict(warning) for warning in warnings]
        return payload


@dataclass(frozen=True)
class SourceRowsLoadResult:
    """Source rows plus parser-level count metadata."""

    rows: list[Any]
    warnings: tuple[CampaignOpportunityWarning, ...]
    source_row_count: int

    @property
    def included_row_count(self) -> int:
        return len(self.rows)

    @property
    def truncated_row_count(self) -> int:
        return max(0, self.source_row_count - self.included_row_count)


class _SourceFieldLookup(Mapping[str, Any]):
    """Mapping wrapper that caches string-key alias lookups for one source row."""

    def __init__(self, row: Mapping[str, Any]) -> None:
        self._row = row
        self._indexed = tuple(
            (
                _normalized_field_key(str(raw_key)),
                _compact_field_key(str(raw_key)),
                value,
            )
            for raw_key, value in row.items()
        )
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        value = self.field_value(key)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __iter__(self):
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return key in self._row
        return self.field_value(key) is not _MISSING

    def get(self, key: str, default: Any = None) -> Any:
        value = self.field_value(key)
        return default if value is _MISSING else value

    def items(self):
        return self._row.items()

    def field_value(self, key: str) -> Any:
        if key in self._row:
            return self._row.get(key)
        if key in self._cache:
            return self._cache[key]
        normalized = _normalized_field_key(key)
        compact = _compact_field_key(key)
        for raw_normalized, raw_compact, value in self._indexed:
            if raw_normalized == normalized or raw_compact == compact:
                self._cache[key] = value
                return value
        self._cache[key] = _MISSING
        return _MISSING


def load_source_campaign_opportunities_from_file(
    path: str | Path,
    *,
    file_format: SourceDataFormat = "auto",
    target_mode: str | None = None,
    max_text_chars: int = 1200,
    default_fields: Mapping[str, Any] | None = None,
) -> CampaignOpportunityLoadResult:
    """Load review/transcript/document rows as campaign opportunities."""

    source = Path(path)
    rows, load_warnings = _load_source_rows(source, file_format=file_format)
    result = source_rows_to_campaign_opportunities(
        rows,
        target_mode=target_mode,
        max_text_chars=max_text_chars,
        default_fields=default_fields,
    )
    return CampaignOpportunityLoadResult(
        opportunities=result.opportunities,
        warnings=load_warnings + result.warnings,
        source=str(source),
    )


def load_source_rows_from_file(
    path: str | Path,
    *,
    file_format: SourceDataFormat = "auto",
) -> list[Any]:
    """Load source rows from a CSV, JSON, or JSONL file without opportunity mapping."""

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format=file_format,
    )
    malformed_jsonl_warnings = [
        warning for warning in warnings if warning.code == "malformed_jsonl_line"
    ]
    if malformed_jsonl_warnings:
        warning = malformed_jsonl_warnings[0]
        line = f"line {warning.row_index}" if warning.row_index is not None else "a line"
        raise ValueError(f"Malformed JSONL source row at {line}: {warning.message}")
    return rows


def load_source_rows_with_warnings_from_file(
    path: str | Path,
    *,
    file_format: SourceDataFormat = "auto",
) -> tuple[list[Any], tuple[CampaignOpportunityWarning, ...]]:
    """Load source rows plus non-fatal load warnings (e.g. skipped prologue rows)."""

    return _load_source_rows(Path(path), file_format=file_format)


def load_csv_source_rows_result_from_file(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> SourceRowsLoadResult:
    """Load CSV source rows with source/included/truncated row counts."""

    return _load_source_csv_rows_result(Path(path), max_rows=max_rows)


def source_rows_to_campaign_opportunities(
    rows: Sequence[Any],
    *,
    target_mode: str | None = None,
    max_text_chars: int = 1200,
    default_fields: Mapping[str, Any] | None = None,
) -> CampaignOpportunityLoadResult:
    """Normalize richer source rows into the existing opportunity contract."""

    opportunities: list[dict[str, Any]] = []
    warnings: list[CampaignOpportunityWarning] = []
    defaults = _clean_default_fields(default_fields)
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            warnings.append(CampaignOpportunityWarning(
                code="row_not_object",
                row_index=index,
                message="Skipped source row because it is not an object.",
            ))
            continue
        row_index = _source_row_input_index(row, index)
        merged_row = _merge_default_fields(defaults, row) if defaults else row
        opportunity, row_warnings = source_row_to_campaign_opportunity(
            merged_row,
            row_index=row_index,
            max_text_chars=max_text_chars,
        )
        warnings.extend(row_warnings)
        if opportunity:
            opportunities.append(_IndexedSourceRow(
                opportunity,
                source_row_index=row_index,
            ))
    normalized = normalize_campaign_opportunity_rows(
        opportunities,
        target_mode=target_mode,
    )
    return CampaignOpportunityLoadResult(
        opportunities=normalized.opportunities,
        warnings=tuple(warnings) + normalized.warnings,
    )


def source_material_to_source_rows(source_material: Any) -> list[Any]:
    """Expand a source-material object, bundle, or row list into source rows."""

    if isinstance(source_material, Mapping):
        if is_faq_output_bundle(source_material):
            return faq_output_to_source_rows(source_material)
        return _source_rows_from_bundle(source_material)
    if isinstance(source_material, Sequence) and not isinstance(
        source_material,
        (str, bytes, bytearray),
    ):
        rows: list[Any] = []
        for item in source_material:
            if isinstance(item, Mapping) and is_faq_output_bundle(item):
                rows.extend(faq_output_to_source_rows(item))
            else:
                rows.append(item)
        return rows
    return []


def parse_default_fields(values: Sequence[str] | None) -> dict[str, str]:
    """Parse repeatable ``key=value`` source-row fallback metadata."""

    out: dict[str, str] = {}
    for raw in values or ():
        if "=" not in str(raw):
            raise ValueError("--default-field values must use key=value")
        key, value = str(raw).split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("--default-field key must be non-empty")
        if value:
            out[key] = value
    return out


def parse_default_fields_or_exit(values: Sequence[str] | None) -> dict[str, str]:
    """Parse CLI defaults and return a concise command-line validation error."""

    try:
        return parse_default_fields(values)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def parse_default_fields_with_booking_url_or_exit(
    values: Sequence[str] | None,
    *,
    booking_url: str | None = None,
) -> dict[str, Any]:
    """Parse CLI defaults and add an optional selling booking URL."""

    defaults: dict[str, Any] = parse_default_fields_or_exit(values)
    cleaned_url = str(booking_url or "").strip()
    if not cleaned_url:
        return defaults
    selling = defaults.get("selling")
    selling_defaults = dict(selling) if isinstance(selling, Mapping) else {}
    defaults["selling"] = {
        **selling_defaults,
        "booking_url": cleaned_url,
    }
    return defaults


def build_source_row_admission_diagnostics(
    rows: Sequence[Any],
    *,
    input_format: str,
    usable_source_row_count: int,
    field_sample_limit: int = _SOURCE_ADMISSION_FIELD_LIMIT,
) -> SourceRowAdmissionDiagnostics:
    """Classify source-row fields using the same aliases as extraction."""

    populated_fields = _populated_source_fields(rows)
    mapped: dict[str, tuple[str, ...]] = {}
    private_fields = _matching_fields(populated_fields, _PRIVATE_TEXT_KEYS)
    categories = (
        ("source_id", _SOURCE_ID_KEYS),
        ("source_text", _TEXT_KEYS),
        ("thread_text", _THREAD_KEYS),
        ("source_title", _SOURCE_TITLE_KEYS),
        ("source_type", _SOURCE_TYPE_KEYS),
        ("company", _COMPANY_KEYS),
        ("vendor", _VENDOR_KEYS),
        ("contact_name", _CONTACT_NAME_KEYS),
        ("contact_email", _CONTACT_EMAIL_KEYS),
        ("contact_title", _CONTACT_TITLE_KEYS),
        ("nps_score", _NPS_SCORE_KEYS),
        ("csat_score", _CSAT_SCORE_KEYS),
        ("pain_points", _PAIN_KEYS),
    )
    matched_fields_by_category: dict[str, tuple[str, ...]] = {}
    for category, aliases in categories:
        fields = _matching_fields(populated_fields, aliases)
        if fields:
            matched_fields_by_category[category] = fields
            mapped[category] = fields[:field_sample_limit]
    known = set(private_fields)
    for fields in matched_fields_by_category.values():
        known.update(fields)
    unmapped = tuple(field for field in populated_fields if field not in known)
    return SourceRowAdmissionDiagnostics(
        input_format=input_format,
        raw_source_row_count=len(rows),
        usable_source_row_count=max(0, int(usable_source_row_count)),
        mapped_fields={key: value for key, value in mapped.items()},
        ignored_private_fields=private_fields[:field_sample_limit],
        populated_unmapped_fields=unmapped[:field_sample_limit],
        field_sample_limit=field_sample_limit,
    )


def _clean_default_fields(default_fields: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in (default_fields or {}).items()
        if str(key).strip() and value not in (None, "", [], {})
    }


def _merge_default_fields(
    defaults: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    row_values = {
        str(key): value
        for key, value in row.items()
        if value not in (None, "", [], {})
    }
    row_canonical_keys = _row_canonical_aliases(row_values)
    return {
        **{
            str(key): value
            for key, value in defaults.items()
            if (_canonical_alias_key(str(key)) or str(key)) not in row_canonical_keys
        },
        **row_values,
    }


def _populated_source_fields(rows: Sequence[Any]) -> tuple[str, ...]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key, value in row.items():
            field = str(key).strip()
            if not field or field in seen or value in (None, "", [], {}):
                continue
            seen.add(field)
            fields.append(field)
    return tuple(fields)


def _source_row_input_index(row: Mapping[str, Any], fallback: int) -> int:
    raw_index = getattr(row, "source_row_index", None)
    if isinstance(raw_index, int) and raw_index > 0:
        return raw_index
    return fallback


def _matching_fields(fields: Sequence[str], aliases: Sequence[str]) -> tuple[str, ...]:
    return tuple(field for field in fields if _field_matches_aliases(field, aliases))


def _field_matches_aliases(field: str, aliases: Sequence[str]) -> bool:
    normalized = _normalized_field_key(field)
    compact = _compact_field_key(field)
    for alias in aliases:
        if (
            _normalized_field_key(alias) == normalized
            or _compact_field_key(alias) == compact
        ):
            return True
    return False


def _canonical_alias_key(key: str) -> str | None:
    normalized_key = _normalized_field_key(key)
    compact_key = _compact_field_key(key)
    for canonical_key, keys in (
        *_CANONICAL_TEXT_ALIAS_KEYS,
        *_CANONICAL_VALUE_ALIAS_KEYS,
    ):
        for alias in keys:
            if (
                _normalized_field_key(alias) == normalized_key
                or _compact_field_key(alias) == compact_key
            ):
                return canonical_key
    return None


def _row_canonical_aliases(row: Mapping[str, Any]) -> set[str]:
    lookup = _SourceFieldLookup(row)
    out: set[str] = set()
    for canonical_key, keys in _CANONICAL_TEXT_ALIAS_KEYS:
        if _first_text(lookup, keys):
            out.add(canonical_key)
    for canonical_key, keys in _CANONICAL_VALUE_ALIAS_KEYS:
        if _has_field_value(lookup, keys):
            out.add(canonical_key)
    return out


def source_row_to_campaign_opportunity(
    row: Mapping[str, Any],
    *,
    row_index: int | None = None,
    max_text_chars: int = 1200,
) -> tuple[dict[str, Any], tuple[CampaignOpportunityWarning, ...]]:
    """Convert one source row while preserving original non-empty fields."""

    lookup = _SourceFieldLookup(row)
    warnings: list[CampaignOpportunityWarning] = []
    if _is_private_source_row(lookup):
        warnings.append(CampaignOpportunityWarning(
            code="private_source_text",
            row_index=row_index,
            field="text",
            message="Skipped source row because it is marked private/internal.",
        ))
        return {}, tuple(warnings)
    text, machine_payload_seen = _source_text(lookup)
    if not text:
        if machine_payload_seen:
            warnings.append(CampaignOpportunityWarning(
                code="machine_source_payload_text",
                row_index=row_index,
                field="text",
                message=(
                    "Skipped source row because the mapped text field contains a "
                    "machine JSON payload, not customer wording."
                ),
            ))
        else:
            warnings.append(CampaignOpportunityWarning(
                code="missing_source_text",
                row_index=row_index,
                field="text",
                message=(
                    "Source row did not contain text, review_text, transcript, "
                    "content, body, quote, complaint, message, description, "
                    "summary, notes, or thread messages."
                ),
            ))
        return {}, tuple(warnings)
    source_id = _first_text(lookup, _SOURCE_ID_KEYS)
    source_type = _first_text(lookup, _SOURCE_TYPE_KEYS) or _infer_source_type(lookup)
    source_title = _first_text(lookup, _SOURCE_TITLE_KEYS)
    opportunity: dict[str, Any] = {}
    for key, value in row.items():
        key_text = str(key)
        if (
            value in (None, "", [], {})
            or _is_source_title_collision_key(key_text)
            or _is_private_source_text_key(key_text)
        ):
            continue
        preserved_value = _preserved_source_value(key_text, value)
        if preserved_value not in (None, "", [], {}):
            opportunity[key_text] = preserved_value
    _copy_alias_text(opportunity, lookup, "company_name", _COMPANY_KEYS)
    _copy_alias_text(opportunity, lookup, "vendor_name", _VENDOR_KEYS)
    _copy_alias_text(opportunity, lookup, "contact_name", _CONTACT_NAME_KEYS)
    _copy_alias_text(opportunity, lookup, "contact_email", _CONTACT_EMAIL_KEYS)
    _copy_alias_text(opportunity, lookup, "contact_title", _CONTACT_TITLE_KEYS)
    _copy_alias_value(opportunity, lookup, "nps_score", _NPS_SCORE_KEYS)
    _copy_alias_value(opportunity, lookup, "csat_score", _CSAT_SCORE_KEYS)
    if source_title:
        opportunity["source_title"] = source_title
    if source_id and "source_id" not in opportunity:
        opportunity["source_id"] = source_id
    if source_id and "id" not in opportunity:
        opportunity["id"] = source_id
    if source_type:
        opportunity["source_type"] = source_type
    pain_points = _first_text_list(lookup, _PAIN_KEYS)
    if pain_points:
        opportunity["pain_points"] = pain_points
    evidence = _source_evidence(
        lookup,
        text=text,
        source_id=source_id,
        source_type=source_type,
        max_text_chars=max_text_chars,
    )
    if evidence:
        opportunity["evidence"] = [evidence]
    return normalize_campaign_opportunity(opportunity), tuple(warnings)


def _load_source_rows(
    path: Path,
    *,
    file_format: SourceDataFormat,
) -> tuple[list[Any], tuple[CampaignOpportunityWarning, ...]]:
    resolved_format = _resolve_format(path, file_format)
    if resolved_format == "csv":
        return _load_source_csv_rows(path)
    if resolved_format == "jsonl":
        rows: list[Any] = []
        warnings: list[CampaignOpportunityWarning] = []
        for line_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            text = line.strip()
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, Mapping):
                        parsed = _IndexedSourceRow(parsed, source_row_index=line_index)
                    rows.append(parsed)
                except json.JSONDecodeError as exc:
                    warnings.append(CampaignOpportunityWarning(
                        code="malformed_jsonl_line",
                        row_index=line_index,
                        field="jsonl",
                        message=(
                            "Skipped JSONL source row because it is not valid "
                            f"JSON ({exc.msg} at column {exc.colno})."
                        ),
                    ))
        return rows, tuple(warnings)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return list(data), ()
    if not isinstance(data, Mapping):
        raise ValueError("Source JSON must be an object or array")
    return _source_rows_from_bundle(data), ()


def _source_rows_from_bundle(
    bundle: Mapping[str, Any],
    *,
    parent_fields: Mapping[str, Any] | None = None,
    depth: int = 0,
) -> list[Any]:
    if depth > _MAX_BUNDLE_DEPTH:
        return []
    lookup = _SourceFieldLookup(bundle)
    inherited = {
        **dict(parent_fields or {}),
        **_safe_parent_fields(bundle),
    }
    rows: list[Any] = []
    for key in _ROW_LIST_KEYS:
        value = _field_value(lookup, key)
        if isinstance(value, Mapping):
            rows.extend(_source_rows_from_bundle(
                value,
                parent_fields=inherited,
                depth=depth + 1,
            ))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rows.extend(_rows_with_parent_fields(value, inherited))
    if rows:
        return rows
    if parent_fields:
        return [{**dict(parent_fields), **dict(bundle)}]
    return [dict(bundle)]


def _rows_with_parent_fields(
    rows: Sequence[Any],
    parent_fields: Mapping[str, Any],
) -> list[Any]:
    out: list[Any] = []
    for row in rows:
        if isinstance(row, Mapping):
            out.append({**dict(parent_fields), **dict(row)})
        else:
            out.append(row)
    return out


def _safe_parent_fields(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in bundle.items()
        if key not in _PARENT_EXCLUDE_KEYS
        and _is_safe_parent_value(value)
    }


def _is_safe_parent_value(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, Mapping)):
        return all(isinstance(item, (str, int, float, bool)) for item in value)
    return False


def _load_source_csv_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], tuple[CampaignOpportunityWarning, ...]]:
    result = _load_source_csv_rows_result(path)
    return result.rows, result.warnings


def _load_source_csv_rows_result(
    path: Path,
    *,
    max_rows: int | None = None,
) -> SourceRowsLoadResult:
    result: CsvDictRowsLoadResult = _load_csv_dict_rows_result(
        path,
        max_rows=max_rows,
    )
    return SourceRowsLoadResult(
        rows=list(result.rows),
        warnings=result.warnings,
        source_row_count=result.source_row_count,
    )


def _resolve_format(
    path: Path,
    file_format: SourceDataFormat,
) -> Literal["json", "jsonl", "csv"]:
    if file_format != "auto":
        return file_format
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    return "json"


def _source_evidence(
    row: Mapping[str, Any],
    *,
    text: str,
    source_id: str,
    source_type: str,
    max_text_chars: int,
) -> dict[str, Any]:
    if not text:
        return {}
    evidence: dict[str, Any] = {
        "text": text[:max_text_chars],
    }
    if source_id:
        evidence["source_id"] = source_id
    if source_type:
        evidence["source_type"] = source_type
    source_title = _first_text(row, _SOURCE_TITLE_KEYS)
    if source_title:
        evidence["source_title"] = source_title
    return evidence


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        text = str(_field_value(row, key) or "").strip()
        if text:
            return text
    return ""


def _source_text(row: Mapping[str, Any]) -> tuple[str, bool]:
    machine_payload_seen = False
    for key in _TEXT_KEYS:
        value = _field_value(row, key)
        text, is_machine_payload = _source_text_value(value)
        if text:
            return text, machine_payload_seen
        if is_machine_payload:
            machine_payload_seen = True
    thread_text, thread_machine_payload_seen = _thread_text(row)
    return thread_text, machine_payload_seen or thread_machine_payload_seen


def _source_text_value(value: Any) -> tuple[str, bool]:
    if value in (None, "", [], {}):
        return "", False
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "", False
        extracted = _customer_message_from_json_payload(text)
        if extracted:
            return extracted, False
        return ("", True) if _is_machine_payload_text(text) else (text, False)
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray))
    ):
        extracted = _customer_message_from_json_value(value)
        return (extracted, False) if extracted else ("", True)
    text = str(value or "").strip()
    return text, False


def _is_machine_payload_text(text: str) -> bool:
    raw = text.strip()
    if not _looks_like_json_container(raw):
        return False
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False
    return _is_machine_payload_value(parsed)


def _customer_message_from_json_payload(text: str) -> str:
    raw = text.strip()
    if not _looks_like_json_container(raw):
        return ""
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return ""
    return _customer_message_from_json_value(parsed)


def _looks_like_json_container(text: str) -> bool:
    if not (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
    ):
        return False
    return True


def _is_machine_payload_value(value: Any) -> bool:
    if not isinstance(value, (Mapping, Sequence)) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return False
    if _customer_message_from_json_value(value):
        return False
    strings = tuple(_json_string_values(value))
    if not strings:
        return True
    return True


def _customer_message_from_json_value(value: Any) -> str:
    for text in _json_customer_message_values(value):
        stripped = text.strip()
        if stripped:
            return stripped
    return ""


def _json_customer_message_values(value: Any, *, limit: int = 20) -> tuple[str, ...]:
    out: list[str] = []

    def visit(item: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if _field_matches_aliases(str(key), _JSON_CUSTOMER_MESSAGE_KEYS):
                    if isinstance(nested, str):
                        text = nested.strip()
                        if text:
                            out.append(text)
                        if len(out) >= limit:
                            return
                    elif isinstance(nested, (Mapping, Sequence)) and not isinstance(
                        nested,
                        (bytes, bytearray),
                    ):
                        visit(nested)
                        if len(out) >= limit:
                            return
                if isinstance(nested, (Mapping, Sequence)) and not isinstance(
                    nested,
                    (str, bytes, bytearray),
                ):
                    visit(nested)
                    if len(out) >= limit:
                        return
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for nested in item:
                visit(nested)
                if len(out) >= limit:
                    return

    visit(value)
    return tuple(out)


def _json_string_values(value: Any, *, limit: int = 50) -> tuple[str, ...]:
    out: list[str] = []

    def visit(item: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
                if len(out) >= limit:
                    return
            return
        if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            for nested in item:
                visit(nested)
                if len(out) >= limit:
                    return

    visit(value)
    return tuple(out)


def _thread_text(row: Mapping[str, Any]) -> tuple[str, bool]:
    machine_payload_seen = False
    for key in _THREAD_KEYS:
        value = row.get(key)
        lines, line_machine_payload_seen = _thread_lines(value)
        if lines:
            return "\n".join(lines), machine_payload_seen
        if line_machine_payload_seen:
            machine_payload_seen = True
    return "", machine_payload_seen


def _thread_lines(value: Any) -> tuple[list[str], bool]:
    if value in (None, "", [], {}):
        return [], False
    if isinstance(value, str):
        text = value.strip()
        return ([text], False) if text else ([], False)
    if isinstance(value, Mapping):
        return _thread_lines([value])
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return [], False
    lines: list[str] = []
    machine_payload_seen = False
    for item in value:
        line, is_machine_payload = _thread_line(item)
        if line:
            lines.append(line)
        if is_machine_payload:
            machine_payload_seen = True
    return lines, machine_payload_seen


def _thread_line(item: Any) -> tuple[str, bool]:
    if isinstance(item, str):
        return item.strip(), False
    if not isinstance(item, Mapping):
        return "", False
    if item.get("public") is False:
        return "", False
    text, machine_payload_seen = _source_text(item)
    if not text:
        return "", machine_payload_seen
    speaker = _first_text(item, _THREAD_SPEAKER_KEYS)
    return (f"{speaker}: {text}" if speaker else text), machine_payload_seen


def _first_text_list(row: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    for key in keys:
        value = _field_value(row, key)
        values = _text_list(value)
        if values:
            return values
    return []


def _text_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        items = (value,)
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _infer_source_type(row: Mapping[str, Any]) -> str:
    for keys, source_type in _SOURCE_TYPE_PRECEDENCE:
        if _has_field_value(row, keys):
            return source_type
    return "document"


def _copy_alias_text(
    opportunity: dict[str, Any],
    row: Mapping[str, Any],
    canonical_key: str,
    keys: Sequence[str],
) -> None:
    if opportunity.get(canonical_key):
        return
    text = _first_text(row, keys)
    if text:
        opportunity[canonical_key] = text


def _copy_alias_value(
    opportunity: dict[str, Any],
    row: Mapping[str, Any],
    canonical_key: str,
    keys: Sequence[str],
) -> None:
    if opportunity.get(canonical_key) not in (None, "", [], {}):
        return
    for key in keys:
        value = _field_value(row, key)
        if value not in (None, "", [], {}):
            opportunity[canonical_key] = value
            return


def _is_source_title_collision_key(key: str) -> bool:
    normalized_key = _normalized_field_key(key)
    compact_key = _compact_field_key(key)
    for alias in _SOURCE_TITLE_COLLISION_KEYS:
        if (
            _normalized_field_key(alias) == normalized_key
            or _compact_field_key(alias) == compact_key
        ):
            return True
    return False


def _is_private_source_text_key(key: str) -> bool:
    normalized_key = _normalized_field_key(key)
    compact_key = _compact_field_key(key)
    for alias in _PRIVATE_TEXT_KEYS:
        if (
            _normalized_field_key(alias) == normalized_key
            or _compact_field_key(alias) == compact_key
        ):
            return True
    return False


def _is_private_source_row(row: Mapping[str, Any]) -> bool:
    for key in _PRIVATE_ROW_KEYS:
        if _is_truthy_marker(_field_value(row, key)):
            return True
    for key in _PUBLIC_ROW_KEYS:
        if _is_falsey_marker(_field_value(row, key)):
            return True
    return False


def _is_truthy_marker(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def _is_falsey_marker(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    text = str(value if value is not None else "").strip().lower()
    return text in {"0", "0.0", "false", "no", "n"}


def _preserved_source_value(key: str, value: Any) -> Any:
    if not _is_thread_source_text_key(key):
        return value
    return _public_thread_value(value)


def _is_thread_source_text_key(key: str) -> bool:
    normalized_key = _normalized_field_key(key)
    compact_key = _compact_field_key(key)
    for alias in _THREAD_KEYS:
        if (
            _normalized_field_key(alias) == normalized_key
            or _compact_field_key(alias) == compact_key
        ):
            return True
    return False


def _public_thread_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return None if value.get("public") is False else dict(value)
    if isinstance(value, str) or not isinstance(value, Sequence) or isinstance(
        value,
        (bytes, bytearray),
    ):
        return value
    out: list[Any] = []
    for item in value:
        public_item = _public_thread_value(item)
        if public_item not in (None, "", [], {}):
            out.append(public_item)
    return out


def _field_value(row: Mapping[str, Any], key: str) -> Any:
    if isinstance(row, _SourceFieldLookup):
        value = row.field_value(key)
        return None if value is _MISSING else value
    if key in row:
        return row.get(key)
    normalized = _normalized_field_key(key)
    compact = _compact_field_key(key)
    for raw_key, value in row.items():
        raw_text = str(raw_key)
        if (
            _normalized_field_key(raw_text) == normalized
            or _compact_field_key(raw_text) == compact
        ):
            return value
    return None


def _has_field_value(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    for key in keys:
        if _field_value(row, key) not in (None, "", [], {}):
            return True
    return False


def _normalized_field_key(key: str) -> str:
    return _FIELD_SEPARATOR_RE.sub("_", key.strip().lower()).strip("_")


def _compact_field_key(key: str) -> str:
    return _FIELD_SEPARATOR_RE.sub("", key.strip().lower())


__all__ = [
    "SourceDataFormat",
    "SourceRowsLoadResult",
    "SourceRowAdmissionDiagnostics",
    "build_source_row_admission_diagnostics",
    "load_csv_source_rows_result_from_file",
    "load_source_campaign_opportunities_from_file",
    "load_source_rows_from_file",
    "load_source_rows_with_warnings_from_file",
    "parse_default_fields",
    "parse_default_fields_with_booking_url_or_exit",
    "parse_default_fields_or_exit",
    "source_material_to_source_rows",
    "source_row_to_campaign_opportunity",
    "source_rows_to_campaign_opportunities",
]
