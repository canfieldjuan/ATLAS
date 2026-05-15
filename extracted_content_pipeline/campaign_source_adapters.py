"""Convert richer host source rows into campaign opportunities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
from typing import Any, Literal

from .campaign_customer_data import (
    CampaignOpportunityLoadResult,
    CampaignOpportunityWarning,
    normalize_campaign_opportunity_rows,
)
from .campaign_opportunities import normalize_campaign_opportunity


SourceDataFormat = Literal["auto", "json", "jsonl", "csv"]

_ROW_LIST_KEYS = (
    "sources",
    "documents",
    "reviews",
    "transcripts",
    "complaints",
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
    "review_id",
    "transcript_id",
    "document_id",
    "ticket_id",
    "case_id",
    "conversation_id",
    "survey_id",
    "response_id",
    "feedback_id",
)
_TEXT_KEYS = (
    "text",
    "review_text",
    "transcript",
    "content",
    "body",
    "quote",
    "complaint",
    "message",
    "description",
    "summary",
    "notes",
    "feedback",
    "feedback_text",
    "response_text",
    "comment_text",
    "open_ended_response",
)
_THREAD_KEYS = ("messages", "comments", "thread", "conversation", "entries")
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
_SOURCE_TYPE_KEYS = ("source_type", "type", "kind")
_SOURCE_TITLE_KEYS = ("source_title", "ticket_subject", "subject", "title", "name")
_SOURCE_TITLE_COLLISION_KEYS = ("subject", "ticket_subject", "title", "name")
_PAIN_KEYS = ("pain_points", "pain_categories", "pain_category", "topic", "category")
_PARENT_EXCLUDE_KEYS = set(_ROW_LIST_KEYS) | set(_SOURCE_TITLE_COLLISION_KEYS)
_MAX_BUNDLE_DEPTH = 8


def load_source_campaign_opportunities_from_file(
    path: str | Path,
    *,
    file_format: SourceDataFormat = "auto",
    target_mode: str | None = None,
    max_text_chars: int = 1200,
) -> CampaignOpportunityLoadResult:
    """Load review/transcript/document rows as campaign opportunities."""

    source = Path(path)
    rows = _load_source_rows(source, file_format=file_format)
    result = source_rows_to_campaign_opportunities(
        rows,
        target_mode=target_mode,
        max_text_chars=max_text_chars,
    )
    return CampaignOpportunityLoadResult(
        opportunities=result.opportunities,
        warnings=result.warnings,
        source=str(source),
    )


def source_rows_to_campaign_opportunities(
    rows: Sequence[Any],
    *,
    target_mode: str | None = None,
    max_text_chars: int = 1200,
) -> CampaignOpportunityLoadResult:
    """Normalize richer source rows into the existing opportunity contract."""

    opportunities: list[dict[str, Any]] = []
    warnings: list[CampaignOpportunityWarning] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            warnings.append(CampaignOpportunityWarning(
                code="row_not_object",
                row_index=index,
                message="Skipped source row because it is not an object.",
            ))
            continue
        opportunity, row_warnings = source_row_to_campaign_opportunity(
            row,
            row_index=index,
            max_text_chars=max_text_chars,
        )
        warnings.extend(row_warnings)
        if opportunity:
            opportunities.append(opportunity)
    normalized = normalize_campaign_opportunity_rows(
        opportunities,
        target_mode=target_mode,
    )
    return CampaignOpportunityLoadResult(
        opportunities=normalized.opportunities,
        warnings=tuple(warnings) + normalized.warnings,
    )


def source_row_to_campaign_opportunity(
    row: Mapping[str, Any],
    *,
    row_index: int | None = None,
    max_text_chars: int = 1200,
) -> tuple[dict[str, Any], tuple[CampaignOpportunityWarning, ...]]:
    """Convert one source row while preserving original non-empty fields."""

    warnings: list[CampaignOpportunityWarning] = []
    text = _source_text(row)
    if not text:
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
    source_id = _first_text(row, _SOURCE_ID_KEYS)
    source_type = _first_text(row, _SOURCE_TYPE_KEYS) or _infer_source_type(row)
    source_title = _first_text(row, _SOURCE_TITLE_KEYS)
    opportunity = {
        str(key): value
        for key, value in row.items()
        if value not in (None, "", [], {}) and key not in _SOURCE_TITLE_COLLISION_KEYS
    }
    if source_title:
        opportunity["source_title"] = source_title
    if source_id and "source_id" not in opportunity:
        opportunity["source_id"] = source_id
    if source_id and "id" not in opportunity:
        opportunity["id"] = source_id
    if source_type:
        opportunity["source_type"] = source_type
    pain_points = _first_text_list(row, _PAIN_KEYS)
    if pain_points:
        opportunity["pain_points"] = pain_points
    evidence = _source_evidence(
        row,
        text=text,
        source_id=source_id,
        source_type=source_type,
        max_text_chars=max_text_chars,
    )
    if evidence:
        opportunity["evidence"] = [evidence]
    return normalize_campaign_opportunity(opportunity), tuple(warnings)


def _load_source_rows(path: Path, *, file_format: SourceDataFormat) -> list[Any]:
    resolved_format = _resolve_format(path, file_format)
    if resolved_format == "csv":
        return _load_source_csv_rows(path)
    if resolved_format == "jsonl":
        rows: list[Any] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                rows.append(json.loads(text))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return list(data)
    if not isinstance(data, Mapping):
        raise ValueError("Source JSON must be an object or array")
    return _source_rows_from_bundle(data)


def _source_rows_from_bundle(
    bundle: Mapping[str, Any],
    *,
    parent_fields: Mapping[str, Any] | None = None,
    depth: int = 0,
) -> list[Any]:
    if depth > _MAX_BUNDLE_DEPTH:
        return []
    inherited = {
        **dict(parent_fields or {}),
        **_safe_parent_fields(bundle),
    }
    rows: list[Any] = []
    for key in _ROW_LIST_KEYS:
        value = bundle.get(key)
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


def _load_source_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rows
        for row in reader:
            rows.append({
                str(key): value
                for key, value in row.items()
                if key is not None and value not in (None, "")
            })
    return rows


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
        text = str(row.get(key) or "").strip()
        if text:
            return text
    return ""


def _source_text(row: Mapping[str, Any]) -> str:
    scalar_text = _first_text(row, _TEXT_KEYS)
    if scalar_text:
        return scalar_text
    return _thread_text(row)


def _thread_text(row: Mapping[str, Any]) -> str:
    for key in _THREAD_KEYS:
        value = row.get(key)
        lines = _thread_lines(value)
        if lines:
            return "\n".join(lines)
    return ""


def _thread_lines(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        return _thread_lines([value])
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    lines: list[str] = []
    for item in value:
        line = _thread_line(item)
        if line:
            lines.append(line)
    return lines


def _thread_line(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, Mapping):
        return ""
    text = _first_text(item, _THREAD_TEXT_KEYS)
    if not text:
        return ""
    speaker = _first_text(item, _THREAD_SPEAKER_KEYS)
    return f"{speaker}: {text}" if speaker else text


def _first_text_list(row: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    for key in keys:
        value = row.get(key)
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
    if row.get("review_text") is not None:
        return "review"
    if row.get("transcript") is not None:
        return "transcript"
    if row.get("complaint") is not None:
        return "complaint"
    if row.get("ticket_id") is not None:
        return "support_ticket"
    if row.get("case_id") is not None:
        return "case"
    if row.get("conversation_id") is not None:
        return "conversation"
    if row.get("nps_score") is not None or row.get("nps") is not None:
        return "nps_response"
    if row.get("csat_score") is not None or row.get("csat") is not None:
        return "csat_response"
    if row.get("survey_id") is not None or row.get("response_id") is not None:
        return "survey_response"
    return "document"


__all__ = [
    "SourceDataFormat",
    "load_source_campaign_opportunities_from_file",
    "source_row_to_campaign_opportunity",
    "source_rows_to_campaign_opportunities",
]
