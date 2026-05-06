"""Convert richer host source rows into campaign opportunities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Literal

from .campaign_customer_data import (
    CampaignOpportunityLoadResult,
    CampaignOpportunityWarning,
    normalize_campaign_opportunity_rows,
)
from .campaign_opportunities import normalize_campaign_opportunity


SourceDataFormat = Literal["auto", "json", "jsonl"]

_ROW_LIST_KEYS = ("sources", "documents", "reviews", "transcripts", "complaints", "rows", "data")
_SOURCE_ID_KEYS = ("source_id", "id", "review_id", "transcript_id", "document_id")
_TEXT_KEYS = ("text", "review_text", "transcript", "content", "body", "quote", "complaint")
_SOURCE_TYPE_KEYS = ("source_type", "type", "kind")
_SOURCE_TITLE_KEYS = ("source_title", "title", "name")
_PAIN_KEYS = ("pain_points", "pain_categories", "pain_category", "topic", "category")


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
    text = _first_text(row, _TEXT_KEYS)
    if not text:
        warnings.append(CampaignOpportunityWarning(
            code="missing_source_text",
            row_index=row_index,
            field="text",
            message=(
                "Source row did not contain text, review_text, transcript, "
                "content, body, quote, or complaint."
            ),
        ))
    source_id = _first_text(row, _SOURCE_ID_KEYS)
    source_type = _first_text(row, _SOURCE_TYPE_KEYS) or _infer_source_type(row)
    opportunity = {
        str(key): value
        for key, value in row.items()
        if value not in (None, "", [], {})
    }
    if source_id and "source_id" not in opportunity:
        opportunity["source_id"] = source_id
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
    for key in _ROW_LIST_KEYS:
        value = data.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
    return [dict(data)]


def _resolve_format(path: Path, file_format: SourceDataFormat) -> Literal["json", "jsonl"]:
    if file_format != "auto":
        return file_format
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
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
    title = _first_text(row, _SOURCE_TITLE_KEYS)
    if title:
        evidence["source_title"] = title
    return evidence


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        text = str(row.get(key) or "").strip()
        if text:
            return text
    return ""


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
    return "document"


__all__ = [
    "SourceDataFormat",
    "load_source_campaign_opportunities_from_file",
    "source_row_to_campaign_opportunity",
    "source_rows_to_campaign_opportunities",
]
