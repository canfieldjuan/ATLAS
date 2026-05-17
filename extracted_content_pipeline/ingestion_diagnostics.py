"""Pre-import diagnostics for AI Content Ops ingestion files."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .campaign_customer_data import (
    CampaignOpportunityLoadResult,
    CustomerDataFormat,
    load_campaign_opportunities_from_file,
    normalize_campaign_opportunity_rows,
)
from .campaign_source_adapters import (
    SourceDataFormat,
    load_source_campaign_opportunities_from_file,
    source_rows_to_campaign_opportunities,
)


IngestionMode = Literal["opportunities", "source_rows"]


_REQUIRED_FIELDS = ("target_id", "company_name", "vendor_name", "evidence")


@dataclass(frozen=True)
class IngestionDiagnosticsReport:
    """Structured readiness report for a host ingestion file."""

    mode: IngestionMode
    source: str
    opportunities: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    warning_counts: Mapping[str, int]
    missing_field_counts: Mapping[str, int]
    source_type_counts: Mapping[str, int]
    sample_limit: int

    @property
    def ok(self) -> bool:
        return bool(self.opportunities) and not self.missing_field_counts.get("target_id", 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "source": self.source,
            "opportunity_count": len(self.opportunities),
            "warning_count": len(self.warnings),
            "warning_counts": dict(sorted(self.warning_counts.items())),
            "missing_field_counts": dict(sorted(self.missing_field_counts.items())),
            "source_type_counts": dict(sorted(self.source_type_counts.items())),
            "samples": [dict(row) for row in self.opportunities[: self.sample_limit]],
            "warnings": list(self.warnings),
        }


def inspect_ingestion_file(
    path: str | Path,
    *,
    source_rows: bool = False,
    file_format: CustomerDataFormat = "auto",
    source_format: SourceDataFormat = "auto",
    target_mode: str | None = "vendor_retention",
    max_source_text_chars: int = 1200,
    sample_limit: int = 3,
) -> IngestionDiagnosticsReport:
    """Inspect a host ingestion file without writing to a database."""

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    if max_source_text_chars < 1:
        raise ValueError("max_source_text_chars must be positive")

    source = Path(path)
    if source_rows:
        loaded = load_source_campaign_opportunities_from_file(
            source,
            file_format=source_format,
            target_mode=target_mode,
            max_text_chars=max_source_text_chars,
        )
        mode: IngestionMode = "source_rows"
    else:
        loaded = load_campaign_opportunities_from_file(
            source,
            file_format=file_format,
            target_mode=target_mode,
        )
        mode = "opportunities"

    return build_ingestion_diagnostics(
        loaded,
        mode=mode,
        source=str(source),
        sample_limit=sample_limit,
    )


def inspect_ingestion_rows(
    rows: Sequence[Any],
    *,
    source_rows: bool = False,
    source: str | None = "api",
    target_mode: str | None = "vendor_retention",
    max_source_text_chars: int = 1200,
    sample_limit: int = 3,
) -> IngestionDiagnosticsReport:
    """Inspect already-loaded host rows without writing to a database."""

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    if max_source_text_chars < 1:
        raise ValueError("max_source_text_chars must be positive")

    if source_rows:
        loaded = source_rows_to_campaign_opportunities(
            rows,
            target_mode=target_mode,
            max_text_chars=max_source_text_chars,
        )
        mode: IngestionMode = "source_rows"
    else:
        loaded = normalize_campaign_opportunity_rows(
            rows,
            target_mode=target_mode,
        )
        mode = "opportunities"

    return build_ingestion_diagnostics(
        loaded,
        mode=mode,
        source=source,
        sample_limit=sample_limit,
    )


def build_ingestion_diagnostics(
    loaded: CampaignOpportunityLoadResult,
    *,
    mode: IngestionMode,
    source: str | None = None,
    sample_limit: int = 3,
) -> IngestionDiagnosticsReport:
    """Build a diagnostics report from normalized opportunity rows."""

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    opportunities = tuple(dict(row) for row in loaded.opportunities)
    warnings = tuple(warning.as_dict() for warning in loaded.warnings)
    return IngestionDiagnosticsReport(
        mode=mode,
        source=source or loaded.source or "",
        opportunities=opportunities,
        warnings=warnings,
        warning_counts=_warning_counts(warnings),
        missing_field_counts=_missing_field_counts(opportunities),
        source_type_counts=_source_type_counts(opportunities),
        sample_limit=sample_limit,
    )


def _warning_counts(warnings: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for warning in warnings:
        code = str(warning.get("code") or "unknown").strip() or "unknown"
        counts[code] += 1
    return dict(counts)


def _missing_field_counts(opportunities: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in opportunities:
        for field in _REQUIRED_FIELDS:
            if not _has_value(row.get(field)):
                counts[field] += 1
    return dict(counts)


def _source_type_counts(opportunities: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in opportunities:
        source_type = _row_source_type(row)
        if source_type:
            counts[source_type] += 1
    return dict(counts)


def _row_source_type(row: Mapping[str, Any]) -> str:
    source_type = str(row.get("source_type") or "").strip()
    if source_type:
        return source_type
    evidence = row.get("evidence")
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        for item in evidence:
            if isinstance(item, Mapping):
                source_type = str(item.get("source_type") or "").strip()
                if source_type:
                    return source_type
    return ""


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


__all__ = [
    "IngestionDiagnosticsReport",
    "IngestionMode",
    "build_ingestion_diagnostics",
    "inspect_ingestion_file",
    "inspect_ingestion_rows",
]
