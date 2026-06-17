"""Pre-import diagnostics for AI Content Ops ingestion files."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

from .campaign_customer_data import (
    CampaignOpportunityLoadResult,
    CsvCustomerDataParseError,
    CustomerDataFormat,
    load_campaign_opportunities_from_file,
    normalize_campaign_opportunity_rows,
)
from .campaign_source_adapters import (
    SourceDataFormat,
    SourceRowAdmissionDiagnostics,
    build_source_row_admission_diagnostics,
    load_source_campaign_opportunities_from_file,
    load_source_rows_with_warnings_from_file,
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
    source_row_admission: SourceRowAdmissionDiagnostics | None = None
    parse_error: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        if self.parse_error:
            return False
        return bool(self.opportunities) and not self.missing_field_counts.get("target_id", 0)

    def as_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.source_row_admission:
            payload["source_row_admission"] = self.source_row_admission.as_dict()
        if self.parse_error:
            payload["parse_error"] = dict(self.parse_error)
        return payload


def inspect_ingestion_file(
    path: str | Path,
    *,
    source_rows: bool = False,
    file_format: CustomerDataFormat = "auto",
    source_format: SourceDataFormat = "auto",
    target_mode: str | None = "vendor_retention",
    max_source_text_chars: int = 1200,
    sample_limit: int = 3,
    default_fields: Mapping[str, Any] | None = None,
) -> IngestionDiagnosticsReport:
    """Inspect a host ingestion file without writing to a database."""

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    if max_source_text_chars < 1:
        raise ValueError("max_source_text_chars must be positive")

    source = Path(path)
    source_row_admission: SourceRowAdmissionDiagnostics | None = None
    try:
        if source_rows:
            if _source_rows_file_is_csv(source, source_format):
                rows, load_warnings = load_source_rows_with_warnings_from_file(
                    source,
                    file_format=source_format,
                )
                converted = source_rows_to_campaign_opportunities(
                    rows,
                    target_mode=target_mode,
                    max_text_chars=max_source_text_chars,
                    default_fields=default_fields,
                )
                source_row_admission = build_source_row_admission_diagnostics(
                    rows,
                    input_format="csv",
                    usable_source_row_count=len(converted.opportunities),
                )
                loaded = CampaignOpportunityLoadResult(
                    opportunities=converted.opportunities,
                    warnings=load_warnings + converted.warnings,
                    source=str(source),
                )
            else:
                loaded = load_source_campaign_opportunities_from_file(
                    source,
                    file_format=source_format,
                    target_mode=target_mode,
                    max_text_chars=max_source_text_chars,
                    default_fields=default_fields,
                )
            mode: IngestionMode = "source_rows"
        else:
            loaded = load_campaign_opportunities_from_file(
                source,
                file_format=file_format,
                target_mode=target_mode,
            )
            mode = "opportunities"
    except CsvCustomerDataParseError as exc:
        mode = "source_rows" if source_rows else "opportunities"
        return _parse_error_report(
            source,
            mode=mode,
            sample_limit=sample_limit,
            parse_error=_csv_parse_error_payload(
                exc,
                location=_parse_error_location(
                    source=source,
                    source_rows=source_rows,
                    file_format=file_format,
                    source_format=source_format,
                ),
            ),
        )
    except json.JSONDecodeError as exc:
        mode = "source_rows" if source_rows else "opportunities"
        return _parse_error_report(
            source,
            mode=mode,
            sample_limit=sample_limit,
            parse_error=_json_parse_error_payload(
                exc,
                location=_parse_error_location(
                    source=source,
                    source_rows=source_rows,
                    file_format=file_format,
                    source_format=source_format,
                ),
            ),
        )
    except UnicodeDecodeError as exc:
        mode = "source_rows" if source_rows else "opportunities"
        return _parse_error_report(
            source,
            mode=mode,
            sample_limit=sample_limit,
            parse_error=_decode_parse_error_payload(
                exc,
                location=_parse_error_location(
                    source=source,
                    source_rows=source_rows,
                    file_format=file_format,
                    source_format=source_format,
                ),
            ),
        )
    except csv.Error as exc:
        mode = "source_rows" if source_rows else "opportunities"
        return _parse_error_report(
            source,
            mode=mode,
            sample_limit=sample_limit,
            parse_error=_raw_csv_parse_error_payload(
                exc,
                location=_parse_error_location(
                    source=source,
                    source_rows=source_rows,
                    file_format=file_format,
                    source_format=source_format,
                ),
            ),
        )

    return build_ingestion_diagnostics(
        loaded,
        mode=mode,
        source=str(source),
        sample_limit=sample_limit,
        source_row_admission=source_row_admission,
    )


def inspect_ingestion_rows(
    rows: Sequence[Any],
    *,
    source_rows: bool = False,
    source: str | None = "api",
    target_mode: str | None = "vendor_retention",
    max_source_text_chars: int = 1200,
    sample_limit: int = 3,
    default_fields: Mapping[str, Any] | None = None,
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
            default_fields=default_fields,
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
    source_row_admission: SourceRowAdmissionDiagnostics | None = None,
    parse_error: Mapping[str, Any] | None = None,
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
        source_row_admission=source_row_admission,
        parse_error=parse_error,
    )


def _parse_error_report(
    source: Path,
    *,
    mode: IngestionMode,
    sample_limit: int,
    parse_error: Mapping[str, Any],
) -> IngestionDiagnosticsReport:
    return IngestionDiagnosticsReport(
        mode=mode,
        source=str(source),
        opportunities=(),
        warnings=(),
        warning_counts={},
        missing_field_counts={},
        source_type_counts={},
        sample_limit=sample_limit,
        parse_error=parse_error,
    )


def _csv_parse_error_payload(
    error: CsvCustomerDataParseError,
    *,
    location: str,
) -> dict[str, Any]:
    payload = error.as_dict()
    payload["location"] = location
    return payload


def _json_parse_error_payload(
    error: json.JSONDecodeError,
    *,
    location: str,
) -> dict[str, Any]:
    return {
        "code": "json_parse_error",
        "message": (
            "JSON customer data could not be parsed "
            f"({error.msg} at line {error.lineno}, column {error.colno})."
        ),
        "how_to_fix": "Export the file again as valid UTF-8 JSON, then upload the new export.",
        "location": location,
        "line": error.lineno,
        "column": error.colno,
    }


def _decode_parse_error_payload(
    error: UnicodeDecodeError,
    *,
    location: str,
) -> dict[str, Any]:
    encoding = error.encoding or "text"
    return {
        "code": "file_decode_error",
        "message": (
            f"Ingestion file could not be decoded as {encoding} text "
            f"at byte {error.start}."
        ),
        "how_to_fix": (
            "Export the file again as valid UTF-8 CSV or JSON, then upload "
            "the new export."
        ),
        "location": location,
        "encoding": error.encoding,
        "byte": error.start,
    }


def _raw_csv_parse_error_payload(
    error: csv.Error,
    *,
    location: str,
) -> dict[str, Any]:
    # csv.Error messages are parser-structural, not uploaded row content.
    return {
        "code": "csv_parse_error",
        "message": f"CSV customer data could not be parsed ({error}).",
        "how_to_fix": (
            "Export the file again as valid CSV with consistent quoting and "
            "cells below the field-size limit."
        ),
        "location": location,
    }


def _parse_error_location(
    *,
    source: Path,
    source_rows: bool,
    file_format: CustomerDataFormat,
    source_format: SourceDataFormat,
) -> str:
    if source_rows:
        if source_format != "auto":
            resolved = source_format
        else:
            resolved = {
                ".csv": "csv",
                ".json": "json",
                ".jsonl": "jsonl",
            }.get(source.suffix.lower(), "unknown")
        return f"source_row_{resolved}"
    if file_format != "auto":
        resolved = file_format
    else:
        resolved = {
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "json",
        }.get(source.suffix.lower(), "unknown")
    return f"customer_data_{resolved}"


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


def _source_rows_file_is_csv(path: Path, source_format: SourceDataFormat) -> bool:
    if source_format == "csv":
        return True
    if source_format != "auto":
        return False
    return path.suffix.lower() == ".csv"


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


__all__ = [
    "IngestionDiagnosticsReport",
    "IngestionMode",
    "build_ingestion_diagnostics",
    "inspect_ingestion_file",
    "inspect_ingestion_rows",
]
