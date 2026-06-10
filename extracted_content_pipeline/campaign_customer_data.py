"""Customer data adapters for standalone campaign generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from typing import Any, Literal

from .campaign_opportunities import (
    normalize_campaign_opportunity,
    opportunity_target_id,
)
from .campaign_ports import TenantScope


CustomerDataFormat = Literal["auto", "json", "csv"]
_CSV_DETECT_DELIMITERS = ",;\t|"
_CSV_SNIFFER_SAMPLE_CHARS = 65_536
_CSV_HEADER_HINTS = frozenset({
    "account",
    "account_name",
    "answer",
    "body",
    "case_id",
    "category",
    "company",
    "company_name",
    "contact_email",
    "conversation_id",
    "conversation_title",
    "created_at",
    "customer_message",
    "description",
    "email",
    "id",
    "initial_message",
    "language",
    "message",
    "opportunity_id",
    "pain_category",
    "queue",
    "requester_comment",
    "resolution",
    "resolution_text",
    "source_id",
    "subject",
    "summary",
    "ticket_number",
    "ticket_id",
    "ticket_subject",
    "title",
    "topic",
    "user_email",
    "vendor",
    "vendor_name",
})


@dataclass(frozen=True)
class CampaignOpportunityWarning:
    """Non-fatal customer-data warning for one loaded opportunity row."""

    code: str
    message: str
    row_index: int | None = None
    field: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.row_index is not None:
            data["row_index"] = self.row_index
        if self.field:
            data["field"] = self.field
        return data


@dataclass(frozen=True)
class CampaignOpportunityLoadResult:
    """Normalized opportunities plus validation warnings from a customer file."""

    opportunities: tuple[dict[str, Any], ...]
    warnings: tuple[CampaignOpportunityWarning, ...] = ()
    source: str | None = None

    def warning_dicts(self) -> list[dict[str, Any]]:
        return [warning.as_dict() for warning in self.warnings]

    def as_payload(
        self,
        *,
        target_mode: str = "vendor_retention",
        channel: str = "email",
        limit: int | None = None,
        scope: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target_mode": target_mode,
            "channel": channel,
            "limit": limit or len(self.opportunities),
            "opportunities": [dict(row) for row in self.opportunities],
        }
        if scope is not None:
            payload["scope"] = dict(scope)
        if self.source:
            payload["source"] = self.source
        if self.warnings:
            payload["opportunity_warnings"] = self.warning_dicts()
        return payload


def load_campaign_opportunities_from_file(
    path: str | Path,
    *,
    file_format: CustomerDataFormat = "auto",
    target_mode: str | None = None,
) -> CampaignOpportunityLoadResult:
    """Load customer campaign opportunities from JSON or CSV."""

    source = Path(path)
    resolved_format = _resolve_format(source, file_format)
    if resolved_format == "csv":
        rows = _load_csv_rows(source)
    else:
        rows = _load_json_rows(source)
    result = normalize_campaign_opportunity_rows(rows, target_mode=target_mode)
    return CampaignOpportunityLoadResult(
        opportunities=result.opportunities,
        warnings=result.warnings,
        source=str(source),
    )


def normalize_campaign_opportunity_rows(
    rows: Sequence[Any],
    *,
    target_mode: str | None = None,
) -> CampaignOpportunityLoadResult:
    """Normalize loose customer rows and collect non-fatal validation warnings."""

    opportunities: list[dict[str, Any]] = []
    warnings: list[CampaignOpportunityWarning] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            warnings.append(
                CampaignOpportunityWarning(
                    code="row_not_object",
                    row_index=index,
                    message="Skipped row because it is not an object.",
                )
            )
            continue
        normalized = normalize_campaign_opportunity(row, target_mode=target_mode)
        if not normalized:
            warnings.append(
                CampaignOpportunityWarning(
                    code="empty_row",
                    row_index=index,
                    message="Skipped row because it did not contain usable values.",
                )
            )
            continue
        opportunities.append(normalized)
        warnings.extend(_validation_warnings(normalized, row_index=index))
    return CampaignOpportunityLoadResult(
        opportunities=tuple(opportunities),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class FileIntelligenceRepository:
    """IntelligenceRepository backed by loaded customer opportunity rows."""

    opportunities: Sequence[Mapping[str, Any]]
    warnings: Sequence[CampaignOpportunityWarning] = ()
    source: str | None = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        file_format: CustomerDataFormat = "auto",
        target_mode: str | None = None,
    ) -> "FileIntelligenceRepository":
        loaded = load_campaign_opportunities_from_file(
            path,
            file_format=file_format,
            target_mode=target_mode,
        )
        return cls(
            opportunities=loaded.opportunities,
            warnings=loaded.warnings,
            source=loaded.source,
        )

    async def read_campaign_opportunities(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        del scope
        rows = [
            normalize_campaign_opportunity(row, target_mode=target_mode)
            for row in self.opportunities
            if _matches_filters(row, filters)
        ]
        return rows[:limit]

    async def read_vendor_targets(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        vendor_name: str | None = None,
    ) -> Sequence[dict[str, Any]]:  # pragma: no cover - protocol filler
        del scope
        del target_mode
        del vendor_name
        return []


def _resolve_format(path: Path, file_format: CustomerDataFormat) -> Literal["json", "csv"]:
    if file_format != "auto":
        return file_format
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl"}:
        return "json"
    raise ValueError(f"Cannot infer customer data format from file suffix: {path}")


def _load_json_rows(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return list(data)
    if not isinstance(data, Mapping):
        raise ValueError("JSON customer data must be an object or array")
    for key in ("opportunities", "rows", "data", "accounts", "customers"):
        value = data.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
    return [dict(data)]


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    return _load_csv_dict_rows(path, value_coercer=_coerce_csv_value)


def _load_csv_dict_rows(
    path: Path,
    *,
    value_coercer: Callable[[Any], Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = _read_csv_text(path)
    if not text.strip():
        raise ValueError("CSV customer data must include a header row.")
    reader = csv.reader(StringIO(text), dialect=_detect_csv_dialect(text))
    raw_rows = list(reader)
    header_index = _csv_header_index(raw_rows)
    if header_index is None:
        raise ValueError("CSV customer data must include a header row.")
    header_width = len(raw_rows[header_index])
    header_fields = tuple(
        (index, str(field or "").strip())
        for index, field in enumerate(raw_rows[header_index])
        if field is not None and str(field).strip()
    )
    if not header_fields:
        raise ValueError("CSV customer data must include a header row.")
    for row_index, values in enumerate(raw_rows[header_index + 1:], start=header_index + 2):
        if not any(str(value or "").strip() for value in values):
            continue
        if len(values) > header_width:
            raise ValueError(
                f"CSV row {row_index} has more cells than the header; "
                "check the delimiter and header row."
            )
        cleaned: dict[str, Any] = {}
        for value_index, key in header_fields:
            value = values[value_index] if value_index < len(values) else ""
            cleaned_key = str(key).strip()
            if not cleaned_key:
                continue
            cleaned_value = value_coercer(value) if value_coercer else value
            if cleaned_value not in (None, ""):
                cleaned[cleaned_key] = cleaned_value
        rows.append(cleaned)
    return rows


def _csv_header_index(rows: Sequence[Sequence[Any]]) -> int | None:
    fallback: int | None = None
    for index, row in enumerate(rows):
        cells = [str(cell or "").strip() for cell in row]
        nonempty = [cell for cell in cells if cell]
        if not nonempty:
            continue
        normalized = {_normalize_csv_header_cell(cell) for cell in nonempty}
        if normalized.intersection(_CSV_HEADER_HINTS):
            return index
        if len(nonempty) < 2:
            continue
        if fallback is None:
            fallback = index
    return fallback


def _normalize_csv_header_cell(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _read_csv_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _detect_csv_dialect(text: str) -> csv.Dialect:
    sample = text[:_CSV_SNIFFER_SAMPLE_CHARS]
    try:
        return _harden_csv_dialect(csv.Sniffer().sniff(
            sample,
            delimiters=_CSV_DETECT_DELIMITERS,
        ))
    except csv.Error:
        return csv.excel


def _harden_csv_dialect(dialect: csv.Dialect) -> type[csv.Dialect]:
    class HardenedDialect(csv.Dialect):
        delimiter = dialect.delimiter
        quotechar = dialect.quotechar or '"'
        escapechar = dialect.escapechar
        doublequote = True
        skipinitialspace = dialect.skipinitialspace
        lineterminator = dialect.lineterminator
        quoting = dialect.quoting
        strict = False

    return HardenedDialect


def _coerce_csv_value(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    if text[0] not in "[{":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _validation_warnings(
    opportunity: Mapping[str, Any],
    *,
    row_index: int,
) -> list[CampaignOpportunityWarning]:
    warnings: list[CampaignOpportunityWarning] = []
    checks = [
        (
            "missing_target_id",
            "target_id",
            not opportunity_target_id(opportunity),
            "Row does not contain a stable target id, email, company, or vendor.",
        ),
        (
            "missing_company_name",
            "company_name",
            not str(opportunity.get("company_name") or "").strip(),
            "Row does not contain a company name.",
        ),
        (
            "missing_vendor_name",
            "vendor_name",
            not str(opportunity.get("vendor_name") or "").strip(),
            "Row does not contain a current/incumbent vendor name.",
        ),
        (
            "missing_contact_email",
            "contact_email",
            not str(opportunity.get("contact_email") or "").strip(),
            "Row does not contain a contact email.",
        ),
    ]
    for code, field, should_warn, message in checks:
        if should_warn:
            warnings.append(
                CampaignOpportunityWarning(
                    code=code,
                    field=field,
                    row_index=row_index,
                    message=message,
                )
            )
    return warnings


def _matches_filters(
    row: Mapping[str, Any],
    filters: Mapping[str, Any] | None,
) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if expected in (None, "", [], {}):
            continue
        actual = row.get(key)
        if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
            expected_values = {str(item).strip().lower() for item in expected}
            if str(actual or "").strip().lower() not in expected_values:
                return False
        elif str(actual or "").strip().lower() != str(expected).strip().lower():
            return False
    return True


__all__ = [
    "CampaignOpportunityLoadResult",
    "CampaignOpportunityWarning",
    "FileIntelligenceRepository",
    "load_campaign_opportunities_from_file",
    "normalize_campaign_opportunity_rows",
]
