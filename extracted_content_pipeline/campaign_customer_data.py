"""Customer data adapters for standalone campaign generation."""

from __future__ import annotations

import codecs
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
_CSV_DETECT_QUOTECHARS = ('"', "'")
_CSV_SNIFFER_SAMPLE_CHARS = 65_536
_CSV_DELIMITER_SAMPLE_LINES = 100
_CSV_DELIMITER_MIN_CONSISTENCY = 0.90
_CSV_NUL_REDECODE_RATIO = 0.10
_CSV_UTF16_NUL_SIDE_RATIO = 0.30
_CSV_REPLACEMENT_WARNING_RATIO = 0.01
_CSV_UTF8_RECOVERY_REPLACEMENT_RATIO = 0.05
_CSV_UTF8_MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00e2\u201a")
_CSV_UTF8_MOJIBAKE_MARKER_TAIL_CHARS = max(
    len(marker) for marker in _CSV_UTF8_MOJIBAKE_MARKERS
) - 1
_CSV_IMPLAUSIBLE_LEGACY_FALLBACK_CHARS = ("\u00ff", "\u00fe")
_CSV_READ_CHUNK_BYTES = 64 * 1024
_CSV_BOM_ENCODINGS = (
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)
_CSV_HEADER_HINTS = frozenset({
    "account",
    "account_name",
    "answer",
    "body",
    "case_id",
    "category",
    "comment",
    "comment_body",
    "comments",
    "company",
    "company_name",
    "contact_email",
    "conversation_id",
    "conversation_title",
    "created_at",
    "customer_message",
    "description",
    "email",
    "history",
    "id",
    "initial_message",
    "language",
    "message",
    "opportunity_id",
    "pain_category",
    "queue",
    "public_comment",
    "public_comments",
    "requester_comment",
    "resolution",
    "resolution_text",
    "source_id",
    "subject",
    "summary",
    "ticket_comments",
    "ticket_history",
    "ticket_number",
    "ticket_id",
    "ticket_subject",
    "title",
    "topic",
    "user_email",
    "vendor",
    "vendor_name",
})
_CSV_MISSING_HEADER_FIX = (
    "Add a header row with column names such as ticket_id, subject, and message, "
    "then export the CSV again."
)
_CSV_ENCODING_FIX = (
    "Export the file again as UTF-8 or UTF-16 CSV, then upload the new export."
)
_CSV_INCONSISTENT_COLUMNS_FIX = (
    "Use one delimiter consistently, keep every row within the header width, "
    "and quote cells that contain commas, tabs, semicolons, pipes, or newlines."
)


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
class CsvCustomerDataParseError(ValueError):
    """Safe structured CSV parser error for user-facing upload feedback."""

    code: str
    message: str
    how_to_fix: str
    row_index: int | None = None

    def __str__(self) -> str:
        return self.message

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "how_to_fix": self.how_to_fix,
        }
        if self.row_index is not None:
            data["row_index"] = self.row_index
        return data


@dataclass(frozen=True)
class _CsvDelimiterCandidate:
    delimiter: str
    quotechar: str
    header_index: int | None
    header_has_hint: bool
    header_width: int
    data_rows: int
    consistent_rows: int
    exact_width_rows: int
    first_offending_row: int | None
    first_collapsed_row: int | None

    @property
    def consistency(self) -> float:
        if not self.data_rows:
            return 1.0
        return self.consistent_rows / self.data_rows

    @property
    def exactness(self) -> float:
        if not self.data_rows:
            return 1.0
        return self.exact_width_rows / self.data_rows

    @property
    def valid(self) -> bool:
        return (
            self.header_index is not None
            and self.header_width >= 1
            and (self.data_rows > 0 or self.header_index == 0)
            and self.first_collapsed_row is None
            and self.consistency >= _CSV_DELIMITER_MIN_CONSISTENCY
        )


@dataclass(frozen=True)
class _CsvEncodingPlan:
    encoding: str
    errors: str = "strict"
    warnings: tuple[CampaignOpportunityWarning, ...] = ()


@dataclass(frozen=True)
class _CsvTextStats:
    char_count: int
    nul_count: int
    replacement_count: int
    mojibake_score: int
    implausible_legacy_count: int

    @property
    def nul_ratio(self) -> float:
        if not self.char_count:
            return 0.0
        return self.nul_count / self.char_count

    @property
    def replacement_ratio(self) -> float:
        if not self.char_count:
            return 0.0
        return self.replacement_count / self.char_count


@dataclass
class _CsvColumnConsistencyState:
    header_index: int
    header_has_hint: bool
    header_width: int
    delimiter: str
    data_rows: int = 0
    consistent_rows: int = 0
    exact_width_rows: int = 0
    first_offending_row: int | None = None
    first_collapsed_row: int | None = None

    def add(self, row_number: int, row: Sequence[Any]) -> None:
        if not any(str(value or "").strip() for value in row):
            return
        self.data_rows += 1
        collapsed = (
            self.header_width >= 2
            and _csv_short_row_uses_competing_delimiter(row, delimiter=self.delimiter)
        )
        if collapsed and self.first_collapsed_row is None:
            self.first_collapsed_row = row_number
        if len(row) == self.header_width:
            self.exact_width_rows += 1
        if _csv_row_matches_delimiter(row, self.header_width, delimiter=self.delimiter):
            self.consistent_rows += 1
        elif self.first_offending_row is None:
            self.first_offending_row = row_number

    def as_candidate(self, *, quotechar: str) -> _CsvDelimiterCandidate:
        return _CsvDelimiterCandidate(
            delimiter=self.delimiter,
            quotechar=quotechar,
            header_index=self.header_index,
            header_has_hint=self.header_has_hint,
            header_width=self.header_width,
            data_rows=self.data_rows,
            consistent_rows=self.consistent_rows,
            exact_width_rows=self.exact_width_rows,
            first_offending_row=self.first_offending_row,
            first_collapsed_row=self.first_collapsed_row,
        )


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
    load_warnings: tuple[CampaignOpportunityWarning, ...] = ()
    if resolved_format == "csv":
        rows, load_warnings = _load_csv_rows(source)
    else:
        rows = _load_json_rows(source)
    result = normalize_campaign_opportunity_rows(rows, target_mode=target_mode)
    return CampaignOpportunityLoadResult(
        opportunities=result.opportunities,
        warnings=load_warnings + result.warnings,
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


def _load_csv_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], tuple[CampaignOpportunityWarning, ...]]:
    return _load_csv_dict_rows(path, value_coercer=_coerce_csv_value)


def _load_csv_dict_rows(
    path: Path,
    *,
    value_coercer: Callable[[Any], Any] | None = None,
) -> tuple[list[dict[str, Any]], tuple[CampaignOpportunityWarning, ...]]:
    rows: list[dict[str, Any]] = []
    encoding_plan = _csv_encoding_plan(path)
    candidate = _select_csv_delimiter_for_path(path, encoding_plan)
    header_index = candidate.header_index
    if header_index is None:
        raise _csv_missing_header_error()

    skipped_leading_count = 0
    first_leading_row: tuple[int, Sequence[Any]] | None = None
    header_fields: tuple[tuple[int, str], ...] = ()
    header_width = 0
    consistency: _CsvColumnConsistencyState | None = None
    header_row_number = header_index + 1
    with _open_csv_text(path, encoding_plan) as handle:
        reader = csv.reader(
            handle,
            dialect=_csv_delimiter_dialect(
                candidate.delimiter,
                quotechar=candidate.quotechar,
            ),
        )
        for row_index, values in enumerate(reader, start=1):
            if row_index < header_row_number:
                if any(str(cell or "").strip() for cell in values):
                    skipped_leading_count += 1
                    if first_leading_row is None:
                        first_leading_row = (row_index, values)
                continue
            if row_index == header_row_number:
                header_width = len(values)
                header_fields = tuple(
                    (index, str(field or "").strip())
                    for index, field in enumerate(values)
                    if field is not None and str(field).strip()
                )
                if not header_fields:
                    raise _csv_missing_header_error()
                consistency = _CsvColumnConsistencyState(
                    header_index=header_index,
                    header_has_hint=candidate.header_has_hint,
                    header_width=header_width,
                    delimiter=candidate.delimiter,
                )
                continue
            if not any(str(value or "").strip() for value in values):
                continue
            if len(values) > header_width:
                raise _csv_inconsistent_columns_error(
                    f"CSV row {row_index} has more cells than the header.",
                    row_index=row_index,
                )
            if consistency is not None:
                consistency.add(row_index, values)
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
    if not header_fields or consistency is None:
        raise _csv_missing_header_error()
    consistency_candidate = consistency.as_candidate(quotechar=candidate.quotechar)
    if not consistency_candidate.valid:
        _raise_csv_delimiter_error(consistency_candidate)
    load_warnings = encoding_plan.warnings + _leading_rows_skipped_warnings(
        skipped_leading_count,
        first_leading_row,
        header_index,
    )
    return rows, load_warnings


def _csv_missing_header_error() -> CsvCustomerDataParseError:
    return CsvCustomerDataParseError(
        code="csv_missing_header",
        message="CSV customer data must include a header row.",
        how_to_fix=_CSV_MISSING_HEADER_FIX,
    )


def _csv_encoding_error(message: str) -> CsvCustomerDataParseError:
    return CsvCustomerDataParseError(
        code="csv_encoding_error",
        message=message,
        how_to_fix=_CSV_ENCODING_FIX,
    )


def _csv_inconsistent_columns_error(
    message: str,
    *,
    row_index: int | None = None,
) -> CsvCustomerDataParseError:
    return CsvCustomerDataParseError(
        code="csv_inconsistent_columns",
        message=message,
        how_to_fix=_CSV_INCONSISTENT_COLUMNS_FIX,
        row_index=row_index,
    )


def _open_csv_text(path: Path, plan: _CsvEncodingPlan):
    return path.open(
        "r",
        encoding=plan.encoding,
        errors=plan.errors,
        newline="",
    )


def _csv_delimiter_sample(path: Path, plan: _CsvEncodingPlan) -> str:
    with _open_csv_text(path, plan) as handle:
        return handle.read(_CSV_SNIFFER_SAMPLE_CHARS)


def _select_csv_delimiter_for_path(
    path: Path,
    plan: _CsvEncodingPlan,
) -> _CsvDelimiterCandidate:
    sample_text = _csv_delimiter_sample(path, plan)
    if sample_text.strip():
        try:
            sample_candidate = _select_csv_delimiter(sample_text)
        except ValueError:
            sample_candidate = None
        if sample_candidate is not None and sample_candidate.header_has_hint:
            return sample_candidate
    return _select_csv_delimiter_from_stream(path, plan)


def _csv_encoding_plan(path: Path) -> _CsvEncodingPlan:
    prefix, byte_count = _csv_byte_prefix_and_count(path)
    if byte_count == 0:
        return _CsvEncodingPlan(encoding="utf-8-sig")
    for bom, encoding in _CSV_BOM_ENCODINGS:
        if prefix.startswith(bom):
            return _csv_checked_encoding_plan(path, encoding=encoding)
    try:
        utf8_stats = _scan_csv_text_stats(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return _csv_utf8_error_encoding_plan(path, prefix=prefix)
    if utf8_stats.nul_ratio >= _CSV_NUL_REDECODE_RATIO:
        inferred = _utf16_encoding_from_nul_bytes(prefix)
        if inferred:
            return _csv_inferred_utf16_encoding_plan(path, encoding=inferred)
        raise _csv_encoding_error(
            "CSV customer data decoded to NUL-heavy text; check the file encoding."
        )
    return _CsvEncodingPlan(
        encoding="utf-8-sig",
        warnings=_csv_decode_warnings_from_stats(
            utf8_stats,
            encoding="utf-8-sig",
        ),
    )


def _csv_checked_encoding_plan(path: Path, *, encoding: str) -> _CsvEncodingPlan:
    try:
        stats = _scan_csv_text_stats(path, encoding=encoding)
    except UnicodeDecodeError as exc:
        raise _csv_encoding_error(
            f"CSV customer data could not be decoded as {encoding}; "
            "check the file encoding."
        ) from exc
    if stats.nul_ratio >= _CSV_NUL_REDECODE_RATIO:
        raise _csv_encoding_error(
            f"CSV customer data decoded as {encoding} but remained NUL-heavy; "
            "check the file encoding."
        )
    return _CsvEncodingPlan(
        encoding=encoding,
        warnings=_csv_decode_warnings_from_stats(stats, encoding=encoding),
    )


def _csv_inferred_utf16_encoding_plan(path: Path, *, encoding: str) -> _CsvEncodingPlan:
    try:
        stats = _scan_csv_text_stats(path, encoding=encoding)
    except UnicodeDecodeError as exc:
        raise _csv_encoding_error(
            f"CSV customer data could not be decoded as {encoding}; "
            "check the file encoding."
        ) from exc
    if stats.nul_ratio >= _CSV_NUL_REDECODE_RATIO:
        raise _csv_encoding_error(
            f"CSV customer data decoded as {encoding} but remained NUL-heavy; "
            "check the file encoding."
        )
    return _CsvEncodingPlan(
        encoding=encoding,
        warnings=_csv_decode_warnings_from_stats(stats, encoding=encoding)
        + (
            CampaignOpportunityWarning(
                code="csv_encoding_inferred",
                field="encoding",
                message=(
                    "CSV contained UTF-16-style NUL bytes without a BOM; "
                    f"decoded as {encoding}."
                ),
            ),
        ),
    )


def _leading_rows_skipped_warnings(
    skipped_count: int,
    first_row: tuple[int, Sequence[Any]] | None,
    header_index: int,
) -> tuple[CampaignOpportunityWarning, ...]:
    if not skipped_count or first_row is None:
        return ()
    first_row_number, first_row_values = first_row
    preview = _csv_row_preview(first_row_values)
    return (
        CampaignOpportunityWarning(
            code="csv_leading_rows_skipped",
            message=(
                f"Skipped {skipped_count} leading row(s) before the CSV header "
                f"on row {header_index + 1}; "
                f"first skipped row {first_row_number}: {preview}"
            ),
            row_index=first_row_number,
        ),
    )


def _csv_row_preview(row: Sequence[Any], *, max_chars: int = 80) -> str:
    text = ", ".join(
        str(cell).strip() for cell in row if str(cell or "").strip()
    )
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _csv_header_index(rows: Sequence[Sequence[Any]]) -> int | None:
    index, _has_hint = _csv_header_index_and_hint(rows)
    return index


def _csv_header_index_and_hint(
    rows: Sequence[Sequence[Any]],
) -> tuple[int | None, bool]:
    fallback: int | None = None
    for index, row in enumerate(rows):
        cells = [str(cell or "").strip() for cell in row]
        nonempty = [cell for cell in cells if cell]
        if not nonempty:
            continue
        if _csv_row_has_header_hint(row):
            return index, True
        if len(nonempty) < 2:
            continue
        if fallback is None:
            fallback = index
    return fallback, False


def _csv_row_has_header_hint(row: Sequence[Any]) -> bool:
    normalized = {
        _normalize_csv_header_cell(str(cell or "").strip())
        for cell in row
        if str(cell or "").strip()
    }
    return bool(normalized.intersection(_CSV_HEADER_HINTS))


def _normalize_csv_header_cell(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _csv_byte_prefix_and_count(path: Path) -> tuple[bytes, int]:
    prefix = bytearray()
    byte_count = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CSV_READ_CHUNK_BYTES)
            if not chunk:
                break
            byte_count += len(chunk)
            if len(prefix) < _CSV_SNIFFER_SAMPLE_CHARS:
                remaining = _CSV_SNIFFER_SAMPLE_CHARS - len(prefix)
                prefix.extend(chunk[:remaining])
    return bytes(prefix), byte_count


def _csv_utf8_error_encoding_plan(path: Path, *, prefix: bytes) -> _CsvEncodingPlan:
    inferred = _utf16_encoding_from_nul_bytes(prefix)
    if inferred:
        return _csv_inferred_utf16_encoding_plan(path, encoding=inferred)
    legacy_encoding, legacy_stats = _legacy_csv_encoding_stats(path)
    recovered_stats = _scan_csv_text_stats(
        path,
        encoding="utf-8-sig",
        errors="replace",
    )
    if (
        recovered_stats.replacement_ratio <= _CSV_UTF8_RECOVERY_REPLACEMENT_RATIO
        and legacy_stats.mojibake_score > recovered_stats.replacement_count
    ):
        return _CsvEncodingPlan(
            encoding="utf-8-sig",
            errors="replace",
            warnings=_csv_decode_warnings_from_stats(
                recovered_stats,
                encoding="utf-8-sig",
                warn_on_any_replacement=True,
            ),
        )
    return _CsvEncodingPlan(
        encoding=legacy_encoding,
        warnings=_csv_decode_warnings_from_stats(
            legacy_stats,
            encoding=legacy_encoding,
        )
        + _legacy_fallback_corruption_warnings(
            path,
            legacy_encoding=legacy_encoding,
            legacy_stats=legacy_stats,
        ),
    )


def _legacy_csv_encoding_stats(path: Path) -> tuple[str, _CsvTextStats]:
    for encoding in ("cp1252", "latin-1"):
        try:
            stats = _scan_csv_text_stats(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        if stats.nul_ratio >= _CSV_NUL_REDECODE_RATIO:
            raise _csv_encoding_error(
                f"CSV customer data decoded as {encoding} but remained NUL-heavy; "
                "check the file encoding."
            )
        return encoding, stats
    raise _csv_encoding_error(
        "CSV customer data could not be decoded as UTF-8, UTF-16, UTF-32, "
        "CP1252, or Latin-1."
    )


def _scan_csv_text_stats(
    path: Path,
    *,
    encoding: str,
    errors: str = "strict",
) -> _CsvTextStats:
    decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
    char_count = 0
    nul_count = 0
    replacement_count = 0
    mojibake_score = 0
    implausible_legacy_count = 0
    marker_tail = ""

    def add_text(text: str) -> None:
        nonlocal char_count
        nonlocal nul_count
        nonlocal replacement_count
        nonlocal mojibake_score
        nonlocal implausible_legacy_count
        nonlocal marker_tail
        if not text:
            return
        char_count += len(text)
        nul_count += text.count("\x00")
        replacement_count += text.count("\ufffd")
        marker_score, marker_tail = _csv_mojibake_marker_score(text, marker_tail)
        mojibake_score += marker_score
        implausible_legacy_count += sum(
            1 for char in text if _is_implausible_legacy_fallback_char(char)
        )

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CSV_READ_CHUNK_BYTES)
            if not chunk:
                break
            add_text(decoder.decode(chunk, final=False))
    add_text(decoder.decode(b"", final=True))
    return _CsvTextStats(
        char_count=char_count,
        nul_count=nul_count,
        replacement_count=replacement_count,
        mojibake_score=mojibake_score,
        implausible_legacy_count=implausible_legacy_count,
    )


def _csv_mojibake_marker_score(text: str, tail: str) -> tuple[int, str]:
    combined = tail + text
    boundary = len(tail)
    score = 0
    for marker in _CSV_UTF8_MOJIBAKE_MARKERS:
        start = combined.find(marker)
        while start != -1:
            if start + len(marker) > boundary:
                score += 1
            start = combined.find(marker, start + 1)
    next_tail = combined[-_CSV_UTF8_MOJIBAKE_MARKER_TAIL_CHARS:]
    return score, next_tail


def _utf16_encoding_from_nul_bytes(data: bytes) -> str | None:
    sample = data[:_CSV_SNIFFER_SAMPLE_CHARS]
    even = sample[0::2]
    odd = sample[1::2]
    if not even or not odd:
        return None
    even_nul_ratio = even.count(0) / len(even)
    odd_nul_ratio = odd.count(0) / len(odd)
    if (
        odd_nul_ratio >= _CSV_UTF16_NUL_SIDE_RATIO
        and even_nul_ratio < _CSV_NUL_REDECODE_RATIO
    ):
        return "utf-16-le"
    if (
        even_nul_ratio >= _CSV_UTF16_NUL_SIDE_RATIO
        and odd_nul_ratio < _CSV_NUL_REDECODE_RATIO
    ):
        return "utf-16-be"
    return None


def _legacy_fallback_corruption_warnings(
    path: Path,
    *,
    legacy_encoding: str,
    legacy_stats: _CsvTextStats,
) -> tuple[CampaignOpportunityWarning, ...]:
    if legacy_stats.mojibake_score:
        return ()
    artifact_count = _legacy_fallback_implausible_artifact_count(
        path,
        legacy_encoding=legacy_encoding,
    )
    if not artifact_count:
        return ()
    return (
        CampaignOpportunityWarning(
            code="csv_encoding_ambiguous",
            field="encoding",
            message=(
                "CSV failed strict UTF-8 decoding; legacy fallback decoded the "
                "bytes, but UTF-8 recovery would contain "
                f"{artifact_count} replacement character(s). Verify the "
                "source encoding before relying on these rows."
            ),
        ),
    )


def _legacy_fallback_implausible_artifact_count(
    path: Path,
    *,
    legacy_encoding: str,
) -> int:
    legacy_decoder = codecs.getincrementaldecoder(legacy_encoding)(errors="strict")
    recovered_decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
    legacy_buffer = ""
    recovered_buffer = ""
    count = 0

    def consume() -> None:
        nonlocal legacy_buffer
        nonlocal recovered_buffer
        nonlocal count
        limit = min(len(legacy_buffer), len(recovered_buffer))
        for legacy_char, recovered_char in zip(
            legacy_buffer[:limit],
            recovered_buffer[:limit],
        ):
            if (
                recovered_char == "\ufffd"
                and _is_implausible_legacy_fallback_char(legacy_char)
            ):
                count += 1
        legacy_buffer = legacy_buffer[limit:]
        recovered_buffer = recovered_buffer[limit:]

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CSV_READ_CHUNK_BYTES)
            if not chunk:
                break
            legacy_buffer += legacy_decoder.decode(chunk, final=False)
            recovered_buffer += recovered_decoder.decode(chunk, final=False)
            consume()
    legacy_buffer += legacy_decoder.decode(b"", final=True)
    recovered_buffer += recovered_decoder.decode(b"", final=True)
    consume()
    return count


def _is_implausible_legacy_fallback_char(char: str) -> bool:
    if not char:
        return False
    if char in _CSV_IMPLAUSIBLE_LEGACY_FALLBACK_CHARS:
        return True
    codepoint = ord(char)
    return 0x80 <= codepoint <= 0x9F


def _csv_decode_warnings_from_stats(
    stats: _CsvTextStats,
    *,
    encoding: str,
    warn_on_any_replacement: bool = False,
) -> tuple[CampaignOpportunityWarning, ...]:
    if not stats.replacement_count:
        return ()
    ratio = stats.replacement_ratio
    if ratio < _CSV_REPLACEMENT_WARNING_RATIO and not warn_on_any_replacement:
        return ()
    return (
        CampaignOpportunityWarning(
            code="csv_replacement_characters",
            field="encoding",
            message=(
                f"CSV decoded as {encoding} but contains {stats.replacement_count} "
                f"Unicode replacement character(s) ({ratio:.1%}); export "
                "the source again with UTF-8 or UTF-16 encoding."
            ),
        ),
    )


def _select_csv_delimiter(text: str) -> _CsvDelimiterCandidate:
    candidates = tuple(
        _score_csv_delimiter(text, delimiter=delimiter, quotechar=quotechar)
        for delimiter in _CSV_DETECT_DELIMITERS
        for quotechar in _CSV_DETECT_QUOTECHARS
    )
    valid = [candidate for candidate in candidates if candidate.valid]
    if valid:
        return max(valid, key=_csv_delimiter_candidate_key)
    if all(candidate.header_index is None for candidate in candidates):
        raise _csv_missing_header_error()
    _raise_csv_delimiter_error(max(candidates, key=_csv_delimiter_candidate_key))


def _select_csv_delimiter_from_stream(
    path: Path,
    plan: _CsvEncodingPlan,
) -> _CsvDelimiterCandidate:
    candidates = tuple(
        _score_csv_delimiter_stream(
            path,
            plan,
            delimiter=delimiter,
            quotechar=quotechar,
        )
        for delimiter in _CSV_DETECT_DELIMITERS
        for quotechar in _CSV_DETECT_QUOTECHARS
    )
    valid = [candidate for candidate in candidates if candidate.valid]
    if valid:
        return max(valid, key=_csv_delimiter_candidate_key)
    if all(candidate.header_index is None for candidate in candidates):
        raise _csv_missing_header_error()
    _raise_csv_delimiter_error(max(candidates, key=_csv_delimiter_candidate_key))


def _score_csv_delimiter(
    text: str,
    *,
    delimiter: str,
    quotechar: str,
) -> _CsvDelimiterCandidate:
    reader = csv.reader(
        StringIO(_csv_delimiter_sample_text(text)),
        dialect=_csv_delimiter_dialect(delimiter, quotechar=quotechar),
    )
    rows = list(reader)
    header_index, header_has_hint = _csv_header_index_and_hint(rows)
    return _csv_delimiter_candidate(
        rows,
        header_index,
        header_has_hint=header_has_hint,
        delimiter=delimiter,
        quotechar=quotechar,
    )


def _score_csv_delimiter_stream(
    path: Path,
    plan: _CsvEncodingPlan,
    *,
    delimiter: str,
    quotechar: str,
) -> _CsvDelimiterCandidate:
    fallback_state: _CsvColumnConsistencyState | None = None
    hint_state: _CsvColumnConsistencyState | None = None
    with _open_csv_text(path, plan) as handle:
        reader = csv.reader(
            handle,
            dialect=_csv_delimiter_dialect(delimiter, quotechar=quotechar),
        )
        for row_index, row in enumerate(reader):
            if hint_state is not None:
                hint_state.add(row_index + 1, row)
                continue
            if not any(str(cell or "").strip() for cell in row):
                continue
            if _csv_row_has_header_hint(row):
                hint_state = _CsvColumnConsistencyState(
                    header_index=row_index,
                    header_has_hint=True,
                    header_width=len(row),
                    delimiter=delimiter,
                )
                continue
            if fallback_state is None:
                nonempty = [cell for cell in row if str(cell or "").strip()]
                if len(nonempty) >= 2:
                    fallback_state = _CsvColumnConsistencyState(
                        header_index=row_index,
                        header_has_hint=False,
                        header_width=len(row),
                        delimiter=delimiter,
                    )
                continue
            fallback_state.add(row_index + 1, row)
    state = hint_state or fallback_state
    if state is None:
        return _CsvDelimiterCandidate(
            delimiter=delimiter,
            quotechar=quotechar,
            header_index=None,
            header_has_hint=False,
            header_width=0,
            data_rows=0,
            consistent_rows=0,
            exact_width_rows=0,
            first_offending_row=None,
            first_collapsed_row=None,
        )
    return state.as_candidate(quotechar=quotechar)


def _csv_delimiter_candidate(
    rows: Sequence[Sequence[Any]],
    header_index: int | None,
    *,
    header_has_hint: bool,
    delimiter: str,
    quotechar: str,
) -> _CsvDelimiterCandidate:
    header_width = len(rows[header_index]) if header_index is not None else 0
    data_rows = 0
    consistent_rows = 0
    exact_width_rows = 0
    first_offending_row: int | None = None
    first_collapsed_row: int | None = None
    if header_index is not None and header_width:
        for row_number, row in enumerate(rows[header_index + 1:], start=header_index + 2):
            if not any(str(value or "").strip() for value in row):
                continue
            data_rows += 1
            collapsed = (
                header_width >= 2
                and _csv_short_row_uses_competing_delimiter(row, delimiter=delimiter)
            )
            if collapsed and first_collapsed_row is None:
                first_collapsed_row = row_number
            if len(row) == header_width:
                exact_width_rows += 1
            if _csv_row_matches_delimiter(row, header_width, delimiter=delimiter):
                consistent_rows += 1
            elif first_offending_row is None:
                first_offending_row = row_number
    return _CsvDelimiterCandidate(
        delimiter=delimiter,
        quotechar=quotechar,
        header_index=header_index,
        header_has_hint=header_has_hint,
        header_width=header_width,
        data_rows=data_rows,
        consistent_rows=consistent_rows,
        exact_width_rows=exact_width_rows,
        first_offending_row=first_offending_row,
        first_collapsed_row=first_collapsed_row,
    )


def _csv_delimiter_candidate_key(
    candidate: _CsvDelimiterCandidate,
) -> tuple[bool, bool, bool, bool, float, float, int, int, int, int, int]:
    return (
        candidate.header_index is not None,
        candidate.header_has_hint,
        candidate.data_rows > 0,
        candidate.first_collapsed_row is None,
        candidate.consistency,
        candidate.exactness,
        candidate.exact_width_rows,
        candidate.header_width,
        candidate.consistent_rows,
        -_CSV_DETECT_DELIMITERS.index(candidate.delimiter),
        -_CSV_DETECT_QUOTECHARS.index(candidate.quotechar),
    )


def _csv_row_matches_delimiter(
    row: Sequence[Any],
    header_width: int,
    *,
    delimiter: str,
) -> bool:
    if len(row) == header_width:
        return True
    if len(row) > header_width:
        return True
    return not _csv_short_row_uses_competing_delimiter(row, delimiter=delimiter)


def _csv_short_row_uses_competing_delimiter(
    row: Sequence[Any],
    *,
    delimiter: str,
) -> bool:
    if len(row) != 1:
        return False
    text = str(row[0] or "")
    return any(
        candidate != delimiter and candidate in text
        for candidate in _CSV_DETECT_DELIMITERS
    )


def _csv_delimiter_sample_text(text: str) -> str:
    return "\n".join(text.splitlines()[:_CSV_DELIMITER_SAMPLE_LINES])


def _csv_delimiter_dialect(
    delimiter: str,
    *,
    quotechar: str,
) -> type[csv.Dialect]:
    return type(
        "CandidateDialect",
        (csv.Dialect,),
        {
            "delimiter": delimiter,
            "quotechar": quotechar,
            "escapechar": None,
            "doublequote": True,
            "skipinitialspace": False,
            "lineterminator": "\n",
            "quoting": csv.QUOTE_MINIMAL,
            "strict": False,
        },
    )


def _validate_csv_column_consistency(
    rows: Sequence[Sequence[Any]],
    header_index: int,
    *,
    delimiter: str,
    quotechar: str,
) -> None:
    candidate = _csv_delimiter_candidate(
        rows,
        header_index,
        header_has_hint=_csv_row_has_header_hint(rows[header_index]),
        delimiter=delimiter,
        quotechar=quotechar,
    )
    if candidate.valid:
        return
    _raise_csv_delimiter_error(candidate)


def _raise_csv_delimiter_error(candidate: _CsvDelimiterCandidate) -> None:
    delimiter_name = {
        ",": "comma",
        ";": "semicolon",
        "\t": "tab",
        "|": "pipe",
    }.get(candidate.delimiter, repr(candidate.delimiter))
    row_hint = (
        f"; first collapsed mixed-delimiter row {candidate.first_collapsed_row}"
        if candidate.first_collapsed_row is not None
        else f"; first inconsistent row {candidate.first_offending_row}"
        if candidate.first_offending_row is not None
        else ""
    )
    raise _csv_inconsistent_columns_error(
        "CSV customer data has inconsistent column counts under the best "
        f"delimiter candidate ({delimiter_name}); "
        f"{candidate.consistent_rows}/{candidate.data_rows} data row(s) "
        f"matched the {candidate.header_width}-column header{row_hint}.",
        row_index=candidate.first_collapsed_row or candidate.first_offending_row,
    )


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
    "CsvCustomerDataParseError",
    "FileIntelligenceRepository",
    "load_campaign_opportunities_from_file",
    "normalize_campaign_opportunity_rows",
]
