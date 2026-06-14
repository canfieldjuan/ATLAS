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
_CSV_DETECT_QUOTECHARS = ('"', "'")
_CSV_SNIFFER_SAMPLE_CHARS = 65_536
_CSV_DELIMITER_SAMPLE_LINES = 100
_CSV_DELIMITER_MIN_CONSISTENCY = 0.90
_CSV_NUL_REDECODE_RATIO = 0.10
_CSV_UTF16_NUL_SIDE_RATIO = 0.30
_CSV_REPLACEMENT_WARNING_RATIO = 0.01
_CSV_UTF8_RECOVERY_REPLACEMENT_RATIO = 0.05
_CSV_UTF8_MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00e2\u201a")
_CSV_IMPLAUSIBLE_LEGACY_FALLBACK_CHARS = ("\u00ff", "\u00fe")
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
class _CsvDelimiterCandidate:
    delimiter: str
    quotechar: str
    header_index: int | None
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
    text, decode_warnings = _read_csv_text(path)
    if not text.strip():
        raise ValueError("CSV customer data must include a header row.")
    candidate = _select_csv_delimiter(text)
    reader = csv.reader(
        StringIO(text),
        dialect=_csv_delimiter_dialect(
            candidate.delimiter,
            quotechar=candidate.quotechar,
        ),
    )
    raw_rows = list(reader)
    header_index = _csv_header_index(raw_rows)
    if header_index is None:
        raise ValueError("CSV customer data must include a header row.")
    _validate_csv_column_consistency(
        raw_rows,
        header_index,
        delimiter=candidate.delimiter,
        quotechar=candidate.quotechar,
    )
    load_warnings = decode_warnings + _leading_rows_skipped_warnings(
        raw_rows,
        header_index,
    )
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
    return rows, load_warnings


def _leading_rows_skipped_warnings(
    raw_rows: Sequence[Sequence[Any]],
    header_index: int,
) -> tuple[CampaignOpportunityWarning, ...]:
    skipped: list[tuple[int, Sequence[Any]]] = []
    for index, row in enumerate(raw_rows[:header_index]):
        if any(str(cell or "").strip() for cell in row):
            skipped.append((index + 1, row))
    if not skipped:
        return ()
    first_row_number, first_row = skipped[0]
    preview = _csv_row_preview(first_row)
    return (
        CampaignOpportunityWarning(
            code="csv_leading_rows_skipped",
            message=(
                f"Skipped {len(skipped)} leading row(s) before the CSV header "
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


def _read_csv_text(
    path: Path,
) -> tuple[str, tuple[CampaignOpportunityWarning, ...]]:
    data = path.read_bytes()
    if not data:
        return "", ()
    for bom, encoding in _CSV_BOM_ENCODINGS:
        if data.startswith(bom):
            return _decode_csv_bytes(data, encoding=encoding)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return _decode_utf8_error_csv_bytes(data)
    inferred = _utf16_encoding_from_nul_pattern(data, text)
    if inferred:
        return _decode_inferred_utf16_csv_bytes(data, encoding=inferred)
    if _nul_ratio(text) >= _CSV_NUL_REDECODE_RATIO:
        raise ValueError(
            "CSV customer data decoded to NUL-heavy text; check the file encoding."
        )
    return text, _csv_decode_warnings(text, encoding="utf-8-sig")


def _decode_legacy_csv_bytes(
    data: bytes,
) -> tuple[str, tuple[CampaignOpportunityWarning, ...]]:
    for encoding in ("cp1252", "latin-1"):
        try:
            return _decode_csv_bytes(data, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        "CSV customer data could not be decoded as UTF-8, UTF-16, UTF-32, "
        "CP1252, or Latin-1."
    )


def _decode_utf8_error_csv_bytes(
    data: bytes,
) -> tuple[str, tuple[CampaignOpportunityWarning, ...]]:
    inferred = _utf16_encoding_from_nul_bytes(data)
    if inferred:
        return _decode_inferred_utf16_csv_bytes(data, encoding=inferred)
    legacy_text, legacy_warnings = _decode_legacy_csv_bytes(data)
    recovered_text = data.decode("utf-8-sig", errors="replace")
    if (
        _replacement_ratio(recovered_text) <= _CSV_UTF8_RECOVERY_REPLACEMENT_RATIO
        and _utf8_mojibake_score(legacy_text) > recovered_text.count("\ufffd")
    ):
        return recovered_text, _csv_decode_warnings(
            recovered_text,
            encoding="utf-8-sig",
            warn_on_any_replacement=True,
        )
    return legacy_text, legacy_warnings + _legacy_fallback_corruption_warnings(
        legacy_text,
        recovered_text,
    )


def _decode_inferred_utf16_csv_bytes(
    data: bytes,
    *,
    encoding: str,
) -> tuple[str, tuple[CampaignOpportunityWarning, ...]]:
    decoded, warnings = _decode_csv_bytes(data, encoding=encoding)
    return decoded, warnings + (
        CampaignOpportunityWarning(
            code="csv_encoding_inferred",
            field="encoding",
            message=(
                "CSV contained UTF-16-style NUL bytes without a BOM; "
                f"decoded as {encoding}."
            ),
        ),
    )


def _decode_csv_bytes(
    data: bytes,
    *,
    encoding: str,
) -> tuple[str, tuple[CampaignOpportunityWarning, ...]]:
    text = data.decode(encoding)
    if _nul_ratio(text) >= _CSV_NUL_REDECODE_RATIO:
        raise ValueError(
            f"CSV customer data decoded as {encoding} but remained NUL-heavy; "
            "check the file encoding."
        )
    return text, _csv_decode_warnings(text, encoding=encoding)


def _utf16_encoding_from_nul_pattern(data: bytes, text: str) -> str | None:
    if _nul_ratio(text) < _CSV_NUL_REDECODE_RATIO:
        return None
    return _utf16_encoding_from_nul_bytes(data)


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


def _nul_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count("\x00") / len(text)


def _replacement_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count("\ufffd") / len(text)


def _utf8_mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in _CSV_UTF8_MOJIBAKE_MARKERS)


def _legacy_fallback_corruption_warnings(
    legacy_text: str,
    recovered_text: str,
) -> tuple[CampaignOpportunityWarning, ...]:
    artifact_count = _legacy_fallback_implausible_artifact_count(
        legacy_text,
        recovered_text,
    )
    if not artifact_count:
        return ()
    if _utf8_mojibake_score(legacy_text):
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
    legacy_text: str,
    recovered_text: str,
) -> int:
    count = 0
    for index, char in enumerate(recovered_text):
        if char != "\ufffd":
            continue
        legacy_char = legacy_text[index] if index < len(legacy_text) else ""
        if not _is_implausible_legacy_fallback_char(legacy_char):
            continue
        count += 1
    return count


def _is_implausible_legacy_fallback_char(char: str) -> bool:
    if not char:
        return False
    if char in _CSV_IMPLAUSIBLE_LEGACY_FALLBACK_CHARS:
        return True
    codepoint = ord(char)
    return 0x80 <= codepoint <= 0x9F


def _csv_decode_warnings(
    text: str,
    *,
    encoding: str,
    warn_on_any_replacement: bool = False,
) -> tuple[CampaignOpportunityWarning, ...]:
    replacement_count = text.count("\ufffd")
    if not replacement_count:
        return ()
    ratio = _replacement_ratio(text)
    if ratio < _CSV_REPLACEMENT_WARNING_RATIO and not warn_on_any_replacement:
        return ()
    return (
        CampaignOpportunityWarning(
            code="csv_replacement_characters",
            field="encoding",
            message=(
                f"CSV decoded as {encoding} but contains {replacement_count} "
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
    header_index = _csv_header_index(rows)
    return _csv_delimiter_candidate(
        rows,
        header_index,
        delimiter=delimiter,
        quotechar=quotechar,
    )


def _csv_delimiter_candidate(
    rows: Sequence[Sequence[Any]],
    header_index: int | None,
    *,
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
        header_width=header_width,
        data_rows=data_rows,
        consistent_rows=consistent_rows,
        exact_width_rows=exact_width_rows,
        first_offending_row=first_offending_row,
        first_collapsed_row=first_collapsed_row,
    )


def _csv_delimiter_candidate_key(
    candidate: _CsvDelimiterCandidate,
) -> tuple[bool, bool, bool, float, float, int, int, int, int]:
    return (
        candidate.header_index is not None,
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
    raise ValueError(
        "CSV customer data has inconsistent column counts under the best "
        f"delimiter candidate ({delimiter_name}); "
        f"{candidate.consistent_rows}/{candidate.data_rows} data row(s) "
        f"matched the {candidate.header_width}-column header{row_hint}. "
        "Check the delimiter and header row."
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
    "FileIntelligenceRepository",
    "load_campaign_opportunities_from_file",
    "normalize_campaign_opportunity_rows",
]
