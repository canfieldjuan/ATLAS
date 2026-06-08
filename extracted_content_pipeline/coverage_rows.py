"""Coverage-row adapters for deterministic content-review evidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from extracted_quality_gate.types import GateSeverity

from .content_pr import CoverageRow, CoverageStatus

_QUALITY_NAMESPACE = "QUALITY-GATE"
_BRAND_VOICE_NAMESPACE = "BRAND-VOICE"
_QUALITY_BLOCKING_FIELDS = ("decision", "verdict", "outcome")
_QUALITY_BLOCKING_VALUES = (
    "approval_required",
    "block",
    "blocked",
    "blocking",
    "deny",
    "denied",
    "fail",
    "failed",
    "failure",
    "reject",
    "rejected",
)


def quality_gate_coverage_rows(
    report: Any,
    *,
    rule_namespace: str = _QUALITY_NAMESPACE,
) -> tuple[CoverageRow, ...]:
    """Map a quality-gate report into Content-PR coverage rows."""

    namespace = _clean(rule_namespace) or _QUALITY_NAMESPACE
    if not (isinstance(report, Mapping) or hasattr(report, "passed")):
        return (_row(namespace, "report", "quality report is present"),)

    passed = _get(report, "passed")
    passed = passed if isinstance(passed, bool) else None
    contradiction = _quality_gate_contradiction(report, passed)
    rows: list[CoverageRow] = []
    for index, finding in enumerate(_items(_get(report, "findings")), start=1):
        code, message, severity, field = _finding_parts(finding)
        if not (code or message or severity):
            rows.append(_row(namespace, f"malformed-finding-{index}", "quality finding is well-formed"))
        elif severity == GateSeverity.BLOCKER:
            evidence = f"{message} ({field})" if message and field else message or field or code
            rows.append(_row(namespace, code or message, message or "quality blocker", status=CoverageStatus.FAIL, evidence=evidence))
        elif severity in (GateSeverity.WARNING, GateSeverity.INFO):
            rows.append(_row(namespace, code or message, message or "quality finding", required=False, status=CoverageStatus.FAIL, evidence=message or code))
        else:
            rows.append(_row(namespace, code or message, "quality finding severity is known"))

    if any(row.required and row.status == CoverageStatus.FAIL for row in rows):
        return tuple(rows)
    if contradiction is not None:
        field, value = contradiction
        return (
            _row(
                namespace,
                f"contradictory-{field}",
                f"quality report {field} agrees with passed flag",
                evidence=f"passed=True but {field}={value}",
            ),
            *rows,
        )
    if passed is True:
        return (_row(namespace, "report", "quality report passes deterministic gates", status=CoverageStatus.PASS, evidence="quality report passed"), *rows)
    if passed is False:
        return (_row(namespace, "report", "quality report passes deterministic gates", status=CoverageStatus.FAIL, evidence="quality report failed without blocker findings"), *rows)
    return (_row(namespace, "report", "quality report passed flag is present"), *rows)


def brand_voice_coverage_rows(
    payload: Any,
    *,
    rule_namespace: str = _BRAND_VOICE_NAMESPACE,
) -> tuple[CoverageRow, ...]:
    """Map a brand-voice audit mapping into Content-PR coverage rows."""

    namespace = _clean(rule_namespace) or _BRAND_VOICE_NAMESPACE
    audit = _brand_voice_audit(payload)
    if audit is None:
        return (_row(namespace, "audit", "brand voice audit is present"),)

    rows = [
        _row(namespace, f"warning:{warning}", "brand voice warning", status=CoverageStatus.FAIL, evidence=warning)
        for warning in _strings(audit.get("warnings"))
    ]
    rows.extend(
        _row(namespace, f"banned-term:{term}", "brand voice banned term", status=CoverageStatus.FAIL, evidence=term)
        for term in _strings(audit.get("banned_terms"))
    )
    if rows:
        return tuple(rows)
    if audit.get("passed") is True:
        return (_row(namespace, "audit", "brand voice audit passes", status=CoverageStatus.PASS, evidence="brand voice audit passed"),)
    if audit.get("passed") is False:
        return (_row(namespace, "audit", "brand voice audit passes", status=CoverageStatus.FAIL, evidence="brand voice audit failed without findings"),)
    return (_row(namespace, "audit", "brand voice audit passed flag is present"),)


def _row(
    namespace: str,
    code: str,
    requirement: str,
    *,
    required: bool = True,
    status: CoverageStatus = CoverageStatus.UNRESOLVED,
    evidence: str = "",
) -> CoverageRow:
    return CoverageRow(
        rule_id=f"{namespace}:{_code(code)}",
        requirement=requirement,
        required=required,
        status=status,
        evidence=evidence or (requirement if status == CoverageStatus.FAIL else ""),
    )


def _finding_parts(value: Any) -> tuple[str, str, str, str]:
    return (
        _clean(_get(value, "code")),
        _clean(_get(value, "message")),
        _clean(_enum_value(_get(value, "severity"))).lower(),
        _clean(_get(value, "field_name")),
    )


def _quality_gate_contradiction(report: Any, passed: bool | None) -> tuple[str, str] | None:
    if passed is not True:
        return None
    for field in _QUALITY_BLOCKING_FIELDS:
        value = _signal_token(_get(report, field))
        if value in _QUALITY_BLOCKING_VALUES:
            return field, value
    return None


def _items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return (value,)


def _brand_voice_audit(payload: Any) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    audit = payload.get("_brand_voice_audit", payload.get("brand_voice_audit", payload))
    return audit if isinstance(audit, Mapping) else None


def _get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)


def _strings(value: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(text for text in (_clean(v) for v in _items(value)) if text))


def _code(value: str) -> str:
    return "-".join("".join(char if char.isalnum() else "-" for char in _clean(value).lower()).split("-")) or "row"


def _enum_value(value: Any) -> Any:
    enum_value = getattr(value, "value", None)
    return enum_value if isinstance(enum_value, str) else value


def _signal_token(value: Any) -> str:
    raw = _clean(_enum_value(value)).lower()
    return "_".join("".join(char if char.isalnum() else "_" for char in raw).split("_"))


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = ["brand_voice_coverage_rows", "quality_gate_coverage_rows"]
