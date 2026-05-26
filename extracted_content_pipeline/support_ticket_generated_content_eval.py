#!/usr/bin/env python3
"""Evaluate generated landing/blog drafts against support-ticket source context."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from extracted_content_pipeline.support_ticket_context_contract import (
        SUPPORT_TICKET_CLUSTER_KEYS,
        SUPPORT_TICKET_SOURCE_COUNT_KEYS,
        is_uploaded_ticket_context,
    )
else:
    from .support_ticket_context_contract import (
        SUPPORT_TICKET_CLUSTER_KEYS,
        SUPPORT_TICKET_SOURCE_COUNT_KEYS,
        is_uploaded_ticket_context,
    )


JsonDict = dict[str, Any]

# From earlier stale smoke benchmark drift; still allowed when source-backed.
_STALE_BENCHMARK_TOKENS = ("186", "78", "42%")
_FRAMING_RE = re.compile(
    r"\b(?:support[-\s]?tickets?|tickets?|faq|help[-\s]?center|answers?)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_UPLOADED_TICKET_TIMEFRAME_RE = re.compile(
    r"\b(?:"
    r"between\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}\s+and|"
    r"(?:over|in|during)\s+the\s+(?:last|past)\s+\d+\s+days|"
    r"(?:last|past)\s+\d+\s+days|"
    r"last\s+90\s+days"
    r")\b",
    re.IGNORECASE,
)
_UNSUPPORTED_UPLOADED_TICKET_CADENCE_RE = re.compile(
    r"\b(?:per|each|every)\s+(?:day|week|month|quarter|year)\b|"
    r"\b(?:daily|weekly|monthly|quarterly|yearly|annually)\b",
    re.IGNORECASE,
)
_PERCENT_CLAIM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<first>\d+(?:\.\d+)?)"
    r"(?:\s*[-\u2013\u2014]\s*(?P<second>\d+(?:\.\d+)?))?\s*%"
)
_UNSUPPORTED_SUPPORT_OUTCOME_CLAIM_PATTERNS = (
    re.compile(
        r"\bsupport\s+tickets(?:\s+for\s+[^.!?\n]{1,80})?\s+drop\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:will|would)\s+(?:reduce|prevent|cut|drop|shrink|eliminate)\b"
        r"[^.!?\n]{0,120}\b(?:support|ticket|queue|churn|retention|customer)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:prevent|prevents|preventing|reduce|reduces|reducing|cut|cuts|"
        r"cutting|eliminate|eliminates|shrinks?|drops?)\b[^.!?\n]{0,120}"
        r"\b(?:support\s+(?:tickets|interactions|volume)|ticket\s+volume|"
        r"support\s+queue)\s+(?:immediately|fastest)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:prevent|prevents|preventing)\s+(?:future\s+)?support\s+"
        r"(?:tickets|interactions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsupport\s+queue\s+will\s+(?:shrink|drop|fall)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcustomers\s+who\s+find\s+answers\b[^.!?\n]{0,160}"
        r"\b(?:stay\s+longer|churn\s+less)\b",
        re.IGNORECASE,
    ),
)


def evaluate_support_ticket_generated_content(
    export: Mapping[str, Any],
    *,
    output: str,
) -> JsonDict:
    """Return deterministic quality checks for one exported generated draft."""

    normalized_output = _normalize_output(output)
    rows = _export_rows(export)
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[JsonDict] = []
    if not rows:
        return _result(
            errors=["export did not include any rows"],
            warnings=[],
            checks=[_check("export_rows_present", False, "error")],
            source_context={},
        )

    row = rows[0]
    context = _source_context(row, output=normalized_output)
    _record(
        checks,
        "support_ticket_context_present",
        bool(context),
        "error",
        errors,
        "export row is missing support-ticket source context",
    )
    if not context:
        return _result(
            errors=errors,
            warnings=warnings,
            checks=checks,
            source_context={},
        )

    text = _generated_text(row, output=normalized_output)
    text_lc = text.lower()
    _record(
        checks,
        "generated_text_present",
        bool(text_lc.strip()),
        "error",
        errors,
        "export row did not contain generated text fields",
    )
    _record(
        checks,
        "support_ticket_framing_present",
        bool(_FRAMING_RE.search(text)),
        "error",
        errors,
        "generated text does not use support-ticket, FAQ, help-center, or answer framing",
    )
    _check_stale_benchmarks(
        checks,
        errors,
        text=text,
        source_context=context,
    )
    _check_count_visibility(
        checks,
        warnings,
        text=text,
        source_context=context,
    )
    _check_uploaded_ticket_timeframe_truthfulness(
        checks,
        errors,
        text=text,
        source_context=context,
    )
    _check_uploaded_ticket_cadence_truthfulness(
        checks,
        errors,
        text=text,
        source_context=context,
    )
    _check_unsupported_percentage_claims(
        checks,
        errors,
        text=text,
        source_context=context,
    )
    _check_unsupported_support_outcome_claims(
        checks,
        errors,
        text=text,
    )
    _check_source_signal_visibility(
        checks,
        errors,
        text_lc=text_lc,
        source_context=context,
    )
    return _result(
        errors=errors,
        warnings=warnings,
        checks=checks,
        source_context=context,
    )


def _normalize_output(output: str) -> str:
    value = str(output or "").strip()
    if value not in {"landing_page", "blog_post"}:
        raise ValueError("--output must be landing_page or blog_post")
    return value


def _export_rows(export: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = export.get("rows") or ()
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        return ()
    return tuple(row for row in rows if isinstance(row, Mapping))


def _source_context(row: Mapping[str, Any], *, output: str) -> Mapping[str, Any]:
    if output == "landing_page":
        metadata = _as_mapping(row.get("metadata"))
        return _as_mapping(metadata.get("source_context"))
    if output == "blog_post":
        return _as_mapping(row.get("data_context"))
    return {}


def _generated_text(row: Mapping[str, Any], *, output: str) -> str:
    if output == "landing_page":
        return _join_text(
            row.get("title"),
            row.get("slug"),
            row.get("hero"),
            row.get("sections"),
            row.get("cta"),
            row.get("meta"),
        )
    if output == "blog_post":
        return _join_text(
            row.get("title"),
            row.get("description"),
            row.get("content"),
            row.get("tags"),
            row.get("charts"),
        )
    return ""


def _check_stale_benchmarks(
    checks: list[JsonDict],
    errors: list[str],
    *,
    text: str,
    source_context: Mapping[str, Any],
) -> None:
    context_text = json.dumps(source_context, sort_keys=True, default=str)
    stale_hits = [
        token
        for token in _STALE_BENCHMARK_TOKENS
        if _benchmark_token_present(text, token)
        and not _benchmark_token_present(context_text, token)
    ]
    passed = not stale_hits
    checks.append({
        "name": "no_stale_benchmark_numbers",
        "passed": passed,
        "level": "error",
        "details": stale_hits,
    })
    if stale_hits:
        errors.append(
            "generated text contains stale benchmark numbers not present in "
            f"source context: {', '.join(stale_hits)}"
        )


def _benchmark_token_present(value: str, token: str) -> bool:
    escaped = re.escape(token)
    return bool(re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9%])", value))


def _check_count_visibility(
    checks: list[JsonDict],
    warnings: list[str],
    *,
    text: str,
    source_context: Mapping[str, Any],
) -> None:
    expected_counts = [
        int(source_context[key])
        for key in SUPPORT_TICKET_SOURCE_COUNT_KEYS
        if _positive_int(source_context.get(key)) is not None
    ]
    visible_counts = [
        count for count in expected_counts if re.search(rf"\b{count}\b", text)
    ]
    passed = not expected_counts or bool(visible_counts)
    checks.append({
        "name": "source_count_visible",
        "passed": passed,
        "level": "warning",
        "details": {
            "expected_counts": expected_counts,
            "visible_counts": visible_counts,
        },
    })
    if not passed:
        warnings.append(
            "generated text does not visibly mention any support-ticket source counts"
        )


def _check_uploaded_ticket_timeframe_truthfulness(
    checks: list[JsonDict],
    errors: list[str],
    *,
    text: str,
    source_context: Mapping[str, Any],
) -> None:
    if not is_uploaded_ticket_context(source_context):
        checks.append({
            "name": "uploaded_ticket_timeframe_truthful",
            "passed": True,
            "level": "error",
            "details": {"applicable": False},
        })
        return
    unsupported_hits = _dedupe(
        [
            match.group(0).strip()
            for match in _UNSUPPORTED_UPLOADED_TICKET_TIMEFRAME_RE.finditer(text)
        ]
    )
    passed = not unsupported_hits
    checks.append({
        "name": "uploaded_ticket_timeframe_truthful",
        "passed": passed,
        "level": "error",
        "details": {"unsupported_timeframes": unsupported_hits},
    })
    if unsupported_hits:
        errors.append(
            "generated text claims a calendar or rolling time window for an "
            "undated uploaded-ticket source: " + ", ".join(unsupported_hits)
        )


def _check_uploaded_ticket_cadence_truthfulness(
    checks: list[JsonDict],
    errors: list[str],
    *,
    text: str,
    source_context: Mapping[str, Any],
) -> None:
    if not is_uploaded_ticket_context(source_context):
        checks.append({
            "name": "uploaded_ticket_cadence_truthful",
            "passed": True,
            "level": "error",
            "details": {"applicable": False},
        })
        return
    unsupported_hits = _dedupe(
        [
            match.group(0).strip()
            for match in _UNSUPPORTED_UPLOADED_TICKET_CADENCE_RE.finditer(text)
        ]
    )
    checks.append({
        "name": "uploaded_ticket_cadence_truthful",
        "passed": not unsupported_hits,
        "level": "error",
        "details": {"unsupported_cadences": unsupported_hits},
    })
    if unsupported_hits:
        errors.append(
            "generated text claims a recurring cadence for an undated "
            "uploaded-ticket source: " + ", ".join(unsupported_hits)
        )


def _check_unsupported_percentage_claims(
    checks: list[JsonDict],
    errors: list[str],
    *,
    text: str,
    source_context: Mapping[str, Any],
) -> None:
    claims = _percentage_claims(text)
    if not claims:
        checks.append({
            "name": "percentage_claims_source_backed",
            "passed": True,
            "level": "error",
            "details": {"claims": [], "unsupported": []},
        })
        return
    allowed = _source_backed_percentages(source_context)
    unsupported = [
        claim["text"]
        for claim in claims
        if not all(value in allowed for value in claim["values"])
    ]
    checks.append({
        "name": "percentage_claims_source_backed",
        "passed": not unsupported,
        "level": "error",
        "details": {
            "claims": [claim["text"] for claim in claims],
            "source_backed_percentages": sorted(allowed),
            "unsupported": unsupported,
        },
    })
    if unsupported:
        errors.append(
            "generated text contains percentage claims not backed by support-ticket "
            "source counts: " + ", ".join(unsupported)
        )


def _percentage_claims(text: str) -> list[JsonDict]:
    claims: list[JsonDict] = []
    for match in _PERCENT_CLAIM_RE.finditer(text):
        values = [_whole_percent(match.group("first"))]
        if match.group("second") is not None:
            values.append(_whole_percent(match.group("second")))
        claims.append({"text": match.group(0).strip(), "values": values})
    return claims


def _source_backed_percentages(source_context: Mapping[str, Any]) -> set[int]:
    counts = [
        count
        for count in (
            _positive_int(source_context.get(key))
            for key in SUPPORT_TICKET_SOURCE_COUNT_KEYS
        )
        if count is not None
    ]
    clusters = _support_ticket_clusters(source_context)
    if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes)):
        for cluster in clusters:
            if isinstance(cluster, Mapping):
                count = _positive_int(cluster.get("count"))
                if count is not None:
                    counts.append(count)
    backed: set[int] = set()
    for numerator in counts:
        for denominator in counts:
            if denominator:
                backed.add(_whole_percent((numerator / denominator) * 100))
    return backed


def _whole_percent(value: str | float) -> int:
    return int(round(float(value)))


def _check_unsupported_support_outcome_claims(
    checks: list[JsonDict],
    errors: list[str],
    *,
    text: str,
) -> None:
    unsupported = _unsupported_support_outcome_claims(text)
    checks.append({
        "name": "support_ticket_outcome_claims_grounded",
        "passed": not unsupported,
        "level": "error",
        "details": {"unsupported_claims": unsupported},
    })
    if unsupported:
        errors.append(
            "generated text guarantees support-ticket or customer outcomes not "
            "backed by uploaded-ticket evidence: " + "; ".join(unsupported)
        )


def _unsupported_support_outcome_claims(text: str) -> list[str]:
    claims: list[str] = []
    for sentence in _sentences(text):
        if any(
            pattern.search(sentence)
            for pattern in _UNSUPPORTED_SUPPORT_OUTCOME_CLAIM_PATTERNS
        ):
            claims.append(sentence)
    return _dedupe(claims)


def _sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [chunk.strip(" -*_`") for chunk in chunks if chunk.strip()]


def _check_source_signal_visibility(
    checks: list[JsonDict],
    errors: list[str],
    *,
    text_lc: str,
    source_context: Mapping[str, Any],
) -> None:
    signals = _source_signals(source_context)
    matched = [signal for signal in signals if _signal_visible(signal, text_lc)]
    passed = bool(matched)
    checks.append({
        "name": "source_signal_visible",
        "passed": passed,
        "level": "error",
        "details": {
            "expected_signal_count": len(signals),
            "matched_signals": matched[:10],
        },
    })
    if not signals:
        errors.append(
            "support-ticket source context did not include clusters, "
            "questions, or wording"
        )
    elif not passed:
        errors.append(
            "generated text does not mention any observed ticket cluster, "
            "customer question, or customer wording example"
        )


def _source_signals(source_context: Mapping[str, Any]) -> list[str]:
    signals: list[str] = []
    clusters = _support_ticket_clusters(source_context)
    if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes)):
        for cluster in clusters:
            if isinstance(cluster, Mapping):
                label = str(cluster.get("label") or "").strip()
                if label:
                    signals.append(label)
    for key in ("faq_questions", "customer_wording_examples"):
        values = source_context.get(key) or ()
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            signals.extend(str(item).strip() for item in values if str(item).strip())
    return _dedupe(signals)


def _signal_visible(signal: str, text_lc: str) -> bool:
    normalized = signal.lower().strip()
    if not normalized:
        return False
    if normalized in text_lc:
        return True
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", normalized)
        if len(word) >= 4
    ]
    if not words:
        return False
    return all(word in text_lc for word in words[:4])


def _result(
    *,
    errors: list[str],
    warnings: list[str],
    checks: list[JsonDict],
    source_context: Mapping[str, Any],
) -> JsonDict:
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "source_context_summary": _source_context_summary(source_context),
    }


def _source_context_summary(source_context: Mapping[str, Any]) -> JsonDict:
    clusters = _support_ticket_clusters(source_context)
    return {
        "source_row_count": source_context.get("source_row_count"),
        "included_ticket_row_count": source_context.get("included_ticket_row_count"),
        "question_like_ticket_count": source_context.get("question_like_ticket_count"),
        "source_period": source_context.get("source_period"),
        "cluster_count": (
            len(clusters)
            if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes))
            else 0
        ),
    }


def _support_ticket_clusters(source_context: Mapping[str, Any]) -> Any:
    for key in SUPPORT_TICKET_CLUSTER_KEYS:
        clusters = source_context.get(key)
        if clusters:
            return clusters
    return None


def _record(
    checks: list[JsonDict],
    name: str,
    passed: bool,
    level: str,
    errors: list[str],
    message: str,
) -> None:
    checks.append({"name": name, "passed": passed, "level": level})
    if not passed:
        errors.append(message)


def _join_text(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        parts.extend(_iter_text(value))
    return "\n".join(parts)


def _iter_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        parts: list[str] = []
        for item in value.values():
            parts.extend(_iter_text(item))
        return parts
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = []
        for item in value:
            parts.extend(_iter_text(item))
        return parts
    return [str(value)]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _load_export(path: Path) -> Mapping[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read export JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"export path must contain JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("export JSON must be an object")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Saved-draft export JSON path")
    parser.add_argument(
        "--output",
        required=True,
        choices=("landing_page", "blog_post"),
        help="Generated asset type contained in the export.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = evaluate_support_ticket_generated_content(
            _load_export(args.path),
            output=args.output,
        )
    except ValueError as exc:
        result = {
            "ok": False,
            "errors": [str(exc)],
            "warnings": [],
            "checks": [],
            "source_context_summary": {},
        }
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
