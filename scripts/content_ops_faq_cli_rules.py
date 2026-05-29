"""Shared CLI parsing for support-ticket FAQ custom rules."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

DOCUMENTATION_TERM_FORMATS = ("auto", "text", "json", "jsonl", "csv")
_DOCUMENTATION_TERM_ROW_KEYS = (
    "documentation_term",
    "documentation_terms",
    "term",
    "heading",
    "title",
    "page_title",
    "name",
    "label",
)
_DOCUMENTATION_TERM_LIST_KEYS = (
    "terms",
    "headings",
    "documents",
    "pages",
    "articles",
    "rows",
    "data",
)
_DOCUMENTATION_TERM_KEY_HINT = ", ".join(_DOCUMENTATION_TERM_ROW_KEYS)
_EMPTY_JSONL_LINE = object()


def parse_documentation_terms(
    inline_terms: list[str],
    files: list[Path],
    file_format: str = "auto",
) -> tuple[str, ...]:
    terms: list[str] = []
    terms.extend(inline_terms)
    for path in files:
        terms.extend(_load_documentation_term_file(path, file_format=file_format))
    return _clean_documentation_terms(terms)


def parse_vocabulary_gap_rules(values: list[str]) -> tuple[tuple[str, ...], ...]:
    rules: list[tuple[str, ...]] = []
    for raw in values or ():
        parts = _parse_vocabulary_gap_rule_terms(str(raw))
        if len(parts) < 2:
            raise SystemExit(
                "--vocabulary-gap-rule must include at least two comma-separated terms"
            )
        rules.append(parts)
    return tuple(rules)


def parse_intent_rules(values: list[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rules: list[tuple[str, tuple[str, ...]]] = []
    for value in values or ():
        topic, separator, raw_keywords = str(value).partition("=")
        topic = topic.strip()
        keywords = _parse_intent_rule_keywords(raw_keywords)
        if not separator or not topic or not keywords:
            raise SystemExit(
                "--intent-rule must use topic=keyword,keyword with at least one keyword"
            )
        rules.append((topic, keywords))
    return tuple(rules)


def load_rule_files(paths: list[Path]) -> dict[str, tuple[Any, ...]]:
    intent_rules: list[tuple[str, tuple[str, ...]]] = []
    vocabulary_gap_rules: list[tuple[str, ...]] = []
    for path in paths:
        payload = _load_rule_file(path)
        intent_rules.extend(_parse_intent_rule_payloads(payload.get("intent_rules", []), path))
        vocabulary_gap_rules.extend(
            _parse_vocabulary_gap_rule_payloads(
                payload.get("vocabulary_gap_rules", []),
                path,
            )
        )
    return {
        "intent_rules": tuple(intent_rules),
        "vocabulary_gap_rules": tuple(vocabulary_gap_rules),
    }


def _parse_vocabulary_gap_rule_terms(value: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        term = part.strip()
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return tuple(terms)


def _parse_intent_rule_keywords(value: str) -> tuple[str, ...]:
    keywords: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        keyword = part.strip()
        key = keyword.lower()
        if not keyword or key in seen:
            continue
        seen.add(key)
        keywords.append(keyword)
    return tuple(keywords)


def _load_documentation_term_file(
    path: Path,
    *,
    file_format: str = "auto",
) -> tuple[str, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"--documentation-term-file not found: {path}") from None
    suffix = path.suffix.lower()
    resolved_format = file_format
    if resolved_format == "auto":
        if suffix == ".json":
            resolved_format = "json"
        elif suffix == ".jsonl":
            resolved_format = "jsonl"
        elif suffix == ".csv":
            resolved_format = "csv"
        else:
            resolved_format = "text"
    if resolved_format == "json":
        terms = _load_documentation_term_json(text, path)
    elif resolved_format == "jsonl":
        terms = _load_documentation_term_jsonl(text, path)
    elif resolved_format == "csv":
        terms = _load_documentation_term_csv(text, path)
    else:
        terms = _load_documentation_term_text(text)
    if not terms:
        raise SystemExit(f"--documentation-term-file contains no terms: {path}")
    return tuple(terms)


def _load_documentation_term_text(text: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _load_documentation_term_json(text: str, path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"--documentation-term-file must be valid JSON: {path}: {exc.msg}"
        ) from None
    terms = tuple(_documentation_terms_from_payload(payload))
    if not terms and _payload_has_unrecognized_term_fields(payload):
        _raise_unrecognized_documentation_term_fields(path)
    return terms


def _load_documentation_term_jsonl(text: str, path: Path) -> tuple[str, ...]:
    saw_unrecognized_fields = False
    terms: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        payload = _parse_documentation_term_jsonl_payload(line, path, line_no)
        if payload is _EMPTY_JSONL_LINE:
            continue
        line_terms = tuple(_documentation_terms_from_payload(payload))
        if not line_terms and _payload_has_unrecognized_term_fields(payload):
            saw_unrecognized_fields = True
        terms.extend(line_terms)
    if not terms and saw_unrecognized_fields:
        _raise_unrecognized_documentation_term_fields(path)
    return tuple(terms)


def _parse_documentation_term_jsonl_payload(
    line: str,
    path: Path,
    line_no: int,
) -> Any:
    text = line.strip()
    if not text:
        return _EMPTY_JSONL_LINE
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "--documentation-term-file must be valid JSONL: "
            f"{path}: line {line_no}: {exc.msg}"
        ) from None


def _load_documentation_term_csv(text: str, path: Path) -> tuple[str, ...]:
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if not rows:
        return ()
    terms = tuple(
        term
        for row in rows
        for term in _documentation_terms_from_mapping(row)
    )
    if not terms:
        _raise_unrecognized_documentation_term_fields(path)
    return terms


def _documentation_terms_from_payload(payload: Any) -> tuple[str, ...]:
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, dict):
        terms = list(_documentation_terms_from_mapping(payload))
        for key in _DOCUMENTATION_TERM_LIST_KEYS:
            value = _case_insensitive_get(payload, key)
            if value is not None:
                terms.extend(_documentation_terms_from_payload(value))
        return tuple(terms)
    if isinstance(payload, list):
        return tuple(
            term
            for item in payload
            for term in _documentation_terms_from_payload(item)
        )
    return ()


def _payload_has_unrecognized_term_fields(payload: Any) -> bool:
    if isinstance(payload, dict):
        has_row_key = any(
            _case_insensitive_has_key(payload, key)
            for key in _DOCUMENTATION_TERM_ROW_KEYS
        )
        list_values = [
            value
            for key in _DOCUMENTATION_TERM_LIST_KEYS
            if (value := _case_insensitive_get(payload, key)) is not None
        ]
        if not has_row_key and not list_values:
            return bool(payload)
        return any(_payload_has_unrecognized_term_fields(value) for value in list_values)
    if isinstance(payload, list):
        return any(_payload_has_unrecognized_term_fields(item) for item in payload)
    return False


def _documentation_terms_from_mapping(
    row: dict[str, Any],
) -> tuple[str, ...]:
    for key in _DOCUMENTATION_TERM_ROW_KEYS:
        value = _case_insensitive_get(row, key)
        if value not in (None, ""):
            if isinstance(value, (dict, list)):
                return _documentation_terms_from_payload(value)
            return (str(value),)
    return ()


def _case_insensitive_get(row: dict[str, Any], key: str) -> Any:
    target = key.lower()
    for raw_key, value in row.items():
        if str(raw_key).lstrip("\ufeff").lower() == target:
            return value
    return None


def _case_insensitive_has_key(row: dict[str, Any], key: str) -> bool:
    target = key.lower()
    return any(str(raw_key).lstrip("\ufeff").lower() == target for raw_key in row)


def _raise_unrecognized_documentation_term_fields(path: Path) -> None:
    raise SystemExit(
        "--documentation-term-file has no recognized term fields: "
        f"{path}; expected one of: {_DOCUMENTATION_TERM_KEY_HINT}"
    )


def _clean_documentation_terms(terms: list[str]) -> tuple[str, ...]:
    out: dict[str, str] = {}
    for term in terms:
        text = " ".join(str(term).split())
        if text:
            out.setdefault(text.lower(), text)
    return tuple(out.values())


def _load_rule_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"--rule-file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--rule-file must be valid JSON: {path}: {exc.msg}") from None
    if not isinstance(payload, dict):
        raise SystemExit(f"--rule-file must contain a JSON object: {path}")
    allowed = {"intent_rules", "vocabulary_gap_rules"}
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise SystemExit(
            f"--rule-file contains unsupported key(s): {', '.join(unknown)}"
        )
    return payload


def _parse_intent_rule_payloads(
    values: Any,
    path: Path,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(values, list):
        raise SystemExit(f"--rule-file intent_rules must be an array: {path}")
    rules: list[tuple[str, tuple[str, ...]]] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise SystemExit(
                f"--rule-file intent_rules[{index}] must be an object: {path}"
            )
        topic = _rule_file_text(
            value.get("topic"),
            path=path,
            label=f"intent_rules[{index}].topic",
            forbidden=("=", ","),
        )
        keywords = value.get("keywords")
        if not isinstance(keywords, list):
            raise SystemExit(
                f"--rule-file intent_rules[{index}].keywords must be an array: {path}"
            )
        parsed_keywords = [
            _rule_file_text(
                keyword,
                path=path,
                label=f"intent_rules[{index}].keywords",
                forbidden=(",",),
            )
            for keyword in keywords
        ]
        rule_text = f"{topic}={','.join(parsed_keywords)}"
        try:
            rules.extend(parse_intent_rules([rule_text]))
        except SystemExit as exc:
            raise SystemExit(
                f"--rule-file intent_rules[{index}] is invalid: {path}: {exc}"
            ) from None
    return tuple(rules)


def _parse_vocabulary_gap_rule_payloads(
    values: Any,
    path: Path,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(values, list):
        raise SystemExit(f"--rule-file vocabulary_gap_rules must be an array: {path}")
    rules: list[tuple[str, ...]] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, list):
            raise SystemExit(
                f"--rule-file vocabulary_gap_rules[{index}] must be an array: {path}"
            )
        aliases = [
            _rule_file_text(
                alias,
                path=path,
                label=f"vocabulary_gap_rules[{index}]",
                forbidden=(",",),
            )
            for alias in value
        ]
        rule_text = ",".join(aliases)
        try:
            rules.extend(parse_vocabulary_gap_rules([rule_text]))
        except SystemExit as exc:
            raise SystemExit(
                f"--rule-file vocabulary_gap_rules[{index}] is invalid: {path}: {exc}"
            ) from None
    return tuple(rules)


def _rule_file_text(
    value: Any,
    *,
    path: Path,
    label: str,
    forbidden: tuple[str, ...],
) -> str:
    if not isinstance(value, str):
        raise SystemExit(f"--rule-file {label} must contain string values: {path}")
    text = value.strip()
    for delimiter in forbidden:
        if delimiter in text:
            raise SystemExit(
                f"--rule-file {label} cannot contain delimiter {delimiter!r}: {path}"
            )
    return text


__all__ = [
    "DOCUMENTATION_TERM_FORMATS",
    "load_rule_files",
    "parse_documentation_terms",
    "parse_intent_rules",
    "parse_vocabulary_gap_rules",
]
