"""Shared CLI parsing for support-ticket FAQ custom rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    "load_rule_files",
    "parse_intent_rules",
    "parse_vocabulary_gap_rules",
]
