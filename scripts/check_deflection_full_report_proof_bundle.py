#!/usr/bin/env python3
"""Fail-closed redaction check for deflection full-report proof bundles."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org", "example.net", "example.test", "invalid"}
CHECK_KEYS = (
    "request_id",
    "result_url",
    "customer_email",
    "absolute_local_path",
    "stripe_checkout_session_id",
    "stripe_payment_intent_id",
    "raw_evidence_quote",
    "source_id_list",
    "private_note",
    "artifact_readability",
)
MATCH_ONLY_SNIPPET_KEYS = {"raw_evidence_quote", "source_id_list", "private_note", "artifact_readability"}

_REQUEST_ID_RE = re.compile(r"\bcontent-ops-[A-Za-z0-9_-]{8,}\b")
_RESULT_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+/"
    r"(?:systems/support-ticket-deflection|services/faq-deflection)/results/"
    r"[^\s\"'<>]+"
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,}|invalid)\b", re.IGNORECASE)
_ABSOLUTE_LOCAL_PATH_RE = re.compile(
    r"(?<![\w:.-])(?:/(?:home|Users|tmp|var|root|workspace|mnt|opt)/[^\s\"'<>]+|[A-Za-z]:\\[^\s\"'<>]+)"
)
_STRIPE_CHECKOUT_RE = re.compile(r"\bcs_(?:test|live)_[A-Za-z0-9]{8,}\b")
_STRIPE_PAYMENT_RE = re.compile(r"\bpi_[A-Za-z0-9]{8,}\b")
_RAW_EVIDENCE_RE = re.compile(r"(?i)(\"evidence_quotes?\"\s*:|`[^`]+`\s*-\s+|raw evidence|complete evidence)")
_SOURCE_ID_LIST_RE = re.compile(
    r"(?im)(\"source_ids?\"\s*:|(?:^|[\n\r,])source_ids?(?:[, \t\r\n]|$)|source ids? \(full list\)|full source[-_ ]id list)"
)
_PRIVATE_NOTE_RE = re.compile(
    r"(?im)(private note|\"public\"\s*:\s*false|\"is_public\"\s*:\s*false|\"visibility\"\s*:\s*\"private\"|(?:^|[\n\r,])(?:public|is_public)\s*,\s*false(?:[, \t\r\n]|$)|(?:^|[\n\r,])visibility\s*,\s*private(?:[, \t\r\n]|$))"
)

DETECTORS = (
    ("request_id", _REQUEST_ID_RE, lambda value: _is_safe_request_id(value)),
    ("result_url", _RESULT_URL_RE, lambda value: _is_safe_result_url(value)),
    ("customer_email", _EMAIL_RE, lambda value: _is_safe_email(value)),
    ("absolute_local_path", _ABSOLUTE_LOCAL_PATH_RE, None),
    ("stripe_checkout_session_id", _STRIPE_CHECKOUT_RE, None),
    ("stripe_payment_intent_id", _STRIPE_PAYMENT_RE, None),
    ("raw_evidence_quote", _RAW_EVIDENCE_RE, None),
    ("source_id_list", _SOURCE_ID_LIST_RE, None),
    ("private_note", _PRIVATE_NOTE_RE, None),
)


@dataclass(frozen=True)
class Finding:
    path: str
    label: str
    snippet: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": _redact_sensitive_text(self.path),
            "label": self.label,
            "snippet": self.snippet,
        }


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            yield child


def iter_artifact_paths(path: Path) -> Iterable[Path]:
    yield path
    if path.is_dir():
        yield from sorted(path.rglob("*"))


def _read_text_or_readability_finding(path: Path, root: Path) -> tuple[str | None, Finding | None]:
    if path.suffix.lower() == ".pdf":
        return None, _readability_finding(path, root, "unsupported_pdf_artifact")
    try:
        data = path.read_bytes()
    except OSError:
        return None, _readability_finding(path, root, "unreadable_artifact")
    if b"\x00" in data:
        return None, _readability_finding(path, root, "binary_artifact")
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, _readability_finding(path, root, "non_utf8_artifact")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _display_path(path: Path, root: Path) -> str:
    if path == root:
        return path.name or "."
    return _relative(path, root)


def _redact_sensitive_text(text: str) -> str:
    text = _RESULT_URL_RE.sub("[result_url]", text)
    text = _REQUEST_ID_RE.sub("[request_id]", text)
    text = _EMAIL_RE.sub("[email]", text)
    text = _STRIPE_CHECKOUT_RE.sub("[stripe_checkout_session_id]", text)
    text = _STRIPE_PAYMENT_RE.sub("[stripe_payment_intent_id]", text)
    text = _ABSOLUTE_LOCAL_PATH_RE.sub("[absolute_local_path]", text)
    return text


def _safe_snippet(key: str, text: str, start: int, end: int) -> str:
    if key in MATCH_ONLY_SNIPPET_KEYS:
        return f"[{key}]"
    snippet = text[max(0, start - 40) : min(len(text), end + 40)]
    snippet = _redact_sensitive_text(snippet)
    return " ".join(snippet.split())


def _is_safe_request_id(value: str) -> bool:
    lowered = value.lower()
    return "synthetic" in lowered or "example" in lowered


def _is_safe_result_url(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if host.endswith((".example.com", ".example.org", ".example.net", ".example.test")):
        return True
    result_id = unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
    return _is_safe_request_id(result_id)


def _is_safe_email(value: str) -> bool:
    domain = value.rsplit("@", 1)[-1].lower()
    return domain in ALLOWED_EMAIL_DOMAINS or domain.endswith(".example.com")


def _regex_findings(
    *,
    key: str,
    pattern: re.Pattern[str],
    path: Path,
    root: Path,
    text: str,
    safe_value: Any = None,
    display_path: str | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    for match in pattern.finditer(text):
        value = match.group(0)
        if safe_value is not None and safe_value(value):
            continue
        findings.append(
            Finding(
                path=display_path or _display_path(path, root),
                label=key,
                snippet=_safe_snippet(key, text, match.start(), match.end()),
            )
        )
    return findings


def _readability_finding(path: Path, root: Path, reason: str) -> Finding:
    return Finding(
        path=_display_path(path, root),
        label="artifact_readability",
        snippet=f"[{reason}]",
    )


def _record_findings(checks: dict[str, dict[str, Any]], findings: list[Finding]) -> None:
    for finding in findings:
        checks[finding.label]["ok"] = False
        checks[finding.label]["findings"].append(finding.as_dict())


def scan_bundle(path: Path) -> dict[str, Any]:
    root = path if path.is_dir() else path.parent
    checks: dict[str, dict[str, Any]] = {
        key: {"ok": True, "findings": []} for key in CHECK_KEYS
    }
    if not path.exists():
        checks["artifact_readability"]["ok"] = False
        checks["artifact_readability"]["findings"].append(
            Finding(path=path.as_posix(), label="artifact_readability", snippet="[missing_bundle_path]").as_dict()
        )
        return _result(scanned_files=0, checks=checks)

    for artifact_path in iter_artifact_paths(path):
        display_path = _display_path(artifact_path, root)
        for key, pattern, safe_value in DETECTORS:
            _record_findings(
                checks,
                _regex_findings(
                    key=key,
                    pattern=pattern,
                    path=artifact_path,
                    root=root,
                    text=display_path,
                    safe_value=safe_value,
                    display_path=display_path,
                ),
            )

    scanned_files = 0
    for file_path in iter_files(path):
        scanned_files += 1
        text, readability_finding = _read_text_or_readability_finding(file_path, root)
        if readability_finding is not None:
            _record_findings(checks, [readability_finding])
            continue
        if text is None:
            continue
        for key, pattern, safe_value in DETECTORS:
            _record_findings(
                checks,
                _regex_findings(
                    key=key,
                    pattern=pattern,
                    path=file_path,
                    root=root,
                    text=text,
                    safe_value=safe_value,
                ),
            )

    if scanned_files == 0:
        _record_findings(
            checks,
            [Finding(path=_display_path(path, root), label="artifact_readability", snippet="[empty_bundle]")],
        )
    return _result(scanned_files=scanned_files, checks=checks)


def _result(*, scanned_files: int, checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = all(check["ok"] for check in checks.values())
    return {
        "schema_version": "deflection_full_report_proof_redaction.v1",
        "ok": ok,
        "policy": "live proof runs may commit sanitized scorecards only; raw live bundles stay uncommitted",
        "scanned_files": scanned_files,
        "checks": checks,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a deflection full-report proof bundle for commit-blocking leaks."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = scan_bundle(args.path)
    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
