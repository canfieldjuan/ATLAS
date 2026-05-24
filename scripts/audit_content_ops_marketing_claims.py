#!/usr/bin/env python3
"""Audit Content Ops marketing copy for unsupported ticket-deflection claims."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS = ("docs/products",)
TEXT_SUFFIXES = {".md", ".mdx", ".txt"}
ALLOW_MARKER = "claim-audit: allow"
ALLOW_COMMENT_RE = re.compile(
    r"<!--\s*claim-audit:\s*allow\s+(?P<reason>[^<>]+?)\s*-->",
    re.IGNORECASE,
)
HELPDESK_PLATFORMS = (
    r"zendesk|intercom|gorgias|freshdesk|help scout|"
    r"salesforce service cloud|hubspot(?: service hub)?|shopify|front|"
    r"jira service management"
)
PUBLISH_TARGETS = r"help[- ]?center|knowledge base|docs site|documentation site"


@dataclass(frozen=True)
class ClaimRule:
    code: str
    pattern: re.Pattern[str]
    message: str


@dataclass(frozen=True)
class ClaimFinding:
    path: Path
    line_no: int
    code: str
    message: str
    line: str


@dataclass(frozen=True)
class ClaimAuditError:
    path: Path
    message: str


@dataclass(frozen=True)
class ClaimAuditReport:
    scanned_files: tuple[Path, ...]
    findings: tuple[ClaimFinding, ...]
    errors: tuple[ClaimAuditError, ...]


RULES: tuple[ClaimRule, ...] = (
    ClaimRule(
        code="AUTO_PUBLISH",
        pattern=re.compile(
            r"\b(auto[- ]?publish(?:es|ing)?|automatically publish(?:es|ing)?|"
            rf"publish(?:es)? (?:directly )?to (?:your )?(?:{PUBLISH_TARGETS})|"
            rf"updates? (?:your )?(?:{PUBLISH_TARGETS}))\b",
            re.IGNORECASE,
        ),
        message="Do not claim automatic help-center publishing.",
    ),
    ClaimRule(
        code="LIVE_HELPDESK_INTEGRATION",
        pattern=re.compile(
            rf"\b(connect(?:s|ed)? (?:to )?(?:{HELPDESK_PLATFORMS})|"
            rf"(?:native )?(?:{HELPDESK_PLATFORMS}) integration)\b",
            re.IGNORECASE,
        ),
        message="Do not claim live help-desk integrations until built.",
    ),
    ClaimRule(
        code="GUARANTEED_DEFLECTION",
        pattern=re.compile(
            r"\b(guarantee(?:d|s)? (?:a )?(?:ticket )?(?:reduction|deflection)|"
            r"(?:cut|reduce|deflect) (?:support )?ticket(?: volume|s)? by \d+%|"
            r"\d+% (?:fewer|less) (?:support )?tickets?)(?=$|[^A-Za-z0-9_])",
            re.IGNORECASE,
        ),
        message="Do not claim guaranteed ticket-volume outcomes.",
    ),
    ClaimRule(
        code="SEMANTIC_CLUSTERING",
        pattern=re.compile(
            r"\b(semantic clustering|semantically cluster(?:s|ed|ing)?|"
            r"embedding(?:s)?[- ]?based clustering)\b",
            re.IGNORECASE,
        ),
        message="Use intent/repeated-issue grouping unless semantic clustering lands.",
    ),
    ClaimRule(
        code="COST_RANKING",
        pattern=re.compile(
            r"\b(rank(?:s|ed|ing)? by cost|cost[- ]?rank(?:s|ed|ing)?|"
            r"support cost ranking)\b",
            re.IGNORECASE,
        ),
        message="Do not claim cost ranking without imported cost/handle-time data.",
    ),
    ClaimRule(
        code="UNBOUNDED_HOSTED_UPLOADS",
        pattern=re.compile(
            r"\b(unlimited (?:ticket|upload|row)s?|"
            r"(?:hosted|self[- ]?serve|synchronous)[^.\n]{0,80}"
            r"(?:50,?000|fifty thousand)|"
            r"(?:50,?000|fifty thousand)[^.\n]{0,80}"
            r"(?:hosted|self[- ]?serve|synchronous))\b",
            re.IGNORECASE,
        ),
        message="Do not claim unbounded or 50k hosted synchronous uploads.",
    ),
)


def _display_path(path: Path, *, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _resolve_repo_path(value: str, *, root: Path) -> Path:
    if not value.strip():
        raise ValueError("path cannot be blank")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {value}")
    if ".." in candidate.parts:
        raise ValueError(f"path traversal is not allowed: {value}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {value}") from exc
    return resolved


def _iter_files(paths: Sequence[str], *, root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for raw_path in paths:
        path = _resolve_repo_path(raw_path, root=root)
        if not path.exists():
            raise ValueError(f"path does not exist: {raw_path}")
        if path.is_file():
            if path.suffix.lower() in TEXT_SUFFIXES:
                files.append(path)
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES:
                files.append(child)
    return tuple(dict.fromkeys(files))


def _allowed(line: str) -> bool:
    match = ALLOW_COMMENT_RE.search(line)
    if match is None:
        return False
    reason = match.group("reason").strip()
    words = re.findall(r"[A-Za-z0-9]+", reason)
    return len(reason) >= 12 and len(words) >= 2


def audit_file(path: Path, *, root: Path) -> tuple[ClaimFinding, ...]:
    findings: list[ClaimFinding] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _allowed(line):
            continue
        for rule in RULES:
            if rule.pattern.search(line):
                findings.append(
                    ClaimFinding(
                        path=_display_path(path, root=root),
                        line_no=line_no,
                        code=rule.code,
                        message=rule.message,
                        line=line.strip(),
                    )
                )
    return tuple(findings)


def audit_paths_report(
    paths: Sequence[str] = DEFAULT_PATHS,
    *,
    root: Path = REPO_ROOT,
) -> ClaimAuditReport:
    files = _iter_files(paths, root=root)
    if not files:
        raise ValueError("no Markdown/text files found to audit")
    findings: list[ClaimFinding] = []
    errors: list[ClaimAuditError] = []
    for path in files:
        try:
            findings.extend(audit_file(path, root=root))
        except (OSError, ValueError) as exc:
            errors.append(
                ClaimAuditError(
                    path=_display_path(path, root=root),
                    message=str(exc),
                )
            )
    return ClaimAuditReport(
        scanned_files=tuple(_display_path(path, root=root) for path in files),
        findings=tuple(findings),
        errors=tuple(errors),
    )


def audit_paths(paths: Sequence[str] = DEFAULT_PATHS, *, root: Path = REPO_ROOT) -> tuple[ClaimFinding, ...]:
    report = audit_paths_report(paths, root=root)
    if report.errors:
        first = report.errors[0]
        raise ValueError(f"could not audit {first.path}: {first.message}")
    return report.findings


def render_findings(findings: Iterable[ClaimFinding]) -> str:
    lines = []
    for finding in findings:
        lines.append(
            f"{finding.path}:{finding.line_no}: {finding.code}: "
            f"{finding.message} [{finding.line}]"
        )
    return "\n".join(lines)


def render_errors(errors: Iterable[ClaimAuditError]) -> str:
    lines = []
    for error in errors:
        lines.append(f"{error.path}: ERROR: {error.message}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="Repo-relative files or directories to audit.",
    )
    args = parser.parse_args(argv)

    try:
        report = audit_paths_report(args.paths, root=REPO_ROOT)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if report.errors:
        if report.findings:
            print(render_findings(report.findings))
        print(render_errors(report.errors), file=sys.stderr)
        return 2
    findings = report.findings
    if findings:
        print(render_findings(findings))
        return 1
    print(
        "OK: no unsupported Content Ops marketing claims found "
        f"({len(report.scanned_files)} files scanned)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
