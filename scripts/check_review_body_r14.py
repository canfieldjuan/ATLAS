#!/usr/bin/env python3
"""Validate R14 evidence in reviewer LGTM bodies.

R14 requires reviewer verdicts to be backed by the checked-out codebase, not
the PR story. This checker owns the review-body shape that is mechanically
checkable before a reviewer posts LGTM: reviewed head, codebase-verification
evidence, not-verified disclosure, and an R14 rule result.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

REVIEWED_HEAD_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\*{0,2}Reviewed head\*{0,2}\s*:\*{0,2}\s*`?([0-9a-f]{7,40})`?\b",
    re.IGNORECASE | re.MULTILINE,
)
CODEBASE_SECTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|(?:[-*]\s*)?\*\*)?Codebase verification\s*\(R14\)\s*:?\*?\*?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
HEADING_RE = re.compile(r"^\s*(#{1,6})\s+\S")
BOLD_HEADING_RE = re.compile(r"^\s*\*\*[^*\n]+:\*\*\s*$")
LABEL_RE_TEMPLATE = r"^[ \t]*(?:[-*][ \t]*)?{label}[ \t]*:[ \t]*(?P<value>.*)$"
LABEL_AT_HEAD_RE_TEMPLATE = (
    r"^[ \t]*(?:[-*][ \t]*)?{label}(?:[ \t]+at[ \t]+HEAD)?"
    r"[ \t]*:[ \t]*(?P<value>.*)$"
)
R14_RULE_PASS_RE = re.compile(
    r"\bR14\b[^\n]*\bPass\b",
    re.IGNORECASE,
)
R14_RULE_NON_PASS_RE = re.compile(
    r"\bR14\b[^\n]*(?:\bFail\b|N[-/ ]?A|Not applicable)\b",
    re.IGNORECASE,
)
AUTO_LGTM_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:Verdict\s*:\s*)?LGTM(?:[.,!?:]|\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

PLACEHOLDER_RE = re.compile(
    r"^\s*(?:|todo|tbd|n/?a|none|not checked|not verified|<[^>]+>|\.\.\.)\s*$",
    re.IGNORECASE,
)
CHANGED_CODE_LABELS = ("Changed code inspected",)
SPOT_CHECK_LABELS = ("Caller/test/artifact spot-checks", "Spot-checks")
NOT_VERIFIED_LABELS = ("Not verified",)


def requires_r14(body: str, verdict: str) -> bool:
    """Return whether this review body must satisfy R14 evidence checks."""
    normalized = verdict.lower()
    if normalized == "lgtm":
        return True
    if normalized in {"non-lgtm", "non_lgtm", "not-lgtm", "not_lgtm"}:
        return False
    return AUTO_LGTM_RE.search(body) is not None


def extract_codebase_section(body: str) -> str | None:
    """Return the Codebase verification (R14) section body, if present."""
    lines = body.splitlines()
    start: int | None = None
    for idx, line in enumerate(lines):
        if CODEBASE_SECTION_RE.match(line):
            start = idx
            break
    if start is None:
        return None

    collected: list[str] = []
    for line in lines[start + 1:]:
        if HEADING_RE.match(line) or BOLD_HEADING_RE.match(line):
            break
        collected.append(line)
    return "\n".join(collected)


def _label_value(
    section: str,
    labels: Sequence[str],
    *,
    allow_at_head: bool = False,
) -> str | None:
    template = LABEL_AT_HEAD_RE_TEMPLATE if allow_at_head else LABEL_RE_TEMPLATE
    for label in labels:
        pattern = re.compile(
            template.format(label=re.escape(label)),
            re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(section)
        if match:
            return match.group("value").strip()
    return None


def _expected_label_text(labels: Sequence[str], *, allow_at_head: bool = False) -> str:
    expected: list[str] = []
    for label in labels:
        expected.append(f"{label}:")
        if allow_at_head:
            expected.append(f"{label} at HEAD:")
    return " or ".join(f"'{label}'" for label in expected)


def _has_evidence_value(value: str | None, *, allow_none: bool = False) -> bool:
    if value is None:
        return False
    if allow_none and value.strip().lower() == "none":
        return True
    return PLACEHOLDER_RE.match(value) is None


def r14_errors(body: str, verdict: str = "auto") -> list[str]:
    """Return R14 evidence errors for a review body."""
    if not requires_r14(body, verdict):
        return []

    errors: list[str] = []
    if not REVIEWED_HEAD_RE.search(body):
        errors.append("missing reviewed head SHA")

    section = extract_codebase_section(body)
    if section is None:
        errors.append("missing Codebase verification (R14) section")
    else:
        changed = _label_value(section, CHANGED_CODE_LABELS, allow_at_head=True)
        if not _has_evidence_value(changed):
            errors.append(
                "missing non-placeholder changed-code evidence "
                f"(expected {_expected_label_text(CHANGED_CODE_LABELS, allow_at_head=True)})"
            )

        spot_checks = _label_value(section, SPOT_CHECK_LABELS, allow_at_head=True)
        if not _has_evidence_value(spot_checks):
            errors.append(
                "missing non-placeholder caller/test/artifact spot-checks "
                f"(expected {_expected_label_text(SPOT_CHECK_LABELS, allow_at_head=True)})"
            )

        not_verified = _label_value(section, NOT_VERIFIED_LABELS)
        if not _has_evidence_value(not_verified, allow_none=True):
            errors.append(
                "missing not-verified disclosure "
                f"(expected {_expected_label_text(NOT_VERIFIED_LABELS)})"
            )

    if R14_RULE_NON_PASS_RE.search(body):
        errors.append("R14 rule result must pass for LGTM")
    elif not R14_RULE_PASS_RE.search(body):
        errors.append("missing passing R14 rule result")

    return errors


def _read_body(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate R14 evidence in reviewer LGTM bodies."
    )
    parser.add_argument("body_file", help="review body file, or '-' for stdin")
    parser.add_argument(
        "--verdict",
        choices=("auto", "lgtm", "non-lgtm"),
        default="auto",
        help="force whether R14 is required; auto detects final LGTM wording",
    )
    args = parser.parse_args(argv)

    try:
        body = _read_body(args.body_file)
    except OSError as exc:
        print(f"review body R14 audit: cannot read {args.body_file}: {exc}", file=sys.stderr)
        return 2

    errors = r14_errors(body, verdict=args.verdict)
    print("review body R14 audit")
    print(f"body file: {args.body_file}")
    print(f"verdict mode: {args.verdict}")
    print("-" * 60)
    if errors:
        for error in errors:
            print(f"  - {error}")
        return 1
    if requires_r14(body, args.verdict):
        print("OK: R14 LGTM evidence is present.")
    else:
        print("OK: R14 evidence not required for this non-LGTM review body.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
