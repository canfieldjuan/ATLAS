#!/usr/bin/env python3
"""Advisory lint for unrecorded operator re-scope decisions in plan docs."""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RULE = (
    "Decision-recording: an operator decision that re-scopes an umbrella issue "
    "is recorded as a comment on that umbrella at decision time; plan docs may "
    "cite only recorded decisions (URL required)."
)
GITHUB_COMMENT_URL_RE = re.compile(
    r"https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/"
    r"(?P<kind>issues|discussions)/(?P<number>\d+)"
    r"#(?:issuecomment|discussioncomment)-\d+"
)
GITHUB_UMBRELLA_URL_RE = re.compile(
    r"https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/"
    r"(?P<kind>issues|discussions)/(?P<number>\d+)"
)
DECISION_FIELDS = ("recorded decision url", "umbrella issue", "scope effect")
PLACEHOLDER_VALUES = {"", "n/a", "na", "todo", "todo/n/a", "none"}


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


def decision_section(text: str) -> str:
    lines: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("### "):
            if line.strip().lower() == "### decision recording":
                in_section = True
                continue
            if in_section:
                break
        elif line.startswith("## ") and in_section:
            break
        if in_section:
            lines.append(line)
    return "\n".join(lines)


def _decision_field_value(section: str, field: str) -> str | None:
    field_lower = field.lower()
    for line in section.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip().lstrip("-*0123456789. ").strip().lower()
        if label != field_lower:
            continue
        return value.strip().rstrip(".")
    return None


def _is_authored_value(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if "todo" in normalized:
        return False
    return normalized not in PLACEHOLDER_VALUES


def plan_cites_rescope_decision(plan_text: str) -> bool:
    section = decision_section(plan_text)
    return any(_is_authored_value(_decision_field_value(section, field)) for field in DECISION_FIELDS)


def plan_has_recorded_decision_url(plan_text: str) -> bool:
    section = decision_section(plan_text)
    decision_url = _decision_field_value(section, "recorded decision url")
    umbrella_url = _decision_field_value(section, "umbrella issue")
    if not _is_authored_value(decision_url) or not _is_authored_value(umbrella_url):
        return False
    decision_match = GITHUB_COMMENT_URL_RE.fullmatch(decision_url.strip())
    umbrella_match = GITHUB_UMBRELLA_URL_RE.fullmatch(umbrella_url.strip())
    if not decision_match or not umbrella_match:
        return False
    return decision_match.groupdict() == umbrella_match.groupdict()


def scan_plans(plan_texts: dict[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    for path, text in sorted(plan_texts.items()):
        if not plan_cites_rescope_decision(text):
            continue
        if plan_has_recorded_decision_url(text):
            continue
        findings.append(Finding(path=path, reason=RULE))
    return findings


def _git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def changed_plan_texts(base: str) -> dict[str, str]:
    names = [
        n
        for n in _git(
            ["diff", "--name-only", "--diff-filter=AMR", f"{base}...HEAD", "--", "plans/PR-*.md"]
        ).splitlines()
        if n
    ]
    return {name: Path(name).read_text(encoding="utf-8") for name in names if Path(name).exists()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="diff base ref")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on findings")
    args = parser.parse_args(argv)

    findings = scan_plans(changed_plan_texts(args.base))
    print("decision-recording lint (advisory)")
    print("-" * 60)
    if not findings:
        print("OK: no unrecorded re-scope decision cited by changed plans.")
        return 0
    for finding in findings:
        print(f"::warning file={finding.path}::{finding.reason}")
        print(f"  {finding.path}: {finding.reason}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
