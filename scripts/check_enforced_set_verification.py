#!/usr/bin/env python3
"""Advisory lint for plan Verification sections that omit CI-equivalent proof."""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RULE = (
    "Enforced-set verification: the plan's Verification section must cite the "
    "CI-equivalent command copied from the enforcing workflow, not a hand-picked "
    "test subset."
)
WORKFLOW_MARKERS = (
    "ci-equivalent command copied from enforcing workflow",
    "copied from enforcing workflow",
)
NO_WORKFLOW_MARKERS = (
    "no enforcing workflow applies",
    "closest local command",
)
PLACEHOLDER_VALUES = {"", "n/a", "na", "todo", "todo/n/a", "none"}
NEGATIVE_EXECUTION_RE = re.compile(
    r"\b(?:not run|not executed|unrun|pending|could not run|unable to run|needs .*dependencies)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


def verification_section(text: str) -> str:
    lines: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("## "):
            if line.strip().lower() == "## verification":
                in_section = True
                continue
            if in_section:
                break
        if in_section:
            lines.append(line)
    return "\n".join(lines)


def _marker_value(section: str, marker: str) -> str | None:
    marker_lower = marker.lower()
    for line in section.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip().lstrip("-*0123456789. ").strip().lower()
        if label != marker_lower:
            continue
        return value.strip().rstrip(".").lower()
    return None


def _is_non_placeholder(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if "todo" in normalized:
        return False
    if NEGATIVE_EXECUTION_RE.search(normalized):
        return False
    return normalized not in PLACEHOLDER_VALUES


def _has_marker_values(section: str, markers: Sequence[str]) -> bool:
    return all(_is_non_placeholder(_marker_value(section, marker)) for marker in markers)


def plan_has_enforced_set_verification(plan_text: str) -> bool:
    section = verification_section(plan_text)
    return _has_marker_values(section, WORKFLOW_MARKERS) or _has_marker_values(
        section, NO_WORKFLOW_MARKERS
    )


def scan_plans(plan_texts: dict[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    for path, text in sorted(plan_texts.items()):
        if plan_has_enforced_set_verification(text):
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
    print("enforced-set verification lint (advisory)")
    print("-" * 60)
    if not findings:
        print("OK: changed plans cite CI-equivalent verification.")
        return 0
    for finding in findings:
        print(f"::warning file={finding.path}::{finding.reason}")
        print(f"  {finding.path}: {finding.reason}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
