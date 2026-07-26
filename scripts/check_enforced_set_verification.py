#!/usr/bin/env python3
"""Advisory lint for plan Verification sections that omit CI-equivalent proof."""
from __future__ import annotations

import argparse
import fnmatch
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
WORKFLOW_COMMAND_MARKER = "ci-equivalent command copied from enforcing workflow"
WORKFLOW_SOURCE_MARKER = "copied from enforcing workflow"
WORKFLOW_MARKERS = (WORKFLOW_COMMAND_MARKER, WORKFLOW_SOURCE_MARKER)
NO_WORKFLOW_REASON_MARKER = "no enforcing workflow applies"
NO_WORKFLOW_COMMAND_MARKER = "closest local command"
NO_WORKFLOW_MARKERS = (NO_WORKFLOW_REASON_MARKER, NO_WORKFLOW_COMMAND_MARKER)
PLACEHOLDER_VALUES = {"", "n/a", "na", "todo", "todo/n/a", "none"}
NEGATIVE_EXECUTION_RE = re.compile(
    r"\b(?:not run|not executed|unrun|pending|could not run|unable to run|skipped|needs .*dependencies)\b",
    re.IGNORECASE,
)
AFFIRMATIVE_EXECUTION_RE = re.compile(
    r"\b(?:passed|success|succeeded|ok|green)\b",
    re.IGNORECASE,
)
WORKFLOW_PATH_RE = re.compile(r"^\s*-\s*[\"']?([^\"'\n]+?)[\"']?\s*$")


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


def _normalize_command(value: str | None) -> str | None:
    if not _is_non_placeholder(value):
        return None
    normalized = value.strip().strip("`").rstrip(".").lower()
    if " - " in normalized:
        normalized = normalized.partition(" - ")[0].strip()
    return normalized or None


def _has_affirmative_result_for(section: str, command: str | None) -> bool:
    if not command:
        return False
    for line in section.splitlines():
        normalized = line.lower().replace("`", "")
        line_command, separator, _result = normalized.lstrip("-*0123456789. ").partition(" - ")
        if not separator or line_command.strip() != command:
            continue
        if NEGATIVE_EXECUTION_RE.search(normalized):
            continue
        if AFFIRMATIVE_EXECUTION_RE.search(normalized):
            return True
    return False


def _has_workflow_verification(section: str) -> bool:
    command = _normalize_command(_marker_value(section, WORKFLOW_COMMAND_MARKER))
    return (
        _has_marker_values(section, WORKFLOW_MARKERS)
        and _has_affirmative_result_for(section, command)
    )


def _has_no_workflow_verification(section: str) -> bool:
    command = _normalize_command(_marker_value(section, NO_WORKFLOW_COMMAND_MARKER))
    return (
        _has_marker_values(section, NO_WORKFLOW_MARKERS)
        and _has_affirmative_result_for(section, command)
    )


def _workflow_path_patterns() -> set[str]:
    patterns: set[str] = set()
    for workflow in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        for line in workflow.read_text(encoding="utf-8").splitlines():
            match = WORKFLOW_PATH_RE.match(line)
            if not match:
                continue
            value = match.group(1).strip()
            if value and not value.startswith((".", "!")):
                patterns.add(value)
    return patterns


def path_has_enforcing_workflow(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in _workflow_path_patterns())


def plan_has_enforced_set_verification(plan_text: str, changed_paths: Sequence[str] = ()) -> bool:
    section = verification_section(plan_text)
    if _has_workflow_verification(section):
        return True
    if _has_no_workflow_verification(section):
        return not any(path_has_enforcing_workflow(path) for path in changed_paths)
    return False


def scan_plans(plan_texts: dict[str, str], changed_paths: Sequence[str] = ()) -> list[Finding]:
    findings: list[Finding] = []
    for path, text in sorted(plan_texts.items()):
        if plan_has_enforced_set_verification(text, changed_paths):
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


def changed_paths(base: str) -> list[str]:
    return [n for n in _git(["diff", "--name-only", f"{base}...HEAD"]).splitlines() if n]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="diff base ref")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on findings")
    args = parser.parse_args(argv)

    findings = scan_plans(changed_plan_texts(args.base), changed_paths(args.base))
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
