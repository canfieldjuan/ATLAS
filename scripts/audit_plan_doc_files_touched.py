#!/usr/bin/env python3
"""Compare a plan doc's declared files against the current git diff."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PATH_PATTERN = re.compile(r"`([^`\n]+)`")
# Optional fix-mode budget declared in the plan *Scope*, e.g. "Max files: 3".
# Read only from the Scope section (so digit-only mentions in other prose do not
# trigger the gate) and fail closed on a malformed value (so a typo cannot
# silently disable the gate).
SCOPE_HEADING_PATTERN = re.compile(r"^##\s+Scope\b")
MAX_FILES_LINE_PATTERN = re.compile(r"(?im)^\s*Max files:\s*(\S.*?)\s*$")


class PlanBudgetError(ValueError):
    """A `Max files:` declaration is present in Scope but is not an integer."""


@dataclass(frozen=True)
class FilesTouchedAudit:
    claimed: set[str]
    actual: set[str]

    @property
    def missing_in_plan(self) -> set[str]:
        return self.actual - self.claimed

    @property
    def extra_in_plan(self) -> set[str]:
        return self.claimed - self.actual

    @property
    def ok(self) -> bool:
        return not self.missing_in_plan and not self.extra_in_plan


def _normalize_heading(line: str) -> str:
    return line.lstrip("#").strip().lower()


def _files_touched_lines(text: str) -> list[str]:
    in_files_touched = False
    lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            in_files_touched = False
        elif line.startswith("### "):
            in_files_touched = _normalize_heading(line) == "files touched"
            continue

        if in_files_touched:
            lines.append(line)
    return lines


def claimed_files_touched(text: str) -> set[str]:
    claimed: set[str] = set()
    for line in _files_touched_lines(text):
        for match in PATH_PATTERN.finditer(line):
            value = match.group(1).strip()
            if value:
                claimed.add(value)
    return claimed


def _scope_section(text: str) -> str:
    """The plan's `## Scope` section body (up to the next `## ` heading)."""
    out: list[str] = []
    in_scope = False
    for line in text.splitlines():
        if line.startswith("## "):
            if SCOPE_HEADING_PATTERN.match(line):
                in_scope = True
                continue
            if in_scope:
                break
        if in_scope:
            out.append(line)
    return "\n".join(out)


def declared_max_files(text: str) -> int | None:
    """The `Max files: N` budget from the plan's Scope, or None if absent.

    Reads only the Scope section, so a digit-only `Max files:` mention in other
    prose or an example does not trigger the budget. Fails closed: a present but
    non-integer value raises `PlanBudgetError` rather than silently disabling the
    gate.
    """
    match = MAX_FILES_LINE_PATTERN.search(_scope_section(text))
    if match is None:
        return None
    raw = match.group(1).strip()
    if not raw.isdigit():
        raise PlanBudgetError(f"malformed 'Max files:' value in Scope: {raw!r}")
    return int(raw)


def actual_diff_files(base_ref: str) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def audit_files_touched(plan_text: str, actual: set[str]) -> FilesTouchedAudit:
    return FilesTouchedAudit(claimed=claimed_files_touched(plan_text), actual=actual)


def _print_paths(label: str, paths: set[str]) -> None:
    for path in sorted(paths):
        print(f"{label:<15} {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a plan doc's declared files against the git diff."
    )
    parser.add_argument("plan", help="path to the plan doc")
    parser.add_argument("base_ref", nargs="?", default="origin/main")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="override the plan's `Max files: N` budget (fix-mode budget cap)",
    )
    args = parser.parse_args(argv)

    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"plan doc not found: {plan_path}", file=sys.stderr)
        return 2

    try:
        actual = actual_diff_files(args.base_ref)
    except RuntimeError as exc:
        print(f"failed to read git diff: {exc}", file=sys.stderr)
        return 2

    plan_text = plan_path.read_text(encoding="utf-8")
    audit = audit_files_touched(plan_text, actual)
    try:
        plan_budget = declared_max_files(plan_text)
    except PlanBudgetError as exc:
        print(f"plan budget error: {exc}", file=sys.stderr)
        return 2
    budget = args.max_files if args.max_files is not None else plan_budget
    over_budget = budget is not None and len(audit.actual) > budget

    print(f"plan doc: {plan_path}")
    print(f"base ref: {args.base_ref}")
    print(f"claimed files: {len(audit.claimed)}")
    print(f"actual files: {len(audit.actual)}")
    if budget is not None:
        print(f"max files: {budget}")
    print("-" * 60)

    if audit.ok and not over_budget:
        print("OK             plan files match git diff")
        return 0

    _print_paths("MISSING", audit.missing_in_plan)
    _print_paths("EXTRA", audit.extra_in_plan)
    if over_budget:
        print(f"OVER BUDGET     {len(audit.actual)} files changed, max {budget}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
