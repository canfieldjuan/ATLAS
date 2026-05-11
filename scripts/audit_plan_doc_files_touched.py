#!/usr/bin/env python3
"""Compare a plan doc's declared files against the current git diff."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PATH_PATTERN = re.compile(r"`([^`\n]+)`")


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


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: audit_plan_doc_files_touched.py PLAN [BASE_REF]", file=sys.stderr)
        return 2

    plan_path = Path(sys.argv[1])
    base_ref = sys.argv[2] if len(sys.argv) == 3 else "origin/main"
    if not plan_path.exists():
        print(f"plan doc not found: {plan_path}", file=sys.stderr)
        return 2

    try:
        actual = actual_diff_files(base_ref)
    except RuntimeError as exc:
        print(f"failed to read git diff: {exc}", file=sys.stderr)
        return 2

    audit = audit_files_touched(plan_path.read_text(encoding="utf-8"), actual)

    print(f"plan doc: {plan_path}")
    print(f"base ref: {base_ref}")
    print(f"claimed files: {len(audit.claimed)}")
    print(f"actual files: {len(audit.actual)}")
    print("-" * 60)

    if audit.ok:
        print("OK             plan files match git diff")
        return 0

    _print_paths("MISSING", audit.missing_in_plan)
    _print_paths("EXTRA", audit.extra_in_plan)
    return 1


if __name__ == "__main__":
    sys.exit(main())
