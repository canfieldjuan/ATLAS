#!/usr/bin/env python3
"""Compare a plan doc's estimated diff size against the current git diff."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TOTAL_PATTERN = re.compile(r"\|\s*\*{0,2}Total\*{0,2}\s*\|\s*\*{0,2}~?([0-9,]+)")


@dataclass(frozen=True)
class DiffSizeAudit:
    estimate: int
    actual: int
    soft_threshold: float = 0.25
    hard_threshold: float = 0.50

    @property
    def drift_ratio(self) -> float:
        if self.estimate == 0:
            return 0.0 if self.actual == 0 else 1.0
        return abs(self.actual - self.estimate) / self.estimate

    @property
    def status(self) -> str:
        if self.drift_ratio > self.hard_threshold:
            return "FAIL"
        if self.drift_ratio > self.soft_threshold:
            return "WARN"
        return "OK"

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"


def _estimated_diff_size_section(text: str) -> str:
    lines: list[str] = []
    in_section = False

    for line in text.splitlines():
        if line.startswith("## "):
            in_section = line[3:].strip().lower() == "estimated diff size"
            continue
        if in_section:
            lines.append(line)
    return "\n".join(lines)


def estimated_total_loc(text: str) -> int | None:
    section = _estimated_diff_size_section(text)
    match = TOTAL_PATTERN.search(section)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def actual_diff_loc(base_ref: str) -> int:
    result = subprocess.run(
        ["git", "diff", "--numstat", f"{base_ref}...HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")

    total = 0
    for line in result.stdout.splitlines():
        added, deleted, *_ = line.split("\t")
        if added == "-" or deleted == "-":
            continue
        total += int(added) + int(deleted)
    return total


def audit_diff_size(plan_text: str, actual: int) -> DiffSizeAudit | None:
    estimate = estimated_total_loc(plan_text)
    if estimate is None:
        return None
    return DiffSizeAudit(estimate=estimate, actual=actual)


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: audit_plan_doc_diff_size.py PLAN [BASE_REF]", file=sys.stderr)
        return 2

    plan_path = Path(sys.argv[1])
    base_ref = sys.argv[2] if len(sys.argv) == 3 else "origin/main"
    if not plan_path.exists():
        print(f"plan doc not found: {plan_path}", file=sys.stderr)
        return 2

    try:
        actual = actual_diff_loc(base_ref)
    except RuntimeError as exc:
        print(f"failed to read git diff: {exc}", file=sys.stderr)
        return 2

    audit = audit_diff_size(plan_path.read_text(encoding="utf-8"), actual)
    if audit is None:
        print("estimated diff size total not found", file=sys.stderr)
        return 2

    print(f"plan doc: {plan_path}")
    print(f"base ref: {base_ref}")
    print(f"estimate loc: {audit.estimate}")
    print(f"actual loc: {audit.actual}")
    print(f"drift: {audit.drift_ratio:.1%}")
    print(f"status: {audit.status}")
    return 0 if audit.ok else 1


if __name__ == "__main__":
    sys.exit(main())
