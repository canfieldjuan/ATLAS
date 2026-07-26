#!/usr/bin/env python3
"""Advisory lint for deployed/default config probing in guard-shaped diffs."""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RULE = (
    "Deployed-config probing: guard PRs must state deployed/default config values "
    "and probe explicit, absent, and default-session shapes; no side effect "
    "before all admissions pass."
)

CODE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".sh"}
BOUNDARY_PATH_PART_RE = re.compile(
    r"(^|[-_./])"
    r"(guard|validat(?:e|or|ion)?|resolver|resolution|admission|intake|claim|"
    r"scope|tenant|auth(?:entication|orization)?)"
    r"($|[-_./])",
    re.IGNORECASE,
)
BOUNDARY_CODE_RE = re.compile(
    r"^(?:[+-]\s*|@@[^@\n]*@@\s*.*?)"
    r"(?:"
    r"(?:export\s+)?(?:async\s+)?function\s+"
    r"[a-zA-Z0-9_$]*(?:guard|validat|resolve|admit|reject|claim|scope|auth)"
    r"[a-zA-Z0-9_$]*\s*\("
    r"|(?:async\s+)?def\s+"
    r"[a-zA-Z0-9_]*(?:guard|validat|resolve|admit|reject|claim|scope|auth)"
    r"[a-zA-Z0-9_]*\s*\("
    r"|(?:const|let|var)\s+"
    r"[a-zA-Z0-9_$]*(?:Guard|Validat|Resolve|Admit|Reject|Claim|Scope|Auth)"
    r"[a-zA-Z0-9_$]*\s*="
    r"|class\s+[a-zA-Z0-9_$]*(?:Guard|Validator|Resolver|Admission|Auth)"
    r"[a-zA-Z0-9_$]*"
    r")",
    re.MULTILINE,
)
CONFIG_FALLBACK_RE = re.compile(
    r"os\.(?:getenv|environ\.get)\([^,\n)]+,\s*[^)\n]+"
    r"|os\.environ\.get\([^,\n)]+,\s*[^)\n]+"
    r"|os\.(?:getenv|environ\.get)\([^)]+\)\s+or\s+[^)\n]+"
    r"|os\.environ\.get\([^)]+\)\s+or\s+[^)\n]+"
    r"|process\.env\.[A-Za-z0-9_]+\s*(?:\|\||\?\?)\s*[^;\n]+"
    r"|process\.env\[[^\]\n]+\]\s*(?:\|\||\?\?)\s*[^;\n]+"
    r"|Deno\.env\.get\([^)\n]+\)\s*(?:\|\||\?\?)\s*[^;\n]+"
    r"|\$\{[A-Za-z_][A-Za-z0-9_]*:(?:-|=)[^}\n]+\}"
)
REQUIRED_PLAN_MARKERS = (
    "deployed-config probing",
    "deployed/default config values",
    "explicit value probe",
    "absent value probe",
    "default-session/default-context probe",
    "side-effect ordering",
)
PROBE_MARKERS = REQUIRED_PLAN_MARKERS[1:]
PLACEHOLDER_VALUES = {"", "n/a", "na", "todo", "todo/n/a", "none"}


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


def _is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        p.name.startswith("test_")
        or "tests/" in path
        or p.name.endswith(("_test.py", ".test.js", ".test.jsx", ".test.ts", ".test.tsx"))
        or p.name.endswith((".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx"))
    )


def _is_process_path(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        path.startswith(("plans/", ".github/"))
        or p.name in {"AGENTS.md", "new_pr_plan.sh"}
        or (path.startswith("scripts/check_") and p.suffix == ".py")
    )


def file_needs_deployed_config_probe(path: str, added: str) -> bool:
    if PurePosixPath(path).suffix not in CODE_SUFFIXES or _is_test_path(path) or _is_process_path(path):
        return False
    return bool(
        CONFIG_FALLBACK_RE.search(added)
        or BOUNDARY_PATH_PART_RE.search(path)
        or BOUNDARY_CODE_RE.search(added)
    )


def _marker_value(plan_text: str, marker: str) -> str | None:
    marker_lower = marker.lower()
    for line in plan_text.splitlines():
        lowered = line.lower()
        if marker_lower not in lowered:
            continue
        if ":" not in line:
            return ""
        return line.split(":", 1)[1].strip().rstrip(".").lower()
    return None


def _is_dispositioned_value(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if "todo" in normalized:
        return False
    return normalized not in PLACEHOLDER_VALUES


def plan_has_deployed_config_probing(plan_text: str) -> bool:
    lowered = plan_text.lower()
    if not all(marker in lowered for marker in REQUIRED_PLAN_MARKERS):
        return False
    return all(_is_dispositioned_value(_marker_value(plan_text, marker)) for marker in PROBE_MARKERS)


def scan_diff(added_by_file: Mapping[str, str], plan_texts: Sequence[str]) -> list[Finding]:
    has_plan_probe = any(plan_has_deployed_config_probing(text) for text in plan_texts)
    findings: list[Finding] = []
    for path, added in sorted(added_by_file.items()):
        if not file_needs_deployed_config_probe(path, added):
            continue
        if has_plan_probe:
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


def _changed_lines(diff_hunk: str) -> str:
    return "\n".join(
        line
        for line in diff_hunk.splitlines()
        if (
            (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
            or line.startswith("@@")
        )
    )


def changed_lines(base: str) -> dict[str, str]:
    names = [n for n in _git(["diff", "--name-only", f"{base}...HEAD"]).splitlines() if n]
    out: dict[str, str] = {}
    for name in names:
        if PurePosixPath(name).suffix in CODE_SUFFIXES:
            out[name] = _changed_lines(_git(["diff", "--unified=3", f"{base}...HEAD", "--", name]))
    return out


def changed_plan_texts(base: str) -> list[str]:
    names = [
        n
        for n in _git(
            ["diff", "--name-only", "--diff-filter=AMR", f"{base}...HEAD", "--", "plans/PR-*.md"]
        ).splitlines()
        if n
    ]
    return [Path(name).read_text(encoding="utf-8") for name in names if Path(name).exists()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="diff base ref")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on findings")
    args = parser.parse_args(argv)

    findings = scan_diff(changed_lines(args.base), changed_plan_texts(args.base))
    print("deployed-config probing lint (advisory)")
    print("-" * 60)
    if not findings:
        print("OK: no guard/config boundary change without deployed-config probing.")
        return 0
    for finding in findings:
        print(f"::warning file={finding.path}::{finding.reason}")
        print(f"  {finding.path}: {finding.reason}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
