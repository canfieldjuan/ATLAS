#!/usr/bin/env python3
"""Advisory lint for boundary-change enumeration plan coverage."""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RULE = (
    "Boundary-change enumeration: a diff changing a guard, validator, resolver, "
    "or admission boundary must ship a plan-doc enumeration before code: "
    "replaced-path behaviors, guard-relevant fields, and every caller x input "
    "shape, each dispositioned."
)

CODE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".sh"}
BOUNDARY_PATH_PART_RE = re.compile(
    r"(^|[-_./])"
    r"(guard|gate|validat(?:e|or|ion)?|normaliz(?:e|er|ation|ing)?|resolver|resolution|admission|intake|"
    r"route|router|routing|classif(?:y|ier|ication)?|dedupe|scope|tenant|auth(?:entication|orization)?)"
    r"($|[-_./])",
    re.IGNORECASE,
)
BOUNDARY_NAME_TOKEN = (
    r"guard|gate|validat|normaliz|resolve|admit|reject|allow|allowed|route|classif|dedupe|scope|auth"
)
BOUNDARY_CODE_RE = re.compile(
    r"^(?:[+-]\s*|@@[^@\n]*@@\s*.*?)"
    r"(?:"
    r"(?:export\s+)?(?:async\s+)?function\s+"
    r"[a-zA-Z0-9_$]*(?:" + BOUNDARY_NAME_TOKEN + r")"
    r"[a-zA-Z0-9_$]*\s*\("
    r"|(?:async\s+)?def\s+"
    r"[a-zA-Z0-9_]*(?:" + BOUNDARY_NAME_TOKEN + r")"
    r"[a-zA-Z0-9_]*\s*\("
    r"|(?:const|let|var)\s+"
    r"[a-zA-Z0-9_$]*(?:Guard|Gate|Validat|Normaliz|Resolve|Admit|Reject|Allow|Allowed|Route|Classif|Dedupe|Scope|Auth)"
    r"[a-zA-Z0-9_$]*\s*="
    r"|class\s+[a-zA-Z0-9_$]*(?:Guard|Validator|Resolver|Admission|Authenticator|"
    r"Authentication|Authorization|AuthGate|AuthProvider|AuthValidator|AuthResolver)"
    r"[a-zA-Z0-9_$]*"
    r"|(?:async\s+)?[a-zA-Z0-9_$]*(?:" + BOUNDARY_NAME_TOKEN + r")"
    r"[a-zA-Z0-9_$]*\s*\([^)]*\)\s*\{"
    r"|(?:function\s+)?[a-zA-Z0-9_]*(?:" + BOUNDARY_NAME_TOKEN + r")"
    r"[a-zA-Z0-9_]*\s*\(\)\s*\{"
    r")",
    re.MULTILINE | re.IGNORECASE,
)
REQUIRED_PLAN_MARKERS = (
    "boundary-change enumeration",
    "replaced-path behaviors",
    "guard-relevant fields",
    "caller x input shape",
)
DISPOSITION_MARKERS = REQUIRED_PLAN_MARKERS[1:]
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
        or path == "scripts/check_boundary_change_enumeration.py"
    )


def file_is_boundary_shaped(path: str, added: str) -> bool:
    if PurePosixPath(path).suffix not in CODE_SUFFIXES or _is_test_path(path) or _is_process_path(path):
        return False
    return bool(BOUNDARY_PATH_PART_RE.search(path) or BOUNDARY_CODE_RE.search(added))


def boundary_enumeration_section(text: str) -> str:
    lines: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("### "):
            if line.strip().lower() == "### boundary-change enumeration":
                in_section = True
                continue
            if in_section:
                break
        elif line.startswith("## ") and in_section:
            break
        if in_section:
            lines.append(line)
    return "\n".join(lines)


def _marker_values(section: str, marker: str) -> list[str]:
    marker_lower = marker.lower()
    values: list[str] = []
    for line in section.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip().lstrip("-*0123456789. ").strip().lower()
        if label != marker_lower and not label.startswith(f"{marker_lower} "):
            continue
        values.append(value.strip().rstrip(".").lower())
    return values


def _has_section_level_not_applicable(section: str) -> bool:
    for line in section.splitlines():
        stripped = line.strip().lower().rstrip(".")
        if stripped.startswith(("-", "*")):
            continue
        if stripped.startswith(("n/a -", "na -", "not applicable -")):
            return True
    return False


def _is_dispositioned_value(value: str | None, *, section_not_applicable: bool = False) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if "todo" in normalized:
        return False
    if section_not_applicable and normalized in {"n/a", "na", "not applicable"}:
        return True
    if normalized.startswith(("n/a -", "na -", "not applicable -")):
        return True
    return normalized not in PLACEHOLDER_VALUES


def plan_has_boundary_enumeration(plan_text: str) -> bool:
    if "boundary-change enumeration" not in plan_text.lower():
        return False
    section = boundary_enumeration_section(plan_text)
    section_not_applicable = _has_section_level_not_applicable(section)
    for marker in DISPOSITION_MARKERS:
        values = _marker_values(section, marker)
        if not values:
            return False
        if not all(
            _is_dispositioned_value(value, section_not_applicable=section_not_applicable)
            for value in values
        ):
            return False
    return True


def _path_mentions(path: str, section: str) -> bool:
    section_lower = section.lower()
    p = PurePosixPath(path)
    candidates = {
        path.lower(),
        p.name.lower(),
        p.stem.lower(),
        p.stem.lower().replace("_", "-"),
        p.stem.lower().replace("-", "_"),
        p.stem.lower().replace("_", " "),
        p.stem.lower().replace("-", " "),
    }
    return any(candidate and candidate in section_lower for candidate in candidates)


def plan_covers_boundary_path(plan_text: str, path: str) -> bool:
    if not plan_has_boundary_enumeration(plan_text):
        return False
    section = boundary_enumeration_section(plan_text)
    if _has_section_level_not_applicable(section):
        return True
    return _path_mentions(path, section)


def scan_diff(added_by_file: Mapping[str, str], plan_texts: Sequence[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path, added in sorted(added_by_file.items()):
        if not file_is_boundary_shaped(path, added):
            continue
        if any(plan_covers_boundary_path(text, path) for text in plan_texts):
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


def changed_added_lines(base: str) -> dict[str, str]:
    names = [n for n in _git(["diff", "--name-only", f"{base}...HEAD"]).splitlines() if n]
    out: dict[str, str] = {}
    for name in names:
        if PurePosixPath(name).suffix in CODE_SUFFIXES:
            out[name] = _changed_lines(_git(["diff", "--unified=0", f"{base}...HEAD", "--", name]))
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

    findings = scan_diff(changed_added_lines(args.base), changed_plan_texts(args.base))
    print("boundary-change enumeration lint (advisory)")
    print("-" * 60)
    if not findings:
        print("OK: no boundary-shaped change without plan enumeration.")
        return 0
    for finding in findings:
        print(f"::warning file={finding.path}::{finding.reason}")
        print(f"  {finding.path}: {finding.reason}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
