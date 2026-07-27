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

CODE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".sh", ".yml", ".yaml"}
CONFIG_FALLBACK_PATHS = (".github/workflows/",)
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
    r"[a-zA-Z0-9_$]*(?:guard|validat|is_valid|resolve|admit|reject|claim|scope|auth)"
    r"[a-zA-Z0-9_$]*\s*\("
    r"|(?:async\s+)?def\s+"
    r"[a-zA-Z0-9_]*(?:guard|validat|is_valid|resolve|admit|reject|claim|scope|auth)"
    r"[a-zA-Z0-9_]*\s*\("
    r"|(?:const|let|var)\s+"
    r"[a-zA-Z0-9_$]*(?:Guard|Validat|IsValid|Resolve|Admit|Reject|Claim|Scope|Auth)"
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
    # Atlas reads configuration through Pydantic Settings, not os.environ
    # (CLAUDE.md: "Never read `os.environ` directly"). A detector that only
    # knows os.getenv cannot fire on this repository's own guards -- including
    # `_default_context()`, the deployed-default read behind the #2216
    # claim-before-reject finding this rule exists to prevent.
    r"|settings(?:\.[A-Za-z_][A-Za-z0-9_]*)+\s+or\s+[^)\n]+"
    r"|[A-Za-z_][A-Za-z0-9_]*_settings(?:\.[A-Za-z_][A-Za-z0-9_]*)+\s+or\s+[^)\n]+"
    r"|getattr\(\s*settings[^,\n]*,[^,\n]+,\s*[^)\n]+\)"
)
CONFIG_KEY_RE = re.compile(
    r"os\.(?:getenv|environ\.get)\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
    r"|os\.environ\.get\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
    r"|process\.env\.([A-Za-z_][A-Za-z0-9_]*)"
    r"|process\.env\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]"
    r"|Deno\.env\.get\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\)"
    r"|\$\{([A-Za-z_][A-Za-z0-9_]*):(?:-|=)[^}\n]+\}"
    # Pydantic Settings attribute path; the leaf name is the config key.
    r"|(?:[A-Za-z_][A-Za-z0-9_]*_)?settings(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r"\.([A-Za-z_][A-Za-z0-9_]*)\s+or\s+"
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
# Exact non-answers. This is a DENYLIST, and that is the whole design rule
# here: a denylist of explicit "I have not filled this in" markers fails toward
# ACCEPTING an unlisted phrasing, which is silence -- the cheap error for an
# advisory check. The removed evidence/negative-outcome patterns were the
# opposite shape, an allowlist of approved wordings that failed toward WARNING
# on correct input. Only add exact non-answers here; never a word that tries to
# judge whether a real sentence is good enough.
PLACEHOLDER_VALUES = {
    "", "n/a", "na", "todo", "todo/n/a", "none", "could-not-determine",
    "could not determine", "unknown",
}
UNRESOLVED_VALUES_RE = re.compile(
    r"\b(?:pending|skipped|not verified|not run|tbd|unknown)\b",
    re.IGNORECASE,
)


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
        path.startswith("plans/")
        or p.name in {"AGENTS.md", "new_pr_plan.sh"}
        or (path.startswith("scripts/check_") and p.suffix == ".py")
    )


def file_needs_deployed_config_probe(path: str, added: str) -> bool:
    if _is_test_path(path) or _is_process_path(path):
        return False
    if CONFIG_FALLBACK_RE.search(added):
        return True
    suffix = PurePosixPath(path).suffix
    name = PurePosixPath(path).name
    if suffix not in CODE_SUFFIXES and not name.startswith("Dockerfile") and not path.startswith(CONFIG_FALLBACK_PATHS):
        return False
    return bool(
        BOUNDARY_PATH_PART_RE.search(path)
        or BOUNDARY_CODE_RE.search(added)
    )


# Every settings attribute leaf and literal getattr name inside a fallback.
# Scoped to the fallback span rather than the whole diff so an unrelated plain
# settings read does not manufacture a key the plan must then probe.
_SETTINGS_LEAF_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*_)?settings(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r"\.([A-Za-z_][A-Za-z0-9_]*)"
)
_GETATTR_LITERAL_RE = re.compile(
    r"getattr\([^,\n]+,\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
)


def config_keys(added: str) -> set[str]:
    keys: set[str] = set()
    for match in CONFIG_KEY_RE.finditer(added):
        for group in match.groups():
            if group:
                keys.add(group)
    # A fallback can name several settings operands, and the getattr form names
    # its key as a literal. Extracting only the first operand -- or none, for
    # getattr -- left an incomplete key set, and an incomplete set lets
    # section_covers_config_keys() accept evidence for a key the diff never
    # touched while the real fallback goes unprobed.
    for fallback in CONFIG_FALLBACK_RE.finditer(added):
        span = fallback.group(0)
        literals = _GETATTR_LITERAL_RE.findall(span)
        if literals:
            # getattr names its key as the literal; the settings expression
            # before it is the container, not a config key, and demanding
            # evidence for it would be a false warning.
            keys.update(literals)
            continue
        keys.update(_SETTINGS_LEAF_RE.findall(span))
    return keys


def deployed_config_section(text: str) -> str:
    lines: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("### "):
            if line.strip().lower() == "### deployed-config probing":
                in_section = True
                continue
            if in_section:
                break
        elif line.startswith("## ") and in_section:
            break
        if in_section:
            lines.append(line)
    return "\n".join(lines)


def _marker_value(section: str, marker: str) -> str | None:
    """Return one marker's disposition, including wrapped continuation lines.

    A disposition that wraps onto an indented Markdown continuation line is one
    value, not a truncated one. Reading only the first physical line dropped the
    remainder, so a complete probe could be reported as missing evidence -- a
    false warning on correct input, which is the expensive direction for an
    advisory check.
    """
    marker_lower = marker.lower()
    lines = section.splitlines()
    for index, line in enumerate(lines):
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip().lstrip("-*0123456789. ").strip().lower()
        if label != marker_lower:
            continue
        parts = [value.strip()]
        for continuation in lines[index + 1:]:
            if not continuation.strip():
                break
            # A new list item or heading ends this value; anything else that is
            # indented is a wrap of it.
            stripped = continuation.lstrip()
            if stripped.startswith(("-", "*", "#")) or not continuation[:1].isspace():
                break
            parts.append(stripped)
        return " ".join(parts).strip().rstrip(".").lower()
    return None


def _is_dispositioned_value(value: str | None, *, marker: str) -> bool:
    """Whether a marker carries a disposition, decided on structure alone.

    This deliberately does NOT judge whether the sentence "states evidence."
    That is an open category, and the earlier attempt to recognize it with word
    lists (evidence verbs, negative-outcome phrases) failed in the expensive
    direction four separate times: `is deployed as X`, a value wrapped onto a
    continuation line, and `no write before admission` are all correct
    dispositions that the recognizer rejected, warning on compliant plans.

    Per `docs/GUARD_CLASS_CLOSURE.md`, an open category is gated on bounded
    mechanical evidence with the ambiguous case defaulting to the cheap side.
    Here the bounded facts are: the marker exists, its value is non-empty and
    not a placeholder, and (checked by the caller) it names the config key the
    diff actually changed. Everything else is the reviewer's judgement, which
    is where judging prose belongs -- this check exists to make an OMISSION
    visible, not to grade writing.
    """
    if value is None:
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    if normalized in PLACEHOLDER_VALUES:
        return False
    # Placeholder/unresolved words are a closed, repo-owned denylist: they mark
    # a row the author has explicitly not filled in yet.
    if "todo" in normalized or UNRESOLVED_VALUES_RE.search(normalized):
        return False
    return True


def plan_has_deployed_config_probing(plan_text: str) -> bool:
    section = deployed_config_section(plan_text)
    if not section:
        return False
    return all(_is_dispositioned_value(_marker_value(section, marker), marker=marker) for marker in PROBE_MARKERS)


def section_covers_config_keys(plan_texts: Sequence[str], keys: set[str]) -> bool:
    if not keys:
        return True
    sections = "\n".join(deployed_config_section(text) for text in plan_texts)
    for key in keys:
        # A Pydantic-Settings key is the attribute name
        # (crm_default_business_context) while the plan naturally names the
        # env var (ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT), which carries the
        # attribute as a trailing segment. Allow that, case-insensitively: for
        # an advisory check the cheap error is staying silent, so a generous
        # mention test beats a false warning that burns trust in the check.
        key_re = re.compile(
            rf"{re.escape(key)}(?![A-Za-z0-9_])", re.IGNORECASE
        )
        for marker in PROBE_MARKERS:
            value = _marker_value(sections, marker) or ""
            if marker == "side-effect ordering" and "no write" in value:
                continue
            if not key_re.search(value):
                return False
    return True


def scan_diff(added_by_file: Mapping[str, str], plan_texts: Sequence[str]) -> list[Finding]:
    has_plan_probe = any(plan_has_deployed_config_probing(text) for text in plan_texts)
    findings: list[Finding] = []
    for path, added in sorted(added_by_file.items()):
        if not file_needs_deployed_config_probe(path, added):
            continue
        keys = config_keys(added)
        if has_plan_probe and section_covers_config_keys(plan_texts, keys):
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
