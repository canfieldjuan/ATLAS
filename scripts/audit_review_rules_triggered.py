#!/usr/bin/env python3
"""Check that a plan's declared reviewer rules cover what the diff triggers.

Mechanizes the path-to-rule trigger table in docs/REVIEWER_RULES.md: it derives
the rule IDs the changed files require and fails when the plan's Review Contract
"Reviewer rules triggered" line omits one. The trigger table is the single
source of truth -- this audit parses it rather than hardcoding a second copy.

Per AGENTS.md section 3g (surface, never silently skip), trigger-table rows that
carry no machine-matchable glob (prose-only descriptions like "invoicing /
billing / payment code") are reported as advisory, not silently dropped.

Live GitHub bot threads are not consulted; this is a static diff-vs-plan check
that runs in the local bundle and CI.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RULES_DOC = "docs/REVIEWER_RULES.md"

TRIGGER_HEADING_RE = re.compile(r"^\s*#+\s*Path-based rule triggers\b", re.IGNORECASE)
NEXT_HEADING_RE = re.compile(r"^\s*#+\s+\S")
TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
BACKTICK_RE = re.compile(r"`([^`]+)`")
RULE_ID_RE = re.compile(r"\bR\d{1,2}\b")
DECLARED_LINE_RE = re.compile(r"reviewer rules triggered\s*:\s*(.*)", re.IGNORECASE)
# Filler words that describe a glob rather than naming a separate trigger
# surface; a comma-segment made only of these is not surfaced as prose.
PROSE_FILLER = {"migrations", "synced", "files", "etc"}
WORD_RE = re.compile(r"[A-Za-z][\w/]*")


def _table_rows(text: str) -> list[str]:
    """Return the data-row cells of the trigger table, if present."""
    lines = text.splitlines()
    in_section = False
    rows: list[str] = []
    for line in lines:
        if TRIGGER_HEADING_RE.match(line):
            in_section = True
            continue
        if in_section and NEXT_HEADING_RE.match(line):
            break
        if in_section:
            match = TABLE_ROW_RE.match(line)
            if match:
                rows.append(match.group(1))
    return rows


def _prose_residual(path_cell: str) -> str:
    """Return the non-glob, non-filler descriptor text of a path cell, if any.

    A trigger row can mix backticked globs with prose that names a separate
    surface (e.g. "login/token/permission code"). After removing the globs, any
    comma-segment that still carries a non-filler word is surfaced so it is not
    silently dropped (AGENTS.md section 3g).
    """
    segments = path_cell.split(",")
    kept: list[str] = []
    for segment in segments:
        without_globs = BACKTICK_RE.sub("", segment).strip()
        words = WORD_RE.findall(without_globs)
        if any(word.lower() not in PROSE_FILLER for word in words):
            kept.append(without_globs)
    return ", ".join(kept)


def parse_trigger_table(text: str):
    """Return (glob_rows, prose_rows).

    glob_rows: list of (glob, frozenset(rule_ids)) for backticked globs.
    prose_rows: list of (description, frozenset(rule_ids)) for triggers with no
    machine-matchable glob -- including the prose portion of a mixed row that
    also has globs (surfaced, not silently skipped).
    """
    glob_rows: list[tuple[str, frozenset[str]]] = []
    prose_rows: list[tuple[str, frozenset[str]]] = []
    for row in _table_rows(text):
        cells = [c.strip() for c in row.split("|")]
        if len(cells) < 2:
            continue
        path_cell, rule_cell = cells[0], cells[1]
        # Skip the header row and its separator.
        if path_cell.lower().startswith("changed path") or set(path_cell) <= set("-: "):
            continue
        rules = frozenset(RULE_ID_RE.findall(rule_cell))
        if not rules:
            continue
        globs = BACKTICK_RE.findall(path_cell)
        for glob in globs:
            glob_rows.append((glob, rules))
        residual = _prose_residual(path_cell) if globs else path_cell
        if residual:
            prose_rows.append((residual, rules))
    return glob_rows, prose_rows


def glob_to_regex(glob: str) -> str:
    pattern = glob
    if pattern.endswith("/"):
        pattern += "**"
    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", ".")
    return escaped


def path_matches(glob: str, path: str) -> bool:
    regex = glob_to_regex(glob)
    if re.fullmatch(regex, path):
        return True
    # Extension-style globs without a slash match the basename anywhere.
    if "/" not in glob and re.fullmatch(regex, path.rsplit("/", 1)[-1]):
        return True
    return False


def required_rules(changed_files: Sequence[str], glob_rows) -> dict[str, list[str]]:
    """Map each required rule id to the changed paths that triggered it."""
    triggered: dict[str, list[str]] = {}
    for path in changed_files:
        for glob, rules in glob_rows:
            if path_matches(glob, path):
                for rule in rules:
                    triggered.setdefault(rule, [])
                    if path not in triggered[rule]:
                        triggered[rule].append(path)
    return triggered


def declared_rules(plan_text: str) -> set[str]:
    """Collect rule ids from the "Reviewer rules triggered" bullet.

    The bullet often wraps across several lines (80-col plans), so gather the
    match line plus its indented continuation lines until the next bullet,
    blank line, heading, or table.
    """
    lines = plan_text.splitlines()
    for idx, line in enumerate(lines):
        match = DECLARED_LINE_RE.search(line)
        if not match:
            continue
        collected = [match.group(1)]
        for cont in lines[idx + 1:]:
            # Only the bullet's own wrapped/indented continuation lines belong to
            # it. A blank line, a non-indented line, or a new bullet/heading/table
            # ends the item -- otherwise unrelated later text (and its rule IDs)
            # could be absorbed and mask an omission.
            if not cont.strip():
                break
            if cont[0] not in " \t":
                break
            if cont.strip()[0] in "-*#|":
                break
            collected.append(cont)
        return set(RULE_ID_RE.findall(" ".join(collected)))
    return set()


def _git_changed_files(base_ref: str) -> list[str]:
    base = subprocess.run(
        ["git", "merge-base", "HEAD", base_ref],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def _discover_plan(base_ref: str) -> list[str]:
    base = subprocess.run(
        ["git", "merge-base", "HEAD", base_ref],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD", "--", "plans/PR-*.md"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def audit(plan_text: str, changed_files, rules_text):
    glob_rows, prose_rows = parse_trigger_table(rules_text)
    triggered = required_rules(changed_files, glob_rows)
    declared = declared_rules(plan_text)
    missing = {rule: paths for rule, paths in triggered.items() if rule not in declared}
    return triggered, missing, prose_rows, declared


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a plan declares the reviewer rules its diff triggers."
    )
    parser.add_argument("base_ref", nargs="?", default="origin/main")
    parser.add_argument("--plan", help="plan doc to check (default: discover changed plans/PR-*.md)")
    parser.add_argument("--reviewer-rules", default=DEFAULT_RULES_DOC)
    args = parser.parse_args(argv)

    try:
        rules_text = (REPO_ROOT / args.reviewer_rules).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"trigger audit: cannot read {args.reviewer_rules}: {exc}", file=sys.stderr)
        return 2

    try:
        changed_files = _git_changed_files(args.base_ref)
        plans = [args.plan] if args.plan else _discover_plan(args.base_ref)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"trigger audit: git error: {exc}", file=sys.stderr)
        return 2

    print("reviewer rules triggered audit")
    print(f"base ref: {args.base_ref}")
    print(f"changed files: {len(changed_files)}")

    if not plans:
        print("no changed plan doc found; skipped (plan-less PR is out of scope)")
        return 0

    drift = False
    for plan in plans:
        try:
            plan_text = (REPO_ROOT / plan if not Path(plan).is_absolute() else Path(plan)).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"trigger audit: cannot read plan {plan}: {exc}", file=sys.stderr)
            return 2
        triggered, missing, prose_rows, declared = audit(plan_text, changed_files, rules_text)
        print("-" * 60)
        print(f"plan: {plan}")
        print(f"declared rules: {', '.join(sorted(declared)) or '(none)'}")
        print(f"triggered by diff: {', '.join(sorted(triggered)) or '(none)'}")
        if missing:
            drift = True
            for rule in sorted(missing):
                paths = ", ".join(missing[rule][:5])
                print(f"  MISSING {rule}: triggered by {paths}; add it to 'Reviewer rules triggered'")
        else:
            print("OK: plan declares every rule the diff triggers.")
        if prose_rows:
            print("advisory -- prose-only trigger rows (not auto-enforced, inspect manually):")
            for desc, rules in prose_rows:
                print(f"  - {desc} -> {', '.join(sorted(rules))}")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
