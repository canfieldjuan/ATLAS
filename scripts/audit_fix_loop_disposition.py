#!/usr/bin/env python3
"""Require a bounded fix-loop disposition before reconciling review findings."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_ai_reconciliation import (
    BULLET_RE,
    DISPOSITION_TOKEN_RE,
    FINDING_SEPARATOR_RE,
    NO_FINDINGS_RE,
    extract_section as extract_ai_reconciliation_section,
)
from audit_pr_body import unfenced_lines

DISPOSITIONS = (
    "fixed-in",
    "waived-duplicate",
    "waived-out-of-scope",
    "waived-speculative",
    "waived-nit",
    "not-applicable",
)
FIX_STRATEGIES = (
    "upstream-root",
    "symptom-only-deferred",
)
BLOCKING_PREDICATES = {
    "contract",
    "ci",
    "security",
    "privacy",
    "money",
    "data",
    "material-correctness",
    "back-compat",
    "performance",
    "claimed-mechanism",
    "not-blocking",
}

HEADING_RE_TEMPLATE = r"(?im)^##\s+{heading}\s*$"
NEXT_SECTION_RE = re.compile(r"(?m)^##\s+")
PLAN_RE = re.compile(r"(?im)^\s*Plan:\s*(?P<path>plans/PR-[A-Za-z0-9._-]+\.md)\s*$")
FIELD_RE = re.compile(r"(?im)^\s*-\s*(?P<name>[A-Za-z][A-Za-z -]*):\s*(?P<value>\S.*)$")
MAX_FILES_RE = re.compile(r"(?im)^\s*Max files:\s*(?P<value>\S.*?)\s*$")
SCOPE_RE = re.compile(r"(?im)^##\s+Scope\b")
PATH_TOKEN_RE = re.compile(r"`([^`\n]+)`|([^,\s]+)")


class ReconciliationItem(NamedTuple):
    root: str
    disposition: str


class PreflightRecord(NamedTuple):
    root: str
    fields: dict[str, str]


def extract_section(body: str, heading: str) -> str | None:
    pattern = re.compile(HEADING_RE_TEMPLATE.format(heading=re.escape(heading)), re.IGNORECASE | re.MULTILINE)
    match = pattern.search(body)
    if match is None:
        return None
    next_match = NEXT_SECTION_RE.search(body, match.end())
    end = next_match.start() if next_match else len(body)
    return body[match.end() : end].strip()


def actionable_reconciliation(section: str | None) -> bool:
    if section is None:
        return False
    if NO_FINDINGS_RE.search(section) and not DISPOSITION_TOKEN_RE.search(section):
        return False
    return bool(DISPOSITION_TOKEN_RE.search(section))


def parse_fields(section: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in FIELD_RE.finditer(section):
        name = " ".join(match.group("name").lower().split())
        fields[name] = match.group("value").strip()
    return fields


def parse_preflight_records(section: str) -> list[PreflightRecord]:
    records: list[PreflightRecord] = []
    current: dict[str, str] = {}
    current_root = ""
    for match in FIELD_RE.finditer(section):
        name = " ".join(match.group("name").lower().split())
        value = match.group("value").strip()
        if name == "root decision":
            if current or current_root:
                records.append(PreflightRecord(root=current_root, fields=current))
            current_root = normalize_root(value)
            current = {name: value}
        elif current or current_root:
            current[name] = value
    if current or current_root:
        records.append(PreflightRecord(root=current_root, fields=current))
    return records


def structural_preflight_section(section: str) -> str:
    return "\n".join(unfenced_lines(section))


def _placeholder(value: str) -> bool:
    return value.strip().lower() in {"", "none", "n/a", "na", "tbd", "todo", "unknown", "?", "-", "--", "..."}


def _source_trace_is_valid(value: str) -> bool:
    parts = [part.strip() for part in value.split("->")]
    if len(parts) < 2:
        return False
    return all(not _placeholder(part) and re.search(r"[A-Za-z0-9]", part) for part in parts)


def trace_contract_errors(fields: dict[str, str]) -> tuple[list[str], str, set[str]]:
    errors: list[str] = []
    for field in ("source trace", "upstream files", "fix strategy"):
        if field not in fields:
            errors.append(f"fix-loop disposition preflight: missing '- {field.title()}: ...'")
    source_trace = fields.get("source trace", "")
    if source_trace and not _source_trace_is_valid(source_trace):
        errors.append(
            "fix-loop disposition preflight: source trace must name the chain "
            "from symptom -> upstream source with non-placeholder endpoints"
        )
    upstream_files = parse_allowed_files(fields.get("upstream files", ""))
    if "upstream files" in fields and not upstream_files:
        errors.append("fix-loop disposition preflight: upstream files must contain repo-relative paths")
    strategy = fields.get("fix strategy", "").lower()
    if strategy and strategy not in FIX_STRATEGIES:
        errors.append(
            "fix-loop disposition preflight: invalid fix strategy "
            f"{strategy!r}; use one of {', '.join(FIX_STRATEGIES)}"
        )
    if strategy == "symptom-only-deferred":
        for field in ("symptom-only reason", "follow-up"):
            if _placeholder(fields.get(field, "")):
                errors.append(
                    "fix-loop disposition preflight: symptom-only-deferred requires "
                    f"'- {field.title()}: ...'"
                )
    return errors, strategy, upstream_files


def disposition_errors(fields: dict[str, str], *, changed_file_set: set[str] | None) -> list[str]:
    errors: list[str] = []
    changed_file_count = len(changed_file_set) if changed_file_set is not None else None
    required = (
        "root decision",
        "blocking predicate",
        "disposition",
        "allowed files",
        "max files",
        "parked hardening",
    )
    for field in required:
        if field not in fields:
            errors.append(f"fix-loop disposition preflight: missing '- {field.title()}: ...'")
    trace_errors, strategy, upstream_files = trace_contract_errors(fields)
    errors.extend(trace_errors)

    disposition = fields.get("disposition", "").lower()
    if disposition and disposition not in DISPOSITIONS:
        errors.append(
            "fix-loop disposition preflight: invalid disposition "
            f"{disposition!r}; use one of {', '.join(DISPOSITIONS)}"
        )

    predicate = fields.get("blocking predicate", "").lower()
    if predicate and predicate not in BLOCKING_PREDICATES:
        errors.append(
            "fix-loop disposition preflight: invalid blocking predicate "
            f"{predicate!r}; use one of {', '.join(sorted(BLOCKING_PREDICATES))}"
        )
    if disposition == "fixed-in" and predicate == "not-blocking":
        errors.append("fix-loop disposition preflight: fixed-in findings need a blocking predicate")
    if disposition.startswith("waived-") and predicate and predicate != "not-blocking":
        errors.append("fix-loop disposition preflight: waived findings must use blocking predicate 'not-blocking'")

    allowed_files = fields.get("allowed files", "")
    if allowed_files.lower() in {"none", "n/a", "na", "tbd"}:
        errors.append("fix-loop disposition preflight: allowed files must name the bounded edit set")
    allowed_set = parse_allowed_files(allowed_files)
    if allowed_files and not allowed_set:
        errors.append("fix-loop disposition preflight: allowed files must contain repo-relative paths")
    raw_max = fields.get("max files", "")
    if raw_max:
        if not raw_max.isdigit():
            errors.append(f"fix-loop disposition preflight: Max files must be an integer, got {raw_max!r}")
        elif int(raw_max) < 1:
            errors.append("fix-loop disposition preflight: Max files must be at least 1")
        elif changed_file_count is not None and changed_file_count > int(raw_max):
            errors.append(
                "fix-loop disposition preflight: changed file count "
                f"{changed_file_count} exceeds Max files {raw_max}"
            )

    root = fields.get("root decision", "")
    if root.lower() in {"none", "n/a", "na", "tbd"}:
        errors.append("fix-loop disposition preflight: root decision must name the reviewed defect class")

    if disposition == "fixed-in":
        if strategy == "upstream-root" and changed_file_set is not None and upstream_files:
            if not changed_file_set.intersection(upstream_files):
                errors.append(
                    "fix-loop disposition preflight: fixed-in upstream-root must change at least one "
                    "declared upstream file"
                )

    parked = fields.get("parked hardening", "")
    if parked.lower() in {"", "tbd"}:
        errors.append("fix-loop disposition preflight: parked hardening must be 'none' or name the parking target")

    return errors


def reconciliation_dispositions(section: str | None) -> set[str]:
    if section is None:
        return set()
    return {match.group("name").lower() for match in DISPOSITION_TOKEN_RE.finditer(section)}


def normalize_root(text: str) -> str:
    root = FINDING_SEPARATOR_RE.sub(" ", text.strip().strip("`"))
    return " ".join(root.lower().split())


def reconciliation_items(section: str | None) -> list[ReconciliationItem]:
    if section is None:
        return []
    items: list[ReconciliationItem] = []
    for raw_line in section.splitlines():
        bullet = BULLET_RE.match(raw_line)
        if bullet is None:
            continue
        item = bullet.group("body").strip()
        disposition = DISPOSITION_TOKEN_RE.search(item)
        if disposition is None:
            continue
        finding = item[: disposition.start()].strip()
        root = normalize_root(finding.strip(" :-"))
        if root:
            items.append(ReconciliationItem(root=root, disposition=disposition.group("name").lower()))
    return items


def parse_allowed_files(value: str) -> set[str]:
    paths: set[str] = set()
    for match in PATH_TOKEN_RE.finditer(value.replace(",", " ")):
        raw = (match.group(1) or match.group(2) or "").strip().strip("`")
        if not raw or raw.startswith("/") or ".." in Path(raw).parts:
            continue
        paths.add(raw)
    return paths


def plan_path_from_body(body: str, repo_root: Path) -> tuple[Path | None, str | None]:
    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
    match = PLAN_RE.fullmatch(first_line)
    if match is None:
        return None, "PR body must start with Plan: plans/PR-<Slice>.md"
    raw = match.group("path")
    path = repo_root / raw
    try:
        resolved_repo = repo_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        return None, f"plan path is not a readable regular file: {raw} ({exc})"
    if not resolved_path.is_relative_to(resolved_repo):
        return None, f"plan path escapes repo root: {raw}"
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != path.anchor):
        return None, f"plan path must not be a symlink: {raw}"
    if not path.is_file():
        return None, f"plan path is not a regular file: {raw}"
    return path, None


def scope_section(plan_text: str) -> str:
    match = SCOPE_RE.search(plan_text)
    if match is None:
        return ""
    next_match = NEXT_SECTION_RE.search(plan_text, match.end())
    end = next_match.start() if next_match else len(plan_text)
    return plan_text[match.end() : end]


def plan_max_files(plan_text: str) -> int | None:
    match = MAX_FILES_RE.search(scope_section(plan_text))
    if match is None:
        return None
    raw = match.group("value").strip()
    if not raw.isdigit():
        raise ValueError(f"plan Scope has malformed Max files value: {raw!r}")
    return int(raw)


def changed_files(base_ref: str, *, repo_root: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "git diff failed").strip())
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def audit_body(body: str, *, repo_root: Path, base_ref: str | None = None, changed_file_set: set[str] | None = None) -> list[str]:
    ai_section = extract_ai_reconciliation_section(body)
    if not actionable_reconciliation(ai_section):
        return []

    errors: list[str] = []
    preflight_raw = extract_section(body, "Fix-loop disposition preflight")
    if preflight_raw is None:
        return [
            "fix-loop disposition preflight missing: PR bodies that reconcile "
            "bot findings must classify fix-vs-waive, allowed files, and Max files before push"
        ]
    preflight = structural_preflight_section(preflight_raw)

    if changed_file_set is None and base_ref:
        try:
            changed_file_set = changed_files(base_ref, repo_root=repo_root)
        except RuntimeError as exc:
            return [f"fix-loop disposition preflight: failed to read changed files: {exc}"]

    records = parse_preflight_records(preflight)
    if not records:
        errors.append("fix-loop disposition preflight: no root decision records found")
        fields = parse_fields(preflight)
        records = [PreflightRecord(root=normalize_root(fields.get("root decision", "")), fields=fields)]

    ai_dispositions = reconciliation_dispositions(ai_section)
    items = reconciliation_items(ai_section)
    records_by_root = {record.root: record for record in records}
    allowed_union: set[str] = set()
    for item in items:
        record = records_by_root.get(item.root)
        if record is None:
            errors.append(
                "fix-loop disposition preflight: missing preflight record for "
                f"AI reconciliation root {item.root!r}"
            )
            continue
        errors.extend(
            disposition_errors(
                record.fields,
                changed_file_set=changed_file_set,
            )
        )
        allowed_union.update(parse_allowed_files(record.fields.get("allowed files", "")))
        disposition = record.fields.get("disposition", "").lower()
        if disposition and disposition != item.disposition:
            errors.append(
                "fix-loop disposition preflight: disposition "
                f"{disposition!r} does not match AI reconciliation disposition "
                f"{item.disposition!r} for root {item.root!r}"
            )

    if changed_file_set is not None and allowed_union:
        extra = sorted(changed_file_set - allowed_union)
        if extra:
            errors.append(
                "fix-loop disposition preflight: changed files outside allowed set: "
                + ", ".join(extra)
            )

    body_max_values = {record.fields.get("max files", "") for record in records}
    numeric_body_max_values = {int(value) for value in body_max_values if value.isdigit()}
    if len(numeric_body_max_values) > 1:
        errors.append(
            "fix-loop disposition preflight: all records must declare the same Max files value"
        )
    plan_path, plan_error = plan_path_from_body(body, repo_root)
    if plan_path is None:
        errors.append(f"fix-loop disposition preflight: {plan_error}")
    else:
        try:
            plan_budget = plan_max_files(plan_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            errors.append(f"fix-loop disposition preflight: {exc}")
        else:
            if plan_budget is None:
                errors.append("fix-loop disposition preflight: plan Scope must declare Max files: N")
            else:
                for body_max in sorted(numeric_body_max_values):
                    if plan_budget != body_max:
                        errors.append(
                            "fix-loop disposition preflight: body Max files "
                            f"{body_max} does not match plan Scope Max files {plan_budget}"
                        )

    return errors


def read_body(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-pr-body-file", required=True, help="PR body file to inspect")
    parser.add_argument("--repo-root", default=".", help="repository root containing the plan")
    parser.add_argument("--base-ref", help="base ref for enforcing Allowed files against the branch diff")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    errors = audit_body(read_body(args.current_pr_body_file), repo_root=repo_root, base_ref=args.base_ref)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("fix-loop disposition preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
