#!/usr/bin/env python3
"""Detect changed-file drift across concurrent PR sessions."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Sequence

LANE_RE = re.compile(r"^\s*Ownership lane:\s*`?([^`\n]+?)`?\s*$", re.IGNORECASE | re.MULTILINE)
LANE_VALUE_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*[a-z0-9]$")
SLICE_PHASE_RE = re.compile(r"^\s*Slice phase:\s*`?([^`\n]+?)`?\s*$", re.IGNORECASE | re.MULTILINE)
SCOPE_SECTION_RE = re.compile(
    r"^##\s+Scope(?:\s+\(this PR\))?\s*$\n?(.*?)(?=^##\s+|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
VALID_SLICE_PHASES = frozenset(
    {
        "vertical slice",
        "functional validation",
        "robust testing",
        "production hardening",
        "product polish",
        "workflow/process",
    }
)


@dataclass(frozen=True)
class OpenPullRequest:
    number: int
    title: str
    head_ref: str
    head_oid: str
    url: str
    files: frozenset[str]
    ownership_lanes: frozenset[str]
    slice_phases: frozenset[str]
    slice_phase_errors: tuple[str, ...]


@dataclass(frozen=True)
class DriftReport:
    branch_files: frozenset[str]
    base_files: frozenset[str]
    base_overlap: frozenset[str]
    open_pr_overlaps: tuple[tuple[OpenPullRequest, frozenset[str]], ...]
    branch_ownership_lanes: frozenset[str]
    branch_slice_phases: frozenset[str]
    open_pr_lane_overlaps: tuple[tuple[OpenPullRequest, frozenset[str]], ...]
    github_status: str
    github_warnings: tuple[str, ...]
    path_errors: tuple[str, ...]
    ownership_errors: tuple[str, ...]
    phase_errors: tuple[str, ...]
    current_pr_phase_errors: tuple[str, ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect changed-file overlap with base updates and open PRs.",
    )
    parser.add_argument(
        "base_ref",
        nargs="?",
        default="origin/main",
        help="base ref to compare against (default: origin/main)",
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="skip GitHub open-PR overlap checks",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(args.base_ref, skip_github=args.skip_github)
    except AuditError as exc:
        print(f"DRIFT audit error: {exc}", file=sys.stderr)
        return 2

    render_report(args.base_ref, report)
    if (
        report.path_errors
        or report.ownership_errors
        or report.phase_errors
        or report.current_pr_phase_errors
        or report.base_overlap
        or report.open_pr_lane_overlaps
    ):
        return 1
    return 0


def build_report(base_ref: str, *, skip_github: bool = False) -> DriftReport:
    ensure_git_ref(base_ref)
    base = git_stdout(["merge-base", "HEAD", base_ref])
    branch_files = changed_files([base, "HEAD"], triple_dot=True)
    base_files = changed_files([base, base_ref], triple_dot=False)
    plan_docs = added_plan_docs(base)
    branch_ownership_lanes, ownership_errors = branch_ownership(plan_docs)
    branch_slice_phases, phase_errors = collect_branch_slice_phases(plan_docs)

    path_errors: list[str] = []
    path_errors.extend(validate_paths(branch_files, "branch diff"))
    path_errors.extend(validate_paths(base_files, f"{base_ref} diff"))

    base_overlap = frozenset(branch_files & base_files)
    open_pr_overlaps: list[tuple[OpenPullRequest, frozenset[str]]] = []
    open_pr_lane_overlaps: list[tuple[OpenPullRequest, frozenset[str]]] = []
    current_pr_phase_errors: list[str] = []
    github_status = "skipped (--skip-github)"
    github_warnings: list[str] = []

    if not skip_github and os.environ.get("ATLAS_SKIP_PR_SESSION_DRIFT_GITHUB") != "1":
        github_status, prs, loaded_github_warnings = load_open_pull_requests(base_ref)
        github_warnings.extend(loaded_github_warnings)
        current_branch = current_head_ref()
        current_oid = current_head_oid()
        for pr in prs:
            is_current_pr = pr.head_ref == current_branch or (pr.head_oid and pr.head_oid == current_oid)
            if is_current_pr:
                current_pr_phase_errors.extend(pr.slice_phase_errors)
                if branch_slice_phases and not pr.slice_phases:
                    current_pr_phase_errors.append("current PR body: missing Slice phase")
                elif (
                    branch_slice_phases
                    and pr.slice_phases
                    and branch_slice_phases.isdisjoint(pr.slice_phases)
                ):
                    current_pr_phase_errors.append(
                        "current PR body Slice phase "
                        f"{format_values(pr.slice_phases)} does not match branch plan phase(s) "
                        f"{format_values(branch_slice_phases)}"
                    )
                continue
            overlap = frozenset(branch_files & pr.files)
            if overlap:
                open_pr_overlaps.append((pr, overlap))
            lane_overlap = frozenset(branch_ownership_lanes & pr.ownership_lanes)
            if lane_overlap:
                open_pr_lane_overlaps.append((pr, lane_overlap))

    return DriftReport(
        branch_files=frozenset(branch_files),
        base_files=frozenset(base_files),
        base_overlap=base_overlap,
        open_pr_overlaps=tuple(open_pr_overlaps),
        branch_ownership_lanes=branch_ownership_lanes,
        branch_slice_phases=branch_slice_phases,
        open_pr_lane_overlaps=tuple(open_pr_lane_overlaps),
        github_status=github_status,
        github_warnings=tuple(github_warnings),
        path_errors=tuple(path_errors),
        ownership_errors=tuple(ownership_errors),
        phase_errors=tuple(phase_errors),
        current_pr_phase_errors=tuple(current_pr_phase_errors),
    )


def ensure_git_ref(ref: str) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AuditError(f"base ref not found: {ref}")


def changed_files(revs: Sequence[str], *, triple_dot: bool) -> set[str]:
    separator = "..." if triple_dot else ".."
    diff_ref = f"{revs[0]}{separator}{revs[1]}"
    output = git_stdout(["diff", "--name-only", diff_ref])
    return {line for line in output.splitlines() if line.strip()}


def added_plan_docs(base: str) -> set[str]:
    output = git_stdout([
        "diff",
        "--name-only",
        "--diff-filter=A",
        f"{base}...HEAD",
        "--",
        "plans/PR-*.md",
    ])
    return {line for line in output.splitlines() if line.strip()}


def branch_ownership(plan_docs: set[str]) -> tuple[frozenset[str], tuple[str, ...]]:
    lanes: set[str] = set()
    errors: list[str] = []
    if not plan_docs:
        return frozenset(), tuple()

    for plan_doc in plan_docs:
        text = git_stdout(["show", f"HEAD:{plan_doc}"])
        doc_lanes, doc_errors = extract_ownership_lanes(text, source=plan_doc)
        lanes.update(doc_lanes)
        errors.extend(doc_errors)
        if not doc_lanes and not doc_errors:
            errors.append(f"{plan_doc}: missing Ownership lane")

    if not lanes and plan_docs:
        errors.append("changed PR plan docs must declare an Ownership lane")
    return frozenset(lanes), tuple(errors)


def collect_branch_slice_phases(plan_docs: set[str]) -> tuple[frozenset[str], tuple[str, ...]]:
    phases: set[str] = set()
    errors: list[str] = []
    if not plan_docs:
        return frozenset(), tuple()

    for plan_doc in plan_docs:
        text = git_stdout(["show", f"HEAD:{plan_doc}"])
        doc_phases, doc_errors = extract_plan_slice_phases(text, source=plan_doc)
        phases.update(doc_phases)
        errors.extend(doc_errors)
        if not doc_phases and not doc_errors:
            errors.append(f"{plan_doc}: missing Slice phase")

    if not phases and plan_docs:
        errors.append("changed PR plan docs must declare a Slice phase")
    return frozenset(phases), tuple(errors)


def extract_ownership_lanes(text: str, *, source: str) -> tuple[frozenset[str], tuple[str, ...]]:
    lanes: set[str] = set()
    errors: list[str] = []
    for raw_lane in LANE_RE.findall(text):
        lane = raw_lane.strip().lower()
        if not LANE_VALUE_RE.match(lane):
            errors.append(
                f"{source}: invalid Ownership lane {raw_lane!r}; "
                "use lowercase letters, numbers, dots, dashes, slashes, or underscores"
            )
            continue
        lanes.add(lane)
    return frozenset(lanes), tuple(errors)


def extract_plan_slice_phases(text: str, *, source: str) -> tuple[frozenset[str], tuple[str, ...]]:
    return extract_slice_phases(scope_section(text), source=source)


def extract_slice_phases(text: str, *, source: str) -> tuple[frozenset[str], tuple[str, ...]]:
    phases: set[str] = set()
    errors: list[str] = []
    for raw_phase in SLICE_PHASE_RE.findall(text):
        phase = normalize_slice_phase(raw_phase)
        if phase not in VALID_SLICE_PHASES:
            errors.append(
                f"{source}: invalid Slice phase {raw_phase!r}; "
                f"use one of: {', '.join(sorted(VALID_SLICE_PHASES))}"
            )
            continue
        phases.add(phase)
    return frozenset(phases), tuple(errors)


def normalize_slice_phase(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().rstrip(".").lower())


def scope_section(text: str) -> str:
    match = SCOPE_SECTION_RE.search(text)
    return match.group(1) if match else ""


def load_open_pull_requests(base_ref: str) -> tuple[str, tuple[OpenPullRequest, ...], tuple[str, ...]]:
    if shutil.which("gh") is None:
        return "skipped (gh not found)", (), ()

    base_branch = github_base_branch_name(base_ref)
    result = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--base",
            base_branch,
            "--limit",
            "100",
            "--json",
            "number,title,headRefName,headRefOid,url",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = one_line(result.stderr) or one_line(result.stdout) or "gh pr list failed"
        return f"skipped ({message})", (), ()

    try:
        rows = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        return f"skipped (gh pr list returned invalid JSON: {exc})", (), ()

    prs: list[OpenPullRequest] = []
    warnings: list[str] = []
    for row in rows:
        pr_number = integer_value(row.get("number"))
        if pr_number is None:
            warnings.append(f"open PR row missing numeric number: {row!r}")
            continue
        pr, metadata_warnings = load_open_pull_request_files(pr_number, row)
        prs.append(pr)
        warnings.extend(metadata_warnings)
        for error in validate_paths(pr.files, f"PR #{pr.number}"):
            warnings.append(error)

    return f"checked {len(prs)} open PR(s)", tuple(prs), tuple(warnings)


def load_open_pull_request_files(number: int, row: dict[str, Any]) -> tuple[OpenPullRequest, tuple[str, ...]]:
    fallback = open_pr_from_row(
        number,
        row,
        files=frozenset(),
        ownership_lanes=frozenset(),
        slice_phases=frozenset(),
        slice_phase_errors=(),
    )
    result = subprocess.run(
        ["gh", "pr", "view", str(number), "--json", "files,body"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = one_line(result.stderr) or one_line(result.stdout) or "gh pr view failed"
        return fallback, (f"skipped PR #{number}: could not read files/body: {message}",)

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return fallback, (f"skipped PR #{number}: files/body JSON is invalid: {exc}",)

    files = {
        str(item.get("path", "")).strip()
        for item in payload.get("files", [])
        if isinstance(item, dict)
    }
    body = str(payload.get("body", "") or "")
    lanes, lane_errors = extract_ownership_lanes(body, source=f"PR #{number} body")
    phases, phase_errors = extract_slice_phases(body, source=f"PR #{number} body")
    warnings = tuple(lane_errors + phase_errors)
    return open_pr_from_row(
        number,
        row,
        files=frozenset(path for path in files if path),
        ownership_lanes=lanes if not lane_errors else frozenset(),
        slice_phases=phases if not phase_errors else frozenset(),
        slice_phase_errors=phase_errors,
    ), warnings


def open_pr_from_row(
    number: int,
    row: dict[str, Any],
    *,
    files: frozenset[str],
    ownership_lanes: frozenset[str],
    slice_phases: frozenset[str],
    slice_phase_errors: tuple[str, ...],
) -> OpenPullRequest:
    return OpenPullRequest(
        number=number,
        title=str(row.get("title", "")).strip(),
        head_ref=str(row.get("headRefName", "")).strip(),
        head_oid=str(row.get("headRefOid", "")).strip(),
        url=str(row.get("url", "")).strip(),
        files=files,
        ownership_lanes=ownership_lanes,
        slice_phases=slice_phases,
        slice_phase_errors=slice_phase_errors,
    )


def github_base_branch_name(base_ref: str) -> str:
    for prefix in ("refs/remotes/origin/", "origin/", "refs/heads/"):
        if base_ref.startswith(prefix):
            return base_ref[len(prefix):]
    return base_ref


def validate_paths(paths: Iterable[str], source: str) -> list[str]:
    errors: list[str] = []
    for path in sorted(paths):
        posix = PurePosixPath(path)
        if not path or posix.is_absolute() or ".." in posix.parts:
            errors.append(f"{source}: unsafe path {path!r}")
    return errors


def render_report(base_ref: str, report: DriftReport) -> None:
    print("cross-session PR drift audit")
    print(f"base ref: {base_ref}")
    print(f"branch changed files: {len(report.branch_files)}")
    print(f"base changed files since branch point: {len(report.base_files)}")
    print(f"branch ownership lanes: {', '.join(sorted(report.branch_ownership_lanes)) or 'none'}")
    print(f"branch slice phases: {', '.join(sorted(report.branch_slice_phases)) or 'none'}")
    print(f"GitHub open PR check: {report.github_status}")
    if report.github_warnings:
        print()
        print("WARN: GitHub metadata skipped or malformed")
        for warning in report.github_warnings:
            print(f"- {warning}")

    if report.path_errors:
        print()
        print("DRIFT: unsafe or malformed file paths detected")
        for error in report.path_errors:
            print(f"- {error}")

    if report.ownership_errors:
        print()
        print("DRIFT: ownership lane contract failed")
        for error in report.ownership_errors:
            print(f"- {error}")

    if report.phase_errors:
        print()
        print("DRIFT: slice phase contract failed")
        for error in report.phase_errors:
            print(f"- {error}")

    if report.current_pr_phase_errors:
        print()
        print("DRIFT: current PR body slice phase contract failed")
        for error in report.current_pr_phase_errors:
            print(f"- {error}")

    if report.base_overlap:
        print()
        print("DRIFT: base branch changed files that this branch also changes")
        for path in sorted(report.base_overlap):
            print(f"- {path}")

    if report.open_pr_overlaps:
        print()
        print("WARN: open PRs change files that this branch also changes")
        for pr, files in report.open_pr_overlaps:
            label = f"#{pr.number}"
            if pr.title:
                label = f"{label} {pr.title}"
            if pr.url:
                label = f"{label} ({pr.url})"
            print(f"- {label}")
            for path in sorted(files):
                print(f"  - {path}")

    if report.open_pr_lane_overlaps:
        print()
        print("DRIFT: open PRs claim the same ownership lane")
        for pr, lanes in report.open_pr_lane_overlaps:
            label = f"#{pr.number}"
            if pr.title:
                label = f"{label} {pr.title}"
            if pr.url:
                label = f"{label} ({pr.url})"
            print(f"- {label}")
            for lane in sorted(lanes):
                print(f"  - {lane}")

    if (
        not report.path_errors
        and not report.ownership_errors
        and not report.phase_errors
        and not report.current_pr_phase_errors
        and not report.base_overlap
        and not report.open_pr_lane_overlaps
    ):
        print("OK: no blocking drift detected")


def format_values(values: Iterable[str]) -> str:
    return ", ".join(sorted(values)) or "none"


def current_head_ref() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def current_head_oid() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def git_stdout(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AuditError(one_line(result.stderr) or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def integer_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def one_line(value: str) -> str:
    return " ".join(value.split())


class AuditError(Exception):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
