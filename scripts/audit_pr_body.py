#!/usr/bin/env python3
"""Audit a PR body against the AGENTS.md section 1b contract.

Most human PR bodies must lead with ``Plan: plans/PR-<Slice-Name>.md``, a
``Slice phase: <phase>`` line, and an ``Ownership lane: <lane>`` line, then
carry these ``##`` sections in order:
Intentional, AI reconciliation, Deferred, Parked hardening, Cold diff
reconstruction, Verification, Mechanical verification, Diff size. The
referenced plan doc must exist in the checkout.
A non-empty Markdown-only human diff may instead lead with ``Docs-only: true``
when the caller supplies a base ref for changed-path validation. Dependabot
keeps its explicit generated-body exemption.

Intended for CI on ``pull_request`` events (the workflow writes
``github.event.pull_request.body`` to a file - no GitHub API call), and for
local use before opening a PR:

    python scripts/audit_pr_body.py /tmp/pr_body.md
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, NamedTuple, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pr_change_policy import (
    ChangeKind,
    ChangePolicyError,
    branch_added_plan_docs,
    classify_changes,
    is_dependabot_author,
)
from audit_ai_reconciliation import reconciliation_errors


ROOT = Path(__file__).resolve().parents[1]
PLAN_LINE_RE = re.compile(r"^Plan:\s+(?P<plan>plans/PR-[A-Za-z0-9._-]+\.md)\s*$")
SLICE_PHASE_RE = re.compile(r"^Slice phase:\s*\S.*$")
LANE_LINE_RE = re.compile(r"^Ownership lane:\s*[a-z0-9][a-z0-9._/-]*[a-z0-9]\s*$")
LANE_PREFIX_RE = re.compile(r"^Ownership lane:\s*")
HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")
FENCE_RE = re.compile(r"^ {0,3}(?P<delimiter>`{3,}|~{3,})(?P<tail>.*)$")
REQUIRED_SECTIONS = (
    "Intentional",
    "AI reconciliation",
    "Deferred",
    "Parked hardening",
    "Cold diff reconstruction",
    "Verification",
    "Mechanical verification",
    "Diff size",
)
DOCS_ONLY_RE = re.compile(r"^Docs-only:\s*true\s*$", re.IGNORECASE)
COMMAND_PREFIX_RE = re.compile(r"^\s*[-*]\s+command\s*:\s*\S", re.IGNORECASE)
MECHANICAL_FIELD_RE = re.compile(
    r"(?P<prefix>^\s*[-*]\s+|\s+-\s+)"
    r"(?P<label>command|result|environment|env)\s*:",
    re.IGNORECASE,
)
FIELD_LABEL_RE = re.compile(r"^(?:result|environment|env)\s*:", re.IGNORECASE)
SKIPPED_PREFIX_RE = re.compile(r"^\s*[-*]\s+skipped(?:\s+checks?)?\s*:\s*\S", re.IGNORECASE)
REASON_FIELD_RE = re.compile(r"\breason\s*:\s*\S", re.IGNORECASE)
BULLET_LINE_RE = re.compile(r"^\s*[-*]\s+\S")
VALID_RESULT_RE = re.compile(
    r"^(?:"
    r"passed?|pass|"
    r"failed?|fail|"
    r"skipped?|skip|"
    r"\d+\s+passed(?:[,\s]+(?:\d+\s+failed|failures?|\d+\s+skipped|skips?))*|"
    r"\d+\s+failed(?:[,\s]+(?:\d+\s+passed|\d+\s+skipped))*|"
    r"\d+\s+skipped"
    r")(?:[\s.;].*)?$",
    re.IGNORECASE,
)
PLACEHOLDER_VALUE_RE = re.compile(r"^(?:todo|tbd|n/?a|none|unknown|nowhere|\.*|-+)\s*\.?$", re.IGNORECASE)
VALID_ENVIRONMENTS = {"local", "office pc", "ci"}


class FenceMarker(NamedTuple):
    is_backtick: bool
    length: int


def is_docs_only_body(body: str) -> bool:
    """Return true only for the explicit planless Markdown-only body marker."""

    first_nonempty = next((line.strip() for line in body.splitlines() if line.strip()), "")
    return DOCS_ONLY_RE.fullmatch(first_nonempty) is not None


def first_nonempty_raw_line_index(lines: list[str]) -> tuple[int, str] | None:
    """Return the first non-empty raw line position before fence elision."""

    for index, line in enumerate(lines):
        if line.strip():
            return index, line.strip()
    return None


def _git_read(args: list[str], *, repo_root: Path) -> tuple[int, bytes]:
    """Run a read-only git command. A missing or broken git binary is an
    infrastructure condition surfaced as a nonzero code (fail-closed for
    every caller), never a traceback."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True
        )
    except OSError:
        return 1, b""
    return proc.returncode, proc.stdout


def resolve_git_ref(ref: str, *, repo_root: Path = ROOT) -> bool:
    """True when ``ref`` resolves to a commit in the local repo. The
    trusted-base workflow fetches the PR head before auditing; an
    unresolvable ref is an infrastructure failure, never a silent pass."""
    code, _ = _git_read(
        ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        repo_root=repo_root,
    )
    return code == 0


def plan_exists_at_ref(ref: str, *, repo_root: Path = ROOT) -> Callable[[str], bool]:
    """Plan-doc existence checker against a git ref (the fetched PR head).
    Trusted-base gate runs execute this script from the BASE checkout, but a
    new plan doc arrives WITH the PR -- so existence is asked of the PR head
    ref by inspection (git ls-tree), never by executing PR code."""

    def _exists(plan: str) -> bool:
        # ls-tree lines are "<mode> <type> <sha>\t<path>". The entry must
        # be a REGULAR-FILE blob: a tree at the path (plans/PR-Foo.md/child)
        # or a symlink (mode 120000, possibly dangling) is not a plan doc.
        # Requiring the 100644/100755 blob prefix mirrors the working-tree
        # regular-file checker.
        code, out = _git_read(["ls-tree", ref, "--", plan], repo_root=repo_root)
        line = out.strip()
        return code == 0 and (
            line.startswith(b"100644 blob") or line.startswith(b"100755 blob")
        )

    return _exists


def plan_exists_in_worktree(plan: str, *, repo_root: Path = ROOT) -> bool:
    """True only for a regular plan file in ``repo_root``.

    ``Path.is_file()`` follows symlinks, but the trusted ref checker rejects
    symlink plan entries. Keep the working-tree fallback equally strict.
    """

    rel = Path(plan)
    if rel.is_absolute() or ".." in rel.parts:
        return False
    current = repo_root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            return False
    return current.is_file()


def unfenced_lines(body: str) -> list[str]:
    """Return body lines with fenced examples made non-structural.

    Blank placeholders preserve the fact that a fenced block appeared between
    two real lines, so a header declaration cannot become adjacent by removing
    an intervening example.
    """

    lines: list[str] = []
    fence_marker: FenceMarker | None = None
    for line in body.splitlines():
        match = FENCE_RE.match(line)
        if fence_marker is not None:
            lines.append("")
            if match is not None and closes_fence(match, fence_marker):
                fence_marker = None
        elif match is not None and opens_fence(match):
            lines.append("")
            fence_marker = open_fence_marker(match)
        else:
            lines.append(line)
    return lines


def line_after(lines: list[str], index: int) -> str:
    """Return one line after ``index`` without assuming it exists."""

    return next(islice(lines, index + 1, index + 2), "")


def section_body_lines(lines: list[str], title: str) -> list[str] | None:
    """Return unfenced lines under a top-level PR-body ``##`` section."""

    start: int | None = None
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if match and match.group("title") == title:
            start = index + 1
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start, len(lines)):
        if HEADING_RE.match(lines[index]):
            end = index
            break
    return lines[start:end]


def mechanical_command_fields(raw_line: str) -> tuple[dict[str, str], set[str]]:
    """Parse a mechanical-verification command bullet into delimited fields."""

    matches = list(MECHANICAL_FIELD_RE.finditer(raw_line))
    fields: dict[str, str] = {}
    duplicates: set[str] = set()
    for index, match in enumerate(matches):
        label = match.group("label").lower()
        if label == "env":
            label = "environment"
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(raw_line)
        value = raw_line[match.end():next_start].strip()
        if label in fields:
            duplicates.add(label)
        else:
            fields[label] = value
    return fields, duplicates


def mechanical_verification_errors(lines: list[str]) -> list[str]:
    """Return failures in the PR body's Mechanical verification section."""

    body_lines = section_body_lines(lines, "Mechanical verification")
    if body_lines is None:
        return []
    errors: list[str] = []
    command_count = 0
    for raw_line in body_lines:
        line = raw_line.strip()
        if not line:
            continue
        if COMMAND_PREFIX_RE.match(raw_line):
            fields, duplicates = mechanical_command_fields(raw_line)
            if duplicates:
                errors.append(
                    "mechanical verification command has duplicate fields; "
                    "use each of Command, Result, and Environment once"
                )
            if "command" not in fields:
                errors.append(
                    "mechanical verification command is missing 'Command: <actual command>'"
                )
            else:
                command = fields["command"].strip()
                if (
                    not command
                    or PLACEHOLDER_VALUE_RE.fullmatch(command)
                    or FIELD_LABEL_RE.match(command)
                ):
                    errors.append(
                        "mechanical verification command has invalid Command; "
                        "name the actual command that ran"
                    )
            if "result" not in fields:
                errors.append(
                    "mechanical verification command is missing 'Result: <outcome>'"
                )
            else:
                result = fields["result"].strip()
                if PLACEHOLDER_VALUE_RE.fullmatch(result) or not VALID_RESULT_RE.fullmatch(result):
                    errors.append(
                        "mechanical verification command has invalid Result; "
                        "use pass/fail/skip wording or counts"
                    )
            if "environment" not in fields:
                errors.append(
                    "mechanical verification command is missing "
                    "'Environment: <local|Office PC|CI>'"
                )
            else:
                env = fields["environment"].strip().lower()
                if PLACEHOLDER_VALUE_RE.fullmatch(env) or env not in VALID_ENVIRONMENTS:
                    errors.append(
                        "mechanical verification command has invalid Environment; "
                        "use local, Office PC, or CI"
                    )
            command_count += 1
            continue
        if SKIPPED_PREFIX_RE.match(raw_line):
            if not REASON_FIELD_RE.search(raw_line):
                errors.append(
                    "mechanical verification skipped check is missing "
                    "'Reason: <why skipped>'"
                )
            continue
        if BULLET_LINE_RE.match(raw_line):
            errors.append(
                "mechanical verification bullets must start with 'Command:' "
                "or 'Skipped:'"
            )
    if command_count == 0:
        errors.append(
            "mechanical verification must list at least one executed "
            "'Command:' with Result and Environment"
        )
    return errors


def open_fence_marker(match: re.Match[str]) -> FenceMarker:
    """Return the opening fence character and minimum closing length."""

    delimiter = match.group("delimiter")
    return FenceMarker(is_backtick=delimiter.startswith("`"), length=len(delimiter))


def opens_fence(match: re.Match[str]) -> bool:
    """Return true when ``match`` is a valid Markdown opening fence."""

    delimiter = match.group("delimiter")
    return not (delimiter.startswith("`") and "`" in match.group("tail"))


def closes_fence(match: re.Match[str], marker: FenceMarker) -> bool:
    """Return true when ``match`` closes the active Markdown fence."""

    delimiter = match.group("delimiter")
    same_kind = delimiter.startswith("`") if marker.is_backtick else delimiter.startswith("~")
    return same_kind and len(delimiter) >= marker.length and not match.group("tail").strip()


def audit_pr_body(
    body: str,
    *,
    root: Path = ROOT,
    plan_exists: Callable[[str], bool] | None = None,
) -> list[str]:
    """Return a list of contract failures (empty means the body passes)."""

    if plan_exists is None:
        def plan_exists(plan: str) -> bool:
            return plan_exists_in_worktree(plan, repo_root=root)

    failures: list[str] = []
    raw_lines = body.splitlines()
    raw_header_start = first_nonempty_raw_line_index(raw_lines)
    lines = unfenced_lines(body)
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        return ["PR body is empty"]

    if raw_header_start is None:
        return ["PR body is empty"]
    first_nonempty_index, first_nonempty_line = raw_header_start
    plan_match = PLAN_LINE_RE.match(first_nonempty_line)
    if plan_match is None:
        failures.append(
            "first non-empty line must be 'Plan: plans/PR-<Slice-Name>.md'"
        )
    else:
        if not plan_exists(plan_match.group("plan")):
            failures.append(
                f"plan doc named in the PR body does not exist: {plan_match.group('plan')}"
            )

    phase_line = line_after(raw_lines, first_nonempty_index)
    if not SLICE_PHASE_RE.match(phase_line.strip()):
        failures.append(
            "missing canonical 'Slice phase: <phase>' line immediately after "
            "'Plan: plans/PR-<Slice-Name>.md'"
        )
    else:
        lane_line = line_after(raw_lines, first_nonempty_index + 1)
        if not LANE_LINE_RE.match(lane_line.strip()):
            failures.append(
                "missing canonical 'Ownership lane: <lowercase-lane>' line immediately "
                "after 'Slice phase: <phase>'"
            )

    first_heading_index = next(
        (index for index, line in enumerate(lines) if HEADING_RE.match(line)),
        len(lines),
    )
    lead_lines = lines[:first_heading_index]
    lane_lines = [line.strip() for line in lead_lines if LANE_PREFIX_RE.match(line.strip())]
    if len(lane_lines) != 1:
        failures.append("full PR body must contain exactly one 'Ownership lane:' line")
    elif not LANE_LINE_RE.match(next(iter(lane_lines))):
        failures.append(
            "Ownership lane must use lowercase letters, numbers, dots, dashes, slashes, or underscores"
        )
    why_lines = [
        line.strip()
        for line in lead_lines
        if line.strip()
        and PLAN_LINE_RE.match(line.strip()) is None
        and SLICE_PHASE_RE.match(line.strip()) is None
        and LANE_PREFIX_RE.match(line.strip()) is None
    ]
    if not why_lines:
        failures.append(
            "missing the one-paragraph why between the lead lines and the "
            "first '##' section"
        )

    headings = [
        match.group("title")
        for line in lines
        if (match := HEADING_RE.match(line))
    ]
    missing = [title for title in REQUIRED_SECTIONS if title not in headings]
    for title in missing:
        failures.append(f"missing required section: ## {title}")
    present_in_order = [title for title in headings if title in REQUIRED_SECTIONS]
    expected_order = [title for title in REQUIRED_SECTIONS if title in headings]
    if not missing and present_in_order != expected_order:
        failures.append(
            "required sections are out of order; expected "
            + " -> ".join(f"## {title}" for title in REQUIRED_SECTIONS)
        )
    ai_reconciliation_lines = section_body_lines(lines, "AI reconciliation")
    if ai_reconciliation_lines is None:
        failures.extend(
            f"AI reconciliation: {error}"
            for error in reconciliation_errors("", require=True)
        )
    else:
        canonical_ai_reconciliation = (
            "## AI reconciliation\n" + "\n".join(ai_reconciliation_lines)
        )
        failures.extend(
            f"AI reconciliation: {error}"
            for error in reconciliation_errors(canonical_ai_reconciliation, require=True)
        )
    failures.extend(mechanical_verification_errors(lines))
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr-author",
        default="",
        help="GitHub PR author login; Dependabot PRs are exempt",
    )
    parser.add_argument(
        "--plan-git-ref",
        default="",
        help=(
            "check plan-doc existence against this git ref (the fetched PR "
            "head) instead of the working tree -- for trusted-base gate runs"
        ),
    )
    parser.add_argument(
        "--repo-root",
        default="",
        help=(
            "repo checkout whose working tree or fetched refs should be "
            "inspected; defaults to this script's checkout"
        ),
    )
    parser.add_argument(
        "--base-ref",
        default="",
        help=(
            "base ref used to validate a Docs-only: true body against the "
            "actual changed paths"
        ),
    )
    parser.add_argument(
        "--head-ref",
        default="HEAD",
        help=(
            "head ref used with --base-ref for Docs-only: true validation; "
            "trusted-base gates pass their fetched PR head"
        ),
    )
    parser.add_argument("body_file", help="path to a file holding the PR body")
    args = parser.parse_args(argv)

    body_path = Path(args.body_file)
    if not body_path.is_file():
        print(f"pr body audit: body file not found: {body_path}", file=sys.stderr)
        return 2
    body = body_path.read_text(encoding="utf-8", errors="replace")

    if is_dependabot_author(args.pr_author):
        print("pr body audit: PASS (Dependabot PR body exempt)")
        return 0

    repo_root = ROOT
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
        if not repo_root.is_dir():
            print(f"pr body audit: repo root not found: {repo_root}", file=sys.stderr)
            return 2

    plan_exists = None
    if args.plan_git_ref:
        if not resolve_git_ref(args.plan_git_ref, repo_root=repo_root):
            print(
                f"pr body audit: plan ref not resolvable: {args.plan_git_ref} "
                "(fetch the PR head before auditing)",
                file=sys.stderr,
            )
            return 2  # infrastructure failure -- never a silent pass
        plan_exists = plan_exists_at_ref(args.plan_git_ref, repo_root=repo_root)

    if is_docs_only_body(body):
        if not args.base_ref:
            print(
                "pr body audit: Docs-only body requires --base-ref for changed-path validation",
                file=sys.stderr,
            )
            return 2
        try:
            classification = classify_changes(
                author=args.pr_author,
                base_ref=args.base_ref,
                head_ref=args.head_ref,
                repo_root=repo_root,
            )
        except ChangePolicyError as exc:
            print(f"pr body audit: {exc}", file=sys.stderr)
            return 2
        if classification.kind is not ChangeKind.DOCS_ONLY:
            print("pr body audit: FAIL (AGENTS.md section 1b contract)")
            print("- Docs-only: true is valid only for a non-empty Markdown-only human diff")
            return 1
        try:
            plans = branch_added_plan_docs(
                args.base_ref,
                head_ref=args.head_ref,
                repo_root=repo_root,
            )
        except ChangePolicyError as exc:
            print(f"pr body audit: {exc}", file=sys.stderr)
            return 2
        if plans:
            print("pr body audit: FAIL (AGENTS.md section 1b contract)")
            print("- Docs-only: true is only for a Markdown-only diff with no branch-added plan")
            return 1
        print("pr body audit: PASS (explicit Markdown-only body exemption)")
        return 0

    failures = audit_pr_body(body, root=repo_root, plan_exists=plan_exists)
    if failures:
        print("pr body audit: FAIL (AGENTS.md section 1b contract)")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if args.base_ref:
        try:
            classification = classify_changes(
                author=args.pr_author,
                base_ref=args.base_ref,
                head_ref=args.head_ref,
                repo_root=repo_root,
            )
            plans = branch_added_plan_docs(
                args.base_ref,
                head_ref=args.head_ref,
                repo_root=repo_root,
            )
        except ChangePolicyError as exc:
            print(f"pr body audit: {exc}", file=sys.stderr)
            return 2
        if classification.kind in (ChangeKind.DOCS_ONLY, ChangeKind.PLAN_REQUIRED):
            body_plan = PLAN_LINE_RE.match(
                next(line.strip() for line in body.splitlines() if line.strip())
            ).group("plan")
            if len(plans) != 1:
                if classification.kind is ChangeKind.DOCS_ONLY:
                    failures.append(
                        "a Markdown-only human diff with a full PR body must add "
                        "exactly one plan; otherwise use Docs-only: true"
                    )
                else:
                    failures.append(
                        "human non-Markdown diff must add exactly one plan before its "
                        "full PR body can be accepted"
                    )
            else:
                sole_plan, = plans
                if body_plan != sole_plan:
                    failures.append(
                        "Plan: line must name the sole branch-added plan for a human "
                        f"full PR body: {sole_plan}"
                    )
    if failures:
        print("pr body audit: FAIL (AGENTS.md section 1b contract)")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("pr body audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
