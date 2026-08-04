#!/usr/bin/env python3
"""Validate Atlas builder PR branch names against the PR body contract."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence

from _pr_change_policy import is_dependabot_author


PLAN_LINE_RE = re.compile(r"^Plan:\s+plans/PR-(?P<slice>[A-Za-z0-9._-]+)\.md\s*$")
DOCS_ONLY_RE = re.compile(r"^Docs-only:\s*true\s*$", re.IGNORECASE)
PR_BRANCH_RE = re.compile(r"^claude/pr-[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")


def plan_slug(slice_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", slice_name).strip("-").lower()
    return slug


def expected_branch_for_body(body: str) -> str | None:
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if DOCS_ONLY_RE.fullmatch(line):
            return None
        match = PLAN_LINE_RE.fullmatch(line)
        if match:
            slug = plan_slug(match.group("slice"))
            if slug:
                return f"claude/pr-{slug}"
            return ""
        return ""
    return ""


def branch_name_errors(
    *,
    branch: str,
    body: str,
    pr_author: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if is_dependabot_author(pr_author):
        return errors
    clean_branch = branch.strip()
    if not clean_branch:
        return ["current checkout is detached; switch to a PR branch first"]
    if not PR_BRANCH_RE.fullmatch(clean_branch):
        errors.append(
            f"branch must match claude/pr-<slice-name>, got {clean_branch!r}"
        )
    expected = expected_branch_for_body(body)
    if expected == "":
        errors.append("PR body must begin with Plan: plans/PR-<Slice>.md or Docs-only: true")
    elif expected is not None and clean_branch != expected:
        errors.append(
            f"branch {clean_branch!r} does not match PR plan branch {expected!r}"
        )
    return errors


def push_refspec_errors(
    *,
    branch: str,
    body: str,
    push_args: Sequence[str],
    pr_author: str | None = None,
) -> list[str]:
    errors = branch_name_errors(branch=branch, body=body, pr_author=pr_author)
    if errors or is_dependabot_author(pr_author):
        return errors

    clean_branch = branch.strip()
    allowed_sources = {"HEAD", clean_branch, f"refs/heads/{clean_branch}"}
    allowed_destinations = {clean_branch, f"refs/heads/{clean_branch}"}
    for arg in push_args:
        refspec = arg.lstrip("+")
        if not _looks_branch_refspec(refspec):
            continue
        if ":" in refspec:
            source, destination = refspec.split(":", 1)
            if source not in allowed_sources:
                errors.append(
                    f"push refspec source {source!r} must be HEAD or {clean_branch!r}"
                )
            if destination not in allowed_destinations:
                errors.append(
                    f"push refspec destination {destination!r} must be {clean_branch!r}"
                )
        elif refspec not in allowed_sources:
            errors.append(f"push refspec {refspec!r} must be HEAD or {clean_branch!r}")
    return errors


def _looks_branch_refspec(value: str) -> bool:
    if not value or value.startswith("-"):
        return False
    return (
        value == "HEAD"
        or value.startswith("HEAD:")
        or value.startswith("claude/")
        or value.startswith("refs/heads/")
        or ":claude/" in value
        or ":refs/heads/" in value
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the current PR branch name against the PR body."
    )
    parser.add_argument("--branch", required=True)
    parser.add_argument("--pr-author", default="")
    parser.add_argument("--push-arg", action="append", default=[])
    parser.add_argument("body_file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        body = args.body_file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"branch name check could not read PR body file: {exc}", file=sys.stderr)
        return 2
    errors = push_refspec_errors(
        branch=args.branch,
        body=body,
        push_args=args.push_arg,
        pr_author=args.pr_author,
    )
    if errors:
        print("PR branch name check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print(f"PR branch name check passed for {args.branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
