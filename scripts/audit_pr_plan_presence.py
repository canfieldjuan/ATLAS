#!/usr/bin/env python3
"""Require a branch-added plan for human PRs that change non-Markdown paths."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit_repo_root import audit_repo_root
from _pr_change_policy import (
    ChangeKind,
    ChangePolicyError,
    branch_added_plan_docs,
    classify_changes,
)


REPO_ROOT = audit_repo_root(__file__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_ref", nargs="?", default="origin/main")
    parser.add_argument(
        "--pr-author",
        default="",
        help="GitHub PR author login; Dependabot keeps its explicit exemption.",
    )
    args = parser.parse_args(argv)

    try:
        classification = classify_changes(
            author=args.pr_author,
            base_ref=args.base_ref,
            repo_root=REPO_ROOT,
        )
    except ChangePolicyError as exc:
        print(f"plan admission audit: {exc}", file=sys.stderr)
        return 2

    print("plan admission audit")
    print(f"base ref: {args.base_ref}")
    print(f"classification: {classification.kind.value}")
    print(f"changed paths: {len(classification.paths)}")

    if classification.kind is ChangeKind.DEPENDABOT:
        print("PASS: Dependabot PR is explicitly exempt from the plan requirement.")
        return 0
    if classification.kind is ChangeKind.NO_CHANGES:
        print("PASS: no changed paths; plan admission is not applicable.")
        return 0
    if classification.kind is ChangeKind.DOCS_ONLY:
        print("PASS: Markdown-only diff is explicitly exempt from the plan requirement.")
        return 0

    try:
        plans = branch_added_plan_docs(args.base_ref, repo_root=REPO_ROOT)
    except ChangePolicyError as exc:
        print(f"plan admission audit: {exc}", file=sys.stderr)
        return 2

    if len(plans) == 1:
        print(f"PASS: required branch-added plan: {plans[0]}")
        return 0

    print("FAIL: a human non-Markdown diff must add exactly one plans/PR-*.md file.")
    if plans:
        print("branch-added plans:")
        for plan in plans:
            print(f"- {plan}")
    else:
        print("branch-added plans: none")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
