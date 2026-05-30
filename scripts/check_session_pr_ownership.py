#!/usr/bin/env python3
"""Check that a PR is owned by the current local builder session."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_FILE = ROOT / "SESSION_STATE.local.md"
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
PR_RE = re.compile(r"#(?P<number>\d+)\b")


@dataclass(frozen=True)
class SessionOwnership:
    owned_pr: int | None
    owned_branch: str
    owned_head: str
    may_touch: frozenset[int]
    must_not_touch: frozenset[int]


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _section_map(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = SECTION_RE.match(line)
        if match:
            current = match.group(1).strip().lower()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def _field(section: Sequence[str], name: str) -> str:
    prefix = f"{name}:"
    for line in section:
        if line.startswith(prefix):
            return _clean(line[len(prefix) :])
    return ""


def _parse_pr_field(value: str) -> int | None:
    if value.lower() == "none":
        return None
    match = PR_RE.search(value)
    if not match:
        return None
    return int(match.group("number"))


def _parse_pr_list(lines: Sequence[str]) -> frozenset[int]:
    numbers: set[int] = set()
    for line in lines:
        match = PR_RE.search(line)
        if match:
            numbers.add(int(match.group("number")))
    return frozenset(numbers)


def parse_session_state(text: str) -> SessionOwnership:
    sections = _section_map(text)
    owned = sections.get("owned active pr", [])
    return SessionOwnership(
        owned_pr=_parse_pr_field(_field(owned, "PR")),
        owned_branch=_field(owned, "Branch"),
        owned_head=_field(owned, "Expected head SHA"),
        may_touch=_parse_pr_list(sections.get("prs this session may touch", [])),
        must_not_touch=_parse_pr_list(sections.get("prs this session must not touch", [])),
    )


def ownership_errors(
    ownership: SessionOwnership,
    *,
    pr: int,
    branch: str,
    head_sha: str = "",
) -> list[str]:
    errors: list[str] = []
    if pr in ownership.must_not_touch:
        errors.append(f"PR #{pr} is listed under PRs This Session Must Not Touch")
        return errors
    if ownership.owned_pr != pr and pr not in ownership.may_touch:
        errors.append(
            f"PR #{pr} is not listed under Owned Active PR or PRs This Session May Touch"
        )
    if ownership.owned_pr == pr:
        if ownership.owned_branch and ownership.owned_branch != branch:
            errors.append(
                f"branch mismatch for PR #{pr}: state has {ownership.owned_branch}, "
                f"target has {branch}"
            )
        expected = ownership.owned_head
        if expected and expected.lower() != "none":
            if not head_sha:
                errors.append(
                    f"head SHA required for PR #{pr} because session state records "
                    f"expected head {expected}"
                )
            elif expected != head_sha:
                errors.append(
                    f"head SHA mismatch for PR #{pr}: state has {expected}, "
                    f"target has {head_sha}"
                )
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate that a target PR is owned by SESSION_STATE.local.md."
    )
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--head-sha", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state_path = args.state_file
    if not state_path.exists():
        print(f"session state file not found: {state_path}", file=sys.stderr)
        return 2
    try:
        ownership = parse_session_state(state_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"could not read session state file: {exc}", file=sys.stderr)
        return 2
    errors = ownership_errors(
        ownership,
        pr=args.pr,
        branch=_clean(args.branch),
        head_sha=_clean(args.head_sha),
    )
    if errors:
        print("session PR ownership check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"session PR ownership check passed for PR #{args.pr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
