#!/usr/bin/env python3
"""Sync a PR plan's machine-checked sections with the current git diff."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiffEntry:
    path: str
    added: int
    deleted: int

    @property
    def loc(self) -> int:
        return self.added + self.deleted


def run_git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def merge_base(base_ref: str, *, cwd: Path) -> str:
    return run_git(["merge-base", "HEAD", base_ref], cwd=cwd).strip()


def git_diff_entries(base_ref: str, *, cwd: Path) -> list[DiffEntry]:
    base = merge_base(base_ref, cwd=cwd)
    tracked = _tracked_entries(base, cwd=cwd)
    untracked = _untracked_entries(cwd=cwd)
    by_path = {entry.path: entry for entry in tracked}
    by_path.update({entry.path: entry for entry in untracked})
    return [by_path[path] for path in sorted(by_path)]


def sync_plan_text(text: str, entries: list[DiffEntry]) -> str:
    text = _replace_files_touched(text, entries)
    return _replace_estimated_diff_size(text, entries)


def _tracked_entries(base: str, *, cwd: Path) -> list[DiffEntry]:
    return _parse_numstat_z(run_git(["diff", "--numstat", "-z", base], cwd=cwd))


def _parse_numstat_z(payload: str) -> list[DiffEntry]:
    entries: list[DiffEntry] = []
    index = 0
    while index < len(payload):
        header_end = payload.find("\0", index)
        if header_end == -1:
            raise ValueError("unexpected git numstat output: missing path terminator")
        header = payload[index:header_end]
        index = header_end + 1
        if not header:
            continue

        parts = header.split("\t", 2)
        if len(parts) != 3:
            raise ValueError("unexpected git numstat output: missing count fields")
        added, deleted, path = parts

        # With -z, Git emits renamed/copied paths as:
        # added<TAB>deleted<TAB><NUL>old<NUL>new<NUL>
        # The plan should list the destination path, matching --name-only.
        if path == "":
            old_end = payload.find("\0", index)
            if old_end == -1:
                raise ValueError("unexpected git numstat output: missing rename source")
            index = old_end + 1
            new_end = payload.find("\0", index)
            if new_end == -1:
                raise ValueError("unexpected git numstat output: missing rename destination")
            path = payload[index:new_end]
            index = new_end + 1

        entries.append(
            DiffEntry(path=path, added=_count_value(added), deleted=_count_value(deleted))
        )
    return entries


def _untracked_entries(*, cwd: Path) -> list[DiffEntry]:
    entries: list[DiffEntry] = []
    for line in run_git(["status", "--porcelain", "--untracked-files=all"], cwd=cwd).splitlines():
        if not line.startswith("?? "):
            continue
        path = line[3:].strip()
        if not path:
            continue
        file_path = cwd / path
        if file_path.is_file():
            entries.append(DiffEntry(path=path, added=_line_count(file_path), deleted=0))
    return entries


def _replace_files_touched(text: str, entries: list[DiffEntry]) -> str:
    lines = text.splitlines()
    scope_index = _find_heading(lines, "## Scope (this PR)")
    if scope_index is None:
        raise ValueError("plan is missing required heading: ## Scope (this PR)")

    next_section = _next_heading(lines, scope_index + 1, "## ")
    files_index = _find_heading(lines, "### Files touched", start=scope_index + 1, end=next_section)
    block = ["### Files touched", "", *[f"- `{entry.path}`" for entry in entries], ""]

    if files_index is None:
        insert_at = next_section if next_section is not None else len(lines)
        return "\n".join(lines[:insert_at] + [""] + block + lines[insert_at:]).rstrip() + "\n"

    end = _next_heading(lines, files_index + 1, "### ", "## ")
    if end is None:
        end = len(lines)
    return "\n".join(lines[:files_index] + block + lines[end:]).rstrip() + "\n"


def _replace_estimated_diff_size(text: str, entries: list[DiffEntry]) -> str:
    lines = text.splitlines()
    section_index = _find_heading(lines, "## Estimated diff size")
    if section_index is None:
        raise ValueError("plan is missing required heading: ## Estimated diff size")

    end = _next_heading(lines, section_index + 1, "## ")
    if end is None:
        end = len(lines)
    block = [
        "## Estimated diff size",
        "",
        "| File | LOC |",
        "|---|---:|",
        *[f"| `{entry.path}` | {entry.loc} |" for entry in entries],
        f"| **Total** | **{sum(entry.loc for entry in entries)}** |",
        "",
    ]
    return "\n".join(lines[:section_index] + block + lines[end:]).rstrip() + "\n"


def _find_heading(
    lines: list[str],
    heading: str,
    *,
    start: int = 0,
    end: int | None = None,
) -> int | None:
    limit = len(lines) if end is None else end
    normalized = heading.strip().lower()
    for index in range(start, limit):
        if lines[index].strip().lower() == normalized:
            return index
    return None


def _next_heading(lines: list[str], start: int, *prefixes: str) -> int | None:
    for index in range(start, len(lines)):
        if any(lines[index].startswith(prefix) for prefix in prefixes):
            return index
    return None


def _count_value(value: str) -> int:
    return 0 if value == "-" else int(value)


def _line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync a PR plan's files-touched and diff-size sections."
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("base_ref", nargs="?", default="origin/main")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the plan would be rewritten instead of writing changes",
    )
    args = parser.parse_args(argv)

    repo_root = Path(run_git(["rev-parse", "--show-toplevel"], cwd=Path.cwd()).strip())
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    if not plan_path.exists():
        print(f"sync_pr_plan.py: plan not found: {args.plan}", file=sys.stderr)
        return 2

    try:
        entries = git_diff_entries(args.base_ref, cwd=repo_root)
        original = plan_path.read_text(encoding="utf-8")
        updated = sync_plan_text(original, entries)
    except (RuntimeError, ValueError) as exc:
        print(f"sync_pr_plan.py: {exc}", file=sys.stderr)
        return 2

    if updated == original:
        print(f"plan already in sync: {args.plan}")
        return 0
    if args.check:
        print(f"plan is out of sync: {args.plan}", file=sys.stderr)
        return 1
    plan_path.write_text(updated, encoding="utf-8")
    print(f"updated plan from git diff: {args.plan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
