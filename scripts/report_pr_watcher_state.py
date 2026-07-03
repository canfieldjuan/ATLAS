#!/usr/bin/env python3
"""Summarize local PR watcher state for an active builder resume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence


def _json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"could not read watcher JSON: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"unreadable watcher JSON: {exc}"
    if not isinstance(value, dict):
        return None, "watcher JSON must be an object"
    return value, None


def _gh_pr_state(pr: object, fallback: str) -> str:
    if not pr:
        return fallback
    try:
        proc = subprocess.run(
            ["gh", "pr", "view", str(pr), "--json", "state", "--jq", ".state"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return fallback
    if proc.returncode != 0:
        return fallback
    return proc.stdout.strip() or fallback


def _truthy(value: Any) -> bool:
    if value is None or value is False or value == "" or value == 0:
        return False
    return bool(value)


def _has_failure_detail(data: dict[str, Any]) -> bool:
    return any(
        [
            _truthy(data.get("head_mismatch")),
            _truthy(data.get("worktree_dirty")),
            _truthy(data.get("merge_error")),
            _truthy(data.get("check_failures")),
            data.get("reconciliation_exit_code") not in {None, 0},
        ]
    )


def _bucket(data: dict[str, Any], *, watcher_state: str, gh_state: str) -> str:
    if _has_failure_detail(data):
        return "attention"
    if gh_state in {"MERGED", "CLOSED"} or watcher_state == "closed":
        return "stale"
    if watcher_state == "pending" or _truthy(data.get("check_pending")):
        return "pending"
    if watcher_state == "ready_for_human_merge":
        return "ready"
    if watcher_state in {"attention", "review_changed"}:
        return "attention"
    return "other"


def _entries(state_dir: Path, *, skip_github: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not state_dir.exists():
        return entries
    for path in sorted(state_dir.glob("*.json")):
        if path.name.endswith(".wake.json"):
            continue
        data, error = _json(path)
        if data is None:
            entries.append(
                {
                    "path": path,
                    "bucket": "attention",
                    "summary": error or "unreadable watcher JSON",
                    "watcher_state": "unreadable",
                    "gh_state": "unknown",
                }
            )
            continue
        pr = data.get("pr") if isinstance(data.get("pr"), dict) else {}
        stored_gh_state = str(pr.get("state") or "unknown")
        gh_state = stored_gh_state if skip_github else _gh_pr_state(pr.get("number"), stored_gh_state)
        watcher_state = str(data.get("state") or "unknown")
        entries.append(
            {
                "path": path,
                "bucket": _bucket(data, watcher_state=watcher_state, gh_state=gh_state),
                "watcher_state": watcher_state,
                "gh_state": gh_state,
                "pr_number": pr.get("number"),
                "title": pr.get("title", ""),
                "head": pr.get("headRefOid", ""),
                "observed": data.get("observed_at", ""),
                "failures": data.get("check_failures") or [],
                "pending": data.get("check_pending") or [],
                "reconciliation": data.get("reconciliation_exit_code"),
                "head_mismatch": data.get("head_mismatch"),
                "worktree_dirty": data.get("worktree_dirty"),
                "merge_error": data.get("merge_error"),
            }
        )
    return entries


def _line(entry: dict[str, Any]) -> str:
    if entry.get("summary"):
        return f"{entry.get('path')} | {entry.get('summary')}"
    bits = [
        f"#{entry.get('pr_number')}",
        str(entry.get("title") or ""),
        f"watcher={entry.get('watcher_state')}",
        f"github={entry.get('gh_state')}",
        f"observed={entry.get('observed')}",
    ]
    failures = entry.get("failures") or []
    pending = entry.get("pending") or []
    if failures:
        bits.append("failures=" + ", ".join(str(item) for item in failures))
    if pending:
        bits.append("pending=" + ", ".join(str(item) for item in pending))
    if entry.get("head_mismatch"):
        bits.append("head_mismatch=true")
    if entry.get("worktree_dirty"):
        bits.append("worktree_dirty=true")
    if entry.get("merge_error"):
        bits.append(f"merge_error={entry.get('merge_error')}")
    reconciliation = entry.get("reconciliation")
    if reconciliation not in {None, 0}:
        bits.append(f"reconciliation_exit_code={reconciliation}")
    return " | ".join(bits)


def render(entries: Sequence[dict[str, Any]]) -> None:
    print("PR watcher ready-state handoff")
    print("-" * 60)
    if not entries:
        print("No watcher state files found.")
        return
    sections = (
        ("ready", "Ready for active-agent merge decision"),
        ("attention", "Needs active-agent attention"),
        ("pending", "Still pending"),
        ("stale", "Stale/closed watcher state to clean up"),
        ("other", "Other watcher states"),
    )
    for bucket, title in sections:
        grouped = [entry for entry in entries if entry.get("bucket") == bucket]
        if not grouped:
            continue
        print()
        print(title)
        for entry in grouped:
            print(f"- {_line(entry)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path.home() / ".local" / "state" / "atlas-pr-watchers",
    )
    parser.add_argument("--skip-github", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    render(_entries(args.state_dir.expanduser(), skip_github=args.skip_github))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
