#!/usr/bin/env python3
"""Summarize local PR watcher state for an active builder resume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from codex_wake_bridge import attention_blockers, readiness_blockers


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


def _gh_pr_metadata(
    pr: object,
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if not pr:
        return fallback, "PR number is missing; live GitHub refresh was not run"
    try:
        proc = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(pr),
                "--json",
                "state,headRefOid,mergeStateStatus,reviewDecision,isDraft",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return fallback, f"live GitHub refresh failed: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "gh pr view failed").strip()
        return fallback, f"live GitHub refresh failed: {detail}"
    try:
        value = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return fallback, f"live GitHub refresh returned invalid JSON: {exc}"
    if not isinstance(value, dict):
        return fallback, "live GitHub refresh did not return an object"
    return {**fallback, **value}, None


def _bucket(data: dict[str, Any], *, watcher_state: str, gh_state: str) -> str:
    if attention_blockers(data):
        return "attention"
    if gh_state in {"MERGED", "CLOSED"} or watcher_state == "closed":
        return "stale"
    if watcher_state == "pending" or bool(data.get("check_pending")):
        return "pending"
    if watcher_state == "ready_for_human_merge":
        return "attention" if readiness_blockers(data) else "ready"
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
        stored_pr = data.get("pr") if isinstance(data.get("pr"), dict) else {}
        if skip_github:
            pr = stored_pr
            github_refresh_error = None
        else:
            pr, github_refresh_error = _gh_pr_metadata(stored_pr.get("number"), stored_pr)
        effective_data = {**data, "pr": pr}
        if github_refresh_error:
            effective_data["view_error"] = github_refresh_error
        gh_state = str(pr.get("state") or "unknown")
        watcher_state = str(data.get("state") or "unknown")
        entries.append(
            {
                "path": path,
                "bucket": _bucket(effective_data, watcher_state=watcher_state, gh_state=gh_state),
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
                "readiness_blockers": readiness_blockers(effective_data)
                if watcher_state == "ready_for_human_merge"
                else [],
                "github_refresh_error": github_refresh_error,
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
    if entry.get("github_refresh_error"):
        bits.append(f"github_refresh_error={entry.get('github_refresh_error')}")
    reconciliation = entry.get("reconciliation")
    if reconciliation not in {None, 0}:
        bits.append(f"reconciliation_exit_code={reconciliation}")
    blockers = entry.get("readiness_blockers") or []
    if blockers:
        bits.append("readiness_blockers=" + "; ".join(str(item) for item in blockers))
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
