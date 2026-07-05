#!/usr/bin/env python3
"""Select the next Codex autonomy issue or record an operator-owned defer."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence


DEFAULT_REPO = "canfieldjuan/ATLAS"
DEFAULT_ALERT_DIR = Path.home() / ".local" / "state" / "atlas-pr-watchers" / "operator-defers"
DEFAULT_ISSUE_LIMIT = 1000
QUEUE_LABEL = "codex"
DEFERRED_LABEL = "deferred"
TRUSTED_COMMENT_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
LANE_RE = re.compile(r"(?im)^\s*Autonomy lane\s*:\s*(?P<value>.+?)\s*$")
PRIORITY_RE = re.compile(r"(?im)^\s*Autonomy priority\s*:\s*(?P<value>-?\d+)\s*$")
DEFER_RE = re.compile(r"(?im)^\s*Autonomy deferred\s*:\s*(true|yes|1)\s*$")
JSON_FIELDS = "number,title,url,body,labels,updatedAt,state,comments"


class QueueError(Exception):
    """Raised for operator-actionable queue failures."""


def _run_gh_json(args: Sequence[str]) -> Any:
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise QueueError(proc.stderr.strip() or proc.stdout.strip() or f"gh exited {proc.returncode}")
    try:
        return json.loads(proc.stdout or "null")
    except json.JSONDecodeError as exc:
        raise QueueError(f"gh returned invalid JSON: {exc}") from exc


def _run_gh(args: Sequence[str]) -> str:
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise QueueError(proc.stderr.strip() or proc.stdout.strip() or f"gh exited {proc.returncode}")
    return proc.stdout.strip()


def _comment_bodies(issue: dict[str, Any]) -> list[str]:
    comments = issue.get("comments")
    if not isinstance(comments, list):
        return []
    bodies: list[str] = []
    for comment in comments:
        if not isinstance(comment, dict) or not isinstance(comment.get("body"), str):
            continue
        association = str(comment.get("authorAssociation") or "").upper()
        if association in TRUSTED_COMMENT_ASSOCIATIONS:
            bodies.append(comment["body"])
    return bodies


def _text_blocks(issue: dict[str, Any]) -> list[str]:
    blocks = [str(issue.get("body") or "")] if QUEUE_LABEL in _labels(issue) else []
    blocks.extend(_comment_bodies(issue))
    return blocks


def _labels(issue: dict[str, Any]) -> set[str]:
    labels = issue.get("labels")
    if not isinstance(labels, list):
        return set()
    names: set[str] = set()
    for label in labels:
        if isinstance(label, dict) and label.get("name") is not None:
            names.add(str(label["name"]).strip().lower())
        elif isinstance(label, str):
            names.add(label.strip().lower())
    return names


def _unique_markers(pattern: re.Pattern[str], blocks: Sequence[str]) -> set[str]:
    values: set[str] = set()
    for block in blocks:
        for match in pattern.finditer(block):
            value = match.group("value").strip()
            if value:
                values.add(value)
    return values


def _issue_metadata(issue: dict[str, Any]) -> dict[str, Any]:
    blocks = _text_blocks(issue)
    lanes = _unique_markers(LANE_RE, blocks)
    priorities = _unique_markers(PRIORITY_RE, blocks)
    if len(lanes) > 1:
        raise QueueError(f"issue #{issue.get('number')} has conflicting Autonomy lane markers: {sorted(lanes)}")
    if len(priorities) > 1:
        raise QueueError(f"issue #{issue.get('number')} has conflicting Autonomy priority markers: {sorted(priorities)}")
    priority = int(next(iter(priorities))) if priorities else 1000
    deferred = DEFERRED_LABEL in _labels(issue) or any(DEFER_RE.search(block) for block in blocks)
    return {
        "lane": next(iter(lanes), ""),
        "priority": priority,
        "deferred": deferred,
    }


def _list_issues(repo: str, *, limit: int = DEFAULT_ISSUE_LIMIT) -> list[dict[str, Any]]:
    value = _run_gh_json([
        "gh",
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--label",
        QUEUE_LABEL,
        "--limit",
        str(limit),
        "--json",
        JSON_FIELDS,
    ])
    if not isinstance(value, list):
        raise QueueError("gh issue list returned a non-list payload")
    return [item for item in value if isinstance(item, dict)]


def select_next_issue(issues: Sequence[dict[str, Any]], *, lane: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for issue in issues:
        meta = _issue_metadata(issue)
        if meta["lane"] != lane or meta["deferred"]:
            continue
        candidates.append(
            {
                "number": issue.get("number"),
                "title": issue.get("title", ""),
                "url": issue.get("url", ""),
                "updatedAt": issue.get("updatedAt", ""),
                "priority": meta["priority"],
            }
        )
    if not candidates:
        raise QueueError(f"no open queued issues found for lane {lane!r}")
    candidates.sort(key=lambda item: (item["priority"], str(item["updatedAt"]), int(item["number"] or 0)))
    top = candidates[0]
    return {
        "ok": True,
        "action": "next",
        "lane": lane,
        "issue_number": top["number"],
        "issue_url": top["url"],
        "title": top["title"],
        "priority": top["priority"],
        "updatedAt": top["updatedAt"],
        "eligible_count": len(candidates),
    }


def _now_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:80] or "defer"


def _quote_freeform(text: str) -> str:
    lines = str(text or "").strip().splitlines() or [""]
    return "\n".join(f"> {line}" for line in lines)


def _defer_comment(*, lane: str, reason: str, source: str) -> str:
    source_block = f"\nSource:\n{_quote_freeform(source)}\n" if source else ""
    return (
        "## Operator-owned defer\n\n"
        f"Autonomy lane: {lane}\n"
        "Autonomy deferred: true\n"
        f"Reason:\n{_quote_freeform(reason)}\n"
        f"{source_block}\n"
        "This records a decision that requires operator input. The active agent should "
        "continue only with other safe queued work."
    )


def _write_alert(*, alert_dir: Path, repo: str, issue: int, lane: str, reason: str, source: str) -> Path:
    alert_dir.mkdir(parents=True, exist_ok=True)
    path = alert_dir / f"{_now_stamp()}-issue-{issue}-{_safe_slug(lane)}.md"
    issue_url = f"https://github.com/{repo}/issues/{issue}"
    text = (
        "# Atlas Operator-Owned Defer\n\n"
        f"- Repository: {repo}\n"
        f"- Issue: #{issue} {issue_url}\n"
        f"- Autonomy lane: {lane}\n"
        f"- Source: {source or 'not provided'}\n"
        f"- Reason: {reason.strip()}\n\n"
        "Action needed: decide the operator-owned fork, then update the issue or assign "
        "a follow-up slice. Until then, agents should continue only with other safe queued work.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def defer_issue(*, repo: str, issue: int, lane: str, reason: str, source: str, alert_dir: Path) -> dict[str, Any]:
    comment = _defer_comment(lane=lane, reason=reason, source=source)
    alert_path = _write_alert(alert_dir=alert_dir, repo=repo, issue=issue, lane=lane, reason=reason, source=source)
    _run_gh(["gh", "issue", "edit", str(issue), "--repo", repo, "--add-label", DEFERRED_LABEL])
    _run_gh(["gh", "issue", "comment", str(issue), "--repo", repo, "--body", comment])
    return {
        "ok": True,
        "action": "defer",
        "repo": repo,
        "lane": lane,
        "issue_number": issue,
        "issue_url": f"https://github.com/{repo}/issues/{issue}",
        "reason": reason.strip(),
        "alert_path": str(alert_path),
    }


def _print_payload(payload: dict[str, Any], *, markdown: bool) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
    if markdown and payload.get("ok"):
        print()
        if payload.get("action") == "next":
            print(f"Next queued issue: #{payload['issue_number']} {payload['title']}")
            print(f"URL: {payload['issue_url']}")
            print(f"Lane: {payload['lane']} | priority: {payload['priority']}")
        elif payload.get("action") == "defer":
            print(f"Deferred operator decision for #{payload['issue_number']}")
            print(f"Alert artifact: {payload['alert_path']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    next_parser = sub.add_parser("next", help="select the next queued issue for a lane")
    next_parser.add_argument("--repo", default=DEFAULT_REPO)
    next_parser.add_argument("--lane", required=True)
    next_parser.add_argument("--limit", type=int, default=DEFAULT_ISSUE_LIMIT)
    next_parser.add_argument("--markdown", action="store_true")

    defer_parser = sub.add_parser("defer", help="record an operator-owned defer")
    defer_parser.add_argument("--repo", default=DEFAULT_REPO)
    defer_parser.add_argument("--issue", type=int, required=True)
    defer_parser.add_argument("--lane", required=True)
    defer_parser.add_argument("--reason", required=True)
    defer_parser.add_argument("--source", default="")
    defer_parser.add_argument("--alert-dir", type=Path, default=DEFAULT_ALERT_DIR)
    defer_parser.add_argument("--markdown", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "next":
            payload = select_next_issue(_list_issues(args.repo, limit=args.limit), lane=args.lane)
            payload["repo"] = args.repo
            _print_payload(payload, markdown=args.markdown)
            return 0
        payload = defer_issue(
            repo=args.repo,
            issue=args.issue,
            lane=args.lane,
            reason=args.reason,
            source=args.source,
            alert_dir=args.alert_dir.expanduser(),
        )
        _print_payload(payload, markdown=args.markdown)
        return 0
    except QueueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
