#!/usr/bin/env python3
"""Live, CI-side enforcement of the AI-finding reconciliation rule (#1328 Phase 5).

The local audit (scripts/audit_ai_reconciliation.py) can only check that the PR
body's reconciliation record is internally well-formed; it cannot see the live
GitHub bot threads. This check closes that half: it fetches the real
Codex/Copilot review threads and fails when the recorded reconciliation
*omits a genuinely open finding* -- i.e. the body claims all-clear (or carries
no reconciliation record at all) while unresolved bot threads still exist.

It deliberately does NOT require every thread to be GitHub-"resolved": that is
self-resolvable by the PR author and would be a gameable rigor gate. It catches
the specific contradiction between a "resolved" body and open reality, which is
exactly the deferred spec from plans/archive/PR-Reviewer-Reconciliation-Audit.md.

Bot findings stay advisory (operating-model gap (b)): nothing here auto-resolves
or auto-applies. It only enforces that a human accounted for what the bots
raised.

Exit codes: 0 = clean (no open bot threads, or the body honestly acknowledges
open findings); 1 = contradiction (open bot threads + an all-clear/absent
record); 2 = usage error or a GitHub API failure (retryable, never a silent
pass).

The body classifier reuses scripts/audit_ai_reconciliation.py by import so the
local and live checks cannot disagree on what a "resolved" record looks like.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_DEFAULT_BOTS = ("copilot", "codex")

_THREADS_QUERY = """
query($owner:String!,$name:String!,$pr:Int!){
  repository(owner:$owner,name:$name){
    pullRequest(number:$pr){
      reviewThreads(first:100){
        nodes{
          isResolved
          isOutdated
          path
          line
          comments(first:1){ nodes{ author{ login } bodyText } }
        }
      }
    }
  }
}
"""


def _load_phase2():
    """Import the local reconciliation auditor so the body classifier matches."""
    path = Path(__file__).resolve().parent / "audit_ai_reconciliation.py"
    spec = importlib.util.spec_from_file_location("audit_ai_reconciliation", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_body(body: str) -> str:
    """Return 'absent' | 'acknowledges_open' | 'claims_clear' | 'unmarked'.

    Uses the Phase-2 section extractor + markers, so "what counts as a resolved
    record" stays defined in exactly one place.
    """
    p2 = _load_phase2()
    section = p2.extract_section(body)
    if section is None:
        return "absent"
    if p2.UNRESOLVED_RE.search(section):
        return "acknowledges_open"
    if p2.RESOLVED_RE.search(section):
        return "claims_clear"
    return "unmarked"


def open_bot_threads(nodes: Sequence[dict], bot_logins: Sequence[str]) -> list[dict]:
    """Return unresolved, non-outdated review threads authored by a known bot.

    `nodes` is the GraphQL reviewThreads node list; pure so it is unit-testable
    without touching GitHub.
    """
    wanted = tuple(b.lower() for b in bot_logins)
    found: list[dict] = []
    for node in nodes or []:
        if node.get("isResolved") or node.get("isOutdated"):
            continue
        comments = ((node.get("comments") or {}).get("nodes")) or []
        author = ""
        snippet = ""
        if comments:
            author = (((comments[0] or {}).get("author") or {}).get("login")) or ""
            snippet = ((comments[0] or {}).get("bodyText") or "").strip().replace("\n", " ")
        if not any(w in author.lower() for w in wanted):
            continue
        found.append(
            {
                "path": node.get("path") or "?",
                "line": node.get("line"),
                "author": author or "?",
                "snippet": (snippet[:120] + "...") if len(snippet) > 120 else snippet,
            }
        )
    return found


def evaluate(nodes: Sequence[dict], body: str, bot_logins: Sequence[str]) -> tuple[int, list[str]]:
    """Core decision (pure). Returns (exit_code, messages)."""
    open_threads = open_bot_threads(nodes, bot_logins)
    if not open_threads:
        return 0, ["OK: no open automated-review (bot) threads on this PR."]

    body_class = classify_body(body)
    if body_class in ("acknowledges_open", "unmarked"):
        # The body does not claim all-clear; the local audit owns blocking an
        # unresolved/unmarked record. No contradiction to flag here.
        return 0, [
            f"OK: {len(open_threads)} open bot thread(s), and the reconciliation "
            f"record does not claim all-clear ({body_class})."
        ]

    if body_class == "claims_clear":
        lead = (
            "reconciliation contradicts reality: the PR body records the "
            "automated-review findings as all fixed/waived, but these bot threads "
            "are still open:"
        )
    else:  # absent
        lead = (
            "no AI reconciliation record found, but these automated-review (bot) "
            "threads are still open and unaccounted for:"
        )
    messages = [lead]
    for t in open_threads:
        loc = t["path"] if t["line"] is None else f"{t['path']}:{t['line']}"
        messages.append(f"  - [{t['author']}] {loc}: {t['snippet']}")
    messages.append(
        "Resolve or explicitly waive (with a reason in the PR body) each finding "
        "before merge (AGENTS.md 4a.1)."
    )
    return 1, messages


def _gh(args: Sequence[str], gh: str) -> str:
    proc = subprocess.run(
        [gh, *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "gh failed").strip())
    return proc.stdout


def fetch_threads(pr: int, owner: str, name: str, gh: str) -> list[dict]:
    out = _gh(
        [
            "api", "graphql",
            "-f", f"query={_THREADS_QUERY}",
            "-F", f"owner={owner}",
            "-F", f"name={name}",
            "-F", f"pr={pr}",
        ],
        gh,
    )
    data = json.loads(out)
    return (
        ((((data.get("data") or {}).get("repository") or {}).get("pullRequest") or {})
         .get("reviewThreads") or {}).get("nodes")
    ) or []


def fetch_body(pr: int, repo: str, gh: str) -> str:
    out = _gh(["pr", "view", str(pr), "--repo", repo, "--json", "body", "-q", ".body"], gh)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=int, help="PR number")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name (defaults to $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--bots",
        default=os.environ.get("ATLAS_REVIEW_BOTS", ",".join(_DEFAULT_BOTS)),
        help="comma-separated bot login substrings (default: copilot,codex)",
    )
    parser.add_argument("--gh", default="gh", help="path to the gh CLI")
    parser.add_argument(
        "--threads-file",
        help="JSON file of reviewThreads nodes (test/dry-run; skips the live fetch)",
    )
    parser.add_argument(
        "--body-file",
        help="PR body file (test/dry-run; skips fetching the live body)",
    )
    args = parser.parse_args(argv)

    bots = [b.strip() for b in (args.bots or "").split(",") if b.strip()]
    if not bots:
        print("live reconciliation: no bot logins configured", file=sys.stderr)
        return 2

    try:
        if args.threads_file:
            nodes = json.loads(Path(args.threads_file).read_text(encoding="utf-8"))
        else:
            if args.pr is None or not args.repo:
                print(
                    "live reconciliation: need --pr and --repo (or $GITHUB_REPOSITORY) "
                    "when not using --threads-file",
                    file=sys.stderr,
                )
                return 2
            owner, _, name = args.repo.partition("/")
            nodes = fetch_threads(args.pr, owner, name, args.gh)

        if args.body_file:
            body = Path(args.body_file).read_text(encoding="utf-8")
        elif args.pr is not None and args.repo:
            body = fetch_body(args.pr, args.repo, args.gh)
        else:
            body = ""
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"live reconciliation: GitHub API/read error: {exc}", file=sys.stderr)
        return 2

    code, messages = evaluate(nodes, body, bots)
    print("live AI reconciliation check")
    print("-" * 60)
    for line in messages:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
