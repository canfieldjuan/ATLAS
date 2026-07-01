"""Markdown digest rendering for the Reddit listening tool (S3, #1934).

Two layers:

- :func:`render_digest` is a pure function of its inputs (candidates,
  replies, a date string) -- no store access, no clock, no I/O -- so
  rendering is deterministic and directly testable.
- :func:`write_digest` queries the real store for the radar (status
  ``new``) and warm replies (unseen), renders, and writes
  ``<digest_dir>/YYYY-MM-DD.md``. Re-running the same day overwrites the
  file, so regeneration is idempotent.

Titles and bodies are external Reddit text: they are escaped before
being placed inside Markdown link syntax so a hostile title like
``](http://evil)`` cannot break out of its link, and newlines are
collapsed so one post cannot forge digest structure.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .store import Candidate, ListeningStore, Reply

_EXCERPT_LENGTH = 140

_MD_ESCAPE = str.maketrans({"[": "\\[", "]": "\\]", "(": "\\(", ")": "\\)"})


def _sanitize_inline(text: str) -> str:
    """Collapse whitespace/newlines and escape Markdown link metachars so
    external Reddit text cannot forge digest structure."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed.translate(_MD_ESCAPE)


def _utc_date(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).date().isoformat()


def _excerpt(body: str) -> str:
    clean = _sanitize_inline(body)
    if len(clean) <= _EXCERPT_LENGTH:
        return clean
    return clean[: _EXCERPT_LENGTH - 3].rstrip() + "..."


def render_digest(
    *,
    candidates: list[Candidate],
    replies: list[Reply],
    generated_on: str,
) -> str:
    """Render the daily digest. Pure: same inputs, same output."""
    lines: list[str] = [f"# Reddit Listening Digest -- {generated_on}", ""]

    lines.append("## Radar (new candidates)")
    lines.append("")
    if candidates:
        for rank, candidate in enumerate(candidates, start=1):
            title = _sanitize_inline(candidate.title)
            topics = ", ".join(candidate.matched_topics) if candidate.matched_topics else "none"
            lines.append(
                f"{rank}. **[{title}]({candidate.url})** -- "
                f"r/{candidate.subreddit} -- score {candidate.final_score:g}"
            )
            lines.append(
                f"   topics: {topics} | posted: {_utc_date(candidate.created_utc)} "
                f"| comments: {candidate.num_comments} | reddit score: {candidate.reddit_score}"
            )
    else:
        lines.append("No new candidates.")
    lines.append("")

    lines.append("## Warm replies")
    lines.append("")
    if replies:
        for reply in replies:
            author = _sanitize_inline(reply.author) if reply.author else "unknown"
            target = "you" if reply.is_reply_to_me else "the thread"
            lines.append(
                f"- {author} replied to {target} on thread {reply.thread_id} "
                f"({_utc_date(reply.created_utc)}): \"{_excerpt(reply.body)}\""
            )
    else:
        lines.append(
            "No tracked-thread activity yet (the reply tracker lands in S5)."
        )
    lines.append("")

    return "\n".join(lines)


def write_digest(
    store: ListeningStore,
    *,
    digest_dir: Path,
    generated_on: str,
    limit: int = 20,
    min_final_score: float | None = None,
) -> Path:
    """Query the store, render, and write ``<digest_dir>/<date>.md``.

    The date arrives as data (``generated_on``, ISO ``YYYY-MM-DD``): the
    clock is the caller's boundary, which keeps this function
    deterministic and replayable for any day.
    """
    candidates = store.list_candidates(
        status="new", min_final_score=min_final_score, limit=limit
    )
    replies = store.list_replies(only_unseen=True, only_to_me=True)
    content = render_digest(
        candidates=candidates, replies=replies, generated_on=generated_on
    )
    digest_dir.mkdir(parents=True, exist_ok=True)
    path = digest_dir / f"{generated_on}.md"
    path.write_text(content, encoding="utf-8")
    return path
