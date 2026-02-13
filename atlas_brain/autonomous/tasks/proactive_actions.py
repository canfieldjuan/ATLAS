"""
Proactive action extractor builtin task.

Scans recent user conversation turns for actionable patterns
("I need to...", "Remind me to...", etc.) and persists them as
proactive_actions for inclusion in the morning briefing.
"""

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.proactive_actions")

# Patterns that indicate an actionable intent
ACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "I need to..." / "I should..." / "I have to..." / "I want to..."
    (re.compile(
        r"[Ii]\s+(?:need|should|have|ought|want)\s+to\s+(.+?)(?:\.|!|$)",
    ), "task"),
    # "Remind me to..."
    (re.compile(
        r"[Rr]emind\s+me\s+to\s+(.+?)(?:\.|!|$)",
    ), "reminder"),
    # "I forgot to..." / "I still need to..."
    (re.compile(
        r"[Ii]\s+(?:forgot|still\s+need)\s+to\s+(.+?)(?:\.|!|$)",
    ), "task"),
    # "Tomorrow I need to..." / "This weekend I should..."
    (re.compile(
        r"(?:[Tt]omorrow|[Tt]his\s+(?:weekend|week|evening))\s+I\s+"
        r"(?:need|should|have|want)\s+to\s+(.+?)(?:\.|!|$)",
    ), "scheduled_task"),
    # "Don't let me forget to..."
    (re.compile(
        r"[Dd]on'?t\s+let\s+me\s+forget\s+to\s+(.+?)(?:\.|!|$)",
    ), "reminder"),
]


def _sha256(text: str) -> str:
    """Deterministic hash for dedup."""
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()


async def run(task: ScheduledTask) -> dict[str, Any]:
    """
    Extract actionable items from recent user conversations.

    Configurable via task.metadata:
        lookback_hours (int): How far back to scan (default 24)
    """
    from ...storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"error": "Database not initialized", "scanned_messages": 0}

    metadata = task.metadata or {}
    hours = metadata.get("lookback_hours", 24)
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

    # Fetch recent user messages
    rows = await pool.fetch(
        """
        SELECT content, created_at, session_id
        FROM conversation_turns
        WHERE role = 'user' AND created_at >= $1
        ORDER BY created_at DESC
        """,
        since,
    )

    # Extract actions via regex
    actions: list[dict[str, Any]] = []
    for row in rows:
        content = row["content"] or ""
        for pattern, action_type in ACTION_PATTERNS:
            for match in pattern.finditer(content):
                text = match.group(1).strip()
                if text:
                    actions.append({
                        "text": text,
                        "type": action_type,
                        "source_time": row["created_at"].isoformat(),
                        "session_id": str(row["session_id"]),
                    })

    # Dedup by normalized text
    seen: set[str] = set()
    unique_actions: list[dict[str, Any]] = []
    for a in actions:
        key = a["text"].lower().strip()
        if key not in seen and len(key) > 5:
            seen.add(key)
            unique_actions.append(a)

    # Persist (skip duplicates via ON CONFLICT)
    stored = 0
    for a in unique_actions:
        try:
            text_hash = _sha256(a["text"])
            sid = a["session_id"]
            result = await pool.execute(
                """
                INSERT INTO proactive_actions
                    (action_text, action_text_hash, action_type,
                     source_time, session_id)
                VALUES ($1, $2, $3, $4, $5::uuid)
                ON CONFLICT (action_text_hash)
                    WHERE status = 'pending'
                DO NOTHING
                """,
                a["text"],
                text_hash,
                a["type"],
                datetime.fromisoformat(a["source_time"]),
                sid,
            )
            # asyncpg returns "INSERT 0 1" on insert, "INSERT 0 0" on conflict
            if result and result.endswith("1"):
                stored += 1
        except Exception as e:
            logger.warning("Failed to store action: %s", e)

    summary = (
        f"Found {len(unique_actions)} actionable items from "
        f"{len(rows)} messages ({stored} new)."
    )
    logger.info("Proactive actions: %s", summary)

    return {
        "scanned_messages": len(rows),
        "actions_found": len(unique_actions),
        "actions_stored": stored,
        "actions": unique_actions[:10],
        "summary": summary,
    }
