"""
Auto-tune user response_style and expertise_level from conversation patterns.

Nightly cron task (2:30 AM) that analyzes recent conversation turns per speaker
and updates user_profiles when strong signals are detected.
"""

import logging
import re
from typing import Any

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.preference_learning")

# Regex patterns for explicit style signals
_BRIEF_PATTERNS = re.compile(
    r"\b(be brief|keep it short|shorter|too long|tldr|tl;dr|just the answer)\b",
    re.IGNORECASE,
)
_DETAILED_PATTERNS = re.compile(
    r"\b(more detail|elaborate|explain more|go deeper|tell me more|in depth)\b",
    re.IGNORECASE,
)
_BEGINNER_PATTERNS = re.compile(
    r"\b(explain simply|eli5|what does .+ mean|i don'?t understand|basic)\b",
    re.IGNORECASE,
)
_EXPERT_PATTERNS = re.compile(
    r"\b(skip the basics|i know|technically|implementation detail|under the hood)\b",
    re.IGNORECASE,
)


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Analyze recent conversations and auto-tune user profiles."""
    from ...storage.database import get_db_pool
    from ...storage.repositories.profile import get_profile_repo

    from ...config import settings

    metadata = task.metadata or {}
    lookback_days = metadata.get("lookback_days", settings.autonomous.preference_learning_lookback_days)
    min_turns = metadata.get("min_turns", settings.autonomous.preference_learning_min_turns)

    pool = get_db_pool()
    if not pool.is_initialized:
        return {
            "error": "Database not initialized",
            "_skip_synthesis": "Preference learning skipped -- database not ready.",
        }

    # Fetch user turns from the last N days grouped by speaker_uuid
    rows = await pool.fetch(
        """
        SELECT ct.speaker_uuid, ct.content
        FROM conversation_turns ct
        WHERE ct.role = 'user'
          AND ct.speaker_uuid IS NOT NULL
          AND ct.created_at >= NOW() - ($1 || ' days')::interval
        ORDER BY ct.speaker_uuid, ct.created_at
        """,
        str(lookback_days),
    )

    # Group by speaker_uuid
    speakers: dict[str, list[str]] = {}
    for row in rows:
        uid = str(row["speaker_uuid"])
        speakers.setdefault(uid, []).append(row["content"] or "")

    profile_repo = get_profile_repo()
    changes: list[dict[str, str]] = []

    for user_uuid_str, messages in speakers.items():
        if len(messages) < min_turns:
            continue

        # Compute signals
        brief_hits = sum(1 for m in messages if _BRIEF_PATTERNS.search(m))
        detailed_hits = sum(1 for m in messages if _DETAILED_PATTERNS.search(m))
        beginner_hits = sum(1 for m in messages if _BEGINNER_PATTERNS.search(m))
        expert_hits = sum(1 for m in messages if _EXPERT_PATTERNS.search(m))

        avg_length = sum(len(m) for m in messages) / len(messages)

        new_style = None
        new_expertise = None

        # Style: explicit mentions take priority, then message length
        if brief_hits >= 3 and brief_hits > detailed_hits:
            new_style = "brief"
        elif detailed_hits >= 3 and detailed_hits > brief_hits:
            new_style = "detailed"
        elif avg_length < 30:
            new_style = "brief"
        elif avg_length > 150:
            new_style = "detailed"

        # Expertise: explicit mentions only
        if beginner_hits >= 3 and beginner_hits > expert_hits:
            new_expertise = "beginner"
        elif expert_hits >= 3 and expert_hits > beginner_hits:
            new_expertise = "expert"

        if not new_style and not new_expertise:
            continue

        try:
            from uuid import UUID
            user_uuid = UUID(user_uuid_str)

            await profile_repo.update_profile(
                user_id=user_uuid,
                response_style=new_style,
                expertise_level=new_expertise,
            )
            change = {"speaker_uuid": user_uuid_str}
            if new_style:
                change["response_style"] = new_style
            if new_expertise:
                change["expertise_level"] = new_expertise
            changes.append(change)
            logger.info("Updated profile for %s: style=%s expertise=%s",
                        user_uuid_str, new_style, new_expertise)
        except Exception as e:
            logger.warning("Failed to update profile for %s: %s", user_uuid_str, e)

    return {
        "users_analyzed": len(speakers),
        "profiles_updated": len(changes),
        "changes": changes,
    }
