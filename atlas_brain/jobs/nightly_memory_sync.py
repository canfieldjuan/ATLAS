"""
Nightly memory sync job.

Processes daily conversation sessions:
1. Loads conversations from PostgreSQL (skip commands)
2. Extracts facts from explicit triggers
3. Deduplicates against existing graph
4. Embeds and stores new facts in GraphRAG
5. Purges old PostgreSQL messages
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

logger = logging.getLogger("atlas.jobs.memory_sync")


@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""
    content: str
    source_turn_id: str
    speaker: Optional[str] = None
    timestamp: Optional[datetime] = None
    trigger_type: str = "explicit"  # explicit, preference, personal_info


class FactExtractor:
    """
    Extracts memorable facts from conversations.

    Uses explicit trigger patterns to identify facts worth storing
    in long-term memory.
    """

    # Explicit memory triggers - user explicitly asks to remember
    EXPLICIT_TRIGGERS = [
        r"remember\s+(?:that\s+)?(.+)",
        r"don'?t\s+forget\s+(?:that\s+)?(.+)",
        r"keep\s+in\s+mind\s+(?:that\s+)?(.+)",
        r"make\s+(?:a\s+)?note\s+(?:that\s+)?(.+)",
        r"save\s+(?:this|that)?\s*[:\-]?\s*(.+)",
    ]

    # Preference patterns - user states preferences
    PREFERENCE_TRIGGERS = [
        r"my\s+favorite\s+(\w+)\s+is\s+(.+)",
        r"i\s+(?:really\s+)?(?:love|like|prefer|enjoy)\s+(.+)",
        r"i\s+(?:don'?t|never)\s+(?:like|want|eat|use)\s+(.+)",
        r"i\s+always\s+(.+)",
        r"i\s+usually\s+(.+)",
    ]

    # Personal info patterns - user shares personal details
    PERSONAL_INFO_TRIGGERS = [
        r"my\s+name\s+is\s+(\w+)",
        r"i(?:'?m|\s+am)\s+(\w+)",  # "I'm Juan" or "I am Juan"
        r"i\s+live\s+(?:in|at)\s+(.+)",
        r"i\s+work\s+(?:at|for)\s+(.+)",
        r"my\s+(?:wife|husband|partner|spouse)(?:'?s)?\s+(?:name\s+is\s+)?(\w+)",
        r"my\s+(?:son|daughter|child|kid)(?:'?s)?\s+(?:name\s+is\s+)?(\w+)",
        r"my\s+(?:dog|cat|pet)(?:'?s)?\s+(?:name\s+is\s+)?(\w+)",
        r"my\s+birthday\s+is\s+(.+)",
        r"i(?:'?m|\s+am)\s+(\d+)\s+years?\s+old",
    ]

    def __init__(self):
        # Compile all patterns
        self._explicit_patterns = [re.compile(p, re.IGNORECASE) for p in self.EXPLICIT_TRIGGERS]
        self._preference_patterns = [re.compile(p, re.IGNORECASE) for p in self.PREFERENCE_TRIGGERS]
        self._personal_patterns = [re.compile(p, re.IGNORECASE) for p in self.PERSONAL_INFO_TRIGGERS]

    def extract_facts(self, text: str, turn_id: str, speaker: str = None, timestamp: datetime = None) -> list[ExtractedFact]:
        """
        Extract facts from a conversation turn.

        Args:
            text: The conversation text
            turn_id: ID of the conversation turn
            speaker: Optional speaker identifier
            timestamp: When the turn occurred

        Returns:
            List of extracted facts
        """
        facts = []

        # Check explicit triggers first (highest priority)
        for pattern in self._explicit_patterns:
            match = pattern.search(text)
            if match:
                fact_content = match.group(1).strip()
                if fact_content:
                    facts.append(ExtractedFact(
                        content=f"User asked to remember: {fact_content}",
                        source_turn_id=turn_id,
                        speaker=speaker,
                        timestamp=timestamp,
                        trigger_type="explicit",
                    ))

        # Check preference triggers
        for pattern in self._preference_patterns:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    fact_content = f"User's favorite {groups[0]} is {groups[1]}"
                else:
                    fact_content = f"User preference: {groups[0]}"
                facts.append(ExtractedFact(
                    content=fact_content.strip(),
                    source_turn_id=turn_id,
                    speaker=speaker,
                    timestamp=timestamp,
                    trigger_type="preference",
                ))

        # Check personal info triggers
        for pattern in self._personal_patterns:
            match = pattern.search(text)
            if match:
                fact_content = text.strip()  # Keep original context for personal info
                facts.append(ExtractedFact(
                    content=fact_content,
                    source_turn_id=turn_id,
                    speaker=speaker,
                    timestamp=timestamp,
                    trigger_type="personal_info",
                ))
                break  # Only one personal info fact per turn

        return facts


class NightlyMemorySync:
    """
    Nightly job to sync conversations to long-term memory.

    Processes:
    1. Today's conversation sessions
    2. Extracts facts using trigger patterns
    3. Deduplicates against existing graph
    4. Stores new facts in GraphRAG
    5. Purges old PostgreSQL data
    """

    def __init__(
        self,
        purge_days: int = None,
        similarity_threshold: float = None,
    ):
        """
        Args:
            purge_days: Delete PostgreSQL messages older than this (default from config)
            similarity_threshold: Skip facts with similarity > this (default from config)
        """
        from ..config import settings

        self.purge_days = purge_days if purge_days is not None else settings.memory.purge_days
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else settings.memory.similarity_threshold
        )
        self.fact_extractor = FactExtractor()
        self._memory_client = None

    def _get_memory_client(self):
        """Lazy load memory client."""
        if self._memory_client is None:
            from ..services.memory import get_memory_client
            self._memory_client = get_memory_client()
        return self._memory_client

    async def run(self, target_date: Optional[datetime] = None) -> dict:
        """
        Run the nightly sync job.

        Args:
            target_date: Date to process (defaults to today)

        Returns:
            Summary of operations performed
        """
        target_date = target_date or datetime.now()
        logger.info("Starting nightly memory sync for %s", target_date.date())

        summary = {
            "date": str(target_date.date()),
            "sessions_processed": 0,
            "turns_scanned": 0,
            "facts_extracted": 0,
            "facts_stored": 0,
            "duplicates_skipped": 0,
            "messages_purged": 0,
            "errors": [],
        }

        try:
            # 1. Load today's sessions
            sessions = await self._load_sessions_for_date(target_date)
            summary["sessions_processed"] = len(sessions)
            logger.info("Found %d sessions to process", len(sessions))

            # 2. Process each session
            all_facts = []
            for session_id in sessions:
                turns = await self._load_conversation_turns(session_id)
                summary["turns_scanned"] += len(turns)

                for turn in turns:
                    # Skip assistant turns and commands
                    if turn.get("role") != "user":
                        continue
                    if turn.get("turn_type") == "command":
                        continue

                    facts = self.fact_extractor.extract_facts(
                        text=turn.get("content", ""),
                        turn_id=str(turn.get("id", "")),
                        speaker=turn.get("speaker_id"),
                        timestamp=turn.get("created_at"),
                    )
                    all_facts.extend(facts)

            summary["facts_extracted"] = len(all_facts)
            logger.info("Extracted %d facts from conversations", len(all_facts))

            # 3. Deduplicate and store facts
            for fact in all_facts:
                try:
                    is_duplicate = await self._check_duplicate(fact)
                    if is_duplicate:
                        summary["duplicates_skipped"] += 1
                        continue

                    await self._store_fact(fact)
                    summary["facts_stored"] += 1
                except Exception as e:
                    logger.warning("Failed to store fact: %s", e)
                    summary["errors"].append(str(e))

            # 4. Purge old messages
            purged = await self._purge_old_messages()
            summary["messages_purged"] = purged

            logger.info(
                "Nightly sync complete: %d facts stored, %d duplicates skipped, %d messages purged",
                summary["facts_stored"],
                summary["duplicates_skipped"],
                summary["messages_purged"],
            )

        except Exception as e:
            logger.error("Nightly sync failed: %s", e)
            summary["errors"].append(str(e))

        return summary

    async def _load_sessions_for_date(self, target_date: datetime) -> list[UUID]:
        """Load session IDs for a specific date."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT id FROM sessions
                WHERE created_at >= $1 AND created_at < $2
                """,
                start_of_day,
                end_of_day,
            )
            return [row["id"] for row in rows]

    async def _load_conversation_turns(self, session_id: UUID) -> list[dict]:
        """Load conversation turns for a session."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, turn_type, speaker_id, created_at
                FROM conversation_turns
                WHERE session_id = $1
                ORDER BY created_at ASC
                """,
                session_id,
            )
            return [dict(row) for row in rows]

    async def _check_duplicate(self, fact: ExtractedFact) -> bool:
        """Check if a similar fact already exists in the graph."""
        try:
            memory_client = self._get_memory_client()
            results = await memory_client.search(
                query=fact.content,
                num_results=1,
            )

            if results and results[0].score >= self.similarity_threshold:
                logger.debug(
                    "Duplicate detected (score=%.2f): %s",
                    results[0].score,
                    fact.content[:50],
                )
                return True
            return False

        except Exception as e:
            logger.warning("Duplicate check failed: %s", e)
            return False  # Store anyway if check fails

    async def _store_fact(self, fact: ExtractedFact) -> None:
        """Store a fact in the knowledge graph."""
        memory_client = self._get_memory_client()

        await memory_client.add_fact(
            fact=fact.content,
            source=f"atlas-nightly-sync:{fact.trigger_type}",
            timestamp=fact.timestamp,
        )

        logger.debug("Stored fact: %s", fact.content[:50])

    async def _purge_old_messages(self) -> int:
        """Purge PostgreSQL messages older than purge_days."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return 0

        cutoff_date = datetime.now() - timedelta(days=self.purge_days)

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM conversation_turns
                WHERE created_at < $1
                """,
                cutoff_date,
            )
            # Parse "DELETE N" response
            count = int(result.split()[-1]) if result else 0
            logger.info("Purged %d messages older than %s", count, cutoff_date.date())
            return count


async def run_nightly_sync(purge_days: int = None):
    """Convenience function to run the nightly sync.

    Args:
        purge_days: Override config purge_days (default uses config setting)
    """
    from ..config import settings

    if not settings.memory.nightly_sync_enabled:
        logger.info("Nightly sync disabled in config")
        return {"status": "disabled"}

    sync = NightlyMemorySync(purge_days=purge_days)
    return await sync.run()


if __name__ == "__main__":
    # Allow running directly: python -m atlas_brain.jobs.nightly_memory_sync
    import sys

    logging.basicConfig(level=logging.INFO)

    purge_days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    result = asyncio.run(run_nightly_sync(purge_days))
    print(f"Sync complete: {result}")
