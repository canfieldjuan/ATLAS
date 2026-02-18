"""
Nightly memory sync job.

Processes daily conversation sessions by sending full conversation
context to Graphiti's /messages endpoint for LLM-powered entity and
relationship extraction. This is the sole write path to the knowledge graph.

Flow:
1. Load conversation sessions from PostgreSQL for the target date
2. For each session, send all user+assistant turns as a batch to Graphiti
3. Purge old PostgreSQL messages
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

logger = logging.getLogger("atlas.jobs.memory_sync")


class NightlyMemorySync:
    """
    Nightly job to sync conversations to long-term memory.

    Sends full conversation sessions to Graphiti's /messages endpoint,
    which handles LLM-powered entity extraction, relationship discovery,
    and deduplication internally.
    """

    def __init__(self, purge_days: int = None):
        """
        Args:
            purge_days: Delete PostgreSQL messages older than this (default from config)
        """
        from ..config import settings

        self.purge_days = purge_days if purge_days is not None else settings.memory.purge_days
        self._rag_client = None

    def _get_rag_client(self):
        """Lazy load RAG client."""
        if self._rag_client is None:
            from ..memory.rag_client import get_rag_client
            self._rag_client = get_rag_client()
        return self._rag_client

    async def run(self, target_date: Optional[datetime] = None) -> dict:
        """
        Run the nightly sync job.

        Args:
            target_date: Date to process (defaults to today)

        Returns:
            Summary of operations performed
        """
        target_date = target_date or datetime.now(timezone.utc)
        logger.info("Starting nightly memory sync for %s", target_date.date())

        summary = {
            "date": str(target_date.date()),
            "sessions_processed": 0,
            "turns_sent": 0,
            "messages_purged": 0,
            "errors": [],
        }

        try:
            # 1. Load un-synced conversation turns, grouped by session
            sessions_turns = await self._load_unsynced_turns()
            logger.info("Found %d sessions with un-synced turns", len(sessions_turns))

            # 2. Process each session
            rag_client = self._get_rag_client()

            for session_id, conversation_turns in sessions_turns.items():
                try:
                    # Collect turn IDs for marking as synced on success
                    turn_ids = [turn["id"] for turn in conversation_turns]

                    # Format turns as Graphiti message objects
                    messages = []
                    for turn in conversation_turns:
                        role = turn.get("role", "user")
                        role_type = "assistant" if role == "assistant" else "user"
                        ts = turn.get("created_at")
                        if ts is not None:
                            timestamp = ts.isoformat() if ts.tzinfo else ts.isoformat() + "Z"
                        else:
                            timestamp = datetime.now(timezone.utc).isoformat()

                        msg = {
                            "content": turn.get("content", ""),
                            "role_type": role_type,
                            "role": turn.get("speaker_id") or role,
                            "timestamp": timestamp,
                        }

                        # Include memory quality signals if present
                        raw_meta = turn.get("metadata")
                        if raw_meta:
                            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                            if isinstance(meta, dict) and "memory_quality" in meta:
                                msg["memory_quality"] = meta["memory_quality"]

                        messages.append(msg)

                    # Send batch to Graphiti
                    result = await rag_client.send_messages(
                        messages=messages,
                    )

                    if result:
                        await self._mark_turns_synced(turn_ids)
                        summary["sessions_processed"] += 1
                        summary["turns_sent"] += len(messages)
                        logger.info(
                            "Session %s: sent %d turns to Graphiti",
                            str(session_id)[:8],
                            len(messages),
                        )
                    else:
                        summary["errors"].append(
                            f"Session {str(session_id)[:8]}: empty response from Graphiti"
                        )

                except Exception as e:
                    logger.warning("Failed to process session %s: %s", session_id, e)
                    summary["errors"].append(f"Session {str(session_id)[:8]}: {e}")

            # 3. Purge old messages
            purged = await self._purge_old_messages()
            summary["messages_purged"] = purged

            logger.info(
                "Nightly sync complete: %d sessions processed, %d turns sent, "
                "%d messages purged",
                summary["sessions_processed"],
                summary["turns_sent"],
                summary["messages_purged"],
            )

        except Exception as e:
            logger.error("Nightly sync failed: %s", e)
            summary["errors"].append(str(e))

        return summary

    async def _load_unsynced_turns(self) -> dict[UUID, list[dict]]:
        """Load non-command conversation turns that have not been synced, grouped by session."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return {}

        conn = await pool.acquire()
        try:
            rows = await conn.fetch(
                """
                SELECT session_id, id, role, content, turn_type, speaker_id,
                       created_at, metadata
                FROM conversation_turns
                WHERE synced_at IS NULL
                  AND turn_type != 'command'
                ORDER BY created_at ASC
                """,
            )

            sessions: dict[UUID, list[dict]] = {}
            for row in rows:
                sid = row["session_id"]
                if sid not in sessions:
                    sessions[sid] = []
                sessions[sid].append(dict(row))

            return sessions
        finally:
            await pool.release(conn)

    async def _mark_turns_synced(self, turn_ids: list[UUID]) -> None:
        """Mark conversation turns as synced after successful GraphRAG send."""
        if not turn_ids:
            return
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return

        conn = await pool.acquire()
        try:
            await conn.execute(
                """
                UPDATE conversation_turns
                SET synced_at = NOW()
                WHERE id = ANY($1::uuid[])
                """,
                turn_ids,
            )
        finally:
            await pool.release(conn)

    async def _purge_old_messages(self) -> int:
        """Purge synced PostgreSQL messages older than purge_days.

        Only deletes turns that have been successfully synced to GraphRAG,
        preventing data loss if sync failed for a session.
        """
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return 0

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.purge_days)

        conn = await pool.acquire()
        try:
            result = await conn.execute(
                """
                DELETE FROM conversation_turns
                WHERE created_at < $1
                  AND synced_at IS NOT NULL
                """,
                cutoff_date,
            )
            # Parse "DELETE N" response
            count = int(result.split()[-1]) if result else 0
            logger.info("Purged %d synced messages older than %s", count, cutoff_date.date())
            return count
        finally:
            await pool.release(conn)


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
