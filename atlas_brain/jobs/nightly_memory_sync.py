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
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import httpx

logger = logging.getLogger("atlas.jobs.memory_sync")

GRAPHITI_CONTAINER = "atlas-graphiti-wrapper"  # fallback; prefer settings.autonomous.nightly_sync_container_name


class NightlyMemorySync:
    """
    Nightly job to sync conversations to long-term memory.

    Sends full conversation sessions to Graphiti's /messages endpoint,
    which handles LLM-powered entity extraction, relationship discovery,
    and deduplication internally.
    """

    def __init__(self, purge_days: int = None, max_turns_per_run: int = None):
        """
        Args:
            purge_days: Delete PostgreSQL messages older than this (default from config)
            max_turns_per_run: Cap turns processed per run to stay within timeout.
                Remaining turns are picked up on the next run.
        """
        from ..config import settings

        self.purge_days = purge_days if purge_days is not None else settings.memory.purge_days
        self.max_turns_per_run = max_turns_per_run if max_turns_per_run is not None else settings.autonomous.nightly_sync_max_turns
        self._rag_client = None

    def _get_rag_client(self):
        """Lazy load RAG client."""
        if self._rag_client is None:
            from ..memory.rag_client import get_rag_client
            self._rag_client = get_rag_client()
        return self._rag_client

    async def _ensure_graphiti_reachable(self) -> bool:
        """Pre-flight check: verify Graphiti is reachable, auto-restart container if not."""
        from ..config import settings

        url = f"{settings.memory.base_url}/healthcheck"

        # First attempt
        if await self._ping_graphiti(url):
            return True

        # Unreachable -- try restarting the container
        container = settings.autonomous.nightly_sync_container_name
        logger.warning(
            "Graphiti unreachable at %s, restarting container '%s'",
            url, container,
        )
        try:
            subprocess.run(
                ["docker", "restart", container],
                capture_output=True, timeout=30,
            )
        except Exception as e:
            logger.error("Failed to restart Graphiti container: %s", e)
            return False

        # Wait for it to come back up (up to 20s)
        for attempt in range(4):
            await asyncio.sleep(5)
            if await self._ping_graphiti(url):
                logger.info("Graphiti recovered after container restart")
                return True

        logger.error("Graphiti still unreachable after container restart")
        return False

    @staticmethod
    async def _ping_graphiti(url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

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
            "turns_remaining": 0,
            "messages_purged": 0,
            "errors": [],
        }

        try:
            # 0. Pre-flight: ensure Graphiti is reachable
            if not await self._ensure_graphiti_reachable():
                summary["errors"].append("Graphiti unreachable after auto-restart attempt")
                return summary

            # 1. Load un-synced conversation turns, grouped by session
            sessions_turns = await self._load_unsynced_turns()
            total_turns = sum(len(t) for t in sessions_turns.values())
            logger.info(
                "Found %d sessions with %d un-synced turns (cap: %d)",
                len(sessions_turns), total_turns, self.max_turns_per_run,
            )

            # 2. Process sessions until turn cap is reached
            rag_client = self._get_rag_client()
            turns_budget = self.max_turns_per_run

            for session_id, conversation_turns in sessions_turns.items():
                if turns_budget <= 0:
                    break
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
                        turns_budget -= len(messages)
                        logger.info(
                            "Session %s: sent %d turns to Graphiti (%d budget left)",
                            str(session_id)[:8],
                            len(messages),
                            max(turns_budget, 0),
                        )
                    else:
                        summary["errors"].append(
                            f"Session {str(session_id)[:8]}: empty response from Graphiti"
                        )

                except Exception as e:
                    logger.warning("Failed to process session %s: %s", session_id, e)
                    summary["errors"].append(f"Session {str(session_id)[:8]}: {e}")

            summary["turns_remaining"] = max(total_turns - summary["turns_sent"], 0)
            if summary["turns_remaining"] > 0:
                logger.info(
                    "%d turns remaining for next run", summary["turns_remaining"],
                )

            # 3. Purge old messages
            purged = await self._purge_old_messages()
            summary["messages_purged"] = purged

            logger.info(
                "Nightly sync complete: %d sessions, %d turns sent, "
                "%d remaining, %d purged",
                summary["sessions_processed"],
                summary["turns_sent"],
                summary["turns_remaining"],
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
