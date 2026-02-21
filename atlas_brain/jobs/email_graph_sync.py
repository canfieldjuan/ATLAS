"""
Email-to-graph sync job.

Daily batch job (6am) that extracts graph-worthy facts from processed
emails and sends them to Graphiti for entity/relationship extraction.

Flow:
1. Query processed_emails where graph_synced_at IS NULL and priority = action_required
2. For each email, fetch full body from Gmail
3. Run qwen3:32b to extract concise factual summary (filter noise)
4. Send clean summaries to Graphiti via rag_client.send_messages()
5. Mark emails as graph_synced_at = NOW()

This is a two-stage LLM pipeline:
  Stage 1 (this job): qwen3:32b extracts graph-worthy facts from raw email
  Stage 2 (graphiti-wrapper): qwen3:32b extracts entities/relationships from clean summary
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("atlas.jobs.email_graph_sync")


class EmailGraphSync:
    """Batch sync processed emails to the knowledge graph."""

    def __init__(self, max_emails_per_run: int = 20):
        from ..config import settings

        self._settings = settings
        self.max_emails_per_run = max_emails_per_run
        self._rag_client = None
        self._llm = None
        self._gmail_client = None
        self._skill_prompt = None

    def _get_rag_client(self):
        if self._rag_client is None:
            from ..memory.rag_client import get_rag_client
            self._rag_client = get_rag_client()
        return self._rag_client

    def _get_llm(self):
        """Get or create a dedicated OllamaLLM instance for qwen3:32b."""
        if self._llm is None:
            from ..services.llm.ollama import OllamaLLM

            model = self._settings.memory.email_graph_model
            base_url = self._settings.llm.ollama_url
            self._llm = OllamaLLM(model=model, base_url=base_url)
            self._llm.load()
            logger.info("Loaded extraction LLM: %s at %s", model, base_url)
        return self._llm

    async def _get_gmail_client(self):
        if self._gmail_client is None:
            from ..autonomous.tasks.gmail_digest import _get_gmail_client
            self._gmail_client = await _get_gmail_client()
        return self._gmail_client

    def _get_skill_prompt(self) -> str:
        if self._skill_prompt is None:
            from ..skills import get_skill_registry
            skill = get_skill_registry().get("digest/email_graph_extract")
            self._skill_prompt = skill.content if skill else ""
        return self._skill_prompt

    async def _load_unsynced_emails(self) -> list[dict[str, Any]]:
        """Load processed emails that haven't been synced to the graph."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        priorities = self._settings.memory.email_graph_priorities

        rows = await pool.fetch(
            """
            SELECT gmail_message_id, sender, subject, category, priority,
                   processed_at
            FROM processed_emails
            WHERE graph_synced_at IS NULL
              AND priority = ANY($1::text[])
              AND processed_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
            ORDER BY processed_at ASC
            LIMIT $2
            """,
            priorities,
            self.max_emails_per_run,
        )

        return [dict(r) for r in rows]

    async def _mark_synced(self, gmail_message_ids: list[str]) -> None:
        """Mark emails as synced to the graph."""
        if not gmail_message_ids:
            return

        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return

        await pool.execute(
            """
            UPDATE processed_emails
            SET graph_synced_at = NOW()
            WHERE gmail_message_id = ANY($1::text[])
            """,
            gmail_message_ids,
        )

    _VALID_SENTIMENTS = {"positive", "neutral", "negative", "urgent"}

    def _parse_extraction(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Parse LLM output into (sentiment, facts).

        Expected format:
            SENTIMENT: negative
            Tia Jackson from Red Cross...

        Returns (sentiment, facts) where either may be None.
        SKIP in the facts position means no graph-worthy content.
        """
        lines = text.strip().splitlines()
        sentiment = None
        fact_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.upper().startswith("SENTIMENT:"):
                label = stripped.split(":", 1)[1].strip().lower()
                if label in self._VALID_SENTIMENTS:
                    sentiment = label
            else:
                fact_lines.append(stripped)

        facts = " ".join(l for l in fact_lines if l).strip()

        if not facts or facts.upper() == "SKIP":
            return sentiment, None

        return sentiment, facts

    async def _extract_facts(
        self,
        sender: str,
        subject: str,
        body_snippet: str,
        category: str,
        received_at: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Run LLM to extract graph-worthy facts and sentiment from an email.

        Returns (sentiment, facts). facts is None if the email should be skipped.
        """
        llm = self._get_llm()
        system_prompt = self._get_skill_prompt()

        user_input = json.dumps({
            "sender": sender,
            "subject": subject,
            "body_snippet": body_snippet,
            "category": category,
            "received_at": received_at,
        })

        from ..services.protocols import Message

        text = await llm.chat_async(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_input),
            ],
            max_tokens=350,
            temperature=0.1,
        )

        # Strip <think> tags (Qwen3 quirk with /no_think)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        return self._parse_extraction(text)

    async def run(self) -> dict:
        """Run the email graph sync job."""
        summary = {
            "emails_processed": 0,
            "emails_synced": 0,
            "emails_skipped": 0,
            "errors": [],
        }

        if not self._settings.memory.enabled:
            summary["_skip_synthesis"] = "Memory system disabled."
            return summary

        if not self._settings.memory.email_graph_sync_enabled:
            summary["_skip_synthesis"] = "Email graph sync disabled."
            return summary

        # Pre-flight: check Graphiti is reachable
        rag_client = self._get_rag_client()
        if not await rag_client.health_check():
            summary["errors"].append("Graphiti unreachable")
            summary["_skip_synthesis"] = "Graphiti unreachable."
            return summary

        # Load unsynced emails
        emails = await self._load_unsynced_emails()
        if not emails:
            summary["_skip_synthesis"] = "No unsynced emails to process."
            return summary

        logger.info("Email graph sync: %d emails to process", len(emails))

        # Get Gmail client for fetching full bodies
        try:
            gmail = await self._get_gmail_client()
        except Exception as e:
            summary["errors"].append(f"Gmail client init failed: {e}")
            summary["_skip_synthesis"] = "Gmail client unavailable."
            return summary

        # skipped_ids: emails with no graph-worthy content -- mark synced, never retry
        # graphiti_ids: emails with extracted facts -- only mark synced if Graphiti succeeds
        skipped_ids: list[str] = []
        graphiti_ids: list[str] = []
        messages_for_graphiti: list[dict] = []

        for email_row in emails:
            msg_id = email_row["gmail_message_id"]
            summary["emails_processed"] += 1

            # Fetch full body from Gmail
            try:
                full_msg = await gmail.get_message_full(msg_id)
            except Exception as e:
                logger.warning("Failed to fetch email %s: %s", msg_id, e)
                summary["errors"].append(f"{msg_id[:8]}: Gmail fetch failed")
                continue

            body_text = (full_msg.get("body_text") or "")[:500]
            sender = full_msg.get("from", email_row.get("sender", ""))
            subject = full_msg.get("subject", email_row.get("subject", ""))
            received_at = email_row.get("processed_at", "")
            if hasattr(received_at, "isoformat"):
                received_at = received_at.isoformat()

            # Stage 1: extract graph-worthy facts + sentiment
            try:
                sentiment, extracted = await self._extract_facts(
                    sender=sender,
                    subject=subject,
                    body_snippet=body_text,
                    category=email_row.get("category", "other"),
                    received_at=str(received_at),
                )
            except Exception as e:
                logger.warning("Extraction failed for %s: %s", msg_id, e)
                summary["errors"].append(f"{msg_id[:8]}: extraction failed")
                continue

            if extracted is None:
                logger.info("Skipped %s (no graph-worthy content)", msg_id)
                summary["emails_skipped"] += 1
                skipped_ids.append(msg_id)
                continue

            logger.info(
                "Extracted facts for %s [sentiment=%s]: %s",
                msg_id, sentiment or "unknown", extracted[:100],
            )

            # Append sentiment to facts so Graphiti indexes it with the entities
            content = extracted
            if sentiment:
                content = f"{extracted} The email sentiment is {sentiment}."

            # Build Graphiti message
            ts = email_row.get("processed_at")
            if ts is not None:
                timestamp = ts.isoformat() if ts.tzinfo else ts.isoformat() + "Z"
            else:
                timestamp = datetime.now(timezone.utc).isoformat()

            messages_for_graphiti.append({
                "content": content,
                "role_type": "system",
                "role": "email_digest",
                "source_description": f"email from {sender}: {subject} [sentiment:{sentiment or 'unknown'}]",
                "timestamp": timestamp,
            })
            graphiti_ids.append(msg_id)

        # Unload Stage 1 model before Graphiti calls Ollama -- frees the runner
        # so Graphiti (via Docker) doesn't hit a stale/busy model instance.
        if self._llm is not None:
            try:
                self._llm.unload()
                self._llm = None
                logger.info("Unloaded extraction LLM before Graphiti send")
            except Exception:
                pass

        # Stage 2: Send batch to Graphiti for entity/relationship extraction.
        # Brief pause lets Ollama's model runner fully settle after Stage 1.
        graphiti_synced = False
        if messages_for_graphiti:
            await asyncio.sleep(3)
            try:
                group_id = (
                    self._settings.memory.email_graph_group_id
                    or self._settings.memory.group_id
                )
                result = await rag_client.send_messages(
                    messages=messages_for_graphiti,
                    group_id=group_id,
                )
                episodes_created = result.get("episodes_created", 0) if result else 0
                if episodes_created > 0:
                    summary["emails_synced"] = len(messages_for_graphiti)
                    graphiti_synced = True
                    logger.info(
                        "Sent %d email extractions to Graphiti: %s",
                        len(messages_for_graphiti),
                        result.get("message", ""),
                    )
                elif result:
                    err_msg = result.get("message", "0 episodes created")
                    logger.warning("Graphiti extraction failed: %s -- will retry tomorrow", err_msg)
                    summary["errors"].append(f"Graphiti: {err_msg}")
                else:
                    summary["errors"].append("Empty response from Graphiti")
            except Exception as e:
                logger.error("Graphiti send failed: %s", e)
                summary["errors"].append(f"Graphiti send failed: {e}")

        # Mark emails as graph-synced:
        # - skipped (no graph content): always mark, no point retrying
        # - graphiti batch: only mark if Graphiti successfully created episodes
        ids_to_mark = skipped_ids + (graphiti_ids if graphiti_synced else [])
        if ids_to_mark:
            await self._mark_synced(ids_to_mark)
            logger.info("Marked %d emails as graph-synced", len(ids_to_mark))

        if not summary["emails_synced"] and not summary["errors"]:
            summary["_skip_synthesis"] = "No emails had graph-worthy content."

        logger.info(
            "Email graph sync complete: %d processed, %d synced, %d skipped, %d errors",
            summary["emails_processed"],
            summary["emails_synced"],
            summary["emails_skipped"],
            len(summary["errors"]),
        )

        return summary


async def run_email_graph_sync() -> dict:
    """Convenience function to run the email graph sync."""
    from ..config import settings

    if not settings.memory.email_graph_sync_enabled:
        logger.info("Email graph sync disabled in config")
        return {"status": "disabled", "_skip_synthesis": "Email graph sync disabled."}

    sync = EmailGraphSync()
    return await sync.run()
