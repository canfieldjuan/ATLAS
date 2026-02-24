"""
Article enrichment pipeline: fetch full article content and classify
via SORAM pressure channels using a triage LLM.

Two-phase pipeline per article:
  Phase 1: httpx GET + trafilatura extract -> content column
  Phase 2: Triage LLM with soram_classification skill -> SORAM + linguistic + entities

Runs on an interval (default 10 min). Returns _skip_synthesis so the
runner does not double-synthesize.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.article_enrichment")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: enrich pending news articles."""
    cfg = settings.external_data
    if not cfg.enabled or not cfg.enrichment_enabled:
        return {"_skip_synthesis": "Article enrichment disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    max_batch = cfg.enrichment_max_per_batch
    max_attempts = cfg.enrichment_max_attempts

    # Fetch articles needing enrichment (pending or fetched but not classified)
    rows = await pool.fetch(
        """
        SELECT id, title, url, summary, matched_keywords,
               enrichment_status, enrichment_attempts, content
        FROM news_articles
        WHERE enrichment_status IN ('pending', 'fetched')
          AND enrichment_attempts < $1
          AND url != ''
        ORDER BY created_at DESC
        LIMIT $2
        """,
        max_attempts,
        max_batch,
    )

    if not rows:
        return {"_skip_synthesis": "No articles to enrich"}

    fetched = 0
    classified = 0
    failed = 0

    for row in rows:
        article_id = row["id"]
        status = row["enrichment_status"]
        attempts = row["enrichment_attempts"]

        fetched_content = None  # tracks content from Phase 1 for immediate Phase 2

        try:
            if status == "pending":
                # Phase 1: fetch content
                fetched_content = await _fetch_article_content(row["url"], cfg)
                if fetched_content:
                    await pool.execute(
                        """
                        UPDATE news_articles
                        SET content = $1,
                            enrichment_status = 'fetched',
                            enrichment_attempts = $2
                        WHERE id = $3
                        """,
                        fetched_content[:cfg.enrichment_content_max_chars],
                        attempts + 1,
                        article_id,
                    )
                    fetched += 1
                    # Continue to phase 2 immediately
                    status = "fetched"
                else:
                    await pool.execute(
                        """
                        UPDATE news_articles
                        SET enrichment_attempts = $1
                        WHERE id = $2
                        """,
                        attempts + 1,
                        article_id,
                    )
                    if attempts + 1 >= max_attempts:
                        await pool.execute(
                            "UPDATE news_articles SET enrichment_status = 'failed' WHERE id = $1",
                            article_id,
                        )
                        failed += 1
                    continue

            if status == "fetched":
                # Phase 2: SORAM classification
                article_content = fetched_content or row["content"]
                if not article_content:
                    article_content = row["summary"] or ""

                classification = await _classify_soram(
                    row["title"],
                    article_content,
                    row["matched_keywords"] or [],
                )

                if classification:
                    await pool.execute(
                        """
                        UPDATE news_articles
                        SET soram_channels = $1,
                            linguistic_indicators = $2,
                            entities_detected = $3,
                            enrichment_status = 'classified',
                            enrichment_attempts = $4,
                            enriched_at = $5
                        WHERE id = $6
                        """,
                        json.dumps(classification.get("soram_channels", {})),
                        json.dumps(classification.get("linguistic_indicators", {})),
                        classification.get("entities", []),
                        attempts + 1,
                        datetime.now(timezone.utc),
                        article_id,
                    )
                    classified += 1
                else:
                    await pool.execute(
                        """
                        UPDATE news_articles
                        SET enrichment_attempts = $1
                        WHERE id = $2
                        """,
                        attempts + 1,
                        article_id,
                    )
                    if attempts + 1 >= max_attempts:
                        await pool.execute(
                            "UPDATE news_articles SET enrichment_status = 'failed' WHERE id = $1",
                            article_id,
                        )
                        failed += 1

        except Exception:
            logger.exception("Failed to enrich article %s", article_id)
            try:
                await pool.execute(
                    """
                    UPDATE news_articles
                    SET enrichment_attempts = enrichment_attempts + 1
                    WHERE id = $1
                    """,
                    article_id,
                )
            except Exception:
                pass
            failed += 1

    logger.info(
        "Article enrichment: %d fetched, %d classified, %d failed (of %d)",
        fetched, classified, failed, len(rows),
    )

    return {
        "_skip_synthesis": "Article enrichment complete",
        "total": len(rows),
        "fetched": fetched,
        "classified": classified,
        "failed": failed,
    }


async def _fetch_article_content(url: str, cfg) -> str | None:
    """Fetch article HTML and extract main content via trafilatura."""
    import httpx

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=cfg.enrichment_fetch_timeout,
        ) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AtlasBot/1.0)",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            resp.raise_for_status()
            html = resp.text

        # trafilatura is CPU-bound; run in thread
        import trafilatura

        content = await asyncio.to_thread(trafilatura.extract, html)
        if content and len(content.strip()) > 50:
            return content.strip()

        logger.debug("trafilatura returned insufficient content for %s", url)
        return None

    except Exception as e:
        logger.debug("Failed to fetch article content from %s: %s", url, e)
        return None


async def _classify_soram(
    title: str,
    content: str,
    matched_keywords: list[str],
) -> dict[str, Any] | None:
    """Classify article via triage LLM using soram_classification skill."""
    from ...skills import get_skill_registry
    from ...services.llm_router import get_triage_llm
    from ...services.protocols import Message

    skill = get_skill_registry().get("digest/soram_classification")
    if not skill:
        logger.warning("Skill 'digest/soram_classification' not found")
        return None

    llm = get_triage_llm()
    if llm is None:
        # Fall back to active LLM
        from ...services import llm_registry

        llm = llm_registry.get_active()
    if llm is None:
        logger.warning("No LLM available for SORAM classification")
        return None

    # Truncate content for the classification call
    truncated = content[:3000] if content else ""

    user_payload = json.dumps({
        "title": title,
        "content": truncated,
        "matched_keywords": matched_keywords,
    })

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=user_payload),
    ]

    try:
        result = llm.chat(
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )
        text = result.get("response", "").strip()
        if not text:
            return None

        # Strip <think> tags (Qwen3 models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Try to parse JSON
        # Try ```json blocks first
        json_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        parsed = json.loads(text)

        # Validate expected structure
        if "soram_channels" not in parsed:
            logger.debug("SORAM classification missing soram_channels field")
            return None

        return parsed

    except json.JSONDecodeError:
        logger.debug("Failed to parse SORAM classification JSON: %s", text[:200])
        return None
    except Exception:
        logger.exception("SORAM classification failed")
        return None
