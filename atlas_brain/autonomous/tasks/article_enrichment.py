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
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.article_enrichment")

_VALID_PRESSURE_DIRECTIONS = frozenset({"building", "steady", "releasing", "unclear"})


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
                    soram = classification.get("soram_channels", {})
                    ling = classification.get("linguistic_indicators", {})
                    entities = classification.get("entities", [])
                    direction = classification.get("pressure_direction")
                    await pool.execute(
                        """
                        UPDATE news_articles
                        SET soram_channels = $1::jsonb,
                            linguistic_indicators = $2::jsonb,
                            entities_detected = $3,
                            pressure_direction = $4,
                            enrichment_status = 'classified',
                            enrichment_attempts = $5,
                            enriched_at = $6
                        WHERE id = $7
                        """,
                        json.dumps(soram),
                        json.dumps(ling),
                        entities,
                        direction,
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
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    truncated = content[:3000] if content else ""

    payload = {
        "title": title,
        "content": truncated,
        "matched_keywords": matched_keywords,
    }

    text = call_llm_with_skill(
        "digest/soram_classification", payload,
        max_tokens=512, temperature=0.1,
        workload="triage",
        try_openrouter=False, auto_activate_ollama=True,
    )
    if not text:
        return None

    parsed = parse_json_response(text, recover_truncated=True)

    # parse_json_response always returns a dict; check for required field
    if "soram_channels" not in parsed:
        logger.debug("SORAM classification missing soram_channels: %s", text[:200])
        return None

    return _validate_classification(parsed)


def _validate_classification(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and clamp SORAM classification values from LLM output."""
    # Validate soram_channels: each value must be float 0.0-1.0
    soram = raw.get("soram_channels", {})
    if isinstance(soram, dict):
        validated_soram = {}
        for key in ("societal", "operational", "regulatory", "alignment", "media"):
            val = soram.get(key, 0.0)
            try:
                validated_soram[key] = max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                validated_soram[key] = 0.0
        raw["soram_channels"] = validated_soram
    else:
        raw["soram_channels"] = {k: 0.0 for k in ("societal", "operational", "regulatory", "alignment", "media")}

    # Validate linguistic_indicators: each value must be bool
    ling = raw.get("linguistic_indicators", {})
    if isinstance(ling, dict):
        validated_ling = {}
        for key in ("permission_shift", "certainty_spike", "linguistic_dissociation",
                     "hedging_withdrawal", "urgency_escalation"):
            validated_ling[key] = bool(ling.get(key, False))
        raw["linguistic_indicators"] = validated_ling
    else:
        raw["linguistic_indicators"] = {k: False for k in (
            "permission_shift", "certainty_spike", "linguistic_dissociation",
            "hedging_withdrawal", "urgency_escalation")}

    # Validate entities: list of non-empty strings
    entities = raw.get("entities", [])
    if isinstance(entities, list):
        raw["entities"] = [str(e) for e in entities[:5] if e and str(e).strip()]
    else:
        raw["entities"] = []

    # Validate pressure_direction
    direction = raw.get("pressure_direction", "unclear")
    if direction not in _VALID_PRESSURE_DIRECTIONS:
        raw["pressure_direction"] = "unclear"

    return raw
