"""
Article enrichment pipeline: fetch full article content and classify
via SORAM pressure channels using the local vLLM enrichment path.

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
from ...services.scraping.universal.html_cleaner import html_to_text
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._google_news import is_google_news_wrapper_url, resolve_google_news_url

logger = logging.getLogger("atlas.autonomous.tasks.article_enrichment")

_VALID_PRESSURE_DIRECTIONS = frozenset({"building", "steady", "releasing", "unclear"})
_DEFAULT_ARTICLE_FAILURE_REASON = "unexpected_exception"
_SORAM_CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "title": "soram_classification",
    "type": "object",
    "properties": {
        "soram_channels": {
            "type": "object",
            "properties": {
                "societal": {"type": "number"},
                "operational": {"type": "number"},
                "regulatory": {"type": "number"},
                "alignment": {"type": "number"},
                "media": {"type": "number"},
            },
            "required": ["societal", "operational", "regulatory", "alignment", "media"],
            "additionalProperties": False,
        },
        "linguistic_indicators": {
            "type": "object",
            "properties": {
                "permission_shift": {"type": "boolean"},
                "certainty_spike": {"type": "boolean"},
                "linguistic_dissociation": {"type": "boolean"},
                "hedging_withdrawal": {"type": "boolean"},
                "urgency_escalation": {"type": "boolean"},
            },
            "required": [
                "permission_shift",
                "certainty_spike",
                "linguistic_dissociation",
                "hedging_withdrawal",
                "urgency_escalation",
            ],
            "additionalProperties": False,
        },
        "entities": {"type": "array", "items": {"type": "string"}},
        "pressure_direction": {"type": "string", "enum": sorted(_VALID_PRESSURE_DIRECTIONS)},
    },
    "required": [
        "soram_channels",
        "linguistic_indicators",
        "entities",
        "pressure_direction",
    ],
    "additionalProperties": False,
}


def _normalize_failure_reason(reason: str | None) -> str:
    """Normalize a nullable failure reason to a short stored tag."""
    reason_text = str(reason or "").strip().lower()
    return reason_text or _DEFAULT_ARTICLE_FAILURE_REASON


def _record_failure_reason(counter: dict[str, int], reason: str | None) -> str:
    """Increment a per-run failure-reason counter and return the stored tag."""
    reason_text = _normalize_failure_reason(reason)
    counter[reason_text] = counter.get(reason_text, 0) + 1
    return reason_text


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
    failure_reasons: dict[str, int] = {}

    for row in rows:
        article_id = row["id"]
        status = row["enrichment_status"]
        attempts = row["enrichment_attempts"]

        fetched_content = None  # tracks content from Phase 1 for immediate Phase 2

        try:
            if status == "pending":
                # Phase 1: fetch content
                fetched_content, fetched_url, fetch_reason = await _fetch_article_content(
                    row["url"],
                    cfg,
                )
                if fetched_content:
                    await pool.execute(
                        """
                        UPDATE news_articles
                        SET content = $1,
                            url = $2,
                            enrichment_status = 'fetched',
                            enrichment_attempts = $3,
                            enrichment_failure_reason = NULL
                        WHERE id = $4
                        """,
                        fetched_content[:cfg.enrichment_content_max_chars],
                        fetched_url or row["url"],
                        attempts + 1,
                        article_id,
                    )
                    fetched += 1
                    # Continue to phase 2 immediately
                    status = "fetched"
                else:
                    stored_reason = _record_failure_reason(failure_reasons, fetch_reason)
                    terminal = attempts + 1 >= max_attempts
                    await pool.execute(
                        """
                        UPDATE news_articles
                        SET url = $1,
                            enrichment_attempts = $2,
                            enrichment_failure_reason = $3,
                            enrichment_status = $4
                        WHERE id = $5
                        """,
                        fetched_url or row["url"],
                        attempts + 1,
                        stored_reason,
                        "failed" if terminal else "pending",
                        article_id,
                    )
                    if terminal:
                        failed += 1
                    continue

            if status == "fetched":
                # Phase 2: SORAM classification
                article_content = fetched_content or row["content"]
                if not article_content:
                    article_content = row["summary"] or ""

                classification, classify_reason = await _classify_soram(
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
                            enriched_at = $6,
                            enrichment_failure_reason = NULL
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
                    stored_reason = _record_failure_reason(failure_reasons, classify_reason)
                    terminal = attempts + 1 >= max_attempts
                    await pool.execute(
                        """
                        UPDATE news_articles
                        SET enrichment_attempts = $1,
                            enrichment_failure_reason = $2,
                            enrichment_status = $3
                        WHERE id = $4
                        """,
                        attempts + 1,
                        stored_reason,
                        "failed" if terminal else "fetched",
                        article_id,
                    )
                    if terminal:
                        failed += 1

        except Exception:
            logger.exception("Failed to enrich article %s", article_id)
            try:
                stored_reason = _record_failure_reason(
                    failure_reasons,
                    _DEFAULT_ARTICLE_FAILURE_REASON,
                )
                terminal = attempts + 1 >= max_attempts
                await pool.execute(
                    """
                    UPDATE news_articles
                    SET enrichment_attempts = $1,
                        enrichment_failure_reason = $2,
                        enrichment_status = $3
                    WHERE id = $4
                    """,
                    attempts + 1,
                    stored_reason,
                    "failed" if terminal else status,
                    article_id,
                )
            except Exception:
                pass
            if terminal:
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
        "failure_reasons": failure_reasons,
    }


async def _fetch_article_content(url: str, cfg) -> tuple[str | None, str, str | None]:
    """Fetch article HTML and extract main content from the publisher page."""
    import httpx

    fetch_url = url
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=cfg.enrichment_fetch_timeout,
        ) as client:
            if is_google_news_wrapper_url(url):
                resolved = await resolve_google_news_url(
                    url,
                    timeout=cfg.enrichment_fetch_timeout,
                    client=client,
                )
                if resolved:
                    fetch_url = resolved
                else:
                    logger.debug("Could not resolve Google News wrapper URL %s", url)

            resp = await client.get(
                fetch_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AtlasBot/1.0)",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            resp.raise_for_status()
            html = resp.text

        content = await asyncio.to_thread(
            html_to_text,
            html,
            cfg.enrichment_content_max_chars,
        )
        if content and len(content.strip()) > 50:
            return content.strip(), fetch_url, None

        logger.debug("html extraction returned insufficient content for %s", fetch_url)
        return None, fetch_url, "fetch_extraction_empty"

    except httpx.TimeoutException as exc:
        logger.debug("Timed out fetching article content from %s: %s", fetch_url, exc)
        return None, fetch_url, "fetch_timeout"
    except httpx.HTTPStatusError as exc:
        logger.debug("HTTP error fetching article content from %s: %s", fetch_url, exc)
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in {401, 403, 429}:
            return None, fetch_url, "fetch_blocked"
        if status_code == 404:
            return None, fetch_url, "fetch_not_found"
        return None, fetch_url, "fetch_http_error"

    except Exception as e:
        logger.debug("Failed to fetch article content from %s: %s", fetch_url, e)
        return None, fetch_url, "fetch_error"


async def _classify_soram(
    title: str,
    content: str,
    matched_keywords: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Classify article via local vLLM using soram_classification skill."""
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    truncated = content[:3000] if content else ""

    payload = {
        "title": title,
        "content": truncated,
        "matched_keywords": matched_keywords,
    }

    usage: dict[str, Any] = {}
    max_output_tokens = max(
        256,
        int(getattr(settings.external_data, "enrichment_classification_max_tokens", 1200)),
    )
    try:
        text = call_llm_with_skill(
            "digest/soram_classification", payload,
            max_tokens=max_output_tokens,
            temperature=0.1,
            workload="vllm",
            response_format={"type": "json_object"},
            guided_json=_SORAM_CLASSIFICATION_JSON_SCHEMA,
            usage_out=usage,
        )
    except Exception:
        logger.exception("SORAM classification LLM call failed")
        return None, "classify_error"
    if usage.get("input_tokens"):
        logger.info("soram_classification LLM tokens: in=%d out=%d model=%s",
                     usage["input_tokens"], usage["output_tokens"], usage.get("model", ""))
    if not text:
        return None, "classify_empty_response"

    parsed = parse_json_response(text, recover_truncated=True)

    # parse_json_response always returns a dict; check for required field
    if "soram_channels" not in parsed:
        logger.debug("SORAM classification missing soram_channels: %s", text[:200])
        return None, "classify_invalid_json"

    return _validate_classification(parsed), None


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
        direction = "unclear"
    raw["pressure_direction"] = direction

    return raw
