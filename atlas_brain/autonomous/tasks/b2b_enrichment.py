"""
B2B review enrichment: extract churn signals from pending reviews via LLM
using the b2b_churn_extraction skill.

Single-pass enrichment (one LLM call per review). Polls b2b_reviews WHERE
enrichment_status = 'pending', calls LLM, stores result in enrichment JSONB
column, sets status to 'enriched'.

Runs on an interval (default 5 min). Returns _skip_synthesis so the
runner does not double-synthesize.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...services.company_normalization import normalize_company_name
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...services.scraping.sources import VERIFIED_SOURCES as _VERIFIED_SOURCES

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")

# Dedicated OpenRouter instance for enrichment -- not shared with the global
# registry singleton (used by synthesis/reasoning workloads).
_enrichment_llm = None


def _get_enrichment_llm():
    """Get or create a dedicated OpenRouter LLM for enrichment."""
    global _enrichment_llm
    target_model = (
        str(settings.b2b_churn.enrichment_openrouter_model or "").strip()
        or str(settings.llm.openrouter_reasoning_model or "").strip()
    )
    or_key = (
        str(settings.b2b_churn.openrouter_api_key or "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("ATLAS_B2B_CHURN_OPENROUTER_API_KEY", "")
    )
    if not or_key or not target_model:
        return None

    if _enrichment_llm is not None and getattr(_enrichment_llm, "model", "") == target_model:
        return _enrichment_llm

    try:
        from ...services.llm.openrouter import OpenRouterLLM
        _enrichment_llm = OpenRouterLLM(model=target_model, api_key=or_key)
        _enrichment_llm.load()
        logger.info("Dedicated enrichment LLM ready: %s", target_model)
        return _enrichment_llm
    except Exception as e:
        logger.warning("Failed to create enrichment LLM: %s", e)
        return None


# ------------------------------------------------------------------
# Hybrid enrichment: cached LLM instances + connection pool
# ------------------------------------------------------------------
_tier2_llm = None


def _get_tier2_llm(cfg):
    """Get or create a cached OpenRouter LLM for Tier 2 hybrid extraction."""
    global _tier2_llm
    target_model = cfg.enrichment_tier2_model
    if not target_model:
        return None

    if _tier2_llm is not None and getattr(_tier2_llm, "model", "") == target_model:
        return _tier2_llm

    or_key = (
        str(settings.b2b_churn.openrouter_api_key or "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("ATLAS_B2B_CHURN_OPENROUTER_API_KEY", "")
    )
    if not or_key:
        return None

    try:
        from ...services.llm.openrouter import OpenRouterLLM
        _tier2_llm = OpenRouterLLM(model=target_model, api_key=or_key)
        _tier2_llm.load()
        logger.info("Dedicated Tier 2 LLM ready: %s", target_model)
        return _tier2_llm
    except Exception as e:
        logger.warning("Failed to create Tier 2 LLM: %s", e)
        return None


_tier1_client = None


def _get_tier1_client(cfg):
    """Get or create a pooled httpx.AsyncClient for Tier 1 vLLM calls."""
    global _tier1_client
    if _tier1_client is not None and not _tier1_client.is_closed:
        return _tier1_client

    import httpx

    _tier1_client = httpx.AsyncClient(
        base_url=cfg.enrichment_tier1_vllm_url,
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
    )
    return _tier1_client


async def _call_vllm_tier1(payload_json: str, cfg, client) -> tuple[dict | None, str | None]:
    """Tier 1 extraction: deterministic fields via local vLLM.

    Returns (result_dict, model_id) or (None, None) on failure.
    """
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier1")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier1' not found")
        return None, None

    model_id = cfg.enrichment_tier1_model
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": cfg.enrichment_tier1_model,
                "messages": [
                    {"role": "system", "content": skill.content},
                    {"role": "user", "content": payload_json},
                ],
                "max_tokens": cfg.enrichment_tier1_max_tokens,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if not text:
            return None, model_id

        from ...pipelines.llm import clean_llm_output, parse_json_response
        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            return parsed, model_id
        return None, model_id
    except (json.JSONDecodeError, ValueError):
        logger.warning("Tier 1 vLLM returned invalid JSON")
        return None, model_id
    except Exception:
        logger.exception("Tier 1 vLLM call failed")
        return None, None


async def _call_cloud_tier2(row, cfg, truncate_length: int) -> tuple[dict | None, str | None]:
    """Tier 2 extraction: interpretive fields via cloud LLM (OpenRouter).

    Returns (result_dict, model_id) or (None, None) on failure.
    """
    from ...skills import get_skill_registry
    from ...services.protocols import Message
    from ...pipelines.llm import clean_llm_output

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier2' not found")
        return None, None

    # Tier 2 must use a cloud model (OpenRouter). Falling back to local would
    # defeat the purpose -- Tier 1 already covers the local path.
    llm = _get_tier2_llm(cfg) or _get_enrichment_llm()
    if llm is None:
        logger.warning("No cloud LLM available for Tier 2 -- returning None (Tier 1 result will get defaults)")
        return None, None

    model_id = getattr(llm, "model_id", None) or getattr(llm, "model", None)

    payload = _build_classify_payload(row, truncate_length)
    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=json.dumps(payload)),
    ]

    try:
        if hasattr(llm, "chat_async"):
            text = await llm.chat_async(messages=messages, max_tokens=cfg.enrichment_max_tokens, temperature=0.1)
        else:
            result = await asyncio.to_thread(
                llm.chat, messages=messages, max_tokens=cfg.enrichment_max_tokens, temperature=0.1,
            )
            text = result.get("response", "").strip()

        if not text:
            return None, model_id
        text = clean_llm_output(text)
        from ...pipelines.llm import parse_json_response
        parsed = parse_json_response(text, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            return parsed, model_id
        return None, model_id
    except (json.JSONDecodeError, ValueError):
        logger.warning("Tier 2 cloud returned invalid JSON for review %s", row["id"])
        return None, model_id
    except Exception:
        logger.exception("Tier 2 cloud LLM call failed")
        return None, None


def _merge_tier1_tier2(tier1: dict, tier2: dict | None) -> dict:
    """Merge Tier 1 (deterministic) + Tier 2 (interpretive) into a single 47-field JSONB.

    Tier 1 provides the base. Tier 2 keys are overlaid on top.
    competitors_mentioned is merged by name (case-insensitive).
    If tier2 is None (failed), apply safe defaults for Tier 2 fields.
    """
    result = dict(tier1)

    if tier2 is None:
        # Tier 2 failed -- apply minimal defaults so validation passes
        signals = result.get("churn_signals", {})
        any_signal = any(signals.get(f) for f in (
            "intent_to_leave", "actively_evaluating", "migration_in_progress",
        ))
        result.setdefault("urgency_score", 5 if any_signal else 3)
        result.setdefault("pain_category", "other")
        result.setdefault("pain_categories", [{"category": "other", "severity": "primary"}])
        result.setdefault("sentiment_trajectory", {"direction": "unknown"})
        result.setdefault("buyer_authority", {"role_type": "unknown", "buying_stage": "unknown",
                                              "has_budget_authority": False, "executive_sponsor_mentioned": False})
        result.setdefault("timeline", {"decision_timeline": "unknown"})
        result.setdefault("contract_context", {"contract_value_signal": "unknown"})
        result.setdefault("insider_signals", None)
        result.setdefault("would_recommend", None)
        result.setdefault("positive_aspects", [])
        result.setdefault("feature_gaps", [])
        # Leave competitors_mentioned as-is from Tier 1 (partial data)
        for comp in result.get("competitors_mentioned", []):
            comp.setdefault("evidence_type", "neutral_mention")
            comp.setdefault("displacement_confidence", "low")
            comp.setdefault("reason_category", None)
        return result

    # --- Tier 2 succeeded: overlay interpretive fields ---
    _TIER2_TOP_LEVEL_KEYS = {
        "urgency_score", "pain_category", "pain_categories",
        "sentiment_trajectory", "buyer_authority", "timeline",
        "contract_context", "insider_signals",
        "would_recommend", "positive_aspects", "feature_gaps",
    }
    for key in _TIER2_TOP_LEVEL_KEYS:
        if key in tier2:
            result[key] = tier2[key]

    # Merge competitors_mentioned by name (case-insensitive)
    tier1_comps = {c["name"].lower(): c for c in result.get("competitors_mentioned", []) if isinstance(c, dict) and "name" in c}
    tier2_comps = tier2.get("competitors_mentioned", []) or []

    merged_comps = []
    seen = set()
    for t2_comp in tier2_comps:
        if not isinstance(t2_comp, dict) or "name" not in t2_comp:
            continue
        key = t2_comp["name"].lower()
        seen.add(key)
        base = dict(tier1_comps.get(key, {"name": t2_comp["name"]}))
        # Overlay Tier 2 fields
        for field in ("evidence_type", "displacement_confidence", "reason_category"):
            if field in t2_comp:
                base[field] = t2_comp[field]
        # Ensure name comes from Tier 1 if available (preserves original casing)
        if key in tier1_comps:
            base["name"] = tier1_comps[key]["name"]
        merged_comps.append(base)

    # Append Tier 1 competitors not in Tier 2 (with defaults)
    for key, t1_comp in tier1_comps.items():
        if key not in seen:
            t1_comp.setdefault("evidence_type", "neutral_mention")
            t1_comp.setdefault("displacement_confidence", "low")
            t1_comp.setdefault("reason_category", None)
            merged_comps.append(t1_comp)

    result["competitors_mentioned"] = merged_comps
    return result


async def _enrich_hybrid(row, cfg, truncate_length: int) -> tuple[dict | None, dict | None, str | None]:
    """Run Tier 1 + Tier 2 extraction in parallel.

    Returns (tier1_result, tier2_result, tier2_model_id).
    """
    payload = _build_classify_payload(row, truncate_length)
    payload_json = json.dumps(payload)

    client = _get_tier1_client(cfg)

    tier1_task = _call_vllm_tier1(payload_json, cfg, client)
    tier2_task = _call_cloud_tier2(row, cfg, truncate_length)

    results = await asyncio.gather(tier1_task, tier2_task, return_exceptions=True)

    # Unpack Tier 1
    if isinstance(results[0], Exception):
        logger.error("Tier 1 raised: %s", results[0])
        tier1, tier1_model = None, None
    else:
        tier1, tier1_model = results[0]

    # Unpack Tier 2
    if isinstance(results[1], Exception):
        logger.error("Tier 2 raised: %s", results[1])
        tier2, tier2_model = None, None
    else:
        tier2, tier2_model = results[1]

    return tier1, tier2, tier2_model


async def _notify_high_urgency(
    vendor_name: str,
    reviewer_company: str,
    urgency: float,
    pain_category: str,
    intent_to_leave: bool,
) -> None:
    """Send ntfy push when a newly enriched review exceeds the urgency threshold."""
    if not settings.alerts.ntfy_enabled:
        return

    import httpx

    url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    company_part = f" at {reviewer_company}" if reviewer_company else ""
    intent_part = " | Intent to leave" if intent_to_leave else ""
    pain_part = f" | Pain: {pain_category}" if pain_category else ""

    message = (
        f"Urgency {urgency:.0f}/10{company_part}\n"
        f"Vendor: {vendor_name}{pain_part}{intent_part}"
    )

    headers: dict[str, str] = {
        "Title": f"High-Urgency Signal: {vendor_name}",
        "Priority": "high",
        "Tags": "rotating_light,b2b,churn",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("ntfy high-urgency alert sent for %s (urgency=%s)", vendor_name, urgency)
    except Exception as exc:
        logger.warning("ntfy high-urgency alert failed for %s: %s", vendor_name, exc)


async def enrich_batch(batch_id: str) -> dict[str, Any]:
    """Enrich all pending reviews from a specific import batch immediately.

    Called inline after scrape insertion so reviews are enriched on arrival
    rather than waiting for the scheduler.
    """
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"skipped": "B2B churn pipeline disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"skipped": "DB not ready"}

    max_attempts = cfg.enrichment_max_attempts

    rows = await pool.fetch(
        """
        WITH batch AS (
            SELECT id
            FROM b2b_reviews
            WHERE import_batch_id = $1
              AND enrichment_status = 'pending'
              AND enrichment_attempts < $2
            FOR UPDATE SKIP LOCKED
        )
        UPDATE b2b_reviews r
        SET enrichment_status = 'enriching'
        FROM batch
        WHERE r.id = batch.id
        RETURNING r.id, r.vendor_name, r.product_name, r.product_category,
                  r.source, r.raw_metadata,
                  r.rating, r.rating_max, r.summary, r.review_text, r.pros, r.cons,
                  r.reviewer_title, r.reviewer_company, r.company_size_raw,
                  r.reviewer_industry, r.enrichment_attempts, r.content_type
        """,
        batch_id,
        max_attempts,
    )

    if not rows:
        return {"total": 0, "enriched": 0, "failed": 0}

    return await _enrich_rows(rows, cfg, pool)


async def _enrich_rows(rows, cfg, pool) -> dict[str, Any]:
    """Enrich a list of claimed rows concurrently."""
    max_attempts = cfg.enrichment_max_attempts
    enriched = 0
    failed = 0

    sem = asyncio.Semaphore(30)

    async def _bounded_enrich(row):
        async with sem:
            return await _enrich_single(pool, row, max_attempts, cfg.enrichment_local_only,
                                        cfg.enrichment_max_tokens, cfg.review_truncate_length)

    results = await asyncio.gather(
        *[_bounded_enrich(row) for row in rows],
        return_exceptions=True,
    )

    for row, result in zip(rows, results):
        if isinstance(result, Exception):
            logger.error("Unexpected enrichment error for %s: %s", row["id"], result, exc_info=result)
            if (row["enrichment_attempts"] + 1) >= max_attempts:
                failed += 1
        elif result:
            enriched += 1
        elif (row["enrichment_attempts"] + 1) >= max_attempts:
            failed += 1

    # Count how many were triaged out in this batch
    no_signal = await pool.fetchval(
        "SELECT count(*) FROM b2b_reviews WHERE id = ANY($1::uuid[]) AND enrichment_status = 'no_signal'",
        [row["id"] for row in rows],
    )

    logger.info(
        "B2B enrichment: %d enriched, %d no_signal, %d failed (of %d)",
        enriched, no_signal or 0, failed, len(rows),
    )

    return {
        "total": len(rows),
        "enriched": enriched,
        "no_signal": no_signal or 0,
        "failed": failed,
    }


async def _queue_version_upgrades(pool) -> int:
    """Reset enrichment_status to 'pending' for reviews scraped with outdated parser versions.

    Compares each review's parser_version against the currently registered
    parser version.  Reviews with older versions are re-queued for enrichment.
    Returns the number of reviews re-queued.
    """
    if not settings.b2b_churn.enrichment_auto_requeue_parser_upgrades:
        logger.info("Parser-version auto requeue disabled; skipping version-upgrade scan")
        return 0
    try:
        from ...services.scraping.parsers import get_all_parsers

        parsers = get_all_parsers()
        if not parsers:
            return 0

        total_requeued = 0
        for source_name, parser in parsers.items():
            current_version = getattr(parser, "version", None)
            if not current_version:
                continue

            # Find enriched reviews with an older parser version
            count = await pool.fetchval(
                """
                WITH updated AS (
                    UPDATE b2b_reviews
                    SET enrichment_status = 'pending',
                        enrichment_attempts = 0
                    WHERE source = $1
                      AND parser_version IS NOT NULL
                      AND parser_version != $2
                      AND enrichment_status IN ('enriched', 'no_signal')
                    RETURNING 1
                )
                SELECT count(*) FROM updated
                """,
                source_name,
                current_version,
            )
            if count and count > 0:
                logger.info(
                    "Re-queued %d %s reviews for re-enrichment (parser %s -> %s)",
                    count, source_name, "old", current_version,
                )
                total_requeued += count

        return total_requeued
    except Exception:
        logger.debug("Version upgrade check skipped", exc_info=True)
        return 0


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: enrich pending B2B reviews (fallback for anything missed)."""
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B churn pipeline disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Auto re-process reviews scraped with outdated parser versions
    requeued = await _queue_version_upgrades(pool)

    max_batch = min(cfg.enrichment_max_per_batch, 500)
    max_attempts = cfg.enrichment_max_attempts

    total_enriched = 0
    total_failed = 0
    total_no_signal = 0
    rounds = 0

    while True:
        rows = await pool.fetch(
            """
            WITH batch AS (
                SELECT id
                FROM b2b_reviews
                WHERE enrichment_status = 'pending'
                  AND enrichment_attempts < $1
                ORDER BY imported_at ASC
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            UPDATE b2b_reviews r
            SET enrichment_status = 'enriching'
            FROM batch
            WHERE r.id = batch.id
            RETURNING r.id, r.vendor_name, r.product_name, r.product_category,
                      r.source, r.raw_metadata,
                      r.rating, r.rating_max, r.summary, r.review_text, r.pros, r.cons,
                      r.reviewer_title, r.reviewer_company, r.company_size_raw,
                      r.reviewer_industry, r.enrichment_attempts, r.content_type
            """,
            max_attempts,
            max_batch,
        )

        if not rows:
            break

        result = await _enrich_rows(rows, cfg, pool)
        total_enriched += result.get("enriched", 0)
        batch_failed = result.get("failed", 0)
        total_failed += batch_failed
        total_no_signal += result.get("no_signal", 0)
        rounds += 1

        # If most of the batch failed, vLLM is likely overwhelmed -- stop the loop
        if batch_failed > len(rows) * 0.5:
            logger.warning("B2B enrichment: >50%% failures in batch (%d/%d), stopping loop",
                           batch_failed, len(rows))
            break

        await asyncio.sleep(2)  # breathing room between batches

    if rounds == 0:
        return {"_skip_synthesis": "No B2B reviews to enrich"}

    result = {
        "enriched": total_enriched,
        "failed": total_failed,
        "no_signal": total_no_signal,
        "rounds": rounds,
        "_skip_synthesis": "B2B enrichment complete",
    }
    if requeued:
        result["version_upgrade_requeued"] = requeued
    return result


_MIN_REVIEW_TEXT_LENGTH = 80  # Skip LLM calls for reviews shorter than this

# Verified review platforms -- every review gets full extraction (skip triage).


async def _enrich_single(pool, row, max_attempts: int, local_only: bool,
                         max_tokens: int, truncate_length: int = 3000) -> bool:
    """Enrich a single B2B review with churn signals. Returns True on success."""
    review_id = row["id"]

    # Skip reviews with insufficient text -- title-only scrapes can't yield 47 fields
    review_text = row.get("review_text") or ""
    if len(review_text) < _MIN_REVIEW_TEXT_LENGTH:
        await pool.execute(
            "UPDATE b2b_reviews SET enrichment_status = 'not_applicable' WHERE id = $1",
            review_id,
        )
        return False

    source = (row.get("source") or "").lower().strip()
    is_verified = source in _VERIFIED_SOURCES

    try:
        if is_verified:
            # Verified sources: skip triage, go straight to full extraction.
            # Every review on these platforms has signal worth extracting.
            logger.debug("Verified source %s for %s -- skipping triage", source, review_id)
        else:
            # Stage 1: Fast triage -- is this review worth full extraction?
            triage, triage_model_id = await asyncio.wait_for(
                _triage_review(row, local_only),
                timeout=30,
            )

            if triage is None:
                # Triage failed (no LLM, skill missing, parse error).
                # Don't fall through to full extraction -- that will likely fail too.
                # Leave as pending for retry on next cycle.
                logger.debug("Triage returned None for %s, deferring to next cycle", review_id)
                await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
                return False

            if not triage.get("signal", True):
                # No churn signal -- skip full extraction
                await pool.execute(
                    """
                    UPDATE b2b_reviews
                    SET enrichment_status = 'no_signal',
                        enrichment_attempts = enrichment_attempts + 1,
                        enrichment = $1,
                        enrichment_model = $3
                    WHERE id = $2
                    """,
                    json.dumps({"triage": triage}),
                    review_id,
                    triage_model_id,
                )
                return False

        # Stage 2: Full 47-field extraction (only for reviews that passed triage)
        cfg = settings.b2b_churn
        if cfg.enrichment_hybrid_enabled:
            tier1, tier2, tier2_model_id = await asyncio.wait_for(
                _enrich_hybrid(row, cfg, truncate_length),
                timeout=120,
            )
            if tier1 is not None:
                result = _merge_tier1_tier2(tier1, tier2)
                model_id = f"hybrid:{cfg.enrichment_tier1_model}+{tier2_model_id or 'none'}"
            else:
                # Tier 1 failed -- fall back to single-pass
                result, model_id = await asyncio.wait_for(
                    _classify_review_async(row, local_only, max_tokens, truncate_length),
                    timeout=120,
                )
        else:
            result, model_id = await asyncio.wait_for(
                _classify_review_async(row, local_only, max_tokens, truncate_length),
                timeout=120,
            )

        if result and _validate_enrichment(result, row):
            # Extract sentiment_trajectory subfields for indexed columns
            st = result.get("sentiment_trajectory") or {}
            st_direction = st.get("direction") if isinstance(st, dict) else None
            st_tenure = st.get("tenure") if isinstance(st, dict) else None
            st_turning = st.get("turning_point") if isinstance(st, dict) else None

            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment = $1,
                    enrichment_status = 'enriched',
                    enrichment_attempts = enrichment_attempts + 1,
                    enriched_at = $2,
                    enrichment_model = $4,
                    sentiment_direction = $5,
                    sentiment_tenure = $6,
                    sentiment_turning_point = $7
                WHERE id = $3
                """,
                json.dumps(result),
                datetime.now(timezone.utc),
                review_id,
                model_id,
                st_direction,
                st_tenure,
                st_turning if st_turning and st_turning != "null" else None,
            )

            # Fire ntfy notification for high-urgency signals (must never
            # break enrichment -- wrapped in its own try/except)
            try:
                urgency = result.get("urgency_score", 0)
                threshold = settings.b2b_churn.high_churn_urgency_threshold
                if urgency >= threshold:
                    signals = result.get("churn_signals", {})
                    await _notify_high_urgency(
                        vendor_name=row["vendor_name"],
                        reviewer_company=row.get("reviewer_company") or "",
                        urgency=urgency,
                        pain_category=result.get("pain_category", ""),
                        intent_to_leave=bool(signals.get("intent_to_leave")),
                    )
            except Exception:
                logger.warning("ntfy notification failed for review %s, enrichment preserved", review_id)

            # Backfill reviewer_company from LLM extraction when parser left it empty
            try:
                _ctx = result.get("reviewer_context") or {}
                _extracted_company = (_ctx.get("company_name") or "").strip()
                if _extracted_company and not (row.get("reviewer_company") or "").strip():
                    _extracted_company_norm = normalize_company_name(_extracted_company) or None
                    await pool.execute(
                        "UPDATE b2b_reviews SET reviewer_company = $1, reviewer_company_norm = $2 WHERE id = $3",
                        _extracted_company,
                        _extracted_company_norm,
                        review_id,
                    )
            except Exception:
                logger.debug("Company name backfill failed for %s (non-fatal)", review_id)

            return True
        else:
            await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
            return False

    except Exception:
        logger.exception("Failed to enrich B2B review %s", review_id)
        try:
            # Reset from 'enriching' back to 'pending' (or 'failed' if exhausted)
            new_status = "failed" if (row["enrichment_attempts"] + 1) >= max_attempts else "pending"
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment_attempts = enrichment_attempts + 1,
                    enrichment_status = $1
                WHERE id = $2
                """,
                new_status, review_id,
            )
        except Exception:
            pass
        return False


def _smart_truncate(text: str, max_len: int = 3000) -> str:
    """Truncate preserving both beginning and end of review text.

    Churn signals often appear at the end ("I'm switching to X next quarter"),
    so naive head-only truncation loses them.
    """
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 15
    return text[:half] + "\n[...truncated...]\n" + text[-half:]


def _build_classify_payload(row, truncate_length: int = 3000) -> dict[str, Any]:
    """Build the JSON payload for the churn extraction skill."""
    review_text = _smart_truncate(row["review_text"] or "", max_len=truncate_length)

    raw_meta = row.get("raw_metadata") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            raw_meta = {}

    return {
        "vendor_name": row["vendor_name"],
        "product_name": row["product_name"] or "",
        "product_category": row["product_category"] or "",
        "source_name": row.get("source") or "",
        "source_weight": raw_meta.get("source_weight", 0.7),
        "source_type": raw_meta.get("source_type", "unknown"),
        "content_type": row.get("content_type") or "review",
        "rating": float(row["rating"]) if row["rating"] is not None else None,
        "rating_max": int(row["rating_max"]),
        "summary": row["summary"] or "",
        "review_text": review_text,
        "pros": row["pros"] or "",
        "cons": row["cons"] or "",
        "reviewer_title": row["reviewer_title"] or "",
        "reviewer_company": row["reviewer_company"] or "",
        "company_size_raw": row["company_size_raw"] or "",
        "reviewer_industry": row["reviewer_industry"] or "",
    }


async def _triage_review(row, local_only: bool) -> tuple[dict[str, Any] | None, str | None]:
    """Fast LLM triage: does this review contain churn signals worth extracting?

    Returns ({"signal": bool, "confidence": int, "reason": str}, model_id) or (None, None) on error.
    ~50 tokens output -- runs 5-10x faster than full extraction.
    """
    from ...skills import get_skill_registry
    from ...services.protocols import Message
    from ...pipelines.llm import get_pipeline_llm, clean_llm_output

    skill = get_skill_registry().get("digest/b2b_churn_triage")
    if not skill:
        # Triage skill missing -- fall through to full extraction
        return None, None

    # Use same dedicated OpenRouter instance as extraction
    llm = None
    if not local_only:
        llm = _get_enrichment_llm()
    if llm is None:
        llm = get_pipeline_llm(prefer_cloud=False, try_openrouter=False, auto_activate_ollama=True)
    if llm is None:
        return None, None

    model_id = getattr(llm, "model_id", None) or getattr(llm, "model", None)

    # Compact payload -- triage doesn't need all fields
    review_text = (row.get("review_text") or "")[:1500]
    payload = json.dumps({
        "vendor_name": row["vendor_name"],
        "source": row.get("source") or "",
        "content_type": row.get("content_type") or "review",
        "rating": float(row["rating"]) if row.get("rating") is not None else None,
        "summary": row.get("summary") or "",
        "review_text": review_text,
        "pros": (row.get("pros") or "")[:300],
        "cons": (row.get("cons") or "")[:300],
    })

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=payload),
    ]

    try:
        if hasattr(llm, "chat_async"):
            text = await llm.chat_async(messages=messages, max_tokens=64, temperature=0.0)
        else:
            result = await asyncio.to_thread(
                llm.chat, messages=messages, max_tokens=64, temperature=0.0,
            )
            text = result.get("response", "").strip()

        if not text:
            return None, model_id
        text = clean_llm_output(text)
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "signal" in parsed:
            return parsed, model_id
    except (json.JSONDecodeError, Exception):
        pass
    # On any failure, return None (fall through to full extraction -- safe default)
    return None, model_id


async def _classify_review_async(row, local_only: bool, max_tokens: int,
                                  truncate_length: int = 3000) -> tuple[dict[str, Any] | None, str | None]:
    """Async LLM call -- no thread pool, pure async for maximum vLLM throughput.

    Returns (enrichment_dict, model_id) tuple.
    """
    from ...skills import get_skill_registry
    from ...services.protocols import Message
    from ...pipelines.llm import get_pipeline_llm, clean_llm_output, parse_json_response

    skill = get_skill_registry().get("digest/b2b_churn_extraction")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction' not found")
        return None, None

    # Primary: Dedicated OpenRouter instance (not shared registry singleton)
    # Fallback: local vLLM/Ollama
    llm = None
    if not local_only:
        llm = _get_enrichment_llm()
    if llm is None:
        llm = get_pipeline_llm(
            prefer_cloud=False,
            try_openrouter=False,
            auto_activate_ollama=True,
        )
    if llm is None:
        logger.warning("No LLM available for B2B churn extraction")
        return None, None

    model_id = getattr(llm, "model_id", None) or getattr(llm, "model", None)

    payload = _build_classify_payload(row, truncate_length)
    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=json.dumps(payload)),
    ]

    try:
        # Use async chat if available (vLLM, httpx-based backends)
        if hasattr(llm, "chat_async"):
            text = await llm.chat_async(messages=messages, max_tokens=max_tokens, temperature=0.1)
        else:
            # Fallback to sync in thread for backends without async
            result = await asyncio.to_thread(
                llm.chat, messages=messages, max_tokens=max_tokens, temperature=0.1
            )
            text = result.get("response", "").strip()

        if not text:
            return None, model_id
        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)
        if not isinstance(parsed, dict) or parsed.get("_parse_fallback"):
            return None, model_id
        return parsed, model_id
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "Failed to parse B2B enrichment JSON for review %s (vendor=%s)",
            row["id"], row["vendor_name"],
        )
        return None, model_id
    except Exception:
        logger.exception("B2B enrichment LLM call failed")
        return None, None


_KNOWN_PAIN_CATEGORIES = {
    "pricing", "features", "reliability", "support", "integration",
    "performance", "security", "ux", "onboarding", "other",
}


def _coerce_bool(value: Any) -> bool | None:
    """Coerce a value to bool. Returns None if unrecognizable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
    return None


_CHURN_SIGNAL_BOOL_FIELDS = (
    "intent_to_leave",
    "actively_evaluating",
    "migration_in_progress",
    "support_escalation",
    "contract_renewal_mentioned",
)

_KNOWN_SEVERITY_LEVELS = {"primary", "secondary", "minor"}
_KNOWN_LOCK_IN_LEVELS = {"high", "medium", "low", "unknown"}
_KNOWN_SENTIMENT_DIRECTIONS = {"declining", "consistently_negative", "improving", "stable_positive", "unknown"}
_KNOWN_ROLE_TYPES = {"economic_buyer", "champion", "evaluator", "end_user", "unknown"}
_KNOWN_ROLE_LEVELS = {"executive", "director", "manager", "ic", "unknown"}
_KNOWN_BUYING_STAGES = {"active_purchase", "evaluation", "renewal_decision", "post_purchase", "unknown"}
_KNOWN_DECISION_TIMELINES = {"immediate", "within_quarter", "within_year", "unknown"}
_KNOWN_CONTRACT_VALUE_SIGNALS = {"enterprise_high", "enterprise_mid", "mid_market", "smb", "unknown"}

# Insider signal validation sets (migration 133)
_KNOWN_CONTENT_TYPES = {"review", "community_discussion", "comment", "insider_account"}
_KNOWN_ORG_HEALTH_LEVELS = {"high", "medium", "low", "unknown"}
_KNOWN_LEADERSHIP_QUALITIES = {"poor", "mixed", "good", "unknown"}
_KNOWN_INNOVATION_CLIMATES = {"stagnant", "declining", "healthy", "unknown"}
_KNOWN_MORALE_LEVELS = {"high", "medium", "low", "unknown"}
_KNOWN_DEPARTURE_TYPES = {"voluntary", "involuntary", "still_employed", "unknown"}

_ROLE_TYPE_ALIASES = {
    "economicbuyer": "economic_buyer",
    "decisionmaker": "economic_buyer",
    "buyer": "economic_buyer",
    "budgetowner": "economic_buyer",
    "executive": "economic_buyer",
    "director": "economic_buyer",
    "champion": "champion",
    "manager": "champion",
    "teamlead": "champion",
    "lead": "champion",
    "evaluator": "evaluator",
    "admin": "evaluator",
    "administrator": "evaluator",
    "analyst": "evaluator",
    "architect": "evaluator",
    "enduser": "end_user",
    "user": "end_user",
    "ic": "end_user",
    "individualcontributor": "end_user",
    "unknown": "unknown",
}

_NOISY_REVIEWER_TITLE_PATTERNS = (
    re.compile(r"^repeat churn signal", re.I),
    re.compile(r"score:\s*\d", re.I),
)
_EXEC_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(vp\b|vice president|director|head of|chief|cfo|ceo|coo|cio|cto|cro|cmo|founder|owner|president)\b",
    re.I,
)
_CHAMPION_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(manager|lead|supervisor|coordinator)\b",
    re.I,
)
_EVALUATOR_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(analyst|architect|engineer|developer|administrator|admin|consultant|specialist)\b",
    re.I,
)
_EXEC_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(ceo|cto|cfo|cio|coo|cmo|cro|chief|founder|owner|president)\b",
    re.I,
)
_DIRECTOR_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(vp|vice president|svp|evp|director|head of)\b",
    re.I,
)
_MANAGER_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(manager|team lead|lead|supervisor|coordinator)\b",
    re.I,
)
_IC_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(engineer|developer|administrator|admin|analyst|specialist|"
    r"consultant|marketer|designer|architect)\b",
    re.I,
)
_ECONOMIC_BUYER_TEXT_PATTERNS = (
    re.compile(
        r"\b(we|i) decided to (switch|move|migrate|leave|replace|renew|buy|adopt|go with)\b",
        re.I,
    ),
    re.compile(r"\bapproved (the )?(purchase|renewal|budget)\b", re.I),
    re.compile(r"\bsigned off on (the )?(purchase|renewal|budget|migration)\b", re.I),
    re.compile(r"\bfinal decision (was|is) to\b", re.I),
)
_CHAMPION_TEXT_PATTERNS = (
    re.compile(r"\b(i|we) recommended\b", re.I),
    re.compile(r"\bchampioned\b", re.I),
    re.compile(r"\bpushed for\b", re.I),
    re.compile(r"\badvocated for\b", re.I),
)
_EVALUATOR_TEXT_PATTERNS = (
    re.compile(r"\bevaluating alternatives\b", re.I),
    re.compile(r"\bcomparing options\b", re.I),
    re.compile(r"\bproof of concept\b", re.I),
    re.compile(r"\bpoc\b", re.I),
    re.compile(r"\bshortlist\b", re.I),
    re.compile(r"\btrialing\b", re.I),
    re.compile(r"\bpiloting\b", re.I),
    re.compile(r"\btasked with evaluating\b", re.I),
)
_END_USER_TEXT_PATTERNS = (
    re.compile(r"\bi use\b", re.I),
    re.compile(r"\bwe use\b", re.I),
    re.compile(r"\bday-to-day\b", re.I),
    re.compile(r"\bdaily use\b", re.I),
    re.compile(r"\buse it for\b", re.I),
)


def _canonical_role_type(value: Any) -> str:
    raw = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    if not raw:
        return "unknown"
    return _ROLE_TYPE_ALIASES.get(raw, "unknown")


def _clean_reviewer_title_for_role_inference(value: Any) -> str:
    title = str(value or "").strip()
    if not title or len(title) > 120:
        return ""
    lowered = title.lower()
    if any(pattern.search(lowered) for pattern in _NOISY_REVIEWER_TITLE_PATTERNS):
        return ""
    return title


def _canonical_role_level(value: Any) -> str:
    raw = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    if raw in {"executive", "exec", "csuite", "cxo"}:
        return "executive"
    if raw in {"director", "vp", "vicepresident", "head"}:
        return "director"
    if raw in {"manager", "lead", "teamlead", "supervisor", "coordinator"}:
        return "manager"
    if raw in {"ic", "individualcontributor", "individual", "user"}:
        return "ic"
    return "unknown"


def _combined_source_text(source_row: dict[str, Any] | None) -> str:
    if not isinstance(source_row, dict):
        return ""
    parts = [
        str(source_row.get("summary") or ""),
        str(source_row.get("review_text") or ""),
        str(source_row.get("pros") or ""),
        str(source_row.get("cons") or ""),
    ]
    return "\n".join(part for part in parts if part).strip()


def _infer_role_level_from_text(reviewer_title: Any, source_row: dict[str, Any] | None) -> str:
    title = _clean_reviewer_title_for_role_inference(reviewer_title)
    if title:
        if re.search(r"\b(cfo|ceo|coo|cio|cto|cro|cmo|chief|founder|owner|president)\b", title, re.I):
            return "executive"
        if re.search(r"\b(vp\b|vice president|svp|evp|director|head of)\b", title, re.I):
            return "director"
        if _CHAMPION_REVIEWER_TITLE_PATTERN.search(title):
            return "manager"
        if _EVALUATOR_REVIEWER_TITLE_PATTERN.search(title):
            return "ic"
    source_text = _combined_source_text(source_row)
    if not source_text:
        return "unknown"
    if _EXEC_ROLE_TEXT_PATTERN.search(source_text):
        return "executive"
    if _DIRECTOR_ROLE_TEXT_PATTERN.search(source_text):
        return "director"
    if _MANAGER_ROLE_TEXT_PATTERN.search(source_text):
        return "manager"
    if _IC_ROLE_TEXT_PATTERN.search(source_text):
        return "ic"
    return "unknown"


def _infer_buyer_role_type_from_text(
    buyer_authority: dict[str, Any],
    source_row: dict[str, Any] | None,
) -> str:
    if not isinstance(source_row, dict):
        return "unknown"
    if str(source_row.get("content_type") or "").strip().lower() == "insider_account":
        return "unknown"
    source_text = _combined_source_text(source_row)
    if not source_text:
        return "unknown"
    for pattern in _ECONOMIC_BUYER_TEXT_PATTERNS:
        if pattern.search(source_text):
            return "economic_buyer"
    for pattern in _CHAMPION_TEXT_PATTERNS:
        if pattern.search(source_text):
            return "champion"
    for pattern in _EVALUATOR_TEXT_PATTERNS:
        if pattern.search(source_text):
            return "evaluator"
    buying_stage = str(buyer_authority.get("buying_stage") or "").strip().lower()
    for pattern in _END_USER_TEXT_PATTERNS:
        if pattern.search(source_text):
            return "evaluator" if buying_stage in {"evaluation", "active_purchase"} else "end_user"
    return "unknown"


def _infer_buyer_role_type(
    buyer_authority: dict[str, Any],
    reviewer_context: dict[str, Any] | None,
    reviewer_title: Any,
    source_row: dict[str, Any] | None = None,
) -> str:
    ctx = reviewer_context if isinstance(reviewer_context, dict) else {}
    role_level = str(ctx.get("role_level") or "").strip().lower()
    buying_stage = str(buyer_authority.get("buying_stage") or "").strip().lower()
    if _coerce_bool(buyer_authority.get("has_budget_authority")) is True:
        return "economic_buyer"
    if _coerce_bool(ctx.get("decision_maker")) is True:
        return "economic_buyer"
    if role_level in {"executive", "director"}:
        return "economic_buyer"
    title = _clean_reviewer_title_for_role_inference(reviewer_title)
    if title and _EXEC_REVIEWER_TITLE_PATTERN.search(title):
        return "economic_buyer"
    if role_level == "manager":
        return "champion"
    if title and _CHAMPION_REVIEWER_TITLE_PATTERN.search(title):
        return "champion"
    if role_level == "ic" and buying_stage in {"evaluation", "active_purchase"}:
        return "evaluator"
    if title and _EVALUATOR_REVIEWER_TITLE_PATTERN.search(title):
        return "evaluator" if buying_stage in {"evaluation", "active_purchase"} else "end_user"
    if role_level == "ic":
        return "end_user"
    return _infer_buyer_role_type_from_text(buyer_authority, source_row)


def _validate_enrichment(result: dict, source_row: dict[str, Any] | None = None) -> bool:
    """Validate enrichment output structure and data consistency."""
    if "churn_signals" not in result:
        return False
    if "urgency_score" not in result:
        return False
    if not isinstance(result.get("churn_signals"), dict):
        return False

    # Type check: urgency_score must be numeric
    urgency = result.get("urgency_score")
    if isinstance(urgency, str):
        try:
            urgency = float(urgency)
            result["urgency_score"] = urgency
        except (ValueError, TypeError):
            logger.warning("urgency_score is non-numeric string: %r", urgency)
            return False

    if not isinstance(urgency, (int, float)):
        logger.warning("urgency_score has unexpected type: %s", type(urgency).__name__)
        return False

    # Range check: 0-10
    if urgency < 0 or urgency > 10:
        logger.warning("urgency_score out of range [0,10]: %s", urgency)
        return False

    # Boolean coercion: churn_signals fields used in ::boolean casts
    signals = result["churn_signals"]
    for field in _CHURN_SIGNAL_BOOL_FIELDS:
        if field in signals:
            coerced = _coerce_bool(signals[field])
            if coerced is None:
                logger.warning("churn_signals.%s unrecognizable bool: %r -- rejecting", field, signals[field])
                return False
            signals[field] = coerced

    # Consistency warning: high urgency with no intent_to_leave
    intent = signals.get("intent_to_leave")
    if urgency >= 9 and intent is False:
        logger.warning(
            "Contradictory: urgency=%s but intent_to_leave=false -- accepting with warning",
            urgency,
        )

    # Boolean coercion: reviewer_context.decision_maker (used in ::boolean cast)
    reviewer_ctx = result.get("reviewer_context")
    if isinstance(reviewer_ctx, dict) and "decision_maker" in reviewer_ctx:
        coerced = _coerce_bool(reviewer_ctx["decision_maker"])
        if coerced is None:
            logger.warning("reviewer_context.decision_maker unrecognizable bool: %r -- rejecting", reviewer_ctx["decision_maker"])
            return False
        reviewer_ctx["decision_maker"] = coerced

    # Boolean coercion: would_recommend (used in ::boolean cast in vendor_churn_scores)
    if "would_recommend" in result:
        coerced = _coerce_bool(result["would_recommend"])
        if coerced is None:
            # null/None is valid (reviewer didn't express preference) -- keep as null
            result["would_recommend"] = None
        else:
            result["would_recommend"] = coerced

    # Type check: competitors_mentioned must be list; items must be dicts with "name"
    competitors = result.get("competitors_mentioned")
    if competitors is not None and not isinstance(competitors, list):
        logger.warning("competitors_mentioned is not a list: %s", type(competitors).__name__)
        result["competitors_mentioned"] = []
    elif isinstance(competitors, list):
        result["competitors_mentioned"] = [
            c for c in competitors
            if isinstance(c, dict) and "name" in c
        ]

    # Validate evidence_type, displacement_confidence, and reason_category on each competitor entry
    _VALID_EVIDENCE_TYPES = {"explicit_switch", "active_evaluation", "implied_preference", "reverse_flow", "neutral_mention"}
    _VALID_DISP_CONFIDENCE = {"high", "medium", "low", "none"}
    _VALID_REASON_CATEGORIES = {"pricing", "features", "reliability", "ux", "support", "integration"}
    _ET_TO_CONTEXT = {
        "explicit_switch": "switched_to",
        "active_evaluation": "considering",
        "implied_preference": "compared",
        "reverse_flow": "switched_from",
        "neutral_mention": "compared",
    }
    for comp in result.get("competitors_mentioned", []):
        # Coerce unknown evidence_type; fall back from legacy context field
        et = comp.get("evidence_type")
        if et not in _VALID_EVIDENCE_TYPES:
            # Map legacy context -> evidence_type
            legacy = comp.get("context", "")
            _CONTEXT_TO_ET = {
                "switched_to": "explicit_switch",
                "considering": "active_evaluation",
                "compared": "implied_preference",
                "switched_from": "reverse_flow",
            }
            comp["evidence_type"] = _CONTEXT_TO_ET.get(legacy, "neutral_mention")

        # Coerce unknown displacement_confidence
        dc = comp.get("displacement_confidence")
        if dc not in _VALID_DISP_CONFIDENCE:
            comp["displacement_confidence"] = "low"

        # Consistency: reverse_flow -> confidence "none"
        if comp["evidence_type"] == "reverse_flow":
            comp["displacement_confidence"] = "none"
        # Consistency: neutral_mention -> confidence at most "low"
        if comp["evidence_type"] == "neutral_mention" and comp.get("displacement_confidence") in ("high", "medium"):
            comp["displacement_confidence"] = "low"

        # Coerce reason_category to taxonomy
        rc = comp.get("reason_category")
        if rc and rc not in _VALID_REASON_CATEGORIES:
            comp["reason_category"] = None

        # Backward compat: populate context from evidence_type
        comp["context"] = _ET_TO_CONTEXT.get(comp["evidence_type"], "compared")

        # Backward compat: populate reason from reason_category + reason_detail
        rc = comp.get("reason_category")
        rd = comp.get("reason_detail")
        if rc and rd:
            comp["reason"] = f"{rc}: {rd}"
        elif rc:
            comp["reason"] = rc
        elif rd:
            comp["reason"] = rd
        # else: keep existing reason if any (legacy data)

    # Type check: quotable_phrases must be list if present
    qp = result.get("quotable_phrases")
    if qp is not None and not isinstance(qp, list):
        logger.warning("quotable_phrases is not a list: %s", type(qp).__name__)
        result["quotable_phrases"] = []

    # Type check: feature_gaps must be list if present
    fg = result.get("feature_gaps")
    if fg is not None and not isinstance(fg, list):
        logger.warning("feature_gaps is not a list: %s", type(fg).__name__)
        result["feature_gaps"] = []

    # Coerce unknown pain_category to "other"
    pain = result.get("pain_category")
    if pain and pain not in _KNOWN_PAIN_CATEGORIES:
        logger.warning("Unknown pain_category: %r -- coercing to 'other'", pain)
        result["pain_category"] = "other"

    # --- New expanded field validation (permissive: coerce, never reject) ---

    # pain_categories: list of {category, severity}
    pc = result.get("pain_categories")
    if pc is not None:
        if not isinstance(pc, list):
            result["pain_categories"] = []
        else:
            cleaned = []
            for item in pc:
                if not isinstance(item, dict):
                    continue
                cat = item.get("category", "other")
                if cat not in _KNOWN_PAIN_CATEGORIES:
                    cat = "other"
                sev = item.get("severity", "minor")
                if sev not in _KNOWN_SEVERITY_LEVELS:
                    sev = "minor"
                cleaned.append({"category": cat, "severity": sev})
            result["pain_categories"] = cleaned

    # budget_signals: dict with known keys
    bs = result.get("budget_signals")
    if bs is not None:
        if not isinstance(bs, dict):
            result["budget_signals"] = {}
        else:
            if "seat_count" in bs and bs["seat_count"] is not None:
                try:
                    seat = int(bs["seat_count"])
                    bs["seat_count"] = seat if 1 <= seat <= 1_000_000 else None
                except (ValueError, TypeError):
                    bs["seat_count"] = None
            if "price_increase_mentioned" in bs:
                coerced = _coerce_bool(bs["price_increase_mentioned"])
                bs["price_increase_mentioned"] = coerced if coerced is not None else False

    # use_case: dict with lists and lock_in_level
    uc = result.get("use_case")
    if uc is not None:
        if not isinstance(uc, dict):
            result["use_case"] = {}
        else:
            if "modules_mentioned" in uc and not isinstance(uc["modules_mentioned"], list):
                uc["modules_mentioned"] = []
            if "integration_stack" in uc and not isinstance(uc["integration_stack"], list):
                uc["integration_stack"] = []
            lil = uc.get("lock_in_level")
            if lil and lil not in _KNOWN_LOCK_IN_LEVELS:
                uc["lock_in_level"] = "unknown"

    # reviewer_context: normalize role level and backfill from title/text
    reviewer_ctx = result.get("reviewer_context")
    if reviewer_ctx is None or not isinstance(reviewer_ctx, dict):
        result["reviewer_context"] = {}
        reviewer_ctx = result["reviewer_context"]
    role_level = _canonical_role_level(reviewer_ctx.get("role_level"))
    if role_level == "unknown":
        role_level = _infer_role_level_from_text(
            (source_row or {}).get("reviewer_title"),
            source_row,
        )
    reviewer_ctx["role_level"] = role_level
    decision_maker = _coerce_bool(reviewer_ctx.get("decision_maker"))
    if decision_maker is None:
        decision_maker = role_level in {"executive", "director"}
    reviewer_ctx["decision_maker"] = decision_maker

    # sentiment_trajectory: dict with direction
    st = result.get("sentiment_trajectory")
    if st is not None:
        if not isinstance(st, dict):
            result["sentiment_trajectory"] = {}
        else:
            d = st.get("direction")
            if d and d not in _KNOWN_SENTIMENT_DIRECTIONS:
                st["direction"] = "unknown"

    # buyer_authority: dict with role_type, booleans, buying_stage
    ba = result.get("buyer_authority")
    if ba is not None:
        if not isinstance(ba, dict):
            result["buyer_authority"] = {}
            ba = result["buyer_authority"]
        reviewer_ctx = (
            result.get("reviewer_context")
            if isinstance(result.get("reviewer_context"), dict)
            else {}
        )
        for bool_field in ("has_budget_authority", "executive_sponsor_mentioned"):
            if bool_field in ba:
                coerced = _coerce_bool(ba[bool_field])
                ba[bool_field] = coerced if coerced is not None else False
        bstage = ba.get("buying_stage")
        if bstage and bstage not in _KNOWN_BUYING_STAGES:
            ba["buying_stage"] = "unknown"
        canonical_rt = _canonical_role_type(ba.get("role_type"))
        if canonical_rt == "unknown":
            ba["role_type"] = _infer_buyer_role_type(
                ba,
                reviewer_ctx,
                (source_row or {}).get("reviewer_title"),
                source_row,
            )
        else:
            ba["role_type"] = canonical_rt
        if ba["role_type"] == "economic_buyer":
            reviewer_ctx["decision_maker"] = True

    # timeline: dict with decision_timeline
    tl = result.get("timeline")
    if tl is not None:
        if not isinstance(tl, dict):
            result["timeline"] = {}
        else:
            dt = tl.get("decision_timeline")
            if dt and dt not in _KNOWN_DECISION_TIMELINES:
                tl["decision_timeline"] = "unknown"

    # contract_context: dict with contract_value_signal
    cc = result.get("contract_context")
    if cc is not None:
        if not isinstance(cc, dict):
            result["contract_context"] = {}
        else:
            cvs = cc.get("contract_value_signal")
            if cvs and cvs not in _KNOWN_CONTRACT_VALUE_SIGNALS:
                cc["contract_value_signal"] = "unknown"

    # content_classification: pass-through string, coerce to known values
    cc_val = result.get("content_classification")
    if cc_val and cc_val not in _KNOWN_CONTENT_TYPES:
        result["content_classification"] = "review"

    # insider_signals: validate structure if present
    insider = result.get("insider_signals")
    if insider is not None:
        if not isinstance(insider, dict):
            result["insider_signals"] = None
        else:
            # org_health: must be dict
            oh = insider.get("org_health")
            if oh is not None and not isinstance(oh, dict):
                insider["org_health"] = {}
            elif isinstance(oh, dict):
                # culture_indicators must be list
                ci = oh.get("culture_indicators")
                if ci is not None and not isinstance(ci, list):
                    oh["culture_indicators"] = []
                # Enum fields: coerce unknowns
                for field, allowed in (
                    ("bureaucracy_level", _KNOWN_ORG_HEALTH_LEVELS),
                    ("leadership_quality", _KNOWN_LEADERSHIP_QUALITIES),
                    ("innovation_climate", _KNOWN_INNOVATION_CLIMATES),
                ):
                    val = oh.get(field)
                    if val and val not in allowed:
                        oh[field] = "unknown"

            # talent_drain: must be dict
            td = insider.get("talent_drain")
            if td is not None and not isinstance(td, dict):
                insider["talent_drain"] = {}
            elif isinstance(td, dict):
                for bool_field in ("departures_mentioned", "layoff_fear"):
                    if bool_field in td:
                        coerced = _coerce_bool(td[bool_field])
                        td[bool_field] = coerced if coerced is not None else False
                morale = td.get("morale")
                if morale and morale not in _KNOWN_MORALE_LEVELS:
                    td["morale"] = "unknown"

            # departure_type: enum
            dt = insider.get("departure_type")
            if dt and dt not in _KNOWN_DEPARTURE_TYPES:
                insider["departure_type"] = "unknown"

    return True


async def _increment_attempts(pool, review_id, current_attempts: int, max_attempts: int) -> None:
    """Bump attempts atomically; reset to pending or mark failed if exhausted."""
    new_status = "failed" if (current_attempts + 1) >= max_attempts else "pending"
    await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_attempts = enrichment_attempts + 1,
            enrichment_status = $1
        WHERE id = $2
        """,
        new_status, review_id,
    )
