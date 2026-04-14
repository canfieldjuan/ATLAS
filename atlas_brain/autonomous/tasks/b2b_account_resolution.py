"""B2B account resolution: resolve anonymous reviewers to named companies.

Runs after enrichment. Reads enriched reviews that lack a resolution row,
runs the deterministic resolver, persists results to b2b_account_resolution,
and backfills reviewer_company on b2b_reviews for high/medium confidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...services.company_normalization import normalize_company_name
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_account_resolution")

_PROFILE_URL_TEMPLATES = {
    "hackernews": "https://news.ycombinator.com/user?id={username}",
    "github": "https://github.com/{username}",
    "reddit": "https://www.reddit.com/user/{username}",
}
_SELF_IDENT_TEXT_PATTERN = (
    r"( i work at | i work for | we work at | we work for | we use this at )"
)
_ACCOUNT_RESOLUTION_ALWAYS_SUPPORTED_SOURCES = frozenset({
    "g2",
    "gartner",
    "trustradius",
    "capterra",
    "software_advice",
    "peerspot",
    "sourceforge",
})


def _build_author_profile_url(source: str, username: str) -> str | None:
    template = _PROFILE_URL_TEMPLATES.get(source)
    if template and username:
        return template.format(username=username)
    return None


def _normalize_account_resolution_excluded_sources(raw: Any) -> list[str]:
    """Keep truly unsupported sources excluded while allowing structured B2B review sources."""
    if not raw:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(part or "").strip() for part in raw]
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        source = str(value or "").strip().lower()
        if not source or source in seen:
            continue
        seen.add(source)
        if source in _ACCOUNT_RESOLUTION_ALWAYS_SUPPORTED_SOURCES:
            continue
        normalized.append(source)
    return normalized


def _unique_usernames(rows: list[dict[str, Any]], limit: int) -> list[str]:
    """Return up to ``limit`` unique non-empty reviewer handles in row order."""
    seen: set[str] = set()
    usernames: list[str] = []
    for row in rows:
        username = str(row.get("reviewer_name") or "").strip()
        if not username or username in seen:
            continue
        seen.add(username)
        usernames.append(username)
        if len(usernames) >= limit:
            break
    return usernames


async def _backfill_review_company_context(
    pool: Any,
    *,
    review_id: Any,
    company_name: str,
    company_norm: str | None,
) -> bool:
    """Promote a resolved company into both raw review fields and enrichment reviewer context."""
    status = await pool.execute(
        """
        UPDATE b2b_reviews
        SET reviewer_company = CASE
                WHEN reviewer_company IS NULL OR reviewer_company = '' THEN $2
                ELSE reviewer_company
            END,
            reviewer_company_norm = CASE
                WHEN reviewer_company IS NULL OR reviewer_company = '' THEN $3
                ELSE reviewer_company_norm
            END,
            enrichment = CASE
                WHEN NULLIF(trim(coalesce(enrichment #>> '{reviewer_context,company_name}', '')), '') IS NULL
                    THEN jsonb_set(
                        COALESCE(enrichment, '{}'::jsonb),
                        '{reviewer_context,company_name}',
                        to_jsonb($2::text),
                        true
                    )
                ELSE enrichment
            END
        WHERE id = $1
          AND (
                reviewer_company IS NULL
             OR reviewer_company = ''
             OR NULLIF(trim(coalesce(enrichment #>> '{reviewer_context,company_name}', '')), '') IS NULL
          )
        """,
        review_id,
        company_name,
        company_norm,
    )
    try:
        return int(str(status).split()[-1]) > 0
    except Exception:
        return False


async def _exclude_unsupported_sources(
    pool: Any,
    *,
    eligible_statuses: list[str],
    excluded_sources: list[str],
) -> int:
    """Mark low-signal sources as excluded so they stop competing for resolution."""
    if not excluded_sources:
        return 0
    status = await pool.execute(
        """
        INSERT INTO b2b_account_resolution (
            review_id, source, source_item_url,
            author_handle, author_profile_url,
            reviewer_company_raw,
            resolved_company_name, normalized_company_name,
            confidence_score, confidence_label,
            resolution_method, resolution_evidence,
            resolution_status, resolved_at
        )
        SELECT
            r.id,
            r.source,
            r.source_url,
            r.reviewer_name,
            NULL,
            r.reviewer_company,
            NULL,
            NULL,
            0.0,
            'unresolved',
            'unsupported_source',
            jsonb_build_object('reason', 'unsupported_source', 'source', r.source),
            'excluded',
            NOW()
        FROM b2b_reviews r
        LEFT JOIN b2b_account_resolution ar ON ar.review_id = r.id
        WHERE r.enrichment_status = ANY($1::text[])
          AND r.duplicate_of_review_id IS NULL
          AND r.enrichment IS NOT NULL
          AND r.source = ANY($2::text[])
          AND ar.id IS NULL
        """,
        eligible_statuses,
        excluded_sources,
    )
    try:
        return int(str(status).split()[-1])
    except Exception:
        return 0


async def _propagate_user_resolutions(pool: Any, backfill_labels: set) -> int:
    """Propagate high/medium confidence resolutions to same-user unresolved reviews.

    If reviewer 'johndoe' has a high-confidence resolution for company 'Stripe'
    on one review, their other unresolved reviews on the same source are
    updated with the same company at slightly lower propagated confidence.
    Skips '[deleted]' accounts (shared username on Reddit deleted posts).
    """
    resolved_users = await pool.fetch("""
        SELECT DISTINCT ON (r.source, r.reviewer_name)
               r.source,
               r.reviewer_name,
               ar.resolved_company_name,
               ar.normalized_company_name,
               ar.confidence_score,
               ar.confidence_label,
               ar.resolution_method
        FROM b2b_reviews r
        JOIN b2b_account_resolution ar ON ar.review_id = r.id
        WHERE r.duplicate_of_review_id IS NULL
          AND ar.resolution_status = 'resolved'
          AND ar.confidence_label IN ('high', 'medium')
          AND r.reviewer_name IS NOT NULL
          AND r.reviewer_name NOT IN ('[deleted]', '', 'deleted')
        ORDER BY r.source, r.reviewer_name, ar.confidence_score DESC
    """)

    propagated = 0
    for user in resolved_users:
        unresolved = await pool.fetch("""
            SELECT r.id AS review_id, r.reviewer_company, ar.id AS ar_id
            FROM b2b_reviews r
            JOIN b2b_account_resolution ar ON ar.review_id = r.id
            WHERE r.duplicate_of_review_id IS NULL
              AND r.source = $1
              AND r.reviewer_name = $2
              AND ar.resolution_status = 'unresolved'
        """, user["source"], user["reviewer_name"])

        if not unresolved:
            continue

        prop_score = round(min(float(user["confidence_score"]) * 0.85, 0.75), 2)
        prop_label = "medium" if prop_score >= 0.6 else "low"
        evidence = json.dumps({
            "signals": [{
                "type": "username_propagation",
                "value": user["resolved_company_name"],
                "confidence": prop_score,
                "source_field": f"reviewer_name:{user['reviewer_name']}",
                "propagated_from_method": user["resolution_method"],
            }],
            "excluded_candidates": [],
        })

        for review in unresolved:
            try:
                await pool.execute("""
                    UPDATE b2b_account_resolution
                    SET resolved_company_name = $2,
                        normalized_company_name = $3,
                        confidence_score = $4,
                        confidence_label = $5,
                        resolution_method = 'username_propagation',
                        resolution_evidence = $6::jsonb,
                        resolution_status = 'resolved',
                        resolved_at = NOW()
                    WHERE id = $1
                """, review["ar_id"], user["resolved_company_name"],
                    user["normalized_company_name"],
                    prop_score, prop_label, evidence)
                propagated += 1

                if prop_label in backfill_labels and not (review["reviewer_company"] or "").strip():
                    await _backfill_review_company_context(
                        pool,
                        review_id=review["review_id"],
                        company_name=user["resolved_company_name"],
                        company_norm=user["normalized_company_name"],
                    )
            except Exception:
                logger.warning(
                    "Username propagation failed for %s", review["ar_id"], exc_info=True,
                )

    return propagated


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task: resolve anonymous review authors to named companies."""
    cfg = settings.b2b_churn
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    batch_size = getattr(cfg, "account_resolution_batch_size", 100)
    backfill_min = getattr(cfg, "account_resolution_backfill_min_confidence", "medium")
    eligible_statuses = list(
        getattr(
            cfg,
            "account_resolution_eligible_statuses",
            ["enriched", "no_signal", "quarantined"],
        ) or ["enriched", "no_signal", "quarantined"]
    )
    source_priority = list(
        getattr(
            cfg,
            "account_resolution_source_priority",
            ["g2", "gartner", "capterra", "software_advice", "trustpilot"],
        ) or ["g2", "gartner", "capterra", "software_advice", "trustpilot"]
    )
    excluded_sources = list(
        _normalize_account_resolution_excluded_sources(
            getattr(
                cfg,
                "account_resolution_excluded_sources",
                ["trustpilot"],
            ) or ["trustpilot"]
        )
    )
    retry_unresolved_sources = list(
        getattr(
            cfg,
            "account_resolution_retry_unresolved_sources",
            ["reddit", "hackernews", "github"],
        ) or ["reddit", "hackernews", "github"]
    )
    retry_interval_hours = int(
        getattr(
            cfg,
            "account_resolution_unresolved_retry_interval_hours",
            24,
        )
    )
    backfill_labels = {"high"}
    if backfill_min == "medium":
        backfill_labels.add("medium")
    elif backfill_min == "low":
        backfill_labels.update(("medium", "low"))

    t0 = time.monotonic()
    excluded_count = await _exclude_unsupported_sources(
        pool,
        eligible_statuses=eligible_statuses,
        excluded_sources=excluded_sources,
    )

    # APPROVED-ENRICHMENT-READ: reviewer_context
    # Reason: account resolution needs company_name from enrichment for identity matching
    rows = await pool.fetch(
        """
        SELECT r.id, r.source, r.source_url, r.reviewer_name,
               r.reviewer_title, r.reviewer_company, r.reviewer_company_norm,
               r.company_size_raw, r.reviewer_industry,
               r.review_text, r.enrichment, r.raw_metadata,
               COALESCE(primary_vm.vendor_name, r.vendor_name) AS vendor_name,
               r.product_category, r.rating, r.rating_max
        FROM b2b_reviews r
        LEFT JOIN b2b_review_vendor_mentions primary_vm
          ON primary_vm.review_id = r.id
         AND primary_vm.is_primary = TRUE
        LEFT JOIN b2b_account_resolution ar ON ar.review_id = r.id
        WHERE r.enrichment_status = ANY($1::text[])
          AND r.duplicate_of_review_id IS NULL
          AND r.enrichment IS NOT NULL
          AND NOT (r.source = ANY($2::text[]))
          AND (
                ar.id IS NULL
             OR (
                    ar.resolution_status = 'excluded'
                AND ar.resolution_method = 'unsupported_source'
                AND NOT (r.source = ANY($2::text[]))
             )
             OR (
                    ar.resolution_status = 'unresolved'
                AND r.source = ANY($5::text[])
                AND COALESCE(ar.resolved_at, NOW() - make_interval(hours => ($6::int + 1)))
                    <= NOW() - make_interval(hours => $6::int)
             )
          )
        ORDER BY
            CASE
                WHEN NULLIF(trim(coalesce(r.reviewer_company, '')), '') IS NOT NULL THEN 0
                WHEN NULLIF(trim(coalesce(r.enrichment #>> '{reviewer_context,company_name}', '')), '') IS NOT NULL THEN 1
                WHEN NULLIF(trim(coalesce(r.reviewer_title, '')), '') IS NOT NULL
                     AND lower(r.reviewer_title) ~ '( at | @ )' THEN 2
                WHEN NULLIF(trim(coalesce(r.reviewer_title, '')), '') IS NOT NULL THEN 3
                WHEN lower(left(coalesce(r.review_text, ''), 500)) ~ $3 THEN 4
                ELSE 5
            END,
            COALESCE(array_position($4::text[], r.source), 999),
            r.enriched_at DESC
        LIMIT $7
        """,
        eligible_statuses,
        excluded_sources,
        _SELF_IDENT_TEXT_PATTERN,
        source_priority,
        retry_unresolved_sources,
        retry_interval_hours,
        batch_size,
    )

    if not rows:
        return {
            "_skip_synthesis": "No reviews pending resolution",
            "excluded_sources": int(excluded_count or 0),
        }

    from ...services.b2b.account_resolver import (
        extract_from_github_profile,
        extract_from_hn_profile,
        extract_from_reddit_profile,
        fetch_github_profile,
        fetch_hn_profile,
        fetch_reddit_profile,
        resolve_review,
    )

    # Pre-fetch profiles for HN, GitHub, and Reddit reviews (parallel, semaphore-limited)
    import httpx

    max_fetches = getattr(cfg, "account_resolution_max_profile_fetches", 50)
    reddit_max_fetches = getattr(cfg, "account_resolution_reddit_max_profile_fetches", 150)
    fetch_concurrency = getattr(cfg, "account_resolution_profile_fetch_concurrency", 10)
    fetch_timeout = getattr(cfg, "account_resolution_profile_fetch_timeout", 5.0)
    reddit_fetch_concurrency = getattr(
        cfg,
        "account_resolution_reddit_profile_fetch_concurrency",
        3,
    )
    reddit_fetch_delay_seconds = getattr(
        cfg,
        "account_resolution_reddit_profile_fetch_delay_seconds",
        0.25,
    )
    reddit_fetch_max_retries = getattr(
        cfg,
        "account_resolution_reddit_profile_fetch_max_retries",
        1,
    )
    reddit_fetch_retry_after_cap_seconds = getattr(
        cfg,
        "account_resolution_reddit_profile_fetch_retry_after_cap_seconds",
        5.0,
    )

    hn_reviews = [r for r in rows if (r["source"] or "") == "hackernews" and r["reviewer_name"]]
    gh_reviews = [r for r in rows if (r["source"] or "") == "github" and r["reviewer_name"]]
    reddit_reviews = [r for r in rows if (r["source"] or "") == "reddit" and r["reviewer_name"]]
    profile_cache: dict[str, dict] = {}

    if hn_reviews or gh_reviews or reddit_reviews:
        # Deduplicate usernames before launching coroutines
        hn_usernames = _unique_usernames([dict(r) for r in hn_reviews], max_fetches)
        gh_usernames = _unique_usernames([dict(r) for r in gh_reviews], max_fetches)
        reddit_usernames = _unique_usernames([dict(r) for r in reddit_reviews], reddit_max_fetches)

        async with httpx.AsyncClient(timeout=fetch_timeout) as http:
            sem = asyncio.Semaphore(fetch_concurrency)
            reddit_sem = asyncio.Semaphore(reddit_fetch_concurrency)

            async def _fetch_hn(username: str) -> None:
                async with sem:
                    profile = await fetch_hn_profile(username, http)
                    if profile:
                        profile_cache[f"hn:{username}"] = profile

            async def _fetch_gh(username: str) -> None:
                async with sem:
                    profile = await fetch_github_profile(username, http)
                    if profile:
                        profile_cache[f"gh:{username}"] = profile

            async def _fetch_reddit(username: str) -> None:
                async with reddit_sem:
                    profile = await fetch_reddit_profile(
                        username,
                        http,
                        request_delay_seconds=reddit_fetch_delay_seconds,
                        max_retries=reddit_fetch_max_retries,
                        retry_after_cap_seconds=reddit_fetch_retry_after_cap_seconds,
                    )
                    if profile:
                        profile_cache[f"reddit:{username}"] = profile

            await asyncio.gather(
                *[_fetch_hn(u) for u in hn_usernames],
                *[_fetch_gh(u) for u in gh_usernames],
                *[_fetch_reddit(u) for u in reddit_usernames],
                return_exceptions=True,
            )

        logger.info(
            "Fetched %d profiles (HN=%d, GH=%d, Reddit=%d) concurrency=%d reddit_concurrency=%d timeout=%.1fs caps=(hn_gh=%d reddit=%d)",
            len(profile_cache),
            sum(1 for k in profile_cache if k.startswith("hn:")),
            sum(1 for k in profile_cache if k.startswith("gh:")),
            sum(1 for k in profile_cache if k.startswith("reddit:")),
            fetch_concurrency,
            reddit_fetch_concurrency,
            fetch_timeout,
            max_fetches,
            reddit_max_fetches,
        )

    # Build vendor blocklist
    vendor_names = {r["vendor_name"] for r in rows if r["vendor_name"]}
    blocked_by_vendor: dict[str, set[str]] = {}
    for vn in vendor_names:
        norm = normalize_company_name(vn)
        blocked_by_vendor[vn] = {norm} if norm else set()

    resolved_count = 0
    unresolved_count = 0
    backfilled_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0
    errors = 0

    for row in rows:
        review_dict = dict(row)
        # Parse JSONB fields
        for field in ("enrichment", "raw_metadata"):
            val = review_dict.get(field)
            if isinstance(val, str):
                try:
                    review_dict[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    review_dict[field] = {}

        vendor = review_dict.get("vendor_name") or ""
        blocked = blocked_by_vendor.get(vendor, set())
        source = (review_dict.get("source") or "").lower()
        username = (review_dict.get("reviewer_name") or "").strip()

        # Inject fetched profile data as extra signals
        if source == "hackernews" and username:
            hn_profile = profile_cache.get(f"hn:{username}")
            if hn_profile:
                review_dict["_hn_profile"] = hn_profile
        elif source == "github" and username:
            gh_profile = profile_cache.get(f"gh:{username}")
            if gh_profile:
                review_dict["_gh_profile"] = gh_profile
        elif source == "reddit" and username:
            reddit_profile = profile_cache.get(f"reddit:{username}")
            if reddit_profile:
                review_dict["_reddit_profile"] = reddit_profile

        try:
            result = resolve_review(
                review_dict,
                vendor_name=vendor,
                blocked_names=blocked,
            )
        except Exception:
            logger.warning(
                "Resolution failed for review %s", row["id"], exc_info=True,
            )
            errors += 1
            continue

        # Determine status
        if result.resolved_company_name:
            status = "resolved"
            resolved_count += 1
        else:
            status = "unresolved"
            unresolved_count += 1

        if result.confidence_label == "high":
            high_count += 1
        elif result.confidence_label == "medium":
            medium_count += 1
        elif result.confidence_label == "low":
            low_count += 1

        # Persist resolution
        try:
            await pool.execute(
                """
                INSERT INTO b2b_account_resolution (
                    review_id, source, source_item_url,
                    author_handle, author_profile_url,
                    reviewer_company_raw,
                    resolved_company_name, normalized_company_name,
                    confidence_score, confidence_label,
                    resolution_method, resolution_evidence,
                    resolution_status, resolved_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13, NOW()
                )
                ON CONFLICT (review_id) DO UPDATE SET
                    resolved_company_name = EXCLUDED.resolved_company_name,
                    normalized_company_name = EXCLUDED.normalized_company_name,
                    confidence_score = EXCLUDED.confidence_score,
                    confidence_label = EXCLUDED.confidence_label,
                    resolution_method = EXCLUDED.resolution_method,
                    resolution_evidence = EXCLUDED.resolution_evidence,
                    resolution_status = EXCLUDED.resolution_status,
                    resolved_at = EXCLUDED.resolved_at
                """,
                row["id"],
                row["source"],
                row["source_url"],
                row["reviewer_name"],
                _build_author_profile_url(source, username),
                row["reviewer_company"],
                result.resolved_company_name,
                result.normalized_company_name,
                result.confidence_score,
                result.confidence_label,
                result.resolution_method,
                json.dumps(result.to_evidence_json()),
                status,
            )
        except Exception:
            logger.warning(
                "Failed to persist resolution for %s", row["id"], exc_info=True,
            )
            errors += 1
            continue

        # Backfill reviewer_company on b2b_reviews
        if (
            result.resolved_company_name
            and result.confidence_label in backfill_labels
        ):
            try:
                changed = await _backfill_review_company_context(
                    pool,
                    review_id=row["id"],
                    company_name=result.resolved_company_name,
                    company_norm=result.normalized_company_name,
                )
                if changed:
                    backfilled_count += 1
            except Exception:
                logger.warning(
                    "Failed to backfill reviewer_company for %s",
                    row["id"], exc_info=True,
                )

    # Username propagation: if a reviewer has a high/medium resolution elsewhere,
    # propagate that company to their other unresolved reviews (same source + username).
    # Skips [deleted] accounts (shared username across multiple Reddit users).
    propagated_count = 0
    try:
        propagated_count = await _propagate_user_resolutions(pool, backfill_labels)
    except Exception:
        logger.warning("Username propagation failed", exc_info=True)

    elapsed = round(time.monotonic() - t0, 1)
    logger.info(
        "Account resolution: %d resolved, %d unresolved, %d backfilled, "
        "%d propagated, %d errors (high=%d med=%d low=%d) %.1fs",
        resolved_count, unresolved_count, backfilled_count,
        propagated_count, errors, high_count, medium_count, low_count, elapsed,
    )

    return {
        "_skip_synthesis": "Account resolution complete",
        "reviews_processed": len(rows),
        "resolved": resolved_count,
        "unresolved": unresolved_count,
        "backfilled": backfilled_count,
        "propagated": propagated_count,
        "high_confidence": high_count,
        "medium_confidence": medium_count,
        "low_confidence": low_count,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "excluded_sources": int(excluded_count or 0),
    }
