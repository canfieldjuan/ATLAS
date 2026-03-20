"""
B2B review scrape intake: poll configured scrape targets, fetch reviews
from G2, Capterra, TrustRadius, Reddit, HackerNews, GitHub, and RSS feeds,
and insert into b2b_reviews for automatic enrichment pickup.

Runs as an autonomous task on a configurable interval (default 1 hour).
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid as _uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ...config import settings
from ...services.company_normalization import normalize_company_name
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...services.scraping.sources import parse_source_allowlist
from ...services.vendor_registry import resolve_vendor_name
from ...services.scraping.source_fit import classify_source_fit, is_source_fit_allowed

logger = logging.getLogger("atlas.autonomous.tasks.b2b_scrape_intake")


# Common date formats returned by review site parsers
_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",         # ISO 8601 with tz
    "%Y-%m-%dT%H:%M:%S",           # ISO 8601 no tz
    "%Y-%m-%d",                      # ISO date only
    "%b %d, %Y",                     # "Feb 15, 2024"
    "%B %d, %Y",                     # "February 15, 2024"
    "%d %b %Y",                      # "15 Feb 2024"
    "%d %B %Y",                      # "15 February 2024"
    "%m/%d/%Y",                      # "02/15/2024"
    "%d/%m/%Y",                      # "15/02/2024" (EU)
]


def _parse_date(raw: Any) -> datetime | None:
    """Parse a date string in various formats.  Returns None on failure."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Fast path: ISO 8601 (most common from APIs)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass

    # Try common formats
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

    # Last resort: extract "Month Day, Year" or similar from longer text
    m = re.search(
        r"(\w+ \d{1,2},?\s+\d{4})",
        s,
    )
    if m:
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"):
            try:
                return datetime.strptime(m.group(1), fmt).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

    return None


def _make_dedup_key(
    source: str,
    vendor_name: str,
    source_review_id: str | None,
    reviewer_name: str | None,
    reviewed_at: str | None,
) -> str:
    """Generate deterministic dedup key for a review.

    Identical logic to api/b2b_reviews.py and scripts/import_b2b_reviews.py.
    """
    if source_review_id:
        raw = f"{source}:{vendor_name}:{source_review_id}"
    else:
        raw = f"{source}:{vendor_name}:{reviewer_name or ''}:{reviewed_at or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Exhaustive mode helpers (ported from scripts/exhaustive_verified_scrape.py)
# ---------------------------------------------------------------------------

def _parse_review_date(raw: str | None) -> date | None:
    """Best-effort parse of a review date string to a date object."""
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
                "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y",
                "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw[:30], fmt).date()
        except (ValueError, TypeError):
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _filter_by_date(reviews: list[dict], cutoff: date) -> tuple[list[dict], int]:
    """Keep reviews with reviewed_at >= cutoff. Returns (kept, dropped_count)."""
    kept = []
    dropped = 0
    for r in reviews:
        d = _parse_review_date(str(r.get("reviewed_at", "")))
        if d is None or d >= cutoff:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def _determine_stop_reason(result, target, date_dropped: int) -> str:
    """Derive stop reason from parser result + heuristics."""
    if result.stop_reason:
        return result.stop_reason
    page_logs = getattr(result, "page_logs", []) or []
    duplicate_pages = sum(1 for pl in page_logs if pl.stop_reason == "duplicate_page")
    if date_dropped > 0:
        return "date_cutoff"
    if result.pages_scraped >= target.max_pages:
        return "page_cap"
    if duplicate_pages > 0:
        return "duplicate_page"
    if not result.reviews and result.errors:
        return "blocked_or_error"
    if not result.reviews:
        return "no_reviews"
    if page_logs and page_logs[-1].stop_reason:
        return page_logs[-1].stop_reason
    return "pages_exhausted"


def _should_persist_page_logs(stats: dict, page_logs: list) -> bool:
    """Decide whether page logs warrant DB persistence."""
    status = stats.get("status", "")
    if status in ("error", "blocked"):
        return True
    stop = stats.get("stop_reason", "")
    if stop in ("blocked_or_error", "exception"):
        return True
    dup_pages = sum(1 for pl in page_logs if pl.stop_reason == "duplicate_page")
    if dup_pages > 0:
        return True
    total_parsed = sum(pl.reviews_parsed for pl in page_logs)
    total_missing = sum(pl.missing_date for pl in page_logs)
    if total_parsed > 0 and total_missing / total_parsed > 0.2:
        return True
    if any(pl.stop_reason in ("blocked_or_throttled", "http_error") for pl in page_logs):
        return True
    return False


async def _persist_page_logs(pool, run_id, page_logs: list) -> None:
    """Write page-level telemetry rows to b2b_scrape_page_logs."""
    for pl in page_logs:
        try:
            oldest_d = None
            newest_d = None
            if pl.oldest_review:
                try:
                    oldest_d = date.fromisoformat(pl.oldest_review)
                except (ValueError, TypeError):
                    pass
            if pl.newest_review:
                try:
                    newest_d = date.fromisoformat(pl.newest_review)
                except (ValueError, TypeError):
                    pass
            await pool.execute(
                """
                INSERT INTO b2b_scrape_page_logs
                    (run_id, page, url, requested_at, status_code, final_url,
                     response_bytes, duration_ms,
                     review_nodes_found, reviews_parsed,
                     missing_date, missing_rating, missing_body, missing_author,
                     oldest_review, newest_review,
                     next_page_found, next_page_url, content_hash,
                     duplicate_reviews, stop_reason, errors)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22::jsonb)
                """,
                run_id,
                pl.page,
                pl.url,
                datetime.fromisoformat(pl.timestamp) if pl.timestamp else datetime.now(timezone.utc),
                pl.status_code,
                pl.final_url or pl.url,
                pl.response_bytes,
                pl.duration_ms,
                pl.review_nodes_found,
                pl.reviews_parsed,
                pl.missing_date,
                pl.missing_rating,
                pl.missing_body,
                pl.missing_author,
                oldest_d,
                newest_d,
                pl.next_page_found,
                pl.next_page_url or None,
                pl.content_hash or None,
                pl.duplicate_reviews,
                pl.stop_reason or None,
                json.dumps(pl.errors[:5] if pl.errors else []),
            )
        except Exception:
            logger.debug("Failed to persist page log page=%d (non-fatal)", pl.page, exc_info=True)


async def _log_scrape_exhaustive(
    pool, target, status: str, stats: dict, result, parser, duration_ms: int,
) -> _uuid.UUID | None:
    """Enriched scrape log for exhaustive mode. Returns run_id or None."""
    proxy_type = "residential" if parser.prefer_residential else "none"
    pv = getattr(parser, "version", None)
    block_type = _classify_block_type(status, list(result.errors) if result else [])

    oldest_d = None
    newest_d = None
    try:
        oldest_d = date.fromisoformat(stats["oldest_review"]) if stats.get("oldest_review") else None
    except (ValueError, TypeError):
        pass
    try:
        newest_d = date.fromisoformat(stats["newest_review"]) if stats.get("newest_review") else None
    except (ValueError, TypeError):
        pass

    page_logs = getattr(result, "page_logs", []) if result else []
    should_persist = bool(page_logs) and _should_persist_page_logs(stats, page_logs)
    duplicate_pages = sum(1 for pl in page_logs if pl.stop_reason == "duplicate_page") if page_logs else 0

    try:
        run_id = await pool.fetchval(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type, parser_version,
                 block_type, stop_reason, oldest_review, newest_review,
                 date_dropped, duplicate_pages, has_page_logs)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17)
            RETURNING id
            """,
            _uuid.UUID(target.id),
            target.source,
            status,
            stats.get("found", 0),
            stats.get("inserted", 0),
            result.pages_scraped if result else 0,
            json.dumps(list(result.errors[:10]) if result and result.errors else []),
            duration_ms,
            proxy_type,
            pv,
            block_type,
            stats.get("stop_reason"),
            oldest_d,
            newest_d,
            stats.get("date_dropped", 0),
            duplicate_pages,
            should_persist,
        )
    except Exception:
        logger.warning("Failed to log exhaustive scrape result", exc_info=True)
        return None

    if should_persist and run_id:
        await _persist_page_logs(pool, run_id, page_logs)

    return run_id


def _review_date_stats(reviews: list[dict]) -> dict:
    """Compute date diagnostics for a batch of reviews."""
    dates = []
    null_count = 0
    for r in reviews:
        raw = r.get("reviewed_at")
        if not raw:
            null_count += 1
            continue
        d = _parse_review_date(str(raw))
        if d:
            dates.append(d)
    return {
        "oldest": str(min(dates)) if dates else None,
        "newest": str(max(dates)) if dates else None,
    }


_INSERT_SQL = """
INSERT INTO b2b_reviews (
    dedup_key, source, source_url, source_review_id,
    vendor_name, product_name, product_category,
    rating, rating_max, summary, review_text, pros, cons,
    reviewer_name, reviewer_title, reviewer_company, reviewer_company_norm,
    company_size_raw, reviewer_industry, reviewed_at,
    import_batch_id, raw_metadata, parser_version,
    content_type, thread_id, comment_depth,
    relevance_score, author_churn_score, source_weight,
    reddit_subreddit, reddit_trending, reddit_flair,
    reddit_is_edited, reddit_is_crosspost, reddit_num_comments,
    reddit_score, reddit_comment_thread_count, reddit_crosspost_subreddits
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23,
    $24, $25, $26,
    $27, $28, $29,
    $30, $31, $32, $33, $34, $35,
    $36, $37, $38::jsonb
)
ON CONFLICT (dedup_key) DO NOTHING
"""

# Resolve parent_review_id for comments after all posts in the batch are inserted
_RESOLVE_PARENT_SQL = """
UPDATE b2b_reviews AS c
SET parent_review_id = p.id
FROM b2b_reviews AS p
WHERE c.import_batch_id = $1
  AND c.content_type = 'comment'
  AND c.parent_review_id IS NULL
  AND p.source = 'reddit'
  AND p.source_review_id = (c.raw_metadata->>'parent_source_review_id')
"""

_TARGET_QUERY = """
SELECT id, source, vendor_name, product_name, product_slug,
       product_category, max_pages, metadata, scrape_mode,
       last_scraped_at, last_scrape_status, last_scrape_reviews,
       last_scrape_runtime_mode, last_scrape_stop_reason,
       last_scrape_oldest_review, last_scrape_newest_review,
       last_scrape_date_cutoff, last_scrape_pages_scraped,
       last_scrape_reviews_found, last_scrape_reviews_filtered,
       last_scrape_date_dropped, last_scrape_duration_ms,
       last_scrape_resume_page
FROM b2b_scrape_targets
WHERE enabled = true
    AND source = ANY($3::text[])
  AND (last_scraped_at IS NULL
       OR last_scraped_at < NOW() - make_interval(hours => scrape_interval_hours))
  AND (last_scrape_status IS NULL
       OR last_scrape_status != 'blocked'
       OR last_scraped_at < NOW() - make_interval(hours => $1))
ORDER BY CASE WHEN last_scraped_at IS NULL THEN 0 ELSE 1 END,
         priority DESC,
         last_scraped_at ASC NULLS FIRST
LIMIT $2
"""


def _filter_targets_by_source_fit(rows: list[dict[str, Any]], cfg) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter scheduled targets by source/category fit, with metadata override."""
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("source_fit_override") == "allow":
            row["_source_fit"] = "override"
            row["_scrape_vertical"] = None
            kept.append(row)
            continue
        decision = classify_source_fit(row.get("source", ""), row.get("product_category"))
        row["_source_fit"] = decision.fit
        row["_scrape_vertical"] = decision.vertical
        row["_source_fit_reason"] = decision.reason
        if (
            metadata.get("source_fit_probation") is True
            and decision.fit == "conditional"
            and cfg.source_fit_allow_probation
        ):
            row["_source_fit"] = "probation"
            kept.append(row)
            continue
        if cfg.source_fit_filter_enabled and not is_source_fit_allowed(
            row.get("source", ""),
            row.get("product_category"),
            allow_conditional=cfg.source_fit_allow_conditional,
        ):
            skipped.append({
                "source": row.get("source"),
                "vendor": row.get("vendor_name"),
                "product_category": row.get("product_category"),
                "vertical": decision.vertical,
                "fit": decision.fit,
                "reason": decision.reason,
            })
            continue
        kept.append(row)
    return kept, skipped


def _coerce_target_metadata(raw_meta: Any) -> dict[str, Any]:
    """Normalize target metadata to a mutable dict."""
    if isinstance(raw_meta, dict):
        return dict(raw_meta)
    if isinstance(raw_meta, str):
        try:
            decoded = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _merge_scrape_raw_metadata(raw_meta: Any, target_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Attach scrape-target provenance to review raw_metadata."""
    merged = _coerce_target_metadata(raw_meta)
    if not target_context:
        return merged
    for key, value in target_context.items():
        if value is not None:
            merged[key] = value
    return merged


def _derive_runtime_scrape_mode(scrape_mode: str, metadata: dict[str, Any]) -> str:
    """Map top-level target mode to the parser runtime mode."""
    _ = metadata
    return "initial" if scrape_mode == "exhaustive" else "incremental"


def _date_to_iso(raw: Any) -> str | None:
    """Convert a date-like value to a canonical YYYY-MM-DD string."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    parsed = _parse_review_date(str(raw))
    return parsed.isoformat() if parsed else None


def _scrape_state_from_row(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Load checkpoint state from explicit target columns."""
    _ = metadata
    explicit = {
        "mode": str(row.get("scrape_mode") or "") or None,
        "runtime_mode": row.get("last_scrape_runtime_mode"),
        "status": row.get("last_scrape_status"),
        "oldest_review": _date_to_iso(row.get("last_scrape_oldest_review")),
        "newest_review": _date_to_iso(row.get("last_scrape_newest_review")),
        "date_cutoff_used": _date_to_iso(row.get("last_scrape_date_cutoff")),
        "pages_scraped": row.get("last_scrape_pages_scraped"),
        "reviews_found": row.get("last_scrape_reviews_found"),
        "reviews_inserted": row.get("last_scrape_reviews"),
        "reviews_filtered": row.get("last_scrape_reviews_filtered"),
        "date_dropped": row.get("last_scrape_date_dropped"),
        "duration_ms": row.get("last_scrape_duration_ms"),
        "stop_reason": row.get("last_scrape_stop_reason"),
        "resume_page": row.get("last_scrape_resume_page"),
    }
    return {key: value for key, value in explicit.items() if value not in (None, "")}


def _incremental_cutoff_from_row(row: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    """Load the stored newest-review checkpoint for incremental reruns."""
    if metadata.get("use_incremental_checkpoint", True) is False:
        return None
    state = _scrape_state_from_row(row, metadata)
    newest_review = state.get("newest_review")
    if newest_review:
        return _date_to_iso(newest_review)
    legacy_state = metadata.get("scrape_state")
    if isinstance(legacy_state, dict):
        return _date_to_iso(legacy_state.get("newest_review"))
    return None


def _prepare_scrape_target(row: dict[str, Any], cfg) -> tuple[Any, str, dict[str, Any]]:
    """Build a parser target with effective mode and cutoff settings applied."""
    from ...services.scraping.parsers import ScrapeTarget

    raw_meta = _coerce_target_metadata(row.get("metadata"))
    scrape_mode = str(row.get("scrape_mode", "incremental") or "incremental")
    runtime_mode = _derive_runtime_scrape_mode(scrape_mode, raw_meta)
    metadata = dict(raw_meta)
    metadata.pop("scrape_mode", None)
    target_metadata = dict(raw_meta)
    target_metadata.pop("scrape_mode", None)
    target_metadata["scrape_mode"] = runtime_mode

    target = ScrapeTarget(
        id=str(row["id"]),
        source=row["source"],
        vendor_name=row["vendor_name"],
        product_name=row["product_name"],
        product_slug=row["product_slug"],
        product_category=row["product_category"],
        max_pages=row["max_pages"],
        metadata=target_metadata,
    )

    if scrape_mode == "exhaustive":
        lookback_days = metadata.get("lookback_days", cfg.exhaustive_lookback_days)
        target.date_cutoff = str(date.today() - timedelta(days=lookback_days))
        if target.max_pages <= 5:
            target.max_pages = metadata.get(
                "exhaustive_max_pages",
                cfg.exhaustive_max_pages_default,
            )
    else:
        target.date_cutoff = _incremental_cutoff_from_row(row, metadata)

    return target, scrape_mode, metadata


def _build_scrape_state(
    metadata: dict[str, Any],
    target,
    scrape_mode: str,
    result,
    *,
    inserted: int,
    filtered_count: int,
    date_dropped: int,
    duration_ms: int,
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build persisted scrape checkpoint state for the target metadata."""
    previous = previous_state if isinstance(previous_state, dict) else {}
    if not previous:
        fallback = metadata.get("scrape_state")
        previous = fallback if isinstance(fallback, dict) else {}
    date_info = _review_date_stats(result.reviews) if result.reviews else {"oldest": None, "newest": None}
    stop_reason = _determine_stop_reason(result, target, date_dropped)

    state = {
        "source": target.source,
        "mode": scrape_mode,
        "runtime_mode": target.metadata.get("scrape_mode"),
        "status": result.status,
        "last_run_at": datetime.now(timezone.utc).isoformat(),
        "oldest_review": date_info["oldest"] or previous.get("oldest_review"),
        "newest_review": date_info["newest"] or previous.get("newest_review"),
        "date_cutoff_used": target.date_cutoff,
        "pages_scraped": result.pages_scraped,
        "reviews_found": len(result.reviews) + filtered_count + date_dropped,
        "reviews_inserted": inserted,
        "reviews_filtered": filtered_count,
        "date_dropped": date_dropped,
        "duration_ms": duration_ms,
        "stop_reason": stop_reason,
    }
    if result.resume_page is not None:
        state["resume_page"] = result.resume_page
    return state


def _scrape_state_db_values(scrape_state: dict[str, Any]) -> dict[str, Any]:
    """Coerce checkpoint state into DB-ready scalar values."""
    return {
        "runtime_mode": str(scrape_state.get("runtime_mode") or "") or None,
        "stop_reason": str(scrape_state.get("stop_reason") or "") or None,
        "oldest_review": _parse_review_date(str(scrape_state.get("oldest_review") or "")),
        "newest_review": _parse_review_date(str(scrape_state.get("newest_review") or "")),
        "date_cutoff": _parse_review_date(str(scrape_state.get("date_cutoff_used") or "")),
        "pages_scraped": scrape_state.get("pages_scraped"),
        "reviews_found": scrape_state.get("reviews_found"),
        "reviews_filtered": scrape_state.get("reviews_filtered"),
        "date_dropped": scrape_state.get("date_dropped"),
        "duration_ms": scrape_state.get("duration_ms"),
        "resume_page": scrape_state.get("resume_page"),
    }


async def _update_target_after_scrape(
    pool,
    target_id: Any,
    status: str,
    inserted: int,
    *,
    metadata: dict[str, Any] | None = None,
    scrape_state: dict[str, Any] | None = None,
) -> None:
    """Persist target status and optional checkpoint metadata."""
    if metadata is None or scrape_state is None:
        await pool.execute(
            """
            UPDATE b2b_scrape_targets
            SET last_scraped_at = NOW(), last_scrape_status = $2,
                last_scrape_reviews = $3, updated_at = NOW()
            WHERE id = $1
            """,
            target_id,
            status,
            inserted,
        )
        return

    updated_metadata = dict(metadata)
    updated_metadata.pop("scrape_state", None)
    state_values = _scrape_state_db_values(scrape_state)
    await pool.execute(
        """
        UPDATE b2b_scrape_targets
        SET last_scraped_at = NOW(), last_scrape_status = $2,
            last_scrape_reviews = $3, metadata = $4::jsonb,
            last_scrape_runtime_mode = $5,
            last_scrape_stop_reason = $6,
            last_scrape_oldest_review = $7,
            last_scrape_newest_review = $8,
            last_scrape_date_cutoff = $9,
            last_scrape_pages_scraped = $10,
            last_scrape_reviews_found = $11,
            last_scrape_reviews_filtered = $12,
            last_scrape_date_dropped = $13,
            last_scrape_duration_ms = $14,
            last_scrape_resume_page = $15,
            updated_at = NOW()
        WHERE id = $1
        """,
        target_id,
        status,
        inserted,
        json.dumps(updated_metadata),
        state_values["runtime_mode"],
        state_values["stop_reason"],
        state_values["oldest_review"],
        state_values["newest_review"],
        state_values["date_cutoff"],
        state_values["pages_scraped"],
        state_values["reviews_found"],
        state_values["reviews_filtered"],
        state_values["date_dropped"],
        state_values["duration_ms"],
        state_values["resume_page"],
    )


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: scrape B2B review sites per configured targets."""
    cfg = settings.b2b_scrape
    if not cfg.enabled:
        return {"_skip_synthesis": True, "skipped": "b2b_scrape disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": True, "skipped": "db not ready"}

    # Import here to avoid circular imports and lazy-load curl_cffi
    from ...services.scraping.client import get_scrape_client
    from ...services.scraping.parsers import ScrapeTarget, get_all_parsers, get_parser
    from ...services.scraping.relevance import STRUCTURED_SOURCES, filter_reviews

    client = get_scrape_client()
    allowed_sources = parse_source_allowlist(cfg.source_allowlist)
    if not allowed_sources:
        allowed_sources = list(get_all_parsers().keys())

    # Fetch due targets.
    # Use minimum per-source cooldown as SQL floor, then post-filter for
    # sources with longer cooldown (Sprint 2: per-source orchestration).
    from ...services.scraping.capabilities import get_all_capabilities, get_capability

    all_caps = get_all_capabilities()
    min_cooldown_hours = min(
        (c.cooldown_minutes / 60 for c in all_caps.values()),
        default=cfg.blocked_cooldown_hours,
    )
    # 0 means unlimited per config docs; Postgres LIMIT 0 returns nothing,
    # so substitute a large cap when the user sets 0.
    target_limit = cfg.max_targets_per_run if cfg.max_targets_per_run > 0 else 2147483647
    raw_targets = await pool.fetch(
        _TARGET_QUERY,
        max(1, int(min_cooldown_hours)),
        target_limit,
        allowed_sources,
    )

    # Post-filter: apply per-source cooldown for blocked targets
    targets = []
    now = datetime.now(timezone.utc)
    for row in raw_targets:
        row = dict(row)
        if row.get("last_scrape_status") == "blocked" and row.get("last_scraped_at"):
            cap = get_capability(row["source"])
            cooldown_min = cap.cooldown_minutes if cap else cfg.blocked_cooldown_hours * 60
            if (now - row["last_scraped_at"]).total_seconds() < cooldown_min * 60:
                continue
        targets.append(row)

    targets, source_fit_skipped = _filter_targets_by_source_fit(targets, cfg)
    if source_fit_skipped:
        logger.info(
            "Scrape intake source-fit policy skipped %d targets",
            len(source_fit_skipped),
        )

    if not targets:
        return {
            "_skip_synthesis": True,
            "targets_due": 0,
            "targets_skipped_source_fit": len(source_fit_skipped),
            "skipped_targets": source_fit_skipped[:20],
        }

    total_reviews = 0
    total_inserted = 0
    results_summary: list[dict] = []
    results_lock = asyncio.Lock()

    # Group targets by source for concurrent scraping.
    # Per-source concurrency read from capability profiles (Sprint 2).
    _DEFAULT_CONCURRENCY = 4

    source_sems: dict[str, asyncio.Semaphore] = {}
    for row in targets:
        src = row["source"]
        if src not in source_sems:
            cap = get_capability(src)
            limit = cap.max_concurrency if cap else _DEFAULT_CONCURRENCY
            source_sems[src] = asyncio.Semaphore(limit)

    async def _scrape_one(row):
        """Scrape a single target with per-source concurrency control."""
        nonlocal total_reviews, total_inserted

        target, scrape_mode, metadata = _prepare_scrape_target(row, cfg)

        parser = get_parser(target.source)
        if not parser:
            logger.warning("No parser for source %r, skipping target %s", target.source, target.id)
            return

        sem = source_sems.get(target.source, asyncio.Semaphore(2))
        async with sem:
            started_at = time.monotonic()
            batch_id = f"scrape_{target.source}_{target.product_slug}_{int(time.time())}"
            client.reset_captcha_stats()
            previous_state = _scrape_state_from_row(row, metadata)

            try:
                result = await parser.scrape(target, client)
            except Exception as exc:
                logger.error("Scrape failed for %s/%s: %s", target.source, target.vendor_name, exc)
                duration_ms = int((time.monotonic() - started_at) * 1000)
                await _log_scrape(
                    pool, target, "failed", 0, 0, 0, [str(exc)], duration_ms, parser,
                    captcha_attempts=client.captcha_attempts,
                    captcha_types=sorted(client.captcha_types_seen) if client.captcha_types_seen else None,
                    captcha_solve_ms=client.captcha_solve_ms_total,
                )
                await _update_target_after_scrape(pool, row["id"], "failed", 0)
                async with results_lock:
                    results_summary.append({
                        "source": target.source,
                        "vendor": target.vendor_name,
                        "status": "failed",
                        "error": str(exc),
                    })
                return

            # Relevance filter
            filtered_count = 0
            if (cfg.relevance_filter_enabled
                    and target.source not in STRUCTURED_SOURCES
                    and result.reviews):
                original_count = len(result.reviews)
                result.reviews, filtered_count = filter_reviews(
                    result.reviews, target.vendor_name, cfg.relevance_threshold,
                )
                if filtered_count:
                    logger.info(
                        "Relevance filter: kept %d/%d for %s/%s",
                        len(result.reviews), original_count,
                        target.source, target.vendor_name,
                    )

            # Exhaustive mode: date filtering
            date_dropped = 0
            if scrape_mode == "exhaustive" and result.reviews and target.date_cutoff:
                cutoff = date.fromisoformat(target.date_cutoff)
                result.reviews, date_dropped = _filter_by_date(result.reviews, cutoff)

            # Insert reviews + fire enrichment immediately (background task)
            inserted = 0
            insert_stats = {
                "inserted": 0,
                "skipped_short": 0,
                "duplicate_or_existing": 0,
                "named_company_reviews": 0,
                "eligible_rows": 0,
            }
            pv = getattr(parser, 'version', None)
            if result.reviews:
                insert_stats = await _insert_reviews(
                    pool,
                    result.reviews,
                    batch_id,
                    parser_version=pv,
                    target_context={
                        "scrape_target_id": str(row["id"]),
                        "scrape_target_source_fit": row.get("_source_fit"),
                        "scrape_target_vertical": row.get("_scrape_vertical"),
                        "scrape_target_probation": row.get("_source_fit") == "probation",
                    },
                )
                inserted = insert_stats["inserted"]

                # Fire enrichment NOW -- don't wait for it, let vLLM chew
                if inserted > 0:
                    asyncio.create_task(
                        _fire_enrichment(batch_id, target.source, target.vendor_name),
                        name=f"enrich_{batch_id}",
                    )

                # Mark synthetic aggregate reviews as not_applicable
                synthetic_keys = [
                    _make_dedup_key(
                        r["source"], r["vendor_name"],
                        r.get("source_review_id"),
                        r.get("reviewer_name"),
                        r.get("reviewed_at"),
                    )
                    for r in result.reviews
                    if r.get("raw_metadata", {}).get("extraction_method") == "jsonld_aggregate"
                ]
                if synthetic_keys:
                    await pool.execute(
                        """
                        UPDATE b2b_reviews
                        SET enrichment_status = 'not_applicable'
                        WHERE dedup_key = ANY($1::text[])
                          AND enrichment_status = 'pending'
                        """,
                        synthetic_keys,
                    )

            duration_ms = int((time.monotonic() - started_at) * 1000)

            # Log + update target
            if scrape_mode == "exhaustive":
                date_info = _review_date_stats(result.reviews) if result.reviews else {"oldest": None, "newest": None}
                stop_reason = _determine_stop_reason(result, target, date_dropped)
                await _log_scrape_exhaustive(
                    pool, target, result.status,
                    {
                        "found": len(result.reviews) + filtered_count + date_dropped,
                        "inserted": inserted,
                        "date_dropped": date_dropped,
                        "stop_reason": stop_reason,
                        "oldest_review": date_info["oldest"],
                        "newest_review": date_info["newest"],
                        "status": result.status,
                    },
                    result, parser, duration_ms,
                )
            else:
                date_info = _review_date_stats(result.reviews) if result.reviews else {"oldest": None, "newest": None}
                stop_reason = _determine_stop_reason(result, target, date_dropped)
                scrape_errors = list(result.errors)
                if filtered_count:
                    scrape_errors.append(f"relevance_filtered={filtered_count}")
                await _log_scrape(
                    pool, target, result.status,
                    len(result.reviews) + filtered_count, inserted, result.pages_scraped,
                    scrape_errors, duration_ms, parser,
                    captcha_attempts=client.captcha_attempts,
                    captcha_types=sorted(client.captcha_types_seen) if client.captcha_types_seen else None,
                    captcha_solve_ms=client.captcha_solve_ms_total,
                    page_logs=getattr(result, "page_logs", []) or [],
                    stop_reason=stop_reason,
                    oldest_review=date_info["oldest"],
                    newest_review=date_info["newest"],
                    date_dropped=date_dropped,
                )
            scrape_state = _build_scrape_state(
                metadata,
                target,
                scrape_mode,
                result,
                inserted=inserted,
                filtered_count=filtered_count,
                date_dropped=date_dropped,
                duration_ms=duration_ms,
                previous_state=previous_state,
            )
            await _update_target_after_scrape(
                pool,
                row["id"],
                result.status,
                inserted,
                metadata=metadata,
                scrape_state=scrape_state,
            )

            async with results_lock:
                total_reviews += len(result.reviews) + filtered_count + date_dropped
                total_inserted += inserted
                entry = {
                    "source": target.source,
                    "vendor": target.vendor_name,
                    "status": result.status,
                    "found": len(result.reviews) + filtered_count + date_dropped,
                    "inserted": inserted,
                    "filtered": filtered_count,
                    "pages": result.pages_scraped,
                    "mode": scrape_mode,
                    "source_fit": row.get("_source_fit"),
                    "vertical": row.get("_scrape_vertical"),
                    "duplicate_or_existing": insert_stats["duplicate_or_existing"],
                    "named_company_reviews": insert_stats["named_company_reviews"],
                    "skipped_short": insert_stats["skipped_short"],
                }
                if date_dropped:
                    entry["date_dropped"] = date_dropped
                results_summary.append(entry)

            logger.info(
                "Scraped %s/%s: %d found, %d inserted (%s) in %dms",
                target.source, target.vendor_name,
                len(result.reviews), inserted, result.status, duration_ms,
            )

    # Split targets by mode
    incremental_targets = [t for t in targets if t.get("scrape_mode", "incremental") == "incremental"]
    exhaustive_targets = [t for t in targets if t.get("scrape_mode") == "exhaustive"]

    # Fire incremental targets concurrently (per-source semaphores handle throttling)
    if incremental_targets:
        logger.info(
            "Scraping %d incremental targets concurrently (%d sources)",
            len(incremental_targets), len(source_sems),
        )
        await asyncio.gather(
            *[_scrape_one(row) for row in incremental_targets],
            return_exceptions=True,
        )

    # Exhaustive targets: sequential per source, sources in parallel
    if exhaustive_targets:
        by_source: dict[str, list] = defaultdict(list)
        for row in exhaustive_targets:
            by_source[row["source"]].append(row)

        async def _run_exhaustive_source(source: str, rows: list):
            for i, row in enumerate(rows):
                await _scrape_one(row)
                if i < len(rows) - 1:
                    await asyncio.sleep(cfg.exhaustive_inter_vendor_delay)

        logger.info(
            "Scraping %d exhaustive targets across %d sources (sequential per source)",
            len(exhaustive_targets), len(by_source),
        )
        await asyncio.gather(
            *[_run_exhaustive_source(src, rows) for src, rows in by_source.items()],
            return_exceptions=True,
        )

    # Enrichment fires as background tasks per-target (see _fire_enrichment).
    # No need to wait -- vLLM handles concurrent requests natively.
    # The b2b_enrichment scheduler task catches any stragglers every 30s.

    return {
        "_skip_synthesis": True,
        "targets_scraped": len(results_summary),
        "targets_skipped_source_fit": len(source_fit_skipped),
        "total_reviews_found": total_reviews,
        "total_reviews_inserted": total_inserted,
        "skipped_targets": source_fit_skipped[:20],
        "results": results_summary,
    }


async def _fire_enrichment(batch_id: str, source: str, vendor: str) -> None:
    """Fire-and-forget enrichment for a single scrape batch.

    Runs as a background asyncio task so scraping continues unblocked.
    vLLM with PagedAttention handles concurrent requests natively.
    """
    try:
        from .b2b_enrichment import enrich_batch
        result = await enrich_batch(batch_id)
        logger.info(
            "Enrichment done for %s/%s: %s",
            source, vendor, result,
        )
    except Exception as exc:
        logger.warning(
            "Background enrichment failed for %s/%s (scheduler retries): %s",
            source, vendor, exc,
        )


_MIN_ENRICHABLE_TEXT_LEN = 80  # Reviews shorter than this can't produce useful enrichment


async def _insert_reviews(
    pool,
    reviews: list[dict],
    batch_id: str,
    parser_version: str | None = None,
    *,
    target_context: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Insert reviews into b2b_reviews with dedup and return batch stats."""
    rows = []
    skipped_short = 0
    for r in reviews:
        # Gate: don't insert reviews with no meaningful text body
        # Combine review_text + pros + cons for length check (some sources
        # put the substance in pros/cons rather than the main body)
        review_text = r.get("review_text") or ""
        pros = r.get("pros") or ""
        cons = r.get("cons") or ""
        combined_len = len(review_text) + len(pros) + len(cons)
        if combined_len < _MIN_ENRICHABLE_TEXT_LEN:
            skipped_short += 1
            continue

        reviewed_at_ts = _parse_date(r.get("reviewed_at"))

        # Resolve to canonical vendor name BEFORE dedup key so both use canonical form
        canonical_vendor = await resolve_vendor_name(r["vendor_name"])

        dedup_key = _make_dedup_key(
            r["source"], canonical_vendor,
            r.get("source_review_id"),
            r.get("reviewer_name"),
            r.get("reviewed_at"),
        )

        reviewer_company = r.get("reviewer_company")
        reviewer_company_norm = normalize_company_name(reviewer_company or "") or None

        raw_metadata = _merge_scrape_raw_metadata(r.get("raw_metadata", {}), target_context)

        rows.append((
            dedup_key,
            r["source"],
            r.get("source_url"),
            r.get("source_review_id"),
            canonical_vendor,
            r.get("product_name"),
            r.get("product_category"),
            r.get("rating"),
            r.get("rating_max") or 5,
            r.get("summary"),
            r["review_text"],
            r.get("pros"),
            r.get("cons"),
            r.get("reviewer_name"),
            r.get("reviewer_title"),
            reviewer_company,
            reviewer_company_norm,
            r.get("company_size_raw"),
            r.get("reviewer_industry"),
            reviewed_at_ts,
            batch_id,
            json.dumps(raw_metadata),
            parser_version,
            # Threading / content-type columns (migration 133)
            r.get("content_type") or "review",
            r.get("thread_id"),
            r.get("comment_depth") or 0,
            # Relevance score (migration 145) -- set by filter_reviews() in raw_metadata
            raw_metadata.get("relevance_score"),
            # Author churn score (migration 148) -- set by Reddit parser
            raw_metadata.get("author_churn_score"),
            # Source weight (migration 152) -- set by all parsers
            raw_metadata.get("source_weight"),
            # Reddit analytics (migration 156) -- only Reddit reviews have these
            raw_metadata.get("subreddit"),
            raw_metadata.get("trending_score"),
            raw_metadata.get("post_flair"),
            raw_metadata.get("is_edited"),
            raw_metadata.get("is_crosspost"),
            raw_metadata.get("num_comments"),
            raw_metadata.get("score"),
            len(raw_metadata.get("comment_threads") or []),
            json.dumps(raw_metadata.get("crosspost_subreddits"))
            if raw_metadata.get("crosspost_subreddits") is not None
            else None,
        ))

    if skipped_short:
        logger.info("Skipped %d reviews with text < %d chars", skipped_short, _MIN_ENRICHABLE_TEXT_LEN)

    if not rows:
        return {
            "inserted": 0,
            "skipped_short": skipped_short,
            "duplicate_or_existing": 0,
            "named_company_reviews": 0,
            "eligible_rows": 0,
        }

    try:
        async with pool.transaction() as conn:
            await conn.executemany(_INSERT_SQL, rows)
    except Exception:
        logger.exception("Failed to insert scraped reviews (batch %s)", batch_id)
        return {
            "inserted": 0,
            "skipped_short": skipped_short,
            "duplicate_or_existing": len(rows),
            "named_company_reviews": 0,
            "eligible_rows": len(rows),
        }

    # Resolve parent_review_id for Reddit comments (looks up parent post by
    # source_review_id stored in raw_metadata.parent_source_review_id)
    # Note: check original review dicts, not `rows` (which are tuples of values)
    has_comments = any(r.get("content_type") == "comment" for r in reviews)
    if has_comments:
        try:
            await pool.execute(_RESOLVE_PARENT_SQL, batch_id)
        except Exception:
            logger.warning("parent_review_id resolution failed for batch %s", batch_id)

    # Count actual inserts
    count_row = await pool.fetchrow(
        """
        SELECT count(*) AS cnt,
               count(*) FILTER (WHERE reviewer_company_norm IS NOT NULL) AS named_company_reviews
        FROM b2b_reviews
        WHERE import_batch_id = $1
        """,
        batch_id,
    )
    inserted = int(count_row["cnt"] or 0) if count_row else 0
    named_company_reviews = int(count_row["named_company_reviews"] or 0) if count_row else 0
    return {
        "inserted": inserted,
        "skipped_short": skipped_short,
        "duplicate_or_existing": max(len(rows) - inserted, 0),
        "named_company_reviews": named_company_reviews,
        "eligible_rows": len(rows),
    }


async def _log_scrape(
    pool, target, status: str, reviews_found: int, reviews_inserted: int,
    pages_scraped: int, errors: list[str], duration_ms: int, parser,
    *, captcha_attempts: int = 0, captcha_types: list[str] | None = None,
    captcha_solve_ms: int = 0,
    page_logs: list | None = None,
    stop_reason: str | None = None,
    oldest_review: str | None = None,
    newest_review: str | None = None,
    date_dropped: int = 0,
) -> None:
    """Insert a record into b2b_scrape_log."""
    proxy_type = "residential" if parser.prefer_residential else "none"
    pv = getattr(parser, 'version', None)
    page_logs = page_logs or []
    duplicate_pages = sum(1 for pl in page_logs if pl.stop_reason == "duplicate_page")
    has_page_logs = bool(page_logs)
    oldest_d = None
    newest_d = None
    try:
        oldest_d = date.fromisoformat(oldest_review) if oldest_review else None
    except (TypeError, ValueError):
        pass
    try:
        newest_d = date.fromisoformat(newest_review) if newest_review else None
    except (TypeError, ValueError):
        pass
    # Classify block type from errors
    block_type = _classify_block_type(status, errors)
    try:
        run_id = await pool.fetchval(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type, parser_version,
                 captcha_attempts, captcha_types, captcha_solve_ms, block_type,
                 stop_reason, oldest_review, newest_review,
                 date_dropped, duplicate_pages, has_page_logs)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            RETURNING id
            """,
            _uuid.UUID(target.id),
            target.source,
            status,
            reviews_found,
            reviews_inserted,
            pages_scraped,
            json.dumps(errors),
            duration_ms,
            proxy_type,
            pv,
            captcha_attempts,
            captcha_types or [],
            captcha_solve_ms if captcha_solve_ms > 0 else None,
            block_type,
            stop_reason,
            oldest_d,
            newest_d,
            date_dropped,
            duplicate_pages,
            has_page_logs,
        )
        if page_logs and run_id:
            await _persist_page_logs(pool, run_id, page_logs)
    except Exception:
        logger.warning("Failed to log scrape result", exc_info=True)


def _classify_block_type(status: str, errors: list[str]) -> str | None:
    """Classify the block type from scrape status and error messages."""
    if status not in ("blocked", "failed"):
        return None
    error_text = " ".join(errors).lower()
    if "captcha" in error_text or "challenge" in error_text:
        return "captcha"
    if "403" in error_text and ("ban" in error_text or "forbidden" in error_text):
        return "ip_ban"
    if "429" in error_text or "rate" in error_text:
        return "rate_limit"
    if "403" in error_text or "blocked" in error_text:
        return "waf"
    if status == "blocked":
        return "unknown"
    return None
