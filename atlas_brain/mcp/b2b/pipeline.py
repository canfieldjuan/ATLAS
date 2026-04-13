"""B2B Churn MCP -- pipeline & source health tools."""
import json
from typing import Optional

from ._shared import (
    _canonical_review_predicate,
    _suppress_predicate,
    logger,
    get_pool,
    VALID_SOURCES,
)
from .server import mcp
from ...services.b2b.source_impact import (
    build_source_impact_ledger,
    get_consumer_wiring_baseline,
    summarize_source_field_baseline,
)

_MAX_IMPACT_WINDOW_DAYS = 3650


@mcp.tool()
async def get_pipeline_status() -> str:
    """
    Enrichment pipeline health snapshot.

    Returns enrichment counts by status, recent imports (last 24h),
    last enrichment timestamp, active scrape targets, and last scrape time.
    """
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        # Enrichment counts by status
        _rev_sup = _suppress_predicate('review')
        status_rows = await pool.fetch(
            f"""
            SELECT enrichment_status, COUNT(*) AS cnt
            FROM b2b_reviews
            WHERE {_rev_sup}
              AND {_canonical_review_predicate()}
            GROUP BY enrichment_status
            """
        )
        enrichment_counts = {r["enrichment_status"]: r["cnt"] for r in status_rows}

        # Recent imports + last enrichment
        stats = await pool.fetchrow(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE imported_at > NOW() - INTERVAL '24 hours') AS recent_imports_24h,
                MAX(enriched_at) AS last_enrichment_at
            FROM b2b_reviews
            WHERE {_rev_sup}
              AND {_canonical_review_predicate()}
            """
        )

        # Scrape targets summary
        scrape_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE enabled) AS active_scrape_targets,
                MAX(last_scraped_at) AS last_scrape_at
            FROM b2b_scrape_targets
            """
        )

        result = {
            "enrichment_counts": enrichment_counts,
            "recent_imports_24h": stats["recent_imports_24h"] if stats else 0,
            "last_enrichment_at": stats["last_enrichment_at"] if stats else None,
            "active_scrape_targets": scrape_stats["active_scrape_targets"] if scrape_stats else 0,
            "last_scrape_at": scrape_stats["last_scrape_at"] if scrape_stats else None,
        }

        return json.dumps({"success": True, **result}, default=str)
    except Exception as exc:
        logger.exception("get_pipeline_status error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_parser_version_status() -> str:
    """
    Show per-source parser version status and count of reviews needing re-extraction.

    Returns for each source: current parser version, total reviews,
    count at current version, count at outdated versions, and count with unknown version.
    Reviews with outdated parser versions are automatically re-queued for enrichment
    on the next enrichment run.
    """
    try:
        from ...services.scraping.parsers import get_all_parsers

        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        parsers = get_all_parsers()
        sources = []
        for source_name, parser in sorted(parsers.items()):
            current_version = getattr(parser, "version", None)
            if not current_version:
                continue

            row = await pool.fetchrow(
                """
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE parser_version = $2) AS current_count,
                       COUNT(*) FILTER (WHERE parser_version IS NOT NULL AND parser_version != $2) AS outdated_count,
                       COUNT(*) FILTER (WHERE parser_version IS NULL) AS unknown_count
                FROM b2b_reviews
                WHERE source = $1
                """,
                source_name, current_version,
            )
            sources.append({
                "source": source_name,
                "current_version": current_version,
                "total_reviews": row["total"],
                "current_version_count": row["current_count"],
                "outdated_version_count": row["outdated_count"],
                "unknown_version_count": row["unknown_count"],
            })

        return json.dumps({"sources": sources, "count": len(sources)}, default=str)
    except Exception:
        logger.exception("get_parser_version_status error")
        return json.dumps({"error": "Internal error"})


_SOURCE_HEALTH_SQL = """
WITH current_window AS (
    SELECT
        source,
        COUNT(*)                                            AS total_scrapes,
        COUNT(*) FILTER (WHERE status = 'success')          AS success_count,
        COUNT(*) FILTER (WHERE status = 'partial')          AS partial_count,
        COUNT(*) FILTER (WHERE status = 'failed')           AS failed_count,
        COUNT(*) FILTER (WHERE status = 'blocked')          AS blocked_count,
        AVG(reviews_found)                                  AS avg_reviews_found,
        AVG(reviews_inserted)                               AS avg_reviews_inserted,
        AVG(duration_ms)                                    AS avg_duration_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
        MAX(started_at) FILTER (WHERE status = 'success')   AS last_success_at,
        MAX(started_at)                                     AS last_scrape_at
    FROM b2b_scrape_log
    WHERE started_at >= NOW() - make_interval(days => $1)
    {source_filter_current}
    GROUP BY source
),
prev_window AS (
    SELECT
        source,
        COUNT(*)                                            AS total_scrapes,
        COUNT(*) FILTER (WHERE status = 'success')          AS success_count,
        COUNT(*) FILTER (WHERE status = 'blocked')          AS blocked_count,
        AVG(reviews_found)                                  AS avg_reviews_found
    FROM b2b_scrape_log
    WHERE started_at >= NOW() - make_interval(days => $1 * 2)
      AND started_at <  NOW() - make_interval(days => $1)
    {source_filter_prev}
    GROUP BY source
),
target_counts AS (
    SELECT source, COUNT(*) FILTER (WHERE enabled) AS active_targets
    FROM b2b_scrape_targets
    {target_filter}
    GROUP BY source
)
SELECT
    c.source, c.total_scrapes, c.success_count, c.partial_count,
    c.failed_count, c.blocked_count, c.avg_reviews_found,
    c.avg_reviews_inserted, c.avg_duration_ms, c.p95_duration_ms,
    c.last_success_at, c.last_scrape_at,
    COALESCE(t.active_targets, 0)  AS active_targets,
    p.total_scrapes                AS prev_total_scrapes,
    p.success_count                AS prev_success_count,
    p.blocked_count                AS prev_blocked_count,
    p.avg_reviews_found            AS prev_avg_reviews_found
FROM current_window c
LEFT JOIN prev_window p USING (source)
LEFT JOIN target_counts t USING (source)
ORDER BY c.total_scrapes DESC
"""


@mcp.tool()
async def get_source_health(
    window_days: int = 7,
    source: Optional[str] = None,
) -> str:
    """
    Per-source scrape reliability metrics with trend comparison.

    Aggregates b2b_scrape_log over a configurable window and computes
    success/block rates, yield, duration, and recency per review source.
    Includes trend deltas vs the previous equivalent window.

    window_days: Aggregation window in days (1-30, default 7)
    source: Filter to a single source (g2, capterra, trustradius, etc.)
    """
    window_days = max(1, min(window_days, 30))
    if source:
        source = source.strip().lower()
        if source not in VALID_SOURCES:
            return json.dumps({
                "success": False,
                "error": f"source must be one of {sorted(s.value for s in VALID_SOURCES)}",
            })

    try:
        from ...services.scraping.sources import display_name as source_display_name

        pool = get_pool()

        if source:
            sql = _SOURCE_HEALTH_SQL.format(
                source_filter_current="AND source = $2",
                source_filter_prev="AND source = $2",
                target_filter="WHERE source = $2",
            )
            rows = await pool.fetch(sql, window_days, source)
        else:
            sql = _SOURCE_HEALTH_SQL.format(
                source_filter_current="",
                source_filter_prev="",
                target_filter="",
            )
            rows = await pool.fetch(sql, window_days)

        def _float(v):
            if v is None:
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        sources_list = []
        for r in rows:
            total = r["total_scrapes"] or 1
            success_rate = round(r["success_count"] / total, 3)
            block_rate = round(r["blocked_count"] / total, 3)

            prev_total = r["prev_total_scrapes"] or 0
            prev_success_rate = round(r["prev_success_count"] / max(prev_total, 1), 3) if prev_total else None
            prev_block_rate = round(r["prev_blocked_count"] / max(prev_total, 1), 3) if prev_total else None

            sources_list.append({
                "source": r["source"],
                "display_name": source_display_name(r["source"]),
                "total_scrapes": r["total_scrapes"],
                "success_count": r["success_count"],
                "partial_count": r["partial_count"],
                "failed_count": r["failed_count"],
                "blocked_count": r["blocked_count"],
                "success_rate": success_rate,
                "block_rate": block_rate,
                "avg_reviews_found": _float(r["avg_reviews_found"]),
                "avg_reviews_inserted": _float(r["avg_reviews_inserted"]),
                "avg_duration_ms": _float(r["avg_duration_ms"]),
                "p95_duration_ms": _float(r["p95_duration_ms"]),
                "last_success_at": r["last_success_at"],
                "last_scrape_at": r["last_scrape_at"],
                "active_targets": r["active_targets"],
                "trend": {
                    "prev_window_scrapes": prev_total,
                    "prev_success_rate": prev_success_rate,
                    "prev_block_rate": prev_block_rate,
                    "prev_avg_reviews_found": _float(r["prev_avg_reviews_found"]),
                    "success_rate_delta": round(success_rate - prev_success_rate, 3) if prev_success_rate is not None else None,
                    "block_rate_delta": round(block_rate - prev_block_rate, 3) if prev_block_rate is not None else None,
                },
            })

        total_scrapes = sum(s["total_scrapes"] for s in sources_list)
        total_success = sum(s["success_count"] for s in sources_list)
        total_blocked = sum(s["blocked_count"] for s in sources_list)

        result = {
            "success": True,
            "window_days": window_days,
            "sources": sources_list,
            "summary": {
                "total_sources": len(sources_list),
                "total_scrapes": total_scrapes,
                "overall_success_rate": round(total_success / max(total_scrapes, 1), 3),
                "overall_block_rate": round(total_blocked / max(total_scrapes, 1), 3),
                "worst_source": min(sources_list, key=lambda s: s["success_rate"])["source"] if sources_list else None,
                "best_source": max(sources_list, key=lambda s: s["success_rate"])["source"] if sources_list else None,
            },
        }

        return json.dumps(result, default=str)
    except Exception as exc:
        logger.exception("get_source_health error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_source_telemetry(
    window_days: int = 7,
    source: Optional[str] = None,
) -> str:
    """
    Get CAPTCHA attempts, solve times, block type distribution, and proxy usage per source.

    window_days: How many days back to look (default 7, max 30)
    source: Filter to a single source (optional)
    """
    window_days = min(max(window_days, 1), 30)
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions = ["started_at >= NOW() - make_interval(days => $1)"]
        params: list = [window_days]
        idx = 2
        if source:
            conditions.append(f"source = ${idx}")
            params.append(source.strip().lower())
            idx += 1

        where = " AND ".join(conditions)
        rows = await pool.fetch(
            f"""
            SELECT
                source,
                COUNT(*)                                                AS total_scrapes,
                SUM(COALESCE(captcha_attempts, 0))                      AS total_captcha_attempts,
                COUNT(*) FILTER (WHERE captcha_attempts > 0)            AS scrapes_with_captcha,
                AVG(captcha_solve_ms) FILTER (WHERE captcha_solve_ms > 0) AS avg_captcha_solve_ms,
                MAX(captcha_solve_ms)                                    AS max_captcha_solve_ms,
                COUNT(*) FILTER (WHERE block_type IS NOT NULL)           AS total_blocks,
                COUNT(*) FILTER (WHERE block_type = 'captcha')           AS blocks_captcha,
                COUNT(*) FILTER (WHERE block_type = 'ip_ban')            AS blocks_ip_ban,
                COUNT(*) FILTER (WHERE block_type = 'rate_limit')        AS blocks_rate_limit,
                COUNT(*) FILTER (WHERE block_type = 'waf')               AS blocks_waf,
                COUNT(*) FILTER (WHERE block_type = 'unknown')           AS blocks_unknown,
                COUNT(*) FILTER (WHERE proxy_type = 'datacenter')        AS proxy_datacenter,
                COUNT(*) FILTER (WHERE proxy_type = 'residential')       AS proxy_residential,
                COUNT(*) FILTER (WHERE proxy_type = 'none')              AS proxy_none
            FROM b2b_scrape_log
            WHERE {where}
            GROUP BY source
            ORDER BY total_captcha_attempts DESC
            """,
            *params,
        )

        sources_out = []
        for r in rows:
            total = r["total_scrapes"] or 1
            sources_out.append({
                "source": r["source"],
                "total_scrapes": r["total_scrapes"],
                "captcha": {
                    "total_attempts": r["total_captcha_attempts"] or 0,
                    "scrapes_with_captcha": r["scrapes_with_captcha"],
                    "captcha_rate": round(r["scrapes_with_captcha"] / total, 3),
                    "avg_solve_ms": round(float(r["avg_captcha_solve_ms"]), 0) if r["avg_captcha_solve_ms"] else None,
                    "max_solve_ms": r["max_captcha_solve_ms"],
                },
                "blocks": {
                    "total": r["total_blocks"],
                    "captcha": r["blocks_captcha"],
                    "ip_ban": r["blocks_ip_ban"],
                    "rate_limit": r["blocks_rate_limit"],
                    "waf": r["blocks_waf"],
                    "unknown": r["blocks_unknown"],
                },
                "proxy_usage": {
                    "datacenter": r["proxy_datacenter"],
                    "residential": r["proxy_residential"],
                    "none": r["proxy_none"],
                },
            })

        return json.dumps({
            "window_days": window_days,
            "sources": sources_out,
            "total_sources": len(sources_out),
        }, default=str)
    except Exception:
        logger.exception("get_source_telemetry error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def get_source_capabilities(
    source: Optional[str] = None,
) -> str:
    """
    Get capability profiles for scrape sources.

    Returns access patterns, anti-bot protection, proxy requirements,
    data quality tier, and concurrency class for each source.

    source: Optional source name to filter (e.g. "g2", "reddit"). Returns all if omitted.
    """
    from ...services.scraping.capabilities import get_all_capabilities, get_capability

    if source:
        source = source.strip().lower()
        profile = get_capability(source)
        if not profile:
            return json.dumps({
                "success": False,
                "error": f"Unknown source '{source}'. Use without source param to list all.",
            })
        return json.dumps({"success": True, "profile": profile.to_dict()})

    all_profiles = get_all_capabilities()
    return json.dumps({
        "success": True,
        "total": len(all_profiles),
        "profiles": [p.to_dict() for p in all_profiles.values()],
    })


@mcp.tool()
async def get_source_impact_ledger(
    source: Optional[str] = None,
    window_days: int = 90,
    include_field_baseline: bool = True,
    include_consumer_wiring: bool = True,
) -> str:
    """
    Return the source-by-pool impact ledger with optional live field and wiring baselines.

    source: Optional source filter
    window_days: Live field-baseline lookback window (default 90)
    include_field_baseline: Include live b2b_reviews field-coverage metrics when DB is ready
    include_consumer_wiring: Include downstream consumer wiring baseline
    """
    window_days = max(1, min(window_days, _MAX_IMPACT_WINDOW_DAYS))
    valid_source_names = sorted(
        str(member.value if hasattr(member, "value") else member)
        for member in VALID_SOURCES
    )
    if source:
        source = source.strip().lower()
        if source not in valid_source_names:
            return json.dumps({
                "success": False,
                "error": f"source must be one of {valid_source_names}",
            })

    try:
        ledger = build_source_impact_ledger(source=source)
        field_baseline = None
        if include_field_baseline:
            pool = get_pool()
            if pool.is_initialized:
                field_baseline = await summarize_source_field_baseline(
                    pool,
                    window_days=window_days,
                    source=source,
                )
            else:
                field_baseline = {
                    "available": False,
                    "reason": "Database not ready",
                    "window_days": window_days,
                    "source_filter": source,
                    "rows": [],
                    "summary": {
                        "total_sources": 0,
                        "total_reviews": 0,
                        "total_enriched_reviews": 0,
                    },
                }

        consumer_wiring = (
            get_consumer_wiring_baseline() if include_consumer_wiring else None
        )
        return json.dumps({
            "success": True,
            "window_days": window_days,
            "source_filter": source,
            "impact_summary": ledger["summary"],
            "sources": ledger["sources"],
            "field_baseline": field_baseline,
            "consumer_wiring": consumer_wiring,
        }, default=str)
    except Exception:
        logger.exception("get_source_impact_ledger error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_operational_overview() -> str:
    """
    Single snapshot combining pipeline status, source health, telemetry, and recent events.

    Returns data summary (total reviews, vendors tracked), enrichment pipeline counts,
    source health summary (7d), CAPTCHA/block telemetry (7d), and 10 most recent change events.
    """
    try:
        import asyncio as _aio
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        pipeline_row, health_rows, telemetry_row, event_rows, review_row = await _aio.gather(
            pool.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE enrichment_status = 'pending')   AS pending,
                    COUNT(*) FILTER (WHERE enrichment_status = 'enriched')  AS enriched,
                    COUNT(*) FILTER (WHERE enrichment_status = 'failed')    AS failed,
                    COUNT(*)                                                 AS total
                FROM b2b_reviews
                WHERE duplicate_of_review_id IS NULL
            """),
            pool.fetch("""
                SELECT source, COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE status = 'success') AS success,
                       COUNT(*) FILTER (WHERE status = 'blocked') AS blocked
                FROM b2b_scrape_log
                WHERE started_at >= NOW() - INTERVAL '7 days'
                GROUP BY source ORDER BY total DESC
            """),
            pool.fetchrow("""
                SELECT
                    SUM(COALESCE(captcha_attempts, 0)) AS captcha_total,
                    COUNT(*) FILTER (WHERE block_type IS NOT NULL) AS blocks_total
                FROM b2b_scrape_log
                WHERE started_at >= NOW() - INTERVAL '7 days'
            """),
            pool.fetch("""
                SELECT vendor_name, event_type, event_date, description
                FROM b2b_change_events
                ORDER BY event_date DESC, created_at DESC
                LIMIT 10
            """),
            pool.fetchrow("""
                SELECT COUNT(*) AS total_reviews,
                       COUNT(DISTINCT vm.vendor_name) AS vendors_tracked
                FROM b2b_reviews r
                JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
                WHERE r.duplicate_of_review_id IS NULL
            """),
        )

        total_scrapes = sum(r["total"] for r in health_rows)
        total_success = sum(r["success"] for r in health_rows)

        return json.dumps({
            "data_summary": {
                "total_reviews": review_row["total_reviews"],
                "vendors_tracked": review_row["vendors_tracked"],
            },
            "pipeline": {
                "pending": pipeline_row["pending"],
                "enriched": pipeline_row["enriched"],
                "failed": pipeline_row["failed"],
                "total": pipeline_row["total"],
            },
            "source_health_7d": {
                "total_scrapes": total_scrapes,
                "overall_success_rate": round(total_success / max(total_scrapes, 1), 3),
                "sources": [{"source": r["source"], "total": r["total"],
                             "success_rate": round(r["success"] / max(r["total"], 1), 3)}
                            for r in health_rows],
            },
            "telemetry_7d": {
                "captcha_attempts": telemetry_row["captcha_total"] or 0,
                "blocks": telemetry_row["blocks_total"] or 0,
            },
            "recent_events": [
                {"vendor": r["vendor_name"], "type": r["event_type"],
                 "date": str(r["event_date"]), "description": r["description"]}
                for r in event_rows
            ],
        }, default=str)
    except Exception:
        logger.exception("get_operational_overview error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def get_parser_health() -> str:
    """
    Show parser version distribution and identify stale reviews.

    Returns per-source counts of reviews by parser version, the latest version
    for each source, and a stale count where reviews were parsed by an older
    version than the latest for that source.
    """
    pool = get_pool()
    if not pool.is_initialized:
        return json.dumps({"success": False, "error": "Database not ready"})
    try:
        rows = await pool.fetch("""
            WITH version_counts AS (
                SELECT source,
                       COALESCE(parser_version, 'unknown') AS parser_version,
                       COUNT(*) AS review_count
                FROM b2b_reviews
                GROUP BY source, COALESCE(parser_version, 'unknown')
            ),
            latest AS (
                SELECT DISTINCT ON (source) source, parser_version AS latest_version
                FROM b2b_scrape_log
                WHERE parser_version IS NOT NULL
                ORDER BY source, started_at DESC
            )
            SELECT vc.source, vc.parser_version, vc.review_count,
                   l.latest_version,
                   (vc.parser_version != COALESCE(l.latest_version, vc.parser_version))
                       AS is_stale
            FROM version_counts vc
            LEFT JOIN latest l USING (source)
            ORDER BY vc.source, vc.review_count DESC
        """)

        sources: dict[str, dict] = {}
        for r in rows:
            src = r["source"]
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "latest_version": r["latest_version"],
                    "versions": [],
                    "total_reviews": 0,
                    "stale_reviews": 0,
                }
            entry = sources[src]
            entry["versions"].append({
                "parser_version": r["parser_version"],
                "review_count": r["review_count"],
                "is_stale": r["is_stale"],
            })
            entry["total_reviews"] += r["review_count"]
            if r["is_stale"]:
                entry["stale_reviews"] += r["review_count"]

        result = sorted(sources.values(), key=lambda x: x["stale_reviews"], reverse=True)
        total_stale = sum(s["stale_reviews"] for s in result)
        return json.dumps({
            "success": True,
            "sources": result,
            "total_stale_reviews": total_stale,
            "total_sources": len(result),
        }, default=str)
    except Exception:
        logger.exception("get_parser_health error")
        return json.dumps({"success": False, "error": "Internal error"})
