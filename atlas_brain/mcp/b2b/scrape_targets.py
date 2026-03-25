"""B2B Churn MCP -- scrape target tools."""
import json
import uuid as _uuid
from typing import Optional

from ._shared import _is_uuid, _safe_json, logger, get_pool, VALID_SOURCES
from .server import mcp


@mcp.tool()
async def list_scrape_targets(
    source: Optional[str] = None,
    scrape_mode: Optional[str] = None,
    enabled_only: bool = True,
    limit: int = 20,
) -> str:
    """
    View scrape target configuration and last run status.

    source: Filter by source (g2, capterra, trustradius, reddit, gartner, getapp, github, hackernews, peerspot, producthunt, quora, rss, slashdot, stackoverflow, trustpilot, youtube)
    scrape_mode: Filter by mode -- "incremental" or "exhaustive" (optional)
    enabled_only: Only show enabled targets (default true)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    if source and source not in VALID_SOURCES:
        return json.dumps({"error": f"source must be one of {sorted(s.value for s in VALID_SOURCES)}", "targets": [], "count": 0})
    if scrape_mode and scrape_mode not in ("incremental", "exhaustive"):
        return json.dumps({"error": "scrape_mode must be 'incremental' or 'exhaustive'", "targets": [], "count": 0})

    try:
        pool = get_pool()
        conditions = []
        params = []
        idx = 1

        if enabled_only:
            conditions.append("enabled = true")

        if source:
            conditions.append(f"source = ${idx}")
            params.append(source)
            idx += 1

        if scrape_mode:
            conditions.append(f"scrape_mode = ${idx}")
            params.append(scrape_mode)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        capped = min(limit, 100)
        params.append(capped)

        rows = await pool.fetch(
            f"""
            SELECT id, source, vendor_name, product_name, product_category,
                   enabled, priority, scrape_mode, last_scraped_at, last_scrape_status,
                   last_scrape_reviews
            FROM b2b_scrape_targets
            {where}
            ORDER BY priority DESC, vendor_name ASC
            LIMIT ${idx}
            """,
            *params,
        )

        targets = [
            {
                "id": str(r["id"]),
                "source": r["source"],
                "vendor_name": r["vendor_name"],
                "product_name": r["product_name"],
                "product_category": r["product_category"],
                "enabled": r["enabled"],
                "priority": r["priority"],
                "scrape_mode": r["scrape_mode"],
                "last_scraped_at": r["last_scraped_at"],
                "last_scrape_status": r["last_scrape_status"],
                "last_scrape_reviews": r["last_scrape_reviews"],
            }
            for r in rows
        ]

        return json.dumps({"targets": targets, "count": len(targets)}, default=str)
    except Exception as exc:
        logger.exception("list_scrape_targets error")
        return json.dumps({"error": "Internal error", "targets": [], "count": 0})


@mcp.tool()
async def add_scrape_target(
    source: str,
    vendor_name: str,
    product_slug: str,
    product_name: Optional[str] = None,
    product_category: Optional[str] = None,
    max_pages: int = 5,
    priority: int = 0,
    scrape_interval_hours: int = 168,
    scrape_mode: str = "incremental",
    metadata_json: Optional[str] = None,
) -> str:
    """
    Add a new scrape target to monitor a vendor on a review source.

    source: Review source -- one of: g2, capterra, trustradius, reddit, gartner,
            getapp, github, hackernews, peerspot, producthunt, quora, rss,
            stackoverflow, trustpilot, twitter, youtube
    vendor_name: Vendor/company name (e.g. "Salesforce")
    product_slug: Format depends on source type:
        URL-slug sources (required -- used in URL construction):
          g2: "salesforce-crm"
          capterra: "61368/Salesforce" (numeric-id/name)
          trustradius: "salesforce-crm"
          gartner: "market-slug/vendor-slug" (slash-separated)
          peerspot: "monday-com"
          getapp: "project-management-software/a/monday-com" (category/a/product)
          producthunt: "my-product" (GraphQL slug)
          trustpilot: "monday.com" (company domain)
          slashdot: "slack"
        Search sources (informational -- vendor_name is used for search):
          reddit, hackernews, github, youtube, stackoverflow, quora, twitter:
          use vendor name as slug (e.g. "salesforce")
        Special:
          rss: full feed URL (e.g. "https://news.google.com/rss/search?q=salesforce")
    product_name: Optional product variant name
    product_category: Category (e.g. "CRM", "Project Management")
    max_pages: Pages to scrape per run (default 5, max 100)
    priority: Higher = scraped first (default 0, max 100)
    scrape_interval_hours: Re-scrape interval (default 168 = weekly, max 8760)
    scrape_mode: "incremental" (shallow, concurrent -- default) or "exhaustive" (deep pagination, sequential)
    metadata_json: Optional JSON string for source-specific config:
        reddit: '{"subreddits": ["sysadmin","projectmanagement"]}'
        twitter: '{"search_terms": ["salesforce down"], "min_likes": 2}'
        youtube: '{"search_terms": ["salesforce review"], "max_videos_per_query": 10}'
        hackernews: '{"min_points": 5, "include_comments": true}'
        github: '{"search_mode": "both", "min_stars": 10}'
        stackoverflow: '{"sites": "stackoverflow,softwarerecs", "min_score": 1}'
        rss: '{"feed_urls": ["https://..."], "keywords": ["migration","switching"]}'
        exhaustive: '{"lookback_days": 365}' (date cutoff for exhaustive mode)
    """
    from ...config import settings
    from ...services.scraping.target_validation import is_source_allowed, validate_target_input

    source = source.strip().lower()
    if source not in VALID_SOURCES:
        return json.dumps({"success": False, "error": f"source must be one of {sorted(s.value for s in VALID_SOURCES)}"})
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name is required"})
    if not product_slug or not product_slug.strip():
        return json.dumps({"success": False, "error": "product_slug is required"})
    if scrape_mode not in ("incremental", "exhaustive"):
        return json.dumps({"success": False, "error": "scrape_mode must be 'incremental' or 'exhaustive'"})
    if not is_source_allowed(source, settings.b2b_scrape.source_allowlist):
        return json.dumps({
            "success": False,
            "error": (
                f"Source '{source}' is currently disabled by "
                "ATLAS_B2B_SCRAPE_SOURCE_ALLOWLIST"
            ),
        })

    try:
        source, product_slug = validate_target_input(source, product_slug)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})

    max_pages = max(1, min(max_pages, 100))
    priority = max(0, min(priority, 100))
    scrape_interval_hours = max(1, min(scrape_interval_hours, 8760))

    # Resolve to canonical vendor name
    from ...services.vendor_registry import resolve_vendor_name
    vendor_name = await resolve_vendor_name(vendor_name)

    meta = {}
    if metadata_json:
        try:
            meta = json.loads(metadata_json)
            if not isinstance(meta, dict):
                return json.dumps({"success": False, "error": "metadata_json must be a JSON object"})
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"success": False, "error": "Invalid metadata_json"})

    try:
        pool = get_pool()

        # Check for duplicate (same source + slug + mode)
        existing = await pool.fetchrow(
            "SELECT id FROM b2b_scrape_targets WHERE source = $1 AND product_slug = $2 AND scrape_mode = $3",
            source, product_slug, scrape_mode,
        )
        if existing:
            return json.dumps({
                "success": False,
                "error": f"Target already exists for {source}/{product_slug}/{scrape_mode} (id: {existing['id']})",
            })

        row = await pool.fetchrow(
            """
            INSERT INTO b2b_scrape_targets
                (source, vendor_name, product_name, product_slug, product_category,
                 max_pages, priority, scrape_interval_hours, scrape_mode, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            RETURNING id, source, vendor_name, product_slug, enabled, priority, scrape_mode
            """,
            source,
            vendor_name.strip(),
            product_name.strip() if product_name else None,
            product_slug,
            product_category.strip() if product_category else None,
            max_pages,
            priority,
            scrape_interval_hours,
            scrape_mode,
            json.dumps(meta),
        )

        return json.dumps({
            "success": True,
            "target": {
                "id": str(row["id"]),
                "source": row["source"],
                "vendor_name": row["vendor_name"],
                "product_slug": row["product_slug"],
                "enabled": row["enabled"],
                "priority": row["priority"],
                "scrape_mode": row["scrape_mode"],
            },
        }, default=str)
    except Exception:
        logger.exception("add_scrape_target error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def manage_scrape_target(
    target_id: str,
    enabled: Optional[bool] = None,
    priority: Optional[int] = None,
    max_pages: Optional[int] = None,
    scrape_interval_hours: Optional[int] = None,
    scrape_mode: Optional[str] = None,
    metadata_json: Optional[str] = None,
) -> str:
    """
    Update a scrape target's settings.

    target_id: UUID of the scrape target (required)
    enabled: Set to true/false to enable/disable
    priority: Set priority (0-100, higher = scraped first)
    max_pages: Pages to scrape per run (1-100)
    scrape_interval_hours: Re-scrape interval in hours (1-8760)
    scrape_mode: "incremental" or "exhaustive"
    metadata_json: Replace source-specific config JSON (e.g. subreddits for reddit)
    """
    if not _is_uuid(target_id):
        return json.dumps({"success": False, "error": "Invalid target_id (must be UUID)"})

    if all(v is None for v in [enabled, priority, max_pages, scrape_interval_hours, scrape_mode, metadata_json]):
        return json.dumps({"success": False, "error": "Provide at least one field to update"})

    if scrape_mode is not None and scrape_mode not in ("incremental", "exhaustive"):
        return json.dumps({"success": False, "error": "scrape_mode must be 'incremental' or 'exhaustive'"})

    try:
        pool = get_pool()

        sets = ["updated_at = NOW()"]
        params = []
        idx = 1

        if enabled is not None:
            sets.append(f"enabled = ${idx}")
            params.append(enabled)
            idx += 1

        if priority is not None:
            sets.append(f"priority = ${idx}")
            params.append(max(0, min(priority, 100)))
            idx += 1

        if max_pages is not None:
            sets.append(f"max_pages = ${idx}")
            params.append(max(1, min(max_pages, 100)))
            idx += 1

        if scrape_interval_hours is not None:
            sets.append(f"scrape_interval_hours = ${idx}")
            params.append(max(1, min(scrape_interval_hours, 8760)))
            idx += 1

        if scrape_mode is not None:
            sets.append(f"scrape_mode = ${idx}")
            params.append(scrape_mode)
            idx += 1

        if metadata_json is not None:
            try:
                meta = json.loads(metadata_json)
                if not isinstance(meta, dict):
                    return json.dumps({"success": False, "error": "metadata_json must be a JSON object"})
            except (json.JSONDecodeError, TypeError):
                return json.dumps({"success": False, "error": "Invalid metadata_json"})
            sets.append(f"metadata = ${idx}::jsonb")
            params.append(json.dumps(meta))
            idx += 1

        params.append(_uuid.UUID(target_id))

        try:
            row = await pool.fetchrow(
                f"""
                UPDATE b2b_scrape_targets
                SET {', '.join(sets)}
                WHERE id = ${idx}
                RETURNING id, source, vendor_name, product_name, product_slug,
                          enabled, priority, max_pages, scrape_interval_hours, scrape_mode
                """,
                *params,
            )
        except Exception as exc:
            if "idx_b2b_scrape_targets_dedup" in str(exc):
                return json.dumps({
                    "success": False,
                    "error": "A target with that source/product_slug/scrape_mode already exists",
                })
            raise

        if not row:
            return json.dumps({"success": False, "error": "Target not found"})

        return json.dumps({
            "success": True,
            "target": {
                "id": str(row["id"]),
                "source": row["source"],
                "vendor_name": row["vendor_name"],
                "product_name": row["product_name"],
                "product_slug": row["product_slug"],
                "enabled": row["enabled"],
                "priority": row["priority"],
                "max_pages": row["max_pages"],
                "scrape_interval_hours": row["scrape_interval_hours"],
                "scrape_mode": row["scrape_mode"],
            },
        }, default=str)
    except Exception:
        logger.exception("manage_scrape_target error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def delete_scrape_target(target_id: str) -> str:
    """
    Delete a scrape target and its associated scrape logs.

    target_id: UUID of the scrape target to delete (required)
    """
    if not _is_uuid(target_id):
        return json.dumps({"success": False, "error": "Invalid target_id (must be UUID)"})

    try:
        pool = get_pool()

        # Get target info before deleting
        row = await pool.fetchrow(
            "SELECT source, vendor_name, product_slug FROM b2b_scrape_targets WHERE id = $1",
            _uuid.UUID(target_id),
        )
        if not row:
            return json.dumps({"success": False, "error": "Target not found"})

        # Delete logs first (FK has no CASCADE), then target -- in a transaction
        async with pool.transaction() as conn:
            await conn.execute(
                "DELETE FROM b2b_scrape_log WHERE target_id = $1",
                _uuid.UUID(target_id),
            )
            await conn.execute(
                "DELETE FROM b2b_scrape_targets WHERE id = $1",
                _uuid.UUID(target_id),
            )

        return json.dumps({
            "success": True,
            "deleted": {
                "id": target_id,
                "source": row["source"],
                "vendor_name": row["vendor_name"],
                "product_slug": row["product_slug"],
            },
        })
    except Exception:
        logger.exception("delete_scrape_target error")
        return json.dumps({"success": False, "error": "Internal error"})
