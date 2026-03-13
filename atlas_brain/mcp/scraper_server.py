"""
Atlas Universal Scraper MCP Server.

Tools:
    scrape_url          -- scrape a single URL and extract data
    scrape_multi        -- scrape multiple URLs with shared schema
    get_scrape_job      -- check job status
    get_scrape_results  -- retrieve extracted data
    list_scrape_jobs    -- list recent jobs

Run:
    python -m atlas_brain.mcp.scraper_server          # stdio (Claude Desktop/Cursor)
    python -m atlas_brain.mcp.scraper_server --sse    # SSE HTTP (port 8063)
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.scraper")


@asynccontextmanager
async def _lifespan(server):
    from ..storage.database import init_database, close_database

    await init_database()
    logger.info("Scraper MCP: database pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-scraper",
    instructions=(
        "Universal web scraper. Extract ANY structured data from ANY website "
        "using LLM-powered extraction. Define what to extract via a schema "
        "(natural language or JSON field definitions). Supports pagination, "
        "JS-rendered pages (Playwright), and parallel multi-site scraping."
    ),
    lifespan=_lifespan,
)


# ── Tools ────────────────────────────────────────────────────────────


@mcp.tool()
async def scrape_url(
    url: str,
    extract: str,
    use_browser: bool = False,
    wait_for_selector: Optional[str] = None,
    max_pages: int = 1,
    pagination_url_pattern: Optional[str] = None,
    pagination_css_selector: Optional[str] = None,
) -> str:
    """Scrape a single URL and extract structured data.

    Args:
        url: The page to scrape.
        extract: What to extract (natural language),
                 e.g. "product name, price, rating, review count".
        use_browser: True for JS-rendered pages (React/Angular/SPA),
                     False for static HTML (faster).
        wait_for_selector: CSS selector to wait for before extracting
                           (only with use_browser=True).
        max_pages: Number of pages to follow (default 1 = single page).
        pagination_url_pattern: URL with {page} placeholder,
                                e.g. "https://example.com/list?page={page}".
        pagination_css_selector: CSS selector for a "next page" link.
    """
    from ..services.scraping.universal.schemas import (
        ExtractionSchema,
        PaginationConfig,
        PaginationStrategy,
        ScrapeJobConfig,
        ScrapeTarget,
    )
    from ..services.scraping.universal.orchestrator import get_universal_scraper
    from ..storage.database import get_db_pool

    # Build pagination config
    if pagination_url_pattern:
        pagination = PaginationConfig(
            strategy=PaginationStrategy.URL_PATTERN,
            url_pattern=pagination_url_pattern,
            max_pages=max_pages,
        )
    elif pagination_css_selector:
        pagination = PaginationConfig(
            strategy=PaginationStrategy.CSS_SELECTOR,
            css_selector=pagination_css_selector,
            max_pages=max_pages,
        )
    else:
        pagination = PaginationConfig(strategy=PaginationStrategy.NONE, max_pages=1)

    config = ScrapeJobConfig(
        name=f"mcp-scrape-{url[:60]}",
        schema=ExtractionSchema(description=extract),
        targets=[
            ScrapeTarget(
                url=url,
                use_browser=use_browser,
                wait_for_selector=wait_for_selector,
                pagination=pagination,
            )
        ],
        concurrency=1,
    )

    try:
        scraper = get_universal_scraper()
        job_id = await scraper.run_job_sync(config)

        pool = get_db_pool()
        rows = await pool.fetch(
            """
            SELECT extracted_data, item_count, page_number, error, duration_ms
            FROM universal_scrape_results WHERE job_id = $1
            ORDER BY page_number
            """,
            job_id,
        )

        all_items: list = []
        for r in rows:
            if r["extracted_data"]:
                data = (
                    json.loads(r["extracted_data"])
                    if isinstance(r["extracted_data"], str)
                    else r["extracted_data"]
                )
                all_items.extend(data)

        return json.dumps(
            {
                "job_id": str(job_id),
                "items": all_items,
                "total_items": len(all_items),
                "pages_scraped": len(rows),
                "errors": [r["error"] for r in rows if r["error"]],
            },
            default=str,
        )
    except Exception as exc:
        logger.exception("scrape_url failed")
        return json.dumps({"error": str(exc), "items": [], "total_items": 0})


@mcp.tool()
async def scrape_multi(
    urls: str,
    extract: str,
    use_browser: bool = False,
    concurrency: int = 3,
) -> str:
    """Scrape multiple URLs in parallel with the same extraction schema.

    For 4+ URLs or multi-page jobs, runs asynchronously and returns the
    job_id. Use get_scrape_job and get_scrape_results to poll for completion.
    For 1-3 single-page URLs, runs synchronously and returns data inline.

    Args:
        urls: Comma-separated list of URLs to scrape.
        extract: What to extract (natural language).
        use_browser: True for JS-rendered pages.
        concurrency: Max parallel scrapes (default 3, max 10).
    """
    from ..services.scraping.universal.schemas import (
        ExtractionSchema,
        ScrapeJobConfig,
        ScrapeTarget,
    )
    from ..services.scraping.universal.orchestrator import get_universal_scraper
    from ..storage.database import get_db_pool

    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    if not url_list:
        return json.dumps({"error": "No URLs provided"})

    targets = [ScrapeTarget(url=u, use_browser=use_browser) for u in url_list]

    config = ScrapeJobConfig(
        name=f"mcp-multi-{len(url_list)}-sites",
        schema=ExtractionSchema(description=extract),
        targets=targets,
        concurrency=min(concurrency, 10),
    )

    try:
        scraper = get_universal_scraper()

        # Small jobs run sync, large jobs run async
        is_small = len(url_list) <= 3
        if is_small:
            job_id = await scraper.run_job_sync(config)
        else:
            job_id = await scraper.run_job(config)
            return json.dumps(
                {
                    "job_id": str(job_id),
                    "status": "running",
                    "message": (
                        f"Job submitted with {len(url_list)} targets. "
                        "Use get_scrape_job and get_scrape_results to poll."
                    ),
                    "total_targets": len(url_list),
                },
                default=str,
            )

        pool = get_db_pool()
        job = await pool.fetchrow(
            "SELECT * FROM universal_scrape_jobs WHERE id = $1", job_id
        )
        results = await pool.fetch(
            """
            SELECT target_url, extracted_data, item_count, error
            FROM universal_scrape_results WHERE job_id = $1
            """,
            job_id,
        )

        by_url: dict[str, list] = {}
        for r in results:
            url = r["target_url"]
            data = (
                json.loads(r["extracted_data"])
                if isinstance(r["extracted_data"], str)
                else r["extracted_data"]
            )
            by_url.setdefault(url, []).extend(data)

        return json.dumps(
            {
                "job_id": str(job_id),
                "status": job["status"],
                "results_by_url": by_url,
                "total_items": job["total_records"],
                "completed": job["completed_targets"],
                "failed": job["failed_targets"],
            },
            default=str,
        )
    except Exception as exc:
        logger.exception("scrape_multi failed")
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def get_scrape_job(job_id: str) -> str:
    """Get the status of a scrape job by its UUID."""
    from ..storage.database import get_db_pool
    from uuid import UUID

    try:
        uid = UUID(job_id)
    except ValueError:
        return json.dumps({"error": "Invalid job_id (must be UUID)"})

    pool = get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT id, name, status, total_targets, completed_targets,
               failed_targets, total_records, error, started_at, finished_at, created_at
        FROM universal_scrape_jobs WHERE id = $1
        """,
        uid,
    )
    if not row:
        return json.dumps({"error": "Job not found"})
    return json.dumps(dict(row), default=str)


@mcp.tool()
async def get_scrape_results(job_id: str, limit: int = 100) -> str:
    """Get extracted data from a completed scrape job."""
    from ..storage.database import get_db_pool
    from uuid import UUID

    try:
        uid = UUID(job_id)
    except ValueError:
        return json.dumps({"error": "Invalid job_id"})

    pool = get_db_pool()
    rows = await pool.fetch(
        """
        SELECT target_url, page_number, extracted_data, item_count, error
        FROM universal_scrape_results WHERE job_id = $1
        ORDER BY target_url, page_number
        LIMIT $2
        """,
        uid,
        min(limit, 500),
    )

    all_items: list = []
    for r in rows:
        data = (
            json.loads(r["extracted_data"])
            if isinstance(r["extracted_data"], str)
            else r["extracted_data"]
        )
        for item in data:
            item["_source_url"] = r["target_url"]
            item["_page"] = r["page_number"]
        all_items.extend(data)

    return json.dumps(
        {"items": all_items, "total": len(all_items)}, default=str
    )


@mcp.tool()
async def list_scrape_jobs(limit: int = 10, status: Optional[str] = None) -> str:
    """List recent scrape jobs, optionally filtered by status."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if status:
        rows = await pool.fetch(
            """
            SELECT id, name, status, total_targets, completed_targets,
                   total_records, created_at, finished_at
            FROM universal_scrape_jobs WHERE status = $1
            ORDER BY created_at DESC LIMIT $2
            """,
            status,
            min(limit, 50),
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, name, status, total_targets, completed_targets,
                   total_records, created_at, finished_at
            FROM universal_scrape_jobs
            ORDER BY created_at DESC LIMIT $1
            """,
            min(limit, 50),
        )
    return json.dumps([dict(r) for r in rows], default=str)


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from ..config import settings
        from .auth import run_sse_with_auth

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.scraper_port
        run_sse_with_auth(mcp, settings.mcp.host, settings.mcp.scraper_port)
    else:
        mcp.run(transport="stdio")
