"""
Multi-target parallel scrape orchestrator.

Manages the full job lifecycle: DB record creation, concurrent target fan-out,
pagination handling, result storage, and status updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from ....storage.database import get_db_pool
from ..browser import get_stealth_browser
from ..client import get_scrape_client
from .engine import extract_from_text
from .html_cleaner import extract_next_page_url, html_to_text
from .schemas import PaginationStrategy, ScrapeJobConfig, ScrapeTarget

logger = logging.getLogger("atlas.services.scraping.universal.orchestrator")


class UniversalScraper:
    """Orchestrates universal scrape jobs with parallel target execution."""

    # ── Public API ───────────────────────────────────────────────────

    async def run_job(self, config: ScrapeJobConfig) -> UUID:
        """Create a job record and start scraping in the background.

        Returns the ``job_id`` immediately.
        """
        job_id = await self._create_job_record(config)
        asyncio.create_task(self._execute_job(job_id, config))
        return job_id

    async def run_job_sync(self, config: ScrapeJobConfig) -> UUID:
        """Create a job record and execute synchronously (blocks until done).

        Useful for MCP tools that return results inline.
        """
        job_id = await self._create_job_record(config)
        await self._execute_job(job_id, config)
        return job_id

    # ── Job record ───────────────────────────────────────────────────

    async def _create_job_record(self, config: ScrapeJobConfig) -> UUID:
        pool = get_db_pool()
        if not pool.is_initialized:
            raise RuntimeError("Database not ready")

        row = await pool.fetchrow(
            """
            INSERT INTO universal_scrape_jobs (name, status, config, total_targets)
            VALUES ($1, 'pending', $2::jsonb, $3)
            RETURNING id
            """,
            config.name,
            config.model_dump_json(by_alias=True),
            len(config.targets),
        )
        return row["id"]

    # ── Job execution ────────────────────────────────────────────────

    async def _execute_job(self, job_id: UUID, config: ScrapeJobConfig) -> None:
        """Fan out targets with concurrency control, update DB on progress."""
        pool = get_db_pool()

        await pool.execute(
            "UPDATE universal_scrape_jobs SET status = 'running', started_at = now() WHERE id = $1",
            job_id,
        )

        sem = asyncio.Semaphore(config.concurrency)
        # Use a lock for counter updates to avoid races with gather
        lock = asyncio.Lock()
        completed = 0
        failed = 0
        total_records = 0

        async def _run_target(target: ScrapeTarget) -> None:
            nonlocal completed, failed, total_records
            async with sem:
                try:
                    records = await self._scrape_single_target(job_id, target, config)
                    async with lock:
                        completed += 1
                        total_records += records
                except Exception as exc:
                    logger.error("Target %s failed: %s", target.url, exc)
                    async with lock:
                        failed += 1
                    # Store the error as a result row
                    await pool.execute(
                        """
                        INSERT INTO universal_scrape_results
                            (job_id, target_url, extracted_data, item_count, error)
                        VALUES ($1, $2, '[]'::jsonb, 0, $3)
                        """,
                        job_id,
                        target.url,
                        str(exc),
                    )

            # Update progress after each target
            async with lock:
                c, f, t = completed, failed, total_records
            await pool.execute(
                """
                UPDATE universal_scrape_jobs
                SET completed_targets = $2, failed_targets = $3, total_records = $4
                WHERE id = $1
                """,
                job_id,
                c,
                f,
                t,
            )

        await asyncio.gather(
            *[_run_target(t) for t in config.targets],
            return_exceptions=True,
        )

        # Final status
        final_status = (
            "failed"
            if failed == len(config.targets)
            else "completed"
        )
        await pool.execute(
            """
            UPDATE universal_scrape_jobs
            SET status = $2, finished_at = now(),
                completed_targets = $3, failed_targets = $4, total_records = $5
            WHERE id = $1
            """,
            job_id,
            final_status,
            completed,
            failed,
            total_records,
        )

    # ── Single target (with pagination) ──────────────────────────────

    async def _scrape_single_target(
        self, job_id: UUID, target: ScrapeTarget, config: ScrapeJobConfig
    ) -> int:
        """Scrape one target through all its pages. Returns total items."""
        from ....config import settings

        pool = get_db_pool()
        domain = urlparse(target.url).hostname or "unknown"
        total_items = 0
        current_url = target.url
        max_pages = min(
            target.pagination.max_pages,
            settings.universal_scrape.max_pages_limit,
        )
        max_chars = settings.universal_scrape.html_max_chars

        for page_num in range(1, max_pages + 1):
            t0 = time.monotonic()

            # 1. Fetch
            html = await self._fetch_page(current_url, target, domain)

            # 2. Clean
            text = html_to_text(html, max_chars=max_chars)
            if not text or len(text.strip()) < 50:
                logger.info(
                    "Page %d of %s has insufficient content, stopping",
                    page_num,
                    target.url,
                )
                break

            # 3. Title
            page_title = self._extract_title(html)

            # 4. LLM extraction
            items, raw_llm = await extract_from_text(
                text,
                config.schema_def,
                workload=config.llm_workload,
                max_tokens=config.llm_max_tokens,
            )

            duration_ms = int((time.monotonic() - t0) * 1000)

            # 5. Store
            await pool.execute(
                """
                INSERT INTO universal_scrape_results
                    (job_id, target_url, page_number, page_title,
                     extracted_data, item_count, raw_llm_response, duration_ms)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
                """,
                job_id,
                target.url,
                page_num,
                page_title,
                json.dumps(items),
                len(items),
                raw_llm if config.store_raw_llm else None,
                duration_ms,
            )
            total_items += len(items)

            # Stop if nothing was extracted
            if not items:
                logger.info(
                    "No items on page %d of %s, stopping pagination",
                    page_num,
                    target.url,
                )
                break

            # 6. Pagination
            if target.pagination.strategy == PaginationStrategy.NONE:
                break
            elif target.pagination.strategy == PaginationStrategy.URL_PATTERN:
                if not target.pagination.url_pattern:
                    break
                current_url = target.pagination.url_pattern.format(
                    page=page_num + 1
                )
            elif target.pagination.strategy == PaginationStrategy.CSS_SELECTOR:
                if not target.pagination.css_selector:
                    break
                next_url = extract_next_page_url(
                    html, target.pagination.css_selector, current_url
                )
                if not next_url:
                    logger.info(
                        "No next-page link on page %d, stopping", page_num
                    )
                    break
                current_url = next_url

        return total_items

    # ── Fetching ─────────────────────────────────────────────────────

    async def _fetch_page(
        self, url: str, target: ScrapeTarget, domain: str
    ) -> str:
        """Fetch page HTML using the configured client."""
        if target.use_browser:
            browser = get_stealth_browser()
            result = await browser.scrape_page(
                url, wait_for_selector=target.wait_for_selector
            )
            return result.html

        client = get_scrape_client()
        resp = await client.get(
            url,
            domain=domain,
            extra_headers=target.extra_headers,
            prefer_residential=target.prefer_residential,
            sticky_session=target.sticky_session,
        )
        return resp.text

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_title(html: str) -> str | None:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else None


# ── Module singleton ─────────────────────────────────────────────────

_instance: UniversalScraper | None = None


def get_universal_scraper() -> UniversalScraper:
    """Return the module-level singleton."""
    global _instance
    if _instance is None:
        _instance = UniversalScraper()
    return _instance


def load_config_file(path: str | Path) -> ScrapeJobConfig:
    """Load a ScrapeJobConfig from a JSON file on disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = json.loads(p.read_text())
    return ScrapeJobConfig.model_validate(raw)
