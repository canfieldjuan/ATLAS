"""
Multi-target parallel scrape orchestrator.

Manages the full job lifecycle: DB record creation, concurrent target fan-out,
pagination handling, result storage, and status updates.

Supports real cancellation, durable job recovery on startup, and correct
partial-failure status reporting.
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
from .url_validation import validate_url

logger = logging.getLogger("atlas.services.scraping.universal.orchestrator")

# Number of consecutive pages with zero extracted items before stopping.
# Prevents false early termination from a single bad page.
_CONSECUTIVE_EMPTY_THRESHOLD = 2


class UniversalScraper:
    """Orchestrates universal scrape jobs with parallel target execution."""

    def __init__(self) -> None:
        # Track running jobs for real cancellation
        self._cancelled: set[UUID] = set()

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

    async def cancel_job(self, job_id: UUID) -> bool:
        """Signal a running job to stop. Returns True if the signal was sent."""
        self._cancelled.add(job_id)
        pool = get_db_pool()
        result = await pool.execute(
            """
            UPDATE universal_scrape_jobs
            SET status = 'cancelled', finished_at = now()
            WHERE id = $1 AND status IN ('pending', 'running')
            """,
            job_id,
        )
        return "UPDATE 1" in result

    def _is_cancelled(self, job_id: UUID) -> bool:
        return job_id in self._cancelled

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

        # Check if already cancelled before starting
        if self._is_cancelled(job_id):
            return

        await pool.execute(
            "UPDATE universal_scrape_jobs SET status = 'running', started_at = now() WHERE id = $1",
            job_id,
        )

        sem = asyncio.Semaphore(config.concurrency)
        lock = asyncio.Lock()
        completed = 0
        failed = 0
        total_records = 0

        async def _run_target(target: ScrapeTarget) -> None:
            nonlocal completed, failed, total_records

            # Check cancellation before starting target
            if self._is_cancelled(job_id):
                return

            async with sem:
                # Re-check after acquiring semaphore
                if self._is_cancelled(job_id):
                    return

                try:
                    records = await self._scrape_single_target(job_id, target, config)
                    async with lock:
                        completed += 1
                        total_records += records
                except Exception as exc:
                    logger.error("Target %s failed: %s", target.url, exc)
                    async with lock:
                        failed += 1
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

            # Update progress (only if not cancelled)
            if not self._is_cancelled(job_id):
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

        # Clean up cancellation tracking
        self._cancelled.discard(job_id)

        # Determine final status — only update if not already cancelled
        row = await pool.fetchrow(
            "SELECT status FROM universal_scrape_jobs WHERE id = $1", job_id
        )
        if row and row["status"] == "cancelled":
            # Already cancelled — don't overwrite
            return

        total_targets = len(config.targets)
        if failed == total_targets:
            final_status = "failed"
        elif failed > 0:
            final_status = "partial_success"
        else:
            final_status = "completed"

        await pool.execute(
            """
            UPDATE universal_scrape_jobs
            SET status = $2, finished_at = now(),
                completed_targets = $3, failed_targets = $4, total_records = $5
            WHERE id = $1 AND status != 'cancelled'
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

        # Pagination loop protection
        seen_urls: set[str] = set()
        consecutive_empty = 0

        for page_num in range(1, max_pages + 1):
            # Check cancellation before each page
            if self._is_cancelled(job_id):
                logger.info("Job %s cancelled, stopping target %s", job_id, target.url)
                break

            # Loop detection
            if current_url in seen_urls:
                logger.warning(
                    "Pagination loop detected on %s (URL repeated: %s), stopping",
                    target.url,
                    current_url,
                )
                break
            seen_urls.add(current_url)

            # Validate paginated URLs (first URL validated by Pydantic, but
            # subsequent URLs from patterns or CSS selectors need checking)
            if page_num > 1:
                try:
                    validate_url(current_url)
                except Exception as exc:
                    logger.warning(
                        "Paginated URL failed validation: %s — %s", current_url, exc
                    )
                    break

            t0 = time.monotonic()
            html: str | None = None
            page_error: str | None = None
            items: list[dict] = []
            raw_llm = ""
            page_title: str | None = None

            try:
                # 1. Fetch (with timeout)
                html = await asyncio.wait_for(
                    self._fetch_page(current_url, target, domain),
                    timeout=120,  # 2 minute hard timeout per page
                )

                # 2. Clean
                text = html_to_text(html, max_chars=max_chars)
                if not text or len(text.strip()) < 50:
                    logger.info(
                        "Page %d of %s has insufficient content",
                        page_num,
                        target.url,
                    )
                    page_error = "insufficient_content"
                else:
                    # 3. Title
                    page_title = self._extract_title(html)

                    # 4. LLM extraction
                    items, raw_llm = await extract_from_text(
                        text,
                        config.schema_def,
                        workload=config.llm_workload,
                        max_tokens=config.llm_max_tokens,
                    )

            except asyncio.TimeoutError:
                page_error = "fetch_timeout"
                logger.warning("Fetch timeout on page %d of %s", page_num, current_url)
            except Exception as exc:
                page_error = str(exc)
                logger.warning(
                    "Page %d of %s failed: %s", page_num, current_url, exc
                )

            duration_ms = int((time.monotonic() - t0) * 1000)

            # 5. Store — use current_url (actual page), not target.url
            await pool.execute(
                """
                INSERT INTO universal_scrape_results
                    (job_id, target_url, page_number, page_title,
                     extracted_data, item_count, raw_llm_response,
                     duration_ms, error)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9)
                ON CONFLICT (job_id, target_url, page_number) DO NOTHING
                """,
                job_id,
                current_url,
                page_num,
                page_title,
                json.dumps(items),
                len(items),
                raw_llm if config.store_raw_llm else None,
                duration_ms,
                page_error,
            )
            total_items += len(items)

            # Track consecutive empty pages (errors count as empty)
            if not items:
                consecutive_empty += 1
                if consecutive_empty >= _CONSECUTIVE_EMPTY_THRESHOLD:
                    logger.info(
                        "%d consecutive empty pages on %s, stopping pagination",
                        consecutive_empty,
                        target.url,
                    )
                    break
            else:
                consecutive_empty = 0

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
                if not target.pagination.css_selector or html is None:
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
        """Fetch page HTML using the configured client.

        Validates redirect targets to prevent SSRF via open redirects.
        """
        from .url_validation import validate_redirect_url

        if target.use_browser:
            browser = get_stealth_browser()
            result = await browser.scrape_page(
                url, wait_for_selector=target.wait_for_selector
            )
            # Browser may have followed redirects — validate final URL
            if result.url and result.url != url:
                validate_redirect_url(result.url, url)
            return result.html

        client = get_scrape_client()
        resp = await client.get(
            url,
            domain=domain,
            extra_headers=target.extra_headers,
            prefer_residential=target.prefer_residential,
            sticky_session=target.sticky_session,
        )
        # curl_cffi may follow redirects — validate final URL
        if hasattr(resp, "url") and resp.url and str(resp.url) != url:
            validate_redirect_url(str(resp.url), url)
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
    """Load a ScrapeJobConfig from a JSON file, restricted to the configured config_dir.

    Resolves the path and verifies it falls within the allowed directory
    to prevent path traversal attacks.
    """
    from ....config import settings

    config_dir = Path(settings.universal_scrape.config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)

    requested = Path(path)
    # If relative, resolve against config_dir
    if not requested.is_absolute():
        resolved = (config_dir / requested).resolve()
    else:
        resolved = requested.resolve()

    # Must be within config_dir
    try:
        resolved.relative_to(config_dir)
    except ValueError:
        raise PermissionError(
            f"Config file must be inside {config_dir}, "
            f"got {resolved}"
        )

    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Not a file: {resolved}")

    raw = json.loads(resolved.read_text())
    return ScrapeJobConfig.model_validate(raw)


async def reconcile_orphaned_jobs() -> int:
    """Mark any running/pending jobs as failed on startup.

    Called during application lifespan to recover from crashes where
    background tasks were lost. Returns the number of jobs reconciled.
    """
    pool = get_db_pool()
    if not pool.is_initialized:
        return 0

    result = await pool.execute(
        """
        UPDATE universal_scrape_jobs
        SET status = 'failed',
            error = 'Process restarted — job was orphaned',
            finished_at = now()
        WHERE status IN ('pending', 'running')
        """
    )
    # Parse "UPDATE N" from result
    try:
        count = int(result.split()[-1])
    except (ValueError, IndexError):
        count = 0

    if count > 0:
        logger.warning("Reconciled %d orphaned universal scrape jobs", count)

    return count
