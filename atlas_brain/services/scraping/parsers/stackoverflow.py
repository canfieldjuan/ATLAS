"""
Stack Overflow parser for B2B review scraping.

Uses the Stack Exchange API v2.3 (api.stackexchange.com) -- free, no auth for
basic queries, 300 requests/day without an API key.

Also queries Software Recommendations SE (softwarerecs.stackexchange.com) which
is highly valuable for "switching from X" and "alternatives to X" discussions.

Signal value: Integration pain points, migration stories, developer complaints,
alternative-seeking behaviour.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from datetime import datetime, timezone
from urllib.parse import quote_plus

from ..client import AntiDetectionClient
from . import (
    ScrapeResult,
    ScrapeTarget,
    apply_date_cutoff,
    page_has_only_known_source_reviews,
    register_parser,
)

logger = logging.getLogger("atlas.services.scraping.parsers.stackoverflow")

_DOMAIN = "api.stackexchange.com"
_BASE_URL = "https://api.stackexchange.com/2.3"
_MIN_TEXT_LEN = 100
_PAGE_SIZE = 50
_MAX_ANSWER_BATCH = 30  # Max question IDs per /answers request (API allows 100)
_MIN_SCORE_FOR_ANSWERS = 5  # Only fetch answers for questions above this score
_THROTTLE_BACKOFF_S = 30  # Seconds to wait on HTTP 502 (API throttle)

# HTML tag stripper (SE API returns HTML in body fields)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Default search suffixes appended to the vendor name
_SEARCH_SUFFIXES = ["", "alternative", "migration"]


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    return html.unescape(_HTML_TAG_RE.sub("", text)).strip()


def _epoch_to_iso(epoch: int | None) -> str | None:
    """Convert a Unix epoch to an ISO-8601 UTC string, or None."""
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    except (OSError, ValueError):
        return None


def _cutoff_epoch(date_cutoff: str | None) -> int | None:
    """Convert a YYYY-MM-DD cutoff into a UTC epoch for Stack Exchange APIs."""
    if not date_cutoff:
        return None
    try:
        dt = datetime.fromisoformat(date_cutoff)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _get_api_key() -> str:
    """Load Stack Exchange API key from config or env."""
    try:
        from ....config import settings
        return settings.b2b_scrape.stackoverflow_api_key
    except Exception:
        pass
    import os
    return os.environ.get("ATLAS_B2B_SCRAPE_STACKOVERFLOW_API_KEY", "")


class StackOverflowParser:
    """Parse Stack Overflow and Software Recommendations SE as B2B churn signals."""

    source_name = "stackoverflow"
    prefer_residential = False
    version = "stackoverflow:1"

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _append_key(self, url: str) -> str:
        """Append API key to URL if configured (raises quota from 300 to 10k/day)."""
        key = _get_api_key()
        if key:
            sep = "&" if "?" in url else "?"
            return f"{url}{sep}key={key}"
        return url

    async def _api_get(
        self,
        client: AntiDetectionClient,
        url: str,
    ):
        """Issue a GET against the SE API, returning parsed JSON or None.

        Handles HTTP 502 (throttle) with a single back-off retry.
        Returns ``(data_dict | None, error_str | None)``.
        """
        for attempt in range(2):
            resp = await client.get(
                url,
                domain=_DOMAIN,
                referer="https://stackoverflow.com/",
                sticky_session=False,
                prefer_residential=False,
            )

            if resp.status_code == 502 and attempt == 0:
                logger.warning("SE API throttled (502), backing off %ds", _THROTTLE_BACKOFF_S)
                await asyncio.sleep(_THROTTLE_BACKOFF_S)
                continue

            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}"

            try:
                return resp.json(), None
            except (ValueError, TypeError):
                return None, "non-parseable JSON"

        return None, "throttled after retry"

    async def _fetch_answers(
        self,
        client: AntiDetectionClient,
        question_ids: list[int],
        site: str,
    ) -> dict[int, list[dict]]:
        """Fetch answers for a batch of question IDs.

        Returns a mapping of question_id -> list of answer dicts.
        """
        if not question_ids:
            return {}

        ids_str = ";".join(str(qid) for qid in question_ids[:_MAX_ANSWER_BATCH])
        url = self._append_key(
            f"{_BASE_URL}/questions/{ids_str}/answers"
            f"?site={site}&sort=votes&order=desc&filter=withbody&pagesize=100"
        )

        data, err = await self._api_get(client, url)
        if data is None:
            logger.warning("Failed to fetch answers for %s: %s", site, err)
            return {}

        result: dict[int, list[dict]] = {}
        for item in data.get("items", []):
            qid = item.get("question_id")
            if qid is not None:
                result.setdefault(qid, []).append(item)
        return result

    # --------------------------------------------------------------------- #
    # Main scrape
    # --------------------------------------------------------------------- #

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape SE sites for questions and answers mentioning the vendor."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        stop_reason = ""

        # Configurable via target.metadata
        include_answers = target.metadata.get("include_answers", True)
        sites = target.metadata.get("sites", ["stackoverflow", "softwarerecs"])
        if isinstance(sites, str):
            sites = [s.strip() for s in sites.split(",")]

        extra_suffixes = target.metadata.get("search_suffixes")
        suffixes = list(_SEARCH_SUFFIXES)
        if extra_suffixes and isinstance(extra_suffixes, list):
            suffixes.extend(extra_suffixes[:3])

        try:
            min_score = int(target.metadata.get("min_score", 1))
        except (ValueError, TypeError):
            min_score = 1

        for site in sites:
            for suffix in suffixes:
                term = f'"{target.vendor_name}"'
                if suffix:
                    term = f'"{target.vendor_name}" {suffix}'

                consecutive_empty = 0
                high_score_qids: list[int] = []
                search_stop_reason = ""

                for page in range(1, target.max_pages + 1):
                    cutoff_epoch = _cutoff_epoch(target.date_cutoff)
                    url = self._append_key(
                        f"{_BASE_URL}/search/advanced"
                        f"?q={quote_plus(term)}"
                        f"&site={site}"
                        f"&sort=votes&order=desc"
                        f"&filter=withbody"
                        f"&pagesize={_PAGE_SIZE}"
                        f"&page={page}"
                        f"{f'&fromdate={cutoff_epoch}' if cutoff_epoch is not None else ''}"
                    )

                    try:
                        before = len(reviews)
                        data, err = await self._api_get(client, url)
                        pages_scraped += 1

                        if data is None:
                            errors.append(f"SE {site} search '{term}' page {page}: {err}")
                            break

                        items = data.get("items", [])
                        if not items:
                            break

                        page_reviews: list[dict] = []
                        for item in items:
                            qid = item.get("question_id")
                            review_id = f"{site}_q_{qid}"

                            if review_id in seen_ids:
                                continue

                            score = item.get("score", 0)
                            if score < min_score:
                                continue

                            body_html = item.get("body") or ""
                            text = _strip_html(body_html)
                            title = item.get("title", "")

                            # Use title + body as full text
                            full_text = f"{title}\n\n{text}" if text else title
                            if len(full_text) < _MIN_TEXT_LEN:
                                continue

                            seen_ids.add(review_id)

                            owner = item.get("owner") or {}
                            tags = item.get("tags", [])

                            page_reviews.append({
                                "source": "stackoverflow",
                                "source_url": item.get("link", f"https://stackoverflow.com/q/{qid}"),
                                "source_review_id": review_id,
                                "vendor_name": target.vendor_name,
                                "product_name": target.product_name,
                                "product_category": target.product_category,
                                "rating": None,
                                "rating_max": 5,
                                "summary": title[:500],
                                "review_text": full_text[:10000],
                                "pros": None,
                                "cons": None,
                                "reviewer_name": owner.get("display_name", ""),
                                "reviewer_title": None,
                                "reviewer_company": None,
                                "company_size_raw": None,
                                "reviewer_industry": None,
                                "reviewed_at": _epoch_to_iso(item.get("creation_date")),
                                "raw_metadata": {
                                    "extraction_method": "api_json",
                                    "source_weight": 0.5,
                                    "source_type": "developer_qa_platform",
                                    "score": score,
                                    "answer_count": item.get("answer_count", 0),
                                    "view_count": item.get("view_count", 0),
                                    "tags": tags,
                                    "site": site,
                                    "search_term": term,
                                    "content_type": "question",
                                },
                            })

                            # Collect high-score question IDs for answer fetch
                            if include_answers and score >= _MIN_SCORE_FOR_ANSWERS:
                                high_score_qids.append(qid)

                        if target.date_cutoff:
                            page_reviews, cutoff_hit = apply_date_cutoff(page_reviews, target.date_cutoff)
                        else:
                            cutoff_hit = False
                        if page_has_only_known_source_reviews(page_reviews, target):
                            search_stop_reason = "known_source_reviews"
                            stop_reason = "known_source_reviews"
                            break
                        reviews.extend(page_reviews)

                        if len(reviews) == before:
                            consecutive_empty += 1
                            if consecutive_empty >= 2:
                                break
                        else:
                            consecutive_empty = 0
                        if cutoff_hit:
                            break

                    except Exception as exc:
                        errors.append(f"SE {site} search '{term}' page {page}: {exc}")
                        logger.warning(
                            "SE scrape failed for '%s' (%s) page %d: %s",
                            term, site, page, exc,
                        )
                        break

                # ----------------------------------------------------------
                # Fetch answers for high-score questions from this search pass
                # ----------------------------------------------------------
                if high_score_qids and search_stop_reason != "known_source_reviews":
                    try:
                        answers_map = await self._fetch_answers(client, high_score_qids, site)
                        pages_scraped += 1  # Count the answers request

                        for qid, answers in answers_map.items():
                            for ans in answers:
                                aid = ans.get("answer_id")
                                review_id = f"{site}_a_{aid}"
                                if review_id in seen_ids:
                                    continue

                                body_html = ans.get("body") or ""
                                text = _strip_html(body_html)
                                if len(text) < _MIN_TEXT_LEN:
                                    continue

                                seen_ids.add(review_id)
                                owner = ans.get("owner") or {}

                                review = {
                                    "source": "stackoverflow",
                                    "source_url": f"https://stackoverflow.com/a/{aid}",
                                    "source_review_id": review_id,
                                    "vendor_name": target.vendor_name,
                                    "product_name": target.product_name,
                                    "product_category": target.product_category,
                                    "rating": None,
                                    "rating_max": 5,
                                    "summary": f"Answer to question {qid} (score {ans.get('score', 0)})",
                                    "review_text": text[:10000],
                                    "pros": None,
                                    "cons": None,
                                    "reviewer_name": owner.get("display_name", ""),
                                    "reviewer_title": None,
                                    "reviewer_company": None,
                                    "company_size_raw": None,
                                    "reviewer_industry": None,
                                    "reviewed_at": _epoch_to_iso(ans.get("creation_date")),
                                    "raw_metadata": {
                                        "extraction_method": "api_json",
                                        "source_weight": 0.5,
                                        "source_type": "developer_qa_platform",
                                        "score": ans.get("score", 0),
                                        "is_accepted": ans.get("is_accepted", False),
                                        "question_id": qid,
                                        "site": site,
                                        "search_term": term,
                                        "content_type": "answer",
                                    },
                                }
                                if target.date_cutoff:
                                    kept_reviews, _ = apply_date_cutoff([review], target.date_cutoff)
                                    if not kept_reviews:
                                        continue
                                    review = kept_reviews[0]
                                reviews.append(review)

                    except Exception as exc:
                        errors.append(f"SE {site} answers fetch: {exc}")
                        logger.warning("SE answer fetch failed for %s: %s", site, exc)

        logger.info(
            "StackOverflow scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            stop_reason=stop_reason,
        )


# Auto-register
register_parser(StackOverflowParser())
