"""
Quora parser for B2B review scraping.

Scrapes Quora search results and question pages for vendor-related opinions.
Quora has NO public API -- must scrape HTML.

Primary: Bright Data Web Unlocker (Quora blocks scrapers heavily).
Fallback: curl_cffi HTTP client with residential proxy.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from urllib.parse import quote_plus

from bs4 import BeautifulSoup, Tag

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.quora")

_DOMAIN = "quora.com"
_BASE_URL = "https://www.quora.com"
_MIN_TEXT_LEN = 100

# Multiple search queries per vendor to maximise coverage
_SEARCH_SUFFIXES = [
    "alternative",
    "switching from",
    "vs",
    "review",
]

# Headers mimicking a real browser session on Quora
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


class QuoraParser:
    """Parse Quora search results and answers for B2B vendor opinions."""

    source_name = "quora"
    prefer_residential = True

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Quora -- Web Unlocker first, then HTTP client fallback."""
        from atlas_brain.config import settings

        # Priority 1: Bright Data Web Unlocker (handles Quora anti-bot automatically)
        if settings.b2b_scrape.web_unlocker_url:
            unlocker_domains = {
                d.strip().lower()
                for d in settings.b2b_scrape.web_unlocker_domains.split(",")
                if d.strip()
            }
            if _DOMAIN in unlocker_domains:
                try:
                    result = await self._scrape_web_unlocker(target)
                    if result.reviews:
                        return result
                    logger.warning(
                        "Web Unlocker for %s returned 0 reviews, falling back",
                        target.vendor_name,
                    )
                except Exception as exc:
                    logger.warning(
                        "Web Unlocker failed for %s: %s -- falling back",
                        target.vendor_name, exc,
                    )

        # Priority 2: curl_cffi HTTP client with residential proxy
        return await self._scrape_http(target, client)

    # ------------------------------------------------------------------
    # Web Unlocker path (Bright Data -- handles anti-bot internally)
    # ------------------------------------------------------------------

    async def _scrape_web_unlocker(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape Quora via Bright Data Web Unlocker proxy."""
        import httpx
        from atlas_brain.config import settings

        proxy_url = settings.b2b_scrape.web_unlocker_url.strip()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        queries = _build_search_queries(target.vendor_name)

        for query in queries:
            url = f"{_BASE_URL}/search?q={quote_plus(query)}&type=answer"

            headers = {
                **_BROWSER_HEADERS,
                "Referer": f"https://www.google.com/search?q={quote_plus(query)}+site:quora.com",
            }

            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url, verify=False, timeout=90.0,
                ) as http:
                    resp = await http.get(url, headers=headers)

                pages_scraped += 1

                if resp.status_code == 403:
                    errors.append(f"Query '{query}': blocked (403) via Web Unlocker")
                    continue
                if resp.status_code != 200:
                    errors.append(f"Query '{query}': HTTP {resp.status_code}")
                    continue

                page_reviews = _parse_search_results(resp.text, target, seen_ids)
                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Query '{query}': {exc}")
                logger.warning(
                    "Quora Web Unlocker query '%s' failed for %s: %s",
                    query, target.vendor_name, exc,
                )

            # Inter-query delay to avoid rate limits
            await asyncio.sleep(random.uniform(3.0, 6.0))

        # Also try direct question URL pattern
        direct_reviews = await self._fetch_direct_question_unlocker(
            target, proxy_url, seen_ids, errors,
        )
        reviews.extend(direct_reviews)
        if direct_reviews:
            pages_scraped += 1

        logger.info(
            "Quora Web Unlocker scrape for %s: %d reviews from %d requests",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    async def _fetch_direct_question_unlocker(
        self,
        target: ScrapeTarget,
        proxy_url: str,
        seen_ids: set[str],
        errors: list[str],
    ) -> list[dict]:
        """Try fetching a direct question URL via Web Unlocker."""
        import httpx

        # Normalise vendor name for URL: "Salesforce CRM" -> "Salesforce-CRM"
        slug = target.vendor_name.replace(" ", "-")
        url = f"{_BASE_URL}/What-are-the-best-alternatives-to-{slug}"

        headers = {
            **_BROWSER_HEADERS,
            "Referer": f"https://www.google.com/search?q=best+alternatives+to+{quote_plus(target.vendor_name)}+quora",
        }

        try:
            async with httpx.AsyncClient(
                proxy=proxy_url, verify=False, timeout=90.0,
            ) as http:
                resp = await http.get(url, headers=headers)

            if resp.status_code != 200:
                return []

            return _parse_question_page(resp.text, target, seen_ids)
        except Exception as exc:
            errors.append(f"Direct question URL: {exc}")
            logger.warning(
                "Quora direct question failed for %s: %s",
                target.vendor_name, exc,
            )
            return []

    # ------------------------------------------------------------------
    # HTTP client path (curl_cffi + residential proxy)
    # ------------------------------------------------------------------

    async def _scrape_http(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Quora via curl_cffi HTTP client with residential proxy."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        queries = _build_search_queries(target.vendor_name)

        for query in queries[:target.max_pages]:
            url = f"{_BASE_URL}/search?q={quote_plus(query)}&type=answer"
            referer = f"https://www.google.com/search?q={quote_plus(query)}+site:quora.com"

            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer=referer,
                    sticky_session=True,
                    prefer_residential=True,
                )
                pages_scraped += 1

                if resp.status_code == 403:
                    errors.append(f"Query '{query}': blocked (403)")
                    continue
                if resp.status_code == 429:
                    errors.append(f"Query '{query}': rate limited (429)")
                    break
                if resp.status_code != 200:
                    errors.append(f"Query '{query}': HTTP {resp.status_code}")
                    continue

                # Guard against non-HTML responses
                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Query '{query}': unexpected content-type ({ct[:40]})")
                    continue

                page_reviews = _parse_search_results(resp.text, target, seen_ids)
                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Query '{query}': {exc}")
                logger.warning(
                    "Quora HTTP query '%s' failed for %s: %s",
                    query, target.vendor_name, exc,
                )

        # Try direct question URL via HTTP client
        direct_reviews = await self._fetch_direct_question_http(
            target, client, seen_ids, errors,
        )
        reviews.extend(direct_reviews)
        if direct_reviews:
            pages_scraped += 1

        logger.info(
            "Quora HTTP scrape for %s: %d reviews from %d requests",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    async def _fetch_direct_question_http(
        self,
        target: ScrapeTarget,
        client: AntiDetectionClient,
        seen_ids: set[str],
        errors: list[str],
    ) -> list[dict]:
        """Try fetching a direct question URL via HTTP client."""
        slug = target.vendor_name.replace(" ", "-")
        url = f"{_BASE_URL}/What-are-the-best-alternatives-to-{slug}"
        referer = f"https://www.google.com/search?q=best+alternatives+to+{quote_plus(target.vendor_name)}+quora"

        try:
            resp = await client.get(
                url,
                domain=_DOMAIN,
                referer=referer,
                sticky_session=True,
                prefer_residential=True,
            )
            if resp.status_code != 200:
                return []

            ct = resp.headers.get("content-type", "")
            if "html" not in ct and "text" not in ct:
                return []

            return _parse_question_page(resp.text, target, seen_ids)
        except Exception as exc:
            errors.append(f"Direct question URL: {exc}")
            logger.warning(
                "Quora direct question HTTP failed for %s: %s",
                target.vendor_name, exc,
            )
            return []


# ------------------------------------------------------------------
# Query building
# ------------------------------------------------------------------

def _build_search_queries(vendor_name: str) -> list[str]:
    """Build multiple search queries for a vendor."""
    return [f"{vendor_name} {suffix}" for suffix in _SEARCH_SUFFIXES]


# ------------------------------------------------------------------
# HTML parsing helpers
# ------------------------------------------------------------------

def _parse_search_results(
    html: str, target: ScrapeTarget, seen_ids: set[str],
) -> list[dict]:
    """Parse Quora search results page and extract answers."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # Quora search results contain answer snippets in various container patterns
    answer_containers = _find_answer_containers(soup)

    for container in answer_containers:
        try:
            review = _parse_answer_container(container, target, seen_ids)
            if review:
                reviews.append(review)
        except Exception:
            logger.debug("Failed to parse Quora answer container", exc_info=True)

    return reviews


def _parse_question_page(
    html: str, target: ScrapeTarget, seen_ids: set[str],
) -> list[dict]:
    """Parse a Quora question page for full answers."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # Extract the question title for context
    question_title = _extract_question_title(soup)

    # Find answer containers on a question page
    answer_containers = _find_answer_containers(soup)

    for container in answer_containers:
        try:
            review = _parse_answer_container(
                container, target, seen_ids, question_title=question_title,
            )
            if review:
                reviews.append(review)
        except Exception:
            logger.debug("Failed to parse Quora question-page answer", exc_info=True)

    return reviews


def _find_answer_containers(soup: BeautifulSoup) -> list[Tag]:
    """Find answer containers using multiple selector strategies.

    Quora's HTML is heavily obfuscated with dynamic class names, so we try
    several selector patterns in priority order.
    """
    # Strategy 1: div containers with Answer-related class fragments
    containers = soup.select('div[class*="Answer"]')
    if containers:
        return containers

    # Strategy 2: q-box containers (Quora's generic card wrapper)
    containers = soup.select("div.q-box")
    if containers:
        # Filter to those that look like answers (contain enough text)
        return [
            c for c in containers
            if len(c.get_text(strip=True)) >= _MIN_TEXT_LEN
        ]

    # Strategy 3: DOM-walking -- look for answer-like content blocks
    # Quora renders answers inside nested divs; find spans with substantial text
    # that sit inside a clickable answer region
    containers = soup.select('div[class*="AnswerBase"], div[class*="answer"]')
    if containers:
        return containers

    # Strategy 4: broadest fallback -- any div with a large text block
    # that contains author credentials patterns
    fallback: list[Tag] = []
    for div in soup.find_all("div", recursive=True):
        text = div.get_text(strip=True)
        if len(text) >= _MIN_TEXT_LEN and _looks_like_answer(div):
            fallback.append(div)
    return fallback[:50]  # Cap to avoid processing the entire DOM


def _looks_like_answer(el: Tag) -> bool:
    """Heuristic: does this element look like a Quora answer block?"""
    # Must have some minimum depth (answers are nested, not top-level)
    if not el.parent or not el.parent.parent:
        return False
    # Should not be a navigation/header/footer element
    tag_classes = " ".join(el.get("class", []))
    if any(skip in tag_classes.lower() for skip in ("nav", "header", "footer", "sidebar", "menu")):
        return False
    # Must contain at least one <span> with text (Quora wraps answer text in spans)
    spans = el.find_all("span", limit=5)
    return any(len(s.get_text(strip=True)) > 30 for s in spans)


def _extract_question_title(soup: BeautifulSoup) -> str | None:
    """Extract the question title from a Quora question page."""
    # Question title selectors
    for selector in (
        'div[class*="question"] span[class*="q-text"]',
        'span[class*="q-text"]',
        'div[class*="QuestionHeader"] span',
        'h1 span',
        'title',
    ):
        el = soup.select_one(selector)
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 10:
                # Clean trailing " - Quora" from <title>
                if text.endswith(" - Quora"):
                    text = text[:-8].strip()
                return text[:500]
    return None


def _parse_answer_container(
    container: Tag,
    target: ScrapeTarget,
    seen_ids: set[str],
    *,
    question_title: str | None = None,
) -> dict | None:
    """Extract a review dict from a single Quora answer container."""
    # Extract answer text -- collect all spans/paragraphs within the container
    answer_text = _extract_answer_text(container)
    if not answer_text or len(answer_text) < _MIN_TEXT_LEN:
        return None

    # Generate a stable review ID from the answer text content
    text_hash = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()[:16]
    review_id = f"quora-{text_hash}"

    if review_id in seen_ids:
        return None
    seen_ids.add(review_id)

    # Extract author info
    author_name, author_credentials = _extract_author_info(container)

    # Extract upvote count from the container
    upvotes = _extract_upvotes(container)

    # Build source URL -- best-effort from any link in the container
    source_url = _extract_answer_url(container) or f"{_BASE_URL}/search?q={quote_plus(target.vendor_name)}"

    # Use question title as summary when available
    summary = question_title[:500] if question_title else None

    return {
        "source": "quora",
        "source_url": source_url,
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": None,
        "rating_max": 5,
        "summary": summary,
        "review_text": answer_text[:10000],
        "pros": None,
        "cons": None,
        "reviewer_name": author_name,
        "reviewer_title": author_credentials,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
        "reviewed_at": None,
        "raw_metadata": {
            "extraction_method": "html",
            "source_weight": 0.5,
            "source_type": "qa_platform",
            "upvotes": upvotes,
            "question_title": question_title,
        },
    }


def _extract_answer_text(container: Tag) -> str:
    """Extract the main answer text from a Quora answer container.

    Quora wraps answer content in deeply nested spans. We collect text
    from the largest contiguous text block within the container.
    """
    # Strategy 1: look for the main answer content span
    for selector in (
        'div[class*="AnswerContent"] span',
        'div[class*="answer_content"] span',
        'span[class*="q-text"]',
        'div.q-text',
    ):
        el = container.select_one(selector)
        if el:
            text = el.get_text(separator=" ", strip=True)
            if len(text) >= _MIN_TEXT_LEN:
                return text

    # Strategy 2: collect all <p> and <span> text within the container
    parts: list[str] = []
    for tag in container.find_all(["p", "span"], recursive=True):
        text = tag.get_text(strip=True)
        # Skip very short fragments (navigation, buttons, metadata)
        if len(text) > 20:
            # Avoid duplicates from nested elements
            if text not in parts:
                parts.append(text)

    if parts:
        # Take the longest contiguous block
        combined = " ".join(parts)
        if len(combined) >= _MIN_TEXT_LEN:
            return combined

    # Strategy 3: raw container text as last resort
    raw = container.get_text(separator=" ", strip=True)
    return raw


def _extract_author_info(container: Tag) -> tuple[str | None, str | None]:
    """Extract author name and credentials from an answer container.

    Returns (author_name, credentials_text).
    """
    author_name: str | None = None
    credentials: str | None = None

    # Author name selectors
    for selector in (
        'a[class*="user"] span',
        'span[class*="author"]',
        'a[class*="ProfileLink"]',
        'div[class*="AuthorInfo"] a',
        'span[class*="name"]',
    ):
        el = container.select_one(selector)
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 1 and len(text) < 100:
                author_name = text
                break

    # Credentials / bio line (e.g. "Software Engineer at Google (2015-present)")
    for selector in (
        'span[class*="credential"]',
        'div[class*="credential"]',
        'span[class*="UserCredential"]',
        'div[class*="AuthorInfo"] span:nth-of-type(2)',
    ):
        el = container.select_one(selector)
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 3 and text != author_name:
                credentials = text[:300]
                break

    return author_name, credentials


def _extract_upvotes(container: Tag) -> int | None:
    """Extract the upvote count from an answer container."""
    for selector in (
        'span[class*="upvote"]',
        'button[class*="upvote"] span',
        'div[class*="VoteButton"] span',
    ):
        el = container.select_one(selector)
        if el:
            text = el.get_text(strip=True).replace(",", "")
            # Handle "1.2K" style counts
            try:
                if text.upper().endswith("K"):
                    return int(float(text[:-1]) * 1000)
                if text.upper().endswith("M"):
                    return int(float(text[:-1]) * 1_000_000)
                return int(text)
            except (ValueError, TypeError):
                continue
    return None


def _extract_answer_url(container: Tag) -> str | None:
    """Try to extract a permalink to this specific answer."""
    # Look for a link that points to a Quora question/answer
    for a_tag in container.find_all("a", href=True, limit=10):
        href = a_tag["href"]
        if href.startswith("/") and not href.startswith("/search"):
            return f"{_BASE_URL}{href}"
        if "quora.com/" in href and "/search" not in href:
            return href
    return None


# Auto-register
register_parser(QuoraParser())
