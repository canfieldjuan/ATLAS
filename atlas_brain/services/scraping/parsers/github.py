"""
GitHub parser for B2B review scraping.

Uses the GitHub REST API (api.github.com) for issue and repository search.
Rate limit: 30 req/min (authenticated), 10 req/min (unauthenticated).

Signal value: Migration tool popularity, bug/complaint issues, adoption trends.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import quote_plus

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.github")

_DOMAIN = "api.github.com"
_BASE_URL = "https://api.github.com"
_PER_PAGE = 50
_MIN_TEXT_LEN = 100  # Min body length for issues
_MIN_REPO_DESC_LEN = 20  # Min description length for repos (intentionally shorter)


class GitHubParser:
    """Parse GitHub issues and repos as B2B churn signals."""

    source_name = "github"
    prefer_residential = False

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape GitHub for issues/repos related to the vendor."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        search_mode = target.metadata.get("search_mode", "both")
        issue_labels = target.metadata.get("issue_labels", "bug,migration")
        min_stars = target.metadata.get("min_stars", 10)

        # Auth token: target metadata > env var > config > unauthenticated
        token = target.metadata.get("github_token") or os.environ.get(
            "ATLAS_B2B_SCRAPE_GITHUB_TOKEN", ""
        )
        if not token:
            try:
                from ...config import settings
                token = settings.b2b_scrape.github_token
            except Exception:
                token = ""

        headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        vendor_encoded = quote_plus(target.vendor_name)

        # --- Issue search ---
        # Search per label separately (OR semantics: bug OR migration, not AND)
        if search_mode in ("issues", "both"):
            labels = [lbl.strip() for lbl in issue_labels.split(",") if lbl.strip()]
            for label in labels:
                for page in range(1, target.max_pages + 1):
                    url = (
                        f"{_BASE_URL}/search/issues"
                        f"?q={vendor_encoded}+label:{label}"
                        f"&sort=reactions&per_page={_PER_PAGE}&page={page}"
                    )

                    try:
                        resp = await client.get(
                            url,
                            domain=_DOMAIN,
                            referer="https://github.com/",
                            sticky_session=False,
                            prefer_residential=False,
                            extra_headers=headers,
                        )
                        pages_scraped += 1

                        if resp.status_code == 403:
                            errors.append(f"GitHub issue search label:{label} page {page}: rate limited (403)")
                            break
                        if resp.status_code != 200:
                            errors.append(f"GitHub issue search label:{label} page {page}: HTTP {resp.status_code}")
                            break

                        data = resp.json()
                        items = data.get("items", [])

                        if not items:
                            break

                        for item in items:
                            repo_full_name = item.get("repository_url", "").rsplit("/repos/", 1)[-1]
                            number = item.get("number", 0)
                            review_id = f"{repo_full_name}#issue_{number}"

                            if review_id in seen_ids:
                                continue

                            body = item.get("body") or ""
                            if len(body) < _MIN_TEXT_LEN:
                                continue

                            seen_ids.add(review_id)

                            reactions = item.get("reactions", {})
                            total_reactions = reactions.get("total_count", 0)

                            reviews.append({
                                "source": "github",
                                "source_url": item.get("html_url", ""),
                                "source_review_id": review_id,
                                "vendor_name": target.vendor_name,
                                "product_name": target.product_name,
                                "product_category": target.product_category,
                                "rating": None,
                                "rating_max": 5,
                                "summary": item.get("title", "")[:500],
                                "review_text": body[:10000],
                                "pros": None,
                                "cons": None,
                                "reviewer_name": (item.get("user") or {}).get("login", ""),
                                "reviewer_title": None,
                                "reviewer_company": None,
                                "company_size_raw": None,
                                "reviewer_industry": None,
                                "reviewed_at": item.get("created_at"),
                                "raw_metadata": {
                                    "extraction_method": "api_json",
                                    "source_weight": 0.4,
                                    "source_type": "issue_tracker",
                                    "reactions": total_reactions,
                                    "comments": item.get("comments", 0),
                                    "repo": repo_full_name,
                                    "labels": [lbl.get("name", "") for lbl in item.get("labels", [])],
                                    "state": item.get("state", ""),
                                },
                            })

                    except Exception as exc:
                        errors.append(f"GitHub issue search label:{label} page {page}: {exc}")
                        logger.warning("GitHub issue search failed label:%s page %d: %s", label, page, exc)
                        break

        # --- Repository search ---
        if search_mode in ("repos", "both"):
            for page in range(1, target.max_pages + 1):
                url = (
                    f"{_BASE_URL}/search/repositories"
                    f"?q={vendor_encoded}+migration+topic:migration"
                    f"+stars:>={min_stars}"
                    f"&sort=stars&per_page={_PER_PAGE}&page={page}"
                )

                try:
                    resp = await client.get(
                        url,
                        domain=_DOMAIN,
                        referer="https://github.com/",
                        sticky_session=False,
                        prefer_residential=False,
                        extra_headers=headers,
                    )
                    pages_scraped += 1

                    if resp.status_code == 403:
                        errors.append(f"GitHub repo search page {page}: rate limited (403)")
                        break
                    if resp.status_code != 200:
                        errors.append(f"GitHub repo search page {page}: HTTP {resp.status_code}")
                        break

                    data = resp.json()
                    items = data.get("items", [])

                    if not items:
                        break

                    for item in items:
                        full_name = item.get("full_name", "")
                        review_id = f"repo_{full_name}"

                        if review_id in seen_ids:
                            continue

                        description = item.get("description") or ""
                        if len(description) < _MIN_REPO_DESC_LEN:
                            continue

                        seen_ids.add(review_id)

                        stars = item.get("stargazers_count", 0)
                        summary = f"{full_name} ({stars} stars)"

                        reviews.append({
                            "source": "github",
                            "source_url": item.get("html_url", ""),
                            "source_review_id": review_id,
                            "vendor_name": target.vendor_name,
                            "product_name": target.product_name,
                            "product_category": target.product_category,
                            "rating": None,
                            "rating_max": 5,
                            "summary": summary[:500],
                            "review_text": description[:10000],
                            "pros": None,
                            "cons": None,
                            "reviewer_name": (item.get("owner") or {}).get("login", ""),
                            "reviewer_title": None,
                            "reviewer_company": None,
                            "company_size_raw": None,
                            "reviewer_industry": None,
                            "reviewed_at": item.get("created_at"),
                            "raw_metadata": {
                                "extraction_method": "api_json",
                                "source_weight": 0.4,
                                "source_type": "repository_signal",
                                "stars": stars,
                                "forks": item.get("forks_count", 0),
                                "open_issues": item.get("open_issues_count", 0),
                                "language": item.get("language"),
                                "topics": item.get("topics", []),
                                "updated_at": item.get("updated_at"),
                            },
                        })

                except Exception as exc:
                    errors.append(f"GitHub repo search page {page}: {exc}")
                    logger.warning("GitHub repo search failed page %d: %s", page, exc)
                    break

        logger.info(
            "GitHub scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# Auto-register
register_parser(GitHubParser())
