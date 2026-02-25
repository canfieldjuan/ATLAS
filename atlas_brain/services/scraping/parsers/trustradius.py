"""
TrustRadius parser for B2B review scraping.

Uses TrustRadius JSON API: /api/v1/products/{slug}/reviews
Rating scale is 1-10 (normalized). Residential proxy recommended.
"""

from __future__ import annotations

import logging
from urllib.parse import quote_plus

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.trustradius")

_DOMAIN = "trustradius.com"
_API_BASE = "https://www.trustradius.com/api/v1/products"
_PAGE_SIZE = 25


class TrustRadiusParser:
    """Parse TrustRadius reviews via their JSON API."""

    source_name = "trustradius"
    prefer_residential = True

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape TrustRadius reviews for the given product."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        for page in range(target.max_pages):
            offset = page * _PAGE_SIZE
            url = (
                f"{_API_BASE}/{target.product_slug}/reviews"
                f"?limit={_PAGE_SIZE}&offset={offset}"
            )
            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+trustradius+reviews"
                if page == 0
                else f"https://www.trustradius.com/products/{target.product_slug}/reviews"
            )

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
                    errors.append(f"Page {page + 1}: blocked (403)")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page + 1}: HTTP {resp.status_code}")
                    continue

                data = resp.json()
                records = data.get("records", [])

                if not records:
                    break  # No more reviews

                for record in records:
                    review = _parse_record(record, target, seen_ids)
                    if review:
                        reviews.append(review)

            except Exception as exc:
                errors.append(f"Page {page + 1}: {exc}")
                logger.warning("TrustRadius page %d failed for %s: %s", page + 1, target.product_slug, exc)
                break

        logger.info(
            "TrustRadius scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


def _parse_record(record: dict, target: ScrapeTarget, seen_ids: set[str]) -> dict | None:
    """Parse a single TrustRadius API review record."""
    review_id = record.get("_id", "")
    if not review_id or review_id in seen_ids:
        return None
    seen_ids.add(review_id)

    # Rating: {"normalized": 9, "possible": 100, "earned": 90}
    rating_obj = record.get("rating", {})
    rating = rating_obj.get("normalized")  # 1-10 scale

    # Review text: synopsis is the main text, verbatims are key quotes
    synopsis = record.get("synopsis", "")
    verbatims = record.get("verbatims", [])
    heading = record.get("heading", "")

    # Build review_text from synopsis + verbatims
    parts = []
    if synopsis:
        parts.append(synopsis)
    if verbatims:
        parts.append("\n".join(verbatims))
    review_text = "\n\n".join(parts)

    if not review_text or len(review_text) < 20:
        return None

    # Date
    reviewed_at = record.get("publishedDate")

    # Reviewer info
    reviewer_title = record.get("reviewerJobType")
    reviewer_dept = record.get("reviewerDepartment")
    if reviewer_title and reviewer_dept:
        reviewer_title = f"{reviewer_title}, {reviewer_dept}"

    slug = record.get("slug", review_id)
    source_url = f"https://www.trustradius.com/reviews/{slug}"

    return {
        "source": "trustradius",
        "source_url": source_url,
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 10,
        "summary": heading[:500] if heading else None,
        "review_text": review_text[:10000],
        "pros": None,
        "cons": None,
        "reviewer_name": None,  # API doesn't expose reviewer name
        "reviewer_title": reviewer_title,
        "reviewer_company": None,
        "company_size_raw": record.get("companySize"),
        "reviewer_industry": record.get("companyIndustry"),
        "reviewed_at": reviewed_at,
        "raw_metadata": {
            "grade": record.get("grade"),
            "rating_earned": rating_obj.get("earned"),
            "rating_possible": rating_obj.get("possible"),
        },
    }


# Auto-register
register_parser(TrustRadiusParser())
