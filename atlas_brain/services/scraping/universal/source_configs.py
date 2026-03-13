"""
Per-source adapter configurations for wiring the universal scraper
into the B2B review pipeline.

Each config defines how to build URLs, what to extract, pagination,
and normalization metadata for a specific review source. Wave 1 covers
deterministic page-based sources; search/API sources stay on legacy parsers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import ExtractionSchema


@dataclass(frozen=True)
class SourceAdapterConfig:
    """Configuration for adapting a specific review source to the universal scraper."""

    source: str
    """Source name matching b2b_scrape_targets.source (e.g. 'trustpilot')."""

    url_template: str
    """URL template with ``{slug}`` placeholder for product_slug."""

    pagination_template: str | None = None
    """URL template with ``{slug}`` and ``{page}`` placeholders.
    None means no pagination support."""

    use_browser: bool = False
    """True to use Playwright/StealthBrowser instead of HTTP client."""

    prefer_residential: bool = True
    """True to prefer residential proxies for this source."""

    extraction_schema: ExtractionSchema = field(
        default_factory=lambda: ExtractionSchema(description="Extract all reviews")
    )
    """LLM extraction prompt describing what to pull from the page."""

    source_weight: float = 0.7
    """Trust weight for raw_metadata (0.0-1.0)."""

    source_type: str = "review_site"
    """Source classification for raw_metadata."""

    rating_max: int = 5
    """Maximum rating value for this source."""

    min_text_length: int = 20
    """Minimum combined text length for a review to be kept."""

    inter_page_delay: tuple[float, float] = (2.0, 5.0)
    """Random delay range (min, max) seconds between paginated requests."""


# ── Wave 1 Configs ───────────────────────────────────────────────────

# Shared extraction prompt fragment for fields all review sources have
_COMMON_FIELDS = (
    "rating (number 1-5), title or headline, full review text, "
    "author/reviewer name, date published (ISO 8601 format if possible), "
    "author company or employer if shown"
)

_STRUCTURED_FIELDS = (
    f"{_COMMON_FIELDS}, "
    "pros or what is most valuable, cons or what needs improvement, "
    "job title, company size, industry"
)


TRUSTPILOT = SourceAdapterConfig(
    source="trustpilot",
    url_template="https://www.trustpilot.com/review/{slug}",
    pagination_template="https://www.trustpilot.com/review/{slug}?page={page}",
    use_browser=False,
    prefer_residential=True,
    extraction_schema=ExtractionSchema(description=(
        "Extract ALL individual user reviews from this page. "
        f"For each review return: {_COMMON_FIELDS}. "
        "Do NOT include aggregate/summary ratings or site navigation text. "
        "Do NOT fabricate reviews — only extract what is actually on the page."
    )),
    source_weight=0.7,
    source_type="consumer_review_platform",
    inter_page_delay=(2.0, 5.0),
)

PEERSPOT = SourceAdapterConfig(
    source="peerspot",
    url_template="https://www.peerspot.com/products/{slug}-reviews",
    pagination_template="https://www.peerspot.com/products/{slug}-reviews?page={page}",
    use_browser=False,
    prefer_residential=True,
    extraction_schema=ExtractionSchema(description=(
        "Extract ALL individual reviews from this enterprise software review page. "
        f"For each review return: {_STRUCTURED_FIELDS}, "
        "years of experience with the product. "
        "If the review has structured Q&A sections (What is most valuable?, "
        "What needs improvement?, etc.), include each section's content. "
        "Combine all Q&A answers into the review text field. "
        "Do NOT fabricate reviews — only extract what is actually on the page."
    )),
    source_weight=0.9,
    source_type="verified_review_platform",
    inter_page_delay=(2.0, 5.0),
)

GETAPP = SourceAdapterConfig(
    source="getapp",
    # product_slug in DB is "{category-slug}/a/{product-slug}"
    url_template="https://www.getapp.com/software/{slug}/reviews/",
    pagination_template="https://www.getapp.com/software/{slug}/reviews/?page={page}",
    use_browser=False,
    prefer_residential=True,
    extraction_schema=ExtractionSchema(description=(
        "Extract ALL individual software reviews from this page. "
        f"For each review return: {_STRUCTURED_FIELDS}. "
        "Do NOT include aggregate ratings or site metadata. "
        "Do NOT fabricate reviews — only extract what is actually on the page."
    )),
    source_weight=0.85,
    source_type="verified_review_platform",
    inter_page_delay=(2.0, 5.0),
)

SOFTWARE_ADVICE = SourceAdapterConfig(
    source="software_advice",
    # product_slug in DB is "{category}/{product-slug}"
    url_template="https://www.softwareadvice.com/{slug}/reviews/",
    pagination_template="https://www.softwareadvice.com/{slug}/reviews/?page={page}",
    use_browser=False,
    prefer_residential=True,
    extraction_schema=ExtractionSchema(description=(
        "Extract ALL individual software reviews from this page. "
        f"For each review return: {_STRUCTURED_FIELDS}. "
        "Do NOT include aggregate ratings or site metadata. "
        "Do NOT fabricate reviews — only extract what is actually on the page."
    )),
    source_weight=0.85,
    source_type="verified_review_platform",
    inter_page_delay=(2.0, 5.0),
)

SOURCEFORGE = SourceAdapterConfig(
    source="sourceforge",
    url_template="https://sourceforge.net/software/product/{slug}/reviews/",
    pagination_template="https://sourceforge.net/software/product/{slug}/reviews/?page={page}",
    use_browser=False,
    prefer_residential=False,  # Datacenter proxies sufficient
    extraction_schema=ExtractionSchema(description=(
        "Extract ALL individual software reviews from this page. "
        f"For each review return: {_STRUCTURED_FIELDS}. "
        "Do NOT include aggregate ratings or site metadata. "
        "Do NOT fabricate reviews — only extract what is actually on the page."
    )),
    source_weight=0.6,
    source_type="review_site",
    min_text_length=80,
    inter_page_delay=(1.0, 2.0),
)


# ── Registry ─────────────────────────────────────────────────────────

_CONFIGS: dict[str, SourceAdapterConfig] = {
    cfg.source: cfg
    for cfg in [TRUSTPILOT, PEERSPOT, GETAPP, SOFTWARE_ADVICE, SOURCEFORGE]
}


def get_source_adapter_config(source: str) -> SourceAdapterConfig | None:
    """Look up adapter config by source name. Returns None for unconfigured sources."""
    return _CONFIGS.get(source)


def get_all_adapter_sources() -> list[str]:
    """Return all sources that have universal adapter configs."""
    return list(_CONFIGS.keys())
