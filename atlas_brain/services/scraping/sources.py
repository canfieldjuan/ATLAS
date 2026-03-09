"""Canonical review source enum and classification sets.

Single source of truth for all B2B review platform identifiers used across
the scraping pipeline, MCP server, enrichment, blog generation, and
intelligence synthesis.

Because ``ReviewSource`` extends ``str``, existing string comparisons
(``"g2" == ReviewSource.G2``) work without changes.
"""

from __future__ import annotations

from enum import Enum


class ReviewSource(str, Enum):
    G2 = "g2"
    CAPTERRA = "capterra"
    TRUSTRADIUS = "trustradius"
    GARTNER = "gartner"
    PEERSPOT = "peerspot"
    GETAPP = "getapp"
    PRODUCTHUNT = "producthunt"
    TRUSTPILOT = "trustpilot"
    REDDIT = "reddit"
    HACKERNEWS = "hackernews"
    GITHUB = "github"
    YOUTUBE = "youtube"
    STACKOVERFLOW = "stackoverflow"
    QUORA = "quora"
    TWITTER = "twitter"
    RSS = "rss"
    SOFTWARE_ADVICE = "software_advice"


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------

_DISPLAY_NAMES: dict[ReviewSource, str] = {
    ReviewSource.G2: "G2",
    ReviewSource.CAPTERRA: "Capterra",
    ReviewSource.TRUSTRADIUS: "TrustRadius",
    ReviewSource.GARTNER: "Gartner",
    ReviewSource.PEERSPOT: "PeerSpot",
    ReviewSource.GETAPP: "GetApp",
    ReviewSource.PRODUCTHUNT: "Product Hunt",
    ReviewSource.TRUSTPILOT: "Trustpilot",
    ReviewSource.REDDIT: "Reddit",
    ReviewSource.HACKERNEWS: "Hacker News",
    ReviewSource.GITHUB: "GitHub",
    ReviewSource.YOUTUBE: "YouTube",
    ReviewSource.STACKOVERFLOW: "Stack Overflow",
    ReviewSource.QUORA: "Quora",
    ReviewSource.TWITTER: "Twitter/X",
    ReviewSource.RSS: "RSS",
    ReviewSource.SOFTWARE_ADVICE: "Software Advice",
}


def display_name(source: str | ReviewSource) -> str:
    """Human-readable label for a source."""
    try:
        member = ReviewSource(source)
    except ValueError:
        return str(source).title()
    return _DISPLAY_NAMES.get(member, member.value.title())


# ---------------------------------------------------------------------------
# Classification sets (frozensets of enum members)
# ---------------------------------------------------------------------------

ALL_SOURCES: frozenset[ReviewSource] = frozenset(
    m for m in ReviewSource if m is not ReviewSource.SOFTWARE_ADVICE
)

SEARCH_SOURCES: frozenset[ReviewSource] = frozenset({
    ReviewSource.REDDIT,
    ReviewSource.HACKERNEWS,
    ReviewSource.GITHUB,
    ReviewSource.YOUTUBE,
    ReviewSource.STACKOVERFLOW,
    ReviewSource.QUORA,
    ReviewSource.TWITTER,
})

SLUG_SOURCES: frozenset[ReviewSource] = frozenset({
    ReviewSource.G2,
    ReviewSource.CAPTERRA,
    ReviewSource.TRUSTRADIUS,
    ReviewSource.GARTNER,
    ReviewSource.PEERSPOT,
    ReviewSource.GETAPP,
    ReviewSource.PRODUCTHUNT,
    ReviewSource.TRUSTPILOT,
})

API_SOURCES: frozenset[ReviewSource] = frozenset({
    ReviewSource.YOUTUBE,
    ReviewSource.STACKOVERFLOW,
    ReviewSource.PRODUCTHUNT,
    ReviewSource.HACKERNEWS,
    ReviewSource.GITHUB,
    ReviewSource.RSS,
})

VERIFIED_SOURCES: frozenset[ReviewSource] = frozenset({
    ReviewSource.G2,
    ReviewSource.CAPTERRA,
    ReviewSource.GARTNER,
    ReviewSource.TRUSTRADIUS,
    ReviewSource.PEERSPOT,
    ReviewSource.GETAPP,
    ReviewSource.SOFTWARE_ADVICE,
    ReviewSource.TRUSTPILOT,
})

STRUCTURED_SOURCES: frozenset[ReviewSource] = frozenset({
    ReviewSource.G2,
    ReviewSource.CAPTERRA,
    ReviewSource.TRUSTRADIUS,
})

EXECUTIVE_SOURCES: frozenset[ReviewSource] = frozenset({
    ReviewSource.G2,
    ReviewSource.CAPTERRA,
    ReviewSource.TRUSTRADIUS,
})

DEFAULT_ALLOWLIST_SOURCES: frozenset[ReviewSource] = frozenset({
    ReviewSource.G2,
    ReviewSource.CAPTERRA,
    ReviewSource.TRUSTRADIUS,
    ReviewSource.TRUSTPILOT,
    ReviewSource.REDDIT,
    ReviewSource.HACKERNEWS,
    ReviewSource.QUORA,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_source_allowlist(raw: str) -> list[str]:
    """Return a normalized source allowlist from a comma-separated string."""
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def is_source_allowed(source: str, allowlist_raw: str) -> bool:
    """Return True when *source* is allowed by the configured allowlist."""
    allowed = parse_source_allowlist(allowlist_raw)
    if not allowed:
        return True
    return source.strip().lower() in allowed
