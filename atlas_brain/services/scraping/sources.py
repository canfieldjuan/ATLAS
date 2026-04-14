"""Canonical review source enum and classification sets.

Single source of truth for all B2B review platform identifiers used across
the scraping pipeline, MCP server, enrichment, blog generation, and
intelligence synthesis.

Because ``ReviewSource`` extends ``str``, existing string comparisons
(``"g2" == ReviewSource.G2``) work without changes.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    SOURCEFORGE = "sourceforge"
    SLASHDOT = "slashdot"


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
    ReviewSource.SOURCEFORGE: "SourceForge",
    ReviewSource.SLASHDOT: "Slashdot",
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

ALL_SOURCES: frozenset[ReviewSource] = frozenset(ReviewSource)

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
    ReviewSource.SOFTWARE_ADVICE,
    ReviewSource.PRODUCTHUNT,
    ReviewSource.TRUSTPILOT,
    ReviewSource.SOURCEFORGE,
    ReviewSource.SLASHDOT,
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

# Structured review platforms bypass the social/content relevance filter, but
# only VERIFIED_SOURCES should receive verified-source confidence treatment.
STRUCTURED_SOURCES: frozenset[ReviewSource] = VERIFIED_SOURCES | frozenset({
    ReviewSource.SOURCEFORGE,
    ReviewSource.SLASHDOT,
})

EXECUTIVE_SOURCES: frozenset[ReviewSource] = frozenset({
    ReviewSource.G2,
    ReviewSource.GARTNER,
    ReviewSource.TRUSTRADIUS,
    ReviewSource.PEERSPOT,
    ReviewSource.GETAPP,
})

DEFAULT_ALLOWLIST_SOURCES: frozenset[ReviewSource] = frozenset({
    ReviewSource.G2,
    ReviewSource.GARTNER,
    ReviewSource.TRUSTRADIUS,
    ReviewSource.PEERSPOT,
    ReviewSource.GETAPP,
    ReviewSource.REDDIT,
    ReviewSource.HACKERNEWS,
    ReviewSource.GITHUB,
    ReviewSource.STACKOVERFLOW,
    ReviewSource.SLASHDOT,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_ACTIONABLE_SOURCES: frozenset[str] = frozenset({
    ReviewSource.TRUSTRADIUS.value,
    ReviewSource.SOFTWARE_ADVICE.value,
})

REQUIRED_SCRAPE_SOURCES: frozenset[str] = REQUIRED_ACTIONABLE_SOURCES | frozenset({
    ReviewSource.CAPTERRA.value,
})

def parse_source_allowlist(raw: str) -> list[str]:
    """Return a normalized source allowlist from a comma-separated string."""
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def with_required_sources(
    sources: Iterable[str],
    required: Iterable[str] | None = None,
) -> list[str]:
    """Append required sources while preserving order and deduplicating."""
    merged: list[str] = []
    seen: set[str] = set()
    required_sources = list(REQUIRED_ACTIONABLE_SOURCES if required is None else required)
    for source in list(sources) + required_sources:
        normalized = str(source).strip().lower()
        if not normalized or normalized in seen:
            continue
        merged.append(normalized)
        seen.add(normalized)
    return merged


def filter_deprecated_sources(
    sources: Iterable[str],
    deprecated: str | Iterable[str] | None,
) -> list[str]:
    """Remove deprecated sources from an allowlist while preserving order."""
    if isinstance(deprecated, str):
        deprecated_set = set(parse_source_allowlist(deprecated))
    else:
        deprecated_set = {
            str(source).strip().lower()
            for source in (deprecated or [])
            if str(source).strip()
        }
    deprecated_set -= REQUIRED_ACTIONABLE_SOURCES
    filtered: list[str] = []
    seen: set[str] = set()
    for source in sources:
        normalized = str(source).strip().lower()
        if not normalized or normalized in deprecated_set or normalized in seen:
            continue
        filtered.append(normalized)
        seen.add(normalized)
    return filtered


def filter_blocked_sources(
    sources: Iterable[str],
    blocked: str | Iterable[str] | None,
) -> list[str]:
    """Remove operationally blocked sources from an allowlist."""
    if isinstance(blocked, str):
        blocked_set = set(parse_source_allowlist(blocked))
    else:
        blocked_set = {
            str(source).strip().lower()
            for source in (blocked or [])
            if str(source).strip()
        }
    filtered: list[str] = []
    seen: set[str] = set()
    for source in sources:
        normalized = str(source).strip().lower()
        if not normalized or normalized in blocked_set or normalized in seen:
            continue
        filtered.append(normalized)
        seen.add(normalized)
    return filtered


def is_source_allowed(source: str, allowlist_raw: str) -> bool:
    """Return True when *source* is allowed by the configured allowlist."""
    allowed = parse_source_allowlist(allowlist_raw)
    if not allowed:
        return True
    return source.strip().lower() in allowed
