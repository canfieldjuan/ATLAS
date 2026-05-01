from __future__ import annotations

from enum import Enum
from typing import Iterable


class ReviewSource(str, Enum):
    G2 = "g2"
    CAPTERRA = "capterra"
    GARTNER = "gartner_peer_insights"
    TRUST_RADIUS = "trustradius"


VERIFIED_SOURCES = {
    ReviewSource.G2,
    ReviewSource.CAPTERRA,
    ReviewSource.GARTNER,
    ReviewSource.TRUST_RADIUS,
}


REQUIRED_ACTIONABLE_SOURCES = VERIFIED_SOURCES


def display_name(source: ReviewSource | str) -> str:
    raw = source.value if isinstance(source, ReviewSource) else str(source)
    return raw.replace("_", " ").title()


def parse_source_allowlist(value: str | None) -> set[ReviewSource]:
    if not value:
        return set(VERIFIED_SOURCES)
    out = set()
    for token in value.split(","):
        token = token.strip().lower()
        for src in ReviewSource:
            if src.value == token:
                out.add(src)
                break
    return out or set(VERIFIED_SOURCES)


def with_required_sources(sources: Iterable[ReviewSource]) -> set[ReviewSource]:
    return set(sources) | set(REQUIRED_ACTIONABLE_SOURCES)


def filter_deprecated_sources(
    sources: Iterable[ReviewSource],
    deprecated: Iterable[str] | None = None,
) -> set[ReviewSource]:
    deprecated_set = {str(x).strip().lower() for x in (deprecated or [])}
    return {src for src in sources if src.value not in deprecated_set}


def filter_blocked_sources(
    sources: Iterable[ReviewSource],
    blocked: Iterable[str] | None = None,
) -> set[ReviewSource]:
    blocked_set = {str(x).strip().lower() for x in (blocked or [])}
    return {src for src in sources if src.value not in blocked_set}
