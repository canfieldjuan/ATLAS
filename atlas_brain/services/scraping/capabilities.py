"""
Source capability profiles for B2B review scraping.

Each profile documents a source's access patterns, anti-bot protection,
proxy requirements, data quality tier, and recommended concurrency.
This is a code-level registry (not a DB table) because capabilities
are properties of the parser implementation, not runtime data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AccessPattern(str, Enum):
    """How the scraper accesses the source."""
    api = "api"
    html_scrape = "html_scrape"
    js_rendered = "js_rendered"
    web_unlocker = "web_unlocker"


class AntiBot(str, Enum):
    """Anti-bot protection level on the source."""
    none = "none"
    datadome = "datadome"
    cloudflare = "cloudflare"
    akamai = "akamai"
    aggressive = "aggressive"


class ProxyClass(str, Enum):
    """Proxy tier required for reliable access."""
    none = "none"
    datacenter = "datacenter"
    residential = "residential"


class DataQuality(str, Enum):
    """Data quality tier of the source."""
    verified = "verified"       # Gated review sites with identity checks
    structured = "structured"   # Structured data but no identity verification
    community = "community"     # Open community posts (Reddit, HN, etc.)
    news = "news"               # News/RSS aggregation


class ConcurrencyClass(str, Enum):
    """Concurrency classification for scheduling."""
    api = "api"   # High concurrency OK (rate limiter handles throttling)
    web = "web"   # Lower concurrency to avoid proxy overload


@dataclass(frozen=True)
class SourceCapabilityProfile:
    """Immutable capability profile for a scrape source."""

    source: str
    access_patterns: tuple[AccessPattern, ...]
    anti_bot: AntiBot
    proxy_class: ProxyClass
    data_quality: DataQuality
    default_rpm: int
    concurrency_class: ConcurrencyClass
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for API/MCP responses."""
        return {
            "source": self.source,
            "access_patterns": [p.value for p in self.access_patterns],
            "anti_bot": self.anti_bot.value,
            "proxy_class": self.proxy_class.value,
            "data_quality": self.data_quality.value,
            "default_rpm": self.default_rpm,
            "concurrency_class": self.concurrency_class.value,
            "notes": self.notes or None,
        }


# ---------------------------------------------------------------------------
# Registry: 16 source profiles from codebase audit
# ---------------------------------------------------------------------------

_PROFILES: dict[str, SourceCapabilityProfile] = {}


def _r(profile: SourceCapabilityProfile) -> None:
    _PROFILES[profile.source] = profile


# Verified review sites (gated, identity checks)
_r(SourceCapabilityProfile(
    source="g2",
    access_patterns=(AccessPattern.web_unlocker, AccessPattern.js_rendered, AccessPattern.html_scrape),
    anti_bot=AntiBot.datadome,
    proxy_class=ProxyClass.residential,
    data_quality=DataQuality.verified,
    default_rpm=6,
    concurrency_class=ConcurrencyClass.web,
))
_r(SourceCapabilityProfile(
    source="capterra",
    access_patterns=(AccessPattern.web_unlocker, AccessPattern.html_scrape),
    anti_bot=AntiBot.cloudflare,
    proxy_class=ProxyClass.residential,
    data_quality=DataQuality.verified,
    default_rpm=8,
    concurrency_class=ConcurrencyClass.web,
))
_r(SourceCapabilityProfile(
    source="trustradius",
    access_patterns=(AccessPattern.web_unlocker, AccessPattern.js_rendered, AccessPattern.html_scrape),
    anti_bot=AntiBot.none,
    proxy_class=ProxyClass.residential,
    data_quality=DataQuality.verified,
    default_rpm=10,
    concurrency_class=ConcurrencyClass.web,
))
_r(SourceCapabilityProfile(
    source="gartner",
    access_patterns=(AccessPattern.web_unlocker, AccessPattern.html_scrape),
    anti_bot=AntiBot.akamai,
    proxy_class=ProxyClass.residential,
    data_quality=DataQuality.verified,
    default_rpm=4,
    concurrency_class=ConcurrencyClass.web,
))
_r(SourceCapabilityProfile(
    source="peerspot",
    access_patterns=(AccessPattern.web_unlocker, AccessPattern.html_scrape),
    anti_bot=AntiBot.cloudflare,
    proxy_class=ProxyClass.residential,
    data_quality=DataQuality.verified,
    default_rpm=4,
    concurrency_class=ConcurrencyClass.web,
))
_r(SourceCapabilityProfile(
    source="getapp",
    access_patterns=(AccessPattern.web_unlocker, AccessPattern.html_scrape),
    anti_bot=AntiBot.cloudflare,
    proxy_class=ProxyClass.residential,
    data_quality=DataQuality.verified,
    default_rpm=8,
    concurrency_class=ConcurrencyClass.web,
))
_r(SourceCapabilityProfile(
    source="trustpilot",
    access_patterns=(AccessPattern.web_unlocker, AccessPattern.html_scrape),
    anti_bot=AntiBot.cloudflare,
    proxy_class=ProxyClass.residential,
    data_quality=DataQuality.verified,
    default_rpm=6,
    concurrency_class=ConcurrencyClass.web,
))

# Community sources (open posts, no identity verification)
_r(SourceCapabilityProfile(
    source="quora",
    access_patterns=(AccessPattern.web_unlocker, AccessPattern.html_scrape),
    anti_bot=AntiBot.aggressive,
    proxy_class=ProxyClass.residential,
    data_quality=DataQuality.community,
    default_rpm=4,
    concurrency_class=ConcurrencyClass.web,
))
_r(SourceCapabilityProfile(
    source="twitter",
    access_patterns=(AccessPattern.web_unlocker,),
    anti_bot=AntiBot.aggressive,
    proxy_class=ProxyClass.residential,
    data_quality=DataQuality.community,
    default_rpm=10,
    concurrency_class=ConcurrencyClass.web,
))
_r(SourceCapabilityProfile(
    source="reddit",
    access_patterns=(AccessPattern.api,),
    anti_bot=AntiBot.none,
    proxy_class=ProxyClass.none,
    data_quality=DataQuality.community,
    default_rpm=30,
    concurrency_class=ConcurrencyClass.api,
))
_r(SourceCapabilityProfile(
    source="hackernews",
    access_patterns=(AccessPattern.api,),
    anti_bot=AntiBot.none,
    proxy_class=ProxyClass.none,
    data_quality=DataQuality.community,
    default_rpm=100,
    concurrency_class=ConcurrencyClass.api,
))
_r(SourceCapabilityProfile(
    source="github",
    access_patterns=(AccessPattern.api,),
    anti_bot=AntiBot.none,
    proxy_class=ProxyClass.none,
    data_quality=DataQuality.community,
    default_rpm=25,
    concurrency_class=ConcurrencyClass.api,
))
_r(SourceCapabilityProfile(
    source="youtube",
    access_patterns=(AccessPattern.api,),
    anti_bot=AntiBot.none,
    proxy_class=ProxyClass.none,
    data_quality=DataQuality.community,
    default_rpm=50,
    concurrency_class=ConcurrencyClass.api,
))
_r(SourceCapabilityProfile(
    source="stackoverflow",
    access_patterns=(AccessPattern.api,),
    anti_bot=AntiBot.none,
    proxy_class=ProxyClass.none,
    data_quality=DataQuality.community,
    default_rpm=25,
    concurrency_class=ConcurrencyClass.api,
))
_r(SourceCapabilityProfile(
    source="producthunt",
    access_patterns=(AccessPattern.api, AccessPattern.html_scrape),
    anti_bot=AntiBot.none,
    proxy_class=ProxyClass.none,
    data_quality=DataQuality.community,
    default_rpm=20,
    concurrency_class=ConcurrencyClass.api,
))

# News/RSS
_r(SourceCapabilityProfile(
    source="rss",
    access_patterns=(AccessPattern.api,),
    anti_bot=AntiBot.none,
    proxy_class=ProxyClass.none,
    data_quality=DataQuality.news,
    default_rpm=10,
    concurrency_class=ConcurrencyClass.api,
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_capability(source: str) -> SourceCapabilityProfile | None:
    """Get capability profile for a source, or None if unknown."""
    return _PROFILES.get(source)


def get_all_capabilities() -> dict[str, SourceCapabilityProfile]:
    """Get all registered capability profiles."""
    return dict(_PROFILES)
