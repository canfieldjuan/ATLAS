"""
Per-domain token-bucket rate limiter for scraping.

Each domain gets its own bucket with a configurable requests-per-minute (RPM) limit.
Callers await `acquire(domain)` which blocks until a token is available.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import B2BScrapeConfig


_DEFAULT_RPM: dict[str, int] = {
    "g2.com": 6,
    "capterra.com": 8,
    "trustradius.com": 10,
    "reddit.com": 30,
    "hn.algolia.com": 100,
    "api.github.com": 25,
    "news.google.com": 10,
}


@dataclass
class _TokenBucket:
    rpm: int
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.rpm)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(float(self.rpm), self.tokens + elapsed * (self.rpm / 60.0))
        self.last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            # Sleep until approximately 1 token is available
            wait = (1.0 - self.tokens) / (self.rpm / 60.0)
            await asyncio.sleep(wait)


class DomainRateLimiter:
    """Per-domain rate limiter using token buckets."""

    def __init__(self, rpm_map: dict[str, int]) -> None:
        self._rpm_map = rpm_map
        self._buckets: dict[str, _TokenBucket] = {}

    @classmethod
    def from_config(cls, cfg: B2BScrapeConfig) -> DomainRateLimiter:
        rpm_map = dict(_DEFAULT_RPM)
        rpm_map["g2.com"] = cfg.g2_rpm
        rpm_map["capterra.com"] = cfg.capterra_rpm
        rpm_map["trustradius.com"] = cfg.trustradius_rpm
        rpm_map["reddit.com"] = cfg.reddit_rpm
        rpm_map["hn.algolia.com"] = cfg.hackernews_rpm
        rpm_map["api.github.com"] = cfg.github_rpm
        rpm_map["news.google.com"] = cfg.rss_rpm
        return cls(rpm_map)

    async def acquire(self, domain: str) -> None:
        """Acquire a rate-limit token for the given domain."""
        if domain not in self._buckets:
            rpm = self._rpm_map.get(domain, 30)  # default 30 RPM for unknown domains
            self._buckets[domain] = _TokenBucket(rpm=rpm)
        await self._buckets[domain].acquire()
