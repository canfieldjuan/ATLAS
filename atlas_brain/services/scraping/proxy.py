"""
Proxy manager for scraping anti-detection.

Manages datacenter and residential proxy pools with rotation, geo-matching,
and sticky sessions for multi-page crawls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import B2BScrapeConfig


@dataclass(frozen=True)
class ProxyConfig:
    """A single proxy endpoint."""

    url: str       # http://user:pass@host:port
    geo: str       # "US", "GB", etc.
    type: str      # "datacenter" or "residential"


_MAX_STICKY_DOMAINS = 1000


class ProxyManager:
    """Manages proxy rotation with sticky session support."""

    def __init__(self, proxies: list[ProxyConfig]) -> None:
        self._all = proxies
        self._datacenter = [p for p in proxies if p.type == "datacenter"]
        self._residential = [p for p in proxies if p.type == "residential"]
        self._sticky: dict[str, ProxyConfig] = {}

    @classmethod
    def from_config(cls, cfg: B2BScrapeConfig) -> ProxyManager:
        proxies: list[ProxyConfig] = []
        for url in cfg.proxy_datacenter_urls.split(","):
            url = url.strip()
            if url:
                proxies.append(ProxyConfig(url=url, geo="US", type="datacenter"))
        for url in cfg.proxy_residential_urls.split(","):
            url = url.strip()
            if url:
                proxies.append(ProxyConfig(url=url, geo=cfg.proxy_residential_geo, type="residential"))
        return cls(proxies)

    @property
    def has_proxies(self) -> bool:
        return len(self._all) > 0

    @property
    def has_residential(self) -> bool:
        return len(self._residential) > 0

    def get_proxy(
        self,
        *,
        domain: str,
        sticky: bool = False,
        prefer_residential: bool = False,
    ) -> ProxyConfig | None:
        """Select a proxy for the given domain.

        Args:
            domain: Target domain (used for sticky session key).
            sticky: Reuse the same proxy for this domain across calls.
            prefer_residential: Prefer residential proxies (for Cloudflare sites).

        Returns:
            ProxyConfig or None if no proxies are configured.
        """
        if not self._all:
            return None

        # Sticky: return previously assigned proxy for this domain
        if sticky and domain in self._sticky:
            return self._sticky[domain]

        # Select pool
        pool = self._all
        if prefer_residential and self._residential:
            pool = self._residential
        elif not prefer_residential and self._datacenter:
            pool = self._datacenter

        proxy = random.choice(pool)

        if sticky:
            # Evict oldest entries if over limit
            while len(self._sticky) >= _MAX_STICKY_DOMAINS:
                oldest = next(iter(self._sticky))
                del self._sticky[oldest]
            self._sticky[domain] = proxy

        return proxy

    def clear_sticky(self, domain: str) -> None:
        """Clear sticky session for a domain."""
        self._sticky.pop(domain, None)
