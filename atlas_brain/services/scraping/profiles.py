"""
Browser profile manager for anti-detection.

Maintains a registry of Chrome/Firefox profiles where each profile is a
consistent bundle: curl_cffi impersonate target, User-Agent string, platform,
and header set. Ensures UA always matches TLS fingerprint.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BrowserProfile:
    """A consistent browser identity bundle."""

    impersonate: str       # curl_cffi target, e.g. "chrome120"
    user_agent: str        # Matching UA string
    platform: str          # "Windows", "Macintosh", "X11"
    sec_ch_ua: str         # Sec-CH-UA header value
    sec_ch_ua_platform: str  # Sec-CH-UA-Platform header value

    def build_headers(
        self,
        *,
        referer: str | None = None,
        proxy_geo: str | None = None,
    ) -> dict[str, str]:
        """Build a complete, realistic browser header set."""
        # Geo-matched Accept-Language
        lang = _GEO_LANG.get(proxy_geo or "US", "en-US,en;q=0.9")

        headers: dict[str, str] = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": lang,
            "Accept-Encoding": "gzip, deflate, br",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site" if referer else "none",
            "Sec-Fetch-User": "?1",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
        }

        # Chrome-specific Sec-CH-UA headers
        if self.impersonate.startswith("chrome"):
            headers["Sec-CH-UA"] = self.sec_ch_ua
            headers["Sec-CH-UA-Mobile"] = "?0"
            headers["Sec-CH-UA-Platform"] = self.sec_ch_ua_platform

        if referer:
            headers["Referer"] = referer

        return headers


# Geo -> Accept-Language mapping
_GEO_LANG: dict[str, str] = {
    "US": "en-US,en;q=0.9",
    "GB": "en-GB,en;q=0.9",
    "CA": "en-CA,en;q=0.9",
    "AU": "en-AU,en;q=0.9",
    "DE": "de-DE,de;q=0.9,en;q=0.8",
    "FR": "fr-FR,fr;q=0.9,en;q=0.8",
}


# Pre-defined profiles -- impersonate target MUST match UA string
_PROFILES: list[BrowserProfile] = [
    BrowserProfile(
        impersonate="chrome120",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        platform="Windows",
        sec_ch_ua='"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        sec_ch_ua_platform='"Windows"',
    ),
    BrowserProfile(
        impersonate="chrome124",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        platform="Windows",
        sec_ch_ua='"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        sec_ch_ua_platform='"Windows"',
    ),
    BrowserProfile(
        impersonate="chrome131",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        platform="Macintosh",
        sec_ch_ua='"Chromium";v="131", "Google Chrome";v="131", "Not_A Brand";v="24"',
        sec_ch_ua_platform='"macOS"',
    ),
    BrowserProfile(
        impersonate="chrome142",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
        platform="Windows",
        sec_ch_ua='"Chromium";v="142", "Google Chrome";v="142", "Not-A.Brand";v="99"',
        sec_ch_ua_platform='"Windows"',
    ),
    BrowserProfile(
        impersonate="firefox133",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
        platform="Windows",
        sec_ch_ua="",
        sec_ch_ua_platform="",
    ),
]


class BrowserProfileManager:
    """Selects browser profiles with weighted randomization.

    Chrome profiles are selected 80% of the time, Firefox 20%.
    """

    def __init__(self) -> None:
        self._chrome = [p for p in _PROFILES if p.impersonate.startswith("chrome")]
        self._firefox = [p for p in _PROFILES if p.impersonate.startswith("firefox")]

    def get_profile(self) -> BrowserProfile:
        """Get a random browser profile (80% Chrome, 20% Firefox)."""
        if self._firefox and random.random() < 0.2:
            return random.choice(self._firefox)
        return random.choice(self._chrome)

    def match_profile(self, user_agent: str) -> BrowserProfile:
        """Find the closest Chrome profile matching a User-Agent string.

        Used after CAPTCHA solving to match the TLS fingerprint (impersonate)
        to the Chrome version the solver used. Cloudflare binds cf_clearance
        to the exact TLS fingerprint, so the impersonate must be close.
        """
        m = re.search(r"Chrome/(\d+)", user_agent)
        if not m:
            return self._chrome[-1]  # default to newest Chrome

        target_ver = int(m.group(1))

        def _chrome_ver(p: BrowserProfile) -> int:
            vm = re.search(r"chrome(\d+)", p.impersonate)
            return int(vm.group(1)) if vm else 0

        return min(self._chrome, key=lambda p: abs(_chrome_ver(p) - target_ver))
