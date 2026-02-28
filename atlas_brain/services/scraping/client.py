"""
Anti-detection HTTP client for B2B review scraping.

Wraps curl_cffi to produce Chrome/Firefox-identical TLS handshakes,
combined with proxy rotation, browser profile consistency, and
per-domain rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
import random

from curl_cffi.requests import AsyncSession, Response

from .captcha import CaptchaType, detect_captcha, get_captcha_proxy, get_captcha_solver, is_captcha_enabled_for_domain
from .profiles import BrowserProfile, BrowserProfileManager
from .proxy import ProxyConfig, ProxyManager
from .rate_limiter import DomainRateLimiter

logger = logging.getLogger("atlas.services.scraping.client")


class AntiDetectionClient:
    """HTTP client with TLS fingerprint spoofing and anti-detection measures."""

    def __init__(
        self,
        *,
        proxy_manager: ProxyManager,
        profile_manager: BrowserProfileManager,
        rate_limiter: DomainRateLimiter,
        min_delay: float = 2.0,
        max_delay: float = 8.0,
        max_retries: int = 2,
    ) -> None:
        self._proxy = proxy_manager
        self._profiles = profile_manager
        self._rate_limiter = rate_limiter
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._max_retries = max_retries
        # Per-domain cookie jars: solved CAPTCHA cookies persist across requests
        self._cookie_jars: dict[str, dict[str, str]] = {}

    def _get_domain_cookies(self, domain: str) -> dict[str, str]:
        """Get the cookie jar for a domain."""
        return self._cookie_jars.setdefault(domain, {})

    def _store_cookies_from_response(self, domain: str, resp: Response) -> None:
        """Capture Set-Cookie headers from a response into the domain jar."""
        jar = self._get_domain_cookies(domain)
        for cookie in resp.cookies.jar:
            jar[cookie.name] = cookie.value

    async def get(
        self,
        url: str,
        *,
        domain: str,
        referer: str | None = None,
        sticky_session: bool = False,
        prefer_residential: bool = False,
    ) -> Response:
        """Fetch a URL with anti-detection measures.

        Args:
            url: Target URL.
            domain: Domain name for rate limiting and proxy selection.
            referer: Referer header (e.g. Google search URL or previous page).
            sticky_session: Reuse same proxy across calls for this domain.
            prefer_residential: Use residential proxy (for Cloudflare sites).

        Returns:
            curl_cffi Response object.
        """
        last_exc: Exception | None = None
        captcha_enabled = is_captcha_enabled_for_domain(domain)
        max_retries = self._max_retries + (1 if captcha_enabled else 0)

        # Profile pinning: once a CAPTCHA is solved, reuse the same profile
        # so the User-Agent matches what was sent to the solver.
        pinned_profile: BrowserProfile | None = None
        # If the solver swapped the UA (e.g. CapSolver requires Chrome 137+),
        # override the header on the retry so cookies match the solved UA.
        override_ua: str | None = None
        # If the solver used a sticky proxy, override the proxy on retry
        # so the retry IP matches (critical for IP-bound cookies like cf_clearance).
        override_proxy: str | None = None

        attempt = 0
        while attempt <= max_retries:
            try:
                # 1. Rate limit
                await self._rate_limiter.acquire(domain)

                # 2. Select browser profile (pinned after CAPTCHA solve)
                profile = pinned_profile or self._profiles.get_profile()

                # 3. Get proxy
                proxy = self._proxy.get_proxy(
                    domain=domain,
                    sticky=sticky_session,
                    prefer_residential=prefer_residential,
                )

                # 4. Build headers
                headers = profile.build_headers(
                    referer=referer,
                    proxy_geo=proxy.geo if proxy else None,
                )
                if override_ua:
                    headers["User-Agent"] = override_ua

                # 5. Random delay (human-like)
                delay = random.uniform(self._min_delay, self._max_delay)
                if attempt > 0:
                    # Exponential backoff on retries
                    delay *= (2 ** attempt)
                await asyncio.sleep(delay)

                # 6. Execute via curl_cffi with matching TLS fingerprint
                domain_cookies = self._get_domain_cookies(domain)
                effective_proxy = override_proxy or (proxy.url if proxy else None)
                async with AsyncSession(impersonate=profile.impersonate) as session:
                    resp = await session.get(
                        url,
                        headers=headers,
                        cookies=domain_cookies if domain_cookies else None,
                        proxy=effective_proxy,
                        timeout=30,
                    )

                # Capture any Set-Cookie headers
                self._store_cookies_from_response(domain, resp)

                # 7. Check for CAPTCHA challenges on 403
                if resp.status_code == 403 and captcha_enabled:
                    captcha_type = detect_captcha(resp.text, resp.status_code)

                    if captcha_type != CaptchaType.NONE:
                        solver = get_captcha_solver()
                        if solver:
                            logger.info(
                                "CAPTCHA detected on %s (%s), solving attempt %d/%d",
                                domain, captcha_type.value,
                                attempt + 1, max_retries + 1,
                            )
                            try:
                                # Use sticky captcha proxy if configured,
                                # otherwise fall back to the current rotating proxy.
                                captcha_proxy = get_captcha_proxy()
                                solve_proxy = captcha_proxy or (proxy.url if proxy else None)
                                solution = await solver.solve(
                                    captcha_type=captcha_type,
                                    page_url=url,
                                    page_html=resp.text,
                                    user_agent=profile.user_agent,
                                    proxy_url=solve_proxy,
                                )
                                # Store solved cookies and pin profile with matching TLS fingerprint
                                self._get_domain_cookies(domain).update(solution.cookies)
                                if solution.user_agent:
                                    # Solver used a different UA — match impersonate to it
                                    pinned_profile = self._profiles.match_profile(solution.user_agent)
                                    override_ua = solution.user_agent
                                else:
                                    pinned_profile = profile
                                # Pin proxy so retry uses same IP as the solve
                                if solution.sticky_proxy:
                                    override_proxy = solution.sticky_proxy
                                logger.info(
                                    "CAPTCHA solved for %s in %dms, retrying (ua_override=%s, proxy_override=%s)",
                                    domain, solution.solve_time_ms,
                                    bool(solution.user_agent), bool(override_proxy),
                                )
                                # Successful solve always gets one more attempt.
                                # Don't increment attempt — the solve itself isn't a retry.
                                last_exc = None
                                continue
                            except Exception as solve_exc:
                                logger.warning(
                                    "CAPTCHA solve failed for %s: %s",
                                    domain, solve_exc,
                                )
                                # Fall through to normal 403 handling

                # Log non-200 responses
                if resp.status_code == 403:
                    logger.warning(
                        "Blocked (403) on %s attempt %d/%d (proxy=%s, profile=%s)",
                        domain, attempt + 1, max_retries + 1,
                        proxy.type if proxy else "none", profile.impersonate,
                    )
                    if attempt < max_retries:
                        # Clear sticky session on block so next attempt uses different proxy
                        self._proxy.clear_sticky(domain)
                        attempt += 1
                        continue
                elif resp.status_code == 429:
                    logger.warning("Rate limited (429) on %s, backing off", domain)
                    await asyncio.sleep(30 + random.uniform(5, 15))
                    if attempt < max_retries:
                        attempt += 1
                        continue

                return resp

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Request failed for %s attempt %d/%d: %s",
                    url, attempt + 1, max_retries + 1, exc,
                )
                if attempt < max_retries:
                    self._proxy.clear_sticky(domain)
                    await asyncio.sleep(random.uniform(2, 5))

            attempt += 1

        raise last_exc or RuntimeError(f"All retries exhausted for {url}")


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_client: AntiDetectionClient | None = None


def get_scrape_client() -> AntiDetectionClient:
    """Get or create the module-level scrape client singleton."""
    global _client
    if _client is None:
        from ...config import settings

        cfg = settings.b2b_scrape
        _client = AntiDetectionClient(
            proxy_manager=ProxyManager.from_config(cfg),
            profile_manager=BrowserProfileManager(),
            rate_limiter=DomainRateLimiter.from_config(cfg),
            min_delay=cfg.min_delay_seconds,
            max_delay=cfg.max_delay_seconds,
            max_retries=cfg.max_retries,
        )
    return _client
