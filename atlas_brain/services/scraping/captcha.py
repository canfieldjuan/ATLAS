"""
CAPTCHA detection and solving for B2B review scraping.

Detects DataDome (G2) and Cloudflare (Capterra) challenges from 403 responses,
then solves them via CapSolver or 2Captcha APIs.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import re
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger("atlas.services.scraping.captcha")

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class CaptchaType(enum.Enum):
    NONE = "none"
    DATADOME = "datadome"        # G2
    CLOUDFLARE = "cloudflare"    # Capterra


def detect_captcha(response_text: str, status_code: int) -> CaptchaType:
    """Inspect a response for known CAPTCHA challenges.

    Args:
        response_text: HTML body of the response.
        status_code: HTTP status code.

    Returns:
        CaptchaType indicating which challenge was found (or NONE).
    """
    if status_code != 403:
        return CaptchaType.NONE

    body = response_text.lower()

    # DataDome: JS challenge page served by captcha-delivery.com
    if "captcha-delivery.com" in body and "var dd=" in body:
        return CaptchaType.DATADOME

    # Cloudflare managed challenge
    if "attention required" in body and "cloudflare" in body:
        return CaptchaType.CLOUDFLARE

    return CaptchaType.NONE


# ---------------------------------------------------------------------------
# DataDome parameter extraction
# ---------------------------------------------------------------------------

_DD_PATTERN = re.compile(
    r"var\s+dd\s*=\s*\{([^}]+)\}",
    re.DOTALL,
)


def _extract_datadome_params(html: str) -> dict[str, str]:
    """Extract cid, hsh, host from DataDome's ``var dd={...}`` JS block."""
    m = _DD_PATTERN.search(html)
    if not m:
        return {}

    raw = m.group(1)
    params: dict[str, str] = {}
    for key in ("cid", "hsh", "host"):
        km = re.search(rf"['\"]?{key}['\"]?\s*:\s*['\"]([^'\"]+)['\"]", raw)
        if km:
            params[key] = km.group(1)
    return params


# ---------------------------------------------------------------------------
# Solution model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptchaSolution:
    """Result of a successful CAPTCHA solve."""

    cookies: dict[str, str]
    solve_time_ms: int


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

# Polling settings
_POLL_INTERVAL_S = 3
_MAX_POLL_ATTEMPTS = 60  # 3s * 60 = 180s max wait


class CaptchaSolver:
    """Solves DataDome and Cloudflare CAPTCHAs via CapSolver or 2Captcha."""

    def __init__(self, provider: str, api_key: str) -> None:
        if provider not in ("capsolver", "2captcha"):
            raise ValueError(f"Unsupported CAPTCHA provider: {provider}")
        self._provider = provider
        self._api_key = api_key

    async def solve(
        self,
        captcha_type: CaptchaType,
        page_url: str,
        page_html: str,
        user_agent: str,
        proxy_url: str | None = None,
    ) -> CaptchaSolution:
        """Solve a detected CAPTCHA challenge.

        Args:
            captcha_type: DATADOME or CLOUDFLARE.
            page_url: The URL that returned the challenge.
            page_html: Full HTML body of the 403 response.
            user_agent: UA string used for the blocked request.
            proxy_url: Proxy URL used (some solvers need it).

        Returns:
            CaptchaSolution with cookies to bypass the challenge.
        """
        t0 = time.monotonic()

        if self._provider == "capsolver":
            cookies = await self._solve_capsolver(
                captcha_type, page_url, page_html, user_agent, proxy_url,
            )
        else:
            cookies = await self._solve_2captcha(
                captcha_type, page_url, page_html, user_agent, proxy_url,
            )

        elapsed = int((time.monotonic() - t0) * 1000)
        logger.info(
            "CAPTCHA solved: type=%s provider=%s time=%dms",
            captcha_type.value, self._provider, elapsed,
        )
        return CaptchaSolution(cookies=cookies, solve_time_ms=elapsed)

    # -- CapSolver ----------------------------------------------------------

    async def _solve_capsolver(
        self,
        captcha_type: CaptchaType,
        page_url: str,
        page_html: str,
        user_agent: str,
        proxy_url: str | None,
    ) -> dict[str, str]:
        base = "https://api.capsolver.com"

        if captcha_type == CaptchaType.DATADOME:
            dd = _extract_datadome_params(page_html)
            if not dd or "cid" not in dd:
                logger.warning("Failed to extract DataDome params from challenge page")
            task: dict = {
                "type": "DatadomeSliderTask" if proxy_url else "DatadomeSliderTaskProxyless",
                "websiteURL": page_url,
                "captchaUrl": f"https://{dd.get('host', 'geo.captcha-delivery.com')}/captcha/?initialCid={dd.get('cid', '')}&hash={dd.get('hsh', '')}",
                "userAgent": user_agent,
            }
            if proxy_url:
                task["proxy"] = proxy_url

        elif captcha_type == CaptchaType.CLOUDFLARE:
            task = {
                "type": "AntiCloudflareTask" if proxy_url else "AntiCloudflareTaskProxyLess",
                "websiteURL": page_url,
                "metadata": {"type": "managed"},
            }
            if proxy_url:
                task["proxy"] = proxy_url
        else:
            raise ValueError(f"Unsupported captcha type for CapSolver: {captcha_type}")

        # Create task
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base}/createTask",
                json={"clientKey": self._api_key, "task": task},
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("errorId", 0) != 0:
            raise RuntimeError(f"CapSolver createTask error: {data.get('errorDescription', data)}")

        task_id = data["taskId"]
        return await self._poll_capsolver(base, task_id)

    async def _poll_capsolver(self, base: str, task_id: str) -> dict[str, str]:
        async with httpx.AsyncClient(timeout=30) as client:
            for _ in range(_MAX_POLL_ATTEMPTS):
                await asyncio.sleep(_POLL_INTERVAL_S)
                resp = await client.post(
                    f"{base}/getTaskResult",
                    json={"clientKey": self._api_key, "taskId": task_id},
                )
                resp.raise_for_status()
                data = resp.json()

                status = data.get("status", "")
                if status == "ready":
                    solution = data.get("solution", {})
                    return _extract_cookies(solution)
                if status == "failed":
                    raise RuntimeError(f"CapSolver task failed: {data.get('errorDescription', data)}")

        raise TimeoutError(f"CapSolver task {task_id} timed out after {_POLL_INTERVAL_S * _MAX_POLL_ATTEMPTS}s")

    # -- 2Captcha ----------------------------------------------------------

    async def _solve_2captcha(
        self,
        captcha_type: CaptchaType,
        page_url: str,
        page_html: str,
        user_agent: str,
        proxy_url: str | None,
    ) -> dict[str, str]:
        base = "https://2captcha.com"

        if captcha_type == CaptchaType.DATADOME:
            dd = _extract_datadome_params(page_html)
            if not dd or "cid" not in dd:
                logger.warning("Failed to extract DataDome params from challenge page (2Captcha path)")
            captcha_url = f"https://{dd.get('host', 'geo.captcha-delivery.com')}/captcha/?initialCid={dd.get('cid', '')}&hash={dd.get('hsh', '')}"
            params: dict = {
                "key": self._api_key,
                "method": "datadome",
                "captcha_url": captcha_url,
                "pageurl": page_url,
                "userAgent": user_agent,
                "json": 1,
            }
            if proxy_url:
                params["proxy"] = proxy_url
                params["proxytype"] = "HTTP"

        elif captcha_type == CaptchaType.CLOUDFLARE:
            params = {
                "key": self._api_key,
                "method": "managed",
                "sitekey": "",  # 2Captcha extracts from page
                "pageurl": page_url,
                "userAgent": user_agent,
                "json": 1,
            }
            if proxy_url:
                params["proxy"] = proxy_url
                params["proxytype"] = "HTTP"
        else:
            raise ValueError(f"Unsupported captcha type for 2Captcha: {captcha_type}")

        # Submit task
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{base}/in.php", data=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != 1:
            raise RuntimeError(f"2Captcha submit error: {data.get('request', data)}")

        task_id = data["request"]
        return await self._poll_2captcha(base, task_id)

    async def _poll_2captcha(self, base: str, task_id: str) -> dict[str, str]:
        async with httpx.AsyncClient(timeout=30) as client:
            for _ in range(_MAX_POLL_ATTEMPTS):
                await asyncio.sleep(_POLL_INTERVAL_S)
                resp = await client.get(
                    f"{base}/res.php",
                    params={
                        "key": self._api_key,
                        "action": "get",
                        "id": task_id,
                        "json": 1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("status") == 1:
                    return _parse_2captcha_solution(data.get("request", ""))
                if data.get("request") not in ("CAPCHA_NOT_READY",):
                    raise RuntimeError(f"2Captcha error: {data.get('request', data)}")

        raise TimeoutError(f"2Captcha task {task_id} timed out after {_POLL_INTERVAL_S * _MAX_POLL_ATTEMPTS}s")


# ---------------------------------------------------------------------------
# Cookie extraction helpers
# ---------------------------------------------------------------------------


def _extract_cookies(solution: dict) -> dict[str, str]:
    """Extract cookies from a CapSolver solution response.

    CapSolver returns cookies in various formats depending on task type:
    - ``{"cookie": "datadome=abc123; ..."}``  (DataDome)
    - ``{"cookies": {"cf_clearance": "abc"}}`` (Cloudflare)
    - ``{"cookie": {"datadome": "abc"}}``
    """
    cookies: dict[str, str] = {}

    raw = solution.get("cookie") or solution.get("cookies")
    if isinstance(raw, dict):
        cookies.update(raw)
    elif isinstance(raw, str):
        # Parse "key=value; key2=value2" format
        for part in raw.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                cookies[k.strip()] = v.strip()

    if not cookies:
        logger.warning("No cookies extracted from solution: %s", solution)
    return cookies


def _parse_2captcha_solution(raw: str) -> dict[str, str]:
    """Parse 2Captcha solution string into cookies dict.

    2Captcha returns cookies as ``key=value; key2=value2`` in the request field.
    """
    cookies: dict[str, str] = {}
    for part in raw.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k.strip()] = v.strip()

    if not cookies:
        logger.warning("No cookies parsed from 2Captcha solution: %s", raw)
    return cookies


# ---------------------------------------------------------------------------
# Module singleton + helpers
# ---------------------------------------------------------------------------

_solver: CaptchaSolver | None = None
_enabled_domains: set[str] = set()
_checked: bool = False


def get_captcha_solver() -> CaptchaSolver | None:
    """Get the module-level CAPTCHA solver, or None if disabled."""
    global _solver, _enabled_domains, _checked
    if _solver is not None:
        return _solver
    if _checked:
        return None

    from ...config import settings

    cfg = settings.b2b_scrape
    _checked = True
    if not cfg.captcha_enabled or not cfg.captcha_api_key:
        return None

    _solver = CaptchaSolver(provider=cfg.captcha_provider, api_key=cfg.captcha_api_key)
    _enabled_domains = {d.strip().lower() for d in cfg.captcha_domains.split(",") if d.strip()}
    logger.info(
        "CAPTCHA solver initialized: provider=%s, domains=%s",
        cfg.captcha_provider, _enabled_domains,
    )
    return _solver


def is_captcha_enabled_for_domain(domain: str) -> bool:
    """Check if CAPTCHA solving is enabled for a specific domain."""
    # Ensure singleton is initialized
    get_captcha_solver()
    return domain.lower() in _enabled_domains
