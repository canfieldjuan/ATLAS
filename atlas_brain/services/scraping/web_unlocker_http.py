"""Shared HTTP helpers for Bright Data Web Unlocker."""

from __future__ import annotations

import httpx

from ...config import settings


def build_web_unlocker_timeout() -> httpx.Timeout:
    total = float(settings.b2b_scrape.web_unlocker_timeout_seconds)
    phase_timeout = min(total, 15.0)
    return httpx.Timeout(
        timeout=total,
        connect=phase_timeout,
        read=total,
        write=phase_timeout,
        pool=phase_timeout,
    )


async def get_with_web_unlocker(
    url: str,
    *,
    headers: dict[str, str],
    domain: str,
) -> httpx.Response:
    proxy_url = str(settings.b2b_scrape.web_unlocker_url or "").strip()
    if not proxy_url:
        raise RuntimeError("Web Unlocker is not configured")
    try:
        async with httpx.AsyncClient(
            proxy=proxy_url,
            verify=False,
            timeout=build_web_unlocker_timeout(),
        ) as http:
            return await http.get(url, headers=headers)
    except httpx.TimeoutException as exc:
        timeout_seconds = settings.b2b_scrape.web_unlocker_timeout_seconds
        raise RuntimeError(
            f"Web Unlocker timed out for {domain} after {timeout_seconds:g}s"
        ) from exc
