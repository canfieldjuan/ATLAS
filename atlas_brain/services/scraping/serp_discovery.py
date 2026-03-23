"""SERP-based URL discovery for blocked review sites.

Uses Bright Data's SERP API to find indexed URLs on sites that block
direct scraping (Quora login wall, GetApp Cloudflare, Twitter/X).
Google has already crawled and indexed the content -- we just need the URLs.

Usage:
    urls = await discover_urls("quora.com", "HubSpot", ["alternative", "problems"])
    # Returns: ["https://www.quora.com/What-are-the-best-alternatives-to-HubSpot", ...]
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger("atlas.services.scraping.serp_discovery")


async def discover_urls(
    site_domain: str,
    vendor_name: str,
    query_suffixes: list[str],
    *,
    max_results_per_query: int = 10,
    serp_api_url: str = "",
    serp_api_token: str = "",
    serp_api_zone: str = "serp_api1",
    timeout: float = 60.0,
) -> list[str]:
    """Discover URLs on a blocked site via Google SERP API.

    Searches Google for ``site:{domain} {vendor} {suffix}`` for each suffix,
    extracts matching URLs, and returns a deduplicated list.

    Returns at most ``max_results_per_query * len(query_suffixes)`` URLs.
    """
    if not serp_api_token:
        from ...config import settings
        serp_api_token = settings.b2b_scrape.serp_api_token
        serp_api_zone = settings.b2b_scrape.serp_api_zone
        serp_api_url = settings.b2b_scrape.serp_api_url

    if not serp_api_token:
        logger.debug("SERP API token not configured, skipping discovery")
        return []

    headers = {
        "Authorization": f"Bearer {serp_api_token}",
        "Content-Type": "application/json",
    }

    seen: set[str] = set()
    urls: list[str] = []

    async with httpx.AsyncClient(timeout=timeout) as http:
        for suffix in query_suffixes:
            query = f"site:{site_domain} {vendor_name} {suffix}"
            google_url = (
                f"https://www.google.com/search"
                f"?q={quote_plus(query)}&num={max_results_per_query}"
            )

            try:
                resp = await http.post(
                    serp_api_url,
                    json={
                        "zone": serp_api_zone,
                        "url": google_url,
                        "format": "raw",
                    },
                    headers=headers,
                )
                if resp.status_code != 200:
                    logger.debug(
                        "SERP API returned %d for query: %s",
                        resp.status_code, query,
                    )
                    continue

                # Extract URLs matching the target domain
                pattern = re.compile(
                    rf"https?://(?:www\.)?{re.escape(site_domain)}/[^\s\"<>]+",
                )
                found = pattern.findall(resp.text)
                for url in found:
                    # Clean tracking params
                    clean = url.split("&")[0].split("?sa=")[0].rstrip("/")
                    if clean not in seen:
                        seen.add(clean)
                        urls.append(clean)

            except Exception as exc:
                logger.warning(
                    "SERP discovery failed for '%s': %s",
                    query, exc,
                )

    logger.info(
        "SERP discovery for %s on %s: %d URLs from %d queries",
        vendor_name, site_domain, len(urls), len(query_suffixes),
    )
    return urls


async def discover_urls_with_snippets(
    site_domain: str,
    vendor_name: str,
    query_suffixes: list[str],
    *,
    max_results_per_query: int = 10,
    serp_api_url: str = "",
    serp_api_token: str = "",
    serp_api_zone: str = "serp_api1",
    timeout: float = 60.0,
) -> list[dict[str, str]]:
    """Discover URLs with Google snippet text (useful for Twitter/X).

    Returns list of {url, snippet} dicts. Uses ``format=json`` to get
    structured Google results with organic[].description containing the
    cached page text (for tweets, this is the tweet text itself).
    """
    if not serp_api_token:
        from ...config import settings
        serp_api_token = settings.b2b_scrape.serp_api_token
        serp_api_zone = settings.b2b_scrape.serp_api_zone
        serp_api_url = settings.b2b_scrape.serp_api_url

    if not serp_api_token:
        return []

    headers = {
        "Authorization": f"Bearer {serp_api_token}",
        "Content-Type": "application/json",
    }

    seen: set[str] = set()
    results: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=timeout) as http:
        for suffix in query_suffixes:
            query = f"site:{site_domain} {vendor_name} {suffix}"
            google_url = (
                f"https://www.google.com/search"
                f"?q={quote_plus(query)}&num={max_results_per_query}"
            )

            try:
                resp = await http.post(
                    serp_api_url,
                    json={
                        "zone": serp_api_zone,
                        "url": google_url,
                        "format": "json",
                    },
                    headers=headers,
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                body = data.get("body")
                if isinstance(body, str):
                    import json as _json
                    try:
                        body = _json.loads(body)
                    except (ValueError, TypeError):
                        continue

                if not isinstance(body, dict):
                    continue

                organic = body.get("organic", [])
                for item in organic:
                    url = item.get("link", "")
                    if not url or site_domain not in url:
                        continue
                    clean = url.split("?sa=")[0].rstrip("/")
                    if clean in seen:
                        continue
                    seen.add(clean)

                    snippet = (
                        item.get("description")
                        or item.get("snippet")
                        or ""
                    )
                    results.append({
                        "url": clean,
                        "snippet": snippet,
                    })

            except Exception as exc:
                logger.warning(
                    "SERP discovery (with snippets) failed for '%s': %s",
                    query, exc,
                )

    logger.info(
        "SERP discovery (snippets) for %s on %s: %d results",
        vendor_name, site_domain, len(results),
    )
    return results
