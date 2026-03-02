"""
News intake: poll news APIs, match against watchlist keywords,
deduplicate, and store in news_articles for daily intelligence.

Supports Mediastack (requires key) and Google News RSS (free fallback).
Runs as an autonomous task on a configurable interval (default 15 min).
"""

import asyncio
import hashlib
import logging
import re
import uuid
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.news_intake")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: poll news and store articles for daily intelligence."""
    cfg = settings.external_data
    if not cfg.enabled or not cfg.news_enabled:
        return {"_skip_synthesis": True, "skipped": "external_data or news disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": True, "skipped": "db not ready"}

    # Load news watchlist items
    rows = await pool.fetch(
        """
        SELECT id, category, name, keywords, metadata
        FROM data_watchlist
        WHERE enabled = true
          AND category IN ('news_topic', 'news_region')
          AND keywords IS NOT NULL
          AND array_length(keywords, 1) > 0
        """
    )
    if not rows:
        return {"_skip_synthesis": True, "fetched": 0, "stored": 0}

    # Build keyword set and watchlist lookup
    all_keywords: set[str] = set()
    watchlist_items = []
    for r in rows:
        kws = [k.lower() for k in (r["keywords"] or [])]
        all_keywords.update(kws)
        watchlist_items.append({
            "id": str(r["id"]),
            "name": r["name"],
            "category": r["category"],
            "keywords": set(kws),
            "metadata": r["metadata"] or {},
        })

    # Also load market watchlist symbols for cross-referencing
    market_symbols = set()
    market_rows = await pool.fetch(
        """
        SELECT LOWER(symbol) AS sym, LOWER(name) AS name
        FROM data_watchlist
        WHERE enabled = true
          AND category IN ('stock', 'etf', 'commodity', 'crypto', 'forex')
          AND symbol IS NOT NULL
        """
    )
    for mr in market_rows:
        market_symbols.add(mr["sym"])
        # Also add name words for matching (e.g. "coffee" from "Coffee Futures")
        for word in mr["name"].split():
            if len(word) > cfg.news_keyword_min_length:
                market_symbols.add(word)

    # Fetch articles
    articles = await _fetch_articles(
        cfg.news_api_provider, list(all_keywords), cfg.news_api_key,
        cfg.news_max_articles_per_poll, cfg,
    )
    if not articles:
        return {"_skip_synthesis": True, "fetched": 0, "stored": 0}

    stored = 0
    for article in articles:
        title_lower = (article.get("title") or "").lower()
        desc_lower = (article.get("description") or "").lower()
        text = f"{title_lower} {desc_lower}"

        # Stage 1: keyword match (word-boundary to avoid "apple pie" matching "apple")
        matched_keywords: set[str] = set()
        matched_watchlist_ids: list[str] = []
        for wl in watchlist_items:
            for kw in wl["keywords"]:
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    matched_keywords.add(kw)
                    if wl["id"] not in matched_watchlist_ids:
                        matched_watchlist_ids.append(wl["id"])

        if not matched_keywords:
            continue

        # Dedup by article URL
        url = article.get("url", "")
        dedup_key = hashlib.sha256(url.encode()).hexdigest()

        # Check if any matched keywords overlap with market symbols
        is_market_moving = bool(matched_keywords & market_symbols)

        title = article.get("title", "")[:500]
        source_name = article.get("source_name", "unknown")
        published_at = article.get("published_at", "")
        summary = (article.get("description") or "")[:500]

        # Store in news_articles for daily intelligence analysis
        try:
            wl_uuids = [uuid.UUID(wid) for wid in matched_watchlist_ids]
            inserted = await pool.fetchval(
                """
                INSERT INTO news_articles (dedup_key, title, source_name, url, published_at,
                                           summary, matched_keywords, matched_watchlist_ids, is_market_related)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::uuid[], $9)
                ON CONFLICT (dedup_key) DO NOTHING
                RETURNING id
                """,
                dedup_key, title, source_name, url, published_at, summary,
                sorted(matched_keywords), wl_uuids, is_market_moving,
            )
        except Exception:
            logger.debug("news_articles insert failed for %s", dedup_key, exc_info=True)
            inserted = None

        if not inserted:
            continue  # already stored

        stored += 1
        logger.info(
            "Stored news article: %s (matched: %s, market=%s)",
            title[:80], ", ".join(sorted(matched_keywords)), is_market_moving,
        )

    return {
        "_skip_synthesis": True,
        "fetched": len(articles),
        "stored": stored,
    }


async def _fetch_articles(
    provider: str,
    keywords: list[str],
    api_key: str | None,
    max_articles: int,
    cfg,
) -> list[dict[str, Any]]:
    """Fetch articles from the configured provider."""
    if provider == "mediastack" and api_key:
        return await _fetch_mediastack(
            keywords, api_key, max_articles,
            max_keywords=cfg.news_max_keywords_per_query,
            timeout=cfg.api_timeout_seconds,
        )
    # Default fallback: Google News RSS (free, no key needed)
    return await _fetch_google_rss(
        keywords, max_articles,
        max_feeds=cfg.news_max_rss_feeds,
        timeout=cfg.api_timeout_seconds,
    )


async def _fetch_mediastack(
    keywords: list[str],
    api_key: str,
    max_articles: int,
    max_keywords: int = 10,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """Fetch from Mediastack API."""
    import httpx

    # Mediastack accepts comma-separated keywords
    kw_str = ",".join(keywords[:max_keywords])

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "http://api.mediastack.com/v1/news",
                params={
                    "access_key": api_key,
                    "keywords": kw_str,
                    "languages": "en",
                    "limit": min(max_articles, 100),
                    "sort": "published_desc",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            logger.warning("Mediastack API error: %s", data["error"].get("message", data["error"]))
            return []

        articles = []
        for a in data.get("data", [])[:max_articles]:
            articles.append({
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "source_name": a.get("source", "unknown"),
                "published_at": a.get("published_at", ""),
            })
        return articles
    except Exception:
        logger.warning("Mediastack fetch failed", exc_info=True)
        return []


async def _fetch_google_rss(
    keywords: list[str],
    max_articles: int,
    max_feeds: int = 5,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """Fetch from Google News RSS feeds (free, no API key)."""
    _max_feeds = max_feeds

    def _sync_parse():
        import feedparser
        from urllib.parse import quote_plus

        articles = []
        seen_urls: set[str] = set()

        for kw in keywords[:_max_feeds]:
            url = f"https://news.google.com/rss/search?q={quote_plus(kw)}&hl=en-US&gl=US&ceid=US:en"
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_articles]:
                    link = entry.get("link", "")
                    if link in seen_urls:
                        continue
                    seen_urls.add(link)
                    articles.append({
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", ""),
                        "url": link,
                        "source_name": entry.get("source", {}).get("title", "Google News"),
                        "published_at": entry.get("published", ""),
                    })
            except Exception as e:
                logger.warning("RSS feed parse failed for keyword: %s", e)

            if len(articles) >= max_articles:
                break

        return articles[:max_articles]

    try:
        return await asyncio.to_thread(_sync_parse)
    except Exception:
        logger.warning("Google RSS fetch failed", exc_info=True)
        return []
