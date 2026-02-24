"""
News intake: poll news APIs, match against watchlist keywords,
deduplicate, and emit news.* events.

Supports NewsAPI.org (requires key) and Google News RSS (free fallback).
Runs as an autonomous task on a configurable interval (default 15 min).
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.news_intake")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: poll news and emit events."""
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
        return {"_skip_synthesis": True, "fetched": 0, "emitted": 0}

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
            if len(word) > 3:
                market_symbols.add(word)

    # Fetch articles
    articles = await _fetch_articles(
        cfg.news_api_provider, list(all_keywords), cfg.news_api_key, cfg.news_max_articles_per_poll
    )
    if not articles:
        return {"_skip_synthesis": True, "fetched": 0, "emitted": 0}

    emitted = 0
    for article in articles:
        title_lower = (article.get("title") or "").lower()
        desc_lower = (article.get("description") or "").lower()
        text = f"{title_lower} {desc_lower}"

        # Stage 1: keyword match
        matched_keywords: set[str] = set()
        matched_watchlist_ids: list[str] = []
        for wl in watchlist_items:
            for kw in wl["keywords"]:
                if kw in text:
                    matched_keywords.add(kw)
                    if wl["id"] not in matched_watchlist_ids:
                        matched_watchlist_ids.append(wl["id"])

        if not matched_keywords:
            continue

        # Dedup by article URL
        url = article.get("url", "")
        dedup_key = hashlib.sha256(url.encode()).hexdigest()

        inserted = await pool.fetchval(
            """
            INSERT INTO data_dedup (source, dedup_key)
            VALUES ('news', $1)
            ON CONFLICT (source, dedup_key) DO NOTHING
            RETURNING id
            """,
            dedup_key,
        )
        if not inserted:
            continue  # already processed

        # Check if any matched keywords overlap with market symbols
        is_market_moving = bool(matched_keywords & market_symbols)

        # Build event payload
        payload = {
            "article_id": dedup_key,
            "title": article.get("title", "")[:500],
            "source_name": article.get("source_name", "unknown"),
            "url": url,
            "published_at": article.get("published_at", ""),
            "summary": (article.get("description") or "")[:500],
            "matched_interests": sorted(matched_keywords),
            "matched_watchlist_ids": matched_watchlist_ids,
        }

        from ...reasoning.producers import emit_if_enabled
        from ...reasoning.events import EventType

        event_type = (
            EventType.NEWS_MARKET_MOVING if is_market_moving
            else EventType.NEWS_SIGNIFICANT
        )

        await emit_if_enabled(
            event_type=event_type,
            source="news_intake",
            payload=payload,
        )
        emitted += 1
        logger.info(
            "News event: [%s] %s (matched: %s)",
            event_type, article.get("title", "")[:80], ", ".join(sorted(matched_keywords)),
        )

    return {
        "_skip_synthesis": True,
        "fetched": len(articles),
        "emitted": emitted,
    }


async def _fetch_articles(
    provider: str,
    keywords: list[str],
    api_key: str | None,
    max_articles: int,
) -> list[dict[str, Any]]:
    """Fetch articles from the configured provider."""
    if provider == "newsapi" and api_key:
        return await _fetch_newsapi(keywords, api_key, max_articles)
    # Default fallback: Google News RSS (free, no key needed)
    return await _fetch_google_rss(keywords, max_articles)


async def _fetch_newsapi(
    keywords: list[str], api_key: str, max_articles: int
) -> list[dict[str, Any]]:
    """Fetch from NewsAPI.org."""
    import httpx

    # Combine keywords with OR for broad search
    query = " OR ".join(keywords[:10])  # API limit on query length

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": min(max_articles, 100),
                    "apiKey": api_key,
                    "language": "en",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for a in data.get("articles", [])[:max_articles]:
            articles.append({
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "source_name": (a.get("source") or {}).get("name", "unknown"),
                "published_at": a.get("publishedAt", ""),
            })
        return articles
    except Exception:
        logger.warning("NewsAPI fetch failed", exc_info=True)
        return []


async def _fetch_google_rss(
    keywords: list[str], max_articles: int
) -> list[dict[str, Any]]:
    """Fetch from Google News RSS feeds (free, no API key)."""
    def _sync_parse():
        import feedparser
        from urllib.parse import quote_plus

        articles = []
        seen_urls: set[str] = set()

        # Fetch a feed per keyword (or small groups)
        for kw in keywords[:5]:  # limit to avoid too many requests
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
            except Exception:
                pass  # individual feed failure is non-fatal

            if len(articles) >= max_articles:
                break

        return articles[:max_articles]

    try:
        return await asyncio.to_thread(_sync_parse)
    except Exception:
        logger.warning("Google RSS fetch failed", exc_info=True)
        return []
