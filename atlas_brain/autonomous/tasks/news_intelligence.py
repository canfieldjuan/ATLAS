"""
News intelligence task — daily pressure signal detection.

Runs once per day (default 5 AM) and:
1. Fetches recent news articles for each monitored topic via NewsAPI.
2. Compares today's article volume to the lookback baseline to compute
   a *velocity* — how much faster is this topic moving than usual?
3. Flags topics whose velocity exceeds the configured threshold as
   *pressure signals* — leading indicators that typically build up before
   the topic becomes mainstream headline news.
4. Returns structured results that are synthesised by the LLM engine into
   a plain-language intelligence briefing.

Requires a NewsAPI.org API key (free tier: 100 req/day).
Set ATLAS_NEWS_API_KEY in your .env file.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.news_intelligence")

# Threshold below which we treat the baseline as "near zero" / emerging topic.
# At < 0.5 articles/day the topic has essentially no established baseline,
# so any recent activity represents genuine emergence rather than acceleration.
_MIN_BASELINE_DAILY = 0.5
# HTTP timeout for NewsAPI calls
_HTTP_TIMEOUT = 15.0


async def run(task: ScheduledTask) -> dict[str, Any]:
    """
    Execute the daily news intelligence analysis.

    Configurable via task.metadata:
        topics (str): Comma-separated topics override (default: settings.news_intel.topics)
        lookback_days (int): History window override (default: settings.news_intel.lookback_days)
    """
    cfg = settings.news_intel
    metadata = task.metadata or {}

    if not cfg.enabled:
        return {"status": "disabled", "message": "News intelligence is disabled (ATLAS_NEWS_ENABLED=false)"}

    if not cfg.api_key:
        return {
            "status": "unconfigured",
            "message": "No NewsAPI key configured. Set ATLAS_NEWS_API_KEY in your .env file.",
        }

    # Allow per-task overrides from metadata
    raw_topics = metadata.get("topics", cfg.topics)
    topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
    lookback_days = int(metadata.get("lookback_days", cfg.lookback_days))
    regions = [r.strip() for r in cfg.regions.split(",") if r.strip()]

    if not topics:
        return {"status": "no_topics", "message": "No topics configured to monitor."}

    logger.info("News intelligence: analysing %d topic(s) over %d-day window", len(topics), lookback_days)

    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []
    topic_results: list[dict] = []

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        for topic in topics:
            result = await _analyse_topic(client, topic, regions, lookback_days, now_utc, cfg)
            topic_results.append(result)
            if result.get("is_signal"):
                signals.append(result)
                logger.info(
                    "Pressure signal detected: '%s' velocity=%.2f recent_count=%d",
                    topic,
                    result.get("velocity", 0),
                    result.get("recent_count", 0),
                )

    summary = _build_summary(signals, topic_results, now_utc)

    return {
        "status": "ok",
        "date": now_utc.strftime("%Y-%m-%d"),
        "topics_analysed": len(topics),
        "signals_detected": len(signals),
        "signals": signals,
        "all_topics": topic_results,
        "summary": summary,
    }


async def _analyse_topic(
    client: httpx.AsyncClient,
    topic: str,
    regions: list[str],
    lookback_days: int,
    now_utc: datetime,
    cfg: Any,
) -> dict[str, Any]:
    """
    Fetch and analyse news volume for a single topic.

    Returns a dict with:
    - topic, velocity, recent_count, baseline_count, is_signal
    - top_headlines: up to 3 article titles from the most-recent day
    """
    # Date boundaries
    lookback_start = (now_utc - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent_start = (now_utc - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build query — optionally prefix with first region
    query_parts = [topic]
    if regions:
        query_parts.insert(0, regions[0])
    query = " ".join(query_parts)

    # Fetch articles for full lookback window
    try:
        resp = await client.get(
            _NEWSAPI_BASE,
            params={
                "q": query,
                "from": lookback_start,
                "sortBy": "publishedAt",
                "language": cfg.languages.split(",")[0].strip(),
                "pageSize": cfg.max_articles_per_topic,
                "apiKey": cfg.api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("NewsAPI HTTP error for topic '%s': %s", topic, exc)
        return _error_result(topic, str(exc))
    except Exception as exc:
        logger.warning("NewsAPI request failed for topic '%s': %s", topic, exc)
        return _error_result(topic, str(exc))

    articles = data.get("articles", [])
    total_fetched = len(articles)

    # Split into recent (last 1 day) vs historical (days 2..lookback)
    recent_articles = []
    historical_articles = []

    for article in articles:
        pub = article.get("publishedAt", "")
        if pub >= recent_start:
            recent_articles.append(article)
        else:
            historical_articles.append(article)

    recent_count = len(recent_articles)
    historical_count = len(historical_articles)

    # Compute velocity: recent vs daily average of historical window
    # Baseline = historical_count / (lookback_days - 1)  (avoid div-zero)
    historical_days = max(lookback_days - 1, 1)
    baseline_daily = historical_count / historical_days

    # Avoid division by zero when baseline is tiny
    if baseline_daily < _MIN_BASELINE_DAILY:
        # If baseline is near zero and recent > 0 it's genuinely emerging
        velocity = float(recent_count) if recent_count > 0 else 0.0
    else:
        velocity = recent_count / baseline_daily

    is_signal = (
        velocity >= cfg.pressure_velocity_threshold
        and recent_count >= cfg.signal_min_articles
    )

    # Top headlines from the most recent day
    top_headlines = [
        a.get("title", "(no title)") for a in recent_articles[:3]
        if a.get("title") and "[Removed]" not in a.get("title", "")
    ]

    return {
        "topic": topic,
        "velocity": round(velocity, 2),
        "recent_count": recent_count,
        "baseline_daily": round(baseline_daily, 2),
        "historical_count": historical_count,
        "total_fetched": total_fetched,
        "is_signal": is_signal,
        "top_headlines": top_headlines,
    }


def _error_result(topic: str, error: str) -> dict[str, Any]:
    return {
        "topic": topic,
        "velocity": 0.0,
        "recent_count": 0,
        "baseline_daily": 0.0,
        "historical_count": 0,
        "total_fetched": 0,
        "is_signal": False,
        "top_headlines": [],
        "error": error,
    }


def _build_summary(signals: list[dict], all_topics: list[dict], now_utc: datetime) -> str:
    """Build a plain-language intelligence briefing."""
    date_str = now_utc.strftime("%B %d, %Y")

    if not all_topics:
        return f"News intelligence ({date_str}): No topics analysed."

    if not signals:
        monitored = ", ".join(t["topic"] for t in all_topics if "error" not in t)
        return (
            f"News intelligence ({date_str}): No pressure signals detected. "
            f"Topics monitored: {monitored}."
        )

    parts = [f"News intelligence ({date_str}): {len(signals)} pressure signal(s) detected."]

    for sig in signals:
        topic = sig["topic"]
        velocity = sig["velocity"]
        recent = sig["recent_count"]
        headlines = sig.get("top_headlines", [])

        pct = int((velocity - 1.0) * 100) if velocity >= 1.0 else 0
        line = f"▲ '{topic}': {recent} article(s) today ({pct}% above baseline)."
        if headlines:
            line += f" Headlines: {headlines[0]}"
            if len(headlines) > 1:
                line += f"; {headlines[1]}"
        parts.append(line)

    return " ".join(parts)
