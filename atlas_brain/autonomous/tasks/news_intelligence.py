"""
News intelligence task — entity-level pressure signal detection.

Runs once per day (default 5 AM) and, for each entity in the configured
watchlist (companies, sports teams, markets, crypto, or custom topics):

1. Fetches recent news articles via NewsAPI.
2. Measures *volume velocity* — how many more articles today vs the baseline?
3. Measures *sentiment shift* — is the tone suddenly more negative/positive?
4. Measures *source diversity* — is the story spreading to new outlets?
5. Computes a composite *pressure score* from all three dimensions.
6. Flags entities whose composite score exceeds the threshold as
   *pre-movement signals* — leading indicators that typically build up
   before a price move, odds shift, or public narrative change.

This is intentionally a proxy-only approach: no ML models, no paid data feeds,
just structured observation of publicly available article patterns.

Requires a NewsAPI.org API key (free tier: 100 req/day).
Set ATLAS_NEWS_API_KEY in your .env file.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.news_intelligence")

_NEWSAPI_BASE = "https://newsapi.org/v2/everything"
_HTTP_TIMEOUT = 15.0

# Minimum baseline below which we treat a topic as "emerging" (no prior coverage).
# At < 0.5 articles/day the entity has no established baseline so any recent
# activity is genuine emergence rather than acceleration.
_MIN_BASELINE_DAILY = 0.5

# Keyword proxies for sentiment scoring (no external NLP dependency).
# These are intentionally broad — the goal is direction, not precision.
_NEGATIVE_KEYWORDS = frozenset([
    "lawsuit", "sue", "sued", "fraud", "scandal", "crash", "collapse",
    "decline", "investigation", "probe", "recall", "warning", "concern",
    "risk", "downgrade", "sell", "loss", "miss", "below", "cut", "fail",
    "ban", "halt", "suspend", "resign", "arrest", "fine", "penalty",
    "breach", "hack", "leak", "layoff", "downfall", "crisis",
])
_POSITIVE_KEYWORDS = frozenset([
    "beat", "upgrade", "record", "growth", "expansion", "partnership",
    "approve", "exceed", "raise", "win", "victory", "surge", "rally",
    "buy", "profit", "gain", "launch", "deal", "merger", "acquire",
    "innovation", "breakthrough", "all-time", "milestone", "best",
])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run(task: ScheduledTask) -> dict[str, Any]:
    """
    Execute the daily news intelligence analysis.

    Configurable via task.metadata:
        watchlist (str): JSON watchlist override
        lookback_days (int): History window override
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

    # Build entity list from watchlist JSON first, fall back to simple topics
    entities = _resolve_entities(
        metadata.get("watchlist", cfg.watchlist),
        metadata.get("topics", cfg.topics),
        cfg.regions,
    )

    if not entities:
        return {"status": "no_entities", "message": "No entities or topics configured to monitor."}

    lookback_days = int(metadata.get("lookback_days", cfg.lookback_days))

    logger.info(
        "News intelligence: analysing %d entity/entities over %d-day window (sentiment=%s, diversity=%s)",
        len(entities), lookback_days, cfg.sentiment_enabled, cfg.source_diversity_enabled,
    )

    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []
    entity_results: list[dict] = []

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        for entity in entities:
            result = await _analyse_entity(client, entity, lookback_days, now_utc, cfg)
            entity_results.append(result)
            if result.get("is_signal"):
                signals.append(result)
                logger.info(
                    "Pre-movement pressure: '%s' (%s) score=%.2f velocity=%.2f sentiment=%.2f diversity=%.2f",
                    entity["name"], entity["type"],
                    result.get("composite_score", 0),
                    result.get("velocity", 0),
                    result.get("sentiment_score", 0),
                    result.get("diversity_score", 0),
                )

    summary = _build_summary(signals, entity_results, now_utc)

    return {
        "status": "ok",
        "date": now_utc.strftime("%Y-%m-%d"),
        "entities_analysed": len(entities),
        "signals_detected": len(signals),
        "signals": signals,
        "all_entities": entity_results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------

def _resolve_entities(watchlist_json: str, topics_csv: str, regions: str) -> list[dict[str, str]]:
    """
    Parse the watchlist JSON into a list of entity dicts.

    Each entity has: name, type, query, ticker (optional).
    Falls back to simple comma-separated topics when the watchlist is empty.
    """
    entities: list[dict[str, str]] = []

    if watchlist_json and watchlist_json.strip():
        try:
            parsed = json.loads(watchlist_json)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("name") and item.get("query"):
                        entities.append({
                            "name": str(item["name"]),
                            "type": str(item.get("type", "custom")),
                            "query": str(item["query"]),
                            "ticker": str(item.get("ticker", "")),
                        })
        except json.JSONDecodeError as exc:
            logger.warning("Invalid watchlist JSON: %s — falling back to topics", exc)

    # Simple-mode fallback
    if not entities and topics_csv and topics_csv.strip():
        region_prefix = regions.split(",")[0].strip() if regions else ""
        for topic in topics_csv.split(","):
            topic = topic.strip()
            if not topic:
                continue
            query = f"{region_prefix} {topic}".strip() if region_prefix else topic
            entities.append({"name": topic, "type": "custom", "query": query, "ticker": ""})

    return entities


# ---------------------------------------------------------------------------
# Per-entity analysis
# ---------------------------------------------------------------------------

async def _analyse_entity(
    client: httpx.AsyncClient,
    entity: dict[str, str],
    lookback_days: int,
    now_utc: datetime,
    cfg: Any,
) -> dict[str, Any]:
    """
    Fetch and analyse news for a single entity.

    Returns structured result including:
    - velocity, sentiment_score, diversity_score, composite_score
    - is_signal, signal_type (volume | sentiment | composite)
    - top_headlines with sentiment direction
    - entity metadata (type, ticker)
    """
    name = entity["name"]
    entity_type = entity["type"]
    ticker = entity["ticker"]
    query = entity["query"]

    lookback_start = (now_utc - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent_start = (now_utc - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")

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
        logger.warning("NewsAPI HTTP error for '%s': %s", name, exc)
        return _error_result(entity, str(exc))
    except Exception as exc:
        logger.warning("NewsAPI request failed for '%s': %s", name, exc)
        return _error_result(entity, str(exc))

    articles = data.get("articles", [])

    # Split: recent (last 24h) vs historical (days 2..lookback)
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

    # ── Volume Velocity ──────────────────────────────────────────────────────
    historical_days = max(lookback_days - 1, 1)
    baseline_daily = historical_count / historical_days

    if baseline_daily < _MIN_BASELINE_DAILY:
        # Emerging entity — any recent activity is meaningful
        velocity = float(recent_count) if recent_count > 0 else 0.0
    else:
        velocity = recent_count / baseline_daily

    # ── Sentiment Shift ──────────────────────────────────────────────────────
    sentiment_score = 0.0
    sentiment_direction = "neutral"
    if cfg.sentiment_enabled and recent_articles:
        sentiment_score, sentiment_direction = _score_sentiment(
            recent_articles, historical_articles
        )

    # ── Source Diversity ─────────────────────────────────────────────────────
    diversity_score = 0.0
    if cfg.source_diversity_enabled:
        diversity_score = _score_diversity(recent_articles, historical_articles, historical_days)

    # ── Composite Pressure Score ─────────────────────────────────────────────
    # Sentiment amplifies the score when tone is shifting (abs value)
    # Diversity amplifies when coverage is spreading to new outlets
    sentiment_factor = 1.0 + abs(sentiment_score) * 0.4
    diversity_factor = 1.0 + diversity_score * 0.3
    composite_score = velocity * sentiment_factor * diversity_factor

    # Signal logic: composite score OR volume-only (if extras disabled)
    is_signal = (
        recent_count >= cfg.signal_min_articles
        and composite_score >= cfg.composite_score_threshold
    )

    # Determine primary signal driver
    if not is_signal:
        signal_type = None
    elif velocity >= cfg.pressure_velocity_threshold:
        signal_type = "volume"
    elif abs(sentiment_score) >= 0.4:
        signal_type = "sentiment"
    else:
        signal_type = "composite"

    # Top headlines (clean, with sentiment labels)
    top_headlines = _extract_headlines(recent_articles)

    return {
        "name": name,
        "type": entity_type,
        "ticker": ticker,
        "query": query,
        # Scores
        "velocity": round(velocity, 2),
        "sentiment_score": round(sentiment_score, 3),
        "sentiment_direction": sentiment_direction,
        "diversity_score": round(diversity_score, 2),
        "composite_score": round(composite_score, 2),
        # Counts
        "recent_count": recent_count,
        "baseline_daily": round(baseline_daily, 2),
        "historical_count": historical_count,
        "total_fetched": len(articles),
        # Signal
        "is_signal": is_signal,
        "signal_type": signal_type,
        "top_headlines": top_headlines,
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_sentiment(
    recent: list[dict], historical: list[dict]
) -> tuple[float, str]:
    """
    Compute a sentiment shift score in [-1, +1].

    Positive = more positive articles than baseline.
    Negative = more negative articles than baseline.

    Uses simple keyword matching (no external NLP) as a directional proxy.
    """
    def _sentiment_ratio(articles: list[dict]) -> float:
        if not articles:
            return 0.0
        pos = neg = 0
        for a in articles:
            text = f"{a.get('title', '')} {a.get('description', '')}".lower()
            if any(kw in text for kw in _NEGATIVE_KEYWORDS):
                neg += 1
            if any(kw in text for kw in _POSITIVE_KEYWORDS):
                pos += 1
        return (pos - neg) / len(articles)

    recent_ratio = _sentiment_ratio(recent)
    hist_ratio = _sentiment_ratio(historical)

    # Shift = how much has sentiment moved from baseline
    shift = recent_ratio - hist_ratio

    if shift > 0.15:
        direction = "positive"
    elif shift < -0.15:
        direction = "negative"
    else:
        direction = "neutral"

    return round(shift, 3), direction


def _score_diversity(
    recent: list[dict], historical: list[dict], historical_days: int
) -> float:
    """
    Source diversity score: how many new/unique domains are covering this story today?

    Score > 1.0 means more unique sources than baseline; higher = spreading faster.
    """
    def _unique_domains(articles: list[dict]) -> int:
        domains: set[str] = set()
        for a in articles:
            url = a.get("url", "")
            try:
                domain = urlparse(url).netloc.lower().lstrip("www.")
                if domain:
                    domains.add(domain)
            except Exception:
                pass
        return len(domains)

    recent_domains = _unique_domains(recent)
    hist_domains = _unique_domains(historical)

    baseline_domains_per_day = hist_domains / max(historical_days, 1) if hist_domains else 0

    if baseline_domains_per_day < 0.5:
        return float(recent_domains) if recent_domains > 0 else 0.0
    return round(recent_domains / baseline_domains_per_day, 2)


def _extract_headlines(articles: list[dict]) -> list[dict]:
    """Extract top 3 clean headlines with sentiment labels."""
    result = []
    for a in articles[:3]:
        title = a.get("title", "")
        if not title or "[Removed]" in title:
            continue
        text = f"{title} {a.get('description', '')}".lower()
        has_neg = any(kw in text for kw in _NEGATIVE_KEYWORDS)
        has_pos = any(kw in text for kw in _POSITIVE_KEYWORDS)
        if has_neg and not has_pos:
            tone = "negative"
        elif has_pos and not has_neg:
            tone = "positive"
        else:
            tone = "neutral"
        result.append({"title": title, "tone": tone, "source": a.get("source", {}).get("name", "")})
    return result


def _error_result(entity: dict[str, str], error: str) -> dict[str, Any]:
    return {
        "name": entity["name"],
        "type": entity["type"],
        "ticker": entity.get("ticker", ""),
        "query": entity["query"],
        "velocity": 0.0,
        "sentiment_score": 0.0,
        "sentiment_direction": "neutral",
        "diversity_score": 0.0,
        "composite_score": 0.0,
        "recent_count": 0,
        "baseline_daily": 0.0,
        "historical_count": 0,
        "total_fetched": 0,
        "is_signal": False,
        "signal_type": None,
        "top_headlines": [],
        "error": error,
    }


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(signals: list[dict], all_entities: list[dict], now_utc: datetime) -> str:
    """Build a plain-language pre-movement intelligence briefing."""
    date_str = now_utc.strftime("%B %d, %Y")
    total = len(all_entities)

    if total == 0:
        return f"Intelligence briefing ({date_str}): No entities analysed."

    if not signals:
        names = ", ".join(e["name"] for e in all_entities if "error" not in e)
        return (
            f"Intelligence briefing ({date_str}): No pre-movement pressure signals detected. "
            f"Entities monitored: {names}."
        )

    parts = [
        f"Intelligence briefing ({date_str}): "
        f"{len(signals)} pre-movement signal(s) detected across {total} watched entities."
    ]

    _TYPE_EMOJI = {
        "company": "📈",
        "sports_team": "🏆",
        "market": "📊",
        "crypto": "₿",
        "custom": "◉",
    }

    for sig in signals:
        name = sig["name"]
        entity_type = sig.get("type", "custom")
        ticker = sig.get("ticker", "")
        velocity = sig.get("velocity", 0)
        composite = sig.get("composite_score", 0)
        sentiment_dir = sig.get("sentiment_direction", "neutral")
        signal_type = sig.get("signal_type", "composite")
        headlines = sig.get("top_headlines", [])
        recent = sig.get("recent_count", 0)
        pct = int((velocity - 1.0) * 100) if velocity >= 1.0 else 0

        icon = _TYPE_EMOJI.get(entity_type, "◉")
        ticker_str = f" ({ticker})" if ticker else ""
        sentiment_str = f", sentiment {sentiment_dir}" if sentiment_dir != "neutral" else ""
        driver = {"volume": "volume spike", "sentiment": "sentiment shift", "composite": "multi-signal"}.get(
            signal_type, "pressure"
        )

        line = (
            f"{icon} {name}{ticker_str}: {driver} — {recent} articles today "
            f"({pct}% above baseline{sentiment_str}, composite score {composite:.1f}×)."
        )
        if headlines:
            line += f" \"{headlines[0]['title']}\""
        parts.append(line)

    return " ".join(parts)
