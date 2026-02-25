"""
News intelligence task — entity-level pressure signal detection.

Runs once per day (default 5 AM) and, for each entity in the configured
watchlist (companies, sports teams, markets, crypto, or custom topics):

1. Fetches recent news articles via NewsAPI.
2. Measures *volume velocity* — how many more articles today vs the baseline?
3. Measures *sentiment shift* — is the tone suddenly more negative/positive?
4. Measures *source diversity* — is the story spreading to new outlets?
5. Measures *linguistic pre-indicators* (behavioral stacking) — are the four
   Chase Hughes pre-event language patterns building?
   - Hedging: "reportedly", "could", "may" — uncertainty before disclosure
   - Deflection: "denies", "refuses to comment" — denial clusters before breaks
   - Insider: "sources say", "people familiar" — information leakage
   - Escalation: "breaking", "urgent", "crisis" — urgency before mainstream hit
6. Computes a composite *pressure score* from all active dimensions.
7. Tracks a *signal streak* — how many consecutive days has this entity been
   elevated? A multi-day streak is far more predictive than a single spike.
8. Detects *cross-entity correlation* — when multiple watched entities of the
   same type signal simultaneously it's a macro event, not individual noise.

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

# ---------------------------------------------------------------------------
# Keyword sets — sentiment (direction) and linguistic pre-indicators (behavioral)
# ---------------------------------------------------------------------------

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

# Chase Hughes pre-indicator linguistic patterns.
# Each set detects one of four behavioral signal dimensions that statistically
# appear in news coverage *before* a meaningful movement is publicly reported.

# Hedging: uncertainty language that builds as unverified information circulates
_HEDGE_KEYWORDS = frozenset([
    "reportedly", "allegedly", "said to be", "could", "may have", "might",
    "possibly", "potential", "expected to", "likely to", "appears to",
    "seems to", "uncertain", "unclear", "unconfirmed", "rumored", "rumour",
    "speculation", "whispers", "whisper", "chatter", "buzz",
])

# Deflection: denial/dismissal clusters that appear right before a story breaks
_DEFLECT_KEYWORDS = frozenset([
    "denies", "deny", "dismisses", "rejects", "refutes", "disputes",
    "refuses to comment", "declines to comment", "no comment",
    "pushes back", "calls unfounded", "calls speculation", "has not responded",
    "did not respond", "declined to respond", "won't say", "not commenting",
    "spokesperson declined",
])

# Insider: sourcing language indicating information leakage before official disclosure
_INSIDER_KEYWORDS = frozenset([
    "sources say", "sources close", "people familiar", "according to sources",
    "person familiar", "insiders say", "insider says", "anonymous sources",
    "speaking on condition", "not authorized to speak", "exclusive report",
    "obtained by", "reviewed by", "seen by", "leaked", "document shows",
    "filing shows", "internal memo",
])

# Escalation: urgency/crisis language in trade press before mainstream pickup
_ESCALATION_KEYWORDS = frozenset([
    "breaking", "urgent", "crisis", "critical", "emergency", "imminent",
    "rapid", "accelerating", "mounting", "intensifying", "escalating",
    "deteriorating", "worsening", "spiraling", "alarming", "dire",
    "dramatic", "drastic", "unprecedented", "pivotal", "decisive",
])

# ---------------------------------------------------------------------------
# Extended linguistic patterns — the three SORAM behavioral dimensions
# ---------------------------------------------------------------------------

# Permission Shifts: moral permission language — grants "right" to act against prior values.
# Appears before coordinated pressure campaigns; primes the audience for action.
_LINGUISTIC_PERMISSION = frozenset([
    "must be stopped", "cannot be tolerated", "for the greater good",
    "no option but", "we have no choice", "deserve what", "duty to",
    "obligation to", "justified in", "entitled to", "for public safety",
    "time to act", "enough is enough", "we must act", "cannot allow",
    "protecting our", "defending our", "threat to our",
])

# Certainty / Moral Panic: absolute language + emotional triggers.
# Spikes before a coordinated narrative is launched — creates urgency via certainty.
_LINGUISTIC_CERTAINTY = frozenset([
    "undeniable", "settled", "irrefutable", "beyond doubt", "unquestionable",
    "without question", "everyone knows", "no one disputes", "clearly proven",
    "definitive proof", "absolute certainty", "fact is", "the truth is",
    "plain and simple", "period", "full stop", "there is no debate",
    "dangerous misinformation", "dangerous conspiracy",
])

# Linguistic Dissociation: we/us → they/them shifts and label-based dehumanization.
# The transition from "people" to categorical labels precedes mobilization events.
_LINGUISTIC_DISSOCIATION = frozenset([
    "these people", "those people", "their kind", "the likes of",
    "that group", "such individuals", "the others", "outsiders",
    "not one of us", "foreign element", "radical element", "extremist element",
    "they believe", "they want", "they are trying", "their agenda",
    "anti-", "pro-", "followers of",
])

# ---------------------------------------------------------------------------
# SORAM Framework keyword sets (Chase Hughes)
# Five societal pressure channels pulled simultaneously before major events
# ---------------------------------------------------------------------------

# Societal: coordinated threat/fear framing across disparate outlets
_SORAM_SOCIETAL = frozenset([
    "threat to", "public threat", "national security", "public safety",
    "misinformation", "disinformation", "conspiracy theory", "dangerous narrative",
    "harmful content", "existential threat", "unprecedented danger",
    "clear and present", "looming threat", "growing threat", "hidden threat",
])

# Operational: drills, exercises, readiness — often precede actual events
_SORAM_OPERATIONAL = frozenset([
    "drill", "exercise", "simulation", "readiness", "preparedness",
    "tabletop", "war game", "wargame", "practice run", "rehearsal",
    "test scenario", "training exercise", "emergency response exercise",
    "live drill", "full-scale exercise", "response exercise",
])

# Regulatory: new laws / emergency powers quietly introduced before a crisis
_SORAM_REGULATORY = frozenset([
    "emergency powers", "executive order", "emergency declaration",
    "emergency regulation", "rule change", "bill introduced",
    "legislation passed", "law enacted", "policy enacted",
    "emergency measure", "temporary measure", "interim rule",
    "regulatory guidance", "new framework", "oversight expansion",
])

# Alignment: scripted consensus — identical phrasing across government, media, tech
_SORAM_ALIGNMENT = frozenset([
    "experts say", "officials confirm", "authorities warn", "scientists agree",
    "studies confirm", "data confirms", "research confirms", "experts agree",
    "consensus", "unanimous", "widely accepted", "fact-checkers",
    "trusted sources confirm", "verified by", "independent experts",
])

# Media Novelty: hijack tactics — constant "breaking" keeps brain in high suggestibility
_SORAM_MEDIA_NOVELTY = frozenset([
    "breaking", "just in", "developing story", "happening now",
    "major breaking", "live updates", "breaking news", "breaking:",
    "shocking new", "stunning revelation", "bombshell", "exclusive:",
    "we're following", "this just in", "urgent update",
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
        signal_history (dict): Persisted streak counters from prior runs
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

    # Restore per-entity streak counters persisted from previous runs
    signal_history: dict[str, int] = metadata.get("signal_history", {})

    logger.info(
        "News intelligence: analysing %d entity/entities over %d-day window "
        "(sentiment=%s, diversity=%s, linguistic=%s, soram=%s, sec_edgar=%s, usaspending=%s)",
        len(entities), lookback_days,
        cfg.sentiment_enabled, cfg.source_diversity_enabled,
        cfg.linguistic_analysis_enabled, cfg.soram_enabled,
        cfg.sec_edgar_enabled, cfg.usaspending_enabled,
    )

    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []
    entity_results: list[dict] = []

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        for entity in entities:
            result = await _analyse_entity(
                client, entity, lookback_days, now_utc, cfg, signal_history
            )
            entity_results.append(result)
            if result.get("is_signal"):
                signals.append(result)
                logger.info(
                    "Pre-movement pressure: '%s' (%s) score=%.2f "
                    "velocity=%.2f sentiment=%.2f diversity=%.2f linguistic=%.2f soram=%.2f streak=%d",
                    entity["name"], entity["type"],
                    result.get("composite_score", 0),
                    result.get("velocity", 0),
                    result.get("sentiment_score", 0),
                    result.get("diversity_score", 0),
                    result.get("linguistic_score", 0),
                    result.get("soram_score", 0),
                    result.get("signal_streak", 0),
                )

    # Update streak counters for persistence
    updated_history = _update_signal_history(signal_history, entity_results, cfg)

    # Detect cross-entity macro correlations
    macro_signals = []
    if cfg.cross_entity_correlation_enabled:
        macro_signals = _detect_macro_correlations(signals, cfg)

    summary = _build_summary(signals, macro_signals, entity_results, now_utc)

    return {
        "status": "ok",
        "date": now_utc.strftime("%Y-%m-%d"),
        "entities_analysed": len(entities),
        "signals_detected": len(signals),
        "signals": signals,
        "macro_signals": macro_signals,
        "all_entities": entity_results,
        "signal_history": updated_history,
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
    signal_history: dict[str, int],
) -> dict[str, Any]:
    """
    Fetch and analyse news for a single entity.

    Returns structured result including:
    - velocity, sentiment_score, diversity_score, linguistic_score, composite_score
    - is_signal, signal_type (volume | sentiment | linguistic | composite)
    - signal_streak (consecutive days with elevated signal)
    - top_headlines with sentiment and linguistic tone labels
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

    # ── Linguistic Pre-Indicator Analysis ────────────────────────────────────
    linguistic_score = 0.0
    linguistic_markers: list[str] = []
    if cfg.linguistic_analysis_enabled and recent_articles:
        linguistic_score, linguistic_markers = _score_linguistic(recent_articles, cfg)

    # ── SORAM Framework ───────────────────────────────────────────────────────
    soram_score = 0.0
    soram_channels: list[str] = []
    if cfg.soram_enabled and recent_articles:
        soram_score, soram_channels = _score_soram(recent_articles, cfg)

    # ── Alternative Data Sources ──────────────────────────────────────────────
    sec_signals: dict[str, Any] = {}
    if cfg.sec_edgar_enabled and entity_type in ("company", "crypto") and ticker:
        sec_signals = await _fetch_sec_edgar_signals(client, ticker, now_utc)

    usaspending_signals: dict[str, Any] = {}
    if cfg.usaspending_enabled and name:
        usaspending_signals = await _fetch_usaspending_signals(client, name, now_utc)

    # ── Composite Pressure Score ─────────────────────────────────────────────
    # Each enabled dimension amplifies the base velocity score.
    # SORAM carries the highest amplification weight — it represents macro
    # societal coordination which is the strongest pre-event signal.
    sentiment_factor = 1.0 + abs(sentiment_score) * 0.4
    diversity_factor = 1.0 + diversity_score * 0.3
    linguistic_factor = 1.0 + linguistic_score * 0.5
    soram_factor = 1.0 + soram_score * 0.6
    # Data source bonus: confirmed external signals boost composite by a flat multiplier
    data_source_bonus = 1.0
    if sec_signals.get("elevated"):
        data_source_bonus += 0.2
    if usaspending_signals.get("elevated"):
        data_source_bonus += 0.15
    composite_score = velocity * sentiment_factor * diversity_factor * linguistic_factor * soram_factor * data_source_bonus

    # Signal logic: composite score must exceed threshold with minimum article count
    is_signal = (
        recent_count >= cfg.signal_min_articles
        and composite_score >= cfg.composite_score_threshold
    )

    # Determine primary signal driver
    if not is_signal:
        signal_type = None
    elif soram_score >= 0.35:
        signal_type = "soram"
    elif linguistic_score >= 0.4:
        signal_type = "linguistic"
    elif velocity >= cfg.pressure_velocity_threshold:
        signal_type = "volume"
    elif abs(sentiment_score) >= 0.4:
        signal_type = "sentiment"
    else:
        signal_type = "composite"

    # ── Signal Streak ────────────────────────────────────────────────────────
    prior_streak = signal_history.get(name, 0)
    signal_streak = (prior_streak + 1) if is_signal else 0
    streak_alert = (
        cfg.signal_streak_enabled
        and signal_streak >= cfg.signal_streak_threshold
    )

    # Top headlines (clean, with sentiment + linguistic labels)
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
        "linguistic_score": round(linguistic_score, 3),
        "linguistic_markers": linguistic_markers,
        "soram_score": round(soram_score, 3),
        "soram_channels": soram_channels,
        "composite_score": round(composite_score, 2),
        # External data signals
        "sec_signals": sec_signals,
        "usaspending_signals": usaspending_signals,
        # Counts
        "recent_count": recent_count,
        "baseline_daily": round(baseline_daily, 2),
        "historical_count": historical_count,
        "total_fetched": len(articles),
        # Signal
        "is_signal": is_signal,
        "signal_type": signal_type,
        "signal_streak": signal_streak,
        "streak_alert": streak_alert,
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
                domain = urlparse(url).netloc.lower().removeprefix("www.")
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


def _score_linguistic(
    recent: list[dict], cfg: Any
) -> tuple[float, list[str]]:
    """
    Score linguistic pre-indicator patterns (behavioral stacking dimension).

    Detects seven language pattern types in recent article headlines/descriptions
    that statistically build before a meaningful movement:
      Original four:
        - Hedging: uncertainty language ("reportedly", "may", "could")
        - Deflection: denial clusters ("denies", "refuses to comment")
        - Insider: source language ("sources say", "people familiar")
        - Escalation: urgency language ("breaking", "crisis", "urgent")
      SORAM behavioral layer (three new):
        - Permission Shifts: moral permission language ("must be stopped", "for the greater good")
        - Certainty/Panic: absolute language ("undeniable", "settled", "no debate")
        - Dissociation: we/us → they/them shifts ("these people", "their kind")

    Returns (score 0-1, list of active marker labels).
    """
    if not recent:
        return 0.0, []

    hedge_hits = deflect_hits = insider_hits = escalation_hits = 0
    permission_hits = certainty_hits = dissociation_hits = 0

    for a in recent:
        text = f"{a.get('title', '')} {a.get('description', '')}".lower()
        if cfg.linguistic_hedge_enabled and any(kw in text for kw in _HEDGE_KEYWORDS):
            hedge_hits += 1
        if cfg.linguistic_deflection_enabled and any(kw in text for kw in _DEFLECT_KEYWORDS):
            deflect_hits += 1
        if cfg.linguistic_insider_enabled and any(kw in text for kw in _INSIDER_KEYWORDS):
            insider_hits += 1
        if cfg.linguistic_escalation_enabled and any(kw in text for kw in _ESCALATION_KEYWORDS):
            escalation_hits += 1
        if cfg.linguistic_permission_enabled and any(kw in text for kw in _LINGUISTIC_PERMISSION):
            permission_hits += 1
        if cfg.linguistic_certainty_enabled and any(kw in text for kw in _LINGUISTIC_CERTAINTY):
            certainty_hits += 1
        if cfg.linguistic_dissociation_enabled and any(kw in text for kw in _LINGUISTIC_DISSOCIATION):
            dissociation_hits += 1

    n = len(recent)
    hedge_ratio       = min(hedge_hits / n, 1.0)
    deflect_ratio     = min(deflect_hits / n, 1.0)
    insider_ratio     = min(insider_hits / n, 1.0)
    escalation_ratio  = min(escalation_hits / n, 1.0)
    permission_ratio  = min(permission_hits / n, 1.0)
    certainty_ratio   = min(certainty_hits / n, 1.0)
    dissociation_ratio = min(dissociation_hits / n, 1.0)

    # Composite linguistic score: average of active dimensions
    active_dimensions = [
        r for r, enabled in [
            (hedge_ratio,        cfg.linguistic_hedge_enabled),
            (deflect_ratio,      cfg.linguistic_deflection_enabled),
            (insider_ratio,      cfg.linguistic_insider_enabled),
            (escalation_ratio,   cfg.linguistic_escalation_enabled),
            (permission_ratio,   cfg.linguistic_permission_enabled),
            (certainty_ratio,    cfg.linguistic_certainty_enabled),
            (dissociation_ratio, cfg.linguistic_dissociation_enabled),
        ] if enabled
    ]
    score = sum(active_dimensions) / max(len(active_dimensions), 1)

    # Build a list of triggered marker labels for the briefing
    markers: list[str] = []
    if cfg.linguistic_hedge_enabled and hedge_ratio >= 0.2:
        markers.append("hedging")
    if cfg.linguistic_deflection_enabled and deflect_ratio >= 0.15:
        markers.append("deflection")
    if cfg.linguistic_insider_enabled and insider_ratio >= 0.15:
        markers.append("insider-sourcing")
    if cfg.linguistic_escalation_enabled and escalation_ratio >= 0.2:
        markers.append("escalation")
    if cfg.linguistic_permission_enabled and permission_ratio >= 0.15:
        markers.append("permission-shift")
    if cfg.linguistic_certainty_enabled and certainty_ratio >= 0.15:
        markers.append("certainty-panic")
    if cfg.linguistic_dissociation_enabled and dissociation_ratio >= 0.15:
        markers.append("dissociation")

    return round(score, 3), markers


def _score_soram(
    recent: list[dict], cfg: Any
) -> tuple[float, list[str]]:
    """
    Score the SORAM framework channels against recent article text.

    SORAM = Societal / Operational / Regulatory / Alignment / Media Novelty.
    Each channel detects a specific type of societal pressure lever that,
    when pulled simultaneously, precedes major coordinated events.

    Returns (score 0-1, list of active channel labels).
    """
    if not recent:
        return 0.0, []

    societal_hits = operational_hits = regulatory_hits = 0
    alignment_hits = novelty_hits = 0

    for a in recent:
        text = f"{a.get('title', '')} {a.get('description', '')}".lower()
        if cfg.soram_societal_enabled and any(kw in text for kw in _SORAM_SOCIETAL):
            societal_hits += 1
        if cfg.soram_operational_enabled and any(kw in text for kw in _SORAM_OPERATIONAL):
            operational_hits += 1
        if cfg.soram_regulatory_enabled and any(kw in text for kw in _SORAM_REGULATORY):
            regulatory_hits += 1
        if cfg.soram_alignment_enabled and any(kw in text for kw in _SORAM_ALIGNMENT):
            alignment_hits += 1
        if cfg.soram_media_novelty_enabled and any(kw in text for kw in _SORAM_MEDIA_NOVELTY):
            novelty_hits += 1

    n = len(recent)
    societal_ratio   = min(societal_hits / n, 1.0)
    operational_ratio = min(operational_hits / n, 1.0)
    regulatory_ratio  = min(regulatory_hits / n, 1.0)
    alignment_ratio   = min(alignment_hits / n, 1.0)
    novelty_ratio     = min(novelty_hits / n, 1.0)

    active_dimensions = [
        r for r, enabled in [
            (societal_ratio,    cfg.soram_societal_enabled),
            (operational_ratio, cfg.soram_operational_enabled),
            (regulatory_ratio,  cfg.soram_regulatory_enabled),
            (alignment_ratio,   cfg.soram_alignment_enabled),
            (novelty_ratio,     cfg.soram_media_novelty_enabled),
        ] if enabled
    ]
    score = sum(active_dimensions) / max(len(active_dimensions), 1)

    # Label active channels (threshold set deliberately low — any presence is notable)
    channels: list[str] = []
    if cfg.soram_societal_enabled and societal_ratio >= 0.1:
        channels.append("S:societal")
    if cfg.soram_operational_enabled and operational_ratio >= 0.1:
        channels.append("O:operational")
    if cfg.soram_regulatory_enabled and regulatory_ratio >= 0.1:
        channels.append("R:regulatory")
    if cfg.soram_alignment_enabled and alignment_ratio >= 0.15:
        channels.append("A:alignment")
    if cfg.soram_media_novelty_enabled and novelty_ratio >= 0.2:
        channels.append("M:novelty")

    return round(score, 3), channels


_SEC_EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
_USASPENDING_BASE = "https://api.usaspending.gov/api/v2/search/spending_by_award/"


async def _fetch_sec_edgar_signals(
    client: httpx.AsyncClient,
    ticker: str,
    now_utc: datetime,
) -> dict[str, Any]:
    """
    Fetch recent SEC 8-K filings for a company ticker via the free EDGAR full-text search API.

    8-K forms are material event disclosures — an elevated count in the last 24h
    compared to the trailing 7-day average signals undisclosed activity.
    No API key required.
    """
    yesterday = (now_utc - timedelta(hours=24)).strftime("%Y-%m-%d")
    week_ago  = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
    today     = now_utc.strftime("%Y-%m-%d")

    try:
        # Recent 24h count
        resp_recent = await client.get(
            _SEC_EDGAR_BASE,
            params={"q": f'"{ticker}"', "dateRange": "custom",
                    "startdt": yesterday, "enddt": today, "forms": "8-K"},
            headers={"User-Agent": "Atlas Intelligence atlas@example.com"},
        )
        resp_recent.raise_for_status()
        recent_count = resp_recent.json().get("hits", {}).get("total", {}).get("value", 0)

        # 7-day baseline
        resp_hist = await client.get(
            _SEC_EDGAR_BASE,
            params={"q": f'"{ticker}"', "dateRange": "custom",
                    "startdt": week_ago, "enddt": yesterday, "forms": "8-K"},
            headers={"User-Agent": "Atlas Intelligence atlas@example.com"},
        )
        resp_hist.raise_for_status()
        hist_count = resp_hist.json().get("hits", {}).get("total", {}).get("value", 0)

        baseline_daily = hist_count / 6  # 6 full days in the trailing window
        elevated = bool(recent_count > 0 and (baseline_daily < 0.3 or recent_count > baseline_daily * 1.5))

        return {
            "source": "sec_edgar",
            "ticker": ticker,
            "recent_8k_count": recent_count,
            "baseline_8k_daily": round(baseline_daily, 2),
            "elevated": elevated,
        }
    except Exception as exc:
        logger.debug("SEC EDGAR fetch failed for '%s': %s", ticker, exc)
        return {"source": "sec_edgar", "ticker": ticker, "elevated": False, "error": str(exc)}


async def _fetch_usaspending_signals(
    client: httpx.AsyncClient,
    entity_name: str,
    now_utc: datetime,
) -> dict[str, Any]:
    """
    Fetch recent government contract awards mentioning the entity via USAspending.gov free API.

    A sudden increase in contract awards indicates business momentum or regulatory attention.
    No API key required.
    """
    yesterday = (now_utc - timedelta(hours=24)).strftime("%Y-%m-%d")
    week_ago  = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
    today     = now_utc.strftime("%Y-%m-%d")

    payload_base = {
        "filters": {
            "keywords": [entity_name],
            "award_type_codes": ["A", "B", "C", "D"],  # contracts
        },
        "fields": ["Award ID", "Recipient Name", "Award Amount", "Action Date"],
        "limit": 5,
        "page": 1,
    }

    try:
        # Recent 24h
        recent_payload = {**payload_base, "filters": {
            **payload_base["filters"],
            "time_period": [{"start_date": yesterday, "end_date": today}],
        }}
        resp_recent = await client.post(_USASPENDING_BASE, json=recent_payload)
        resp_recent.raise_for_status()
        recent_count = resp_recent.json().get("page_metadata", {}).get("count", 0)

        # 7-day baseline
        hist_payload = {**payload_base, "filters": {
            **payload_base["filters"],
            "time_period": [{"start_date": week_ago, "end_date": yesterday}],
        }}
        resp_hist = await client.post(_USASPENDING_BASE, json=hist_payload)
        resp_hist.raise_for_status()
        hist_count = resp_hist.json().get("page_metadata", {}).get("count", 0)

        baseline_daily = hist_count / 6
        elevated = bool(recent_count > 0 and (baseline_daily < 0.2 or recent_count > baseline_daily * 1.5))

        return {
            "source": "usaspending",
            "entity": entity_name,
            "recent_award_count": recent_count,
            "baseline_award_daily": round(baseline_daily, 2),
            "elevated": elevated,
        }
    except Exception as exc:
        logger.debug("USAspending fetch failed for '%s': %s", entity_name, exc)
        return {"source": "usaspending", "entity": entity_name, "elevated": False, "error": str(exc)}


def _extract_headlines(articles: list[dict]) -> list[dict]:
    """Extract top 3 clean headlines with sentiment and linguistic tone labels."""
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

        # Flag if any linguistic or SORAM pre-indicator pattern is present
        has_linguistic = (
            any(kw in text for kw in _HEDGE_KEYWORDS)
            or any(kw in text for kw in _DEFLECT_KEYWORDS)
            or any(kw in text for kw in _INSIDER_KEYWORDS)
            or any(kw in text for kw in _ESCALATION_KEYWORDS)
            or any(kw in text for kw in _LINGUISTIC_PERMISSION)
            or any(kw in text for kw in _LINGUISTIC_CERTAINTY)
            or any(kw in text for kw in _LINGUISTIC_DISSOCIATION)
            or any(kw in text for kw in _SORAM_SOCIETAL)
            or any(kw in text for kw in _SORAM_OPERATIONAL)
            or any(kw in text for kw in _SORAM_REGULATORY)
            or any(kw in text for kw in _SORAM_ALIGNMENT)
        )

        result.append({
            "title": title,
            "tone": tone,
            "linguistic": has_linguistic,
            "source": a.get("source", {}).get("name", ""),
        })
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
        "linguistic_score": 0.0,
        "linguistic_markers": [],
        "soram_score": 0.0,
        "soram_channels": [],
        "composite_score": 0.0,
        "sec_signals": {},
        "usaspending_signals": {},
        "recent_count": 0,
        "baseline_daily": 0.0,
        "historical_count": 0,
        "total_fetched": 0,
        "is_signal": False,
        "signal_type": None,
        "signal_streak": 0,
        "streak_alert": False,
        "top_headlines": [],
        "error": error,
    }


# ---------------------------------------------------------------------------
# Signal history (streak tracking)
# ---------------------------------------------------------------------------

def _update_signal_history(
    prior_history: dict[str, int],
    entity_results: list[dict],
    cfg: Any,
) -> dict[str, int]:
    """
    Update the per-entity streak counter dict.

    Each entity's streak increments when is_signal=True and resets to 0
    when the signal clears. Persisted via task.metadata between runs.
    """
    if not cfg.signal_streak_enabled:
        return {}
    updated = dict(prior_history)
    for result in entity_results:
        if "error" in result:
            continue
        name = result["name"]
        updated[name] = result.get("signal_streak", 0)
    return updated


# ---------------------------------------------------------------------------
# Cross-entity correlation detection
# ---------------------------------------------------------------------------

def _detect_macro_correlations(signals: list[dict], cfg: Any) -> list[dict]:
    """
    Detect macro signals by grouping simultaneous per-entity signals by type.

    When >= cross_entity_min_signals entities of the same type are all
    signalling on the same run, it indicates a sector-wide macro event rather
    than individual noise.
    """
    if not cfg.cross_entity_correlation_enabled or len(signals) < cfg.cross_entity_min_signals:
        return []

    # Group signals by entity type
    by_type: dict[str, list[dict]] = {}
    for sig in signals:
        etype = sig.get("type", "custom")
        by_type.setdefault(etype, []).append(sig)

    macro: list[dict] = []
    for etype, group in by_type.items():
        if len(group) >= cfg.cross_entity_min_signals:
            avg_score = sum(s.get("composite_score", 0) for s in group) / len(group)
            macro.append({
                "type": etype,
                "entity_count": len(group),
                "entities": [s["name"] for s in group],
                "avg_composite_score": round(avg_score, 2),
                "interpretation": (
                    f"Macro {etype} signal — {len(group)} entities signalling simultaneously "
                    f"(avg score {avg_score:.1f}×). Likely sector-wide or macro catalyst."
                ),
            })
    return macro


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    signals: list[dict],
    macro_signals: list[dict],
    all_entities: list[dict],
    now_utc: datetime,
) -> str:
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
        linguistic_markers = sig.get("linguistic_markers", [])
        soram_channels = sig.get("soram_channels", [])
        sec_elevated = sig.get("sec_signals", {}).get("elevated", False)
        usaspending_elevated = sig.get("usaspending_signals", {}).get("elevated", False)
        streak = sig.get("signal_streak", 0)
        streak_alert = sig.get("streak_alert", False)
        headlines = sig.get("top_headlines", [])
        recent = sig.get("recent_count", 0)
        pct = int((velocity - 1.0) * 100) if velocity >= 1.0 else 0

        icon = _TYPE_EMOJI.get(entity_type, "◉")
        ticker_str = f" ({ticker})" if ticker else ""
        sentiment_str = f", sentiment {sentiment_dir}" if sentiment_dir != "neutral" else ""
        linguistic_str = f", [{', '.join(linguistic_markers)}]" if linguistic_markers else ""
        soram_str = f", SORAM [{', '.join(soram_channels)}]" if soram_channels else ""
        data_str = ""
        if sec_elevated and usaspending_elevated:
            data_str = ", +SEC 8-K +gov contracts"
        elif sec_elevated:
            data_str = ", +SEC 8-K elevated"
        elif usaspending_elevated:
            data_str = ", +gov contracts elevated"
        streak_str = f" ⚠ {streak}-day streak" if streak_alert else (f" ({streak}d streak)" if streak > 1 else "")
        driver = {
            "volume": "volume spike",
            "sentiment": "sentiment shift",
            "linguistic": "linguistic pre-indicators",
            "soram": "SORAM pressure",
            "composite": "multi-signal",
        }.get(signal_type, "pressure")

        line = (
            f"{icon} {name}{ticker_str}: {driver} — {recent} articles today "
            f"({pct}% above baseline{sentiment_str}{linguistic_str}{soram_str}{data_str}, "
            f"composite score {composite:.1f}×){streak_str}."
        )
        if headlines:
            line += f" \"{headlines[0]['title']}\""
        parts.append(line)

    # Append macro correlation alerts
    for macro in macro_signals:
        parts.append(
            f"🌐 MACRO: {macro['interpretation']}"
        )

    return " ".join(parts)
