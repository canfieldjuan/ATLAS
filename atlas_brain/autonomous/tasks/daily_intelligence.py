"""
Daily intelligence analysis: gather accumulated market, news, and business
data over a configurable window, feed to LLM with prior reasoning journal
entries, and persist structured conclusions.

Runs once daily (default 8 PM). Handles its own LLM call, journal
persistence, and ntfy notification -- returns _skip_synthesis so the
runner does not double-synthesize.
"""

import asyncio
import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.daily_intelligence")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: daily deep intelligence analysis."""
    cfg = settings.external_data
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "Daily intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    window_days = cfg.intelligence_analysis_window_days
    max_prior = cfg.intelligence_max_prior_sessions
    today = date.today()

    # Skip if we already have a journal entry for today
    existing = await pool.fetchrow(
        "SELECT id FROM reasoning_journal WHERE session_date = $1 LIMIT 1",
        today,
    )
    if existing:
        return {"_skip_synthesis": f"Intelligence journal already exists for {today}"}

    # Gather all 6 data sources in parallel
    (
        market_data, news_articles, business_ctx,
        graph_ctx, prior_reasoning, pressure_baselines,
    ) = await asyncio.gather(
        _fetch_market_data(pool, window_days),
        _fetch_news_articles(pool, window_days),
        _fetch_business_context(pool, window_days),
        _fetch_graph_context(),
        _fetch_prior_reasoning(pool, max_prior),
        _fetch_pressure_baselines(pool),
        return_exceptions=True,
    )

    # Convert exceptions to empty values
    if isinstance(market_data, Exception):
        logger.warning("Market data fetch failed: %s", market_data)
        market_data = []
    if isinstance(news_articles, Exception):
        logger.warning("News articles fetch failed: %s", news_articles)
        news_articles = []
    if isinstance(business_ctx, Exception):
        logger.warning("Business context fetch failed: %s", business_ctx)
        business_ctx = {}
    if isinstance(graph_ctx, Exception):
        logger.warning("Graph context fetch failed: %s", graph_ctx)
        graph_ctx = []
    if isinstance(prior_reasoning, Exception):
        logger.warning("Prior reasoning fetch failed: %s", prior_reasoning)
        prior_reasoning = []
    if isinstance(pressure_baselines, Exception):
        logger.warning("Pressure baselines fetch failed: %s", pressure_baselines)
        pressure_baselines = []

    # Run behavioral risk sensors on article text (secondary signal)
    if news_articles and isinstance(news_articles, list):
        news_articles = _run_risk_sensors(news_articles)

    # Check if there's enough data to analyze
    total_data_points = len(market_data) + len(news_articles)
    if total_data_points == 0 and not business_ctx and not prior_reasoning:
        return {"_skip_synthesis": "No data to analyze"}

    # Pre-compute temporal correlations (article + market move within 4h window)
    temporal_correlations = _compute_temporal_correlations(news_articles, market_data)

    # Build the user message payload
    payload = {
        "date": str(today),
        "analysis_window_days": window_days,
        "market_data": market_data,
        "news_articles": news_articles,
        "business_context": business_ctx,
        "graph_context": graph_ctx,
        "pressure_baselines": pressure_baselines,
        "prior_reasoning": prior_reasoning,
        "temporal_correlations": temporal_correlations,
    }

    # Load skill and call LLM
    from ...pipelines.llm import call_llm_with_skill, parse_json_response

    analysis = call_llm_with_skill(
        "digest/daily_intelligence", payload,
        max_tokens=cfg.intelligence_max_tokens, temperature=0.4,
    )
    if not analysis:
        return {"_skip_synthesis": "LLM analysis failed"}

    # Parse structured output from LLM
    parsed = parse_json_response(analysis, recover_truncated=True)

    # Persist to reasoning_journal
    pressure_readings = parsed.get("pressure_readings", [])
    try:
        await pool.execute(
            """
            INSERT INTO reasoning_journal (
                session_date, analysis_type, analysis_window_days,
                raw_data_summary, reasoning_output, key_insights,
                connections_found, recommendations, market_summary,
                news_summary, business_implications, pressure_readings
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            today,
            "daily",
            window_days,
            json.dumps({
                "market_count": len(market_data),
                "news_count": len(news_articles),
                "business_keys": list(business_ctx.keys()) if isinstance(business_ctx, dict) else [],
                "graph_facts": len(graph_ctx),
                "prior_sessions": len(prior_reasoning),
                "pressure_baselines": len(pressure_baselines),
            }),
            parsed.get("analysis_text", analysis),
            json.dumps(parsed.get("key_insights", [])),
            json.dumps(parsed.get("connections_found", [])),
            json.dumps(parsed.get("recommendations", [])),
            json.dumps(parsed.get("market_summary", {})),
            json.dumps(parsed.get("news_summary", {})),
            json.dumps(parsed.get("business_implications", [])),
            json.dumps(pressure_readings),
        )
        logger.info("Stored reasoning journal entry for %s", today)
    except Exception:
        logger.exception("Failed to store reasoning journal entry")

    # Upsert entity pressure baselines (with delta-clamping against prior)
    if cfg.pressure_enabled and pressure_readings:
        await _upsert_pressure_baselines(pool, pressure_readings, pressure_baselines)

    # Send ntfy notification
    from ...pipelines.notify import send_pipeline_notification

    await send_pipeline_notification(
        parsed.get("analysis_text", analysis), task,
        title="Atlas: Daily Intelligence",
        default_tags="brain,chart_with_upwards_trend",
        parsed=parsed,
    )

    return {
        "_skip_synthesis": "Daily intelligence complete",
        "date": str(today),
        "market_snapshots": len(market_data),
        "news_articles": len(news_articles),
        "prior_sessions": len(prior_reasoning),
        "insights": len(parsed.get("key_insights", [])),
        "connections": len(parsed.get("connections_found", [])),
    }


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_market_data(pool, window_days: int) -> list[dict[str, Any]]:
    """Fetch market snapshots grouped by symbol over the analysis window."""
    rows = await pool.fetch(
        """
        SELECT ms.symbol, ms.price, ms.change_pct, ms.volume, ms.snapshot_at,
               dw.name, dw.category
        FROM market_snapshots ms
        JOIN data_watchlist dw ON dw.symbol = ms.symbol AND dw.enabled = true
        WHERE ms.snapshot_at > NOW() - make_interval(days => $1)
        ORDER BY ms.symbol, ms.snapshot_at DESC
        """,
        window_days,
    )
    return [
        {
            "symbol": r["symbol"],
            "name": r["name"],
            "category": r["category"],
            "price": float(r["price"]) if r["price"] else None,
            "change_pct": float(r["change_pct"]) if r["change_pct"] else None,
            "volume": r["volume"],
            "snapshot_at": r["snapshot_at"].isoformat() if r["snapshot_at"] else None,
        }
        for r in rows
    ]


async def _fetch_news_articles(pool, window_days: int) -> list[dict[str, Any]]:
    """Fetch stored news articles from the analysis window (with enrichment data)."""
    rows = await pool.fetch(
        """
        SELECT title, source_name, url, published_at, summary,
               matched_keywords, is_market_related, created_at,
               content, soram_channels, linguistic_indicators,
               entities_detected, pressure_direction
        FROM news_articles
        WHERE created_at > NOW() - make_interval(days => $1)
        ORDER BY created_at DESC
        LIMIT 100
        """,
        window_days,
    )
    result = []
    for r in rows:
        article: dict[str, Any] = {
            "title": r["title"],
            "source": r["source_name"],
            "summary": r["summary"],
            "matched_keywords": r["matched_keywords"] or [],
            "is_market_related": r["is_market_related"],
            "published_at": r["published_at"],
        }
        # Enrichment fields (may be NULL if not yet enriched)
        if r["content"]:
            article["content"] = r["content"][:2000]  # truncate for LLM context
        soram = r["soram_channels"]
        if isinstance(soram, str):
            try:
                soram = json.loads(soram)
            except (json.JSONDecodeError, TypeError):
                soram = None
        if soram:
            article["soram_channels"] = soram
        ling = r["linguistic_indicators"]
        if isinstance(ling, str):
            try:
                ling = json.loads(ling)
            except (json.JSONDecodeError, TypeError):
                ling = None
        if ling:
            article["linguistic_indicators"] = ling
        entities = r["entities_detected"]
        if entities:
            article["entities_detected"] = list(entities)
        if r.get("pressure_direction"):
            article["pressure_direction"] = r["pressure_direction"]
        result.append(article)
    return result


def _run_risk_sensors(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run behavioral risk sensors on article text as secondary signal.

    Adds sensor_analysis dict to each article that has content,
    including per-sensor results and cross-correlation.
    """
    try:
        from ...tools.risk_sensors import (
            alignment_sensor_tool,
            operational_urgency_tool,
            negotiation_rigidity_tool,
            correlate,
        )
    except Exception:
        logger.debug("Risk sensors not available, skipping", exc_info=True)
        return articles

    for article in articles:
        text = article.get("content") or article.get("summary") or ""
        if not text or len(text) < 50:
            continue

        try:
            alignment = alignment_sensor_tool.analyze(text)
            urgency = operational_urgency_tool.analyze(text)
            rigidity = negotiation_rigidity_tool.analyze(text)
            cross = correlate(alignment, urgency, rigidity)

            # Cross-validate sensors against SORAM classification.
            # Sensors are bag-of-words -- they fire on quoted language
            # in neutral reporting just like direct adversarial text.
            # SORAM classification (LLM-driven) understands context.
            # Use SORAM to assign confidence to sensor readings.
            soram = article.get("soram_channels", {})
            confidence = _sensor_confidence(soram, cross)

            article["sensor_analysis"] = {
                "alignment_triggered": alignment["triggered"],
                "urgency_triggered": urgency["triggered"],
                "rigidity_triggered": rigidity["triggered"],
                "composite_risk_level": cross["composite_risk_level"],
                "sensor_count": cross["sensor_count"],
                "patterns": [r["label"] for r in cross["relationships"]],
                "confidence": confidence,
                "confidence_note": _sensor_confidence_note(soram, cross, confidence),
            }
        except Exception:
            logger.debug("Sensor analysis failed for article: %s", article.get("title", ""), exc_info=True)

    return articles


def _sensor_confidence(soram: dict, cross: dict) -> str:
    """Rate sensor confidence using SORAM classification as ground truth.

    Sensors are term-frequency based (context-blind). SORAM classification
    is LLM-based (context-aware). When sensors fire but SORAM says the
    article is mostly media-channel reporting (not operational or
    alignment-heavy), the sensor hits are likely from quoted language
    in neutral coverage -- downgrade confidence.

    Returns: "high", "medium", or "low"
    """
    if not soram or cross.get("sensor_count", 0) == 0:
        return "low"

    # Which SORAM channels support the sensor readings?
    operational = soram.get("operational", 0.0)
    alignment = soram.get("alignment", 0.0)
    media = soram.get("media", 0.0)
    societal = soram.get("societal", 0.0)

    # Sensors measure adversarial/urgency/rigidity language.
    # These are most meaningful when O or A channels are active
    # (actual operational/alignment content, not just media reporting).
    supporting_signal = max(operational, alignment, societal)
    reporting_signal = media

    # High confidence: SORAM confirms the content is operational/alignment
    # AND sensors fired
    if supporting_signal >= 0.5 and cross.get("sensor_count", 0) >= 2:
        return "high"

    # Medium: some SORAM support, or sensors strongly triggered
    if supporting_signal >= 0.3:
        return "medium"

    # Low: article is mostly media-channel (reporting about events)
    # but sensors fired on quoted/described language
    if reporting_signal >= 0.6 and supporting_signal < 0.3:
        return "low"

    return "medium"


def _sensor_confidence_note(soram: dict, cross: dict, confidence: str) -> str:
    """Generate a short explanation of why sensor confidence is what it is."""
    if confidence == "high":
        return "SORAM confirms operational/alignment content -- sensor readings reflect direct signals"
    if confidence == "low":
        if not soram:
            return "No SORAM classification available -- sensor readings unvalidated"
        media = soram.get("media", 0.0)
        if media >= 0.6:
            return "Article is primarily media reporting -- sensor triggers likely from quoted language, not direct signals"
        return "Weak SORAM support for sensor-detected patterns"
    return "Moderate SORAM support -- sensor readings are plausible but not confirmed"


def _compute_temporal_correlations(
    articles: list[dict[str, Any]],
    market_data: list[dict[str, Any]],
    window_hours: float = 4.0,
) -> list[dict[str, Any]]:
    """Find article/market pairs within a time window.

    When a news article and an abnormal price move co-occur within
    ``window_hours``, the LLM should evaluate causation direction
    (news-first = information asymmetry, price-first = insider/algo).
    """
    from datetime import timedelta
    from dateutil.parser import isoparse

    correlations: list[dict[str, Any]] = []

    # Parse article timestamps
    art_times: list[tuple[datetime, dict]] = []
    for a in articles:
        ts = a.get("published_at")
        if not ts:
            continue
        try:
            dt = isoparse(ts) if isinstance(ts, str) else ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            art_times.append((dt, a))
        except (ValueError, TypeError):
            continue

    # Parse market snapshots with abnormal moves (>2% change)
    mkt_moves: list[tuple[datetime, dict]] = []
    for m in market_data:
        change = m.get("change_pct")
        if change is None or abs(float(change)) < 2.0:
            continue
        ts = m.get("snapshot_at")
        if not ts:
            continue
        try:
            dt = isoparse(ts) if isinstance(ts, str) else ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            mkt_moves.append((dt, m))
        except (ValueError, TypeError):
            continue

    if not art_times or not mkt_moves:
        return correlations

    window = timedelta(hours=window_hours)

    for art_dt, art in art_times:
        for mkt_dt, mkt in mkt_moves:
            gap = abs((art_dt - mkt_dt).total_seconds())
            if gap <= window.total_seconds():
                art_first = art_dt <= mkt_dt
                correlations.append({
                    "article_title": art.get("title", ""),
                    "symbol": mkt.get("symbol", ""),
                    "change_pct": mkt.get("change_pct"),
                    "gap_hours": round(gap / 3600, 1),
                    "direction": "news_before_price" if art_first else "price_before_news",
                    "implication": (
                        "Information asymmetry -- news preceded price move"
                        if art_first
                        else "Possible insider/algorithmic activity -- price moved before news"
                    ),
                })

    # Deduplicate and limit
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for c in sorted(correlations, key=lambda x: x["gap_hours"]):
        key = (c["article_title"][:50], c["symbol"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
        if len(unique) >= 20:
            break

    return unique


async def _fetch_business_context(pool, window_days: int) -> dict[str, Any]:
    """Fetch recent business activity: appointments, invoices, emails, interactions."""
    ctx: dict[str, Any] = {}

    # Appointments
    try:
        rows = await pool.fetch(
            """
            SELECT id, contact_id, service_type, start_time, status
            FROM appointments
            WHERE start_time > NOW() - make_interval(days => $1)
            ORDER BY start_time DESC
            LIMIT 20
            """,
            window_days,
        )
        ctx["appointments"] = [
            {
                "service_type": r["service_type"],
                "scheduled_at": r["start_time"].isoformat() if r["start_time"] else None,
                "status": r["status"],
            }
            for r in rows
        ]
    except Exception:
        ctx["appointments"] = []

    # Invoices
    try:
        rows = await pool.fetch(
            """
            SELECT id, contact_id, total_amount, status, due_date, created_at
            FROM invoices
            WHERE created_at > NOW() - make_interval(days => $1)
            ORDER BY created_at DESC
            LIMIT 20
            """,
            window_days,
        )
        ctx["invoices"] = [
            {
                "total_amount": float(r["total_amount"]) if r["total_amount"] else None,
                "status": r["status"],
                "due_date": r["due_date"].isoformat() if r["due_date"] else None,
            }
            for r in rows
        ]
    except Exception:
        ctx["invoices"] = []

    # Processed emails
    try:
        rows = await pool.fetch(
            """
            SELECT sender, subject, category, intent, replyable, processed_at
            FROM processed_emails
            WHERE processed_at > NOW() - make_interval(days => $1)
            ORDER BY processed_at DESC
            LIMIT 30
            """,
            window_days,
        )
        ctx["emails"] = [
            {
                "sender": r["sender"],
                "subject": r["subject"],
                "category": r["category"],
                "intent": r["intent"],
            }
            for r in rows
        ]
    except Exception:
        ctx["emails"] = []

    # Contact interactions
    try:
        rows = await pool.fetch(
            """
            SELECT contact_id, interaction_type, summary, created_at
            FROM contact_interactions
            WHERE created_at > NOW() - make_interval(days => $1)
            ORDER BY created_at DESC
            LIMIT 20
            """,
            window_days,
        )
        ctx["interactions"] = [
            {
                "type": r["interaction_type"],
                "notes": (r["summary"] or "")[:200],
            }
            for r in rows
        ]
    except Exception:
        ctx["interactions"] = []

    return ctx


async def _fetch_graph_context() -> list[dict[str, Any]]:
    """Fetch relevant facts from knowledge graph."""
    try:
        from ...memory.rag_client import get_rag_client

        rag = get_rag_client()
        result = await rag.search(
            "business obligations, financial patterns, recurring contacts",
            max_facts=10,
        )
        if result and result.facts:
            return [
                {"fact": f.fact, "source": f.name}
                for f in result.facts
                if f.fact
            ]
    except Exception:
        logger.debug("Graph context fetch failed", exc_info=True)
    return []


async def _fetch_prior_reasoning(pool, max_sessions: int) -> list[dict[str, Any]]:
    """Fetch prior reasoning journal entries (most recent first)."""
    rows = await pool.fetch(
        """
        SELECT session_date, reasoning_output, key_insights,
               connections_found, recommendations, market_summary,
               news_summary, business_implications, pressure_readings
        FROM reasoning_journal
        ORDER BY session_date DESC
        LIMIT $1
        """,
        max_sessions,
    )
    result = []
    for r in rows:
        entry: dict[str, Any] = {
            "session_date": str(r["session_date"]),
            "reasoning_output": (r["reasoning_output"] or "")[:1000],
            "key_insights": r["key_insights"] if isinstance(r["key_insights"], list) else [],
            "connections_found": r["connections_found"] if isinstance(r["connections_found"], list) else [],
            "recommendations": r["recommendations"] if isinstance(r["recommendations"], list) else [],
            "market_summary": r["market_summary"] if isinstance(r["market_summary"], dict) else {},
            "news_summary": r["news_summary"] if isinstance(r["news_summary"], dict) else {},
            "business_implications": r["business_implications"] if isinstance(r["business_implications"], list) else [],
        }
        pr = r["pressure_readings"]
        if isinstance(pr, str):
            try:
                pr = json.loads(pr)
            except (json.JSONDecodeError, TypeError):
                pr = []
        if isinstance(pr, list) and pr:
            entry["pressure_readings"] = pr
        result.append(entry)
    return result


async def _fetch_pressure_baselines(pool) -> list[dict[str, Any]]:
    """Fetch current entity pressure baselines (highest pressure first)."""
    rows = await pool.fetch(
        """
        SELECT entity_name, entity_type, pressure_score, sentiment_drift,
               narrative_frequency, soram_breakdown, linguistic_signals,
               last_computed_at
        FROM entity_pressure_baselines
        ORDER BY pressure_score DESC
        LIMIT 50
        """,
    )
    result = []
    for r in rows:
        entry: dict[str, Any] = {
            "entity_name": r["entity_name"],
            "entity_type": r["entity_type"],
            "pressure_score": float(r["pressure_score"]) if r["pressure_score"] is not None else 0.0,
            "sentiment_drift": float(r["sentiment_drift"]) if r["sentiment_drift"] is not None else 0.0,
            "narrative_frequency": r["narrative_frequency"] or 0,
            "last_computed_at": r["last_computed_at"].isoformat() if r["last_computed_at"] else None,
        }
        soram = r["soram_breakdown"]
        if isinstance(soram, str):
            try:
                soram = json.loads(soram)
            except (json.JSONDecodeError, TypeError):
                soram = {}
        entry["soram_breakdown"] = soram if isinstance(soram, dict) else {}
        ling = r["linguistic_signals"]
        if isinstance(ling, str):
            try:
                ling = json.loads(ling)
            except (json.JSONDecodeError, TypeError):
                ling = {}
        entry["linguistic_signals"] = ling if isinstance(ling, dict) else {}
        result.append(entry)
    return result


# Suffixes stripped during entity name normalization (case-insensitive)
_CORP_SUFFIXES = re.compile(
    r',?\s+(?:Inc\.?|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|LLC|PLC|SA|AG|NV|SE|Group|Holdings?)\.?$',
    re.IGNORECASE,
)


def _normalize_entity_name(name: str) -> str:
    """Canonicalize entity name to prevent duplicate pressure baselines.

    Strips corporate suffixes (Inc, Corp, Co, Ltd, etc.), collapses whitespace,
    and title-cases. "Boeing Co" and "Boeing Company" both become "Boeing".
    """
    name = name.strip()
    if not name:
        return name
    # Strip corporate suffixes (may need multiple passes for "X Holdings Inc")
    for _ in range(2):
        cleaned = _CORP_SUFFIXES.sub('', name).strip()
        if cleaned == name:
            break
        name = cleaned
    # Collapse whitespace and title-case
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


async def _upsert_pressure_baselines(
    pool,
    pressure_readings: list[dict[str, Any]],
    prior_baselines: list[dict[str, Any]] | None = None,
) -> None:
    """Upsert entity pressure baselines with delta-clamping.

    The LLM sees yesterday's score and tends to anchor to it, inheriting any
    prior overestimate. Delta-clamping limits the per-day change to +/-2 points
    unless sensor composite risk (HIGH/CRITICAL) justifies a larger jump.
    """
    # Build lookup: (normalized_name, entity_type) -> prior_pressure_score
    prior_scores: dict[tuple[str, str], float] = {}
    for b in prior_baselines or []:
        key = (_normalize_entity_name(b.get("entity_name", "")), b.get("entity_type", "company"))
        if key[0]:
            prior_scores[key] = float(b.get("pressure_score", 0.0))

    MAX_DELTA = 2.0  # max change per day without sensor support
    SENSOR_SUPPORTED_DELTA = 5.0  # max change when HIGH/CRITICAL sensor composite

    now = datetime.now(timezone.utc)
    upserted = 0
    clamped = 0
    for reading in pressure_readings:
        raw_name = reading.get("entity_name")
        if not raw_name:
            continue
        entity_name = _normalize_entity_name(raw_name)
        if not entity_name:
            continue
        entity_type = reading.get("entity_type", "company")
        raw_score = float(reading.get("pressure_score", 0.0))

        # Delta-clamp against prior baseline
        key = (entity_name, entity_type)
        if key in prior_scores:
            prior = prior_scores[key]
            delta = raw_score - prior
            # Sensor composite from the reading's own sensor_analysis or note
            # HIGH/CRITICAL allows larger jumps (e.g., sudden crisis)
            sensor_level = str(reading.get("sensor_composite", "")).upper()
            max_d = SENSOR_SUPPORTED_DELTA if sensor_level in ("HIGH", "CRITICAL") else MAX_DELTA
            if abs(delta) > max_d:
                clamped_score = prior + max_d * (1.0 if delta > 0 else -1.0)
                clamped_score = max(0.0, min(10.0, clamped_score))
                logger.debug(
                    "Clamped %s pressure: LLM=%.1f prior=%.1f -> %.1f (max_delta=%.1f)",
                    entity_name, raw_score, prior, clamped_score, max_d,
                )
                raw_score = clamped_score
                clamped += 1

        # Clamp to valid range
        raw_score = max(0.0, min(10.0, raw_score))

        try:
            await pool.execute(
                """
                INSERT INTO entity_pressure_baselines (
                    entity_name, entity_type, pressure_score, sentiment_drift,
                    narrative_frequency, soram_breakdown, linguistic_signals,
                    last_computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (entity_name, entity_type) DO UPDATE SET
                    pressure_score = EXCLUDED.pressure_score,
                    sentiment_drift = EXCLUDED.sentiment_drift,
                    narrative_frequency = EXCLUDED.narrative_frequency,
                    soram_breakdown = EXCLUDED.soram_breakdown,
                    linguistic_signals = EXCLUDED.linguistic_signals,
                    last_computed_at = EXCLUDED.last_computed_at
                """,
                entity_name,
                entity_type,
                raw_score,
                reading.get("sentiment_drift", 0.0),
                reading.get("narrative_frequency", 0),
                json.dumps(reading.get("soram_breakdown", {})),
                json.dumps(reading.get("linguistic_signals", {})),
                now,
            )
            upserted += 1
        except Exception:
            logger.debug("Failed to upsert pressure baseline for %s", entity_name, exc_info=True)

    if upserted:
        logger.info("Upserted %d entity pressure baselines (%d clamped)", upserted, clamped)


