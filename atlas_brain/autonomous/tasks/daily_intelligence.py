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

    # Check if there's enough data to analyze
    total_data_points = len(market_data) + len(news_articles)
    if total_data_points == 0 and not business_ctx and not prior_reasoning:
        return {"_skip_synthesis": "No data to analyze"}

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
    }

    # Load skill and call LLM
    analysis = await _run_llm_analysis(payload, cfg.intelligence_max_tokens)
    if not analysis:
        return {"_skip_synthesis": "LLM analysis failed"}

    # Parse structured output from LLM
    parsed = _parse_analysis(analysis)

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

    # Upsert entity pressure baselines
    if cfg.pressure_enabled and pressure_readings:
        await _upsert_pressure_baselines(pool, pressure_readings)

    # Send ntfy notification
    analysis_text = parsed.get("analysis_text", analysis)
    await _send_notification(analysis_text, task)

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
               content, soram_channels, linguistic_indicators, entities_detected
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
        result.append(article)
    return result


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


async def _upsert_pressure_baselines(
    pool, pressure_readings: list[dict[str, Any]]
) -> None:
    """Upsert entity pressure baselines from LLM pressure readings."""
    now = datetime.now(timezone.utc)
    upserted = 0
    for reading in pressure_readings:
        entity_name = reading.get("entity_name")
        if not entity_name:
            continue
        entity_type = reading.get("entity_type", "company")
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
                reading.get("pressure_score", 0.0),
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
        logger.info("Upserted %d entity pressure baselines", upserted)


# ------------------------------------------------------------------
# LLM analysis
# ------------------------------------------------------------------


async def _run_llm_analysis(payload: dict[str, Any], max_tokens: int) -> str | None:
    """Load skill, call LLM, return raw text response.

    Prefers Anthropic (triage LLM) for reliable JSON output.
    Falls back to local Ollama if Anthropic is unavailable.
    """
    from ...skills import get_skill_registry
    from ...services import llm_registry
    from ...services.llm_router import get_triage_llm
    from ...services.protocols import Message

    skill = get_skill_registry().get("digest/daily_intelligence")
    if not skill:
        logger.warning("Skill 'digest/daily_intelligence' not found")
        return None

    # Prefer Anthropic for structured JSON output
    llm = get_triage_llm()
    if llm:
        logger.debug("Using triage LLM (Anthropic) for daily intelligence")
    else:
        llm = llm_registry.get_active()
    if llm is None:
        try:
            from ...config import settings as _settings
            llm_registry.activate(
                "ollama",
                model=_settings.llm.ollama_model,
                base_url=_settings.llm.ollama_url,
            )
            llm = llm_registry.get_active()
            logger.info("Auto-activated Ollama LLM for daily intelligence")
        except Exception as e:
            logger.warning("Could not auto-activate LLM: %s", e)
    if llm is None:
        logger.warning("No active LLM for daily intelligence")
        return None

    messages = [
        Message(role="system", content=skill.content),
        Message(
            role="user",
            content=json.dumps(payload, indent=2, default=str),
        ),
    ]

    try:
        result = llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,
        )
        text = result.get("response", "").strip()
        if not text:
            logger.warning("LLM returned empty response for daily intelligence")
            return None

        # Strip <think> tags (Qwen3 models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    except Exception:
        logger.exception("LLM call failed for daily intelligence")
        return None


def _parse_analysis(raw_text: str) -> dict[str, Any]:
    """Extract structured JSON from LLM response. Falls back to raw text."""
    # Try to find JSON block in the response
    # Look for ```json ... ``` first
    json_match = re.search(r"```json\s*(.*?)```", raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to parse the entire response as JSON
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Try to find any JSON object in the response
    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    # Try to recover truncated JSON (output cut off by max_tokens)
    recovered = _recover_truncated_json(raw_text)
    if recovered:
        return recovered

    # Fallback: wrap raw text
    return {"analysis_text": raw_text}


def _recover_truncated_json(raw_text: str) -> dict[str, Any] | None:
    """Attempt to recover a JSON object from truncated LLM output.

    When max_tokens cuts off output mid-JSON, we try closing open
    structures progressively to salvage whatever fields were complete.
    """
    # Find the start of JSON
    start = raw_text.find("{")
    if start < 0:
        return None

    text = raw_text[start:]

    # Try closing open brackets/braces from the end
    for trim in range(0, min(len(text), 500), 1):
        candidate = text if trim == 0 else text[:-trim]
        # Count unclosed structures
        opens = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if opens <= 0 and open_brackets <= 0:
            continue
        # Build closing sequence
        suffix = "]" * max(open_brackets, 0) + "}" * max(opens, 0)
        try:
            result = json.loads(candidate + suffix)
            if isinstance(result, dict) and result.get("analysis_text"):
                logger.info(
                    "Recovered truncated JSON (trimmed %d chars, closed %d braces)",
                    trim, opens + open_brackets,
                )
                return result
        except json.JSONDecodeError:
            continue

    return None


# ------------------------------------------------------------------
# Notification
# ------------------------------------------------------------------


async def _send_notification(analysis_text: str, task: ScheduledTask) -> None:
    """Send ntfy push notification with analysis summary."""
    from ...config import settings as _settings

    if not _settings.autonomous.notify_results:
        return
    if not _settings.alerts.ntfy_enabled:
        return
    if (task.metadata or {}).get("notify") is False:
        return

    title = "Atlas: Daily Intelligence"
    priority = (task.metadata or {}).get("notify_priority", "default")
    tags = (task.metadata or {}).get("notify_tags", "brain,chart_with_upwards_trend")

    try:
        from ...tools.notify import notify_tool

        await notify_tool._send_notification(
            message=analysis_text[:4000],  # ntfy has a ~4KB limit
            title=title,
            priority=priority,
            tags=tags,
        )
        logger.info("Sent daily intelligence notification")
    except Exception:
        logger.warning("Failed to send daily intelligence notification", exc_info=True)
