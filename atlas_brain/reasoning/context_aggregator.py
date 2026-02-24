"""Cross-domain context aggregator for the reasoning agent.

Pulls data from CRM, email, voice, calendar, SMS, and GraphRAG into
a single context dict. Each source has a 2s timeout; total cap is 10s.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger("atlas.reasoning.context_aggregator")

_SOURCE_TIMEOUT = 2.0  # per-source timeout
_TOTAL_TIMEOUT = 10.0  # total aggregation timeout


async def aggregate_context(
    entity_type: Optional[str],
    entity_id: Optional[str],
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Gather cross-domain context for an entity.

    Returns a dict with keys: crm, emails, voice, calendar, sms,
    graph_facts, recent_events. Missing sources return empty values.
    """
    ctx: dict[str, Any] = {
        "crm": None,
        "emails": [],
        "voice": [],
        "calendar": [],
        "sms": [],
        "graph_facts": [],
        "recent_events": [],
        "market_data": [],
        "recent_news": [],
    }

    if not entity_type or not entity_id:
        # Entity-less events (news, market) still get external context
        if event_type.startswith(("news.", "market.")):
            ext_tasks = {
                "market_data": _fetch_market_context(),
                "recent_news": _fetch_news_context(),
            }
            try:
                results = await asyncio.wait_for(
                    _gather_with_timeouts(ext_tasks), timeout=_TOTAL_TIMEOUT
                )
                ctx.update(results)
            except asyncio.TimeoutError:
                logger.warning("External context aggregation timed out")
        return ctx

    tasks = {
        "crm": _fetch_crm(entity_id),
        "emails": _fetch_emails(entity_id),
        "voice": _fetch_voice(entity_id),
        "calendar": _fetch_calendar(entity_id),
        "sms": _fetch_sms(entity_id),
        "recent_events": _fetch_recent_events(entity_type, entity_id),
        "market_data": _fetch_market_context(),
        "recent_news": _fetch_news_context(),
    }

    try:
        results = await asyncio.wait_for(
            _gather_with_timeouts(tasks), timeout=_TOTAL_TIMEOUT
        )
        ctx.update(results)
    except asyncio.TimeoutError:
        logger.warning("Context aggregation timed out (%.0fs cap)", _TOTAL_TIMEOUT)

    return ctx


async def _gather_with_timeouts(
    tasks: dict[str, Any],
) -> dict[str, Any]:
    """Run tasks concurrently, each with its own timeout."""
    results: dict[str, Any] = {}

    async def _run(name: str, coro):
        try:
            results[name] = await asyncio.wait_for(coro, timeout=_SOURCE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.debug("Context source '%s' timed out", name)
        except Exception:
            logger.debug("Context source '%s' failed", name, exc_info=True)

    await asyncio.gather(*[_run(n, c) for n, c in tasks.items()])
    return results


# ------------------------------------------------------------------
# Individual source fetchers
# ------------------------------------------------------------------


async def _fetch_crm(entity_id: str) -> Optional[dict[str, Any]]:
    """Fetch CRM context for a contact."""
    try:
        from ..services.customer_context import get_customer_context_service
        svc = get_customer_context_service()
        ctx = await svc.get_context(entity_id)
        return ctx if ctx else None
    except Exception:
        return None


async def _fetch_emails(entity_id: str) -> list[dict[str, Any]]:
    """Fetch recent processed emails for a contact (30-day window)."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    rows = await pool.fetch(
        """
        SELECT gmail_message_id, sender, subject, category, intent,
               replyable, created_at
        FROM processed_emails
        WHERE contact_id = $1
          AND created_at > NOW() - INTERVAL '30 days'
        ORDER BY created_at DESC
        LIMIT 20
        """,
        entity_id,
    )
    return [dict(r) for r in rows]


async def _fetch_voice(entity_id: str) -> list[dict[str, Any]]:
    """Fetch recent voice turns mentioning this contact."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    rows = await pool.fetch(
        """
        SELECT session_id, role, content, action_type, created_at
        FROM conversation_turns
        WHERE assistant_metadata->>'contact_id' = $1
          AND created_at > NOW() - INTERVAL '7 days'
        ORDER BY created_at DESC
        LIMIT 10
        """,
        entity_id,
    )
    return [dict(r) for r in rows]


async def _fetch_calendar(entity_id: str) -> list[dict[str, Any]]:
    """Fetch upcoming appointments for a contact."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    rows = await pool.fetch(
        """
        SELECT id, contact_id, service_type, scheduled_at, status,
               calendar_event_id
        FROM appointments
        WHERE contact_id = $1
          AND scheduled_at > NOW() - INTERVAL '7 days'
        ORDER BY scheduled_at ASC
        LIMIT 10
        """,
        entity_id,
    )
    return [dict(r) for r in rows]


async def _fetch_sms(entity_id: str) -> list[dict[str, Any]]:
    """Fetch recent SMS messages for a contact."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    try:
        rows = await pool.fetch(
            """
            SELECT id, direction, body, status, created_at
            FROM sms_messages
            WHERE contact_id = $1
              AND created_at > NOW() - INTERVAL '30 days'
            ORDER BY created_at DESC
            LIMIT 10
            """,
            entity_id,
        )
        return [dict(r) for r in rows]
    except Exception:
        # sms_messages table may not exist
        return []


async def _fetch_recent_events(
    entity_type: str, entity_id: str
) -> list[dict[str, Any]]:
    """Fetch recent atlas_events for the same entity."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    rows = await pool.fetch(
        """
        SELECT id, event_type, source, payload, created_at
        FROM atlas_events
        WHERE entity_type = $1
          AND entity_id = $2
          AND created_at > NOW() - INTERVAL '7 days'
        ORDER BY created_at DESC
        LIMIT 20
        """,
        entity_type,
        entity_id,
    )
    return [dict(r) for r in rows]


async def _fetch_market_context() -> list[dict[str, Any]]:
    """Fetch recent significant market moves (24h window)."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    try:
        rows = await pool.fetch(
            """
            SELECT ms.symbol, ms.price, ms.change_pct, ms.volume, ms.snapshot_at,
                   dw.name, dw.category, dw.threshold_pct
            FROM market_snapshots ms
            JOIN data_watchlist dw ON dw.symbol = ms.symbol AND dw.enabled = true
            WHERE ms.snapshot_at > NOW() - INTERVAL '24 hours'
              AND ms.change_pct IS NOT NULL
              AND ABS(ms.change_pct) >= COALESCE(dw.threshold_pct, 5.0)
            ORDER BY ABS(ms.change_pct) DESC
            LIMIT 20
            """
        )
        return [dict(r) for r in rows]
    except Exception:
        logger.debug("Market context fetch failed", exc_info=True)
        return []


async def _fetch_news_context() -> list[dict[str, Any]]:
    """Fetch recent news events (24h window)."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    try:
        rows = await pool.fetch(
            """
            SELECT id, event_type, payload, created_at
            FROM atlas_events
            WHERE event_type LIKE 'news.%'
              AND created_at > NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC
            LIMIT 20
            """
        )
        return [dict(r) for r in rows]
    except Exception:
        logger.debug("News context fetch failed", exc_info=True)
        return []
