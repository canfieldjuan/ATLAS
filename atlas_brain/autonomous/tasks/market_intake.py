"""
Market data intake: poll prices for watchlist symbols, detect significant
moves, record snapshots, emit market.* events.

Uses Alpha Vantage (free tier: 25 req/day) by default. Runs as an
autonomous task on a configurable interval (default 5 min).
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.market_intake")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: poll market prices and emit events."""
    cfg = settings.external_data
    if not cfg.enabled or not cfg.market_enabled:
        return {"_skip_synthesis": True, "skipped": "external_data or market disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": True, "skipped": "db not ready"}

    # Check market hours if configured
    if cfg.market_hours_only and not _is_market_hours():
        return {"_skip_synthesis": True, "skipped": "outside market hours"}

    # Load watchlist symbols
    rows = await pool.fetch(
        """
        SELECT id, category, symbol, name, threshold_pct, metadata
        FROM data_watchlist
        WHERE enabled = true
          AND category IN ('stock', 'etf', 'commodity', 'crypto', 'forex')
          AND symbol IS NOT NULL
        """
    )
    if not rows:
        return {"_skip_synthesis": True, "symbols_checked": 0, "emitted": 0}

    watchlist = {r["symbol"]: dict(r) for r in rows}
    symbols = list(watchlist.keys())

    # Fetch prices
    quotes = await _fetch_prices(
        symbols, cfg.market_api_provider, cfg.market_api_key, cfg.api_timeout_seconds
    )
    if not quotes:
        return {"_skip_synthesis": True, "symbols_checked": len(symbols), "emitted": 0, "error": "no quotes"}

    # Record snapshots
    snapshot_rows = []
    for sym, q in quotes.items():
        snapshot_rows.append((
            sym,
            q["price"],
            q.get("change_pct"),
            q.get("volume"),
        ))

    if snapshot_rows:
        await pool.executemany(
            """
            INSERT INTO market_snapshots (symbol, price, change_pct, volume)
            VALUES ($1, $2, $3, $4)
            """,
            snapshot_rows,
        )

    # Detect significant moves
    emitted = 0
    for sym, q in quotes.items():
        wl = watchlist.get(sym)
        if not wl:
            continue

        threshold = wl.get("threshold_pct") or cfg.market_default_threshold_pct
        change_pct = q.get("change_pct")
        if change_pct is None or abs(change_pct) < threshold:
            continue

        # Dedup: one alert per symbol per direction per day
        direction = "up" if change_pct > 0 else "down"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dedup_key = f"{sym}:{today}:{direction}"

        inserted = await pool.fetchval(
            """
            INSERT INTO data_dedup (source, dedup_key)
            VALUES ('market', $1)
            ON CONFLICT (source, dedup_key) DO NOTHING
            RETURNING id
            """,
            dedup_key,
        )
        if not inserted:
            continue  # already emitted today

        # Emit event
        from ...reasoning.producers import emit_if_enabled

        payload = {
            "symbol": sym,
            "name": wl["name"],
            "asset_type": wl["category"],
            "current_price": float(q["price"]),
            "previous_close": float(q.get("previous_close", 0)),
            "change_pct": round(float(change_pct), 2),
            "threshold_pct": float(threshold),
            "matched_watchlist_id": str(wl["id"]),
        }
        from ...reasoning.events import EventType

        await emit_if_enabled(
            event_type=EventType.MARKET_SIGNIFICANT_MOVE,
            source="market_intake",
            payload=payload,
        )
        emitted += 1
        logger.info(
            "Market alert: %s %s %.2f%% (threshold %.1f%%)",
            sym, direction, change_pct, threshold,
        )

    return {
        "_skip_synthesis": True,
        "symbols_checked": len(symbols),
        "snapshots_recorded": len(snapshot_rows),
        "emitted": emitted,
    }


def _is_market_hours() -> bool:
    """Check if current time is within US market hours (9:30-16:00 ET)."""
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


async def _fetch_prices(
    symbols: list[str], provider: str, api_key: str | None,
    timeout: float = 20.0,
) -> dict[str, dict[str, Any]]:
    """Fetch current prices for symbols.

    Returns {symbol: {price, change_pct, volume, previous_close}}.
    """
    if provider == "alpha_vantage":
        return await _fetch_alpha_vantage(symbols, api_key, timeout)
    elif provider == "finnhub":
        return await _fetch_finnhub(symbols, api_key, timeout)
    else:
        logger.warning("Unknown market provider: %s", provider)
        return {}


async def _fetch_alpha_vantage(
    symbols: list[str], api_key: str | None,
    timeout: float = 20.0,
) -> dict[str, dict[str, Any]]:
    """Fetch via Alpha Vantage GLOBAL_QUOTE endpoint.

    Free tier: 25 requests/day.  Each symbol is one request.
    Response key "Global Quote" contains:
      01. symbol, 02. open, 03. high, 04. low, 05. price,
      06. volume, 07. latest trading day, 08. previous close,
      09. change, 10. change percent
    """
    if not api_key:
        logger.warning("Alpha Vantage API key not configured (ATLAS_EXTERNAL_DATA_MARKET_API_KEY)")
        return {}

    import httpx

    results = {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        for sym in symbols:
            try:
                resp = await client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "GLOBAL_QUOTE",
                        "symbol": sym,
                        "apikey": api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                quote = data.get("Global Quote")
                if not quote:
                    # Rate-limit or invalid symbol
                    note = data.get("Note") or data.get("Information") or ""
                    if note:
                        logger.debug("Alpha Vantage note for %s: %s", sym, note[:120])
                    continue

                price = float(quote.get("05. price", 0))
                prev_close = float(quote.get("08. previous close", 0))
                if not price or not prev_close:
                    continue

                change_pct_str = quote.get("10. change percent", "0%")
                change_pct = float(change_pct_str.rstrip("%"))

                volume_str = quote.get("06. volume")
                volume = int(volume_str) if volume_str else None

                results[sym] = {
                    "price": price,
                    "previous_close": prev_close,
                    "change_pct": change_pct,
                    "volume": volume,
                }
            except Exception:
                logger.debug("Alpha Vantage fetch failed for %s", sym, exc_info=True)

    return results


async def _fetch_finnhub(
    symbols: list[str], api_key: str | None,
    timeout: float = 20.0,
) -> dict[str, dict[str, Any]]:
    """Fetch via Finnhub REST API."""
    if not api_key:
        logger.warning("Finnhub API key not configured")
        return {}

    import httpx

    results = {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        for sym in symbols:
            try:
                resp = await client.get(
                    "https://finnhub.io/api/v1/quote",
                    params={"symbol": sym, "token": api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                price = data.get("c")  # current
                prev_close = data.get("pc")  # previous close
                if price and prev_close and prev_close != 0:
                    results[sym] = {
                        "price": price,
                        "previous_close": prev_close,
                        "change_pct": ((price - prev_close) / prev_close) * 100,
                        "volume": None,
                    }
            except Exception:
                logger.debug("Finnhub fetch failed for %s", sym, exc_info=True)

    return results
