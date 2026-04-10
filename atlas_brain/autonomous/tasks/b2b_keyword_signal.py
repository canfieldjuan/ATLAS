"""
B2B keyword search volume signal collection via Google Trends.

Polls pytrends for churn-indicative queries per vendor (e.g. "Salesforce
alternative", "cancel Salesforce"). Computes rolling 4-week averages and
spike detection. Results stored in b2b_keyword_signals for downstream
aggregation by b2b_churn_intelligence.

Runs weekly (default Monday 6 AM).
"""

import asyncio
import importlib
import inspect
import json
import logging
from datetime import date, timedelta
from typing import Any

from ...config import settings
from ...services.vendor_registry import resolve_vendor_name
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._b2b_shared import read_vendor_top_competitor_map

logger = logging.getLogger("atlas.autonomous.tasks.b2b_keyword_signal")

# Keyword templates: key is the template name, value is the format string
_QUERY_TEMPLATES = {
    "alternative": "{vendor} alternative",
    "vs_competitor": "{vendor} vs {competitor}",
    "cancel": "cancel {vendor}",
    "migrate": "migrate from {vendor}",
    "pricing": "{vendor} pricing",
}


def _patch_pytrends_retry_compat(pytrends_request_module: Any) -> None:
    """Adapt pytrends' Retry usage for urllib3>=2 without touching global deps."""
    retry_cls = getattr(pytrends_request_module, "Retry", None)
    if retry_cls is None:
        return
    try:
        params = inspect.signature(retry_cls.__init__).parameters
    except (TypeError, ValueError):
        return
    if "method_whitelist" in params or "allowed_methods" not in params:
        return

    class CompatRetry(retry_cls):
        def __init__(self, *args: Any, method_whitelist: Any = None, **kwargs: Any) -> None:
            if method_whitelist is not None and "allowed_methods" not in kwargs:
                kwargs["allowed_methods"] = method_whitelist
            super().__init__(*args, **kwargs)

    pytrends_request_module.Retry = CompatRetry


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Collect Google Trends keyword signals for tracked B2B vendors."""
    cfg = settings.b2b_churn
    if not cfg.keyword_signal_enabled:
        return {"_skip_synthesis": True, "skipped": "keyword signal disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": True, "skipped": "db not ready"}

    try:
        pytrends_request = importlib.import_module("pytrends.request")
    except ImportError:
        logger.error("pytrends not installed -- pip install pytrends")
        return {"_skip_synthesis": True, "skipped": "pytrends not installed"}
    _patch_pytrends_retry_compat(pytrends_request)
    TrendReq = pytrends_request.TrendReq

    # Load enabled vendors
    rows = await pool.fetch(
        "SELECT DISTINCT vendor_name FROM b2b_scrape_targets WHERE enabled = true"
    )
    vendors_raw = [r["vendor_name"] for r in rows]
    if not vendors_raw:
        return {"_skip_synthesis": True, "skipped": "no enabled vendors"}
    # Canonicalize and deduplicate vendor names
    seen: set[str] = set()
    vendors: list[str] = []
    for v in vendors_raw:
        canon = await resolve_vendor_name(v)
        if canon not in seen:
            seen.add(canon)
            vendors.append(canon)

    vendors = vendors[: cfg.keyword_max_vendors_per_run]

    # Fetch top competitor per vendor for "vs" queries
    competitor_map = await read_vendor_top_competitor_map(
        pool,
        vendor_names=vendors,
    )

    # Current ISO week start (Monday)
    today = date.today()
    snapshot_week = today - timedelta(days=today.weekday())

    pytrends = TrendReq(hl="en-US", tz=360, retries=2, backoff_factor=1.0)
    geo = cfg.keyword_geo
    threshold_pct = cfg.keyword_spike_threshold_pct
    delay = cfg.keyword_query_delay_seconds

    vendors_processed = 0
    keywords_upserted = 0
    spikes_found = 0

    for vendor in vendors:
        # Build keyword list for this vendor
        keywords: list[tuple[str, str]] = []  # (template_name, keyword_string)
        for tmpl_name, tmpl in _QUERY_TEMPLATES.items():
            if tmpl_name == "vs_competitor":
                competitor = competitor_map.get(vendor)
                if not competitor:
                    continue
                keywords.append((tmpl_name, tmpl.format(vendor=vendor, competitor=competitor)))
            else:
                keywords.append((tmpl_name, tmpl.format(vendor=vendor)))

        if not keywords:
            continue

        # Query pytrends (max 5 keywords per call)
        # pytrends uses requests (synchronous) -- run in thread to avoid blocking
        kw_strings = [kw for _, kw in keywords]

        def _query_trends() -> "pd.DataFrame | None":
            pytrends.build_payload(kw_strings, timeframe="today 3-m", geo=geo)
            return pytrends.interest_over_time()

        try:
            df = await asyncio.to_thread(_query_trends)
        except Exception as exc:
            exc_name = type(exc).__name__
            if "429" in str(exc) or "TooManyRequests" in exc_name:
                logger.warning("Google Trends rate limited at vendor %s, stopping", vendor)
                break
            logger.warning("pytrends query failed for %s: %s", vendor, exc)
            await asyncio.sleep(delay)
            continue

        if df is None or df.empty:
            # No data -- record zeros
            for tmpl_name, kw in keywords:
                await _upsert_signal(
                    pool, vendor, kw, tmpl_name, 0, snapshot_week, threshold_pct, {}
                )
                keywords_upserted += 1
            vendors_processed += 1
            await asyncio.sleep(delay)
            continue

        # Extract latest week values
        for tmpl_name, kw in keywords:
            kw_raw: dict = {}
            if kw in df.columns:
                volume = int(df[kw].iloc[-1])
                kw_raw[kw] = df[kw].tolist()
            else:
                volume = 0

            is_spike = await _upsert_signal(
                pool, vendor, kw, tmpl_name, volume, snapshot_week,
                threshold_pct, kw_raw,
            )
            keywords_upserted += 1
            if is_spike:
                spikes_found += 1

        vendors_processed += 1
        logger.info("Keyword signals for %s: %d keywords", vendor, len(keywords))

        # Rate limit protection
        await asyncio.sleep(delay)

    return {
        "_skip_synthesis": True,
        "vendors_processed": vendors_processed,
        "keywords_upserted": keywords_upserted,
        "spikes_found": spikes_found,
    }


async def _upsert_signal(
    pool,
    vendor: str,
    keyword: str,
    query_template: str,
    volume: int,
    snapshot_week: date,
    threshold_pct: float,
    raw_response: dict,
) -> bool:
    """Upsert a keyword signal row. Returns True if it's a spike."""
    # Compute rolling 4-week average from prior snapshots
    prior_rows = await pool.fetch(
        """
        SELECT volume_relative
        FROM b2b_keyword_signals
        WHERE vendor_name = $1
          AND keyword = $2
          AND snapshot_week < $3
        ORDER BY snapshot_week DESC
        LIMIT 4
        """,
        vendor, keyword, snapshot_week,
    )

    rolling_avg = None
    volume_change_pct = None
    is_spike = False

    if prior_rows:
        prior_values = [r["volume_relative"] for r in prior_rows]
        rolling_avg = round(sum(prior_values) / len(prior_values), 1)
        if rolling_avg > 0:
            volume_change_pct = round((volume - rolling_avg) / rolling_avg * 100, 2)
            is_spike = volume_change_pct >= threshold_pct

    await pool.execute(
        """
        INSERT INTO b2b_keyword_signals (
            vendor_name, keyword, query_template, volume_relative,
            rolling_avg_4w, volume_change_pct, is_spike,
            snapshot_week, raw_response
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (vendor_name, keyword, snapshot_week) DO UPDATE SET
            volume_relative = EXCLUDED.volume_relative,
            rolling_avg_4w = EXCLUDED.rolling_avg_4w,
            volume_change_pct = EXCLUDED.volume_change_pct,
            is_spike = EXCLUDED.is_spike,
            snapshot_at = NOW(),
            raw_response = EXCLUDED.raw_response
        """,
        vendor, keyword, query_template, volume,
        rolling_avg, volume_change_pct, is_spike,
        snapshot_week, json.dumps(raw_response),
    )

    return is_spike
