from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import (
    anomaly_detection,
    b2b_scrape_intake,
    market_intake,
    news_intake,
    reasoning_tick,
)
from atlas_brain.storage import database as storage_database
from atlas_brain.storage.models import ScheduledTask


def _task(name: str) -> ScheduledTask:
    return ScheduledTask(
        id=uuid4(),
        name=name,
        task_type="builtin",
        schedule_type="interval",
        interval_seconds=300,
        prompt=None,
    )


class _MarketPool:
    def __init__(self, rows):
        self.is_initialized = True
        self._rows = rows
        self.inserted = []

    async def fetch(self, _query):
        return self._rows

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def executemany(self, _query, rows):
        self.inserted.extend(rows)


class _AnomalyPool:
    def __init__(self, pattern_count, recent_events=None):
        self.is_initialized = True
        self._pattern_count = pattern_count
        self._recent_events = recent_events or []

    async def fetchval(self, _query):
        return self._pattern_count

    async def fetch(self, _query, _cutoff):
        return self._recent_events

    async def fetchrow(self, _query, _person, _transition, _dow):
        return None


class _NewsPool:
    def __init__(self, watchlist_rows=None, market_rows=None):
        self.is_initialized = True
        self.watchlist_rows = watchlist_rows or []
        self.market_rows = market_rows or []

    async def fetch(self, query):
        if "category IN ('news_topic', 'news_region')" in query:
            return self.watchlist_rows
        if "category IN ('stock', 'etf', 'commodity', 'crypto', 'forex')" in query:
            return self.market_rows
        return []


class _ReasoningTickPool:
    def __init__(self, rows=None):
        self.is_initialized = True
        self.rows = rows or []
        self.updated = []

    async def fetch(self, _query, _limit):
        return self.rows

    async def execute(self, _query, result_json, event_id):
        self.updated.append((result_json, event_id))
        return "UPDATE 1"


class _ScrapePool:
    def __init__(self, target_rows=None):
        self.is_initialized = True
        self.target_rows = target_rows or []

    async def fetch(self, query, *_args):
        if "FROM scrape_targets" in query:
            return self.target_rows
        return []


@pytest.mark.asyncio
async def test_market_intake_disabled_emits_trigger_reason(monkeypatch):
    monkeypatch.setattr(market_intake.settings.external_data, "enabled", False, raising=False)
    monkeypatch.setattr(market_intake.settings.external_data, "market_enabled", False, raising=False)

    result = await market_intake.run(_task("market_intake"))

    assert result["trigger_reason"] == "Market intake disabled"
    assert result["skip_reason"] == "Market intake disabled"
    assert result["skipped"] == "external_data or market disabled"


@pytest.mark.asyncio
async def test_market_intake_success_emits_specific_trigger_reason(monkeypatch):
    pool = _MarketPool(
        [
            {
                "id": uuid4(),
                "category": "stock",
                "symbol": "MSFT",
                "name": "Microsoft",
                "threshold_pct": 3.0,
                "metadata": {},
            }
        ]
    )
    monkeypatch.setattr(market_intake.settings.external_data, "enabled", True, raising=False)
    monkeypatch.setattr(market_intake.settings.external_data, "market_enabled", True, raising=False)
    monkeypatch.setattr(market_intake.settings.external_data, "market_hours_only", False, raising=False)
    monkeypatch.setattr(market_intake, "get_db_pool", lambda: pool)

    async def _fake_fetch_prices(_symbols, _provider, _api_key, _timeout):
        return {
            "MSFT": {
                "price": 100.0,
                "change_pct": 5.2,
                "volume": 12345,
            }
        }

    monkeypatch.setattr(market_intake, "_fetch_prices", _fake_fetch_prices)

    result = await market_intake.run(_task("market_intake"))

    assert result["trigger_reason"] == "Significant market moves detected"
    assert result["skip_reason"] == "Market snapshots recorded"
    assert result["snapshots_recorded"] == 1
    assert result["significant_moves"] == 1
    assert len(pool.inserted) == 1


@pytest.mark.asyncio
async def test_anomaly_detection_no_patterns_emits_trigger_reason(monkeypatch):
    monkeypatch.setattr(storage_database, "get_db_pool", lambda: _AnomalyPool(pattern_count=0))

    result = await anomaly_detection.run(_task("anomaly_detection"))

    assert result["trigger_reason"] == "Anomaly detection skipped -- no learned patterns yet."
    assert result["skip_reason"] == "Anomaly detection skipped -- no learned patterns yet."
    assert result["note"] == "No learned patterns yet"


@pytest.mark.asyncio
async def test_anomaly_detection_no_anomalies_emits_trigger_reason(monkeypatch):
    pool = _AnomalyPool(pattern_count=3, recent_events=[])
    monkeypatch.setattr(storage_database, "get_db_pool", lambda: pool)

    async def _no_device_anomalies(_pool, _current_minutes, _today_str):
        return []

    monkeypatch.setattr(anomaly_detection, "_check_device_anomalies", _no_device_anomalies)

    result = await anomaly_detection.run(_task("anomaly_detection"))

    assert result["trigger_reason"] == "No anomalies detected"
    assert result["checked"] == 0
    assert result["anomalies"] == 0


@pytest.mark.asyncio
async def test_news_intake_disabled_emits_trigger_reason(monkeypatch):
    monkeypatch.setattr(news_intake.settings.external_data, "enabled", False, raising=False)
    monkeypatch.setattr(news_intake.settings.external_data, "news_enabled", False, raising=False)

    result = await news_intake.run(_task("news_intake"))

    assert result["trigger_reason"] == "News intake disabled"
    assert result["skip_reason"] == "News intake disabled"


@pytest.mark.asyncio
async def test_news_intake_no_articles_emits_trigger_reason(monkeypatch):
    pool = _NewsPool(
        watchlist_rows=[
            {
                "id": uuid4(),
                "name": "Cloud",
                "category": "news_topic",
                "keywords": ["cloud"],
                "metadata": {},
            }
        ]
    )
    monkeypatch.setattr(news_intake.settings.external_data, "enabled", True, raising=False)
    monkeypatch.setattr(news_intake.settings.external_data, "news_enabled", True, raising=False)
    monkeypatch.setattr(news_intake, "get_db_pool", lambda: pool)

    async def _no_articles(*_args, **_kwargs):
        return []

    monkeypatch.setattr(news_intake, "_fetch_articles", _no_articles)

    result = await news_intake.run(_task("news_intake"))

    assert result["trigger_reason"] == "News intake skipped -- no articles fetched"
    assert result["skip_reason"] == "News intake skipped -- no articles fetched"


@pytest.mark.asyncio
async def test_reasoning_tick_no_rows_emits_trigger_reason(monkeypatch):
    monkeypatch.setattr(reasoning_tick, "get_db_pool", lambda: _ReasoningTickPool(rows=[]))
    monkeypatch.setattr(reasoning_tick.settings.reasoning, "enabled", True, raising=False)

    result = await reasoning_tick.run(_task("reasoning_tick"))

    assert result["trigger_reason"] == "No pending reasoning events"
    assert result["skip_reason"] == "No pending reasoning events"
    assert result["picked_up"] == 0


@pytest.mark.asyncio
async def test_b2b_scrape_intake_no_targets_emits_trigger_reason(monkeypatch):
    pool = _ScrapePool(target_rows=[])
    monkeypatch.setattr(b2b_scrape_intake.settings.b2b_scrape, "enabled", True, raising=False)
    monkeypatch.setattr(b2b_scrape_intake, "get_db_pool", lambda: pool)
    monkeypatch.setattr(b2b_scrape_intake, "parse_source_allowlist", lambda _value: ["g2"])

    import atlas_brain.services.scraping.client as scraping_client
    import atlas_brain.services.scraping.capabilities as scraping_capabilities
    import atlas_brain.services.scraping.parsers as scraping_parsers

    monkeypatch.setattr(scraping_client, "get_scrape_client", lambda: object())
    monkeypatch.setattr(scraping_capabilities, "get_all_capabilities", lambda: {"g2": type("Cap", (), {"cooldown_minutes": 60, "max_concurrency": 2})()})
    monkeypatch.setattr(scraping_capabilities, "get_capability", lambda _source: type("Cap", (), {"cooldown_minutes": 60, "max_concurrency": 2})())
    monkeypatch.setattr(scraping_parsers, "get_all_parsers", lambda: {"g2": object()})
    monkeypatch.setattr(scraping_parsers, "get_parser", lambda _source: object())

    result = await b2b_scrape_intake.run(_task("b2b_scrape_intake"))

    assert result["trigger_reason"] == "No scrape targets due"
    assert result["skip_reason"] == "No scrape targets due"
    assert result["targets_due"] == 0
