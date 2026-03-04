"""Tests for B2B keyword search volume signal collection task."""

import json
import sys
from datetime import date, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pre-mock heavy deps before importing the task module
# ---------------------------------------------------------------------------
for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "starlette", "starlette.requests",
    "asyncpg",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_pool(fetch_return=None):
    """Build a mock asyncpg pool."""
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=fetch_return or [])
    pool.execute = AsyncMock()
    return pool


def _mock_pool_seq(*fetch_sequence):
    """Build a mock pool with ordered fetch returns, defaulting to [] when exhausted."""
    pool = MagicMock()
    pool.is_initialized = True
    pool.execute = AsyncMock()
    items = list(fetch_sequence)

    async def _fetch(*args, **kwargs):
        if items:
            return items.pop(0)
        return []

    pool.fetch = AsyncMock(side_effect=_fetch)
    return pool


def _mock_cfg(**overrides):
    """Build a mock B2BChurnConfig."""
    defaults = {
        "keyword_signal_enabled": True,
        "keyword_spike_threshold_pct": 50.0,
        "keyword_query_delay_seconds": 0.0,  # No delay in tests
        "keyword_max_vendors_per_run": 20,
        "keyword_geo": "US",
    }
    defaults.update(overrides)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_vendor_row(name: str) -> dict:
    return {"vendor_name": name}


def _make_competitor_row(vendor: str, competitor: str) -> dict:
    return {"vendor_name": vendor, "top_competitor": competitor}


def _make_prior_row(volume: int) -> dict:
    return {"volume_relative": volume}


def _make_task() -> MagicMock:
    """Build a minimal ScheduledTask mock."""
    task = MagicMock()
    task.name = "b2b_keyword_signal"
    return task


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestKeywordSignalTask:
    """Tests for b2b_keyword_signal.run()."""

    # -- Early returns --

    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        cfg = _mock_cfg(keyword_signal_enabled=False)
        with patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.settings") as mock_settings:
            mock_settings.b2b_churn = cfg
            from atlas_brain.autonomous.tasks.b2b_keyword_signal import run
            result = await run(_make_task())
        assert result["skipped"] == "keyword signal disabled"
        assert result["_skip_synthesis"] is True

    @pytest.mark.asyncio
    async def test_skip_when_db_not_ready(self):
        cfg = _mock_cfg()
        pool = _mock_pool()
        pool.is_initialized = False
        with (
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.settings") as mock_settings,
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.get_db_pool", return_value=pool),
        ):
            mock_settings.b2b_churn = cfg
            from atlas_brain.autonomous.tasks.b2b_keyword_signal import run
            result = await run(_make_task())
        assert result["skipped"] == "db not ready"

    @pytest.mark.asyncio
    async def test_skip_when_no_vendors(self):
        cfg = _mock_cfg()
        pool = _mock_pool(fetch_return=[])
        with (
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.settings") as mock_settings,
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.get_db_pool", return_value=pool),
        ):
            mock_settings.b2b_churn = cfg
            from atlas_brain.autonomous.tasks.b2b_keyword_signal import run
            result = await run(_make_task())
        assert result["skipped"] == "no enabled vendors"

    # -- Template generation --

    def test_query_templates_all_present(self):
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _QUERY_TEMPLATES
        assert "alternative" in _QUERY_TEMPLATES
        assert "vs_competitor" in _QUERY_TEMPLATES
        assert "cancel" in _QUERY_TEMPLATES
        assert "migrate" in _QUERY_TEMPLATES
        assert "pricing" in _QUERY_TEMPLATES

    def test_template_formatting_without_competitor(self):
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _QUERY_TEMPLATES
        vendor = "Salesforce"
        keywords = []
        for tmpl_name, tmpl in _QUERY_TEMPLATES.items():
            if tmpl_name == "vs_competitor":
                continue  # No competitor available
            keywords.append((tmpl_name, tmpl.format(vendor=vendor)))
        assert len(keywords) == 4
        assert ("alternative", "Salesforce alternative") in keywords
        assert ("cancel", "cancel Salesforce") in keywords
        assert ("migrate", "migrate from Salesforce") in keywords
        assert ("pricing", "Salesforce pricing") in keywords

    def test_template_formatting_with_competitor(self):
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _QUERY_TEMPLATES
        vendor = "Salesforce"
        competitor = "HubSpot"
        tmpl = _QUERY_TEMPLATES["vs_competitor"]
        result = tmpl.format(vendor=vendor, competitor=competitor)
        assert result == "Salesforce vs HubSpot"

    # -- Vendor limit --

    @pytest.mark.asyncio
    async def test_vendor_limit_cap(self):
        """Vendors are capped at keyword_max_vendors_per_run."""
        cfg = _mock_cfg(keyword_max_vendors_per_run=2)
        vendors = [_make_vendor_row(f"Vendor{i}") for i in range(5)]

        # Mock pytrends to return empty DataFrame
        mock_pytrends_cls = MagicMock()
        mock_pytrends_inst = MagicMock()
        mock_pytrends_inst.build_payload = MagicMock()
        mock_pytrends_inst.interest_over_time = MagicMock(return_value=None)
        mock_pytrends_cls.return_value = mock_pytrends_inst

        # Use _mock_pool_seq: vendors, competitors, then [] for all prior-row lookups
        pool = _mock_pool_seq(vendors, [])

        with (
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.settings") as mock_settings,
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.get_db_pool", return_value=pool),
            patch.dict(sys.modules, {"pytrends": MagicMock(), "pytrends.request": MagicMock(TrendReq=mock_pytrends_cls)}),
        ):
            mock_settings.b2b_churn = cfg
            from atlas_brain.autonomous.tasks.b2b_keyword_signal import run
            result = await run(_make_task())
        assert result["vendors_processed"] == 2

    # -- Main flow with data --

    @pytest.mark.asyncio
    async def test_successful_run_with_trends_data(self):
        """Full happy path: 1 vendor, pytrends returns data, upserts happen."""
        cfg = _mock_cfg()
        vendors = [_make_vendor_row("Acme")]
        competitors = [_make_competitor_row("Acme", "Globex")]

        # Build a fake DataFrame-like object using a class to avoid MagicMock __getitem__ issues
        class FakeDataFrame:
            empty = False
            columns = ["Acme alternative", "Acme vs Globex", "cancel Acme", "migrate from Acme", "Acme pricing"]

            def __getitem__(self, key):
                col = MagicMock()
                col.iloc.__getitem__ = MagicMock(return_value=75)
                col.tolist = MagicMock(return_value=[50, 60, 70, 75])
                return col

            def __contains__(self, key):
                return key in self.columns

        mock_df = FakeDataFrame()

        mock_pytrends_cls = MagicMock()
        mock_pytrends_inst = MagicMock()
        mock_pytrends_inst.build_payload = MagicMock()
        mock_pytrends_inst.interest_over_time = MagicMock(return_value=mock_df)
        mock_pytrends_cls.return_value = mock_pytrends_inst

        # Use _mock_pool_seq: vendors, competitors, then [] for all prior-row lookups
        pool = _mock_pool_seq(vendors, competitors)

        with (
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.settings") as mock_settings,
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.get_db_pool", return_value=pool),
            patch.dict(sys.modules, {"pytrends": MagicMock(), "pytrends.request": MagicMock(TrendReq=mock_pytrends_cls)}),
        ):
            mock_settings.b2b_churn = cfg
            from atlas_brain.autonomous.tasks.b2b_keyword_signal import run
            result = await run(_make_task())

        assert result["vendors_processed"] == 1
        assert result["keywords_upserted"] == 5
        # With no prior rows, no spikes detected
        assert result["spikes_found"] == 0
        # 5 execute calls for 5 keywords
        assert pool.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_empty_dataframe_records_zeros(self):
        """When pytrends returns None/empty, all keywords get volume=0."""
        cfg = _mock_cfg()
        vendors = [_make_vendor_row("Acme")]

        mock_pytrends_cls = MagicMock()
        mock_pytrends_inst = MagicMock()
        mock_pytrends_inst.build_payload = MagicMock()
        mock_pytrends_inst.interest_over_time = MagicMock(return_value=None)
        mock_pytrends_cls.return_value = mock_pytrends_inst

        # Use _mock_pool_seq: vendors, competitors (empty), then [] for prior rows
        pool = _mock_pool_seq(vendors, [])

        with (
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.settings") as mock_settings,
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.get_db_pool", return_value=pool),
            patch.dict(sys.modules, {"pytrends": MagicMock(), "pytrends.request": MagicMock(TrendReq=mock_pytrends_cls)}),
        ):
            mock_settings.b2b_churn = cfg
            from atlas_brain.autonomous.tasks.b2b_keyword_signal import run
            result = await run(_make_task())

        assert result["vendors_processed"] == 1
        # 4 keywords (no competitor = no vs_competitor)
        assert result["keywords_upserted"] == 4
        assert result["spikes_found"] == 0

    # -- Rate limit handling --

    @pytest.mark.asyncio
    async def test_rate_limit_stops_processing(self):
        """429 from pytrends should break out of vendor loop."""
        cfg = _mock_cfg()
        vendors = [_make_vendor_row("V1"), _make_vendor_row("V2")]

        mock_pytrends_cls = MagicMock()
        mock_pytrends_inst = MagicMock()
        mock_pytrends_inst.build_payload = MagicMock()
        mock_pytrends_inst.interest_over_time = MagicMock(
            side_effect=Exception("429 Too Many Requests")
        )
        mock_pytrends_cls.return_value = mock_pytrends_inst

        pool = _mock_pool_seq(vendors, [])

        with (
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.settings") as mock_settings,
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.get_db_pool", return_value=pool),
            patch.dict(sys.modules, {"pytrends": MagicMock(), "pytrends.request": MagicMock(TrendReq=mock_pytrends_cls)}),
        ):
            mock_settings.b2b_churn = cfg
            from atlas_brain.autonomous.tasks.b2b_keyword_signal import run
            result = await run(_make_task())

        # 429 breaks the loop, no vendors processed
        assert result["vendors_processed"] == 0
        assert result["keywords_upserted"] == 0

    @pytest.mark.asyncio
    async def test_non_429_error_continues(self):
        """Non-rate-limit errors should continue to next vendor."""
        cfg = _mock_cfg()
        vendors = [_make_vendor_row("V1"), _make_vendor_row("V2")]

        call_count = 0

        def _side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return None  # V2 returns empty

        mock_pytrends_cls = MagicMock()
        mock_pytrends_inst = MagicMock()
        mock_pytrends_inst.build_payload = MagicMock()
        mock_pytrends_inst.interest_over_time = MagicMock(side_effect=_side_effect)
        mock_pytrends_cls.return_value = mock_pytrends_inst

        # Use _mock_pool_seq: vendors, competitors (empty), then [] for prior rows
        pool = _mock_pool_seq(vendors, [])

        with (
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.settings") as mock_settings,
            patch("atlas_brain.autonomous.tasks.b2b_keyword_signal.get_db_pool", return_value=pool),
            patch.dict(sys.modules, {"pytrends": MagicMock(), "pytrends.request": MagicMock(TrendReq=mock_pytrends_cls)}),
        ):
            mock_settings.b2b_churn = cfg
            from atlas_brain.autonomous.tasks.b2b_keyword_signal import run
            result = await run(_make_task())

        # V1 failed (non-429), V2 processed with empty data
        assert result["vendors_processed"] == 1


class TestUpsertSignal:
    """Tests for _upsert_signal spike detection and rolling average."""

    @pytest.mark.asyncio
    async def test_no_prior_data_no_spike(self):
        """First ever data point: no rolling avg, no spike."""
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _upsert_signal

        pool = _mock_pool(fetch_return=[])
        is_spike = await _upsert_signal(
            pool, "Acme", "Acme alternative", "alternative",
            80, date(2026, 3, 2), 50.0, {},
        )
        assert is_spike is False
        # Verify execute was called with correct args
        args = pool.execute.call_args[0]
        assert args[1] == "Acme"           # vendor
        assert args[2] == "Acme alternative"  # keyword
        assert args[3] == "alternative"     # template
        assert args[4] == 80               # volume
        assert args[5] is None             # rolling_avg (no prior data)
        assert args[6] is None             # volume_change_pct
        assert args[7] is False            # is_spike

    @pytest.mark.asyncio
    async def test_spike_detected(self):
        """Volume 80 vs rolling avg 50 = 60% change >= 50% threshold = spike."""
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _upsert_signal

        prior = [_make_prior_row(50), _make_prior_row(50), _make_prior_row(50), _make_prior_row(50)]
        pool = _mock_pool(fetch_return=prior)
        is_spike = await _upsert_signal(
            pool, "Acme", "Acme alternative", "alternative",
            80, date(2026, 3, 2), 50.0, {},
        )
        assert is_spike is True
        args = pool.execute.call_args[0]
        assert args[5] == 50.0             # rolling_avg
        assert args[6] == 60.0             # (80-50)/50*100
        assert args[7] is True             # is_spike

    @pytest.mark.asyncio
    async def test_no_spike_below_threshold(self):
        """Volume 70 vs rolling avg 50 = 40% change < 50% threshold = no spike."""
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _upsert_signal

        prior = [_make_prior_row(50), _make_prior_row(50)]
        pool = _mock_pool(fetch_return=prior)
        is_spike = await _upsert_signal(
            pool, "Acme", "Acme alternative", "alternative",
            70, date(2026, 3, 2), 50.0, {},
        )
        assert is_spike is False

    @pytest.mark.asyncio
    async def test_spike_at_exact_threshold(self):
        """Exactly at threshold should be flagged as spike."""
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _upsert_signal

        prior = [_make_prior_row(100)]
        pool = _mock_pool(fetch_return=prior)
        # 150 vs 100 = 50% change == 50% threshold
        is_spike = await _upsert_signal(
            pool, "Acme", "Acme alternative", "alternative",
            150, date(2026, 3, 2), 50.0, {},
        )
        assert is_spike is True

    @pytest.mark.asyncio
    async def test_rolling_avg_zero_no_spike(self):
        """When rolling avg is 0, division by zero is avoided, no spike."""
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _upsert_signal

        prior = [_make_prior_row(0), _make_prior_row(0)]
        pool = _mock_pool(fetch_return=prior)
        is_spike = await _upsert_signal(
            pool, "Acme", "Acme alternative", "alternative",
            50, date(2026, 3, 2), 50.0, {},
        )
        assert is_spike is False
        args = pool.execute.call_args[0]
        assert args[5] == 0.0              # rolling_avg
        assert args[6] is None             # volume_change_pct (skipped due to 0 avg)

    @pytest.mark.asyncio
    async def test_rolling_avg_with_mixed_values(self):
        """Rolling avg computed correctly from mixed prior values."""
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _upsert_signal

        prior = [_make_prior_row(40), _make_prior_row(60), _make_prior_row(80)]
        pool = _mock_pool(fetch_return=prior)
        # avg = (40+60+80)/3 = 60.0
        # volume 120 vs 60 = 100% change
        is_spike = await _upsert_signal(
            pool, "Acme", "Acme alternative", "alternative",
            120, date(2026, 3, 2), 50.0, {},
        )
        assert is_spike is True
        args = pool.execute.call_args[0]
        assert args[5] == 60.0             # rolling_avg
        assert args[6] == 100.0            # (120-60)/60*100

    @pytest.mark.asyncio
    async def test_raw_response_serialized_as_json(self):
        """raw_response dict is JSON-serialized before DB insert."""
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _upsert_signal

        pool = _mock_pool(fetch_return=[])
        raw = {"Acme alternative": [10, 20, 30]}
        await _upsert_signal(
            pool, "Acme", "Acme alternative", "alternative",
            30, date(2026, 3, 2), 50.0, raw,
        )
        args = pool.execute.call_args[0]
        assert args[9] == json.dumps(raw)  # $9 = raw_response as JSON string

    @pytest.mark.asyncio
    async def test_snapshot_week_passed_to_db(self):
        """snapshot_week is passed correctly as $8."""
        from atlas_brain.autonomous.tasks.b2b_keyword_signal import _upsert_signal

        pool = _mock_pool(fetch_return=[])
        week = date(2026, 3, 2)
        await _upsert_signal(
            pool, "Acme", "Acme alternative", "alternative",
            0, week, 50.0, {},
        )
        args = pool.execute.call_args[0]
        assert args[8] == week


class TestSnapshotWeekComputation:
    """Test ISO week start (Monday) calculation."""

    def test_monday_is_itself(self):
        """A Monday should be its own snapshot_week."""
        monday = date(2026, 3, 2)  # Monday
        assert monday.weekday() == 0
        snapshot = monday - timedelta(days=monday.weekday())
        assert snapshot == monday

    def test_wednesday_maps_to_monday(self):
        """Wednesday Mar 4 should map to Monday Mar 2."""
        wednesday = date(2026, 3, 4)
        snapshot = wednesday - timedelta(days=wednesday.weekday())
        assert snapshot == date(2026, 3, 2)

    def test_sunday_maps_to_monday(self):
        """Sunday Mar 8 should map to Monday Mar 2."""
        sunday = date(2026, 3, 8)
        snapshot = sunday - timedelta(days=sunday.weekday())
        assert snapshot == date(2026, 3, 2)
