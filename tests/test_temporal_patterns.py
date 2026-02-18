"""Tests for temporal pattern context generation.

Covers:
- format_minutes() utility (boundary values, rounding, clamping)
- get_temporal_context() async function (DB queries, caching, failure handling)
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.utils.time import format_minutes

# Patch target for the lazy import of get_db_pool inside get_temporal_context
_PATCH_DB_POOL = "atlas_brain.storage.database.get_db_pool"
_PATCH_DATE = "atlas_brain.orchestration.temporal.date"


# ---------------------------------------------------------------------------
# Fixture: reset module-level cache between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_temporal_cache():
    from atlas_brain.orchestration import temporal

    temporal._cache.clear()
    temporal._failure_until = 0.0
    yield
    temporal._cache.clear()
    temporal._failure_until = 0.0


# ---------------------------------------------------------------------------
# Helper: build a mock DB pool
# ---------------------------------------------------------------------------


def _make_pool(*, is_initialized: bool = True, rows: list | None = None):
    """Return a mock pool with async fetch() and is_initialized property."""
    pool = MagicMock()
    pool.is_initialized = is_initialized
    pool.fetch = AsyncMock(return_value=rows if rows is not None else [])
    return pool


# ---------------------------------------------------------------------------
# TestFormatMinutes
# ---------------------------------------------------------------------------


class TestFormatMinutes:
    """Test format_minutes() boundary values, rounding, and clamping."""

    def test_midnight(self):
        assert format_minutes(0) == "12:00 AM"

    def test_one_am(self):
        assert format_minutes(60) == "1:00 AM"

    def test_noon(self):
        assert format_minutes(720) == "12:00 PM"

    def test_one_pm(self):
        assert format_minutes(780) == "1:00 PM"

    def test_round_exact(self):
        # 420 minutes = 7:00 AM, rounding to 5 should not change it
        assert format_minutes(420, round_to=5) == "7:00 AM"

    def test_round_up(self):
        # 423/5 = 84.6 -> round(84.6) = 85 -> 85*5 = 425 -> 7:05 AM
        assert format_minutes(423, round_to=5) == "7:05 AM"

    def test_round_down(self):
        # 422/5 = 84.4 -> round(84.4) = 84 -> 84*5 = 420 -> 7:00 AM
        assert format_minutes(422, round_to=5) == "7:00 AM"

    def test_last_minute_of_day(self):
        assert format_minutes(1439) == "11:59 PM"

    def test_negative_clamped_to_zero(self):
        assert format_minutes(-10) == "12:00 AM"

    def test_over_max_clamped(self):
        assert format_minutes(2000) == "11:59 PM"

    def test_rounding_past_boundary_clamped(self):
        # 1438/5 = 287.6 -> round = 288 -> 288*5 = 1440 -> clamped to 1439
        assert format_minutes(1438, round_to=5) == "11:59 PM"

    def test_negative_rounding_clamped(self):
        # -3/5 = -0.6 -> round = -1 -> -1*5 = -5 -> clamped to 0
        assert format_minutes(-3, round_to=5) == "12:00 AM"

    def test_various_times(self):
        assert format_minutes(90) == "1:30 AM"
        assert format_minutes(750) == "12:30 PM"
        assert format_minutes(1080) == "6:00 PM"


# ---------------------------------------------------------------------------
# TestGetTemporalContext -- DB returns results
# ---------------------------------------------------------------------------


class TestGetTemporalContextResults:
    """Test get_temporal_context when DB returns pattern rows."""

    @pytest.mark.asyncio
    async def test_single_person_single_pattern(self):
        rows = [
            {"person_name": "Juan", "pattern_type": "wake_up", "median_minutes": 420},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)  # Monday
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result = await get_temporal_context()

        assert "Juan's routine" in result
        assert "wake_up ~7:00 AM" in result
        assert "(Mon)" in result

    @pytest.mark.asyncio
    async def test_single_person_multiple_patterns(self):
        rows = [
            {"person_name": "Juan", "pattern_type": "wake_up", "median_minutes": 420},
            {"person_name": "Juan", "pattern_type": "sleep", "median_minutes": 1380},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)  # Monday
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result = await get_temporal_context()

        assert "Juan's routine (Mon):" in result
        assert "wake_up ~7:00 AM" in result
        assert "sleep ~11:00 PM" in result

    @pytest.mark.asyncio
    async def test_multiple_people(self):
        rows = [
            {"person_name": "Alice", "pattern_type": "wake_up", "median_minutes": 360},
            {"person_name": "Bob", "pattern_type": "wake_up", "median_minutes": 480},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 18)  # Wednesday
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result = await get_temporal_context()

        assert "Alice's routine (Wed):" in result
        assert "Bob's routine (Wed):" in result
        assert "wake_up ~6:00 AM" in result
        assert "wake_up ~8:00 AM" in result


# ---------------------------------------------------------------------------
# TestGetTemporalContext -- empty / not initialized
# ---------------------------------------------------------------------------


class TestGetTemporalContextEmpty:
    """Test get_temporal_context when DB returns no rows or is unavailable."""

    @pytest.mark.asyncio
    async def test_empty_results(self):
        pool = _make_pool(rows=[])

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result = await get_temporal_context()

        assert result == ""

    @pytest.mark.asyncio
    async def test_db_not_initialized(self):
        pool = _make_pool(is_initialized=False)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result = await get_temporal_context()

        assert result == ""
        # fetch() should never be called when pool is not initialized
        pool.fetch.assert_not_called()


# ---------------------------------------------------------------------------
# TestGetTemporalContext -- caching
# ---------------------------------------------------------------------------


class TestGetTemporalContextCaching:
    """Test that results are cached and the cache invalidates properly."""

    @pytest.mark.asyncio
    async def test_second_call_uses_cache(self):
        rows = [
            {"person_name": "Juan", "pattern_type": "wake_up", "median_minutes": 420},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result1 = await get_temporal_context()
            result2 = await get_temporal_context()

        assert result1 == result2
        assert "Juan's routine" in result1
        # DB should be queried exactly once; the second call hits cache
        pool.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_result_cached(self):
        """Even empty results should be cached (avoids repeated DB queries)."""
        pool = _make_pool(rows=[])

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result1 = await get_temporal_context()
            result2 = await get_temporal_context()

        assert result1 == ""
        assert result2 == ""
        pool.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_eviction_on_date_change(self):
        """Cache entries from a previous day should be evicted."""
        from atlas_brain.orchestration import temporal

        # Pre-populate cache with yesterday's entry
        yesterday = date(2026, 2, 15)  # Sunday
        yesterday_key = (yesterday, yesterday.weekday(), 5)
        temporal._cache[yesterday_key] = "stale data"

        rows = [
            {"person_name": "Juan", "pattern_type": "wake_up", "median_minutes": 420},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            today = date(2026, 2, 16)  # Monday
            mock_date.today.return_value = today
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            result = await temporal.get_temporal_context()

        # Yesterday's entry should have been evicted
        assert yesterday_key not in temporal._cache
        # Today's entry should be present
        today_key = (today, today.weekday(), 5)
        assert today_key in temporal._cache
        assert "Juan's routine" in result

    @pytest.mark.asyncio
    async def test_different_min_samples_separate_cache(self):
        """Different min_samples values produce separate cache entries."""
        rows = [
            {"person_name": "Juan", "pattern_type": "wake_up", "median_minutes": 420},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            await get_temporal_context(min_samples=5)
            await get_temporal_context(min_samples=10)

        # Two different cache keys means two DB queries
        assert pool.fetch.call_count == 2


# ---------------------------------------------------------------------------
# TestGetTemporalContext -- failure suppression
# ---------------------------------------------------------------------------


class TestGetTemporalContextFailure:
    """Test failure handling and suppression."""

    @pytest.mark.asyncio
    async def test_exception_returns_empty_string(self):
        pool = _make_pool()
        pool.fetch.side_effect = RuntimeError("connection refused")

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result = await get_temporal_context()

        assert result == ""

    @pytest.mark.asyncio
    async def test_exception_sets_failure_until(self):
        import time as _time

        from atlas_brain.orchestration import temporal

        pool = _make_pool()
        pool.fetch.side_effect = RuntimeError("connection refused")

        before = _time.monotonic()
        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            await temporal.get_temporal_context()

        # _failure_until should be ~60 seconds after the call
        assert temporal._failure_until >= before + 59
        assert temporal._failure_until <= before + 62

    @pytest.mark.asyncio
    async def test_failure_suppression_skips_db(self):
        """When _failure_until is in the future, DB is not queried."""
        import time as _time

        from atlas_brain.orchestration import temporal

        # Set failure suppression to far in the future
        temporal._failure_until = _time.monotonic() + 3600

        pool = _make_pool(rows=[
            {"person_name": "Juan", "pattern_type": "wake_up", "median_minutes": 420},
        ])

        with patch(_PATCH_DB_POOL, return_value=pool):
            result = await temporal.get_temporal_context()

        assert result == ""
        # DB pool should never be obtained or queried
        pool.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_suppression_expired(self):
        """When _failure_until is in the past, DB is queried normally."""
        from atlas_brain.orchestration import temporal

        # Set failure suppression to the past (already expired)
        temporal._failure_until = 0.0

        rows = [
            {"person_name": "Juan", "pattern_type": "wake_up", "median_minutes": 420},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            result = await temporal.get_temporal_context()

        assert "Juan's routine" in result
        pool.fetch.assert_called_once()


# ---------------------------------------------------------------------------
# TestGetTemporalContext -- day of week
# ---------------------------------------------------------------------------


class TestGetTemporalContextDayOfWeek:
    """Test that the correct day_of_week is passed to the DB query."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_date, expected_dow, expected_label",
        [
            (date(2026, 2, 16), 0, "Mon"),   # Monday
            (date(2026, 2, 17), 1, "Tue"),   # Tuesday
            (date(2026, 2, 18), 2, "Wed"),   # Wednesday
            (date(2026, 2, 19), 3, "Thu"),   # Thursday
            (date(2026, 2, 20), 4, "Fri"),   # Friday
            (date(2026, 2, 21), 5, "Sat"),   # Saturday
            (date(2026, 2, 22), 6, "Sun"),   # Sunday
        ],
    )
    async def test_day_of_week_passed_to_query(
        self, test_date, expected_dow, expected_label
    ):
        rows = [
            {"person_name": "Juan", "pattern_type": "wake_up", "median_minutes": 420},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = test_date
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result = await get_temporal_context(min_samples=3)

        # Verify the correct dow was passed as the first positional arg to fetch
        call_args = pool.fetch.call_args
        assert call_args[0][1] == expected_dow  # $1 = day_of_week
        assert call_args[0][2] == 3             # $2 = min_samples

        # Verify the day label appears in the output
        assert f"({expected_label})" in result


# ---------------------------------------------------------------------------
# TestGetTemporalContext -- min_samples parameter
# ---------------------------------------------------------------------------


class TestGetTemporalContextMinSamples:
    """Test that min_samples is correctly forwarded to the query."""

    @pytest.mark.asyncio
    async def test_default_min_samples(self):
        pool = _make_pool(rows=[])

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            await get_temporal_context()

        call_args = pool.fetch.call_args
        assert call_args[0][2] == 5  # default min_samples

    @pytest.mark.asyncio
    async def test_custom_min_samples(self):
        pool = _make_pool(rows=[])

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            await get_temporal_context(min_samples=20)

        call_args = pool.fetch.call_args
        assert call_args[0][2] == 20


# ---------------------------------------------------------------------------
# TestGetTemporalContext -- output formatting
# ---------------------------------------------------------------------------


class TestGetTemporalContextFormatting:
    """Test the exact shape of the formatted output."""

    @pytest.mark.asyncio
    async def test_multiple_people_newline_separated(self):
        rows = [
            {"person_name": "Alice", "pattern_type": "wake_up", "median_minutes": 360},
            {"person_name": "Alice", "pattern_type": "sleep", "median_minutes": 1320},
            {"person_name": "Bob", "pattern_type": "wake_up", "median_minutes": 480},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)  # Monday
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result = await get_temporal_context()

        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "Alice's routine (Mon): wake_up ~6:00 AM, sleep ~10:00 PM"
        assert lines[1] == "Bob's routine (Mon): wake_up ~8:00 AM"

    @pytest.mark.asyncio
    async def test_rounding_applied_in_output(self):
        """median_minutes are rounded to nearest 5 in the formatted output."""
        rows = [
            {"person_name": "Juan", "pattern_type": "commute", "median_minutes": 523},
        ]
        pool = _make_pool(rows=rows)

        with patch(_PATCH_DB_POOL, return_value=pool), patch(
            _PATCH_DATE,
        ) as mock_date:
            mock_date.today.return_value = date(2026, 2, 16)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            from atlas_brain.orchestration.temporal import get_temporal_context

            result = await get_temporal_context()

        # 523/5 = 104.6 -> round = 105 -> 105*5 = 525 -> 8:45 AM
        assert "commute ~8:45 AM" in result
