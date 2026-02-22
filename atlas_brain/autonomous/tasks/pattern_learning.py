"""
Temporal pattern learning task.

Nightly cron (2:00 AM) that computes arrival/departure/wake/sleep norms
from presence_events and sessions over the last 30 days, then upserts
statistics into the temporal_patterns table.
"""

import logging
import statistics
from typing import Any

from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.pattern_learning")


def _minutes_since_midnight(dt) -> int:
    """Extract minutes since midnight from a datetime."""
    return dt.hour * 60 + dt.minute


def _compute_stats(minutes_list: list[int]) -> dict[str, int]:
    """Compute median, stddev, min, max for a list of minute-of-day values."""
    med = int(statistics.median(minutes_list))
    std = int(statistics.stdev(minutes_list)) if len(minutes_list) >= 2 else 60
    # Floor stddev at 15 to prevent false positives on very consistent patterns
    std = max(std, 15)
    return {
        "median": med,
        "stddev": std,
        "earliest": min(minutes_list),
        "latest": max(minutes_list),
        "count": len(minutes_list),
    }


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Learn temporal patterns from presence and session data."""
    from ...storage.database import get_db_pool

    from ...config import settings

    metadata = task.metadata or {}
    lookback_days = metadata.get("lookback_days", settings.autonomous.pattern_learning_lookback_days)

    pool = get_db_pool()
    if not pool.is_initialized:
        return {
            "error": "Database not initialized",
            "_skip_synthesis": "Pattern learning skipped -- database not ready.",
        }

    # Verify temporal_patterns table exists (migration 025)
    try:
        await pool.fetchval("SELECT 1 FROM temporal_patterns LIMIT 0")
    except Exception:
        return {
            "error": "temporal_patterns table not ready (migration 025 pending)",
            "_skip_synthesis": "Pattern learning skipped -- migration pending.",
        }

    # --- Arrival/departure patterns from presence_events ---
    presence_rows = await pool.fetch(
        """
        SELECT person_name,
               transition,
               extract(isodow from created_at)::int - 1 AS dow,
               extract(hour from created_at)::int * 60
                   + extract(minute from created_at)::int AS minutes
        FROM presence_events
        WHERE created_at >= NOW() - ($1 || ' days')::interval
          AND person_name IS NOT NULL
        """,
        str(lookback_days),
    )

    # Group: (person_name, transition, dow) -> [minutes]
    presence_groups: dict[tuple[str, str, int], list[int]] = {}
    for row in presence_rows:
        key = (row["person_name"], row["transition"], row["dow"])
        presence_groups.setdefault(key, []).append(row["minutes"])

    # --- Wake/sleep patterns from sessions ---
    session_rows = await pool.fetch(
        """
        SELECT session_date,
               min(started_at) AS first_activity,
               max(last_activity_at) AS last_activity
        FROM sessions
        WHERE session_date >= (CURRENT_DATE - ($1 || ' days')::interval)
          AND started_at IS NOT NULL
        GROUP BY session_date
        """,
        str(lookback_days),
    )

    # Group: (dow) -> [wake_minutes] and [sleep_minutes]
    wake_groups: dict[int, list[int]] = {}
    sleep_groups: dict[int, list[int]] = {}
    for row in session_rows:
        # Compute day-of-week in Python (0=Mon, 6=Sun)
        dow = row["session_date"].weekday()
        if row["first_activity"]:
            wake_groups.setdefault(dow, []).append(
                _minutes_since_midnight(row["first_activity"])
            )
        if row["last_activity"]:
            sleep_groups.setdefault(dow, []).append(
                _minutes_since_midnight(row["last_activity"])
            )

    # --- Upsert all patterns ---
    upsert_count = 0
    persons = set()

    # Presence patterns (arrival / departure)
    for (person, transition, dow), minutes_list in presence_groups.items():
        if len(minutes_list) < 2:
            continue
        stats = _compute_stats(minutes_list)
        persons.add(person)

        await pool.execute(
            """
            INSERT INTO temporal_patterns
                (id, person_name, pattern_type, day_of_week,
                 median_minutes, stddev_minutes, sample_count,
                 earliest_minutes, latest_minutes, updated_at)
            VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (person_name, pattern_type, day_of_week)
            DO UPDATE SET
                median_minutes = $4,
                stddev_minutes = $5,
                sample_count = $6,
                earliest_minutes = $7,
                latest_minutes = $8,
                updated_at = NOW()
            """,
            person, transition, dow,
            stats["median"], stats["stddev"], stats["count"],
            stats["earliest"], stats["latest"],
        )
        upsert_count += 1

    # Wake/sleep patterns (attributed to "system" since sessions aren't per-person)
    system_person = "_system"
    for dow, minutes_list in wake_groups.items():
        if len(minutes_list) < 2:
            continue
        stats = _compute_stats(minutes_list)
        persons.add(system_person)

        await pool.execute(
            """
            INSERT INTO temporal_patterns
                (id, person_name, pattern_type, day_of_week,
                 median_minutes, stddev_minutes, sample_count,
                 earliest_minutes, latest_minutes, updated_at)
            VALUES (gen_random_uuid(), $1, 'wake', $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (person_name, pattern_type, day_of_week)
            DO UPDATE SET
                median_minutes = $3,
                stddev_minutes = $4,
                sample_count = $5,
                earliest_minutes = $6,
                latest_minutes = $7,
                updated_at = NOW()
            """,
            system_person, dow,
            stats["median"], stats["stddev"], stats["count"],
            stats["earliest"], stats["latest"],
        )
        upsert_count += 1

    for dow, minutes_list in sleep_groups.items():
        if len(minutes_list) < 2:
            continue
        stats = _compute_stats(minutes_list)

        await pool.execute(
            """
            INSERT INTO temporal_patterns
                (id, person_name, pattern_type, day_of_week,
                 median_minutes, stddev_minutes, sample_count,
                 earliest_minutes, latest_minutes, updated_at)
            VALUES (gen_random_uuid(), $1, 'sleep', $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (person_name, pattern_type, day_of_week)
            DO UPDATE SET
                median_minutes = $3,
                stddev_minutes = $4,
                sample_count = $5,
                earliest_minutes = $6,
                latest_minutes = $7,
                updated_at = NOW()
            """,
            system_person, dow,
            stats["median"], stats["stddev"], stats["count"],
            stats["earliest"], stats["latest"],
        )
        upsert_count += 1

    logger.info("Pattern learning: %d persons, %d patterns upserted", len(persons), upsert_count)
    return {"persons": len(persons), "patterns_upserted": upsert_count}
