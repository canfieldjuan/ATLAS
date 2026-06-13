import pytest
from pathlib import Path


class FakeMigrationPool:
    def __init__(self, records=None):
        self.records = list(records or [])
        self.inserted = []

    async def fetchval(self, query, *args):
        normalized = " ".join(query.split())
        if "WHERE name = $1" in normalized:
            name = args[0]
            for version, record_name in self.records:
                if record_name == name:
                    return version
            return None
        if "WHERE version = $1" in normalized:
            version = args[0]
            for record_version, name in self.records:
                if record_version == version:
                    return name
            return None
        if "MIN(version)" in normalized:
            if not self.records:
                return -1
            min_version = min(version for version, _ in self.records)
            return min_version - 1 if min_version < 0 else -1
        raise AssertionError(f"Unexpected fetchval query: {query}")

    async def execute(self, query, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("INSERT INTO schema_migrations"):
            record = (args[0], args[1])
            self.records.append(record)
            self.inserted.append(record)
            return
        raise AssertionError(f"Unexpected execute query: {query}")


@pytest.mark.asyncio
async def test_record_migration_uses_prefix_version_when_available():
    from atlas_brain.storage.migrations import _record_migration

    pool = FakeMigrationPool(records=[(1, "001_initial_schema")])

    await _record_migration(pool, "247_b2b_vendor_witness_packets.sql")

    assert pool.inserted == [(247, "247_b2b_vendor_witness_packets")]


@pytest.mark.asyncio
async def test_record_migration_uses_negative_version_on_prefix_collision():
    from atlas_brain.storage.migrations import _record_migration

    pool = FakeMigrationPool(records=[
        (76, "076_saas_accounts"),
        (230, "230_scrape_target_checkpoints"),
    ])

    await _record_migration(pool, "076_consumer_analytics_views.sql")
    await _record_migration(pool, "230_b2b_reasoning_synthesis.sql")

    assert pool.inserted == [
        (-1, "076_consumer_analytics_views"),
        (-2, "230_b2b_reasoning_synthesis"),
    ]


def test_find_duplicate_migration_prefixes_detects_repo_collisions():
    from atlas_brain.storage.migrations import _find_duplicate_migration_prefixes

    duplicates = _find_duplicate_migration_prefixes([
        Path("270_b2b_watchlist_views.sql"),
        Path("271_b2b_watchlist_view_alert_thresholds.sql"),
        Path("272_b2b_opportunity_dispositions.sql"),
        Path("272_b2b_watchlist_alert_events.sql"),
    ])

    assert duplicates == {
        272: [
            "272_b2b_opportunity_dispositions.sql",
            "272_b2b_watchlist_alert_events.sql",
        ]
    }


def test_repo_migration_prefix_collisions_are_only_historical_exceptions():
    from atlas_brain.storage.migrations import MIGRATIONS_DIR, _find_duplicate_migration_prefixes

    duplicates = _find_duplicate_migration_prefixes(sorted(MIGRATIONS_DIR.glob("*.sql")))

    # Accepted historical prefix collisions: each pair already shipped on main
    # with the same numeric prefix (parallel-session migrations that landed
    # together). The allowlist documents them so this test still catches a NEW
    # collision; 281/282/283/298 were added after the allowlist was last updated.
    assert duplicates == {
        76: [
            "076_consumer_analytics_views.sql",
            "076_saas_accounts.sql",
        ],
        230: [
            "230_b2b_reasoning_synthesis.sql",
            "230_scrape_target_checkpoints.sql",
        ],
        281: [
            "281_b2b_report_subscription_filter_payload.sql",
            "281_b2b_watchlist_alert_reopen_count.sql",
        ],
        282: [
            "282_b2b_enrichment_stage_runs.sql",
            "282_b2b_materialization_run_id.sql",
        ],
        283: [
            "283_b2b_enrichment_stage_runs_work_fingerprint.sql",
            "283_b2b_report_subscription_delivery_blocked_freshness.sql",
        ],
        298: [
            "298_b2b_review_vendor_mentions_drop_duplicate_rows.sql",
            "298_b2b_watchlist_preview_alert_policy.sql",
        ],
    }
