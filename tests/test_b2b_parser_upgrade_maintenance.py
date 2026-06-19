from unittest.mock import AsyncMock, MagicMock

import pytest

from tests._module_stub import stub_missing_module

# Stub asyncpg only when it is genuinely not importable, so CI (where asyncpg
# is installed) imports the real module and no sibling test is poisoned. The
# production task module is imported inline in each test below; this top-level
# stub guarantees asyncpg is resolvable before those imports run.
stub_missing_module(
    "asyncpg",
    attributes={"connect": object, "Connection": object, "Record": dict},
)


@pytest.mark.asyncio
async def test_parser_upgrade_maintenance_skips_when_disabled(monkeypatch):
    import atlas_brain.autonomous.tasks.b2b_parser_upgrade_maintenance as mod

    monkeypatch.setattr(
        mod,
        "settings",
        MagicMock(b2b_scrape=MagicMock(parser_upgrade_maintenance_enabled=False)),
    )

    result = await mod.run(MagicMock())

    assert result == {"_skip_synthesis": "Parser-upgrade maintenance disabled"}


@pytest.mark.asyncio
async def test_parser_upgrade_maintenance_skips_when_db_not_ready(monkeypatch):
    import atlas_brain.autonomous.tasks.b2b_parser_upgrade_maintenance as mod

    cfg = MagicMock(
        parser_upgrade_maintenance_enabled=True,
    )
    monkeypatch.setattr(mod, "settings", MagicMock(b2b_scrape=cfg))
    monkeypatch.setattr(mod, "get_db_pool", lambda: MagicMock(is_initialized=False))

    result = await mod.run(MagicMock())

    assert result == {"_skip_synthesis": "DB not ready"}


@pytest.mark.asyncio
async def test_parser_upgrade_maintenance_runs_drain_with_configured_args(monkeypatch):
    import atlas_brain.autonomous.tasks.b2b_parser_upgrade_maintenance as mod

    cfg = MagicMock(
        parser_upgrade_maintenance_enabled=True,
        parser_upgrade_maintenance_sources="trustradius,gartner",
        parser_upgrade_maintenance_limit_targets=7,
        parser_upgrade_maintenance_run_max_pages=4,
        parser_upgrade_maintenance_run_scrape_mode="exhaustive",
        parser_upgrade_maintenance_deep_sources="trustradius,capterra",
        parser_upgrade_maintenance_deep_min_parser_backlog_reviews=24,
        parser_upgrade_maintenance_deep_run_max_pages=8,
        parser_upgrade_maintenance_deep_min_stable_pages_scraped=3,
        parser_upgrade_maintenance_deep_max_targets_per_batch=2,
        parser_upgrade_maintenance_recent_cooldown_hours=18,
        parser_upgrade_maintenance_drain_max_batches=9,
    )
    monkeypatch.setattr(mod, "settings", MagicMock(b2b_scrape=cfg))
    monkeypatch.setattr(mod, "get_db_pool", lambda: MagicMock(is_initialized=True))

    captured = {}

    async def _fake_run_drain(args):
        captured["args"] = args
        return {
            "sources": ["trustradius", "gartner"],
            "batches_run": 2,
            "requested_targets": 0,
            "deferred_blocked_targets": 0,
            "run_started": 6,
        }

    monkeypatch.setattr(mod.parser_upgrade_runner, "_run_drain", _fake_run_drain)

    result = await mod.run(MagicMock())

    assert captured["args"].sources == "trustradius,gartner"
    assert captured["args"].limit_targets == 7
    assert captured["args"].run_now is True
    assert captured["args"].run_now_mode == "direct"
    assert captured["args"].run_max_pages == 4
    assert captured["args"].run_scrape_mode == "exhaustive"
    assert captured["args"].deep_sources == "trustradius,capterra"
    assert captured["args"].deep_min_parser_backlog_reviews == 24
    assert captured["args"].deep_run_max_pages == 8
    assert captured["args"].deep_min_stable_pages_scraped == 3
    assert captured["args"].deep_max_targets_per_batch == 2
    assert captured["args"].recent_cooldown_hours == 18
    assert captured["args"].drain is True
    assert captured["args"].drain_max_batches == 9
    assert result["_skip_synthesis"] is True
    assert result["requested_targets"] == 0
