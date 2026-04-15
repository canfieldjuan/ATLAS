import sys
import types
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


if "asyncpg" not in sys.modules:
    asyncpg_module = types.ModuleType("asyncpg")
    asyncpg_module.connect = object
    asyncpg_module.Connection = object
    asyncpg_module.Record = dict
    sys.modules["asyncpg"] = asyncpg_module

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from plan_parser_upgrade_rescrape_targets import (  # noqa: E402
    _build_parser,
    _campaign_metadata,
    _collect_candidates,
    _clear_blocked_parser_upgrade_deferred,
    _effective_run_overrides,
    _has_stable_pagination_signal,
    _is_blocked_parser_upgrade_deferred,
    _is_blocked_target,
    _is_runnable_target,
    _is_in_recent_cooldown,
    _mark_blocked_parser_upgrade_deferred,
    _parse_csv,
    _recent_scrape_penalty,
    _should_fallback_to_direct,
    _source_rank_weight,
    _target_priority_score,
    _run_drain,
)


def test_parse_csv_normalizes_values():
    assert _parse_csv(" G2, Gartner ,,software_advice ") == ["g2", "gartner", "software_advice"]


def test_source_rank_weight_biases_identity_sources():
    assert _source_rank_weight("trustradius") > _source_rank_weight("g2")
    assert _source_rank_weight("g2") > _source_rank_weight("software_advice")
    assert _source_rank_weight("gartner") > _source_rank_weight("software_advice")
    assert _source_rank_weight("g2") > _source_rank_weight("trustpilot")


def test_target_priority_score_prefers_outdated_then_missing_then_source():
    high_value_source = {
        "source": "g2",
        "outdated_reviews": 5,
        "missing_parser_version_reviews": 20,
        "vendor_name": "Slack",
        "target_id": "a",
    }
    lower_value_source = {
        "source": "software_advice",
        "outdated_reviews": 20,
        "missing_parser_version_reviews": 5,
        "vendor_name": "Zendesk",
        "target_id": "b",
    }
    assert _target_priority_score(high_value_source) > _target_priority_score(lower_value_source)


def test_recent_scrape_penalty_applies_to_fresh_targets():
    fresh = datetime.now(timezone.utc) - timedelta(minutes=15)
    stale = datetime.now(timezone.utc) - timedelta(hours=14)

    assert _recent_scrape_penalty(fresh) == -1
    assert _recent_scrape_penalty(stale) == 0


def test_is_in_recent_cooldown_filters_fresh_targets():
    fresh = datetime.now(timezone.utc) - timedelta(minutes=15)
    stale = datetime.now(timezone.utc) - timedelta(hours=14)

    assert _is_in_recent_cooldown(fresh) is True
    assert _is_in_recent_cooldown(stale) is False


def test_campaign_metadata_records_rescrape_request():
    metadata = _campaign_metadata(
        {"existing": True},
        current_version="g2:3",
        outdated_reviews=11,
        missing_reviews=7,
        requested_at=__import__("datetime").datetime(2026, 4, 14, tzinfo=__import__("datetime").timezone.utc),
    )

    assert metadata["existing"] is True
    assert metadata["parser_upgrade_rescrape"]["requested"] is True
    assert metadata["parser_upgrade_rescrape"]["current_parser_version"] == "g2:3"
    assert metadata["parser_upgrade_rescrape"]["outdated_reviews"] == 11
    assert metadata["parser_upgrade_rescrape"]["missing_parser_version_reviews"] == 7


def test_is_runnable_target_skips_disabled_and_blocked_by_default():
    assert _is_runnable_target({"enabled": False, "last_scrape_status": "success"}, include_blocked=False) is False
    assert _is_runnable_target({"enabled": True, "last_scrape_status": "blocked"}, include_blocked=False) is False
    assert _is_runnable_target({"enabled": True, "last_scrape_status": "blocked"}, include_blocked=True) is True
    assert _is_runnable_target({"enabled": True, "last_scrape_status": "success"}, include_blocked=False) is True


def test_is_blocked_target_recognizes_blocked_status():
    assert _is_blocked_target({"last_scrape_status": "blocked"}) is True
    assert _is_blocked_target({"last_scrape_status": "success"}) is False


def test_blocked_parser_upgrade_defer_round_trip():
    deferred_at = datetime.now(timezone.utc)
    metadata = _mark_blocked_parser_upgrade_deferred(
        {"existing": True},
        deferred_at=deferred_at,
        reason="last_scrape_status_blocked",
    )

    assert _is_blocked_parser_upgrade_deferred(metadata, cooldown_hours=168) is True
    assert metadata["parser_upgrade_rescrape"]["blocked_deferred_reason"] == "last_scrape_status_blocked"

    cleared = _clear_blocked_parser_upgrade_deferred(metadata)
    assert _is_blocked_parser_upgrade_deferred(cleared, cooldown_hours=168) is False


@pytest.mark.asyncio
async def test_collect_candidates_ignores_disabled_targets(monkeypatch):
    class FakeConn:
        async def fetch(self, *args, **kwargs):
            return [
                {
                    "target_id": "enabled-target",
                    "source": "capterra",
                    "vendor_name": "Freshdesk",
                    "product_name": "Freshdesk",
                    "product_slug": "freshdesk",
                    "product_category": "Help Desk",
                    "enabled": True,
                    "priority": 90,
                    "max_pages": 30,
                    "scrape_mode": "exhaustive",
                    "last_scraped_at": None,
                    "last_scrape_status": "success",
                    "metadata": {},
                    "outdated_reviews": 0,
                    "missing_parser_version_reviews": 4,
                    "processed_reviews": 4,
                },
                {
                    "target_id": "disabled-target",
                    "source": "capterra",
                    "vendor_name": "Slack",
                    "product_name": "Slack",
                    "product_slug": "slack",
                    "product_category": "Communication",
                    "enabled": False,
                    "priority": 90,
                    "max_pages": 30,
                    "scrape_mode": "exhaustive",
                    "last_scraped_at": None,
                    "last_scrape_status": "blocked",
                    "metadata": {},
                    "outdated_reviews": 0,
                    "missing_parser_version_reviews": 11,
                    "processed_reviews": 11,
                },
            ]

    class FakeParser:
        version = "capterra:2"

    monkeypatch.setattr(
        "plan_parser_upgrade_rescrape_targets.get_all_parsers",
        lambda: {"capterra": FakeParser()},
    )

    rows = await _collect_candidates(FakeConn(), sources=["capterra"])

    assert len(rows) == 1
    assert rows[0]["target_id"] == "enabled-target"


@pytest.mark.asyncio
async def test_collect_candidates_skips_deferred_sources(monkeypatch):
    class FakeConn:
        async def fetch(self, *args, **kwargs):
            return [
                {
                    "target_id": "g2-target",
                    "source": "g2",
                    "vendor_name": "Slack",
                    "product_name": "Slack",
                    "product_slug": "slack",
                    "product_category": "Communication",
                    "enabled": True,
                    "priority": 90,
                    "max_pages": 50,
                    "scrape_mode": "exhaustive",
                    "last_scraped_at": None,
                    "last_scrape_status": "success",
                    "metadata": {},
                    "outdated_reviews": 10,
                    "missing_parser_version_reviews": 0,
                    "processed_reviews": 10,
                }
            ]

    class FakeParser:
        version = "g2:3"

    monkeypatch.setattr(
        "plan_parser_upgrade_rescrape_targets.get_all_parsers",
        lambda: {"g2": FakeParser()},
    )
    monkeypatch.setattr(
        "plan_parser_upgrade_rescrape_targets.settings",
        MagicMock(b2b_scrape=MagicMock(parser_upgrade_deferred_sources="g2")),
    )

    rows = await _collect_candidates(FakeConn(), sources=["g2"])

    assert rows == []


def test_should_fallback_to_direct_for_local_api_connection_refusal():
    result = {
        "status": "failed",
        "error": {"detail": "Connection refused"},
    }

    assert _should_fallback_to_direct(result) is True


def test_should_fallback_to_direct_for_stale_governance_failure():
    result = {
        "status": "failed",
        "http_status": 400,
        "error": {"detail": "Source 'capterra' is currently disabled by B2B scrape source governance"},
    }

    assert _should_fallback_to_direct(result) is True


def test_should_not_fallback_for_normal_success():
    result = {
        "status": "completed",
        "result": {"status": "success"},
    }

    assert _should_fallback_to_direct(result) is False


def test_build_parser_accepts_direct_run_overrides():
    parser = _build_parser()
    args = parser.parse_args([
        "--run-now",
        "--run-now-mode", "direct",
        "--run-max-pages", "3",
        "--run-scrape-mode", "incremental",
    ])

    assert args.run_now is True
    assert args.run_now_mode == "direct"
    assert args.run_max_pages == 3
    assert args.run_scrape_mode == "incremental"
    assert args.recent_cooldown_hours == 12


def test_build_parser_accepts_deep_run_overrides():
    parser = _build_parser()
    args = parser.parse_args([
        "--deep-sources", "trustradius,capterra",
        "--deep-min-parser-backlog-reviews", "24",
        "--deep-run-max-pages", "8",
        "--deep-min-stable-pages-scraped", "3",
        "--deep-max-targets-per-batch", "2",
    ])

    assert args.deep_sources == "trustradius,capterra"
    assert args.deep_min_parser_backlog_reviews == 24
    assert args.deep_run_max_pages == 8
    assert args.deep_min_stable_pages_scraped == 3
    assert args.deep_max_targets_per_batch == 2


def test_has_stable_pagination_signal_requires_clean_recent_page_walk():
    stable = {
        "last_scrape_status": "success",
        "last_scrape_pages_scraped": 3,
        "last_scrape_reviews_found": 21,
        "last_scrape_stop_reason": "page_cap",
    }
    unstable = {
        "last_scrape_status": "failed",
        "last_scrape_pages_scraped": 0,
        "last_scrape_reviews_found": 0,
        "last_scrape_stop_reason": None,
    }

    assert _has_stable_pagination_signal(stable, min_pages_scraped=3, baseline_run_max_pages=3) is True
    assert _has_stable_pagination_signal(unstable, min_pages_scraped=3, baseline_run_max_pages=3) is False


def test_effective_run_overrides_only_deepens_qualified_targets():
    args = argparse.Namespace(
        run_max_pages=3,
        run_scrape_mode="exhaustive",
        deep_sources="trustradius,capterra",
        deep_min_parser_backlog_reviews=20,
        deep_run_max_pages=8,
        deep_min_stable_pages_scraped=3,
        deep_max_targets_per_batch=2,
    )
    qualified = {
        "source": "trustradius",
        "outdated_reviews": 12,
        "missing_parser_version_reviews": 10,
        "last_scrape_status": "success",
        "last_scrape_pages_scraped": 3,
        "last_scrape_reviews_found": 25,
        "last_scrape_stop_reason": "page_cap",
    }
    unqualified = {
        "source": "gartner",
        "outdated_reviews": 15,
        "missing_parser_version_reviews": 10,
        "last_scrape_status": "failed",
        "last_scrape_pages_scraped": 0,
        "last_scrape_reviews_found": 0,
        "last_scrape_stop_reason": None,
    }

    max_pages, scrape_mode, deep = _effective_run_overrides(qualified, args, deep_targets_used=0)
    assert max_pages == 8
    assert scrape_mode == "exhaustive"
    assert deep is True

    max_pages, scrape_mode, deep = _effective_run_overrides(unqualified, args, deep_targets_used=0)
    assert max_pages == 3
    assert scrape_mode == "exhaustive"
    assert deep is False

    max_pages, scrape_mode, deep = _effective_run_overrides(qualified, args, deep_targets_used=2)
    assert max_pages == 3
    assert scrape_mode == "exhaustive"
    assert deep is False


def test_build_parser_accepts_drain_mode():
    parser = _build_parser()
    args = parser.parse_args([
        "--run-now",
        "--drain",
        "--drain-max-batches", "9",
    ])

    assert args.drain is True
    assert args.drain_max_batches == 9


@pytest.mark.asyncio
async def test_run_drain_stops_when_queue_is_empty(monkeypatch):
    calls = []
    responses = [
        {
            "sources": ["software_advice"],
            "requested_targets": 5,
            "deferred_blocked_targets": 0,
            "applied": 0,
            "run_started": 5,
            "targets": [{"vendor_name": "Zoom"}],
            "run_results": [{"vendor_name": "Zoom"}],
        },
        {
            "sources": ["software_advice"],
            "requested_targets": 0,
            "deferred_blocked_targets": 0,
            "applied": 0,
            "run_started": 0,
            "targets": [],
            "run_results": [],
        },
        {
            "sources": ["software_advice"],
            "requested_targets": 0,
            "deferred_blocked_targets": 0,
            "applied": 0,
            "run_started": 0,
            "targets": [],
            "run_results": [],
        },
    ]

    async def fake_run(args):
        calls.append({"run_now": args.run_now, "apply": args.apply, "drain": args.drain})
        return responses[len(calls) - 1]

    monkeypatch.setattr("plan_parser_upgrade_rescrape_targets._run", fake_run)

    args = argparse.Namespace(
        run_now=True,
        apply=False,
        drain=True,
        drain_max_batches=10,
    )
    result = await _run_drain(args)

    assert result["drain"] is True
    assert result["batches_run"] == 2
    assert result["requested_targets"] == 0
    assert len(result["batches"]) == 2
    assert calls[-1] == {"run_now": False, "apply": False, "drain": False}


@pytest.mark.asyncio
async def test_run_drain_stops_when_nothing_runnable_started(monkeypatch):
    calls = []
    responses = [
        {
            "sources": ["software_advice"],
            "requested_targets": 4,
            "deferred_blocked_targets": 0,
            "applied": 0,
            "run_started": 0,
            "targets": [{"vendor_name": "Blocked Vendor"}],
            "run_results": [],
        },
        {
            "sources": ["software_advice"],
            "requested_targets": 4,
            "deferred_blocked_targets": 0,
            "applied": 0,
            "run_started": 0,
            "targets": [{"vendor_name": "Blocked Vendor"}],
            "run_results": [],
        },
    ]

    async def fake_run(args):
        calls.append({"run_now": args.run_now, "apply": args.apply, "drain": args.drain})
        return responses[len(calls) - 1]

    monkeypatch.setattr("plan_parser_upgrade_rescrape_targets._run", fake_run)

    args = argparse.Namespace(
        run_now=True,
        apply=False,
        drain=True,
        drain_max_batches=10,
    )
    result = await _run_drain(args)

    assert result["batches_run"] == 1
    assert result["requested_targets"] == 4
    assert calls[-1] == {"run_now": False, "apply": False, "drain": False}
