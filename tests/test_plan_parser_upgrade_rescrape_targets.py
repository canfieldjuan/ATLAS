import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path


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
    _is_runnable_target,
    _is_in_recent_cooldown,
    _parse_csv,
    _recent_scrape_penalty,
    _should_fallback_to_direct,
    _source_rank_weight,
    _target_priority_score,
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
