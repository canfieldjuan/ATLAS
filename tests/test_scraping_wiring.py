"""Tests for scraper wiring outside parser-local behavior."""

import sys
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


def test_rate_limiter_includes_x_domain_from_config():
    from atlas_brain.services.scraping.rate_limiter import DomainRateLimiter

    cfg = MagicMock()
    cfg.g2_rpm = 6
    cfg.capterra_rpm = 8
    cfg.trustradius_rpm = 10
    cfg.reddit_rpm = 30
    cfg.hackernews_rpm = 100
    cfg.github_rpm = 25
    cfg.rss_rpm = 10
    cfg.gartner_rpm = 4
    cfg.trustpilot_rpm = 6
    cfg.getapp_rpm = 8
    cfg.twitter_rpm = 10
    cfg.producthunt_rpm = 20
    cfg.youtube_rpm = 50
    cfg.quora_rpm = 4
    cfg.stackoverflow_rpm = 25
    cfg.peerspot_rpm = 4
    cfg.software_advice_rpm = 8
    cfg.sourceforge_rpm = 12
    cfg.slashdot_rpm = 8

    limiter = DomainRateLimiter.from_config(cfg)
    assert limiter._rpm_map["x.com"] == 10
    assert limiter._rpm_map["sourceforge.net"] == 12
    assert limiter._rpm_map["slashdot.org"] == 8


def test_structured_sources_extend_verified_review_platforms_without_claiming_verification():
    from atlas_brain.services.scraping.sources import ReviewSource, STRUCTURED_SOURCES
    from atlas_brain.services.scraping.sources import VERIFIED_SOURCES

    assert VERIFIED_SOURCES < STRUCTURED_SOURCES
    assert ReviewSource.SOFTWARE_ADVICE in STRUCTURED_SOURCES
    assert ReviewSource.SOURCEFORGE in STRUCTURED_SOURCES
    assert ReviewSource.SLASHDOT in STRUCTURED_SOURCES
    assert ReviewSource.SOURCEFORGE not in VERIFIED_SOURCES
    assert ReviewSource.SLASHDOT not in VERIFIED_SOURCES


def test_default_source_allowlist_includes_developer_sources_and_excludes_noise():
    from atlas_brain.services.scraping.sources import is_source_allowed

    allowlist = (
        "g2,capterra,trustradius,gartner,peerspot,"
        "getapp,software_advice,trustpilot,reddit,hackernews,github,stackoverflow"
    )

    assert is_source_allowed("getapp", allowlist)
    assert is_source_allowed("github", allowlist)
    assert is_source_allowed("stackoverflow", allowlist)
    assert not is_source_allowed("quora", allowlist)
    assert not is_source_allowed("twitter", allowlist)
    assert not is_source_allowed("sourceforge", allowlist)
    assert is_source_allowed("software_advice", allowlist)


def test_capabilities_match_getapp_twitter_sourceforge_and_slashdot_profiles():
    from atlas_brain.services.scraping.capabilities import AccessPattern, ProxyClass, get_capability

    getapp = get_capability("getapp")
    twitter = get_capability("twitter")
    sourceforge = get_capability("sourceforge")
    slashdot = get_capability("slashdot")

    assert getapp is not None
    assert twitter is not None
    assert sourceforge is not None
    assert slashdot is not None
    assert AccessPattern.js_rendered in getapp.access_patterns
    assert getapp.fallback_chain == ("web_unlocker", "js_rendered", "html_scrape")
    assert twitter.fallback_chain == ("js_rendered", "html_scrape")
    assert sourceforge.fallback_chain == ("html_scrape",)
    assert slashdot.fallback_chain == ("html_scrape",)
    assert slashdot.proxy_class == ProxyClass.residential


def test_builtin_task_registry_includes_scrape_target_pruning():
    from atlas_brain.autonomous.tasks import _BUILTIN_TASKS

    assert ("b2b_scrape_target_pruning", "run", "b2b_scrape_target_pruning") in _BUILTIN_TASKS


def test_source_fit_policy_keeps_crm_github_conditional_and_cloud_github_core():
    from atlas_brain.services.scraping.source_fit import classify_source_fit, is_source_fit_allowed

    crm = classify_source_fit("github", "CRM")
    cloud = classify_source_fit("github", "Cloud Infrastructure")

    assert crm.fit == "conditional"
    assert crm.vertical == "crm_support_marketing"
    assert is_source_fit_allowed("github", "CRM") is True
    assert cloud.fit == "core"
    assert cloud.vertical == "cloud_devops_security"
    assert is_source_fit_allowed("github", "Cloud Infrastructure") is True


def test_source_fit_policy_treats_communication_stackoverflow_as_core():
    from atlas_brain.services.scraping.source_fit import classify_source_fit, is_source_fit_allowed

    comm_so = classify_source_fit("stackoverflow", "Communication")
    comm_gh = classify_source_fit("github", "Communication")
    comm_sf = classify_source_fit("sourceforge", "Communication")

    assert comm_so.fit == "core"
    assert comm_so.vertical == "communication"
    assert is_source_fit_allowed("stackoverflow", "Communication") is True
    assert comm_gh.fit == "core"
    assert comm_gh.vertical == "communication"
    assert is_source_fit_allowed("github", "Communication") is True
    assert comm_sf.fit == "avoid"
    assert is_source_fit_allowed("sourceforge", "Communication") is False


def test_source_fit_policy_keeps_project_stackoverflow_conditional():
    from atlas_brain.services.scraping.source_fit import classify_source_fit, is_source_fit_allowed

    project_so = classify_source_fit("stackoverflow", "Project Management")
    project_gh = classify_source_fit("github", "Project Management")
    project_sf = classify_source_fit("sourceforge", "Project Management")

    assert project_so.fit == "conditional"
    assert project_so.vertical == "project_collaboration"
    assert is_source_fit_allowed("stackoverflow", "Project Management") is True
    assert project_gh.fit == "conditional"
    assert project_gh.vertical == "project_collaboration"
    assert is_source_fit_allowed("github", "Project Management") is True
    assert project_sf.fit == "avoid"
    assert is_source_fit_allowed("sourceforge", "Project Management") is False


@pytest.mark.asyncio
async def test_list_targets_normalizes_metadata_and_exposes_scrape_state():

    from atlas_brain.api.b2b_scrape import list_targets

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        return_value=[
            {
                "id": "target-1",
                "source": "reddit",
                "vendor_name": "Asana",
                "product_name": "Asana",
                "product_slug": "asana-reddit",
                "product_category": "Project Management",
                "max_pages": 15,
                "enabled": True,
                "priority": 15,
                "scrape_mode": "incremental",
                "last_scraped_at": "2026-03-19T23:01:14+00:00",
                "last_scrape_status": "success",
                "last_scrape_reviews": 0,
                "last_scrape_runtime_mode": "incremental",
                "last_scrape_stop_reason": "page_cap",
                "last_scrape_oldest_review": "2026-02-22",
                "last_scrape_newest_review": "2026-03-19",
                "last_scrape_date_cutoff": "2026-03-19",
                "last_scrape_pages_scraped": 29,
                "last_scrape_reviews_found": 66,
                "last_scrape_reviews_filtered": 56,
                "last_scrape_date_dropped": 0,
                "last_scrape_duration_ms": 23491,
                "last_scrape_resume_page": None,
                "scrape_interval_hours": 168,
                "metadata": "{\"subreddits\": \"asana,projectmanagement\", \"scrape_mode\": \"initial\", \"scrape_state\": {\"newest_review\": \"2026-03-10\"}}",
                "created_at": "2026-03-03T05:54:29+00:00",
                "updated_at": "2026-03-19T23:01:14+00:00",
            }
        ]
    )

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        rows = await list_targets(source=None, enabled_only=True)

    row = rows[0]
    assert row["metadata"]["subreddits"] == "asana,projectmanagement"
    assert "scrape_state" not in row["metadata"]
    assert "scrape_mode" not in row["metadata"]
    assert row["scrape_state"]["newest_review"] == "2026-03-19"
    assert row["scrape_state"]["stop_reason"] == "page_cap"
    assert row["last_scrape_pages_scraped"] == 29


def test_scrape_intake_filters_poor_source_fit_targets():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _filter_targets_by_source_fit

    cfg = MagicMock(source_fit_filter_enabled=True, source_fit_allow_conditional=True)
    kept, skipped = _filter_targets_by_source_fit(
        [
            {
                "source": "github",
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "metadata": {},
            },
            {
                "source": "g2",
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "metadata": {},
            },
            {
                "source": "github",
                "vendor_name": "Datadog",
                "product_category": "Cloud Infrastructure",
                "metadata": {},
            },
        ],
        cfg,
    )

    assert [row["source"] for row in kept] == ["github", "g2", "github"]
    assert kept[0]["_source_fit"] == "conditional"
    assert kept[1]["_source_fit"] == "core"
    assert kept[2]["_source_fit"] == "core"
    assert skipped == []


def test_scrape_intake_allows_conditional_probation_target():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _filter_targets_by_source_fit

    cfg = MagicMock(
        source_fit_filter_enabled=True,
        source_fit_allow_conditional=False,
        source_fit_allow_probation=True,
    )
    kept, skipped = _filter_targets_by_source_fit(
        [
            {
                "source": "twitter",
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "metadata": {"source_fit_probation": True},
            }
        ],
        cfg,
    )

    assert len(kept) == 1
    assert kept[0]["_source_fit"] == "probation"
    assert skipped == []


def test_merge_scrape_raw_metadata_adds_target_provenance():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _merge_scrape_raw_metadata

    merged = _merge_scrape_raw_metadata(
        {"source_weight": 0.7},
        {
            "scrape_target_id": "target-1",
            "scrape_target_source_fit": "probation",
            "scrape_target_vertical": "crm_support_marketing",
            "scrape_target_probation": True,
        },
    )

    assert merged["source_weight"] == 0.7
    assert merged["scrape_target_id"] == "target-1"
    assert merged["scrape_target_source_fit"] == "probation"
    assert merged["scrape_target_vertical"] == "crm_support_marketing"
    assert merged["scrape_target_probation"] is True


def test_coverage_planner_prefers_specific_category_over_b2b_software():
    from atlas_brain.services.scraping.target_planning import collapse_inventory_rows

    rows = [
        {
            "vendor_name": "Amazon Web Services",
            "product_category": "B2B Software",
            "total_reviews_analyzed": 700,
            "confidence_score": 0.9,
        },
        {
            "vendor_name": "Amazon Web Services",
            "product_category": "Cloud Infrastructure",
            "total_reviews_analyzed": 500,
            "confidence_score": 0.8,
        },
    ]

    collapsed = collapse_inventory_rows(rows)
    assert collapsed == [
        {
            "vendor_name": "Amazon Web Services",
            "product_category": "Cloud Infrastructure",
            "total_reviews_analyzed": 500,
            "confidence_score": 0.8,
            "last_computed_at": None,
            "inventory_source": "b2b_product_profiles",
        }
    ]


@pytest.mark.asyncio
async def test_source_yield_prune_dry_run_selects_candidates():
    from atlas_brain.services.scraping.source_yield import prune_low_yield_targets

    target_id = uuid4()
    pool = AsyncMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "target_id": target_id,
                "source": "twitter",
                "vendor_name": "BigCommerce",
                "product_slug": "bigcommerce",
                "product_category": "Ecommerce",
                "enabled": True,
                "metadata": {"region": "us"},
                "runs_observed": 3,
                "inserted_sum": 0,
                "last_run_at": None,
                "statuses": ["partial", "success", "success"],
            }
        ]
    )
    pool.execute = AsyncMock(return_value="UPDATE 1")

    result = await prune_low_yield_targets(
        pool,
        source="twitter",
        lookback_runs=3,
        min_runs=2,
        max_inserted_total=0,
        max_disable_per_run=10,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["requested"] == 1
    assert result["disabled"] == 1
    assert result["targets"][0]["target_id"] == str(target_id)
    pool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_source_yield_prune_apply_disables_selected_targets():
    from atlas_brain.services.scraping.source_yield import prune_low_yield_targets

    pool = AsyncMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "target_id": uuid4(),
                "source": "twitter",
                "vendor_name": "Tableau",
                "product_slug": "tableau",
                "product_category": "Analytics",
                "enabled": True,
                "metadata": {},
                "runs_observed": 3,
                "inserted_sum": 0,
                "last_run_at": None,
                "statuses": ["partial", "partial", "partial"],
            }
        ]
    )
    pool.execute = AsyncMock(return_value="UPDATE 1")

    result = await prune_low_yield_targets(
        pool,
        source="twitter",
        lookback_runs=3,
        min_runs=2,
        max_inserted_total=0,
        max_disable_per_run=1,
        dry_run=False,
    )

    assert result["dry_run"] is False
    assert result["requested"] == 1
    assert result["disabled"] == 1
    pool.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_disable_low_yield_source_endpoint_uses_config_defaults(monkeypatch):
    from atlas_brain.api.b2b_scrape import (
        DisableLowYieldSourceRequest,
        disable_low_yield_source_targets,
    )

    pool = AsyncMock()
    pool.is_initialized = True

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.prune_low_yield_targets", new_callable=AsyncMock) as mock_prune:
            mock_prune.return_value = {"disabled": 0, "requested": 0, "targets": []}
            from atlas_brain.config import settings

            monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_source", "twitter")
            monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_lookback_runs", 3)
            monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_min_runs", 2)
            monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_max_inserted_total", 0)
            monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_max_disable_per_run", 25)

            result = await disable_low_yield_source_targets(DisableLowYieldSourceRequest())

    assert result["disabled"] == 0
    kwargs = mock_prune.await_args.kwargs
    assert kwargs["source"] == "twitter"
    assert kwargs["lookback_runs"] == 3
    assert kwargs["min_runs"] == 2
    assert kwargs["max_inserted_total"] == 0
    assert kwargs["max_disable_per_run"] == 25
    assert kwargs["dry_run"] is True


@pytest.mark.asyncio
async def test_scrape_target_pruning_task_uses_shared_policy(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_target_pruning import run
    from atlas_brain.config import settings

    pool = AsyncMock()
    pool.is_initialized = True

    monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_enabled", True)
    monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_source", "twitter")
    monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_lookback_runs", 3)
    monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_min_runs", 2)
    monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_max_inserted_total", 0)
    monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_max_disable_per_run", 25)
    monkeypatch.setattr(settings.b2b_scrape, "source_low_yield_pruning_dry_run", False)

    with patch(
        "atlas_brain.autonomous.tasks.b2b_scrape_target_pruning.get_db_pool",
        return_value=pool,
    ):
        with patch(
            "atlas_brain.autonomous.tasks.b2b_scrape_target_pruning.prune_low_yield_targets",
            new_callable=AsyncMock,
        ) as mock_prune:
            mock_prune.return_value = {
                "source": "twitter",
                "requested": 1,
                "disabled": 1,
                "dry_run": False,
                "targets": [],
            }
            result = await run(MagicMock())

    assert result["disabled"] == 1
    assert result["_skip_synthesis"] is True
    kwargs = mock_prune.await_args.kwargs
    assert kwargs["source"] == "twitter"
    assert kwargs["lookback_runs"] == 3
    assert kwargs["min_runs"] == 2


def test_coverage_planner_flags_missing_core_and_poor_fit_targets():
    from atlas_brain.services.scraping.target_planning import build_scrape_coverage_plan

    plan = build_scrape_coverage_plan(
        [
            {
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "total_reviews_analyzed": 300,
                "confidence_score": 0.8,
            },
            {
                "vendor_name": "Datadog",
                "product_category": "Cloud Infrastructure",
                "total_reviews_analyzed": 250,
                "confidence_score": 0.7,
            },
        ],
        [
            {
                "id": "t-1",
                "source": "github",
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "product_slug": "hubspot",
                "enabled": True,
                "scrape_mode": "incremental",
                "priority": 5,
                "metadata": {},
            },
            {
                "id": "t-2",
                "source": "g2",
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "product_slug": "hubspot",
                "enabled": True,
                "scrape_mode": "incremental",
                "priority": 5,
                "metadata": {},
            },
        ],
        allowed_sources=["g2", "github", "reddit"],
    )

    assert plan["summary"]["poor_fit_enabled_targets"] == 0
    assert plan["poor_fit_enabled_targets"] == []
    missing = {(row["vendor_name"], row["source"]) for row in plan["missing_core_targets"]}
    assert ("HubSpot", "reddit") in missing
    assert ("Datadog", "github") in missing
    assert ("Datadog", "reddit") in missing
    datadog_github = next(row for row in plan["missing_core_targets"] if row["vendor_name"] == "Datadog" and row["source"] == "github")
    assert datadog_github["auto_seedable"] is True
    assert datadog_github["requires_product_slug"] is False
    assert datadog_github["suggested_product_slug"] == "datadog"


def test_coverage_planner_exposes_verified_software_advice_seed():
    from atlas_brain.services.scraping.target_planning import build_scrape_coverage_plan

    plan = build_scrape_coverage_plan(
        [
            {
                "vendor_name": "Jira",
                "product_category": "Project Management",
                "total_reviews_analyzed": 450,
                "confidence_score": 0.8,
            }
        ],
        [
            {
                "id": "t-1",
                "source": "g2",
                "vendor_name": "Jira",
                "product_category": "Project Management",
                "product_slug": "jira",
                "enabled": True,
                "scrape_mode": "incremental",
                "priority": 5,
                "metadata": {},
            }
        ],
        allowed_sources=["g2", "software_advice"],
    )

    jira_sa = next(row for row in plan["missing_core_targets"] if row["source"] == "software_advice")
    assert jira_sa["can_seed_now"] is True
    assert jira_sa["verified_product_slug"] == "project-management/atlassian-jira-profile"
    assert jira_sa["verified_product_name"] == "Jira"


def test_coverage_planner_exposes_conditional_probation_opportunities():
    from atlas_brain.services.scraping.target_planning import build_scrape_coverage_plan

    plan = build_scrape_coverage_plan(
        [
            {
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "total_reviews_analyzed": 300,
                "confidence_score": 0.8,
            }
        ],
        [
            {
                "id": "t-1",
                "source": "g2",
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "product_slug": "hubspot",
                "enabled": True,
                "scrape_mode": "incremental",
                "priority": 5,
                "metadata": {},
            }
        ],
        allowed_sources=["g2", "twitter"],
    )

    assert plan["summary"]["conditional_opportunities"] == 1
    opportunity = plan["conditional_opportunities"][0]
    assert opportunity["source"] == "twitter"
    assert opportunity["can_probation_now"] is True
    assert opportunity["auto_seedable"] is True
    assert opportunity["suggested_product_slug"] == "hubspot"


def test_coverage_planner_skips_unsupported_vendor_level_software_advice_target():
    from atlas_brain.services.scraping.target_planning import build_scrape_coverage_plan

    plan = build_scrape_coverage_plan(
        [
            {
                "vendor_name": "Amazon Web Services",
                "product_category": "Cloud Infrastructure",
                "total_reviews_analyzed": 539,
                "confidence_score": 0.84,
            }
        ],
        [],
        allowed_sources=["software_advice"],
    )

    assert plan["summary"]["conditional_opportunities"] == 0
    assert plan["conditional_opportunities"] == []


def test_coverage_planner_prefers_profile_inventory_source_over_signal_fallback():
    from atlas_brain.services.scraping.target_planning import collapse_inventory_rows

    rows = [
        {
            "vendor_name": "HubSpot",
            "product_category": "CRM",
            "total_reviews_analyzed": 120,
            "confidence_score": 0.8,
            "last_computed_at": "2026-03-18T00:00:00+00:00",
            "inventory_source": "b2b_product_profiles",
        },
        {
            "vendor_name": "HubSpot",
            "product_category": "CRM",
            "total_reviews_analyzed": 400,
            "confidence_score": 0.0,
            "last_computed_at": None,
            "inventory_source": "b2b_churn_signals",
        },
    ]

    collapsed = collapse_inventory_rows(rows)
    assert collapsed[0]["inventory_source"] == "b2b_product_profiles"


def test_coverage_planner_reports_mixed_inventory_sources():
    from atlas_brain.services.scraping.target_planning import build_scrape_coverage_plan

    plan = build_scrape_coverage_plan(
        [
            {
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "total_reviews_analyzed": 300,
                "confidence_score": 0.8,
                "last_computed_at": None,
                "inventory_source": "b2b_product_profiles",
            },
            {
                "vendor_name": "OpenProject",
                "product_category": "Project Management",
                "total_reviews_analyzed": 80,
                "confidence_score": 0.0,
                "last_computed_at": None,
                "inventory_source": "b2b_churn_signals",
            },
        ],
        [],
        allowed_sources=["g2"],
    )

    assert plan["inventory_source"] == "mixed"
    assert plan["inventory_source_breakdown"] == {
        "b2b_churn_signals": 1,
        "b2b_product_profiles": 1,
    }


def test_target_query_prioritizes_never_scraped_targets_before_repeats():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _TARGET_QUERY

    assert "CASE WHEN last_scraped_at IS NULL THEN 0 ELSE 1 END" in _TARGET_QUERY


@pytest.mark.asyncio
async def test_coverage_plan_endpoint_uses_product_profiles_inventory():

    from atlas_brain.api.b2b_scrape import coverage_plan

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "vendor_name": "HubSpot",
                    "product_category": "CRM",
                    "total_reviews_analyzed": 300,
                    "confidence_score": 0.8,
                    "last_computed_at": None,
                }
            ],
            [],
            [
                {
                    "id": "t-1",
                    "source": "g2",
                    "vendor_name": "HubSpot",
                    "product_category": "CRM",
                    "product_slug": "hubspot",
                    "enabled": True,
                    "scrape_mode": "incremental",
                    "priority": 5,
                    "max_pages": 5,
                    "scrape_interval_hours": 168,
                    "metadata": {},
                }
            ],
        ]
    )

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        result = await coverage_plan(limit=10)

    assert result["inventory_source"] == "b2b_product_profiles"
    assert result["vendors_considered"] == 1
    assert result["summary"]["missing_core_targets"] >= 0


@pytest.mark.asyncio
async def test_seed_missing_core_endpoint_inserts_verified_target():

    from atlas_brain.api.b2b_scrape import SeedMissingCoreRequest, seed_missing_core_targets

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "vendor_name": "Jira",
                    "product_category": "Project Management",
                    "total_reviews_analyzed": 450,
                    "confidence_score": 0.8,
                    "last_computed_at": None,
                }
            ],
            [],
            [
                {
                    "id": "t-1",
                    "source": "g2",
                    "vendor_name": "Jira",
                    "product_name": "Jira",
                    "product_category": "Project Management",
                    "product_slug": "jira",
                    "enabled": True,
                    "scrape_mode": "incremental",
                    "priority": 5,
                    "max_pages": 3,
                    "scrape_interval_hours": 168,
                    "metadata": {},
                }
            ],
        ]
    )
    pool.fetchrow = AsyncMock(return_value={"id": "seed-1"})

    cfg = MagicMock()
    cfg.source_allowlist = "g2,software_advice"

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            with patch("atlas_brain.api.b2b_scrape.resolve_vendor_name", new=AsyncMock(return_value="Jira")):
                result = await seed_missing_core_targets(SeedMissingCoreRequest(dry_run=False))

    assert result["applied"] == 1
    assert result["actions"][0]["source"] == "software_advice"
    assert result["actions"][0]["product_slug"] == "project-management/atlassian-jira-profile"
    pool.fetchrow.assert_awaited_once()


@pytest.mark.asyncio
async def test_onboard_vendor_targets_endpoint_bootstraps_net_new_vendor():
    from atlas_brain.api.b2b_scrape import (
        VendorOnboardingTargetsRequest,
        onboard_vendor_targets,
    )

    pool = AsyncMock()
    pool.is_initialized = True

    expected = {
        "status": "dry_run",
        "requested": 2,
        "applied": 2,
        "matched_vendors": ["Microsoft Defender for Endpoint"],
        "unmatched_vendors": [],
        "inventory_source": "manual_bootstrap",
        "inventory_source_breakdown": {"manual_bootstrap": 1},
        "bootstrap_used": True,
        "actions": [
            {"source": "reddit", "vendor_name": "Microsoft Defender for Endpoint"},
            {"source": "hackernews", "vendor_name": "Microsoft Defender for Endpoint"},
        ],
    }

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch(
            "atlas_brain.api.b2b_scrape.provision_vendor_onboarding_targets",
            new=AsyncMock(return_value=expected),
        ) as provision:
            result = await onboard_vendor_targets(
                VendorOnboardingTargetsRequest(
                    vendor_name="Microsoft Defender for Endpoint",
                    product_category="Cybersecurity",
                    dry_run=True,
                )
            )

    assert result == expected
    provision.assert_awaited_once_with(
        pool,
        "Microsoft Defender for Endpoint",
        product_category="Cybersecurity",
        source_slug_overrides={},
        dry_run=True,
        limit=200,
    )


@pytest.mark.asyncio
async def test_fetch_coverage_inputs_uses_shared_signal_inventory_adapter():
    from atlas_brain.services.scraping.target_provisioning import fetch_coverage_inputs

    pool = AsyncMock()
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "vendor_name": "Linear",
                    "product_category": "Project Management",
                    "total_reviews_analyzed": 125,
                    "confidence_score": 0.9,
                    "last_computed_at": None,
                    "inventory_source": "b2b_product_profiles",
                }
            ],
            [],
        ]
    )
    signal_adapter = AsyncMock(
        return_value=[
            {
                "vendor_name": "OpenProject",
                "product_category": "Project Management",
                "total_reviews_analyzed": 80,
                "confidence_score": 0.0,
                "last_computed_at": None,
                "inventory_source": "b2b_churn_signals",
            }
        ]
    )

    with patch(
        "atlas_brain.autonomous.tasks._b2b_shared.read_vendor_scorecard_inventory_rows",
        signal_adapter,
    ):
        inventory, targets = await fetch_coverage_inputs(pool)

    signal_adapter.assert_awaited_once_with(pool)
    assert [row["inventory_source"] for row in inventory] == [
        "b2b_product_profiles",
        "b2b_churn_signals",
    ]
    assert targets == []
    assert pool.fetch.await_count == 2


@pytest.mark.asyncio
async def test_provision_missing_core_targets_uses_signal_inventory_fallback():
    from atlas_brain.services.scraping.target_provisioning import (
        provision_missing_core_targets_for_vendors,
    )

    pool = AsyncMock()
    pool.fetch = AsyncMock(
        side_effect=[
            [],
            [
                {
                    "vendor_name": "OpenProject",
                    "product_category": "Project Management",
                    "total_reviews_analyzed": 80,
                    "confidence_score": 0.0,
                    "last_computed_at": None,
                    "inventory_source": "b2b_churn_signals",
                }
            ],
            [],
        ]
    )
    pool.fetchrow = AsyncMock(return_value={"id": "seed-1"})

    cfg = MagicMock()
    cfg.source_allowlist = "reddit"

    with patch(
        "atlas_brain.services.scraping.target_provisioning.settings",
        MagicMock(b2b_scrape=cfg),
    ):
        with patch(
            "atlas_brain.services.scraping.target_provisioning.resolve_vendor_name",
            new=AsyncMock(return_value="OpenProject"),
        ):
            result = await provision_missing_core_targets_for_vendors(
                pool,
                ["OpenProject"],
                dry_run=False,
            )

    assert result["status"] == "applied"
    assert result["applied"] == 1
    assert result["matched_vendors"] == ["OpenProject"]
    assert result["actions"][0]["source"] == "reddit"
    pool.fetchrow.assert_awaited_once()


@pytest.mark.asyncio
async def test_disable_poor_fit_endpoint_disables_targets():

    from atlas_brain.api.b2b_scrape import DisablePoorFitRequest, disable_poor_fit_targets

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "vendor_name": "HubSpot",
                    "product_category": "CRM",
                    "total_reviews_analyzed": 300,
                    "confidence_score": 0.8,
                    "last_computed_at": None,
                }
            ],
            [],
            [
                {
                    "id": "t-1",
                    "source": "sourceforge",
                    "vendor_name": "HubSpot",
                    "product_name": "HubSpot",
                    "product_category": "CRM",
                    "product_slug": "hubspot",
                    "enabled": True,
                    "scrape_mode": "incremental",
                    "priority": 5,
                    "max_pages": 5,
                    "scrape_interval_hours": 168,
                    "metadata": {},
                }
            ],
        ]
    )
    pool.execute = AsyncMock(return_value="UPDATE 1")

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        result = await disable_poor_fit_targets(DisablePoorFitRequest(dry_run=False))

    assert result["disabled"] == 1
    assert result["targets"][0]["source"] == "sourceforge"
    pool.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_probation_telemetry_endpoint_summarizes_rates():
    from atlas_brain.api.b2b_scrape import probation_telemetry

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "target_id": "target-1",
                    "vendor_name": "HubSpot",
                    "source": "twitter",
                    "product_name": "HubSpot",
                    "product_slug": "hubspot",
                    "product_category": "Marketing Automation",
                    "priority": 3,
                    "max_pages": 3,
                    "scrape_interval_hours": 336,
                    "last_scraped_at": None,
                    "last_scrape_status": "success",
                    "last_scrape_reviews": 4,
                    "metadata": {
                        "source_fit_probation_reason": "vertical_conditional_source",
                        "source_fit_probation_seeded_at": "2026-03-19T00:00:00+00:00",
                    },
                    "runs_total": 2,
                    "reviews_found_total": 10,
                    "reviews_inserted_total": 4,
                    "last_run_at": None,
                }
            ],
            [
                {
                    "target_id": "target-1",
                    "tracked_reviews": 4,
                    "named_company_reviews": 3,
                    "actionable_reviews": 2,
                }
            ],
            [
                {
                    "target_id": "target-1",
                    "company_signal_reviews": 1,
                }
            ],
        ]
    )

    cfg = MagicMock()
    cfg.source_fit_probation_telemetry_lookback_days = 30
    cfg.source_fit_probation_actionable_urgency_min = 7.0

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            result = await probation_telemetry(limit=100, lookback_days=None)

    assert result["basis"] == "raw_source_target_provenance"
    assert result["probation_targets"] == 1
    assert result["summary"]["targets_with_runs"] == 1
    target = result["targets"][0]
    assert target["scrape_yield_rate"] == 0.4
    assert target["duplicate_noise_rate"] == 0.6
    assert target["named_company_hit_rate"] == 0.75
    assert target["actionable_review_rate"] == 0.5
    assert target["company_signal_hit_rate"] == 0.25
    assert target["vertical"] == "crm_support_marketing"


@pytest.mark.asyncio
async def test_disable_low_yield_probation_endpoint_disables_failed_zero_yield_targets():
    from atlas_brain.api.b2b_scrape import (
        DisableLowYieldProbationRequest,
        disable_low_yield_probation_targets,
    )

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "target_id": "target-1",
                    "vendor_name": "HubSpot",
                    "source": "twitter",
                    "product_name": "HubSpot",
                    "product_slug": "hubspot",
                    "product_category": "Marketing Automation",
                    "priority": 3,
                    "max_pages": 3,
                    "scrape_interval_hours": 72,
                    "last_scraped_at": None,
                    "last_scrape_status": "failed",
                    "last_scrape_reviews": 0,
                    "metadata": {
                        "source_fit_probation_reason": "vertical_conditional_source",
                        "source_fit_probation_seeded_at": "2026-03-19T00:00:00+00:00",
                    },
                    "runs_total": 1,
                    "reviews_found_total": 0,
                    "reviews_inserted_total": 0,
                    "last_run_at": None,
                },
                {
                    "target_id": "target-2",
                    "vendor_name": "Datadog",
                    "source": "twitter",
                    "product_name": "Datadog",
                    "product_slug": "datadog",
                    "product_category": "Cloud Infrastructure",
                    "priority": 3,
                    "max_pages": 3,
                    "scrape_interval_hours": 72,
                    "last_scraped_at": None,
                    "last_scrape_status": "failed",
                    "last_scrape_reviews": 2,
                    "metadata": {
                        "source_fit_probation_reason": "vertical_conditional_source",
                        "source_fit_probation_seeded_at": "2026-03-19T00:00:00+00:00",
                    },
                    "runs_total": 1,
                    "reviews_found_total": 5,
                    "reviews_inserted_total": 2,
                    "last_run_at": None,
                },
            ],
            [],
            [],
        ]
    )
    pool.execute = AsyncMock(return_value="UPDATE 1")

    cfg = MagicMock()
    cfg.source_fit_probation_telemetry_lookback_days = 30
    cfg.source_fit_probation_actionable_urgency_min = 7.0

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            result = await disable_low_yield_probation_targets(
                DisableLowYieldProbationRequest(
                    dry_run=False,
                    sources=["twitter"],
                    verticals=["crm_support_marketing"],
                )
            )

    assert result["disabled"] == 1
    assert result["targets"][0]["target_id"] == "target-1"
    pool.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_promote_probation_endpoint_promotes_useful_target():
    from atlas_brain.api.b2b_scrape import (
        PromoteProbationTargetsRequest,
        promote_probation_targets,
    )

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "target_id": "target-1",
                    "vendor_name": "HubSpot",
                    "source": "twitter",
                    "product_name": "HubSpot",
                    "product_slug": "hubspot",
                    "product_category": "Marketing Automation",
                    "priority": 3,
                    "max_pages": 3,
                    "scrape_interval_hours": 72,
                    "last_scraped_at": None,
                    "last_scrape_status": "success",
                    "last_scrape_reviews": 4,
                    "metadata": {
                        "source_fit_probation_reason": "vertical_conditional_source",
                        "source_fit_probation_seeded_at": "2026-03-19T00:00:00+00:00",
                    },
                    "runs_total": 2,
                    "reviews_found_total": 10,
                    "reviews_inserted_total": 4,
                    "last_run_at": None,
                }
            ],
            [
                {
                    "target_id": "target-1",
                    "tracked_reviews": 4,
                    "named_company_reviews": 2,
                    "actionable_reviews": 2,
                }
            ],
            [
                {
                    "target_id": "target-1",
                    "company_signal_reviews": 1,
                }
            ],
            [
                {
                    "id": "target-1",
                    "source": "twitter",
                    "vendor_name": "HubSpot",
                    "product_name": "HubSpot",
                    "product_category": "Marketing Automation",
                    "product_slug": "hubspot",
                    "enabled": True,
                    "scrape_mode": "incremental",
                    "priority": 3,
                    "max_pages": 3,
                    "scrape_interval_hours": 72,
                    "metadata": {"source_fit_probation": True},
                },
                {
                    "id": "baseline-1",
                    "source": "twitter",
                    "vendor_name": "Mailchimp",
                    "product_name": "Mailchimp",
                    "product_category": "Marketing Automation",
                    "product_slug": "mailchimp",
                    "enabled": True,
                    "scrape_mode": "incremental",
                    "priority": 8,
                    "max_pages": 9,
                    "scrape_interval_hours": 168,
                    "metadata": {},
                },
            ],
        ]
    )
    pool.execute = AsyncMock(return_value="UPDATE 1")

    cfg = MagicMock()
    cfg.source_fit_probation_telemetry_lookback_days = 30
    cfg.source_fit_probation_actionable_urgency_min = 7.0

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            result = await promote_probation_targets(
                PromoteProbationTargetsRequest(
                    dry_run=False,
                    sources=["twitter"],
                    verticals=["crm_support_marketing"],
                )
            )

    assert result["promoted"] == 1
    target = result["targets"][0]
    assert target["target_id"] == "target-1"
    assert target["promotion_defaults"]["priority"] == 8
    assert target["promotion_defaults"]["max_pages"] == 9
    assert target["promotion_defaults"]["scrape_interval_hours"] == 168
    pool.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_seed_conditional_probation_endpoint_inserts_capped_target():
    from atlas_brain.api.b2b_scrape import SeedConditionalProbationRequest, seed_conditional_probation_targets

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "vendor_name": "HubSpot",
                    "product_category": "CRM",
                    "total_reviews_analyzed": 300,
                    "confidence_score": 0.8,
                    "last_computed_at": None,
                }
            ],
            [],
            [
                {
                    "id": "t-1",
                    "source": "g2",
                    "vendor_name": "HubSpot",
                    "product_name": "HubSpot",
                    "product_category": "CRM",
                    "product_slug": "hubspot",
                    "enabled": True,
                    "scrape_mode": "incremental",
                    "priority": 10,
                    "max_pages": 15,
                    "scrape_interval_hours": 168,
                    "metadata": {},
                }
            ],
        ]
    )
    pool.fetchrow = AsyncMock(return_value={"id": "probation-1"})

    cfg = MagicMock()
    cfg.source_allowlist = "g2,twitter"
    cfg.source_fit_probation_priority = 3
    cfg.source_fit_probation_max_pages = 3
    cfg.source_fit_probation_scrape_interval_hours = 336

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            with patch("atlas_brain.api.b2b_scrape.resolve_vendor_name", new=AsyncMock(return_value="HubSpot")):
                result = await seed_conditional_probation_targets(
                    SeedConditionalProbationRequest(dry_run=False)
                )

    assert result["applied"] == 1
    target = result["targets"][0]
    assert target["source"] == "twitter"
    assert target["priority"] == 3
    assert target["max_pages"] == 3
    assert target["scrape_interval_hours"] == 336
    assert target["metadata"]["source_fit_probation"] is True
    pool.fetchrow.assert_awaited_once()


@pytest.mark.asyncio
async def test_seed_conditional_probation_endpoint_filters_by_vertical():
    from atlas_brain.api.b2b_scrape import SeedConditionalProbationRequest, seed_conditional_probation_targets

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "vendor_name": "HubSpot",
                    "product_category": "CRM",
                    "total_reviews_analyzed": 300,
                    "confidence_score": 0.8,
                    "last_computed_at": None,
                },
                {
                    "vendor_name": "Asana",
                    "product_category": "Project Management",
                    "total_reviews_analyzed": 250,
                    "confidence_score": 0.8,
                    "last_computed_at": None,
                },
            ],
            [],
            [
                {
                    "id": "crm-1",
                    "source": "twitter",
                    "vendor_name": "HubSpot",
                    "product_name": "HubSpot",
                    "product_category": "CRM",
                    "product_slug": "hubspot",
                    "enabled": False,
                    "scrape_mode": "incremental",
                    "priority": 10,
                    "max_pages": 15,
                    "scrape_interval_hours": 168,
                    "metadata": {},
                },
                {
                    "id": "pm-1",
                    "source": "twitter",
                    "vendor_name": "Asana",
                    "product_name": "Asana",
                    "product_category": "Project Management",
                    "product_slug": "asana",
                    "enabled": False,
                    "scrape_mode": "incremental",
                    "priority": 10,
                    "max_pages": 15,
                    "scrape_interval_hours": 168,
                    "metadata": {},
                },
            ],
        ]
    )
    pool.execute = AsyncMock(return_value="UPDATE 1")

    cfg = MagicMock()
    cfg.source_allowlist = "twitter"
    cfg.source_fit_probation_priority = 3
    cfg.source_fit_probation_max_pages = 3
    cfg.source_fit_probation_scrape_interval_hours = 336

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            result = await seed_conditional_probation_targets(
                SeedConditionalProbationRequest(
                    dry_run=False,
                    verticals=["crm_support_marketing"],
                )
            )

    assert result["applied"] == 1
    assert result["targets"][0]["vendor_name"] == "HubSpot"
    assert result["targets"][0]["source"] == "twitter"
    pool.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_seed_conditional_probation_endpoint_reenables_with_caps():
    from atlas_brain.api.b2b_scrape import SeedConditionalProbationRequest, seed_conditional_probation_targets

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "vendor_name": "HubSpot",
                    "product_category": "CRM",
                    "total_reviews_analyzed": 300,
                    "confidence_score": 0.8,
                    "last_computed_at": None,
                }
            ],
            [],
            [
                {
                    "id": "t-2",
                    "source": "twitter",
                    "vendor_name": "HubSpot",
                    "product_name": "HubSpot",
                    "product_category": "CRM",
                    "product_slug": "hubspot",
                    "enabled": False,
                    "scrape_mode": "exhaustive",
                    "priority": 12,
                    "max_pages": 20,
                    "scrape_interval_hours": 24,
                    "metadata": {},
                }
            ],
        ]
    )
    pool.execute = AsyncMock(return_value="UPDATE 1")

    cfg = MagicMock()
    cfg.source_allowlist = "g2,twitter"
    cfg.source_fit_probation_priority = 3
    cfg.source_fit_probation_max_pages = 3
    cfg.source_fit_probation_scrape_interval_hours = 336

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            result = await seed_conditional_probation_targets(
                SeedConditionalProbationRequest(dry_run=False)
            )

    assert result["applied"] == 1
    target = result["targets"][0]
    assert target["action"] == "enable_existing_probation"
    assert target["priority"] == 3
    assert target["max_pages"] == 3
    assert target["scrape_interval_hours"] == 336
    pool.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_scrape_log_persists_page_logs():

    from atlas_brain.api.b2b_scrape import _write_scrape_log
    from atlas_brain.services.scraping.parsers import log_page

    pool = AsyncMock()
    pool.fetchval = AsyncMock(return_value="run-123")
    parser = MagicMock(prefer_residential=True, version="getapp:test")
    page_logs = [log_page(1, "https://example.com/reviews", status_code=200)]

    with patch("atlas_brain.autonomous.tasks.b2b_scrape_intake._persist_page_logs", new=AsyncMock()) as persist:
        run_id = await _write_scrape_log(
            pool,
            "00000000-0000-0000-0000-000000000001",
            "getapp",
            "failed",
            0,
            0,
            1,
            ["blocked"],
            123,
            parser,
            page_logs=page_logs,
        )

    assert run_id == "run-123"
    persist.assert_awaited_once_with(pool, "run-123", page_logs)


@pytest.mark.asyncio
async def test_manual_scrape_log_persists_incremental_boundaries():

    from atlas_brain.api.b2b_scrape import _write_scrape_log

    pool = AsyncMock()
    pool.fetchval = AsyncMock(return_value="run-123")
    parser = MagicMock(prefer_residential=False, version="reddit:test")

    await _write_scrape_log(
        pool,
        "00000000-0000-0000-0000-000000000001",
        "reddit",
        "success",
        66,
        0,
        29,
        [],
        20841,
        parser,
        stop_reason="page_cap",
        oldest_review="2026-02-22",
        newest_review="2026-03-19",
        date_dropped=0,
    )

    sql_args = pool.fetchval.await_args.args
    assert "stop_reason" in sql_args[0]
    assert sql_args[15] == "page_cap"
    assert str(sql_args[16]) == "2026-02-22"
    assert str(sql_args[17]) == "2026-03-19"
    assert sql_args[18] == 0
    assert sql_args[19] == 0
    assert sql_args[20] is False


@pytest.mark.asyncio
async def test_scheduled_scrape_log_persists_incremental_boundaries():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _log_scrape

    pool = AsyncMock()
    pool.fetchval = AsyncMock(return_value="run-456")
    parser = MagicMock(prefer_residential=False, version="reddit:test")
    target = MagicMock(id="00000000-0000-0000-0000-000000000001", source="reddit")

    await _log_scrape(
        pool,
        target,
        "success",
        66,
        0,
        29,
        [],
        20841,
        parser,
        stop_reason="page_cap",
        oldest_review="2026-02-22",
        newest_review="2026-03-19",
        date_dropped=0,
    )

    sql_args = pool.fetchval.await_args.args
    assert "stop_reason" in sql_args[0]
    assert sql_args[15] == "page_cap"
    assert str(sql_args[16]) == "2026-02-22"
    assert str(sql_args[17]) == "2026-03-19"
    assert sql_args[18] == 0
    assert sql_args[19] == 0
    assert sql_args[20] is False


@pytest.mark.asyncio
async def test_trigger_scrape_incremental_path_uses_boundary_helpers():

    from atlas_brain.api.b2b_scrape import trigger_scrape

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "id": "00000000-0000-0000-0000-000000000001",
            "source": "reddit",
            "vendor_name": "Asana",
            "product_name": "Asana",
            "product_slug": "asana-reddit",
            "product_category": "Project Management",
            "max_pages": 15,
            "metadata": {},
            "scrape_mode": "incremental",
        }
    )

    parser = MagicMock()
    parser.version = "reddit:test"
    parser.scrape = AsyncMock(
        return_value=MagicMock(
            reviews=[],
            status="success",
            pages_scraped=5,
            errors=[],
            page_logs=[],
            stop_reason="page_cap",
        )
    )
    target = MagicMock(
        source="reddit",
        vendor_name="Asana",
        date_cutoff=None,
        product_slug="asana-reddit",
        metadata={"scrape_mode": "incremental"},
    )

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.services.scraping.parsers.get_parser", return_value=parser):
            with patch("atlas_brain.services.scraping.client.get_scrape_client", return_value=MagicMock()):
                with patch(
                    "atlas_brain.autonomous.tasks.b2b_scrape_intake._prepare_scrape_target",
                    return_value=(target, "incremental", {}),
                ):
                    with patch(
                        "atlas_brain.autonomous.tasks.b2b_scrape_intake._build_scrape_state",
                        return_value={"runtime_mode": "incremental"},
                    ):
                        with patch(
                            "atlas_brain.autonomous.tasks.b2b_scrape_intake._update_target_after_scrape",
                            new=AsyncMock(),
                        ) as update_target:
                            with patch(
                                "atlas_brain.api.b2b_scrape._write_scrape_log",
                                new=AsyncMock(),
                            ) as write_log:
                                result = await trigger_scrape(
                                    "00000000-0000-0000-0000-000000000001"
                                )

    assert result["status"] == "success"
    assert result["scrape_mode"] == "incremental"
    write_kwargs = write_log.await_args.kwargs
    assert write_kwargs["stop_reason"] == "page_cap"
    assert write_kwargs["oldest_review"] is None
    assert write_kwargs["newest_review"] is None
    update_target.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_probation_batch_runs_due_targets_only():

    from atlas_brain.api.b2b_scrape import RunProbationBatchRequest, run_probation_batch

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        return_value=[
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "source": "g2",
                "vendor_name": "HubSpot",
                "last_scraped_at": None,
                "priority": 3,
            }
        ]
    )

    cfg = MagicMock()
    cfg.source_allowlist = "g2"

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            with patch(
                "atlas_brain.api.b2b_scrape.trigger_scrape",
                new=AsyncMock(return_value={"target_id": "00000000-0000-0000-0000-000000000001", "status": "success"}),
            ) as trigger:
                result = await run_probation_batch(RunProbationBatchRequest(limit=5, due_only=True))

    assert result["selected"] == 1
    assert result["succeeded"] == 1
    assert result["skipped_disallowed_sources"] == 0
    trigger.assert_awaited_once_with("00000000-0000-0000-0000-000000000001")
    sql = pool.fetch.await_args.args[0]
    assert "source_fit_probation" in sql
    assert "source = ANY(" in sql
    assert "make_interval(hours => scrape_interval_hours)" in sql


@pytest.mark.asyncio
async def test_run_probation_batch_skips_disallowed_sources():
    from atlas_brain.api.b2b_scrape import RunProbationBatchRequest, run_probation_batch

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        return_value=[
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "source": "twitter",
                "vendor_name": "HubSpot",
                "last_scraped_at": None,
                "priority": 3,
            }
        ]
    )

    cfg = MagicMock()
    cfg.source_allowlist = "g2"

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            with patch("atlas_brain.api.b2b_scrape.trigger_scrape", new=AsyncMock()) as trigger:
                result = await run_probation_batch(RunProbationBatchRequest(limit=5, due_only=True))

    assert result["selected"] == 0
    assert result["attempted"] == 0
    assert result["skipped_disallowed_sources"] == 1
    trigger.assert_not_awaited()


@pytest.mark.asyncio
async def test_trigger_scrape_all_rejects_disallowed_source_filter():
    from fastapi import HTTPException
    from atlas_brain.api.b2b_scrape import trigger_scrape_all

    pool = AsyncMock()
    pool.is_initialized = True

    cfg = MagicMock()
    cfg.source_allowlist = "g2"

    with patch("atlas_brain.api.b2b_scrape.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_scrape.settings", MagicMock(b2b_scrape=cfg)):
            with pytest.raises(HTTPException) as exc:
                await trigger_scrape_all(source="twitter")

    assert exc.value.status_code == 400
    assert "ATLAS_B2B_SCRAPE_SOURCE_ALLOWLIST" in str(exc.value.detail)
