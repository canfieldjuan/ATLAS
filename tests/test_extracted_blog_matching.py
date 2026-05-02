from __future__ import annotations

import pytest

from extracted_content_pipeline.autonomous.tasks._blog_matching import (
    _blog_base_url,
    _role_topic_boosts,
    fetch_relevant_blog_posts,
)


class _Pool:
    def __init__(self, rows):
        self.rows = rows

    async def fetch(self, query, *args):
        return self.rows


class _CapturingPool(_Pool):
    def __init__(self, rows):
        super().__init__(rows)
        self.last_query = None

    async def fetch(self, query, *args):
        self.last_query = query
        return await super().fetch(query, *args)


@pytest.mark.asyncio
async def test_blog_matching_uses_extracted_b2b_base_url(monkeypatch) -> None:
    monkeypatch.setenv("EXTRACTED_B2B_BLOG_BASE_URL", "https://content.example.com/")
    rows = [
        {
            "title": "Zendesk Migration Guide",
            "slug": "zendesk-migration-guide",
            "topic_type": "migration_guide",
            "status": "published",
            "tags": ["customer-support"],
            "data_context": {},
        }
    ]

    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        vendor_name="Zendesk",
    )

    assert posts[0]["url"] == "https://content.example.com/blog/zendesk-migration-guide"


def test_blog_base_url_falls_back_without_settings_shape(monkeypatch) -> None:
    monkeypatch.delenv("EXTRACTED_B2B_BLOG_BASE_URL", raising=False)
    monkeypatch.delenv("EXTRACTED_BLOG_BASE_URL", raising=False)

    assert _blog_base_url("b2b")


@pytest.mark.asyncio
async def test_unknown_pipeline_returns_empty_list() -> None:
    posts = await fetch_relevant_blog_posts(
        _Pool([]),
        pipeline="unknown",
        vendor_name="Zendesk",
    )

    assert posts == []


@pytest.mark.asyncio
async def test_alternative_vendor_matching_avoids_substring_false_positive() -> None:
    rows = [
        {
            "title": "CRM Landscape 2026",
            "slug": "crm-landscape-2026-03",
            "topic_type": "market_landscape",
            "tags": ["crm"],
            "data_context": {},
        },
        {
            "title": "Salesforce vs SAP",
            "slug": "salesforce-vs-sap-2026-03",
            "topic_type": "vendor_showdown",
            "tags": ["crm", "comparison"],
            "data_context": {"topic_ctx": {"vendor_a": "Salesforce", "vendor_b": "SAP"}},
        },
    ]

    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        alternative_vendors=["sap"],
        limit=5,
    )

    assert len(posts) == 1
    assert posts[0]["title"] == "Salesforce vs SAP"


@pytest.mark.asyncio
async def test_alternative_vendor_matching_supports_exact_vendor_b_match() -> None:
    rows = [
        {
            "title": "Cloud Vendor Showdown",
            "slug": "cloud-vendor-showdown-2026-03",
            "topic_type": "vendor_showdown",
            "tags": ["cloud"],
            "data_context": {"topic_ctx": {"vendor_b": "Google Cloud Platform"}},
        },
    ]

    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        alternative_vendors=["google cloud platform"],
        limit=3,
    )

    assert len(posts) == 1
    assert posts[0]["topic_type"] == "vendor_showdown"


@pytest.mark.asyncio
async def test_blog_matching_draft_fallback_is_opt_in() -> None:
    rows = [
        {
            "title": "Asana vs Monday.com",
            "slug": "asana-vs-mondaycom-2026-03",
            "topic_type": "vendor_showdown",
            "status": "draft",
            "tags": ["project-management", "comparison"],
            "data_context": {"topic_ctx": {"vendor_a": "Asana", "vendor_b": "Monday.com"}},
        },
    ]

    no_fallback = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        alternative_vendors=["asana"],
        limit=3,
    )
    with_fallback = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        alternative_vendors=["asana"],
        include_drafts=True,
        limit=3,
    )

    assert no_fallback == []
    assert len(with_fallback) == 1
    assert with_fallback[0]["url"].endswith("/blog/asana-vs-mondaycom-2026-03")


@pytest.mark.asyncio
async def test_blog_matching_prefers_published_over_drafts() -> None:
    rows = [
        {
            "title": "Asana vs Monday.com (Draft)",
            "slug": "asana-vs-mondaycom-2026-03-draft",
            "topic_type": "vendor_showdown",
            "status": "draft",
            "tags": ["project-management", "comparison"],
            "data_context": {"topic_ctx": {"vendor_a": "Asana", "vendor_b": "Monday.com"}},
        },
        {
            "title": "Asana vs Monday.com",
            "slug": "asana-vs-mondaycom-2026-03",
            "topic_type": "vendor_showdown",
            "status": "published",
            "tags": ["project-management", "comparison"],
            "data_context": {"topic_ctx": {"vendor_a": "Asana", "vendor_b": "Monday.com"}},
        },
    ]

    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        alternative_vendors=["asana"],
        include_drafts=True,
        limit=3,
    )

    assert len(posts) == 1
    assert posts[0]["url"].endswith("/blog/asana-vs-mondaycom-2026-03")


@pytest.mark.asyncio
async def test_blog_matching_prefers_relevant_draft_over_unrelated_published_post() -> None:
    rows = [
        {
            "title": "The Real Cost of Salesforce",
            "slug": "real-cost-of-salesforce-2026-03",
            "topic_type": "pricing_reality_check",
            "status": "published",
            "tags": ["crm", "pricing"],
            "data_context": {"pain_distribution": "pricing"},
        },
        {
            "title": "CrowdStrike vs SentinelOne",
            "slug": "crowdstrike-vs-sentinelone-2026-03",
            "topic_type": "vendor_showdown",
            "status": "draft",
            "tags": ["security", "comparison"],
            "data_context": {"topic_ctx": {"vendor_a": "CrowdStrike", "vendor_b": "SentinelOne"}},
        },
    ]

    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        vendor_name="CrowdStrike",
        alternative_vendors=["SentinelOne"],
        pain_categories=["pricing"],
        include_drafts=True,
        limit=3,
    )

    assert len(posts) == 2
    assert posts[0]["topic_type"] == "vendor_showdown"
    assert posts[0]["url"].endswith("/blog/crowdstrike-vs-sentinelone-2026-03")


@pytest.mark.asyncio
async def test_blog_matching_orders_recent_drafts_by_created_at_when_enabled() -> None:
    pool = _CapturingPool([])

    await fetch_relevant_blog_posts(
        pool,
        pipeline="b2b",
        alternative_vendors=["sentinelone"],
        include_drafts=True,
        limit=3,
    )

    assert "COALESCE(published_at, created_at) DESC" in pool.last_query


def test_role_topic_boosts_by_persona() -> None:
    assert _role_topic_boosts("CFO").get("pricing_reality_check", 0) >= 3
    assert _role_topic_boosts("CTO / Head of Engineering").get("migration_guide", 0) >= 3
    assert _role_topic_boosts(None) == {}
    assert _role_topic_boosts("Barista") == {}


@pytest.mark.asyncio
async def test_contact_role_boosts_matching_post_above_unrelated() -> None:
    rows = [
        {
            "title": "Zendesk vs Freshdesk",
            "slug": "zendesk-vs-freshdesk-2026-03",
            "topic_type": "vendor_showdown",
            "status": "published",
            "tags": ["customer-support"],
            "data_context": {"topic_ctx": {"vendor_a": "Zendesk", "vendor_b": "Freshdesk"}},
        },
        {
            "title": "The Real Cost of Zendesk",
            "slug": "real-cost-of-zendesk-2026-03",
            "topic_type": "pricing_reality_check",
            "status": "published",
            "tags": ["customer-support", "pricing"],
            "data_context": {"pain_distribution": "pricing"},
        },
    ]

    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        vendor_name="Zendesk",
        contact_role="VP Finance",
        limit=3,
    )

    assert len(posts) == 2
    assert posts[0]["topic_type"] == "pricing_reality_check"


@pytest.mark.asyncio
async def test_no_contact_role_preserves_standard_ranking() -> None:
    rows = [
        {
            "title": "Zendesk Migration Guide",
            "slug": "zendesk-migration-guide-2026-03",
            "topic_type": "migration_guide",
            "status": "published",
            "tags": ["customer-support"],
            "data_context": {},
        },
        {
            "title": "Zendesk vs Freshdesk",
            "slug": "zendesk-vs-freshdesk-2026-03",
            "topic_type": "vendor_showdown",
            "status": "published",
            "tags": ["customer-support"],
            "data_context": {"topic_ctx": {"vendor_a": "Zendesk", "vendor_b": "Freshdesk"}},
        },
    ]

    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        vendor_name="Zendesk",
        alternative_vendors=["Freshdesk"],
        contact_role=None,
        limit=3,
    )

    assert posts[0]["topic_type"] == "vendor_showdown"


@pytest.mark.asyncio
async def test_diminishing_returns_multi_pain_post_scores_higher() -> None:
    rows = [
        {
            "title": "Zendesk Pricing Problems",
            "slug": "zendesk-pricing-2026-03",
            "topic_type": "pricing_reality_check",
            "status": "published",
            "tags": ["pricing", "customer-support"],
            "data_context": {"pain_distribution": "pricing support"},
        },
        {
            "title": "Zendesk Support Issues",
            "slug": "zendesk-support-2026-03",
            "topic_type": "churn_report",
            "status": "published",
            "tags": ["support", "customer-support"],
            "data_context": {"pain_distribution": "support"},
        },
    ]

    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        pain_categories=["pricing", "support"],
        limit=3,
    )

    assert len(posts) == 2
    assert posts[0]["url"].endswith("/blog/zendesk-pricing-2026-03")


@pytest.mark.asyncio
async def test_single_pain_match_without_break() -> None:
    rows = [
        {
            "title": "Zendesk Migration Guide",
            "slug": "zendesk-migration-2026-03",
            "topic_type": "migration_guide",
            "status": "published",
            "tags": ["migration"],
            "data_context": {},
        },
    ]

    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        vendor_name="Zendesk",
        pain_categories=["pricing", "support", "features", "onboarding"],
        limit=3,
    )

    assert len(posts) == 1
