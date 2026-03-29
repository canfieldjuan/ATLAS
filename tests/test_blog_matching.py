import pytest

from atlas_brain.autonomous.tasks._blog_matching import (
    fetch_relevant_blog_posts,
    _role_topic_boosts,
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
async def test_alternative_vendor_matching_avoids_substring_false_positive():
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
async def test_alternative_vendor_matching_supports_exact_vendor_b_match():
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
async def test_blog_matching_draft_fallback_is_opt_in():
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
async def test_blog_matching_prefers_published_over_drafts():
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
async def test_blog_matching_prefers_relevant_draft_over_unrelated_published_post():
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
async def test_blog_matching_orders_recent_drafts_by_created_at_when_enabled():
    pool = _CapturingPool([])

    await fetch_relevant_blog_posts(
        pool,
        pipeline="b2b",
        alternative_vendors=["sentinelone"],
        include_drafts=True,
        limit=3,
    )

    assert "COALESCE(published_at, created_at) DESC" in pool.last_query


# ---------------------------------------------------------------------------
# Role-boost tests
# ---------------------------------------------------------------------------

def test_role_topic_boosts_cfo_prefers_pricing():
    boosts = _role_topic_boosts("CFO")
    assert boosts.get("pricing_reality_check", 0) >= 3


def test_role_topic_boosts_cto_prefers_migration():
    boosts = _role_topic_boosts("CTO / Head of Engineering")
    assert boosts.get("migration_guide", 0) >= 3


def test_role_topic_boosts_none_returns_empty():
    assert _role_topic_boosts(None) == {}


def test_role_topic_boosts_unknown_role_returns_empty():
    assert _role_topic_boosts("Barista") == {}


@pytest.mark.asyncio
async def test_contact_role_boosts_matching_post_above_unrelated():
    """A pricing post ranks first for a VP Finance even when a vendor-name match exists."""
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
    # "VP Finance": "finance" is a substring → matches finance keyword → pricing boost +3
    # pricing_reality_check: vendor name match (+3) + role boost (+3) = 6
    # vendor_showdown: vendor name match (+3) only = 3
    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        vendor_name="Zendesk",
        contact_role="VP Finance",
        limit=3,
    )

    assert len(posts) == 2
    # Finance role should boost pricing_reality_check above vendor_showdown
    assert posts[0]["topic_type"] == "pricing_reality_check"


@pytest.mark.asyncio
async def test_no_contact_role_does_not_change_ranking():
    """Without a contact_role, the standard scoring order is preserved."""
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
    # alternative_vendor match (+5) beats plain name match (+3), so showdown wins
    posts_no_role = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        vendor_name="Zendesk",
        alternative_vendors=["Freshdesk"],
        contact_role=None,
        limit=3,
    )
    assert posts_no_role[0]["topic_type"] == "vendor_showdown"


# ---------------------------------------------------------------------------
# Diminishing-returns pain scoring tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_diminishing_returns_multi_pain_post_scores_higher():
    """A post matching 2 pains should outscore a post matching only 1."""
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
    # First post matches both "pricing" and "support" (score 4+2=6)
    # Second post matches only "support" (score 4)
    posts = await fetch_relevant_blog_posts(
        _Pool(rows),
        pipeline="b2b",
        pain_categories=["pricing", "support"],
        limit=3,
    )

    assert len(posts) == 2
    assert posts[0]["url"].endswith("/blog/zendesk-pricing-2026-03")


@pytest.mark.asyncio
async def test_single_pain_match_without_break():
    """Passing many pains does not break when only one matches (no break regression)."""
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
    # Should still return the post (vendor name match) without raising
    assert len(posts) == 1
