import pytest

from atlas_brain.autonomous.tasks._blog_matching import fetch_relevant_blog_posts


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
