from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.landing_page_postgres import (
    PostgresLandingPageRepository,
)
from extracted_content_pipeline.landing_page_ports import (
    LandingPageDraft,
    LandingPageSection,
)


class _Pool:
    def __init__(self):
        self.fetchval_results: list[object] = []
        self.fetch_rows: list[dict] = []
        self.fetchval_calls: list[dict] = []
        self.fetch_calls: list[dict] = []
        self.execute_calls: list[dict] = []
        # asyncpg returns command tags like "UPDATE 1" / "UPDATE 0";
        # tests can override this to simulate misses.
        self.execute_result: object = "UPDATE 1"

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": query, "args": args})
        return self.fetchval_results.pop(0)

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": query, "args": args})
        return self.execute_result


def _draft() -> LandingPageDraft:
    return LandingPageDraft(
        campaign_name="acme-q3-launch",
        persona="VP Engineering at mid-market SaaS",
        value_prop="Cut renewal pricing leakage by 40%",
        title="Acme Q3: Stop Renewal Surprises",
        slug="acme-q3-launch",
        hero={
            "headline": "Stop renewal surprises",
            "subheadline": "Acme catches pricing pressure before it becomes churn",
            "cta_label": "Book a 15-min demo",
            "cta_url": "/demo",
        },
        sections=(
            LandingPageSection(
                id="problem",
                title="The Problem",
                body_markdown="Renewal pricing is the #1 driver of unplanned churn.",
                metadata={"order": 1},
            ),
            LandingPageSection(
                id="solution",
                title="How Acme Helps",
                body_markdown="We surface pressure signals 90 days early.",
                metadata={"order": 2},
            ),
        ),
        cta={"label": "Book a 15-min demo", "url": "/demo", "variant": "primary"},
        meta={
            "title_tag": "Stop Renewal Surprises | Acme",
            "description": "Acme catches renewal pressure 90 days early.",
            "og_image_url": "https://cdn.acme.com/og/q3.png",
        },
        reference_ids=("r1", "r2"),
        metadata={"generation_model": "fake-llm", "confidence": 0.84},
    )


@pytest.mark.asyncio
async def test_save_drafts_persists_each_draft_and_returns_ids() -> None:
    pool = _Pool()
    pool.fetchval_results = ["lp-uuid-1"]
    repo = PostgresLandingPageRepository(pool)

    saved = await repo.save_drafts([_draft()], scope=TenantScope(account_id="acct-1"))

    assert saved == ("lp-uuid-1",)
    assert len(pool.fetchval_calls) == 1
    args = pool.fetchval_calls[0]["args"]
    # account_id, campaign_name, persona, value_prop, title, slug, hero, sections, cta, meta, reference_ids, metadata
    assert args[0] == "acct-1"
    assert args[1] == "acme-q3-launch"
    assert args[2] == "VP Engineering at mid-market SaaS"
    assert args[3] == "Cut renewal pricing leakage by 40%"
    assert args[4] == "Acme Q3: Stop Renewal Surprises"
    assert args[5] == "acme-q3-launch"
    hero_payload = json.loads(args[6])
    assert hero_payload["headline"] == "Stop renewal surprises"
    sections_payload = json.loads(args[7])
    assert [s["id"] for s in sections_payload] == ["problem", "solution"]
    assert sections_payload[0]["metadata"] == {"order": 1}
    cta_payload = json.loads(args[8])
    assert cta_payload["label"] == "Book a 15-min demo"
    meta_payload = json.loads(args[9])
    assert meta_payload["title_tag"].startswith("Stop")
    reference_ids_payload = json.loads(args[10])
    assert reference_ids_payload == ["r1", "r2"]
    metadata_payload = json.loads(args[11])
    assert metadata_payload["campaign_name"] == "acme-q3-launch"
    assert metadata_payload["scope"]["account_id"] == "acct-1"
    assert metadata_payload["confidence"] == 0.84


@pytest.mark.asyncio
async def test_save_drafts_handles_empty_account_id_with_default_scope() -> None:
    pool = _Pool()
    pool.fetchval_results = ["lp-uuid-2"]
    repo = PostgresLandingPageRepository(pool)

    await repo.save_drafts([_draft()], scope=TenantScope())

    assert pool.fetchval_calls[0]["args"][0] == ""


@pytest.mark.asyncio
async def test_save_drafts_coerces_dict_sections_via_helper() -> None:
    pool = _Pool()
    pool.fetchval_results = ["lp-uuid-3"]
    repo = PostgresLandingPageRepository(pool)

    draft = LandingPageDraft(
        campaign_name="acme",
        persona="",
        value_prop="",
        title="title",
        slug="slug",
        sections=(
            {"id": "raw_section", "title": "From Dict", "body_markdown": "body"},
        ),  # type: ignore[arg-type]
    )

    await repo.save_drafts([draft], scope=TenantScope())

    sections_payload = json.loads(pool.fetchval_calls[0]["args"][7])
    assert sections_payload[0]["id"] == "raw_section"
    assert sections_payload[0]["title"] == "From Dict"


@pytest.mark.asyncio
async def test_list_drafts_filters_by_status_campaign_and_slug() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "campaign_name": "acme-q3-launch",
            "persona": "VP Engineering",
            "value_prop": "Stop churn",
            "title": "Acme Q3",
            "slug": "acme-q3-launch",
            "hero": json.dumps({"headline": "Stop renewal surprises"}),
            "sections": json.dumps([
                {
                    "id": "problem",
                    "title": "The Problem",
                    "body_markdown": "Renewal pricing churn.",
                    "metadata": {"order": 1},
                }
            ]),
            "cta": json.dumps({"label": "Book demo", "url": "/demo"}),
            "meta": json.dumps({"title_tag": "Stop Surprises | Acme"}),
            "reference_ids": json.dumps(["r1"]),
            "metadata": json.dumps({"confidence": 0.7}),
        }
    ]
    repo = PostgresLandingPageRepository(pool)

    drafts = await repo.list_drafts(
        scope=TenantScope(account_id="acct-1"),
        status="draft",
        campaign_name="acme-q3-launch",
        slug="acme-q3-launch",
        limit=20,
    )

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.campaign_name == "acme-q3-launch"
    assert draft.title == "Acme Q3"
    assert draft.slug == "acme-q3-launch"
    assert draft.hero == {"headline": "Stop renewal surprises"}
    assert draft.sections[0].id == "problem"
    assert draft.cta["label"] == "Book demo"
    assert draft.reference_ids == ("r1",)
    assert draft.metadata == {"confidence": 0.7}

    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "account_id = $1" in sql
    assert "status = $2" in sql
    assert "campaign_name = $3" in sql
    assert "slug = $4" in sql
    assert "LIMIT $5" in sql
    assert args == ("acct-1", "draft", "acme-q3-launch", "acme-q3-launch", 20)


@pytest.mark.asyncio
async def test_list_drafts_handles_pre_decoded_jsonb_columns() -> None:
    """asyncpg with the json codec installed delivers JSONB pre-decoded."""
    pool = _Pool()
    pool.fetch_rows = [
        {
            "campaign_name": "acme",
            "persona": "vp",
            "value_prop": "vp",
            "title": "title",
            "slug": "slug",
            "hero": {"headline": "Hero text"},  # already a dict
            "sections": [{"id": "s1", "title": "T", "body_markdown": "B"}],  # already a list
            "cta": {"label": "L"},  # already a dict
            "meta": {"title_tag": "tt"},  # already a dict
            "reference_ids": ["r1", "r2"],  # already a list
            "metadata": {"key": "value"},  # already a dict
        }
    ]
    repo = PostgresLandingPageRepository(pool)

    drafts = await repo.list_drafts(scope=TenantScope())

    assert drafts[0].hero == {"headline": "Hero text"}
    assert drafts[0].sections[0].id == "s1"
    assert drafts[0].cta == {"label": "L"}
    assert drafts[0].meta == {"title_tag": "tt"}
    assert drafts[0].reference_ids == ("r1", "r2")
    assert drafts[0].metadata == {"key": "value"}


@pytest.mark.asyncio
async def test_update_status_returns_true_on_hit_and_runs_scoped_update() -> None:
    pool = _Pool()
    pool.execute_result = "UPDATE 1"
    repo = PostgresLandingPageRepository(pool)

    hit = await repo.update_status(
        "lp-uuid-1",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert hit is True
    assert len(pool.execute_calls) == 1
    args = pool.execute_calls[0]["args"]
    assert args == ("lp-uuid-1", "approved", "acct-1")
    sql = pool.execute_calls[0]["query"]
    assert "UPDATE landing_pages" in sql
    assert "account_id = $3" in sql


@pytest.mark.asyncio
async def test_update_status_returns_false_on_miss() -> None:
    """Wrong landing_page_id or wrong tenant returns False so callers can branch."""
    pool = _Pool()
    pool.execute_result = "UPDATE 0"
    repo = PostgresLandingPageRepository(pool)

    hit = await repo.update_status(
        "missing-id",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert hit is False


@pytest.mark.asyncio
async def test_update_status_treats_non_asyncpg_command_tags_as_success() -> None:
    """Test fakes / alt drivers that return 'OK' or None don't crash; default to True."""
    pool = _Pool()
    pool.execute_result = None
    repo = PostgresLandingPageRepository(pool)

    hit = await repo.update_status("any-id", "approved", scope=TenantScope())
    assert hit is True


@pytest.mark.asyncio
async def test_save_drafts_rejects_non_mapping_non_section_input_with_clear_error() -> None:
    """A host passing a string / number / None as a section item gets a clear TypeError."""
    pool = _Pool()
    pool.fetchval_results = ["lp-uuid-x"]
    repo = PostgresLandingPageRepository(pool)

    bad_draft = LandingPageDraft(
        campaign_name="acme",
        persona="",
        value_prop="",
        title="title",
        slug="slug",
        sections=("not_a_section_or_mapping",),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="LandingPageDraft.sections entries must be"):
        await repo.save_drafts([bad_draft], scope=TenantScope())


@pytest.mark.asyncio
async def test_save_drafts_returns_empty_tuple_for_empty_input() -> None:
    pool = _Pool()
    repo = PostgresLandingPageRepository(pool)

    saved = await repo.save_drafts([], scope=TenantScope(account_id="acct-1"))

    assert saved == ()
    assert pool.fetchval_calls == []
