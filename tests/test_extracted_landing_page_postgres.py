from __future__ import annotations

import json
import os
from uuid import uuid4

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.landing_page_postgres import (
    LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY,
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


class _PublicLandingPagePool(_Pool):
    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        landing_page_id = args[0] if args else None
        if "status = 'approved'" not in str(query):
            return self.fetch_rows
        return [
            row for row in self.fetch_rows
            if row.get("id") == landing_page_id and row.get("status") == "approved"
        ]


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
async def test_get_draft_filters_by_id_and_scope() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "status": "draft",
            "campaign_name": "acme",
            "persona": "vp",
            "value_prop": "vp",
            "title": "title",
            "slug": "slug",
            "hero": {"headline": "Hero text"},
            "sections": [{"id": "s1", "title": "T", "body_markdown": "B"}],
            "cta": {"label": "L"},
            "meta": {"title_tag": "tt"},
            "reference_ids": ["r1"],
            "metadata": {"key": "value"},
        }
    ]
    repo = PostgresLandingPageRepository(pool)

    draft = await repo.get_draft(
        "11111111-1111-1111-1111-111111111111",
        scope=TenantScope(account_id="acct-1"),
    )

    assert draft is not None
    assert draft.id == "11111111-1111-1111-1111-111111111111"
    assert draft.status == "draft"
    assert draft.hero["headline"] == "Hero text"
    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "FROM landing_pages" in sql
    assert "id = $1" in sql
    assert "account_id = $2" in sql
    assert args == ("11111111-1111-1111-1111-111111111111", "acct-1")


@pytest.mark.asyncio
async def test_get_draft_returns_none_on_miss() -> None:
    pool = _Pool()
    repo = PostgresLandingPageRepository(pool)

    draft = await repo.get_draft(
        "11111111-1111-1111-1111-111111111111",
        scope=TenantScope(account_id="acct-1"),
    )

    assert draft is None


@pytest.mark.asyncio
async def test_update_draft_updates_editable_fields_and_returns_row() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "status": "draft",
            "campaign_name": "acme",
            "persona": "vp",
            "value_prop": "vp",
            "title": "updated title",
            "slug": "updated-slug",
            "hero": {"headline": "Updated hero"},
            "sections": [{"id": "s1", "title": "T", "body_markdown": "B"}],
            "cta": {"label": "Book"},
            "meta": {"title_tag": "Updated"},
            "reference_ids": ["r1"],
            "metadata": {"key": "value"},
        }
    ]
    repo = PostgresLandingPageRepository(pool)
    draft = LandingPageDraft(
        campaign_name="acme",
        persona="vp",
        value_prop="vp",
        title="updated title",
        slug="updated-slug",
        hero={"headline": "Updated hero"},
        sections=(LandingPageSection(id="s1", title="T", body_markdown="B"),),
        cta={"label": "Book"},
        meta={"title_tag": "Updated"},
        reference_ids=("r1",),
        metadata={"key": "value"},
    )

    updated = await repo.update_draft(
        "11111111-1111-1111-1111-111111111111",
        draft,
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated is not None
    assert updated.title == "updated title"
    assert updated.slug == "updated-slug"
    assert updated.status == "draft"
    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "UPDATE landing_pages" in sql
    assert "status <> 'approved'" in sql
    assert "metadata = $10::jsonb" in sql
    assert "$11::text IS NULL" in sql
    assert "RETURNING id" in sql
    assert args[0:4] == (
        "11111111-1111-1111-1111-111111111111",
        "acct-1",
        "updated title",
        "updated-slug",
    )
    assert json.loads(args[4]) == {"headline": "Updated hero"}
    assert json.loads(args[5])[0]["id"] == "s1"
    assert json.loads(args[6]) == {"label": "Book"}
    assert json.loads(args[7]) == {"title_tag": "Updated"}
    assert json.loads(args[8]) == ["r1"]
    assert json.loads(args[9]) == {"key": "value"}
    assert args[10] is None


@pytest.mark.asyncio
async def test_update_draft_can_be_fenced_by_repair_claim_token() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "status": "draft",
            "campaign_name": "acme",
            "persona": "vp",
            "value_prop": "vp",
            "title": "updated title",
            "slug": "updated-slug",
            "hero": {"headline": "Updated hero"},
            "sections": [{"id": "s1", "title": "T", "body_markdown": "B"}],
            "cta": {"label": "Book"},
            "meta": {"title_tag": "Updated"},
            "reference_ids": ["r1"],
            "metadata": {
                LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY: {
                    "token": "claim-token",
                    "expires_at_epoch": 9999999999,
                }
            },
        }
    ]
    repo = PostgresLandingPageRepository(pool)

    updated = await repo.update_draft(
        "11111111-1111-1111-1111-111111111111",
        _draft(),
        scope=TenantScope(account_id="acct-1"),
        repair_claim_token="claim-token",
    )

    assert updated is not None
    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "metadata #>>" in sql
    assert LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY in sql
    assert args[10] == "claim-token"


@pytest.mark.asyncio
async def test_update_draft_returns_none_on_miss_or_approved_row() -> None:
    pool = _Pool()
    repo = PostgresLandingPageRepository(pool)

    updated = await repo.update_draft(
        "11111111-1111-1111-1111-111111111111",
        _draft(),
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated is None
    assert len(pool.fetch_calls) == 1


@pytest.mark.asyncio
async def test_claim_repair_sets_tokenized_metadata_and_returns_row() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "status": "draft",
            "campaign_name": "acme",
            "persona": "vp",
            "value_prop": "vp",
            "title": "title",
            "slug": "slug",
            "hero": {"headline": "Hero"},
            "sections": [{"id": "s1", "title": "T", "body_markdown": "B"}],
            "cta": {"label": "Book"},
            "meta": {"title_tag": "Title"},
            "reference_ids": ["r1"],
            "metadata": {
                LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY: {
                    "token": "claim-token",
                    "expires_at_epoch": 9999999999,
                }
            },
        }
    ]
    repo = PostgresLandingPageRepository(pool)

    claimed = await repo.claim_repair(
        "11111111-1111-1111-1111-111111111111",
        token="claim-token",
        scope=TenantScope(account_id="acct-1"),
    )

    assert claimed is not None
    assert claimed.metadata[LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY]["token"] == (
        "claim-token"
    )
    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "UPDATE landing_pages" in sql
    assert "jsonb_set" in sql
    assert "status <> 'approved'" in sql
    assert "expires_at_epoch" in sql
    assert args[0:2] == (
        "11111111-1111-1111-1111-111111111111",
        "acct-1",
    )
    claim_payload = json.loads(args[2])
    assert claim_payload["token"] == "claim-token"
    assert claim_payload["expires_at_epoch"] > claim_payload["claimed_at_epoch"]
    assert args[3] == "claim-token"


@pytest.mark.asyncio
async def test_claim_repair_returns_none_when_claim_update_misses() -> None:
    pool = _Pool()
    repo = PostgresLandingPageRepository(pool)

    claimed = await repo.claim_repair(
        "11111111-1111-1111-1111-111111111111",
        token="claim-token",
        scope=TenantScope(account_id="acct-1"),
    )

    assert claimed is None
    assert len(pool.fetch_calls) == 1


@pytest.mark.asyncio
async def test_release_repair_removes_matching_tokenized_claim() -> None:
    pool = _Pool()
    repo = PostgresLandingPageRepository(pool)

    released = await repo.release_repair(
        "11111111-1111-1111-1111-111111111111",
        token="claim-token",
        scope=TenantScope(account_id="acct-1"),
    )

    assert released is True
    sql = pool.execute_calls[0]["query"]
    args = pool.execute_calls[0]["args"]
    assert "UPDATE landing_pages" in sql
    assert f"- '{LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY}'" in sql
    assert "metadata #>>" in sql
    assert args == (
        "11111111-1111-1111-1111-111111111111",
        "acct-1",
        "claim-token",
    )


@pytest.mark.asyncio
async def test_release_repair_returns_false_when_token_misses() -> None:
    pool = _Pool()
    pool.execute_result = "UPDATE 0"
    repo = PostgresLandingPageRepository(pool)

    released = await repo.release_repair(
        "11111111-1111-1111-1111-111111111111",
        token="wrong-token",
        scope=TenantScope(account_id="acct-1"),
    )

    assert released is False
    assert len(pool.execute_calls) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_landing_page_repair_claim_contract_against_postgres() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("EXTRACTED_DATABASE_URL or DATABASE_URL is required")

    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    repo = PostgresLandingPageRepository(pool)
    landing_page_id = str(uuid4())
    scope = TenantScope(account_id=f"acct-{uuid4().hex}")
    other_scope = TenantScope(account_id=f"acct-{uuid4().hex}")

    try:
        await pool.execute(
            """
            INSERT INTO landing_pages (
                id, account_id, campaign_name, persona, value_prop, title, slug,
                hero, sections, cta, meta, reference_ids, metadata, status
            )
            VALUES (
                $1::uuid, $2, $3, $4, $5, $6, $7,
                $8::jsonb, $9::jsonb, $10::jsonb, $11::jsonb,
                $12::jsonb, $13::jsonb, 'draft'
            )
            """,
            landing_page_id,
            scope.account_id,
            "repair-contract",
            "Owner",
            "Clear support gaps",
            "Repair Contract",
            "repair-contract",
            json.dumps({"headline": "Repair contract"}),
            json.dumps([{"id": "s1", "title": "Problem", "body_markdown": "Body"}]),
            json.dumps({"label": "Book", "url": "https://example.com"}),
            json.dumps({"title_tag": "Repair Contract"}),
            json.dumps(["r1"]),
            json.dumps({}),
        )

        first = await repo.claim_repair(
            landing_page_id,
            token="token-a",
            scope=scope,
        )
        assert first is not None

        blocked = await repo.claim_repair(
            landing_page_id,
            token="token-b",
            scope=scope,
        )
        assert blocked is None

        same_token = await repo.claim_repair(
            landing_page_id,
            token="token-a",
            scope=scope,
        )
        assert same_token is not None

        await pool.execute(
            f"""
            UPDATE landing_pages
               SET metadata = jsonb_set(
                       metadata,
                       '{{{LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY},expires_at_epoch}}',
                       '0'::jsonb,
                       true
                   )
             WHERE id = $1::uuid
            """,
            landing_page_id,
        )
        stolen = await repo.claim_repair(
            landing_page_id,
            token="token-b",
            scope=scope,
        )
        assert stolen is not None

        cross_tenant = await repo.claim_repair(
            landing_page_id,
            token="other-tenant",
            scope=other_scope,
        )
        assert cross_tenant is None

        stale_update = await repo.update_draft(
            landing_page_id,
            _draft(),
            scope=scope,
            repair_claim_token="token-a",
        )
        assert stale_update is None

        wrong_release = await repo.release_repair(
            landing_page_id,
            token="token-a",
            scope=scope,
        )
        assert wrong_release is False
        persisted_token = await pool.fetchval(
            f"""
            SELECT metadata #>>
                   '{{{LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY},token}}'
              FROM landing_pages
             WHERE id = $1::uuid
            """,
            landing_page_id,
        )
        assert persisted_token == "token-b"

        right_release = await repo.release_repair(
            landing_page_id,
            token="token-b",
            scope=scope,
        )
        assert right_release is True
        claim_exists = await pool.fetchval(
            f"""
            SELECT COALESCE(metadata, '{{}}'::jsonb)
                   ? '{LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY}'
              FROM landing_pages
             WHERE id = $1::uuid
            """,
            landing_page_id,
        )
        assert claim_exists is False
    finally:
        await pool.execute(
            "DELETE FROM landing_pages WHERE id = $1::uuid",
            landing_page_id,
        )
        await pool.close()


@pytest.mark.asyncio
async def test_get_public_approved_draft_filters_by_id_and_approved_status() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "status": "approved",
            "campaign_name": "acme",
            "persona": "vp",
            "value_prop": "vp",
            "title": "title",
            "slug": "slug",
            "hero": {"headline": "Hero text"},
            "sections": [{"id": "s1", "title": "T", "body_markdown": "B"}],
            "cta": {"label": "L"},
            "meta": {"title_tag": "tt"},
            "reference_ids": ["r1"],
            "metadata": {"key": "value"},
        }
    ]
    repo = PostgresLandingPageRepository(pool)

    draft = await repo.get_public_approved_draft(
        "11111111-1111-1111-1111-111111111111"
    )

    assert draft is not None
    assert draft.id == "11111111-1111-1111-1111-111111111111"
    assert draft.status == "approved"
    assert draft.slug == "slug"
    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "FROM landing_pages" in sql
    assert "id = $1" in sql
    assert "status = 'approved'" in sql
    assert args == ("11111111-1111-1111-1111-111111111111",)


@pytest.mark.asyncio
async def test_get_public_approved_draft_returns_none_on_miss() -> None:
    pool = _Pool()
    repo = PostgresLandingPageRepository(pool)

    draft = await repo.get_public_approved_draft(
        "11111111-1111-1111-1111-111111111111"
    )

    assert draft is None


@pytest.mark.asyncio
async def test_get_public_approved_draft_hides_non_approved_row() -> None:
    pool = _PublicLandingPagePool()
    pool.fetch_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "status": "draft",
            "campaign_name": "acme",
            "persona": "vp",
            "value_prop": "vp",
            "title": "title",
            "slug": "slug",
            "hero": {"headline": "Hero text"},
            "sections": [{"id": "s1", "title": "T", "body_markdown": "B"}],
            "cta": {"label": "L"},
            "meta": {"title_tag": "tt"},
            "reference_ids": ["r1"],
            "metadata": {"key": "value"},
        }
    ]
    repo = PostgresLandingPageRepository(pool)

    draft = await repo.get_public_approved_draft(
        "11111111-1111-1111-1111-111111111111"
    )

    assert draft is None
    sql = pool.fetch_calls[0]["query"]
    assert "status = 'approved'" in sql


@pytest.mark.asyncio
async def test_list_public_sitemap_candidates_uses_public_projection() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "status": "approved",
            "campaign_name": "acme",
            "persona": "vp",
            "value_prop": "vp",
            "title": "title",
            "slug": "slug",
            "hero": {"headline": "Hero text"},
            "sections": [{"id": "s1", "title": "T", "body_markdown": "B"}],
            "cta": {"label": "L"},
            "meta": {"title_tag": "tt"},
            "reference_ids": ["r1"],
            "metadata": {"key": "value"},
        }
    ]
    repo = PostgresLandingPageRepository(pool)

    candidates = await repo.list_public_sitemap_candidates()

    assert len(candidates) == 1
    assert candidates[0].status == "approved"
    assert candidates[0].slug == "slug"
    assert candidates[0].to_policy_draft().metadata == {}
    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "FROM landing_pages" in sql
    assert "status = 'approved'" in sql
    assert "ORDER BY updated_at DESC, created_at DESC" in sql
    assert "metadata" not in sql
    assert "LIMIT" not in sql
    assert args == ()


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
