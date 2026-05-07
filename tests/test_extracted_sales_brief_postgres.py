from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.sales_brief_postgres import (
    PostgresSalesBriefRepository,
)
from extracted_content_pipeline.sales_brief_ports import (
    SalesBriefDraft,
    SalesBriefSection,
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


def _draft() -> SalesBriefDraft:
    return SalesBriefDraft(
        target_id="opp-42",
        target_mode="opportunity",
        brief_type="pre_call",
        title="Pre-call brief: Acme renewal",
        headline="Renewal Q3 -- 90-day pressure window opens this week",
        sections=(
            SalesBriefSection(
                id="account_context",
                title="Account Context",
                body_markdown="Mid-market SaaS, Series C, 350 seats.",
                metadata={"order": 1},
            ),
            SalesBriefSection(
                id="signals",
                title="Recent Signals",
                body_markdown="Two competitor evals in last 30 days.",
                claim_ids=("c1", "c2"),
                evidence_ids=("e1",),
                metadata={"order": 2},
            ),
            SalesBriefSection(
                id="talking_points",
                title="Talking Points",
                body_markdown="Lead with renewal-pressure framing.",
                metadata={"order": 3},
            ),
        ),
        reference_ids=("r1", "r2"),
        metadata={"generation_model": "fake-llm", "confidence": 0.82},
    )


@pytest.mark.asyncio
async def test_save_drafts_persists_each_draft_and_returns_ids() -> None:
    pool = _Pool()
    pool.fetchval_results = ["sb-uuid-1"]
    repo = PostgresSalesBriefRepository(pool)

    saved = await repo.save_drafts([_draft()], scope=TenantScope(account_id="acct-1"))

    assert saved == ("sb-uuid-1",)
    assert len(pool.fetchval_calls) == 1
    args = pool.fetchval_calls[0]["args"]
    # account_id, target_id, target_mode, brief_type, title, headline,
    # sections, reference_ids, metadata
    assert args[0] == "acct-1"
    assert args[1] == "opp-42"
    assert args[2] == "opportunity"
    assert args[3] == "pre_call"
    assert args[4] == "Pre-call brief: Acme renewal"
    assert args[5].startswith("Renewal Q3")
    sections_payload = json.loads(args[6])
    assert [s["id"] for s in sections_payload] == [
        "account_context",
        "signals",
        "talking_points",
    ]
    assert sections_payload[1]["claim_ids"] == ["c1", "c2"]
    assert sections_payload[1]["evidence_ids"] == ["e1"]
    reference_ids_payload = json.loads(args[7])
    assert reference_ids_payload == ["r1", "r2"]
    metadata_payload = json.loads(args[8])
    assert metadata_payload["target_id"] == "opp-42"
    assert metadata_payload["target_mode"] == "opportunity"
    assert metadata_payload["scope"]["account_id"] == "acct-1"
    assert metadata_payload["confidence"] == 0.82


@pytest.mark.asyncio
async def test_save_drafts_handles_empty_account_id_with_default_scope() -> None:
    pool = _Pool()
    pool.fetchval_results = ["sb-uuid-2"]
    repo = PostgresSalesBriefRepository(pool)

    await repo.save_drafts([_draft()], scope=TenantScope())

    assert pool.fetchval_calls[0]["args"][0] == ""


@pytest.mark.asyncio
async def test_save_drafts_coerces_dict_sections_via_helper() -> None:
    pool = _Pool()
    pool.fetchval_results = ["sb-uuid-3"]
    repo = PostgresSalesBriefRepository(pool)

    draft = SalesBriefDraft(
        target_id="opp-1",
        target_mode="opportunity",
        brief_type="pre_call",
        title="title",
        headline="headline",
        sections=(
            {"id": "raw_section", "title": "From Dict", "body_markdown": "body"},
        ),  # type: ignore[arg-type]
    )

    await repo.save_drafts([draft], scope=TenantScope())

    sections_payload = json.loads(pool.fetchval_calls[0]["args"][6])
    assert sections_payload[0]["id"] == "raw_section"
    assert sections_payload[0]["title"] == "From Dict"


@pytest.mark.asyncio
async def test_list_drafts_filters_by_status_target_mode_and_brief_type() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "target_id": "opp-42",
            "target_mode": "opportunity",
            "brief_type": "pre_call",
            "title": "Pre-call brief: Acme",
            "headline": "Renewal pressure window opens",
            "sections": json.dumps([
                {
                    "id": "account_context",
                    "title": "Account Context",
                    "body_markdown": "Mid-market SaaS",
                    "claim_ids": [],
                    "evidence_ids": [],
                    "metadata": {"order": 1},
                }
            ]),
            "reference_ids": json.dumps(["r1"]),
            "metadata": json.dumps({"confidence": 0.7}),
        }
    ]
    repo = PostgresSalesBriefRepository(pool)

    drafts = await repo.list_drafts(
        scope=TenantScope(account_id="acct-1"),
        status="draft",
        target_mode="opportunity",
        brief_type="pre_call",
        limit=20,
    )

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.target_id == "opp-42"
    assert draft.brief_type == "pre_call"
    assert draft.title == "Pre-call brief: Acme"
    assert draft.headline.startswith("Renewal pressure")
    assert draft.sections[0].id == "account_context"
    assert draft.reference_ids == ("r1",)
    assert draft.metadata == {"confidence": 0.7}

    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "account_id = $1" in sql
    assert "status = $2" in sql
    assert "target_mode = $3" in sql
    assert "brief_type = $4" in sql
    assert "LIMIT $5" in sql
    assert args == ("acct-1", "draft", "opportunity", "pre_call", 20)


@pytest.mark.asyncio
async def test_list_drafts_handles_pre_decoded_jsonb_columns() -> None:
    """asyncpg with the json codec installed delivers JSONB pre-decoded."""
    pool = _Pool()
    pool.fetch_rows = [
        {
            "target_id": "opp-1",
            "target_mode": "opportunity",
            "brief_type": "pre_call",
            "title": "title",
            "headline": "headline",
            "sections": [
                {"id": "s1", "title": "T", "body_markdown": "B", "claim_ids": [], "evidence_ids": []}
            ],  # already a list
            "reference_ids": ["r1", "r2"],  # already a list
            "metadata": {"key": "value"},  # already a dict
        }
    ]
    repo = PostgresSalesBriefRepository(pool)

    drafts = await repo.list_drafts(scope=TenantScope())

    assert drafts[0].sections[0].id == "s1"
    assert drafts[0].reference_ids == ("r1", "r2")
    assert drafts[0].metadata == {"key": "value"}


@pytest.mark.asyncio
async def test_update_status_returns_true_on_hit_and_runs_scoped_update() -> None:
    pool = _Pool()
    pool.execute_result = "UPDATE 1"
    repo = PostgresSalesBriefRepository(pool)

    hit = await repo.update_status(
        "sb-uuid-1",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert hit is True
    assert len(pool.execute_calls) == 1
    args = pool.execute_calls[0]["args"]
    assert args == ("sb-uuid-1", "approved", "acct-1")
    sql = pool.execute_calls[0]["query"]
    assert "UPDATE sales_briefs" in sql
    assert "account_id = $3" in sql


@pytest.mark.asyncio
async def test_update_status_returns_false_on_miss() -> None:
    """Wrong brief_id or wrong tenant returns False so callers can branch."""
    pool = _Pool()
    pool.execute_result = "UPDATE 0"
    repo = PostgresSalesBriefRepository(pool)

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
    repo = PostgresSalesBriefRepository(pool)

    hit = await repo.update_status("any-id", "approved", scope=TenantScope())
    assert hit is True


@pytest.mark.asyncio
async def test_save_drafts_rejects_non_mapping_non_section_input_with_clear_error() -> None:
    """A host passing a string / number / None as a section item gets a clear TypeError."""
    pool = _Pool()
    pool.fetchval_results = ["sb-uuid-x"]
    repo = PostgresSalesBriefRepository(pool)

    bad_draft = SalesBriefDraft(
        target_id="opp-1",
        target_mode="opportunity",
        brief_type="pre_call",
        title="title",
        headline="headline",
        sections=("not_a_section_or_mapping",),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="SalesBriefDraft.sections entries must be"):
        await repo.save_drafts([bad_draft], scope=TenantScope())


@pytest.mark.asyncio
async def test_save_drafts_returns_empty_tuple_for_empty_input() -> None:
    pool = _Pool()
    repo = PostgresSalesBriefRepository(pool)

    saved = await repo.save_drafts([], scope=TenantScope(account_id="acct-1"))

    assert saved == ()
    assert pool.fetchval_calls == []
