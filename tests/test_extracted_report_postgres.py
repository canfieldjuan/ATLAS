from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.report_ports import ReportDraft, ReportSection
from extracted_content_pipeline.report_postgres import PostgresReportRepository


class _Pool:
    def __init__(self):
        self.fetchval_results: list[object] = []
        self.fetch_rows: list[dict] = []
        self.fetchval_calls: list[dict] = []
        self.fetch_calls: list[dict] = []
        self.execute_calls: list[dict] = []

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": query, "args": args})
        return self.fetchval_results.pop(0)

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": query, "args": args})
        return "OK"


def _draft() -> ReportDraft:
    return ReportDraft(
        target_id="acme",
        target_mode="vendor",
        report_type="vendor_pressure",
        title="Acme: Q3 Pressure Report",
        summary="Pricing renewal pressure dominates the displacement signal.",
        sections=(
            ReportSection(
                id="executive_summary",
                title="Executive Summary",
                body_markdown="Renewal pricing is the dominant churn driver.",
                claim_ids=("c1",),
                evidence_ids=("r1", "r2"),
            ),
            ReportSection(
                id="drivers",
                title="Pressure Drivers",
                body_markdown="Onboarding friction is a secondary driver.",
                claim_ids=("c2",),
                evidence_ids=("t9",),
            ),
        ),
        reference_ids=("r1", "r2", "t9"),
        metadata={"generation_model": "fake-llm", "confidence": 0.84},
    )


@pytest.mark.asyncio
async def test_save_drafts_persists_each_draft_and_returns_ids() -> None:
    pool = _Pool()
    pool.fetchval_results = ["report-uuid-1"]
    repo = PostgresReportRepository(pool)

    saved = await repo.save_drafts([_draft()], scope=TenantScope(account_id="acct-1"))

    assert saved == ("report-uuid-1",)
    assert len(pool.fetchval_calls) == 1
    args = pool.fetchval_calls[0]["args"]
    # Argument order matches the INSERT param list:
    # account_id, target_id, target_mode, report_type, title, summary, sections, reference_ids, metadata
    assert args[0] == "acct-1"
    assert args[1] == "acme"
    assert args[2] == "vendor"
    assert args[3] == "vendor_pressure"
    assert args[4] == "Acme: Q3 Pressure Report"
    assert "Pricing renewal" in args[5]
    sections_payload = json.loads(args[6])
    assert [s["id"] for s in sections_payload] == ["executive_summary", "drivers"]
    assert sections_payload[0]["evidence_ids"] == ["r1", "r2"]
    reference_ids_payload = json.loads(args[7])
    assert reference_ids_payload == ["r1", "r2", "t9"]
    metadata_payload = json.loads(args[8])
    assert metadata_payload["target_id"] == "acme"
    assert metadata_payload["scope"]["account_id"] == "acct-1"
    assert metadata_payload["confidence"] == 0.84


@pytest.mark.asyncio
async def test_save_drafts_handles_empty_account_id_with_default_scope() -> None:
    pool = _Pool()
    pool.fetchval_results = ["report-uuid-2"]
    repo = PostgresReportRepository(pool)

    await repo.save_drafts([_draft()], scope=TenantScope())

    assert pool.fetchval_calls[0]["args"][0] == ""


@pytest.mark.asyncio
async def test_save_drafts_handles_dict_section_input_via_coercion() -> None:
    """Hosts may pass dict-shaped sections; the repo coerces them defensively."""
    pool = _Pool()
    pool.fetchval_results = ["report-uuid-3"]
    repo = PostgresReportRepository(pool)

    draft = ReportDraft(
        target_id="acme",
        target_mode="vendor",
        report_type="vendor_pressure",
        title="title",
        summary="summary",
        sections=(
            {
                "id": "raw_section",
                "title": "From Dict",
                "body_markdown": "body",
                "claim_ids": ["c1"],
                "evidence_ids": ["r1"],
            },
        ),  # type: ignore[arg-type]
    )

    await repo.save_drafts([draft], scope=TenantScope())

    sections_payload = json.loads(pool.fetchval_calls[0]["args"][6])
    assert sections_payload[0]["id"] == "raw_section"
    assert sections_payload[0]["claim_ids"] == ["c1"]


@pytest.mark.asyncio
async def test_list_drafts_filters_by_status_and_target_mode() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "target_id": "acme",
            "target_mode": "vendor",
            "report_type": "vendor_pressure",
            "title": "Acme report",
            "summary": "summary text",
            "sections": json.dumps([
                {
                    "id": "summary",
                    "title": "Executive Summary",
                    "body_markdown": "body",
                    "claim_ids": ["c1"],
                    "evidence_ids": ["r1"],
                }
            ]),
            "reference_ids": json.dumps(["r1"]),
            "metadata": json.dumps({"confidence": 0.7}),
        }
    ]
    repo = PostgresReportRepository(pool)

    drafts = await repo.list_drafts(
        scope=TenantScope(account_id="acct-1"),
        status="draft",
        target_mode="vendor",
        report_type="vendor_pressure",
        limit=20,
    )

    assert len(drafts) == 1
    assert drafts[0].target_id == "acme"
    assert drafts[0].sections[0].id == "summary"
    assert drafts[0].sections[0].claim_ids == ("c1",)
    assert drafts[0].reference_ids == ("r1",)
    assert drafts[0].metadata == {"confidence": 0.7}

    sql = pool.fetch_calls[0]["query"]
    args = pool.fetch_calls[0]["args"]
    assert "account_id = $1" in sql
    assert "status = $2" in sql
    assert "target_mode = $3" in sql
    assert "report_type = $4" in sql
    assert "LIMIT $5" in sql
    assert args == ("acct-1", "draft", "vendor", "vendor_pressure", 20)


@pytest.mark.asyncio
async def test_list_drafts_handles_pre_decoded_jsonb_columns() -> None:
    """asyncpg with the json codec installed delivers JSONB pre-decoded."""
    pool = _Pool()
    pool.fetch_rows = [
        {
            "target_id": "acme",
            "target_mode": "vendor",
            "report_type": "vendor_pressure",
            "title": "title",
            "summary": "summary",
            "sections": [{"id": "s1", "title": "T", "body_markdown": "B"}],  # already a list
            "reference_ids": ["r1", "r2"],  # already a list
            "metadata": {"key": "value"},  # already a dict
        }
    ]
    repo = PostgresReportRepository(pool)

    drafts = await repo.list_drafts(scope=TenantScope())

    assert drafts[0].sections[0].id == "s1"
    assert drafts[0].reference_ids == ("r1", "r2")
    assert drafts[0].metadata == {"key": "value"}


@pytest.mark.asyncio
async def test_update_status_runs_scoped_update() -> None:
    pool = _Pool()
    repo = PostgresReportRepository(pool)

    await repo.update_status(
        "report-uuid-1",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert len(pool.execute_calls) == 1
    args = pool.execute_calls[0]["args"]
    assert args == ("report-uuid-1", "approved", "acct-1")
    sql = pool.execute_calls[0]["query"]
    assert "UPDATE reports" in sql
    assert "account_id = $3" in sql


@pytest.mark.asyncio
async def test_save_drafts_returns_empty_tuple_for_empty_input() -> None:
    pool = _Pool()
    repo = PostgresReportRepository(pool)

    saved = await repo.save_drafts([], scope=TenantScope(account_id="acct-1"))

    assert saved == ()
    assert pool.fetchval_calls == []
