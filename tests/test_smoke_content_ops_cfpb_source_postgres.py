from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_cfpb_source_postgres.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_cfpb_source_postgres",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class _Pool:
    def __init__(
        self,
        *,
        opportunity_rows=None,
        saved_draft_rows=None,
        existing_relations=None,
    ):
        self.opportunity_rows = list(opportunity_rows or [])
        self.saved_draft_rows = list(saved_draft_rows or [])
        self.existing_relations = set(
            existing_relations or ("campaign_opportunities", "b2b_campaigns")
        )
        self.fetch_calls = []
        self.execute_calls = []
        self.fetchval_calls = []
        self.fetchval_results = ["campaign-1", "campaign-2"]
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        if len(self.fetch_calls) == 1:
            return self.opportunity_rows
        return self.saved_draft_rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": str(query), "args": args})
        return "EXECUTE"

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": str(query), "args": args})
        if "to_regclass" in str(query):
            return args[0] if args and args[0] in self.existing_relations else None
        return self.fetchval_results.pop(0)

    async def close(self):
        self.closed = True


async def _return_pool(pool):
    return pool


def _source_row():
    return {
        "id": "cfpb-1",
        "source_id": "cfpb-1",
        "source": "cfpb",
        "source_system": "cfpb",
        "source_type": "support_ticket",
        "vendor_name": "Example Bank",
        "text": "The bank kept charging fees after I closed the account.",
        "pain_category": "Fees",
        "source_title": "Checking account - Fees",
        "source_url": "https://example.test/cfpb/cfpb-1",
    }


def _opportunity_row():
    return {
        "target_id": "cfpb-1",
        "company_name": "Acme Logistics",
        "vendor_name": "Example Bank",
        "contact_email": "ops@example.com",
        "contact_name": "Jordan Lee",
        "pain_points": ["Fees"],
        "evidence": [
            {
                "source_id": "cfpb-1",
                "source_type": "support_ticket",
                "text": "The bank kept charging fees after I closed the account.",
            }
        ],
        "raw_payload": {
            "target_id": "cfpb-1",
            "company_name": "Acme Logistics",
            "vendor_name": "Example Bank",
            "contact_email": "ops@example.com",
            "source_type": "support_ticket",
        },
    }


def _saved_draft_row(*, body=None):
    return {
        "id": "campaign-1",
        "subject": "Acme Logistics: Fees",
        "body": body or "<p>Teams evaluating Example Bank are reporting pain around Fees.</p>",
        "target_mode": "vendor_retention",
        "channel": "email_cold",
        "metadata": {
            "target_id": "cfpb-1",
            "source_opportunity": {
                "target_id": "cfpb-1",
            },
        },
    }


def _args(**overrides):
    values = {
        "company": "Example Bank",
        "product": None,
        "issue": "Fees",
        "search_term": "fees",
        "date_received_min": None,
        "date_received_max": None,
        "api_url": "https://example.test/cfpb",
        "limit": 1,
        "max_rows_scanned": 5,
        "timeout": 3.5,
        "source_system": "cfpb",
        "source_type": "support_ticket",
        "target_mode": "vendor_retention",
        "channels": "email_cold",
        "min_drafts": None,
        "allow_ingestion_warnings": False,
        "default_field": [
            "company_name=Acme Logistics",
            "contact_email=ops@example.com",
            "contact_name=Jordan Lee",
        ],
        "account_id": "acct-smoke",
        "user_id": None,
        "opportunity_table": "campaign_opportunities",
        "keep_existing_opportunities": False,
        "forbidden_phrase": list(smoke.DEFAULT_FORBIDDEN_PHRASES),
        "output_source_rows": None,
        "output_result": None,
        "json": False,
        "database_url": "postgres://example",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
async def test_cfpb_source_postgres_smoke_imports_and_persists(monkeypatch, tmp_path):
    pool = _Pool(
        opportunity_rows=[_opportunity_row()],
        saved_draft_rows=[_saved_draft_row()],
    )
    fetch_calls = []
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    def fake_fetch(**kwargs):
        fetch_calls.append(kwargs)
        return [_source_row()]

    monkeypatch.setattr(smoke, "fetch_cfpb_source_rows", fake_fetch)

    code, payload = await smoke.run_cfpb_source_postgres_smoke(
        _args(),
        source_rows_path=tmp_path / "cfpb_sources.jsonl",
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["source"] == "cfpb"
    assert payload["source_rows"] == 1
    assert payload["import"]["inserted"] == 1
    assert payload["import"]["replace_existing"] is True
    assert payload["drafts"]["generated"] == 1
    assert payload["saved_drafts"][0]["target_id"] == "cfpb-1"
    assert pool.closed is True
    assert fetch_calls[0]["api_url"] == "https://example.test/cfpb"
    assert fetch_calls[0]["timeout"] == 3.5
    assert fetch_calls[0]["source_type"] == "support_ticket"
    assert "DELETE FROM \"campaign_opportunities\"" in pool.execute_calls[0]["query"]
    assert any("INSERT INTO b2b_campaigns" in call["query"] for call in pool.fetchval_calls)
    assert pool.fetch_calls[0]["args"] == ("vendor_retention", "acct-smoke", "cfpb-1", 1)


@pytest.mark.asyncio
async def test_cfpb_source_postgres_smoke_fails_before_import_when_schema_missing(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(existing_relations={"b2b_campaigns"})
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    monkeypatch.setattr(smoke, "fetch_cfpb_source_rows", lambda **_kwargs: [_source_row()])

    code, payload = await smoke.run_cfpb_source_postgres_smoke(
        _args(),
        source_rows_path=tmp_path / "cfpb_sources.jsonl",
    )

    assert code == 1
    assert any("required Content Ops table(s) missing" in error for error in payload["errors"])
    assert any("campaign_opportunities" in error for error in payload["errors"])
    assert pool.execute_calls == []
    assert all("INSERT INTO b2b_campaigns" not in call["query"] for call in pool.fetchval_calls)


@pytest.mark.asyncio
async def test_cfpb_source_postgres_smoke_can_keep_existing_opportunities(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(
        opportunity_rows=[_opportunity_row()],
        saved_draft_rows=[_saved_draft_row()],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    monkeypatch.setattr(smoke, "fetch_cfpb_source_rows", lambda **_kwargs: [_source_row()])

    code, payload = await smoke.run_cfpb_source_postgres_smoke(
        _args(keep_existing_opportunities=True),
        source_rows_path=tmp_path / "cfpb_sources.jsonl",
    )

    assert code == 0
    assert payload["import"]["replace_existing"] is False
    assert all("DELETE FROM" not in call["query"] for call in pool.execute_calls)


@pytest.mark.asyncio
async def test_cfpb_source_postgres_smoke_fails_on_ingestion_warnings(
    monkeypatch,
    tmp_path,
):
    pool = _Pool()
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    monkeypatch.setattr(smoke, "fetch_cfpb_source_rows", lambda **_kwargs: [_source_row()])

    code, payload = await smoke.run_cfpb_source_postgres_smoke(
        _args(default_field=[]),
        source_rows_path=tmp_path / "cfpb_sources.jsonl",
    )

    assert code == 1
    assert any("ingestion inspection produced warnings" in error for error in payload["errors"])
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_cfpb_source_postgres_smoke_fails_on_forbidden_persisted_body(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(
        opportunity_rows=[_opportunity_row()],
        saved_draft_rows=[_saved_draft_row(body="<p>Acme appears to be weighing Example Bank.</p>")],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    monkeypatch.setattr(smoke, "fetch_cfpb_source_rows", lambda **_kwargs: [_source_row()])

    code, payload = await smoke.run_cfpb_source_postgres_smoke(
        _args(),
        source_rows_path=tmp_path / "cfpb_sources.jsonl",
    )

    assert code == 1
    assert any("forbidden phrase" in error for error in payload["errors"])
