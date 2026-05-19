from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types

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
        "id": "cfpb:1",
        "source_id": "cfpb:1",
        "source": "cfpb",
        "source_system": "cfpb",
        "source_type": "support_ticket",
        "complaint_id": "1",
        "vendor_name": "Example Bank",
        "text": "The bank kept charging fees after I closed the account.",
        "pain_category": "Fees",
        "source_title": "Checking account - Fees",
        "source_url": "https://example.test/cfpb/1",
    }


def _opportunity_row():
    return {
        "target_id": "cfpb:1",
        "company_name": "Acme Logistics",
        "vendor_name": "Example Bank",
        "contact_email": "ops@example.com",
        "contact_name": "Jordan Lee",
        "pain_points": ["Fees"],
        "evidence": [
            {
                "source_id": "cfpb:1",
                "source_type": "support_ticket",
                "text": "The bank kept charging fees after I closed the account.",
            }
        ],
        "raw_payload": {
            "target_id": "cfpb:1",
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
            "target_id": "cfpb:1",
            "source_opportunity": {
                "target_id": "cfpb:1",
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
        "llm": "offline",
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


def test_parse_args_defaults_to_offline_llm():
    args = smoke._parse_args([
        "--account-id",
        "acct-smoke",
        "--database-url",
        "postgres://example",
    ])

    assert args.llm == "offline"


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
    assert payload["saved_drafts"][0]["target_id"] == "cfpb:1"
    assert pool.closed is True
    assert fetch_calls[0]["api_url"] == "https://example.test/cfpb"
    assert fetch_calls[0]["timeout"] == 3.5
    assert fetch_calls[0]["source_type"] == "support_ticket"
    assert "DELETE FROM \"campaign_opportunities\"" in pool.execute_calls[0]["query"]
    assert any("INSERT INTO b2b_campaigns" in call["query"] for call in pool.fetchval_calls)
    assert pool.fetch_calls[0]["args"] == ("vendor_retention", "acct-smoke", "cfpb:1", 1)


@pytest.mark.asyncio
async def test_cfpb_source_postgres_smoke_pipeline_llm_wires_provider_ports(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(opportunity_rows=[_saved_draft_row()])
    llm = object()
    skills = object()
    calls = []
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    monkeypatch.setattr(smoke, "fetch_cfpb_source_rows", lambda **_kwargs: [_source_row()])
    monkeypatch.setattr(smoke, "create_pipeline_llm_client", lambda: llm)
    monkeypatch.setattr(smoke, "get_skill_registry", lambda: skills)

    async def generate_with_ports(**kwargs):
        calls.append(kwargs)
        return {
            "requested": 1,
            "generated": 1,
            "skipped": 0,
            "reasoning_contexts_used": 0,
            "saved_ids": ["campaign-1"],
            "errors": [],
        }

    monkeypatch.setattr(smoke, "generate_imported_target_drafts", generate_with_ports)

    code, payload = await smoke.run_cfpb_source_postgres_smoke(
        _args(llm="pipeline"),
        source_rows_path=tmp_path / "cfpb_sources.jsonl",
    )

    assert code == 0
    assert payload["ok"] is True
    assert calls[0]["llm"] is llm
    assert calls[0]["skills"] is skills


@pytest.mark.asyncio
async def test_cfpb_source_postgres_smoke_fails_before_import_when_schema_missing(
    monkeypatch,
    tmp_path,
):
    pool = _Pool(existing_relations={"b2b_campaigns"})
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    monkeypatch.setattr(smoke, "fetch_cfpb_source_rows", lambda **_kwargs: [_source_row()])

    code, payload = await smoke.run_cfpb_source_postgres_smoke(
        _args(forbidden_phrase=["appears to be weighing"]),
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
        _args(forbidden_phrase=["appears to be weighing"]),
        source_rows_path=tmp_path / "cfpb_sources.jsonl",
    )

    assert code == 1
    assert any("forbidden phrase" in error for error in payload["errors"])


def test_default_database_url_falls_back_to_atlas_settings(monkeypatch):
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    fake_config = types.SimpleNamespace(
        db_settings=types.SimpleNamespace(dsn="postgres://settings")
    )
    monkeypatch.setitem(sys.modules, "atlas_brain.storage.config", fake_config)

    assert smoke._default_database_url() == "postgres://settings"


@pytest.mark.asyncio
async def test_main_loads_dotenv_before_database_url_default(monkeypatch, tmp_path, capsys):
    pool = _Pool(
        opportunity_rows=[_opportunity_row()],
        saved_draft_rows=[_saved_draft_row()],
    )
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    monkeypatch.setattr(smoke, "fetch_cfpb_source_rows", lambda **_kwargs: [_source_row()])

    def fake_load_dotenv(path, override=False):
        if path.name == ".env.local":
            monkeypatch.setenv("EXTRACTED_DATABASE_URL", "postgres://dotenv")
        return True

    monkeypatch.setattr(smoke, "load_dotenv", fake_load_dotenv)

    code = await smoke._main([
        "--account-id",
        "acct-smoke",
        "--default-field",
        "company_name=Acme Logistics",
        "--default-field",
        "contact_email=ops@example.com",
        "--output-source-rows",
        str(tmp_path / "cfpb_sources.jsonl"),
        "--channels",
        "email_cold",
        "--json",
    ])

    assert code == 0
    assert capsys.readouterr().out
