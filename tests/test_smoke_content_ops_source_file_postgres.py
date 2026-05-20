from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_source_file_postgres.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_source_file_postgres",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
SUPPORT_TICKET_CSV = ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"


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
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        if self.opportunity_rows and len(self.fetch_calls) == 1:
            return self.opportunity_rows
        return self.saved_draft_rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": str(query), "args": args})
        return "EXECUTE"

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": str(query), "args": args})
        if "to_regclass" in str(query):
            return args[0] if args and args[0] in self.existing_relations else None
        return "campaign-1"

    async def close(self):
        self.closed = True


async def _return_pool(pool):
    return pool


def _opportunity_row(target_id="ticket-acme-1"):
    return {
        "target_id": target_id,
        "company_name": "Acme Logistics",
        "vendor_name": "HubSpot",
        "contact_email": "ops@example.com",
        "pain_points": ["reporting friction"],
        "evidence": [
            {
                "source_id": target_id,
                "source_type": "support_ticket",
                "text": "The operations team cannot export campaign attribution data.",
            }
        ],
        "raw_payload": {
            "target_id": target_id,
            "source_type": "support_ticket",
        },
    }


def _saved_draft_row(target_id="ticket-acme-1", *, body=None):
    return {
        "id": f"campaign-{target_id}",
        "subject": "Acme Logistics: reporting friction",
        "body": body or "<p>Teams evaluating HubSpot are reporting export friction.</p>",
        "target_mode": "vendor_retention",
        "channel": "email_cold",
        "metadata": {
            "target_id": target_id,
            "source_opportunity": {
                "target_id": target_id,
            },
        },
    }


def _args(**overrides):
    values = {
        "path": SUPPORT_TICKET_CSV,
        "source_format": "csv",
        "target_mode": "vendor_retention",
        "channels": "email_cold",
        "min_source_rows": 2,
        "min_drafts": None,
        "allow_ingestion_warnings": False,
        "default_field": [],
        "account_id": "acct-smoke",
        "user_id": None,
        "opportunity_table": "campaign_opportunities",
        "keep_existing_opportunities": False,
        "forbidden_phrase": list(smoke.DEFAULT_FORBIDDEN_PHRASES),
        "output_result": None,
        "json": False,
        "database_url": "postgres://example",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
async def test_source_file_postgres_smoke_imports_and_persists(monkeypatch):
    pool = _Pool(
        saved_draft_rows=[
            _saved_draft_row("ticket-acme-1"),
            _saved_draft_row("ticket-acme-2"),
            _saved_draft_row("ticket-northstar-1"),
            _saved_draft_row("ticket-northstar-2"),
        ],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    async def fake_generate_imported_target_drafts(**kwargs):
        assert kwargs["target_ids"] == [
            "ticket-acme-1",
            "ticket-acme-2",
            "ticket-northstar-1",
            "ticket-northstar-2",
        ]
        return {
            "requested": 4,
            "generated": 4,
            "skipped": 0,
            "reasoning_contexts_used": 0,
            "saved_ids": [
                "campaign-ticket-acme-1",
                "campaign-ticket-acme-2",
                "campaign-ticket-northstar-1",
                "campaign-ticket-northstar-2",
            ],
            "errors": [],
        }

    monkeypatch.setattr(
        smoke,
        "generate_imported_target_drafts",
        fake_generate_imported_target_drafts,
    )

    code, payload = await smoke.run_source_file_postgres_smoke(_args())

    assert code == 0
    assert payload["ok"] is True
    assert payload["source_rows"] == 4
    assert payload["import"]["inserted"] == 4
    assert payload["drafts"]["generated"] == 4
    assert [row["target_id"] for row in payload["saved_drafts"]] == [
        "ticket-acme-1",
        "ticket-acme-2",
        "ticket-northstar-1",
        "ticket-northstar-2",
    ]
    assert pool.closed is True
    assert "DELETE FROM \"campaign_opportunities\"" in pool.execute_calls[0]["query"]
    assert "INSERT INTO \"campaign_opportunities\"" in pool.execute_calls[1]["query"]


@pytest.mark.asyncio
async def test_source_file_postgres_smoke_fails_before_import_when_schema_missing(monkeypatch):
    pool = _Pool(existing_relations={"b2b_campaigns"})
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_source_file_postgres_smoke(_args())

    assert code == 1
    assert any("required Content Ops table(s) missing" in error for error in payload["errors"])
    assert any("campaign_opportunities" in error for error in payload["errors"])
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_source_file_postgres_smoke_fails_on_wrong_persisted_target(monkeypatch):
    pool = _Pool(
        saved_draft_rows=[_saved_draft_row("wrong-ticket"), _saved_draft_row("ticket-northstar-1")],
    )
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    async def fake_generate_imported_target_drafts(**_kwargs):
        return {
            "requested": 2,
            "generated": 2,
            "skipped": 0,
            "reasoning_contexts_used": 0,
            "saved_ids": ["campaign-wrong-ticket", "campaign-ticket-northstar-1"],
            "errors": [],
        }

    monkeypatch.setattr(
        smoke,
        "generate_imported_target_drafts",
        fake_generate_imported_target_drafts,
    )

    code, payload = await smoke.run_source_file_postgres_smoke(_args())

    assert code == 1
    assert any("persisted draft target_id was not imported" in error for error in payload["errors"])


def test_default_database_url_falls_back_to_atlas_settings(monkeypatch):
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    fake_config = types.SimpleNamespace(
        db_settings=types.SimpleNamespace(dsn="postgres://settings")
    )
    monkeypatch.setitem(sys.modules, "atlas_brain.storage.config", fake_config)

    assert smoke._default_database_url() == "postgres://settings"
