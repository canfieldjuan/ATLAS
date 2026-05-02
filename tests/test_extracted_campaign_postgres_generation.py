from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_ports import LLMResponse, TenantScope
from extracted_content_pipeline.campaign_postgres_generation import (
    generate_campaign_drafts_from_postgres,
    tenant_scope_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_postgres_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_extracted_campaign_generation_postgres",
        ROOT / "scripts/run_extracted_campaign_generation_postgres.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self):
        self.fetch_rows = []
        self.fetchval_results = ["campaign-1"]
        self.fetch_calls = []
        self.fetchval_calls = []
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": query, "args": args})
        return self.fetchval_results.pop(0)

    async def close(self):
        self.closed = True


class _LLM:
    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        self.calls.append({
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": dict(metadata or {}),
        })
        return LLMResponse(
            content=json.dumps({
                "subject": "Acme pricing signal",
                "body": "<p>Pricing pressure is showing up.</p>",
                "cta": "Review the account",
            }),
            model="test-model",
        )


class _Skills:
    def __init__(self):
        self.calls = []

    def get_prompt(self, name):
        self.calls.append(name)
        return "Mode={target_mode}; opportunity={opportunity_json}"


@pytest.mark.asyncio
async def test_generate_campaign_drafts_from_postgres_reads_generates_and_saves():
    pool = _Pool()
    pool.fetch_rows = [
        {
            "target_id": "opp-1",
            "company_name": "Acme",
            "vendor_name": "HubSpot",
            "contact_email": "buyer@example.com",
            "pain_points": ["pricing"],
            "competitors": ["Salesforce"],
            "raw_payload": {"custom_segment": "enterprise"},
        }
    ]
    llm = _LLM()
    skills = _Skills()

    result = await generate_campaign_drafts_from_postgres(
        pool,
        scope={"account_id": "acct-1", "user_id": "user-1"},
        target_mode="vendor_retention",
        channel="email",
        limit=5,
        filters={"vendor_name": "HubSpot"},
        llm=llm,
        skills=skills,
    )

    assert result.as_dict() == {
        "requested": 1,
        "generated": 1,
        "skipped": 0,
        "saved_ids": ["campaign-1"],
        "errors": [],
    }
    assert "FROM \"campaign_opportunities\"" in pool.fetch_calls[0]["query"]
    assert pool.fetch_calls[0]["args"] == ("vendor_retention", "acct-1", "HubSpot", 5)
    assert '"custom_segment":"enterprise"' in llm.calls[0]["messages"][0].content
    save_call = pool.fetchval_calls[0]
    assert "INSERT INTO b2b_campaigns" in save_call["query"]
    assert save_call["args"][:9] == (
        "Acme",
        "HubSpot",
        None,
        "vendor_retention",
        "email",
        "Acme pricing signal",
        "<p>Pricing pressure is showing up.</p>",
        "Review the account",
        "buyer@example.com",
    )
    metadata = json.loads(save_call["args"][9])
    assert metadata["scope"] == {"account_id": "acct-1", "user_id": "user-1"}
    assert metadata["source_opportunity"]["custom_segment"] == "enterprise"
    assert save_call["args"][10] == "test-model"


def test_tenant_scope_from_mapping_accepts_mapping_and_existing_scope():
    scope = tenant_scope_from_mapping({
        "account_id": "acct-1",
        "user_id": "user-1",
        "allowed_vendors": ["HubSpot"],
        "roles": ["admin"],
    })

    assert scope == TenantScope(
        account_id="acct-1",
        user_id="user-1",
        allowed_vendors=("HubSpot",),
        roles=("admin",),
    )
    assert tenant_scope_from_mapping(scope) is scope
    assert tenant_scope_from_mapping(None) == TenantScope()


@pytest.mark.asyncio
async def test_postgres_runner_cli_wires_pool_and_offline_dependencies(monkeypatch, capsys):
    postgres_cli = _load_postgres_cli_module()
    pool = _Pool()
    pool.fetch_rows = [
        {
            "target_id": "opp-1",
            "company_name": "Acme",
            "vendor_name": "HubSpot",
            "contact_email": "buyer@example.com",
            "pain_points": ["pricing"],
            "raw_payload": {},
        }
    ]
    created_urls = []

    async def create_pool(database_url):
        created_urls.append(database_url)
        return pool

    monkeypatch.setattr(postgres_cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        postgres_cli.sys,
        "argv",
        [
            "run",
            "--database-url",
            "postgres://example",
            "--account-id",
            "acct-1",
            "--target-mode",
            "vendor_retention",
            "--limit",
            "1",
            "--llm",
            "offline",
        ],
    )

    exit_code = await postgres_cli._main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert output["generated"] == 1
    assert output["saved_ids"] == ["campaign-1"]
    assert pool.closed is True


@pytest.mark.asyncio
async def test_postgres_runner_cli_requires_database_url(monkeypatch):
    postgres_cli = _load_postgres_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(postgres_cli.sys, "argv", ["run"])

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await postgres_cli._main()
