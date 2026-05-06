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
from extracted_content_pipeline.campaign_visibility import read_jsonl_visibility_events
from extracted_content_pipeline.services.single_pass_reasoning_provider import (
    SinglePassCampaignReasoningProvider,
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


@pytest.mark.asyncio
async def test_generate_campaign_drafts_from_postgres_supports_channel_expansion():
    pool = _Pool()
    pool.fetchval_results = ["campaign-1", "campaign-2"]
    pool.fetch_rows = [
        {
            "target_id": "opp-1",
            "company_name": "Acme",
            "vendor_name": "HubSpot",
            "raw_payload": {},
        }
    ]

    result = await generate_campaign_drafts_from_postgres(
        pool,
        scope={"account_id": "acct-1"},
        target_mode="vendor_retention",
        channel="email",
        channels=("email_cold", "email_followup"),
        limit=1,
        llm=_LLM(),
        skills=_Skills(),
    )

    assert result.generated == 2
    assert result.saved_ids == ("campaign-1", "campaign-2")
    assert [call["args"][4] for call in pool.fetchval_calls] == [
        "email_cold",
        "email_followup",
    ]
    followup_metadata = json.loads(pool.fetchval_calls[1]["args"][9])
    assert followup_metadata["source_opportunity"]["channel"] == "email_followup"
    assert followup_metadata["source_opportunity"]["cold_email_context"] == {
        "subject": "Acme pricing signal",
        "body": "<p>Pricing pressure is showing up.</p>",
    }


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


def test_postgres_runner_cli_accepts_skills_root(tmp_path) -> None:
    postgres_cli = _load_postgres_cli_module()
    skill_path = tmp_path / "digest" / "b2b_campaign_generation.md"
    skill_path.parent.mkdir()
    skill_path.write_text("Custom DB prompt {opportunity_json}", encoding="utf-8")

    args = postgres_cli._parse_args([
        "--database-url",
        "postgres://example",
        "--skills-root",
        str(tmp_path),
    ])
    overrides = postgres_cli._dependency_overrides(args)

    assert overrides["skills"].get_prompt("digest/b2b_campaign_generation") == (
        "Custom DB prompt {opportunity_json}"
    )


@pytest.mark.asyncio
async def test_postgres_runner_cli_wires_pool_offline_and_reasoning_context(
    monkeypatch,
    capsys,
    tmp_path,
):
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
    reasoning_path = tmp_path / "reasoning.json"
    reasoning_path.write_text(
        json.dumps({
            "contexts": [
                {
                    "target_id": "opp-1",
                    "reasoning_context": {
                        "wedge": "renewal pressure",
                        "confidence": "high",
                    },
                }
            ]
        }),
        encoding="utf-8",
    )
    visibility_path = tmp_path / "visibility.jsonl"
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
            "--reasoning-context",
            str(reasoning_path),
            "--visibility-jsonl",
            str(visibility_path),
        ],
    )

    exit_code = await postgres_cli._main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert output["generated"] == 1
    assert output["saved_ids"] == ["campaign-1"]
    metadata = json.loads(pool.fetchval_calls[0]["args"][9])
    assert metadata["source_opportunity"]["reasoning_context"] == {
        "confidence": "high",
        "wedge": "renewal pressure",
    }
    assert pool.closed is True
    events = read_jsonl_visibility_events(visibility_path)
    assert [row["event_type"] for row in events] == [
        "campaign_operation_started",
        "campaign_operation_completed",
    ]
    assert events[0]["payload"]["operation"] == "draft_generation"
    assert events[0]["payload"]["account_id"] == "acct-1"
    assert events[1]["payload"]["result"] == {
        "error_count": 0,
        "generated": 1,
        "requested": 1,
        "saved_ids_count": 1,
        "skipped": 0,
    }


@pytest.mark.asyncio
async def test_postgres_runner_cli_requires_database_url(monkeypatch):
    postgres_cli = _load_postgres_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(postgres_cli.sys, "argv", ["run"])

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await postgres_cli._main()


def test_postgres_runner_cli_uses_provider_port_loader(tmp_path) -> None:
    postgres_cli = _load_postgres_cli_module()
    reasoning_path = tmp_path / "reasoning.json"
    reasoning_path.write_text("[]", encoding="utf-8")

    calls = []

    def _fake_loader(path):
        calls.append(path)
        return "provider-port"

    postgres_cli.load_reasoning_provider_port = _fake_loader

    args = postgres_cli._parse_args([
        "--database-url",
        "postgres://example",
        "--reasoning-context",
        str(reasoning_path),
    ])
    overrides = postgres_cli._dependency_overrides(args)

    assert calls == [reasoning_path]
    assert overrides["reasoning_context"] == "provider-port"


def test_postgres_runner_cli_wires_single_pass_reasoning(monkeypatch, tmp_path) -> None:
    postgres_cli = _load_postgres_cli_module()
    llm = _LLM()
    skills = _Skills()
    skill_path = tmp_path / "digest" / "b2b_campaign_generation.md"
    skill_path.parent.mkdir()
    skill_path.write_text("Custom DB prompt {opportunity_json}", encoding="utf-8")

    monkeypatch.setattr(postgres_cli, "create_pipeline_llm_client", lambda: llm)
    monkeypatch.setattr(postgres_cli, "get_skill_registry", lambda root=None: skills)

    args = postgres_cli._parse_args([
        "--database-url",
        "postgres://example",
        "--skills-root",
        str(tmp_path),
        "--single-pass-reasoning",
        "--reasoning-skill-name",
        "digest/custom_reasoning",
        "--reasoning-max-tokens",
        "321",
        "--reasoning-temperature",
        "0.3",
        "--no-reasoning-source-opportunity",
    ])
    overrides = postgres_cli._dependency_overrides(args)

    provider = overrides["reasoning_context"]
    assert isinstance(provider, SinglePassCampaignReasoningProvider)
    assert provider.llm is llm
    assert provider.skills is skills
    assert provider.config.skill_name == "digest/custom_reasoning"
    assert provider.config.max_tokens == 321
    assert provider.config.temperature == 0.3
    assert provider.config.include_source_opportunity is False
    assert overrides["llm"] is llm
    assert overrides["skills"] is skills


def test_postgres_runner_cli_rejects_conflicting_reasoning_modes(tmp_path) -> None:
    postgres_cli = _load_postgres_cli_module()
    reasoning_path = tmp_path / "reasoning.json"
    reasoning_path.write_text("[]", encoding="utf-8")

    args = postgres_cli._parse_args([
        "--database-url",
        "postgres://example",
        "--reasoning-context",
        str(reasoning_path),
        "--single-pass-reasoning",
    ])

    with pytest.raises(SystemExit, match="cannot be combined"):
        postgres_cli._dependency_overrides(args)


def test_postgres_runner_cli_rejects_offline_single_pass_reasoning() -> None:
    postgres_cli = _load_postgres_cli_module()
    args = postgres_cli._parse_args([
        "--database-url",
        "postgres://example",
        "--llm",
        "offline",
        "--single-pass-reasoning",
    ])

    with pytest.raises(SystemExit, match="requires --llm pipeline"):
        postgres_cli._dependency_overrides(args)
