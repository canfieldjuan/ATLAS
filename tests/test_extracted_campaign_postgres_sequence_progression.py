from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_ports import LLMResponse
from extracted_content_pipeline.campaign_postgres_sequence_progression import (
    progress_campaign_sequences_from_postgres,
)
from extracted_content_pipeline.campaign_sequence_progression import (
    CampaignSequenceProgressionConfig,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/progress_extracted_campaign_sequences.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "progress_extracted_campaign_sequences",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self, due=None, previous=None) -> None:
        self.due = list(due or [])
        self.previous = list(previous or [])
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        if "FROM campaign_sequences" in str(query):
            return self.due
        if "FROM b2b_campaigns" in str(query):
            return self.previous
        return []

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((str(query), args))
        return "campaign-1"

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        return "OK"

    async def close(self):
        self.closed = True


class _LLM:
    def __init__(self) -> None:
        self.calls = []

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        self.calls.append({
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": metadata,
        })
        return LLMResponse(
            content=json.dumps({
                "subject": "Following up",
                "body": "<p>Second note</p>",
                "cta": "Book time",
                "angle_reasoning": "Opened previous email.",
            }),
            model="test-model",
        )


class _Skills:
    def __init__(self) -> None:
        self.calls = []

    def get_prompt(self, name):
        self.calls.append(name)
        return (
            "Company {company_name}; step {current_step}/{max_steps}; "
            "ctx {company_context}; sell {selling_context}; "
            "eng {engagement_summary}; prev {previous_emails}"
        )


def _sequence(**overrides):
    row = {
        "id": "sequence-1",
        "company_name": "Acme",
        "batch_id": "batch-1",
        "recipient_email": "buyer@example.com",
        "current_step": 1,
        "max_steps": 4,
        "open_count": 1,
        "click_count": 0,
        "last_sent_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "company_context": {
            "recipient_type": "vendor_retention",
            "selling": {"product_name": "Atlas"},
        },
        "selling_context": {"value_prop": "research-backed outreach"},
    }
    row.update(overrides)
    return row


def _previous(**overrides):
    row = {
        "step_number": 1,
        "subject": "First note",
        "body": "<p>Hello</p>",
        "status": "sent",
        "opened_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "clicked_at": None,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_postgres_sequence_progression_runner_queues_due_followup() -> None:
    pool = _Pool(due=[_sequence()], previous=[_previous()])
    llm = _LLM()
    skills = _Skills()

    result = await progress_campaign_sequences_from_postgres(
        pool,
        llm=llm,
        skills=skills,
        config=CampaignSequenceProgressionConfig(
            batch_limit=2,
            from_email="sales@example.com",
        ),
    )

    assert result.as_dict() == {
        "due_sequences": 1,
        "progressed": 1,
        "skipped": 0,
        "disabled": False,
    }
    assert pool.fetch_calls[0][1][1] == 2
    insert_query, insert_args = pool.fetchval_calls[0]
    assert "INSERT INTO b2b_campaigns" in insert_query
    assert insert_args[:9] == (
        "sequence-1",
        "Acme",
        "batch-1",
        "Following up",
        "<p>Second note</p>",
        "Book time",
        2,
        "buyer@example.com",
        "sales@example.com",
    )
    assert any("UPDATE campaign_sequences" in call[0] for call in pool.execute_calls)
    assert sum("INSERT INTO campaign_audit_log" in call[0] for call in pool.execute_calls) == 2
    assert skills.calls == ["digest/b2b_vendor_sequence"]
    assert llm.calls[0]["metadata"]["sequence_id"] == "sequence-1"


@pytest.mark.asyncio
async def test_postgres_sequence_progression_runner_limit_zero_returns_empty_without_query() -> None:
    pool = _Pool(due=[_sequence()])

    result = await progress_campaign_sequences_from_postgres(
        pool,
        llm=_LLM(),
        skills=_Skills(),
        limit=0,
    )

    assert result.as_dict() == {
        "due_sequences": 0,
        "progressed": 0,
        "skipped": 0,
        "disabled": False,
    }
    assert pool.fetch_calls == []
    assert pool.fetchval_calls == []
    assert pool.execute_calls == []


def test_sequence_cli_parses_env_defaults(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEQUENCE_LIMIT", "7")
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEQUENCE_MAX_STEPS", "6")
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEQUENCE_FROM_EMAIL", "sales@example.com")
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEQUENCE_TEMPERATURE", "0.2")
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEQUENCE_LLM", " offline ")

    args = cli._parse_args(["--database-url", "postgres://example"])

    assert args.limit == 7
    assert args.max_steps == 6
    assert args.from_email == "sales@example.com"
    assert args.temperature == 0.2
    assert args.llm == "offline"


def test_sequence_cli_rejects_invalid_integer_env(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEQUENCE_LIMIT", "many")

    with pytest.raises(SystemExit) as exc_info:
        cli._parse_args(["--database-url", "postgres://example"])

    message = str(exc_info.value)
    assert "Invalid integer for EXTRACTED_CAMPAIGN_SEQUENCE_LIMIT" in message
    assert "'many'" in message


def test_sequence_cli_rejects_invalid_float_env(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEQUENCE_TEMPERATURE", "warm")

    with pytest.raises(SystemExit) as exc_info:
        cli._parse_args(["--database-url", "postgres://example"])

    message = str(exc_info.value)
    assert "Invalid float for EXTRACTED_CAMPAIGN_SEQUENCE_TEMPERATURE" in message
    assert "'warm'" in message


def test_sequence_cli_rejects_unknown_llm_env(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEQUENCE_LLM", "remote")

    with pytest.raises(SystemExit) as exc_info:
        cli._parse_args(["--database-url", "postgres://example"])

    assert "Invalid --llm" in str(exc_info.value)
    assert "remote" in str(exc_info.value)


def test_sequence_cli_validates_positive_limits() -> None:
    cli = _load_cli_module()
    args = cli._parse_args([
        "--database-url",
        "postgres://example",
        "--limit",
        "1",
        "--max-steps",
        "1",
    ])
    cli._validate_args(args)

    args.limit = 0
    with pytest.raises(SystemExit, match="Invalid --limit"):
        cli._validate_args(args)
    args.limit = 1
    args.max_steps = 0
    with pytest.raises(SystemExit, match="Invalid --max-steps"):
        cli._validate_args(args)


def test_sequence_cli_builds_config_and_offline_llm() -> None:
    cli = _load_cli_module()
    args = cli._parse_args([
        "--database-url",
        "postgres://example",
        "--llm",
        "offline",
        "--limit",
        "3",
        "--max-steps",
        "5",
        "--from-email",
        "sales@example.com",
        "--onboarding-product-name",
        "Atlas Ops",
        "--temperature",
        "0.1",
    ])

    config = cli._config_from_args(args)

    assert config.batch_limit == 3
    assert config.max_steps == 5
    assert config.from_email == "sales@example.com"
    assert config.onboarding_product_name == "Atlas Ops"
    assert config.temperature == 0.1
    assert cli._llm_from_args(args).__class__.__name__ == "DeterministicCampaignLLM"
