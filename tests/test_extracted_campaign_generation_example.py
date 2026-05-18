from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_example import (
    generate_campaign_drafts_from_payload,
)
from extracted_content_pipeline.campaign_ports import LLMResponse
from extracted_content_pipeline.campaign_reasoning_data import (
    FileCampaignReasoningContextProvider,
    load_reasoning_provider_port,
)
from extracted_content_pipeline.services.single_pass_reasoning_provider import (
    SinglePassCampaignReasoningProvider,
)
from extracted_content_pipeline.services.multi_pass_reasoning_provider import (
    MultiPassCampaignReasoningProvider,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PAYLOAD = (
    ROOT / "extracted_content_pipeline/examples/campaign_generation_payload.json"
)
EXAMPLE_SOURCE_ROWS = (
    ROOT / "extracted_content_pipeline/examples/campaign_source_rows.jsonl"
)
EXAMPLE_SOURCE_BUNDLE = (
    ROOT / "extracted_content_pipeline/examples/campaign_source_bundle.json"
)
CLI = ROOT / "scripts/run_extracted_campaign_generation_example.py"


def _load_example_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_extracted_campaign_generation_example",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _InjectedLLM:
    def __init__(self) -> None:
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
                "subject": "Injected subject",
                "body": "<p>Injected body</p>",
            }),
            model="injected-model",
        )


class _InjectedSkills:
    def get_prompt(self, name):
        return f"Injected prompt for {name} without payload placeholders."


@pytest.mark.asyncio
async def test_example_generates_drafts_from_customer_opportunity_payload() -> None:
    payload = {
        "scope": {"account_id": "acct-1", "allowed_vendors": ["HubSpot"]},
        "target_mode": "vendor_retention",
        "limit": 1,
        "opportunities": [
            {
                "id": "opp-1",
                "company": "Acme Logistics",
                "vendor": "HubSpot",
                "email": "ops@example.com",
                "title": "VP Revenue Operations",
                "pain_category": "pricing pressure",
                "competitor": "Salesforce, Zoho",
                "custom_segment": "enterprise logistics",
            }
        ],
    }

    result = await generate_campaign_drafts_from_payload(payload)

    assert result["result"] == {
        "requested": 1,
        "generated": 1,
        "skipped": 0,
        "reasoning_contexts_used": 0,
        "saved_ids": ["draft-1"],
        "errors": [],
    }
    assert result["llm_model"] == "offline-deterministic"
    draft = result["drafts"][0]
    assert draft["id"] == "draft-1"
    assert draft["target_id"] == "opp-1"
    assert draft["target_mode"] == "vendor_retention"
    assert draft["channel"] == "email"
    assert draft["subject"] == "Acme Logistics: pricing pressure"
    assert "Acme Logistics appears to be weighing HubSpot" in draft["body"]
    source = draft["metadata"]["source_opportunity"]
    assert source["company_name"] == "Acme Logistics"
    assert source["vendor_name"] == "HubSpot"
    assert source["contact_email"] == "ops@example.com"
    assert source["contact_title"] == "VP Revenue Operations"
    assert source["pain_points"] == ["pricing pressure"]
    assert source["competitors"] == ["Salesforce", "Zoho"]
    assert source["custom_segment"] == "enterprise logistics"


@pytest.mark.asyncio
async def test_example_accepts_host_llm_and_skill_store_overrides() -> None:
    llm = _InjectedLLM()
    payload = {
        "target_mode": "vendor_retention",
        "limit": 1,
        "opportunities": [
            {"id": "opp-1", "company": "Acme Logistics", "vendor": "HubSpot"}
        ],
    }

    result = await generate_campaign_drafts_from_payload(
        payload,
        llm=llm,
        skills=_InjectedSkills(),
    )

    assert result["llm_model"] == "injected-model"
    assert result["drafts"][0]["subject"] == "Injected subject"
    assert "without payload placeholders" in llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert "target_mode=vendor_retention" in user_prompt
    assert '"company_name":"Acme Logistics"' in user_prompt


@pytest.mark.asyncio
async def test_example_generates_cold_and_followup_channels_from_payload() -> None:
    payload = {
        "target_mode": "vendor_retention",
        "channels": ["email_cold", "email_followup"],
        "limit": 1,
        "opportunities": [
            {"id": "opp-1", "company": "Acme Logistics", "vendor": "HubSpot"}
        ],
    }

    result = await generate_campaign_drafts_from_payload(payload)

    assert result["result"]["generated"] == 2
    assert result["result"]["saved_ids"] == ["draft-1", "draft-2"]
    assert [draft["channel"] for draft in result["drafts"]] == [
        "email_cold",
        "email_followup",
    ]
    followup = result["drafts"][1]
    assert followup["metadata"]["source_opportunity"]["cold_email_context"] == {
        "subject": result["drafts"][0]["subject"],
        "body": result["drafts"][0]["body"],
    }


@pytest.mark.asyncio
async def test_example_accepts_file_backed_reasoning_provider() -> None:
    provider = FileCampaignReasoningContextProvider.from_payload({
        "contexts": [
            {
                "target_id": "opp-1",
                "reasoning_context": {
                    "wedge": "renewal pressure",
                    "confidence": "high",
                },
                "campaign_reasoning_context": {
                    "proof_points": [{"label": "pricing_mentions", "value": 12}]
                },
            }
        ]
    })
    payload = {
        "target_mode": "vendor_retention",
        "limit": 1,
        "opportunities": [
            {"id": "opp-1", "company": "Acme Logistics", "vendor": "HubSpot"}
        ],
    }

    result = await generate_campaign_drafts_from_payload(
        payload,
        reasoning_context=provider,
    )

    source = result["drafts"][0]["metadata"]["source_opportunity"]
    assert source["reasoning_context"]["wedge"] == "renewal pressure"
    assert source["campaign_reasoning_context"]["proof_points"][0]["label"] == (
        "pricing_mentions"
    )


@pytest.mark.asyncio
async def test_example_respects_limit_and_normalizes_multiple_rows() -> None:
    payload = json.loads(EXAMPLE_PAYLOAD.read_text(encoding="utf-8"))

    result = await generate_campaign_drafts_from_payload({**payload, "limit": 1})

    assert result["result"]["requested"] == 1
    assert result["result"]["generated"] == 1
    assert len(result["drafts"]) == 1
    assert result["drafts"][0]["target_id"] == "opp-acme-hubspot"


@pytest.mark.asyncio
async def test_example_rejects_payload_without_opportunities_array() -> None:
    with pytest.raises(ValueError, match="opportunities array"):
        await generate_campaign_drafts_from_payload({"opportunities": "not-an-array"})


def test_campaign_generation_example_cli_outputs_draft_json() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI), str(EXAMPLE_PAYLOAD), "--limit", "1", "--llm", "offline"],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)

    assert result["result"]["generated"] == 1
    assert result["result"]["saved_ids"] == ["draft-1"]
    assert result["drafts"][0]["target_id"] == "opp-acme-hubspot"
    assert result["drafts"][0]["metadata"]["generation_model"] == "offline-deterministic"


def test_campaign_generation_example_cli_can_enable_quality_revalidation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(EXAMPLE_PAYLOAD),
            "--limit",
            "1",
            "--llm",
            "offline",
            "--quality-revalidation",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    metadata = result["drafts"][0]["metadata"]
    assert metadata["campaign_revalidation"]["audit"]["status"] == "pass"


def test_campaign_generation_example_cli_generates_from_source_rows() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(EXAMPLE_SOURCE_ROWS),
            "--source-rows",
            "--source-format",
            "jsonl",
            "--channels",
            "email_cold,email_followup",
            "--limit",
            "1",
            "--llm",
            "offline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["result"]["generated"] == 2
    assert [draft["channel"] for draft in result["drafts"]] == [
        "email_cold",
        "email_followup",
    ]
    source = result["drafts"][0]["metadata"]["source_opportunity"]
    assert source["target_id"] == "review-acme-1"
    assert source["evidence"][0]["source_type"] == "review"
    assert source["contact_email"] == "ops@example.com"


def test_campaign_generation_example_cli_applies_source_default_fields(tmp_path) -> None:
    source_path = tmp_path / "g2_sources.jsonl"
    source_path.write_text(
        json.dumps({
            "id": "g2-review-1",
            "vendor": "Slack",
            "review_text": "Search gets slow once message history grows.",
        }),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(source_path),
            "--source-rows",
            "--source-format",
            "jsonl",
            "--default-field",
            "company_name=Acme Logistics",
            "--default-field",
            "contact_email=ops@example.com",
            "--channels",
            "email_cold",
            "--limit",
            "1",
            "--llm",
            "offline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    source = json.loads(completed.stdout)["drafts"][0]["metadata"]["source_opportunity"]
    assert source["company_name"] == "Acme Logistics"
    assert source["contact_email"] == "ops@example.com"


def test_campaign_generation_example_cli_generates_from_source_bundle_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(EXAMPLE_SOURCE_BUNDLE),
            "--source-rows",
            "--source-format",
            "json",
            "--channels",
            "email_cold,email_followup",
            "--limit",
            "1",
            "--llm",
            "offline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["result"]["generated"] == 2
    source = result["drafts"][0]["metadata"]["source_opportunity"]
    assert source["target_id"] == "bundle-review-acme-1"
    assert source["company_name"] == "Acme Logistics"
    assert source["evidence"][0]["source_type"] == "review"


def test_campaign_generation_example_cli_rejects_invalid_source_text_limit(tmp_path) -> None:
    source_path = tmp_path / "customer_sources.json"
    source_path.write_text("[]", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(source_path),
            "--source-rows",
            "--max-source-text-chars",
            "0",
            "--llm",
            "offline",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--max-source-text-chars must be positive" in completed.stderr


def test_campaign_generation_example_cli_accepts_reasoning_context_file(tmp_path) -> None:
    reasoning_path = tmp_path / "reasoning.json"
    reasoning_path.write_text(
        json.dumps({
            "contexts": [
                {
                    "target_id": "opp-acme-hubspot",
                    "reasoning_context": {
                        "wedge": "renewal pressure",
                        "confidence": "high",
                    },
                }
            ]
        }),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(EXAMPLE_PAYLOAD),
            "--limit",
            "1",
            "--llm",
            "offline",
            "--reasoning-context",
            str(reasoning_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    source = result["drafts"][0]["metadata"]["source_opportunity"]
    assert source["reasoning_context"]["wedge"] == "renewal pressure"
    assert source["reasoning_context"]["confidence"] == "high"


def test_campaign_generation_example_cli_accepts_skills_root(tmp_path) -> None:
    example_cli = _load_example_cli_module()
    skill_path = tmp_path / "digest" / "b2b_campaign_generation.md"
    skill_path.parent.mkdir()
    skill_path.write_text("Custom host prompt {opportunity_json}", encoding="utf-8")
    args = example_cli._parse_args([
        str(EXAMPLE_PAYLOAD),
        "--skills-root",
        str(tmp_path),
    ])

    overrides = example_cli._dependency_overrides(args)

    assert overrides["skills"].get_prompt("digest/b2b_campaign_generation") == (
        "Custom host prompt {opportunity_json}"
    )


def test_campaign_generation_example_cli_wires_single_pass_reasoning(
    monkeypatch,
    tmp_path,
) -> None:
    example_cli = _load_example_cli_module()
    llm = _InjectedLLM()
    skills = _InjectedSkills()
    skill_path = tmp_path / "digest" / "b2b_campaign_generation.md"
    skill_path.parent.mkdir()
    skill_path.write_text("Custom host prompt {opportunity_json}", encoding="utf-8")

    monkeypatch.setattr(
        "extracted_content_pipeline.campaign_llm_client.create_pipeline_llm_client",
        lambda: llm,
    )
    monkeypatch.setattr(
        "extracted_content_pipeline.skills.registry.get_skill_registry",
        lambda root=None: skills,
    )

    args = example_cli._parse_args([
        str(EXAMPLE_PAYLOAD),
        "--llm",
        "pipeline",
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
    overrides = example_cli._dependency_overrides(args)

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


def test_campaign_generation_example_cli_wires_multi_pass_reasoning(
    monkeypatch,
) -> None:
    example_cli = _load_example_cli_module()
    llm = _InjectedLLM()

    monkeypatch.setattr(
        "extracted_content_pipeline.campaign_llm_client.create_pipeline_llm_client",
        lambda: llm,
    )

    args = example_cli._parse_args([
        str(EXAMPLE_PAYLOAD),
        "--llm",
        "pipeline",
        "--multi-pass-reasoning",
        "--multi-pass-pack-name",
        "campaign/content",
        "--multi-pass-depth",
        "L3",
        "--multi-pass-max-continuations",
        "2",
        "--multi-pass-disable-chain",
    ])
    overrides = example_cli._dependency_overrides(args)

    provider = overrides["reasoning_context"]
    assert isinstance(provider, MultiPassCampaignReasoningProvider)
    assert provider._ports.llm is llm
    assert provider._config.pack_name == "campaign/content"
    assert provider._config.default_depth == "L3"
    assert provider._config.max_continuations == 2
    assert provider._config.enable_multi_pass is False
    assert overrides["llm"] is llm


def test_campaign_generation_example_cli_rejects_conflicting_reasoning_modes(
    tmp_path,
) -> None:
    example_cli = _load_example_cli_module()
    reasoning_path = tmp_path / "reasoning.json"
    reasoning_path.write_text("[]", encoding="utf-8")

    args = example_cli._parse_args([
        str(EXAMPLE_PAYLOAD),
        "--llm",
        "pipeline",
        "--reasoning-context",
        str(reasoning_path),
        "--single-pass-reasoning",
    ])

    with pytest.raises(SystemExit, match="cannot be combined"):
        example_cli._dependency_overrides(args)


def test_campaign_generation_example_cli_rejects_offline_single_pass_reasoning() -> None:
    example_cli = _load_example_cli_module()
    args = example_cli._parse_args([
        str(EXAMPLE_PAYLOAD),
        "--llm",
        "offline",
        "--single-pass-reasoning",
    ])

    with pytest.raises(SystemExit, match="requires --llm pipeline"):
        example_cli._dependency_overrides(args)


def test_campaign_generation_example_cli_rejects_offline_multi_pass_reasoning() -> None:
    example_cli = _load_example_cli_module()
    args = example_cli._parse_args([
        str(EXAMPLE_PAYLOAD),
        "--llm",
        "offline",
        "--multi-pass-reasoning",
    ])

    with pytest.raises(SystemExit, match="requires --llm pipeline"):
        example_cli._dependency_overrides(args)


def test_campaign_generation_example_cli_rejects_invalid_multi_pass_depth() -> None:
    example_cli = _load_example_cli_module()

    with pytest.raises(SystemExit):
        example_cli._parse_args([
            str(EXAMPLE_PAYLOAD),
            "--multi-pass-reasoning",
            "--multi-pass-depth",
            "L6",
        ])


def test_campaign_generation_example_cli_rejects_negative_multi_pass_continuations() -> None:
    example_cli = _load_example_cli_module()
    args = example_cli._parse_args([
        str(EXAMPLE_PAYLOAD),
        "--llm",
        "pipeline",
        "--multi-pass-reasoning",
        "--multi-pass-max-continuations",
        "-1",
    ])

    with pytest.raises(SystemExit, match="must be >= 0"):
        example_cli._dependency_overrides(args)


@pytest.mark.asyncio
async def test_example_accepts_provider_port_loader(tmp_path) -> None:
    reasoning_path = tmp_path / "reasoning_port.json"
    reasoning_path.write_text(
        json.dumps({
            "contexts": [
                {
                    "target_id": "opp-1",
                    "reasoning_context": {"wedge": "renewal pressure", "confidence": "high"},
                    "campaign_reasoning_context": {
                        "proof_points": [{"label": "pricing_mentions", "value": 12}]
                    },
                }
            ]
        }),
        encoding="utf-8",
    )
    provider = load_reasoning_provider_port(reasoning_path)
    payload = {
        "target_mode": "vendor_retention",
        "limit": 1,
        "opportunities": [
            {"id": "opp-1", "company": "Acme Logistics", "vendor": "HubSpot"}
        ],
    }

    result = await generate_campaign_drafts_from_payload(payload, reasoning_context=provider)

    source = result["drafts"][0]["metadata"]["source_opportunity"]
    assert source["reasoning_context"]["wedge"] == "renewal pressure"
    assert source["campaign_reasoning_context"]["proof_points"][0]["label"] == "pricing_mentions"
