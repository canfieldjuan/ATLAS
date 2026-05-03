from __future__ import annotations

import json

import pytest

import extracted_content_pipeline.campaign_generation as campaign_generation_module
from extracted_content_pipeline.campaign_generation import (
    CampaignGenerationConfig,
    CampaignGenerationService,
    opportunity_target_id,
    parse_campaign_draft_response,
)
from extracted_content_pipeline.campaign_opportunities import (
    normalize_campaign_opportunity,
)
from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)


class _Intelligence:
    def __init__(self, opportunities):
        self.opportunities = opportunities
        self.calls = []

    async def read_campaign_opportunities(self, *, scope, target_mode, limit, filters=None):
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": filters,
        })
        return self.opportunities

    async def read_vendor_targets(self, *, scope, target_mode, vendor_name=None):  # pragma: no cover
        raise AssertionError("not used")


class _Campaigns:
    def __init__(self):
        self.saved = []

    async def save_drafts(self, drafts, *, scope):
        self.saved.append({"drafts": list(drafts), "scope": scope})
        return [f"draft-{index + 1}" for index, _ in enumerate(drafts)]

    async def list_due_sends(self, *, limit, now):  # pragma: no cover - protocol filler
        raise AssertionError("not used")

    async def mark_sent(self, *, campaign_id, result, sent_at):  # pragma: no cover
        raise AssertionError("not used")

    async def mark_cancelled(self, *, campaign_id, reason, metadata=None):  # pragma: no cover
        raise AssertionError("not used")

    async def mark_send_failed(self, *, campaign_id, error, metadata=None):  # pragma: no cover
        raise AssertionError("not used")

    async def record_webhook_event(self, event):  # pragma: no cover
        raise AssertionError("not used")

    async def refresh_analytics(self):  # pragma: no cover
        raise AssertionError("not used")


class _LLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        self.calls.append({
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": metadata,
        })
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(
            content=response,
            model="test-model",
            usage={"input_tokens": 11, "output_tokens": 7},
        )


class _Skills:
    def __init__(self, prompts):
        self.prompts = prompts
        self.calls = []

    def get_prompt(self, name):
        self.calls.append(name)
        return self.prompts.get(name)


class _ReasoningProvider:
    def __init__(self, context):
        self.context = context
        self.calls = []

    async def read_campaign_reasoning_context(
        self,
        *,
        scope,
        target_id,
        target_mode,
        opportunity,
    ):
        self.calls.append({
            "scope": scope,
            "target_id": target_id,
            "target_mode": target_mode,
            "opportunity": dict(opportunity),
        })
        context = self.context
        if isinstance(context, list):
            context = context.pop(0)
        if isinstance(context, Exception):
            raise context
        return context


def _service(opportunities, responses, *, config=None, prompts=None, reasoning_context=None):
    intelligence = _Intelligence(opportunities)
    campaigns = _Campaigns()
    llm = _LLM(responses)
    default_prompts = {
        "digest/b2b_campaign_generation": (
            "Mode={target_mode}; opportunity={opportunity_json}"
        )
    }
    skills = _Skills(default_prompts if prompts is None else prompts)
    service = CampaignGenerationService(
        intelligence=intelligence,
        campaigns=campaigns,
        llm=llm,
        skills=skills,
        reasoning_context=reasoning_context,
        config=config,
    )
    return service, intelligence, campaigns, llm, skills


def test_parse_campaign_draft_response_accepts_fenced_json_and_alias_body():
    parsed = parse_campaign_draft_response(
        '```json\n{"subject":"Hi","email_body":"Body","cta":"Reply"}\n```'
    )

    assert parsed == {
        "subject": "Hi",
        "email_body": "Body",
        "cta": "Reply",
        "body": "Body",
    }


def test_parse_campaign_draft_response_finds_json_inside_prose():
    parsed = parse_campaign_draft_response(
        'Draft: {"subject":"Hi","content":"Body"}'
    )

    assert parsed == {"subject": "Hi", "content": "Body", "body": "Body"}


def test_parse_campaign_draft_response_accepts_first_item_from_array():
    parsed = parse_campaign_draft_response(
        '[{"subject":"Hi","body":"Body"},{"subject":"Second","body":"No"}]'
    )

    assert parsed == {"subject": "Hi", "body": "Body"}


def test_opportunity_target_id_prefers_stable_ids_then_names():
    assert opportunity_target_id({"target_id": "target-1", "company_name": "Acme"}) == "target-1"
    assert opportunity_target_id({"id": "row-1"}) == "row-1"
    assert opportunity_target_id({"company_name": "Acme"}) == "Acme"
    assert opportunity_target_id({"company": "Acme"}) == "Acme"
    assert opportunity_target_id({}) == ""


def test_normalize_campaign_opportunity_adds_customer_data_contract_fields():
    normalized = normalize_campaign_opportunity(
        {
            "id": "opp-1",
            "company": " Acme ",
            "vendor": "HubSpot",
            "email": "buyer@example.com",
            "title": "VP Revenue",
            "pain_category": "pricing",
            "competitor": "Salesforce, Zoho",
            "opportunity_score": "84.0",
            "custom_segment": "enterprise",
        },
        target_mode="vendor_retention",
    )

    assert normalized["target_id"] == "opp-1"
    assert normalized["company_name"] == "Acme"
    assert normalized["vendor_name"] == "HubSpot"
    assert normalized["contact_email"] == "buyer@example.com"
    assert normalized["contact_title"] == "VP Revenue"
    assert normalized["target_mode"] == "vendor_retention"
    assert normalized["pain_points"] == ["pricing"]
    assert normalized["competitors"] == ["Salesforce", "Zoho"]
    assert normalized["opportunity_score"] == 84
    assert normalized["custom_segment"] == "enterprise"


@pytest.mark.asyncio
async def test_generate_reads_opportunities_prompts_llm_and_saves_drafts():
    scope = TenantScope(account_id="acct-1")
    opportunity = {
        "id": "opp-1",
        "company_name": "Acme",
        "pain": "pricing pressure",
    }
    service, intelligence, campaigns, llm, skills = _service(
        [opportunity],
        [json.dumps({
            "subject": "Acme pricing signal",
            "body": "<p>Pricing note</p>",
            "cta": "Book time",
            "angle_reasoning": "Pricing complaints are rising.",
        })],
    )

    result = await service.generate(
        scope=scope,
        target_mode="churning_company",
        limit=5,
        filters={"vendor": "HubSpot"},
    )

    assert result.as_dict() == {
        "requested": 1,
        "generated": 1,
        "skipped": 0,
        "saved_ids": ["draft-1"],
        "errors": [],
    }
    assert intelligence.calls == [{
        "scope": scope,
        "target_mode": "churning_company",
        "limit": 5,
        "filters": {"vendor": "HubSpot"},
    }]
    assert skills.calls == ["digest/b2b_campaign_generation"]
    llm_call = llm.calls[0]
    assert llm_call["max_tokens"] == 1200
    assert llm_call["temperature"] == 0.4
    assert '"company_name":"Acme"' in llm_call["messages"][0].content
    assert '"target_id":"opp-1"' in llm_call["messages"][0].content
    assert '"target_mode":"churning_company"' in llm_call["messages"][0].content
    assert '"company_name":"Acme"' in llm_call["messages"][1].content
    assert "target_mode=churning_company" in llm_call["messages"][1].content
    assert "channel=email" in llm_call["messages"][1].content
    assert llm_call["metadata"] == {
        "target_mode": "churning_company",
        "target_id": "opp-1",
        "channel": "email",
        "skill_name": "digest/b2b_campaign_generation",
    }
    draft = campaigns.saved[0]["drafts"][0]
    assert draft.target_id == "opp-1"
    assert draft.target_mode == "churning_company"
    assert draft.channel == "email"
    assert draft.subject == "Acme pricing signal"
    assert draft.body == "<p>Pricing note</p>"
    assert draft.metadata["cta"] == "Book time"
    assert draft.metadata["generation_model"] == "test-model"
    assert draft.metadata["source_opportunity"]["target_id"] == "opp-1"
    assert draft.metadata["source_opportunity"]["target_mode"] == "churning_company"
    assert draft.metadata["source_opportunity"]["channel"] == "email"
    assert draft.metadata["source_opportunity"]["company_name"] == "Acme"
    assert draft.metadata["source_opportunity"]["pain"] == "pricing pressure"


@pytest.mark.asyncio
async def test_generate_includes_opportunity_payload_when_skill_has_no_placeholders():
    service, _, _, llm, _ = _service(
        [{"id": "opp-1", "company_name": "Acme", "pain": "pricing pressure"}],
        ['{"subject":"Hi","body":"Body"}'],
        prompts={
            "digest/b2b_campaign_generation": (
                "You receive normalized opportunity JSON and return a draft."
            )
        },
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.generated == 1
    assert "opportunity=" not in llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert "target_mode=vendor_retention" in user_prompt
    assert "channel=email" in user_prompt
    assert '"target_id":"opp-1"' in user_prompt
    assert '"company_name":"Acme"' in user_prompt
    assert '"target_mode":"vendor_retention"' in user_prompt


@pytest.mark.asyncio
async def test_generate_expands_configured_channels_and_passes_cold_context():
    config = CampaignGenerationConfig(channels=("email_cold", "email_followup"))
    service, _, campaigns, llm, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [
            json.dumps({"subject": "Cold subject", "body": "<p>Cold body</p>"}),
            json.dumps({"subject": "Follow-up subject", "body": "<p>Follow-up body</p>"}),
        ],
        config=config,
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.as_dict() == {
        "requested": 1,
        "generated": 2,
        "skipped": 0,
        "saved_ids": ["draft-1", "draft-2"],
        "errors": [],
    }
    cold, followup = campaigns.saved[0]["drafts"]
    assert cold.channel == "email_cold"
    assert cold.subject == "Cold subject"
    assert followup.channel == "email_followup"
    assert followup.subject == "Follow-up subject"
    assert cold.metadata["source_opportunity"]["channel"] == "email_cold"
    assert followup.metadata["source_opportunity"]["channel"] == "email_followup"
    assert followup.metadata["source_opportunity"]["cold_email_context"] == {
        "subject": "Cold subject",
        "body": "<p>Cold body</p>",
    }
    assert [call["metadata"]["channel"] for call in llm.calls] == [
        "email_cold",
        "email_followup",
    ]
    assert "channel=email_cold" in llm.calls[0]["messages"][1].content
    assert "channel=email_followup" in llm.calls[1]["messages"][1].content
    assert "cold_email_context" in llm.calls[1]["messages"][0].content


@pytest.mark.asyncio
async def test_generate_accepts_comma_string_channels_defensively():
    config = CampaignGenerationConfig(channels="email_cold,email_followup")  # type: ignore[arg-type]
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [
            json.dumps({"subject": "Cold subject", "body": "<p>Cold body</p>"}),
            json.dumps({"subject": "Follow-up subject", "body": "<p>Follow-up body</p>"}),
        ],
        config=config,
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 2
    assert [draft.channel for draft in campaigns.saved[0]["drafts"]] == [
        "email_cold",
        "email_followup",
    ]


@pytest.mark.asyncio
async def test_generate_uses_host_reasoning_context_without_pool_compression_import():
    scope = TenantScope(account_id="acct-1")
    opportunity = {"id": "opp-1", "company_name": "Acme"}
    provider = _ReasoningProvider(
        CampaignReasoningContext(
            anchor_examples={
                "outlier_or_named_account": (
                    {"witness_id": "w1", "excerpt_text": "Pricing drove evaluation."},
                )
            },
            witness_highlights=(
                {"witness_id": "w1", "excerpt_text": "Pricing drove evaluation."},
            ),
            reference_ids={"witness_ids": ("w1",)},
            account_signals=({"company": "Acme", "primary_pain": "pricing"},),
            timing_windows=({"window_type": "renewal", "anchor": "Q3"},),
            proof_points=({"label": "pricing_mentions", "value": 12},),
        )
    )
    service, _, campaigns, llm, _ = _service(
        [opportunity],
        ['{"subject":"Hi","body":"Body"}'],
        reasoning_context=provider,
    )

    result = await service.generate(scope=scope, target_mode="vendor_retention")

    assert result.generated == 1
    assert provider.calls[0]["scope"] == scope
    assert provider.calls[0]["target_id"] == "opp-1"
    assert provider.calls[0]["target_mode"] == "vendor_retention"
    assert provider.calls[0]["opportunity"]["target_id"] == "opp-1"
    assert provider.calls[0]["opportunity"]["target_mode"] == "vendor_retention"
    assert provider.calls[0]["opportunity"]["company_name"] == "Acme"
    prompt = llm.calls[0]["messages"][0].content
    assert "reasoning_context" in prompt
    assert "Pricing drove evaluation." in prompt
    draft = campaigns.saved[0]["drafts"][0]
    assert draft.metadata["reasoning_reference_ids"] == {"witness_ids": ["w1"]}
    assert draft.metadata["reasoning_context"]["account_signals"][0]["company"] == "Acme"
    assert draft.metadata["source_opportunity"]["reasoning_context"]["proof_points"][0]["label"] == "pricing_mentions"


@pytest.mark.asyncio
async def test_generate_defers_base_reasoning_normalization_when_provider_returns_context(
    monkeypatch,
):
    scope = TenantScope(account_id="acct-1")
    opportunity = {
        "id": "opp-1",
        "company_name": "Acme",
        "reasoning_context": {"wedge": "base context should not be read first"},
    }
    provider = _ReasoningProvider(
        CampaignReasoningContext(
            proof_points=({"label": "pricing_mentions", "value": 12},),
        )
    )
    real_normalize = campaign_generation_module.normalize_campaign_reasoning_context

    def normalize_spy(value):
        if (
            isinstance(value, dict)
            and value.get("id") == "opp-1"
            and "campaign_reasoning_context" not in value
        ):
            raise AssertionError("base opportunity normalized before provider context")
        return real_normalize(value)

    monkeypatch.setattr(
        campaign_generation_module,
        "normalize_campaign_reasoning_context",
        normalize_spy,
    )
    service, _, campaigns, _, _ = _service(
        [opportunity],
        ['{"subject":"Hi","body":"Body"}'],
        reasoning_context=provider,
    )

    result = await service.generate(scope=scope, target_mode="vendor_retention")

    assert result.generated == 1
    draft = campaigns.saved[0]["drafts"][0]
    assert draft.metadata["reasoning_context"]["proof_points"][0]["label"] == "pricing_mentions"


@pytest.mark.asyncio
async def test_generate_preserves_existing_canonical_reasoning_context():
    scope = TenantScope(account_id="acct-1")
    opportunity = {
        "id": "opp-1",
        "company_name": "Acme",
        "reasoning_context": {
            "wedge": "renewal pressure",
            "confidence": "high",
            "summary": "Acme is reviewing vendors before renewal.",
        },
    }
    provider = _ReasoningProvider(
        CampaignReasoningContext(
            witness_highlights=(
                {"witness_id": "w1", "excerpt_text": "Pricing drove evaluation."},
            ),
            proof_points=({"label": "pricing_mentions", "value": 12},),
        )
    )
    service, _, campaigns, llm, _ = _service(
        [opportunity],
        ['{"subject":"Hi","body":"Body"}'],
        reasoning_context=provider,
    )

    result = await service.generate(scope=scope, target_mode="vendor_retention")

    assert result.generated == 1
    prompt = llm.calls[0]["messages"][0].content
    assert "renewal pressure" in prompt
    assert "campaign_reasoning_context" in prompt
    source = campaigns.saved[0]["drafts"][0].metadata["source_opportunity"]
    metadata_context = campaigns.saved[0]["drafts"][0].metadata["reasoning_context"]
    assert source["reasoning_context"]["wedge"] == "renewal pressure"
    assert metadata_context["wedge"] == "renewal pressure"
    assert metadata_context["proof_points"][0]["label"] == "pricing_mentions"
    assert metadata_context["witness_highlights"][0]["witness_id"] == "w1"
    assert (
        source["reasoning_context"]["campaign_reasoning_context"]["proof_points"][0]["label"]
        == "pricing_mentions"
    )
    assert source["campaign_reasoning_context"]["witness_highlights"][0]["witness_id"] == "w1"


@pytest.mark.asyncio
async def test_generate_uses_provider_canonical_reasoning_context_without_other_evidence():
    provider = _ReasoningProvider(
        {
            "reasoning_context": {
                "wedge": "renewal pressure",
                "confidence": "high",
                "summary": "Acme is reviewing vendors before renewal.",
                "key_signals": ["pricing_mentions", "renewal_window"],
                "falsification": {"missing": ["fresh account signals"]},
            }
        }
    )
    service, _, campaigns, llm, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        ['{"subject":"Hi","body":"Body"}'],
        reasoning_context=provider,
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.generated == 1
    prompt = llm.calls[0]["messages"][0].content
    assert "renewal pressure" in prompt
    assert "Acme is reviewing vendors before renewal." in prompt
    assert "pricing_mentions" in prompt
    source = campaigns.saved[0]["drafts"][0].metadata["source_opportunity"]
    metadata_context = campaigns.saved[0]["drafts"][0].metadata["reasoning_context"]
    assert source["reasoning_context"]["wedge"] == "renewal pressure"
    assert source["reasoning_context"]["key_signals"] == [
        "pricing_mentions",
        "renewal_window",
    ]
    assert source["reasoning_context"]["falsification"] == {
        "missing": ["fresh account signals"]
    }
    assert metadata_context["key_signals"] == ["pricing_mentions", "renewal_window"]
    assert source["campaign_reasoning_context"]["confidence"] == "high"


@pytest.mark.asyncio
async def test_generate_uses_custom_config_and_omits_source_opportunity():
    config = CampaignGenerationConfig(
        skill_name="custom",
        channel="linkedin",
        max_tokens=300,
        temperature=0.2,
        include_source_opportunity=False,
    )
    service, _, campaigns, llm, skills = _service(
        [{"company_name": "Acme"}],
        ['{"subject":"Hi","body":"Body"}'],
        config=config,
        prompts={"custom": "custom prompt {target_mode} {opportunity}"},
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 1
    assert skills.calls == ["custom"]
    assert llm.calls[0]["max_tokens"] == 300
    assert llm.calls[0]["temperature"] == 0.2
    draft = campaigns.saved[0]["drafts"][0]
    assert draft.channel == "linkedin"
    assert "source_opportunity" not in draft.metadata


@pytest.mark.asyncio
async def test_generate_skips_missing_target_and_unparseable_responses():
    service, _, campaigns, _, _ = _service(
        [
            {"pain": "missing id"},
            {"id": "opp-2"},
        ],
        ["not-json"],
    )

    result = await service.generate(scope=TenantScope(), target_mode="churning_company")

    assert result.generated == 0
    assert result.skipped == 2
    assert result.errors[0]["reason"] == "missing_target_id"
    assert result.errors[1] == {
        "target_id": "opp-2",
        "channel": "email",
        "reason": "unparseable_response",
    }
    assert campaigns.saved == []


@pytest.mark.asyncio
async def test_generate_continues_after_llm_error():
    service, _, campaigns, _, _ = _service(
        [
            {"id": "opp-1"},
            {"id": "opp-2"},
        ],
        [
            RuntimeError("provider down"),
            '{"subject":"Hi","body":"Body"}',
        ],
    )

    result = await service.generate(scope=TenantScope(), target_mode="churning_company")

    assert result.generated == 1
    assert result.skipped == 1
    assert result.errors == (
        {"target_id": "opp-1", "channel": "email", "reason": "provider down"},
    )
    assert campaigns.saved[0]["drafts"][0].target_id == "opp-2"


@pytest.mark.asyncio
async def test_generate_continues_after_reasoning_context_provider_error():
    provider = _ReasoningProvider([RuntimeError("reasoning provider down"), None])
    service, _, campaigns, llm, _ = _service(
        [
            {"id": "opp-1"},
            {"id": "opp-2"},
        ],
        ['{"subject":"Hi","body":"Body"}'],
        reasoning_context=provider,
    )

    result = await service.generate(scope=TenantScope(), target_mode="churning_company")

    assert result.generated == 1
    assert result.skipped == 1
    assert result.errors == (
        {"target_id": "opp-1", "reason": "reasoning provider down"},
    )
    assert len(llm.calls) == 1
    assert campaigns.saved[0]["drafts"][0].target_id == "opp-2"


@pytest.mark.asyncio
async def test_generate_raises_clear_error_when_skill_missing():
    service, _, _, _, _ = _service(
        [{"id": "opp-1"}],
        ['{"subject":"Hi","body":"Body"}'],
        prompts={},
    )

    with pytest.raises(ValueError, match="Campaign generation skill not found"):
        await service.generate(scope=TenantScope(), target_mode="churning_company")
