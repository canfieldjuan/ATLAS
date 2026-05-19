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
        if isinstance(response, dict):
            return LLMResponse(
                content=response["content"],
                model=response.get("model", "test-model"),
                usage=response.get("usage", {}),
            )
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
        "reasoning_contexts_used": 0,
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
        "attempt_no": 1,
    }
    draft = campaigns.saved[0]["drafts"][0]
    assert draft.target_id == "opp-1"
    assert draft.target_mode == "churning_company"
    assert draft.channel == "email"
    assert draft.subject == "Acme pricing signal"
    assert draft.body == "<p>Pricing note</p>"
    assert draft.metadata["cta"] == "Book time"
    assert draft.metadata["generation_model"] == "test-model"
    assert draft.metadata["generation_parse_attempts"] == 1
    assert draft.metadata["source_opportunity"]["target_id"] == "opp-1"
    assert draft.metadata["source_opportunity"]["target_mode"] == "churning_company"
    assert draft.metadata["source_opportunity"]["channel"] == "email"
    assert draft.metadata["source_opportunity"]["company_name"] == "Acme"
    assert draft.metadata["source_opportunity"]["pain"] == "pricing pressure"


@pytest.mark.asyncio
async def test_generate_skips_draft_with_placeholder_url_in_body():
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [json.dumps({
            "subject": "Acme signal",
            "body": '<p>Read the analysis: <a href="https://example.com/report">Report</a></p>',
            "cta": "Read the report",
        })],
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors == ({
        "target_id": "opp-1",
        "channel": "email",
        "reason": "placeholder_url",
    },)
    assert campaigns.saved == []


@pytest.mark.asyncio
async def test_generate_skips_draft_with_placeholder_url_in_cta():
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [json.dumps({
            "subject": "Acme signal",
            "body": "<p>Pricing note</p>",
            "cta": "Book time: http://localhost:3000/demo",
        })],
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors == ({
        "target_id": "opp-1",
        "channel": "email",
        "reason": "placeholder_url",
    },)
    assert campaigns.saved == []


@pytest.mark.asyncio
async def test_generate_skips_scheme_less_placeholder_urls():
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [json.dumps({
            "subject": "Acme signal",
            "body": "Read the analysis at example.com/report",
            "cta": "Book time at localhost:3000/demo",
        })],
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors == ({
        "target_id": "opp-1",
        "channel": "email",
        "reason": "placeholder_url",
    },)
    assert campaigns.saved == []


@pytest.mark.asyncio
async def test_generate_skips_scheme_less_placeholder_url_with_www_prefix():
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [json.dumps({
            "subject": "Acme signal",
            "body": "Read the summary at www.example.com/updates",
            "cta": "Keep building",
        })],
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors == ({
        "target_id": "opp-1",
        "channel": "email",
        "reason": "placeholder_url",
    },)
    assert campaigns.saved == []


@pytest.mark.asyncio
async def test_generate_skips_placeholder_url_before_sentence_punctuation():
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [json.dumps({
            "subject": "Acme signal",
            "body": "Read the summary at example.com.",
            "cta": "Book time at localhost, then bring your team.",
        })],
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors == ({
        "target_id": "opp-1",
        "channel": "email",
        "reason": "placeholder_url",
    },)
    assert campaigns.saved == []


@pytest.mark.asyncio
async def test_generate_skips_placeholder_subdomain_url():
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [json.dumps({
            "subject": "Acme signal",
            "body": "Read the summary at https://demo.example.com/report",
            "cta": "Keep building",
        })],
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors == ({
        "target_id": "opp-1",
        "channel": "email",
        "reason": "placeholder_url",
    },)
    assert campaigns.saved == []


@pytest.mark.asyncio
async def test_generate_does_not_treat_email_address_as_placeholder_url():
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [json.dumps({
            "subject": "Acme signal",
            "body": "Coordinate with ops@example.com before rollout.",
            "cta": "Book time",
        })],
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
    )

    assert result.generated == 1
    assert result.skipped == 0
    assert result.errors == ()
    assert len(campaigns.saved) == 1


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
        "reasoning_contexts_used": 0,
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
    result_dict = result.as_dict()
    assert result_dict["reasoning_contexts_used"] == 1
    assert result_dict["consumed_reasoning_contexts"][0]["proof_points"][0]["label"] == (
        "pricing_mentions"
    )
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
    assert result.as_dict()["reasoning_contexts_used"] == 1
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
        channels=("linkedin",),  # supported multi-channel field
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
async def test_generate_quality_revalidation_is_opt_in() -> None:
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        ['{"subject":"Hi [Name]","body":"Body"}'],
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 1
    assert campaigns.saved[0]["drafts"][0].subject == "Hi [Name]"
    assert "campaign_revalidation" not in campaigns.saved[0]["drafts"][0].metadata


@pytest.mark.asyncio
async def test_generate_quality_revalidation_blocks_failed_drafts() -> None:
    config = CampaignGenerationConfig(quality_revalidation_enabled=True)
    service, _, campaigns, _, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        ['{"subject":"Hi [Name]","body":"Body"}'],
        config=config,
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors == (
        {
            "target_id": "opp-1",
            "channel": "email",
            "reason": "quality_revalidation_failed",
            "quality_revalidation": {
                "status": "fail",
                "blocking_issues": ["placeholder_token"],
                "primary_blocker": "placeholder_token",
            },
        },
    )
    assert campaigns.saved == []


@pytest.mark.asyncio
async def test_generate_quality_revalidation_failure_includes_proof_term_details() -> None:
    config = CampaignGenerationConfig(quality_revalidation_enabled=True)
    opportunity = {
        "id": "opp-1",
        "company_name": "Acme",
        "anchor_examples": {
            "pricing": [
                {"excerpt_text": "Pricing drove evaluation."},
            ]
        },
    }
    service, _, campaigns, _, _ = _service(
        [opportunity],
        ['{"subject":"Pricing signal","body":"Generic body."}'],
        config=config,
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 0
    assert result.skipped == 1
    error = result.errors[0]
    assert error["reason"] == "quality_revalidation_failed"
    assert error["quality_revalidation"]["blocking_issues"] == ["missing_anchor_support"]
    assert error["quality_revalidation"]["primary_blocker"] == "missing_anchor_support"
    assert error["quality_revalidation"]["unused_proof_terms"] == [
        "Pricing drove evaluation."
    ]
    assert campaigns.saved == []


@pytest.mark.asyncio
async def test_generate_quality_revalidation_preserves_pass_audit_metadata() -> None:
    config = CampaignGenerationConfig(quality_revalidation_enabled=True)
    opportunity = {
        "id": "opp-1",
        "company_name": "Acme",
        "anchor_examples": {
            "pricing": [
                {"excerpt_text": "Pricing drove evaluation."},
            ]
        },
    }
    service, _, campaigns, _, _ = _service(
        [opportunity],
        [
            json.dumps({
                "subject": "Pricing signal",
                "body": "Pricing drove evaluation.",
            })
        ],
        config=config,
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 1
    metadata = campaigns.saved[0]["drafts"][0].metadata
    audit = metadata["campaign_revalidation"]["audit"]
    assert audit["status"] == "pass"
    assert audit["used_proof_terms"] == ["Pricing drove evaluation."]
    assert metadata["campaign_revalidation"]["metadata"]["campaign_proof_terms"] == [
        "Pricing drove evaluation."
    ]


@pytest.mark.asyncio
async def test_generate_quality_revalidation_adds_prompt_proof_terms() -> None:
    config = CampaignGenerationConfig(quality_revalidation_enabled=True)
    opportunity = {
        "id": "opp-1",
        "company_name": "Acme",
        "anchor_examples": {
            "pricing": [
                {"excerpt_text": "Pricing drove evaluation."},
            ]
        },
    }
    service, _, campaigns, llm, _ = _service(
        [opportunity],
        [
            json.dumps({
                "subject": "Pricing signal",
                "body": "Pricing drove evaluation.",
            })
        ],
        config=config,
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 1
    prompt = llm.calls[0]["messages"][0].content
    assert '"campaign_proof_terms":["Pricing drove evaluation."]' in prompt
    source = campaigns.saved[0]["drafts"][0].metadata["source_opportunity"]
    assert source["campaign_proof_terms"] == ["Pricing drove evaluation."]


@pytest.mark.asyncio
async def test_generate_quality_revalidation_preserves_existing_prompt_terms() -> None:
    config = CampaignGenerationConfig(
        quality_revalidation_enabled=True,
        quality_prompt_proof_term_limit=2,
    )
    opportunity = {
        "id": "opp-1",
        "company_name": "Acme",
        "campaign_proof_terms": ["Host supplied proof."],
        "anchor_examples": {
            "pricing": [
                {"excerpt_text": "Pricing drove evaluation."},
            ]
        },
    }
    service, _, campaigns, llm, _ = _service(
        [opportunity],
        [
            json.dumps({
                "subject": "Proof signal",
                "body": "Host supplied proof.",
            })
        ],
        config=config,
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 1
    prompt = llm.calls[0]["messages"][0].content
    assert (
        '"campaign_proof_terms":["Host supplied proof.","Pricing drove evaluation."]'
        in prompt
    )
    audit = campaigns.saved[0]["drafts"][0].metadata["campaign_revalidation"]["audit"]
    assert audit["used_proof_terms"] == ["Host supplied proof."]


@pytest.mark.asyncio
async def test_generate_quality_revalidation_allows_zero_prompt_terms() -> None:
    config = CampaignGenerationConfig(
        quality_revalidation_enabled=True,
        quality_prompt_proof_term_limit=0,
    )
    opportunity = {
        "id": "opp-1",
        "company_name": "Acme",
        "anchor_examples": {
            "pricing": [
                {"excerpt_text": "Pricing drove evaluation."},
            ]
        },
    }
    service, _, campaigns, llm, _ = _service(
        [opportunity],
        [
            json.dumps({
                "subject": "Pricing signal",
                "body": "Pricing drove evaluation.",
            })
        ],
        config=config,
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 1
    prompt = llm.calls[0]["messages"][0].content
    assert "campaign_proof_terms" not in prompt
    source = campaigns.saved[0]["drafts"][0].metadata["source_opportunity"]
    assert "campaign_proof_terms" not in source


@pytest.mark.asyncio
async def test_generate_quality_revalidation_zero_limit_removes_existing_prompt_terms() -> None:
    config = CampaignGenerationConfig(
        quality_revalidation_enabled=True,
        quality_prompt_proof_term_limit=0,
    )
    opportunity = {
        "id": "opp-1",
        "company_name": "Acme",
        "campaign_proof_terms": ["Host supplied proof."],
        "anchor_examples": {
            "pricing": [
                {"excerpt_text": "Pricing drove evaluation."},
            ]
        },
    }
    service, _, campaigns, llm, _ = _service(
        [opportunity],
        [
            json.dumps({
                "subject": "Pricing signal",
                "body": "Pricing drove evaluation.",
            })
        ],
        config=config,
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 1
    prompt = llm.calls[0]["messages"][0].content
    assert "campaign_proof_terms" not in prompt
    source = campaigns.saved[0]["drafts"][0].metadata["source_opportunity"]
    assert "campaign_proof_terms" not in source


@pytest.mark.asyncio
async def test_generate_skips_missing_target_and_unparseable_responses():
    service, _, campaigns, _, _ = _service(
        [
            {"pain": "missing id"},
            {"id": "opp-2"},
        ],
        ["not-json", "still-not-json"],
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
async def test_generate_retries_unparseable_response_once_by_default():
    service, _, campaigns, llm, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [
            "not-json",
            '{"subject":"Recovered","body":"Recovered body"}',
        ],
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 1
    assert result.skipped == 0
    assert [call["metadata"]["attempt_no"] for call in llm.calls] == [1, 2]
    retry_prompt = llm.calls[1]["messages"][1].content
    assert "previous response was not valid campaign JSON" in retry_prompt
    assert "not-json" in retry_prompt
    draft = campaigns.saved[0]["drafts"][0]
    assert draft.subject == "Recovered"
    assert draft.metadata["generation_parse_attempts"] == 2


@pytest.mark.asyncio
async def test_generate_accumulates_usage_across_parse_retry_attempts():
    service, _, campaigns, _llm, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        [
            {
                "content": "not-json",
                "model": "first-model",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
            {
                "content": '{"subject":"Recovered","body":"Recovered body"}',
                "model": "final-model",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        ],
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 1
    draft = campaigns.saved[0]["drafts"][0]
    assert draft.metadata["generation_model"] == "final-model"
    assert draft.metadata["generation_usage"] == {
        "input_tokens": 12,
        "output_tokens": 5,
    }
    assert draft.metadata["generation_parse_attempts"] == 2


@pytest.mark.asyncio
async def test_generate_parse_retry_attempts_can_be_disabled():
    config = CampaignGenerationConfig(parse_retry_attempts=0)
    service, _, campaigns, llm, _ = _service(
        [{"id": "opp-1", "company_name": "Acme"}],
        ["not-json"],
        config=config,
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention")

    assert result.generated == 0
    assert result.skipped == 1
    assert len(llm.calls) == 1
    assert result.errors == (
        {"target_id": "opp-1", "channel": "email", "reason": "unparseable_response"},
    )
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


# -----------------------
# PR-OptionA-1: per-call channels override (plan-as-execution-contract)
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_channels_override_wins_over_construction_config():
    """When the executor threads `channels=` from step.config, the call wins
    over the construction-time config. PR-OptionA-1 makes this load-bearing."""

    service, _intel, campaigns, llm, _skills = _service(
        [{"id": "opp-1"}],
        ['{"subject":"Hi","body":"Body"}'],
        config=CampaignGenerationConfig(channels=("email_cold", "email_followup")),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="churning_company",
        channels=["email_cold"],  # plan picks cold-only
    )

    # The construction-time config had two channels; the per-call override
    # narrowed it to one, so the LLM is invoked exactly once.
    assert len(llm.calls) == 1
    drafts = campaigns.saved[0]["drafts"]
    assert len(drafts) == 1
    assert drafts[0].channel == "email_cold"


@pytest.mark.asyncio
async def test_generate_per_call_channels_override_falls_back_when_none():
    """A None override leaves the construction-time channels config in place."""

    service, _intel, campaigns, llm, _skills = _service(
        [{"id": "opp-1"}],
        [
            '{"subject":"Hi","body":"Body"}',
            '{"subject":"Hi","body":"Body"}',
        ],
        config=CampaignGenerationConfig(channels=("email_cold", "email_followup")),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="churning_company",
        channels=None,  # no per-call override
    )

    # Both channels from the construction-time config were used.
    assert len(llm.calls) == 2
    assert {d.channel for d in campaigns.saved[0]["drafts"]} == {"email_cold", "email_followup"}


@pytest.mark.asyncio
async def test_generate_per_call_channels_override_empty_falls_back_to_default():
    """An empty override is treated as "no override", not "no channels"."""

    service, _intel, campaigns, llm, _skills = _service(
        [{"id": "opp-1"}],
        [
            '{"subject":"Hi","body":"Body"}',
            '{"subject":"Hi","body":"Body"}',
        ],
        config=CampaignGenerationConfig(channels=("email_cold", "email_followup")),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="churning_company",
        channels=[],  # empty override -> fall back, don't zero out
    )

    assert len(llm.calls) == 2
    assert {d.channel for d in campaigns.saved[0]["drafts"]} == {"email_cold", "email_followup"}


# -----------------------
# PR-OptionA-2: per-call temperature/max_tokens/parse_retry_attempts overrides
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_llm_tuning_overrides_win_over_construction_config():
    """Per-call temperature/max_tokens/parse_retry_attempts kwargs override
    the construction-time config; the LLM client receives the overridden
    values and the retry budget honors the override."""

    service, _, _, llm, _ = _service(
        [{"id": "opp-1"}],
        [
            "not parseable",  # forces a retry
            "still not parseable",
            '{"subject":"Hi","body":"Body"}',
        ],
        config=CampaignGenerationConfig(
            temperature=0.4,
            max_tokens=1200,
            parse_retry_attempts=0,  # construction default would be 1 LLM call
            channels=("email_cold",),
        ),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="churning_company",
        temperature=0.95,  # plan-supplied
        max_tokens=2048,
        parse_retry_attempts=2,  # 1 + 2 retries = 3 calls total
    )

    # 3 calls = override raised retry budget from 0 to 2.
    assert len(llm.calls) == 3
    # Every call honors the per-call temperature / max_tokens override.
    for call in llm.calls:
        assert call["temperature"] == 0.95
        assert call["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_generate_llm_tuning_kwargs_none_falls_back_to_construction_config():
    """A None override leaves the construction-time config in place."""

    service, _, _, llm, _ = _service(
        [{"id": "opp-1"}],
        ['{"subject":"Hi","body":"Body"}'],
        config=CampaignGenerationConfig(
            temperature=0.7,
            max_tokens=999,
            channels=("email_cold",),
        ),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="churning_company",
        temperature=None,
        max_tokens=None,
        parse_retry_attempts=None,
    )

    assert llm.calls[0]["temperature"] == 0.7
    assert llm.calls[0]["max_tokens"] == 999


# -----------------------
# PR-OptionA-3: per-call quality + retry-excerpt overrides
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_quality_revalidation_enabled_override():
    """When the executor passes quality_revalidation_enabled=False, the
    revalidation step is skipped even though config has it on (and vice
    versa). The smoking-gun: an operator picking 'skip revalidation' in
    the control surface actually skips it."""

    valid = '{"subject":"Hi","body":"Body"}'
    service, _, _, llm, _ = _service(
        [{"id": "opp-1"}],
        [valid],
        config=CampaignGenerationConfig(
            quality_revalidation_enabled=True,  # construction default
            channels=("email_cold",),
        ),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="churning_company",
        quality_revalidation_enabled=False,  # override to skip
    )

    # When revalidation is skipped, no revalidation LLM call happens; only
    # the initial generation LLM call. (Construction-time True would have
    # forced an extra revalidation call for each generated draft.)
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_generate_per_call_parse_retry_response_excerpt_chars_override():
    """The override controls how much of a prior invalid response gets
    embedded in the retry user prompt."""

    long_invalid = "X" * 5000
    service, _, _, llm, _ = _service(
        [{"id": "opp-1"}],
        [long_invalid, '{"subject":"Hi","body":"Body"}'],
        config=CampaignGenerationConfig(
            channels=("email_cold",),
            parse_retry_attempts=1,
            parse_retry_response_excerpt_chars=200,  # construction default
        ),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="churning_company",
        parse_retry_response_excerpt_chars=50,  # tighter cap
    )

    # Second LLM call's user message should contain at most 50 chars of
    # the prior invalid response.
    retry_user_prompt = llm.calls[1]["messages"][1].content
    # Prompt format: "...Previous response excerpt:\n<excerpt>"
    assert "XXX" in retry_user_prompt  # excerpt is present
    excerpt_section = retry_user_prompt.split("excerpt:")[1].lstrip()
    # Excerpt is clipped to 50 chars; the construction-time default of 200
    # would have shown 200 chars.
    assert len(excerpt_section.rstrip()) <= 50


@pytest.mark.asyncio
async def test_generate_per_call_quality_prompt_proof_term_limit_override():
    """The override caps how many proof terms reach the prompt. With 5 terms
    in the opportunity and an override of 2, only 2 terms should land in the
    system prompt's serialized opportunity payload. Catches typos in the
    deep helper chain (_with_quality_prompt_terms -> _campaign_proof_terms)
    where the threaded param could be dropped."""

    service, _, _, llm, _ = _service(
        [{"id": "opp-1", "campaign_proof_terms": ["t1", "t2", "t3", "t4", "t5"]}],
        ['{"subject":"Hi","body":"Body"}'],
        config=CampaignGenerationConfig(
            quality_revalidation_enabled=True,
            quality_prompt_proof_term_limit=5,  # construction default would allow all 5
            channels=("email_cold",),
        ),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="churning_company",
        quality_prompt_proof_term_limit=2,  # override caps at 2
    )

    # Proof terms surface in the system prompt via the opportunity JSON.
    system_prompt = llm.calls[0]["messages"][0].content
    found_terms = [term for term in ("t1", "t2", "t3", "t4", "t5") if term in system_prompt]
    assert len(found_terms) <= 2, (
        f"override should cap proof terms at 2; saw {len(found_terms)}: {found_terms}"
    )
