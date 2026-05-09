from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.sales_brief_generation import (
    SalesBriefGenerationConfig,
    SalesBriefGenerationService,
    parse_sales_brief_response,
)
from extracted_content_pipeline.sales_brief_ports import SalesBriefDraft


# -----------------------
# Fake ports (mirror tests/test_extracted_report_generation.py:24-120)
# -----------------------


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


class _SalesBriefs:
    def __init__(self):
        self.saved = []

    async def save_drafts(self, drafts, *, scope):
        self.saved.append({"drafts": list(drafts), "scope": scope})
        return [f"sb-{index + 1}" for index, _ in enumerate(drafts)]

    async def list_drafts(self, *, scope, status=None, target_mode=None, brief_type=None, limit=None):  # pragma: no cover
        raise AssertionError("not used")

    async def update_status(self, brief_id, status, *, scope):  # pragma: no cover
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
            "metadata": dict(metadata or {}),
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
        })
        return self.context


def _opportunity():
    return {
        "id": "opp-1",
        "target_id": "vendor-acme",
        "vendor_name": "Acme",
        "company_name": "Acme",
        "evidence": [{"quote": "Renewal pricing is too high"}],
    }


def _valid_brief_json(*, title="Pre-call brief: Acme renewal", brief_type="pre_call"):
    return json.dumps({
        "title": title,
        "headline": "Renewal Q3 -- 90-day pressure window opens this week",
        "brief_type": brief_type,
        "sections": [
            {
                "id": "account_context",
                "title": "Account Context",
                "body_markdown": "Mid-market SaaS, Series C, 350 seats.",
                "claim_ids": ["c1"],
                "evidence_ids": ["r1"],
                "metadata": {"order": 1},
            },
            {
                "id": "talking_points",
                "title": "Talking Points",
                "body_markdown": "Lead with renewal-pressure framing.",
                "claim_ids": [],
                "evidence_ids": [],
                "metadata": {"order": 2},
            },
        ],
        "reference_ids": ["r1"],
        "confidence": 0.82,
    })


def _service(
    *,
    opportunities=None,
    responses=None,
    prompts=None,
    reasoning_context=None,
    config=None,
):
    intelligence = _Intelligence(opportunities or [_opportunity()])
    sales_briefs = _SalesBriefs()
    llm = _LLM(responses or [_valid_brief_json()])
    if prompts is None:
        prompts = {
            "digest/sales_brief_generation": "TEMPLATE for {target_mode}: {opportunity_json}"
        }
    skills = _Skills(prompts)
    reasoning_provider = (
        _ReasoningProvider(reasoning_context) if reasoning_context is not None else None
    )
    service = SalesBriefGenerationService(
        intelligence=intelligence,
        sales_briefs=sales_briefs,
        llm=llm,
        skills=skills,
        reasoning_context=reasoning_provider,
        config=config or SalesBriefGenerationConfig(),
    )
    return service, intelligence, sales_briefs, llm, skills, reasoning_provider


# -----------------------
# parse_sales_brief_response
# -----------------------


def test_parse_sales_brief_response_strips_code_fences_and_extracts_first_object() -> None:
    text = "```json\n" + _valid_brief_json() + "\n```"
    parsed = parse_sales_brief_response(text)
    assert parsed is not None
    assert parsed["title"] == "Pre-call brief: Acme renewal"
    assert parsed["sections"][0]["id"] == "account_context"


def test_parse_sales_brief_response_returns_none_when_required_fields_missing() -> None:
    assert parse_sales_brief_response("") is None
    assert parse_sales_brief_response('{"title": "x"}') is None  # no sections
    assert parse_sales_brief_response('{"title": "x", "sections": []}') is None  # empty sections


def test_parse_sales_brief_response_handles_braces_inside_string_values() -> None:
    """body_markdown with template syntax must not break the parser."""
    payload = json.dumps({
        "title": "Brief",
        "headline": "punchy",
        "sections": [
            {
                "id": "s1",
                "title": "Templates",
                "body_markdown": "Pricing tiers: {basic, pro} models. Use {project_id} placeholders.",
            }
        ],
    })
    parsed = parse_sales_brief_response(payload)
    assert parsed is not None
    assert "{basic, pro}" in parsed["sections"][0]["body_markdown"]


def test_parse_sales_brief_response_strips_think_blocks_before_decoding() -> None:
    payload = "<think>thinking aloud...</think>\n" + _valid_brief_json()
    parsed = parse_sales_brief_response(payload)
    assert parsed is not None
    assert parsed["title"] == "Pre-call brief: Acme renewal"


def test_parse_sales_brief_response_accepts_missing_headline_for_quality_pack_to_judge() -> None:
    """Missing headline isn't a parser-level rejection -- the quality pack
    fires ``no_headline`` so callers see exactly what was wrong."""
    payload = json.dumps({
        "title": "Brief",
        # headline intentionally omitted
        "sections": [{"id": "s1", "title": "T", "body_markdown": "b"}],
    })
    parsed = parse_sales_brief_response(payload)
    assert parsed is not None
    assert parsed["title"] == "Brief"


# -----------------------
# Service: generation
# -----------------------


@pytest.mark.asyncio
async def test_generate_persists_one_brief_per_opportunity_via_save_drafts() -> None:
    service, intelligence, sales_briefs, llm, _skills, _rp = _service()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert result.requested == 1
    assert result.generated == 1
    assert result.skipped == 0
    assert result.saved_ids == ("sb-1",)
    assert len(llm.calls) == 1
    assert llm.calls[0]["metadata"]["asset_type"] == "sales_brief"
    assert llm.calls[0]["metadata"]["target_mode"] == "vendor"
    saved_drafts = sales_briefs.saved[0]["drafts"]
    assert len(saved_drafts) == 1
    draft = saved_drafts[0]
    assert isinstance(draft, SalesBriefDraft)
    assert draft.target_id == "vendor-acme"
    assert draft.target_mode == "vendor"
    assert draft.brief_type == "pre_call"
    assert draft.title == "Pre-call brief: Acme renewal"
    assert draft.headline.startswith("Renewal Q3")
    assert draft.sections[0].id == "account_context"
    assert draft.metadata["generation_model"] == "test-model"
    assert draft.metadata["confidence"] == 0.82


@pytest.mark.asyncio
async def test_generate_substitutes_opportunity_json_into_system_prompt_only() -> None:
    """Opportunity payload appears in system content and NOT in user content."""
    service, _intel, _sb, llm, _skills, _rp = _service(
        prompts={"digest/sales_brief_generation": "TEMPLATE {target_mode}: {opportunity_json}"},
    )

    await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    system_msg = llm.calls[0]["messages"][0].content
    user_msg = llm.calls[0]["messages"][1].content
    assert '"target_id":"vendor-acme"' in system_msg
    assert "vendor-acme" not in user_msg


@pytest.mark.asyncio
async def test_generate_skips_unparseable_response() -> None:
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(
        responses=["not json garbage", "still not json"]
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert sales_briefs.saved == []
    assert any(err.get("reason") == "unparseable_response" for err in result.errors)


@pytest.mark.asyncio
async def test_generate_retries_unparseable_response_once_by_default() -> None:
    service, _intel, sales_briefs, llm, _skills, _rp = _service(
        responses=["not json garbage", _valid_brief_json()]
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert result.generated == 1
    assert len(llm.calls) == 2
    assert llm.calls[0]["metadata"]["attempt_no"] == 1
    assert llm.calls[1]["metadata"]["attempt_no"] == 2
    assert "Previous response excerpt:\nnot json garbage" in llm.calls[1]["messages"][1].content
    draft = sales_briefs.saved[0]["drafts"][0]
    assert draft.metadata["generation_parse_attempts"] == 2


@pytest.mark.asyncio
async def test_generate_can_disable_parse_retry() -> None:
    service, _intel, sales_briefs, llm, _skills, _rp = _service(
        responses=["not json garbage", _valid_brief_json()],
        config=SalesBriefGenerationConfig(parse_retry_attempts=0),
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert result.generated == 0
    assert len(llm.calls) == 1
    assert sales_briefs.saved == []
    assert any(err.get("reason") == "unparseable_response" for err in result.errors)


@pytest.mark.asyncio
async def test_generate_accumulates_usage_across_parse_retry_attempts() -> None:
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(
        responses=[
            {
                "content": "not json garbage",
                "model": "first-model",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
            {
                "content": _valid_brief_json(),
                "model": "final-model",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        ]
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert result.generated == 1
    draft = sales_briefs.saved[0]["drafts"][0]
    assert draft.metadata["generation_model"] == "final-model"
    assert draft.metadata["generation_usage"] == {"input_tokens": 12, "output_tokens": 5}
    assert draft.metadata["generation_parse_attempts"] == 2


@pytest.mark.asyncio
async def test_generate_skips_when_quality_pack_blocks_no_references() -> None:
    """A brief with no reference_ids and no per-section evidence is blocked."""
    bad_response = json.dumps({
        "title": "Brief",
        "headline": "punchy",
        "brief_type": "pre_call",
        "sections": [
            {"id": "s1", "title": "T", "body_markdown": "b", "evidence_ids": []}
        ],
        "reference_ids": [],
    })
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(responses=[bad_response])

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert sales_briefs.saved == []
    assert any(err.get("reason") == "quality_blocked" for err in result.errors)
    assert any("no_references" in b for b in result.errors[0]["blockers"])


@pytest.mark.asyncio
async def test_generate_blocks_when_response_omits_headline() -> None:
    """End-to-end: parser accepts the candidate, quality pack fires no_headline."""
    response = json.dumps({
        "title": "Pre-call brief: Acme",
        # headline omitted
        "sections": [
            {"id": "s1", "title": "T", "body_markdown": "b", "evidence_ids": ["r1"]}
        ],
        "reference_ids": ["r1"],
    })
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(responses=[response])

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert result.generated == 0
    assert sales_briefs.saved == []
    assert any("no_headline" in b for b in result.errors[0]["blockers"])


@pytest.mark.asyncio
async def test_generate_skips_when_target_id_missing() -> None:
    # No id-shaped key and no name-shaped key -> opportunity_target_id returns ""
    bad_opp = {"evidence": [{"quote": "no target"}]}
    service, _intel, sales_briefs, llm, _skills, _rp = _service(opportunities=[bad_opp])

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert sales_briefs.saved == []
    assert llm.calls == []  # no LLM round-trip
    assert any(err.get("reason") == "missing_target_id" for err in result.errors)


@pytest.mark.asyncio
async def test_generate_consumes_reasoning_context_via_provider() -> None:
    """When a CampaignReasoningContextProvider is supplied, its context is fetched per opp."""
    context = CampaignReasoningContext(
        top_theses=({"claim": "Renewal pricing", "confidence": 0.9, "source_ids": ["r1"]},),
        canonical_reasoning={"summary": "narrative summary"},
    )
    service, _intel, _sb, _llm, _skills, reasoning_provider = _service(
        reasoning_context=context
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert reasoning_provider is not None
    assert len(reasoning_provider.calls) == 1
    call = reasoning_provider.calls[0]
    assert call["target_id"] == "vendor-acme"
    assert call["target_mode"] == "vendor"
    assert result.as_dict()["reasoning_contexts_used"] == 1


@pytest.mark.asyncio
async def test_generate_raises_value_error_when_skill_prompt_missing() -> None:
    service, _intel, _sb, _llm, _skills, _rp = _service(prompts={})

    with pytest.raises(ValueError, match="Sales-brief generation skill not found"):
        await service.generate(
            scope=TenantScope(account_id="acct-1"),
            target_mode="vendor",
        )


@pytest.mark.asyncio
async def test_generate_defaults_brief_type_when_llm_omits_it() -> None:
    """Config.default_brief_type fills in when the LLM doesn't supply brief_type."""
    response = json.dumps({
        "title": "Brief",
        "headline": "punchy",
        # brief_type omitted
        "sections": [
            {"id": "s1", "title": "T", "body_markdown": "b", "evidence_ids": ["r1"]}
        ],
        "reference_ids": ["r1"],
    })
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(
        responses=[response],
        config=SalesBriefGenerationConfig(default_brief_type="discovery"),
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
    )

    assert result.generated == 1
    draft = sales_briefs.saved[0]["drafts"][0]
    assert draft.brief_type == "discovery"


@pytest.mark.asyncio
async def test_generate_aggregates_section_evidence_into_reference_ids_no_duplicates() -> None:
    """reference_ids should be the union of explicit list + per-section evidence ids."""
    response = json.dumps({
        "title": "Brief",
        "headline": "punchy",
        "brief_type": "pre_call",
        "sections": [
            {
                "id": "s1",
                "title": "T",
                "body_markdown": "b",
                "evidence_ids": ["r1", "r2"],
            },
            {
                "id": "s2",
                "title": "U",
                "body_markdown": "c",
                "evidence_ids": ["r2", "r3"],  # r2 dup
            },
        ],
        "reference_ids": ["r1", "r4"],
    })
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(responses=[response])

    await service.generate(scope=TenantScope(account_id="acct-1"), target_mode="vendor")

    draft = sales_briefs.saved[0]["drafts"][0]
    # Order preserved: explicit list first, then section evidence ids in insertion order
    assert draft.reference_ids == ("r1", "r4", "r2", "r3")


# -----------------------
# PR-OptionA-1: per-call default_brief_type override
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_default_brief_type_override_when_llm_omits_type():
    """When the LLM doesn't include a `brief_type` in its JSON, the per-call
    override (from the plan's step.config) wins over the construction-time
    default."""

    response = json.dumps({
        "title": "Acme",
        "headline": "elevator",
        # brief_type intentionally omitted
        "sections": [{"id": "s1", "title": "S", "body_markdown": "b", "evidence_ids": ["r1"]}],
        "reference_ids": ["r1"],
    })
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(
        responses=[response],
        config=SalesBriefGenerationConfig(default_brief_type="pre_call"),
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
        default_brief_type="renewal",  # plan-supplied override
    )

    saved_drafts = sales_briefs.saved[0]["drafts"]
    assert saved_drafts[0].brief_type == "renewal"


@pytest.mark.asyncio
async def test_generate_llm_supplied_brief_type_still_wins_over_per_call_override():
    """The per-call override only fills in when the LLM omits `brief_type`."""

    service, _intel, sales_briefs, _llm, _skills, _rp = _service(
        responses=[_valid_brief_json(brief_type="pre_call")],
        config=SalesBriefGenerationConfig(default_brief_type="pre_call"),
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
        default_brief_type="renewal",  # ignored: LLM supplied pre_call
    )

    saved_drafts = sales_briefs.saved[0]["drafts"]
    assert saved_drafts[0].brief_type == "pre_call"


@pytest.mark.asyncio
async def test_generate_per_call_default_brief_type_falls_back_when_none():
    """A None override leaves the construction-time default in place."""

    response = json.dumps({
        "title": "Acme",
        "headline": "elevator",
        "sections": [{"id": "s1", "title": "S", "body_markdown": "b", "evidence_ids": ["r1"]}],
        "reference_ids": ["r1"],
    })
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(
        responses=[response],
        config=SalesBriefGenerationConfig(default_brief_type="pre_call"),
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
        default_brief_type=None,
    )

    saved_drafts = sales_briefs.saved[0]["drafts"]
    assert saved_drafts[0].brief_type == "pre_call"


# -----------------------
# PR-OptionA-2: per-call temperature/max_tokens/parse_retry_attempts overrides
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_llm_tuning_overrides_win_over_construction_config():
    """Resolved-value param reaches the LLM, not self._config.X."""

    service, _intel, _briefs, llm, _skills, _rp = _service(
        responses=[
            "not parseable",
            _valid_brief_json(),
        ],
        config=SalesBriefGenerationConfig(
            temperature=0.3,
            max_tokens=4096,
            parse_retry_attempts=0,
        ),
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
        temperature=0.95,
        max_tokens=2048,
        parse_retry_attempts=1,
    )

    assert len(llm.calls) == 2
    for call in llm.calls:
        assert call["temperature"] == 0.95
        assert call["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_generate_llm_tuning_kwargs_none_falls_back_to_construction_config():
    service, _intel, _briefs, llm, _skills, _rp = _service(
        responses=[_valid_brief_json()],
        config=SalesBriefGenerationConfig(temperature=0.7, max_tokens=999),
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
        temperature=None,
        max_tokens=None,
        parse_retry_attempts=None,
    )

    assert llm.calls[0]["temperature"] == 0.7
    assert llm.calls[0]["max_tokens"] == 999


# -----------------------
# PR-OptionA-3: per-call parse_retry_response_excerpt_chars override
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_parse_retry_response_excerpt_chars_override():
    long_invalid = "X" * 5000
    service, _intel, _briefs, llm, _skills, _rp = _service(
        responses=[long_invalid, _valid_brief_json()],
        config=SalesBriefGenerationConfig(
            parse_retry_attempts=1,
            parse_retry_response_excerpt_chars=200,
        ),
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
        parse_retry_response_excerpt_chars=50,
    )

    retry_user_prompt = llm.calls[1]["messages"][1].content
    assert "XXX" in retry_user_prompt
    excerpt_section = retry_user_prompt.split("excerpt:")[1].lstrip()
    assert len(excerpt_section.rstrip()) <= 50


# -----------------------
# PR-OptionA-4: per-call quality_gates_enabled override
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_quality_gates_enabled_false_skips_quality_gate() -> None:
    """quality_gates_enabled=False short-circuits the quality gate, even on
    parses that would normally fail (e.g., missing reference_ids)."""

    bad_response = json.dumps({
        "title": "Brief",
        "headline": "punchy",
        "brief_type": "pre_call",
        "sections": [
            {"id": "s1", "title": "T", "body_markdown": "b", "evidence_ids": []}
        ],
        "reference_ids": [],  # would normally hit no_references blocker
    })
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(responses=[bad_response])

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
        quality_gates_enabled=False,
    )

    # Gate skipped -> the bad parse is persisted.
    assert result.generated == 1
    assert len(sales_briefs.saved[0]["drafts"]) == 1


@pytest.mark.asyncio
async def test_generate_per_call_quality_gates_enabled_true_still_blocks() -> None:
    bad_response = json.dumps({
        "title": "Brief",
        "headline": "punchy",
        "brief_type": "pre_call",
        "sections": [
            {"id": "s1", "title": "T", "body_markdown": "b", "evidence_ids": []}
        ],
        "reference_ids": [],
    })
    service, _intel, sales_briefs, _llm, _skills, _rp = _service(responses=[bad_response])

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor",
        quality_gates_enabled=True,
    )

    assert result.generated == 0
    assert sales_briefs.saved == []
