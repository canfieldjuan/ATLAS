from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationConfig,
    LandingPageGenerationService,
    parse_landing_page_response,
)
from extracted_content_pipeline.landing_page_ports import (
    LandingPageDraft,
    MarketingCampaign,
)


# -----------------------
# Fake ports
# -----------------------


class _LandingPages:
    def __init__(self):
        self.saved = []

    async def save_drafts(self, drafts, *, scope):
        self.saved.append({"drafts": list(drafts), "scope": scope})
        return [f"lp-{index + 1}" for index, _ in enumerate(drafts)]

    async def list_drafts(self, *, scope, status=None, campaign_name=None, slug=None, limit=None):  # pragma: no cover
        raise AssertionError("not used")

    async def update_status(self, landing_page_id, status, *, scope):  # pragma: no cover
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

    async def read_campaign_reasoning_context(self, *, scope, target_id, target_mode, opportunity):
        self.calls.append({
            "scope": scope,
            "target_id": target_id,
            "target_mode": target_mode,
        })
        return self.context


def _campaign() -> MarketingCampaign:
    return MarketingCampaign(
        name="acme-q3-launch",
        persona="VP Engineering at mid-market SaaS",
        value_prop="Cut renewal pricing leakage by 40%",
        vendors=("Acme",),
        tags=("growth", "renewal"),
    )


def _valid_response(*, slug="acme-q3-launch") -> str:
    return json.dumps({
        "title": "Acme Q3: Stop Renewal Surprises",
        "slug": slug,
        "hero": {
            "headline": "Stop renewal surprises",
            "subheadline": "Acme catches pressure 90 days early",
            "cta_label": "Book a 15-min demo",
            "cta_url": "/demo",
        },
        "sections": [
            {
                "id": "problem",
                "title": "The Problem",
                "body_markdown": "Renewal pricing is the #1 churn driver.",
                "metadata": {"order": 1},
            },
            {
                "id": "solution",
                "title": "The Solution",
                "body_markdown": "Acme surfaces pressure signals in the first 30 days.",
                "metadata": {"order": 2},
            },
        ],
        "cta": {"label": "Book a 15-min demo", "url": "/demo", "variant": "primary"},
        "meta": {
            "title_tag": "Stop Renewal Surprises | Acme",
            "description": "Acme catches renewal pressure 90 days early so you can prevent unplanned churn.",
        },
        "reference_ids": ["customer-logo-1"],
    })


def _service(*, responses=None, prompts=None, reasoning_context=None, config=None):
    landing_pages = _LandingPages()
    llm = _LLM(responses or [_valid_response()])
    if prompts is None:
        prompts = {"digest/landing_page_generation": "TEMPLATE: {campaign_json}"}
    skills = _Skills(prompts)
    reasoning_provider = (
        _ReasoningProvider(reasoning_context) if reasoning_context is not None else None
    )
    service = LandingPageGenerationService(
        landing_pages=landing_pages,
        llm=llm,
        skills=skills,
        reasoning_context=reasoning_provider,
        config=config or LandingPageGenerationConfig(),
    )
    return service, landing_pages, llm, skills, reasoning_provider


# -----------------------
# parse_landing_page_response
# -----------------------


def test_parse_landing_page_response_strips_code_fences_and_extracts_first_object() -> None:
    text = "```json\n" + _valid_response() + "\n```"
    parsed = parse_landing_page_response(text)
    assert parsed is not None
    assert parsed["title"] == "Acme Q3: Stop Renewal Surprises"
    assert parsed["sections"][0]["id"] == "problem"


def test_parse_landing_page_response_returns_none_when_required_fields_missing() -> None:
    assert parse_landing_page_response("") is None
    assert parse_landing_page_response('{"title": "x"}') is None  # no sections
    assert parse_landing_page_response('{"title": "x", "sections": []}') is None  # empty sections


def test_parse_landing_page_response_handles_braces_inside_string_values() -> None:
    """body_markdown with template syntax / set notation must not break the parser."""
    payload = json.dumps({
        "title": "Title",
        "slug": "slug",
        "hero": {"headline": "h", "subheadline": "s", "cta_label": "L", "cta_url": "/u"},
        "sections": [
            {
                "id": "s1",
                "title": "Templates",
                "body_markdown": "Pricing tiers: {basic, pro} models. Use {project_id} placeholders.",
            }
        ],
        "cta": {"label": "L", "url": "/u"},
        "meta": {},
        "reference_ids": [],
    })
    parsed = parse_landing_page_response(payload)
    assert parsed is not None
    assert "{basic, pro}" in parsed["sections"][0]["body_markdown"]


def test_parse_landing_page_response_strips_think_blocks_before_decoding() -> None:
    payload = "<think>thinking aloud about the campaign...</think>\n" + _valid_response()
    parsed = parse_landing_page_response(payload)
    assert parsed is not None
    assert parsed["title"] == "Acme Q3: Stop Renewal Surprises"


def test_parse_landing_page_response_accepts_missing_hero_for_quality_pack_to_judge() -> None:
    """Missing/non-mapping hero is NOT a parser-level rejection -- the quality pack
    raises ``no_hero_headline`` so callers see exactly what was wrong rather than a
    generic ``unparseable_response``."""
    payload = json.dumps({
        "title": "Title",
        "slug": "slug",
        # hero intentionally omitted
        "sections": [{"id": "s1", "title": "T", "body_markdown": "b"}],
        "cta": {"label": "L", "url": "/u"},
        "meta": {},
    })
    parsed = parse_landing_page_response(payload)
    assert parsed is not None
    assert parsed["title"] == "Title"


@pytest.mark.asyncio
async def test_generate_blocks_with_no_hero_headline_when_response_omits_hero() -> None:
    """End-to-end: parser accepts the candidate, quality pack fires no_hero_headline."""
    response = json.dumps({
        "title": "Acme Q3: Stop Renewal Surprises",
        "slug": "acme-q3-launch",
        # hero omitted -> must surface as a quality blocker, not unparseable_response
        "sections": [{"id": "s1", "title": "T", "body_markdown": "b"}],
        "cta": {"label": "L", "url": "/u"},
        "meta": {},
    })
    service, landing_pages, _llm, _skills, _rp = _service(responses=[response])

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert landing_pages.saved == []
    assert result.errors[0]["reason"] == "quality_blocked"
    blockers = result.errors[0]["blockers"]
    assert any("no_hero_headline" in b for b in blockers)


# -----------------------
# Service: generation
# -----------------------


@pytest.mark.asyncio
async def test_generate_persists_one_landing_page_per_call_via_save_drafts() -> None:
    service, landing_pages, llm, _skills, _rp = _service()

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.requested == 1
    assert result.generated == 1
    assert result.skipped == 0
    assert result.saved_ids == ("lp-1",)
    assert len(llm.calls) == 1
    assert llm.calls[0]["metadata"]["asset_type"] == "landing_page"
    assert llm.calls[0]["metadata"]["target_mode"] == "marketing_campaign"
    saved_drafts = landing_pages.saved[0]["drafts"]
    assert len(saved_drafts) == 1
    draft = saved_drafts[0]
    assert isinstance(draft, LandingPageDraft)
    assert draft.campaign_name == "acme-q3-launch"
    assert draft.persona == "VP Engineering at mid-market SaaS"
    assert draft.title == "Acme Q3: Stop Renewal Surprises"
    assert draft.slug == "acme-q3-launch"
    assert draft.hero["headline"] == "Stop renewal surprises"
    assert draft.cta["label"] == "Book a 15-min demo"
    assert draft.sections[0].id == "problem"
    assert draft.metadata["generation_model"] == "test-model"


@pytest.mark.asyncio
async def test_generate_substitutes_campaign_json_into_system_prompt_only() -> None:
    """Campaign payload appears in system content via {campaign_json} and NOT in user content."""
    service, _lp, llm, _skills, _rp = _service(
        prompts={"digest/landing_page_generation": "C={campaign_json}"},
    )

    await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    system_msg = llm.calls[0]["messages"][0].content
    user_msg = llm.calls[0]["messages"][1].content
    assert '"name":"acme-q3-launch"' in system_msg
    assert '"name":"acme-q3-launch"' not in user_msg
    assert "acme-q3-launch" not in user_msg


@pytest.mark.asyncio
async def test_generate_skips_unparseable_response() -> None:
    service, landing_pages, _llm, _skills, _rp = _service(responses=["not json garbage"])

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.saved == []
    assert any(err.get("reason") == "unparseable_response" for err in result.errors)


@pytest.mark.asyncio
async def test_generate_skips_when_quality_pack_blocks() -> None:
    """Response missing CTA gets blocked by the quality pack."""
    bad_response = json.dumps({
        "title": "ok",
        "slug": "ok",
        "hero": {"headline": "h", "subheadline": "s", "cta_label": "L", "cta_url": "/u"},
        "sections": [{"id": "s1", "title": "T", "body_markdown": "b"}],
        "cta": {},  # missing CTA -> no_cta blocker
        "meta": {},
        "reference_ids": [],
    })
    service, landing_pages, _llm, _skills, _rp = _service(responses=[bad_response])

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.saved == []
    assert any(err.get("reason") == "quality_blocked" for err in result.errors)
    blockers = result.errors[0]["blockers"]
    assert any("no_cta" in b for b in blockers)


@pytest.mark.asyncio
async def test_generate_skips_when_campaign_name_empty() -> None:
    service, landing_pages, llm, _skills, _rp = _service()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        campaign=MarketingCampaign(name=""),
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.saved == []
    assert llm.calls == []  # no LLM round-trip
    assert any(err.get("reason") == "missing_campaign_name" for err in result.errors)


@pytest.mark.asyncio
async def test_generate_consumes_reasoning_context_via_provider() -> None:
    """When a CampaignReasoningContextProvider is supplied, its context is fetched per campaign."""
    context = CampaignReasoningContext(
        top_theses=({"claim": "Renewal pricing", "confidence": 0.9, "source_ids": ["r1"]},),
        canonical_reasoning={"summary": "narrative summary"},
    )
    service, _lp, _llm, _skills, reasoning_provider = _service(reasoning_context=context)

    await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert reasoning_provider is not None
    assert len(reasoning_provider.calls) == 1
    call = reasoning_provider.calls[0]
    assert call["target_id"] == "acme-q3-launch"
    assert call["target_mode"] == "marketing_campaign"


@pytest.mark.asyncio
async def test_generate_raises_value_error_when_skill_prompt_missing() -> None:
    service, _lp, _llm, _skills, _rp = _service(prompts={})

    with pytest.raises(ValueError, match="Landing-page generation skill not found"):
        await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())


@pytest.mark.asyncio
async def test_generate_blocks_when_llm_omits_slug() -> None:
    """Empty slug from the LLM hits the quality pack's no_slug blocker."""
    response = _valid_response(slug="")
    service, landing_pages, _llm, _skills, _rp = _service(responses=[response])

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert landing_pages.saved == []
    blockers = result.errors[0]["blockers"]
    assert any("no_slug" in b for b in blockers)
