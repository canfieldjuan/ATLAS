from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.content_image_provider import (
    ContentImageAsset,
)
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationConfig,
    LandingPageGenerationService,
    _normalize_section_answer_visibility,
    _quality_repair_guidance,
    parse_landing_page_response,
)
from extracted_content_pipeline.landing_page_ports import (
    LandingPageDraft,
    LandingPageSection,
    MarketingCampaign,
)


# -----------------------
# Fake ports
# -----------------------


class _LandingPages:
    def __init__(self):
        self.saved = []
        self.updated = []

    async def save_drafts(self, drafts, *, scope):
        self.saved.append({"drafts": list(drafts), "scope": scope})
        return [f"lp-{index + 1}" for index, _ in enumerate(drafts)]

    async def update_draft(self, landing_page_id, draft, *, scope):
        self.updated.append({
            "id": landing_page_id,
            "draft": draft,
            "scope": scope,
        })
        return draft

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

    async def read_campaign_reasoning_context(self, *, scope, target_id, target_mode, opportunity):
        self.calls.append({
            "scope": scope,
            "target_id": target_id,
            "target_mode": target_mode,
        })
        return self.context


class _ImageProvider:
    def __init__(self, asset=None, error: Exception | None = None):
        self.asset = asset
        self.error = error
        self.calls = []

    async def select_image(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.asset


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
                "body_markdown": (
                    "Renewal surprises create preventable churn because "
                    "teams see pricing pressure too late. Renewal pricing is "
                    "the #1 churn driver."
                ),
                "metadata": {
                    "order": 1,
                    "kind": "problem",
                    "primary_question": "Why do renewal surprises cause churn?",
                    "answer_summary": (
                        "Renewal surprises create preventable churn because "
                        "teams see pricing pressure too late."
                    ),
                },
            },
            {
                "id": "solution",
                "title": "The Solution",
                "body_markdown": (
                    "Acme catches renewal pressure early by surfacing "
                    "risk signals during the first 30 days. Acme surfaces "
                    "pressure signals in the first 30 days."
                ),
                "metadata": {
                    "order": 2,
                    "kind": "solution",
                    "primary_question": "How does Acme catch renewal pressure?",
                    "answer_summary": (
                        "Acme catches renewal pressure early by surfacing "
                        "risk signals during the first 30 days."
                    ),
                },
            },
            {
                "id": "implementation_questions",
                "title": "Implementation questions",
                "body_markdown": (
                    "VP Engineering teams can review pricing, rollout, "
                    "and risk questions before booking time with sales. "
                    "That keeps the renewal review moving with clearer "
                    "answers."
                ),
                "metadata": {
                    "order": 3,
                    "kind": "objection",
                    "primary_question": "What should teams know before rollout?",
                    "answer_summary": (
                        "VP Engineering teams can review pricing, rollout, "
                        "and risk questions before booking time with sales."
                    ),
                },
            },
        ],
        "cta": {"label": "Book a 15-min demo", "url": "/demo", "variant": "primary"},
        "meta": {
            "title_tag": "Stop Renewal Surprises | Acme",
            "description": "Acme catches renewal pressure 90 days early so you can prevent unplanned churn.",
        },
        "reference_ids": ["customer-logo-1"],
    })


def _draft_from_response(
    response: str | dict[str, object],
    *,
    landing_page_id: str = "11111111-1111-1111-1111-111111111111",
    status: str = "quality_blocked",
    metadata: dict[str, object] | None = None,
) -> LandingPageDraft:
    payload = json.loads(response) if isinstance(response, str) else dict(response)
    return LandingPageDraft(
        id=landing_page_id,
        status=status,
        campaign_name="acme-q3-launch",
        persona="VP Engineering at mid-market SaaS",
        value_prop="Cut renewal pricing leakage by 40%",
        title=str(payload["title"]),
        slug=str(payload["slug"]),
        hero=dict(payload.get("hero") or {}),
        sections=tuple(
            LandingPageSection(
                id=str(section.get("id") or ""),
                title=str(section.get("title") or ""),
                body_markdown=str(section.get("body_markdown") or ""),
                metadata=dict(section.get("metadata") or {}),
            )
            for section in payload.get("sections") or ()
            if isinstance(section, dict)
        ),
        cta=dict(payload.get("cta") or {}),
        meta=dict(payload.get("meta") or {}),
        reference_ids=tuple(str(item) for item in payload.get("reference_ids") or ()),
        metadata=dict(metadata or {}),
    )


def _service(
    *,
    responses=None,
    prompts=None,
    reasoning_context=None,
    config=None,
    image_provider=None,
):
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
        image_provider=image_provider,
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
    assert parsed["sections"][0]["metadata"]["kind"] == "problem"


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


def test_quality_repair_guidance_explains_section_semantics_contract() -> None:
    guidance = _quality_repair_guidance(("geo_readiness:section_semantics",))

    assert "metadata.kind must be one of" in guidance
    assert "body_markdown must start with the exact metadata.answer_summary" in guidance
    assert "problem, solution, how_it_works, faq, or objection" in guidance


def test_normalize_section_answer_visibility_prepends_model_summary() -> None:
    parsed = {
        "title": "FAQ Report",
        "sections": [
            {
                "id": "problem",
                "title": "Support Problems Become Cancellations",
                "body_markdown": "Small teams keep answering repeat questions.",
                "metadata": {
                    "kind": "problem",
                    "primary_question": "Why do repeat support questions matter?",
                    "answer_summary": (
                        "Repeat support questions matter because they show where "
                        "customers cannot find answers before frustration grows."
                    ),
                },
            }
        ],
    }

    normalized = _normalize_section_answer_visibility(parsed)

    body = normalized["sections"][0]["body_markdown"]
    assert body.startswith(parsed["sections"][0]["metadata"]["answer_summary"])
    assert "Small teams keep answering repeat questions." in body


@pytest.mark.asyncio
async def test_generate_rejects_direct_quality_repair_attempt_above_contract_max() -> None:
    service, _landing_pages, _llm, _skills, _rp = _service()

    with pytest.raises(
        ValueError,
        match="landing_page_quality_repair_attempts must be at most 10",
    ):
        await service.generate(
            scope=TenantScope(account_id="acct-1"),
            campaign=_campaign(),
            quality_repair_attempts=11,
        )


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
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=[response],
        config=LandingPageGenerationConfig(quality_repair_attempts=0),
    )

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
    assert draft.sections[0].metadata["kind"] == "problem"
    assert (
        draft.sections[0].metadata["primary_question"]
        == "Why do renewal surprises cause churn?"
    )
    assert "preventable churn" in draft.sections[0].metadata["answer_summary"]
    assert draft.metadata["generation_model"] == "test-model"


@pytest.mark.asyncio
async def test_generate_attaches_optional_landing_page_image_before_save() -> None:
    payload = json.loads(_valid_response())
    payload["hero"]["image_url"] = ""
    payload["hero"]["image_alt"] = ""
    payload["meta"] = {"og_image_url": None}
    image_provider = _ImageProvider(
        ContentImageAsset(
            url="https://images.example.com/landing.jpg",
            provider="unsplash",
            alt_text="Support team reviewing customer questions",
            attribution_name="Ada Lens",
            attribution_url="https://unsplash.example.com/@ada",
            source_id="photo-1",
        )
    )
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=[json.dumps(payload)],
        config=LandingPageGenerationConfig(quality_gates_enabled=False),
        image_provider=image_provider,
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 1
    draft = landing_pages.saved[0]["drafts"][0]
    assert draft.hero["image_url"] == "https://images.example.com/landing.jpg"
    assert draft.hero["image_alt"] == "Support team reviewing customer questions"
    assert draft.meta["og_image_url"] == "https://images.example.com/landing.jpg"
    assert draft.metadata["content_image"]["provider"] == "unsplash"
    assert image_provider.calls[0].asset_type == "landing_page"
    assert image_provider.calls[0].slot == "hero"
    assert "Acme Q3" in image_provider.calls[0].title


@pytest.mark.asyncio
async def test_generate_keeps_landing_page_when_image_provider_fails() -> None:
    service, landing_pages, _llm, _skills, _rp = _service(
        image_provider=_ImageProvider(error=RuntimeError("image service unavailable"))
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 1
    draft = landing_pages.saved[0]["drafts"][0]
    assert "image_url" not in draft.hero
    assert "content_image" not in draft.metadata


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
async def test_generate_threads_seo_geo_aeo_context_into_system_prompt_payload() -> None:
    service, _lp, llm, _skills, _rp = _service(
        prompts={"digest/landing_page_generation": "C={campaign_json}"},
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        campaign=MarketingCampaign(
            name="support-faq-report",
            persona="10-50 person SaaS team",
            value_prop="Turn repeat support tickets into customer-ready FAQs",
            context={
                "target_keyword": "support ticket FAQ",
                "secondary_keywords": ["reduce repeat support tickets"],
                "search_intent": "Find a low-friction way to turn old tickets into help-center answers.",
                "primary_entity": "FAQ Report",
                "audience_entity": "small SaaS team",
                "objections": ["Will this publish automatically?"],
                "faq_questions": ["What happens after upload?"],
                "source_period": "Last 90 days of support tickets",
                "has_dated_window": True,
                "source_row_count": 4,
                "included_ticket_row_count": 4,
                "skipped_ticket_row_count": 0,
                "truncated_ticket_row_count": 0,
                "question_like_ticket_count": 2,
                "top_ticket_clusters": [{"label": "reporting friction", "count": 2}],
                "customer_wording_examples": [{
                    "source_id": "ticket-1",
                    "source_title": "Reporting dashboard export",
                    "pain_category": "reporting friction",
                    "text": "We cannot export the reporting dashboard for analysts.",
                }],
                "support_ticket_resolution_evidence_present": True,
                "support_ticket_resolution_evidence_count": 1,
                "support_ticket_resolution_examples": [{
                    "source_id": "ticket-1",
                    "source_title": "Reporting dashboard export",
                    "text": "Open Reports, choose Export, then select CSV.",
                }],
                "has_measured_outcomes": True,
                "measured_outcome_count": 1,
                "measured_outcome_examples": [{
                    "source_id": "ticket-1",
                    "source_title": "Reporting dashboard export",
                    "text": "Repeat reporting tickets fell from 9 to 4.",
                }],
                "internal_links": ["/systems/ai-content-ops/intake"],
                "cta_label": "Upload Ticket CSV -- Free Analysis",
                "cta_url": "/systems/ai-content-ops/intake",
            },
        ),
    )

    system_msg = llm.calls[0]["messages"][0].content
    user_msg = llm.calls[0]["messages"][1].content
    assert '"target_keyword":"support ticket FAQ"' in system_msg
    assert '"secondary_keywords":["reduce repeat support tickets"]' in system_msg
    assert '"primary_entity":"FAQ Report"' in system_msg
    assert '"source_row_count":4' in system_msg
    assert '"has_dated_window":true' in system_msg
    assert '"top_ticket_clusters":[{"label":"reporting friction","count":2}]' in system_msg
    assert '"support_ticket_resolution_evidence_present":true' in system_msg
    assert '"support_ticket_resolution_evidence_count":1' in system_msg
    assert '"has_measured_outcomes":true' in system_msg
    assert '"measured_outcome_count":1' in system_msg
    assert "We cannot export the reporting dashboard for analysts." in system_msg
    assert "Open Reports, choose Export, then select CSV." in system_msg
    assert "Repeat reporting tickets fell from 9 to 4." in system_msg
    assert '"cta_url":"/systems/ai-content-ops/intake"' in system_msg
    assert "support ticket FAQ" not in user_msg


def test_build_draft_preserves_support_ticket_source_context_metadata() -> None:
    service, _lp, _llm, _skills, _rp = _service()
    campaign = MarketingCampaign(
        name="support-faq-report",
        persona="10-50 person SaaS team",
        value_prop="Turn repeat support tickets into customer-ready FAQs",
        context={
            "source_row_count": 4,
            "included_ticket_row_count": 4,
            "skipped_ticket_row_count": 0,
            "truncated_ticket_row_count": 0,
            "question_like_ticket_count": 2,
            "has_dated_window": False,
            "top_ticket_clusters": [{"label": "reporting friction", "count": 2}],
            "customer_wording_examples": [{
                "source_id": "ticket-1",
                "source_title": "Reporting dashboard export",
                "pain_category": "reporting friction",
                "text": "We cannot export the reporting dashboard for analysts.",
            }],
            "support_ticket_resolution_evidence_present": False,
            "support_ticket_resolution_evidence_count": 0,
            "support_ticket_resolution_examples": [],
            "has_measured_outcomes": False,
            "measured_outcome_count": 0,
            "measured_outcome_examples": [],
            "support_ticket_source_summary": {"unsafe": "should not be copied"},
            "target_account": "should not be copied",
        },
    )

    draft = service._build_draft(  # noqa: SLF001 - source metadata contract
        parse_landing_page_response(_valid_response()) or {},
        campaign=campaign,
    )

    assert draft.metadata["source_context"] == {
        "source_row_count": 4,
        "included_ticket_row_count": 4,
        "skipped_ticket_row_count": 0,
        "truncated_ticket_row_count": 0,
        "question_like_ticket_count": 2,
        "has_dated_window": False,
        "top_ticket_clusters": [{"label": "reporting friction", "count": 2}],
        "customer_wording_examples": [{
            "source_id": "ticket-1",
            "source_title": "Reporting dashboard export",
            "pain_category": "reporting friction",
            "text": "We cannot export the reporting dashboard for analysts.",
        }],
        "support_ticket_resolution_evidence_present": False,
        "support_ticket_resolution_evidence_count": 0,
        "has_measured_outcomes": False,
        "measured_outcome_count": 0,
    }
    assert "target_account" not in draft.metadata["source_context"]
    assert "support_ticket_source_summary" not in draft.metadata["source_context"]


@pytest.mark.asyncio
async def test_generate_skips_unparseable_response() -> None:
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=["not json garbage", "still not json"]
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.saved == []
    assert any(err.get("reason") == "unparseable_response" for err in result.errors)


@pytest.mark.asyncio
async def test_generate_retries_unparseable_response_once_by_default() -> None:
    service, landing_pages, llm, _skills, _rp = _service(
        responses=["not json garbage", _valid_response()]
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 1
    assert len(llm.calls) == 2
    assert llm.calls[0]["metadata"]["attempt_no"] == 1
    assert llm.calls[1]["metadata"]["attempt_no"] == 2
    assert "Previous response excerpt:\nnot json garbage" in llm.calls[1]["messages"][1].content
    draft = landing_pages.saved[0]["drafts"][0]
    assert draft.metadata["generation_parse_attempts"] == 2


@pytest.mark.asyncio
async def test_generate_can_disable_parse_retry() -> None:
    service, landing_pages, llm, _skills, _rp = _service(
        responses=["not json garbage", _valid_response()],
        config=LandingPageGenerationConfig(parse_retry_attempts=0),
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert len(llm.calls) == 1
    assert landing_pages.saved == []
    assert any(err.get("reason") == "unparseable_response" for err in result.errors)


@pytest.mark.asyncio
async def test_generate_accumulates_usage_across_parse_retry_attempts() -> None:
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=[
            {
                "content": "not json garbage",
                "model": "first-model",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
            {
                "content": _valid_response(),
                "model": "final-model",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        ]
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 1
    draft = landing_pages.saved[0]["drafts"][0]
    assert draft.metadata["generation_model"] == "final-model"
    assert draft.metadata["generation_usage"] == {"input_tokens": 12, "output_tokens": 5}
    assert draft.metadata["generation_parse_attempts"] == 2


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
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=[bad_response],
        config=LandingPageGenerationConfig(quality_repair_attempts=0),
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.saved == []
    assert any(err.get("reason") == "quality_blocked" for err in result.errors)
    blockers = result.errors[0]["blockers"]
    assert any("no_cta" in b for b in blockers)


@pytest.mark.asyncio
async def test_generate_skips_when_shared_readiness_blocks() -> None:
    response = json.loads(_valid_response())
    response["meta"].pop("description")
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=[json.dumps(response)],
        config=LandingPageGenerationConfig(quality_repair_attempts=0),
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.saved == []
    assert result.errors[0]["reason"] == "quality_blocked"
    assert "seo_aeo_readiness:meta_description" in result.errors[0]["blockers"]


@pytest.mark.asyncio
async def test_generate_quality_blocked_error_includes_failed_candidate_snapshot() -> None:
    response = json.loads(_valid_response())
    response["sections"][0]["title"] = "Benefits"
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=[json.dumps(response)],
        config=LandingPageGenerationConfig(quality_repair_attempts=0),
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.saved == []
    error = result.errors[0]
    assert error["reason"] == "quality_blocked"
    assert "geo_readiness:section_semantics" in error["blockers"]
    snapshot = error["failed_candidate"]
    assert snapshot["title"] == "Acme Q3: Stop Renewal Surprises"
    assert snapshot["slug"] == "acme-q3-launch"
    assert snapshot["hero_headline"] == "Stop renewal surprises"
    assert snapshot["cta_url"] == "/demo"
    assert snapshot["section_count"] == 3
    assert snapshot["sections_truncated"] is False
    first_section = snapshot["sections"][0]
    assert first_section["title"] == "Benefits"
    assert first_section["kind"] == "problem"
    assert first_section["primary_question"] == "Why do renewal surprises cause churn?"
    assert first_section["answer_summary_word_count"] >= 6
    assert first_section["body_starts_with_answer_summary"] is True


@pytest.mark.asyncio
async def test_generate_repairs_quality_blocked_response_once_and_persists() -> None:
    """Parsed JSON that fails the quality gate gets one targeted repair pass."""
    bad_response = json.dumps({
        "title": "ok",
        "slug": "ok",
        "hero": {"headline": "h", "subheadline": "s", "cta_label": "L", "cta_url": "/u"},
        "sections": [{"id": "s1", "title": "T", "body_markdown": "b"}],
        "cta": {},  # missing CTA -> no_cta blocker
        "meta": {},
        "reference_ids": [],
    })
    service, landing_pages, llm, _skills, _rp = _service(
        responses=[
            {
                "content": bad_response,
                "model": "first-model",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
            {
                "content": _valid_response(),
                "model": "final-model",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        ],
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 1
    assert result.saved_ids == ("lp-1",)
    assert len(llm.calls) == 2
    assert llm.calls[0]["metadata"]["quality_repair_attempt_no"] == 0
    assert llm.calls[1]["metadata"]["quality_repair_attempt_no"] == 1
    repair_prompt = llm.calls[1]["messages"][1].content
    assert "failed the deterministic quality gate" in repair_prompt
    assert "no_cta" in repair_prompt
    draft = landing_pages.saved[0]["drafts"][0]
    assert draft.metadata["generation_model"] == "final-model"
    assert draft.metadata["generation_usage"] == {"input_tokens": 12, "output_tokens": 5}
    assert draft.metadata["generation_parse_attempts"] == 2
    assert draft.metadata["generation_quality_repair_attempts"] == 1
    assert result.quality_repair_history[0]["attempt"] == 0
    assert result.quality_repair_history[0]["passed"] is False
    assert any(
        "no_cta" in blocker
        for blocker in result.quality_repair_history[0]["blockers"]
    )
    assert result.quality_repair_history[1] == {
        "attempt": 1,
        "passed": True,
        "blockers": (),
        "repair_issues": (),
    }
    assert result.as_dict()["quality_repair_history"] == [
        dict(item) for item in result.quality_repair_history
    ]
    assert (
        draft.metadata["generation_quality_repair_history"]
        == result.quality_repair_history
    )


@pytest.mark.asyncio
async def test_generate_repairs_shared_readiness_blocker_once_and_persists() -> None:
    needs_repair = json.loads(_valid_response())
    needs_repair["meta"].pop("description")
    service, landing_pages, llm, _skills, _rp = _service(
        responses=[
            {
                "content": json.dumps(needs_repair),
                "model": "first-model",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
            {
                "content": _valid_response(),
                "model": "final-model",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        ],
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 1
    assert result.saved_ids == ("lp-1",)
    assert len(llm.calls) == 2
    repair_prompt = llm.calls[1]["messages"][1].content
    assert "seo_aeo_readiness:meta_description" in repair_prompt
    assert landing_pages.saved[0]["drafts"][0].metadata[
        "generation_quality_repair_attempts"
    ] == 1
    assert result.quality_repair_history[0]["blockers"] == (
        "seo_aeo_readiness:meta_description",
    )


@pytest.mark.asyncio
async def test_repair_draft_sends_current_draft_and_repair_issues_and_updates_same_row() -> None:
    needs_repair = json.loads(_valid_response())
    needs_repair["meta"].pop("description")
    draft = _draft_from_response(
        needs_repair,
        metadata={"source": "review_drawer"},
    )
    service, landing_pages, llm, _skills, _rp = _service(
        responses=[
            {
                "content": _valid_response(),
                "model": "repair-model",
                "usage": {"input_tokens": 9, "output_tokens": 4},
            },
        ],
    )

    result = await service.repair_draft(
        scope=TenantScope(account_id="acct-1"),
        draft=draft,
    )

    assert result.requested == 1
    assert result.generated == 1
    assert result.skipped == 0
    assert result.saved_ids == (draft.id,)
    assert landing_pages.saved == []
    assert len(landing_pages.updated) == 1
    assert landing_pages.updated[0]["id"] == draft.id
    repaired = landing_pages.updated[0]["draft"]
    assert repaired.id == draft.id
    assert repaired.status == "draft"
    assert repaired.metadata["source"] == "review_drawer"
    assert repaired.metadata["generation_model"] == "repair-model"
    assert repaired.metadata["generation_usage"] == {
        "input_tokens": 9,
        "output_tokens": 4,
    }
    assert repaired.metadata["generation_parse_attempts"] == 1
    assert repaired.metadata["generation_quality_repair_attempts"] == 1
    assert repaired.metadata["saved_draft_repair_source_id"] == draft.id
    assert result.quality_repair_history[0]["attempt"] == 0
    assert result.quality_repair_history[0]["passed"] is False
    assert result.quality_repair_history[0]["blockers"] == (
        "seo_aeo_readiness:meta_description",
    )
    assert result.quality_repair_history[1] == {
        "attempt": 1,
        "passed": True,
        "blockers": (),
        "repair_issues": (),
    }
    system_prompt = llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert '"repair_mode":"saved_draft"' in system_prompt
    assert '"current_draft"' in system_prompt
    assert '"repair_issues":["seo_aeo_readiness:meta_description"]' in system_prompt
    assert "seo_aeo_readiness:meta_description" in user_prompt


@pytest.mark.asyncio
async def test_repair_draft_applies_brand_voice_to_repaired_landing_page() -> None:
    needs_repair = json.loads(_valid_response())
    needs_repair["meta"].pop("description")
    draft = _draft_from_response(needs_repair)
    service, landing_pages, llm, _skills, _rp = _service(
        responses=[_valid_response()],
    )

    result = await service.repair_draft(
        scope=TenantScope(account_id="acct-1"),
        draft=draft,
        brand_voice_profile_id="acme-main",
        brand_voice={
            "id": "acme-main",
            "account_id": "acct-1",
            "name": "Acme main voice",
            "descriptors": ["plainspoken"],
            "exemplars": ["You explain the tradeoff before the CTA."],
            "banned_terms": ["synergy"],
        },
    )

    assert result.generated == 1
    system_prompt = llm.calls[0]["messages"][0].content
    assert "## Brand voice" in system_prompt
    assert "plainspoken" in system_prompt
    assert "You explain the tradeoff before the CTA." in system_prompt
    repaired = landing_pages.updated[0]["draft"]
    assert repaired.metadata["brand_voice_profile"] == {
        "id": "acme-main",
        "account_id": "acct-1",
        "name": "Acme main voice",
        "descriptors": ["plainspoken"],
    }
    assert repaired.metadata["brand_voice_audit"]["passed"] is True


@pytest.mark.asyncio
async def test_repair_draft_rejects_approved_draft_without_llm_call() -> None:
    draft = _draft_from_response(_valid_response(), status="approved")
    service, landing_pages, llm, _skills, _rp = _service()

    result = await service.repair_draft(
        scope=TenantScope(account_id="acct-1"),
        draft=draft,
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors == ({
        "landing_page_id": draft.id,
        "reason": "approved_draft_not_repairable",
    },)
    assert llm.calls == []
    assert landing_pages.updated == []


@pytest.mark.asyncio
async def test_repair_draft_returns_quality_blocked_when_no_repair_attempts() -> None:
    needs_repair = json.loads(_valid_response())
    needs_repair["meta"].pop("description")
    draft = _draft_from_response(needs_repair)
    service, landing_pages, llm, _skills, _rp = _service(
        config=LandingPageGenerationConfig(quality_repair_attempts=0),
    )

    result = await service.repair_draft(
        scope=TenantScope(account_id="acct-1"),
        draft=draft,
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["reason"] == "quality_blocked"
    assert result.errors[0]["blockers"] == (
        "seo_aeo_readiness:meta_description",
    )
    assert result.quality_repair_history[0]["attempt"] == 0
    assert result.quality_repair_history[0]["passed"] is False
    assert llm.calls == []
    assert landing_pages.updated == []


@pytest.mark.asyncio
async def test_repair_draft_noops_when_existing_draft_already_passes() -> None:
    draft = _draft_from_response(_valid_response(), status="draft")
    service, landing_pages, llm, _skills, _rp = _service()

    result = await service.repair_draft(
        scope=TenantScope(account_id="acct-1"),
        draft=draft,
    )

    assert result.generated == 0
    assert result.skipped == 0
    assert result.saved_ids == (draft.id,)
    assert result.quality_repair_history == ({
        "attempt": 0,
        "passed": True,
        "blockers": (),
        "repair_issues": (),
    },)
    assert llm.calls == []
    assert landing_pages.updated == []


@pytest.mark.asyncio
async def test_repair_draft_reports_blockers_when_repair_response_will_not_parse() -> None:
    needs_repair = json.loads(_valid_response())
    needs_repair["meta"].pop("description")
    draft = _draft_from_response(needs_repair)
    service, landing_pages, llm, _skills, _rp = _service(
        responses=["not valid json"],
        config=LandingPageGenerationConfig(parse_retry_attempts=0),
    )

    result = await service.repair_draft(
        scope=TenantScope(account_id="acct-1"),
        draft=draft,
    )

    expected_blockers = ("seo_aeo_readiness:meta_description",)
    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.updated == []
    assert len(llm.calls) == 1
    assert result.errors[0]["reason"] == "unparseable_response"
    assert result.errors[0]["blockers"] == expected_blockers
    assert result.errors[0]["quality_blockers"] == expected_blockers
    assert result.errors[0]["quality_repair_history"] == result.quality_repair_history
    assert result.quality_repair_history[0]["blockers"] == expected_blockers


@pytest.mark.asyncio
async def test_repair_draft_reports_blockers_when_repair_provider_fails() -> None:
    needs_repair = json.loads(_valid_response())
    needs_repair["meta"].pop("description")
    draft = _draft_from_response(needs_repair)
    service, landing_pages, llm, _skills, _rp = _service(
        responses=[RuntimeError("provider unavailable")],
    )

    result = await service.repair_draft(
        scope=TenantScope(account_id="acct-1"),
        draft=draft,
    )

    expected_blockers = ("seo_aeo_readiness:meta_description",)
    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.updated == []
    assert len(llm.calls) == 1
    assert result.errors[0]["reason"] == "provider unavailable"
    assert result.errors[0]["blockers"] == expected_blockers
    assert result.errors[0]["quality_repair_history"] == result.quality_repair_history
    assert result.quality_repair_history[0]["blockers"] == expected_blockers


@pytest.mark.asyncio
async def test_generate_reports_quality_blockers_after_repair_attempt_fails() -> None:
    bad_response = json.dumps({
        "title": "ok",
        "slug": "ok",
        "hero": {"headline": "h", "subheadline": "s", "cta_label": "L", "cta_url": "/u"},
        "sections": [{"id": "s1", "title": "T", "body_markdown": "b"}],
        "cta": {},
        "meta": {},
        "reference_ids": [],
    })
    service, landing_pages, llm, _skills, _rp = _service(
        responses=[bad_response, bad_response],
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert landing_pages.saved == []
    assert len(llm.calls) == 2
    assert result.errors[0]["reason"] == "quality_blocked"
    assert result.errors[0]["quality_repair_attempts"] == 1
    assert any("no_cta" in blocker for blocker in result.errors[0]["blockers"])
    assert result.errors[0]["quality_repair_history"] == result.quality_repair_history
    assert [row["attempt"] for row in result.quality_repair_history] == [0, 1]
    assert all(row["passed"] is False for row in result.quality_repair_history)


@pytest.mark.asyncio
async def test_generate_reports_quality_blockers_when_repair_response_will_not_parse() -> None:
    bad_response = json.dumps({
        "title": "ok",
        "slug": "ok",
        "hero": {"headline": "h", "subheadline": "s", "cta_label": "L", "cta_url": "/u"},
        "sections": [{"id": "s1", "title": "T", "body_markdown": "b"}],
        "cta": {},
        "meta": {},
        "reference_ids": [],
    })
    service, landing_pages, llm, _skills, _rp = _service(
        responses=[bad_response, "not valid json"],
        config=LandingPageGenerationConfig(parse_retry_attempts=0),
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert result.skipped == 1
    assert landing_pages.saved == []
    assert len(llm.calls) == 2
    assert result.errors[0]["reason"] == "unparseable_response"
    assert any("no_cta" in blocker for blocker in result.errors[0]["quality_blockers"])
    assert result.errors[0]["quality_repair_history"] == result.quality_repair_history
    snapshot = result.errors[0]["failed_candidate"]
    assert snapshot["title"] == "ok"
    assert snapshot["section_count"] == 1
    assert snapshot["sections"][0]["title"] == "T"
    assert len(result.quality_repair_history) == 1
    assert result.quality_repair_history[0]["attempt"] == 0
    assert result.quality_repair_history[0]["passed"] is False


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

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        campaign=_campaign(),
    )

    assert reasoning_provider is not None
    assert len(reasoning_provider.calls) == 1
    call = reasoning_provider.calls[0]
    assert call["target_id"] == "acme-q3-launch"
    assert call["target_mode"] == "marketing_campaign"
    result_dict = result.as_dict()
    assert result_dict["reasoning_contexts_used"] == 1
    assert result_dict["consumed_reasoning_contexts"][0]["top_theses"][0]["claim"] == (
        "Renewal pricing"
    )


@pytest.mark.asyncio
async def test_generate_raises_value_error_when_skill_prompt_missing() -> None:
    service, _lp, _llm, _skills, _rp = _service(prompts={})

    with pytest.raises(ValueError, match="Landing-page generation skill not found"):
        await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())


@pytest.mark.asyncio
async def test_generate_blocks_when_llm_omits_slug() -> None:
    """Empty slug from the LLM hits the quality pack's no_slug blocker."""
    response = _valid_response(slug="")
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=[response],
        config=LandingPageGenerationConfig(quality_repair_attempts=0),
    )

    result = await service.generate(scope=TenantScope(account_id="acct-1"), campaign=_campaign())

    assert result.generated == 0
    assert landing_pages.saved == []
    blockers = result.errors[0]["blockers"]
    assert any("no_slug" in b for b in blockers)


# -----------------------
# PR-OptionA-2: per-call temperature/max_tokens/parse_retry_attempts overrides
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_llm_tuning_overrides_win_over_construction_config():
    """Resolved-value param reaches the LLM, not self._config.X."""

    service, _lp, llm, _skills, _rp = _service(
        responses=[
            "not parseable",
            _valid_response(),
        ],
        config=LandingPageGenerationConfig(
            temperature=0.3,
            max_tokens=4096,
            parse_retry_attempts=0,
        ),
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        campaign=_campaign(),
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
    service, _lp, llm, _skills, _rp = _service(
        responses=[_valid_response()],
        config=LandingPageGenerationConfig(temperature=0.7, max_tokens=999),
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        campaign=_campaign(),
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
    service, _lp, llm, _skills, _rp = _service(
        responses=[long_invalid, _valid_response()],
        config=LandingPageGenerationConfig(
            parse_retry_attempts=1,
            parse_retry_response_excerpt_chars=200,
        ),
    )

    await service.generate(
        scope=TenantScope(account_id="acct-1"),
        campaign=_campaign(),
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
    """When the executor passes quality_gates_enabled=False, the service
    short-circuits the quality gate. A response that would normally fail
    the gate (e.g., missing CTA) now lands as generated."""

    bad_response = json.dumps({
        "title": "ok",
        "slug": "ok",
        "hero": {"headline": "h", "subheadline": "s", "cta_label": "L", "cta_url": "/u"},
        "sections": [{"id": "s1", "title": "T", "body_markdown": "b"}],
        "cta": {},  # would normally hit no_cta blocker
        "meta": {},
        "reference_ids": [],
    })
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=[bad_response],
        config=LandingPageGenerationConfig(quality_repair_attempts=0),
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        campaign=_campaign(),
        quality_gates_enabled=False,
    )

    # Gate skipped -> the bad parse is persisted.
    assert result.generated == 1
    assert len(landing_pages.saved[0]["drafts"]) == 1


@pytest.mark.asyncio
async def test_generate_per_call_quality_gates_enabled_true_still_blocks() -> None:
    """When the executor passes True (or None falls back to construction
    default), the gate still runs. Behavior parity with prior PRs."""

    bad_response = json.dumps({
        "title": "ok",
        "slug": "ok",
        "hero": {"headline": "h", "subheadline": "s", "cta_label": "L", "cta_url": "/u"},
        "sections": [{"id": "s1", "title": "T", "body_markdown": "b"}],
        "cta": {},
        "meta": {},
        "reference_ids": [],
    })
    service, landing_pages, _llm, _skills, _rp = _service(
        responses=[bad_response],
        config=LandingPageGenerationConfig(quality_repair_attempts=0),
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        campaign=_campaign(),
        quality_gates_enabled=True,
    )

    assert result.generated == 0
    assert landing_pages.saved == []
