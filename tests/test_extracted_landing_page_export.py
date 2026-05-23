from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.landing_page_export import (
    LandingPageDraftExportResult,
    export_landing_page_drafts,
    public_landing_page_draft_row,
    public_landing_page_robots,
)
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationConfig,
    LandingPageGenerationService,
)
from extracted_content_pipeline.landing_page_ports import (
    LandingPageDraft,
    LandingPageSection,
    MarketingCampaign,
)


class _Repository:
    def __init__(self, drafts=None) -> None:
        self.drafts = tuple(drafts or ())
        self.list_calls: list[dict] = []

    async def save_drafts(self, drafts, *, scope):
        raise NotImplementedError

    async def list_drafts(
        self,
        *,
        scope,
        status=None,
        campaign_name=None,
        slug=None,
        limit=None,
    ):
        self.list_calls.append({
            "scope": scope,
            "status": status,
            "campaign_name": campaign_name,
            "slug": slug,
            "limit": limit,
        })
        return self.drafts

    async def update_status(self, landing_page_id, status, *, scope):
        raise NotImplementedError


class _SavingRepository(_Repository):
    async def save_drafts(self, drafts, *, scope):
        self.drafts = tuple(drafts)
        return tuple(f"landing-page-{index + 1}" for index, _ in enumerate(drafts))


class _LLM:
    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        return LLMResponse(
            content=json.dumps({
                "title": "Acme launch for VP Engineering",
                "slug": "acme-launch",
                "hero": {
                    "headline": "Stop renewal surprises",
                    "subheadline": (
                        "VP Engineering teams catch pressure early before "
                        "renewal risk creates more customer churn."
                    ),
                    "cta_label": "Book a demo",
                    "cta_url": "/demo",
                },
                "sections": [
                    {
                        "id": "problem",
                        "title": "Renewal pressure creates customer risk",
                        "body_markdown": (
                            "VP Engineering teams face renewal pressure when "
                            "customer problems stay hidden until pricing "
                            "reviews. Pricing pressure is rising."
                        ),
                        "metadata": {
                            "kind": "problem",
                            "primary_question": "Why does renewal pressure matter?",
                            "answer_summary": (
                                "VP Engineering teams face renewal pressure "
                                "when customer problems stay hidden until "
                                "pricing reviews."
                            ),
                        },
                    },
                    {
                        "id": "solution",
                        "title": "Catch pressure before renewal review",
                        "body_markdown": (
                            "VP Engineering teams catch pressure early by "
                            "turning customer signals into answers before "
                            "renewal review."
                        ),
                        "metadata": {
                            "kind": "solution",
                            "primary_question": "How does Acme help?",
                            "answer_summary": (
                                "VP Engineering teams catch pressure early "
                                "by turning customer signals into answers."
                            ),
                        },
                    },
                    {
                        "id": "objections",
                        "title": "Questions before rollout",
                        "body_markdown": (
                            "VP Engineering teams can review rollout and "
                            "pricing questions before booking time with the "
                            "team."
                        ),
                        "metadata": {
                            "kind": "objection",
                            "primary_question": "What should teams know first?",
                            "answer_summary": (
                                "VP Engineering teams can review rollout and "
                                "pricing questions before booking time."
                            ),
                        },
                    },
                ],
                "cta": {"label": "Book a demo", "url": "/demo"},
                "meta": {
                    "title_tag": "Acme Launch for VP Engineering",
                    "description": (
                        "Acme helps VP Engineering teams catch renewal "
                        "pressure early so customer risk is easier to see."
                    ),
                },
                "reference_ids": ["r1"],
            }),
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


class _Skills:
    def get_prompt(self, name):
        return "TEMPLATE {campaign_json}"


class _ReasoningProvider:
    async def read_campaign_reasoning_context(
        self,
        *,
        scope,
        target_id,
        target_mode,
        opportunity,
    ):
        return CampaignReasoningContext(
            canonical_reasoning={
                "wedge": "price_squeeze",
                "confidence": "high",
                "summary": "Pricing pressure creates displacement risk.",
            },
        )


def _draft(**overrides) -> LandingPageDraft:
    draft = LandingPageDraft(
        campaign_name="acme-q3-launch",
        persona="VP Engineering",
        value_prop="Cut renewal leakage",
        title="Acme Q3 launch",
        slug="acme-q3-launch",
        hero={"headline": "Stop renewal surprises"},
        sections=(
            LandingPageSection(
                id="problem",
                title="Problem",
                body_markdown="Renewal pricing is the main signal.",
                metadata={"order": 1},
            ),
        ),
        cta={"label": "Book a demo", "url": "/demo"},
        meta={"title_tag": "Acme Q3 launch"},
        reference_ids=("r1", "r2"),
        metadata={
            "generation_usage": {
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
            "generation_parse_attempts": 2,
            "generation_quality_repair_attempts": 1,
            "generation_quality_repair_history": [
                {
                    "attempt": 0,
                    "passed": False,
                    "blockers": ["missing proof"],
                    "repair_issues": ["add evidence"],
                },
                {
                    "attempt": 1,
                    "passed": True,
                    "blockers": [],
                    "repair_issues": [],
                },
            ],
            "reasoning_context": {
                "wedge": "price_squeeze",
                "confidence": "high",
            },
        },
    )
    return LandingPageDraft(
        campaign_name=overrides.get("campaign_name", draft.campaign_name),
        persona=overrides.get("persona", draft.persona),
        value_prop=overrides.get("value_prop", draft.value_prop),
        title=overrides.get("title", draft.title),
        slug=overrides.get("slug", draft.slug),
        hero=overrides.get("hero", draft.hero),
        sections=overrides.get("sections", draft.sections),
        cta=overrides.get("cta", draft.cta),
        meta=overrides.get("meta", draft.meta),
        reference_ids=overrides.get("reference_ids", draft.reference_ids),
        metadata=overrides.get("metadata", draft.metadata),
        id=overrides.get("id", draft.id),
        status=overrides.get("status", draft.status),
    )


def _ready_draft(**overrides) -> LandingPageDraft:
    sections = (
        LandingPageSection(
            id="problem",
            title="Support problems that keep coming back",
            body_markdown=(
                "VP Engineering teams catch pressure early when repeated "
                "support issues are turned into visible answers before "
                "renewal risk builds. Support problems keep coming back when "
                "teams cannot see where customers get stuck."
            ),
            metadata={
                "order": 1,
                "kind": "problem",
                "primary_question": "Why do support problems become renewal risk?",
                "answer_summary": (
                    "VP Engineering teams catch pressure early when repeated "
                    "support issues are turned into visible answers before "
                    "renewal risk builds."
                ),
            },
        ),
        LandingPageSection(
            id="solution",
            title="A workflow for catching pressure early",
            body_markdown=(
                "The solution helps VP Engineering teams catch pressure early "
                "by turning repeated support signals into a clear page and "
                "follow-up workflow."
            ),
            metadata={
                "order": 2,
                "kind": "solution",
                "primary_question": "How does the workflow catch pressure early?",
                "answer_summary": (
                    "The solution helps VP Engineering teams catch pressure "
                    "early by turning repeated support signals into a clear "
                    "page and follow-up workflow."
                ),
            },
        ),
        LandingPageSection(
            id="buyer_questions",
            title="Questions buyers ask before rollout",
            body_markdown=(
                "VP Engineering buyers can answer implementation, pricing, "
                "and security questions before they have to email the team. "
                "This keeps the conversion path clear during rollout review."
            ),
            metadata={
                "order": 3,
                "kind": "objection",
                "primary_question": "What should buyers know before rollout?",
                "answer_summary": (
                    "VP Engineering buyers can answer implementation, "
                    "pricing, and security questions before they have to "
                    "email the team."
                ),
            },
        ),
    )
    draft = LandingPageDraft(
        campaign_name="acme-support-retention",
        persona="VP Engineering",
        value_prop="Catch pressure early",
        title="Acme support retention page",
        slug="acme-support-retention",
        hero={
            "headline": "Catch support pressure before renewal risk builds",
            "subheadline": (
                "A landing page for VP Engineering teams that turns repeat "
                "support signals into clearer answers before customers drift."
            ),
            "cta_label": "Book a demo",
            "cta_url": "/landing",
        },
        sections=sections,
        cta={"label": "Book a demo", "url": "/landing"},
        meta={
            "title_tag": "Acme Support Retention for VP Engineering",
            "description": (
                "See how VP Engineering teams catch support pressure early and "
                "turn repeated customer questions into clearer retention pages."
            ),
            "og_title": "Acme Support Retention",
        },
        reference_ids=("r1",),
        metadata={
            "generation_usage": {
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
        },
    )
    return LandingPageDraft(
        campaign_name=overrides.get("campaign_name", draft.campaign_name),
        persona=overrides.get("persona", draft.persona),
        value_prop=overrides.get("value_prop", draft.value_prop),
        title=overrides.get("title", draft.title),
        slug=overrides.get("slug", draft.slug),
        hero=overrides.get("hero", draft.hero),
        sections=overrides.get("sections", draft.sections),
        cta=overrides.get("cta", draft.cta),
        meta=overrides.get("meta", draft.meta),
        reference_ids=overrides.get("reference_ids", draft.reference_ids),
        metadata=overrides.get("metadata", draft.metadata),
        id=overrides.get("id", draft.id),
        status=overrides.get("status", draft.status),
    )


@pytest.mark.asyncio
async def test_export_landing_page_drafts_passes_filters_to_repository() -> None:
    repo = _Repository(drafts=[_draft()])

    result = await export_landing_page_drafts(
        repo,
        scope={"account_id": "acct_1", "user_id": "user_1"},
        status="approved",
        campaign_name="acme-q3-launch",
        slug="acme-q3-launch",
        limit=7,
    )

    call = repo.list_calls[0]
    assert isinstance(call["scope"], TenantScope)
    assert call["scope"].account_id == "acct_1"
    assert call["scope"].user_id == "user_1"
    assert call["status"] == "approved"
    assert call["campaign_name"] == "acme-q3-launch"
    assert call["slug"] == "acme-q3-launch"
    assert call["limit"] == 7
    assert result.limit == 7
    assert result.filters == {
        "status": "approved",
        "account_id": "acct_1",
        "campaign_name": "acme-q3-launch",
        "slug": "acme-q3-launch",
    }


@pytest.mark.asyncio
async def test_export_landing_page_drafts_derives_review_summary_fields() -> None:
    result = await export_landing_page_drafts(
        _Repository(drafts=[_draft()]),
        scope=TenantScope(account_id="acct_1"),
    )

    row = result.rows[0]
    assert row["campaign_name"] == "acme-q3-launch"
    assert row["section_count"] == 1
    assert row["reference_count"] == 2
    assert row["generation_input_tokens"] == 12
    assert row["generation_output_tokens"] == 6
    assert row["generation_total_tokens"] == 18
    assert row["generation_parse_attempts"] == 2
    assert row["generation_quality_repair_attempts"] == 1
    assert row["generation_quality_repair_history"] == [
        {
            "attempt": 0,
            "passed": False,
            "blockers": ["missing proof"],
            "repair_issues": ["add evidence"],
        },
        {
            "attempt": 1,
            "passed": True,
            "blockers": [],
            "repair_issues": [],
        },
    ]
    assert row["reasoning_context_used"] is True
    assert row["reasoning_wedge"] == "price_squeeze"
    assert row["reasoning_confidence"] == "high"
    assert row["passed_output_checks"] == 3
    assert row["seo_aeo_readiness"]["status"] == "needs_review"
    assert row["geo_readiness"]["status"] == "needs_review"


def test_public_landing_page_draft_row_uses_renderer_allowlist() -> None:
    row = public_landing_page_draft_row(
        _draft(metadata={
            "scope": {"account_id": "acct_1", "user_id": "user_1"},
            "generation_usage": {"input_tokens": 12, "output_tokens": 6},
            "reasoning_context": {"wedge": "price_squeeze"},
        })
    )

    assert set(row) == {
        "id",
        "slug",
        "title",
        "persona",
        "value_prop",
        "hero",
        "sections",
        "cta",
        "meta",
        "robots",
        "structured_data",
    }
    assert row["slug"] == "acme-q3-launch"
    assert row["robots"] == "noindex,follow"
    assert row["structured_data"]["@context"] == "https://schema.org"
    assert "metadata" not in row
    assert "reference_ids" not in row
    assert "generation_input_tokens" not in row
    assert "reasoning_context_used" not in row
    assert "seo_aeo_readiness" not in row
    assert "geo_readiness" not in row


def test_public_landing_page_robots_indexes_only_approved_ready_pages() -> None:
    assert public_landing_page_robots(
        _ready_draft(status="approved")
    ) == "index,follow"


def test_public_landing_page_robots_keeps_non_approved_pages_noindex() -> None:
    assert public_landing_page_robots(
        _ready_draft(status="draft")
    ) == "noindex,follow"


def test_public_landing_page_robots_keeps_incomplete_pages_noindex() -> None:
    assert public_landing_page_robots(
        _draft(status="approved")
    ) == "noindex,follow"


def test_public_landing_page_robots_keeps_placeholder_cta_noindex() -> None:
    assert public_landing_page_robots(
        _ready_draft(status="approved", cta={"label": "Book a demo", "url": "/demo"})
    ) == "noindex,follow"


@pytest.mark.parametrize(
    "url",
    ["/demo/", "/demo?utm=1", "/demo#hero", "/demo/?utm=1"],
)
def test_public_landing_page_robots_keeps_demo_cta_variants_noindex(
    url: str,
) -> None:
    assert public_landing_page_robots(
        _ready_draft(status="approved", cta={"label": "Book a demo", "url": url})
    ) == "noindex,follow"


@pytest.mark.parametrize(
    "url",
    ["/demos", "/demo/foo", "/demo-page", "/"],
)
def test_public_landing_page_robots_indexes_demo_near_misses(url: str) -> None:
    # Guards the tight `== "/demo"` boundary: real routes whose path only
    # resembles the placeholder must stay indexable. A future switch to a
    # prefix match (e.g. startswith("/demo")) would silently de-index these.
    assert public_landing_page_robots(
        _ready_draft(status="approved", cta={"label": "Book a demo", "url": url})
    ) == "index,follow"


@pytest.mark.asyncio
async def test_export_landing_page_drafts_surfaces_ready_readiness() -> None:
    result = await export_landing_page_drafts(
        _Repository(drafts=[_ready_draft()]),
        scope=TenantScope(account_id="acct_1"),
    )

    row = result.rows[0]
    assert row["passed_output_checks"] == 8
    assert row["output_checks"] == {
        "title_tag": True,
        "meta_description": True,
        "slug_quality": True,
        "metadata_consistency": True,
        "answer_first_hero": True,
        "problem_solution_clarity": True,
        "audience_specificity": True,
        "objection_coverage": True,
    }
    assert row["seo_aeo_readiness"] == {
        "status": "ready",
        "passed": 8,
        "total": 8,
        "missing": [],
        "checks": row["output_checks"],
    }
    assert row["geo_readiness"] == {
        "status": "ready",
        "passed": 7,
        "total": 7,
        "missing": [],
        "checks": {
            "offer_entity_clarity": True,
            "audience_entity_clarity": True,
            "answer_extractability": True,
            "section_semantics": True,
            "trust_signal_visibility": True,
            "conversion_path_clarity": True,
            "claim_safety": True,
        },
    }
    assert row["structured_data"]["@context"] == "https://schema.org"
    assert [node["@type"] for node in row["structured_data"]["@graph"]] == [
        "WebPage",
        "FAQPage",
    ]
    assert len(row["structured_data"]["@graph"][1]["mainEntity"]) == 3


@pytest.mark.asyncio
async def test_export_landing_page_drafts_surfaces_incomplete_readiness() -> None:
    result = await export_landing_page_drafts(
        _Repository(drafts=[
            _ready_draft(
                persona="teams",
                slug="landing-page",
                hero={"headline": "Launch faster"},
                sections=(
                    LandingPageSection(
                        id="overview",
                        title="Overview",
                        body_markdown="TODO {{claim}} 42% faster",
                    ),
                ),
                cta={"label": "Book a demo", "url": "#"},
                meta={"title_tag": "Launch"},
                reference_ids=(),
            )
        ]),
        scope=TenantScope(account_id="acct_1"),
    )

    row = result.rows[0]
    assert row["passed_output_checks"] == 1
    assert row["seo_aeo_readiness"]["status"] == "needs_review"
    assert row["seo_aeo_readiness"]["missing"] == [
        "title_tag",
        "meta_description",
        "slug_quality",
        "answer_first_hero",
        "problem_solution_clarity",
        "audience_specificity",
        "objection_coverage",
    ]
    assert row["geo_readiness"]["status"] == "needs_review"
    assert row["geo_readiness"]["missing"] == [
        "audience_entity_clarity",
        "answer_extractability",
        "section_semantics",
        "trust_signal_visibility",
        "conversion_path_clarity",
        "claim_safety",
    ]


@pytest.mark.asyncio
async def test_export_landing_page_drafts_scores_section_metadata_readiness() -> None:
    result = await export_landing_page_drafts(
        _Repository(drafts=[
            _ready_draft(
                hero={
                    "headline": "Start the review",
                    "subheadline": "For busy operators.",
                    "cta_label": "Book a demo",
                    "cta_url": "/demo",
                },
                sections=(
                    LandingPageSection(
                        id="customer_friction",
                        title="Customer friction before renewal",
                        body_markdown=(
                            "VP Engineering teams catch pressure early by "
                            "turning repeated support issues into visible "
                            "answers before renewal risk builds. Customers "
                            "stop waiting for the same basic answer."
                        ),
                        metadata={
                            "kind": "problem",
                            "primary_question": (
                                "Why does repeated customer friction matter?"
                            ),
                            "answer_summary": (
                                "VP Engineering teams catch pressure early by "
                                "turning repeated support issues into visible "
                                "answers before renewal risk builds."
                            ),
                        },
                    ),
                    LandingPageSection(
                        id="clearer_answers",
                        title="Clearer answers before customers wait",
                        body_markdown=(
                            "VP Engineering teams use a workflow to turn "
                            "repeated support signals into clearer answers "
                            "before customers wait on another reply."
                        ),
                        metadata={
                            "kind": "solution",
                            "primary_question": (
                                "How do teams turn repeat tickets into answers?"
                            ),
                            "answer_summary": (
                                "VP Engineering teams use a workflow to turn "
                                "repeated support signals into clearer answers "
                                "before customers wait on another reply."
                            ),
                        },
                    ),
                    LandingPageSection(
                        id="before_rollout",
                        title="Before rollout questions",
                        body_markdown=(
                            "VP Engineering buyers can answer implementation "
                            "and pricing concerns before they book time with "
                            "the team. That keeps review moving without "
                            "another support handoff."
                        ),
                        metadata={
                            "kind": "objection",
                            "primary_question": "What should buyers know first?",
                            "answer_summary": (
                                "VP Engineering buyers can answer "
                                "implementation and pricing concerns before "
                                "they book time with the team."
                            ),
                        },
                    ),
                ),
            )
        ]),
        scope=TenantScope(account_id="acct_1"),
    )

    row = result.rows[0]
    assert row["seo_aeo_readiness"]["checks"]["answer_first_hero"] is False
    assert row["seo_aeo_readiness"]["checks"]["problem_solution_clarity"] is True
    assert row["seo_aeo_readiness"]["checks"]["objection_coverage"] is True
    assert row["geo_readiness"]["checks"]["answer_extractability"] is True
    assert row["geo_readiness"]["checks"]["section_semantics"] is True


@pytest.mark.asyncio
async def test_export_landing_page_drafts_defaults_summary_fields_without_metadata() -> None:
    result = await export_landing_page_drafts(
        _Repository(drafts=[_draft(metadata={})]),
        limit=1,
    )

    row = result.rows[0]
    assert row["generation_input_tokens"] is None
    assert row["generation_output_tokens"] is None
    assert row["generation_total_tokens"] is None
    assert row["generation_parse_attempts"] is None
    assert row["generation_quality_repair_attempts"] is None
    assert row["generation_quality_repair_history"] is None
    assert row["reasoning_context_used"] is False
    assert row["reasoning_wedge"] is None
    assert row["reasoning_confidence"] is None


@pytest.mark.asyncio
async def test_generated_landing_page_export_includes_reasoning_summary_fields() -> None:
    repo = _SavingRepository()
    service = LandingPageGenerationService(
        landing_pages=repo,
        llm=_LLM(),
        skills=_Skills(),
        reasoning_context=_ReasoningProvider(),
        config=LandingPageGenerationConfig(),
    )

    generated = await service.generate(
        scope=TenantScope(account_id="acct_1"),
        campaign=MarketingCampaign(
            name="acme-launch",
            persona="VP Engineering",
            value_prop="Catch pressure early",
        ),
    )
    exported = await export_landing_page_drafts(
        repo,
        scope=TenantScope(account_id="acct_1"),
    )

    assert generated.reasoning_contexts_used == 1
    row = exported.rows[0]
    assert row["reasoning_context_used"] is True
    assert row["reasoning_wedge"] == "price_squeeze"
    assert row["reasoning_confidence"] == "high"


@pytest.mark.asyncio
async def test_export_landing_page_drafts_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit must be non-negative"):
        await export_landing_page_drafts(_Repository(), limit=-1)


def test_landing_page_draft_export_result_renders_dict_and_csv() -> None:
    result = LandingPageDraftExportResult(
        rows=(
            {
                "campaign_name": "acme",
                "persona": "VP Engineering",
                "value_prop": "Catch pressure early",
                "title": "Acme page",
                "slug": "acme-page",
                "section_count": 1,
                "reference_count": 1,
                "generation_input_tokens": 12,
                "generation_output_tokens": 6,
                "generation_total_tokens": 18,
                "generation_parse_attempts": 1,
                "generation_quality_repair_attempts": 1,
                "generation_quality_repair_history": [
                    {
                        "attempt": 1,
                        "passed": True,
                        "blockers": [],
                        "repair_issues": [],
                    },
                ],
                "reasoning_context_used": True,
                "reasoning_wedge": "price_squeeze",
                "reasoning_confidence": "high",
                "passed_output_checks": 8,
                "output_checks": {"title_tag": True},
                "seo_aeo_readiness": {"status": "ready"},
                "geo_readiness": {"status": "ready"},
                "structured_data": {
                    "@context": "https://schema.org",
                    "@graph": [{"@type": "WebPage", "name": "Acme page"}],
                },
                "hero": {"headline": "Stop surprises"},
                "sections": [{"id": "problem"}],
                "cta": {"label": "Book a demo"},
                "meta": {"title_tag": "Acme page"},
                "reference_ids": ["r1"],
                "metadata": {"scope": {"account_id": "acct_1"}},
            },
        ),
        limit=1,
        filters={"status": "draft"},
    )

    as_dict = result.as_dict()
    csv_text = result.as_csv()

    assert as_dict["count"] == 1
    assert as_dict["rows"][0]["reasoning_wedge"] == "price_squeeze"
    assert "campaign_name,persona,value_prop" in csv_text
    assert "generation_input_tokens,generation_output_tokens" in csv_text
    assert (
        "generation_quality_repair_attempts,generation_quality_repair_history"
        in csv_text
    )
    assert "reasoning_context_used,reasoning_wedge,reasoning_confidence" in csv_text
    assert (
        "passed_output_checks,output_checks,seo_aeo_readiness,geo_readiness,"
        "structured_data"
    ) in csv_text
    assert '"[{""attempt"":1,""passed"":true' in csv_text
    assert '""@context"":""https://schema.org"",""@graph""' in csv_text
    assert "price_squeeze" in csv_text
