from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.blog_generation import (
    BlogPostGenerationConfig,
    BlogPostGenerationService,
    _blog_failure_candidate_snapshot,
    _blog_quality_repair_guidance,
    _is_support_ticket_blog_context,
    _normalize_blog_metadata,
    _quality_policy_for_context,
    parse_blog_post_response,
    support_ticket_descriptive_blog_contract,
)
from extracted_content_pipeline.blog_ports import BlogPostDraft
from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.content_image_provider import ContentImageAsset
from extracted_quality_gate.types import QualityPolicy


class _Blueprints:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def read_blog_blueprints(self, *, scope, target_mode, limit, filters=None):
        self.calls.append({
            "scope": scope,
            "target_mode": target_mode,
            "limit": limit,
            "filters": filters,
        })
        return self.rows


class _BlogPosts:
    def __init__(self):
        self.saved = []

    async def save_drafts(self, drafts, *, scope):
        self.saved.append({"drafts": list(drafts), "scope": scope})
        return [f"blog-{index + 1}" for index, _ in enumerate(drafts)]

    async def list_drafts(self, *, scope, status=None, topic_type=None, limit=None):  # pragma: no cover
        raise AssertionError("not used")

    async def update_status(self, blog_post_id, status, *, scope):  # pragma: no cover
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
            usage={"input_tokens": 13, "output_tokens": 17},
        )


class _Skills:
    def __init__(self, prompts):
        self.prompts = prompts
        self.calls = []

    def get_prompt(self, name):
        self.calls.append(name)
        return self.prompts.get(name)


def _blueprint():
    return {
        "id": "bp-1",
        "slug": "hubspot-pricing-pressure",
        "topic": "HubSpot pricing pressure",
        "topic_type": "vendor_alternative",
        "suggested_title": "HubSpot Pricing Pressure",
        "data_context": {"review_period": ""},
        "available_charts": [{"chart_id": "pricing", "title": "Pricing"}],
    }


def _support_ticket_blueprint():
    return {
        "id": "support-ticket-bp-1",
        "slug": "support-ticket-faq-gaps",
        "topic": "Support-ticket questions customers keep asking",
        "topic_type": "content_ops_support_ticket_faq",
        "suggested_title": "Support-ticket FAQ gaps",
        "data_context": {
            "review_period": "last 90 days",
            "source_period": "Last 90 days of support tickets",
            "source": "support_ticket_provider",
            "source_row_count": 3,
            "included_ticket_row_count": 3,
            "question_like_ticket_count": 3,
            "top_clusters": [
                {"label": "account", "count": 2},
                {"label": "reporting", "count": 1},
            ],
            "category": "support tickets",
            "topic": "Support-ticket questions customers keep asking",
        },
        "available_charts": [],
    }


def _valid_content() -> str:
    body = (
        "## How is HubSpot pricing pressure changing buyer shortlists?\n\n"
        "HubSpot pricing pressure is visible in the last 90 days of review patterns, "
        "especially across 214 reviews where buyers describe renewal friction, budget "
        "concerns, and comparison shopping. The answer is changing buyer shortlists "
        "because teams are checking renewal costs before they commit to another "
        "contract.\n\n"
        "## How should teams read the HubSpot pricing evidence?\n\n"
        "HubSpot pricing evidence should be read as a renewal-risk signal, not as "
        "proof that every buyer has the same problem. The useful pattern is that "
        "customers keep using similar wording about budget pressure, contract terms, "
        "and alternative comparisons when they explain why pricing has become harder "
        "to justify, so HubSpot pricing pressure stays visible in the section's "
        "answer before the supporting details continue."
    )
    return body + "\n\n{{chart:pricing}}\n"


def _valid_support_ticket_content(extra: str = "") -> str:
    body = (
        "## What do repeat support tickets show?\n\n"
        "In the last 90 days, the uploaded 3 support tickets show account "
        "and reporting questions that customers keep asking. The clearest "
        "answer is that these teams need FAQ entries written in customer "
        "wording before another customer has to email support for the same "
        "basic answer.\n\n"
        "## Which FAQ gaps should the team fix first?\n\n"
        "Account questions appear in 2 support tickets, while reporting "
        "questions appear in 1 support ticket. That makes account access the "
        "first FAQ gap to clean up, followed by reporting export instructions "
        "that customers can find before they open another support ticket."
    )
    if extra:
        body = f"{body}\n\n{extra}"
    return body


def _large_support_ticket_descriptive_content(word_count: int = 1_000) -> str:
    sentence = (
        "Customers asked support about account email changes, billing exports, "
        "and report access in the uploaded ticket rows. "
    )
    words = sentence.split()
    repeated = (words * ((word_count // len(words)) + 1))[:word_count]
    midpoint = max(1, len(repeated) // 2)
    return (
        "## What do the uploaded support tickets show?\n\n"
        + " ".join(repeated[:midpoint])
        + "\n\n## Which repeated questions need clearer answers?\n\n"
        + " ".join(repeated[midpoint:])
    )


def _valid_blog_json(**overrides):
    payload = {
        "title": "HubSpot Pricing Pressure Is Changing Buyer Shortlists",
        "slug": "hubspot-pricing-pressure",
        "description": "How renewal pressure changes buyer shortlists.",
        "seo_title": "HubSpot Pricing Pressure",
        "seo_description": "A data-backed look at HubSpot pricing pressure.",
        "target_keyword": "hubspot pricing pressure",
        "secondary_keywords": ["hubspot alternatives", "crm renewal pricing"],
        "faq": [
            {
                "question": "Why is HubSpot pricing pressure changing shortlists?",
                "answer": "Teams are comparing renewal costs earlier.",
            },
            {
                "question": "Who should review alternatives?",
                "answer": "Teams facing budget pressure should compare options.",
            },
            {
                "question": "What should buyers check?",
                "answer": "Buyers should check renewal terms and support needs.",
            },
        ],
        "topic_type": "vendor_alternative",
        "content": _valid_content(),
        "charts": [{"chart_id": "pricing", "title": "Pricing"}],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _valid_support_ticket_blog_json(**overrides):
    payload = {
        "title": "Support-ticket FAQ gaps customers keep hitting",
        "slug": "support-ticket-faq-gaps",
        "description": "How repeat support tickets point to missing FAQ answers.",
        "seo_title": "Support-ticket FAQ Gaps",
        "seo_description": "How small teams can turn support tickets into FAQ answers.",
        "target_keyword": "support ticket FAQ gaps",
        "secondary_keywords": ["support ticket FAQ", "customer support answers"],
        "faq": [
            {
                "question": "What do repeat support tickets show?",
                "answer": "They show which answers customers cannot find.",
            },
            {
                "question": "Which FAQ gaps should the team fix first?",
                "answer": "Start with the highest-volume repeated questions.",
            },
            {
                "question": "Why use customer wording?",
                "answer": "Customers search with their own words, not internal labels.",
            },
        ],
        "topic_type": "content_ops_support_ticket_faq",
        "content": _valid_support_ticket_content(),
        "charts": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


class _ReasoningProvider:
    """Fake CampaignReasoningContextProvider for blog tests.

    Mirrors the shape used by tests/test_extracted_landing_page_generation.py.
    """

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
            "opportunity": dict(opportunity or {}),
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


def _service(
    *,
    rows=None,
    responses=None,
    prompts=None,
    config=None,
    reasoning_context=None,
    image_provider=None,
):
    blueprints = _Blueprints(rows or [_blueprint()])
    blog_posts = _BlogPosts()
    llm = _LLM(responses or [_valid_blog_json()])
    if prompts is None:
        prompts = {"digest/blog_post_generation": "Write from {blueprint_json}"}
    skills = _Skills(prompts)
    service = BlogPostGenerationService(
        blueprints=blueprints,
        blog_posts=blog_posts,
        llm=llm,
        skills=skills,
        reasoning_context=reasoning_context,
        image_provider=image_provider,
        config=config or BlogPostGenerationConfig(
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            )
        ),
    )
    return service, blueprints, blog_posts, llm, skills


def test_parse_blog_post_response_strips_code_fences() -> None:
    parsed = parse_blog_post_response("```json\n" + _valid_blog_json() + "\n```")

    assert parsed is not None
    assert parsed["title"].startswith("HubSpot Pricing")
    assert parsed["content"] == _valid_content().strip()


def test_parse_blog_post_response_returns_none_when_required_fields_missing() -> None:
    """PR-Audit-MINOR-Batch-2: parser only requires ``title`` as the
    "is this a candidate" filter; ``content`` is delegated to the
    quality pack (empty content fires ``content_too_short``)."""
    assert parse_blog_post_response("") is None
    # Missing title -> not a candidate.
    assert parse_blog_post_response('{"content": "No title"}') is None
    # Missing content -> still a candidate; the quality pack judges it.
    parsed = parse_blog_post_response('{"title": "No content"}')
    assert parsed is not None
    assert parsed["title"] == "No content"
    assert parsed["content"] == ""


def test_normalize_blog_metadata_trims_overlong_seo_title() -> None:
    parsed = json.loads(_valid_blog_json(seo_title="A" * 61))

    normalized = _normalize_blog_metadata(parsed, quality_policy=None)

    assert len(normalized["seo_title"]) == 60
    assert normalized["seo_title"] == "A" * 60


def test_blog_quality_repair_guidance_explains_citation_safety() -> None:
    guidance = _blog_quality_repair_guidance([
        "seo_title_too_long:61_chars_max_60",
        "geo_citation_safety_failed",
    ])

    assert "Shorten `seo_title` to 60 characters or fewer" in guidance
    assert "Remove unresolved placeholders" in guidance
    assert "unsupported claims" in guidance
    assert "visible chart IDs" in guidance


def test_blog_failure_candidate_snapshot_is_bounded() -> None:
    parsed = json.loads(_valid_blog_json(content="alpha " * 200))
    parsed["_parse_attempts"] = 2
    parsed["_quality_repair_attempts"] = 1

    snapshot = _blog_failure_candidate_snapshot(parsed, excerpt_chars=25)

    assert snapshot["title"] == "HubSpot Pricing Pressure Is Changing Buyer Shortlists"
    assert snapshot["slug"] == "hubspot-pricing-pressure"
    assert snapshot["seo_title"] == "HubSpot Pricing Pressure"
    assert snapshot["target_keyword"] == "hubspot pricing pressure"
    assert snapshot["topic_type"] == "vendor_alternative"
    assert snapshot["word_count"] == 200
    assert snapshot["generation_parse_attempts"] == 2
    assert snapshot["generation_quality_repair_attempts"] == 1
    assert snapshot["content_excerpt_head"] == ("alpha " * 200)[:25]
    assert snapshot["content_excerpt_tail"] == ("alpha " * 200).strip()[-25:]
    assert snapshot["content_truncated"] is True


@pytest.mark.parametrize(
    "data_context",
    [
        {"source": "support_ticket_provider"},
        {"provider": "support_ticket_upload"},
        {"source_period": "Last 90 days of support tickets"},
        {"source_period": "support-ticket upload"},
        {"category": "support tickets"},
        {"topic": "support ticket FAQ gaps"},
        {"included_ticket_row_count": 3},
        {"question_like_ticket_count": 2},
        {"top_ticket_clusters": [{"label": "account", "count": 2}]},
        {"top_clusters": [{"label": "reporting", "count": 1}]},
    ],
)
def test_support_ticket_blog_context_detection_markers_bite(
    data_context: dict[str, object],
) -> None:
    assert _is_support_ticket_blog_context(data_context) is True


@pytest.mark.parametrize(
    "data_context",
    [
        {},
        {"source": "review_provider"},
        {"top_clusters": "pricing, onboarding"},
        {"top_ticket_clusters": "account"},
        {"top_clusters": []},
        {"top_ticket_clusters": []},
        {"included_ticket_row_count": 0},
        {"question_like_ticket_count": ""},
    ],
)
def test_support_ticket_blog_context_detection_rejects_false_positives(
    data_context: dict[str, object],
) -> None:
    assert _is_support_ticket_blog_context(data_context) is False


def test_support_ticket_descriptive_blog_contract_requires_no_outcome_or_resolution_evidence() -> None:
    contract = support_ticket_descriptive_blog_contract({
        "source": "support_ticket_provider",
        "source_row_count": 36,
        "included_ticket_row_count": 36,
        "has_measured_outcomes": False,
        "support_ticket_resolution_evidence_present": False,
        "top_clusters": [
            {"label": "account access", "count": 2},
            {"label": "reporting export", "count": 1},
        ],
        "faq_questions": [
            "How do I change the account email?",
            "Why is the report export missing columns?",
        ],
        "customer_wording_examples": [
            {
                "source_id": "ticket-1",
                "pain_category": "account access",
                "text": "How do I change the account email?",
            }
        ],
    })

    assert contract["support_ticket_blog_mode"] == "descriptive_no_outcome"
    assert "observed support-ticket clusters" in contract["allowed_claims"][0]
    assert "future ticket reduction or deflection" in contract["forbidden_claims"]
    assert (
        "claims that FAQ entries help customers find answers or avoid tickets"
        in contract["forbidden_claims"]
    )
    assert (
        "fixed calendar windows, rolling periods, or future tracking intervals "
        "when uploaded tickets are undated"
        in contract["forbidden_claims"]
    )
    assert contract["draft_answer_guidance"].startswith("Draft answer -")
    assert [section["heading"] for section in contract["required_section_outline"]] == [
        "What the uploaded support tickets show",
        "Which FAQ gaps should be reviewed first",
        "Draft FAQ shells to verify",
        "What to measure after publishing",
    ]
    assert contract["draft_faq_shells"][0] == {
        "cluster": "account access",
        "observed_ticket_count": 2,
        "draft_question": "How do I change the account email?",
        "answer_shell": (
            "Draft answer - support team should add the verified resolution "
            "before publishing."
        ),
        "verification_needed": [
            "verified resolution",
            "approved customer-facing wording",
            "support owner review",
        ],
        "source_ids": ["ticket-1"],
    }
    assert contract["draft_faq_shells"][1]["draft_question"] == (
        "Why is the report export missing columns?"
    )
    assert contract["measurement_guidance"] == [
        "Track new tickets by the same observed cluster labels after publishing.",
        "Review FAQ page traffic and customer feedback as signals to inspect.",
        "Compare future tickets against the observed clusters without claiming causality.",
        (
            "Do not add fixed day, week, month, 30-day, 60-day, or 90-day "
            "checkpoints unless the uploaded tickets include a dated source window."
        ),
    ]
    assert support_ticket_descriptive_blog_contract({
        "source": "support_ticket_provider",
        "has_measured_outcomes": True,
        "support_ticket_resolution_evidence_present": False,
    }) == {}
    assert support_ticket_descriptive_blog_contract({
        "source": "support_ticket_provider",
        "has_measured_outcomes": False,
        "support_ticket_resolution_evidence_present": True,
    }) == {}


def test_support_ticket_draft_shells_skip_aggregate_buckets_and_unrelated_examples() -> None:
    contract = support_ticket_descriptive_blog_contract({
        "source": "support_ticket_provider",
        "source_row_count": 28,
        "included_ticket_row_count": 28,
        "has_measured_outcomes": False,
        "support_ticket_resolution_evidence_present": False,
        "top_clusters": [
            {"label": "login issues", "count": 8},
            {"label": "billing questions", "count": 5},
            {"label": "remaining", "count": 9},
            {"label": "uncategorized", "count": 6},
            {"label": "shipping delays", "count": 4},
        ],
        "faq_questions": [
            "How do I reset login access?",
            "Why is shipping delayed?",
        ],
        "customer_wording_examples": [
            {
                "source_id": "ticket-login",
                "pain_category": "login issues",
                "text": "How do I reset login access?",
            },
            {
                "source_id": "ticket-shipping",
                "pain_category": "shipping delays",
                "text": "Why is shipping delayed?",
            },
        ],
    })

    shells = contract["draft_faq_shells"]
    assert [shell["cluster"] for shell in shells] == [
        "login issues",
        "billing questions",
        "shipping delays",
    ]
    assert shells[1] == {
        "cluster": "billing questions",
        "observed_ticket_count": 5,
        "draft_question": "What should the team verify for billing questions?",
        "answer_shell": (
            "Draft answer - support team should add the verified resolution "
            "before publishing."
        ),
        "verification_needed": [
            "verified resolution",
            "approved customer-facing wording",
            "support owner review",
        ],
    }


def test_small_support_ticket_blog_context_uses_compact_quality_policy() -> None:
    policy = _quality_policy_for_context(
        {
            "data_context": {
                "source": "support_ticket_provider",
                "source_row_count": 4,
                "included_ticket_row_count": 4,
                "has_measured_outcomes": False,
                "support_ticket_resolution_evidence_present": False,
            }
        },
        base_policy=None,
    )

    assert policy is not None
    assert policy.thresholds["min_words"] == 700
    assert policy.thresholds["target_words"] == 1100
    assert policy.metadata["support_ticket_small_upload"] is True


def test_small_support_ticket_blog_policy_preserves_explicit_thresholds() -> None:
    base = QualityPolicy(
        name="custom",
        thresholds={"min_words": 20, "target_words": 30, "pass_score": 0},
    )

    policy = _quality_policy_for_context(
        {
            "data_context": {
                "source": "support_ticket_provider",
                "source_row_count": 4,
                "included_ticket_row_count": 4,
                "has_measured_outcomes": False,
                "support_ticket_resolution_evidence_present": False,
            }
        },
        base_policy=base,
    )

    assert policy is not None
    assert policy.thresholds["min_words"] == 20
    assert policy.thresholds["target_words"] == 30
    assert policy.thresholds["pass_score"] == 0


def test_demo_sized_support_ticket_blog_context_uses_compact_quality_policy() -> None:
    policy = _quality_policy_for_context(
        {
            "data_context": {
                "source": "support_ticket_provider",
                "source_row_count": 36,
                "included_ticket_row_count": 36,
                "has_measured_outcomes": False,
                "support_ticket_resolution_evidence_present": False,
            }
        },
        base_policy=None,
    )

    assert policy is not None
    assert policy.thresholds["min_words"] == 700
    assert policy.thresholds["target_words"] == 1100
    assert policy.metadata["support_ticket_small_upload"] is True


def test_support_ticket_blog_context_uses_compact_policy_when_included_count_is_small() -> None:
    policy = _quality_policy_for_context(
        {
            "data_context": {
                "source": "support_ticket_provider",
                "source_row_count": 250,
                "included_ticket_row_count": 30,
                "has_measured_outcomes": False,
                "support_ticket_resolution_evidence_present": False,
            }
        },
        base_policy=None,
    )

    assert policy is not None
    assert policy.thresholds["min_words"] == 700
    assert policy.thresholds["target_words"] == 1100
    assert policy.metadata["support_ticket_small_upload"] is True


def test_small_support_ticket_blog_policy_does_not_apply_with_outcome_evidence() -> None:
    base = QualityPolicy(name="custom", thresholds={"pass_score": 0})

    policy = _quality_policy_for_context(
        {
            "data_context": {
                "source": "support_ticket_provider",
                "source_row_count": 4,
                "included_ticket_row_count": 4,
                "has_measured_outcomes": True,
                "support_ticket_resolution_evidence_present": False,
            }
        },
        base_policy=base,
    )

    assert policy is base


def test_small_support_ticket_blog_policy_does_not_apply_to_large_uploads() -> None:
    base = QualityPolicy(name="custom", thresholds={"pass_score": 0})

    policy = _quality_policy_for_context(
        {
            "data_context": {
                "source": "support_ticket_provider",
                "source_row_count": 250,
                "included_ticket_row_count": 75,
                "has_measured_outcomes": False,
                "support_ticket_resolution_evidence_present": False,
            }
        },
        base_policy=base,
    )

    assert policy is base


@pytest.mark.parametrize(
    "data_context",
    [
        {
            "source": "support_ticket_provider",
            "has_measured_outcomes": False,
            "support_ticket_resolution_evidence_present": False,
        },
        {
            "source": "support_ticket_provider",
            "source_row_count": 0,
            "included_ticket_row_count": "",
            "has_measured_outcomes": False,
            "support_ticket_resolution_evidence_present": False,
        },
    ],
)
def test_small_support_ticket_blog_policy_requires_positive_row_count(
    data_context: dict[str, object],
) -> None:
    base = QualityPolicy(name="custom", thresholds={"pass_score": 0})

    policy = _quality_policy_for_context(
        {"data_context": data_context},
        base_policy=base,
    )

    assert policy is base


@pytest.mark.asyncio
async def test_generate_persists_blog_drafts_via_ports() -> None:
    service, blueprints, blog_posts, llm, skills = _service()
    scope = TenantScope(account_id="acct-1")

    result = await service.generate(
        scope=scope,
        target_mode="vendor_retention",
        limit=1,
        filters={"topic": "pricing"},
    )

    assert result.generated == 1
    assert result.saved_ids == ("blog-1",)
    assert blueprints.calls == [{
        "scope": scope,
        "target_mode": "vendor_retention",
        "limit": 1,
        "filters": {"topic": "pricing"},
    }]
    assert skills.calls == ["digest/blog_post_generation"]
    assert llm.calls[0]["metadata"]["asset_type"] == "blog_post"
    draft = blog_posts.saved[0]["drafts"][0]
    assert isinstance(draft, BlogPostDraft)
    assert draft.slug == "hubspot-pricing-pressure"
    assert draft.metadata["target_keyword"] == "hubspot pricing pressure"
    assert draft.metadata["generation_model"] == "test-model"


@pytest.mark.asyncio
async def test_generate_attaches_optional_blog_cover_image_before_save() -> None:
    image_provider = _ImageProvider(
        ContentImageAsset(
            url="https://images.example.com/blog.jpg",
            provider="unsplash",
            alt_text="Pricing dashboard",
            attribution_name="Ada Lens",
            attribution_url="https://unsplash.example.com/@ada",
            source_id="photo-1",
        )
    )
    service, _blueprints, blog_posts, _llm, _skills = _service(
        image_provider=image_provider
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    draft = blog_posts.saved[0]["drafts"][0]
    assert draft.metadata["cover_image"]["url"] == "https://images.example.com/blog.jpg"
    assert draft.metadata["cover_image"]["provider"] == "unsplash"
    assert image_provider.calls[0].asset_type == "blog_post"
    assert image_provider.calls[0].slot == "cover"
    assert "HubSpot" in image_provider.calls[0].title


@pytest.mark.asyncio
async def test_generate_keeps_blog_post_when_image_provider_fails() -> None:
    service, _blueprints, blog_posts, _llm, _skills = _service(
        image_provider=_ImageProvider(error=RuntimeError("image service unavailable"))
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    draft = blog_posts.saved[0]["drafts"][0]
    assert "cover_image" not in draft.metadata


@pytest.mark.asyncio
async def test_generate_sends_blueprint_in_user_message_when_template_has_no_placeholder() -> None:
    service, _blueprints, _blog_posts, llm, _skills = _service(
        prompts={"digest/blog_post_generation": "Write a post."}
    )

    await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert "blueprint JSON" in llm.calls[0]["messages"][1].content
    assert "HubSpot pricing pressure" in llm.calls[0]["messages"][1].content


@pytest.mark.asyncio
async def test_generate_keeps_dynamic_blueprint_out_of_system_prompt() -> None:
    service, _blueprints, _blog_posts, llm, _skills = _service(
        prompts={"digest/blog_post_generation": "Write from {blueprint_json}"}
    )

    await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    system_prompt = llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert "the blueprint JSON supplied in the user message" in system_prompt
    assert "HubSpot pricing pressure" not in system_prompt
    assert '"topic":"HubSpot pricing pressure"' in user_prompt


@pytest.mark.asyncio
async def test_generate_blocks_low_quality_posts_without_saving() -> None:
    service, _blueprints, blog_posts, _llm, _skills = _service(
        responses=[_valid_blog_json(content="Too short.")],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_repair_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 70},
            )
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["reason"] == "quality_blocked"
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_blocks_missing_seo_aeo_fields_without_saving() -> None:
    payload = json.loads(_valid_blog_json())
    for key in (
        "seo_title",
        "seo_description",
        "target_keyword",
        "secondary_keywords",
        "faq",
    ):
        payload.pop(key)
    service, _blueprints, blog_posts, _llm, _skills = _service(
        responses=[json.dumps(payload)],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_repair_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            )
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["reason"] == "quality_blocked"
    assert set(result.errors[0]["blockers"]) >= {
        "missing_seo_title",
        "missing_seo_description",
        "missing_target_keyword",
        "missing_secondary_keywords",
        "too_few_faq_entries:0_need_3",
    }
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_blocks_missing_geo_contract_without_saving() -> None:
    payload = json.loads(_valid_blog_json())
    payload["content"] = (
        "## How is HubSpot pricing pressure changing shortlists?\n\n"
        "HubSpot pricing pressure changes shortlists because buyers compare "
        "renewal terms before they commit to another contract."
    )
    service, _blueprints, blog_posts, _llm, _skills = _service(
        responses=[json.dumps(payload)],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_repair_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            )
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["reason"] == "quality_blocked"
    assert set(result.errors[0]["blockers"]) >= {
        "geo_citable_section_structure_missing",
        "geo_evidence_specificity_missing",
        "geo_freshness_context_missing",
    }
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_repairs_geo_quality_block_with_retry_budget() -> None:
    payload = json.loads(_valid_blog_json())
    payload["content"] = (
        "## How is HubSpot pricing pressure changing shortlists?\n\n"
        "HubSpot pricing pressure changes shortlists because buyers compare "
        "renewal terms before they commit to another contract."
    )
    service, _blueprints, blog_posts, llm, _skills = _service(
        responses=[json.dumps(payload), _valid_blog_json()],
        config=BlogPostGenerationConfig(
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            )
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 1
    assert result.skipped == 0
    assert result.errors == ()
    assert len(llm.calls) == 2
    retry_prompt = llm.calls[1]["messages"][1].content
    assert "Quality blockers:" in retry_prompt
    assert "geo_citable_section_structure_missing" in retry_prompt
    assert llm.calls[1]["metadata"]["quality_repair_attempt_no"] == 1
    draft = blog_posts.saved[0]["drafts"][0]
    assert draft.metadata["generation_usage"] == {"input_tokens": 26, "output_tokens": 34}
    assert draft.metadata["generation_parse_attempts"] == 1
    assert draft.metadata["generation_quality_repair_attempts"] == 1


@pytest.mark.asyncio
async def test_generate_blocks_support_ticket_generated_content_failure_without_saving() -> None:
    service, _blueprints, blog_posts, _llm, _skills = _service(
        rows=[_support_ticket_blueprint()],
        responses=[
            _valid_support_ticket_blog_json(
                content=_valid_support_ticket_content(
                    "These answers can reduce repeat tickets by 30-45%."
                )
            )
        ],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_repair_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["reason"] == "quality_blocked"
    assert any(
        blocker.startswith("support_ticket_generated_content:")
        and "percentage claims not backed" in blocker
        for blocker in result.errors[0]["blockers"]
    )
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_repairs_support_ticket_generated_content_failure() -> None:
    service, _blueprints, blog_posts, llm, _skills = _service(
        rows=[_support_ticket_blueprint()],
        responses=[
            _valid_support_ticket_blog_json(
                content=_valid_support_ticket_content(
                    "These answers can reduce repeat tickets by 30-45%."
                )
            ),
            _valid_support_ticket_blog_json(),
        ],
        config=BlogPostGenerationConfig(
            quality_repair_attempts=1,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    assert result.skipped == 0
    assert result.errors == ()
    retry_prompt = llm.calls[1]["messages"][1].content
    assert "support_ticket_generated_content:" in retry_prompt
    assert "percentage claims not backed" in retry_prompt
    assert "Do not invent calendar windows" in retry_prompt
    draft = blog_posts.saved[0]["drafts"][0]
    assert draft.metadata["generation_quality_repair_attempts"] == 1


@pytest.mark.asyncio
async def test_generate_saves_descriptive_support_ticket_blog_without_outcome_or_resolution_evidence() -> None:
    descriptive_content = _valid_support_ticket_content(
        "Teams with fewer tickets may have a simpler support queue, but this "
        "upload does not prove ticket reduction. Customers often go to billing "
        "questions first when account wording is unclear. You can export your "
        "data from most analytics tools. For this uploaded product, the draft "
        "answer stays as: Draft answer - support team should add the verified "
        "resolution before publishing. Faster resolution for customers is "
        "something to measure after publishing, not an outcome these tickets prove."
    )
    service, _blueprints, blog_posts, _llm, _skills = _service(
        rows=[_support_ticket_blueprint()],
        responses=[_valid_support_ticket_blog_json(content=descriptive_content)],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_repair_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    assert result.skipped == 0
    assert result.errors == ()
    draft = blog_posts.saved[0]["drafts"][0]
    assert not draft.data_context.get("support_ticket_resolution_evidence_present")
    assert not draft.data_context.get("has_measured_outcomes")
    assert draft.data_context["support_ticket_blog_mode"] == "descriptive_no_outcome"
    assert "forbidden_claims" in draft.data_context


@pytest.mark.asyncio
async def test_generate_puts_support_ticket_descriptive_contract_in_prompt() -> None:
    service, _blueprints, _blog_posts, llm, _skills = _service(
        rows=[_support_ticket_blueprint()],
        responses=[_valid_support_ticket_blog_json()],
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    system_prompt = llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert '"support_ticket_blog_mode":"descriptive_no_outcome"' not in system_prompt
    assert '"support_ticket_blog_mode":"descriptive_no_outcome"' in user_prompt
    assert '"allowed_claims":' in user_prompt
    assert '"forbidden_claims":' in user_prompt
    assert '"draft_answer_guidance":' in user_prompt
    assert '"required_section_outline":' in user_prompt
    assert '"draft_faq_shells":' in user_prompt
    assert '"measurement_guidance":' in user_prompt
    assert "Support-ticket descriptive mode instructions:" in user_prompt
    assert "Do not rank tied clusters by business impact" in user_prompt
    assert (
        "Use `data_context.required_section_outline` as the H2 section order"
        in user_prompt
    )
    assert "Measurement language must be observational only" in user_prompt
    assert "metadata, FAQ metadata, tags, and chart copy" in user_prompt


@pytest.mark.asyncio
async def test_generate_recomputes_stale_support_ticket_descriptive_contract() -> None:
    blueprint = _support_ticket_blueprint()
    blueprint["data_context"]["support_ticket_blog_mode"] = "stale_non_descriptive"
    service, _blueprints, _blog_posts, llm, _skills = _service(
        rows=[blueprint],
        responses=[_valid_support_ticket_blog_json()],
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    system_prompt = llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert '"support_ticket_blog_mode":"descriptive_no_outcome"' not in system_prompt
    assert '"support_ticket_blog_mode":"descriptive_no_outcome"' in user_prompt
    assert "stale_non_descriptive" not in system_prompt
    assert "stale_non_descriptive" not in user_prompt
    assert "Support-ticket descriptive mode instructions:" in user_prompt


@pytest.mark.asyncio
async def test_generate_clears_stale_descriptive_contract_for_outcome_backed_context() -> None:
    blueprint = _support_ticket_blueprint()
    blueprint["data_context"].update({
        "has_measured_outcomes": True,
        "support_ticket_blog_mode": "descriptive_no_outcome",
        "allowed_claims": ["stale allowed"],
        "forbidden_claims": ["stale forbidden"],
        "draft_answer_guidance": "stale draft guidance",
        "required_section_outline": [{"heading": "stale heading"}],
        "draft_faq_shells": [{"draft_question": "stale question"}],
        "measurement_guidance": ["stale measurement"],
    })
    service, _blueprints, blog_posts, llm, _skills = _service(
        rows=[blueprint],
        responses=[_valid_support_ticket_blog_json()],
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    system_prompt = llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    draft = blog_posts.saved[0]["drafts"][0]
    assert "support_ticket_blog_mode" not in draft.data_context
    assert "allowed_claims" not in draft.data_context
    assert "forbidden_claims" not in draft.data_context
    assert "draft_answer_guidance" not in draft.data_context
    assert "required_section_outline" not in draft.data_context
    assert "draft_faq_shells" not in draft.data_context
    assert "measurement_guidance" not in draft.data_context
    assert "descriptive_no_outcome" not in system_prompt
    assert "descriptive_no_outcome" not in user_prompt
    assert "stale allowed" not in system_prompt
    assert "stale allowed" not in user_prompt
    assert "stale heading" not in system_prompt
    assert "stale heading" not in user_prompt
    assert "stale question" not in system_prompt
    assert "stale question" not in user_prompt
    assert "stale measurement" not in system_prompt
    assert "stale measurement" not in user_prompt
    assert "Support-ticket descriptive mode instructions:" not in user_prompt


@pytest.mark.asyncio
async def test_quality_repair_prompt_keeps_support_ticket_descriptive_contract() -> None:
    bad_content = _valid_support_ticket_content(
        "These answers can reduce repeat tickets by 30-45%."
    )
    service, _blueprints, _blog_posts, llm, _skills = _service(
        rows=[_support_ticket_blueprint()],
        responses=[
            _valid_support_ticket_blog_json(content=bad_content),
            _valid_support_ticket_blog_json(),
        ],
        config=BlogPostGenerationConfig(
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            )
        ),
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    repair_system_prompt = llm.calls[1]["messages"][0].content
    retry_prompt = llm.calls[1]["messages"][1].content
    assert '"support_ticket_blog_mode":"descriptive_no_outcome"' not in repair_system_prompt
    assert '"support_ticket_blog_mode":"descriptive_no_outcome"' in retry_prompt
    assert "follow its `allowed_claims`, `forbidden_claims`, and `draft_answer_guidance`" in retry_prompt
    assert "Support-ticket descriptive mode instructions:" in retry_prompt
    assert "Do not rank tied clusters by business impact" in retry_prompt
    assert (
        "Use `data_context.required_section_outline` as the H2 section order"
        in retry_prompt
    )


@pytest.mark.asyncio
async def test_generate_large_support_ticket_context_keeps_default_word_floor() -> None:
    service, _blueprints, blog_posts, _llm, _skills = _service(
        rows=[_support_ticket_blueprint()],
        responses=[
            _valid_support_ticket_blog_json(
                content=_large_support_ticket_descriptive_content(),
            )
        ],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_repair_attempts=0,
        ),
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
        data_context={
            "source": "support_ticket_provider",
            "source_row_count": 250,
            "included_ticket_row_count": 75,
            "question_like_ticket_count": 75,
            "has_measured_outcomes": False,
            "support_ticket_resolution_evidence_present": False,
            "support_ticket_resolution_evidence_count": 0,
        },
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["reason"] == "quality_blocked"
    assert any(
        blocker.startswith("content_too_short:")
        and blocker.endswith("_words_need_1500")
        for blocker in result.errors[0]["blockers"]
    )
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_keeps_trusted_support_ticket_context_over_model_context() -> None:
    service, _blueprints, blog_posts, _llm, _skills = _service(
        rows=[_support_ticket_blueprint()],
        responses=[
            _valid_support_ticket_blog_json(
                data_context={
                    "source": "review_provider",
                    "included_ticket_row_count": 999,
                    "model_note": "kept",
                }
            )
        ],
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    draft = blog_posts.saved[0]["drafts"][0]
    assert draft.data_context["source"] == "support_ticket_provider"
    assert draft.data_context["included_ticket_row_count"] == 3
    assert draft.data_context["model_note"] == "kept"


@pytest.mark.asyncio
async def test_generate_blocks_support_ticket_draft_after_repair_budget_exhausted() -> None:
    bad_content = _valid_support_ticket_content(
        "These answers can reduce repeat tickets by 30-45%."
    )
    service, _blueprints, blog_posts, llm, _skills = _service(
        rows=[_support_ticket_blueprint()],
        responses=[
            _valid_support_ticket_blog_json(content=bad_content),
            _valid_support_ticket_blog_json(content=bad_content),
        ],
        config=BlogPostGenerationConfig(
            quality_repair_attempts=1,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["reason"] == "quality_blocked"
    assert len(llm.calls) == 2
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_does_not_run_support_ticket_gate_for_other_blog_contexts() -> None:
    service, _blueprints, blog_posts, _llm, _skills = _service(
        rows=[_blueprint()],
        responses=[
            _valid_blog_json(
                content=_valid_content()
                + "\n\nThese changes can reduce repeat tickets by 30-45%."
            )
        ],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_repair_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
    )

    assert result.generated == 1
    assert result.skipped == 0
    assert result.errors == ()
    assert blog_posts.saved[0]["drafts"][0].slug == "hubspot-pricing-pressure"


@pytest.mark.asyncio
async def test_generate_quality_repair_prompt_explains_known_blockers() -> None:
    payload = json.loads(_valid_blog_json())
    payload["title"] = "Buyer Shortlists Are Changing"
    payload["seo_title"] = "A" * 61
    payload["content"] = (
        "## Why are buyer shortlists changing?\n\n"
        "Buyers are comparing renewal costs before they commit to another contract."
    )
    service, _blueprints, _blog_posts, llm, _skills = _service(
        responses=[json.dumps(payload), _valid_blog_json()],
        config=BlogPostGenerationConfig(
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            )
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 1
    assert len(llm.calls) == 2
    retry_prompt = llm.calls[1]["messages"][1].content
    assert "Quality blockers:" in retry_prompt
    assert "content_too_short:" in retry_prompt
    assert "geo_entity_clarity_missing" in retry_prompt
    assert "geo_citable_section_structure_missing" in retry_prompt
    assert "Expand `content` to at least 1500 words" in retry_prompt
    assert "include the exact current `target_keyword` string" in retry_prompt
    assert "repeat that exact phrase naturally" in retry_prompt
    assert "Replace vague H2 headings" in retry_prompt
    assert "`Summary`" in retry_prompt
    assert "specific question or answer headings" in retry_prompt
    assert "Rewrite at least two H2 sections" in retry_prompt
    assert "first paragraph immediately after each of those H2 headings" in retry_prompt
    assert "must be 40-120 words" in retry_prompt
    assert "exact `target_keyword` string from the previous JSON" in retry_prompt
    assert "Do not rely on the title, introduction, blockquotes, bullets" in retry_prompt


def test_blog_quality_repair_guidance_uses_compact_support_ticket_word_floor() -> None:
    guidance = _blog_quality_repair_guidance(("content_too_short:512_words_need_700",))

    assert "at least 700 words" in guidance
    assert "compact support-ticket brief shape" in guidance
    assert "1500-2200" not in guidance


@pytest.mark.asyncio
async def test_generate_uses_multiple_quality_repair_attempts() -> None:
    first_payload = json.loads(_valid_blog_json())
    first_payload["content"] = (
        "## Why are buyer shortlists changing?\n\n"
        "Buyers are comparing renewal costs before they commit to another contract."
    )
    second_payload = json.loads(_valid_blog_json())
    second_payload["title"] = "Buyer Shortlist Changes"
    second_payload["content"] = (
        _valid_content()
        .replace("HubSpot pricing pressure", "Buyer renewal pressure")
        .replace("HubSpot pricing evidence", "Buyer renewal evidence")
    )
    service, _blueprints, blog_posts, llm, _skills = _service(
        responses=[
            json.dumps(first_payload),
            json.dumps(second_payload),
            _valid_blog_json(),
        ],
        config=BlogPostGenerationConfig(
            quality_repair_attempts=2,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 1
    assert result.errors == ()
    assert len(llm.calls) == 3
    assert llm.calls[1]["metadata"]["quality_repair_attempt_no"] == 1
    assert llm.calls[2]["metadata"]["quality_repair_attempt_no"] == 2
    draft = blog_posts.saved[0]["drafts"][0]
    assert draft.metadata["generation_quality_repair_attempts"] == 2


@pytest.mark.asyncio
async def test_generate_does_not_repair_quality_block_without_repair_budget() -> None:
    payload = json.loads(_valid_blog_json())
    payload["content"] = (
        "## How is HubSpot pricing pressure changing shortlists?\n\n"
        "HubSpot pricing pressure changes shortlists because buyers compare "
        "renewal terms before they commit to another contract."
    )
    service, _blueprints, blog_posts, llm, _skills = _service(
        responses=[json.dumps(payload), _valid_blog_json()],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_repair_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            )
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["reason"] == "quality_blocked"
    assert result.errors[0]["failed_candidate"]["title"] == (
        "HubSpot Pricing Pressure Is Changing Buyer Shortlists"
    )
    assert result.errors[0]["failed_candidate"]["word_count"] == 24
    assert len(llm.calls) == 1
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_reports_unparseable_quality_repair_response() -> None:
    payload = json.loads(_valid_blog_json())
    payload["content"] = (
        "## How is HubSpot pricing pressure changing shortlists?\n\n"
        "HubSpot pricing pressure changes shortlists because buyers compare "
        "renewal terms before they commit to another contract."
    )
    service, _blueprints, blog_posts, llm, _skills = _service(
        responses=[json.dumps(payload), "not valid repair json"],
        config=BlogPostGenerationConfig(
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["blueprint_id"] == "bp-1"
    assert result.errors[0]["reason"] == "quality_repair_unparseable"
    assert "geo_citable_section_structure_missing" in result.errors[0]["blockers"]
    assert result.errors[0]["quality_repair_attempt_no"] == 1
    assert result.errors[0]["failed_candidate"]["title"] == (
        "HubSpot Pricing Pressure Is Changing Buyer Shortlists"
    )
    assert result.errors[0]["failed_candidate"]["word_count"] == 24
    assert len(llm.calls) == 2
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_quality_repair_failure_skips_one_blueprint_not_batch() -> None:
    payload = json.loads(_valid_blog_json())
    payload["content"] = (
        "## How is HubSpot pricing pressure changing shortlists?\n\n"
        "HubSpot pricing pressure changes shortlists because buyers compare "
        "renewal terms before they commit to another contract."
    )
    rows = [
        _blueprint(),
        {**_blueprint(), "id": "bp-2", "slug": "hubspot-pricing-pressure-2"},
    ]
    service, _blueprints, blog_posts, llm, _skills = _service(
        rows=rows,
        responses=[
            json.dumps(payload),
            RuntimeError("repair backend timeout"),
            _valid_blog_json(slug="hubspot-pricing-pressure-2"),
        ],
        config=BlogPostGenerationConfig(
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            )
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=2)

    assert result.generated == 1
    assert result.skipped == 1
    assert result.errors == ({
        "blueprint_id": "bp-1",
        "reason": "quality_repair_failed",
        "error": "repair backend timeout",
        "error_type": "RuntimeError",
    },)
    assert len(llm.calls) == 3
    assert blog_posts.saved[0]["drafts"][0].slug == "hubspot-pricing-pressure-2"


@pytest.mark.asyncio
async def test_generate_routes_missing_content_to_quality_blocked_not_unparseable() -> None:
    payload = json.loads(_valid_blog_json())
    payload.pop("content")
    service, _blueprints, blog_posts, _llm, _skills = _service(
        responses=[json.dumps(payload)],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_repair_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 70},
            )
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 0
    assert result.skipped == 1
    assert result.errors[0]["reason"] == "quality_blocked"
    assert any("content_too_short" in blocker for blocker in result.errors[0]["blockers"])
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_reports_unparseable_responses() -> None:
    service, _blueprints, blog_posts, _llm, _skills = _service(
        responses=["not json", "still not json"]
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 0
    assert result.errors == ({"blueprint_id": "bp-1", "reason": "unparseable_response"},)
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_retries_unparseable_response_once_by_default() -> None:
    service, _blueprints, blog_posts, llm, _skills = _service(
        responses=["not json", _valid_blog_json()]
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 1
    assert len(llm.calls) == 2
    assert llm.calls[0]["metadata"]["attempt_no"] == 1
    assert llm.calls[1]["metadata"]["attempt_no"] == 2
    assert "Previous response excerpt:\nnot json" in llm.calls[1]["messages"][1].content
    draft = blog_posts.saved[0]["drafts"][0]
    assert draft.metadata["generation_parse_attempts"] == 2


@pytest.mark.asyncio
async def test_generate_can_disable_parse_retry() -> None:
    service, _blueprints, blog_posts, llm, _skills = _service(
        responses=["not json", _valid_blog_json()],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 0
    assert result.errors == ({"blueprint_id": "bp-1", "reason": "unparseable_response"},)
    assert len(llm.calls) == 1
    assert blog_posts.saved == []


@pytest.mark.asyncio
async def test_generate_accumulates_usage_across_parse_retry_attempts() -> None:
    service, _blueprints, blog_posts, _llm, _skills = _service(
        responses=[
            {
                "content": "not json",
                "model": "first-model",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
            {
                "content": _valid_blog_json(),
                "model": "final-model",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        ]
    )

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert result.generated == 1
    draft = blog_posts.saved[0]["drafts"][0]
    assert draft.metadata["generation_model"] == "final-model"
    assert draft.metadata["generation_usage"] == {"input_tokens": 12, "output_tokens": 5}
    assert draft.metadata["generation_parse_attempts"] == 2


# -----------------------
# PR-OptionA-2: per-call temperature/max_tokens/parse_retry_attempts overrides
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_llm_tuning_overrides_win_over_construction_config():
    """Resolved-value param reaches the LLM, not self._config.X."""

    service, _bps, _drafts, llm, _skills = _service(
        responses=[
            "not parseable",
            _valid_blog_json(),
        ],
        config=BlogPostGenerationConfig(
            temperature=0.3,
            max_tokens=4096,
            parse_retry_attempts=0,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    await service.generate(
        scope=TenantScope(),
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
    service, _bps, _drafts, llm, _skills = _service(
        responses=[_valid_blog_json()],
        config=BlogPostGenerationConfig(
            temperature=0.7,
            max_tokens=999,
            quality_policy=QualityPolicy(
                name="blog_post",
                thresholds={"min_words": 20, "target_words": 20, "pass_score": 0},
            ),
        ),
    )

    await service.generate(
        scope=TenantScope(),
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
    service, _bps, _drafts, llm, _skills = _service(
        responses=[long_invalid, _valid_blog_json()],
        config=BlogPostGenerationConfig(
            parse_retry_attempts=1,
            parse_retry_response_excerpt_chars=200,
        ),
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor",
        parse_retry_response_excerpt_chars=50,
    )

    retry_user_prompt = llm.calls[1]["messages"][1].content
    assert "XXX" in retry_user_prompt
    excerpt_section = retry_user_prompt.split("excerpt:")[1].lstrip()
    assert len(excerpt_section.rstrip()) <= 50


# -----------------------
# PR-Audit-MINOR-Batch-2: slug length cap + parser-strictness loosening
# -----------------------


def test_slugify_truncates_to_100_chars():
    """A long title produces a clipped slug, not a 2000-char monstrosity."""
    from extracted_content_pipeline.blog_generation import _slugify

    long_title = "this is a very long blog post title " * 20  # ~700 chars
    slug = _slugify(long_title)
    assert len(slug) <= 100
    # No trailing hyphen artifact from the truncation point.
    assert not slug.endswith("-")


def test_slugify_short_input_unchanged():
    from extracted_content_pipeline.blog_generation import _slugify

    assert _slugify("Short Title") == "short-title"
    assert _slugify("") == "blog-post"
    assert _slugify(None) == "blog-post"


def test_parse_blog_post_response_accepts_missing_content_for_quality_pack_to_judge():
    """End-to-end shape: a JSON candidate with title but no content
    is still returned by the parser; the executor passes it to the
    quality pack which fires ``content_too_short`` on empty content.
    Pre-fix, missing content collapsed to ``unparseable_response``."""
    parsed = parse_blog_post_response('{"title": "Has title only"}')
    assert parsed is not None
    assert parsed["title"] == "Has title only"
    # Content is normalized to "" -- quality pack handles the rest.
    assert parsed["content"] == ""


# -----------------------
# PR-Blog-Prompt-Cache-Stability: per-call topic stays in the user prompt
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_topic_stays_out_of_system_prompt():
    service, _bps, _drafts, llm, _skills = _service(
        prompts={
            "digest/blog_post_generation": "Focus: {topic}\n\nWrite from {blueprint_json}"
        }
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
        topic="Renewal pricing pressure on mid-market SaaS",
    )

    system_prompt = llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert "the operator-supplied topic provided in the user message" in system_prompt
    assert "Renewal pricing pressure on mid-market SaaS" not in system_prompt
    assert "Operator-supplied topic focus: Renewal pricing pressure on mid-market SaaS" in user_prompt


@pytest.mark.asyncio
async def test_generate_no_topic_keeps_system_prompt_stable_without_user_topic():
    service, _bps, _drafts, llm, _skills = _service(
        prompts={
            "digest/blog_post_generation": "Focus: {topic}\n\nWrite from {blueprint_json}"
        }
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
        topic=None,
    )

    system_prompt = llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert "{topic}" not in system_prompt
    assert "Focus: the operator-supplied topic provided in the user message" in system_prompt
    assert "Operator-supplied topic focus:" not in user_prompt


@pytest.mark.asyncio
async def test_generate_topic_no_placeholder_no_op():
    """Hosts on the prior prompt without ``{topic}`` are unaffected."""

    service, _bps, _drafts, llm, _skills = _service(
        prompts={"digest/blog_post_generation": "Write from {blueprint_json}"}
    )

    await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=1,
        topic="Some topic",
    )

    system_prompt = llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert "Some topic" not in system_prompt
    assert "Operator-supplied topic focus: Some topic" in user_prompt
    assert "{topic}" not in system_prompt


# -----------------------
# PR-Blog-Reasoning-Parity: blog generator accepts an optional
# reasoning_context provider and merges its payload into the
# blueprint before the LLM call. Mirrors the pattern in
# tests/test_extracted_landing_page_generation.py.
# -----------------------


@pytest.mark.asyncio
async def test_generate_no_reasoning_provider_passes_blueprint_unchanged() -> None:
    """The default (no provider wired) path leaves the blueprint
    untouched -- the LLM sees the original JSON, no reasoning_context
    field is injected."""

    service, _, _, llm, _ = _service()  # no reasoning_context kwarg

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    system_prompt = llm.calls[0]["messages"][0].content
    assert "reasoning_context" not in system_prompt
    assert "campaign_reasoning_context" not in system_prompt
    assert result.as_dict()["reasoning_contexts_used"] == 0


@pytest.mark.asyncio
async def test_generate_with_reasoning_provider_merges_context_into_blueprint() -> None:
    """When the provider returns non-empty context, the blueprint
    JSON sent to the LLM gains a ``reasoning_context`` payload, and
    the draft metadata records the reasoning provider tier."""

    reasoning = _ReasoningProvider(
        CampaignReasoningContext(
            top_theses=(
                {
                    "claim": "Renewal pricing rose 22 percent",
                    "confidence": 0.9,
                    "source_ids": ["r1"],
                },
            ),
            canonical_reasoning={
                "summary": "HubSpot pricing pressure intensifies in Q3.",
            },
        )
    )
    service, _, blog_posts, llm, _ = _service(reasoning_context=reasoning)

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    # Provider was called with the blueprint id + the fixed reasoning
    # target_mode (not the call-site target_mode).
    assert reasoning.calls
    call = reasoning.calls[0]
    assert call["target_id"] == "bp-1"
    assert call["target_mode"] == "blog_blueprint"

    # LLM saw the merged blueprint JSON in the per-run user prompt.
    system_prompt = llm.calls[0]["messages"][0].content
    user_prompt = llm.calls[0]["messages"][1].content
    assert "reasoning_context" not in system_prompt
    assert "reasoning_context" in user_prompt
    assert "campaign_reasoning_context" in user_prompt
    assert "Renewal pricing rose 22 percent" in user_prompt
    result_dict = result.as_dict()
    assert result_dict["reasoning_contexts_used"] == 1
    assert result_dict["consumed_reasoning_contexts"][0]["top_theses"][0]["claim"] == (
        "Renewal pricing rose 22 percent"
    )

    # Draft metadata captured reasoning signal.
    drafts = blog_posts.saved[0]["drafts"]
    assert drafts
    metadata = drafts[0].metadata
    # Metadata fields come from campaign_reasoning_context_metadata; at
    # minimum the provider tier should be visible to consumers.
    assert any("reasoning" in str(key) for key in metadata.keys()), metadata


@pytest.mark.asyncio
async def test_generate_with_reasoning_provider_returning_empty_is_noop() -> None:
    """When the provider returns no content, the blueprint is not
    enriched -- ``reasoning_context`` does not appear in the prompt
    and no reasoning metadata is added to the draft."""

    reasoning = _ReasoningProvider(None)  # provider returns nothing
    service, _, blog_posts, llm, _ = _service(reasoning_context=reasoning)

    result = await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert reasoning.calls  # the provider was consulted
    system_prompt = llm.calls[0]["messages"][0].content
    assert "reasoning_context" not in system_prompt
    assert "campaign_reasoning_context" not in system_prompt
    assert result.as_dict()["reasoning_contexts_used"] == 0

    drafts = blog_posts.saved[0]["drafts"]
    assert drafts
    metadata = drafts[0].metadata
    # No reasoning-shaped metadata fields when the context was empty.
    assert not any("reasoning" in str(key) for key in metadata.keys()), metadata
