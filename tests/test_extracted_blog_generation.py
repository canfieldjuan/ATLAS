from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.blog_generation import (
    BlogPostGenerationConfig,
    BlogPostGenerationService,
    parse_blog_post_response,
)
from extracted_content_pipeline.blog_ports import BlogPostDraft
from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    LLMResponse,
    TenantScope,
)
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


def _valid_content() -> str:
    body = " ".join([
        "Teams",
        "describe",
        "pricing",
        "pressure",
        "during",
        "renewals",
        "and",
        "compare",
        "alternatives",
        "against",
        "implementation",
        "effort",
        "support",
        "needs",
        "and",
        "budget",
        "planning",
        "cycles",
        "before",
        "migration",
        "decisions",
        "are",
        "made",
        "carefully",
        "today",
    ])
    return body + "\n\n{{chart:pricing}}\n"


def _valid_blog_json(**overrides):
    payload = {
        "title": "HubSpot Pricing Pressure Is Changing Buyer Shortlists",
        "slug": "hubspot-pricing-pressure",
        "description": "How renewal pressure changes buyer shortlists.",
        "seo_title": "HubSpot Pricing Pressure",
        "seo_description": "A data-backed look at HubSpot pricing pressure.",
        "target_keyword": "hubspot pricing pressure",
        "secondary_keywords": ["hubspot alternatives", "crm renewal pricing"],
        "topic_type": "vendor_alternative",
        "content": _valid_content(),
        "charts": [{"chart_id": "pricing", "title": "Pricing"}],
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


def _service(
    *,
    rows=None,
    responses=None,
    prompts=None,
    config=None,
    reasoning_context=None,
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
async def test_generate_sends_blueprint_in_user_message_when_template_has_no_placeholder() -> None:
    service, _blueprints, _blog_posts, llm, _skills = _service(
        prompts={"digest/blog_post_generation": "Write a post."}
    )

    await service.generate(scope=TenantScope(), target_mode="vendor_retention", limit=1)

    assert "blueprint JSON" in llm.calls[0]["messages"][1].content
    assert "HubSpot pricing pressure" in llm.calls[0]["messages"][1].content


@pytest.mark.asyncio
async def test_generate_blocks_low_quality_posts_without_saving() -> None:
    service, _blueprints, blog_posts, _llm, _skills = _service(
        responses=[_valid_blog_json(content="Too short.")],
        config=BlogPostGenerationConfig(
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
async def test_generate_routes_missing_content_to_quality_blocked_not_unparseable() -> None:
    payload = json.loads(_valid_blog_json())
    payload.pop("content")
    service, _blueprints, blog_posts, _llm, _skills = _service(
        responses=[json.dumps(payload)],
        config=BlogPostGenerationConfig(
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
        config=BlogPostGenerationConfig(temperature=0.7, max_tokens=999),
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
# PR-Blog-Topic-Per-Call: per-call topic kwarg substitutes into the prompt
# -----------------------


@pytest.mark.asyncio
async def test_generate_per_call_topic_substitutes_into_prompt():
    """Operator-supplied topic reaches the system prompt via the
    ``{topic}`` placeholder."""

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
    assert "Focus: Renewal pricing pressure on mid-market SaaS" in system_prompt


@pytest.mark.asyncio
async def test_generate_no_topic_resolves_placeholder_to_empty_string():
    """No-topic case: the ``{topic}`` placeholder still gets substituted
    (to ``""``), keeping the prompt structurally clean."""

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
    # Placeholder substituted with empty string -- "Focus: \n\n..." remains.
    assert "{topic}" not in system_prompt
    assert "Focus: \n" in system_prompt or "Focus:\n" in system_prompt


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
    assert "Some topic" not in system_prompt  # no placeholder, no substitution
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

    # LLM saw the merged blueprint JSON.
    system_prompt = llm.calls[0]["messages"][0].content
    assert "reasoning_context" in system_prompt
    assert "campaign_reasoning_context" in system_prompt
    assert "Renewal pricing rose 22 percent" in system_prompt
    assert result.as_dict()["reasoning_contexts_used"] == 1

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
