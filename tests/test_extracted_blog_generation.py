from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.blog_generation import (
    BlogPostGenerationConfig,
    BlogPostGenerationService,
    parse_blog_post_response,
)
from extracted_content_pipeline.blog_ports import BlogPostDraft
from extracted_content_pipeline.campaign_ports import LLMResponse, TenantScope
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


def _service(*, rows=None, responses=None, prompts=None, config=None):
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
    assert parse_blog_post_response("") is None
    assert parse_blog_post_response('{"title": "No content"}') is None
    assert parse_blog_post_response('{"content": "No title"}') is None


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
