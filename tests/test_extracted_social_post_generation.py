from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import LLMResponse, TenantScope
from extracted_content_pipeline.social_post_generation import (
    SocialPostGenerationConfig,
    SocialPostGenerationService,
    parse_social_post_response,
)
from extracted_content_pipeline.social_post_ports import SocialPostDraft


class _LLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        self.calls.append({
            "messages": tuple(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": dict(metadata or {}),
        })
        content = self.responses.pop(0) if self.responses else "{}"
        return LLMResponse(
            content=content,
            model="test-model",
            usage={"input_tokens": 11, "output_tokens": 7},
        )


class _Skills:
    def __init__(self, prompt: str | None = "Social prompt\n{brand_voice}") -> None:
        self.prompt = prompt
        self.calls: list[str] = []

    def get_prompt(self, name: str) -> str | None:
        self.calls.append(name)
        return self.prompt


class _SocialPostRepository:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def save_drafts(
        self,
        drafts: list[SocialPostDraft] | tuple[SocialPostDraft, ...],
        *,
        scope: TenantScope,
    ) -> tuple[str, ...]:
        self.calls.append({"drafts": tuple(drafts), "scope": scope})
        return ("social-post-db-id-1",)


@pytest.mark.asyncio
async def test_social_post_service_generates_evidence_backed_posts() -> None:
    service = SocialPostGenerationService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-1",
                "reviewer_company": "Acme Logistics",
                "vendor": "HubSpot",
                "review_text": "Pricing became hard to justify after renewal.",
                "pain_category": "pricing pressure",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    assert payload["target_mode"] == "vendor_retention"
    assert payload["warnings"] == []
    assert payload["posts"] == [
        {
            "id": "review-1",
            "channel": "linkedin",
            "text": (
                "Customer evidence for HubSpot flags pricing pressure. "
                "Source note: \"Pricing became hard to justify after renewal.\" "
                "Use this proof point to sharpen the next landing page, blog "
                "post, or sales brief."
            ),
            "source_id": "review-1",
            "source_type": "review",
            "target_id": "review-1",
            "company_name": "Acme Logistics",
            "vendor_name": "HubSpot",
            "pain_points": ["pricing pressure"],
        }
    ]
    assert payload["saved_ids"] == []


@pytest.mark.asyncio
async def test_social_post_service_persists_generated_drafts_when_repository_is_configured() -> None:
    repository = _SocialPostRepository()
    service = SocialPostGenerationService(social_posts=repository)

    result = await service.generate(
        scope=TenantScope(account_id="acct-1", user_id="user-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-1",
                "reviewer_company": "Acme Logistics",
                "vendor": "HubSpot",
                "review_text": "Pricing became hard to justify after renewal.",
                "pain_category": "pricing pressure",
            }
        ],
    )

    assert result.saved_ids == ("social-post-db-id-1",)
    assert result.as_dict()["saved_ids"] == ["social-post-db-id-1"]
    assert len(repository.calls) == 1
    call = repository.calls[0]
    assert call["scope"] == TenantScope(account_id="acct-1", user_id="user-1")
    drafts = call["drafts"]
    assert isinstance(drafts, tuple)
    draft = drafts[0]
    assert draft.target_id == "review-1"
    assert draft.target_mode == "vendor_retention"
    assert draft.channel == "linkedin"
    assert draft.source_id == "review-1"
    assert draft.source_type == "review"
    assert draft.company_name == "Acme Logistics"
    assert draft.vendor_name == "HubSpot"
    assert draft.pain_points == ("pricing pressure",)
    assert draft.metadata["source_post"]["id"] == "review-1"


@pytest.mark.asyncio
async def test_social_post_service_skips_unusable_rows_with_warnings() -> None:
    service = SocialPostGenerationService()

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=["not a row", {"review_id": "missing-text"}],
    )

    payload = result.as_dict()
    assert payload["generated"] == 0
    assert [warning["code"] for warning in payload["warnings"]] == [
        "row_not_object",
        "missing_source_text",
        "missing_social_post_evidence",
    ]


@pytest.mark.asyncio
async def test_social_post_service_keeps_truncated_posts_within_configured_bound() -> None:
    config = SocialPostGenerationConfig(max_post_chars=120)
    service = SocialPostGenerationService(config=config)

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-long",
                "vendor": "HubSpot",
                "review_text": " ".join(["renewal pressure"] * 80),
                "pain_category": "pricing pressure",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    assert len(payload["posts"][0]["text"]) <= config.max_post_chars
    assert payload["posts"][0]["text"].endswith("...")


@pytest.mark.asyncio
async def test_social_post_service_applies_limit_to_usable_rows() -> None:
    service = SocialPostGenerationService()

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=2,
        source_material=[
            {
                "review_id": "review-1",
                "vendor": "HubSpot",
                "review_text": "Pricing became hard to justify.",
                "pain_category": "pricing pressure",
            },
            {
                "review_id": "review-2",
                "vendor": "Zendesk",
                "review_text": "Support queues took too long to triage.",
                "pain_category": "slow support",
            },
            {
                "review_id": "review-3",
                "vendor": "Intercom",
                "review_text": "Reporting did not explain where leads came from.",
                "pain_category": "unclear reporting",
            },
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 2
    assert [post["source_id"] for post in payload["posts"]] == ["review-1", "review-2"]


def test_parse_social_post_response_accepts_fenced_json() -> None:
    assert parse_social_post_response(
        '```json\n{"channel":"linkedin","text":"Use the proof point."}\n```'
    ) == {"channel": "linkedin", "text": "Use the proof point."}
    assert parse_social_post_response('{"channel":"linkedin"}') is None


@pytest.mark.asyncio
async def test_social_post_service_rewrites_with_brand_voice_and_persists_metadata() -> None:
    repository = _SocialPostRepository()
    llm = _LLM(
        '{"channel":"linkedin","text":"You can use this renewal proof without guessing."}'
    )
    skills = _Skills()
    service = SocialPostGenerationService(
        social_posts=repository,
        llm=llm,
        skills=skills,
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1", user_id="user-1"),
        target_mode="vendor_retention",
        brand_voice={
            "id": "voice-1",
            "account_id": "acct-1",
            "name": "Plain voice",
            "descriptors": ["plainspoken"],
            "preferred_pov": "second_person",
        },
        source_material=[
            {
                "review_id": "review-1",
                "reviewer_company": "Acme Logistics",
                "vendor": "HubSpot",
                "review_text": "Pricing became hard to justify after renewal.",
                "pain_category": "pricing pressure",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    assert payload["posts"][0]["text"] == (
        "You can use this renewal proof without guessing."
    )
    assert payload["posts"][0]["_brand_voice_profile"]["id"] == "voice-1"
    assert payload["posts"][0]["_brand_voice_audit"]["passed"] is True
    assert payload["saved_ids"] == ["social-post-db-id-1"]
    assert skills.calls == ["digest/social_post_generation"]
    llm_call = llm.calls[0]
    assert llm_call["metadata"] == {
        "asset_type": "social_post",
        "target_mode": "vendor_retention",
        "source_id": "review-1",
    }
    system_prompt = llm_call["messages"][0].content
    assert "## Brand voice" in system_prompt
    assert "plainspoken" in system_prompt
    user_prompt = llm_call["messages"][1].content
    assert '"source_id":"review-1"' in user_prompt
    draft = repository.calls[0]["drafts"][0]
    assert draft.text == "You can use this renewal proof without guessing."
    assert draft.metadata["brand_voice_profile"]["id"] == "voice-1"
    assert draft.metadata["brand_voice_audit"]["passed"] is True
    assert draft.metadata["generation_usage"]["input_tokens"] == 11


@pytest.mark.asyncio
async def test_social_post_service_retries_unparseable_llm_response() -> None:
    llm = _LLM(
        "not json",
        '{"channel":"linkedin","text":"You can turn the review into a grounded post."}',
    )
    service = SocialPostGenerationService(llm=llm, skills=_Skills())

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        brand_voice={
            "id": "voice-1",
            "account_id": "acct-1",
            "descriptors": ["direct"],
        },
        source_material=[
            {
                "review_id": "review-1",
                "vendor": "HubSpot",
                "review_text": "Pricing became hard to justify after renewal.",
            }
        ],
    )

    assert result.generated == 1
    assert len(llm.calls) == 2
    retry_prompt = llm.calls[1]["messages"][1].content
    assert "Previous response excerpt:\nnot json" in retry_prompt


@pytest.mark.asyncio
async def test_social_post_service_fails_closed_when_brand_voice_has_no_llm() -> None:
    service = SocialPostGenerationService()

    with pytest.raises(ValueError, match="requires configured LLM and skill store"):
        await service.generate(
            scope=TenantScope(account_id="acct-1"),
            target_mode="vendor_retention",
            brand_voice={
                "id": "voice-1",
                "account_id": "acct-1",
                "descriptors": ["direct"],
            },
            source_material=[
                {
                    "review_id": "review-1",
                    "vendor": "HubSpot",
                    "review_text": "Pricing became hard to justify after renewal.",
                }
            ],
        )


@pytest.mark.asyncio
async def test_social_post_service_fails_closed_when_prompt_is_missing() -> None:
    service = SocialPostGenerationService(llm=_LLM("{}"), skills=_Skills(prompt=None))

    with pytest.raises(ValueError, match="Social post generation skill not found"):
        await service.generate(
            scope=TenantScope(account_id="acct-1"),
            target_mode="vendor_retention",
            brand_voice={
                "id": "voice-1",
                "account_id": "acct-1",
                "descriptors": ["direct"],
            },
            source_material=[
                {
                    "review_id": "review-1",
                    "vendor": "HubSpot",
                    "review_text": "Pricing became hard to justify after renewal.",
                }
            ],
        )
