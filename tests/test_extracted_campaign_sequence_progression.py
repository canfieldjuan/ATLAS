from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from extracted_content_pipeline.campaign_ports import LLMResponse
from extracted_content_pipeline.campaign_sequence_context import SequenceContextLimits
from extracted_content_pipeline.campaign_sequence_progression import (
    CampaignSequenceProgressionConfig,
    CampaignSequenceProgressionService,
    build_engagement_summary,
    build_previous_emails,
    parse_generated_sequence_step,
    sequence_skill_name,
    target_mode_for_recipient_type,
)


class _Clock:
    def __init__(self):
        self.value = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    def now(self):
        return self.value


class _SequenceRepo:
    def __init__(self, due=None, previous=None):
        self.due = due or []
        self.previous = previous or []
        self.list_due_calls = []
        self.previous_calls = []
        self.queued = []
        self.marked = []

    async def list_due_sequences(self, *, limit, now):
        self.list_due_calls.append({"limit": limit, "now": now})
        return self.due

    async def list_previous_campaigns(self, *, sequence_id, limit):
        self.previous_calls.append({"sequence_id": sequence_id, "limit": limit})
        return self.previous

    async def queue_sequence_step(self, *, sequence, content, from_email, queued_at):
        campaign_id = f"campaign-{len(self.queued) + 1}"
        self.queued.append({
            "sequence": sequence,
            "content": content,
            "from_email": from_email,
            "queued_at": queued_at,
            "campaign_id": campaign_id,
        })
        return campaign_id

    async def mark_sequence_step(self, *, sequence_id, current_step, updated_at):
        self.marked.append({
            "sequence_id": sequence_id,
            "current_step": current_step,
            "updated_at": updated_at,
        })


class _Skills:
    def __init__(self, prompts):
        self.prompts = prompts
        self.calls = []

    def get_prompt(self, name):
        self.calls.append(name)
        return self.prompts.get(name)


class _LLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        self.calls.append({
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": metadata,
        })
        return LLMResponse(content=self.content, model="test-model")


class _Audit:
    def __init__(self):
        self.events = []

    async def record(self, event_type, *, campaign_id=None, sequence_id=None, metadata=None):
        self.events.append({
            "event_type": event_type,
            "campaign_id": campaign_id,
            "sequence_id": sequence_id,
            "metadata": metadata,
        })


def _sequence(**overrides):
    data = {
        "id": "sequence-1",
        "company_name": "Acme",
        "recipient_email": "buyer@example.com",
        "current_step": 1,
        "max_steps": 4,
        "open_count": 2,
        "click_count": 0,
        "last_sent_at": datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc),
        "company_context": {
            "recipient_type": "vendor_retention",
            "category": "crm",
            "selling": {"product_name": "Atlas"},
            "reasoning_anchor_examples": {"pricing": ["hidden from prompt"]},
        },
        "selling_context": {"value_prop": "research-backed outreach"},
    }
    data.update(overrides)
    return data


def _previous(**overrides):
    data = {
        "step_number": 1,
        "subject": "First note",
        "body": "<p>Hello <b>there</b>.</p>",
        "status": "sent",
        "opened_at": datetime(2026, 4, 30, tzinfo=timezone.utc),
        "clicked_at": None,
    }
    data.update(overrides)
    return data


def _service(seq, *, llm_content=None, skills=None, previous=None, config=None):
    repo = _SequenceRepo(due=[seq], previous=previous or [_previous()])
    audit = _Audit()
    clock = _Clock()
    service = CampaignSequenceProgressionService(
        sequences=repo,
        llm=_LLM(
            llm_content
            or json.dumps({
                "subject": "Following up",
                "body": "<p>Second note</p>",
                "cta": "Book time",
                "angle_reasoning": "Opened prior note",
            })
        ),
        skills=skills
        or _Skills({
            "digest/b2b_vendor_sequence": (
                "Company {company_name}; step {current_step}/{max_steps}; "
                "days {days_since_last}; product {product_name}; "
                "ctx {company_context}; sell {selling_context}; "
                "eng {engagement_summary}; prev {previous_emails}"
            )
        }),
        audit=audit,
        clock=clock,
        config=config
        or CampaignSequenceProgressionConfig(
            batch_limit=3,
            from_email="seller@example.com",
            onboarding_product_name="Fallback Product",
            context_limits=SequenceContextLimits(
                prompt_max_tokens=123,
                prompt_email_body_preview_chars=40,
            ),
        ),
    )
    return service, repo, audit, clock


def test_engagement_summary_includes_counts_reply_and_step_breakdown():
    summary = build_engagement_summary(
        {
            "open_count": 1,
            "click_count": 2,
            "reply_received_at": datetime.now(timezone.utc),
            "reply_intent": "interested",
            "reply_summary": "Asked for pricing.",
        },
        [_previous(clicked_at=datetime.now(timezone.utc))],
    )

    assert "Opened 1 time(s)" in summary
    assert "Clicked 2 time(s)" in summary
    assert "Reply received (interested): Asked for pricing." in summary
    assert "- Step 1: Opened, Clicked" in summary


def test_previous_emails_formats_preview_and_engagement():
    rendered = build_previous_emails([
        _previous(body="<p>Hello&nbsp;buyer</p>", clicked_at=None),
        _previous(step_number=2, subject="", body="", opened_at=None),
    ])

    assert "--- Step 1 (status: sent) ---" in rendered
    assert "Subject: First note" in rendered
    assert "Preview: Hello buyer" in rendered
    assert "Engagement: Opened" in rendered
    assert "Subject: (no subject)" in rendered
    assert "Engagement: No opens or clicks recorded" in rendered


def test_skill_and_target_mode_mapping():
    assert sequence_skill_name("onboarding") == "digest/b2b_onboarding_sequence"
    assert sequence_skill_name("amazon_seller") == "digest/amazon_seller_campaign_sequence"
    assert sequence_skill_name("vendor_retention") == "digest/b2b_vendor_sequence"
    assert sequence_skill_name("challenger_intel") == "digest/b2b_challenger_sequence"
    assert sequence_skill_name(None) == "digest/b2b_campaign_sequence"

    assert target_mode_for_recipient_type("amazon_seller") == "amazon_seller"
    assert target_mode_for_recipient_type("vendor_retention") == "vendor_retention"
    assert target_mode_for_recipient_type("challenger_intel") == "challenger_intel"
    assert target_mode_for_recipient_type(None) == "churning_company"


def test_parse_generated_sequence_step_accepts_fenced_json_and_think_tags():
    parsed = parse_generated_sequence_step(
        '<think>notes</think>\n```json\n{"subject":"Hi","body":"Body"}\n```'
    )

    assert parsed == {"subject": "Hi", "body": "Body"}


def test_parse_generated_sequence_step_finds_json_inside_prose():
    parsed = parse_generated_sequence_step(
        'Sure, here it is: {"subject":"Hi","body":"Body"} Thanks.'
    )

    assert parsed == {"subject": "Hi", "body": "Body"}


@pytest.mark.asyncio
async def test_generate_next_step_uses_skill_prompt_context_and_llm_budget():
    service, repo, _, _ = _service(_sequence())

    content = await service.generate_next_step(repo.due[0], repo.previous)

    assert content["subject"] == "Following up"
    assert content["_recipient_type"] == "vendor_retention"
    llm_call = service._llm.calls[0]
    system_prompt = llm_call["messages"][0].content
    assert "Company Acme" in system_prompt
    assert "step 2/4" in system_prompt
    assert "days 2" in system_prompt
    assert "product Fallback Product" in system_prompt
    assert "hidden from prompt" not in system_prompt
    assert llm_call["max_tokens"] == 123
    assert llm_call["temperature"] == 0.7


@pytest.mark.asyncio
async def test_generate_next_step_uses_amazon_seller_placeholders():
    seq = _sequence(
        company_context={
            "recipient_type": "amazon_seller",
            "seller_name": "Acme Seller",
            "category": "supplements",
            "category_intelligence": {"top_pain_points": [{"pain": "taste"}]},
        }
    )
    service, repo, _, _ = _service(
        seq,
        skills=_Skills({
            "digest/amazon_seller_campaign_sequence": (
                "{recipient_name}|{recipient_company}|{recipient_type}|"
                "{category}|{category_intelligence}"
            )
        }),
    )

    content = await service.generate_next_step(repo.due[0], repo.previous)

    assert content["_recipient_type"] == "amazon_seller"
    assert content["_category"] == "supplements"
    prompt = service._llm.calls[0]["messages"][0].content
    assert prompt.startswith("Acme Seller|Acme Seller|amazon_seller|supplements|")
    assert "taste" in prompt


@pytest.mark.asyncio
async def test_progress_due_queues_followup_and_marks_sequence():
    service, repo, audit, clock = _service(_sequence())

    result = await service.progress_due()

    assert result.as_dict() == {
        "due_sequences": 1,
        "progressed": 1,
        "skipped": 0,
        "disabled": False,
    }
    assert repo.list_due_calls == [{"limit": 3, "now": clock.value}]
    assert repo.previous_calls == [{"sequence_id": "sequence-1", "limit": 4}]
    queued = repo.queued[0]
    assert queued["from_email"] == "seller@example.com"
    assert queued["queued_at"] == clock.value
    assert queued["content"]["step_number"] == 2
    assert queued["content"]["target_mode"] == "vendor_retention"
    assert repo.marked == [{
        "sequence_id": "sequence-1",
        "current_step": 2,
        "updated_at": clock.value,
    }]
    assert [event["event_type"] for event in audit.events] == ["generated", "queued"]
    assert audit.events[0]["campaign_id"] == "campaign-1"
    assert audit.events[0]["metadata"]["subject"] == "Following up"


@pytest.mark.asyncio
async def test_progress_due_skips_when_skill_missing():
    service, repo, audit, _ = _service(_sequence(), skills=_Skills({}))

    result = await service.progress_due()

    assert result.progressed == 0
    assert result.skipped == 1
    assert repo.queued == []
    assert repo.marked == []
    assert audit.events == []


@pytest.mark.asyncio
async def test_progress_due_skips_when_llm_returns_unparseable_content():
    service, repo, audit, _ = _service(_sequence(), llm_content="not json")

    result = await service.progress_due()

    assert result.progressed == 0
    assert result.skipped == 1
    assert repo.queued == []
    assert repo.marked == []
    assert audit.events == []


@pytest.mark.asyncio
async def test_progress_due_returns_disabled_result_without_repo_touch():
    service, repo, _, _ = _service(
        _sequence(),
        config=CampaignSequenceProgressionConfig(enabled=False),
    )

    result = await service.progress_due()

    assert result.as_dict() == {
        "due_sequences": 0,
        "progressed": 0,
        "skipped": 0,
        "disabled": True,
    }
    assert repo.list_due_calls == []
