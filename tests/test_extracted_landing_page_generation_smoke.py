from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import pytest

from extracted_content_pipeline.campaign_ports import LLMResponse, TenantScope
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)
from extracted_content_pipeline.landing_page_export import (
    landing_page_draft_export_row,
    public_landing_page_robots,
)
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationService,
)
from extracted_content_pipeline.landing_page_ports import LandingPageDraft


class _LandingPageStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    async def save_drafts(self, drafts, *, scope):
        self.saved.append({"drafts": list(drafts), "scope": scope})
        return [f"lp-smoke-{index + 1}" for index, _draft in enumerate(drafts)]

    async def list_drafts(self, **_kwargs):  # pragma: no cover - not used by smoke
        raise AssertionError("list_drafts should not be called")

    async def get_draft(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("get_draft should not be called")

    async def update_draft(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("update_draft should not be called")

    async def get_public_approved_draft(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("get_public_approved_draft should not be called")

    async def list_public_sitemap_candidates(self, **_kwargs):  # pragma: no cover
        raise AssertionError("list_public_sitemap_candidates should not be called")

    async def update_status(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("update_status should not be called")

    async def update_statuses(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("update_statuses should not be called")


class _LLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        self.calls.append({
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": dict(metadata or {}),
        })
        return LLMResponse(
            content=self.responses.pop(0),
            model="smoke-llm",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


class _Skills:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_prompt(self, name: str) -> str:
        self.calls.append(name)
        return "Generate from this campaign JSON: {campaign_json}"


def _landing_page_response(*, include_description: bool) -> str:
    payload: dict[str, Any] = {
        "title": "FAQ Report for Small SaaS Support Teams",
        "slug": "faq-report-small-saas-support-teams",
        "hero": {
            "headline": "Turn repeat support tickets into answers customers can find",
            "subheadline": (
                "FAQ Report helps 10-50 person SaaS teams turn the last 90 days "
                "of support tickets into clearer answers before customers email, "
                "complain, or cancel."
            ),
            "cta_label": "Upload Ticket CSV -- Free Analysis",
            "cta_url": "/systems/ai-content-ops/intake",
        },
        "sections": [
            {
                "id": "problem",
                "title": "Support Problems Become Cancellations",
                "body_markdown": (
                    "Repeat support tickets show where customers get stuck "
                    "before frustration turns into cancellation. Small SaaS "
                    "support teams lose time when customers ask the same setup, "
                    "billing, and account questions again and again."
                ),
                "metadata": {
                    "order": 1,
                    "kind": "problem",
                    "primary_question": (
                        "Why do repeat support tickets put retention at risk?"
                    ),
                    "answer_summary": (
                        "Repeat support tickets show where customers get stuck "
                        "before frustration turns into cancellation."
                    ),
                },
            },
            {
                "id": "solution",
                "title": "How FAQ Report Helps",
                "body_markdown": (
                    "FAQ Report clusters support tickets, ranks repeat questions, "
                    "and turns customer wording into answers. Your team can review "
                    "and publish those help-center answers after the report is ready."
                ),
                "metadata": {
                    "order": 2,
                    "kind": "solution",
                    "primary_question": "How does FAQ Report turn tickets into answers?",
                    "answer_summary": (
                        "FAQ Report clusters support tickets, ranks repeat "
                        "questions, and turns customer wording into answers."
                    ),
                },
            },
            {
                "id": "questions",
                "title": "Questions Before You Upload",
                "body_markdown": (
                    "No. Your team reviews, edits, and publishes the FAQ answers "
                    "when they are ready. Upload a CSV from your support platform, "
                    "review the FAQ entries, edit the wording, and publish the "
                    "answers when they are ready."
                ),
                "metadata": {
                    "order": 3,
                    "kind": "faq",
                    "primary_question": "Does FAQ Report publish automatically?",
                    "answer_summary": (
                        "No. Your team reviews, edits, and publishes the FAQ "
                        "answers when they are ready."
                    ),
                },
            },
        ],
        "cta": {
            "label": "Upload Ticket CSV -- Free Analysis",
            "url": "/systems/ai-content-ops/intake",
            "variant": "primary",
        },
        "meta": {
            "title_tag": "FAQ Report for Small SaaS Support Teams",
        },
        "reference_ids": ["support-ticket-csv-90-days"],
    }
    if include_description:
        payload["meta"]["description"] = (
            "FAQ Report turns your last 90 days of support tickets into clear "
            "help-center answers for small SaaS teams."
        )
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_landing_page_generation_smoke_repairs_and_exports_ready_page() -> None:
    store = _LandingPageStore()
    llm = _LLM([
        _landing_page_response(include_description=False),
        _landing_page_response(include_description=True),
    ])
    service = LandingPageGenerationService(
        landing_pages=store,
        llm=llm,
        skills=_Skills(),
    )

    result = await execute_content_ops_from_mapping(
        {
            "outputs": ["landing_page"],
            "inputs": {
                "campaign_name": "FAQ Report",
                "offer": (
                    "Turn repeat support tickets into customer-ready FAQ answers"
                ),
                "audience": "10-50 person SaaS support team",
                "target_keyword": "support ticket FAQ",
                "secondary_keywords": [
                    "reduce repeat support tickets",
                    "help center answers",
                ],
                "search_intent": (
                    "Find a low-friction way to turn old support tickets into "
                    "answers customers can use."
                ),
                "primary_entity": "FAQ Report",
                "audience_entity": "small SaaS support team",
                "objections": ["Will this publish automatically?"],
                "faq_questions": ["What happens after I upload the CSV?"],
                "source_period": "Last 90 days of support tickets",
                "internal_links": ["/systems/ai-content-ops/intake"],
                "cta_label": "Upload Ticket CSV -- Free Analysis",
                "cta_url": "/systems/ai-content-ops/intake",
                "unrelated_input": "must not reach the landing page prompt",
            },
        },
        services=ContentOpsExecutionServices(landing_page=service),
        scope=TenantScope(account_id="acct-smoke"),
    )

    assert result["status"] == "completed"
    step = result["steps"][0]
    assert step["output"] == "landing_page"
    assert step["status"] == "completed"
    assert step["result"]["generated"] == 1
    assert step["result"]["saved_ids"] == ["lp-smoke-1"]
    assert len(llm.calls) == 2
    assert llm.calls[1]["metadata"]["quality_repair_attempt_no"] == 1
    assert "seo_aeo_readiness:meta_description" in (
        step["result"]["quality_repair_history"][0]["blockers"]
    )

    system_prompt = llm.calls[0]["messages"][0].content
    assert '"target_keyword":"support ticket FAQ"' in system_prompt
    assert '"primary_entity":"FAQ Report"' in system_prompt
    assert '"cta_url":"/systems/ai-content-ops/intake"' in system_prompt
    assert "unrelated_input" not in system_prompt

    assert len(store.saved) == 1
    saved_draft = store.saved[0]["drafts"][0]
    assert isinstance(saved_draft, LandingPageDraft)
    assert saved_draft.metadata["generation_quality_repair_attempts"] == 1
    assert saved_draft.metadata["generation_parse_attempts"] == 2
    assert saved_draft.meta["description"].startswith("FAQ Report turns")

    export_row = landing_page_draft_export_row(saved_draft)
    assert export_row["seo_aeo_readiness"]["status"] == "ready"
    assert export_row["geo_readiness"]["status"] == "ready"
    assert export_row["structured_data"]["@graph"][0]["@type"] == "WebPage"
    assert export_row["structured_data"]["@graph"][1]["@type"] == "FAQPage"

    approved = replace(saved_draft, id="lp-smoke-1", status="approved")
    assert public_landing_page_robots(approved) == "index,follow"
    placeholder_cta = replace(approved, cta={**approved.cta, "url": "/demo?utm=1"})
    assert public_landing_page_robots(placeholder_cta) == "noindex,follow"
