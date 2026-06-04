from __future__ import annotations

import json
import logging
from typing import cast

import pytest

pytest.importorskip("fastapi")

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from extracted_content_pipeline.api.generated_assets import (
    GeneratedAssetApiConfig,
    create_generated_asset_router,
    create_public_landing_page_router,
)
import extracted_content_pipeline.api.generated_assets as asset_api
from extracted_content_pipeline.campaign_ports import LLMResponse, TenantScope
from extracted_content_pipeline.faq_macro_writeback import (
    MacroPublishResult,
    MacroPublishStatus,
)


BATCH_REPORT_ID_1 = "11111111-1111-1111-1111-111111111111"
BATCH_REPORT_ID_2 = "22222222-2222-2222-2222-222222222222"
BATCH_REPORT_ID_MISSING = "33333333-3333-3333-3333-333333333333"


class _Pool:
    def __init__(
        self,
        rows=None,
        *,
        execute_result: str = "UPDATE 1",
        initialized: bool = True,
    ) -> None:
        self.rows = list(rows or [])
        self.execute_result = execute_result
        self.is_initialized = initialized
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        return self.execute_result


class _FAQMacroPublishHistoryPool(_Pool):
    def __init__(self, *, draft_row=None, attempt_rows=None) -> None:
        super().__init__(rows=[])
        self.draft_row = dict(draft_row) if draft_row is not None else None
        self.attempt_rows = list(attempt_rows or [])

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        if "FROM ticket_faq_markdown" in str(query):
            return [self.draft_row] if self.draft_row is not None else []
        if "FROM ticket_faq_macro_publish_attempts" in str(query):
            return self.attempt_rows
        return []


class _PublicLandingPagePool(_Pool):
    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        landing_page_id = args[0] if args else None
        if "status = 'approved'" not in str(query):
            return self.rows
        return [
            row for row in self.rows
            if row.get("id") == landing_page_id and row.get("status") == "approved"
        ]


class _EditableLandingPagePool(_Pool):
    def __init__(self, row=None) -> None:
        super().__init__(rows=[])
        self.row = dict(row) if row is not None else None

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        if "UPDATE landing_pages" not in str(query):
            if (
                self.row is not None
                and "account_id" in self.row
                and self.row["account_id"] != args[1]
            ):
                return []
            return [] if self.row is None else [self.row]
        if "jsonb_set" in str(query):
            if (
                self.row is None
                or self.row.get("status") == "approved"
                or not getattr(self, "lock_acquired", True)
                or (
                    "account_id" in self.row
                    and self.row["account_id"] != args[1]
                )
            ):
                return []
            metadata = dict(self.row.get("metadata") or {})
            metadata[asset_api.LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY] = json.loads(
                args[2]
            )
            self.row["metadata"] = metadata
            return [self.row]
        if (
            self.row is None
            or self.row.get("status") == "approved"
            or (
                "account_id" in self.row
                and self.row["account_id"] != args[1]
            )
        ):
            return []
        repair_claim_token = args[10] if len(args) > 10 else None
        if repair_claim_token is not None:
            metadata = dict(self.row.get("metadata") or {})
            claim = metadata.get(asset_api.LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY)
            if not isinstance(claim, dict) or claim.get("token") != repair_claim_token:
                return []
        self.row.update({
            "id": args[0],
            "title": args[2],
            "slug": args[3],
            "hero": json.loads(args[4]),
            "sections": json.loads(args[5]),
            "cta": json.loads(args[6]),
            "meta": json.loads(args[7]),
            "reference_ids": json.loads(args[8]),
            "metadata": json.loads(args[9]),
            "status": "draft",
        })
        return [self.row]

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        if asset_api.LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY in str(query):
            if getattr(self, "force_release_miss", False):
                return "UPDATE 0"
            metadata = dict((self.row or {}).get("metadata") or {})
            claim = metadata.get(asset_api.LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY)
            if isinstance(claim, dict) and claim.get("token") == args[2]:
                metadata.pop(asset_api.LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY, None)
                if self.row is not None:
                    self.row["metadata"] = metadata
                return "UPDATE 1"
            return "UPDATE 0"
        return self.execute_result


class _LockingEditableLandingPagePool(_EditableLandingPagePool):
    def __init__(self, row=None, *, lock_acquired: bool = True) -> None:
        super().__init__(row)
        self.lock_acquired = lock_acquired
        self.lock_calls: list[tuple[str, tuple[object, ...]]] = []
        self.acquire_calls = 0
        self.release_calls = 0
        self.released_connections = []
        self.unlocked = False

    async def acquire(self):
        self.acquire_calls += 1
        raise AssertionError("repair should not hold a checked-out lock connection")

    async def release(self, conn):
        self.release_calls += 1
        self.released_connections.append(conn)


class _LLM:
    def __init__(self, responses) -> None:
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
        return LLMResponse(
            content=response,
            model="test-model",
            usage={"input_tokens": 9, "output_tokens": 4},
        )


class _Skills:
    def __init__(self, prompt: str = "TEMPLATE: {campaign_json}") -> None:
        self.prompt = prompt
        self.calls = []

    def get_prompt(self, name):
        self.calls.append(name)
        return self.prompt


class _MacroProvider:
    def __init__(self, *, status: str = "published", error: str = "") -> None:
        self.status = status
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def publish(self, macros, *, scope: TenantScope):
        captured = tuple(macros)
        self.calls.append({"macros": captured, "scope": scope})
        return tuple(
            MacroPublishResult(
                macro=macro,
                status=cast(MacroPublishStatus, self.status),
                external_id=f"external-{index}" if not self.error else "",
                error=self.error,
            )
            for index, macro in enumerate(captured, start=1)
        )


def _report_row():
    return {
        "id": "report-uuid-1",
        "status": "draft",
        "target_id": "vendor-acme",
        "target_mode": "vendor_retention",
        "report_type": "vendor_pressure",
        "title": "Acme report",
        "summary": "Pricing pressure dominates.",
        "sections": [{"id": "summary", "title": "Summary", "body_markdown": "Body"}],
        "reference_ids": ["r1"],
        "metadata": {
            "generation_usage": {"input_tokens": 10, "output_tokens": 5},
            "reasoning_context": {"wedge": "price_squeeze", "confidence": "high"},
        },
    }


def _blog_post_row():
    return {
        "id": "blog-post-uuid-1",
        "status": "draft",
        "slug": "acme-pricing-pressure",
        "title": "Acme Pricing Pressure",
        "description": "Pricing pressure dominates.",
        "topic_type": "vendor_alternative",
        "tags": ["pricing"],
        "content": (
            "## Why is Acme pricing pressure showing up?\n\n"
            "Acme pricing pressure is visible in the last 90 days of review "
            "patterns, especially across 214 reviews where buyers describe "
            "renewal friction, budget concerns, and comparison shopping. The "
            "answer is that Acme buyers are not only comparing features; they "
            "are checking whether the contract still fits the budget before "
            "another renewal cycle starts.\n\n"
            "## How should teams read the Acme pricing evidence?\n\n"
            "Acme pricing evidence should be read as a renewal-risk signal, not "
            "as proof that every buyer has the same problem. The useful pattern "
            "is that customers keep using similar wording about budget pressure, "
            "contract terms, and alternative comparisons when they explain why "
            "pricing has become harder to justify."
        ),
        "charts": [],
        "data_context": {
            "vendor": "Acme",
            "review_period": "last 90 days",
            "_metadata": {
                "generation_usage": {"input_tokens": 9, "output_tokens": 4},
                "reasoning_context": {"wedge": "price_squeeze", "confidence": "high"},
            }
        },
        "llm_model": "fake-llm",
        "seo_title": "Acme Pricing Pressure 2026",
        "seo_description": "Acme pricing pressure from recent review data.",
        "target_keyword": "acme pricing pressure",
        "secondary_keywords": ["acme pricing"],
        "faq": [
            {"question": "Why is Acme pricing a concern?", "answer": "Pricing appears in reviews."},
            {"question": "Who should compare alternatives?", "answer": "Teams under budget pressure."},
            {"question": "What should buyers check?", "answer": "Renewal and support terms."},
        ],
    }


def _landing_page_row():
    return {
        "id": "landing-page-uuid-1",
        "status": "draft",
        "campaign_name": "acme-launch",
        "persona": "VP Engineering",
        "value_prop": "Catch pressure early",
        "title": "Acme landing page",
        "slug": "acme-launch",
        "hero": {"headline": "Stop surprises"},
        "sections": [{"id": "problem", "title": "Problem", "body_markdown": "Body"}],
        "cta": {"label": "Book a demo"},
        "meta": {"title_tag": "Acme landing page"},
        "reference_ids": ["r1"],
        "metadata": {},
    }


def _ready_landing_page_row():
    return {
        "id": "11111111-1111-1111-1111-111111111111",
        "status": "approved",
        "campaign_name": "acme-support-retention",
        "persona": "VP Engineering",
        "value_prop": "Catch pressure early",
        "title": "Acme support retention page",
        "slug": "acme-support-retention",
        "hero": {
            "headline": "Catch support pressure before renewal risk builds",
            "subheadline": (
                "A landing page for VP Engineering teams that turns repeat "
                "support signals into clearer answers before customers drift."
            ),
            "cta_label": "Book a demo",
            "cta_url": "/landing",
        },
        "sections": [
            {
                "id": "problem",
                "title": "Support problems that keep coming back",
                "body_markdown": (
                    "VP Engineering teams catch pressure early when repeated "
                    "support issues are turned into visible answers before "
                    "renewal risk builds. Support problems keep coming back "
                    "when teams cannot see where customers get stuck."
                ),
                "metadata": {
                    "kind": "problem",
                    "primary_question": "Why do support problems become renewal risk?",
                    "answer_summary": (
                        "VP Engineering teams catch pressure early when repeated "
                        "support issues are turned into visible answers before "
                        "renewal risk builds."
                    ),
                },
            },
            {
                "id": "solution",
                "title": "A workflow for catching pressure early",
                "body_markdown": (
                    "The solution helps VP Engineering teams catch pressure early "
                    "by turning repeated support signals into a clear page and "
                    "follow-up workflow."
                ),
                "metadata": {
                    "kind": "solution",
                    "primary_question": "How does the workflow catch pressure early?",
                    "answer_summary": (
                        "The solution helps VP Engineering teams catch pressure "
                        "early by turning repeated support signals into a clear "
                        "page and follow-up workflow."
                    ),
                },
            },
            {
                "id": "buyer_questions",
                "title": "Questions buyers ask before rollout",
                "body_markdown": (
                    "VP Engineering buyers can answer implementation, pricing, "
                    "and security questions before they have to email the team. "
                    "This keeps the conversion path clear during rollout review."
                ),
                "metadata": {
                    "kind": "objection",
                    "primary_question": "What should buyers know before rollout?",
                    "answer_summary": (
                        "VP Engineering buyers can answer implementation, "
                        "pricing, and security questions before they have to "
                        "email the team."
                    ),
                },
            },
        ],
        "cta": {"label": "Book a demo", "url": "/landing"},
        "meta": {
            "title_tag": "Acme Support Retention for VP Engineering",
            "description": (
                "See how VP Engineering teams catch support pressure early and "
                "turn repeated customer questions into clearer retention pages."
            ),
            "og_title": "Acme Support Retention",
        },
        "reference_ids": ["r1"],
        "metadata": {},
    }


def _landing_page_generation_response(row: dict[str, object]) -> str:
    return json.dumps({
        "title": row["title"],
        "slug": row["slug"],
        "hero": row["hero"],
        "sections": row["sections"],
        "cta": row["cta"],
        "meta": row["meta"],
        "reference_ids": row["reference_ids"],
    })


def _sales_brief_row():
    return {
        "id": "brief-uuid-1",
        "status": "draft",
        "target_id": "vendor-acme",
        "target_mode": "vendor_retention",
        "brief_type": "pre_call",
        "title": "Acme brief",
        "headline": "Renewal pressure opens this week",
        "sections": [{"id": "context", "title": "Context", "body_markdown": "Body"}],
        "reference_ids": ["r1"],
        "metadata": {
            "generation_usage": {"input_tokens": 8, "output_tokens": 4},
            "reasoning_context": {"wedge": "support_erosion", "confidence": "medium"},
        },
    }


def _social_post_row():
    return {
        "id": "social-post-uuid-1",
        "status": "draft",
        "target_id": "review-1",
        "target_mode": "review",
        "channel": "linkedin",
        "text": "Acme support teams can spot churn risk before renewal.",
        "source_id": "source-1",
        "source_type": "review",
        "company_name": "Acme",
        "vendor_name": "Zendesk",
        "pain_points": ["slow response", "renewal pressure"],
        "metadata": {"source_url": "https://example.test/reviews/1"},
    }


def _ad_copy_row():
    return {
        "id": "ad-copy-uuid-1",
        "status": "draft",
        "target_id": "review-1",
        "target_mode": "review",
        "channel": "paid_social",
        "format": "single_image",
        "headline": "Zendesk proof: slow response",
        "primary_text": (
            "When Zendesk buyers mention slow response, use the proof."
        ),
        "cta": "See the proof",
        "source_id": "source-1",
        "source_type": "review",
        "company_name": "Acme",
        "vendor_name": "Zendesk",
        "pain_points": ["slow response", "renewal pressure"],
        "metadata": {"source_url": "https://example.test/reviews/1"},
    }


def _quote_card_row():
    return {
        "id": "quote-card-uuid-1",
        "status": "draft",
        "target_id": "review-1",
        "target_mode": "review",
        "theme": "customer_proof",
        "quote": "Pricing became hard to justify after renewal.",
        "attribution": "Acme",
        "headline": "Customer proof for Zendesk",
        "supporting_text": "Use this quote to frame slow response.",
        "source_id": "source-1",
        "source_type": "review",
        "company_name": "Acme",
        "vendor_name": "Zendesk",
        "pain_points": ["slow response", "renewal pressure"],
        "metadata": {"source_url": "https://example.test/reviews/1"},
    }


def _stat_card_row():
    return {
        "id": "stat-card-uuid-1",
        "status": "draft",
        "target_id": "review-1",
        "target_mode": "review",
        "theme": "customer_metric",
        "metric_label": "NPS score",
        "metric_value": 42,
        "metric_display": "42",
        "claim": "NPS score: 42",
        "headline": "Customer metric for Zendesk",
        "supporting_text": "Use this stat to frame slow response.",
        "evidence": "NPS score dropped to 42 after renewal.",
        "source_id": "source-1",
        "source_type": "review",
        "company_name": "Acme",
        "vendor_name": "Zendesk",
        "pain_points": ["slow response", "renewal pressure"],
        "metadata": {"source_url": "https://example.test/reviews/1"},
    }


def _ticket_faq_row():
    return {
        "id": "faq-uuid-1",
        "status": "draft",
        "target_id": "acct_1",
        "target_mode": "support_account",
        "title": "Support FAQ",
        "markdown": "# Support FAQ\n\n## How do I reset login?",
        "items": [{"question": "How do I reset login?", "answer": "Use the reset link."}],
        "source_count": 3,
        "ticket_source_count": 2,
        "output_checks": {"uses_user_vocabulary": True, "has_action_items": True},
        "warnings": [{"code": "thin_evidence"}],
        "metadata": {},
    }


def _publishable_ticket_faq_row():
    row = _ticket_faq_row()
    row.update({
        "id": "11111111-1111-1111-1111-111111111111",
        "status": "approved",
        "items": [{
            "faq_item_id": "faq-item-1",
            "topic": "billing",
            "question": "Why was I charged twice?",
            "resolution_text": "Open Billing and compare settled charges.",
            "answer_evidence_status": "resolution_evidence",
            "source_ids": ["ticket-1"],
        }],
        "metadata": {"corpus_id": "corpus-1"},
    })
    return row


def _publish_attempt_row(**overrides):
    row = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "faq_draft_id": "11111111-1111-1111-1111-111111111111",
        "draft_status": "approved",
        "ok": True,
        "publishable_count": 1,
        "skipped_count": 0,
        "published_count": 1,
        "updated_count": 0,
        "failed_count": 0,
        "pending_reconcile_count": 0,
        "draft_status_updated": True,
        "skipped": [],
        "results": [{
            "status": "published",
            "external_id": "external-1",
            "external_url": "https://example.zendesk.com/macros/1",
        }],
        "created_at": "2026-05-30 12:00:00+00:00",
    }
    row.update(overrides)
    return row


def _client(
    pool,
    *,
    scope=None,
    llm=None,
    skills=None,
    macro_provider=None,
    config: GeneratedAssetApiConfig | None = None,
    dependencies=None,
) -> TestClient:
    app = FastAPI()

    async def pool_provider():
        return pool

    async def scope_provider():
        return scope

    async def llm_provider():
        return llm

    async def skills_provider():
        return skills

    async def macro_publish_provider():
        return macro_provider

    app.include_router(
        create_generated_asset_router(
            pool_provider=pool_provider,
            scope_provider=scope_provider if scope is not None else None,
            llm_provider=llm_provider if llm is not None else None,
            skills_provider=skills_provider if skills is not None else None,
            macro_publish_provider=(
                macro_publish_provider if macro_provider is not None else None
            ),
            config=config,
            dependencies=dependencies,
        )
    )
    return TestClient(app)


def _public_client(
    pool,
    *,
    config: GeneratedAssetApiConfig | None = None,
) -> TestClient:
    app = FastAPI()

    async def pool_provider():
        return pool

    app.include_router(
        create_public_landing_page_router(
            pool_provider=pool_provider,
            config=config,
        )
    )
    return TestClient(app)


def test_generated_asset_router_lists_report_drafts_with_filters() -> None:
    pool = _Pool(rows=[_report_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/report/drafts"
        "?target_mode=vendor_retention&report_type=vendor_pressure&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["rows"][0]["id"] == "report-uuid-1"
    assert body["rows"][0]["status"] == "draft"
    assert body["rows"][0]["title"] == "Acme report"
    assert body["rows"][0]["reasoning_wedge"] == "price_squeeze"
    query, args = pool.fetch_calls[0]
    assert "FROM reports" in query
    assert args == ("acct_1", "draft", "vendor_retention", "vendor_pressure", 5)


def test_generated_asset_router_lists_blog_post_drafts_with_filters() -> None:
    pool = _Pool(rows=[_blog_post_row()])
    draft_id = "11111111-1111-1111-1111-111111111111"

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/blog_post/drafts"
        f"?topic_type=vendor_alternative&id={draft_id}&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["rows"][0]["id"] == "blog-post-uuid-1"
    assert body["rows"][0]["status"] == "draft"
    assert body["rows"][0]["slug"] == "acme-pricing-pressure"
    assert body["rows"][0]["reasoning_wedge"] == "price_squeeze"
    assert body["rows"][0]["passed_output_checks"] == 6
    assert body["rows"][0]["seo_aeo_readiness"]["status"] == "ready"
    assert body["rows"][0]["geo_readiness"]["status"] == "ready"
    assert body["rows"][0]["geo_readiness"]["passed"] == 7
    query, args = pool.fetch_calls[0]
    assert "FROM blog_posts" in query
    assert "id = ANY($4::uuid[])" in query
    assert args == ("acct_1", "draft", "vendor_alternative", [draft_id], 5)


def test_generated_asset_router_lists_landing_page_drafts_with_readiness() -> None:
    pool = _Pool(rows=[_landing_page_row()])
    draft_id = "22222222-2222-2222-2222-222222222222"

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/landing_page/drafts"
        f"?campaign_name=acme-launch&slug=acme-launch&id={draft_id}&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    row = body["rows"][0]
    assert row["id"] == "landing-page-uuid-1"
    assert row["passed_output_checks"] == 3
    assert row["seo_aeo_readiness"]["status"] == "needs_review"
    assert row["geo_readiness"]["status"] == "needs_review"
    assert row["geo_readiness"]["checks"]["trust_signal_visibility"] is True
    query, args = pool.fetch_calls[0]
    assert "FROM landing_pages" in query
    assert "id = ANY($5::uuid[])" in query
    assert args == ("acct_1", "draft", "acme-launch", "acme-launch", [draft_id], 5)


def test_generated_asset_router_exports_landing_page_csv() -> None:
    pool = _Pool(rows=[_landing_page_row()])
    draft_id = "33333333-3333-3333-3333-333333333333"

    response = _client(pool).get(
        "/content-assets/landing_page/drafts/export"
        f"?format=csv&campaign_name=acme-launch&slug=acme-launch&id={draft_id}"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "content_assets_landing_page.csv" in response.headers["content-disposition"]
    assert "campaign_name,persona,value_prop" in response.text
    assert "Acme landing page" in response.text
    query, args = pool.fetch_calls[0]
    assert "FROM landing_pages" in query
    assert "id = ANY($5::uuid[])" in query
    assert args == ("", "draft", "acme-launch", "acme-launch", [draft_id], 20)


def test_generated_asset_router_rejects_id_filter_for_unsupported_asset() -> None:
    pool = _Pool(rows=[_report_row()])

    response = _client(pool).get(
        "/content-assets/report/drafts?id=11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "id filters are only supported for blog_post and landing_page"
    )
    assert pool.fetch_calls == []


def test_generated_asset_router_updates_landing_page_draft_with_readiness() -> None:
    row = {**_ready_landing_page_row(), "status": "rejected"}
    pool = _EditableLandingPagePool(row)

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).patch(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111",
        json={
            "title": "Edited support retention page",
            "meta": {
                **row["meta"],
                "title_tag": "Edited Support Retention for VP Engineering",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "11111111-1111-1111-1111-111111111111"
    assert body["title"] == "Edited support retention page"
    assert body["status"] == "draft"
    assert body["seo_aeo_readiness"]["status"] == "ready"
    assert body["geo_readiness"]["status"] == "ready"
    assert len(pool.fetch_calls) == 2
    select_query, select_args = pool.fetch_calls[0]
    assert "FROM landing_pages" in select_query
    assert select_args == (
        "11111111-1111-1111-1111-111111111111",
        "acct_1",
    )
    update_query, update_args = pool.fetch_calls[1]
    assert "UPDATE landing_pages" in update_query
    assert "status <> 'approved'" in update_query
    assert update_args[0:4] == (
        "11111111-1111-1111-1111-111111111111",
        "acct_1",
        "Edited support retention page",
        "acme-support-retention",
    )


def test_generated_asset_router_blocks_approved_landing_page_edit() -> None:
    pool = _EditableLandingPagePool(_ready_landing_page_row())

    response = _client(pool).patch(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111",
        json={"title": "Edited title"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "approved landing pages cannot be edited"
    assert len(pool.fetch_calls) == 1


def test_generated_asset_router_returns_404_for_missing_landing_page_edit() -> None:
    pool = _EditableLandingPagePool()

    response = _client(pool).patch(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111",
        json={"title": "Edited title"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Landing page draft not found"
    assert len(pool.fetch_calls) == 1


def test_generated_asset_router_returns_404_for_cross_tenant_landing_page_edit() -> None:
    row = {**_ready_landing_page_row(), "status": "draft", "account_id": "acct_1"}
    pool = _EditableLandingPagePool(row)

    response = _client(pool, scope=TenantScope(account_id="acct_2")).patch(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111",
        json={"title": "Edited title"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Landing page draft not found"
    assert len(pool.fetch_calls) == 1


def test_generated_asset_router_ignores_status_and_metadata_mass_assignment() -> None:
    row = {
        **_ready_landing_page_row(),
        "status": "quality_blocked",
        "metadata": {
            "scope": {"account_id": "acct_1", "user_id": "user_1"},
            "generation_usage": {"input_tokens": 10},
        },
    }
    pool = _EditableLandingPagePool(row)

    response = _client(pool, scope=TenantScope(account_id="acct_1")).patch(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111",
        json={
            "title": "Edited support retention page",
            "status": "approved",
            "metadata": {"scope": {"account_id": "acct_2", "user_id": "user_2"}},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Edited support retention page"
    assert body["status"] == "draft"
    assert body["metadata"]["scope"] == {"account_id": "acct_1", "user_id": "user_1"}
    update_query, update_args = pool.fetch_calls[1]
    assert "metadata = $10::jsonb" in update_query
    persisted_metadata = json.loads(update_args[9])
    assert persisted_metadata["scope"] == {"account_id": "acct_1", "user_id": "user_1"}
    assert "acct_2" not in json.dumps(persisted_metadata)
    assert "approved" not in update_args


def test_generated_asset_router_rejects_non_landing_page_edit() -> None:
    pool = _Pool()

    response = _client(pool).patch(
        "/content-assets/blog_post/drafts/"
        "11111111-1111-1111-1111-111111111111",
        json={"title": "Edited title"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "only landing_page drafts can be edited"
    assert pool.fetch_calls == []


def test_generated_asset_router_rejects_empty_landing_page_edit_payload() -> None:
    pool = _EditableLandingPagePool({**_ready_landing_page_row(), "status": "draft"})

    response = _client(pool).patch(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111",
        json={"campaign_name": "not-editable"},
    )

    assert response.status_code == 400
    assert "payload must include at least one editable landing page field" in (
        response.json()["detail"]
    )


def test_generated_asset_router_repairs_landing_page_draft_and_returns_review_row() -> None:
    repaired_row = {**_ready_landing_page_row(), "status": "quality_blocked"}
    needs_repair = {
        **repaired_row,
        "meta": {
            key: value
            for key, value in repaired_row["meta"].items()
            if key != "description"
        },
    }
    pool = _LockingEditableLandingPagePool(needs_repair)
    llm = _LLM([_landing_page_generation_response(repaired_row)])
    skills = _Skills()

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
        llm=llm,
        skills=skills,
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "11111111-1111-1111-1111-111111111111"
    assert body["status"] == "draft"
    assert body["seo_aeo_readiness"]["status"] == "ready"
    assert body["geo_readiness"]["status"] == "ready"
    assert body["repair_result"]["generated"] == 1
    assert body["repair_result"]["saved_ids"] == [
        "11111111-1111-1111-1111-111111111111"
    ]
    assert len(pool.fetch_calls) == 4
    select_query, select_args = pool.fetch_calls[0]
    assert "FROM landing_pages" in select_query
    assert select_args == (
        "11111111-1111-1111-1111-111111111111",
        "acct_1",
    )
    claim_query, claim_args = pool.fetch_calls[1]
    assert "jsonb_set" in claim_query
    claim_payload = json.loads(claim_args[2])
    assert claim_payload["token"]
    assert claim_payload["expires_at_epoch"] > claim_payload["claimed_at_epoch"]
    update_query, update_args = pool.fetch_calls[2]
    assert "UPDATE landing_pages" in update_query
    assert update_args[0:2] == (
        "11111111-1111-1111-1111-111111111111",
        "acct_1",
    )
    persisted_metadata = json.loads(update_args[9])
    assert persisted_metadata["saved_draft_repair_source_id"] == (
        "11111111-1111-1111-1111-111111111111"
    )
    assert persisted_metadata["generation_quality_repair_attempts"] == 1
    assert update_args[10] == claim_payload["token"]
    system_prompt = llm.calls[0]["messages"][0].content
    assert '"repair_mode":"saved_draft"' in system_prompt
    assert '"repair_issues":["seo_aeo_readiness:meta_description"]' in system_prompt
    assert skills.calls == ["digest/landing_page_generation"]
    assert pool.acquire_calls == 0
    assert pool.release_calls == 0
    assert pool.released_connections == []
    assert pool.lock_calls == []
    assert len(pool.execute_calls) == 1
    assert asset_api.LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY not in body["metadata"]


def test_generated_asset_router_repairs_without_advisory_lock_connection(caplog) -> None:
    repaired_row = {**_ready_landing_page_row(), "status": "quality_blocked"}
    needs_repair = {
        **repaired_row,
        "meta": {
            key: value
            for key, value in repaired_row["meta"].items()
            if key != "description"
        },
    }
    pool = _EditableLandingPagePool(needs_repair)
    llm = _LLM([_landing_page_generation_response(repaired_row)])
    skills = _Skills()
    caplog.set_level(logging.WARNING, logger=asset_api.__name__)

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
        llm=llm,
        skills=skills,
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 200
    assert response.json()["repair_result"]["generated"] == 1
    assert len(llm.calls) == 1
    assert not any(
        "repair advisory lock skipped" in record.getMessage()
        for record in caplog.records
    )
    assert len(pool.execute_calls) == 1
    assert asset_api.LANDING_PAGE_REPAIR_CLAIM_METADATA_KEY not in (
        response.json()["metadata"]
    )


def test_generated_asset_router_logs_repair_claim_release_miss(caplog) -> None:
    repaired_row = {**_ready_landing_page_row(), "status": "quality_blocked"}
    needs_repair = {
        **repaired_row,
        "meta": {
            key: value
            for key, value in repaired_row["meta"].items()
            if key != "description"
        },
    }
    pool = _EditableLandingPagePool(needs_repair)
    pool.force_release_miss = True
    llm = _LLM([_landing_page_generation_response(repaired_row)])
    skills = _Skills()
    caplog.set_level(logging.WARNING, logger=asset_api.__name__)

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
        llm=llm,
        skills=skills,
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 200
    assert any(
        "landing page repair claim release missed" in record.getMessage()
        and record.account_id == "acct_1"
        and record.landing_page_id == "11111111-1111-1111-1111-111111111111"
        and record.outcome == "success"
        for record in caplog.records
    )


def test_generated_asset_router_blocks_concurrent_landing_page_repair_before_llm() -> None:
    row = {**_ready_landing_page_row(), "status": "quality_blocked"}
    row["meta"] = {
        key: value
        for key, value in row["meta"].items()
        if key != "description"
    }
    pool = _LockingEditableLandingPagePool(row, lock_acquired=False)
    llm = _LLM([])
    skills = _Skills()

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
        llm=llm,
        skills=skills,
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "landing page draft repair already in progress"
    assert len(pool.fetch_calls) == 2
    assert "jsonb_set" in pool.fetch_calls[1][0]
    assert pool.acquire_calls == 0
    assert pool.release_calls == 0
    assert pool.released_connections == []
    assert pool.lock_calls == []
    assert pool.unlocked is False
    assert llm.calls == []
    assert skills.calls == []


def test_generated_asset_router_blocks_concurrent_repair_before_provider_resolution() -> None:
    row = {**_ready_landing_page_row(), "status": "quality_blocked"}
    row["meta"] = {
        key: value
        for key, value in row["meta"].items()
        if key != "description"
    }
    pool = _LockingEditableLandingPagePool(row, lock_acquired=False)

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "landing page draft repair already in progress"
    assert len(pool.fetch_calls) == 2
    assert "jsonb_set" in pool.fetch_calls[1][0]
    assert pool.execute_calls == []


def test_generated_asset_router_repair_requires_llm_provider() -> None:
    row = {**_ready_landing_page_row(), "status": "quality_blocked"}
    row["meta"] = {"title_tag": row["meta"]["title_tag"]}
    pool = _EditableLandingPagePool(row)

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
        skills=_Skills(),
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM unavailable"
    assert len(pool.fetch_calls) == 2
    assert "jsonb_set" in pool.fetch_calls[1][0]
    assert len(pool.execute_calls) == 1


def test_generated_asset_router_repair_requires_skills_provider() -> None:
    row = {**_ready_landing_page_row(), "status": "quality_blocked"}
    row["meta"] = {"title_tag": row["meta"]["title_tag"]}
    pool = _EditableLandingPagePool(row)
    llm = _LLM([])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
        llm=llm,
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Landing page generation skills unavailable"
    assert llm.calls == []
    assert len(pool.fetch_calls) == 2
    assert "jsonb_set" in pool.fetch_calls[1][0]
    assert len(pool.execute_calls) == 1


def test_generated_asset_router_returns_404_for_cross_tenant_landing_page_repair() -> None:
    row = {
        **_ready_landing_page_row(),
        "status": "quality_blocked",
        "account_id": "acct_1",
    }
    row["meta"] = {"title_tag": row["meta"]["title_tag"]}
    pool = _EditableLandingPagePool(row)
    llm = _LLM([])
    skills = _Skills()

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_2"),
        llm=llm,
        skills=skills,
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Landing page draft not found"
    assert len(pool.fetch_calls) == 1
    assert llm.calls == []
    assert skills.calls == []


def test_generated_asset_router_returns_repair_result_when_repair_fails() -> None:
    needs_repair = {**_ready_landing_page_row(), "status": "quality_blocked"}
    needs_repair["meta"] = {
        key: value
        for key, value in needs_repair["meta"].items()
        if key != "description"
    }
    pool = _LockingEditableLandingPagePool(needs_repair)
    llm = _LLM([_landing_page_generation_response(needs_repair)])
    skills = _Skills()

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
        llm=llm,
        skills=skills,
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["message"] == "landing page draft could not be repaired"
    assert detail["repair_result"]["generated"] == 0
    assert detail["repair_result"]["skipped"] == 1
    assert detail["repair_result"]["errors"][0]["reason"] == "quality_blocked"
    assert detail["repair_result"]["errors"][0]["blockers"] == [
        "seo_aeo_readiness:meta_description"
    ]
    assert len(pool.fetch_calls) == 2
    assert "jsonb_set" in pool.fetch_calls[1][0]
    assert pool.acquire_calls == 0
    assert pool.release_calls == 0
    assert pool.released_connections == []
    assert pool.lock_calls == []
    assert pool.unlocked is False
    assert len(pool.execute_calls) == 1
    assert len(llm.calls) == 1
    assert skills.calls == ["digest/landing_page_generation"]


def test_generated_asset_router_blocks_approved_landing_page_repair() -> None:
    pool = _EditableLandingPagePool(_ready_landing_page_row())
    llm = _LLM([])

    response = _client(
        pool,
        llm=llm,
        skills=_Skills(),
    ).post(
        "/content-assets/landing_page/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "approved landing pages cannot be repaired"
    assert llm.calls == []


def test_generated_asset_router_rejects_non_landing_page_repair() -> None:
    pool = _Pool()

    response = _client(pool).post(
        "/content-assets/blog_post/drafts/"
        "11111111-1111-1111-1111-111111111111/repair"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "only landing_page drafts can be repaired"
    assert pool.fetch_calls == []


def test_generated_asset_router_returns_public_approved_landing_page() -> None:
    row = {**_landing_page_row(), "status": "approved"}
    row["id"] = "11111111-1111-1111-1111-111111111111"
    row["meta"] = {
        "title_tag": "Acme landing page",
        "description": "A public approved landing page for Acme.",
    }
    row["reference_ids"] = ["internal-ref-1"]
    row["metadata"] = {
        "scope": {"account_id": "acct_1", "user_id": "user_1"},
        "generation_usage": {"input_tokens": 10, "output_tokens": 5},
        "reasoning_context": {"wedge": "support_gap", "confidence": 0.8},
    }
    pool = _Pool(rows=[row])

    response = _public_client(pool).get(
        "/content-assets/landing_page/public/"
        "11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "11111111-1111-1111-1111-111111111111"
    assert body["slug"] == "acme-launch"
    assert body["structured_data"]["@context"] == "https://schema.org"
    assert set(body) == {
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
    assert body["robots"] == "noindex,follow"
    assert "status" not in body
    assert "metadata" not in body
    assert "reference_ids" not in body
    assert "generation_input_tokens" not in body
    assert "reasoning_context_used" not in body
    assert "seo_aeo_readiness" not in body
    query, args = pool.fetch_calls[0]
    assert "FROM landing_pages" in query
    assert "status = 'approved'" in query
    assert args == ("11111111-1111-1111-1111-111111111111",)


def test_generated_asset_router_indexes_public_ready_landing_page() -> None:
    pool = _Pool(rows=[_ready_landing_page_row()])

    response = _public_client(pool).get(
        "/content-assets/landing_page/public/"
        "11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["robots"] == "index,follow"
    assert "seo_aeo_readiness" not in body
    assert "geo_readiness" not in body


def test_generated_asset_router_keeps_placeholder_cta_landing_page_noindex() -> None:
    row = _ready_landing_page_row()
    row["cta"] = {"label": "Book a demo", "url": "/demo"}
    pool = _Pool(rows=[row])

    response = _public_client(pool).get(
        "/content-assets/landing_page/public/"
        "11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 200
    assert response.json()["robots"] == "noindex,follow"


@pytest.mark.parametrize(
    "url",
    ["/demo/", "/demo?utm=1", "/demo#hero", "/demo/?utm=1"],
)
def test_generated_asset_router_keeps_demo_cta_variants_noindex(url: str) -> None:
    row = _ready_landing_page_row()
    row["cta"] = {"label": "Book a demo", "url": url}
    pool = _Pool(rows=[row])

    response = _public_client(pool).get(
        "/content-assets/landing_page/public/"
        "11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 200
    assert response.json()["robots"] == "noindex,follow"


def test_generated_asset_router_sitemap_includes_only_indexable_landing_pages() -> None:
    incomplete = {**_landing_page_row(), "status": "approved"}
    incomplete["id"] = "22222222-2222-2222-2222-222222222222"
    incomplete["slug"] = "not-ready"
    ready = _ready_landing_page_row()
    ready["metadata"] = {
        "scope": {"account_id": "acct_1", "user_id": "user_1"},
        "generation_usage": {"input_tokens": 10, "output_tokens": 5},
        "reasoning_context": {"wedge": "support_gap", "confidence": 0.8},
    }
    rejected = {**_ready_landing_page_row(), "status": "rejected"}
    rejected["id"] = "33333333-3333-3333-3333-333333333333"
    rejected["slug"] = "rejected-ready"
    pool = _Pool(rows=[ready, incomplete, rejected])

    response = _public_client(pool).get(
        "/content-assets/landing_page/public/sitemap.xml"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert (
        "<loc>http://testserver/lp/"
        "11111111-1111-1111-1111-111111111111/"
        "acme-support-retention</loc>"
    ) in response.text
    assert "not-ready" not in response.text
    assert "rejected-ready" not in response.text
    assert "acct_1" not in response.text
    assert "user_1" not in response.text
    assert "generation_usage" not in response.text
    assert "seo_aeo_readiness" not in response.text
    assert "geo_readiness" not in response.text
    query, args = pool.fetch_calls[0]
    assert "FROM landing_pages" in query
    assert "status = 'approved'" in query
    assert "metadata" not in query
    assert args == ()


def test_generated_asset_router_sitemap_applies_limit_after_readiness_filter() -> None:
    incomplete = {**_landing_page_row(), "status": "approved"}
    incomplete["id"] = "22222222-2222-2222-2222-222222222222"
    incomplete["slug"] = "not-ready"
    ready = _ready_landing_page_row()
    later_ready = _ready_landing_page_row()
    later_ready["id"] = "33333333-3333-3333-3333-333333333333"
    later_ready["slug"] = "later-ready"
    pool = _Pool(rows=[incomplete, ready, later_ready])

    response = _public_client(
        pool,
        config=GeneratedAssetApiConfig(public_sitemap_limit=1),
    ).get("/content-assets/landing_page/public/sitemap.xml")

    assert response.status_code == 200
    assert "not-ready" not in response.text
    assert "acme-support-retention" in response.text
    assert "later-ready" not in response.text
    assert response.text.count("<url>") == 1
    query, args = pool.fetch_calls[0]
    assert "LIMIT" not in query
    assert args == ()


def test_generated_asset_router_sitemap_uses_configured_public_base_url() -> None:
    pool = _Pool(rows=[_ready_landing_page_row()])

    response = _public_client(
        pool,
        config=GeneratedAssetApiConfig(
            public_landing_page_base_url="https://example.com/"
        ),
    ).get("/content-assets/landing_page/public/sitemap.xml")

    assert response.status_code == 200
    assert (
        "<loc>https://example.com/lp/"
        "11111111-1111-1111-1111-111111111111/"
        "acme-support-retention</loc>"
    ) in response.text


def test_generated_asset_router_hides_non_public_landing_page() -> None:
    pool = _Pool(rows=[])

    response = _public_client(pool).get(
        "/content-assets/landing_page/public/"
        "11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Landing page not found"
    assert len(pool.fetch_calls) == 1


def test_generated_asset_router_hides_non_approved_landing_page_row() -> None:
    row = {**_landing_page_row(), "status": "draft"}
    row["id"] = "11111111-1111-1111-1111-111111111111"
    pool = _PublicLandingPagePool(rows=[row])

    response = _public_client(pool).get(
        "/content-assets/landing_page/public/"
        "11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Landing page not found"
    assert len(pool.fetch_calls) == 1


def test_generated_asset_router_rejects_invalid_public_landing_page_id() -> None:
    pool = _Pool()

    response = _public_client(pool).get("/content-assets/landing_page/public/not-a-uuid")

    assert response.status_code == 404
    assert response.json()["detail"] == "Landing page not found"
    assert pool.fetch_calls == []


def test_generated_asset_router_exports_sales_brief_json_without_status_filter() -> None:
    pool = _Pool(rows=[_sales_brief_row()])

    response = _client(pool).get(
        "/content-assets/sales_brief/drafts/export"
        "?format=json&status=&target_mode=vendor_retention&brief_type=pre_call"
    )

    assert response.status_code == 200
    row = response.json()["rows"][0]
    assert row["brief_type"] == "pre_call"
    assert row["reasoning_wedge"] == "support_erosion"
    query, args = pool.fetch_calls[0]
    assert "FROM sales_briefs" in query
    assert "status = " not in query
    assert args == ("", "vendor_retention", "pre_call", 20)


def test_generated_asset_router_lists_ticket_faq_drafts_with_filters() -> None:
    pool = _Pool(rows=[_ticket_faq_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/faq_markdown/drafts"
        "?target_mode=support_account&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["rows"][0]["id"] == "faq-uuid-1"
    assert body["rows"][0]["title"] == "Support FAQ"
    assert body["rows"][0]["passed_output_checks"] == 2
    query, args = pool.fetch_calls[0]
    assert "FROM ticket_faq_markdown" in query
    assert args == ("acct_1", "draft", "support_account", 5)


def test_generated_asset_router_exports_ticket_faq_csv() -> None:
    pool = _Pool(rows=[_ticket_faq_row()])

    response = _client(pool).get(
        "/content-assets/faq_markdown/drafts/export"
        "?format=csv&target_mode=support_account"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "content_assets_faq_markdown.csv" in response.headers["content-disposition"]
    assert "target_id,target_mode,title" in response.text
    assert "Support FAQ" in response.text
    query, args = pool.fetch_calls[0]
    assert "FROM ticket_faq_markdown" in query
    assert args == ("", "draft", "support_account", 20)


def test_generated_asset_router_lists_social_post_drafts_with_filters() -> None:
    pool = _Pool(rows=[_social_post_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/social_post/drafts"
        "?target_mode=review&channel=linkedin&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    row = body["rows"][0]
    assert row["id"] == "social-post-uuid-1"
    assert row["channel"] == "linkedin"
    assert row["text"] == "Acme support teams can spot churn risk before renewal."
    assert row["pain_point_count"] == 2
    query, args = pool.fetch_calls[0]
    assert "FROM social_posts" in query
    assert args == ("acct_1", "draft", "review", "linkedin", 5)


def test_generated_asset_router_exports_social_post_csv() -> None:
    pool = _Pool(rows=[_social_post_row()])

    response = _client(pool).get(
        "/content-assets/social_post/drafts/export"
        "?format=csv&status=&target_mode=review&channel=linkedin"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "content_assets_social_post.csv" in response.headers["content-disposition"]
    assert "target_id,target_mode,channel" in response.text
    assert "Acme support teams can spot churn risk before renewal." in response.text
    query, args = pool.fetch_calls[0]
    assert "FROM social_posts" in query
    assert "status = " not in query
    assert args == ("", "review", "linkedin", 20)


def test_generated_asset_router_lists_ad_copy_drafts_with_filters() -> None:
    pool = _Pool(rows=[_ad_copy_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/ad_copy/drafts"
        "?target_mode=review&channel=paid_social&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    row = body["rows"][0]
    assert row["id"] == "ad-copy-uuid-1"
    assert row["channel"] == "paid_social"
    assert row["format"] == "single_image"
    assert row["headline"] == "Zendesk proof: slow response"
    assert row["cta"] == "See the proof"
    assert row["pain_point_count"] == 2
    query, args = pool.fetch_calls[0]
    assert "FROM ad_copy_drafts" in query
    assert args == ("acct_1", "draft", "review", "paid_social", 5)


def test_generated_asset_router_exports_ad_copy_csv() -> None:
    pool = _Pool(rows=[_ad_copy_row()])

    response = _client(pool).get(
        "/content-assets/ad_copy/drafts/export"
        "?format=csv&status=&target_mode=review&channel=paid_social"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "content_assets_ad_copy.csv" in response.headers["content-disposition"]
    assert "target_id,target_mode,channel,format" in response.text
    assert "Zendesk proof: slow response" in response.text
    query, args = pool.fetch_calls[0]
    assert "FROM ad_copy_drafts" in query
    assert "status = " not in query
    assert args == ("", "review", "paid_social", 20)


def test_generated_asset_router_lists_quote_card_drafts_with_filters() -> None:
    pool = _Pool(rows=[_quote_card_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/quote_card/drafts"
        "?target_mode=review&theme=customer_proof&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    row = body["rows"][0]
    assert row["id"] == "quote-card-uuid-1"
    assert row["theme"] == "customer_proof"
    assert row["quote"] == "Pricing became hard to justify after renewal."
    assert row["attribution"] == "Acme"
    assert row["headline"] == "Customer proof for Zendesk"
    assert row["pain_point_count"] == 2
    query, args = pool.fetch_calls[0]
    assert "FROM quote_card_drafts" in query
    assert args == ("acct_1", "draft", "review", "customer_proof", 5)


def test_generated_asset_router_exports_quote_card_csv() -> None:
    pool = _Pool(rows=[_quote_card_row()])

    response = _client(pool).get(
        "/content-assets/quote_card/drafts/export"
        "?format=csv&status=&target_mode=review&theme=customer_proof"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "content_assets_quote_card.csv" in response.headers["content-disposition"]
    assert "target_id,target_mode,theme" in response.text
    assert "Pricing became hard to justify after renewal." in response.text
    query, args = pool.fetch_calls[0]
    assert "FROM quote_card_drafts" in query
    assert "status = " not in query
    assert args == ("", "review", "customer_proof", 20)


def test_generated_asset_router_exports_quote_card_visual_html() -> None:
    pool = _Pool(rows=[_quote_card_row()])

    response = _client(pool).get(
        "/content-assets/quote_card/drafts/export"
        "?format=html&status=approved&target_mode=review&theme=customer_proof"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "content_assets_quote_card.html" in response.headers["content-disposition"]
    assert '<article class="visual-card quote-card"' in response.text
    assert "&quot;Pricing became hard to justify after renewal.&quot;" in response.text
    assert "Customer proof for Zendesk" in response.text
    query, args = pool.fetch_calls[0]
    assert "FROM quote_card_drafts" in query
    assert args == ("", "approved", "review", "customer_proof", 20)


def test_generated_asset_router_lists_stat_card_drafts_with_filters() -> None:
    pool = _Pool(rows=[_stat_card_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1"),
    ).get(
        "/content-assets/stat_card/drafts"
        "?target_mode=review&theme=customer_metric&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    row = body["rows"][0]
    assert row["id"] == "stat-card-uuid-1"
    assert row["theme"] == "customer_metric"
    assert row["metric_label"] == "NPS score"
    assert row["metric_display"] == "42"
    assert row["claim"] == "NPS score: 42"
    assert row["evidence"] == "NPS score dropped to 42 after renewal."
    assert row["pain_point_count"] == 2
    query, args = pool.fetch_calls[0]
    assert "FROM stat_card_drafts" in query
    assert args == ("acct_1", "draft", "review", "customer_metric", 5)


def test_generated_asset_router_exports_stat_card_csv() -> None:
    pool = _Pool(rows=[_stat_card_row()])

    response = _client(pool).get(
        "/content-assets/stat_card/drafts/export"
        "?format=csv&status=&target_mode=review&theme=customer_metric"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "content_assets_stat_card.csv" in response.headers["content-disposition"]
    assert "target_id,target_mode,theme" in response.text
    assert "NPS score: 42" in response.text
    query, args = pool.fetch_calls[0]
    assert "FROM stat_card_drafts" in query
    assert "status = " not in query
    assert args == ("", "review", "customer_metric", 20)


def test_generated_asset_router_exports_stat_card_visual_html_with_escaped_text() -> None:
    row = _stat_card_row()
    row["claim"] = "NPS <script>alert(1)</script>"
    row["evidence"] = 'NPS used <img src=x onerror="alert(1)"> after renewal.'
    pool = _Pool(rows=[row])

    response = _client(pool).get(
        "/content-assets/stat_card/drafts/export"
        "?format=html&status=approved&target_mode=review&theme=customer_metric"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "content_assets_stat_card.html" in response.headers["content-disposition"]
    assert '<article class="visual-card stat-card"' in response.text
    assert "<script>" not in response.text
    assert "<img" not in response.text
    assert "NPS &lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert (
        "NPS used &lt;img src=x onerror=&quot;alert(1)&quot;&gt; after renewal."
        in response.text
    )
    query, args = pool.fetch_calls[0]
    assert "FROM stat_card_drafts" in query
    assert args == ("", "approved", "review", "customer_metric", 20)


def test_generated_asset_router_reviews_report_with_host_defined_status() -> None:
    pool = _Pool()

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/report/drafts/review",
        json={"id": "report-uuid-1", "status": "published"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "account_id": "acct_1",
        "asset": "report",
        "id": "report-uuid-1",
        "status": "published",
        "updated": True,
    }
    query, args = pool.execute_calls[0]
    assert "UPDATE reports" in query
    assert args == ("report-uuid-1", "published", "acct_1")


def test_generated_asset_router_reviews_social_post() -> None:
    pool = _Pool()

    response = _client(pool, scope={"account_id": "acct_1"}).post(
        "/content-assets/social_post/drafts/review",
        json={"id": "social-post-uuid-1", "status": "approved"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "account_id": "acct_1",
        "asset": "social_post",
        "id": "social-post-uuid-1",
        "status": "approved",
        "updated": True,
    }
    query, args = pool.execute_calls[0]
    assert "UPDATE social_posts" in query
    assert args == ("social-post-uuid-1", "approved", "acct_1")


def test_generated_asset_router_reviews_ad_copy() -> None:
    pool = _Pool()

    response = _client(pool, scope={"account_id": "acct_1"}).post(
        "/content-assets/ad_copy/drafts/review",
        json={"id": "ad-copy-uuid-1", "status": "approved"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "account_id": "acct_1",
        "asset": "ad_copy",
        "id": "ad-copy-uuid-1",
        "status": "approved",
        "updated": True,
    }
    query, args = pool.execute_calls[0]
    assert "UPDATE ad_copy_drafts" in query
    assert args == ("ad-copy-uuid-1", "approved", "acct_1")


def test_generated_asset_router_reviews_quote_card() -> None:
    pool = _Pool()

    response = _client(pool, scope={"account_id": "acct_1"}).post(
        "/content-assets/quote_card/drafts/review",
        json={"id": "quote-card-uuid-1", "status": "approved"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "account_id": "acct_1",
        "asset": "quote_card",
        "id": "quote-card-uuid-1",
        "status": "approved",
        "updated": True,
    }
    query, args = pool.execute_calls[0]
    assert "UPDATE quote_card_drafts" in query
    assert args == ("quote-card-uuid-1", "approved", "acct_1")


def test_generated_asset_router_reviews_stat_card() -> None:
    pool = _Pool()

    response = _client(pool, scope={"account_id": "acct_1"}).post(
        "/content-assets/stat_card/drafts/review",
        json={"id": "stat-card-uuid-1", "status": "approved"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "account_id": "acct_1",
        "asset": "stat_card",
        "id": "stat-card-uuid-1",
        "status": "approved",
        "updated": True,
    }
    query, args = pool.execute_calls[0]
    assert "UPDATE stat_card_drafts" in query
    assert args == ("stat-card-uuid-1", "approved", "acct_1")


def test_generated_asset_router_reviews_ticket_faq_with_host_defined_status() -> None:
    row = _ticket_faq_row()
    row["id"] = "11111111-1111-1111-1111-111111111111"
    row["status"] = "approved"
    row["metadata"] = {"corpus_id": "corpus-1"}
    pool = _Pool(rows=[row])

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/faq_markdown/drafts/review",
        json={"id": "11111111-1111-1111-1111-111111111111", "status": "approved"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "account_id": "acct_1",
        "asset": "faq_markdown",
        "id": "11111111-1111-1111-1111-111111111111",
        "status": "approved",
        "updated": True,
    }
    query, args = pool.fetch_calls[0]
    assert "UPDATE ticket_faq_markdown" in query
    assert args == ("11111111-1111-1111-1111-111111111111", "approved", "acct_1")
    delete_query, delete_args = pool.execute_calls[0]
    assert "DELETE FROM ticket_faq_search_documents" in delete_query
    assert delete_args == ("acct_1", "corpus-1", "11111111-1111-1111-1111-111111111111")
    insert_query, insert_args = pool.execute_calls[1]
    assert "INSERT INTO ticket_faq_search_documents" in insert_query
    assert insert_args[:6] == (
        "acct_1",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
        "acct_1",
        "support_account",
        "approved",
    )


def test_generated_asset_router_publishes_ticket_faq_macros() -> None:
    pool = _Pool(rows=[_publishable_ticket_faq_row()])
    provider = _MacroProvider()

    response = _client(
        pool,
        scope={"account_id": "acct_1", "user_id": "user_1"},
        macro_provider=provider,
    ).post(
        "/content-assets/faq_markdown/drafts/"
        "11111111-1111-1111-1111-111111111111/publish-macros"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == "acct_1"
    assert body["asset"] == "faq_markdown"
    assert body["ok"] is True
    assert body["publishable_count"] == 1
    assert body["draft_status_updated"] is True
    assert provider.calls[0]["scope"] == TenantScope(
        account_id="acct_1",
        user_id="user_1",
    )
    macro = provider.calls[0]["macros"][0]
    assert macro.title == "Why was I charged twice?"
    assert macro.body == "Open Billing and compare settled charges."
    get_query, get_args = pool.fetch_calls[0]
    assert "FROM ticket_faq_markdown" in get_query
    assert get_args == ("11111111-1111-1111-1111-111111111111", "acct_1")
    update_query, update_args = pool.fetch_calls[1]
    assert "UPDATE ticket_faq_markdown" in update_query
    assert update_args == ("11111111-1111-1111-1111-111111111111", "published", "acct_1")
    attempt_query, attempt_args = next(
        call for call in pool.execute_calls
        if "ticket_faq_macro_publish_attempts" in call[0]
    )
    assert "INSERT INTO ticket_faq_macro_publish_attempts" in attempt_query
    assert attempt_args[:11] == (
        "acct_1",
        "11111111-1111-1111-1111-111111111111",
        "approved",
        True,
        1,
        0,
        1,
        0,
        0,
        0,
        True,
    )


def test_generated_asset_router_publish_macros_requires_provider() -> None:
    pool = _Pool(rows=[_publishable_ticket_faq_row()])

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/faq_markdown/drafts/"
        "11111111-1111-1111-1111-111111111111/publish-macros"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "FAQ macro publish provider unavailable"


def test_generated_asset_router_publish_macros_is_faq_only() -> None:
    pool = _Pool()
    provider = _MacroProvider()

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
        macro_provider=provider,
    ).post(
        "/content-assets/report/drafts/"
        "11111111-1111-1111-1111-111111111111/publish-macros"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "only faq_markdown drafts can publish macros"
    assert provider.calls == []
    assert pool.fetch_calls == []


def test_generated_asset_router_publish_macros_surfaces_pending_reconcile() -> None:
    pool = _Pool(rows=[_publishable_ticket_faq_row()])
    provider = _MacroProvider(
        status="failed",
        error="zendesk_macro_mapping_pending_reconcile",
    )

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
        macro_provider=provider,
    ).post(
        "/content-assets/faq_markdown/drafts/"
        "11111111-1111-1111-1111-111111111111/publish-macros"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["failed_count"] == 1
    assert body["pending_reconcile_count"] == 1
    assert body["draft_status_updated"] is False
    assert body["results"][0]["error"] == "zendesk_macro_mapping_pending_reconcile"
    assert len(pool.fetch_calls) == 1
    attempt_query, attempt_args = next(
        call for call in pool.execute_calls
        if "ticket_faq_macro_publish_attempts" in call[0]
    )
    assert "INSERT INTO ticket_faq_macro_publish_attempts" in attempt_query
    assert attempt_args[:11] == (
        "acct_1",
        "11111111-1111-1111-1111-111111111111",
        "approved",
        False,
        1,
        0,
        0,
        0,
        1,
        1,
        False,
    )


def test_generated_asset_router_lists_faq_macro_publish_attempts() -> None:
    pool = _FAQMacroPublishHistoryPool(
        draft_row=_publishable_ticket_faq_row(),
        attempt_rows=[_publish_attempt_row()],
    )

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).get(
        "/content-assets/faq_markdown/drafts/"
        "11111111-1111-1111-1111-111111111111/publish-macro-attempts?limit=3"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == "acct_1"
    assert body["asset"] == "faq_markdown"
    assert body["faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert body["count"] == 1
    assert body["limit"] == 3
    assert body["attempts"][0] == {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "faq_id": "11111111-1111-1111-1111-111111111111",
        "draft_status": "approved",
        "ok": True,
        "publishable_count": 1,
        "skipped_count": 0,
        "published_count": 1,
        "updated_count": 0,
        "failed_count": 0,
        "pending_reconcile_count": 0,
        "draft_status_updated": True,
        "skipped": [],
        "results": [{
            "status": "published",
            "external_id": "external-1",
            "external_url": "https://example.zendesk.com/macros/1",
        }],
        "created_at": "2026-05-30 12:00:00+00:00",
    }
    get_query, get_args = pool.fetch_calls[0]
    assert "FROM ticket_faq_markdown" in get_query
    assert get_args == ("11111111-1111-1111-1111-111111111111", "acct_1")
    attempts_query, attempts_args = pool.fetch_calls[1]
    assert "FROM ticket_faq_macro_publish_attempts" in attempts_query
    assert attempts_args == ("acct_1", "11111111-1111-1111-1111-111111111111", 3)


def test_generated_asset_router_publish_attempts_is_faq_only() -> None:
    pool = _FAQMacroPublishHistoryPool(
        draft_row=_publishable_ticket_faq_row(),
        attempt_rows=[_publish_attempt_row()],
    )

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).get(
        "/content-assets/report/drafts/"
        "11111111-1111-1111-1111-111111111111/publish-macro-attempts"
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "only faq_markdown drafts can list macro publish attempts"
    )
    assert pool.fetch_calls == []


def test_generated_asset_router_publish_attempts_404s_missing_faq() -> None:
    pool = _FAQMacroPublishHistoryPool(
        draft_row=None,
        attempt_rows=[_publish_attempt_row()],
    )

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).get(
        "/content-assets/faq_markdown/drafts/"
        "11111111-1111-1111-1111-111111111111/publish-macro-attempts"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "FAQ draft not found"
    assert len(pool.fetch_calls) == 1
    get_query, get_args = pool.fetch_calls[0]
    assert "FROM ticket_faq_markdown" in get_query
    assert get_args == ("11111111-1111-1111-1111-111111111111", "acct_1")


def test_generated_asset_router_reviews_blog_post_with_host_defined_status() -> None:
    pool = _Pool()

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/blog_post/drafts/review",
        json={"id": "blog-post-uuid-1", "status": "published"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "account_id": "acct_1",
        "asset": "blog_post",
        "id": "blog-post-uuid-1",
        "status": "published",
        "updated": True,
    }
    query, args = pool.execute_calls[0]
    assert "UPDATE blog_posts" in query
    assert args == ("blog-post-uuid-1", "published", "acct_1")


def test_generated_asset_router_returns_miss_without_hiding_result() -> None:
    pool = _Pool(execute_result="UPDATE 0")

    response = _client(pool).post(
        "/content-assets/sales_brief/drafts/review",
        json={"asset_id": "brief-uuid-1", "status": "ready_for_call"},
    )

    assert response.status_code == 200
    assert response.json()["updated"] is False
    query, args = pool.execute_calls[0]
    assert "UPDATE sales_briefs" in query
    assert args == ("brief-uuid-1", "ready_for_call", "")


def test_generated_asset_router_batch_reviews_reports() -> None:
    pool = _Pool(rows=[{"id": BATCH_REPORT_ID_1}, {"id": BATCH_REPORT_ID_2}])

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/report/drafts/review-batch",
        json={
            "ids": [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2],
            "status": "approved",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "account_id": "acct_1",
        "asset": "report",
        "ids": [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2],
        "status": "approved",
        "updated": 2,
        "updated_ids": [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2],
        "missing_ids": [],
    }
    assert pool.execute_calls == []
    assert len(pool.fetch_calls) == 1
    query, args = pool.fetch_calls[0]
    assert "UPDATE reports" in query
    assert "RETURNING id" in query
    assert args == ([BATCH_REPORT_ID_1, BATCH_REPORT_ID_2], "approved", "acct_1")


def test_generated_asset_router_batch_reviews_social_posts() -> None:
    pool = _Pool(rows=[{"id": BATCH_REPORT_ID_1}, {"id": BATCH_REPORT_ID_2}])

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/social_post/drafts/review-batch",
        json={
            "ids": [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2],
            "status": "rejected",
        },
    )

    assert response.status_code == 200
    assert response.json()["asset"] == "social_post"
    assert response.json()["updated"] == 2
    assert response.json()["updated_ids"] == [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2]
    query, args = pool.fetch_calls[0]
    assert "UPDATE social_posts" in query
    assert "RETURNING id" in query
    assert args == ([BATCH_REPORT_ID_1, BATCH_REPORT_ID_2], "rejected", "acct_1")
    assert pool.execute_calls == []


def test_generated_asset_router_batch_reviews_ad_copy() -> None:
    pool = _Pool(rows=[{"id": BATCH_REPORT_ID_1}, {"id": BATCH_REPORT_ID_2}])

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/ad_copy/drafts/review-batch",
        json={
            "ids": [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2],
            "status": "rejected",
        },
    )

    assert response.status_code == 200
    assert response.json()["asset"] == "ad_copy"
    assert response.json()["updated"] == 2
    assert response.json()["updated_ids"] == [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2]
    query, args = pool.fetch_calls[0]
    assert "UPDATE ad_copy_drafts" in query
    assert "RETURNING id" in query
    assert args == ([BATCH_REPORT_ID_1, BATCH_REPORT_ID_2], "rejected", "acct_1")
    assert pool.execute_calls == []


def test_generated_asset_router_batch_reviews_quote_cards() -> None:
    pool = _Pool(rows=[{"id": BATCH_REPORT_ID_1}, {"id": BATCH_REPORT_ID_2}])

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/quote_card/drafts/review-batch",
        json={
            "ids": [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2],
            "status": "rejected",
        },
    )

    assert response.status_code == 200
    assert response.json()["asset"] == "quote_card"
    assert response.json()["updated"] == 2
    assert response.json()["updated_ids"] == [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2]
    query, args = pool.fetch_calls[0]
    assert "UPDATE quote_card_drafts" in query
    assert "RETURNING id" in query
    assert args == ([BATCH_REPORT_ID_1, BATCH_REPORT_ID_2], "rejected", "acct_1")
    assert pool.execute_calls == []


def test_generated_asset_router_batch_reviews_stat_cards() -> None:
    pool = _Pool(rows=[{"id": BATCH_REPORT_ID_1}, {"id": BATCH_REPORT_ID_2}])

    response = _client(
        pool,
        scope={"account_id": "acct_1"},
    ).post(
        "/content-assets/stat_card/drafts/review-batch",
        json={
            "ids": [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2],
            "status": "rejected",
        },
    )

    assert response.status_code == 200
    assert response.json()["asset"] == "stat_card"
    assert response.json()["updated"] == 2
    assert response.json()["updated_ids"] == [BATCH_REPORT_ID_1, BATCH_REPORT_ID_2]
    query, args = pool.fetch_calls[0]
    assert "UPDATE stat_card_drafts" in query
    assert "RETURNING id" in query
    assert args == ([BATCH_REPORT_ID_1, BATCH_REPORT_ID_2], "rejected", "acct_1")
    assert pool.execute_calls == []


def test_generated_asset_router_batch_reviews_reports_misses() -> None:
    pool = _Pool(rows=[])

    response = _client(pool).post(
        "/content-assets/report/drafts/review-batch",
        json={"ids": [BATCH_REPORT_ID_1], "status": "approved"},
    )

    assert response.status_code == 200
    assert response.json()["updated"] == 0
    assert response.json()["updated_ids"] == []
    assert response.json()["missing_ids"] == [BATCH_REPORT_ID_1]
    assert len(pool.fetch_calls) == 1
    assert pool.execute_calls == []


def test_generated_asset_router_batch_review_partial_update() -> None:
    pool = _Pool(rows=[{"id": BATCH_REPORT_ID_1}])

    response = _client(pool, scope={"account_id": "acct_1"}).post(
        "/content-assets/report/drafts/review-batch",
        json={
            "ids": [BATCH_REPORT_ID_1, BATCH_REPORT_ID_MISSING],
            "status": "approved",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["updated"] == 1
    assert body["updated_ids"] == [BATCH_REPORT_ID_1]
    assert body["missing_ids"] == [BATCH_REPORT_ID_MISSING]
    assert len(pool.fetch_calls) == 1
    assert pool.execute_calls == []


def test_generated_asset_router_batch_review_treats_invalid_uuid_as_missing() -> None:
    pool = _Pool(rows=[{"id": BATCH_REPORT_ID_1}])

    response = _client(pool, scope={"account_id": "acct_1"}).post(
        "/content-assets/report/drafts/review-batch",
        json={
            "ids": [BATCH_REPORT_ID_1, "not-a-uuid"],
            "status": "approved",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["updated_ids"] == [BATCH_REPORT_ID_1]
    assert body["missing_ids"] == ["not-a-uuid"]
    assert len(pool.fetch_calls) == 1
    query, args = pool.fetch_calls[0]
    assert "UPDATE reports" in query
    assert args == (
        [BATCH_REPORT_ID_1],
        "approved",
        "acct_1",
    )
    assert pool.execute_calls == []


def test_generated_asset_router_batch_review_all_invalid_ids_skip_sql() -> None:
    pool = _Pool()

    response = _client(pool, scope={"account_id": "acct_1"}).post(
        "/content-assets/report/drafts/review-batch",
        json={"ids": ["not-a-uuid"], "status": "approved"},
    )

    assert response.status_code == 200
    assert response.json()["updated_ids"] == []
    assert response.json()["missing_ids"] == ["not-a-uuid"]
    assert pool.fetch_calls == []
    assert pool.execute_calls == []


def test_generated_asset_router_batch_review_enforces_configured_cap() -> None:
    response = _client(
        _Pool(),
        config=GeneratedAssetApiConfig(max_batch_size=1),
    ).post(
        "/content-assets/report/drafts/review-batch",
        json={"ids": ["report-uuid-1", "report-uuid-2"], "status": "approved"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "batch size exceeds max_batch_size (1)"


def test_generated_asset_router_batch_review_rejects_empty_ids() -> None:
    response = _client(_Pool()).post(
        "/content-assets/report/drafts/review-batch",
        json={"ids": ["", "  "], "status": "approved"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "ids must be a non-empty list"


def test_generated_asset_router_rejects_unknown_asset() -> None:
    response = _client(_Pool()).get("/content-assets/podcast_episode/drafts")

    assert response.status_code == 400
    assert "asset must be one of" in response.json()["detail"]


def test_generated_asset_router_rejects_unknown_asset_before_pool_resolution() -> None:
    app = FastAPI()
    calls = 0

    async def pool_provider():
        nonlocal calls
        calls += 1
        raise AssertionError("pool provider should not be touched")

    app.include_router(create_generated_asset_router(pool_provider=pool_provider))

    response = TestClient(app).get("/content-assets/podcast_episode/drafts")

    assert response.status_code == 400
    assert "asset must be one of" in response.json()["detail"]
    assert calls == 0


def test_generated_asset_router_rejects_empty_review_status() -> None:
    response = _client(_Pool()).post(
        "/content-assets/report/drafts/review",
        json={"id": "report-uuid-1", "status": ""},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "status is required"


def test_generated_asset_router_rejects_unknown_export_format() -> None:
    response = _client(_Pool(rows=[_report_row()])).get(
        "/content-assets/report/drafts/export?format=xlsx"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "format must be csv, json, or html"


def test_generated_asset_router_rejects_visual_export_for_non_card_assets() -> None:
    response = _client(_Pool(rows=[_report_row()])).get(
        "/content-assets/report/drafts/export?format=html"
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "format html is only supported for quote_card and stat_card"
    )


def test_generated_asset_router_requires_database() -> None:
    response = _client(_Pool(initialized=False)).get("/content-assets/report/drafts")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable"


def test_generated_asset_router_honors_host_dependencies() -> None:
    pool = _Pool(rows=[_report_row()])

    def require_auth():
        raise HTTPException(status_code=403, detail="forbidden")

    response = _client(pool, dependencies=[Depends(require_auth)]).get(
        "/content-assets/report/drafts"
    )

    assert response.status_code == 403
    assert pool.fetch_calls == []


def test_public_landing_page_router_does_not_use_review_router_dependencies() -> None:
    pool = _Pool(rows=[])

    def require_auth():
        raise HTTPException(status_code=403, detail="forbidden")

    app = FastAPI()

    async def pool_provider():
        return pool

    app.include_router(
        create_generated_asset_router(
            pool_provider=pool_provider,
            dependencies=[Depends(require_auth)],
        )
    )
    app.include_router(create_public_landing_page_router(pool_provider=pool_provider))

    response = TestClient(app).get(
        "/content-assets/landing_page/public/"
        "11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 404
    assert len(pool.fetch_calls) == 1


def test_generated_asset_api_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_limit must be positive"):
        GeneratedAssetApiConfig(max_limit=0)

    with pytest.raises(ValueError, match="default_limit must be less"):
        GeneratedAssetApiConfig(default_limit=5, max_limit=4)

    with pytest.raises(ValueError, match="max_batch_size must be positive"):
        GeneratedAssetApiConfig(max_batch_size=0)

    with pytest.raises(ValueError, match="public_sitemap_limit must be positive"):
        GeneratedAssetApiConfig(public_sitemap_limit=0)


def test_generated_asset_router_requires_fastapi(monkeypatch) -> None:
    monkeypatch.setattr(asset_api, "_FASTAPI_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(RuntimeError, match="FastAPI is required"):
        create_generated_asset_router(pool_provider=lambda: None)

    with pytest.raises(RuntimeError, match="FastAPI is required"):
        create_public_landing_page_router(pool_provider=lambda: None)
