"""Route-level tests for truthful blog/report artifact fields."""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

import atlas_brain.api.b2b_tenant_dashboard as tenant_dashboard_api
import atlas_brain.api.b2b_evidence as evidence_api
import atlas_brain.api.blog_admin as blog_admin_api
from atlas_brain.auth.dependencies import AuthUser, require_auth
from atlas_brain.autonomous.tasks.b2b_blog_post_generation import PostBlueprint


def _auth_user() -> AuthUser:
    return AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )


def test_blog_admin_routes_return_truth_fields(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    created_at = datetime(2026, 3, 30, 18, 0, tzinfo=timezone.utc)
    draft_id = uuid4()
    list_row = {
        "id": draft_id,
        "slug": "switch-to-shopify-2026-03",
        "title": "Migration Guide: Why Teams Are Switching to Shopify",
        "topic_type": "migration_guide",
        "status": "rejected",
        "llm_model": "openai/gpt-oss-120b",
        "created_at": created_at,
        "published_at": None,
        "rejected_at": created_at,
        "rejection_reason": "unsupported_data_claim:Magento",
        "quality_score": 64,
        "quality_threshold": None,
        "blocker_count": 1,
        "warning_count": 2,
        "latest_failure_step": "quality_gate",
        "latest_error_code": "quality_gate_rejection",
        "latest_error_summary": "unsupported_data_claim:Magento",
        "unresolved_issue_count": 3,
        "data_context": {
            "latest_quality_audit": {
                "status": "fail",
                "boundary": "generation",
                "failure_explanation": {
                    "boundary": "generation",
                    "primary_blocker": "unsupported_data_claim:Magento",
                    "cause_type": "unsupported_claim",
                    "blocking_issues": ["unsupported_data_claim:Magento"],
                    "warnings": [],
                    "matched_groups": [],
                    "available_groups": [],
                    "missing_groups": [],
                    "required_proof_terms": [],
                    "used_proof_terms": [],
                    "unused_proof_terms": [],
                    "missing_inputs": [],
                    "missing_primary_inputs": [],
                    "context_sources": ["data_context"],
                },
            },
            "latest_first_pass_quality_audit": {
                "status": "fail",
                "boundary": "generation_first_pass",
                "failure_explanation": {
                    "boundary": "generation_first_pass",
                    "primary_blocker": "unsupported_data_claim:Magento",
                    "cause_type": "unsupported_claim",
                },
            },
        },
    }
    detail_row = {
        **list_row,
        "description": "Shopify migration guide",
        "tags": ["shopify", "migration"],
        "content": "# Heading\n\nBody",
        "charts": [],
        "data_context": {
            "vendor": "Shopify",
            "latest_quality_audit": list_row["data_context"]["latest_quality_audit"],
            "latest_first_pass_quality_audit": list_row["data_context"]["latest_first_pass_quality_audit"],
        },
        "cta": {
            "headline": "See the full migration brief",
            "body": "Get the full report before renewal.",
            "button_text": "Book briefing",
        },
        "reviewer_notes": "Needs cleanup",
        "source_report_date": created_at.date(),
        "seo_title": "SEO title",
        "seo_description": "SEO desc",
        "target_keyword": "switch to shopify",
        "secondary_keywords": ["shopify migration"],
        "faq": [],
        "related_slugs": [],
    }

    class Pool:
        is_initialized = True

        async def fetch(self, *_args):
            return [list_row]

        async def fetchrow(self, query, *_args):
            if "WHERE id = $1" in query:
                return detail_row
            return None

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        list_res = client.get("/admin/blog/drafts")
        detail_res = client.get(f"/admin/blog/drafts/{draft_id}")

    assert list_res.status_code == 200
    body = list_res.json()
    assert body[0]["status"] == "rejected"
    assert body[0]["latest_failure_step"] == "quality_gate"
    assert body[0]["unresolved_issue_count"] == 3
    assert body[0]["failure_explanation"]["cause_type"] == "unsupported_claim"
    assert body[0]["first_pass_failure_explanation"]["primary_blocker"] == "unsupported_data_claim:Magento"

    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["rejection_reason"] == "unsupported_data_claim:Magento"
    assert detail["latest_error_summary"] == "unsupported_data_claim:Magento"
    assert detail["unresolved_issue_count"] == 3
    assert detail["failure_explanation"]["primary_blocker"] == "unsupported_data_claim:Magento"
    assert detail["first_pass_failure_explanation"]["primary_blocker"] == "unsupported_data_claim:Magento"
    assert detail["first_pass_quality_audit"]["boundary"] == "generation_first_pass"
    assert detail["cta"]["headline"] == "See the full migration brief"


def test_blog_admin_summary_returns_quality_rollup(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetch(self, query, *_args):
            if "GROUP BY status" in query:
                return [{"status": "draft", "count": 2}, {"status": "rejected", "count": 1}]
            if "GROUP BY latest_error_summary" in query:
                return [{"reason": "unsupported_data_claim:Magento", "count": 2}]
            if "GROUP BY step" in query:
                return [{"step": "quality_gate", "count": 2}]
            return []

        async def fetchrow(self, query, *_args):
            if "WITH blog_state AS" in query:
                return {
                    "clean": 1,
                    "warning_only": 2,
                    "failing": 3,
                    "unresolved": 4,
                    "blocker_total": 5,
                    "warning_total": 6,
                }
            return None

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.get("/admin/blog/drafts/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["by_status"]["draft"] == 2
    assert body["quality"]["clean"] == 1
    assert body["quality"]["failing"] == 3
    assert body["quality"]["blocker_total"] == 5
    assert body["quality"]["top_blockers"][0]["reason"] == "unsupported_data_claim:Magento"
    assert body["quality"]["by_failure_step"][0]["step"] == "quality_gate"


def test_blog_quality_trends_returns_daily_rollups(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetch(self, query, *_args):
            if "WITH blocker_rows AS" in query:
                return [
                    {"day": "2026-03-29", "reason": "unsupported_data_claim:Magento", "cnt": 1},
                    {"day": "2026-03-30", "reason": "unsupported_data_claim:Magento", "cnt": 2},
                ]
            if "blocker_total" in query:
                return [
                    {"day": "2026-03-29", "blocker_total": 1},
                    {"day": "2026-03-30", "blocker_total": 2},
                ]
            return []

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.get("/admin/blog/quality-trends")

    assert response.status_code == 200
    body = response.json()
    assert body["top_blockers"][0]["reason"] == "unsupported_data_claim:Magento"
    assert body["totals_by_day"][1]["blocker_total"] == 2


def test_b2b_evidence_router_uses_b2b_trial_gate(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            if "SELECT MAX(as_of_date) AS as_of_date" in query:
                assert args == ("Salesforce", 90, date.today())
                return {"as_of_date": date(2026, 4, 1)}
            if "COUNT(*) AS total FROM b2b_vendor_witnesses" in query:
                assert args == ("Salesforce", 90, date(2026, 4, 1))
                return {"total": 1}
            return None

        async def fetch(self, query, *args):
            if "FROM b2b_vendor_witnesses" in query and "GROUP BY pain_category" not in query:
                assert args == ("Salesforce", 90, date(2026, 4, 1), 50, 0)
                return [
                    {
                        "witness_id": "w1",
                        "review_id": uuid4(),
                        "witness_type": "pain_signal",
                        "excerpt_text": "Switching due to slow support.",
                        "source": "g2",
                        "reviewed_at": datetime(2026, 4, 1, tzinfo=timezone.utc),
                        "reviewer_company": "Acme",
                        "reviewer_title": "VP IT",
                        "pain_category": "support",
                        "competitor": None,
                        "salience_score": 0.9,
                        "specificity_score": 0.8,
                        "selection_reason": "high_signal",
                        "signal_tags": ["support"],
                        "as_of_date": datetime(2026, 4, 1, tzinfo=timezone.utc).date(),
                    }
                ]
            if "GROUP BY pain_category, source, witness_type" in query:
                assert args == ("Salesforce", 90, date(2026, 4, 1))
            return []

    monkeypatch.setattr(evidence_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        "atlas_brain.services.vendor_registry.resolve_vendor_name",
        AsyncMock(return_value="Salesforce"),
    )

    with TestClient(app) as client:
        response = client.get("/b2b/evidence/witnesses?vendor_name=Salesforce")

    assert response.status_code == 200
    body = response.json()
    assert body["vendor_name"] == "Salesforce"
    assert body["as_of_date"] == "2026-04-01"
    assert body["analysis_window_days"] == 90
    assert body["total"] == 1
    assert body["witnesses"][0]["witness_id"] == "w1"


def test_b2b_evidence_witness_detail_uses_requested_snapshot(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            if "SELECT MAX(as_of_date) AS as_of_date" in query:
                assert args == ("Salesforce", 30, date(2026, 3, 31))
                return {"as_of_date": date(2026, 3, 30)}
            if "FROM b2b_vendor_witnesses w" in query:
                assert args == ("Salesforce", "w1", 30, date(2026, 3, 30))
                return {
                    "witness_id": "w1",
                    "review_id": uuid4(),
                    "witness_type": "pain_signal",
                    "excerpt_text": "Switching due to slow support.",
                    "source": "g2",
                    "reviewed_at": datetime(2026, 3, 28, tzinfo=timezone.utc),
                    "reviewer_company": "Acme",
                    "reviewer_title": "VP IT",
                    "pain_category": "support",
                    "competitor": None,
                    "salience_score": 0.9,
                    "specificity_score": 0.8,
                    "selection_reason": "high_signal",
                    "signal_tags": ["support"],
                    "as_of_date": date(2026, 3, 30),
                    "review_text": "Full review text",
                    "summary": "Summary",
                    "pros": None,
                    "cons": None,
                    "rating": 2,
                    "review_source": "g2",
                    "source_url": None,
                    "enrichment": None,
                    "reviewer_name": "Pat",
                    "enrichment_status": "completed",
                }
            return None

    monkeypatch.setattr(evidence_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        evidence_api,
        "resolve_vendor_name",
        AsyncMock(return_value="Salesforce"),
    )

    with TestClient(app) as client:
        response = client.get(
            "/b2b/evidence/witnesses/w1?vendor_name=Salesforce&as_of_date=2026-03-31&window_days=30"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["witness"]["witness_id"] == "w1"
    assert body["witness"]["as_of_date"] == "2026-03-30"


def test_b2b_evidence_router_rejects_invalid_as_of_date(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

    monkeypatch.setattr(evidence_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.get("/b2b/evidence/vault?vendor_name=Salesforce&as_of_date=2026-99-99")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid as_of_date; expected YYYY-MM-DD"


def test_b2b_evidence_trace_bounds_diff_lookup_to_target_date(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            if "FROM b2b_reasoning_synthesis" in query:
                return None
            if "FROM b2b_vendor_reasoning_packets" in query:
                return None
            if "FROM reasoning_evidence_diffs" in query:
                assert args == ("Salesforce", date(2026, 3, 31))
                return {
                    "computed_date": date(2026, 3, 30),
                    "confirmed_count": 4,
                    "contradicted_count": 1,
                    "novel_count": 2,
                    "missing_count": 0,
                    "diff_ratio": 0.25,
                    "decision": "stable",
                    "has_core_contradiction": False,
                }
            return None

        async def fetch(self, query, *_args):
            if "FROM b2b_vendor_witnesses" in query:
                return []
            return []

    monkeypatch.setattr(evidence_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        evidence_api,
        "resolve_vendor_name",
        AsyncMock(return_value="Salesforce"),
    )

    with TestClient(app) as client:
        response = client.get("/b2b/evidence/trace?vendor_name=Salesforce&as_of_date=2026-03-31")

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["has_diff"] is True
    assert body["trace"]["evidence_diff"]["decision"] == "stable"
    assert body["trace"]["evidence_diff"]["computed_date"] == "2026-03-30"


def test_push_to_crm_routes_high_intent_payload(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetch(self, *_args):
            return [
                {
                    "id": "sub-1",
                    "url": "https://example.com/webhook",
                    "secret": "secret",
                    "account_id": _auth_user().account_id,
                    "channel": "crm_hubspot",
                    "auth_header": None,
                }
            ]

    captured: dict[str, object] = {}
    log_push = AsyncMock()
    
    async def fake_deliver(pool, sub, event_type, envelope, payload_bytes, cfg):
        await log_push(pool, sub["id"], event_type, envelope)
        return True

    def fake_format_for_channel(channel, envelope):
        captured["channel"] = channel
        captured["envelope"] = envelope
        return b"{}"

    monkeypatch.setattr(tenant_dashboard_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        "atlas_brain.services.b2b.webhook_dispatcher._format_for_channel",
        fake_format_for_channel,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.webhook_dispatcher._deliver_single",
        fake_deliver,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.webhook_dispatcher._log_crm_push",
        log_push,
    )

    with TestClient(app) as client:
        response = client.post(
            "/b2b/tenant/push-to-crm",
            json={
                "opportunities": [
                    {
                        "company": "Acme",
                        "vendor": "Salesforce",
                        "urgency": 8.5,
                        "pain": "pricing",
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert response.json()["pushed"] == 1
    assert response.json()["failed"] == []
    assert captured["channel"] == "crm_hubspot"
    envelope = captured["envelope"]
    assert isinstance(envelope, dict)
    assert envelope["event"] == "high_intent_push"
    assert envelope["vendor"] == "Salesforce"
    assert envelope["data"]["company_name"] == "Acme"
    assert "company" not in envelope["data"]
    assert log_push.await_count == 1


def test_push_to_crm_skips_payloads_above_size_limit(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetch(self, *_args):
            return [
                {
                    "id": "sub-1",
                    "url": "https://example.com/webhook",
                    "secret": "secret",
                    "account_id": _auth_user().account_id,
                    "channel": "crm_hubspot",
                    "auth_header": None,
                }
            ]

    deliver = AsyncMock(return_value=True)
    log_push = AsyncMock()

    monkeypatch.setattr(tenant_dashboard_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        tenant_dashboard_api.settings.b2b_webhook,
        "max_payload_bytes",
        4,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.webhook_dispatcher._format_for_channel",
        lambda *_args, **_kwargs: b"oversized-payload",
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.webhook_dispatcher._deliver_single",
        deliver,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.webhook_dispatcher._log_crm_push",
        log_push,
    )

    with TestClient(app) as client:
        response = client.post(
            "/b2b/tenant/push-to-crm",
            json={
                "opportunities": [
                    {
                        "company": "Acme",
                        "vendor": "Salesforce",
                        "urgency": 8.5,
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert response.json()["pushed"] == 0
    assert response.json()["failed"] == [
        {"company": "Acme", "vendor": "Salesforce", "reason": "payload_too_large"}
    ]
    deliver.assert_not_awaited()
    log_push.assert_not_awaited()


def test_blog_quality_diagnostics_returns_grouped_failures(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetch(self, query, *_args):
            if "AS status_value" in query:
                return [{"status_value": "draft", "cnt": 1}, {"status_value": "rejected", "cnt": 2}]
            if "AS boundary" in query:
                return [{"boundary": "publish", "cnt": 2}]
            if "AS cause_type" in query:
                return [{"cause_type": "unsupported_claim", "cnt": 2}]
            if "AS reason" in query:
                return [{"reason": "unsupported_data_claim:Magento", "cnt": 2}]
            if "missing_input.input AS input" in query:
                return [{"input": "reasoning_anchor_examples", "cnt": 1}]
            if "AS topic_type" in query:
                return [{"topic_type": "migration_guide", "cnt": 2}]
            if "AS subject" in query:
                return [{"subject": "Shopify", "cnt": 2}]
            if "COALESCE(rejection_count, 0) AS rejection_count" in query:
                return [
                    {
                        "slug": "clickup-deep-dive-2026-04",
                        "status": "rejected",
                        "rejection_count": 3,
                        "rejected_at": datetime.now(timezone.utc) - timedelta(days=2),
                    },
                    {
                        "slug": "asana-deep-dive-2026-04",
                        "status": "rejected",
                        "rejection_count": 1,
                        "rejected_at": datetime.now(timezone.utc) - timedelta(hours=2),
                    },
                ]
            return []

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.get("/admin/blog/quality-diagnostics")

    assert response.status_code == 200
    body = response.json()
    assert body["active_failure_count"] == 1
    assert body["rejected_failure_count"] == 2
    assert body["current_blocked_slug_count"] == 2
    assert body["retry_limit_blocked_slug_count"] == 1
    assert body["cooldown_blocked_slug_count"] == 1
    assert body["by_status"][0]["status"] == "draft"
    assert body["by_boundary"][0]["boundary"] == "publish"
    assert body["by_cause_type"][0]["cause_type"] == "unsupported_claim"
    assert body["top_primary_blockers"][0]["reason"] == "unsupported_data_claim:Magento"
    assert body["by_topic_type"][0]["topic_type"] == "migration_guide"
    assert body["top_subjects"][0]["subject"] == "Shopify"
    assert body["top_blocked_slugs"][0]["slug"] == "clickup-deep-dive-2026-04"
    assert body["top_blocked_slugs"][0]["reason"] == "retry_limit"
    assert body["top_blocked_slugs"][1]["reason"] == "rejection_cooldown"


def test_blog_quality_diagnostics_status_counts_ignore_top_n_limit(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetch(self, query, *_args):
            if "AS status_value" in query:
                return [
                    {"status_value": "draft", "cnt": 1},
                    {"status_value": "rejected", "cnt": 2},
                ]
            if "AS boundary" in query:
                return [{"boundary": "publish", "cnt": 2}]
            if "AS cause_type" in query:
                return [{"cause_type": "unsupported_claim", "cnt": 2}]
            if "AS reason" in query:
                return [{"reason": "unsupported_data_claim:Magento", "cnt": 2}]
            if "missing_input.input AS input" in query:
                return [{"input": "reasoning_anchor_examples", "cnt": 1}]
            if "AS topic_type" in query:
                return [{"topic_type": "migration_guide", "cnt": 2}]
            if "AS subject" in query:
                return [{"subject": "Shopify", "cnt": 2}]
            if "COALESCE(rejection_count, 0) AS rejection_count" in query:
                return [
                    {
                        "slug": "clickup-deep-dive-2026-04",
                        "status": "rejected",
                        "rejection_count": 3,
                        "rejected_at": datetime.now(timezone.utc) - timedelta(days=2),
                    },
                    {
                        "slug": "asana-deep-dive-2026-04",
                        "status": "rejected",
                        "rejection_count": 1,
                        "rejected_at": datetime.now(timezone.utc) - timedelta(hours=2),
                    },
                ]
            return []

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.get("/admin/blog/quality-diagnostics?top_n=1")

    assert response.status_code == 200
    body = response.json()
    assert body["active_failure_count"] == 1
    assert body["rejected_failure_count"] == 2
    assert body["current_blocked_slug_count"] == 2
    assert len(body["by_status"]) == 2
    assert len(body["top_blocked_slugs"]) == 1


def test_blog_publish_route_blocks_failed_revalidation(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    draft_id = uuid4()
    created_at = datetime(2026, 3, 30, 18, 0, tzinfo=timezone.utc)
    execute_calls = []
    row = {
        "id": draft_id,
        "slug": "switch-to-shopify-2026-03",
        "title": "Migration Guide: Why Teams Are Switching to Shopify",
        "description": "Shopify migration guide",
        "topic_type": "migration_guide",
        "status": "draft",
        "llm_model": "anthropic/claude-sonnet-4-5",
        "created_at": created_at,
        "published_at": None,
        "rejected_at": None,
        "rejection_reason": None,
        "quality_score": None,
        "quality_threshold": None,
        "blocker_count": 0,
        "warning_count": 0,
        "latest_failure_step": None,
        "latest_error_code": None,
        "latest_error_summary": None,
        "tags": ["shopify", "migration"],
        "content": "<p>There is broad migration pressure in the market.</p>",
        "charts": [],
        "data_context": {
            "vendor": "Shopify",
            "reasoning_anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "witness:r1:0",
                        "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "BigCommerce",
                        "pain_category": "pricing",
                    },
                ],
            },
            "reasoning_witness_highlights": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "BigCommerce",
                    "pain_category": "pricing",
                },
            ],
            "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
        },
        "reviewer_notes": None,
        "source_report_date": created_at.date(),
    }

    class Pool:
        is_initialized = True

        async def fetch(self, *_args):
            return []

        async def fetchrow(self, query, *_args):
            if "SELECT * FROM blog_posts WHERE id = $1" in query:
                return row
            return None

        async def execute(self, query, *args):
            execute_calls.append((query, args))
            return "UPDATE 1"

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(blog_admin_api, "record_attempt", AsyncMock())
    monkeypatch.setattr(blog_admin_api, "emit_event", AsyncMock())

    with TestClient(app) as client:
        response = client.post(f"/admin/blog/drafts/{draft_id}/publish")

    assert response.status_code == 409
    assert "witness_specificity:" in response.json()["detail"] or "content_too_short" in response.json()["detail"]
    assert any("data_context = $1::jsonb" in call[0] and "latest_failure_step = $2" in call[0] for call in execute_calls)


def test_blog_publish_route_blocks_unresolved_critical_warnings(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    draft_id = uuid4()
    created_at = datetime(2026, 3, 30, 18, 0, tzinfo=timezone.utc)
    execute_calls = []
    row = {
        "id": draft_id,
        "slug": "switch-to-shopify-2026-03",
        "title": "Migration Guide: Why Teams Are Switching to Shopify",
        "description": "Shopify migration guide",
        "topic_type": "migration_guide",
        "status": "draft",
        "llm_model": "anthropic/claude-sonnet-4-5",
        "created_at": created_at,
        "published_at": None,
        "rejected_at": None,
        "rejection_reason": None,
        "quality_score": None,
        "quality_threshold": None,
        "blocker_count": 0,
        "warning_count": 0,
        "latest_failure_step": None,
        "latest_error_code": None,
        "latest_error_summary": None,
        "tags": ["shopify", "migration"],
        "content": "<p>Body</p>",
        "charts": [],
        "data_context": {"vendor": "Shopify"},
        "reviewer_notes": None,
        "source_report_date": created_at.date(),
    }

    class Pool:
        is_initialized = True

        async def fetch(self, *_args):
            return []

        async def fetchrow(self, query, *_args):
            if "SELECT * FROM blog_posts WHERE id = $1" in query:
                return row
            return None

        async def execute(self, query, *args):
            execute_calls.append((query, args))
            return "UPDATE 1"

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(blog_admin_api, "record_attempt", AsyncMock())
    monkeypatch.setattr(blog_admin_api, "emit_event", AsyncMock())

    def _fake_quality_gate(_blueprint, content):
        return dict(content), {
            "status": "pass",
            "score": 82,
            "threshold": 70,
            "blocking_issues": [],
            "warnings": ["unsupported_data_claim:Magento"],
        }

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._apply_blog_quality_gate",
        _fake_quality_gate,
    )

    with TestClient(app) as client:
        response = client.post(f"/admin/blog/drafts/{draft_id}/publish")

    assert response.status_code == 409
    assert "critical_warning_unresolved:unsupported_data_claim:Magento" in response.json()["detail"]
    assert any("data_context = $1::jsonb" in call[0] and "latest_failure_step = $2" in call[0] for call in execute_calls)


def test_tenant_report_routes_return_truth_fields(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    user = _auth_user()
    app.dependency_overrides[require_auth] = lambda: user

    report_id = uuid4()
    created_at = datetime(2026, 3, 30, 18, 0, tzinfo=timezone.utc)
    list_row = {
        "id": report_id,
        "report_date": created_at.date(),
        "report_type": "challenger_brief",
        "executive_summary": "Summary",
        "vendor_filter": "Shopify",
        "category_filter": "Ecommerce",
        "status": "failed",
        "created_at": created_at,
        "latest_failure_step": "no_data",
        "latest_error_code": "no_data",
        "latest_error_summary": "Challenger brief has no displacement mentions",
        "blocker_count": 1,
        "warning_count": 0,
        "unresolved_issue_count": 2,
        "quality_status": None,
        "quality_score": None,
    }
    detail_row = {
        "id": report_id,
        "report_date": created_at.date(),
        "report_type": "challenger_brief",
        "vendor_filter": "Shopify",
        "category_filter": "Ecommerce",
        "executive_summary": "Summary",
        "intelligence_data": {"data_stale": False},
        "data_density": {"sources_present": 0},
        "status": "failed",
        "latest_failure_step": "no_data",
        "latest_error_code": "no_data",
        "latest_error_summary": "Challenger brief has no displacement mentions",
        "blocker_count": 1,
        "warning_count": 0,
        "llm_model": "pipeline_deterministic",
        "created_at": created_at,
        "account_id": None,
    }

    class Pool:
        async def fetch(self, *_args):
            return [list_row]

        async def fetchrow(self, query, *_args):
            if "SELECT * FROM b2b_intelligence WHERE id = $1" in query:
                return detail_row
            return None

        async def fetchval(self, query, *_args):
            if "SELECT COUNT(*)" in query:
                return 2
            return None

    monkeypatch.setattr(tenant_dashboard_api, "_pool_or_503", lambda: Pool())

    with TestClient(app) as client:
        list_res = client.get("/b2b/tenant/reports")
        detail_res = client.get(f"/b2b/tenant/reports/{report_id}")

    assert list_res.status_code == 200
    list_body = list_res.json()
    assert list_body["reports"][0]["status"] == "failed"
    assert list_body["reports"][0]["latest_failure_step"] == "no_data"
    assert list_body["reports"][0]["unresolved_issue_count"] == 2

    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["status"] == "failed"
    assert detail["latest_error_summary"] == "Challenger brief has no displacement mentions"
    assert detail["unresolved_issue_count"] == 2


def test_blog_manual_generate_persists_first_pass_audit(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

    record_attempt = AsyncMock()
    llm = object()
    captured = {}

    async def _fake_build_manual_topic_ctx(_pool, vendor_name, topic_type, vendor_b=None, category=None):
        assert vendor_name == "Jira"
        assert topic_type == "vendor_deep_dive"
        assert vendor_b is None
        assert category is None
        return {"vendor": vendor_name, "category": "Project Management", "slug": "jira-deep-dive-2026-04"}

    async def _fake_gather_data(_pool, _topic_type, _topic_ctx):
        return {"data_context": {}}

    async def _fake_load_pool_layers_for_blog(_pool, _topic_type, _topic_ctx, data):
        data["data_context"]["reasoning_anchor_examples"] = {
            "outlier_or_named_account": [{"excerpt_text": "Named account switching at renewal"}]
        }
        data["data_context"]["reasoning_witness_highlights"] = [
            {"excerpt_text": "Switching trigger surfaced in review evidence"}
        ]
        data["data_context"]["reasoning_reference_ids"] = {
            "vendor_core_reasoning": ["review:1"]
        }

    def _fake_check_data_sufficiency(_topic_type, _data):
        return {"sufficient": True}

    def _fake_build_blueprint(_topic_type, topic_ctx, data):
        return PostBlueprint(
            topic_type="vendor_deep_dive",
            slug=topic_ctx["slug"],
            suggested_title="Jira Deep Dive",
            tags=["jira"],
            data_context={"topic_ctx": dict(topic_ctx), **dict(data.get("data_context") or {})},
            sections=[],
            charts=[],
        )

    async def _fake_generate_content_async(_llm, _blueprint, _max_tokens, **_kwargs):
        return {
            "title": "Jira",
            "description": "Jira deep dive",
            "content": "# Jira\n\n" + " ".join(["Jira"] * 2200),
        }

    async def _fake_enforce_blog_quality_async(_llm, _blueprint, content, _max_tokens, **_kwargs):
        return content, {
            "status": "pass",
            "score": 86,
            "threshold": 70,
            "blocking_issues": [],
            "warnings": ["too_few_sourced_quotes"],
            "_retry_requested": True,
            "_first_pass_report": {
                "status": "fail",
                "score": 61,
                "threshold": 70,
                "blocking_issues": ["unsupported_data_claim:Magento"],
                "warnings": ["too_few_sourced_quotes"],
                "failure_explanation": {
                    "boundary": "generation_first_pass",
                    "primary_blocker": "unsupported_data_claim:Magento",
                },
            },
        }

    async def _fake_assemble_and_store(_pool, blueprint, _content, _llm, *, run_id=None, attempt_no=None):
        captured["data_context"] = dict(blueprint.data_context)
        captured["run_id"] = run_id
        captured["attempt_no"] = attempt_no
        return "post-123"

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(blog_admin_api, "record_attempt", record_attempt)
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **_kwargs: llm,
    )
    monkeypatch.setattr(
        "atlas_brain.api.blog_admin.settings",
        SimpleNamespace(
            b2b_churn=SimpleNamespace(blog_post_max_tokens=4096, blog_post_openrouter_model="test-model")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation.build_manual_topic_ctx",
        _fake_build_manual_topic_ctx,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._gather_data",
        _fake_gather_data,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_data_sufficiency",
        _fake_check_data_sufficiency,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._load_pool_layers_for_blog",
        _fake_load_pool_layers_for_blog,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._build_blueprint",
        _fake_build_blueprint,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_blueprint_sufficiency",
        lambda *_args, **_kwargs: {"sufficient": True},
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._generate_content_async",
        _fake_generate_content_async,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._enforce_blog_quality_async",
        _fake_enforce_blog_quality_async,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._get_blog_slug_block_reason",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._assemble_and_store",
        _fake_assemble_and_store,
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/blog/generate",
            json={"vendor_name": "Jira", "topic_type": "vendor_deep_dive"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["post_id"] == "post-123"
    assert body["first_pass_failure_explanation"]["primary_blocker"] == "unsupported_data_claim:Magento"
    assert captured["attempt_no"] == 1
    assert captured["run_id"]
    assert captured["data_context"]["latest_first_pass_quality_audit"]["boundary"] == "generation_first_pass"
    assert captured["data_context"]["reasoning_anchor_examples"]["outlier_or_named_account"][0]["excerpt_text"] == "Named account switching at renewal"
    assert captured["data_context"]["reasoning_reference_ids"]["vendor_core_reasoning"] == ["review:1"]
    record_attempt.assert_awaited_once()
    assert record_attempt.await_args.kwargs["stage"] == "quality_gate_first_pass"


def test_blog_manual_generate_backfills_missing_length_fields(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

    llm = object()
    captured = {}
    article_body = "# SentinelOne\n\n" + " ".join(["SentinelOne"] * 1700)

    async def _fake_build_manual_topic_ctx(_pool, vendor_name, topic_type, vendor_b=None, category=None):
        assert vendor_name == "SentinelOne"
        assert topic_type == "migration_guide"
        assert vendor_b is None
        assert category == "Security"
        return {
            "vendor": vendor_name,
            "category": category,
            "slug": "switch-to-sentinelone-2026-04",
        }

    async def _fake_gather_data(_pool, _topic_type, _topic_ctx):
        return {"data_context": {"vendor": "SentinelOne", "category": "Security"}}

    async def _fake_load_pool_layers_for_blog(_pool, _topic_type, _topic_ctx, data):
        data["data_context"]["reasoning_anchor_examples"] = {
            "counterevidence": [{"excerpt_text": "Pricing concern surfaced before renewal"}]
        }
        data["data_context"]["reasoning_witness_highlights"] = [
            {"excerpt_text": "Teams switched after contract review"}
        ]
        data["data_context"]["reasoning_reference_ids"] = {
            "vendor_core_reasoning": ["review:sentinelone:1"]
        }

    def _fake_check_data_sufficiency(_topic_type, _data):
        return {"sufficient": True}

    def _fake_build_blueprint(_topic_type, topic_ctx, data):
        return PostBlueprint(
            topic_type="migration_guide",
            slug=topic_ctx["slug"],
            suggested_title="Switch to SentinelOne",
            tags=["sentinelone", "migration"],
            data_context={"topic_ctx": dict(topic_ctx), **dict(data.get("data_context") or {})},
            sections=[],
            charts=[],
        )

    async def _fake_generate_content_async(_llm, _blueprint, _max_tokens, **_kwargs):
        return {
            "title": "Switch to SentinelOne",
            "description": "Migration guide",
            "content": article_body,
        }

    async def _fake_enforce_blog_quality_async(_llm, _blueprint, content, _max_tokens, **_kwargs):
        return content, {
            "status": "pass",
            "score": 94,
            "threshold": 70,
            "blocking_issues": [],
            "warnings": ["content_below_seo_target_2500_words"],
            "fixes_applied": ["removed_unsupported_claim_lines:2"],
            "_retry_requested": True,
            "_first_pass_report": {
                "status": "fail",
                "score": 63,
                "threshold": 70,
                "blocking_issues": ["content_too_short:1340_words_need_2000"],
                "warnings": [],
            },
        }

    async def _fake_assemble_and_store(_pool, blueprint, _content, _llm, *, run_id=None, attempt_no=None):
        captured["data_context"] = dict(blueprint.data_context)
        captured["run_id"] = run_id
        captured["attempt_no"] = attempt_no
        return "post-456"

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **_kwargs: llm,
    )
    monkeypatch.setattr(
        "atlas_brain.api.blog_admin.settings",
        SimpleNamespace(
            b2b_churn=SimpleNamespace(blog_post_max_tokens=4096, blog_post_openrouter_model="test-model")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation.build_manual_topic_ctx",
        _fake_build_manual_topic_ctx,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._gather_data",
        _fake_gather_data,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_data_sufficiency",
        _fake_check_data_sufficiency,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._load_pool_layers_for_blog",
        _fake_load_pool_layers_for_blog,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._build_blueprint",
        _fake_build_blueprint,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_blueprint_sufficiency",
        lambda *_args, **_kwargs: {"sufficient": True},
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._generate_content_async",
        _fake_generate_content_async,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._enforce_blog_quality_async",
        _fake_enforce_blog_quality_async,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._get_blog_slug_block_reason",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._assemble_and_store",
        _fake_assemble_and_store,
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/blog/generate",
            json={
                "vendor_name": "SentinelOne",
                "topic_type": "migration_guide",
                "category": "Security",
            },
        )

    assert response.status_code == 200
    assert captured["attempt_no"] == 1
    assert captured["run_id"]
    assert captured["data_context"]["generation_length_policy"]["min_words"] == 1500
    assert captured["data_context"]["generation_length_policy"]["target_words"] == 2100
    audit = captured["data_context"]["latest_quality_audit"]
    assert audit["boundary"] == "manual_generate"
    assert audit["word_count"] == len(article_body.split())
    assert audit["min_words_required"] == 1500
    assert audit["target_words"] == 2100
    assert "content_below_seo_target_2100_words" in audit["warnings"]
    assert "content_below_seo_target_2500_words" not in audit["warnings"]


def test_manual_blog_generate_blocks_recent_rejected_slug(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **_kwargs: SimpleNamespace(model_name="anthropic/test"),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation.build_manual_topic_ctx",
        AsyncMock(return_value={"vendor": "Jira", "category": "Project Management", "slug": "jira-deep-dive-2026-04"}),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._gather_data",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._load_pool_layers_for_blog",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_data_sufficiency",
        lambda *_args, **_kwargs: {"sufficient": True},
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._build_blueprint",
        lambda *_args, **_kwargs: PostBlueprint(
            topic_type="vendor_deep_dive",
            slug="jira-deep-dive-2026-04",
            suggested_title="Jira Deep Dive",
            tags=["jira"],
            data_context={"vendor": "Jira"},
            sections=[],
            charts=[],
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_blueprint_sufficiency",
        lambda *_args, **_kwargs: {"sufficient": True},
    )
    generate_content = AsyncMock(return_value={"title": "ignored"})
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._generate_content_async",
        generate_content,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._get_blog_slug_block_reason",
        AsyncMock(return_value="rejection_cooldown"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/blog/generate",
            json={"vendor_name": "Jira", "topic_type": "vendor_deep_dive"},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["block_reason"] == "rejection_cooldown"
    assert response.json()["detail"]["requires_force_retry"] is True
    assert response.json()["detail"]["cooldown_active"] is True
    generate_content.assert_not_awaited()


def test_manual_blog_generate_persists_quality_rejection(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **_kwargs: SimpleNamespace(model_name="anthropic/test"),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation.build_manual_topic_ctx",
        AsyncMock(return_value={"vendor": "Jira", "category": "Project Management", "slug": "jira-deep-dive-2026-04"}),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._gather_data",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._load_pool_layers_for_blog",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_data_sufficiency",
        lambda *_args, **_kwargs: {"sufficient": True},
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._build_blueprint",
        lambda *_args, **_kwargs: PostBlueprint(
            topic_type="vendor_deep_dive",
            slug="jira-deep-dive-2026-04",
            suggested_title="Jira Deep Dive",
            tags=["jira"],
            data_context={"vendor": "Jira"},
            sections=[],
            charts=[],
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_blueprint_sufficiency",
        lambda *_args, **_kwargs: {"sufficient": True},
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._get_blog_slug_block_reason",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._generate_content_async",
        AsyncMock(return_value={"title": "Jira Deep Dive", "content": "body"}),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._enforce_blog_quality_async",
        AsyncMock(return_value=(None, {"_retry_requested": False, "_rejected_content": {"title": "Jira Deep Dive", "content": "rejected draft body"}})),
    )
    upsert = AsyncMock(return_value="row-1")
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._upsert_blog_post_state",
        upsert,
    )
    monkeypatch.setattr(blog_admin_api, "record_attempt", AsyncMock())
    monkeypatch.setattr(blog_admin_api, "emit_event", AsyncMock())

    with TestClient(app) as client:
        response = client.post(
            "/admin/blog/generate",
            json={"vendor_name": "Jira", "topic_type": "vendor_deep_dive"},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "Generated content failed quality gate"
    assert upsert.await_args.kwargs["status"] == "rejected"
    assert upsert.await_args.kwargs["failure_step"] == "quality_gate"
    assert upsert.await_args.kwargs["content"]["content"] == "rejected draft body"
    blog_admin_api.record_attempt.assert_awaited()
    blog_admin_api.emit_event.assert_awaited()


def test_manual_blog_generate_blocks_insufficient_blueprint(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **_kwargs: SimpleNamespace(model_name="anthropic/test"),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation.build_manual_topic_ctx",
        AsyncMock(return_value={"vendor": "Jira", "category": "Project Management", "slug": "jira-deep-dive-2026-04"}),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._gather_data",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._load_pool_layers_for_blog",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_data_sufficiency",
        lambda *_args, **_kwargs: {"sufficient": True},
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._build_blueprint",
        lambda *_args, **_kwargs: PostBlueprint(
            topic_type="vendor_deep_dive",
            slug="jira-deep-dive-2026-04",
            suggested_title="Jira Deep Dive",
            tags=["jira"],
            data_context={"vendor": "Jira"},
            sections=[],
            charts=[],
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation._check_blueprint_sufficiency",
        lambda *_args, **_kwargs: {"sufficient": False, "reason": "Only 4 blueprint sections (need 6+)"},
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/blog/generate",
            json={"vendor_name": "Jira", "topic_type": "vendor_deep_dive"},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "Insufficient blueprint coverage"
    assert response.json()["detail"]["reason"] == "Only 4 blueprint sections (need 6+)"
