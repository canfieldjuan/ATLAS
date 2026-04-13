"""Route-level tests for truthful blog/report artifact fields."""

import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

import atlas_brain.api.b2b_tenant_dashboard as tenant_dashboard_api
import atlas_brain.api.b2b_campaigns as campaigns_api
import atlas_brain.api.b2b_dashboard as dashboard_api
import atlas_brain.api.b2b_evidence as evidence_api
import atlas_brain.api.blog_admin as blog_admin_api
from atlas_brain.auth.dependencies import AuthUser, require_auth
from atlas_brain.autonomous.tasks.b2b_blog_post_generation import PostBlueprint


def _auth_user(plan: str = "b2b_pro") -> AuthUser:
    return AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan=plan,
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
    user = _auth_user()
    app.dependency_overrides[require_auth] = lambda: user

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            if "SELECT MAX(as_of_date) AS as_of_date" in query:
                assert args == ("Salesforce", 30, date.today())
                return {"as_of_date": date(2026, 4, 1)}
            if "COUNT(*) AS total FROM b2b_vendor_witnesses" in query:
                assert "FROM b2b_vendor_witnesses w" in query
                assert "WHERE w.vendor_name = $1" in query
                assert "WHERE vendor_name = $1" not in query
                assert args == (
                    "Salesforce",
                    30,
                    date(2026, 4, 1),
                    evidence_api._uuid.UUID(user.account_id),
                )
                return {"total": 1}
            return None

        async def fetch(self, query, *args):
            if "LIMIT $5 OFFSET $6" in query:
                assert "FROM b2b_vendor_witnesses w" in query
                assert "WHERE w.vendor_name = $1" in query
                assert "WHERE vendor_name = $1" not in query
                assert args == (
                    "Salesforce",
                    30,
                    date(2026, 4, 1),
                    evidence_api._uuid.UUID(user.account_id),
                    50,
                    0,
                )
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
            if "GROUP BY w.pain_category, w.source, w.witness_type" in query:
                assert args == (
                    "Salesforce",
                    30,
                    date(2026, 4, 1),
                    evidence_api._uuid.UUID(user.account_id),
                )
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
    assert body["analysis_window_days"] == 30
    assert body["total"] == 1
    assert body["witnesses"][0]["witness_id"] == "w1"


def test_b2b_evidence_annotations_list_uses_b2b_gate_and_canonical_vendor(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    user = _auth_user()
    app.dependency_overrides[require_auth] = lambda: user

    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            assert "FROM b2b_evidence_annotations" in query
            assert args == (
                evidence_api._uuid.UUID(user.account_id),
                "Salesforce",
                "pin",
            )
            return [
                {
                    "id": uuid4(),
                    "witness_id": "w1",
                    "vendor_name": "Salesforce",
                    "annotation_type": "pin",
                    "note_text": None,
                    "created_at": datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 4, 8, 12, 5, tzinfo=timezone.utc),
                }
            ]

    monkeypatch.setattr(evidence_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        evidence_api,
        "resolve_vendor_name",
        AsyncMock(return_value="Salesforce"),
    )

    with TestClient(app) as client:
        response = client.get("/b2b/evidence/annotations?vendor_name=salesforce&annotation_type=pin")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["annotations"][0]["vendor_name"] == "Salesforce"
    assert body["annotations"][0]["annotation_type"] == "pin"


def test_b2b_evidence_set_annotation_validates_witness_and_upserts(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    user = _auth_user()
    app.dependency_overrides[require_auth] = lambda: user

    class Pool:
        is_initialized = True

        async def fetchval(self, query, *args):
            assert "FROM b2b_vendor_witnesses" in query
            assert args == ("w1", "Salesforce")
            return 1

        async def fetchrow(self, query, *args):
            assert "INSERT INTO b2b_evidence_annotations" in query
            assert args[1] == evidence_api._uuid.UUID(user.account_id)
            assert args[2] == "w1"
            assert args[3] == "Salesforce"
            assert args[4] == "pin"
            return {
                "id": uuid4(),
                "witness_id": "w1",
                "vendor_name": "Salesforce",
                "annotation_type": "pin",
                "note_text": None,
                "created_at": datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
            }

    monkeypatch.setattr(evidence_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        evidence_api,
        "resolve_vendor_name",
        AsyncMock(return_value="Salesforce"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/b2b/evidence/annotations",
            json={
                "witness_id": "w1",
                "vendor_name": "salesforce",
                "annotation_type": "pin",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["witness_id"] == "w1"
    assert body["vendor_name"] == "Salesforce"
    assert body["annotation_type"] == "pin"


def test_b2b_evidence_set_annotation_rejects_blank_text_before_db_touch(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    resolve_vendor = AsyncMock(return_value="Salesforce")
    monkeypatch.setattr(evidence_api, "get_db_pool", _boom)
    monkeypatch.setattr(evidence_api, "resolve_vendor_name", resolve_vendor)

    with TestClient(app) as client:
        blank_witness = client.post(
            "/b2b/evidence/annotations",
            json={
                "witness_id": "   ",
                "vendor_name": "salesforce",
                "annotation_type": "pin",
            },
        )
        blank_vendor = client.post(
            "/b2b/evidence/annotations",
            json={
                "witness_id": "w1",
                "vendor_name": "   ",
                "annotation_type": "pin",
            },
        )

    assert blank_witness.status_code == 422
    assert blank_witness.json()["detail"] == "witness_id is required"
    assert blank_vendor.status_code == 422
    assert blank_vendor.json()["detail"] == "vendor_name is required"
    resolve_vendor.assert_not_awaited()


def test_b2b_evidence_remove_annotations_ignores_blank_ids_before_db_touch(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    monkeypatch.setattr(evidence_api, "get_db_pool", _boom)

    with TestClient(app) as client:
        response = client.post(
            "/b2b/evidence/annotations/remove",
            json={"witness_ids": ["   ", "\t"]},
        )

    assert response.status_code == 200
    assert response.json() == {"removed": 0}


def test_b2b_evidence_set_annotation_rejects_unknown_witness(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetchval(self, query, *args):
            assert "FROM b2b_vendor_witnesses" in query
            assert args == ("missing", "Salesforce")
            return None

    monkeypatch.setattr(evidence_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        evidence_api,
        "resolve_vendor_name",
        AsyncMock(return_value="Salesforce"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/b2b/evidence/annotations",
            json={
                "witness_id": "missing",
                "vendor_name": "salesforce",
                "annotation_type": "flag",
            },
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Witness not found for vendor"


def test_b2b_evidence_list_witnesses_rejects_invalid_as_of_date_before_db_touch(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    resolve_vendor = AsyncMock(return_value="Salesforce")
    monkeypatch.setattr(evidence_api, "get_db_pool", _boom)
    monkeypatch.setattr(evidence_api, "resolve_vendor_name", resolve_vendor)

    with TestClient(app) as client:
        response = client.get("/b2b/evidence/witnesses?vendor_name=Salesforce&as_of_date=2026-99-99")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid as_of_date; expected YYYY-MM-DD"
    resolve_vendor.assert_not_awaited()


def test_b2b_evidence_read_routes_reject_blank_vendor_before_db_touch(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    resolve_vendor = AsyncMock(return_value="Salesforce")
    monkeypatch.setattr(evidence_api, "get_db_pool", _boom)
    monkeypatch.setattr(evidence_api, "resolve_vendor_name", resolve_vendor)

    with TestClient(app) as client:
        witness_response = client.get("/b2b/evidence/witnesses?vendor_name=   ")
        vault_response = client.get("/b2b/evidence/vault?vendor_name=   ")

    assert witness_response.status_code == 422
    assert witness_response.json()["detail"] == "vendor_name is required"
    assert vault_response.status_code == 422
    assert vault_response.json()["detail"] == "vendor_name is required"
    resolve_vendor.assert_not_awaited()


def test_b2b_evidence_list_annotations_normalizes_blank_optional_filters(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    user = _auth_user()
    app.dependency_overrides[require_auth] = lambda: user

    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            assert "vendor_name = $" not in query
            assert "annotation_type = $" not in query
            assert args == (evidence_api._uuid.UUID(user.account_id),)
            return []

    resolve_vendor = AsyncMock(return_value="Salesforce")
    monkeypatch.setattr(evidence_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(evidence_api, "resolve_vendor_name", resolve_vendor)

    with TestClient(app) as client:
        response = client.get(
            "/b2b/evidence/annotations",
            params={"vendor_name": "   ", "annotation_type": "   "},
        )

    assert response.status_code == 200
    assert response.json() == {"annotations": [], "count": 0}
    resolve_vendor.assert_not_awaited()


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


def test_b2b_evidence_vault_route_uses_shared_vendor_intelligence_reader(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            assert "FROM b2b_vendor_witnesses" in query
            assert args == ("Salesforce", date(2026, 3, 30), 30)
            return {"total": 4}

    reader = AsyncMock(
        return_value={
            "vendor_name": "Salesforce",
            "as_of_date": date(2026, 3, 30),
            "analysis_window_days": 30,
            "schema_version": 2,
            "created_at": datetime(2026, 3, 30, 18, 0, tzinfo=timezone.utc),
            "vault": {
                "weakness_evidence": [{"category": "pricing"}],
                "strength_evidence": [{"category": "ecosystem"}],
                "company_signals": [{"company_name": "Acme"}],
                "metric_snapshot": {"avg_urgency": 7.1},
                "provenance": {"sources": ["g2"]},
            },
        },
    )

    monkeypatch.setattr(evidence_api, "get_db_pool", lambda: Pool())
    monkeypatch.setattr(
        evidence_api,
        "resolve_vendor_name",
        AsyncMock(return_value="Salesforce"),
    )
    monkeypatch.setattr(evidence_api, "_read_vendor_intelligence_record", reader)

    with TestClient(app) as client:
        response = client.get(
            "/b2b/evidence/vault?vendor_name=Salesforce&as_of_date=2026-03-31&window_days=30"
        )

    assert response.status_code == 200
    reader.assert_awaited_once()
    body = response.json()
    assert body["vendor_name"] == "Salesforce"
    assert body["as_of_date"] == "2026-03-30"
    assert body["witness_count"] == 4
    assert body["metric_snapshot"]["avg_urgency"] == 7.1


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


def test_b2b_evidence_trace_reads_prompt_payload_packet_counts(monkeypatch):
    app = FastAPI()
    app.include_router(evidence_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            if "FROM b2b_reasoning_synthesis" in query:
                return None
            if "FROM b2b_vendor_reasoning_packets" in query:
                return {
                    "vendor_name": "Salesforce",
                    "as_of_date": date(2026, 3, 30),
                    "analysis_window_days": 30,
                    "schema_version": "witness_packet_v1",
                    "evidence_hash": "hash-1",
                    "packet": {
                        "payload": {
                            "witness_pack": [{"witness_id": "full-1"}],
                            "section_packets": {
                                "anchor_examples": {"common_pattern": ["full-1", "full-2"]},
                                "segment_packet": {},
                            },
                        },
                        "prompt_payload": {
                            "witness_pack": [{"witness_id": "prompt-1"}],
                            "section_packets": {
                                "anchor_examples": {"common_pattern": ["prompt-1"]},
                            },
                        },
                    },
                    "created_at": datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc),
                }
            if "FROM reasoning_evidence_diffs" in query:
                return None
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
    assert body["trace"]["reasoning_packet"]["witness_pack_size"] == 1
    assert body["trace"]["reasoning_packet"]["section_count"] == 1


def test_campaign_outcome_routes_require_growth_plan():
    app = FastAPI()
    app.include_router(campaigns_api.router)
    app.dependency_overrides[require_auth] = lambda: _auth_user(plan="b2b_trial")
    sequence_id = uuid4()

    with TestClient(app) as client:
        post_response = client.post(
            f"/b2b/campaigns/sequences/{sequence_id}/outcome",
            json={"outcome": "meeting_booked"},
        )
        get_response = client.get(f"/b2b/campaigns/sequences/{sequence_id}/outcome")

    assert post_response.status_code == 403
    assert "b2b_growth" in post_response.json()["detail"]
    assert get_response.status_code == 403
    assert "b2b_growth" in get_response.json()["detail"]


def test_company_timeline_scopes_to_tracked_vendors(monkeypatch):
    app = FastAPI()
    app.include_router(campaigns_api.router)
    user = _auth_user(plan="b2b_growth")
    app.dependency_overrides[require_auth] = lambda: user
    captured_queries: list[tuple[str, tuple]] = []

    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            captured_queries.append((query, args))
            return []

    monkeypatch.setattr(campaigns_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.get("/b2b/campaigns/company-timeline?company=Acme&vendor=Salesforce")

    assert response.status_code == 200
    assert len(captured_queries) == 3
    assert all("tracked_vendors" in query for query, _ in captured_queries)
    assert all(args[-1] == user.account_id or args[-2] == user.account_id for _, args in captured_queries)


def test_dashboard_signal_effectiveness_requires_growth_plan():
    app = FastAPI()
    app.include_router(dashboard_api.router)
    app.dependency_overrides[require_auth] = lambda: _auth_user(plan="b2b_trial")

    with TestClient(app) as client:
        response = client.get("/b2b/dashboard/signal-effectiveness")

    assert response.status_code == 403
    assert "b2b_growth" in response.json()["detail"]


def test_dashboard_outcome_distribution_scopes_by_vendor(monkeypatch):
    app = FastAPI()
    app.include_router(dashboard_api.router)
    user = _auth_user(plan="b2b_growth")
    app.dependency_overrides[require_auth] = lambda: user
    captured = {}

    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            captured["query"] = query
            captured["args"] = args
            return [
                {
                    "outcome": "deal_won",
                    "count": 2,
                    "total_revenue": 10000,
                    "first_recorded": datetime(2026, 4, 1, tzinfo=timezone.utc),
                    "last_recorded": datetime(2026, 4, 7, tzinfo=timezone.utc),
                }
            ]

    monkeypatch.setattr(dashboard_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.get("/b2b/dashboard/outcome-distribution?vendor_name=Salesforce")

    assert response.status_code == 200
    assert "bc.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid)" in captured["query"]
    assert "cs.company_name IN" not in captured["query"]
    assert captured["args"] == (user.account_id, "Salesforce")
    body = response.json()
    assert body["total_sequences"] == 2
    assert body["buckets"][0]["outcome"] == "deal_won"


def test_push_to_crm_routes_high_intent_payload(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    signal_id = uuid4()

    class Pool:
        is_initialized = True

        async def fetch(self, query, *_args):
            if "FROM b2b_webhook_subscriptions" in query:
                captured['subscription_query'] = query
                return [
                    {
                        "id": "sub-1",
                        "url": "https://example.com/webhook",
                        "secret": "secret",
                        "account_id": _auth_user().account_id,
                        "channel": "crm_hubspot",
                        "auth_header": None,
                        "event_types": ["high_intent_push"],
                    }
                ]
            if "FROM b2b_company_signals" in query:
                return [
                    {
                        "id": signal_id,
                        "company_name": "Acme, Inc.",
                        "vendor_name": " Salesforce ",
                    }
                ]
            raise AssertionError(f"Unexpected query: {query}")

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
                        "role_type": "revops",
                        "buying_stage": "evaluation",
                        "contract_end": "2026-06-30",
                        "decision_timeline": "renewal in 60 days",
                        "competitor_context": "HubSpot",
                        "primary_quote": "We need better renewal controls.",
                        "trust_tier": "high",
                        "source": "reddit",
                        "review_id": "review-1",
                        "company_size": "51-200",
                        "company_country": "US",
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert response.json()["pushed"] == 1
    assert response.json()["failed"] == []
    assert captured['subscription_query'].count('high_intent_push') >= 1
    assert captured["channel"] == "crm_hubspot"
    envelope = captured["envelope"]
    assert isinstance(envelope, dict)
    assert envelope["event"] == "high_intent_push"
    assert envelope["vendor"] == "Salesforce"
    assert envelope["data"]["company_name"] == "Acme"
    assert envelope["data"]["company_signal_id"] == str(signal_id)
    assert envelope["data"]["role_type"] == "revops"
    assert envelope["data"]["competitor_context"] == "HubSpot"
    assert envelope["data"]["primary_quote"] == "We need better renewal controls."
    assert envelope["data"]["trust_tier"] == "high"
    assert "company" not in envelope["data"]
    assert log_push.await_count == 1


def test_push_to_crm_omits_company_signal_id_without_canonical_match(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    class Pool:
        is_initialized = True

        async def fetch(self, query, *_args):
            if "FROM b2b_webhook_subscriptions" in query:
                return [
                    {
                        "id": "sub-1",
                        "url": "https://example.com/webhook",
                        "secret": "secret",
                        "account_id": _auth_user().account_id,
                        "channel": "crm_hubspot",
                        "auth_header": None,
                        "event_types": ["high_intent_push"],
                    }
                ]
            if "FROM b2b_company_signals" in query:
                return []
            raise AssertionError(f"Unexpected query: {query}")

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
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert response.json()["pushed"] == 1
    assert response.json()["failed"] == []
    envelope = captured["envelope"]
    assert isinstance(envelope, dict)
    assert envelope["data"]["company_name"] == "Acme"
    assert "company_signal_id" not in envelope["data"]
    assert log_push.await_count == 1


def test_push_to_crm_rejects_blank_company_and_vendor_before_db_touch(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    def fail_get_db_pool():
        raise AssertionError("DB pool should not be acquired for blank push-to-crm fields")

    monkeypatch.setattr(tenant_dashboard_api, "get_db_pool", fail_get_db_pool)

    with TestClient(app) as client:
        response = client.post(
            "/b2b/tenant/push-to-crm",
            json={
                "opportunities": [
                    {
                        "company": "   ",
                        "vendor": "   ",
                        "urgency": 8.5,
                    }
                ]
            },
        )

    assert response.status_code == 422


def test_push_to_crm_trims_payload_fields_before_lookup_and_delivery(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    signal_id = uuid4()

    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            if "FROM b2b_webhook_subscriptions" in query:
                return [
                    {
                        "id": "sub-1",
                        "url": "https://example.com/webhook",
                        "secret": "secret",
                        "account_id": _auth_user().account_id,
                        "channel": "crm_hubspot",
                        "auth_header": None,
                        "event_types": ["high_intent_push"],
                    }
                ]
            if "FROM b2b_company_signals" in query:
                assert args[0] == ["acme"]
                assert args[1] == ["salesforce"]
                return [
                    {
                        "company_name": "Acme",
                        "vendor_name": "Salesforce",
                        "id": signal_id,
                    }
                ]
            raise AssertionError(f"Unexpected query: {query}")

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
                        "company": "  Acme  ",
                        "vendor": "  Salesforce  ",
                        "urgency": 8.5,
                        "pain": "  pricing  ",
                        "competitor_context": "  HubSpot  ",
                        "primary_quote": "  We need better renewal controls.  ",
                        "trust_tier": "  high  ",
                        "source": "  reddit  ",
                        "review_id": "  review-1  ",
                        "alternatives": ["  HubSpot  ", "   ", "  Close  "],
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert response.json()["pushed"] == 1
    assert response.json()["failed"] == []
    envelope = captured["envelope"]
    assert isinstance(envelope, dict)
    assert envelope["vendor"] == "Salesforce"
    assert envelope["data"]["company_name"] == "Acme"
    assert envelope["data"]["pain"] == "pricing"
    assert envelope["data"]["competitor_context"] == "HubSpot"
    assert envelope["data"]["primary_quote"] == "We need better renewal controls."
    assert envelope["data"]["trust_tier"] == "high"
    assert envelope["data"]["source"] == "reddit"
    assert envelope["data"]["review_id"] == "review-1"
    assert envelope["data"]["alternatives"] == ["HubSpot", "Close"]
    assert envelope["data"]["company_signal_id"] == str(signal_id)


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
                    "event_types": ["high_intent_push"],
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


def test_upsert_report_subscription_preserves_next_delivery_when_schedule_unchanged(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    existing_next_delivery_at = datetime(2026, 4, 9, 15, 0, tzinfo=timezone.utc)
    captured_execute_args = {}

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            if "SELECT enabled, delivery_frequency, next_delivery_at" in query:
                return {
                    "enabled": True,
                    "delivery_frequency": "weekly",
                    "next_delivery_at": existing_next_delivery_at,
                }
            if "FROM b2b_report_subscriptions s" in query:
                return {
                    "id": uuid4(),
                    "scope_type": "library",
                    "scope_key": "library",
                    "scope_label": "Recurring brief library",
                    "filter_payload": {},
                    "report_id": None,
                    "delivery_frequency": "weekly",
                    "deliverable_focus": "all",
                    "freshness_policy": "fresh_or_monitor",
                    "recipient_emails": ["ops@example.com"],
                    "delivery_note": "Keep the current cadence",
                    "enabled": True,
                    "next_delivery_at": existing_next_delivery_at,
                    "created_at": existing_next_delivery_at,
                    "updated_at": existing_next_delivery_at,
                    "last_delivery_status": "sent",
                    "last_delivery_at": existing_next_delivery_at,
                    "last_delivery_summary": "Delivered",
                    "last_delivery_error": None,
                    "last_delivery_report_count": 2,
                }
            return None

        async def execute(self, query, *args):
            if "INSERT INTO b2b_report_subscriptions" in query:
                captured_execute_args["args"] = args
            return "INSERT 0 1"

    monkeypatch.setattr(tenant_dashboard_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.put(
            "/b2b/tenant/report-subscriptions/library/library",
            json={
                "scope_label": "Recurring brief library",
                "delivery_frequency": "weekly",
                "deliverable_focus": "all",
                "freshness_policy": "fresh_or_monitor",
                "recipients": ["ops@example.com"],
                "delivery_note": "Keep the current cadence",
                "enabled": True,
            },
        )

    assert response.status_code == 200
    assert captured_execute_args["args"][12] == existing_next_delivery_at


def test_upsert_report_subscription_recomputes_next_delivery_when_frequency_changes(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    existing_next_delivery_at = datetime(2026, 4, 9, 15, 0, tzinfo=timezone.utc)
    captured_execute_args = {}

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            if "SELECT enabled, delivery_frequency, next_delivery_at" in query:
                return {
                    "enabled": True,
                    "delivery_frequency": "weekly",
                    "next_delivery_at": existing_next_delivery_at,
                }
            if "FROM b2b_report_subscriptions s" in query:
                recomputed_next = captured_execute_args["args"][12]
                return {
                    "id": uuid4(),
                    "scope_type": "library",
                    "scope_key": "library",
                    "scope_label": "Recurring brief library",
                    "filter_payload": {},
                    "report_id": None,
                    "delivery_frequency": "monthly",
                    "deliverable_focus": "all",
                    "freshness_policy": "fresh_or_monitor",
                    "recipient_emails": ["ops@example.com"],
                    "delivery_note": "Move to monthly",
                    "enabled": True,
                    "next_delivery_at": recomputed_next,
                    "created_at": existing_next_delivery_at,
                    "updated_at": existing_next_delivery_at,
                    "last_delivery_status": None,
                    "last_delivery_at": None,
                    "last_delivery_summary": None,
                    "last_delivery_error": None,
                    "last_delivery_report_count": 0,
                }
            return None

        async def execute(self, query, *args):
            if "INSERT INTO b2b_report_subscriptions" in query:
                captured_execute_args["args"] = args
            return "INSERT 0 1"

    monkeypatch.setattr(tenant_dashboard_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.put(
            "/b2b/tenant/report-subscriptions/library/library",
            json={
                "scope_label": "Recurring brief library",
                "delivery_frequency": "monthly",
                "deliverable_focus": "all",
                "freshness_policy": "fresh_or_monitor",
                "recipients": ["ops@example.com"],
                "delivery_note": "Move to monthly",
                "enabled": True,
            },
        )

    assert response.status_code == 200
    recomputed_next = captured_execute_args["args"][12]
    assert recomputed_next is not None
    assert recomputed_next != existing_next_delivery_at
    assert recomputed_next > datetime.now(timezone.utc)


def test_get_report_subscription_uses_latest_delivery_attempt_not_live_preference(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    latest_attempt_at = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *_args):
            if "FROM b2b_report_subscriptions s" in query:
                assert "ORDER BY delivered_at DESC NULLS LAST" in query
                assert "CASE WHEN delivery_mode = 'live'" not in query
                return {
                    "id": uuid4(),
                    "scope_type": "library",
                    "scope_key": "library",
                    "scope_label": "Recurring brief library",
                    "filter_payload": {},
                    "report_id": None,
                    "delivery_frequency": "weekly",
                    "deliverable_focus": "all",
                    "freshness_policy": "fresh_or_monitor",
                    "recipient_emails": ["ops@example.com"],
                    "delivery_note": "Review recent state",
                    "enabled": True,
                    "next_delivery_at": latest_attempt_at + timedelta(days=7),
                    "created_at": latest_attempt_at - timedelta(days=14),
                    "updated_at": latest_attempt_at,
                    "last_delivery_status": "dry_run",
                    "last_delivery_at": latest_attempt_at,
                    "last_delivery_summary": "Dry run preview completed",
                    "last_delivery_error": "",
                    "last_delivery_report_count": 1,
                }
            return None

    monkeypatch.setattr(tenant_dashboard_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.get("/b2b/tenant/report-subscriptions/library/library")

    assert response.status_code == 200
    subscription = response.json()["subscription"]
    assert subscription["last_delivery_status"] == "dry_run"
    assert subscription["last_delivery_summary"] == "Dry run preview completed"
    assert subscription["last_delivery_at"] == latest_attempt_at.isoformat()


def test_upsert_report_subscription_library_view_persists_filter_payload(monkeypatch):
    app = FastAPI()
    app.include_router(tenant_dashboard_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    captured_execute_args = {}

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            if "SELECT enabled, delivery_frequency, next_delivery_at" in query:
                return None
            if "FROM b2b_report_subscriptions s" in query:
                return {
                    "id": uuid4(),
                    "scope_type": "library_view",
                    "scope_key": "library-view--type-battle_card--vendor-zendesk--quality-sales_ready",
                    "scope_label": "Battle Cards - Zendesk Library",
                    "filter_payload": {
                        "report_type": "battle_card",
                        "vendor_filter": "Zendesk",
                        "quality_status": "sales_ready",
                        "freshness_state": "stale",
                        "review_state": "blocked",
                    },
                    "report_id": None,
                    "delivery_frequency": "weekly",
                    "deliverable_focus": "battle_cards",
                    "freshness_policy": "fresh_or_monitor",
                    "recipient_emails": ["ops@example.com"],
                    "delivery_note": "Only reviewed assets",
                    "enabled": True,
                    "next_delivery_at": datetime(2026, 4, 17, 15, 0, tzinfo=timezone.utc),
                    "created_at": datetime(2026, 4, 10, 15, 0, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 4, 10, 15, 0, tzinfo=timezone.utc),
                    "last_delivery_status": None,
                    "last_delivery_at": None,
                    "last_delivery_summary": None,
                    "last_delivery_error": None,
                    "last_delivery_report_count": 0,
                }
            return None

        async def execute(self, query, *args):
            if "INSERT INTO b2b_report_subscriptions" in query:
                captured_execute_args["args"] = args
            return "INSERT 0 1"

    monkeypatch.setattr(tenant_dashboard_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.put(
            "/b2b/tenant/report-subscriptions/library_view/library-view--type-battle_card--vendor-zendesk--quality-sales_ready--freshness-stale--review-blocked",
            json={
                "scope_label": "Battle Cards - Zendesk Library",
                "filter_payload": {
                    "report_type": "battle_card",
                    "vendor_filter": "Zendesk",
                    "quality_status": "sales_ready",
                    "freshness_state": "stale",
                    "review_state": "blocked",
                },
                "delivery_frequency": "weekly",
                "deliverable_focus": "battle_cards",
                "freshness_policy": "fresh_or_monitor",
                "recipients": ["ops@example.com"],
                "delivery_note": "Only reviewed assets",
                "enabled": True,
            },
        )

    assert response.status_code == 200
    assert json.loads(captured_execute_args["args"][5]) == {
        "report_type": "battle_card",
        "vendor_filter": "Zendesk",
        "quality_status": "sales_ready",
        "freshness_state": "stale",
        "review_state": "blocked",
    }
    subscription = response.json()["subscription"]
    assert subscription["scope_type"] == "library_view"
    assert subscription["filter_payload"] == {
        "report_type": "battle_card",
        "vendor_filter": "Zendesk",
        "quality_status": "sales_ready",
        "freshness_state": "stale",
        "review_state": "blocked",
    }


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


def test_blog_draft_evidence_uses_canonical_review_basis(monkeypatch):
    app = FastAPI()
    app.include_router(blog_admin_api.router)
    app.dependency_overrides[require_auth] = _auth_user

    captured = {}

    class Pool:
        is_initialized = True

        async def fetchrow(self, query, *_args):
            if "SELECT data_context, source_report_date FROM blog_posts" in query:
                return {
                    "data_context": {"vendor_name": "Shopify"},
                    "source_report_date": date(2026, 4, 1),
                }
            return None

        async def fetch(self, query, *_args):
            captured["query"] = query
            return [{
                "id": uuid4(),
                "vendor_name": "Shopify",
                "reviewer_company": "Acme Corp",
                "summary": "Too expensive",
                "review_text": "We are considering a switch.",
                "pain_category": "pricing",
                "urgency_score": 8,
                "source": "g2",
                "reviewed_at": datetime(2026, 3, 20, tzinfo=timezone.utc),
                "reviewer_title": "VP Operations",
                "company_size_raw": "201-500",
                "industry": "Retail",
            }]

    monkeypatch.setattr(blog_admin_api, "get_db_pool", lambda: Pool())

    with TestClient(app) as client:
        response = client.get(f"/admin/blog/drafts/{uuid4()}/evidence")

    assert response.status_code == 200
    body = response.json()
    assert body["basis"] == "canonical_reviews"
    assert body["count"] == 1
    assert "JOIN b2b_review_vendor_mentions vm" in captured["query"]
    assert "duplicate_of_review_id IS NULL" in captured["query"]


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
        "data_stale": False,
        "quality_status": None,
        "quality_score": None,
        "report_subscription_id": None,
        "report_subscription_scope_type": None,
        "report_subscription_scope_key": None,
        "report_subscription_scope_label": None,
        "report_subscription_enabled": None,
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
