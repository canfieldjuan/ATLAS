from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from atlas_brain.api import billing
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices
from extracted_content_pipeline.deflection_report_access import (
    InMemoryDeflectionReportArtifactStore,
)
from extracted_content_pipeline.faq_deflection_report import FAQDeflectionReportService


def _route(router: Any, path: str, method: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


def _source_material() -> list[dict[str, str]]:
    return [
        {
            "source_id": "ticket-export-1",
            "source_type": "support_ticket",
            "source_title": "Export attribution",
            "text": "How do I export attribution reports?",
            "resolution_text": (
                "Open Analytics, choose Attribution, then click Download report"
            ),
        },
        {
            "source_id": "ticket-export-2",
            "source_type": "support_ticket",
            "source_title": "Report download",
            "text": "Where is the report download for attribution exports?",
            "resolution_text": (
                "Open Analytics, choose Attribution, then click Download report"
            ),
        },
        {
            "source_id": "ticket-sso-1",
            "source_type": "support_ticket",
            "source_title": "SSO setup",
            "text": "How do I enable SSO for my team?",
        },
        {
            "source_id": "ticket-sso-2",
            "source_type": "support_ticket",
            "source_title": "Team login",
            "text": "Can I turn on SSO for all users?",
        },
    ]


def _session(
    *,
    account_id: str,
    request_id: str,
    session_id: str = "cs_test_paid_flow",
) -> SimpleNamespace:
    metadata = {
        "source": "content_ops_deflection_report",
        "account_id": account_id,
        "request_id": request_id,
    }
    return SimpleNamespace(
        id=session_id,
        mode="payment",
        payment_status="paid",
        amount_total=150000,
        currency="usd",
        metadata=metadata,
        to_dict=lambda: {
            "id": session_id,
            "mode": "payment",
            "payment_status": "paid",
            "amount_total": 150000,
            "currency": "usd",
            "metadata": dict(metadata),
        },
    )


class _BillingPool:
    def __init__(self, store: InMemoryDeflectionReportArtifactStore) -> None:
        self.is_initialized = True
        self.store = store
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.delivery_rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append((query, args))
        return None

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        if "FROM content_ops_deflection_reports" not in query:
            raise AssertionError(query)
        account_id, request_id = args
        record = await self.store.get_artifact_record(
            account_id=str(account_id),
            request_id=str(request_id),
        )
        if record is None:
            return None
        return {
            "account_id": record.account_id,
            "request_id": record.request_id,
            "snapshot": record.snapshot,
            "artifact": record.artifact,
            "paid": record.paid,
            "payment_reference": record.payment_reference,
            "delivery_email": record.delivery_email,
        }

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((query, args))
        if "UPDATE content_ops_deflection_reports" in query:
            account_id, request_id, payment_reference = args
            marked = await self.store.mark_paid(
                account_id=str(account_id),
                request_id=str(request_id),
                payment_reference=str(payment_reference or ""),
            )
            return "UPDATE 1" if marked else "UPDATE 0"
        if "INSERT INTO content_ops_deflection_report_deliveries" in query:
            account_id, request_id, payment_reference = args
            self.delivery_rows[(str(account_id), str(request_id))] = {
                "payment_reference": payment_reference,
                "delivery_status": "pending",
            }
            return "INSERT 0 1"
        if "INSERT INTO billing_events" in query:
            return "INSERT 0 1"
        raise AssertionError(query)


@pytest.mark.asyncio
async def test_deflection_paid_flow_locks_snapshot_until_stripe_webhook_unlocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_id = "00000000-0000-0000-0000-000000000115"
    store = InMemoryDeflectionReportArtifactStore()
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            prefix="/ops",
            tags=("ops",),
            deflection_snapshot_top_n=2,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_deflection_report=FAQDeflectionReportService(),
        ),
        deflection_report_store_provider=lambda: store,
        scope_provider=lambda: {"account_id": account_id},
    )

    execute = _route(router, "/ops/execute", "POST")
    executed = await execute.endpoint(
        {
            "outputs": ["faq_deflection_report"],
            "limit": 2,
            "inputs": {
                "source_material": _source_material(),
                "faq_documentation_terms": (
                    "Download report",
                    "Single sign-on setup",
                ),
                "faq_vocabulary_gap_rules": (
                    ("export", "Download report"),
                    ("SSO", "Single sign-on setup"),
                ),
                "contact_email": "buyer@example.com",
            },
        }
    )

    request_id = executed["request_id"]
    gated_result = executed["steps"][0]["result"]
    assert gated_result == {
        "request_id": request_id,
        "snapshot": gated_result["snapshot"],
        "full_report": {
            "status": "locked",
            "reason": "payment_required",
        },
    }
    assert gated_result["snapshot"]["summary"] == {
        "generated": 3,
        "drafted_answer_count": 1,
        "no_proven_answer_count": 2,
        "repeat_ticket_count": 4,
    }
    encoded_gated_result = str(gated_result)
    assert "markdown" not in encoded_gated_result
    assert "faq_result" not in encoded_gated_result
    assert "ticket-export-1" not in encoded_gated_result
    assert "buyer@example.com" not in encoded_gated_result

    snapshot_route = _route(router, "/ops/deflection-reports/{request_id}/snapshot", "GET")
    assert await snapshot_route.endpoint(request_id=request_id) == gated_result["snapshot"]

    artifact_route = _route(router, "/ops/deflection-reports/{request_id}/artifact", "GET")
    with pytest.raises(billing.HTTPException) as locked:
        await artifact_route.endpoint(request_id=request_id)
    assert locked.value.status_code == 403

    event = SimpleNamespace(
        id="evt_deflection_paid_flow",
        type="checkout.session.completed",
        data=SimpleNamespace(object=_session(account_id=account_id, request_id=request_id)),
    )

    class _Webhook:
        @staticmethod
        def construct_event(_body: bytes, _sig: str, _secret: str) -> Any:
            return event

    billing_pool = _BillingPool(store)
    request = SimpleNamespace(
        headers={"stripe-signature": "valid"},
        body=lambda: _body(),
    )

    async def _body() -> bytes:
        return b"{}"

    monkeypatch.setitem(sys.modules, "stripe", SimpleNamespace(Webhook=_Webhook, api_key=""))
    monkeypatch.setattr(billing.settings.saas_auth, "stripe_secret_key", "sk_test")
    monkeypatch.setattr(
        billing.settings.saas_auth,
        "stripe_webhook_secret",
        "whsec_test",
    )
    monkeypatch.setattr(billing, "get_db_pool", lambda: billing_pool)

    assert await billing.stripe_webhook(request) == {"status": "ok"}
    assert billing_pool.fetchval_calls[0][1] == ("evt_deflection_paid_flow",)
    assert "UPDATE content_ops_deflection_reports" in billing_pool.execute_calls[0][0]
    assert "INSERT INTO content_ops_deflection_report_deliveries" in billing_pool.execute_calls[1][0]
    assert billing_pool.delivery_rows[(account_id, request_id)] == {
        "payment_reference": "cs_test_paid_flow",
        "delivery_status": "pending",
    }
    assert "INSERT INTO billing_events" in billing_pool.execute_calls[2][0]

    unlocked = await artifact_route.endpoint(request_id=request_id)
    assert unlocked["summary"]["drafted_answer_count"] == 1
    assert unlocked["summary"]["no_proven_answer_count"] == 2
    assert "## Support Tax Confirmation" in unlocked["markdown"]
    assert "## Publishable Help-Center Copy From Proven Resolutions" in unlocked["markdown"]
    assert "## No Proven Answer Yet" in unlocked["markdown"]
    assert unlocked["faq_result"]["items"][0]["source_ids"]
