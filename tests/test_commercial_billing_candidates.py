"""Contract tests for the side-effect-free EOM commercial billing preview."""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from atlas_brain.services.calendar_provider import CalendarEvent
from atlas_brain.services.commercial_billing_candidates import (
    CommercialBillingCandidateService,
    CommercialBillingCandidatesUnavailableError,
    CommercialBillingCandidatesValidationError,
)


class _ReadOnlyServiceRepository:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[bool] = []
        self.write_attempts = 0

    async def list_active(self, auto_invoice_only: bool = False) -> list[dict]:
        self.calls.append(auto_invoice_only)
        return self.rows

    async def mark_invoiced(self, *_args, **_kwargs) -> None:
        self.write_attempts += 1
        raise AssertionError("candidate preview must not mark a service invoiced")


class _ReadOnlyCalendar:
    def __init__(self, events: list[object], *, error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.calls: list[tuple[datetime, datetime, str | None]] = []
        self.write_attempts = 0

    async def list_events(
        self,
        start: datetime,
        end: datetime,
        calendar_id: str | None = None,
    ) -> list[object]:
        self.calls.append((start, end, calendar_id))
        if self.error is not None:
            raise self.error
        return self.events

    async def create_event(self, *_args, **_kwargs) -> None:
        self.write_attempts += 1
        raise AssertionError("candidate preview must not create calendar events")


class _ReadOnlyCRM:
    def __init__(
        self,
        customers: dict[UUID, dict | None],
        recipients: dict[UUID, dict],
        *,
        customer_error: Exception | None = None,
        recipient_error: Exception | None = None,
    ) -> None:
        self.customers = customers
        self.recipients = recipients
        self.customer_error = customer_error
        self.recipient_error = recipient_error
        self.customer_calls: list[UUID] = []
        self.recipient_calls: list[UUID] = []
        self.write_attempts = 0

    async def get_eom_payment_customer(self, contact_id: UUID) -> dict | None:
        self.customer_calls.append(contact_id)
        if self.customer_error is not None:
            raise self.customer_error
        return self.customers.get(contact_id)

    async def get_billing_recipient(self, contact_id: UUID) -> dict:
        self.recipient_calls.append(contact_id)
        if self.recipient_error is not None:
            raise self.recipient_error
        return self.recipients[contact_id]

    async def log_interaction(self, *_args, **_kwargs) -> None:
        self.write_attempts += 1
        raise AssertionError("candidate preview must not log CRM interactions")


def _service_row(
    contact_id: UUID,
    *,
    service_id: UUID | None = None,
    name: str = "Office cleaning",
    rate: object = Decimal("48.25"),
    rate_label: str = "Per Visit",
    tax_rate: object = Decimal("0.0330"),
    keyword: str = "Acme",
    calendar_id: str | None = None,
) -> dict:
    return {
        "id": service_id or uuid4(),
        "contact_id": contact_id,
        "service_name": name,
        "rate": rate,
        "rate_label": rate_label,
        "tax_rate": tax_rate,
        "calendar_keyword": keyword,
        "calendar_id": calendar_id,
    }


def _customer(contact_id: UUID, *, customer_type: str = "commercial") -> dict:
    return {
        "contact_id": contact_id,
        "customer_name": "Acme Office",
        "customer_type": customer_type,
        "recipient_email": "billing@example.test",
    }


def _recipient(contact_id: UUID, *, eligible: bool = True, reason: str | None = None) -> dict:
    return {
        "contactId": str(contact_id),
        "displayName": "Acme Accounts Payable" if eligible else None,
        "email": "billing@example.test" if eligible else None,
        "eligible": eligible,
        "reason": reason,
    }


def _event(
    event_id: str,
    day: int,
    *,
    summary: str = "Acme Office cleaning",
    location: str | None = "100 Main St",
    status: str = "confirmed",
) -> CalendarEvent:
    start = datetime(2026, 3, day, 16, 0, tzinfo=timezone.utc)
    return CalendarEvent(
        uid=event_id,
        summary=summary,
        start=start,
        end=start.replace(hour=17),
        calendar_id="commercial-calendar",
        location=location,
        status=status,
    )


def _candidate_service(
    rows: list[dict],
    events: list[object],
    *,
    customers: dict[UUID, dict | None] | None = None,
    recipients: dict[UUID, dict] | None = None,
    calendar_error: Exception | None = None,
    crm_customer_error: Exception | None = None,
    crm_recipient_error: Exception | None = None,
) -> tuple[
    CommercialBillingCandidateService,
    _ReadOnlyServiceRepository,
    _ReadOnlyCalendar,
    _ReadOnlyCRM,
]:
    repository = _ReadOnlyServiceRepository(rows)
    calendar = _ReadOnlyCalendar(events, error=calendar_error)
    contact_ids = {
        UUID(str(row["contact_id"]))
        for row in rows
        if row.get("contact_id") is not None
    }
    crm = _ReadOnlyCRM(
        customers or {contact_id: _customer(contact_id) for contact_id in contact_ids},
        recipients or {contact_id: _recipient(contact_id) for contact_id in contact_ids},
        customer_error=crm_customer_error,
        recipient_error=crm_recipient_error,
    )
    return (
        CommercialBillingCandidateService(
            customer_service_repository=repository,
            calendar_provider_loader=lambda: calendar,
            crm_provider_loader=lambda: crm,
            calendar_id="commercial-calendar",
        ),
        repository,
        calendar,
        crm,
    )


@pytest.mark.asyncio
async def test_per_visit_preview_is_exact_and_has_no_financial_or_delivery_effects():
    contact_id = uuid4()
    service, repository, calendar, crm = _candidate_service(
        [_service_row(contact_id, rate=48.25)],
        [_event("evt-2", 3), _event("evt-1", 3), _event("evt-3", 10)],
    )

    preview = await service.preview(billing_period="2026-03")

    assert preview["billingPeriod"] == "2026-03"
    assert preview["calendarId"] == "commercial-calendar"
    assert preview["summary"] == {"blockedCandidateCount": 1, "candidateCount": 1}
    candidate = preview["candidates"][0]
    assert candidate["customer"] == {
        "contactId": str(contact_id),
        "customerType": "commercial",
        "displayName": "Acme Office",
    }
    assert candidate["recipient"] == {
        "contactId": str(contact_id),
        "displayName": "Acme Accounts Payable",
        "email": "billing@example.test",
    }
    assert candidate["deliveryMethod"] is None
    assert candidate["subtotalCents"] == 14_475
    assert candidate["taxRateBasisPoints"] == 330
    assert candidate["taxCents"] == 478
    assert candidate["totalCents"] == 14_953
    assert candidate["lineItems"] == [
        {
            "amountCents": 9_650,
            "description": "Office cleaning",
            "eventIds": ["evt-1", "evt-2"],
            "locations": ["100 Main St"],
            "quantity": 2,
            "quantityUnit": "visit",
            "rateCents": 4_825,
            "serviceId": str(repository.rows[0]["id"]),
            "sourceDate": "2026-03-03",
        },
        {
            "amountCents": 4_825,
            "description": "Office cleaning",
            "eventIds": ["evt-3"],
            "locations": ["100 Main St"],
            "quantity": 1,
            "quantityUnit": "visit",
            "rateCents": 4_825,
            "serviceId": str(repository.rows[0]["id"]),
            "sourceDate": "2026-03-10",
        },
    ]
    assert candidate["blockers"] == [
        {
            "code": "missing_billing_delivery_preference",
            "eventIds": [],
            "message": "No explicit billing delivery preference is recorded.",
            "serviceId": None,
        }
    ]
    assert len(candidate["sourceFingerprint"]) == 64
    assert repository.calls == [True]
    assert calendar.calls[0][2] == "commercial-calendar"
    assert crm.customer_calls == [contact_id]
    assert crm.recipient_calls == [contact_id]
    assert repository.write_attempts == calendar.write_attempts == crm.write_attempts == 0


@pytest.mark.asyncio
async def test_preview_retries_are_identical_and_source_changes_make_it_stale():
    contact_id = uuid4()
    row = _service_row(contact_id, rate=48.25)
    service, repository, calendar, crm = _candidate_service(
        [row], [_event("evt-1", 3)]
    )

    first = await service.preview(billing_period="2026-03")
    second = await service.preview(billing_period="2026-03")
    assert second == first

    row["rate"] = Decimal("49.25")
    changed_rate = await service.preview(billing_period="2026-03")
    assert changed_rate["candidates"][0]["sourceFingerprint"] != first["candidates"][0]["sourceFingerprint"]

    row["id"] = uuid4()
    changed_service_identity = await service.preview(billing_period="2026-03")
    assert (
        changed_service_identity["candidates"][0]["sourceFingerprint"]
        != changed_rate["candidates"][0]["sourceFingerprint"]
    )

    crm.recipients[contact_id] = {
        **crm.recipients[contact_id],
        "email": "new-ap@example.test",
    }
    changed_recipient = await service.preview(billing_period="2026-03")
    assert (
        changed_recipient["candidates"][0]["sourceFingerprint"]
        != changed_service_identity["candidates"][0]["sourceFingerprint"]
    )

    calendar.events.append(_event("evt-2", 10, location="Second Site"))
    changed_calendar = await service.preview(billing_period="2026-03")
    assert (
        changed_calendar["candidates"][0]["sourceFingerprint"]
        != changed_recipient["candidates"][0]["sourceFingerprint"]
    )
    assert repository.write_attempts == calendar.write_attempts == crm.write_attempts == 0


@pytest.mark.asyncio
async def test_preview_surfaces_billing_blockers_without_creating_a_fake_invoice():
    commercial = uuid4()
    residential = uuid4()
    rows = [
        _service_row(
            commercial,
            name="Bad rate",
            rate=Decimal("12.345"),
            keyword="Bad rate",
        ),
        _service_row(
            commercial,
            name="Hourly",
            rate_label="Per Hour",
            keyword="Hours",
        ),
        _service_row(
            commercial,
            name="No calendar evidence",
            keyword="Not found",
        ),
        _service_row(
            commercial,
            name="Unsupported rate label",
            rate_label="Per Shift",
            keyword="Unsupported",
        ),
        _service_row(
            commercial,
            name="Flat monthly service",
            rate_label="Per Month",
            keyword="Flat",
        ),
        _service_row(
            residential,
            name="Residential service",
            rate_label="Per Month",
            keyword="Residential",
        ),
    ]
    customers = {
        commercial: _customer(commercial),
        residential: _customer(residential, customer_type="residential"),
    }
    recipients = {
        commercial: _recipient(commercial, eligible=False, reason="no_email"),
        residential: _recipient(residential),
    }
    service, repository, calendar, crm = _candidate_service(
        rows,
        [_event("evt-hours", 5, summary="Hours cleaning")],
        customers=customers,
        recipients=recipients,
    )

    preview = await service.preview(billing_period="2026-03")
    commercial_candidate = next(
        item
        for item in preview["candidates"]
        if item["customer"]["contactId"] == str(commercial)
    )
    assert {blocker["code"] for blocker in commercial_candidate["blockers"]} >= {
        "invalid_rate",
        "invalid_rate_label",
        "missing_billing_delivery_preference",
        "missing_billing_email",
        "missing_calendar_service_evidence",
        "missing_hours",
        "zero_or_invalid_total",
    }
    assert commercial_candidate["totalCents"] is None
    line_items = {
        item["description"]: item for item in commercial_candidate["lineItems"]
    }
    assert line_items["Hourly"]["amountCents"] is None
    assert (
        line_items["Flat monthly service"]["quantity"],
        line_items["Flat monthly service"]["quantityUnit"],
        line_items["Flat monthly service"]["amountCents"],
    ) == (1, "month", 4_825)

    residential_candidate = next(
        item
        for item in preview["candidates"]
        if item["customer"]["contactId"] == str(residential)
    )
    assert {blocker["code"] for blocker in residential_candidate["blockers"]} >= {
        "customer_not_commercial"
    }
    assert residential_candidate["totalCents"] == 4_984
    assert repository.write_attempts == calendar.write_attempts == crm.write_attempts == 0


@pytest.mark.asyncio
async def test_preview_blocks_ambiguous_keyword_matches_without_double_counting():
    first_contact = uuid4()
    second_contact = uuid4()
    first_service = _service_row(
        first_contact,
        name="Acme",
        rate=Decimal("10.00"),
        keyword="Acme",
    )
    second_service = _service_row(
        second_contact,
        name="Acme Office",
        rate=Decimal("20.00"),
        keyword="Acme Office",
    )
    service, _, _, _ = _candidate_service(
        [first_service, second_service],
        [_event("evt-ambiguous", 7, summary="Acme Office cleaning")],
    )

    preview = await service.preview(billing_period="2026-03")
    candidates = {item["customer"]["contactId"]: item for item in preview["candidates"]}
    first = candidates[str(first_contact)]
    second = candidates[str(second_contact)]
    assert {blocker["code"] for blocker in first["blockers"]} >= {
        "ambiguous_calendar_service_match",
        "zero_or_invalid_total",
    }
    assert {blocker["code"] for blocker in second["blockers"]} >= {
        "ambiguous_calendar_service_match"
    }
    assert first["lineItems"] == []
    assert second["lineItems"][0]["amountCents"] == 2_000
    assert second["totalCents"] is None
    assert all(
        source["eventId"] == "evt-ambiguous"
        for candidate in (first, second)
        for source in candidate["sourceEvents"]
    )


@pytest.mark.asyncio
async def test_preview_reports_missing_customer_and_malformed_calendar_evidence():
    contact_id = uuid4()
    row = _service_row(contact_id)
    malformed = SimpleNamespace(
        uid="",
        summary="Acme Office cleaning",
        start=datetime(2026, 3, 4, tzinfo=timezone.utc),
        end=datetime(2026, 3, 4, 1, tzinfo=timezone.utc),
        status="confirmed",
    )
    service, _, _, _ = _candidate_service(
        [row],
        [malformed],
        customers={contact_id: None},
        recipients={contact_id: _recipient(contact_id)},
    )

    candidate = (await service.preview(billing_period="2026-03"))["candidates"][0]
    assert {blocker["code"] for blocker in candidate["blockers"]} >= {
        "missing_canonical_customer",
        "missing_calendar_service_evidence",
        "source_evidence_invalid",
        "zero_or_invalid_total",
    }
    assert candidate["recipient"] == {
        "contactId": str(contact_id),
        "displayName": None,
        "email": None,
    }


@pytest.mark.asyncio
async def test_invalid_period_and_source_outage_fail_without_a_write_or_partial_preview():
    contact_id = uuid4()
    service, repository, calendar, crm = _candidate_service(
        [_service_row(contact_id)],
        [_event("evt-1", 3)],
    )

    with pytest.raises(CommercialBillingCandidatesValidationError):
        await service.preview(billing_period="2026-3")
    assert repository.calls == []
    assert calendar.calls == []
    assert crm.customer_calls == []

    unavailable_service, unavailable_repository, unavailable_calendar, unavailable_crm = (
        _candidate_service(
            [_service_row(contact_id)],
            [],
            calendar_error=OSError("calendar unavailable"),
        )
    )
    with pytest.raises(CommercialBillingCandidatesUnavailableError):
        await unavailable_service.preview(billing_period="2026-03")
    assert unavailable_repository.calls == [True]
    assert unavailable_calendar.calls
    assert unavailable_crm.customer_calls == []
    assert unavailable_repository.write_attempts == 0
    assert unavailable_calendar.write_attempts == 0
    assert unavailable_crm.write_attempts == 0

    unavailable_calendar.error = None
    recovered = await unavailable_service.preview(billing_period="2026-03")
    assert recovered["summary"] == {"blockedCandidateCount": 1, "candidateCount": 1}

    crm_unavailable_service, crm_unavailable_repository, crm_unavailable_calendar, crm_unavailable = (
        _candidate_service(
            [_service_row(contact_id)],
            [_event("evt-1", 3)],
            crm_customer_error=RuntimeError("canonical CRM unavailable"),
        )
    )
    with pytest.raises(CommercialBillingCandidatesUnavailableError):
        await crm_unavailable_service.preview(billing_period="2026-03")
    assert crm_unavailable_repository.calls == [True]
    assert crm_unavailable_calendar.calls
    assert crm_unavailable.customer_calls == [contact_id]
    assert crm_unavailable.recipient_calls == []
    assert (
        crm_unavailable_repository.write_attempts
        == crm_unavailable_calendar.write_attempts
        == crm_unavailable.write_attempts
        == 0
    )

    crm_unavailable.customer_error = None
    crm_recovered = await crm_unavailable_service.preview(billing_period="2026-03")
    assert crm_recovered["summary"] == {
        "blockedCandidateCount": 1,
        "candidateCount": 1,
    }

    recipient_unavailable_service, recipient_unavailable_repository, recipient_unavailable_calendar, recipient_unavailable = (
        _candidate_service(
            [_service_row(contact_id)],
            [_event("evt-1", 3)],
            crm_recipient_error=RuntimeError("billing recipient unavailable"),
        )
    )
    with pytest.raises(CommercialBillingCandidatesUnavailableError):
        await recipient_unavailable_service.preview(billing_period="2026-03")
    assert recipient_unavailable_repository.calls == [True]
    assert recipient_unavailable_calendar.calls
    assert recipient_unavailable.customer_calls == [contact_id]
    assert recipient_unavailable.recipient_calls == [contact_id]
    assert (
        recipient_unavailable_repository.write_attempts
        == recipient_unavailable_calendar.write_attempts
        == recipient_unavailable.write_attempts
        == 0
    )

    recipient_unavailable.recipient_error = None
    recipient_recovered = await recipient_unavailable_service.preview(
        billing_period="2026-03"
    )
    assert recipient_recovered["summary"] == {
        "blockedCandidateCount": 1,
        "candidateCount": 1,
    }


@pytest.mark.asyncio
async def test_preview_passes_no_calendar_identifier_when_the_config_is_absent():
    contact_id = uuid4()
    repository = _ReadOnlyServiceRepository([_service_row(contact_id)])
    calendar = _ReadOnlyCalendar([_event("evt-1", 3)])
    crm = _ReadOnlyCRM({contact_id: _customer(contact_id)}, {contact_id: _recipient(contact_id)})
    service = CommercialBillingCandidateService(
        customer_service_repository=repository,
        calendar_provider_loader=lambda: calendar,
        crm_provider_loader=lambda: crm,
        calendar_id="",
    )

    candidate = (await service.preview(billing_period="2026-03"))["candidates"][0]
    assert calendar.calls[0][2] is None
    assert candidate["calendarId"] is None

    configured_service = CommercialBillingCandidateService(
        customer_service_repository=repository,
        calendar_provider_loader=lambda: calendar,
        crm_provider_loader=lambda: crm,
        calendar_id="second-commercial-calendar",
    )
    configured_candidate = (
        await configured_service.preview(billing_period="2026-03")
    )["candidates"][0]
    assert configured_candidate["sourceFingerprint"] != candidate["sourceFingerprint"]


def _route_app(service: object) -> tuple[FastAPI, str]:
    from atlas_brain.api.invoicing import auth as receivables_auth
    from atlas_brain.api.invoicing import receivables as routes
    from atlas_brain.eom_api.auth import generate_receivables_service_token

    generated = generate_receivables_service_token()
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[receivables_auth.get_receivables_api_config] = lambda: (
        SimpleNamespace(
            receivables_api_enabled=True,
            receivables_service_token="",
            receivables_service_token_sha256=generated.sha256,
        )
    )
    app.dependency_overrides[routes.get_commercial_billing_candidate_service] = (
        lambda: service
    )
    return app, generated.token


class _RoutePreviewService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    async def preview(self, *, billing_period: str) -> dict:
        self.calls.append(billing_period)
        if self.error is not None:
            raise self.error
        return {"billingPeriod": billing_period, "candidates": []}


@pytest.mark.asyncio
async def test_full_provider_route_authenticates_before_preview_and_validates_period():
    service = _RoutePreviewService()
    app, token = _route_app(service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://receivables.test",
    ) as client:
        path = "/receivables/commercial-billing-candidates"
        assert (await client.get(path)).status_code == 401
        assert service.calls == []
        headers = {"Authorization": f"Bearer {token}"}
        assert (
            await client.get(
                path,
                params={"billing_period": "2026-03"},
                headers={"Authorization": "Bearer wrong-token"},
            )
        ).status_code == 401
        assert service.calls == []
        malformed = await client.get(
            path,
            params={"billing_period": "2026-3"},
            headers=headers,
        )
        assert malformed.status_code == 422
        assert service.calls == []
        response = await client.get(
            path,
            params={"billing_period": "2026-03"},
            headers=headers,
        )

    assert response.status_code == 200, response.text
    assert response.json() == {"billingPeriod": "2026-03", "candidates": []}
    assert service.calls == ["2026-03"]


@pytest.mark.asyncio
async def test_full_provider_route_returns_stable_source_unavailable_response():
    service = _RoutePreviewService(
        error=CommercialBillingCandidatesUnavailableError("calendar evidence unavailable")
    )
    app, token = _route_app(service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://receivables.test",
    ) as client:
        response = await client.get(
            "/receivables/commercial-billing-candidates",
            params={"billing_period": "2026-03"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "commercial_billing_candidates_unavailable",
            "message": "calendar evidence unavailable",
        }
    }
    assert service.calls == ["2026-03"]


def test_preview_service_does_not_import_the_writeful_scheduler_or_delivery_stack():
    """A dependency-free source guard prevents accidental preview side effects."""

    import atlas_brain.services.commercial_billing_candidates as candidates

    imports = {
        alias.name
        for node in ast.walk(ast.parse(inspect.getsource(candidates)))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        {
            f"{node.module}.{alias.name}" if node.module else alias.name
            for node in ast.walk(ast.parse(inspect.getsource(candidates)))
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
    )
    forbidden_fragments = {
        "monthly_invoice_generation",
        "invoice_pdf",
        "email_provider",
        "gmail",
        "notification",
        "invoice",
    }
    assert not any(
        fragment in imported
        for fragment in forbidden_fragments
        for imported in imports
    )


def test_invoicing_workflow_enrolls_the_candidate_contract_for_pr_and_main_push():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "atlas_invoicing_checks.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count('"atlas_brain/services/commercial_billing_candidates.py"') == 2
    assert workflow.count('"tests/test_commercial_billing_candidates.py"') == 2
    assert "tests/test_commercial_billing_candidates.py \\" in workflow
