"""Monkeypatched unit proof for the legacy monthly writer's cross-pipeline
recurring-invoice dedup (migration 385 / ATLAS #2363).

Deliberately lighter than tests/test_legacy_monthly_autoinvoice_writer_harness.py
(the armed, real-Postgres harness): every dependency run() reaches is faked, so
this proves only that run() calls get_by_contact_and_period per bundle and
skips-without-creating on a hit, not the SQL/index behavior itself -- that is
proven separately, against a real database, in
tests/test_invoice_repository.py::test_real_postgres_billing_period_dedup_scoping_and_void_exclusion.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest


class _FakeCalendarProvider:
    async def list_events(self, start, end, calendar_id=None):
        return []


class _FakeCRMProvider:
    async def get_contact(self, contact_id):
        return {"full_name": "Test Customer", "email": None, "phone": None, "address": None}


class _FakeServiceRepo:
    def __init__(self, services):
        self._services = services

    async def list_active(self, auto_invoice_only=True):
        return self._services


class _FakeInvoiceRepo:
    def __init__(self, cross_pipeline_hits, *, raises_for: frozenset[str] = frozenset()):
        self._cross_pipeline_hits = cross_pipeline_hits
        self._raises_for = raises_for
        self.get_by_contact_and_period_calls: list[tuple[str, str]] = []
        self.create_calls: list[dict] = []

    async def get_by_source_ref(self, source_ref):
        return None

    async def get_by_contact_and_period(self, contact_id, billing_period):
        key = (str(contact_id), billing_period)
        self.get_by_contact_and_period_calls.append(key)
        if str(contact_id) in self._raises_for:
            raise RuntimeError("transient read error")
        return self._cross_pipeline_hits.get(key)

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return {
            "id": uuid4(),
            "invoice_number": f"INV-TEST-{len(self.create_calls)}",
            "total_amount": 100.0,
        }


def _per_month_service(*, contact_id: UUID, name: str, keyword: str) -> dict:
    return {
        "id": uuid4(),
        "contact_id": contact_id,
        "service_name": name,
        "rate": 100.0,
        "rate_label": "Per Month",
        "calendar_keyword": keyword,
        "tax_rate": 0.0,
        "auto_invoice": True,
    }


@pytest.mark.asyncio
async def test_legacy_writer_skips_contact_the_new_pipeline_already_invoiced(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
):
    """One contact already invoiced by the new pipeline for this period is
    skipped without a create() call; a second, un-invoiced contact in the
    same run is created normally -- the negative control proving the check
    discriminates per contact, run inline in the same pass."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import run
    from atlas_brain.config import settings
    from atlas_brain.storage.models import ScheduledTask
    import atlas_brain.services.calendar_provider as calendar_provider_mod
    import atlas_brain.services.crm_provider as crm_provider_mod
    import atlas_brain.storage.repositories.customer_service as customer_service_mod
    import atlas_brain.storage.repositories.invoice as invoice_repo_mod

    already_invoiced_contact = uuid4()
    fresh_contact = uuid4()
    period = "2026-04"

    services = [
        _per_month_service(
            contact_id=already_invoiced_contact,
            name="Office Cleaning", keyword="Already Invoiced Co",
        ),
        _per_month_service(
            contact_id=fresh_contact,
            name="Office Cleaning", keyword="Fresh Co",
        ),
    ]
    fake_inv_repo = _FakeInvoiceRepo(
        cross_pipeline_hits={
            (str(already_invoiced_contact), period): {
                "source": "eom_commercial_billing",
                "invoice_number": "INV-2026-Apr-0099",
            },
        },
    )

    monkeypatch.setattr(settings.invoicing, "enabled", True)
    monkeypatch.setattr(settings.invoicing, "auto_invoice_enabled", True)
    monkeypatch.setattr(settings.invoicing, "auto_invoice_review_mode", True)
    monkeypatch.setattr(settings.invoicing, "auto_invoice_save_path", str(tmp_path))
    monkeypatch.setattr(
        calendar_provider_mod, "get_calendar_provider", lambda: _FakeCalendarProvider()
    )
    monkeypatch.setattr(crm_provider_mod, "get_crm_provider", lambda: _FakeCRMProvider())
    monkeypatch.setattr(
        customer_service_mod,
        "get_customer_service_repo",
        lambda: _FakeServiceRepo(services),
    )
    monkeypatch.setattr(invoice_repo_mod, "get_invoice_repo", lambda: fake_inv_repo)

    task = ScheduledTask(
        id=uuid4(),
        name="monthly_invoice_generation",
        task_type="builtin",
        schedule_type="cron",
        cron_expression="0 8 1 * *",
        metadata={"billing_month": period},
    )

    result = await run(task)

    assert result["invoices_created"] == 1
    assert result["invoices_skipped_dedup"] == 1
    assert len(fake_inv_repo.create_calls) == 1
    assert fake_inv_repo.create_calls[0]["contact_id"] == fresh_contact

    # The pre-check ran for BOTH contacts, not only the one that hit.
    assert set(fake_inv_repo.get_by_contact_and_period_calls) == {
        (str(already_invoiced_contact), period),
        (str(fresh_contact), period),
    }


@pytest.mark.asyncio
async def test_legacy_writer_fails_closed_when_cross_pipeline_check_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
):
    """Codex finding #8 (round 3): a quarantined historical collision is
    protected only by invoices_billing_period_reservations, not by the
    partial unique index (both leave billing_period = NULL). So unlike the
    pre-existing get_by_source_ref check -- whose own failure mode is safe,
    backstopped by a real DB constraint either way -- a transient error from
    get_by_contact_and_period must skip this contact this run (fail closed),
    not fall through to create() (fail open), which would admit the exact
    unprotected duplicate the reservation table exists to prevent. Negative
    control: a second, healthy contact in the same run still creates
    normally -- the failure is per-contact, not a run-wide abort."""
    from atlas_brain.autonomous.tasks.monthly_invoice_generation import run
    from atlas_brain.config import settings
    from atlas_brain.storage.models import ScheduledTask
    import atlas_brain.services.calendar_provider as calendar_provider_mod
    import atlas_brain.services.crm_provider as crm_provider_mod
    import atlas_brain.storage.repositories.customer_service as customer_service_mod
    import atlas_brain.storage.repositories.invoice as invoice_repo_mod

    flaky_contact = uuid4()
    healthy_contact = uuid4()
    period = "2026-04"

    services = [
        _per_month_service(
            contact_id=flaky_contact,
            name="Office Cleaning", keyword="Flaky Co",
        ),
        _per_month_service(
            contact_id=healthy_contact,
            name="Office Cleaning", keyword="Healthy Co",
        ),
    ]
    fake_inv_repo = _FakeInvoiceRepo(
        cross_pipeline_hits={},
        raises_for=frozenset({str(flaky_contact)}),
    )

    monkeypatch.setattr(settings.invoicing, "enabled", True)
    monkeypatch.setattr(settings.invoicing, "auto_invoice_enabled", True)
    monkeypatch.setattr(settings.invoicing, "auto_invoice_review_mode", True)
    monkeypatch.setattr(settings.invoicing, "auto_invoice_save_path", str(tmp_path))
    monkeypatch.setattr(
        calendar_provider_mod, "get_calendar_provider", lambda: _FakeCalendarProvider()
    )
    monkeypatch.setattr(crm_provider_mod, "get_crm_provider", lambda: _FakeCRMProvider())
    monkeypatch.setattr(
        customer_service_mod,
        "get_customer_service_repo",
        lambda: _FakeServiceRepo(services),
    )
    monkeypatch.setattr(invoice_repo_mod, "get_invoice_repo", lambda: fake_inv_repo)

    task = ScheduledTask(
        id=uuid4(),
        name="monthly_invoice_generation",
        task_type="builtin",
        schedule_type="cron",
        cron_expression="0 8 1 * *",
        metadata={"billing_month": period},
    )

    result = await run(task)

    assert result["invoices_created"] == 1
    assert result["invoices_skipped_dedup"] == 0
    assert result["invoices_skipped_dedup_check_failed"] == 1
    assert len(fake_inv_repo.create_calls) == 1
    assert fake_inv_repo.create_calls[0]["contact_id"] == healthy_contact
    assert flaky_contact not in {c["contact_id"] for c in fake_inv_repo.create_calls}
