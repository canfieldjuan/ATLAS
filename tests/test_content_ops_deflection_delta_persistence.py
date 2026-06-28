import json
from datetime import datetime, timezone

import pytest

from extracted_content_pipeline.deflection_report_access import (
    DeflectionDeltaBatchSummary,
    DeflectionDeltaReadError,
    DeflectionReportListRecord,
    InMemoryDeflectionReportArtifactStore,
    PostgresDeflectionReportArtifactStore,
    compute_and_save_previous_deflection_delta,
    compute_and_save_recent_deflection_deltas,
    deflection_delta_read_payload,
    fetch_paid_deflection_delta,
)


def _snapshot(label: str) -> dict[str, object]:
    return {"summary": {"generated": 1}, "top_questions": [{"question": label}]}


def _row(
    key: str,
    *,
    status: str = "Needs answer",
    ticket_count: int = 2,
    cost: float = 27.0,
) -> dict[str, object]:
    return {
        "repeat_key": key,
        "cluster_id": key,
        "identity_basis": "question_topic",
        "identity_confidence": "high",
        "question": f"Question {key}",
        "status": status,
        "owner_lane": "help_center",
        "fix_type": "create_missing_answer",
        "ticket_count": ticket_count,
        "estimated_support_cost": cost,
        "csat_signal": {
            "status": "insufficient_data",
            "csat_present_count": 0,
            "negative_csat_ticket_count": 0,
            "numeric_average": None,
        },
    }


def _model(
    *rows: dict[str, object],
    start: str = "2026-05-01",
    end: str = "2026-05-31",
) -> dict[str, object]:
    return {
        "schema_version": "deflection.v1",
        "title": "Support Ticket Deflection Report",
        "summary": {
            "source_date_start": start,
            "source_date_end": end,
            "source_window_days": 31,
        },
        "sections": [
            {
                "id": "backlog_table",
                "title": "Backlog",
                "priority": 90,
                "surfaces": ["web"],
                "default_limit": 25,
                "required_data": ["items"],
                "snapshot_safe_fields": [],
                "data": {"items": list(rows)},
            }
        ],
    }


def _artifact(model: dict[str, object]) -> dict[str, object]:
    return {"report_model": model}


async def _save(
    store: InMemoryDeflectionReportArtifactStore,
    *,
    account_id: str = "acct-1",
    request_id: str,
    model: dict[str, object] | None = None,
    paid: bool = True,
    delivery_email: str | None = None,
    created_at: datetime,
    paid_at: datetime | None = None,
) -> None:
    await store.save_report(
        account_id=account_id,
        request_id=request_id,
        snapshot=_snapshot(request_id),
        artifact=_artifact(model or _model(_row(request_id))),
        delivery_email=delivery_email,
    )
    store._created_at_by_key[(account_id, request_id)] = created_at
    if paid:
        assert await store.mark_paid(account_id=account_id, request_id=request_id)
        if paid_at is not None:
            store._paid_at_by_key[(account_id, request_id)] = paid_at


@pytest.mark.asyncio
async def test_in_memory_select_previous_paid_report_is_scoped_paid_and_ordered() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        request_id="older",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="previous",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="unpaid-between",
        paid=False,
        created_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="current",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="newer",
        created_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-2",
        request_id="other-tenant",
        created_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
    )

    selected = await store.select_previous_paid_report(
        account_id="acct-1",
        current_request_id="current",
    )

    assert selected is not None
    assert selected.request_id == "previous"
    assert await store.select_previous_paid_report(
        account_id="acct-1",
        current_request_id="missing",
    ) is None
    assert await store.mark_unpaid(account_id="acct-1", request_id="current")
    assert await store.select_previous_paid_report(
        account_id="acct-1",
        current_request_id="current",
    ) is None


@pytest.mark.asyncio
async def test_select_previous_paid_report_prefers_prior_source_window() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        request_id="april-window",
        model=_model(
            _row("repeat_1"),
            start="2026-04-01",
            end="2026-04-30",
        ),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="same-may-window-rerun",
        model=_model(
            _row("repeat_1"),
            start="2026-05-01",
            end="2026-05-31",
        ),
        created_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="current",
        model=_model(
            _row("repeat_1"),
            start="2026-05-01",
            end="2026-05-31",
        ),
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    selected = await store.select_previous_paid_report(
        account_id="acct-1",
        current_request_id="current",
    )

    assert selected is not None
    assert selected.request_id == "april-window"


@pytest.mark.asyncio
async def test_select_previous_paid_report_ignores_invalid_candidate_source_dates() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        request_id="valid-april-window",
        model=_model(
            _row("repeat_1"),
            start="2026-04-01",
            end="2026-04-30",
        ),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="invalid-calendar-window",
        model=_model(
            _row("repeat_1"),
            start="2026-04-01",
            end="2026-04-99",
        ),
        created_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="current",
        model=_model(
            _row("repeat_1"),
            start="2026-05-01",
            end="2026-05-31",
        ),
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    selected = await store.select_previous_paid_report(
        account_id="acct-1",
        current_request_id="current",
    )

    assert selected is not None
    assert selected.request_id == "valid-april-window"


@pytest.mark.asyncio
async def test_select_previous_paid_report_falls_back_when_source_dates_missing() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    current_model = _model(_row("repeat_1"))
    assert isinstance(current_model["summary"], dict)
    current_model["summary"].pop("source_date_start")
    await _save(
        store,
        request_id="older",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="previous",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="current",
        model=current_model,
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    selected = await store.select_previous_paid_report(
        account_id="acct-1",
        current_request_id="current",
    )

    assert selected is not None
    assert selected.request_id == "previous"


@pytest.mark.asyncio
async def test_select_previous_paid_report_falls_back_when_current_source_date_invalid() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        request_id="older-source-window",
        model=_model(
            _row("repeat_1"),
            start="2026-04-01",
            end="2026-04-30",
        ),
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="previous-created",
        model=_model(
            _row("repeat_1"),
            start="2026-05-01",
            end="2026-05-31",
        ),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="current",
        model=_model(
            _row("repeat_1"),
            start="2026-05-99",
            end="2026-05-31",
        ),
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    selected = await store.select_previous_paid_report(
        account_id="acct-1",
        current_request_id="current",
    )

    assert selected is not None
    assert selected.request_id == "previous-created"


@pytest.mark.asyncio
async def test_in_memory_paid_account_discovery_orders_by_paid_activity() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-old-report-recent-pay",
        request_id="old-report",
        created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-new-report-old-pay",
        request_id="new-report",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-unpaid",
        request_id="newer-unpaid",
        paid=False,
        created_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )

    assert await store.list_paid_report_accounts(limit=10) == (
        "acct-old-report-recent-pay",
        "acct-new-report-old-pay",
    )


@pytest.mark.asyncio
async def test_in_memory_paid_account_discovery_filters_entitled_accounts() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-entitled-old",
        request_id="old-report",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-paid-not-entitled",
        request_id="newer-report",
        created_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-entitled-new",
        request_id="new-report",
        created_at=datetime(2026, 5, 3, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )

    accounts = await store.list_paid_report_accounts(
        limit=10,
        account_ids=(
            "acct-entitled-old",
            "acct-entitled-new",
            "acct-missing",
        ),
    )

    assert accounts == ("acct-entitled-new", "acct-entitled-old")
    assert await store.count_paid_report_accounts(
        account_ids=("acct-entitled-old", "acct-entitled-new")
    ) == 2


@pytest.mark.asyncio
async def test_in_memory_paid_report_listing_orders_by_paid_activity_window() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        request_id="old-report-recent-pay",
        created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="new-report-old-pay",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="newest-report-oldest-pay",
        created_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    paid_rows = await store.list_reports(account_id="acct-1", limit=2, paid=True)
    all_rows = await store.list_reports(account_id="acct-1", limit=2)

    assert [row.request_id for row in paid_rows] == [
        "old-report-recent-pay",
        "new-report-old-pay",
    ]
    assert [row.request_id for row in all_rows] == [
        "newest-report-oldest-pay",
        "new-report-old-pay",
    ]


@pytest.mark.asyncio
async def test_compute_and_save_previous_delta_persists_pair_payload() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    baseline_model = _model(_row("repeat_1", ticket_count=2, cost=27.0))
    current_model = _model(_row("repeat_1", ticket_count=5, cost=67.5))
    await _save(
        store,
        request_id="baseline",
        model=baseline_model,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="current",
        model=current_model,
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    record = await compute_and_save_previous_deflection_delta(
        store,
        account_id="acct-1",
        current_request_id="current",
    )

    assert record is not None
    assert record.account_id == "acct-1"
    assert record.current_request_id == "current"
    assert record.baseline_request_id == "baseline"
    assert record.delta["schema_version"] == "deflection_delta.v1"
    assert record.delta["summary"]["matched_item_count"] == 1
    assert record.delta["summary"]["growing_count"] == 1
    stored = await store.get_deflection_delta(
        account_id="acct-1",
        current_request_id="current",
        baseline_request_id="baseline",
    )
    assert stored is not None
    assert json.loads(json.dumps(stored.delta)) == stored.delta


@pytest.mark.asyncio
async def test_recent_delta_batch_discovers_paid_accounts_and_stays_tenant_scoped() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-1",
        request_id="acct-1-baseline",
        model=_model(_row("repeat_1", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-1",
        request_id="acct-1-current",
        model=_model(_row("repeat_1", ticket_count=5, cost=67.5)),
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-2",
        request_id="acct-2-current",
        model=_model(_row("repeat_2", ticket_count=4, cost=54.0)),
        created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-3",
        request_id="acct-3-unpaid",
        paid=False,
        created_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
    )

    accounts = await store.list_paid_report_accounts(limit=10)
    summary = await compute_and_save_recent_deflection_deltas(
        store,
        account_limit=10,
        reports_per_account=10,
    )

    assert accounts == ("acct-2", "acct-1")
    assert summary == DeflectionDeltaBatchSummary(
        accounts_scanned=2,
        reports_scanned=3,
        deltas_saved=1,
        skipped_no_delta=2,
        failed=0,
    )
    assert await store.get_deflection_delta(
        account_id="acct-1",
        current_request_id="acct-1-current",
        baseline_request_id="acct-1-baseline",
    )
    assert await store.get_deflection_delta(
        account_id="acct-2",
        current_request_id="acct-2-current",
        baseline_request_id="acct-1-current",
    ) is None


@pytest.mark.asyncio
async def test_recent_delta_batch_scans_only_entitled_paid_accounts() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-entitled",
        request_id="entitled-baseline",
        model=_model(_row("repeat_entitled", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-entitled",
        request_id="entitled-current",
        model=_model(_row("repeat_entitled", ticket_count=5, cost=67.5)),
        delivery_email="entitled@example.com",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-paid-not-entitled",
        request_id="unentitled-baseline",
        model=_model(_row("repeat_unentitled", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-paid-not-entitled",
        request_id="unentitled-current",
        model=_model(_row("repeat_unentitled", ticket_count=7, cost=94.5)),
        delivery_email="unentitled@example.com",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
    )

    summary = await compute_and_save_recent_deflection_deltas(
        store,
        entitled_account_ids=("acct-entitled",),
        account_limit=10,
        reports_per_account=10,
    )

    assert summary == DeflectionDeltaBatchSummary(
        accounts_scanned=1,
        reports_scanned=2,
        deltas_saved=1,
        delta_deliveries_enqueued=1,
        skipped_no_delta=1,
        failed=0,
    )
    assert await store.get_deflection_delta(
        account_id="acct-entitled",
        current_request_id="entitled-current",
        baseline_request_id="entitled-baseline",
    )
    assert await store.get_deflection_delta(
        account_id="acct-paid-not-entitled",
        current_request_id="unentitled-current",
        baseline_request_id="unentitled-baseline",
    ) is None
    assert store._delta_delivery_keys == {
        ("acct-entitled", "entitled-current", "entitled-baseline")
    }


@pytest.mark.asyncio
async def test_recent_delta_batch_account_scope_scans_only_requested_account() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-target",
        request_id="target-baseline",
        model=_model(_row("target_repeat", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-target",
        request_id="target-current",
        model=_model(_row("target_repeat", ticket_count=5, cost=67.5)),
        delivery_email="target@example.com",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-other",
        request_id="other-baseline",
        model=_model(_row("other_repeat", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-other",
        request_id="other-current",
        model=_model(_row("other_repeat", ticket_count=6, cost=81.0)),
        delivery_email="other@example.com",
        created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
    )

    summary = await compute_and_save_recent_deflection_deltas(
        store,
        account_id=" acct-target ",
        account_limit=1,
        reports_per_account=10,
    )

    assert summary == DeflectionDeltaBatchSummary(
        accounts_scanned=1,
        reports_scanned=2,
        deltas_saved=1,
        delta_deliveries_enqueued=1,
        skipped_no_delta=1,
        failed=0,
    )
    assert summary.account_limit_reached is False
    assert await store.get_deflection_delta(
        account_id="acct-target",
        current_request_id="target-current",
        baseline_request_id="target-baseline",
    )
    assert await store.get_deflection_delta(
        account_id="acct-other",
        current_request_id="other-current",
        baseline_request_id="other-baseline",
    ) is None
    assert store._delta_delivery_keys == {
        ("acct-target", "target-current", "target-baseline")
    }


@pytest.mark.asyncio
async def test_recent_delta_batch_current_request_scope_uses_only_checked_report() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-target",
        request_id="oldest",
        model=_model(_row("repeat", ticket_count=1, cost=13.5)),
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-target",
        request_id="checked-current",
        model=_model(_row("repeat", ticket_count=3, cost=40.5)),
        delivery_email="checked@example.com",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-target",
        request_id="newer-current",
        model=_model(_row("repeat", ticket_count=5, cost=67.5)),
        delivery_email="newer@example.com",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
    )

    summary = await compute_and_save_recent_deflection_deltas(
        store,
        account_id="acct-target",
        current_request_id="checked-current",
        reports_per_account=100,
    )

    assert summary == DeflectionDeltaBatchSummary(
        accounts_scanned=1,
        reports_scanned=1,
        deltas_saved=1,
        delta_deliveries_enqueued=1,
        skipped_no_delta=0,
        failed=0,
    )
    assert await store.get_deflection_delta(
        account_id="acct-target",
        current_request_id="checked-current",
        baseline_request_id="oldest",
    )
    assert await store.get_deflection_delta(
        account_id="acct-target",
        current_request_id="newer-current",
        baseline_request_id="checked-current",
    ) is None
    assert store._delta_delivery_keys == {
        ("acct-target", "checked-current", "oldest")
    }


@pytest.mark.asyncio
async def test_recent_delta_batch_enqueues_delivery_for_current_report_email() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-1",
        request_id="baseline",
        model=_model(_row("repeat_1", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-1",
        request_id="current",
        model=_model(_row("repeat_1", ticket_count=5, cost=67.5)),
        delivery_email="buyer@example.com",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )

    first = await compute_and_save_recent_deflection_deltas(store)
    second = await compute_and_save_recent_deflection_deltas(store)

    assert first.deltas_saved == 1
    assert first.delta_deliveries_enqueued == 1
    assert second.deltas_saved == 1
    assert second.delta_deliveries_enqueued == 0
    assert store._delta_delivery_keys == {("acct-1", "current", "baseline")}


@pytest.mark.asyncio
async def test_recent_delta_batch_marks_saturated_account_and_report_windows() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-1",
        request_id="acct-1-baseline",
        model=_model(_row("repeat_1", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-1",
        request_id="acct-1-current",
        model=_model(_row("repeat_1", ticket_count=5, cost=67.5)),
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-2",
        request_id="acct-2-current",
        model=_model(_row("repeat_2", ticket_count=4, cost=54.0)),
        created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    summary = await compute_and_save_recent_deflection_deltas(
        store,
        account_limit=1,
        reports_per_account=2,
    )

    assert summary.accounts_scanned == 1
    assert summary.reports_scanned == 2
    assert summary.deltas_saved == 1
    assert summary.skipped_no_delta == 1
    assert summary.failed == 0
    assert summary.account_limit_reached is True
    assert summary.account_limit_overflow is True
    assert summary.reports_per_account_limit_reached is True
    assert summary.reports_per_account_limit_overflow is False
    assert summary.report_limit_reached_accounts == ("acct-1",)
    assert summary.report_limit_overflow_accounts == ()


@pytest.mark.asyncio
async def test_recent_delta_batch_keeps_exact_fill_saturation_out_of_overflow() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-1",
        request_id="acct-1-baseline",
        model=_model(_row("repeat_1", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-1",
        request_id="acct-1-current",
        model=_model(_row("repeat_1", ticket_count=5, cost=67.5)),
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )

    summary = await compute_and_save_recent_deflection_deltas(
        store,
        account_limit=1,
        reports_per_account=2,
    )

    assert summary.accounts_scanned == 1
    assert summary.reports_scanned == 2
    assert summary.account_limit_reached is True
    assert summary.account_limit_overflow is False
    assert summary.reports_per_account_limit_reached is True
    assert summary.reports_per_account_limit_overflow is False
    assert summary.report_limit_reached_accounts == ("acct-1",)
    assert summary.report_limit_overflow_accounts == ()


@pytest.mark.asyncio
async def test_recent_delta_batch_marks_report_window_overflow() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        account_id="acct-1",
        request_id="acct-1-oldest",
        model=_model(_row("repeat_1", ticket_count=1, cost=13.5)),
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 8, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-1",
        request_id="acct-1-middle",
        model=_model(_row("repeat_1", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-1",
        request_id="acct-1-newest",
        model=_model(_row("repeat_1", ticket_count=5, cost=67.5)),
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        paid_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )

    summary = await compute_and_save_recent_deflection_deltas(
        store,
        account_limit=10,
        reports_per_account=2,
    )

    assert summary.accounts_scanned == 1
    assert summary.reports_scanned == 2
    assert summary.account_limit_reached is False
    assert summary.account_limit_overflow is False
    assert summary.reports_per_account_limit_reached is True
    assert summary.reports_per_account_limit_overflow is True
    assert summary.report_limit_reached_accounts == ("acct-1",)
    assert summary.report_limit_overflow_accounts == ("acct-1",)


@pytest.mark.asyncio
async def test_recent_delta_batch_logs_per_report_failures(caplog) -> None:
    class _FailingStore:
        async def list_paid_report_accounts(
            self,
            *,
            limit: int | None = 100,
            account_ids: tuple[str, ...] | None = None,
        ) -> tuple[str, ...]:
            assert account_ids is None
            return ("acct-1",)

        async def list_reports(
            self,
            *,
            account_id: str,
            limit: int | None = 25,
            paid: bool | None = None,
        ) -> tuple[DeflectionReportListRecord, ...]:
            return (
                DeflectionReportListRecord(
                    account_id=account_id,
                    request_id="broken-report",
                    snapshot={},
                    paid=True,
                ),
            )

        async def get_artifact_record(self, *, account_id: str, request_id: str) -> None:
            raise RuntimeError("delta source exploded")

    caplog.set_level("WARNING", logger="extracted_content_pipeline.deflection_report_access")

    summary = await compute_and_save_recent_deflection_deltas(
        _FailingStore(),
        account_limit=10,
        reports_per_account=10,
    )

    assert summary == DeflectionDeltaBatchSummary(
        accounts_scanned=1,
        reports_scanned=1,
        deltas_saved=0,
        skipped_no_delta=0,
        failed=1,
    )
    assert "account=acct-1 report=broken-report" in caplog.text


@pytest.mark.asyncio
async def test_fetch_paid_delta_requires_paid_source_reports() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        request_id="baseline",
        model=_model(_row("repeat_1", ticket_count=2, cost=27.0)),
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="current",
        model=_model(_row("repeat_1", ticket_count=5, cost=67.5)),
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    saved = await compute_and_save_previous_deflection_delta(
        store,
        account_id="acct-1",
        current_request_id="current",
    )
    assert saved is not None

    fetched = await fetch_paid_deflection_delta(
        store,
        account_id="acct-1",
        current_request_id="current",
    )
    payload = deflection_delta_read_payload(fetched)
    assert payload["schema_version"] == "deflection_delta_read.v1"
    assert payload["current_request_id"] == "current"
    assert payload["baseline_request_id"] == "baseline"
    assert payload["delta"]["schema_version"] == "deflection_delta.v1"

    assert await store.mark_unpaid(account_id="acct-1", request_id="baseline")
    with pytest.raises(DeflectionDeltaReadError) as baseline_locked:
        await fetch_paid_deflection_delta(
            store,
            account_id="acct-1",
            current_request_id="current",
            baseline_request_id="baseline",
        )
    assert baseline_locked.value.code == "baseline_report_locked"

    assert await store.mark_paid(account_id="acct-1", request_id="baseline")
    assert await store.mark_unpaid(account_id="acct-1", request_id="current")
    with pytest.raises(DeflectionDeltaReadError) as current_locked:
        await fetch_paid_deflection_delta(
            store,
            account_id="acct-1",
            current_request_id="current",
        )
    assert current_locked.value.code == "current_report_locked"


@pytest.mark.asyncio
async def test_fetch_paid_delta_requires_stored_pair_for_bound_tenant() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        request_id="baseline",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="current",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        account_id="acct-2",
        request_id="baseline",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    with pytest.raises(DeflectionDeltaReadError) as missing_delta:
        await fetch_paid_deflection_delta(
            store,
            account_id="acct-1",
            current_request_id="current",
            baseline_request_id="baseline",
        )
    assert missing_delta.value.code == "delta_not_found"

    with pytest.raises(DeflectionDeltaReadError) as missing_baseline:
        await fetch_paid_deflection_delta(
            store,
            account_id="acct-1",
            current_request_id="current",
            baseline_request_id="other-tenant-baseline",
        )
    assert missing_baseline.value.code == "baseline_report_not_found"


@pytest.mark.asyncio
async def test_fetch_paid_delta_rejects_unsupported_delta_schema() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        request_id="baseline",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    await _save(
        store,
        request_id="current",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await store.save_deflection_delta(
        account_id="acct-1",
        current_request_id="current",
        baseline_request_id="baseline",
        delta={"schema_version": "future_delta.v2", "items": [{"raw": "nope"}]},
    )

    with pytest.raises(DeflectionDeltaReadError) as exc:
        await fetch_paid_deflection_delta(
            store,
            account_id="acct-1",
            current_request_id="current",
            baseline_request_id="baseline",
        )

    assert exc.value.code == "unsupported_delta_schema"


@pytest.mark.asyncio
async def test_compute_and_save_previous_delta_fails_closed_without_paid_models() -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await _save(
        store,
        request_id="current-only",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert await compute_and_save_previous_deflection_delta(
        store,
        account_id="acct-1",
        current_request_id="current-only",
    ) is None

    await _save(
        store,
        request_id="baseline-invalid",
        model={"schema_version": "legacy.v0"},
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    assert await compute_and_save_previous_deflection_delta(
        store,
        account_id="acct-1",
        current_request_id="current-only",
    ) is None


@pytest.mark.asyncio
async def test_deflection_delta_store_rejects_same_current_and_baseline() -> None:
    store = InMemoryDeflectionReportArtifactStore()

    with pytest.raises(ValueError, match="must differ"):
        await store.save_deflection_delta(
            account_id="acct-1",
            current_request_id="same",
            baseline_request_id="same",
            delta={"schema_version": "deflection_delta.v1"},
        )
    with pytest.raises(ValueError, match="reports must exist"):
        await store.save_deflection_delta(
            account_id="acct-1",
            current_request_id="current",
            baseline_request_id="baseline",
            delta={"schema_version": "deflection_delta.v1"},
        )


@pytest.mark.asyncio
async def test_postgres_delta_methods_are_account_scoped_and_jsonb_encoded() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
            self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
            self.fetchrow_calls.append((query, args))
            if "WITH current_report" in query:
                return {
                    "account_id": "acct-1",
                    "request_id": "baseline",
                    "snapshot": json.dumps(_snapshot("baseline")),
                    "artifact": json.dumps(_artifact(_model(_row("repeat_1")))),
                    "paid": True,
                    "payment_reference": None,
                    "delivery_email": None,
                }
            return {
                "account_id": "acct-1",
                "current_request_id": "current",
                "baseline_request_id": "baseline",
                "delta": json.dumps({"schema_version": "deflection_delta.v1"}),
                "created_at": "2026-06-01T00:00:00Z",
                "updated_at": "2026-06-01T00:00:00Z",
            }

        async def execute(self, query: str, *args: object) -> str:
            self.execute_calls.append((query, args))
            return "INSERT 0 1"

    pool = _Pool()
    store = PostgresDeflectionReportArtifactStore(pool=pool)

    selected = await store.select_previous_paid_report(
        account_id=" acct-1 ",
        current_request_id=" current ",
    )
    await store.save_deflection_delta(
        account_id="acct-1",
        current_request_id="current",
        baseline_request_id="baseline",
        delta={"schema_version": "deflection_delta.v1"},
    )
    stored = await store.get_deflection_delta(
        account_id="acct-1",
        current_request_id="current",
        baseline_request_id="baseline",
    )
    paid_stored = await store.get_paid_deflection_delta(
        account_id="acct-1",
        current_request_id="current",
        baseline_request_id="baseline",
    )
    enqueued = await store.enqueue_deflection_delta_delivery(
        account_id="acct-1",
        current_request_id="current",
        baseline_request_id="baseline",
        delivery_email="buyer@example.com",
    )

    assert selected is not None
    select_query, select_args = pool.fetchrow_calls[0]
    assert select_args == ("acct-1", "current")
    assert "WHERE account_id = $1" in select_query
    assert "reports.account_id = $1" in select_query
    assert "reports.paid = true" in select_query
    assert "reports.created_at < current_report.created_at" in select_query
    assert "artifact #>> '{report_model,summary,source_date_start}'" in select_query
    assert "artifact #>> '{report_model,summary,source_date_end}'" in select_query
    assert "to_date(source_date_start, 'YYYY-MM-DD')" in select_query
    assert "to_date(source_date_end, 'YYYY-MM-DD')" in select_query
    assert "to_char(to_date(source_date_start, 'YYYY-MM-DD')" in select_query
    assert "to_char(to_date(source_date_end, 'YYYY-MM-DD')" in select_query
    assert "source_date_end < current_source_date_start" in select_query
    assert "current_source_year % 400" not in select_query
    assert "source_end_year % 400" not in select_query
    insert_query, insert_args = pool.execute_calls[0]
    assert "INSERT INTO content_ops_deflection_deltas" in insert_query
    assert "ON CONFLICT (account_id, current_request_id, baseline_request_id)" in insert_query
    assert insert_args[:3] == ("acct-1", "current", "baseline")
    assert json.loads(str(insert_args[3])) == {"schema_version": "deflection_delta.v1"}
    assert stored is not None
    assert stored.delta == {"schema_version": "deflection_delta.v1"}
    assert paid_stored is not None
    paid_query, paid_args = pool.fetchrow_calls[2]
    assert paid_args == ("acct-1", "current", "baseline")
    assert "FROM content_ops_deflection_deltas deltas" in paid_query
    assert "JOIN content_ops_deflection_reports current_report" in paid_query
    assert "JOIN content_ops_deflection_reports baseline_report" in paid_query
    assert "current_report.paid = true" in paid_query
    assert "baseline_report.paid = true" in paid_query
    enqueue_query, enqueue_args = pool.execute_calls[1]
    assert enqueued is True
    assert "INSERT INTO content_ops_deflection_delta_deliveries" in enqueue_query
    assert "ON CONFLICT (account_id, current_request_id, baseline_request_id)" in enqueue_query
    assert "DO UPDATE" in enqueue_query
    assert "delivery_email = EXCLUDED.delivery_email" in enqueue_query
    assert "source_report_not_paid" in enqueue_query
    assert "delta_no_longer_sendable" in enqueue_query
    assert "delivery_status = 'pending'" in enqueue_query
    assert "delivery_status = 'failed'" in enqueue_query
    assert "delivery_status = 'delivered'" not in enqueue_query
    assert enqueue_args == ("acct-1", "current", "baseline", "buyer@example.com")


@pytest.mark.asyncio
async def test_postgres_paid_account_discovery_is_paid_scoped_ordered_and_bounded() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
            self.fetch_calls.append((query, args))
            return [{"account_id": "acct-2"}, {"account_id": "acct-1"}]

    pool = _Pool()
    store = PostgresDeflectionReportArtifactStore(pool=pool)

    accounts = await store.list_paid_report_accounts(limit=7)

    assert accounts == ("acct-2", "acct-1")
    query, args = pool.fetch_calls[0]
    assert args == (7,)
    assert "WHERE paid = true" in query
    assert "GROUP BY account_id" in query
    assert (
        "ORDER BY MAX(COALESCE(paid_at, updated_at, created_at)) DESC, account_id ASC"
        in query
    )
    assert "LIMIT $1" in query


@pytest.mark.asyncio
async def test_postgres_paid_count_probes_are_paid_scoped() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetchrow(self, query: str, *args: object) -> dict[str, object]:
            self.fetchrow_calls.append((query, args))
            return {"count": 3}

    pool = _Pool()
    store = PostgresDeflectionReportArtifactStore(pool=pool)

    account_count = await store.count_paid_report_accounts()
    report_count = await store.count_paid_reports(account_id=" acct-1 ")

    assert account_count == 3
    assert report_count == 3
    account_query, account_args = pool.fetchrow_calls[0]
    assert account_args == ()
    assert "COUNT(DISTINCT account_id)" in account_query
    assert "WHERE paid = true" in account_query
    report_query, report_args = pool.fetchrow_calls[1]
    assert report_args == ("acct-1",)
    assert "COUNT(*)" in report_query
    assert "WHERE account_id = $1" in report_query
    assert "AND paid = true" in report_query


@pytest.mark.asyncio
async def test_postgres_paid_report_listing_orders_by_paid_activity() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
            self.fetch_calls.append((query, args))
            return [
                {
                    "account_id": "acct-1",
                    "request_id": "old-report-recent-pay",
                    "snapshot": json.dumps(_snapshot("old-report-recent-pay")),
                    "paid": True,
                    "delivery_email": None,
                    "created_at": "2026-03-01T00:00:00Z",
                    "updated_at": "2026-06-10T00:00:00Z",
                },
            ]

    pool = _Pool()
    store = PostgresDeflectionReportArtifactStore(pool=pool)

    reports = await store.list_reports(account_id=" acct-1 ", limit=2, paid=True)

    assert [row.request_id for row in reports] == ["old-report-recent-pay"]
    query, args = pool.fetch_calls[0]
    assert args == ("acct-1", True, 2)
    assert "AND paid = $2" in query
    assert (
        "ORDER BY COALESCE(paid_at, updated_at, created_at) DESC, request_id ASC"
        in query
    )
    assert "ORDER BY created_at DESC" not in query


@pytest.mark.asyncio
async def test_postgres_unpaid_report_listing_keeps_created_ordering() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

        async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
            self.fetch_calls.append((query, args))
            return []

    pool = _Pool()
    store = PostgresDeflectionReportArtifactStore(pool=pool)

    reports = await store.list_reports(account_id="acct-1", limit=3, paid=False)

    assert reports == ()
    query, args = pool.fetch_calls[0]
    assert args == ("acct-1", False, 3)
    assert "AND paid = $2" in query
    assert "ORDER BY created_at DESC, request_id ASC" in query
