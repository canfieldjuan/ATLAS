"""Tests for the witness-quality maintenance service.

Covers:
  - run_witness_quality_maintenance: combines backfill + audit and
    derives the alert decision.
  - Alert decision logic (_surface_fillable_summary): zero-fillable
    surfaces don't trigger; any positive fillable across any surface
    flips the flag.
  - The autonomous task handler in
    atlas_brain/autonomous/tasks/b2b_witness_quality_maintenance.py:
    metadata coercion, _skip_synthesis hint, alert dispatch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from atlas_brain.services import witness_quality_maintenance as svc


def test_surface_fillable_summary_zero_when_all_clean():
    audit = {
        "surfaces": [
            {"surface": "a", "fillable_missing_fields": 0, "witness_objects": 10},
            {"surface": "b", "fillable_missing_fields": 0, "witness_objects": 5},
        ]
    }
    total, leaking = svc._surface_fillable_summary(audit)
    assert total == 0
    assert leaking == []


def test_surface_fillable_summary_picks_only_leaking_surfaces():
    audit = {
        "surfaces": [
            {"surface": "a", "fillable_missing_fields": 0, "witness_objects": 10},
            {"surface": "b", "fillable_missing_fields": 7, "witness_objects": 5},
            {"surface": "c", "fillable_missing_fields": 3, "witness_objects": 12},
        ]
    }
    total, leaking = svc._surface_fillable_summary(audit)
    assert total == 10
    assert {item["surface"] for item in leaking} == {"b", "c"}
    # Counts preserved
    by_name = {item["surface"]: item for item in leaking}
    assert by_name["b"]["fillable_missing_fields"] == 7
    assert by_name["c"]["witness_objects"] == 12


def test_surface_fillable_summary_handles_missing_keys():
    audit = {"surfaces": [{"surface": "a"}]}  # no fillable_missing_fields key
    total, leaking = svc._surface_fillable_summary(audit)
    assert total == 0
    assert leaking == []


@pytest.mark.asyncio
async def test_run_witness_quality_maintenance_clean_state(monkeypatch):
    """When the audit reports zero fillable, the service returns alert=False."""
    backfill_result = {
        "apply": True,
        "days": 30,
        "limit": None,
        "overwrite": False,
        "tables": {
            "b2b_reasoning_synthesis": {"changed_rows": 0, "fields_written": 0},
            "b2b_cross_vendor_reasoning_synthesis": {"changed_rows": 0, "fields_written": 0},
            "b2b_intelligence": {"changed_rows": 0, "fields_written": 0},
        },
    }
    audit_result = {
        "days": 30,
        "row_limit": 500,
        "quality_fields": list(svc.summarize_witness_field_propagation.__defaults__ or []),
        "surfaces": [
            {"surface": "b2b_intelligence:battle_card", "fillable_missing_fields": 0, "witness_objects": 100},
            {"surface": "b2b_reasoning_synthesis", "fillable_missing_fields": 0, "witness_objects": 50},
        ],
    }
    monkeypatch.setattr(svc, "run_backfill", AsyncMock(return_value=backfill_result))
    monkeypatch.setattr(
        svc,
        "summarize_witness_field_propagation",
        AsyncMock(return_value=audit_result),
    )

    result = await svc.run_witness_quality_maintenance(
        pool=object(),
        days=30,
        apply=True,
    )
    assert result["alert_triggered"] is False
    assert result["fillable_missing_fields"] == 0
    assert result["leaking_surfaces"] == []
    assert result["backfill"] is backfill_result
    assert result["audit"] is audit_result


@pytest.mark.asyncio
async def test_run_witness_quality_maintenance_alert_when_audit_leaks(monkeypatch):
    """When the audit reports any fillable_missing > 0, service flags it."""
    backfill_result = {
        "apply": True, "days": 30, "limit": None, "overwrite": False,
        "tables": {},
    }
    audit_result = {
        "days": 30,
        "surfaces": [
            {"surface": "b2b_intelligence:challenger_brief", "fillable_missing_fields": 12, "witness_objects": 50},
            {"surface": "b2b_reasoning_synthesis", "fillable_missing_fields": 0, "witness_objects": 30},
        ],
    }
    monkeypatch.setattr(svc, "run_backfill", AsyncMock(return_value=backfill_result))
    monkeypatch.setattr(
        svc,
        "summarize_witness_field_propagation",
        AsyncMock(return_value=audit_result),
    )

    result = await svc.run_witness_quality_maintenance(pool=object(), days=30, apply=True)
    assert result["alert_triggered"] is True
    assert result["fillable_missing_fields"] == 12
    assert len(result["leaking_surfaces"]) == 1
    assert result["leaking_surfaces"][0]["surface"] == "b2b_intelligence:challenger_brief"


# ---------------------------------------------------------------------------
# Task handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_handler_no_pool_short_circuits(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_witness_quality_maintenance as task_mod

    monkeypatch.setattr(task_mod, "get_db_pool", lambda: None)

    class _Task:
        metadata: dict[str, Any] = {}

    result = await task_mod.run(_Task())
    assert result["_skip_synthesis"] == "DB pool unavailable"
    assert result["alert_triggered"] is False


@pytest.mark.asyncio
async def test_task_handler_passes_metadata_overrides(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_witness_quality_maintenance as task_mod

    captured: dict = {}

    async def _fake_maintenance(pool, *, days, apply, overwrite, audit_row_limit):
        captured["days"] = days
        captured["apply"] = apply
        captured["overwrite"] = overwrite
        captured["audit_row_limit"] = audit_row_limit
        return {
            "fillable_missing_fields": 0,
            "alert_triggered": False,
            "backfill": {"tables": {}},
            "audit": {"surfaces": []},
            "leaking_surfaces": [],
        }

    monkeypatch.setattr(task_mod, "get_db_pool", lambda: object())
    monkeypatch.setattr(task_mod, "run_witness_quality_maintenance", _fake_maintenance)

    class _Task:
        metadata = {"days": 7, "apply": False, "overwrite": True, "audit_row_limit": 100}

    result = await task_mod.run(_Task())
    assert captured == {"days": 7, "apply": False, "overwrite": True, "audit_row_limit": 100}
    assert "_skip_synthesis" in result


@pytest.mark.asyncio
async def test_task_handler_alerts_on_leak(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_witness_quality_maintenance as task_mod

    async def _fake_maintenance(pool, *, days, apply, overwrite, audit_row_limit):
        return {
            "fillable_missing_fields": 5,
            "alert_triggered": True,
            "leaking_surfaces": [
                {"surface": "b2b_intelligence:battle_card", "fillable_missing_fields": 5, "witness_objects": 20},
            ],
            "backfill": {"tables": {"b2b_intelligence": {"changed_rows": 1, "fields_written": 0}}},
            "audit": {"surfaces": []},
        }

    sent: dict = {}

    async def _fake_send_alert(*, fillable_total, leaking):
        sent["fillable_total"] = fillable_total
        sent["leaking"] = leaking

    monkeypatch.setattr(task_mod, "get_db_pool", lambda: object())
    monkeypatch.setattr(task_mod, "run_witness_quality_maintenance", _fake_maintenance)
    monkeypatch.setattr(task_mod, "_send_alert", _fake_send_alert)

    class _Task:
        metadata: dict[str, Any] = {}

    result = await task_mod.run(_Task())
    assert result["alert_triggered"] is True
    assert sent["fillable_total"] == 5
    assert sent["leaking"][0]["surface"] == "b2b_intelligence:battle_card"


@pytest.mark.asyncio
async def test_task_handler_no_alert_when_clean(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_witness_quality_maintenance as task_mod

    async def _fake_maintenance(pool, *, days, apply, overwrite, audit_row_limit):
        return {
            "fillable_missing_fields": 0,
            "alert_triggered": False,
            "leaking_surfaces": [],
            "backfill": {"tables": {}},
            "audit": {"surfaces": []},
        }

    sent_count = 0

    async def _fake_send_alert(*, fillable_total, leaking):
        nonlocal sent_count
        sent_count += 1

    monkeypatch.setattr(task_mod, "get_db_pool", lambda: object())
    monkeypatch.setattr(task_mod, "run_witness_quality_maintenance", _fake_maintenance)
    monkeypatch.setattr(task_mod, "_send_alert", _fake_send_alert)

    class _Task:
        metadata: dict[str, Any] = {}

    result = await task_mod.run(_Task())
    assert result["alert_triggered"] is False
    assert sent_count == 0  # silent success path
