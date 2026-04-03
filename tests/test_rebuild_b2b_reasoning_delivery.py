import argparse
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "rebuild_b2b_reasoning_delivery.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_rebuild_b2b_reasoning_delivery_module",
    _SCRIPT_PATH,
)
assert _SPEC and _SPEC.loader
rebuild_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rebuild_mod)


@pytest.mark.asyncio
async def test_run_reasoning_rebuild_keeps_empty_scope_unforced(monkeypatch):
    fake_run = AsyncMock(return_value={"ok": True})
    monkeypatch.setitem(
        sys.modules,
        "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis",
        SimpleNamespace(run=fake_run),
    )

    result = await rebuild_mod._run_reasoning_rebuild(
        vendors=[],
        rebuild_cross_vendor=True,
    )

    assert result == {"ok": True}
    task = fake_run.await_args.args[0]
    assert task.metadata == {"force_cross_vendor": True}


@pytest.mark.asyncio
async def test_run_reasoning_rebuild_forces_only_scoped_vendors(monkeypatch):
    fake_run = AsyncMock(return_value={"ok": True})
    monkeypatch.setitem(
        sys.modules,
        "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis",
        SimpleNamespace(run=fake_run),
    )

    await rebuild_mod._run_reasoning_rebuild(
        vendors=["Zendesk", "Freshdesk"],
        rebuild_cross_vendor=False,
    )

    task = fake_run.await_args.args[0]
    assert task.metadata == {
        "force": True,
        "force_cross_vendor": False,
        "test_vendors": ["Zendesk", "Freshdesk"],
    }


@pytest.mark.asyncio
async def test_main_skips_reasoning_rebuild_when_no_stale_vendors(capsys, monkeypatch):
    monkeypatch.setattr(rebuild_mod, "init_database", AsyncMock())
    monkeypatch.setattr(rebuild_mod, "get_db_pool", Mock(return_value=object()))
    monkeypatch.setattr(rebuild_mod, "close_database", AsyncMock())
    monkeypatch.setattr(rebuild_mod, "_discover_stale_vendors", AsyncMock(return_value=[]))
    monkeypatch.setattr(rebuild_mod, "_summarize_report_coverage", AsyncMock(return_value=[]))
    run_rebuild = AsyncMock(return_value={"unexpected": True})
    monkeypatch.setattr(rebuild_mod, "_run_reasoning_rebuild", run_rebuild)
    run_refresh = AsyncMock(return_value={"reports_persisted": 1})
    monkeypatch.setattr(rebuild_mod, "_run_report_refresh", run_refresh)

    args = argparse.Namespace(
        apply=True,
        vendors="",
        limit=None,
        reports_only=False,
        cross_vendor_only=False,
        vendor_reasoning_only=False,
        full_refresh=False,
        include_reports=False,
        report_types="weekly_churn_feed",
        rebuild_cross_vendor=False,
    )

    await rebuild_mod._main(args)

    run_rebuild.assert_not_awaited()
    run_refresh.assert_not_awaited()
    payload = json.loads(capsys.readouterr().out)
    assert payload["reasoning_rebuild"] == {
        "_skip_synthesis": "No stale vendors selected",
    }


@pytest.mark.asyncio
async def test_main_reports_only_refreshes_reports_without_reasoning(capsys, monkeypatch):
    monkeypatch.setattr(rebuild_mod, "init_database", AsyncMock())
    monkeypatch.setattr(rebuild_mod, "get_db_pool", Mock(return_value=object()))
    monkeypatch.setattr(rebuild_mod, "close_database", AsyncMock())
    monkeypatch.setattr(rebuild_mod, "_discover_stale_vendors", AsyncMock(return_value=[]))
    monkeypatch.setattr(rebuild_mod, "_summarize_report_coverage", AsyncMock(return_value=[]))
    run_rebuild = AsyncMock(return_value={"unexpected": True})
    monkeypatch.setattr(rebuild_mod, "_run_reasoning_rebuild", run_rebuild)
    run_refresh = AsyncMock(return_value={"reports_persisted": 1})
    monkeypatch.setattr(rebuild_mod, "_run_report_refresh", run_refresh)

    args = argparse.Namespace(
        apply=True,
        vendors="",
        limit=None,
        reports_only=True,
        cross_vendor_only=False,
        vendor_reasoning_only=False,
        full_refresh=False,
        include_reports=False,
        report_types="weekly_churn_feed",
        rebuild_cross_vendor=False,
    )

    await rebuild_mod._main(args)

    run_rebuild.assert_not_awaited()
    run_refresh.assert_awaited_once()
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "reports_only"
    assert payload["reasoning_rebuild"] == {"_skip_synthesis": "reports-only mode"}
    assert payload["report_refresh"]["weekly_churn_feed"] == {"reports_persisted": 1}


@pytest.mark.asyncio
async def test_main_cross_vendor_only_preserves_selected_vendor_scope(capsys, monkeypatch):
    monkeypatch.setattr(rebuild_mod, "init_database", AsyncMock())
    monkeypatch.setattr(rebuild_mod, "get_db_pool", Mock(return_value=object()))
    monkeypatch.setattr(rebuild_mod, "close_database", AsyncMock())
    monkeypatch.setattr(rebuild_mod, "_discover_stale_vendors", AsyncMock(return_value=[]))
    monkeypatch.setattr(rebuild_mod, "_summarize_report_coverage", AsyncMock(return_value=[]))
    run_rebuild = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(rebuild_mod, "_run_reasoning_rebuild", run_rebuild)
    run_refresh = AsyncMock(return_value={"reports_persisted": 1})
    monkeypatch.setattr(rebuild_mod, "_run_report_refresh", run_refresh)

    args = argparse.Namespace(
        apply=True,
        vendors="Zendesk",
        limit=None,
        reports_only=False,
        cross_vendor_only=True,
        vendor_reasoning_only=False,
        full_refresh=False,
        include_reports=False,
        report_types="weekly_churn_feed",
        rebuild_cross_vendor=False,
    )

    await rebuild_mod._main(args)

    run_rebuild.assert_awaited_once_with(vendors=["Zendesk"], rebuild_cross_vendor=True)
    run_refresh.assert_not_awaited()
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "cross_vendor_only"
