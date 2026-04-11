import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "test_intelligence_subset.py"
_SPEC = importlib.util.spec_from_file_location("test_intelligence_subset", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.asyncio
async def test_main_exits_when_b2b_churn_disabled(monkeypatch, capsys):
    init_db = AsyncMock()
    close_db = AsyncMock()
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]))

    monkeypatch.setattr(_MODULE, "init_database", init_db)
    monkeypatch.setattr(_MODULE, "close_database", close_db)
    monkeypatch.setattr(_MODULE, "get_db_pool", lambda: pool)
    monkeypatch.setattr(
        _MODULE,
        "settings",
        SimpleNamespace(
            b2b_churn=SimpleNamespace(
                enabled=False,
                intelligence_llm_backend="anthropic",
            )
        ),
    )

    with pytest.raises(SystemExit) as exc:
        await _MODULE._main(Namespace(vendors="Shopify,Salesforce", report="vendor_scorecard", skip_intelligence=False))

    assert exc.value.code == 1
    init_db.assert_awaited_once()
    close_db.assert_awaited_once()
    assert "ERROR: B2B churn pipeline disabled. Set ATLAS_B2B_CHURN_ENABLED=true" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_main_skips_intelligence_and_runs_reports(monkeypatch, capsys):
    init_db = AsyncMock()
    close_db = AsyncMock()
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]))
    run_intelligence = AsyncMock()
    run_reports = AsyncMock()

    monkeypatch.setattr(_MODULE, "init_database", init_db)
    monkeypatch.setattr(_MODULE, "close_database", close_db)
    monkeypatch.setattr(_MODULE, "get_db_pool", lambda: pool)
    monkeypatch.setattr(_MODULE, "_run_intelligence", run_intelligence)
    monkeypatch.setattr(_MODULE, "_run_reports", run_reports)
    monkeypatch.setattr(
        _MODULE,
        "settings",
        SimpleNamespace(
            b2b_churn=SimpleNamespace(
                enabled=True,
                intelligence_llm_backend="anthropic",
            )
        ),
    )

    await _MODULE._main(Namespace(vendors="Shopify, Salesforce", report="vendor_scorecard", skip_intelligence=True))

    init_db.assert_awaited_once()
    run_intelligence.assert_not_called()
    run_reports.assert_awaited_once_with(["Shopify", "Salesforce"], "vendor_scorecard")
    pool.fetch.assert_awaited_once()
    close_db.assert_awaited_once()
    output = capsys.readouterr().out
    assert "[1/2] Skipping intelligence refresh; reusing current signal state" in output
    assert "No signals found for ['Shopify', 'Salesforce']. Intelligence may not have run yet." in output
