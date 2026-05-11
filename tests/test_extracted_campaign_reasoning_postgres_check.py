from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

from extracted_content_pipeline.campaign_ports import CampaignReasoningContext


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_extracted_campaign_reasoning_postgres.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_extracted_campaign_reasoning_postgres",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Repository:
    def __init__(self, context: CampaignReasoningContext | None) -> None:
        self.context = context
        self.calls: list[dict[str, Any]] = []

    async def read_campaign_reasoning_context(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self.context


def test_opportunity_from_args_drops_blank_values() -> None:
    module = _load_script_module()
    args = argparse.Namespace(
        company_name=" Acme ",
        company="",
        contact_email=" buyer@example.com ",
        email=None,
        vendor_name="HubSpot",
        vendor=" ",
    )

    assert module._opportunity_from_args(args) == {
        "company_name": "Acme",
        "contact_email": "buyer@example.com",
        "vendor_name": "HubSpot",
    }


@pytest.mark.asyncio
async def test_check_reasoning_context_returns_ok_payload() -> None:
    module = _load_script_module()
    repo = _Repository(
        CampaignReasoningContext(
            top_theses=({"claim": "Renewal pricing"},),
            proof_points=({"label": "source_material", "value": "pricing"},),
        )
    )

    result = await module._check_reasoning_context(
        repo,
        account_id="acct-1",
        target_id="opp-1",
        target_mode="vendor_retention",
        opportunity={"company_name": "Acme"},
    )

    assert result["status"] == "ok"
    assert result["context"]["top_theses"][0]["claim"] == "Renewal pricing"
    assert repo.calls[0]["scope"].account_id == "acct-1"
    assert repo.calls[0]["target_id"] == "opp-1"
    assert repo.calls[0]["opportunity"] == {"company_name": "Acme"}


@pytest.mark.asyncio
async def test_check_reasoning_context_returns_missing_payload() -> None:
    module = _load_script_module()

    result = await module._check_reasoning_context(
        _Repository(None),
        account_id="acct-1",
        target_id="opp-404",
        target_mode="vendor_retention",
        opportunity={},
    )

    assert result == {
        "status": "missing",
        "target_id": "opp-404",
        "target_mode": "vendor_retention",
    }


@pytest.mark.asyncio
async def test_cli_boundary_returns_one_when_no_context_matches(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()

    class _Pool:
        closed = False

        async def close(self) -> None:
            self.closed = True

    pool = _Pool()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_extracted_campaign_reasoning_postgres.py",
            "--database-url",
            "postgres://example",
            "--account-id",
            "acct-1",
            "--target-id",
            "opp-404",
        ],
    )
    async def _pool_factory(_dsn: str) -> _Pool:
        return pool

    monkeypatch.setattr(module, "_create_pool", _pool_factory)
    monkeypatch.setattr(
        module,
        "PostgresCampaignReasoningContextRepository",
        lambda **_kwargs: _Repository(None),
    )

    assert await module._main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing reasoning context for target_id=opp-404" in captured.err
    assert pool.closed is True
