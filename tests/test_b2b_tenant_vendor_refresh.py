from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_refresh_tenant_vendor_pipeline_runs_scoped_chain(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace()
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001", email="ops@example.com")
    stage_mock = AsyncMock(
        side_effect=[
            {
                "task_name": "b2b_churn_intelligence",
                "triggered": True,
                "already_running": False,
                "skipped": False,
                "skip_reason": None,
                "result": {"execution_id": "core"},
            },
            {
                "task_name": "b2b_reasoning_synthesis",
                "triggered": True,
                "already_running": False,
                "skipped": False,
                "skip_reason": None,
                "result": {"execution_id": "reasoning"},
            },
            {
                "task_name": "b2b_battle_cards",
                "triggered": True,
                "already_running": False,
                "skipped": False,
                "skip_reason": None,
                "result": {"execution_id": "battle"},
            },
            {
                "task_name": "b2b_accounts_in_motion",
                "triggered": True,
                "already_running": False,
                "skipped": False,
                "skip_reason": None,
                "result": {"execution_id": "accounts"},
            },
        ]
    )

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod, "_require_b2b_product", lambda _user: None)
    monkeypatch.setattr(mod, "_resolve_tracked_vendor_for_view", AsyncMock(return_value="Zendesk"))
    monkeypatch.setattr(mod, "_run_vendor_refresh_stage", stage_mock)
    monkeypatch.setattr(mod, "_stage_blocks_followups", lambda _stage: False)

    result = await mod.refresh_tenant_vendor_pipeline(
        "  Zendesk  ",
        mod.VendorRefreshRequest(force_reasoning=True),
        user=user,
    )

    assert result["vendor_name"] == "Zendesk"
    assert result["triggered_by"] == "api:ops@example.com"
    assert stage_mock.await_count == 4
    assert stage_mock.await_args_list[0].kwargs["task_name"] == "b2b_churn_intelligence"
    assert stage_mock.await_args_list[0].kwargs["vendor_name"] == "Zendesk"
    assert stage_mock.await_args_list[0].kwargs["scope_trigger"] == "tenant_manual_request"
    assert stage_mock.await_args_list[1].kwargs["extra_metadata"] == {"force": True}


@pytest.mark.asyncio
async def test_refresh_tenant_vendor_pipeline_rejects_untracked_vendor(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace()
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001", email="ops@example.com")

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod, "_require_b2b_product", lambda _user: None)
    monkeypatch.setattr(mod, "_resolve_tracked_vendor_for_view", AsyncMock(return_value=None))

    with pytest.raises(mod.HTTPException) as exc:
        await mod.refresh_tenant_vendor_pipeline(
            "Zendesk",
            mod.VendorRefreshRequest(),
            user=user,
        )

    assert exc.value.status_code == 403
    assert exc.value.detail == "Vendor must be in your tracked vendor list"
