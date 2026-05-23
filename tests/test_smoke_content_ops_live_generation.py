from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_execution import (
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_live_generation.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_live_generation",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class _Lifecycle:
    def __init__(self, *, fail_init: bool = False) -> None:
        self.initialized = False
        self.closed = False
        self.fail_init = fail_init

    async def init(self) -> None:
        self.initialized = True
        if self.fail_init:
            raise RuntimeError("database unavailable")

    async def close(self) -> None:
        self.closed = True


class _LandingPageService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {
            "requested": 1,
            "generated": 1,
            "skipped": 0,
            "saved_ids": ["lp-live-smoke-1"],
            "quality_repair_history": [
                {
                    "attempt": 0,
                    "passed": True,
                    "blockers": [],
                    "repair_issues": [],
                }
            ],
        }


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "account_id": "acct-live-smoke",
        "user_id": "user-live-smoke",
        "env_file": [],
        "input_json": None,
        "input": [],
        "quality_repair_attempts": 1,
        "no_quality_gates": False,
        "output_result": None,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
async def test_live_generation_smoke_executes_landing_page_through_real_executor() -> None:
    lifecycle = _Lifecycle()
    service = _LandingPageService()

    code, result = await smoke.run_content_ops_live_generation_smoke(
        _args(input=["cta_url=/custom-intake", 'faq_questions=["How long does it take?"]']),
        init_database_fn=lifecycle.init,
        close_database_fn=lifecycle.close,
        services_factory=lambda: ContentOpsExecutionServices(landing_page=service),
        executor=execute_content_ops_from_mapping,
        tenant_scope_cls=TenantScope,
    )

    assert code == 0
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["configured_outputs"] == ["landing_page"]
    assert result["execution"]["status"] == "completed"
    assert result["execution"]["steps"][0]["result"]["saved_ids"] == [
        "lp-live-smoke-1"
    ]
    assert lifecycle.initialized is True
    assert lifecycle.closed is True
    assert len(service.calls) == 1

    call = service.calls[0]
    assert call["scope"].account_id == "acct-live-smoke"
    assert call["scope"].user_id == "user-live-smoke"
    assert call["campaign"].name == "FAQ Report"
    assert call["campaign"].context["cta_url"] == "/custom-intake"
    assert call["campaign"].context["faq_questions"] == ["How long does it take?"]
    assert call["quality_repair_attempts"] == 1
    assert call["quality_gates_enabled"] is True


@pytest.mark.asyncio
async def test_live_generation_smoke_fails_before_execute_when_landing_page_unwired() -> None:
    lifecycle = _Lifecycle()
    executor_called = False

    async def _executor(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal executor_called
        executor_called = True
        return {}

    code, result = await smoke.run_content_ops_live_generation_smoke(
        _args(),
        init_database_fn=lifecycle.init,
        close_database_fn=lifecycle.close,
        services_factory=lambda: ContentOpsExecutionServices(),
        executor=_executor,
        tenant_scope_cls=TenantScope,
    )

    assert code == 1
    assert result["ok"] is False
    assert result["configured_outputs"] == []
    assert result["execution"] is None
    assert result["errors"] == [
        "landing_page service is not configured; check Atlas DB initialization "
        "and pipeline LLM/OpenRouter credentials"
    ]
    assert executor_called is False
    assert lifecycle.initialized is True
    assert lifecycle.closed is True


@pytest.mark.asyncio
async def test_live_generation_smoke_reports_failed_quality_gate() -> None:
    lifecycle = _Lifecycle()

    async def _executor(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "steps": [
                {
                    "output": "landing_page",
                    "status": "completed",
                    "result": {
                        "saved_ids": ["lp-live-smoke-1"],
                        "quality_repair_history": [{"passed": False}],
                    },
                }
            ],
        }

    code, result = await smoke.run_content_ops_live_generation_smoke(
        _args(),
        init_database_fn=lifecycle.init,
        close_database_fn=lifecycle.close,
        services_factory=lambda: ContentOpsExecutionServices(
            landing_page=_LandingPageService()
        ),
        executor=_executor,
        tenant_scope_cls=TenantScope,
    )

    assert code == 1
    assert result["ok"] is False
    assert result["errors"] == ["landing_page quality gate did not pass"]


@pytest.mark.asyncio
async def test_live_generation_smoke_reports_database_initialization_failure() -> None:
    lifecycle = _Lifecycle(fail_init=True)

    code, result = await smoke.run_content_ops_live_generation_smoke(
        _args(),
        init_database_fn=lifecycle.init,
        close_database_fn=lifecycle.close,
        services_factory=lambda: ContentOpsExecutionServices(
            landing_page=_LandingPageService()
        ),
        executor=execute_content_ops_from_mapping,
        tenant_scope_cls=TenantScope,
    )

    assert code == 1
    assert result["ok"] is False
    assert result["execution"] is None
    assert result["errors"] == ["RuntimeError: database unavailable"]
    assert lifecycle.initialized is True
    assert lifecycle.closed is True
