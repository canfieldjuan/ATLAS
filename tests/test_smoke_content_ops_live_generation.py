from __future__ import annotations

import argparse
import importlib.util
import json
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


class _BlogPostService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {
            "requested": 1,
            "generated": 1,
            "skipped": 0,
            "saved_ids": ["blog-live-smoke-1"],
            "errors": [],
        }


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "output": "landing_page",
        "account_id": "acct-live-smoke",
        "user_id": "user-live-smoke",
        "target_mode": "vendor_retention",
        "env_file": [],
        "input_json": None,
        "support_ticket_csv": None,
        "blog_blueprint_json": None,
        "input": [],
        "quality_repair_attempts": 1,
        "no_quality_gates": False,
        "output_result": None,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _support_ticket_csv(tmp_path: Path) -> Path:
    path = tmp_path / "support_tickets.csv"
    path.write_text(
        "\n".join([
            "Ticket ID,Subject,Description,Pain Category",
            (
                "ticket-login-1,Login email change,"
                "How do I change my login email?,account"
            ),
            (
                "ticket-export-1,Campaign export,"
                "How do we export campaign attribution data before renewal?,reporting"
            ),
        ]),
        encoding="utf-8",
    )
    return path


def _non_ticket_csv(tmp_path: Path) -> Path:
    path = tmp_path / "accounts.csv"
    path.write_text(
        "\n".join([
            "Company,Website,Notes",
            "Acme,https://example.com,Prospect account",
        ]),
        encoding="utf-8",
    )
    return path


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
async def test_live_generation_smoke_packages_support_ticket_csv_for_landing_page(
    tmp_path: Path,
) -> None:
    lifecycle = _Lifecycle()
    service = _LandingPageService()

    code, result = await smoke.run_content_ops_live_generation_smoke(
        _args(support_ticket_csv=_support_ticket_csv(tmp_path)),
        init_database_fn=lifecycle.init,
        close_database_fn=lifecycle.close,
        services_factory=lambda: ContentOpsExecutionServices(landing_page=service),
        executor=execute_content_ops_from_mapping,
        tenant_scope_cls=TenantScope,
    )

    assert code == 0
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["payload"]["inputs"]["target_keyword"] == "support ticket FAQ report"
    assert result["payload"]["inputs"]["source_period"] == (
        "Last 90 days of support tickets"
    )
    assert result["payload"]["inputs"]["faq_questions"] == [
        "How do I change my login email?",
        "How do we export campaign attribution data before renewal?",
    ]
    assert len(service.calls) == 1

    call = service.calls[0]
    assert call["campaign"].name == "FAQ Report"
    assert call["campaign"].persona == (
        "Small teams answering repeat support questions"
    )
    assert call["campaign"].context["cta_label"] == (
        "Upload Ticket CSV -- Free Analysis"
    )
    assert call["campaign"].context["faq_questions"] == [
        "How do I change my login email?",
        "How do we export campaign attribution data before renewal?",
    ]


@pytest.mark.asyncio
async def test_live_generation_smoke_rejects_non_ticket_csv(tmp_path: Path) -> None:
    lifecycle = _Lifecycle()
    service = _LandingPageService()

    code, result = await smoke.run_content_ops_live_generation_smoke(
        _args(support_ticket_csv=_non_ticket_csv(tmp_path)),
        init_database_fn=lifecycle.init,
        close_database_fn=lifecycle.close,
        services_factory=lambda: ContentOpsExecutionServices(landing_page=service),
        executor=execute_content_ops_from_mapping,
        tenant_scope_cls=TenantScope,
    )

    assert code == 1
    assert result["ok"] is False
    assert result["execution"] is None
    assert result["errors"] == [
        "ValueError: --support-ticket-csv did not contain "
        "support-ticket-shaped rows; include ticket id, subject, and "
        "description/message fields."
    ]
    assert service.calls == []
    assert lifecycle.initialized is True
    assert lifecycle.closed is True


@pytest.mark.asyncio
async def test_live_generation_smoke_seeds_and_executes_blog_post_through_real_executor() -> None:
    lifecycle = _Lifecycle()
    service = _BlogPostService()
    seeded: list[dict[str, Any]] = []

    async def _seed_blog_blueprint(args: argparse.Namespace, scope: TenantScope) -> dict[str, Any]:
        seeded.append({"args": args, "scope": scope})
        return {
            "saved_ids": ["bp-live-smoke-1"],
            "slug": "content-ops-blog-live-smoke-acct-live-smoke",
            "target_mode": args.target_mode,
            "topic_type": "complaint_roundup",
        }

    code, result = await smoke.run_content_ops_live_generation_smoke(
        _args(
            output="blog_post",
            input=["topic=Support ticket FAQ gaps"],
        ),
        init_database_fn=lifecycle.init,
        close_database_fn=lifecycle.close,
        services_factory=lambda: ContentOpsExecutionServices(blog_post=service),
        executor=execute_content_ops_from_mapping,
        tenant_scope_cls=TenantScope,
        blog_blueprint_seed_fn=_seed_blog_blueprint,
    )

    assert code == 0
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["configured_outputs"] == ["blog_post"]
    assert result["seeded_blog_blueprint"]["saved_ids"] == ["bp-live-smoke-1"]
    assert result["execution"]["status"] == "completed"
    assert result["execution"]["steps"][0]["result"]["saved_ids"] == [
        "blog-live-smoke-1"
    ]
    assert lifecycle.initialized is True
    assert lifecycle.closed is True
    assert len(seeded) == 1
    assert seeded[0]["scope"].account_id == "acct-live-smoke"
    assert len(service.calls) == 1

    call = service.calls[0]
    assert call["scope"].account_id == "acct-live-smoke"
    assert call["target_mode"] == "vendor_retention"
    assert call["limit"] == 1
    assert call["filters"] == {
        "topic_type": "complaint_roundup",
        "slug": "content-ops-blog-live-smoke-acct-live-smoke",
    }
    assert call["topic"] == "Support ticket FAQ gaps"
    assert call["quality_gates_enabled"] is True


@pytest.mark.asyncio
async def test_live_generation_smoke_packages_support_ticket_csv_for_blog_post(
    tmp_path: Path,
) -> None:
    lifecycle = _Lifecycle()
    service = _BlogPostService()
    seeded: list[dict[str, Any]] = []

    async def _seed_blog_blueprint(args: argparse.Namespace, scope: TenantScope) -> dict[str, Any]:
        seeded.append({"args": args, "scope": scope})
        return {
            "saved_ids": ["bp-support-ticket-smoke-1"],
            "slug": "content-ops-support-ticket-live-smoke-acct-live-smoke",
            "target_mode": args.target_mode,
            "topic_type": "content_ops_support_ticket_faq",
            "topic": "Support-ticket questions customers keep asking",
        }

    code, result = await smoke.run_content_ops_live_generation_smoke(
        _args(
            output="blog_post",
            support_ticket_csv=_support_ticket_csv(tmp_path),
        ),
        init_database_fn=lifecycle.init,
        close_database_fn=lifecycle.close,
        services_factory=lambda: ContentOpsExecutionServices(blog_post=service),
        executor=execute_content_ops_from_mapping,
        tenant_scope_cls=TenantScope,
        blog_blueprint_seed_fn=_seed_blog_blueprint,
    )

    assert code == 0
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["payload"]["inputs"]["filters"] == {
        "topic_type": "content_ops_support_ticket_faq",
        "slug": "content-ops-support-ticket-live-smoke-acct-live-smoke",
    }
    assert len(seeded) == 1
    assert len(service.calls) == 1

    call = service.calls[0]
    assert call["filters"] == {
        "topic_type": "content_ops_support_ticket_faq",
        "slug": "content-ops-support-ticket-live-smoke-acct-live-smoke",
    }
    assert call["topic"] == "Support-ticket questions customers keep asking"


def test_blog_blueprint_json_loader_accepts_one_custom_blueprint(tmp_path: Path) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps({
            "title": "FAQ gaps for onboarding tickets",
            "sections": [{"id": "onboarding", "heading": "Onboarding gaps"}],
            "data_context": {"audience": "small support team"},
        }),
        encoding="utf-8",
    )

    blueprint, warnings = smoke._load_single_blog_blueprint_from_file(
        path,
        target_mode="vendor_retention",
    )

    assert warnings == []
    assert blueprint.target_mode == "vendor_retention"
    assert blueprint.topic_type == "content_ops_live_smoke"
    assert blueprint.slug == "faq-gaps-for-onboarding-tickets"
    assert blueprint.suggested_title == "FAQ gaps for onboarding tickets"
    assert blueprint.payload["sections"] == [
        {"id": "onboarding", "heading": "Onboarding gaps"}
    ]


def test_blog_blueprint_json_loader_rejects_multiple_blueprints(tmp_path: Path) -> None:
    path = tmp_path / "blueprints.json"
    path.write_text(
        json.dumps({
            "blueprints": [
                {"title": "First blueprint"},
                {"title": "Second blueprint"},
            ]
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must load exactly one blueprint"):
        smoke._load_single_blog_blueprint_from_file(
            path,
            target_mode="vendor_retention",
        )


def test_blog_payload_alignment_uses_seeded_topic_type_and_title() -> None:
    payload = smoke._payload_from_args(_args(output="blog_post"))

    smoke._align_blog_payload_to_seed(
        payload,
        {
            "topic_type": "custom_smoke_topic",
            "slug": "faq-gaps-for-onboarding-tickets",
            "topic": "FAQ gaps for onboarding tickets",
        },
    )

    assert payload["inputs"]["filters"] == {
        "topic_type": "custom_smoke_topic",
        "slug": "faq-gaps-for-onboarding-tickets",
    }
    assert payload["inputs"]["topic"] == "FAQ gaps for onboarding tickets"


def test_blog_payload_alignment_replaces_support_ticket_provider_default_topic() -> None:
    payload = {
        "outputs": ["blog_post"],
        "inputs": {
            "topic": smoke.SUPPORT_TICKET_BLOG_TOPIC,
            "filters": {"topic_type": smoke.SUPPORT_TICKET_BLOG_TOPIC_TYPE},
        },
    }

    smoke._align_blog_payload_to_seed(
        payload,
        {
            "topic_type": "custom_support_ticket_topic",
            "slug": "custom-support-ticket-blueprint",
            "topic": "Custom support-ticket article",
        },
    )

    assert payload["inputs"]["filters"] == {
        "topic_type": "custom_support_ticket_topic",
        "slug": "custom-support-ticket-blueprint",
    }
    assert payload["inputs"]["topic"] == "Custom support-ticket article"


def test_blog_payload_alignment_preserves_custom_operator_topic() -> None:
    payload = {
        "outputs": ["blog_post"],
        "inputs": {
            "topic": "Operator supplied smoke topic",
            "filters": {"topic_type": smoke.SUPPORT_TICKET_BLOG_TOPIC_TYPE},
        },
    }

    smoke._align_blog_payload_to_seed(
        payload,
        {
            "topic_type": "custom_support_ticket_topic",
            "slug": "custom-support-ticket-blueprint",
            "topic": "Custom support-ticket article",
        },
    )

    assert payload["inputs"]["topic"] == "Operator supplied smoke topic"


def test_support_ticket_blog_blueprint_payload_uses_csv_counts(tmp_path: Path) -> None:
    rows = smoke._load_csv_rows(_support_ticket_csv(tmp_path))

    payload = smoke._support_ticket_blog_blueprint_payload(rows)

    serialized = json.dumps(payload, sort_keys=True)
    assert "186" not in serialized
    assert "78" not in serialized
    assert "42%" not in serialized
    assert payload["data_context"]["source_row_count"] == 2
    assert payload["data_context"]["question_like_ticket_count"] == 2
    assert payload["data_context"]["top_clusters"] == [
        {"label": "account", "count": 1},
        {"label": "reporting", "count": 1},
    ]
    first_section = payload["sections"][0]
    assert first_section["key_stats"] == {
        "support_ticket_rows": 2,
        "question_like_rows": 2,
        "cluster_count": 2,
    }
    assert "2 support-ticket rows" in first_section["data_summary"]


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
