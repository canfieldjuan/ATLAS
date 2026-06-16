from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_macro_sandbox_lifecycle.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_faq_macro_sandbox_lifecycle",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
lifecycle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lifecycle
SPEC.loader.exec_module(lifecycle)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"confirm_create_draft": False}, "missing_confirm_create_draft"),
        ({"confirm_live_zendesk_write": False}, "missing_confirm_live_zendesk_write"),
        ({"confirm_live_zendesk_delete": False}, "missing_confirm_live_zendesk_delete"),
        ({"expected_zendesk_base_url": ""}, "missing_expected_zendesk_base_url"),
    ],
)
async def test_lifecycle_skips_before_opening_pool_for_missing_live_guards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, Any],
    reason: str,
) -> None:
    async def fail_create_pool(database_url: str) -> object:
        raise AssertionError(f"pool should not open for {database_url}")

    output = tmp_path / f"{reason}.json"
    monkeypatch.setattr(lifecycle, "_create_pool", fail_create_pool)

    args = _args(**overrides)
    code = await lifecycle._main(_argv(args, output=output))

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == lifecycle.SKIPPED_EXIT
    assert payload["ok"] is False
    assert payload["skipped"] is True
    assert payload["stage"] == "preflight"
    assert payload["not_run_reason"] == reason


@pytest.mark.asyncio
async def test_lifecycle_runs_write_then_cleanup_with_seeded_faq_id() -> None:
    calls: list[dict[str, Any]] = []
    pool = object()

    async def write_stage(
        args: argparse.Namespace,
        stage_pool: object,
    ) -> tuple[int, dict[str, Any]]:
        calls.append({"stage": "write", "pool": stage_pool, "args": args})
        return 0, {
            "ok": True,
            "skipped": False,
            "faq_id": "faq-seed-123",
            "zendesk_base_url": "https://sandbox.zendesk.com",
        }

    async def cleanup_stage(
        args: argparse.Namespace,
        stage_pool: object,
    ) -> tuple[int, dict[str, Any]]:
        calls.append({"stage": "cleanup", "pool": stage_pool, "args": args})
        return 0, {
            "ok": True,
            "skipped": False,
            "faq_id": args.faq_id,
            "zendesk_base_url": "https://sandbox.zendesk.com",
            "deleted_faq_count": 1,
            "errors": [],
        }

    code, payload = await lifecycle.run_sandbox_lifecycle_smoke(
        _args(user_id="operator-1"),
        pool,
        write_stage=write_stage,
        cleanup_stage=cleanup_stage,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["skipped"] is False
    assert payload["cleanup_skipped"] is False
    assert payload["stage"] == "complete"
    assert payload["faq_id"] == "faq-seed-123"
    assert payload["zendesk_base_url"] == "https://sandbox.zendesk.com"
    assert [call["stage"] for call in calls] == ["write", "cleanup"]
    assert calls[0]["pool"] is pool
    assert calls[1]["pool"] is pool
    cleanup_args = calls[1]["args"]
    assert cleanup_args.database_url == "postgres://example"
    assert cleanup_args.account_id == "acct-1"
    assert cleanup_args.faq_id == "faq-seed-123"
    assert cleanup_args.expected_zendesk_base_url == "https://sandbox.zendesk.com"
    assert cleanup_args.execute is True
    assert cleanup_args.confirm_live_zendesk_delete is True


@pytest.mark.asyncio
async def test_lifecycle_write_failure_stops_before_cleanup() -> None:
    cleanup_calls = 0

    async def write_stage(
        args: argparse.Namespace,
        pool: object,
    ) -> tuple[int, dict[str, Any]]:
        return 1, {
            "ok": False,
            "skipped": False,
            "error": "macro_publish_failed",
        }

    async def cleanup_stage(
        args: argparse.Namespace,
        pool: object,
    ) -> tuple[int, dict[str, Any]]:
        nonlocal cleanup_calls
        cleanup_calls += 1
        return 0, {"ok": True}

    code, payload = await lifecycle.run_sandbox_lifecycle_smoke(
        _args(),
        object(),
        write_stage=write_stage,
        cleanup_stage=cleanup_stage,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["stage"] == "write"
    assert payload["errors"] == ["macro_publish_failed"]
    assert payload["cleanup"] is None
    assert cleanup_calls == 0


@pytest.mark.asyncio
async def test_lifecycle_write_missing_faq_id_stops_before_cleanup() -> None:
    cleanup_calls = 0

    async def write_stage(
        args: argparse.Namespace,
        pool: object,
    ) -> tuple[int, dict[str, Any]]:
        return 0, {"ok": True, "faq_id": " "}

    async def cleanup_stage(
        args: argparse.Namespace,
        pool: object,
    ) -> tuple[int, dict[str, Any]]:
        nonlocal cleanup_calls
        cleanup_calls += 1
        return 0, {"ok": True}

    code, payload = await lifecycle.run_sandbox_lifecycle_smoke(
        _args(),
        object(),
        write_stage=write_stage,
        cleanup_stage=cleanup_stage,
    )

    assert code == 1
    assert payload["stage"] == "write"
    assert payload["errors"] == ["write_faq_id_missing"]
    assert cleanup_calls == 0


@pytest.mark.asyncio
async def test_lifecycle_cleanup_failure_surfaces_cleanup_payload() -> None:
    async def write_stage(
        args: argparse.Namespace,
        pool: object,
    ) -> tuple[int, dict[str, Any]]:
        return 0, {
            "ok": True,
            "faq_id": "faq-seed-123",
            "zendesk_base_url": "https://sandbox.zendesk.com",
        }

    async def cleanup_stage(
        args: argparse.Namespace,
        pool: object,
    ) -> tuple[int, dict[str, Any]]:
        return 1, {
            "ok": False,
            "skipped": False,
            "faq_id": args.faq_id,
            "zendesk_base_url": "https://sandbox.zendesk.com",
            "errors": ["zendesk_macro_delete_failed"],
        }

    code, payload = await lifecycle.run_sandbox_lifecycle_smoke(
        _args(),
        object(),
        write_stage=write_stage,
        cleanup_stage=cleanup_stage,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["stage"] == "cleanup"
    assert payload["faq_id"] == "faq-seed-123"
    assert payload["errors"] == ["zendesk_macro_delete_failed"]
    assert payload["cleanup"]["errors"] == ["zendesk_macro_delete_failed"]


@pytest.mark.asyncio
async def test_lifecycle_cleanup_skip_after_write_is_not_top_level_skipped() -> None:
    async def write_stage(
        args: argparse.Namespace,
        pool: object,
    ) -> tuple[int, dict[str, Any]]:
        return 0, {"ok": True, "faq_id": "faq-seed-123"}

    async def cleanup_stage(
        args: argparse.Namespace,
        pool: object,
    ) -> tuple[int, dict[str, Any]]:
        return lifecycle.SKIPPED_EXIT, {
            "ok": False,
            "skipped": True,
            "not_run_reason": "zendesk_credentials_missing",
            "faq_id": args.faq_id,
        }

    code, payload = await lifecycle.run_sandbox_lifecycle_smoke(
        _args(),
        object(),
        write_stage=write_stage,
        cleanup_stage=cleanup_stage,
    )

    assert code == lifecycle.SKIPPED_EXIT
    assert payload["ok"] is False
    assert payload["skipped"] is False
    assert payload["cleanup_skipped"] is True
    assert payload["stage"] == "cleanup"
    assert payload["faq_id"] == "faq-seed-123"
    assert payload["cleanup"]["not_run_reason"] == "zendesk_credentials_missing"


@pytest.mark.asyncio
async def test_lifecycle_main_closes_pool_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pool:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    pool = Pool()

    async def create_pool(database_url: str) -> Pool:
        return pool

    async def run(
        args: argparse.Namespace,
        stage_pool: Pool,
    ) -> tuple[int, dict[str, Any]]:
        assert stage_pool is pool
        return 0, {"ok": True, "skipped": False}

    monkeypatch.setattr(lifecycle, "_create_pool", create_pool)
    monkeypatch.setattr(lifecycle, "run_sandbox_lifecycle_smoke", run)

    code = await lifecycle._main(_argv(_args()))

    assert code == 0
    assert pool.closed is True


def test_lifecycle_wrapper_does_not_define_zendesk_transport_or_provider() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "ZendeskMacroPublishProvider" not in source
    assert "ZendeskHTTPMacroTransport" not in source
    assert "httpx" not in source


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "database_url": "postgres://example",
        "account_id": "acct-1",
        "user_id": "",
        "expected_zendesk_base_url": "https://sandbox.zendesk.com",
        "target_id": "macro-writeback-live-smoke-seed",
        "target_mode": "support_ticket_faq",
        "title": "Zendesk macro writeback sandbox lifecycle smoke seed",
        "question": "How do I refund a duplicate charge?",
        "resolution_text": (
            "Open Billing, select the duplicate charge, and click Refund payment. "
            "Confirm the refund and tell the customer it may take 3-5 business days."
        ),
        "confirm_create_draft": True,
        "confirm_live_zendesk_write": True,
        "confirm_live_zendesk_delete": True,
        "output": None,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _argv(args: argparse.Namespace, *, output: Path | None = None) -> list[str]:
    argv = [
        "--database-url",
        args.database_url,
        "--account-id",
        args.account_id,
        "--expected-zendesk-base-url",
        args.expected_zendesk_base_url,
        "--question",
        args.question,
        "--resolution-text",
        args.resolution_text,
    ]
    if args.user_id:
        argv.extend(["--user-id", args.user_id])
    if args.confirm_create_draft:
        argv.append("--confirm-create-draft")
    if args.confirm_live_zendesk_write:
        argv.append("--confirm-live-zendesk-write")
    if args.confirm_live_zendesk_delete:
        argv.append("--confirm-live-zendesk-delete")
    if output is not None:
        argv.extend(["--output", str(output)])
    return argv
