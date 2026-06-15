from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_macro_sandbox_e2e.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_faq_macro_sandbox_e2e",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
e2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e2e)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"confirm_create_draft": False}, "missing_confirm_create_draft"),
        ({"confirm_live_zendesk_write": False}, "missing_confirm_live_zendesk_write"),
        ({"expected_zendesk_base_url": ""}, "missing_expected_zendesk_base_url"),
    ],
)
async def test_e2e_skips_before_opening_pool_for_missing_live_guards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, Any],
    reason: str,
) -> None:
    async def fail_create_pool(database_url: str) -> object:
        raise AssertionError(f"pool should not open for {database_url}")

    output = tmp_path / f"{reason}.json"
    monkeypatch.setattr(e2e, "_create_pool", fail_create_pool)

    args = _args(**overrides)
    code = await e2e._main(_argv(args, output=output))

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == e2e.SKIPPED_EXIT
    assert payload["ok"] is False
    assert payload["skipped"] is True
    assert payload["stage"] == "preflight"
    assert payload["not_run_reason"] == reason


@pytest.mark.asyncio
async def test_e2e_runs_seed_then_live_smoke_with_seeded_faq_id() -> None:
    calls: list[dict[str, Any]] = []
    pool = object()

    async def seed_stage(args: argparse.Namespace, stage_pool: object) -> tuple[int, dict[str, Any]]:
        calls.append({"stage": "seed", "pool": stage_pool, "args": args})
        return 0, {
            "ok": True,
            "faq_id": "faq-seed-123",
            "publishable_count": 1,
        }

    async def live_stage(args: argparse.Namespace, stage_pool: object) -> tuple[int, dict[str, Any]]:
        calls.append({"stage": "live", "pool": stage_pool, "args": args})
        return 0, {
            "ok": True,
            "skipped": False,
            "faq_id": args.faq_id,
            "zendesk_base_url": "https://sandbox.zendesk.com",
            "summary": {"published_count": 1},
            "errors": [],
        }

    code, payload = await e2e.run_sandbox_e2e_smoke(
        _args(user_id="operator-1"),
        pool,
        seed_stage=seed_stage,
        live_stage=live_stage,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["skipped"] is False
    assert payload["live_smoke_skipped"] is False
    assert payload["stage"] == "complete"
    assert payload["faq_id"] == "faq-seed-123"
    assert payload["zendesk_base_url"] == "https://sandbox.zendesk.com"
    assert [call["stage"] for call in calls] == ["seed", "live"]
    assert calls[0]["pool"] is pool
    assert calls[1]["pool"] is pool
    live_args = calls[1]["args"]
    assert live_args.faq_id == "faq-seed-123"
    assert live_args.account_id == "acct-1"
    assert live_args.user_id == "operator-1"
    assert live_args.expected_zendesk_base_url == "https://sandbox.zendesk.com"
    assert live_args.confirm_live_zendesk_write is True


@pytest.mark.asyncio
async def test_e2e_seed_failure_stops_before_live_smoke() -> None:
    live_calls = 0

    async def seed_stage(args: argparse.Namespace, pool: object) -> tuple[int, dict[str, Any]]:
        return 1, {
            "ok": False,
            "skipped": False,
            "error": "seed_faq_draft_save_failed",
        }

    async def live_stage(args: argparse.Namespace, pool: object) -> tuple[int, dict[str, Any]]:
        nonlocal live_calls
        live_calls += 1
        return 0, {"ok": True}

    code, payload = await e2e.run_sandbox_e2e_smoke(
        _args(),
        object(),
        seed_stage=seed_stage,
        live_stage=live_stage,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["stage"] == "seed"
    assert payload["errors"] == ["seed_faq_draft_save_failed"]
    assert payload["live_smoke"] is None
    assert live_calls == 0


@pytest.mark.asyncio
async def test_e2e_seed_missing_faq_id_stops_before_live_smoke() -> None:
    live_calls = 0

    async def seed_stage(args: argparse.Namespace, pool: object) -> tuple[int, dict[str, Any]]:
        return 0, {"ok": True, "faq_id": " "}

    async def live_stage(args: argparse.Namespace, pool: object) -> tuple[int, dict[str, Any]]:
        nonlocal live_calls
        live_calls += 1
        return 0, {"ok": True}

    code, payload = await e2e.run_sandbox_e2e_smoke(
        _args(),
        object(),
        seed_stage=seed_stage,
        live_stage=live_stage,
    )

    assert code == 1
    assert payload["stage"] == "seed"
    assert payload["errors"] == ["seed_faq_id_missing"]
    assert live_calls == 0


@pytest.mark.asyncio
async def test_e2e_live_failure_surfaces_live_payload() -> None:
    async def seed_stage(args: argparse.Namespace, pool: object) -> tuple[int, dict[str, Any]]:
        return 0, {"ok": True, "faq_id": "faq-seed-123"}

    async def live_stage(args: argparse.Namespace, pool: object) -> tuple[int, dict[str, Any]]:
        return 1, {
            "ok": False,
            "skipped": False,
            "faq_id": args.faq_id,
            "zendesk_base_url": "https://sandbox.zendesk.com",
            "errors": ["macro_publish_failed"],
        }

    code, payload = await e2e.run_sandbox_e2e_smoke(
        _args(),
        object(),
        seed_stage=seed_stage,
        live_stage=live_stage,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["stage"] == "live_smoke"
    assert payload["faq_id"] == "faq-seed-123"
    assert payload["errors"] == ["macro_publish_failed"]
    assert payload["live_smoke"]["errors"] == ["macro_publish_failed"]


@pytest.mark.asyncio
async def test_e2e_live_skip_after_seed_is_not_top_level_skipped() -> None:
    async def seed_stage(args: argparse.Namespace, pool: object) -> tuple[int, dict[str, Any]]:
        return 0, {"ok": True, "faq_id": "faq-seed-123"}

    async def live_stage(args: argparse.Namespace, pool: object) -> tuple[int, dict[str, Any]]:
        return e2e.SKIPPED_EXIT, {
            "ok": False,
            "skipped": True,
            "not_run_reason": "zendesk_credentials_missing",
            "faq_id": args.faq_id,
        }

    code, payload = await e2e.run_sandbox_e2e_smoke(
        _args(),
        object(),
        seed_stage=seed_stage,
        live_stage=live_stage,
    )

    assert code == e2e.SKIPPED_EXIT
    assert payload["ok"] is False
    assert payload["skipped"] is False
    assert payload["live_smoke_skipped"] is True
    assert payload["stage"] == "live_smoke"
    assert payload["faq_id"] == "faq-seed-123"
    assert payload["live_smoke"]["not_run_reason"] == "zendesk_credentials_missing"


@pytest.mark.asyncio
async def test_e2e_main_closes_pool_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class Pool:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    pool = Pool()

    async def create_pool(database_url: str) -> Pool:
        return pool

    async def run(args: argparse.Namespace, stage_pool: Pool) -> tuple[int, dict[str, Any]]:
        assert stage_pool is pool
        return 0, {"ok": True, "skipped": False}

    monkeypatch.setattr(e2e, "_create_pool", create_pool)
    monkeypatch.setattr(e2e, "run_sandbox_e2e_smoke", run)

    code = await e2e._main(_argv(_args()))

    assert code == 0
    assert pool.closed is True


def test_e2e_wrapper_does_not_define_zendesk_transport_or_provider() -> None:
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
        "title": "Zendesk macro writeback sandbox E2E smoke seed",
        "question": "How do I refund a duplicate charge?",
        "resolution_text": (
            "Open Billing, select the duplicate charge, and click Refund payment. "
            "Confirm the refund and tell the customer it may take 3-5 business days."
        ),
        "confirm_create_draft": True,
        "confirm_live_zendesk_write": True,
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
    if output is not None:
        argv.extend(["--output", str(output)])
    return argv
