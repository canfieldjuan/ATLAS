from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Sequence

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/seed_faq_macro_writeback_live_smoke_draft.py"
SPEC = importlib.util.spec_from_file_location(
    "seed_faq_macro_writeback_live_smoke_draft",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
seed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seed)


class _FAQRepo:
    def __init__(
        self,
        *,
        saved_ids: Sequence[str] = ("faq-seed-1",),
        approve_result: bool = True,
    ) -> None:
        self.saved_ids = tuple(saved_ids)
        self.approve_result = approve_result
        self.save_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []

    async def save_drafts(
        self,
        drafts: Sequence[TicketFAQDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        self.save_calls.append({"drafts": tuple(drafts), "scope": scope})
        return self.saved_ids

    async def update_status(
        self,
        faq_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        self.update_calls.append({
            "faq_id": faq_id,
            "status": status,
            "scope": scope,
        })
        return self.approve_result


@pytest.mark.asyncio
async def test_seed_skips_before_opening_pool_when_confirmation_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_create_pool(database_url: str) -> object:
        raise AssertionError(f"pool should not open for {database_url}")

    output = tmp_path / "seed.json"
    monkeypatch.setattr(seed, "_create_pool", fail_create_pool)

    code = await seed._main([
        "--database-url",
        "postgres://example",
        "--account-id",
        "acct-1",
        "--output",
        str(output),
    ])

    assert code == seed.SKIPPED_EXIT
    assert '"not_run_reason": "missing_confirm_create_draft"' in output.read_text()


@pytest.mark.asyncio
async def test_seed_creates_one_approved_publishable_draft() -> None:
    repo = _FAQRepo()

    code, payload = await seed.seed_live_smoke_draft(
        _args(user_id="operator-1"),
        pool=object(),
        faq_repository=repo,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["faq_id"] == "faq-seed-1"
    assert payload["draft_status"] == "approved"
    assert payload["publishable_count"] == 1
    assert payload["macro_titles"] == ["How do I refund a duplicate charge?"]
    assert repo.save_calls[0]["scope"] == TenantScope(
        account_id="acct-1",
        user_id="operator-1",
    )
    saved_draft = repo.save_calls[0]["drafts"][0]
    assert saved_draft.status == ""
    assert saved_draft.metadata["macro_writeback_live_smoke_seed"] is True
    assert saved_draft.metadata["zendesk_write"] is False
    assert saved_draft.items[0]["answer_evidence_status"] == "resolution_evidence"
    assert repo.update_calls == [
        {
            "faq_id": "faq-seed-1",
            "status": "approved",
            "scope": TenantScope(account_id="acct-1", user_id="operator-1"),
        }
    ]


@pytest.mark.asyncio
async def test_seed_outputs_next_command_without_raw_database_url() -> None:
    code, payload = await seed.seed_live_smoke_draft(
        _args(
            database_url="postgres://user:secret@example/db",
            expected_zendesk_base_url="https://sandbox.zendesk.com",
        ),
        pool=object(),
        faq_repository=_FAQRepo(saved_ids=("faq with spaces",)),
    )

    assert code == 0
    command = payload["next_command"]
    assert "postgres://user:secret@example/db" not in command
    assert '--database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}"' in command
    assert "--faq-id 'faq with spaces'" in command
    assert "--expected-zendesk-base-url https://sandbox.zendesk.com" in command
    assert "--confirm-live-zendesk-write" in command


@pytest.mark.asyncio
async def test_seed_reports_save_failure_without_approval_attempt() -> None:
    repo = _FAQRepo(saved_ids=())

    code, payload = await seed.seed_live_smoke_draft(
        _args(),
        pool=object(),
        faq_repository=repo,
    )

    assert code == 1
    assert payload["error"] == "seed_faq_draft_save_failed"
    assert repo.save_calls
    assert repo.update_calls == []


@pytest.mark.asyncio
async def test_seed_reports_approval_failure() -> None:
    repo = _FAQRepo(approve_result=False)

    code, payload = await seed.seed_live_smoke_draft(
        _args(),
        pool=object(),
        faq_repository=repo,
    )

    assert code == 1
    assert payload["faq_id"] == "faq-seed-1"
    assert payload["error"] == "seed_faq_draft_approve_failed"


@pytest.mark.asyncio
async def test_seed_reports_non_publishable_override() -> None:
    code, payload = await seed.seed_live_smoke_draft(
        _args(resolution_text=""),
        pool=object(),
        faq_repository=_FAQRepo(),
    )

    assert code == 1
    assert payload["error"] == "seed_faq_draft_not_publishable"
    assert payload["preview"]["publishable_count"] == 0
    assert payload["preview"]["skipped"][0]["reason"] == "missing_resolution_body"


def test_seed_script_does_not_import_zendesk_transport() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "ZendeskMacroPublishProvider" not in source
    assert "ZendeskHTTPMacroTransport" not in source
    assert "faq_macro_writeback_zendesk" not in source


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "database_url": "postgres://example",
        "account_id": "acct-1",
        "user_id": "",
        "target_id": "macro-writeback-live-smoke-seed",
        "target_mode": "support_ticket_faq",
        "title": "Zendesk macro writeback live smoke seed",
        "question": "How do I refund a duplicate charge?",
        "resolution_text": (
            "Open Billing, select the duplicate charge, and click Refund payment. "
            "Confirm the refund and tell the customer it may take 3-5 business days."
        ),
        "expected_zendesk_base_url": "",
        "confirm_create_draft": True,
        "output": None,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)
