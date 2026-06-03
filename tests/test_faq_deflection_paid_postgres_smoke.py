from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_deflection_paid_postgres.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_deflection_paid_postgres",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class _Pool:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.delete_result = "DELETE 1"

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((query, args))
        if "INSERT INTO content_ops_deflection_reports" in query:
            account_id, request_id, snapshot, artifact, delivery_email = args
            key = (str(account_id), str(request_id))
            existing = self.rows.get(key, {})
            self.rows[key] = {
                "account_id": str(account_id),
                "request_id": str(request_id),
                "snapshot": snapshot,
                "artifact": artifact,
                "paid": bool(existing.get("paid")),
                "payment_reference": existing.get("payment_reference"),
                "delivery_email": delivery_email,
            }
            return "INSERT 0 1"
        if "UPDATE content_ops_deflection_reports" in query:
            account_id, request_id, payment_reference = args
            row = self.rows.get((str(account_id), str(request_id)))
            if row is None:
                return "UPDATE 0"
            row["paid"] = True
            row["payment_reference"] = payment_reference
            return "UPDATE 1"
        if "DELETE FROM content_ops_deflection_reports" in query:
            account_id, request_id = args
            if self.delete_result == "DELETE 1":
                self.rows.pop((str(account_id), str(request_id)), None)
            return self.delete_result
        raise AssertionError(query)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrow_calls.append((query, args))
        account_id, request_id = args
        row = self.rows.get((str(account_id), str(request_id)))
        return dict(row) if row is not None else None


@pytest.mark.asyncio
async def test_deflection_paid_postgres_smoke_skips_before_pool_without_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_create_pool(database_url: str) -> object:
        raise AssertionError(f"pool should not be opened for {database_url}")

    output = tmp_path / "result.json"
    monkeypatch.setattr(smoke, "_create_pool", fail_create_pool)

    code = await smoke._main([
        "--database-url",
        "postgres://example",
        "--account-id",
        _ACCOUNT_ID,
        "--output",
        str(output),
    ])

    assert code == smoke.SKIPPED_EXIT
    assert '"not_run_reason": "missing_confirm_postgres_write"' in output.read_text()


@pytest.mark.asyncio
async def test_deflection_paid_postgres_smoke_rejects_non_ephemeral_request_id() -> None:
    pool = _Pool()

    code, payload = await smoke.run_deflection_paid_postgres_smoke(
        _args(request_id="real-customer-request"),
        pool,
    )

    assert code == smoke.SKIPPED_EXIT
    assert payload["not_run_reason"] == "request_id_not_ephemeral"
    assert pool.execute_calls == []
    assert pool.fetchrow_calls == []


@pytest.mark.asyncio
async def test_deflection_paid_postgres_smoke_rejects_invalid_account_id() -> None:
    pool = _Pool()

    code, payload = await smoke.run_deflection_paid_postgres_smoke(
        _args(account_id="not-a-uuid"),
        pool,
    )

    assert code == smoke.SKIPPED_EXIT
    assert payload["not_run_reason"] == "invalid_account_id"
    assert pool.execute_calls == []
    assert pool.fetchrow_calls == []


@pytest.mark.asyncio
async def test_deflection_paid_postgres_smoke_refuses_existing_ephemeral_request() -> None:
    pool = _Pool()
    pool.rows[(_ACCOUNT_ID, "smoke-existing")] = {
        "account_id": _ACCOUNT_ID,
        "request_id": "smoke-existing",
        "snapshot": "{}",
        "artifact": "{}",
        "paid": False,
        "payment_reference": None,
    }

    code, payload = await smoke.run_deflection_paid_postgres_smoke(
        _args(request_id="smoke-existing"),
        pool,
    )

    assert code == smoke.SKIPPED_EXIT
    assert payload["not_run_reason"] == "ephemeral_request_already_exists"
    assert pool.execute_calls == []
    assert len(pool.fetchrow_calls) == 1


@pytest.mark.asyncio
async def test_deflection_paid_postgres_smoke_seeds_locks_marks_paid_and_rereads() -> None:
    pool = _Pool()

    code, payload = await smoke.run_deflection_paid_postgres_smoke(
        _args(
            request_id="smoke-paid-flow",
            payment_reference="cs_smoke_paid_flow",
        ),
        pool,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["skipped"] is False
    assert payload["account_id"] == _ACCOUNT_ID
    assert payload["request_id"] == "smoke-paid-flow"
    assert payload["payment_reference"] == "cs_smoke_paid_flow"
    assert payload["locked_before_paid"] is True
    assert payload["paid_after_checkout"] is True
    assert payload["snapshot_summary"] == {
        "drafted_answer_count": 1,
        "generated": 1,
        "no_proven_answer_count": 0,
        "repeat_ticket_count": 1,
    }
    assert payload["artifact_summary"] == payload["snapshot_summary"]
    assert len(pool.execute_calls) == 2
    assert "INSERT INTO content_ops_deflection_reports" in pool.execute_calls[0][0]
    assert "UPDATE content_ops_deflection_reports" in pool.execute_calls[1][0]
    assert pool.execute_calls[1][1] == (
        _ACCOUNT_ID,
        "smoke-paid-flow",
        "cs_smoke_paid_flow",
    )
    saved = pool.rows[(_ACCOUNT_ID, "smoke-paid-flow")]
    assert saved["paid"] is True
    assert saved["payment_reference"] == "cs_smoke_paid_flow"


@pytest.mark.asyncio
async def test_deflection_paid_postgres_smoke_can_cleanup_successful_smoke_row() -> None:
    pool = _Pool()

    code, payload = await smoke.run_deflection_paid_postgres_smoke(
        _args(
            request_id="smoke-cleanup",
            payment_reference="cs_smoke_cleanup",
            cleanup_on_success=True,
        ),
        pool,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["cleanup_requested"] is True
    assert payload["cleanup_deleted"] is True
    assert (_ACCOUNT_ID, "smoke-cleanup") not in pool.rows
    assert len(pool.execute_calls) == 3
    assert "DELETE FROM content_ops_deflection_reports" in pool.execute_calls[2][0]
    assert pool.execute_calls[2][1] == (_ACCOUNT_ID, "smoke-cleanup")


@pytest.mark.asyncio
async def test_deflection_paid_postgres_smoke_reports_cleanup_miss() -> None:
    pool = _Pool()
    pool.delete_result = "DELETE 0"

    code, payload = await smoke.run_deflection_paid_postgres_smoke(
        _args(
            request_id="smoke-cleanup-miss",
            payment_reference="cs_smoke_cleanup_miss",
            cleanup_on_success=True,
        ),
        pool,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["error"] == "cleanup_failed"
    assert payload["cleanup_requested"] is True
    assert payload["cleanup_deleted"] is False
    assert payload["paid_after_checkout"] is True
    assert pool.rows[(_ACCOUNT_ID, "smoke-cleanup-miss")]["paid"] is True


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "database_url": "postgres://example",
        "account_id": _ACCOUNT_ID,
        "request_id": "",
        "payment_reference": "",
        "amount_cents": 150000,
        "currency": "usd",
        "confirm_postgres_write": True,
        "cleanup_on_success": False,
        "output": None,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


_ACCOUNT_ID = "00000000-0000-0000-0000-000000000116"
