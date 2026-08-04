"""Durable private receipts for EOM operator scripts."""

from __future__ import annotations

import asyncio
import json
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import import_eom_customers_live as calendar_import  # noqa: E402
import sync_eom_portal_customers as portal_sync  # noqa: E402
from eom_execution_receipt import EomExecutionReceipt, run_receipted  # noqa: E402
from test_eom_live_calendar_import import StubPool, _record  # noqa: E402
from test_sync_eom_portal_customers import SyncPool, _customer  # noqa: E402


UUID_A = "11111111-1111-4111-8111-111111111111"
UUID_B = "22222222-2222-4222-8222-222222222222"


def _payload(path: Path) -> dict:
    return json.loads(path.read_text())


def _private(path: Path) -> bool:
    return stat.S_IMODE(path.stat().st_mode) == 0o600


def test_existing_receipt_directory_must_be_private(tmp_path):
    unsafe = tmp_path / "unsafe-receipts"
    unsafe.mkdir()
    unsafe.chmod(0o755)
    try:
        with pytest.raises(ValueError, match="private mode 0700"):
            EomExecutionReceipt(
                receipt_dir=unsafe,
                tool="import_eom_customers_live",
                mode="write",
                script_path=REPO / "scripts" / "import_eom_customers_live.py",
                receipt_id=UUID_A,
            )
    finally:
        unsafe.chmod(0o700)


def test_receipt_publishes_private_source_bound_non_pii_payload(tmp_path):
    receipt = EomExecutionReceipt(
        receipt_dir=tmp_path,
        tool="import_eom_customers_live",
        mode="write",
        script_path=REPO / "scripts" / "import_eom_customers_live.py",
        receipt_id=UUID_A,
    )
    assert receipt.in_progress_path.exists()
    assert _private(receipt.in_progress_path)

    receipt.record_outcome_counts({"created": 1, "errors": 0})
    receipt.record_changed_contact_id(UUID_B)
    receipt.record_demotions(demoted=0, eligible=3, kept=2)
    final = receipt.finalize(0)

    assert final.exists()
    assert _private(final)
    assert not receipt.in_progress_path.exists()
    payload = _payload(final)
    assert payload["schema_version"] == 1
    assert payload["receipt_id"] == UUID_A
    assert payload["tool"] == "import_eom_customers_live"
    assert payload["mode"] == "write"
    assert payload["git_sha"]
    assert len(payload["script_hash_sha256"]) == 64
    assert payload["exit_code"] == 0
    assert payload["outcome_counts"] == {"created": 1, "errors": 0}
    assert payload["changed_contact_ids"] == [UUID_B]
    assert payload["demotion_totals"] == {"demoted": 0, "eligible": 3, "kept": 2}
    serialized = json.dumps(payload)
    for forbidden in (
        "customer@example.com",
        "217-555-1212",
        "123 Main",
        "token",
        "baseUrl",
        str(tmp_path),
    ):
        assert forbidden not in serialized


def test_receipt_finalization_never_overwrites_existing_final(tmp_path):
    first = EomExecutionReceipt(
        receipt_dir=tmp_path,
        tool="sync_eom_portal_customers",
        mode="apply",
        script_path=REPO / "scripts" / "sync_eom_portal_customers.py",
        receipt_id=UUID_A,
    )
    first_final = first.finalize(0)
    first_payload = first_final.read_text()

    second = EomExecutionReceipt(
        receipt_dir=tmp_path,
        tool="sync_eom_portal_customers",
        mode="apply",
        script_path=REPO / "scripts" / "sync_eom_portal_customers.py",
        receipt_id=UUID_A,
    )
    with pytest.raises(FileExistsError):
        second.finalize(0)
    assert first_final.read_text() == first_payload


def test_indeterminate_mutation_keeps_in_progress_and_no_final(tmp_path):
    receipt = EomExecutionReceipt(
        receipt_dir=tmp_path,
        tool="import_eom_customers_live",
        mode="write",
        script_path=REPO / "scripts" / "import_eom_customers_live.py",
        receipt_id=UUID_A,
    )

    async def interrupted():
        async with receipt.mutation_boundary():
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_receipted(receipt, lambda: asyncio.run(interrupted()))

    assert receipt.in_progress_path.exists()
    assert not list(tmp_path.glob("eom-*.exit-*.json"))
    assert _payload(receipt.in_progress_path)["indeterminate"] is True


def test_indeterminate_mutation_note_is_attached_to_original_interrupt(tmp_path):
    receipt = EomExecutionReceipt(
        receipt_dir=tmp_path,
        tool="import_eom_customers_live",
        mode="write",
        script_path=REPO / "scripts" / "import_eom_customers_live.py",
        receipt_id=UUID_A,
    )

    async def interrupted():
        async with receipt.mutation_boundary():
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        run_receipted(receipt, lambda: asyncio.run(interrupted()))

    assert "left in-progress receipt" in "\n".join(
        getattr(raised.value, "__notes__", [])
    )


def test_ordinary_mutation_exception_finalizes_failure_receipt(tmp_path):
    receipt = EomExecutionReceipt(
        receipt_dir=tmp_path,
        tool="import_eom_customers_live",
        mode="write",
        script_path=REPO / "scripts" / "import_eom_customers_live.py",
        receipt_id=UUID_A,
    )

    async def failed_mutation():
        async with receipt.mutation_boundary():
            raise RuntimeError("database timeout")

    with pytest.raises(RuntimeError, match="database timeout"):
        run_receipted(receipt, lambda: asyncio.run(failed_mutation()))

    final = receipt.final_path_for(1)
    assert final.exists()
    assert not receipt.in_progress_path.exists()
    payload = _payload(final)
    assert payload["exit_code"] == 1
    assert payload["indeterminate"] is False


def test_calendar_live_write_requires_receipt_before_runtime_work(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("provider/database work must not start")

    monkeypatch.setattr(calendar_import, "resolve_calendar_ids", fail_if_called)
    with pytest.raises(SystemExit):
        calendar_import.main(["--calendar", "residential"])


def test_portal_apply_requires_receipt_before_runtime_work(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("login/runtime work must not start")

    monkeypatch.setattr(portal_sync, "portal_login", fail_if_called)
    with pytest.raises(SystemExit):
        portal_sync.main(["--apply"])


class _ReceiptSpy:
    def __init__(self):
        self.changed = []
        self.counts = []
        self.demotions = None
        self.boundary_open = False
        self.changed_recorded_inside_boundary = []

    async def __aenter__(self):
        self.boundary_open = True
        return self

    async def __aexit__(self, *_exc):
        self.boundary_open = False
        return False

    def mutation_boundary(self):
        return self

    def record_changed_contact_id(self, contact_id):
        self.changed.append(str(contact_id))
        self.changed_recorded_inside_boundary.append(self.boundary_open)

    def record_outcome_counts(self, counts):
        self.counts.append(dict(counts))

    def record_demotions(self, **totals):
        self.demotions = totals


class _CalendarCRM:
    async def search_contacts(self, **_kwargs):
        return []

    async def create_contact(self, data):
        self.created = data
        return {"id": UUID_A, "_was_created": True, "status": data["status"]}

    async def log_interaction(self, **_kwargs):
        return {"inserted": True}


def test_calendar_import_records_changed_contacts_and_counts():
    receipt = _ReceiptSpy()
    outcome = asyncio.run(
        calendar_import.import_one(
            _record(), _CalendarCRM(), StubPool(rows=[None, None]), receipt=receipt
        )
    )
    assert outcome == "created"
    assert UUID_A in receipt.changed
    assert receipt.changed_recorded_inside_boundary
    assert all(receipt.changed_recorded_inside_boundary)

    counts = asyncio.run(
        calendar_import.run_import([], dry_run=True, receipt=receipt)
    )
    assert counts["import-planned"] == 0
    assert receipt.counts[-1]["import-planned"] == 0


class _PortalCRM:
    async def create_contact(self, data, *, merge_existing=True):
        self.created = (data, merge_existing)
        return {"id": UUID_A, "_was_created": True}


def test_portal_sync_records_changed_contacts_and_demotion_totals():
    receipt = _ReceiptSpy()
    outcome, contact_id = asyncio.run(
        portal_sync.sync_one(
            _customer(primaryPhone=None, primaryEmail=None, sites=[]),
            _PortalCRM(),
            SyncPool(rows=[None]),
            apply=True,
            receipt=receipt,
        )
    )
    assert (outcome, contact_id) == ("created", UUID_A)
    assert receipt.changed == [UUID_A]

    pool = SyncPool(demotion_rows=[
        {"id": UUID_B, "full_name": "Moved Away", "tags": []},
        {
            "id": "33333333-3333-4333-8333-333333333333",
            "full_name": "Still Here",
            "tags": [],
            "email": "kept@example.com",
        },
    ])
    demoted, eligible = asyncio.run(
        portal_sync.demote_unmatched(
            pool,
            set(),
            apply=True,
            guard_keys={
                "phones": set(),
                "emails": {"kept@example.com"},
                "addrs": set(),
                "names": set(),
            },
            receipt=receipt,
        )
    )
    assert (demoted, eligible) == (1, 2)
    assert UUID_B in receipt.changed
    assert receipt.changed_recorded_inside_boundary
    assert all(receipt.changed_recorded_inside_boundary)
    assert receipt.demotions == {"demoted": 1, "eligible": 2, "kept": 1}
