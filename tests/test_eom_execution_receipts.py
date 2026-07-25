"""Execution-receipt contract for the two EOM reconciliation CLIs."""

from __future__ import annotations

import hashlib
import asyncio
import json
import stat
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO / "tests"))

import import_eom_customers_live as calendar_import  # noqa: E402
import sync_eom_portal_customers as portal_sync  # noqa: E402
from test_eom_live_calendar_import import (  # noqa: E402
    StubCRM,
    StubPool,
    _record,
)
from eom_execution_receipt import (  # noqa: E402
    EomExecutionReceipt,
    run_receipted,
)

GIT_SHA = "a" * 40
CONTACT_A = "11111111-1111-1111-1111-111111111111"
CONTACT_B = "22222222-2222-2222-2222-222222222222"


def _receipt(tmp_path: Path, **overrides) -> EomExecutionReceipt:
    options = {
        "receipt_dir": tmp_path,
        "tool": "import_eom_customers_live",
        "mode": "write",
        "script_path": SCRIPTS / "import_eom_customers_live.py",
        "receipt_id": uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        "started_at_utc": "2026-07-25T05:00:00Z",
        "git_sha": GIT_SHA,
    }
    options.update(overrides)
    return EomExecutionReceipt(**options)


def _load_only_final(receipt_dir: Path) -> tuple[Path, dict]:
    final_paths = list(receipt_dir.glob("*.exit-*.json"))
    assert len(final_paths) == 1
    return final_paths[0], json.loads(final_paths[0].read_text())


def test_receipt_is_private_source_bound_and_allowlisted(tmp_path):
    receipt = _receipt(tmp_path)
    assert receipt.in_progress_path.exists()
    assert stat.S_IMODE(receipt.in_progress_path.stat().st_mode) == 0o600

    receipt.set_outcome_counts(
        {"created": 2, "updated": 1, "unchanged": 3, "errors": 0}
    )
    receipt.record_changed_contact_id(CONTACT_B)
    receipt.record_changed_contact_id(CONTACT_A)
    receipt.record_changed_contact_id(CONTACT_A)
    final_path = receipt.finalize(0)

    assert final_path.exists()
    assert not receipt.in_progress_path.exists()
    assert stat.S_IMODE(final_path.stat().st_mode) == 0o600
    payload = json.loads(final_path.read_text())
    assert set(payload) == {
        "schema_version",
        "receipt_id",
        "tool",
        "mode",
        "started_at_utc",
        "ended_at_utc",
        "git_sha",
        "script_sha256",
        "exit_code",
        "outcome_counts",
        "portal_totals",
        "changed_contact_ids",
    }
    assert payload["git_sha"] == GIT_SHA
    assert payload["script_sha256"] == hashlib.sha256(
        (SCRIPTS / "import_eom_customers_live.py").read_bytes()
    ).hexdigest()
    assert payload["changed_contact_ids"] == [CONTACT_A, CONTACT_B]
    assert payload["exit_code"] == 0


def test_payload_api_rejects_unknown_counts_and_non_uuid_identifiers(tmp_path):
    receipt = _receipt(tmp_path)
    with pytest.raises(ValueError, match="unsupported outcome"):
        receipt.set_outcome_counts({"customer_name": 1})
    with pytest.raises(ValueError, match="badly formed"):
        receipt.record_changed_contact_id("not-a-uuid")


def test_calendar_dry_run_receipt_counts_planned_records():
    class Recorder:
        counts = None

        def set_outcome_counts(self, counts):
            self.counts = dict(counts)

    recorder = Recorder()
    counts = asyncio.run(
        calendar_import.run_import([_record(), _record()], True, receipt=recorder)
    )
    assert counts["import-planned"] == 2
    assert recorder.counts["import-planned"] == 2


def test_final_collision_never_overwrites_and_preserves_in_progress(tmp_path):
    receipt = _receipt(tmp_path)
    collision = receipt.final_path_for(0)
    collision.write_text("existing artifact\n")

    with pytest.raises(FileExistsError):
        receipt.finalize(0)

    assert collision.read_text() == "existing artifact\n"
    assert receipt.in_progress_path.exists()
    assert stat.S_IMODE(receipt.in_progress_path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("failure", "expected_exit"),
    [
        (SystemExit(7), 7),
        (RuntimeError("failed"), 1),
        (KeyboardInterrupt(), 130),
    ],
)
def test_exception_exit_is_finalized_before_reraising(
    tmp_path, failure, expected_exit
):
    receipt = _receipt(tmp_path)

    def fail():
        raise failure

    with pytest.raises(type(failure)):
        run_receipted(receipt, fail)
    final_path, payload = _load_only_final(tmp_path)
    assert f".exit-{expected_exit}.json" in final_path.name
    assert payload["exit_code"] == expected_exit
    assert payload["ended_at_utc"].endswith("Z")


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (calendar_import, []),
        (portal_sync, ["--apply"]),
    ],
)
def test_write_modes_require_receipt_before_async_runtime(module, argv, monkeypatch):
    entered = False

    async def forbidden_run(*_args, **_kwargs):
        nonlocal entered
        entered = True
        return 0

    monkeypatch.setattr(module, "run", forbidden_run)
    with pytest.raises(SystemExit) as raised:
        module.main(argv)
    assert raised.value.code == 2
    assert entered is False


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (calendar_import, ["--dry-run"]),
        (portal_sync, []),
    ],
)
def test_dry_runs_may_omit_receipt(module, argv, monkeypatch):
    observed = []

    async def fake_run(_args, receipt=None):
        observed.append(receipt)
        return 0

    monkeypatch.setattr(module, "run", fake_run)
    assert module.main(argv) == 0
    assert observed == [None]


@pytest.mark.parametrize(
    ("module", "argv", "tool", "mode"),
    [
        (
            calendar_import,
            ["--dry-run"],
            "import_eom_customers_live",
            "dry-run",
        ),
        (
            portal_sync,
            ["--apply"],
            "sync_eom_portal_customers",
            "apply",
        ),
    ],
)
def test_real_cli_creates_in_progress_before_runtime_and_finalizes(
    module, argv, tool, mode, monkeypatch, tmp_path
):
    async def fake_run(_args, receipt=None):
        assert receipt is not None
        assert receipt.in_progress_path.exists()
        assert stat.S_IMODE(receipt.in_progress_path.stat().st_mode) == 0o600
        receipt.set_outcome_counts({"errors": 0})
        if tool == "sync_eom_portal_customers":
            receipt.set_portal_totals({"demoted": 1, "eligible": 2, "kept": 1})
        receipt.record_changed_contact_id(CONTACT_A)
        return 0

    monkeypatch.setattr(module, "run", fake_run)
    assert module.main([*argv, "--receipt-dir", str(tmp_path)]) == 0

    _path, payload = _load_only_final(tmp_path)
    assert payload["tool"] == tool
    assert payload["mode"] == mode
    assert payload["changed_contact_ids"] == [CONTACT_A]
    assert payload["exit_code"] == 0


class _Recorder:
    def __init__(self):
        self.contact_ids = []
        self.portal_totals = {}

    def record_changed_contact_id(self, contact_id):
        self.contact_ids.append(str(contact_id))

    def set_portal_totals(self, totals):
        self.portal_totals = dict(totals)


def test_portal_demotion_records_returned_contact_uuid():
    class Pool:
        async def fetch(self, *_args):
            return [
                {
                    "id": CONTACT_A,
                    "full_name": "redacted",
                    "tags": [],
                    "phone": None,
                    "email": None,
                    "address": None,
                }
            ]

        async def fetchrow(self, *_args):
            return {"id": CONTACT_A}

    recorder = _Recorder()
    totals = {}
    demoted, eligible = asyncio.run(
        portal_sync.demote_unmatched(
            Pool(),
            set(),
            apply=True,
            guard_keys={"phones": set(), "emails": set(), "addrs": set(), "names": set()},
            receipt=recorder,
            receipt_totals=totals,
        )
    )
    assert (demoted, eligible) == (1, 1)
    assert totals == {"demoted": 1, "eligible": 1, "kept": 0}
    assert recorder.contact_ids == [CONTACT_A]


def test_calendar_create_records_changed_contact_id():
    recorder = _Recorder()
    outcome = asyncio.run(
        calendar_import.import_one(
            _record(),
            StubCRM(),
            StubPool(rows=[None, None]),
            receipt=recorder,
        )
    )
    assert outcome == "created"
    assert set(recorder.contact_ids) == {"new-id"}
