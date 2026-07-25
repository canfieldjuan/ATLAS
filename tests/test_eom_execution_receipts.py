"""Execution-receipt contract for the EOM live Calendar importer."""

from __future__ import annotations

import hashlib
import asyncio
import json
import os
import py_compile
import shutil
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
import eom_execution_receipt as receipt_module  # noqa: E402
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

        def assert_healthy(self):
            return None

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
    recovery_payload = json.loads(receipt.in_progress_path.read_text())
    assert recovery_payload["ended_at_utc"] is None
    assert recovery_payload["exit_code"] is None


@pytest.mark.parametrize(
    ("failure", "expected_exit"),
    [
        (SystemExit(), 0),
        (SystemExit(False), 0),
        (SystemExit(True), 1),
        (SystemExit(7), 7),
        (SystemExit(-1), 255),
        (SystemExit(256), 0),
        (SystemExit(513), 1),
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


def test_preexisting_receipt_directory_must_not_be_writable_by_others(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)

    with pytest.raises(ValueError, match="must not be writable by other users"):
        _receipt(shared)


def test_receipt_directory_must_support_hard_links(tmp_path, monkeypatch):
    monkeypatch.setattr(
        os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("hard links are unsupported")
        ),
    )

    with pytest.raises(ValueError, match="must support hard links"):
        _receipt(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_receipt_mode_is_independent_of_restrictive_umask(tmp_path):
    previous_umask = os.umask(0o777)
    try:
        receipt = _receipt(tmp_path)
        assert stat.S_IMODE(receipt.in_progress_path.stat().st_mode) == 0o600
        final_path = receipt.finalize(0)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(final_path.stat().st_mode) == 0o600


def test_initial_in_progress_entry_is_directory_fsynced(tmp_path, monkeypatch):
    fsynced_types = []
    real_fsync = os.fsync

    def recording_fsync(descriptor):
        fsynced_types.append(stat.S_IFMT(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    receipt = _receipt(tmp_path)

    assert fsynced_types[-2:] == [stat.S_IFREG, stat.S_IFDIR]
    receipt.finalize(0)


def test_recorded_mutation_evidence_is_durable_before_finalization(
    tmp_path, monkeypatch
):
    fsynced_types = []
    real_fsync = os.fsync

    def recording_fsync(descriptor):
        fsynced_types.append(stat.S_IFMT(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    receipt = _receipt(tmp_path)
    fsynced_types.clear()

    receipt.record_changed_contact_id(CONTACT_A)
    payload = json.loads(receipt.in_progress_path.read_text())
    assert payload["changed_contact_ids"] == [CONTACT_A]
    assert payload["ended_at_utc"] is None
    assert fsynced_types == [stat.S_IFREG, stat.S_IFDIR]

    receipt.set_outcome_counts({"updated": 1})
    payload = json.loads(receipt.in_progress_path.read_text())
    assert payload["outcome_counts"] == {"updated": 1}

def test_receipt_rejects_dirty_tracked_worktree(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "operator.py"
    dependency = repo / "dependency.py"
    script.write_text("from dependency import VALUE\n")
    dependency.write_text("VALUE = 1\n")
    subprocess_options = {
        "cwd": repo,
        "check": True,
        "capture_output": True,
        "text": True,
    }
    receipt_module.subprocess.run(["git", "init", "-q"], **subprocess_options)
    receipt_module.subprocess.run(["git", "add", "."], **subprocess_options)
    receipt_module.subprocess.run(
        [
            "git",
            "-c",
            "user.name=Receipt Test",
            "-c",
            "user.email=receipt-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        **subprocess_options,
    )
    dependency.write_text("VALUE = 2\n")
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_dir.chmod(0o700)

    with pytest.raises(
        RuntimeError, match="requires a clean worktree"
    ):
        EomExecutionReceipt(
            receipt_dir=receipt_dir,
            tool="import_eom_customers_live",
            mode="write",
            script_path=script,
        )


def test_receipt_rejects_untracked_import_shadow(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "operator.py"
    script.write_text("import httpx\n")
    subprocess_options = {
        "cwd": repo,
        "check": True,
        "capture_output": True,
        "text": True,
    }
    receipt_module.subprocess.run(["git", "init", "-q"], **subprocess_options)
    receipt_module.subprocess.run(["git", "add", "."], **subprocess_options)
    receipt_module.subprocess.run(
        [
            "git",
            "-c",
            "user.name=Receipt Test",
            "-c",
            "user.email=receipt-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        **subprocess_options,
    )
    (scripts / "httpx.py").write_text("raise RuntimeError('shadowed')\n")
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_dir.chmod(0o700)

    with pytest.raises(
        RuntimeError, match="requires a clean worktree"
    ):
        EomExecutionReceipt(
            receipt_dir=receipt_dir,
            tool="import_eom_customers_live",
            mode="write",
            script_path=script,
        )


def _calendar_process_fixture(tmp_path, ignored_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("eom_execution_receipt.py", "import_eom_customers_live.py"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    (scripts / "import_calendar_contacts.py").write_text(
        '"""Minimal fixture; source preflight must run before this import."""\n'
    )
    (repo / ".gitignore").write_text(f"/{ignored_path}\n")
    subprocess_options = {
        "cwd": repo,
        "check": True,
        "capture_output": True,
        "text": True,
    }
    receipt_module.subprocess.run(["git", "init", "-q"], **subprocess_options)
    receipt_module.subprocess.run(["git", "add", "."], **subprocess_options)
    receipt_module.subprocess.run(
        [
            "git",
            "-c",
            "user.name=Receipt Test",
            "-c",
            "user.email=receipt-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        **subprocess_options,
    )
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_dir.chmod(0o700)
    return repo, scripts, receipt_dir


def _run_receipted_calendar(
    repo, scripts, receipt_dir, *, isolated=True, env=None
):
    command = [sys.executable]
    if isolated:
        command.append("-I")
    command.extend(
        [
            str(scripts / "import_eom_customers_live.py"),
            "--dry-run",
            "--receipt-dir",
            str(receipt_dir),
        ]
    )
    return receipt_module.subprocess.run(
        command,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_process_receipt_requires_isolated_python_startup(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )

    result = _run_receipted_calendar(
        repo, scripts, receipt_dir, isolated=False
    )

    assert result.returncode == 2
    assert "requires isolated Python startup" in result.stderr
    assert list(receipt_dir.iterdir()) == []


def test_isolated_process_blocks_sitecustomize_before_preflight(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, "sitecustomize.py"
    )
    marker = tmp_path / "sitecustomize-executed"
    shadow = repo / "sitecustomize.py"
    shadow.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "Path(__file__).unlink()\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)

    result = _run_receipted_calendar(
        repo, scripts, receipt_dir, env=env
    )

    assert result.returncode != 0
    assert "rejects ignored Python import shadows" in result.stderr
    assert not marker.exists()
    assert shadow.exists()
    assert list(receipt_dir.iterdir()) == []


def test_process_preflight_blocks_self_removing_shadow_before_import(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, "scripts/json.py"
    )
    marker = tmp_path / "shadow-executed"
    shadow = scripts / "json.py"
    shadow.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "Path(__file__).unlink()\n"
        "raise RuntimeError('ignored shadow executed')\n"
    )

    result = _run_receipted_calendar(repo, scripts, receipt_dir)

    assert result.returncode != 0
    assert "rejects ignored Python import shadows" in result.stderr
    assert not marker.exists()
    assert shadow.exists()
    assert list(receipt_dir.iterdir()) == []


@pytest.mark.parametrize("package_kind", ("regular", "namespace"))
def test_process_preflight_rejects_ignored_package_symlink(
    tmp_path, package_kind
):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, "scripts/shadow_package"
    )
    target = tmp_path / f"{package_kind}-target"
    target.mkdir()
    if package_kind == "regular":
        (target / "__init__.py").write_text("VALUE = 'unreviewed'\n")
    else:
        (target / "child.py").write_text("VALUE = 'unreviewed'\n")
    (scripts / "shadow_package").symlink_to(target, target_is_directory=True)

    result = _run_receipted_calendar(repo, scripts, receipt_dir)

    assert result.returncode != 0
    assert "rejects ignored Python import shadows" in result.stderr
    assert list(receipt_dir.iterdir()) == []


def test_process_preflight_compares_skip_worktree_source_to_head(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )
    dependency = scripts / "import_calendar_contacts.py"
    marker = tmp_path / "tracked-shadow-executed"
    receipt_module.subprocess.run(
        [
            "git",
            "update-index",
            "--skip-worktree",
            "scripts/import_calendar_contacts.py",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    dependency.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
    )
    status = receipt_module.subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    result = _run_receipted_calendar(repo, scripts, receipt_dir)

    assert result.returncode != 0
    assert "tracked Python source to match HEAD" in result.stderr
    assert not marker.exists()
    assert list(receipt_dir.iterdir()) == []


def test_receipt_rejects_ignored_bytecode_for_tracked_source(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "operator.py"
    dependency = scripts / "dependency.py"
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    script.write_text("from dependency import VALUE\n")
    dependency.write_text("VALUE = 1\n")
    subprocess_options = {
        "cwd": repo,
        "check": True,
        "capture_output": True,
        "text": True,
    }
    receipt_module.subprocess.run(["git", "init", "-q"], **subprocess_options)
    receipt_module.subprocess.run(["git", "add", "."], **subprocess_options)
    receipt_module.subprocess.run(
        [
            "git",
            "-c",
            "user.name=Receipt Test",
            "-c",
            "user.email=receipt-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        **subprocess_options,
    )
    clean_stat = dependency.stat()
    dependency.write_text("VALUE = 2\n")
    py_compile.compile(str(dependency), doraise=True)
    dependency.write_text("VALUE = 1\n")
    os.utime(
        dependency,
        ns=(clean_stat.st_atime_ns, clean_stat.st_mtime_ns),
    )
    assert (
        receipt_module.subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            **subprocess_options,
        ).stdout
        == ""
    )
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_dir.chmod(0o700)

    with pytest.raises(
        RuntimeError, match="rejects cached bytecode"
    ):
        EomExecutionReceipt(
            receipt_dir=receipt_dir,
            tool="import_eom_customers_live",
            mode="write",
            script_path=script,
        )


def test_entrypoint_bootstraps_source_trust_before_local_imports():
    calendar_source = (
        SCRIPTS / "import_eom_customers_live.py"
    ).read_text()

    calendar_disable = calendar_source.index("sys.dont_write_bytecode = True")
    assert calendar_disable < calendar_source.index(
        "import import_calendar_contacts as ics"
    )
    source_preflight = calendar_source.index(
        "_receipt_module, _validated_git_sha = _trusted_receipt_module"
    )
    assert source_preflight < calendar_source.index(
        "import import_calendar_contacts as ics"
    )


def test_clean_real_entrypoint_does_not_create_or_reject_own_bytecode(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("eom_execution_receipt.py", "import_eom_customers_live.py"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    (scripts / "import_calendar_contacts.py").write_text(
        '"""Minimal import-only fixture for the EOM entrypoint smoke test."""\n'
    )
    services = repo / "atlas_brain" / "services"
    services.mkdir(parents=True)
    (repo / "atlas_brain" / "__init__.py").write_text("")
    (services / "__init__.py").write_text("")
    (services / "calendar_provider.py").write_text(
        "class GoogleCalendarProvider:\n"
        "    async def list_events(self, **_kwargs):\n"
        "        return []\n"
        "    async def aclose(self):\n"
        "        return None\n"
    )
    subprocess_options = {
        "cwd": repo,
        "check": True,
        "capture_output": True,
        "text": True,
    }
    receipt_module.subprocess.run(["git", "init", "-q"], **subprocess_options)
    receipt_module.subprocess.run(["git", "add", "."], **subprocess_options)
    receipt_module.subprocess.run(
        [
            "git",
            "-c",
            "user.name=Receipt Test",
            "-c",
            "user.email=receipt-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        **subprocess_options,
    )
    calendar_receipts = tmp_path / "calendar-receipts"
    calendar_receipts.mkdir()
    calendar_receipts.chmod(0o700)
    env = dict(os.environ)
    env.update(
        {
            "EOM_CALENDAR_COMMERCIAL": "commercial-fixture",
            "EOM_CALENDAR_RESIDENTIAL": "residential-fixture",
            "EOM_CALENDAR_ONE_TIME": "one-time-fixture",
            "PYTHONPATH": str(repo),
        }
    )
    receipt_module.subprocess.run(
        [
            sys.executable,
            "-I",
            str(scripts / "import_eom_customers_live.py"),
            "--dry-run",
            "--receipt-dir",
            str(calendar_receipts),
        ],
        env=env,
        **subprocess_options,
    )
    assert list(repo.rglob("*.pyc")) == []
    assert len(list(calendar_receipts.glob("*.exit-0.json"))) == 1


def test_failed_publication_sync_removes_new_final_link(tmp_path, monkeypatch):
    receipt = _receipt(tmp_path)
    final_path = receipt.final_path_for(0)

    def fail_directory_sync():
        raise OSError("storage sync failed")

    monkeypatch.setattr(receipt, "_fsync_directory", fail_directory_sync)
    with pytest.raises(OSError, match="storage sync failed"):
        receipt.finalize(0)

    assert not final_path.exists()
    recovery_payload = json.loads(receipt.in_progress_path.read_text())
    assert recovery_payload["ended_at_utc"] is None
    assert recovery_payload["exit_code"] is None


def test_receipt_directory_must_preexist(tmp_path):
    with pytest.raises(ValueError, match="must already exist"):
        _receipt(tmp_path / "missing")


def test_evidence_failure_is_deferred_until_finalize(tmp_path, monkeypatch):
    receipt = _receipt(tmp_path)
    calls = []

    def fail_persist():
        calls.append("persist")
        raise OSError("storage failed")

    monkeypatch.setattr(receipt, "_persist_in_progress", fail_persist)
    receipt.record_changed_contact_id(CONTACT_A)
    calls.append("reconciliation-completed")

    assert calls == ["persist", "reconciliation-completed"]
    with pytest.raises(RuntimeError, match="durably record"):
        receipt.finalize(0)


def test_initial_evidence_failure_stops_before_first_mutation(
    tmp_path, monkeypatch
):
    receipt = _receipt(tmp_path)
    started = []

    def fail_persist():
        raise OSError("storage failed")

    async def forbidden_import(*_args, **_kwargs):
        started.append("mutation")
        return "created"

    monkeypatch.setattr(receipt, "_persist_in_progress", fail_persist)
    monkeypatch.setattr(calendar_import, "import_one", forbidden_import)

    with pytest.raises(RuntimeError, match="durably record"):
        asyncio.run(
            calendar_import.run_import(
                [_record()], dry_run=False, receipt=receipt
            )
        )

    assert started == []


def test_mid_contact_evidence_failure_stops_before_next_contact(
    tmp_path, monkeypatch
):
    from atlas_brain.services import crm_provider
    from atlas_brain.storage import database

    receipt = _receipt(tmp_path)
    real_persist = receipt._persist_in_progress
    persist_calls = 0
    started = []

    def fail_after_initial_counts():
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            real_persist()
            return
        raise OSError("storage failed")

    async def record_one(rec, _crm, _pool, receipt=None):
        started.append(rec)
        receipt.record_changed_contact_id(CONTACT_A)
        return "created"

    monkeypatch.setattr(
        receipt, "_persist_in_progress", fail_after_initial_counts
    )
    monkeypatch.setattr(crm_provider, "get_crm_provider", object)
    monkeypatch.setattr(database, "get_db_pool", object)
    monkeypatch.setattr(calendar_import, "import_one", record_one)

    with pytest.raises(RuntimeError, match="durably record"):
        asyncio.run(
            calendar_import.run_import(
                [_record(phone="2175550101"), _record(phone="2175550102")],
                dry_run=False,
                receipt=receipt,
            )
        )

    assert len(started) == 1


def test_write_mode_requires_receipt_before_async_runtime(monkeypatch):
    entered = False

    async def forbidden_run(*_args, **_kwargs):
        nonlocal entered
        entered = True
        return 0

    monkeypatch.setattr(calendar_import, "run", forbidden_run)
    with pytest.raises(SystemExit) as raised:
        calendar_import.main([])
    assert raised.value.code == 2
    assert entered is False


def test_dry_run_may_omit_receipt(monkeypatch):
    observed = []

    async def fake_run(_args, receipt=None):
        observed.append(receipt)
        return 0

    monkeypatch.setattr(calendar_import, "run", fake_run)
    assert calendar_import.main(["--dry-run"]) == 0
    assert observed == [None]


def test_real_cli_creates_in_progress_before_runtime_and_finalizes(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        receipt_module, "establish_source_trust", lambda _root: GIT_SHA
    )

    async def fake_run(_args, receipt=None):
        assert receipt is not None
        assert receipt.in_progress_path.exists()
        assert stat.S_IMODE(receipt.in_progress_path.stat().st_mode) == 0o600
        receipt.set_outcome_counts({"errors": 0})
        receipt.record_changed_contact_id(CONTACT_A)
        return 0

    monkeypatch.setattr(calendar_import, "run", fake_run)
    assert calendar_import.main(
        ["--dry-run", "--receipt-dir", str(tmp_path)]
    ) == 0

    _path, payload = _load_only_final(tmp_path)
    assert payload["tool"] == "import_eom_customers_live"
    assert payload["mode"] == "dry-run"
    assert payload["changed_contact_ids"] == [CONTACT_A]
    assert payload["exit_code"] == 0


class _Recorder:
    def __init__(self):
        self.contact_ids = []

    def record_changed_contact_id(self, contact_id):
        self.contact_ids.append(str(contact_id))


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


def test_calendar_interaction_only_write_records_changed_contact_id():
    rec = _record(phone="(217) 555-9999")
    existing = {
        "id": CONTACT_A,
        **calendar_import.record_to_contact_data(rec),
    }

    class InteractionCRM(StubCRM):
        async def log_interaction(self, **kwargs):
            self.interactions.append(kwargs)
            return {"id": "interaction-1", "inserted": True}

    recorder = _Recorder()
    outcome = asyncio.run(
        calendar_import.import_one(
            rec,
            InteractionCRM(scoped_hit=existing),
            StubPool(),
            receipt=recorder,
        )
    )

    assert outcome == "unchanged"
    assert recorder.contact_ids == [CONTACT_A]

def test_calendar_race_merge_records_contact_even_when_followup_is_unchanged(
    monkeypatch,
):
    class RaceCRM(StubCRM):
        async def create_contact(self, data):
            self.created.append(data)
            return {
                "id": CONTACT_A,
                "_was_created": False,
                **calendar_import.record_to_contact_data(_record()),
            }

    async def unchanged(_pool, existing, _data):
        return str(existing["id"]), "unchanged"

    monkeypatch.setattr(calendar_import, "_update_matched", unchanged)
    recorder = _Recorder()
    outcome = asyncio.run(
        calendar_import.import_one(
            _record(),
            RaceCRM(),
            StubPool(rows=[None, None]),
            receipt=recorder,
        )
    )

    assert outcome == "unchanged"
    assert recorder.contact_ids == [CONTACT_A]
