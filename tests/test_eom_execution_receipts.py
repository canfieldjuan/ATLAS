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
import time
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


def test_receipt_directory_swap_cannot_receive_later_writes(tmp_path):
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_dir.chmod(0o700)
    receipt = _receipt(receipt_dir)

    original_directory = tmp_path / "receipts-original"
    receipt_dir.rename(original_directory)
    receipt_dir.mkdir()
    receipt_dir.chmod(0o700)

    receipt.record_changed_contact_id(CONTACT_A)

    assert list(receipt_dir.iterdir()) == []
    assert (original_directory / receipt.in_progress_path.name).exists()
    with pytest.raises(RuntimeError, match="durably record") as raised:
        receipt.assert_healthy()
    assert "receipt directory changed" in str(raised.value.__cause__)


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


def _calendar_process_fixture(tmp_path, ignored_path, extra_files=None):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("eom_execution_receipt.py", "import_eom_customers_live.py"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    (scripts / "import_calendar_contacts.py").write_text(
        '"""Minimal fixture; source preflight must run before this import."""\n'
    )
    for relative_path, source in (extra_files or {}).items():
        destination = repo / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
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
    repo,
    scripts,
    receipt_dir,
    *,
    isolated=True,
    env=None,
    entrypoint="scripts/import_eom_customers_live.py",
    dry_run=True,
    include_receipt=True,
    extra_args=(),
):
    reviewed_sha = receipt_module.subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    launcher_source = receipt_module.subprocess.run(
        ["git", "show", f"{reviewed_sha}:scripts/eom_execution_receipt.py"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    command = [sys.executable]
    if isolated:
        command.append("-I")
    command.extend(
        [
            "-",
            "--launch-reviewed",
            "--reviewed-git-sha",
            reviewed_sha,
            entrypoint,
        ]
    )
    if dry_run:
        command.append("--dry-run")
    if include_receipt:
        command.extend(["--receipt-dir", str(receipt_dir)])
    command.extend(extra_args)
    return receipt_module.subprocess.run(
        command,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        input=launcher_source,
        env=env,
    )


def test_source_trust_pins_one_revision_for_validation_and_receipt(
    tmp_path, monkeypatch
):
    completed_process = receipt_module.subprocess.CompletedProcess
    resolved_sha = "b" * 40
    sha_calls = 0
    tracked_revisions = []

    def fake_run(command, **_kwargs):
        return completed_process(command, 0, stdout="", stderr="")

    def fake_git_sha(_repo_root):
        nonlocal sha_calls
        sha_calls += 1
        return resolved_sha

    def fake_tracked_entries(_repo_root, revision):
        tracked_revisions.append(revision)
        return []

    monkeypatch.setattr(receipt_module.subprocess, "run", fake_run)
    monkeypatch.setattr(receipt_module, "_git_sha", fake_git_sha)
    monkeypatch.setattr(
        receipt_module, "_tracked_python_entries", fake_tracked_entries
    )

    assert receipt_module.establish_source_trust(tmp_path) == resolved_sha
    assert sha_calls == 1
    assert tracked_revisions == [resolved_sha]


def test_git_attestation_subprocesses_disable_replacement_objects(
    tmp_path, monkeypatch
):
    completed_process = receipt_module.subprocess.CompletedProcess
    observed_git_commands = []
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong-worktree"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(tmp_path / "fsmonitor-hook"))

    def fake_run(command, **kwargs):
        if command[0] == "git":
            observed_git_commands.append(
                (tuple(command), kwargs.get("env", {}))
            )
        if "rev-parse" in command and "--verify" in command:
            stdout = GIT_SHA + "\n"
        elif kwargs.get("text") is False:
            stdout = b""
        else:
            stdout = ""
        return completed_process(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(receipt_module.subprocess, "run", fake_run)

    assert receipt_module.establish_source_trust(tmp_path) == GIT_SHA
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    receipt_module._materialize_reviewed_python(tmp_path, GIT_SHA, snapshot)

    assert observed_git_commands
    assert all(
        env.get("GIT_NO_REPLACE_OBJECTS") == "1"
        for _command, env in observed_git_commands
    )
    assert all(
        "core.fsmonitor=false" in command
        for command, _env in observed_git_commands
    )
    assert all(
        "GIT_DIR" not in env
        and "GIT_WORK_TREE" not in env
        and "GIT_CONFIG_KEY_0" not in env
        and "GIT_CONFIG_VALUE_0" not in env
        and env["GIT_CONFIG_GLOBAL"] == os.devnull
        and env["GIT_CONFIG_COUNT"] == "0"
        for _command, env in observed_git_commands
    )
    assert any(
        "cat-file" in command
        for command, _env in observed_git_commands
    )


def _write_process_fixture_files(tmp_path, *, wait_for_rewrite=False):
    mutation_marker = tmp_path / "trusted-mutation"
    ready_marker = tmp_path / "calendar-fetch-started"
    release_marker = tmp_path / "dependency-rewrite-complete"
    wait_block = ""
    if wait_for_rewrite:
        wait_block = (
            f"        Path({str(ready_marker)!r}).write_text('ready')\n"
            f"        while not Path({str(release_marker)!r}).exists():\n"
            "            await asyncio.sleep(0.01)\n"
        )
    files = {
        "scripts/import_calendar_contacts.py": (
            "from dataclasses import dataclass\n"
            "import re\n"
            "_PHONE_RE = re.compile(r'.*\\\\d.*')\n"
            "_EMAIL_RE = re.compile(r'.+@.+')\n"
            "@dataclass\n"
            "class CustomerRecord:\n"
            "    name: str\n"
            "    address: str\n"
            "    phone: str | None\n"
            "    email: str | None\n"
            "    contact_name: str | None\n"
            "    notes: str\n"
            "    tags: list\n"
            "    contact_type: str\n"
            "    source_calendar: str\n"
            "    last_event_date: object\n"
            "    event_count: int\n"
            "    cancelled: bool\n"
            "def _strip_html(value): return value\n"
            "def _clean_summary(value, _commercial):\n"
            "    return value.replace(' - CANCELLED', ''), 'CANCELLED' in value\n"
            "def _normalize_address(value): return value.strip()\n"
            "def _extract_email(_value): return None\n"
            "def _extract_contact_name(_value): return None\n"
            "def _extract_phone(value):\n"
            "    digits = ''.join(char for char in value if char.isdigit())\n"
            "    return digits or None\n"
        ),
        "atlas_brain/__init__.py": "",
        "atlas_brain/services/__init__.py": "",
        "atlas_brain/storage/__init__.py": "",
        "atlas_brain/services/calendar_provider.py": (
            "import asyncio\n"
            "from datetime import datetime, timezone\n"
            "from pathlib import Path\n"
            "class Event:\n"
            "    status = 'confirmed'\n"
            "    summary = 'Snapshot Customer'\n"
            "    location = '123 Snapshot Lane, Effingham, IL'\n"
            "    description = '2175550101'\n"
            "    start = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)\n"
            "class GoogleCalendarProvider:\n"
            "    async def list_events(self, **_kwargs):\n"
            f"{wait_block}"
            "        return [Event()]\n"
            "    async def aclose(self): return None\n"
        ),
        "atlas_brain/services/crm_provider.py": (
            "from pathlib import Path\n"
            "CONTACT_ID = '11111111-1111-1111-1111-111111111111'\n"
            "class CRM:\n"
            "    async def search_contacts(self, **_kwargs): return []\n"
            "    async def create_contact(self, _payload):\n"
            f"        Path({str(mutation_marker)!r}).write_text(CONTACT_ID)\n"
            "        return {'id': CONTACT_ID, '_was_created': True}\n"
            "    async def log_interaction(self, **_kwargs):\n"
            "        return {'inserted': False}\n"
            "def get_crm_provider(): return CRM()\n"
        ),
        "atlas_brain/storage/database.py": (
            "class Pool:\n"
            "    async def initialize(self): return None\n"
            "    async def fetchrow(self, query, *args):\n"
            "        if 'SELECT status' in query:\n"
            "            return {'status': 'active', "
            "'business_context_id': 'effingham_maids'}\n"
            "        if 'UPDATE contacts' in query:\n"
            "            return {'id': args[0]}\n"
            "        return None\n"
            "_POOL = Pool()\n"
            "def get_db_pool(): return _POOL\n"
        ),
    }
    return files, mutation_marker, ready_marker, release_marker


def test_process_receipt_requires_isolated_python_startup(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )

    result = _run_receipted_calendar(
        repo, scripts, receipt_dir, isolated=False
    )

    assert result.returncode == 1
    assert "requires isolated Python startup" in result.stderr
    assert list(receipt_dir.iterdir()) == []


def test_reviewed_launcher_snapshot_is_private_and_read_only(tmp_path):
    repo, scripts, _receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )
    snapshot = tmp_path / "reviewed-python"
    snapshot.mkdir(mode=0o700)

    try:
        receipt_module._materialize_reviewed_python(
            repo, receipt_module._git_sha(repo), snapshot
        )
        reviewed_entrypoint = snapshot / "scripts" / (
            "import_eom_customers_live.py"
        )
        assert reviewed_entrypoint.read_bytes() == (
            scripts / "import_eom_customers_live.py"
        ).read_bytes()
        assert stat.S_IMODE(snapshot.stat().st_mode) == 0o500
        assert stat.S_IMODE(reviewed_entrypoint.stat().st_mode) == 0o400
    finally:
        receipt_module._remove_reviewed_python(snapshot)

    assert not snapshot.exists()


def test_direct_receipted_entrypoint_requires_reviewed_launcher(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )

    result = receipt_module.subprocess.run(
        [
            sys.executable,
            "-I",
            str(scripts / "import_eom_customers_live.py"),
            "--dry-run",
            "--receipt-dir",
            str(receipt_dir),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "requires the reviewed SHA launcher" in result.stderr
    assert list(receipt_dir.iterdir()) == []


def test_reviewed_launcher_rejects_unallowlisted_entrypoint(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )

    result = _run_receipted_calendar(
        repo,
        scripts,
        receipt_dir,
        entrypoint="scripts/unreviewed_operator.py",
    )

    assert result.returncode == 1
    assert "unsupported reviewed EOM entrypoint" in result.stderr
    assert list(receipt_dir.iterdir()) == []


def test_reviewed_launcher_rejects_sha_that_no_longer_matches_checkout(tmp_path):
    repo, _scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )
    old_sha = receipt_module.subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    launcher_source = receipt_module.subprocess.run(
        ["git", "show", f"{old_sha}:scripts/eom_execution_receipt.py"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (repo / "README.md").write_text("new reviewed head\n")
    receipt_module.subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt_module.subprocess.run(
        [
            "git",
            "-c",
            "user.name=Receipt Test",
            "-c",
            "user.email=receipt-test@example.invalid",
            "commit",
            "-qm",
            "move head",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = receipt_module.subprocess.run(
        [
            sys.executable,
            "-I",
            "-",
            "--launch-reviewed",
            "--reviewed-git-sha",
            old_sha,
            "scripts/import_eom_customers_live.py",
            "--dry-run",
            "--receipt-dir",
            str(receipt_dir),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        input=launcher_source,
    )

    assert result.returncode != 0
    assert "resolve the same Git SHA" in result.stderr
    assert list(receipt_dir.iterdir()) == []


def test_reviewed_launcher_rejects_write_without_receipt_before_import(
    tmp_path
):
    marker = tmp_path / "local-import-executed"
    extra_files = {
        "scripts/import_calendar_contacts.py": (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
        )
    }
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral", extra_files=extra_files
    )

    result = _run_receipted_calendar(
        repo,
        scripts,
        receipt_dir,
        dry_run=False,
        include_receipt=False,
    )

    assert result.returncode == 2
    assert "live writes require --receipt-dir" in result.stderr
    assert not marker.exists()
    assert list(receipt_dir.iterdir()) == []


def test_receipted_invalid_arguments_finalize_exit_2_before_runtime(
    tmp_path
):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )

    result = _run_receipted_calendar(
        repo,
        scripts,
        receipt_dir,
        extra_args=("--calendar", "invalid-calendar"),
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    final_path, payload = _load_only_final(receipt_dir)
    assert final_path.name.endswith(".exit-2.json")
    assert payload["exit_code"] == 2


def test_receipt_policy_rejects_abbreviated_protected_options_before_import(
    tmp_path
):
    marker = tmp_path / "local-import-executed"
    extra_files = {
        "scripts/import_calendar_contacts.py": (
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
        )
    }
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral", extra_files=extra_files
    )

    result = _run_receipted_calendar(
        repo,
        scripts,
        receipt_dir,
        dry_run=False,
        include_receipt=False,
        extra_args=("--dry-r", "--receipt-d", str(receipt_dir)),
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --dry-r" in result.stderr
    assert not marker.exists()
    assert list(receipt_dir.iterdir()) == []


def test_receipt_policy_rejects_duplicate_receipt_dir_before_receipt(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )
    second_receipt_dir = tmp_path / "other-receipts"
    second_receipt_dir.mkdir()
    second_receipt_dir.chmod(0o700)

    result = _run_receipted_calendar(
        repo,
        scripts,
        receipt_dir,
        extra_args=("--receipt-dir", str(second_receipt_dir)),
    )

    assert result.returncode == 2
    assert "--receipt-dir may be supplied only once" in result.stderr
    assert list(receipt_dir.iterdir()) == []
    assert list(second_receipt_dir.iterdir()) == []


def test_reviewed_launcher_write_mutates_and_finalizes_receipt(tmp_path):
    extra_files, mutation_marker, _ready, _release = (
        _write_process_fixture_files(tmp_path)
    )
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral", extra_files=extra_files
    )
    env = {
        **os.environ,
        "EOM_CALENDAR_RESIDENTIAL": "residential-fixture",
    }

    result = _run_receipted_calendar(
        repo,
        scripts,
        receipt_dir,
        dry_run=False,
        extra_args=("--calendar", "residential"),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert mutation_marker.read_text() == CONTACT_A
    final_path, payload = _load_only_final(receipt_dir)
    assert final_path.name.endswith(".exit-0.json")
    assert payload["git_sha"] == receipt_module._git_sha(repo)
    assert payload["outcome_counts"]["created"] == 1
    assert payload["changed_contact_ids"] == [CONTACT_A]


def test_reviewed_launcher_cleanup_failure_preserves_finalized_receipt(
    tmp_path
):
    extra_files, _mutation_marker, _ready, _release = (
        _write_process_fixture_files(tmp_path)
    )
    launcher_source = (SCRIPTS / "eom_execution_receipt.py").read_text()
    extra_files["scripts/eom_execution_receipt.py"] = launcher_source.replace(
        "def _remove_reviewed_python(snapshot_root: Path) -> None:\n"
        "    \"\"\"Restore owner permissions only long enough to remove the snapshot.\"\"\"\n",
        "def _remove_reviewed_python(snapshot_root: Path) -> None:\n"
        "    \"\"\"Restore owner permissions only long enough to remove the snapshot.\"\"\"\n"
        "    raise OSError('cleanup failed after finalization')\n",
        1,
    )
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral", extra_files=extra_files
    )
    process_tmp = tmp_path / "process-tmp"
    process_tmp.mkdir()
    env = {
        **os.environ,
        "EOM_CALENDAR_RESIDENTIAL": "residential-fixture",
        "TMPDIR": str(process_tmp),
    }

    result = _run_receipted_calendar(
        repo,
        scripts,
        receipt_dir,
        extra_args=("--calendar", "residential"),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "could not remove reviewed Python snapshot" in result.stderr
    final_path, payload = _load_only_final(receipt_dir)
    assert final_path.name.endswith(".exit-0.json")
    assert payload["exit_code"] == 0
    for snapshot in process_tmp.glob("atlas-eom-reviewed-python-*"):
        receipt_module._remove_reviewed_python(snapshot)


def test_reviewed_launcher_write_ignores_concurrent_worktree_rewrite(
    tmp_path
):
    extra_files, mutation_marker, ready_marker, release_marker = (
        _write_process_fixture_files(tmp_path, wait_for_rewrite=True)
    )
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral", extra_files=extra_files
    )
    reviewed_sha = receipt_module.subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    launcher_source = receipt_module.subprocess.run(
        ["git", "show", f"{reviewed_sha}:scripts/eom_execution_receipt.py"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untrusted_marker = tmp_path / "untrusted-dependency-executed"
    dependency = repo / "atlas_brain" / "services" / "crm_provider.py"
    env = {
        **os.environ,
        "EOM_CALENDAR_RESIDENTIAL": "residential-fixture",
    }
    command = [
        sys.executable,
        "-I",
        "-",
        "--launch-reviewed",
        "--reviewed-git-sha",
        reviewed_sha,
        "scripts/import_eom_customers_live.py",
        "--receipt-dir",
        str(receipt_dir),
        "--calendar",
        "residential",
    ]
    process = receipt_module.subprocess.Popen(
        command,
        cwd=repo,
        stdin=receipt_module.subprocess.PIPE,
        stdout=receipt_module.subprocess.PIPE,
        stderr=receipt_module.subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        process.stdin.write(launcher_source)
        process.stdin.close()
        process.stdin = None
        deadline = time.monotonic() + 10
        while not ready_marker.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                raise AssertionError("reviewed process did not reach Calendar fetch")
            time.sleep(0.01)
        assert ready_marker.exists()
        dependency.write_text(
            "from pathlib import Path\n"
            f"Path({str(untrusted_marker)!r}).write_text('executed')\n"
            "raise RuntimeError('mutable dependency executed')\n"
        )
        release_marker.write_text("continue")
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == 0, (stdout, stderr)
    assert mutation_marker.read_text() == CONTACT_A
    assert not untrusted_marker.exists()
    _final_path, payload = _load_only_final(receipt_dir)
    assert payload["git_sha"] == receipt_module._git_sha(repo)
    assert payload["changed_contact_ids"] == [CONTACT_A]


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


def test_reviewed_launcher_rejects_self_restoring_entrypoint(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )
    marker = tmp_path / "entrypoint-executed"
    entrypoint = scripts / "import_eom_customers_live.py"
    entrypoint.write_text(
        "from pathlib import Path\n"
        "import subprocess\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "subprocess.run(\n"
        "    ['git', 'checkout', '--', 'scripts/import_eom_customers_live.py'],\n"
        "    check=True,\n"
        ")\n"
    )

    result = _run_receipted_calendar(repo, scripts, receipt_dir)

    assert result.returncode != 0
    assert "requires a clean worktree" in result.stderr
    assert not marker.exists()
    assert "entrypoint-executed" in entrypoint.read_text()
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


@pytest.mark.parametrize(
    "module_name", ("shadow_module.py", "cached_shadow.pyc")
)
def test_process_preflight_rejects_ignored_module_file_symlink(
    tmp_path, module_name
):
    ignored_path = f"scripts/{module_name}"
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ignored_path
    )
    target = tmp_path / f"{module_name}.target"
    target.write_text("VALUE = 'unreviewed'\n")
    (scripts / module_name).symlink_to(target)

    result = _run_receipted_calendar(repo, scripts, receipt_dir)

    assert result.returncode != 0
    assert "rejects ignored Python import shadows" in result.stderr
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


def test_reviewed_launcher_rejects_git_replacement_refs(tmp_path):
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral"
    )
    subprocess_options = {
        "cwd": repo,
        "check": True,
        "capture_output": True,
        "text": True,
    }
    base_sha = receipt_module.subprocess.run(
        ["git", "rev-parse", "HEAD"], **subprocess_options
    ).stdout.strip()
    (repo / "replacement-note.txt").write_text("replacement\n")
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
            "replacement",
        ],
        **subprocess_options,
    )
    replacement_sha = receipt_module.subprocess.run(
        ["git", "rev-parse", "HEAD"], **subprocess_options
    ).stdout.strip()
    receipt_module.subprocess.run(
        ["git", "checkout", "-q", base_sha], **subprocess_options
    )
    receipt_module.subprocess.run(
        ["git", "replace", base_sha, replacement_sha], **subprocess_options
    )

    result = _run_receipted_calendar(repo, scripts, receipt_dir)

    assert result.returncode != 0
    assert "rejects Git replacement refs" in result.stderr
    assert list(receipt_dir.iterdir()) == []


def test_process_preflight_compares_skip_worktree_source_to_reviewed_revision(
    tmp_path
):
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
    assert "tracked Python source to match the reviewed Git revision" in result.stderr
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


def test_reviewed_launcher_authenticates_entrypoint_before_execution():
    launcher_source = (
        SCRIPTS / "eom_execution_receipt.py"
    ).read_text()
    calendar_source = (
        SCRIPTS / "import_eom_customers_live.py"
    ).read_text()

    source_preflight = launcher_source.index(
        "git_sha = establish_source_trust("
    )
    snapshot_materialization = launcher_source.index(
        "_materialize_reviewed_python(repo_root, git_sha, snapshot_root)"
    )
    reviewed_source_read = launcher_source.index(
        "source = source_path.read_bytes()"
    )
    entrypoint_execution = launcher_source.index(
        'exec(compile(source, str(source_path), "exec"), namespace)'
    )
    assert (
        source_preflight
        < snapshot_materialization
        < reviewed_source_read
        < entrypoint_execution
    )

    bootstrap_admission = calendar_source.index(
        '_receipt_module = sys.modules.get("eom_execution_receipt")'
    )
    assert bootstrap_admission < calendar_source.index(
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
    result = _run_receipted_calendar(
        repo, scripts, calendar_receipts, env=env
    )
    assert result.returncode == 0, result.stderr
    assert list(repo.rglob("*.pyc")) == []
    assert len(list(calendar_receipts.glob("*.exit-0.json"))) == 1


def test_reviewed_launcher_disables_configured_fsmonitor_during_preflight(
    tmp_path
):
    extra_files, _mutation_marker, _ready, _release = (
        _write_process_fixture_files(tmp_path)
    )
    repo, scripts, receipt_dir = _calendar_process_fixture(
        tmp_path, ".cache/neutral", extra_files=extra_files
    )
    fsmonitor_marker = tmp_path / "fsmonitor-executed"
    hook = tmp_path / "fsmonitor-hook.sh"
    hook.write_text(
        "#!/bin/sh\n"
        f"printf executed > {str(fsmonitor_marker)!r}\n"
        "exit 0\n"
    )
    hook.chmod(0o700)
    receipt_module.subprocess.run(
        ["git", "config", "core.fsmonitor", str(hook)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    env = {
        **os.environ,
        "EOM_CALENDAR_COMMERCIAL": "commercial-fixture",
        "EOM_CALENDAR_RESIDENTIAL": "residential-fixture",
        "EOM_CALENDAR_ONE_TIME": "one-time-fixture",
    }

    result = _run_receipted_calendar(
        repo, scripts, receipt_dir, env=env
    )

    assert result.returncode == 0, result.stderr
    assert not fsmonitor_marker.exists()


def test_failed_publication_sync_removes_new_final_link(tmp_path, monkeypatch):
    receipt = _receipt(tmp_path)
    final_path = receipt.final_path_for(0)
    sync_final_link_states = []

    def fail_directory_sync():
        sync_final_link_states.append(final_path.exists())
        if len(sync_final_link_states) == 1:
            raise OSError("storage sync failed")

    monkeypatch.setattr(receipt, "_fsync_directory", fail_directory_sync)
    with pytest.raises(OSError, match="storage sync failed"):
        receipt.finalize(0)

    assert sync_final_link_states == [True, False]
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
    monkeypatch.setattr(calendar_import, "import_one", record_one)

    with pytest.raises(RuntimeError, match="durably record"):
        asyncio.run(
            calendar_import.run_import(
                [_record(phone="2175550101"), _record(phone="2175550102")],
                dry_run=False,
                receipt=receipt,
                _dependencies=(object(), object()),
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
                "_was_updated": True,
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


def test_calendar_race_merge_does_not_record_unconfirmed_provider_update(
    monkeypatch,
):
    class RaceCRM(StubCRM):
        async def create_contact(self, data):
            self.created.append(data)
            return {
                "id": CONTACT_A,
                "_was_created": False,
                "_was_updated": False,
                **calendar_import.record_to_contact_data(_record()),
            }

    async def skipped(_pool, existing, _data):
        return str(existing["id"]), "skipped"

    monkeypatch.setattr(calendar_import, "_update_matched", skipped)
    recorder = _Recorder()
    outcome = asyncio.run(
        calendar_import.import_one(
            _record(),
            RaceCRM(),
            StubPool(rows=[None, None]),
            receipt=recorder,
        )
    )

    assert outcome == "skipped"
    assert recorder.contact_ids == []


def test_calendar_interaction_insert_is_recorded_before_cancellable_emit():
    rec = _record(phone="(217) 555-9999")
    existing = {
        "id": CONTACT_A,
        **calendar_import.record_to_contact_data(rec),
    }

    class CancelAfterInsertCRM(StubCRM):
        async def log_interaction(self, **kwargs):
            self.interactions.append(kwargs)
            kwargs["after_insert"]()
            raise KeyboardInterrupt()

    recorder = _Recorder()
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(
            calendar_import.import_one(
                rec,
                CancelAfterInsertCRM(scoped_hit=existing),
                StubPool(),
                receipt=recorder,
            )
        )

    assert recorder.contact_ids == [CONTACT_A]
