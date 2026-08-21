from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_migration_content_integrity.py"
SPEC = importlib.util.spec_from_file_location("check_migration_content_integrity", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FakeReadOnlyTransaction:
    def __init__(self, connection: "FakeConnection", readonly: bool):
        self.connection = connection
        self.readonly = readonly

    async def __aenter__(self) -> "FakeReadOnlyTransaction":
        self.connection.transaction_readonly.append(self.readonly)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeConnection:
    def __init__(self, records: list[tuple[str, str | None]]):
        self.records = records
        self.queries: list[str] = []
        self.transaction_readonly: list[bool] = []
        self.execute_calls: list[str] = []
        self.closed = False

    def transaction(self, *, readonly: bool = False) -> FakeReadOnlyTransaction:
        return FakeReadOnlyTransaction(self, readonly)

    async def fetch(self, query: str):
        self.queries.append(query)
        assert query == "SELECT name, content_sha256 FROM schema_migrations"
        return [
            {"name": name, "content_sha256": content_sha256}
            for name, content_sha256 in self.records
        ]

    async def execute(self, query: str, *args) -> None:
        self.execute_calls.append(query)
        raise AssertionError("read-only provenance preflight must not execute SQL")

    async def close(self) -> None:
        self.closed = True


def _write_migration(directory: Path, name: str, content: bytes) -> Path:
    path = directory / f"{name}.sql"
    path.write_bytes(content)
    return path


@pytest.mark.asyncio
async def test_preflight_reports_unresolved_drift_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verified = _write_migration(tmp_path, "900_verified", b"SELECT 'verified';\n")
    _write_migration(tmp_path, "901_legacy", b"SELECT 'legacy';\n")
    _write_migration(tmp_path, "902_mismatched", b"SELECT 'mismatched';\n")
    connection = FakeConnection([
        ("900_verified", hashlib.sha256(verified.read_bytes()).hexdigest()),
        ("901_legacy", None),
        ("902_mismatched", "not-the-current-digest"),
        ("903_missing_source", "f" * 64),
    ])

    async def connect_read_only() -> FakeConnection:
        return connection

    monkeypatch.setattr(module, "_connect_read_only", connect_read_only)

    code = await module._main(migrations_dir=tmp_path)

    payload = json.loads(capsys.readouterr().out)
    assert code == module.UNRESOLVED_DRIFT_EXIT
    assert payload == {
        "check_completed": True,
        "counts": {
            "legacy_unverified": 1,
            "mismatched": 1,
            "missing_source": 1,
            "verified": 1,
        },
        "database_target": module.db_settings.target_label,
        "exit_code": module.UNRESOLVED_DRIFT_EXIT,
        "report": {
            "legacy_unverified": ["901_legacy"],
            "mismatched": ["902_mismatched"],
            "missing_source": ["903_missing_source"],
            "verified": ["900_verified"],
        },
        "status": "unresolved_drift",
    }
    assert connection.queries == ["SELECT name, content_sha256 FROM schema_migrations"]
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []
    assert connection.closed is True


@pytest.mark.asyncio
async def test_preflight_keeps_legacy_evidence_visible_without_treating_it_as_drift(
    tmp_path: Path,
) -> None:
    _write_migration(tmp_path, "901_legacy", b"SELECT 'legacy';\n")
    connection = FakeConnection([("901_legacy", None)])

    code, payload = await module.run_migration_content_integrity_preflight(
        connection,
        migrations_dir=tmp_path,
    )

    assert code == 0
    assert payload["status"] == "legacy_unverified"
    assert payload["report"] == {
        "verified": [],
        "legacy_unverified": ["901_legacy"],
        "mismatched": [],
        "missing_source": [],
    }
    assert connection.transaction_readonly == [True]
    assert connection.execute_calls == []


@pytest.mark.asyncio
async def test_main_redacts_database_failure_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def connect_read_only() -> FakeConnection:
        raise RuntimeError("connection details must not appear in preflight output")

    monkeypatch.setattr(module, "_connect_read_only", connect_read_only)

    code = await module._main()

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert code == module.COULD_NOT_DETERMINE_EXIT
    assert payload == {
        "check_completed": False,
        "database_target": module.db_settings.target_label,
        "error_type": "RuntimeError",
        "exit_code": module.COULD_NOT_DETERMINE_EXIT,
        "status": "could_not_determine",
    }
    assert "connection details" not in output


@pytest.mark.asyncio
async def test_connection_defaults_to_read_only_without_printing_connection_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}
    sentinel = object()

    async def connect(**kwargs):
        captured_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setitem(sys.modules, "asyncpg", types.SimpleNamespace(connect=connect))

    connection = await module._connect_read_only()

    assert connection is sentinel
    server_settings = captured_kwargs["server_settings"]
    assert isinstance(server_settings, dict)
    assert server_settings["default_transaction_read_only"] == "on"


def test_main_displays_the_safe_target_without_opening_a_connection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def should_not_connect() -> FakeConnection:
        raise AssertionError("--show-target must not connect")

    monkeypatch.setattr(module, "_connect_read_only", should_not_connect)

    code = module.main(["--show-target"])

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "database_target": module.db_settings.target_label,
        "status": "target_displayed",
    }


@pytest.mark.parametrize(
    ("argv", "status"),
    [
        ([], "target_confirmation_required"),
        (["--expected-target", "other-safe-target"], "target_confirmation_mismatch"),
    ],
)
def test_main_rejects_unconfirmed_or_mismatched_target_before_connection(
    argv: list[str],
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def should_not_connect() -> FakeConnection:
        raise AssertionError("target admission must happen before connection")

    monkeypatch.setattr(module, "_connect_read_only", should_not_connect)

    code = module.main(argv)

    assert code == module.COULD_NOT_DETERMINE_EXIT
    assert json.loads(capsys.readouterr().out) == {
        "check_completed": False,
        "database_target": module.db_settings.target_label,
        "exit_code": module.COULD_NOT_DETERMINE_EXIT,
        "status": status,
    }


def test_main_passes_a_matching_target_to_the_async_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    async def fake_main(*, database_target: str) -> int:
        observed.append(database_target)
        return 0

    monkeypatch.setattr(module, "_main", fake_main)

    code = module.main(["--expected-target", module.db_settings.target_label])

    assert code == 0
    assert observed == [module.db_settings.target_label]
