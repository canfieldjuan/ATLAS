from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

import extracted_content_pipeline.api.control_surfaces as api_module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "smoke_content_ops_ingestion_file_route_inprocess_load.py"

pytestmark = pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "smoke_content_ops_ingestion_file_route_inprocess_load",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cfpb_rows(path: Path, row_count: int) -> None:
    rows = (
        {
            "complaint_id": f"cfpb-{index}",
            "product": "Credit reporting or other personal consumer reports",
            "issue": "Incorrect information on your report",
            "consumer_complaint_narrative": (
                "My credit report still shows an account I already disputed "
                "and I need help understanding the next step."
            ),
        }
        for index in range(row_count)
    )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _default_args(path: Path, result_path: Path) -> list[str]:
    return [
        str(path),
        "--source-format",
        "jsonl",
        "--source",
        "cfpb-route-inprocess-load-test",
        "--min-source-rows",
        "3",
        "--default-field",
        "company_name=CFPB Public Archive",
        "--default-field",
        "vendor_name=CFPB",
        "--default-field",
        "contact_email=cfpb-public-archive@example.invalid",
        "--account-id",
        "acct-route-inprocess-load",
        "--database-url",
        "postgresql://atlas@localhost:5433/atlas",
        "--output-result",
        str(result_path),
    ]


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _Connection:
    def __init__(self, delay_seconds: float = 0.02) -> None:
        self.delay_seconds = delay_seconds
        self.executed = []

    def transaction(self):
        return _Transaction()

    async def execute(self, query, *args):
        await asyncio.sleep(self.delay_seconds)
        self.executed.append((str(query), args))
        return "EXECUTE"


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        self.closed = False
        self.is_initialized = True

    def acquire(self):
        return _Acquire(self.connection)

    async def close(self):
        self.closed = True


class _SharedAdmissionGate:
    max_concurrency = 1

    def __init__(self) -> None:
        self.active = 0
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire(self) -> bool:
        self.acquire_calls += 1
        if self.active >= 1:
            return False
        self.active += 1
        return True

    async def release(self) -> None:
        self.release_calls += 1
        self.active = 0


def test_inprocess_load_runner_writes_admission_gate_result(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    _write_cfpb_rows(source_path, 3)
    connection = _Connection()
    pool = _Pool(connection)

    async def create_pool(database_url: str):
        assert database_url == "postgresql://atlas@localhost:5433/atlas"
        return pool

    monkeypatch.setattr(module, "_create_pool", create_pool)

    code = module.main([
        *_default_args(source_path, result_path),
        "--concurrency",
        "3",
        "--import-max-concurrency",
        "1",
        "--min-successes",
        "1",
        "--expect-at-capacity-min",
        "1",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["ok"] is True
    assert payload["concurrency"] == 3
    assert payload["import_max_concurrency"] == 1
    assert payload["summary"]["successes"] == 1
    assert payload["summary"]["at_capacity"] == 2
    assert payload["summary"]["unexpected_failures"] == 0
    assert payload["summary"]["inserted"] == 3
    assert payload["summary"]["status_counts"] == {"200": 1, "429": 2}
    assert payload["summary"]["reason_counts"] == {
        "content_ops_ingestion_import_at_capacity": 2
    }
    assert pool.closed is True
    insert_args = [args for query, args in connection.executed if "INSERT INTO" in query]
    assert len(insert_args) == 3
    assert {args[0] for args in insert_args} == {"acct-route-inprocess-load"}


def test_inprocess_load_runner_can_use_postgres_host_admission_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    _write_cfpb_rows(source_path, 3)
    pool = _Pool(_Connection())
    gate = _SharedAdmissionGate()
    builder_calls: list[dict[str, object]] = []

    async def create_pool(database_url: str):
        del database_url
        return pool

    def build_gate(*, max_concurrency, pool_provider):
        builder_calls.append({
            "max_concurrency": max_concurrency,
            "pool": pool_provider(),
        })
        return gate

    monkeypatch.setattr(module, "_create_pool", create_pool)
    monkeypatch.setattr(module, "build_content_ops_import_admission_gate", build_gate)

    code = module.main([
        *_default_args(source_path, result_path),
        "--admission-provider",
        "postgres",
        "--concurrency",
        "3",
        "--import-max-concurrency",
        "1",
        "--min-successes",
        "1",
        "--expect-at-capacity-min",
        "1",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["ok"] is True
    assert payload["admission_provider"] == "postgres"
    assert builder_calls == [
        {"max_concurrency": 1, "pool": pool},
        {"max_concurrency": 1, "pool": pool},
        {"max_concurrency": 1, "pool": pool},
    ]
    assert gate.acquire_calls == 3
    assert gate.release_calls == 1
    assert payload["summary"]["successes"] == 1
    assert payload["summary"]["at_capacity"] == 2
    assert payload["summary"]["reason_counts"] == {
        "content_ops_ingestion_import_at_capacity": 2
    }


def test_inprocess_load_runner_fails_when_expected_429_missing(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    _write_cfpb_rows(source_path, 1)
    pool = _Pool(_Connection(delay_seconds=0))

    async def create_pool(database_url: str):
        del database_url
        return pool

    monkeypatch.setattr(module, "_create_pool", create_pool)

    code = module.main([
        *_default_args(source_path, result_path),
        "--min-source-rows",
        "1",
        "--concurrency",
        "1",
        "--import-max-concurrency",
        "1",
        "--min-successes",
        "1",
        "--expect-at-capacity-min",
        "1",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    assert payload["summary"]["successes"] == 1
    assert payload["summary"]["at_capacity"] == 0
    assert payload["errors"] == [
        "expected at least 1 admission 429 response(s), got 0"
    ]


def test_inprocess_load_runner_fails_when_inserted_rows_below_minimum(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    _write_cfpb_rows(source_path, 1)
    pool = _Pool(_Connection(delay_seconds=0))

    async def create_pool(database_url: str):
        del database_url
        return pool

    monkeypatch.setattr(module, "_create_pool", create_pool)

    code = module.main([
        *_default_args(source_path, result_path),
        "--concurrency",
        "1",
        "--import-max-concurrency",
        "1",
        "--min-successes",
        "1",
        "--expect-at-capacity-min",
        "0",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    assert payload["min_source_rows"] == 3
    assert payload["summary"]["successes"] == 1
    assert payload["summary"]["inserted"] == 1
    assert payload["errors"] == [
        "expected at least 3 inserted source row(s), got 1"
    ]


def test_inprocess_load_runner_requires_database_url(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    _write_cfpb_rows(source_path, 1)
    monkeypatch.setattr(module, "_default_database_url", lambda: None)

    with pytest.raises(SystemExit) as exc:
        module.main([
            str(source_path),
            "--source-format",
            "jsonl",
            "--default-field",
            "company_name=CFPB Public Archive",
            "--default-field",
            "vendor_name=CFPB",
            "--default-field",
            "contact_email=cfpb-public-archive@example.invalid",
            "--account-id",
            "acct-route-inprocess-load",
            "--database-url",
            "",
        ])

    assert "Missing --database-url" in str(exc.value)
