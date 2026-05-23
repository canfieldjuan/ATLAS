from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import extracted_content_pipeline.api.control_surfaces as api_module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "smoke_content_ops_ingestion_file_route.py"

pytestmark = pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("smoke_content_ops_ingestion_file_route", SCRIPT_PATH)
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
        "cfpb-route-test",
        "--min-source-rows",
        "3",
        "--default-field",
        "company_name=CFPB Public Archive",
        "--default-field",
        "vendor_name=CFPB",
        "--default-field",
        "contact_email=cfpb-public-archive@example.invalid",
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
    def __init__(self) -> None:
        self.executed = []

    def transaction(self):
        return _Transaction()

    async def execute(self, query, *args):
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


def test_file_route_smoke_writes_success_result_for_cfpb_rows(tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    _write_cfpb_rows(source_path, 3)

    code = module.main(_default_args(source_path, result_path))

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["ok"] is True
    assert payload["route"] == "/ops/ingestion/files/import"
    assert payload["diagnostics"]["opportunity_count"] == 3
    assert payload["diagnostics"]["warning_count"] == 0
    assert payload["diagnostics"]["source_type_counts"] == {"complaint": 3}
    assert payload["import"]["dry_run"] is True
    assert payload["import"]["inserted"] == 3
    assert payload["import"]["target_id_count"] == 3


def test_file_route_smoke_writes_failure_result_for_row_cap(tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    _write_cfpb_rows(source_path, 10001)

    code = module.main(_default_args(source_path, result_path))

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    assert payload["status_code"] == 413
    assert "max 10000 rows" in payload["detail"]


def test_file_route_smoke_requires_public_dataset_defaults(tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    _write_cfpb_rows(source_path, 1)

    with pytest.raises(SystemExit) as exc:
        module.main([
            str(source_path),
            "--source-format",
            "jsonl",
            "--default-field",
            "company_name=CFPB Public Archive",
        ])

    assert "vendor_name" in str(exc.value)
    assert "contact_email" in str(exc.value)


def test_file_route_smoke_write_mode_requires_account_id(tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    _write_cfpb_rows(source_path, 1)

    with pytest.raises(SystemExit) as exc:
        module.main([
            *_default_args(source_path, tmp_path / "result.json"),
            "--write",
            "--database-url",
            "postgresql://atlas@localhost:5433/atlas",
        ])

    assert "--account-id is required" in str(exc.value)


def test_file_route_smoke_write_mode_uses_pool_provider(monkeypatch, tmp_path: Path) -> None:
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
        "--write",
        "--account-id",
        "acct-route-write",
        "--user-id",
        "user-route-write",
        "--database-url",
        "postgresql://atlas@localhost:5433/atlas",
        "--replace-existing",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["ok"] is True
    assert payload["dry_run"] is False
    assert payload["account_id"] == "acct-route-write"
    assert payload["import"]["dry_run"] is False
    assert payload["import"]["inserted"] == 3
    assert payload["import"]["replace_existing"] is True
    assert pool.closed is True
    assert any("DELETE FROM" in query for query, _args in connection.executed)
    insert_args = [args for query, args in connection.executed if "INSERT INTO" in query]
    assert len(insert_args) == 3
    assert {args[0] for args in insert_args} == {"acct-route-write"}


def test_file_route_smoke_write_mode_writes_pool_failure(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    _write_cfpb_rows(source_path, 1)

    async def create_pool(database_url: str):
        del database_url
        raise RuntimeError("pool unavailable")

    monkeypatch.setattr(module, "_create_pool", create_pool)

    code = module.main([
        *_default_args(source_path, result_path),
        "--write",
        "--account-id",
        "acct-route-write",
        "--database-url",
        "postgresql://atlas@localhost:5433/atlas",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    assert payload["dry_run"] is False
    assert payload["status_code"] == 500
    assert payload["errors"] == ["RuntimeError: pool unavailable"]
