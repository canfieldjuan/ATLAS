from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_seeded_route_e2e.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_faq_search_seeded_route_e2e", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _args(**overrides):
    values = {
        "database_url": "postgresql://example/atlas",
        "base_url": "https://atlas.example.com",
        "token": "token-123",
        "account_id": "acct-1",
        "corpora_per_account": 2,
        "documents_per_corpus": 3,
        "seed_iterations": 12,
        "route_requests": 12,
        "concurrency": 4,
        "pool_size": 2,
        "route": "/api/v1/content-ops/faq-deflection-search",
        "timeout": 10.0,
        "max_error_rate": 0.0,
        "max_p95_ms": None,
        "max_single_request_ms": None,
        "artifact_dir": None,
        "output_result": None,
        "keep_data": False,
        "json": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_cases(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_args_reports_missing_required_fields_and_bad_numbers():
    errors = smoke._validate_args(
        _args(
            database_url="",
            base_url="",
            token="",
            account_id="",
            route_requests=0,
            timeout=0,
            max_error_rate=2,
            max_p95_ms=0,
        )
    )

    assert errors == [
        "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL",
        "ATLAS_API_BASE_URL or --base-url is required",
        "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required",
        "ATLAS_FAQ_SEARCH_ACCOUNT_ID or --account-id is required",
        "--route-requests must be positive",
        "--timeout must be positive",
        "--max-error-rate must be between 0 and 1",
        "--max-p95-ms must be positive",
    ]


def test_seed_command_keeps_data_and_writes_route_cases(tmp_path):
    args = _args(account_id="hosted-acct", artifact_dir=tmp_path)
    case_file = tmp_path / "cases.json"
    cleanup_manifest = tmp_path / "cleanup.json"
    seed_result = tmp_path / "seed.json"

    command = smoke._seed_command(
        args,
        case_file=case_file,
        cleanup_manifest=cleanup_manifest,
        seed_result=seed_result,
    )

    assert str(smoke.SEED_SCRIPT) in command
    assert "--keep-data" in command
    assert command[command.index("--account-id") + 1] == "hosted-acct"
    assert command[command.index("--route-case-file-output") + 1] == str(case_file)
    assert command[command.index("--cleanup-manifest-output") + 1] == str(cleanup_manifest)
    assert command[command.index("--output-result") + 1] == str(seed_result)


def test_route_command_uses_seeded_case_file_and_budgets(tmp_path):
    args = _args(max_p95_ms=50, max_single_request_ms=80)
    case_file = tmp_path / "cases.json"
    route_result = tmp_path / "route.json"

    command = smoke._route_command(args, case_file=case_file, route_result=route_result)

    assert str(smoke.ROUTE_SCRIPT) in command
    assert command[command.index("--case-file") + 1] == str(case_file)
    assert command[command.index("--max-p95-ms") + 1] == "50"
    assert command[command.index("--max-single-request-ms") + 1] == "80"


def test_faq_ids_from_cleanup_manifest_deduplicates_expected_ids(tmp_path):
    manifest = tmp_path / "cleanup.json"
    _write_cases(
        manifest,
        {
            "faq_ids": [
                "11111111-1111-1111-1111-111111111111",
                "11111111-1111-1111-1111-111111111111",
            ]
        },
    )

    faq_ids, errors = smoke._faq_ids_from_cleanup_manifest(manifest)

    assert errors == []
    assert faq_ids == ["11111111-1111-1111-1111-111111111111"]


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ("{bad json", "cleanup manifest must contain JSON: Expecting property name enclosed in double quotes"),
        ([], "cleanup manifest must contain a JSON object"),
        ({}, "cleanup manifest faq_ids must be a list"),
        (
            {"faq_ids": [""]},
            "cleanup manifest faq_ids[0] must be a non-empty string",
        ),
        (
            {"faq_ids": [1]},
            "cleanup manifest faq_ids[0] must be a non-empty string",
        ),
    ],
)
def test_faq_ids_from_cleanup_manifest_rejects_bad_shapes(tmp_path, payload, expected_error):
    manifest = tmp_path / "cleanup.json"
    if isinstance(payload, str):
        manifest.write_text(payload, encoding="utf-8")
    else:
        _write_cases(manifest, payload)

    faq_ids, errors = smoke._faq_ids_from_cleanup_manifest(manifest)

    assert faq_ids == []
    assert expected_error in errors


def test_faq_ids_from_cleanup_manifest_reports_unreadable_file():
    faq_ids, errors = smoke._faq_ids_from_cleanup_manifest(Path("/tmp/atlas-missing-e2e-cleanup.json"))

    assert faq_ids == []
    assert errors
    assert errors[0].startswith("cleanup manifest could not be read:")


@pytest.mark.parametrize(("delete_status", "expected"), [("DELETE 2", 2), (" DELETE 0 ", 0)])
def test_deleted_row_count_accepts_delete_tags(delete_status, expected):
    assert smoke._deleted_row_count(delete_status) == expected


@pytest.mark.parametrize(
    "delete_status",
    [
        "UPDATE 2",
        "DELETE",
        "DELETE two",
        "DELETE -1",
        "DELETE 1 extra",
        "",
        ["DELETE", 1],
        None,
    ],
)
def test_deleted_row_count_rejects_malformed_tags(delete_status):
    assert smoke._deleted_row_count(delete_status) is None


@pytest.mark.asyncio
async def test_cleanup_seeded_faqs_noops_without_ids():
    assert await smoke._cleanup_seeded_faqs("postgresql://example", []) == {
        "ok": True,
        "requested_faq_ids": 0,
        "deleted_faq_ids": 0,
        "delete_status": None,
        "error": None,
    }


@pytest.mark.asyncio
async def test_cleanup_seeded_faqs_reports_actual_delete_rowcount(monkeypatch):
    class FakePool:
        def __init__(self):
            self.closed = False
            self.deleted_ids = None

        async def execute(self, _query, faq_ids):
            self.deleted_ids = faq_ids
            return "DELETE 1"

        async def close(self):
            self.closed = True

    fake_pool = FakePool()

    async def _create_pool(**_kwargs):
        return fake_pool

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    payload = await smoke._cleanup_seeded_faqs(
        "postgresql://example",
        [
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
        ],
    )

    assert payload == {
        "ok": True,
        "requested_faq_ids": 2,
        "deleted_faq_ids": 1,
        "delete_status": "DELETE 1",
        "error": None,
    }
    assert fake_pool.deleted_ids == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    assert fake_pool.closed is True


@pytest.mark.asyncio
async def test_cleanup_seeded_faqs_surfaces_malformed_delete_status(monkeypatch):
    class FakePool:
        async def execute(self, _query, _faq_ids):
            return "UPDATE 1"

        async def close(self):
            return None

    async def _create_pool(**_kwargs):
        return FakePool()

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    payload = await smoke._cleanup_seeded_faqs(
        "postgresql://example",
        ["11111111-1111-1111-1111-111111111111"],
    )

    assert payload == {
        "ok": True,
        "requested_faq_ids": 1,
        "deleted_faq_ids": None,
        "delete_status": "UPDATE 1",
        "error": None,
    }


def test_main_writes_preflight_result(tmp_path, capsys):
    result_path = tmp_path / "result.json"

    code = smoke.main([
        "--database-url",
        "",
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--account-id",
        "acct-1",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["phase"] == "preflight"
    assert payload["preflight_errors"] == [
        "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL"
    ]
    assert json.loads(capsys.readouterr().out)["phase"] == "preflight"


def test_main_runs_seed_route_and_cleanup(tmp_path, monkeypatch):
    calls = []

    def _fake_run_command(command):
        calls.append(command)
        if str(smoke.SEED_SCRIPT) in command:
            cleanup_manifest = Path(command[command.index("--cleanup-manifest-output") + 1])
            _write_cases(
                cleanup_manifest,
                {"faq_ids": ["11111111-1111-1111-1111-111111111111"]},
            )
        return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}

    async def _fake_cleanup(_database_url, faq_ids):
        return {
            "ok": True,
            "requested_faq_ids": len(faq_ids),
            "deleted_faq_ids": len(faq_ids),
            "delete_status": f"DELETE {len(faq_ids)}",
            "error": None,
        }

    monkeypatch.setattr(smoke, "_run_command", _fake_run_command)
    monkeypatch.setattr(smoke, "_cleanup_seeded_faqs", _fake_cleanup)
    result_path = tmp_path / "result.json"

    code = smoke.main([
        "--database-url",
        "postgresql://example/atlas",
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--account-id",
        "acct-1",
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--output-result",
        str(result_path),
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["ok"] is True
    assert payload["cleanup"]["deleted_faq_ids"] == 1
    assert len(calls) == 2


def test_main_route_failure_still_cleans_up(tmp_path, monkeypatch):
    calls = []

    def _fake_run_command(command):
        calls.append(command)
        if str(smoke.SEED_SCRIPT) in command:
            cleanup_manifest = Path(command[command.index("--cleanup-manifest-output") + 1])
            _write_cases(
                cleanup_manifest,
                {"faq_ids": ["11111111-1111-1111-1111-111111111111"]},
            )
            return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}
        return {"ok": False, "returncode": 1, "stdout_tail": "", "stderr_tail": "bad route"}

    async def _fake_cleanup(_database_url, faq_ids):
        return {
            "ok": True,
            "requested_faq_ids": len(faq_ids),
            "deleted_faq_ids": len(faq_ids),
            "delete_status": f"DELETE {len(faq_ids)}",
            "error": None,
        }

    monkeypatch.setattr(smoke, "_run_command", _fake_run_command)
    monkeypatch.setattr(smoke, "_cleanup_seeded_faqs", _fake_cleanup)

    code = smoke.main([
        "--database-url",
        "postgresql://example/atlas",
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--account-id",
        "acct-1",
        "--artifact-dir",
        str(tmp_path / "artifacts"),
    ])

    assert code == 1
    assert len(calls) == 2


def test_main_reports_cleanup_failure(tmp_path, monkeypatch):
    def _fake_run_command(command):
        if str(smoke.SEED_SCRIPT) in command:
            cleanup_manifest = Path(command[command.index("--cleanup-manifest-output") + 1])
            _write_cases(
                cleanup_manifest,
                {"faq_ids": ["11111111-1111-1111-1111-111111111111"]},
            )
        return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}

    async def _fake_cleanup(_database_url, _faq_ids):
        return {
            "ok": False,
            "requested_faq_ids": 1,
            "deleted_faq_ids": 0,
            "delete_status": None,
            "error": "cleanup failed",
        }

    monkeypatch.setattr(smoke, "_run_command", _fake_run_command)
    monkeypatch.setattr(smoke, "_cleanup_seeded_faqs", _fake_cleanup)
    result_path = tmp_path / "result.json"

    code = smoke.main([
        "--database-url",
        "postgresql://example/atlas",
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--account-id",
        "acct-1",
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--output-result",
        str(result_path),
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["cleanup"] == {
        "ok": False,
        "requested_faq_ids": 1,
        "deleted_faq_ids": 0,
        "delete_status": None,
        "error": "cleanup failed",
    }
