from __future__ import annotations

import importlib.util
import json
from pathlib import Path
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
    seed_result = tmp_path / "seed.json"

    command = smoke._seed_command(args, case_file=case_file, seed_result=seed_result)

    assert str(smoke.SEED_SCRIPT) in command
    assert "--keep-data" in command
    assert command[command.index("--account-id") + 1] == "hosted-acct"
    assert command[command.index("--route-case-file-output") + 1] == str(case_file)
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


def test_faq_ids_from_case_file_deduplicates_expected_ids(tmp_path):
    case_file = tmp_path / "cases.json"
    _write_cases(
        case_file,
        [
            {"expected_first_faq_id": "11111111-1111-1111-1111-111111111111"},
            {"expected_first_faq_id": "11111111-1111-1111-1111-111111111111"},
            {"query": "miss case"},
        ],
    )

    faq_ids, errors = smoke._faq_ids_from_case_file(case_file)

    assert errors == []
    assert faq_ids == ["11111111-1111-1111-1111-111111111111"]


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ("{bad json", "case file must contain JSON: Expecting property name enclosed in double quotes"),
        ({}, "case file must contain a JSON list"),
        ([[]], "case[0] must be an object"),
        (
            [{"expected_first_faq_id": ""}],
            "case[0].expected_first_faq_id must be a non-empty string",
        ),
        (
            [{"expected_first_faq_id": 1}],
            "case[0].expected_first_faq_id must be a non-empty string",
        ),
    ],
)
def test_faq_ids_from_case_file_rejects_bad_shapes(tmp_path, payload, expected_error):
    case_file = tmp_path / "cases.json"
    if isinstance(payload, str):
        case_file.write_text(payload, encoding="utf-8")
    else:
        _write_cases(case_file, payload)

    faq_ids, errors = smoke._faq_ids_from_case_file(case_file)

    assert faq_ids == []
    assert expected_error in errors


def test_faq_ids_from_case_file_reports_unreadable_file():
    faq_ids, errors = smoke._faq_ids_from_case_file(Path("/tmp/atlas-missing-e2e-cases.json"))

    assert faq_ids == []
    assert errors
    assert errors[0].startswith("case file could not be read:")


@pytest.mark.asyncio
async def test_cleanup_seeded_faqs_noops_without_ids():
    assert await smoke._cleanup_seeded_faqs("postgresql://example", []) == {
        "ok": True,
        "deleted_faq_ids": 0,
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
            case_file = Path(command[command.index("--route-case-file-output") + 1])
            _write_cases(
                case_file,
                [{"expected_first_faq_id": "11111111-1111-1111-1111-111111111111"}],
            )
        return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}

    async def _fake_cleanup(_database_url, faq_ids):
        return {"ok": True, "deleted_faq_ids": len(faq_ids), "error": None}

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
            case_file = Path(command[command.index("--route-case-file-output") + 1])
            _write_cases(
                case_file,
                [{"expected_first_faq_id": "11111111-1111-1111-1111-111111111111"}],
            )
            return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}
        return {"ok": False, "returncode": 1, "stdout_tail": "", "stderr_tail": "bad route"}

    async def _fake_cleanup(_database_url, faq_ids):
        return {"ok": True, "deleted_faq_ids": len(faq_ids), "error": None}

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
            case_file = Path(command[command.index("--route-case-file-output") + 1])
            _write_cases(
                case_file,
                [{"expected_first_faq_id": "11111111-1111-1111-1111-111111111111"}],
            )
        return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}

    async def _fake_cleanup(_database_url, _faq_ids):
        return {"ok": False, "deleted_faq_ids": 0, "error": "cleanup failed"}

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
        "deleted_faq_ids": 0,
        "error": "cleanup failed",
    }
