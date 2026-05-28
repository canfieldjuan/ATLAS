from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_seeded_route_e2e.py"
RUNBOOK = ROOT / "docs/extraction/validation/content_ops_faq_seeded_route_e2e_runbook.md"
HOST_RUNBOOK = ROOT / "extracted_content_pipeline/docs/host_install_runbook.md"
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
        "max_case_error_rate": None,
        "max_case_p95_ms": None,
        "max_case_single_request_ms": None,
        "detail_route": "",
        "artifact_dir": None,
        "output_result": None,
        "keep_data": False,
        "skip_detail_check": False,
        "json": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_cases(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _runbook_seeded_e2e_command_args() -> list[str]:
    doc = RUNBOOK.read_text(encoding="utf-8")
    marker = "python scripts/smoke_content_ops_faq_search_seeded_route_e2e.py"
    start = doc.index(marker)
    end = doc.index("```", start)
    command = doc[start:end].replace("\\\n", " ")
    parts = shlex.split(command)
    assert parts[:2] == [
        "python",
        "scripts/smoke_content_ops_faq_search_seeded_route_e2e.py",
    ]
    return parts[2:]


def _write_child_result(command, *, ok: bool = True) -> None:
    if "--output-result" not in command:
        return
    result_path = Path(command[command.index("--output-result") + 1])
    if str(smoke.SEED_SCRIPT) in command:
        _write_cases(
            result_path,
            {
                "ok": ok,
                "run_id": "seed-run-1",
                "requests": {"total": 12, "iterations": 12, "concurrency": 4},
                "seed": {
                    "accounts": 1,
                    "corpora_per_account": 2,
                    "documents_per_corpus": 3,
                    "search_cases": 2,
                },
                "setup": {"ok": ok, "phase": "complete"},
                "cleanup": {"ok": True, "attempted": False, "error": None},
                "pool_close": {"ok": True, "attempted": True, "error": None},
                "latency": {"count": 12, "p50_ms": 1.0, "p95_ms": 2.0, "max_ms": 3.0},
                "latency_budget": {"ok": True, "checks": [], "failures": []},
                "isolation": {"count": 0, "items": [], "truncated": False},
                "elapsed_seconds": 1.25,
            },
        )
        return
    if str(smoke.ROUTE_SCRIPT) in command:
        _write_cases(
            result_path,
            {
                "ok": ok,
                "phase": "complete",
                "requests": {"total": 12, "configured": 12, "concurrency": 4},
                "latency": {"count": 12, "p50_ms": 4.0, "p95_ms": 5.0, "max_ms": 6.0},
                "errors": {
                    "count": 0 if ok else 1,
                    "rate": 0.0 if ok else 0.083333,
                    "items": [] if ok else [{"index": 0, "errors": ["bad route"]}],
                    "truncated": False,
                },
                "budgets": {"ok": ok, "checks": [], "failures": [] if ok else ["error_rate exceeded 0.0"]},
                "cases": {
                    "total": 1,
                    "case_file": "route-cases.json",
                    "items": [],
                    "truncated": False,
                },
                "preflight_errors": [],
                "elapsed_seconds": 0.75,
            },
        )
        return
    if str(smoke.CONTRACT_SCRIPT) in command:
        _write_cases(
            result_path,
            {
                "ok": ok,
                "phase": "complete",
                "count": 1 if ok else 0,
                "detail_checked": ok,
                "detail_faq_id": "11111111-1111-1111-1111-111111111111" if ok else "",
                "search_elapsed_ms": 7.0,
                "detail_elapsed_ms": 8.0,
                "total_elapsed_ms": 15.0,
                "errors": [] if ok else ["detail failed"],
            },
        )


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


def test_seeded_route_e2e_runbook_command_matches_parser():
    parsed = smoke._build_parser().parse_args(_runbook_seeded_e2e_command_args())

    assert parsed.database_url == "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}"
    assert parsed.base_url == "$ATLAS_API_BASE_URL"
    assert parsed.token == "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}"
    assert parsed.account_id == "$ATLAS_FAQ_SEARCH_ACCOUNT_ID"
    assert parsed.corpora_per_account == 2
    assert parsed.documents_per_corpus == 3
    assert parsed.seed_iterations == 12
    assert parsed.route_requests == 40
    assert parsed.concurrency == 8
    assert parsed.pool_size == 2
    assert parsed.max_error_rate == 0.0
    assert parsed.max_case_error_rate == 0.0
    assert parsed.max_p95_ms == 1500.0
    assert parsed.max_single_request_ms == 3000.0
    assert parsed.max_case_p95_ms == 1500.0
    assert parsed.max_case_single_request_ms == 3000.0
    assert str(parsed.output_result) == "/tmp/faq-search-seeded-route-e2e-result.json"


def test_host_runbook_links_seeded_route_e2e_validation_runbook():
    doc = HOST_RUNBOOK.read_text(encoding="utf-8")
    relative_path = "docs/extraction/validation/content_ops_faq_seeded_route_e2e_runbook.md"

    assert relative_path in doc
    assert RUNBOOK.exists()


def test_validate_args_rejects_invalid_case_budgets():
    assert smoke._validate_args(_args(max_case_error_rate=-0.1)) == [
        "--max-case-error-rate must be between 0 and 1"
    ]
    assert smoke._validate_args(_args(max_case_error_rate=1.1)) == [
        "--max-case-error-rate must be between 0 and 1"
    ]
    assert smoke._validate_args(_args(max_case_p95_ms=0.0)) == [
        "--max-case-p95-ms must be positive"
    ]
    assert smoke._validate_args(_args(max_case_single_request_ms=-1.0)) == [
        "--max-case-single-request-ms must be positive"
    ]
    assert smoke._validate_args(_args(max_case_p95_ms=float("inf"))) == [
        "--max-case-p95-ms must be finite"
    ]
    assert smoke._validate_args(_args(max_p95_ms=float("inf"))) == [
        "--max-p95-ms must be finite"
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
    args = _args(
        max_p95_ms=50,
        max_single_request_ms=80,
        max_case_error_rate=0.25,
        max_case_p95_ms=25,
        max_case_single_request_ms=40,
    )
    case_file = tmp_path / "cases.json"
    route_result = tmp_path / "route.json"

    command = smoke._route_command(args, case_file=case_file, route_result=route_result)

    assert str(smoke.ROUTE_SCRIPT) in command
    assert command[command.index("--case-file") + 1] == str(case_file)
    assert command[command.index("--max-p95-ms") + 1] == "50"
    assert command[command.index("--max-single-request-ms") + 1] == "80"
    assert command[command.index("--max-case-error-rate") + 1] == "0.25"
    assert command[command.index("--max-case-p95-ms") + 1] == "25"
    assert command[command.index("--max-case-single-request-ms") + 1] == "40"


def test_parser_accepts_case_budgets():
    parsed = smoke._build_parser().parse_args(
        [
            "--database-url",
            "postgresql://example/atlas",
            "--base-url",
            "https://atlas.example.com",
            "--token",
            "token-123",
            "--account-id",
            "acct-1",
            "--max-case-error-rate",
            "0",
            "--max-case-p95-ms",
            "1500",
            "--max-case-single-request-ms",
            "3000",
        ]
    )

    assert parsed.max_case_error_rate == 0.0
    assert parsed.max_case_p95_ms == 1500.0
    assert parsed.max_case_single_request_ms == 3000.0


def test_detail_case_from_route_cases_selects_first_hit_case(tmp_path):
    case_file = tmp_path / "route-cases.json"
    _write_cases(
        case_file,
        [
            {"query": "saml domain verification", "limit": 5, "require_results": False},
            {
                "query": "export attribution report",
                "corpus_id": "corp-1",
                "status": "approved",
                "limit": 3,
                "require_results": True,
                "expected_first_account_id": "acct-1",
                "expected_detail_account_id": "acct-1",
                "expected_detail_target_id": "support-corp-1",
                "expected_detail_target_mode": "support_account",
                "expected_detail_title": "FAQ Search Smoke",
                "expected_detail_status": "approved",
            },
        ],
    )

    detail_case, errors = smoke._detail_case_from_route_cases(case_file)

    assert errors == []
    assert detail_case == {
        "query": "export attribution report",
        "corpus_id": "corp-1",
        "status": "approved",
        "limit": 3,
        "expected_detail_account_id": "acct-1",
        "expected_detail_target_id": "support-corp-1",
        "expected_detail_target_mode": "support_account",
        "expected_detail_title": "FAQ Search Smoke",
        "expected_detail_status": "approved",
    }


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ("{bad json", "route case file must contain JSON: Expecting property name enclosed in double quotes"),
        ({}, "route case file must contain a non-empty JSON list"),
        ([], "route case file must contain a non-empty JSON list"),
        ([[]], "route case[0] must be an object"),
        ([{"query": "export", "limit": 5, "require_results": "yes"}], "route case[0].require_results must be a boolean"),
        ([{"query": "workspace ownership", "limit": 5, "require_results": False}], "route case file must include a require_results case for detail check"),
        ([{"query": "", "limit": 5, "require_results": True, "expected_first_account_id": "acct-1"}], "route case[0].query must be a non-empty string"),
        ([{"query": "export", "limit": 5, "require_results": True}], "route case[0].expected_first_account_id must be a non-empty string"),
        ([{"query": "export", "limit": 5, "require_results": True, "expected_first_account_id": ""}], "route case[0].expected_first_account_id must be a non-empty string"),
        ([{"query": "export", "corpus_id": 1, "limit": 5, "require_results": True, "expected_first_account_id": "acct-1"}], "route case[0].corpus_id must be a string"),
        ([{"query": "export", "status": 1, "limit": 5, "require_results": True, "expected_first_account_id": "acct-1"}], "route case[0].status must be a string"),
        ([{"query": "export", "limit": "5", "require_results": True, "expected_first_account_id": "acct-1"}], "route case[0].limit must be a positive integer"),
        ([{"query": "export", "limit": True, "require_results": True, "expected_first_account_id": "acct-1"}], "route case[0].limit must be a positive integer"),
        ([{"query": "export", "limit": 5, "require_results": True, "expected_first_account_id": "acct-1", "expected_detail_account_id": ""}], "route case[0].expected_detail_account_id must be a non-empty string"),
        ([{"query": "export", "limit": 5, "require_results": True, "expected_first_account_id": "acct-1", "expected_detail_account_id": "acct-1", "expected_detail_target_id": 1}], "route case[0].expected_detail_target_id must be a non-empty string"),
    ],
)
def test_detail_case_from_route_cases_rejects_bad_shapes(tmp_path, payload, expected_error):
    case_file = tmp_path / "route-cases.json"
    if isinstance(payload, str):
        case_file.write_text(payload, encoding="utf-8")
    else:
        _write_cases(case_file, payload)

    detail_case, errors = smoke._detail_case_from_route_cases(case_file)

    assert detail_case is None
    assert expected_error in errors


def test_detail_case_from_route_cases_reports_unreadable_file():
    detail_case, errors = smoke._detail_case_from_route_cases(Path("/tmp/atlas-missing-route-cases.json"))

    assert detail_case is None
    assert errors
    assert errors[0].startswith("route case file could not be read:")


def test_detail_command_uses_contract_checker_and_seeded_case(tmp_path):
    args = _args(detail_route="/api/v2/faqs/{faq_id}/full")
    detail_result = tmp_path / "detail.json"

    command = smoke._detail_command(
        args,
        detail_case={
            "query": "export attribution report",
            "corpus_id": "corp-1",
            "status": "approved",
            "limit": 3,
            "expected_detail_account_id": "acct-1",
            "expected_detail_target_id": "support-corp-1",
            "expected_detail_target_mode": "support_account",
            "expected_detail_title": "FAQ Search Smoke",
            "expected_detail_status": "approved",
        },
        detail_result=detail_result,
    )

    assert str(smoke.CONTRACT_SCRIPT) in command
    assert command[command.index("--query") + 1] == "export attribution report"
    assert command[command.index("--corpus-id") + 1] == "corp-1"
    assert command[command.index("--status") + 1] == "approved"
    assert command[command.index("--limit") + 1] == "3"
    assert "--require-results" in command
    assert "--require-detail" in command
    assert command[command.index("--detail-route") + 1] == "/api/v2/faqs/{faq_id}/full"
    assert command[command.index("--expected-detail-account-id") + 1] == "acct-1"
    assert command[command.index("--expected-detail-target-id") + 1] == "support-corp-1"
    assert command[command.index("--expected-detail-target-mode") + 1] == "support_account"
    assert command[command.index("--expected-detail-title") + 1] == "FAQ Search Smoke"
    assert command[command.index("--expected-detail-status") + 1] == "approved"
    assert command[command.index("--output-result") + 1] == str(detail_result)


def test_compact_child_result_artifact_summarizes_route_without_full_case_items(tmp_path):
    artifact = tmp_path / "route-result.json"
    _write_cases(
        artifact,
        {
            "ok": True,
            "phase": "complete",
            "requests": {"total": 4, "configured": 4, "concurrency": 2},
            "latency": {"count": 4, "p50_ms": 1.0, "p95_ms": 2.0, "max_ms": 3.0},
            "errors": {
                "count": 6,
                "rate": 0.5,
                "items": [{"index": index, "errors": [f"failure {index}"]} for index in range(6)],
                "truncated": False,
            },
            "budgets": {"ok": False, "failures": ["error_rate exceeded 0.0"]},
            "cases": {
                "total": 2,
                "case_file": "route-cases.json",
                "items": [{"query": "one"}, {"query": "two"}],
                "truncated": False,
            },
            "elapsed_seconds": 0.5,
        },
    )

    payload = smoke._compact_child_result_artifact(artifact, kind="route")

    assert payload["ok"] is True
    assert payload["available"] is True
    assert payload["requests"]["total"] == 4
    assert payload["errors"]["count"] == 6
    assert len(payload["errors"]["items"]) == 5
    assert payload["errors"]["truncated"] is True
    assert payload["cases"] == {
        "total": 2,
        "case_file": "route-cases.json",
        "truncated": False,
    }


def test_compact_child_result_artifact_rejects_non_boolean_ok(tmp_path):
    artifact = tmp_path / "route-result.json"
    _write_cases(
        artifact,
        {
            "ok": "false",
            "phase": "complete",
            "requests": {"total": 1, "configured": 1, "concurrency": 1},
            "errors": {"count": 0, "rate": 0.0, "items": [], "truncated": False},
        },
    )

    payload = smoke._compact_child_result_artifact(artifact, kind="route")

    assert payload["ok"] is False
    assert payload["available"] is True
    assert payload["artifact_errors"] == ["result artifact ok must be a boolean"]


@pytest.mark.parametrize(
    ("contents", "expected_error"),
    [
        ("{bad json", "result artifact must contain JSON"),
        ("[]", "result artifact must contain a JSON object"),
    ],
)
def test_compact_child_result_artifact_rejects_malformed_artifacts(
    tmp_path,
    contents,
    expected_error,
):
    artifact = tmp_path / "child-result.json"
    artifact.write_text(contents, encoding="utf-8")

    payload = smoke._compact_child_result_artifact(artifact, kind="route")

    assert payload["ok"] is False
    assert payload["available"] is False
    assert payload["path"] == str(artifact)
    assert payload["errors"][0].startswith(expected_error)


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
        "errors": [],
    }


@pytest.mark.asyncio
async def test_cleanup_seeded_faqs_reports_actual_delete_rowcount(monkeypatch):
    class FakePool:
        def __init__(self):
            self.closed = False
            self.deleted_ids = None

        async def execute(self, _query, faq_ids):
            self.deleted_ids = faq_ids
            return "DELETE 2"

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
        "deleted_faq_ids": 2,
        "delete_status": "DELETE 2",
        "errors": [],
    }
    assert fake_pool.deleted_ids == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    assert fake_pool.closed is True


@pytest.mark.asyncio
async def test_cleanup_seeded_faqs_fails_when_delete_rowcount_mismatches(monkeypatch):
    class FakePool:
        async def execute(self, _query, _faq_ids):
            return "DELETE 1"

        async def close(self):
            return None

    async def _create_pool(**_kwargs):
        return FakePool()

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    payload = await smoke._cleanup_seeded_faqs(
        "postgresql://example",
        [
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
        ],
    )

    assert payload == {
        "ok": False,
        "requested_faq_ids": 2,
        "deleted_faq_ids": 1,
        "delete_status": "DELETE 1",
        "errors": ["cleanup deleted 1 FAQ rows but requested 2"],
    }


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
        "ok": False,
        "requested_faq_ids": 1,
        "deleted_faq_ids": None,
        "delete_status": "UPDATE 1",
        "errors": ["cleanup delete status is not parseable: 'UPDATE 1'"],
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
    assert payload["seed"] == {
        "ok": False,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "skipped": True,
        "not_run_reason": "preflight_failed",
    }
    assert payload["route"] == {
        "ok": False,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "skipped": True,
        "not_run_reason": "preflight_failed",
    }
    assert payload["detail"] == {
        "ok": False,
        "returncode": None,
        "skipped": True,
        "not_run_reason": "preflight_failed",
    }
    assert payload["artifact_cleanup"] == {"ok": True, "attempted": False, "error": None}
    assert json.loads(capsys.readouterr().out)["phase"] == "preflight"


def test_main_runs_seed_route_and_cleanup(tmp_path, monkeypatch):
    calls = []

    def _fake_run_command(command):
        calls.append(command)
        _write_child_result(command)
        if str(smoke.SEED_SCRIPT) in command:
            cleanup_manifest = Path(command[command.index("--cleanup-manifest-output") + 1])
            route_cases = Path(command[command.index("--route-case-file-output") + 1])
            _write_cases(
                route_cases,
                [{
                    "query": "export attribution report",
                    "corpus_id": "corp-1",
                    "status": "approved",
                    "limit": 5,
                    "require_results": True,
                    "expected_first_account_id": "acct-1",
                    "expected_detail_account_id": "acct-1",
                    "expected_detail_target_id": "support-corp-1",
                    "expected_detail_target_mode": "support_account",
                    "expected_detail_title": "FAQ Search Smoke",
                    "expected_detail_status": "approved",
                }],
            )
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
            "errors": [],
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
    assert payload["seed"]["result_artifact"]["run_id"] == "seed-run-1"
    assert payload["route"]["result_artifact"]["requests"]["total"] == 12
    assert payload["detail"]["result_artifact"]["detail_checked"] is True
    assert payload["detail"]["ok"] is True
    assert payload["artifacts"]["detail_result"].endswith("detail-result.json")
    assert payload["cleanup"]["deleted_faq_ids"] == 1
    assert payload["artifact_cleanup"] == {"ok": True, "attempted": False, "error": None}
    assert len(calls) == 3
    assert str(smoke.CONTRACT_SCRIPT) in calls[2]


def test_main_fails_when_successful_seed_missing_result_artifact(tmp_path, monkeypatch):
    calls = []

    def _fake_run_command(command):
        calls.append(command)
        assert str(smoke.SEED_SCRIPT) in command
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
            "errors": [],
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
    assert payload["seed"]["ok"] is False
    assert payload["seed"]["returncode"] == 0
    assert payload["seed"]["result_artifact"]["available"] is False
    assert payload["seed"]["result_artifact"]["errors"][0].startswith(
        "result artifact could not be read:"
    )
    assert payload["route"]["not_run_reason"] == "seed_failed"
    assert payload["cleanup"]["deleted_faq_ids"] == 1
    assert len(calls) == 1


def test_main_seed_failure_marks_route_not_run_and_still_cleans_up(tmp_path, monkeypatch):
    calls = []

    def _fake_run_command(command):
        calls.append(command)
        if str(smoke.SEED_SCRIPT) in command:
            _write_child_result(command)
            cleanup_manifest = Path(command[command.index("--cleanup-manifest-output") + 1])
            _write_cases(
                cleanup_manifest,
                {"faq_ids": ["11111111-1111-1111-1111-111111111111"]},
            )
            return {"ok": False, "returncode": 1, "stdout_tail": "", "stderr_tail": "bad seed"}
        return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}

    async def _fake_cleanup(_database_url, faq_ids):
        return {
            "ok": True,
            "requested_faq_ids": len(faq_ids),
            "deleted_faq_ids": len(faq_ids),
            "delete_status": f"DELETE {len(faq_ids)}",
            "errors": [],
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
    assert payload["route"] == {
        "ok": False,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "skipped": True,
        "not_run_reason": "seed_failed",
    }
    assert payload["detail"]["not_run_reason"] == "seed_failed"
    assert payload["cleanup"]["ok"] is True
    assert len(calls) == 1


def test_main_route_failure_still_cleans_up(tmp_path, monkeypatch):
    calls = []

    def _fake_run_command(command):
        calls.append(command)
        if str(smoke.SEED_SCRIPT) in command:
            _write_child_result(command)
            cleanup_manifest = Path(command[command.index("--cleanup-manifest-output") + 1])
            route_cases = Path(command[command.index("--route-case-file-output") + 1])
            _write_cases(
                route_cases,
                [{
                    "query": "export attribution report",
                    "corpus_id": "corp-1",
                    "status": "approved",
                    "limit": 5,
                    "require_results": True,
                    "expected_first_account_id": "acct-1",
                    "expected_detail_account_id": "acct-1",
                    "expected_detail_target_id": "support-corp-1",
                    "expected_detail_target_mode": "support_account",
                    "expected_detail_title": "FAQ Search Smoke",
                    "expected_detail_status": "approved",
                }],
            )
            _write_cases(
                cleanup_manifest,
                {"faq_ids": ["11111111-1111-1111-1111-111111111111"]},
            )
            return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}
        _write_child_result(command, ok=False)
        return {"ok": False, "returncode": 1, "stdout_tail": "", "stderr_tail": "bad route"}

    async def _fake_cleanup(_database_url, faq_ids):
        return {
            "ok": True,
            "requested_faq_ids": len(faq_ids),
            "deleted_faq_ids": len(faq_ids),
            "delete_status": f"DELETE {len(faq_ids)}",
            "errors": [],
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
    assert payload["route"]["ok"] is False
    assert payload["route"]["returncode"] == 1
    assert payload["route"]["stderr_tail"] == "bad route"
    assert payload["route"]["result_artifact"]["errors"]["count"] == 1
    assert payload["detail"] == {
        "ok": False,
        "returncode": None,
        "skipped": True,
        "not_run_reason": "route_failed",
    }
    assert len(calls) == 2
    assert all(str(smoke.CONTRACT_SCRIPT) not in command for command in calls)


def test_main_can_skip_detail_check_for_liveness_runs(tmp_path, monkeypatch):
    calls = []

    def _fake_run_command(command):
        calls.append(command)
        _write_child_result(command)
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
            "errors": [],
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
        "--skip-detail-check",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["detail"] == {
        "ok": True,
        "returncode": None,
        "skipped": True,
        "not_run_reason": "skip_detail_check",
    }
    assert len(calls) == 2
    assert all(str(smoke.CONTRACT_SCRIPT) not in command for command in calls)


def test_main_reports_cleanup_failure(tmp_path, monkeypatch):
    def _fake_run_command(command):
        _write_child_result(command)
        if str(smoke.SEED_SCRIPT) in command:
            cleanup_manifest = Path(command[command.index("--cleanup-manifest-output") + 1])
            route_cases = Path(command[command.index("--route-case-file-output") + 1])
            _write_cases(
                route_cases,
                [{
                    "query": "export attribution report",
                    "corpus_id": "corp-1",
                    "status": "approved",
                    "limit": 5,
                    "require_results": True,
                    "expected_first_account_id": "acct-1",
                    "expected_detail_account_id": "acct-1",
                    "expected_detail_target_id": "support-corp-1",
                    "expected_detail_target_mode": "support_account",
                    "expected_detail_title": "FAQ Search Smoke",
                    "expected_detail_status": "approved",
                }],
            )
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
            "errors": ["cleanup failed"],
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
        "errors": ["cleanup failed"],
    }


def test_main_reports_artifact_cleanup_failure_without_masking_seed_failure(
    tmp_path,
    monkeypatch,
    capsys,
):
    class _TempDirectory:
        def __init__(self, **_kwargs):
            self.name = str(tmp_path / "temp-artifacts")

        def cleanup(self):
            raise OSError("artifact cleanup failed")

    def _fake_run_command(command):
        assert str(smoke.SEED_SCRIPT) in command
        return {"ok": False, "returncode": 1, "stdout_tail": "", "stderr_tail": "bad seed"}

    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", _TempDirectory)
    monkeypatch.setattr(smoke, "_run_command", _fake_run_command)
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
        "--output-result",
        str(result_path),
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    assert payload["seed"]["ok"] is False
    assert payload["seed"]["returncode"] == 1
    assert payload["seed"]["stdout_tail"] == ""
    assert payload["seed"]["stderr_tail"] == "bad seed"
    assert payload["seed"]["result_artifact"]["available"] is False
    assert payload["seed"]["result_artifact"]["errors"][0].startswith(
        "result artifact could not be read:"
    )
    assert payload["route"]["not_run_reason"] == "seed_failed"
    assert payload["detail"]["not_run_reason"] == "seed_failed"
    assert payload["cleanup"] == {
        "ok": True,
        "requested_faq_ids": 0,
        "deleted_faq_ids": 0,
        "delete_status": None,
        "errors": [],
    }
    assert payload["artifact_cleanup"] == {
        "ok": False,
        "attempted": True,
        "error": {
            "type": "OSError",
            "message": "artifact cleanup failed",
        },
    }
    output = capsys.readouterr().out
    assert "artifact_cleanup=False" in output
    assert "artifact_cleanup_error=OSError: artifact cleanup failed" in output
