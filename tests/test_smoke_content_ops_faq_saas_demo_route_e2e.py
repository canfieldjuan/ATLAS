from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_saas_demo_route_e2e.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_faq_saas_demo_route_e2e", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_args(tmp_path: Path) -> list[str]:
    return [
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
        str(tmp_path / "result.json"),
    ]


def _fake_runner(calls: list[list[str]], *, route_ok: bool = True, seed_faq_id: str = "faq-123"):
    def _run(command):
        calls.append(list(command))
        if str(smoke.SEED_SCRIPT) in command and "--cleanup-faq-id" not in command:
            seed_result = Path(command[command.index("--output-result") + 1])
            case_file = Path(command[command.index("--route-case-file-output") + 1])
            _write_json(
                seed_result,
                {
                    "ok": True,
                    "phase": "seed",
                    "faq_id": seed_faq_id,
                    "corpus_id": "synthetic-b2b-saas-demo",
                    "target_id": "support-synthetic-b2b-saas-demo",
                    "status": "approved",
                    "source_count": 36,
                    "generated_items": 9,
                    "search": {
                        "query": "export attribution reports",
                        "count": 1,
                        "matched_seeded_faq": True,
                    },
                    "route_case_file": {
                        "ok": True,
                        "path": str(case_file),
                        "cases": 1,
                    },
                },
            )
            _write_json(
                case_file,
                [{
                    "query": "export attribution reports",
                    "limit": 5,
                    "require_results": True,
                    "expected_first_faq_id": seed_faq_id,
                    "expected_detail_account_id": "acct-1",
                    "expected_detail_target_id": "support-synthetic-b2b-saas-demo",
                    "expected_detail_target_mode": "support_account",
                    "expected_detail_title": "Synthetic B2B SaaS Support FAQ Demo",
                    "expected_detail_status": "approved",
                }],
            )
            return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}
        if str(smoke.ROUTE_SCRIPT) in command:
            route_result = Path(command[command.index("--output-result") + 1])
            _write_json(
                route_result,
                {
                    "ok": route_ok,
                    "phase": "complete",
                    "requests": {"total": 40, "concurrency": 8},
                    "cases": {"total": 1, "summaries": []},
                    "detail": {"checked": 40, "failures": 0 if route_ok else 1},
                    "budgets": {"ok": route_ok, "failures": [] if route_ok else ["detail failed"]},
                    "errors": [] if route_ok else ["detail failed"],
                },
            )
            return {"ok": route_ok, "returncode": 0 if route_ok else 1, "stdout_tail": "", "stderr_tail": ""}
        if str(smoke.SEED_SCRIPT) in command and "--cleanup-faq-id" in command:
            cleanup_result = Path(command[command.index("--output-result") + 1])
            _write_json(
                cleanup_result,
                {
                    "ok": True,
                    "phase": "cleanup",
                    "account_id": "acct-1",
                    "faq_id": command[command.index("--cleanup-faq-id") + 1],
                    "deleted_faq_ids": 1,
                    "delete_status": "DELETE 1",
                },
            )
            return {"ok": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""}
        raise AssertionError(f"unexpected command: {command}")

    return _run


def test_validate_args_fails_closed_for_missing_host_inputs() -> None:
    args = smoke._build_parser().parse_args([
        "--database-url",
        "",
        "--base-url",
        "",
        "--token",
        "",
        "--account-id",
        "",
        "--route-requests",
        "0",
        "--max-detail-ms",
        "0",
    ])

    assert smoke._validate_args(args) == [
        "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL",
        "ATLAS_API_BASE_URL or --base-url is required",
        "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required",
        "ATLAS_FAQ_SEARCH_ACCOUNT_ID, ATLAS_ACCOUNT_ID, or --account-id is required",
        "--route-requests must be positive",
        "--max-detail-ms must be positive",
    ]


def test_main_writes_preflight_result_before_exit(tmp_path, capsys) -> None:
    result_path = tmp_path / "preflight.json"

    code = smoke.main([
        "--database-url",
        "",
        "--base-url",
        "",
        "--token",
        "",
        "--account-id",
        "",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["phase"] == "preflight"
    assert payload["seed"]["not_run_reason"] == "preflight_failed"
    assert payload["route"]["not_run_reason"] == "preflight_failed"
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_main_runs_seed_route_and_cleanup(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls))

    code = smoke.main(_base_args(tmp_path))

    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert code == 0
    assert payload["ok"] is True
    assert payload["seed"]["result_artifact"]["faq_id"] == "faq-123"
    assert payload["seed"]["result_artifact"]["route_case_file"] == {
        "ok": True,
        "path": calls[0][calls[0].index("--route-case-file-output") + 1],
        "cases": 1,
        "error": None,
    }
    assert payload["route"]["result_artifact"]["detail"]["checked"] == 40
    assert payload["cleanup"]["result_artifact"]["faq_id"] == "faq-123"
    assert payload["cleanup"]["result_artifact"]["account_id"] == "acct-1"
    assert payload["cleanup"]["result_artifact"]["deleted_faq_ids"] == 1
    assert payload["cleanup"]["result_artifact"]["delete_status"] == "DELETE 1"
    assert "--route-case-file-output" in calls[0]
    assert calls[1][calls[1].index("--case-file") + 1] == calls[0][calls[0].index("--route-case-file-output") + 1]
    assert calls[2][calls[2].index("--cleanup-faq-id") + 1] == "faq-123"


def test_main_cleans_up_when_route_fails(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls, route_ok=False))

    code = smoke.main(_base_args(tmp_path))

    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert code == 1
    assert payload["route"]["ok"] is False
    assert payload["cleanup"]["ok"] is True
    assert len(calls) == 3


def test_main_keep_data_skips_cleanup(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls))

    code = smoke.main([*_base_args(tmp_path), "--keep-data"])

    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert code == 0
    assert payload["cleanup"] == {
        "ok": True,
        "skipped": True,
        "not_run_reason": "keep_data",
    }
    assert len(calls) == 2


def test_main_missing_seed_faq_id_fails_cleanup_closed(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls, seed_faq_id=""))

    code = smoke.main(_base_args(tmp_path))

    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert code == 1
    assert payload["errors"] == ["seed result faq_id must be a non-empty string"]
    assert payload["cleanup"] == {
        "ok": False,
        "skipped": True,
        "not_run_reason": "missing_seed_faq_id",
    }
    assert len(calls) == 2
