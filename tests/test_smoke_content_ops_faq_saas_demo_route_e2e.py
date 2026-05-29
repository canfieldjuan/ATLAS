from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
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


def _clear_atlas_db_env(monkeypatch) -> None:
    for name in (
        "ATLAS_DB_HOST",
        "ATLAS_DB_PORT",
        "ATLAS_DB_DATABASE",
        "ATLAS_DB_USER",
        "ATLAS_DB_PASSWORD",
        "ATLAS_DB_SOCKET_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def _fake_runner(
    calls: list[list[str]],
    *,
    route_ok: bool = True,
    route_artifact_ok: bool | None = None,
    route_artifact_body: str | None = None,
    route_writes_artifact: bool = True,
    seed_faq_id: str = "faq-123",
):
    if route_artifact_ok is None:
        route_artifact_ok = route_ok

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
            if route_artifact_body is not None:
                route_result.parent.mkdir(parents=True, exist_ok=True)
                route_result.write_text(route_artifact_body, encoding="utf-8")
            elif route_writes_artifact:
                _write_json(
                    route_result,
                    {
                        "ok": route_artifact_ok,
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


def test_default_database_url_prefers_url_env(monkeypatch) -> None:
    monkeypatch.setenv("EXTRACTED_DATABASE_URL", "postgresql://env/atlas")
    monkeypatch.setenv("DATABASE_URL", "postgresql://database-url/atlas")

    assert smoke._default_database_url() == "postgresql://env/atlas"


def test_default_database_url_falls_back_to_atlas_db_settings(monkeypatch) -> None:
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ATLAS_DB_HOST", "settings-host")
    monkeypatch.setenv("ATLAS_DB_PORT", "6543")
    monkeypatch.setenv("ATLAS_DB_DATABASE", "atlas_settings")
    monkeypatch.setenv("ATLAS_DB_USER", "atlas_user")
    monkeypatch.setenv("ATLAS_DB_PASSWORD", "atlas_pass")
    monkeypatch.delenv("ATLAS_DB_SOCKET_PATH", raising=False)

    assert (
        smoke._default_database_url()
        == "postgresql://atlas_user:atlas_pass@settings-host:6543/atlas_settings"
    )


def test_default_database_url_ignores_implicit_atlas_db_defaults(monkeypatch) -> None:
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _clear_atlas_db_env(monkeypatch)

    assert smoke._default_database_url() == ""


def test_blank_database_url_uses_guarded_fallback(monkeypatch) -> None:
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ATLAS_DB_HOST", "settings-host")
    monkeypatch.setenv("ATLAS_DB_PORT", "6543")
    monkeypatch.setenv("ATLAS_DB_DATABASE", "atlas_settings")
    monkeypatch.setenv("ATLAS_DB_USER", "atlas_user")
    monkeypatch.setenv("ATLAS_DB_PASSWORD", "atlas_pass")
    monkeypatch.delenv("ATLAS_DB_SOCKET_PATH", raising=False)
    args = smoke._build_parser().parse_args(["--database-url", ""])

    smoke._normalize_args(args)

    assert args.database_url == "postgresql://atlas_user:atlas_pass@settings-host:6543/atlas_settings"


def test_blank_database_url_stays_missing_without_target(monkeypatch) -> None:
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _clear_atlas_db_env(monkeypatch)
    args = smoke._build_parser().parse_args(["--database-url", ""])

    smoke._normalize_args(args)

    assert args.database_url == ""


def test_script_preflight_uses_atlas_db_settings_fallback(tmp_path) -> None:
    result_path = tmp_path / "preflight.json"
    env = os.environ.copy()
    env.pop("EXTRACTED_DATABASE_URL", None)
    env.pop("DATABASE_URL", None)
    env.pop("ATLAS_API_BASE_URL", None)
    env.pop("ATLAS_B2B_JWT", None)
    env.pop("ATLAS_TOKEN", None)
    env.pop("ATLAS_FAQ_SEARCH_ACCOUNT_ID", None)
    env.pop("ATLAS_ACCOUNT_ID", None)
    env.update(
        {
            "ATLAS_DB_HOST": "db-settings-host",
            "ATLAS_DB_PORT": "5432",
            "ATLAS_DB_DATABASE": "atlas_settings",
            "ATLAS_DB_USER": "atlas_settings_user",
            "ATLAS_DB_PASSWORD": "atlas_settings_password",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--preflight-only",
            "--json",
            "--output-result",
            str(result_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert completed.returncode == 2
    assert payload["required_inputs"]["database_url"] == {"present": True}
    assert "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL" not in (
        payload["preflight_errors"]
    )


@pytest.mark.parametrize(
    ("base_url", "expected_error"),
    [
        (
            "atlas.example.com",
            "--base-url must be an absolute HTTP(S) URL for hosted proof",
        ),
        (
            "ftp://atlas.example.com",
            "--base-url must be an absolute HTTP(S) URL for hosted proof",
        ),
        (
            "http://[::1",
            "--base-url must be an absolute HTTP(S) URL for hosted proof",
        ),
        (
            "http://localhost:8000",
            "--base-url must point to a deployed host; local hosts are not accepted for hosted proof",
        ),
        (
            "http://127.0.0.1:8000",
            "--base-url must point to a deployed host; local hosts are not accepted for hosted proof",
        ),
        (
            "http://0.0.0.0:8000",
            "--base-url must point to a deployed host; local hosts are not accepted for hosted proof",
        ),
        (
            "http://[::1]:8000",
            "--base-url must point to a deployed host; local hosts are not accepted for hosted proof",
        ),
    ],
)
def test_validate_args_fails_closed_for_non_hosted_base_url(
    base_url: str,
    expected_error: str,
) -> None:
    args = smoke._build_parser().parse_args([
        "--database-url",
        "postgresql://example/atlas",
        "--base-url",
        base_url,
        "--token",
        "token-123",
        "--account-id",
        "acct-1",
    ])

    assert smoke._validate_args(args) == [expected_error]


def test_main_local_base_url_writes_preflight_result_before_exit(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls))
    result_path = tmp_path / "local-base-url-preflight.json"

    code = smoke.main([
        "--database-url",
        "postgresql://example/atlas",
        "--base-url",
        "http://127.0.0.1:8000",
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
        "--base-url must point to a deployed host; local hosts are not accepted for hosted proof"
    ]
    assert payload["required_inputs"]["base_url"] == {"present": True}
    assert payload["seed"]["not_run_reason"] == "preflight_failed"
    assert payload["route"]["not_run_reason"] == "preflight_failed"
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert calls == []


def test_main_malformed_base_url_writes_preflight_result_before_exit(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls))
    result_path = tmp_path / "malformed-base-url-preflight.json"

    code = smoke.main([
        "--database-url",
        "postgresql://example/atlas",
        "--base-url",
        "http://[::1",
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
        "--base-url must be an absolute HTTP(S) URL for hosted proof"
    ]
    assert payload["seed"]["not_run_reason"] == "preflight_failed"
    assert payload["route"]["not_run_reason"] == "preflight_failed"
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert calls == []


def test_main_writes_preflight_result_before_exit(tmp_path, capsys, monkeypatch) -> None:
    result_path = tmp_path / "preflight.json"
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _clear_atlas_db_env(monkeypatch)

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
    assert payload["required_inputs"] == {
        "database_url": {"present": False},
        "base_url": {"present": False},
        "token": {"present": False},
        "account_id": {"present": False},
    }
    assert payload["seed"]["not_run_reason"] == "preflight_failed"
    assert payload["route"]["not_run_reason"] == "preflight_failed"
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_main_preflight_only_reports_ready_inputs_without_running_children(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls))
    result_path = tmp_path / "preflight-ready.json"

    code = smoke.main([*_base_args(tmp_path), "--preflight-only", "--output-result", str(result_path), "--json"])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    stdout_payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["phase"] == "preflight"
    assert payload["preflight_errors"] == []
    assert payload["required_inputs"] == {
        "database_url": {"present": True},
        "base_url": {"present": True},
        "token": {"present": True},
        "account_id": {"present": True},
    }
    assert payload["seed"]["not_run_reason"] == "preflight_only"
    assert payload["route"]["not_run_reason"] == "preflight_only"
    assert payload["cleanup"]["not_run_reason"] == "preflight_only"
    assert stdout_payload["required_inputs"] == payload["required_inputs"]
    assert calls == []


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


def test_main_fails_when_successful_route_omits_result_artifact(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls, route_writes_artifact=False))

    code = smoke.main(_base_args(tmp_path))

    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert code == 1
    assert payload["route"]["ok"] is True
    assert payload["route"]["result_artifact"]["available"] is False
    assert len(payload["errors"]) == 1
    assert payload["errors"][0].startswith("route result could not be read:")
    assert str(tmp_path / "artifacts" / "route-result.json") in payload["errors"][0]
    assert payload["cleanup"]["ok"] is True
    assert len(calls) == 3


def test_main_fails_when_successful_route_writes_malformed_artifact(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls, route_artifact_body="{"))

    code = smoke.main(_base_args(tmp_path))

    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert code == 1
    assert payload["route"]["ok"] is True
    assert payload["route"]["result_artifact"]["available"] is False
    assert payload["route"]["result_artifact"]["ok"] is False
    assert payload["errors"] == ["route result must contain JSON: Expecting property name enclosed in double quotes"]
    assert payload["cleanup"]["ok"] is True
    assert len(calls) == 3


def test_main_fails_when_successful_route_artifact_reports_not_ok(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(smoke, "_run_command", _fake_runner(calls, route_artifact_ok=False))

    code = smoke.main(_base_args(tmp_path))

    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert code == 1
    assert payload["route"]["ok"] is True
    assert payload["route"]["result_artifact"]["available"] is True
    assert payload["route"]["result_artifact"]["ok"] is False
    assert payload["errors"] == ["route result artifact ok was not true"]
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
