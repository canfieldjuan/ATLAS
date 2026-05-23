from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "smoke_content_ops_ingestion_file_route_multiprocess_load.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "smoke_content_ops_ingestion_file_route_multiprocess_load",
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


def _default_args(path: Path, result_path: Path, output_dir: Path) -> list[str]:
    return [
        str(path),
        "--source-format",
        "jsonl",
        "--source",
        "cfpb-route-multiprocess-load-test",
        "--min-source-rows",
        "3",
        "--default-field",
        "company_name=CFPB Public Archive",
        "--default-field",
        "vendor_name=CFPB",
        "--default-field",
        "contact_email=cfpb-public-archive@example.invalid",
        "--account-id",
        "acct-route-multiprocess-load",
        "--database-url",
        "postgresql://atlas@localhost:5433/atlas",
        "--output-dir",
        str(output_dir),
        "--output-result",
        str(result_path),
    ]


def test_multiprocess_load_builds_child_command(tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    output_dir = tmp_path / "children"
    _write_cfpb_rows(source_path, 3)
    args = module._parse_args([
        *_default_args(source_path, result_path, output_dir),
        "--processes",
        "2",
        "--child-concurrency",
        "4",
        "--child-import-max-concurrency",
        "2",
        "--admission-provider",
        "postgres",
        "--replace-existing",
    ])

    spec = module._child_spec(args, index=1)
    command = spec["command"]

    assert spec["account_id"] == "acct-route-multiprocess-load-p2"
    assert spec["source"] == "cfpb-route-multiprocess-load-test-p2"
    assert str(module._CHILD_SCRIPT) in command
    assert command[command.index("--account-id") + 1] == "acct-route-multiprocess-load-p2"
    assert command[command.index("--source") + 1] == "cfpb-route-multiprocess-load-test-p2"
    assert command[command.index("--concurrency") + 1] == "4"
    assert command[command.index("--import-max-concurrency") + 1] == "2"
    assert command[command.index("--admission-provider") + 1] == "postgres"
    assert command[command.index("--database-url") + 1] == "postgresql://atlas@localhost:5433/atlas"
    assert "--replace-existing" in command
    assert command.count("--default-field") == 3


def test_multiprocess_load_writes_aggregate_result(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    output_dir = tmp_path / "children"
    _write_cfpb_rows(source_path, 3)

    async def run_child(command, *, result_path):
        account_id = command[command.index("--account-id") + 1]
        child_payload = {
            "ok": True,
            "summary": {
                "successes": 1,
                "at_capacity": 1,
                "unexpected_failures": 0,
                "inserted": 3,
            },
        }
        result_path.write_text(json.dumps(child_payload) + "\n", encoding="utf-8")
        return {
            "result_path": str(result_path),
            "returncode": 0,
            "ok": True,
            "summary": dict(child_payload["summary"]),
            "stdout_tail": f"account={account_id}",
            "stderr_tail": "",
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(module, "_run_child_process", run_child)

    code = module.main([
        *_default_args(source_path, result_path, output_dir),
        "--processes",
        "2",
        "--child-concurrency",
        "2",
        "--child-import-max-concurrency",
        "1",
        "--admission-provider",
        "postgres",
        "--min-total-successes",
        "2",
        "--expect-total-at-capacity-min",
        "2",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["ok"] is True
    assert payload["admission_provider"] == "postgres"
    assert payload["summary"] == {
        "processes": 2,
        "successful_processes": 2,
        "capacity_only_processes": 0,
        "failed_processes": 0,
        "successes": 2,
        "at_capacity": 2,
        "unexpected_failures": 0,
        "inserted": 6,
        "returncode_counts": {"0": 2},
    }
    assert len(payload["children"]) == 2
    assert (output_dir / "child_1.json").exists()
    assert (output_dir / "child_2.json").exists()


def test_multiprocess_load_can_accept_capacity_only_children(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    output_dir = tmp_path / "children"
    _write_cfpb_rows(source_path, 3)

    async def run_child(command, *, result_path):
        account_id = command[command.index("--account-id") + 1]
        capacity_only = account_id.endswith("-p2")
        summary = {
            "successes": 0 if capacity_only else 1,
            "at_capacity": 2 if capacity_only else 1,
            "unexpected_failures": 0,
            "inserted": 0 if capacity_only else 3,
        }
        return {
            "result_path": str(result_path),
            "returncode": 1 if capacity_only else 0,
            "ok": not capacity_only,
            "summary": summary,
            "errors": [
                "expected at least 3 inserted source row(s), got 0",
                "expected at least 1 success(es), got 0",
            ] if capacity_only else [],
            "stdout_tail": "",
            "stderr_tail": "",
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(module, "_run_child_process", run_child)

    code = module.main([
        *_default_args(source_path, result_path, output_dir),
        "--processes",
        "2",
        "--min-total-successes",
        "1",
        "--expect-total-at-capacity-min",
        "3",
        "--allow-capacity-only-children",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["ok"] is True
    assert payload["allow_capacity_only_children"] is True
    assert payload["summary"]["successful_processes"] == 2
    assert payload["summary"]["capacity_only_processes"] == 1
    assert payload["summary"]["failed_processes"] == 0
    assert payload["summary"]["successes"] == 1
    assert payload["summary"]["at_capacity"] == 3


def test_multiprocess_load_fails_on_child_failure(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    source_path = tmp_path / "cfpb_rows.jsonl"
    result_path = tmp_path / "result.json"
    output_dir = tmp_path / "children"
    _write_cfpb_rows(source_path, 3)

    async def run_child(command, *, result_path):
        account_id = command[command.index("--account-id") + 1]
        failed = account_id.endswith("-p2")
        summary = {
            "successes": 0 if failed else 1,
            "at_capacity": 0 if failed else 1,
            "unexpected_failures": 1 if failed else 0,
            "inserted": 0 if failed else 3,
        }
        return {
            "result_path": str(result_path),
            "returncode": 1 if failed else 0,
            "ok": not failed,
            "summary": summary,
            "stdout_tail": "",
            "stderr_tail": "TooManyConnectionsError" if failed else "",
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(module, "_run_child_process", run_child)

    code = module.main([
        *_default_args(source_path, result_path, output_dir),
        "--processes",
        "2",
        "--min-total-successes",
        "2",
        "--expect-total-at-capacity-min",
        "2",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    assert payload["summary"]["failed_processes"] == 1
    assert payload["summary"]["capacity_only_processes"] == 0
    assert payload["summary"]["unexpected_failures"] == 1
    assert payload["errors"] == [
        "failed child process count: 1",
        "unexpected child failure count: 1",
        "expected at least 2 total success(es), got 1",
        "expected at least 2 total admission 429 response(s), got 1",
    ]


def test_multiprocess_load_requires_database_url(monkeypatch, tmp_path: Path) -> None:
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
            "acct-route-multiprocess-load",
            "--database-url",
            "",
        ])

    assert "Missing --database-url" in str(exc.value)
