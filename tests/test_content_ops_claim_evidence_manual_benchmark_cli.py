from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_content_ops_claim_evidence_manual_benchmark.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_content_ops_claim_evidence_manual_benchmark",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_row(triple_id: str, expected: bool, difficulty: str) -> dict[str, object]:
    return {
        "triple_id": triple_id,
        "claim_id": f"claim-{triple_id}",
        "claim_text": f"Claim statement {triple_id}",
        "evidence_quote": f"evidence {triple_id}",
        "source_id": f"source-{triple_id}",
        "expected_supports": expected,
        "difficulty": difficulty,
    }


def _packet(model_id: str, triple_id: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "triple_id": triple_id,
        "claim_id": f"claim-{triple_id}",
        "source_id": f"source-{triple_id}",
        "difficulty": "easy",
        "contract_version": "verify_claim_evidence.v1",
        "prompt": f"Prompt for {triple_id}",
        "response_schema": {"type": "object"},
    }


def _response(supports: bool) -> dict[str, object]:
    return {
        "supports": supports,
        "confidence": 5,
        "reason": "the quote directly supports the judgment",
    }


def _response_row(
    model_id: str,
    triple_id: str,
    supports: bool,
    *,
    run_type: str = "main",
    run_id: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_id": model_id,
        "triple_id": triple_id,
        "contract_version": "verify_claim_evidence.v1",
        "response": _response(supports),
    }
    if run_type != "main":
        row["run_type"] = run_type
    if run_id:
        row["run_id"] = run_id
    return row


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    fixture_path = tmp_path / "fixture.json"
    packets_path = tmp_path / "packets.json"
    responses_path = tmp_path / "responses.json"
    _write_json(
        fixture_path,
        [_fixture_row("easy", True, "easy"), _fixture_row("hard", False, "hard")],
    )
    _write_json(
        packets_path,
        [
            _packet("claude", "easy"),
            _packet("claude", "hard"),
            _packet("gpt", "easy"),
            _packet("gpt", "hard"),
        ],
    )
    rows = [
        _response_row("claude", "easy", True),
        _response_row("claude", "hard", False),
        _response_row("gpt", "easy", True),
        _response_row("gpt", "hard", False),
    ]
    for model_id in ("claude", "gpt"):
        for run_id in ("rerun-1", "rerun-2"):
            rows.extend(
                [
                    _response_row(
                        model_id,
                        "easy",
                        True,
                        run_type="stability",
                        run_id=run_id,
                    ),
                    _response_row(
                        model_id,
                        "hard",
                        False,
                        run_type="stability",
                        run_id=run_id,
                    ),
                ]
            )
    _write_json(responses_path, rows)
    return fixture_path, packets_path, responses_path


def test_manual_benchmark_writes_recorded_responses_and_result_artifact(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path, packets_path, responses_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "benchmark-output"

    exit_code, payload = cli.run_manual_benchmark_from_files(
        fixture_path,
        packets_path,
        responses_path,
        output_dir,
    )

    recorded_path = output_dir / "recorded_responses.json"
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["go_no_go"] == "go"
    assert payload["recorded_responses_output"] == str(recorded_path)
    assert payload["import_payload"]["response_count"] == 12
    assert payload["artifact_payload"]["ok"] is True
    assert recorded_path.exists()
    assert (output_dir / "claim_evidence_result.json").exists()
    assert (output_dir / "claim_evidence_result.md").exists()


def test_manual_benchmark_import_failure_stops_before_artifact_write(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path, packets_path, responses_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "benchmark-output"
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    _write_json(responses_path, responses[:3] + responses[4:])

    exit_code, payload = cli.run_manual_benchmark_from_files(
        fixture_path,
        packets_path,
        responses_path,
        output_dir,
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["errors"] == ["response import failed"]
    assert "missing main response for gpt/hard" in payload["import_payload"]["errors"]
    assert (output_dir / "recorded_responses.json").exists() is False
    assert (output_dir / "claim_evidence_result.json").exists() is False


def test_manual_benchmark_rejects_symlink_output_dir_without_writing(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path, packets_path, responses_path = _write_inputs(tmp_path)
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    output_dir = tmp_path / "benchmark-output"
    output_dir.symlink_to(external_dir, target_is_directory=True)

    exit_code, payload = cli.run_manual_benchmark_from_files(
        fixture_path,
        packets_path,
        responses_path,
        output_dir,
    )

    assert exit_code == 2
    assert payload["errors"] == [f"output_dir is a symlink: {output_dir}"]
    assert list(external_dir.iterdir()) == []


def test_manual_benchmark_rejects_symlink_recorded_output_before_import(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path, packets_path, responses_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "benchmark-output"
    external_path = tmp_path / "external.json"
    recorded_output = output_dir / "recorded_responses.json"
    output_dir.mkdir()
    external_path.write_text("external stays intact", encoding="utf-8")
    recorded_output.symlink_to(external_path)

    exit_code, payload = cli.run_manual_benchmark_from_files(
        fixture_path,
        packets_path,
        responses_path,
        output_dir,
        recorded_responses_output=recorded_output,
    )

    assert exit_code == 2
    assert payload["errors"] == [
        f"recorded responses output is a symlink: {recorded_output}"
    ]
    assert external_path.read_text(encoding="utf-8") == "external stays intact"


def test_manual_benchmark_refuses_existing_result_artifact_before_import(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path, packets_path, responses_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "benchmark-output"
    output_dir.mkdir()
    existing_result = output_dir / "claim_evidence_result.json"
    existing_result.write_text('{"go_no_go": "go"}\n', encoding="utf-8")

    exit_code, payload = cli.run_manual_benchmark_from_files(
        fixture_path,
        packets_path,
        responses_path,
        output_dir,
    )

    assert exit_code == 2
    assert payload["errors"] == [
        f"output_dir already contains benchmark artifact file: {existing_result}"
    ]
    assert existing_result.read_text(encoding="utf-8") == '{"go_no_go": "go"}\n'
    assert (output_dir / "recorded_responses.json").exists() is False


@pytest.mark.parametrize(
    ("recorded_name", "expected_error"),
    (
        ("fixture", "recorded responses output must differ from fixture path"),
        (
            "claim_evidence_result.json",
            "recorded responses output must not overwrite artifact file",
        ),
    ),
)
def test_manual_benchmark_rejects_unsafe_recorded_response_paths(
    tmp_path: Path,
    recorded_name: str,
    expected_error: str,
) -> None:
    cli = _load_cli_module()
    fixture_path, packets_path, responses_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "benchmark-output"
    recorded_output = (
        fixture_path if recorded_name == "fixture" else output_dir / recorded_name
    )

    exit_code, payload = cli.run_manual_benchmark_from_files(
        fixture_path,
        packets_path,
        responses_path,
        output_dir,
        recorded_responses_output=recorded_output,
    )

    assert exit_code == 2
    assert any(error.startswith(expected_error) for error in payload["errors"])
    assert (output_dir / "claim_evidence_result.json").exists() is False


def test_manual_benchmark_direct_script_invocation_prints_json_envelope(
    tmp_path: Path,
) -> None:
    fixture_path, packets_path, responses_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "benchmark-output"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(fixture_path),
            str(packets_path),
            str(responses_path),
            str(output_dir),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ok"] is True
    assert payload["go_no_go"] == "go"
