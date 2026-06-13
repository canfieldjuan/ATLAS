from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_content_ops_claim_evidence_artifact.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_content_ops_claim_evidence_artifact",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(triple_id: str, expected: bool, difficulty: str) -> dict[str, object]:
    return {
        "triple_id": triple_id,
        "claim_id": f"claim-{triple_id}",
        "claim_text": f"Claim statement {triple_id}",
        "evidence_quote": f"evidence {triple_id}",
        "source_id": f"source-{triple_id}",
        "expected_supports": expected,
        "difficulty": difficulty,
    }


def _response(supports: bool, confidence: int = 5) -> dict[str, object]:
    return {
        "supports": supports,
        "confidence": confidence,
        "reason": "the evidence quote directly supports the judgment",
    }


def _run(model_id: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "responses": {
            "easy": _response(True),
            "hard": _response(False),
        },
    }


def _responses_payload() -> dict[str, object]:
    return {
        "model_runs": [_run("claude"), _run("gpt")],
        "stability_runs_by_model_id": {
            "claude": [_run("claude"), _run("claude")],
            "gpt": [_run("gpt"), _run("gpt")],
        },
    }


def _write_fixture(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                _row("easy", True, "easy"),
                _row("hard", False, "hard"),
            ]
        ),
        encoding="utf-8",
    )


def test_cli_writes_result_artifact_directory_from_recorded_responses(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    responses_path = tmp_path / "responses.json"
    output_dir = tmp_path / "artifact"
    _write_fixture(fixture_path)
    responses_path.write_text(json.dumps(_responses_payload()), encoding="utf-8")

    exit_code, payload = cli.build_claim_evidence_artifact_from_files(
        fixture_path,
        responses_path,
        output_dir,
    )

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["go_no_go"] == "go"
    assert payload["errors"] == []
    assert [file["path"] for file in payload["files"]] == [
        "claim_evidence_result.json",
        "claim_evidence_result.md",
    ]
    result_json = output_dir / "claim_evidence_result.json"
    result_markdown = output_dir / "claim_evidence_result.md"
    assert result_json.exists()
    assert result_markdown.exists()
    assert json.loads(result_json.read_text(encoding="utf-8"))["go_no_go"] == "go"
    assert "Claim Evidence Benchmark Result" in result_markdown.read_text(
        encoding="utf-8"
    )


def test_cli_rejects_malformed_response_json_without_writing(tmp_path: Path) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    responses_path = tmp_path / "responses.json"
    output_dir = tmp_path / "artifact"
    _write_fixture(fixture_path)
    responses_path.write_text("{", encoding="utf-8")

    exit_code, payload = cli.build_claim_evidence_artifact_from_files(
        fixture_path,
        responses_path,
        output_dir,
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["errors"] == ["responses json is malformed: Expecting property name enclosed in double quotes"]
    assert output_dir.exists() is False


def test_cli_rejects_duplicate_normalized_stability_model_ids(tmp_path: Path) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    responses_path = tmp_path / "responses.json"
    output_dir = tmp_path / "artifact"
    _write_fixture(fixture_path)
    payload = _responses_payload()
    flipped_run = {
        "model_id": "gpt",
        "responses": {
            "easy": _response(False),
            "hard": _response(True),
        },
    }
    payload["stability_runs_by_model_id"][" gpt "] = [flipped_run, flipped_run]
    responses_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code, result = cli.build_claim_evidence_artifact_from_files(
        fixture_path,
        responses_path,
        output_dir,
    )

    assert exit_code == 1
    assert result["ok"] is False
    assert result["errors"] == ["stability model_id duplicated: gpt"]
    assert output_dir.exists() is False


def test_cli_writes_no_go_artifact_for_unknown_response_triple(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    responses_path = tmp_path / "responses.json"
    output_dir = tmp_path / "artifact"
    _write_fixture(fixture_path)
    payload = _responses_payload()
    payload["model_runs"][0]["responses"]["ghost"] = _response(True)
    responses_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code, result = cli.build_claim_evidence_artifact_from_files(
        fixture_path,
        responses_path,
        output_dir,
    )

    assert exit_code == 1
    assert result["ok"] is False
    assert result["go_no_go"] == "no_go"
    assert result["errors"] == []
    assert result["artifact_errors"] == ["claude: unknown row triple_id: ghost"]
    assert (output_dir / "claim_evidence_result.json").exists()


def test_cli_reports_writer_failure_without_silent_success(tmp_path: Path) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    responses_path = tmp_path / "responses.json"
    output_path = tmp_path / "artifact-file"
    _write_fixture(fixture_path)
    responses_path.write_text(json.dumps(_responses_payload()), encoding="utf-8")
    output_path.write_text("already a file", encoding="utf-8")

    exit_code, payload = cli.build_claim_evidence_artifact_from_files(
        fixture_path,
        responses_path,
        output_path,
    )

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["files"] == []
    assert payload["errors"] == [f"output_dir is not a directory: {output_path}"]
    assert output_path.read_text(encoding="utf-8") == "already a file"


def test_direct_script_invocation_prints_json_envelope(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    responses_path = tmp_path / "responses.json"
    output_dir = tmp_path / "artifact"
    _write_fixture(fixture_path)
    responses_path.write_text(json.dumps(_responses_payload()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(fixture_path),
            str(responses_path),
            str(output_dir),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert result.returncode == 0
    assert result.stderr == ""
    assert output["ok"] is True
    assert output["output_dir"] == str(output_dir)
