from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "import_content_ops_claim_evidence_prompt_responses.py"
ARTIFACT_SCRIPT = ROOT / "scripts" / "run_content_ops_claim_evidence_artifact.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "import_content_ops_claim_evidence_prompt_responses",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _fixture_row(
    triple_id: str,
    expected: bool,
    difficulty: str,
) -> dict[str, object]:
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


def _packets() -> list[dict[str, object]]:
    return [
        _packet("claude", "easy"),
        _packet("claude", "hard"),
        _packet("gpt", "easy"),
        _packet("gpt", "hard"),
    ]


def _main_rows() -> list[dict[str, object]]:
    return [
        _response_row("claude", "easy", True),
        _response_row("claude", "hard", False),
        _response_row("gpt", "easy", True),
        _response_row("gpt", "hard", False),
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_importer_writes_recorded_response_json_without_labels(tmp_path: Path) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    responses_path = tmp_path / "responses.json"
    fixture_path = tmp_path / "fixture.json"
    output_path = tmp_path / "recorded" / "responses.json"
    artifact_dir = tmp_path / "artifact"
    rows = _main_rows() + [
        _response_row("claude", "easy", True, run_type="stability", run_id="rerun-1"),
        _response_row("claude", "hard", False, run_type="stability", run_id="rerun-1"),
        _response_row("claude", "easy", True, run_type="stability", run_id="rerun-2"),
        _response_row("claude", "hard", False, run_type="stability", run_id="rerun-2"),
        _response_row("gpt", "easy", True, run_type="stability", run_id="rerun-1"),
        _response_row("gpt", "hard", False, run_type="stability", run_id="rerun-1"),
        _response_row("gpt", "easy", True, run_type="stability", run_id="rerun-2"),
        _response_row("gpt", "hard", False, run_type="stability", run_id="rerun-2"),
    ]
    _write_json(packets_path, _packets())
    _write_json(responses_path, rows)
    _write_json(
        fixture_path,
        [_fixture_row("easy", True, "easy"), _fixture_row("hard", False, "hard")],
    )

    exit_code, payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        output_path,
    )

    recorded = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload == {
        "ok": True,
        "output_path": str(output_path),
        "model_run_count": 2,
        "stability_model_count": 2,
        "response_count": 12,
        "errors": [],
    }
    assert [run["model_id"] for run in recorded["model_runs"]] == ["claude", "gpt"]
    assert recorded["model_runs"][0]["responses"] == {
        "easy": _response(True),
        "hard": _response(False),
    }
    assert len(recorded["stability_runs_by_model_id"]["claude"]) == 2
    assert "expected_supports" not in output_path.read_text(encoding="utf-8")

    artifact_result = subprocess.run(
        [
            sys.executable,
            str(ARTIFACT_SCRIPT),
            str(fixture_path),
            str(output_path),
            str(artifact_dir),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    artifact_payload = json.loads(artifact_result.stdout)
    assert artifact_result.returncode == 0
    assert artifact_payload["ok"] is True
    assert (artifact_dir / "claim_evidence_result.json").exists()


def test_importer_accepts_jsonl_packet_and_response_rows(tmp_path: Path) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    output_path = tmp_path / "responses-out.json"
    packets_path.write_text(
        "\n".join(json.dumps(row) for row in _packets()) + "\n",
        encoding="utf-8",
    )
    responses_path.write_text(
        "\n".join(json.dumps(row) for row in _main_rows()) + "\n",
        encoding="utf-8",
    )

    exit_code, payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        output_path,
    )

    assert exit_code == 0
    assert payload["model_run_count"] == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["model_runs"][1][
        "model_id"
    ] == "gpt"


def test_importer_rejects_missing_main_coverage_without_writing(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    responses_path = tmp_path / "responses.json"
    output_path = tmp_path / "responses-out.json"
    _write_json(packets_path, _packets())
    _write_json(responses_path, _main_rows()[:-1])

    exit_code, payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        output_path,
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["errors"] == ["missing main response for gpt/hard"]
    assert output_path.exists() is False


def test_importer_rejects_duplicate_main_and_stability_rows(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    responses_path = tmp_path / "responses.json"
    output_path = tmp_path / "responses-out.json"
    rows = _main_rows()
    rows.append(_response_row("claude", "easy", True))
    rows.append(_response_row("gpt", "easy", True, run_type="stability", run_id="r1"))
    rows.append(_response_row("gpt", "easy", True, run_type="stability", run_id="r1"))
    _write_json(packets_path, _packets())
    _write_json(responses_path, rows)

    exit_code, payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        output_path,
    )

    assert exit_code == 1
    assert payload["errors"] == [
        "response row 5: duplicate main row: claude/easy",
        "response row 7: duplicate stability row: gpt/r1/easy",
    ]
    assert output_path.exists() is False


def test_importer_rejects_unmatched_rows_and_invalid_responses(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    responses_path = tmp_path / "responses.json"
    output_path = tmp_path / "responses-out.json"
    invalid = _main_rows()
    invalid[0] = _response_row("unknown", "easy", True)
    invalid[1]["response"] = {
        "supports": "yes",
        "confidence": 6,
        "reason": " ",
        "extra": True,
    }
    _write_json(packets_path, _packets())
    _write_json(responses_path, invalid)

    exit_code, payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        output_path,
    )

    assert exit_code == 1
    assert payload["errors"] == [
        "response row 1: no exported packet for unknown/easy/verify_claim_evidence.v1",
        "response row 2.response unexpected fields: extra",
        "response row 2.response supports missing",
        "response row 2.response confidence must be an integer from 1 to 5",
        "response row 2.response reason missing",
        "missing main response for claude/easy",
        "missing main response for claude/hard",
    ]
    assert output_path.exists() is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_supports", True),
        ("claim_text", "contaminated claim text"),
        ("evidence_quote", "contaminated evidence quote"),
        ("claim_id", "claim-easy"),
        ("source_id", "source-easy"),
        ("difficulty", "easy"),
    ],
)
def test_importer_rejects_fixture_fields_in_returned_response_rows(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    responses_path = tmp_path / "responses.json"
    output_path = tmp_path / "responses-out.json"
    contaminated_rows = _main_rows()
    contaminated_rows[0][field] = value
    _write_json(packets_path, _packets())
    _write_json(responses_path, contaminated_rows)

    exit_code, payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        output_path,
    )

    assert exit_code == 1
    assert payload["errors"] == [
        f"response row 1: response row carries fixture/label fields: {field}",
        "missing main response for claude/easy",
    ]
    assert output_path.exists() is False


def test_importer_rejects_non_utf8_input_and_symlink_output(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    responses_path = tmp_path / "responses.json"
    output_path = tmp_path / "responses-out.json"
    external_path = tmp_path / "external.json"
    _write_json(packets_path, _packets())
    responses_path.write_bytes(b"\xff\xfe")

    read_exit, read_payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        output_path,
    )

    assert read_exit == 2
    assert len(read_payload["errors"]) == 1
    assert read_payload["errors"][0].startswith(
        f"responses file could not be read: {responses_path}: "
    )
    assert output_path.exists() is False

    _write_json(responses_path, _main_rows())
    external_path.write_text("external stays intact", encoding="utf-8")
    output_path.symlink_to(external_path)

    write_exit, write_payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        output_path,
    )

    assert write_exit == 2
    assert write_payload["errors"] == [f"output path is a symlink: {output_path}"]
    assert external_path.read_text(encoding="utf-8") == "external stays intact"


def test_importer_rejects_output_path_matching_input_path(tmp_path: Path) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    responses_path = tmp_path / "responses.json"
    _write_json(packets_path, _packets())
    _write_json(responses_path, _main_rows())
    original_responses = responses_path.read_text(encoding="utf-8")

    response_exit, response_payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        responses_path,
    )
    packet_exit, packet_payload = cli.import_prompt_responses_from_files(
        packets_path,
        responses_path,
        packets_path,
    )

    assert response_exit == 2
    assert response_payload["errors"] == ["output path must differ from responses path"]
    assert packet_exit == 2
    assert packet_payload["errors"] == ["output path must differ from packets path"]
    assert responses_path.read_text(encoding="utf-8") == original_responses


def test_direct_script_invocation_prints_json_envelope(tmp_path: Path) -> None:
    packets_path = tmp_path / "packets.json"
    responses_path = tmp_path / "responses.json"
    output_path = tmp_path / "responses-out.json"
    _write_json(packets_path, _packets())
    _write_json(responses_path, _main_rows())

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(packets_path),
            str(responses_path),
            str(output_path),
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
    assert output["response_count"] == 4
    assert output_path.exists()
