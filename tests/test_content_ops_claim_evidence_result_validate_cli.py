from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from extracted_content_pipeline.claim_evidence_benchmark import (
    BenchmarkThresholds,
    ClaimEvidenceModelRun,
    ClaimEvidenceResponse,
    ClaimEvidenceRunRow,
    ClaimEvidenceTriple,
    EASY,
    HARD,
    VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION,
    build_claim_evidence_result_artifact,
    claim_evidence_result_artifact_files,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_content_ops_claim_evidence_result.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "validate_content_ops_claim_evidence_result",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response(supports: bool, confidence: int = 5) -> ClaimEvidenceResponse:
    return ClaimEvidenceResponse(
        supports=supports,
        confidence=confidence,
        reason="the evidence quote directly supports the judgment",
    )


def _run(model_id: str) -> ClaimEvidenceModelRun:
    return ClaimEvidenceModelRun(
        model_id=model_id,
        rows=(
            ClaimEvidenceRunRow(
                model_id=model_id,
                triple_id="easy",
                contract_version=VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION,
                response=_response(True),
                errors=(),
            ),
            ClaimEvidenceRunRow(
                model_id=model_id,
                triple_id="hard",
                contract_version=VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION,
                response=_response(False),
                errors=(),
            ),
        ),
        errors=(),
    )


def _artifact_json(*, passing: bool = True) -> str:
    thresholds = (
        BenchmarkThresholds(
            easy_accuracy_min=1.0,
            hard_accuracy_min=1.0,
            inter_model_agreement_min=1.0,
            intra_model_stability_min=1.0,
            high_confidence_accuracy_min=0.90,
        )
        if passing
        else BenchmarkThresholds(
            easy_accuracy_min=1.0,
            hard_accuracy_min=1.0,
            inter_model_agreement_min=1.0,
            intra_model_stability_min=1.0,
            high_confidence_accuracy_min=1.0,
        )
    )
    gpt_run = _run("gpt")
    claude_run = _run("claude")
    artifact = build_claim_evidence_result_artifact(
        (
            ClaimEvidenceTriple(
                triple_id="easy",
                claim_id="claim-easy",
                claim_text="Claim statement easy",
                evidence_quote="evidence easy",
                source_id="source-easy",
                expected_supports=True,
                difficulty=EASY,
            ),
            ClaimEvidenceTriple(
                triple_id="hard",
                claim_id="claim-hard",
                claim_text="Claim statement hard",
                evidence_quote="evidence hard",
                source_id="source-hard",
                expected_supports=False,
                difficulty=HARD,
            ),
        ),
        (gpt_run, claude_run),
        stability_runs_by_model_id={
            "gpt": (gpt_run, gpt_run),
            "claude": (claude_run, claude_run),
        },
        thresholds=thresholds,
    )
    return claim_evidence_result_artifact_files(artifact)[0].content


def test_validate_cli_accepts_saved_go_result_artifact(tmp_path: Path) -> None:
    cli = _load_cli_module()
    result_path = tmp_path / "claim_evidence_result.json"
    result_path.write_text(_artifact_json(), encoding="utf-8")

    exit_code, payload = cli.validate_claim_evidence_result_file(result_path)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["go_no_go"] == "go"
    assert payload["artifact_errors"] == []
    assert payload["verdict_failures"] == []
    assert payload["markdown_written"] is False


def test_validate_cli_returns_no_go_for_saved_threshold_failure(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    result_path = tmp_path / "claim_evidence_result.json"
    result_path.write_text(_artifact_json(passing=False), encoding="utf-8")

    exit_code, payload = cli.validate_claim_evidence_result_file(result_path)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["go_no_go"] == "no_go"
    assert payload["artifact_errors"] == []
    assert payload["verdict_failures"] == [
        "claude: high-confidence accuracy 1.000 not above 1.000",
        "gpt: high-confidence accuracy 1.000 not above 1.000",
    ]


def test_validate_cli_fails_closed_on_malformed_result_json(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    result_path = tmp_path / "claim_evidence_result.json"
    result_path.write_text("{", encoding="utf-8")

    exit_code, payload = cli.validate_claim_evidence_result_file(result_path)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["artifact_errors"] == [
        "result artifact json is malformed: Expecting property name enclosed in double quotes"
    ]
    assert payload["verdict_failures"] == payload["artifact_errors"]


def test_validate_cli_reports_missing_file_as_io_error(tmp_path: Path) -> None:
    cli = _load_cli_module()
    result_path = tmp_path / "missing.json"

    exit_code, payload = cli.validate_claim_evidence_result_file(result_path)

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["errors"] == [f"result artifact file not found: {result_path}"]
    assert payload["artifact_errors"] == []


def test_validate_cli_writes_markdown_report(tmp_path: Path) -> None:
    cli = _load_cli_module()
    result_path = tmp_path / "claim_evidence_result.json"
    markdown_path = tmp_path / "nested" / "claim_evidence_result.md"
    result_path.write_text(_artifact_json(), encoding="utf-8")

    exit_code, payload = cli.validate_claim_evidence_result_file(
        result_path,
        markdown_output=markdown_path,
    )

    assert exit_code == 0
    assert payload["markdown_written"] is True
    assert payload["markdown_output"] == str(markdown_path)
    assert "# Claim Evidence Benchmark Result" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_validate_cli_rejects_markdown_directory_output(tmp_path: Path) -> None:
    cli = _load_cli_module()
    result_path = tmp_path / "claim_evidence_result.json"
    markdown_path = tmp_path / "markdown-dir"
    result_path.write_text(_artifact_json(), encoding="utf-8")
    markdown_path.mkdir()

    exit_code, payload = cli.validate_claim_evidence_result_file(
        result_path,
        markdown_output=markdown_path,
    )

    assert exit_code == 2
    assert payload["ok"] is True
    assert payload["markdown_written"] is False
    assert payload["errors"] == [f"markdown output path is a directory: {markdown_path}"]


def test_direct_script_invocation_prints_json_envelope(tmp_path: Path) -> None:
    result_path = tmp_path / "claim_evidence_result.json"
    markdown_path = tmp_path / "claim_evidence_result.md"
    result_path.write_text(_artifact_json(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(result_path),
            "--markdown-output",
            str(markdown_path),
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
    assert output["markdown_written"] is True
    assert markdown_path.exists()
