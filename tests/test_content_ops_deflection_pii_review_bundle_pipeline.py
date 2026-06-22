from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_deflection_pii_review_bundle_pipeline.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLI = _load_script(SCRIPT, "run_deflection_pii_review_bundle_pipeline")


RAW_TOKENS = (
    "Alice Baker",
    "alice.baker@example.com",
    "202-555-0188",
    "ORD-98765",
    "99 Real Street",
    "1977-06-05",
    "111-22-3333",
    "4242 4242 4242 4242",
)


def _valid_source() -> dict:
    return {
        "schema_version": "deflection_pii_labeled_source.v1",
        "records": [
            {
                "fields": {
                    "subject": "Refund for Alice Baker",
                    "customer_message": (
                        "Email alice.baker@example.com or call 202-555-0188. "
                        "Order ORD-98765 ships to 99 Real Street."
                    ),
                    "agent_reply": (
                        "Customer Alice Baker can keep CVE-2021-44228 and ISO 27001 "
                        "references in the report. DOB 1977-06-05 was removed."
                    ),
                    "private_note": "SSN 111-22-3333 and card 4242 4242 4242 4242.",
                    "source_id": "ticket-eval-safe-001",
                },
                "labels": [
                    {
                        "span": "Alice Baker",
                        "class": "person_name",
                        "origin_field": "subject",
                        "name_subtype": "cue_less",
                    },
                    {
                        "span": "alice.baker@example.com",
                        "class": "email",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "202-555-0188",
                        "class": "phone",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "ORD-98765",
                        "class": "order_id",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "99 Real Street",
                        "class": "street_address",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "Alice Baker",
                        "class": "person_name",
                        "origin_field": "agent_reply",
                        "name_subtype": "cue_prefixed",
                    },
                    {
                        "span": "1977-06-05",
                        "class": "dob",
                        "origin_field": "agent_reply",
                    },
                    {
                        "span": "111-22-3333",
                        "class": "ssn",
                        "origin_field": "private_note",
                    },
                    {
                        "span": "4242 4242 4242 4242",
                        "class": "payment_card",
                        "origin_field": "private_note",
                    },
                ],
                "must_survive": [
                    {
                        "span": "CVE-2021-44228",
                        "origin_field": "agent_reply",
                        "reason": "security_reference",
                    },
                    {
                        "span": "ISO 27001",
                        "origin_field": "agent_reply",
                        "reason": "compliance_reference",
                    },
                    {
                        "span": "ticket-eval-safe-001",
                        "origin_field": "source_id",
                        "reason": "tenant_source_id",
                    },
                ],
            }
        ],
    }


def _write_source(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pipeline_args(tmp_path: Path, source: Path, candidate: Path | None = None) -> list[str]:
    return [
        str(source),
        "--review-bundle-dir",
        str(tmp_path / "bundle"),
        "--candidate-output",
        str(candidate or tmp_path / "candidate" / "corpus.json"),
    ]


def _stderr_payload(capsys: pytest.CaptureFixture[str]) -> dict:
    captured = capsys.readouterr()
    assert captured.out == ""
    return json.loads(captured.err)


def _assert_no_raw_echo(text: str) -> None:
    for token in RAW_TOKENS:
        assert token not in text


def test_pipeline_builds_scores_and_promotes_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate" / "corpus.json"
    bundle_dir = tmp_path / "bundle"
    _write_source(source, _valid_source())

    assert CLI.main(_pipeline_args(tmp_path, source, candidate)) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["schema_version"] == CLI.PIPELINE_SCHEMA_VERSION
    assert payload["steps"] == {"build": "ok", "promote": "ok", "score": "ok"}
    assert payload["candidate_output"] == str(candidate)
    assert payload["ticket_count"] == 1
    assert payload["label_count"] == 9
    assert candidate.exists()
    assert (bundle_dir / CLI.build_cli.REVIEW_BUNDLE_MANIFEST_NAME).exists()
    assert (bundle_dir / CLI.score_cli.REVIEW_BUNDLE_SCORE_NAME).exists()
    assert (bundle_dir / CLI.score_cli.REVIEW_BUNDLE_SCORE_MARKDOWN_NAME).exists()
    manifest = json.loads(
        (bundle_dir / CLI.build_cli.REVIEW_BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "ok"
    assert manifest["score_status"] == "ok"
    rendered = captured.out + candidate.read_text(encoding="utf-8")
    _assert_no_raw_echo(rendered)


def test_pipeline_stops_before_score_and_promote_when_build_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate" / "corpus.json"
    invalid = _valid_source()
    invalid["records"][0]["labels"][0]["span"] = ""
    _write_source(source, invalid)

    assert CLI.main(_pipeline_args(tmp_path, source, candidate)) == 1

    payload = _stderr_payload(capsys)
    bundle_dir = tmp_path / "bundle"
    assert payload["failed_step"] == "build"
    assert "label_missing_span" in [error["code"] for error in payload["errors"]]
    assert not candidate.exists()
    assert not (bundle_dir / CLI.build_cli.REVIEW_BUNDLE_ARTIFACT_NAME).exists()
    assert not (bundle_dir / CLI.score_cli.REVIEW_BUNDLE_SCORE_NAME).exists()
    _assert_no_raw_echo(json.dumps(payload, sort_keys=True))


def test_pipeline_clears_stale_score_files_before_failed_rebuild(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    valid_source = tmp_path / "valid-source.json"
    invalid_source = tmp_path / "invalid-source.json"
    first_candidate = tmp_path / "candidate" / "first.json"
    second_candidate = tmp_path / "candidate" / "second.json"
    _write_source(valid_source, _valid_source())
    invalid = _valid_source()
    invalid["records"][0]["labels"][0]["span"] = ""
    _write_source(invalid_source, invalid)
    bundle_dir = tmp_path / "bundle"

    assert CLI.main(_pipeline_args(tmp_path, valid_source, first_candidate)) == 0
    capsys.readouterr()
    assert (bundle_dir / CLI.score_cli.REVIEW_BUNDLE_SCORE_NAME).exists()
    assert (bundle_dir / CLI.score_cli.REVIEW_BUNDLE_SCORE_MARKDOWN_NAME).exists()

    assert CLI.main(_pipeline_args(tmp_path, invalid_source, second_candidate)) == 1

    payload = _stderr_payload(capsys)
    assert payload["failed_step"] == "build"
    assert "label_missing_span" in [error["code"] for error in payload["errors"]]
    assert not (bundle_dir / CLI.score_cli.REVIEW_BUNDLE_SCORE_NAME).exists()
    assert not (bundle_dir / CLI.score_cli.REVIEW_BUNDLE_SCORE_MARKDOWN_NAME).exists()
    assert not second_candidate.exists()


def test_pipeline_rejects_candidate_output_same_as_input_under_force(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.json"
    source_payload = _valid_source()
    _write_source(source, source_payload)

    args = _pipeline_args(tmp_path, source, source)
    args.append("--force")
    assert CLI.main(args) == 1

    payload = _stderr_payload(capsys)
    assert payload["failed_step"] == "preflight"
    assert "candidate_output_same_as_input" in [
        error["code"] for error in payload["errors"]
    ]
    assert json.loads(source.read_text(encoding="utf-8")) == source_payload
    assert not (tmp_path / "bundle" / CLI.build_cli.REVIEW_BUNDLE_ARTIFACT_NAME).exists()


def test_pipeline_rejects_input_at_reserved_bundle_artifact_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source = bundle_dir / CLI.build_cli.REVIEW_BUNDLE_ARTIFACT_NAME
    candidate = tmp_path / "candidate" / "corpus.json"
    source_payload = _valid_source()
    _write_source(source, source_payload)

    assert CLI.main(_pipeline_args(tmp_path, source, candidate)) == 1

    payload = _stderr_payload(capsys)
    assert payload["failed_step"] == "preflight"
    assert "input_reserved_bundle_artifact" in [
        error["code"] for error in payload["errors"]
    ]
    assert json.loads(source.read_text(encoding="utf-8")) == source_payload
    assert not candidate.exists()


def test_pipeline_stops_before_promote_when_score_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate" / "corpus.json"
    _write_source(source, _valid_source())
    original_score = CLI.score_cli.main

    def corrupt_then_score(argv: list[str] | None = None) -> int:
        assert argv is not None
        bundle_dir = Path(argv[argv.index("--review-bundle-dir") + 1])
        (bundle_dir / CLI.score_cli.REVIEW_BUNDLE_CORPUS_NAME).write_text(
            "{not json",
            encoding="utf-8",
        )
        return original_score(argv)

    monkeypatch.setattr(CLI.score_cli, "main", corrupt_then_score)

    assert CLI.main(_pipeline_args(tmp_path, source, candidate)) == 1

    payload = _stderr_payload(capsys)
    bundle_dir = tmp_path / "bundle"
    assert payload["failed_step"] == "score"
    assert "corpus_load_failed" in [error["code"] for error in payload["errors"]]
    assert (bundle_dir / CLI.score_cli.REVIEW_BUNDLE_SCORE_NAME).exists()
    assert not candidate.exists()
    _assert_no_raw_echo(json.dumps(payload, sort_keys=True))


def test_pipeline_stops_on_promote_no_clobber(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate" / "corpus.json"
    _write_source(source, _valid_source())
    candidate.parent.mkdir(parents=True)
    candidate.write_text("already here\n", encoding="utf-8")

    assert CLI.main(_pipeline_args(tmp_path, source, candidate)) == 1

    payload = _stderr_payload(capsys)
    assert payload["failed_step"] == "promote"
    assert "output_exists" in [error["code"] for error in payload["errors"]]
    assert candidate.read_text(encoding="utf-8") == "already here\n"


def test_pipeline_force_can_replace_existing_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate" / "corpus.json"
    _write_source(source, _valid_source())
    candidate.parent.mkdir(parents=True)
    candidate.write_text("already here\n", encoding="utf-8")

    args = _pipeline_args(tmp_path, source, candidate)
    args.append("--force")
    assert CLI.main(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "already here" not in candidate.read_text(encoding="utf-8")
