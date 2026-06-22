from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_deflection_pii_recall.py"
BUILD_SCRIPT = ROOT / "scripts" / "build_deflection_pii_surrogate_eval_corpus.py"
TINY_FIXTURE = (
    ROOT
    / "docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json"
)
SPEC = importlib.util.spec_from_file_location("score_deflection_pii_recall", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLI
SPEC.loader.exec_module(CLI)
BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_deflection_pii_surrogate_eval_corpus_for_score_tests",
    BUILD_SCRIPT,
)
assert BUILD_SPEC is not None
assert BUILD_SPEC.loader is not None
BUILD_CLI = importlib.util.module_from_spec(BUILD_SPEC)
sys.modules[BUILD_SPEC.name] = BUILD_CLI
BUILD_SPEC.loader.exec_module(BUILD_CLI)


def _tiny_corpus() -> dict:
    return json.loads(TINY_FIXTURE.read_text(encoding="utf-8"))


def test_review_bundle_corpus_filename_matches_builder_bundle_artifact() -> None:
    assert CLI.REVIEW_BUNDLE_CORPUS_NAME == BUILD_CLI.REVIEW_BUNDLE_ARTIFACT_NAME
    assert CLI.REVIEW_BUNDLE_MANIFEST_NAME == BUILD_CLI.REVIEW_BUNDLE_MANIFEST_NAME
    assert (
        CLI.REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION
        == BUILD_CLI.REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION
    )


def test_review_bundle_mode_writes_score_artifacts_without_span_echo(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / CLI.REVIEW_BUNDLE_CORPUS_NAME).write_text(
        TINY_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manifest_path = bundle_dir / CLI.REVIEW_BUNDLE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps({
            "schema_version": CLI.REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "status": "ok",
            "files": {
                "source_intake_summary": {
                    "path": BUILD_CLI.REVIEW_BUNDLE_SUMMARY_NAME,
                    "present": True,
                    "ok": True,
                    "schema_version": "deflection_pii_source_intake_summary.v1",
                },
                "surrogate_eval_corpus": {
                    "path": CLI.REVIEW_BUNDLE_CORPUS_NAME,
                    "present": True,
                    "schema_version": "deflection_pii_eval_corpus.v1",
                    "ticket_count": 3,
                    "label_count": 11,
                },
            },
        }),
        encoding="utf-8",
    )

    assert CLI.main(["--review-bundle-dir", str(bundle_dir)]) == 0

    score_path = bundle_dir / CLI.REVIEW_BUNDLE_SCORE_NAME
    markdown_path = bundle_dir / CLI.REVIEW_BUNDLE_SCORE_MARKDOWN_NAME
    summary = json.loads(score_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rendered = json.dumps(summary, sort_keys=True) + markdown + json.dumps(manifest, sort_keys=True)

    assert summary["status"] == "ok"
    assert summary["schema_version"] == "deflection_pii_recall_score.v1"
    assert summary["input"]["ticket_count"] == 3
    assert manifest["schema_version"] == CLI.REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["score_status"] == "ok"
    assert manifest["files"]["surrogate_eval_corpus"]["path"] == CLI.REVIEW_BUNDLE_CORPUS_NAME
    assert manifest["files"]["recall_score"] == {
        "blocking_error_codes": [],
        "headline": {
            "deferred_open_set_name_leak_count": 1,
            "free_high_severity_gate_eligible_leak_count": 0,
            "free_high_severity_leak_count": 1,
        },
        "path": CLI.REVIEW_BUNDLE_SCORE_NAME,
        "present": True,
        "schema_version": CLI.SCORE_SCHEMA_VERSION,
        "status": "ok",
    }
    assert manifest["files"]["recall_score_markdown"] == {
        "path": CLI.REVIEW_BUNDLE_SCORE_MARKDOWN_NAME,
        "present": True,
    }
    assert "# Deflection PII Recall Advisory" in markdown
    assert str(bundle_dir) not in rendered
    for raw in (
        "Maya Chen",
        "Jordan Lee",
        "Taylor Brooks",
        "Mary Jane Watson",
        "Watson",
    ):
        assert raw not in rendered


def test_review_bundle_mode_rejects_split_output_flags(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        CLI._parse_args([
            "--review-bundle-dir",
            str(tmp_path / "bundle"),
            "--output",
            str(tmp_path / "score.json"),
        ])

    assert exc.value.code == 2


def test_review_bundle_mode_rejects_existing_file_path_without_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_path = tmp_path / "bundle-file"
    bundle_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        CLI._parse_args(["--review-bundle-dir", str(bundle_path)])

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "--review-bundle-dir must be a directory" in captured.err
    assert str(bundle_path) not in captured.err


def test_review_bundle_mode_missing_corpus_writes_sanitized_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = tmp_path / "bundle"

    assert CLI.main(["--review-bundle-dir", str(bundle_dir)]) == 1

    captured = capsys.readouterr()
    score_path = bundle_dir / CLI.REVIEW_BUNDLE_SCORE_NAME
    markdown_path = bundle_dir / CLI.REVIEW_BUNDLE_SCORE_MARKDOWN_NAME
    manifest_path = bundle_dir / CLI.REVIEW_BUNDLE_MANIFEST_NAME
    summary = json.loads(score_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rendered = json.dumps(summary, sort_keys=True) + markdown + json.dumps(manifest, sort_keys=True) + captured.err

    assert summary["status"] == "failed"
    assert summary["blocking_error_codes"] == ["corpus_load_failed"]
    assert manifest["status"] == "unknown"
    assert manifest["score_status"] == "failed"
    assert manifest["files"]["recall_score"]["blocking_error_codes"] == ["corpus_load_failed"]
    assert manifest["files"]["recall_score_markdown"] == {
        "path": CLI.REVIEW_BUNDLE_SCORE_MARKDOWN_NAME,
        "present": True,
    }
    assert "corpus_load_failed" in markdown
    assert str(bundle_dir) not in rendered
    assert CLI.REVIEW_BUNDLE_CORPUS_NAME not in rendered


def test_review_bundle_mode_preserves_blocked_build_errors_on_score_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    manifest_path = bundle_dir / CLI.REVIEW_BUNDLE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps({
            "schema_version": CLI.REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "status": "blocked",
            "blocking_error_codes": [
                "label_span_not_found",
                "alice.baker@example.com",
            ],
            "files": {
                "source_intake_summary": {
                    "path": BUILD_CLI.REVIEW_BUNDLE_SUMMARY_NAME,
                    "present": True,
                    "ok": False,
                    "schema_version": "deflection_pii_source_intake_summary.v1",
                },
                "source_intake_markdown": {
                    "path": BUILD_CLI.REVIEW_BUNDLE_MARKDOWN_NAME,
                    "present": True,
                },
                "surrogate_eval_corpus": {
                    "path": CLI.REVIEW_BUNDLE_CORPUS_NAME,
                    "present": False,
                },
            },
        }),
        encoding="utf-8",
    )

    assert CLI.main(["--review-bundle-dir", str(bundle_dir)]) == 1

    captured = capsys.readouterr()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rendered = json.dumps(manifest, sort_keys=True) + captured.err

    assert manifest["status"] == "blocked"
    assert manifest["blocking_error_codes"] == ["label_span_not_found"]
    assert manifest["files"]["surrogate_eval_corpus"] == {
        "path": CLI.REVIEW_BUNDLE_CORPUS_NAME,
        "present": False,
    }
    assert manifest["files"]["recall_score"]["blocking_error_codes"] == ["corpus_load_failed"]
    assert "label_span_not_found" in rendered
    assert "corpus_load_failed" in rendered
    assert "alice.baker@example.com" not in rendered


def test_review_bundle_mode_clears_stale_corpus_inventory_on_load_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    manifest_path = bundle_dir / CLI.REVIEW_BUNDLE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps({
            "schema_version": CLI.REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "status": "ok",
            "blocking_error_codes": [],
            "files": {
                "source_intake_summary": {
                    "path": BUILD_CLI.REVIEW_BUNDLE_SUMMARY_NAME,
                    "present": True,
                    "ok": True,
                    "schema_version": "deflection_pii_source_intake_summary.v1",
                },
                "surrogate_eval_corpus": {
                    "path": CLI.REVIEW_BUNDLE_CORPUS_NAME,
                    "present": True,
                    "schema_version": "deflection_pii_eval_corpus.v1",
                    "ticket_count": 3,
                    "label_count": 11,
                },
            },
        }),
        encoding="utf-8",
    )

    assert CLI.main(["--review-bundle-dir", str(bundle_dir)]) == 1

    captured = capsys.readouterr()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rendered = json.dumps(manifest, sort_keys=True) + captured.err
    corpus_entry = manifest["files"]["surrogate_eval_corpus"]

    assert manifest["status"] == "blocked"
    assert manifest["score_status"] == "failed"
    assert corpus_entry == {
        "path": CLI.REVIEW_BUNDLE_CORPUS_NAME,
        "present": False,
    }
    assert manifest["files"]["recall_score"]["blocking_error_codes"] == ["corpus_load_failed"]
    assert "ticket_count" not in corpus_entry
    assert "label_count" not in corpus_entry
    assert "corpus_load_failed" in rendered


def test_review_bundle_mode_non_utf8_corpus_writes_sanitized_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / CLI.REVIEW_BUNDLE_CORPUS_NAME).write_bytes(b"\xff\xfe\x00")

    assert CLI.main(["--review-bundle-dir", str(bundle_dir)]) == 1

    captured = capsys.readouterr()
    score_path = bundle_dir / CLI.REVIEW_BUNDLE_SCORE_NAME
    markdown_path = bundle_dir / CLI.REVIEW_BUNDLE_SCORE_MARKDOWN_NAME
    summary = json.loads(score_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    rendered = json.dumps(summary, sort_keys=True) + markdown + captured.err

    assert summary["status"] == "failed"
    assert summary["blocking_error_codes"] == ["corpus_load_failed"]
    assert "corpus_load_failed" in markdown
    assert "UnicodeDecodeError" not in rendered
    assert str(bundle_dir) not in rendered
    assert "deflection-pii-surrogate-eval-corpus.json" not in rendered


def test_tiny_fixture_scores_all_surfaces_without_echoing_spans() -> None:
    summary = CLI.score_corpus(_tiny_corpus())

    assert summary["status"] == "ok"
    assert summary["input"] == {
        "schema_version": "deflection_pii_eval_corpus.v1",
        "ticket_count": 3,
        "label_count": 11,
        "must_survive_count": 3,
    }
    paid_pdf = summary["surface_generation"]["paid_pdf"]
    if paid_pdf["rendered"]:
        assert paid_pdf["skipped"] is False
        assert paid_pdf["byte_count"] > 1000
    else:
        assert paid_pdf == {
            "rendered": False,
            "skipped": True,
            "skip_reason": "missing_optional_dependency:fpdf",
            "byte_count": 0,
            "scored_text_source": None,
        }
    assert set(summary["surfaces"]) == {
        "free_snapshot",
        "free_teaser",
        "paid_artifact",
        "paid_pdf",
    }
    assert summary["surfaces"]["free_snapshot"]["email"]["expected"] == 1
    assert summary["surfaces"]["paid_artifact"]["ssn"] == {
        "expected": 1,
        "redacted": 1,
        "leaks": 0,
        "recall": 1.0,
    }
    assert summary["surfaces"]["paid_artifact"]["payment_card"] == {
        "expected": 1,
        "redacted": 1,
        "leaks": 0,
        "recall": 1.0,
    }
    assert summary["surfaces"]["paid_artifact"]["dob"] == {
        "expected": 1,
        "redacted": 1,
        "leaks": 0,
        "recall": 1.0,
    }
    assert set(summary["person_name"]) == {"cue_less", "cue_prefixed"}
    assert summary["headline"] == {
        "deferred_open_set_name_leak_count": 1,
        "free_high_severity_gate_eligible_leak_count": 0,
        "free_high_severity_gate_eligible_pass": True,
        "free_high_severity_leak_count": 1,
        "free_high_severity_pass": False,
    }
    if paid_pdf["rendered"]:
        assert summary["person_name"]["cue_prefixed"] == {
            "expected": 7,
            "redacted": 7,
            "leaks": 0,
            "recall": 1.0,
        }
    else:
        assert summary["person_name"]["cue_prefixed"] == {
            "expected": 4,
            "redacted": 4,
            "leaks": 0,
            "recall": 1.0,
        }
    assert summary["person_name"]["cue_less"]["leaks"] > 0
    assert summary["must_survive"]["violation_count"] == 0
    assert summary["leak_samples"]
    assert all(
        sample["surrogate_id"] != "person_name-002"
        for sample in summary["leak_samples"]
    )
    assert all(
        sample["surrogate_id"] not in {"ssn-001", "payment_card-001"}
        for sample in summary["leak_samples"]
    )
    assert all(
        sample["surrogate_id"] != "person_name-003"
        for sample in summary["leak_samples"]
    )
    assert all(
        sample["surrogate_id"] != "person_name-004"
        for sample in summary["leak_samples"]
    )
    assert all(
        sample["surrogate_id"] != "dob-001"
        for sample in summary["leak_samples"]
    )
    assert all(
        sample["leak_kind"] != "partial_name_token"
        for sample in summary["leak_samples"]
    )
    assert all("span" not in sample for sample in summary["leak_samples"])
    assert all("token" not in sample for sample in summary["leak_samples"])
    rendered_samples = json.dumps(summary["leak_samples"], sort_keys=True)
    assert "Mary" not in rendered_samples
    assert "Jane" not in rendered_samples
    assert "Watson" not in rendered_samples


def test_partial_name_token_detection_reports_token_residue() -> None:
    leak_kind = CLI._label_leak_kind(
        label={"class": "person_name", "span": "Mary Jane Watson"},
        baseline_text="Customer Mary Jane Watson Report plan was upgraded.",
        scrubbed_text="Customer [redacted-name] Watson Report plan was upgraded.",
    )

    assert leak_kind == "partial_name_token"


def test_partial_name_token_detection_reports_two_letter_surname_residue() -> None:
    leak_kind = CLI._label_leak_kind(
        label={"class": "person_name", "span": "Amy Rae Li"},
        baseline_text="Customer Amy Rae Li Report plan was upgraded.",
        scrubbed_text="Customer [redacted-name] Li Report plan was upgraded.",
    )

    assert leak_kind == "partial_name_token"


def test_partial_name_token_detection_ignores_common_two_letter_words() -> None:
    leak_kind = CLI._label_leak_kind(
        label={"class": "person_name", "span": "Amy Rae To"},
        baseline_text="Customer Amy Rae To Report plan was upgraded.",
        scrubbed_text="Customer [redacted-name] To Report plan was upgraded.",
    )

    assert leak_kind == ""


@pytest.mark.parametrize(
    "scrubbed_text",
    (
        "Customer [redacted-name] Li-ion battery plan was upgraded.",
        "Customer [redacted-name] Li's plan was upgraded.",
        "Customer [redacted-name] O'Li plan was upgraded.",
        "Customer [redacted-name] Link plan was upgraded.",
    ),
)
def test_partial_name_token_detection_ignores_short_token_compounds(
    scrubbed_text: str,
) -> None:
    leak_kind = CLI._label_leak_kind(
        label={"class": "person_name", "span": "Amy Rae Li"},
        baseline_text="Customer Amy Rae Li Report plan was upgraded.",
        scrubbed_text=scrubbed_text,
    )

    assert leak_kind == ""


def test_resolved_partial_name_token_does_not_cross_contaminate_tickets() -> None:
    corpus = _tiny_corpus()
    corpus["tickets"].append(
        {
            "fields": {
                "agent_reply": "Customer Alice Watson Premium was upgraded.",
                "customer_message": "Premium update completed.",
                "source_id": "ticket-eval-safe-003",
                "subject": "Premium update",
            },
            "labels": [
                {
                    "class": "person_name",
                    "name_subtype": "cue_prefixed",
                    "origin_field": "agent_reply",
                    "severity": "high",
                    "span": "Alice Watson",
                    "surrogate_id": "person_name-004",
                }
            ],
            "must_survive": [],
            "ticket_id": "pii-eval-003",
        }
    )

    summary = CLI.score_corpus(corpus)

    assert summary["status"] == "ok"
    assert [
        sample
        for sample in summary["leak_samples"]
        if sample["surrogate_id"] == "person_name-004"
    ] == []
    assert all(
        sample["surrogate_id"] != "person_name-003"
        for sample in summary["leak_samples"]
    )


def test_forced_leak_reports_surface_and_surrogate_without_span() -> None:
    corpus = _tiny_corpus()
    ticket = corpus["tickets"][0]
    ticket["fields"]["subject"] = "UNSCRUBBED_SENTINEL refund status"
    ticket["labels"].append(
        {
            "class": "custom_token",
            "origin_field": "subject",
            "severity": "low",
            "span": "UNSCRUBBED_SENTINEL",
            "surrogate_id": "custom-token-001",
        }
    )

    summary = CLI.score_corpus(corpus)

    matching = [
        sample
        for sample in summary["leak_samples"]
        if sample["surrogate_id"] == "custom-token-001"
    ]
    expected_surfaces = {
        "free_snapshot",
        "paid_artifact",
    }
    if summary["surface_generation"]["paid_pdf"]["rendered"]:
        expected_surfaces.add("paid_pdf")
    assert {sample["surface"] for sample in matching} >= expected_surfaces
    assert all("span" not in sample for sample in matching)
    assert "UNSCRUBBED_SENTINEL" not in json.dumps(
        summary["leak_samples"],
        sort_keys=True,
    )


def test_gate_eligible_headline_excludes_only_deferred_open_set_names() -> None:
    corpus = _tiny_corpus()
    ticket = corpus["tickets"][0]
    ticket["fields"]["subject"] = (
        f"{ticket['fields']['subject']} UNSCRUBBED_HIGH_SEV"
    )
    ticket["labels"].append(
        {
            "class": "custom_token",
            "origin_field": "subject",
            "severity": "high",
            "span": "UNSCRUBBED_HIGH_SEV",
            "surrogate_id": "custom-high-sev-001",
        }
    )

    summary = CLI.score_corpus(corpus)

    assert summary["headline"] == {
        "deferred_open_set_name_leak_count": 1,
        "free_high_severity_gate_eligible_leak_count": 1,
        "free_high_severity_gate_eligible_pass": False,
        "free_high_severity_leak_count": 2,
        "free_high_severity_pass": False,
    }
    assert {
        sample["surrogate_id"]
        for sample in summary["leak_samples"]
    } >= {"custom-high-sev-001", "person_name-001"}
    assert "UNSCRUBBED_HIGH_SEV" not in json.dumps(
        summary["leak_samples"],
        sort_keys=True,
    )


def test_must_survive_violation_reports_precision_loss() -> None:
    corpus = _tiny_corpus()
    ticket = corpus["tickets"][0]
    ticket["must_survive"].append(
        {
            "origin_field": "customer_message",
            "reason": "forced_email_precision_probe",
            "span": "alex.rivera@example.test",
        }
    )

    summary = CLI.score_corpus(corpus)

    assert summary["must_survive"]["violation_count"] > 0
    assert {
        violation["reason"]
        for violation in summary["must_survive"]["violations"]
    } >= {"forced_email_precision_probe"}


def test_invalid_corpus_fails_with_safe_error_codes() -> None:
    summary = CLI.score_corpus({"schema_version": "wrong", "tickets": []})

    assert summary["status"] == "failed"
    assert set(summary["blocking_error_codes"]) == {
        "corpus_empty_tickets",
        "corpus_schema_version_mismatch",
    }
    assert "alex.rivera@example.test" not in json.dumps(summary, sort_keys=True)


@pytest.mark.parametrize(
    ("bad_label", "error_code"),
    (
        (
            {
                "class": "email",
                "origin_field": "customer_message",
                "severity": "high",
                "surrogate_id": "email-bad",
            },
            "label_missing_span",
        ),
        (
            {
                "class": "email",
                "origin_field": "customer_message",
                "severity": "high",
                "span": "",
                "surrogate_id": "email-bad",
            },
            "label_missing_span",
        ),
        (
            {
                "class": "email",
                "origin_field": "customer_message",
                "severity": "high",
                "span": 12345,
                "surrogate_id": "email-bad",
            },
            "label_missing_span",
        ),
        ("not-a-label-object", "label_not_object"),
    ),
)
def test_malformed_label_span_fails_closed_without_scoring(
    bad_label: object,
    error_code: str,
) -> None:
    corpus = _tiny_corpus()
    corpus["tickets"][0]["labels"].append(bad_label)

    summary = CLI.score_corpus(corpus)

    assert summary["status"] == "failed"
    assert summary["blocking_error_codes"] == [error_code]
    assert summary["errors"] == [
        {
            "code": error_code,
            "ticket_index": 1,
            "label_index": 10,
        }
    ]
    assert "email-bad" not in json.dumps(summary, sort_keys=True)


@pytest.mark.parametrize(
    ("bad_record", "error_code"),
    (
        (
            {
                "origin_field": "customer_message",
                "reason": "bad_precision_record",
            },
            "must_survive_missing_span",
        ),
        (
            {
                "origin_field": "customer_message",
                "reason": "bad_precision_record",
                "span": "   ",
            },
            "must_survive_missing_span",
        ),
        (
            {
                "origin_field": "customer_message",
                "reason": "bad_precision_record",
                "span": 12345,
            },
            "must_survive_missing_span",
        ),
        ("not-a-must-survive-object", "must_survive_not_object"),
    ),
)
def test_malformed_must_survive_span_fails_closed_without_scoring(
    bad_record: object,
    error_code: str,
) -> None:
    corpus = _tiny_corpus()
    corpus["tickets"][0]["must_survive"].append(bad_record)

    summary = CLI.score_corpus(corpus)

    assert summary["status"] == "failed"
    assert summary["blocking_error_codes"] == [error_code]
    assert summary["errors"] == [
        {
            "code": error_code,
            "ticket_index": 1,
            "record_index": 4,
        }
    ]
    assert "bad_precision_record" not in json.dumps(summary, sort_keys=True)


def test_missing_pdf_dependency_skips_paid_pdf_scoring(monkeypatch) -> None:
    error = ModuleNotFoundError("No module named 'fpdf'")
    error.name = "fpdf"
    monkeypatch.setattr(CLI, "_PDF_RENDERER_IMPORT_ERROR", error)
    monkeypatch.setattr(CLI, "_artifact_report_model_pdf_markdown", None)
    monkeypatch.setattr(CLI, "render_deflection_full_report_pdf", None)

    summary = CLI.score_corpus(_tiny_corpus())

    assert summary["status"] == "ok"
    assert summary["surface_generation"]["paid_pdf"] == {
        "rendered": False,
        "skipped": True,
        "skip_reason": "missing_optional_dependency:fpdf",
        "byte_count": 0,
        "scored_text_source": None,
    }
    assert summary["surfaces"]["paid_pdf"] == {}


def test_cli_writes_markdown_summary_without_raw_spans(tmp_path: Path) -> None:
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"

    exit_code = CLI.main([
        "--output",
        str(json_path),
        "--markdown-output",
        str(markdown_path),
    ])

    assert exit_code == 0
    written = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert written["headline"]["free_high_severity_gate_eligible_leak_count"] == 0
    assert "# Deflection PII Recall Advisory" in markdown
    assert "| Gate-eligible free high-severity leaks | 0 |" in markdown
    assert "| Deferred open-set name leaks | 1 |" in markdown
    cue_less_expected = (
        4 if written["surface_generation"]["paid_pdf"]["rendered"] else 3
    )
    assert f"| cue_less | {cue_less_expected} | 0 | {cue_less_expected} | 0 |" in markdown
    assert "| Violation count | 0 |" in markdown
    assert "person_name-001" in markdown
    for raw_value in (
        "Maya Chen",
        "Jordan Lee",
        "Mary Jane Watson",
        "Amy Rae Li",
        "alex.rivera@example.test",
        "555-010-4301",
        "1990-04-17",
        "123-45-6789",
        "4111 1111 1111 1111",
        "CVE-2021-44228",
        "ISO 27001",
        "ticket-eval-safe-001",
    ):
        assert raw_value not in markdown


def test_cli_writes_failure_summary_and_returns_nonzero(tmp_path: Path) -> None:
    corpus_path = tmp_path / "bad.json"
    output_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    corpus_path.write_text('{"schema_version": "wrong", "tickets": []}', encoding="utf-8")

    exit_code = CLI.main([
        "--corpus",
        str(corpus_path),
        "--output",
        str(output_path),
        "--markdown-output",
        str(markdown_path),
    ])

    assert exit_code == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "failed"
    assert written["blocking_error_codes"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Blocking Errors" in markdown
    assert "corpus_empty_tickets" in markdown
