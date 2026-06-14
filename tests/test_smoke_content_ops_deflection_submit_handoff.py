from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import urllib.error

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_deflection_submit_handoff.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_deflection_submit_handoff", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


SNAPSHOT = {
    "summary": {
        "generated": 2,
        "drafted_answer_count": 1,
        "no_proven_answer_count": 1,
        "repeat_ticket_count": 4,
    },
    "top_questions": [
        {
            "rank": 1,
            "question": "How do I export reports?",
            "ticket_count": 3,
            "weighted_frequency": 3,
            "customer_wording": "How do I export reports?",
        }
    ],
    "locked_questions": [{"rank": 2, "ticket_count": 1}],
}


@pytest.fixture(autouse=True)
def _isolate_deflection_submit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ATLAS_API_BASE_URL",
        "ATLAS_B2B_JWT",
        "ATLAS_TOKEN",
        "ATLAS_ACCOUNT_ID",
        "ATLAS_FAQ_SEARCH_ACCOUNT_ID",
        "ATLAS_DEFLECTION_SUBMIT_CSV_FILE",
        "ATLAS_DEFLECTION_SUBMIT_BLOB_URL",
        "ATLAS_DEFLECTION_SUPPORT_PLATFORM",
        "ATLAS_DEFLECTION_COMPANY_NAME",
        "ATLAS_DEFLECTION_CONTACT_EMAIL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ATLAS_DISABLE_DOTENV", "1")


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-secret",
        "--account-id",
        "acct-123",
        "--blob-url",
        "https://blob.example.com/tickets.csv?sig=signed-secret",
        "--support-platform",
        "zendesk",
        "--company-name",
        "Acme Co.",
        "--contact-email",
        "lead@example.com",
        "--output-result",
        str(tmp_path / "result.json"),
    ]


def _base_csv_args(tmp_path: Path, csv_file: Path) -> list[str]:
    return [
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-secret",
        "--account-id",
        "acct-123",
        "--csv-file",
        str(csv_file),
        "--support-platform",
        "zendesk",
        "--company-name",
        "Acme Co.",
        "--contact-email",
        "lead@example.com",
        "--output-result",
        str(tmp_path / "result.json"),
    ]


def _write_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "tickets.csv"
    csv_file.write_text(
        "ticket_id,subject,body\n"
        "1,Export reports,How do I export reports?\n",
        encoding="utf-8",
    )
    return csv_file


def test_load_dotenv_files_honors_disable_flag(monkeypatch) -> None:
    calls: list[Path] = []
    monkeypatch.setenv("ATLAS_DISABLE_DOTENV", "1")
    monkeypatch.setattr(smoke, "load_dotenv", lambda path, **_kwargs: calls.append(path))

    smoke._load_dotenv_files()

    assert calls == []


def _submit_payload(
    *,
    request_id: str = "content-ops-123",
    snapshot: Any = SNAPSHOT,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_metadata = {
        "source": "portfolio_deflection_submit",
        "source_row_count": 3,
        "submitted_row_count": 3,
        "truncated_row_count": 0,
        "max_source_material_rows": 1000,
        "blob_bytes": 200,
        "support_platform": "zendesk",
    }
    if metadata is not None:
        base_metadata = metadata
    return {
        "status": "completed",
        "request_id": request_id,
        "steps": [
            {
                "output": "faq_deflection_report",
                "status": "completed",
                "result": {
                    "request_id": request_id,
                    "snapshot": snapshot,
                    "full_report": {
                        "status": "locked",
                        "reason": "payment_required",
                    },
                },
            }
        ],
        "input_provider": {
            "provider": "portfolio_deflection_submit",
            "metadata": base_metadata,
            "warnings": [],
        },
    }


class FakeResponse:
    def __init__(self, status: int, payload: Any):
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")


def _fake_open(sequence: list[tuple[int, Any]], calls: list[dict[str, Any]]):
    def _open(request, *, timeout):
        headers = dict(request.header_items())
        header_lookup = {key.lower(): value for key, value in headers.items()}
        content_type = header_lookup.get("content-type", "")
        body_bytes = request.data or b""
        body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
        body = None
        if body_bytes and content_type.startswith("application/json"):
            body = json.loads(body_text)
        calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "headers": headers,
            "content_type": content_type,
            "body": body,
            "body_text": body_text,
            "body_bytes": body_bytes,
            "timeout": timeout,
        })
        status, payload = sequence.pop(0)
        return FakeResponse(status, payload)

    return _open


def _patch_open(monkeypatch, sequence: list[tuple[int, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(smoke, "_open_http_request", _fake_open(sequence, calls))
    return calls


def test_validate_args_fails_closed_for_missing_and_unsafe_inputs() -> None:
    args = smoke._build_parser().parse_args([
        "--base-url",
        "http://127.0.0.1:8000",
        "--token",
        "",
        "--blob-url",
        "https://user:pass@blob.example.com/tickets.csv",
        "--support-platform",
        "desk",
        "--company-name",
        "",
        "--contact-email",
        "",
        "--limit",
        str(smoke.SUBMIT_ROW_LIMIT_MAX + 1),
        "--timeout",
        "0",
        "--submit-path",
        "relative",
        "--snapshot-path-template",
        "/snapshot",
    ])

    assert smoke._validate_args(args) == [
        "--base-url must be an absolute HTTPS URL for hosted proof",
        "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required",
        "ATLAS_ACCOUNT_ID, ATLAS_FAQ_SEARCH_ACCOUNT_ID, or --account-id is required",
        "--blob-url must not include credentials",
        "--support-platform must be one of: help_scout, intercom, other, zendesk",
        "ATLAS_DEFLECTION_COMPANY_NAME or --company-name is required",
        "ATLAS_DEFLECTION_CONTACT_EMAIL or --contact-email is required",
        f"--limit must be between 1 and {smoke.SUBMIT_ROW_LIMIT_MAX}",
        "--timeout must be a positive finite number",
        "--submit-path must start with /",
        "--snapshot-path-template must include {request_id}",
    ]


def test_validate_args_rejects_negative_volume_gates(tmp_path) -> None:
    flags = (
        "--min-uploaded-bytes",
        "--min-source-row-count",
        "--min-submitted-row-count",
        "--min-generated-questions",
        "--min-repeat-ticket-count",
        "--min-top-question-count",
    )
    args = smoke._build_parser().parse_args([
        *_base_args(tmp_path),
        *(item for flag in flags for item in (flag, "-1")),
    ])

    assert smoke._validate_args(args) == [f"{flag} must be zero or greater" for flag in flags]


def test_validate_args_fails_closed_for_missing_csv_file(tmp_path) -> None:
    args = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-secret",
        "--account-id",
        "acct-123",
        "--csv-file",
        str(tmp_path / "missing.csv"),
        "--support-platform",
        "zendesk",
        "--company-name",
        "Acme Co.",
        "--contact-email",
        "lead@example.com",
    ])

    assert smoke._validate_args(args) == [
        "--csv-file must point to a readable support-ticket CSV file"
    ]


def test_validate_args_defaults_to_checked_fixture_when_no_source_is_provided() -> None:
    args = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-secret",
        "--account-id",
        "acct-123",
        "--support-platform",
        "zendesk",
        "--company-name",
        "Acme Co.",
        "--contact-email",
        "lead@example.com",
    ])

    assert smoke._validate_args(args) == []
    assert args.csv_file == smoke.DEFAULT_CSV_FILE


def test_validate_args_keeps_explicit_blob_source_over_default_fixture() -> None:
    args = smoke._build_parser().parse_args(_base_args(Path("/tmp")))

    assert smoke._validate_args(args) == []
    assert args.csv_file is None


def test_validate_submit_envelope_accepts_locked_gated_result() -> None:
    request_id, snapshot, diagnostics, errors = smoke._validate_submit_envelope(_submit_payload())

    assert errors == []
    assert request_id == "content-ops-123"
    assert snapshot == SNAPSHOT
    assert diagnostics["provider"] == "portfolio_deflection_submit"
    assert diagnostics["metadata"]["source_row_count"] == 3


def test_validate_submit_envelope_accepts_generated_questions_without_customer_wording() -> None:
    snapshot = {
        **SNAPSHOT,
        "top_questions": [
            {
                **SNAPSHOT["top_questions"][0],
                "question_source": "source_policy",
                "customer_wording": "",
            },
            {
                **SNAPSHOT["top_questions"][0],
                "rank": 2,
                "question": "What are customers asking about exports?",
                "question_source": "topic_fallback",
                "customer_wording": "",
            },
            {
                **SNAPSHOT["top_questions"][0],
                "rank": 3,
                "question": "What should I do about export reporting?",
                "customer_wording": "",
            },
        ],
    }

    _request_id, _snapshot, _diagnostics, errors = smoke._validate_submit_envelope(
        _submit_payload(snapshot=snapshot)
    )

    assert errors == []


@pytest.mark.parametrize("question_source", ["customer_wording", "internal_taxonomy"])
def test_validate_submit_envelope_requires_customer_wording_for_customer_sources(
    question_source: str,
) -> None:
    top_question = {
        **SNAPSHOT["top_questions"][0],
        "customer_wording": "",
        "question_source": question_source,
    }
    snapshot = {
        **SNAPSHOT,
        "top_questions": [top_question],
    }

    _request_id, _snapshot, _diagnostics, errors = smoke._validate_submit_envelope(
        _submit_payload(snapshot=snapshot)
    )

    assert any(
        "snapshot.top_questions[0].customer_wording must be non-empty "
        "for customer-wording questions" in error
        for error in errors
    )


def test_validate_submit_envelope_requires_json_blob_byte_counter() -> None:
    payload = _submit_payload(metadata={
        "source": "portfolio_deflection_submit",
        "source_row_count": 3,
        "submitted_row_count": 3,
        "truncated_row_count": 0,
        "max_source_material_rows": 1000,
        "support_platform": "zendesk",
    })

    _request_id, _snapshot, _diagnostics, errors = smoke._validate_submit_envelope(payload)

    assert (
        "input_provider.metadata.blob_bytes must be a positive integer for json_blob_url submit"
        in errors
    )


def test_validate_submit_envelope_requires_multipart_upload_byte_counter() -> None:
    payload = _submit_payload()

    _request_id, _snapshot, _diagnostics, errors = smoke._validate_submit_envelope(
        payload,
        submit_mode="multipart",
    )

    assert (
        "input_provider.metadata.uploaded_bytes must be a positive integer for multipart submit"
        in errors
    )


def test_validate_submit_envelope_rejects_missing_deflection_step() -> None:
    payload = _submit_payload()
    payload["steps"] = []

    request_id, snapshot, _diagnostics, errors = smoke._validate_submit_envelope(payload)

    assert request_id == "content-ops-123"
    assert snapshot is None
    assert "submit response must include faq_deflection_report step" in errors


def test_validate_submit_envelope_rejects_snapshot_leaks() -> None:
    snapshot = {
        **SNAPSHOT,
        "top_questions": [
            {
                **SNAPSHOT["top_questions"][0],
                "evidence": [{"source_id": "ticket-1"}],
                "answer": "Open Reports.",
            }
        ],
    }

    _request_id, _snapshot, _diagnostics, errors = smoke._validate_submit_envelope(
        _submit_payload(snapshot=snapshot)
    )

    assert any("leaked forbidden fields" in error for error in errors)


def test_validate_submit_envelope_rejects_singular_source_id_leak() -> None:
    snapshot = {
        **SNAPSHOT,
        "top_questions": [
            {
                **SNAPSHOT["top_questions"][0],
                "source_id": "ticket-1",
            }
        ],
    }

    _request_id, _snapshot, _diagnostics, errors = smoke._validate_submit_envelope(
        _submit_payload(snapshot=snapshot)
    )

    assert any("$.top_questions[0].source_id" in error for error in errors)


def test_stale_multipart_submit_route_detector_identifies_json_body_422() -> None:
    response = smoke.HttpJsonResponse(
        status=422,
        payload={
            "detail": [
                {
                    "type": "model_attributes_type",
                    "loc": ["body"],
                    "msg": "Input should be a valid dictionary or object to extract fields from",
                }
            ]
        },
        raw_text="",
    )

    errors = smoke._stale_multipart_submit_route_errors(response, submit_mode="multipart")

    assert errors == [
        "deployed submit route rejected multipart as a JSON body; "
        "expected multipart CSV Request route, so the host is likely "
        "serving stale route code or importing a stale extracted_content_pipeline"
    ]


def test_stale_multipart_submit_route_detector_rejects_json_mode() -> None:
    response = smoke.HttpJsonResponse(
        status=422,
        payload={"detail": [{"type": "model_attributes_type", "loc": ["body"]}]},
        raw_text="",
    )

    assert smoke._stale_multipart_submit_route_errors(
        response,
        submit_mode="json_blob_url",
    ) == []


def test_stale_multipart_submit_route_detector_rejects_unrelated_422() -> None:
    response = smoke.HttpJsonResponse(
        status=422,
        payload={"detail": [{"type": "missing", "loc": ["body", "csv_file"]}]},
        raw_text="",
    )

    assert smoke._stale_multipart_submit_route_errors(response, submit_mode="multipart") == []


def test_run_success_posts_submit_and_probes_snapshot_and_locked_artifact(monkeypatch, tmp_path):
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open(
            [
                (200, _submit_payload()),
                (200, SNAPSHOT),
                (403, {"detail": "payment_required"}),
            ],
            calls,
        ),
    )
    args = smoke._build_parser().parse_args(_base_args(tmp_path))

    summary = smoke.run(args)

    assert summary["ok"] is True
    assert summary["request_id"] == "content-ops-123"
    assert summary["blob_host"] == "blob.example.com"
    assert [call["method"] for call in calls] == ["POST", "GET", "GET"]
    assert calls[0]["url"] == "https://atlas.example.com/api/v1/content-ops/deflection-reports/submit"
    assert calls[0]["body"]["blob_url"] == "https://blob.example.com/tickets.csv?sig=signed-secret"
    assert calls[0]["body"]["support_platform"] == "zendesk"
    assert "limit" not in calls[0]["body"]
    assert calls[1]["url"].endswith("/content-ops-123/snapshot")
    assert calls[2]["url"].endswith("/content-ops-123/artifact")
    serialized = json.dumps(summary)
    assert "token-secret" not in serialized
    assert "signed-secret" not in serialized


def test_run_success_posts_multipart_csv_and_probes_locked_artifact(monkeypatch, tmp_path):
    csv_file = _write_csv(tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open(
            [
                (
                    200,
                    _submit_payload(
                        metadata={
                            "source": "portfolio_deflection_submit",
                            "source_row_count": 3,
                            "submitted_row_count": 3,
                            "truncated_row_count": 0,
                            "max_source_material_rows": 1000,
                            "uploaded_bytes": csv_file.stat().st_size,
                            "support_platform": "zendesk",
                        }
                    ),
                ),
                (200, SNAPSHOT),
                (403, {"detail": "payment_required"}),
            ],
            calls,
        ),
    )
    args = smoke._build_parser().parse_args(_base_csv_args(tmp_path, csv_file))

    summary = smoke.run(args)

    assert summary["ok"] is True
    assert summary["submit_mode"] == "multipart"
    assert summary["csv_file_size"] == csv_file.stat().st_size
    assert summary["blob_host"] == ""
    assert calls[0]["body"] is None
    assert calls[0]["content_type"].startswith("multipart/form-data; boundary=")
    assert 'name="csv_file"; filename="tickets.csv"' in calls[0]["body_text"]
    assert "How do I export reports?" in calls[0]["body_text"]
    assert 'name="support_platform"' in calls[0]["body_text"]
    assert 'name="limit"' not in calls[0]["body_text"]
    assert "blob_url" not in calls[0]["body_text"]
    assert summary["submit"]["diagnostics"]["metadata"]["uploaded_bytes"] == csv_file.stat().st_size
    serialized = json.dumps(summary)
    assert "token-secret" not in serialized
    assert "How do I export reports?" not in serialized


def test_run_sends_explicit_large_submit_limit_when_requested(monkeypatch, tmp_path):
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open(
            [
                (
                    200,
                    _submit_payload(
                        metadata={
                            "source": "portfolio_deflection_submit",
                            "source_row_count": 25000,
                            "submitted_row_count": 25000,
                            "truncated_row_count": 0,
                            "max_source_material_rows": 25000,
                            "blob_bytes": 200,
                            "support_platform": "zendesk",
                        }
                    ),
                ),
                (200, SNAPSHOT),
                (403, {"detail": "payment_required"}),
            ],
            calls,
        ),
    )
    args = smoke._build_parser().parse_args([*_base_args(tmp_path), "--limit", "25000"])

    summary = smoke.run(args)

    assert summary["ok"] is True
    assert calls[0]["body"]["limit"] == 25000
    assert summary["submit"]["diagnostics"]["metadata"]["max_source_material_rows"] == 25000


def _full_volume_snapshot() -> dict[str, Any]:
    return {
        **SNAPSHOT,
        "summary": {
            **SNAPSHOT["summary"],
            "generated": 37,
            "repeat_ticket_count": 35386,
        },
        "top_questions": [
            {
                **SNAPSHOT["top_questions"][0],
                "rank": index,
                "ticket_count": 1000 - index,
            }
            for index in range(1, 6)
        ],
    }


def _observed_full_volume_snapshot() -> dict[str, Any]:
    return {
        **SNAPSHOT,
        "summary": {
            **SNAPSHOT["summary"],
            "generated": 1659,
            "repeat_ticket_count": 27384,
        },
        "top_questions": [
            {
                **SNAPSHOT["top_questions"][0],
                "rank": index,
                "ticket_count": 2000 - index,
            }
            for index in range(1, 6)
        ],
    }


def _observed_full_volume_metadata() -> dict[str, Any]:
    return {
        "source": "portfolio_deflection_submit",
        "source_row_count": 40383,
        "submitted_row_count": 40383,
        "truncated_row_count": 0,
        "max_source_material_rows": 40383,
        "blob_bytes": 52428276,
        "support_platform": "zendesk",
    }


def test_run_passes_calibrated_full_volume_cfpb_profile(
    monkeypatch,
    tmp_path,
):
    snapshot = _observed_full_volume_snapshot()
    _patch_open(
        monkeypatch,
        [
            (
                200,
                _submit_payload(
                    snapshot=snapshot,
                    metadata=_observed_full_volume_metadata(),
                ),
            ),
            (200, snapshot),
            (403, {"detail": "payment_required"}),
        ],
    )

    summary = smoke.run(
        smoke._build_parser().parse_args([
            *_base_args(tmp_path),
            "--volume-gate-profile",
            "full-volume-cfpb",
        ])
    )

    assert summary["ok"] is True
    assert summary["volume_gates"] == {
        "profile": "full-volume-cfpb",
        "configured": {
            "uploaded_bytes": 50000000,
            "source_row_count": 30000,
            "submitted_row_count": 30000,
            "generated_questions": 30,
            "repeat_ticket_count": 25000,
            "top_question_count": 5,
        },
        "actual": {
            "uploaded_bytes": 52428276,
            "source_row_count": 40383,
            "submitted_row_count": 40383,
            "generated_questions": 1659,
            "repeat_ticket_count": 27384,
            "top_question_count": 5,
        },
        "ok": True,
        "errors": [],
    }


def test_run_fails_calibrated_full_volume_cfpb_profile_for_tiny_fixture(
    monkeypatch,
    tmp_path,
):
    _patch_open(
        monkeypatch,
        [
            (200, _submit_payload()),
            (200, SNAPSHOT),
            (403, {"detail": "payment_required"}),
        ],
    )

    summary = smoke.run(
        smoke._build_parser().parse_args([
            *_base_args(tmp_path),
            "--volume-gate-profile",
            "full-volume-cfpb",
        ])
    )

    assert summary["ok"] is False
    assert summary["volume_gates"]["profile"] == "full-volume-cfpb"
    assert summary["volume_gates"]["configured"]["repeat_ticket_count"] == 25000
    assert (
        "volume gate repeat_ticket_count expected >= 25000, got 4"
        in summary["volume_gates"]["errors"]
    )


def test_run_explicit_repeat_gate_overrides_full_volume_cfpb_profile(
    monkeypatch,
    tmp_path,
):
    snapshot = _observed_full_volume_snapshot()
    _patch_open(
        monkeypatch,
        [
            (
                200,
                _submit_payload(
                    snapshot=snapshot,
                    metadata=_observed_full_volume_metadata(),
                ),
            ),
            (200, snapshot),
            (403, {"detail": "payment_required"}),
        ],
    )

    summary = smoke.run(
        smoke._build_parser().parse_args([
            *_base_args(tmp_path),
            "--volume-gate-profile",
            "full-volume-cfpb",
            "--min-repeat-ticket-count",
            "30000",
        ])
    )

    assert summary["ok"] is False
    assert summary["volume_gates"]["configured"]["repeat_ticket_count"] == 30000
    assert summary["volume_gates"]["errors"] == [
        "volume gate repeat_ticket_count expected >= 30000, got 27384"
    ]


def test_run_passes_configured_full_volume_gates(monkeypatch, tmp_path):
    snapshot = _full_volume_snapshot()
    metadata = {
        "source": "portfolio_deflection_submit",
        "source_row_count": 35386,
        "submitted_row_count": 35386,
        "truncated_row_count": 0,
        "max_source_material_rows": 35386,
        "blob_bytes": 52363054,
        "support_platform": "zendesk",
    }
    _patch_open(
        monkeypatch,
        [
            (200, _submit_payload(snapshot=snapshot, metadata=metadata)),
            (200, snapshot),
            (403, {"detail": "payment_required"}),
        ],
    )
    flags = (
        "--min-uploaded-bytes",
        "50000000",
        "--min-source-row-count",
        "30000",
        "--min-submitted-row-count",
        "30000",
        "--min-generated-questions",
        "30",
        "--min-repeat-ticket-count",
        "30000",
        "--min-top-question-count",
        "5",
    )
    summary = smoke.run(smoke._build_parser().parse_args([*_base_args(tmp_path), *flags]))

    assert summary["ok"] is True
    assert summary["volume_gates"]["ok"] is True
    assert summary["volume_gates"]["actual"] == {
        "uploaded_bytes": 52363054,
        "source_row_count": 35386,
        "submitted_row_count": 35386,
        "generated_questions": 37,
        "repeat_ticket_count": 35386,
        "top_question_count": 5,
    }
    serialized = json.dumps(summary)
    assert "token-secret" not in serialized
    assert "signed-secret" not in serialized


@pytest.mark.parametrize(
    ("flag", "gate", "minimum", "actual"),
    [
        ("--min-uploaded-bytes", "uploaded_bytes", "201", 200),
        ("--min-source-row-count", "source_row_count", "4", 3),
        ("--min-submitted-row-count", "submitted_row_count", "4", 3),
        ("--min-generated-questions", "generated_questions", "3", 2),
        ("--min-repeat-ticket-count", "repeat_ticket_count", "5", 4),
        ("--min-top-question-count", "top_question_count", "2", 1),
    ],
)
def test_run_fails_each_configured_volume_gate_when_tiny_fixture_passes_shape(
    monkeypatch,
    tmp_path,
    flag: str,
    gate: str,
    minimum: str,
    actual: int,
):
    calls = _patch_open(
        monkeypatch,
        [
            (200, _submit_payload()),
            (200, SNAPSHOT),
            (403, {"detail": "payment_required"}),
        ],
    )
    summary = smoke.run(
        smoke._build_parser().parse_args([*_base_args(tmp_path), flag, minimum])
    )

    assert summary["ok"] is False
    assert summary["volume_gates"]["ok"] is False
    assert summary["volume_gates"]["errors"] == [
        f"volume gate {gate} expected >= {minimum}, got {actual}"
    ]
    assert summary["errors"] == summary["volume_gates"]["errors"]
    assert len(calls) == 3


def test_run_fails_volume_gate_when_metric_is_missing(monkeypatch, tmp_path):
    snapshot = {
        **SNAPSHOT,
        "summary": {
            "generated": 2,
            "drafted_answer_count": 1,
            "no_proven_answer_count": 1,
        },
    }
    _patch_open(
        monkeypatch,
        [
            (200, _submit_payload(snapshot=snapshot)),
            (200, snapshot),
            (403, {"detail": "payment_required"}),
        ],
    )
    summary = smoke.run(
        smoke._build_parser().parse_args([
            *_base_args(tmp_path),
            "--min-repeat-ticket-count",
            "1",
        ])
    )

    assert summary["ok"] is False
    assert summary["volume_gates"]["errors"] == [
        "volume gate repeat_ticket_count expected >= 1, got missing"
    ]


def test_run_marks_configured_volume_gates_skipped_when_envelope_fails(
    monkeypatch,
    tmp_path,
):
    payload = _submit_payload(metadata={
        "source": "portfolio_deflection_submit",
        "source_row_count": 3,
        "submitted_row_count": 3,
        "truncated_row_count": 0,
        "max_source_material_rows": 1000,
        "support_platform": "zendesk",
    })
    calls = _patch_open(monkeypatch, [(200, payload)])
    summary = smoke.run(
        smoke._build_parser().parse_args([
            *_base_args(tmp_path),
            "--min-generated-questions",
            "1",
        ])
    )

    assert summary["ok"] is False
    assert summary["volume_gates"] == {
        "configured": {"generated_questions": 1},
        "actual": {},
        "ok": False,
        "skipped": True,
        "not_run_reason": "submit_failed",
        "errors": [],
    }
    assert (
        "input_provider.metadata.blob_bytes must be a positive integer "
        "for json_blob_url submit"
    ) in summary["errors"]
    assert len(calls) == 1


def test_run_fails_with_stale_multipart_submit_route_diagnostic(monkeypatch, tmp_path):
    csv_file = _write_csv(tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open(
            [
                (
                    422,
                    {
                        "detail": [
                            {
                                "type": "model_attributes_type",
                                "loc": ["body"],
                                "msg": (
                                    "Input should be a valid dictionary or object to "
                                    "extract fields from"
                                ),
                            }
                        ]
                    },
                )
            ],
            calls,
        ),
    )
    args = smoke._build_parser().parse_args(_base_csv_args(tmp_path, csv_file))

    summary = smoke.run(args)

    assert summary["ok"] is False
    assert len(calls) == 1
    assert summary["snapshot"]["not_run_reason"] == "submit_failed"
    assert summary["artifact"]["not_run_reason"] == "submit_failed"
    assert (
        "deployed submit route rejected multipart as a JSON body; expected multipart CSV "
        "Request route, so the host is likely serving stale route code or importing a "
        "stale extracted_content_pipeline"
    ) in summary["errors"]
    assert "submit status must be 200, got 422" in summary["errors"]
    serialized = json.dumps(summary)
    assert "token-secret" not in serialized
    assert "How do I export reports?" not in serialized


def test_run_fails_closed_when_snapshot_does_not_match_submit(monkeypatch, tmp_path):
    calls: list[dict[str, Any]] = []
    changed_snapshot = {
        **SNAPSHOT,
        "summary": {**SNAPSHOT["summary"], "generated": 99},
    }
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open(
            [
                (200, _submit_payload()),
                (200, changed_snapshot),
                (403, {"detail": "payment_required"}),
            ],
            calls,
        ),
    )
    args = smoke._build_parser().parse_args(_base_args(tmp_path))

    summary = smoke.run(args)

    assert summary["ok"] is False
    assert "snapshot response must match submit snapshot" in summary["errors"]


def test_run_fails_closed_when_unpaid_artifact_is_released(monkeypatch, tmp_path):
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open(
            [
                (200, _submit_payload()),
                (200, SNAPSHOT),
                (200, {"markdown": "# Paid report"}),
            ],
            calls,
        ),
    )
    args = smoke._build_parser().parse_args(_base_args(tmp_path))

    summary = smoke.run(args)

    assert summary["ok"] is False
    assert "unpaid artifact status must be 403, got 200" in summary["errors"]


def test_run_stops_after_failed_submit(monkeypatch, tmp_path):
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open([(500, {"detail": "server error"})], calls),
    )
    args = smoke._build_parser().parse_args(_base_args(tmp_path))

    summary = smoke.run(args)

    assert summary["ok"] is False
    assert len(calls) == 1
    assert "submit status must be 200, got 500" in summary["errors"]
    assert summary["snapshot"]["not_run_reason"] == "submit_failed"


def test_json_request_preserves_http_error_status(monkeypatch):
    def _raise_http_error(_request, *, timeout):
        raise urllib.error.HTTPError(
            "https://atlas.example.com/probe",
            403,
            "Forbidden",
            {},
            None,
        )

    monkeypatch.setattr(smoke, "_open_http_request", _raise_http_error)

    response = smoke._json_request(
        "GET",
        "https://atlas.example.com/probe",
        token="secret",
        timeout=1.0,
    )

    assert response.status == 403
    assert response.payload is None
    assert response.errors == ()


def test_main_preflight_writes_redacted_result(tmp_path, capsys):
    result_path = tmp_path / "preflight.json"

    exit_code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-secret",
        "--account-id",
        "acct-123",
        "--blob-url",
        "https://blob.example.com/tickets.csv?sig=signed-secret",
        "--support-platform",
        "zendesk",
        "--company-name",
        "Acme Co.",
        "--contact-email",
        "lead@example.com",
        "--preflight-only",
        "--json",
        "--output-result",
        str(result_path),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["submit"]["not_run_reason"] == "preflight_only"
    serialized = captured.out + result_path.read_text(encoding="utf-8")
    assert "token-secret" not in serialized
    assert "signed-secret" not in serialized


def test_main_multipart_preflight_records_mode_without_csv_contents(tmp_path, capsys):
    csv_file = _write_csv(tmp_path)
    result_path = tmp_path / "preflight.json"

    exit_code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-secret",
        "--account-id",
        "acct-123",
        "--csv-file",
        str(csv_file),
        "--support-platform",
        "zendesk",
        "--company-name",
        "Acme Co.",
        "--contact-email",
        "lead@example.com",
        "--preflight-only",
        "--json",
        "--output-result",
        str(result_path),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["submit_mode"] == "multipart"
    assert payload["csv_file_size"] == csv_file.stat().st_size
    serialized = captured.out + result_path.read_text(encoding="utf-8")
    assert "token-secret" not in serialized
    assert "How do I export reports?" not in serialized
