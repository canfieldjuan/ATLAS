from __future__ import annotations

import csv
import http.client
import json
import tempfile
from io import StringIO
from pathlib import Path
import socket
import urllib.error
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices
from extracted_content_pipeline.deflection_report_access import (
    InMemoryDeflectionReportArtifactStore,
)
from extracted_content_pipeline.faq_deflection_report import FAQDeflectionReportService


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_FIXTURE_DIR = (
    ROOT
    / "extracted_content_pipeline"
    / "examples"
    / "support_ticket_provider_exports"
)
ZENDESK_THREAD_SAMPLE = ROOT / "tests/fixtures/zendesk_full_thread_seed_sample.json"


def _route(router: Any, path: str, method: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


class _BlobResponse:
    def __init__(self, data: bytes, *, status: int = 200) -> None:
        self._data = data
        self._offset = 0
        self.status = status
        self.closed = False
        self.read_sizes: list[int] = []

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        if self._offset >= len(self._data):
            return b""
        if size < 0:
            size = len(self._data) - self._offset
        end = min(len(self._data), self._offset + size)
        chunk = self._data[self._offset:end]
        self._offset = end
        return chunk

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_BlobResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Upload:
    def __init__(self, data: bytes, *, filename: str = "tickets.csv") -> None:
        self._data = data
        self._offset = 0
        self.filename = filename
        self.read_sizes: list[int] = []

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        if self._offset >= len(self._data):
            return b""
        if size < 0:
            size = len(self._data) - self._offset
        end = min(len(self._data), self._offset + size)
        chunk = self._data[self._offset:end]
        self._offset = end
        return chunk


class _FormRequest:
    def __init__(
        self,
        form: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._form = form
        self.headers = headers or {"content-type": "multipart/form-data; boundary=test"}
        self.form_kwargs: dict[str, Any] | None = None

    async def form(self, **kwargs: Any) -> dict[str, Any]:
        self.form_kwargs = dict(kwargs)
        return self._form


def _csv_bytes(rows: list[str]) -> bytes:
    return ("\n".join(rows) + "\n").encode("utf-8")


def _csv_dict_bytes(rows: list[dict[str, str]]) -> bytes:
    out = StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=tuple(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


@pytest.mark.asyncio
async def test_deflection_submit_upload_copy_streams_chunks_not_whole_limit() -> None:
    data = b"a" * (api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES + 17)
    upload = _Upload(data)

    with tempfile.NamedTemporaryFile() as handle:
        byte_count = await api_module._copy_deflection_submit_upload_to_tempfile(
            upload,
            handle,
            max_bytes=len(data) + 1024,
        )
        handle.flush()
        handle.seek(0)
        staged = handle.read()

    assert byte_count == len(data)
    assert staged == data
    assert upload.read_sizes
    assert all(
        size == api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES
        for size in upload.read_sizes
    )
    assert len(data) + 1025 not in upload.read_sizes


@pytest.mark.asyncio
async def test_deflection_submit_upload_copy_rejects_oversize_mid_stream() -> None:
    data = b"a" * (api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES + 2)
    upload = _Upload(data)

    with tempfile.NamedTemporaryFile() as handle:
        with pytest.raises(api_module.HTTPException) as exc_info:
            await api_module._copy_deflection_submit_upload_to_tempfile(
                upload,
                handle,
                max_bytes=api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES + 1,
            )

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == {
        "reason": "deflection_submit_csv_too_large",
        "max_file_bytes": api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES + 1,
    }
    assert upload.read_sizes == [
        api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES,
        api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES,
    ]


@pytest.mark.asyncio
async def test_deflection_submit_full_thread_upload_copy_streams_chunks_not_whole_limit() -> None:
    data = (
        b'{"tickets":[{"ticket":{"id":"zd-large","subject":"'
        + (b"Billing refund " * 6000)
        + b'","description":"How do I get a refund?"},"comments":[]}]}'
    )
    upload = _Upload(data, filename="zendesk-thread.json")

    with tempfile.NamedTemporaryFile() as handle:
        byte_count = await api_module._copy_deflection_submit_upload_to_tempfile(
            upload,
            handle,
            max_bytes=len(data) + 1024,
            too_large_reason="deflection_submit_full_thread_too_large",
            read_error_detail="Uploaded Zendesk full-thread JSON could not be read.",
            stage_error_detail="Uploaded Zendesk full-thread JSON could not be staged.",
        )
        handle.flush()
        handle.seek(0)
        staged = handle.read()

    assert byte_count == len(data)
    assert staged == data
    assert upload.read_sizes
    assert all(
        size == api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES
        for size in upload.read_sizes
    )
    assert len(data) + 1025 not in upload.read_sizes


def _install_blob(
    monkeypatch: pytest.MonkeyPatch,
    data: bytes,
) -> list[tuple[str, str, int]]:
    calls: list[tuple[str, str, int]] = []

    def _open(target: Any, *, timeout: int) -> _BlobResponse:
        calls.append((target.url, target.connect_host, timeout))
        return _BlobResponse(data)

    monkeypatch.setattr(api_module, "_open_https_blob_request", _open)
    return calls


def _install_public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def _getaddrinfo(host: str, port: int, *, type: int):
        return [(
            socket.AF_INET,
            type,
            6,
            "",
            ("93.184.216.34", port),
        )]

    monkeypatch.setattr(api_module.socket, "getaddrinfo", _getaddrinfo)


def test_deflection_submit_blob_csv_streams_response_to_tempfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    message = "How do I export reports? " * 4000
    data = _csv_bytes(
        ["ticket_id,message"]
        + [f"ticket-{index},{message}" for index in range(12)]
    )
    response = _BlobResponse(data)

    monkeypatch.setattr(
        api_module,
        "_open_https_blob_request",
        lambda _target, *, timeout: response,
    )

    rows, byte_count, load_warnings = api_module._load_deflection_submit_blob_rows_sync(
        "https://portfolio.example/blob/tickets.csv",
        max_bytes=len(data) + 1024,
        importer_mode="csv",
    )

    assert byte_count == len(data)
    assert load_warnings == ()
    assert rows[0]["ticket_id"] == "ticket-0"
    assert response.closed
    assert response.read_sizes
    assert all(
        size == api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES
        for size in response.read_sizes
    )
    assert len(data) + 1025 not in response.read_sizes


def test_deflection_submit_blob_full_thread_streams_response_to_tempfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    data = json.dumps({
        "tickets": [{
            "ticket": {
                "id": "zd-large-thread",
                "subject": "Large refund thread",
                "description": "How do I get a refund for duplicate billing?",
                "requester_id": "requester-1",
                "status": "solved",
                "satisfaction_rating": {"score": "good"},
            },
            "comments": [
                {
                    "author_id": "requester-1",
                    "public": True,
                    "plain_body": "The duplicate invoice still appears on my account.",
                },
                {
                    "author_id": "agent-1",
                    "public": True,
                    "plain_body": "We refunded the duplicate invoice. "
                    + ("The refund appears in 5-10 business days. " * 4000),
                },
            ],
        }],
    }).encode("utf-8")
    response = _BlobResponse(data)

    monkeypatch.setattr(
        api_module,
        "_open_https_blob_request",
        lambda _target, *, timeout: response,
    )

    rows, byte_count, load_warnings = api_module._load_deflection_submit_blob_rows_sync(
        "https://portfolio.example/blob/zendesk-thread.json",
        max_bytes=len(data) + 1024,
        importer_mode="full_thread",
    )

    assert byte_count == len(data)
    assert load_warnings == ()
    assert rows[0]["ticket_id"] == "zd-large-thread"
    assert rows[0]["ticket_status"] == "solved"
    assert rows[0]["satisfaction_rating"] == "good"
    assert "refunded the duplicate invoice" in rows[0]["resolution_text"]
    assert response.closed
    assert response.read_sizes
    assert all(
        size == api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES
        for size in response.read_sizes
    )
    assert len(data) + 1025 not in response.read_sizes


def test_copy_bounded_https_blob_rejects_oversize_before_writing_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    data = b"a" * (api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES + 2)
    response = _BlobResponse(data)

    monkeypatch.setattr(
        api_module,
        "_open_https_blob_request",
        lambda _target, *, timeout: response,
    )

    with tempfile.NamedTemporaryFile() as handle:
        with pytest.raises(api_module.HTTPException) as exc_info:
            api_module._copy_bounded_https_blob_to_tempfile(
                "https://portfolio.example/blob/tickets.csv",
                handle,
                max_bytes=api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES + 1,
            )
        handle.flush()
        handle.seek(0)
        staged = handle.read()

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == {
        "reason": "deflection_submit_blob_too_large",
        "max_file_bytes": api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES + 1,
    }
    assert staged == b"a" * api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES
    assert response.closed
    assert response.read_sizes == [
        api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES,
        api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES,
    ]


@pytest.mark.asyncio
async def test_deflection_submit_upload_parses_bom_semicolon_csv() -> None:
    data = (
        "\ufeffticket_id;subject;message\n"
        "ticket-1;Export help;How do I export attribution reports?\n"
    ).encode("utf-8")

    rows, byte_count, load_warnings = await api_module._load_deflection_submit_upload_rows(
        _Upload(data),
        max_bytes=1024,
    )

    assert byte_count == len(data)
    assert load_warnings == ()
    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": "How do I export attribution reports?",
    }]


@pytest.mark.asyncio
async def test_deflection_submit_upload_parses_utf16_bom_csv() -> None:
    data = (
        "ticket_id,subject,message\n"
        "ticket-1,Export help,How do I export attribution reports?\n"
    ).encode("utf-16")

    rows, byte_count, load_warnings = await api_module._load_deflection_submit_upload_rows(
        _Upload(data),
        max_bytes=2048,
    )

    assert byte_count == len(data)
    assert load_warnings == ()
    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": "How do I export attribution reports?",
    }]


@pytest.mark.asyncio
async def test_deflection_submit_upload_parses_embedded_quotes_and_newlines() -> None:
    data = _csv_dict_bytes([
        {
            "ticket_id": "ticket-quoted-1",
            "subject": 'Need help with "classes"',
            "message": (
                'The portal says "classes" are missing.\n'
                "I already tried the setup wizard."
            ),
            "answer": 'Open Settings, choose "Classes", then select Sync.',
        },
    ])

    rows, byte_count, load_warnings = await api_module._load_deflection_submit_upload_rows(
        _Upload(data),
        max_bytes=2048,
    )

    assert byte_count == len(data)
    assert load_warnings == ()
    assert rows == [{
        "ticket_id": "ticket-quoted-1",
        "subject": 'Need help with "classes"',
        "message": (
            'The portal says "classes" are missing.\n'
            "I already tried the setup wizard."
        ),
        "answer": 'Open Settings, choose "Classes", then select Sync.',
    }]


@pytest.mark.asyncio
async def test_deflection_submit_upload_skips_provider_metadata_header() -> None:
    data = _csv_bytes([
        "Zendesk ticket export",
        "ticket_id,subject,message",
        "ticket-1,Export help,How do I export attribution reports?",
    ])

    rows, byte_count, load_warnings = await api_module._load_deflection_submit_upload_rows(
        _Upload(data),
        max_bytes=1024,
    )

    assert byte_count == len(data)
    assert len(load_warnings) == 1
    assert load_warnings[0]["code"] == "csv_leading_rows_skipped"
    assert load_warnings[0]["row_index"] == 1
    assert "Zendesk ticket export" in load_warnings[0]["message"]
    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": "How do I export attribution reports?",
    }]


@pytest.mark.asyncio
async def test_deflection_submit_upload_load_result_caps_rows_and_counts_source() -> None:
    data = _csv_bytes([
        "ticket_id,subject,message",
        "ticket-1,Export help,How do I export attribution reports?",
        "ticket-2,Billing help,Where can I download invoices?",
        "ticket-3,Password help,How do I reset my password?",
    ])

    loaded = await api_module._load_deflection_submit_upload_rows(
        _Upload(data),
        max_bytes=2048,
        max_rows=2,
    )

    assert loaded.source_row_count == 3
    assert [row["ticket_id"] for row in loaded.rows] == ["ticket-1", "ticket-2"]
    rows, byte_count, load_warnings = loaded
    assert rows == loaded.rows
    assert byte_count == len(data)
    assert load_warnings == ()


@pytest.mark.asyncio
async def test_deflection_submit_upload_structures_missing_header_csv_error() -> None:
    data = _csv_bytes([
        "Need export help",
        "Invoice portal is wrong",
    ])

    with pytest.raises(api_module.HTTPException) as exc:
        await api_module._load_deflection_submit_upload_rows(
            _Upload(data),
            max_bytes=1024,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == {
        "code": "csv_missing_header",
        "message": "CSV customer data must include a header row.",
        "how_to_fix": (
            "Add a header row with column names such as ticket_id, subject, "
            "and message, then export the CSV again."
        ),
        "reason": "deflection_submit_csv_parse_error",
    }


@pytest.mark.asyncio
async def test_deflection_submit_upload_structures_encoding_csv_error() -> None:
    data = b"\x00" * 128

    with pytest.raises(api_module.HTTPException) as exc:
        await api_module._load_deflection_submit_upload_rows(
            _Upload(data),
            max_bytes=1024,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["reason"] == "deflection_submit_csv_parse_error"
    assert exc.value.detail["code"] == "csv_encoding_error"
    assert "NUL-heavy" in exc.value.detail["message"]
    assert "UTF-8 or UTF-16" in exc.value.detail["how_to_fix"]


@pytest.mark.asyncio
async def test_deflection_submit_upload_structures_malformed_bom_csv_error() -> None:
    data = b"\xff\xfeticket_id\x00\xff"

    with pytest.raises(api_module.HTTPException) as exc:
        await api_module._load_deflection_submit_upload_rows(
            _Upload(data),
            max_bytes=1024,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["reason"] == "deflection_submit_csv_parse_error"
    assert exc.value.detail["code"] == "csv_encoding_error"
    assert "utf-16" in exc.value.detail["message"]
    assert "UTF-8 or UTF-16" in exc.value.detail["how_to_fix"]


def test_deflection_submit_blob_structures_inconsistent_columns_csv_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    _install_blob(
        monkeypatch,
        _csv_bytes([
            "ticket_id,subject,message",
            "ticket-1,Export help,How do I export reports?",
            "ticket-2,Billing help,Invoice portal is wrong,lead@example.com",
        ]),
    )

    with pytest.raises(api_module.HTTPException) as exc:
        api_module._load_deflection_submit_blob_rows_sync(
            "https://portfolio.example/blob/tickets.csv",
            max_bytes=2048,
        )

    detail_json = json.dumps(exc.value.detail)
    assert exc.value.status_code == 400
    assert exc.value.detail["reason"] == "deflection_submit_csv_parse_error"
    assert exc.value.detail["code"] == "csv_inconsistent_columns"
    assert exc.value.detail["row_index"] == 3
    assert "more cells than the header" in exc.value.detail["message"]
    assert "lead@example.com" not in detail_json


def _router(
    store: InMemoryDeflectionReportArtifactStore,
):
    return create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(
            prefix="/ops",
            tags=("ops",),
            deflection_snapshot_top_n=2,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_deflection_report=FAQDeflectionReportService(),
        ),
        deflection_report_store_provider=lambda: store,
        scope_provider=lambda: {"account_id": "acct-portfolio-submit"},
    )


@pytest.mark.asyncio
async def test_deflection_submit_fetches_blob_and_returns_locked_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    csv_data = _csv_bytes([
        "ticket_id,subject,message,resolution_text,pain_category",
        (
            "ticket-export-1,Export attribution,"
            "How do I export attribution reports?,"
            "Open Analytics and click Download report.,exports"
        ),
        (
            "ticket-export-2,Report download,"
            "Where is the report download for attribution exports?,"
            "Open Analytics and click Download report.,exports"
        ),
        "ticket-sso-1,SSO setup,How do I enable SSO for my team?,,auth",
    ])
    calls = _install_blob(monkeypatch, csv_data)
    store = InMemoryDeflectionReportArtifactStore()
    router = _router(store)

    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    payload = await submit.endpoint({
        "blob_url": "https://portfolio.example/blob/tickets.csv",
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
        "limit": 2,
    })

    assert calls == [(
        "https://portfolio.example/blob/tickets.csv",
        "93.184.216.34",
        api_module._DEFLECTION_SUBMIT_FETCH_TIMEOUT_SECONDS,
    )]
    assert payload["status"] == "completed"
    assert payload["request_id"].startswith("content-ops-")
    gated_result = payload["steps"][0]["result"]
    assert gated_result == {
        "request_id": payload["request_id"],
        "snapshot": gated_result["snapshot"],
        "full_report": {
            "status": "locked",
            "reason": "payment_required",
        },
    }
    assert "markdown" not in str(gated_result)
    assert "ticket-export-1" not in str(gated_result)
    assert "lead@acme.example" not in str(payload)
    assert payload["input_provider"]["metadata"] == {
        "blob_bytes": len(csv_data),
        "cluster_quality": {
            "cluster_count": 1,
            "clustered_row_count": 2,
            "largest_cluster_count": 2,
            "singleton_cluster_count": 0,
            "uncategorized_row_count": 0,
        },
        "included_row_count": 2,
        "max_source_material_rows": 2,
        "skipped_row_count": 0,
        "source": "portfolio_deflection_submit",
        "source_period": "Uploaded support tickets",
        "source_row_count": 3,
        "submitted_row_count": 2,
        "support_platform": "zendesk",
        "support_ticket_resolution_evidence_count": 2,
        "support_ticket_resolution_evidence_present": True,
        "top_ticket_clusters": [{"label": "exports", "count": 2}],
        "truncated_row_count": 1,
    }
    assert payload["input_provider"]["warnings"] == [{
        "code": "deflection_submit_rows_truncated",
        "message": "Used first 2 support-ticket rows out of 3.",
        "row_count": 3,
        "max_rows": 2,
        "truncated_row_count": 1,
    }]

    snapshot = _route(router, "/ops/deflection-reports/{request_id}/snapshot", "GET")
    snapshot_payload = await snapshot.endpoint(request_id=payload["request_id"])
    assert snapshot_payload == gated_result["snapshot"]
    assert snapshot_payload["summary"]["support_ticket_resolution_evidence_count"] == 2
    assert snapshot_payload["summary"]["support_ticket_resolution_evidence_present"] is True
    assert "lead@acme.example" not in str(snapshot_payload)
    assert "lead@acme.example" not in str(gated_result["snapshot"])

    record = await store.get_artifact_record(
        account_id="acct-portfolio-submit",
        request_id=payload["request_id"],
    )
    assert record is not None
    assert record.delivery_email == "lead@acme.example"

    artifact = _route(router, "/ops/deflection-reports/{request_id}/artifact", "GET")
    with pytest.raises(api_module.HTTPException) as locked:
        await artifact.endpoint(request_id=payload["request_id"])
    assert locked.value.status_code == 403

    paid = _route(router, "/ops/deflection-reports/{request_id}/paid", "POST")
    assert await paid.endpoint(
        payload=api_module.DeflectionReportPaidModel(
            payment_reference="checkout-session:test"
        ),
        request_id=payload["request_id"],
    ) == {"request_id": payload["request_id"], "paid": True}
    artifact_payload = await artifact.endpoint(request_id=payload["request_id"])
    assert "lead@acme.example" not in str(artifact_payload)


@pytest.mark.asyncio
async def test_deflection_submit_accepts_zendesk_full_thread_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    artifact_data = ZENDESK_THREAD_SAMPLE.read_bytes()
    calls = _install_blob(monkeypatch, artifact_data)
    store = InMemoryDeflectionReportArtifactStore()
    router = _router(store)

    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    payload = await submit.endpoint({
        "blob_url": "https://portfolio.example/blob/zendesk-thread.json",
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
        "importer_mode": "full_thread",
    })

    assert calls == [(
        "https://portfolio.example/blob/zendesk-thread.json",
        "93.184.216.34",
        api_module._DEFLECTION_SUBMIT_FETCH_TIMEOUT_SECONDS,
    )]
    assert payload["status"] == "completed"
    metadata = payload["input_provider"]["metadata"]
    assert metadata["source_row_count"] == 4
    assert metadata["submitted_row_count"] == 4
    assert metadata["included_row_count"] == 4
    assert metadata["blob_bytes"] == len(artifact_data)
    assert metadata["importer_mode"] == "full_thread"
    assert metadata["support_platform"] == "zendesk"
    assert metadata["support_ticket_resolution_evidence_present"] is True
    assert metadata["support_ticket_resolution_evidence_count"] == 2
    assert metadata["ticket_status_present"] is True
    assert metadata["ticket_status_present_count"] == 4
    assert metadata["ticket_status_summary"] == {"resolved": 1, "open": 3}
    assert metadata["csat_present"] is True
    assert metadata["csat_present_count"] == 1
    assert metadata["csat_score_count"] == 0
    assert metadata["csat_score_average"] is None
    assert payload["input_provider"]["warnings"] == []
    assert "Internal note" not in json.dumps(payload)
    assert "A member of the support team will get back to you" not in json.dumps(payload)
    gated_result = payload["steps"][0]["result"]
    assert gated_result["full_report"] == {
        "status": "locked",
        "reason": "payment_required",
    }
    assert "markdown" not in str(gated_result)

    artifact = _route(router, "/ops/deflection-reports/{request_id}/artifact", "GET")
    with pytest.raises(api_module.HTTPException) as locked:
        await artifact.endpoint(request_id=payload["request_id"])
    assert locked.value.status_code == 403

    paid = _route(router, "/ops/deflection-reports/{request_id}/paid", "POST")
    assert await paid.endpoint(
        payload=api_module.DeflectionReportPaidModel(
            payment_reference="checkout-session:zendesk-full-thread"
        ),
        request_id=payload["request_id"],
    ) == {"request_id": payload["request_id"], "paid": True}

    artifact_payload = await artifact.endpoint(request_id=payload["request_id"])
    summary = artifact_payload["summary"]
    assert summary["drafted_answer_count"] == 1
    assert summary["support_ticket_resolution_evidence_present"] is True
    assert summary["support_ticket_resolution_evidence_count"] == 1
    assert summary["no_proven_answer_count"] == 0
    assert artifact_payload["faq_result"]["items"][0]["answer_evidence_status"] == (
        "resolution_evidence"
    )
    assert list(artifact_payload["faq_result"]["items"][0]["source_ids"]) == ["2"]
    assert "duplicate billing event and refunded the extra charge" in (
        artifact_payload["markdown"]
    )
    assert "Internal note" not in json.dumps(artifact_payload)
    assert "A member of the support team will get back to you" not in json.dumps(
        artifact_payload
    )
    assert "We sent the standard resolution steps" not in artifact_payload["markdown"]
    assert "This is still broken after trying the steps" not in artifact_payload[
        "markdown"
    ]
    assert "lead@acme.example" not in json.dumps(artifact_payload)


@pytest.mark.asyncio
async def test_deflection_submit_accepts_multipart_csv_bytes() -> None:
    csv_data = _csv_bytes([
        "ticket_id,subject,message,resolution_text,pain_category",
        (
            "ticket-export-1,Export attribution,"
            "How do I export attribution reports?,"
            "Open Analytics and click Download report.,exports"
        ),
        (
            "ticket-export-2,Report download,"
            "Where is the report download for attribution exports?,"
            "Open Analytics and click Download report.,exports"
        ),
        "ticket-sso-1,SSO setup,How do I enable SSO for my team?,,auth",
    ])
    store = InMemoryDeflectionReportArtifactStore()
    router = _router(store)

    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    request = _FormRequest({
        "csv_file": _Upload(csv_data),
        "support_platform": "help-scout",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
        "limit": "2",
    })
    payload = await submit.endpoint(request)

    assert request.form_kwargs == {
        "max_part_size": api_module._MAX_DEFLECTION_SUBMIT_BLOB_BYTES
    }
    assert payload["status"] == "completed"
    assert payload["request_id"].startswith("content-ops-")
    gated_result = payload["steps"][0]["result"]
    assert gated_result["request_id"] == payload["request_id"]
    assert gated_result["full_report"] == {
        "status": "locked",
        "reason": "payment_required",
    }
    assert "markdown" not in str(gated_result)
    assert "ticket-export-1" not in str(gated_result)
    assert payload["input_provider"]["metadata"] == {
        "cluster_quality": {
            "cluster_count": 1,
            "clustered_row_count": 2,
            "largest_cluster_count": 2,
            "singleton_cluster_count": 0,
            "uncategorized_row_count": 0,
        },
        "included_row_count": 2,
        "max_source_material_rows": 2,
        "skipped_row_count": 0,
        "source": "portfolio_deflection_submit",
            "source_period": "Uploaded support tickets",
            "source_row_count": 3,
            "submitted_row_count": 2,
            "support_platform": "help_scout",
            "support_ticket_resolution_evidence_count": 2,
            "support_ticket_resolution_evidence_present": True,
            "top_ticket_clusters": [{"label": "exports", "count": 2}],
            "truncated_row_count": 1,
            "uploaded_bytes": len(csv_data),
    }

    snapshot = _route(router, "/ops/deflection-reports/{request_id}/snapshot", "GET")
    assert await snapshot.endpoint(request_id=payload["request_id"]) == gated_result["snapshot"]


@pytest.mark.asyncio
async def test_deflection_submit_defaults_to_all_rows_that_fit_upload_guard() -> None:
    row_count = api_module._MAX_INGESTION_ROWS + 5
    csv_data = _csv_bytes([
        "ticket_id,subject,message,resolution_text,pain_category",
        *(
            (
                f"ticket-export-{index},Export reports,"
                f"How do I export report batch {index}?,"
                "Open Analytics and click Download report.,exports"
            )
            for index in range(row_count)
        ),
    ])
    store = InMemoryDeflectionReportArtifactStore()
    router = _router(store)

    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    request = _FormRequest({
        "csv_file": _Upload(csv_data),
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
    })
    payload = await submit.endpoint(request)

    metadata = payload["input_provider"]["metadata"]
    assert metadata["source_row_count"] == row_count
    assert metadata["submitted_row_count"] == row_count
    assert metadata["included_row_count"] == row_count
    assert metadata["truncated_row_count"] == 0
    assert metadata["max_source_material_rows"] == row_count
    assert payload["input_provider"]["warnings"] == []
    assert payload["steps"][0]["result"]["full_report"] == {
        "status": "locked",
        "reason": "payment_required",
    }


@pytest.mark.asyncio
async def test_deflection_submit_request_limit_becomes_csv_parse_cap() -> None:
    csv_data = _csv_bytes([
        "ticket_id,subject,message,resolution_text,pain_category",
        (
            "ticket-export-1,Export attribution,"
            "How do I export attribution reports?,"
            "Open Analytics and click Download report.,exports"
        ),
        (
            "ticket-export-2,Report download,"
            "Where is the report download for attribution exports?,"
            "Open Analytics and click Download report.,exports"
        ),
        "ticket-sso-1,SSO setup,How do I enable SSO for my team?,,auth",
    ])

    request = _FormRequest({
        "csv_file": _Upload(csv_data),
        "support_platform": "help-scout",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
        "limit": "2",
    })

    (
        data,
        rows,
        byte_count,
        byte_count_key,
        load_warnings,
        parser_source_row_count,
    ) = await api_module._load_deflection_submit_rows_from_request(
        request,
        max_bytes=2048,
    )

    assert data["limit"] == 2
    assert byte_count == len(csv_data)
    assert byte_count_key == "uploaded_bytes"
    assert load_warnings == ()
    assert parser_source_row_count == 3
    assert [row["ticket_id"] for row in rows] == [
        "ticket-export-1",
        "ticket-export-2",
    ]


@pytest.mark.asyncio
async def test_deflection_submit_filters_non_english_huggingface_shaped_rows() -> None:
    csv_data = _csv_dict_bytes([
        {
            "subject": "Instructions for Returning Products",
            "body": "How do I return a recent purchase from your store?",
            "answer": (
                "Items should be returned within 30 days in their original "
                "condition. Start the return in the online returns portal."
            ),
            "language": "en",
            "queue": "Returns and Exchanges",
        },
        {
            "subject": "Synchronisationsproblem",
            "body": "Ich erfahre Schwierigkeiten bei der Synchronisation.",
            "answer": "Bitte senden Sie aktuelle Fehlerprotokolle.",
            "language": "de",
            "queue": "Technical Support",
        },
    ])
    store = InMemoryDeflectionReportArtifactStore()
    router = _router(store)

    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    request = _FormRequest({
        "csv_file": _Upload(csv_data),
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
    })
    payload = await submit.endpoint(request)

    metadata = payload["input_provider"]["metadata"]
    assert metadata["loaded_source_row_count"] == 2
    assert metadata["source_row_count"] == 2
    assert metadata["submitted_row_count"] == 1
    assert metadata["truncated_row_count"] == 0
    assert metadata["included_row_count"] == 1
    assert metadata["language_filtered_row_count"] == 1
    assert metadata["support_ticket_resolution_evidence_count"] == 1
    assert payload["input_provider"]["warnings"] == [{
        "code": "deflection_submit_non_english_rows_filtered",
        "message": "Skipped 1 non-English support-ticket rows.",
        "row_count": 2,
        "filtered_row_count": 1,
    }]
    snapshot = payload["steps"][0]["result"]["snapshot"]
    assert snapshot["summary"]["support_ticket_resolution_evidence_count"] == 1
    assert snapshot["summary"]["drafted_answer_count"] == 1
    assert "Synchronisationsproblem" not in str(payload)


@pytest.mark.asyncio
async def test_deflection_submit_preserves_parser_truncation_after_language_filter() -> None:
    csv_data = _csv_dict_bytes([
        {
            "ticket_id": "ticket-export-1",
            "subject": "Export reports",
            "body": "How do I export attribution reports?",
            "answer": "Open Analytics and click Download report.",
            "language": "en",
            "queue": "Exports",
        },
        {
            "ticket_id": "ticket-sync-de",
            "subject": "Synchronisationsproblem",
            "body": "Ich erfahre Schwierigkeiten bei der Synchronisation.",
            "answer": "Bitte senden Sie aktuelle Fehlerprotokolle.",
            "language": "de",
            "queue": "Technical Support",
        },
        {
            "ticket_id": "ticket-sso-1",
            "subject": "SSO setup",
            "body": "How do I enable SSO for my team?",
            "answer": "Open Settings, then configure SAML SSO.",
            "language": "en",
            "queue": "Authentication",
        },
    ])
    store = InMemoryDeflectionReportArtifactStore()
    router = _router(store)

    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    request = _FormRequest({
        "csv_file": _Upload(csv_data),
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
        "limit": "2",
    })
    payload = await submit.endpoint(request)

    metadata = payload["input_provider"]["metadata"]
    assert metadata["loaded_source_row_count"] == 3
    assert metadata["source_row_count"] == 3
    assert metadata["submitted_row_count"] == 1
    assert metadata["truncated_row_count"] == 1
    assert metadata["max_source_material_rows"] == 2
    assert metadata["language_filtered_row_count"] == 1
    warnings = payload["input_provider"]["warnings"]
    assert [warning["code"] for warning in warnings] == [
        "deflection_submit_non_english_rows_filtered",
        "deflection_submit_rows_truncated",
    ]
    assert warnings[1]["row_count"] == 3
    assert warnings[1]["max_rows"] == 2
    assert warnings[1]["truncated_row_count"] == 1
    assert "ticket-sso-1" not in str(payload)


@pytest.mark.asyncio
async def test_deflection_submit_rejects_all_non_english_language_marked_rows() -> None:
    csv_data = _csv_dict_bytes([
        {
            "subject": "Synchronisationsproblem",
            "body": "Ich erfahre Schwierigkeiten bei der Synchronisation.",
            "answer": "Bitte senden Sie aktuelle Fehlerprotokolle.",
            "language": "de",
        },
    ])
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint(_FormRequest({
            "csv_file": _Upload(csv_data),
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        }))

    assert exc.value.status_code == 400
    assert exc.value.detail == {
        "reason": "deflection_submit_no_english_rows",
        "message": "Submitted CSV did not contain English support-ticket rows.",
        "source_row_count": 1,
        "language_filtered_row_count": 1,
    }


@pytest.mark.asyncio
async def test_deflection_submit_surfaces_skipped_prologue_row_warning() -> None:
    csv_data = _csv_bytes([
        "Zendesk ticket export",
        "ticket_id,subject,message,resolution_text",
        (
            "ticket-1,Export attribution,"
            "How do I export attribution reports?,"
            "Open Analytics and click Download report."
        ),
        (
            "ticket-2,Report download,"
            "Where is the report download for attribution exports?,"
            "Open Analytics and click Download report."
        ),
    ])
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    request = _FormRequest({
        "csv_file": _Upload(csv_data),
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
    })

    payload = await submit.endpoint(request)

    metadata = payload["input_provider"]["metadata"]
    assert metadata["source_row_count"] == 2
    assert metadata["submitted_row_count"] == 2
    warnings = payload["input_provider"]["warnings"]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "csv_leading_rows_skipped"
    assert warnings[0]["row_index"] == 1
    assert "Zendesk ticket export" in warnings[0]["message"]


@pytest.mark.parametrize(
    "value",
    [
        "en",
        "EN",
        "eng",
        "English",
        "en-US",
        "en_GB",
        "English (US)",
        "English (United Kingdom)",
        "english(us)",
        "en (US)",
    ],
)
def test_is_english_language_accepts_common_and_display_forms(value: str) -> None:
    assert api_module._is_english_language(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "de",
        "German",
        "Deutsch (Deutschland)",
        "es-MX",
        "fr_FR",
        "(US)",
    ],
)
def test_is_english_language_rejects_non_english_forms(value: str) -> None:
    assert api_module._is_english_language(value) is False


@pytest.mark.asyncio
async def test_deflection_submit_surfaces_cluster_preview_for_messy_untagged_csv() -> None:
    csv_data = _csv_bytes([
        "ticket_id,subject,message",
        "zd-1,Password reset help,<p>How do I reset my password from the login screen?</p>",
        "zd-2,Password reset not working,I cannot reset my password from the login screen",
        "hs-1,Change email address,Where do I update my email?",
        "hs-2,Update account email,Need to change email address",
    ])
    store = InMemoryDeflectionReportArtifactStore()
    router = _router(store)

    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    request = _FormRequest({
        "csv_file": _Upload(csv_data),
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
    })
    payload = await submit.endpoint(request)

    metadata = payload["input_provider"]["metadata"]
    assert metadata["top_ticket_clusters"] == [
        {"label": "login password reset", "count": 2},
        {"label": "email update", "count": 2},
    ]
    assert metadata["cluster_quality"] == {
        "cluster_count": 2,
        "clustered_row_count": 4,
        "largest_cluster_count": 2,
        "singleton_cluster_count": 0,
        "uncategorized_row_count": 0,
    }
    assert metadata["support_ticket_resolution_evidence_count"] == 0
    assert metadata["support_ticket_resolution_evidence_present"] is False

    snapshot = payload["steps"][0]["result"]["snapshot"]
    assert snapshot["summary"]["support_ticket_resolution_evidence_count"] == 0
    assert snapshot["summary"]["support_ticket_resolution_evidence_present"] is False
    assert [
        (item["ticket_count"], item["customer_wording"])
        for item in snapshot["top_questions"]
    ] == [
        (2, "Password reset help How do I reset my password from the login screen?"),
        (2, "Change email address Where do I update my email?"),
    ]
    assert "<p>" not in str(snapshot)
    assert payload["steps"][0]["result"]["full_report"] == {
        "status": "locked",
        "reason": "payment_required",
    }

    stored_snapshot = _route(router, "/ops/deflection-reports/{request_id}/snapshot", "GET")
    assert await stored_snapshot.endpoint(request_id=payload["request_id"]) == snapshot


@pytest.mark.asyncio
async def test_deflection_submit_reports_cluster_preview_skip_for_large_untagged_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extracted_content_pipeline import support_ticket_clustering as clustering

    monkeypatch.setattr(clustering, "MAX_TOKEN_SET_CLUSTER_ROWS", 3)
    csv_data = _csv_bytes([
        "ticket_id,subject,message",
        "zd-1,Password reset help,How do I reset my password?",
        "zd-2,Change email address,Where do I update my email?",
        "zd-3,Invoice download missing,Where can I download my invoice?",
        "zd-4,Export report broken,The report export button does nothing",
    ])
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    request = _FormRequest({
        "csv_file": _Upload(csv_data),
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
    })

    payload = await submit.endpoint(request)

    metadata = payload["input_provider"]["metadata"]
    assert metadata["cluster_preview_skipped"] is True
    assert metadata["cluster_preview_token_set_row_count"] == 4
    assert metadata["included_row_count"] == 4
    assert metadata["cluster_quality"]["uncategorized_row_count"] == 4
    skip_warnings = [
        warning
        for warning in payload["input_provider"]["warnings"]
        if warning["code"] == "cluster_preview_skipped_large_upload"
    ]
    assert len(skip_warnings) == 1
    assert skip_warnings[0]["row_count"] == 4
    assert skip_warnings[0]["max_token_set_rows"] == 3


@pytest.mark.asyncio
async def test_deflection_submit_accepts_provider_export_fixture_with_resolution_diagnostics() -> None:
    csv_data = (
        PROVIDER_FIXTURE_DIR / "zendesk_full_thread_export.csv"
    ).read_bytes()
    store = InMemoryDeflectionReportArtifactStore()
    router = _router(store)

    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    request = _FormRequest({
        "csv_file": _Upload(csv_data, filename="zendesk_full_thread_export.csv"),
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
    })
    payload = await submit.endpoint(request)

    metadata = payload["input_provider"]["metadata"]
    assert metadata["source_row_count"] == 3
    assert metadata["submitted_row_count"] == 3
    assert metadata["included_row_count"] == 3
    assert metadata["skipped_row_count"] == 0
    assert metadata["truncated_row_count"] == 0
    assert metadata["support_platform"] == "zendesk"
    assert metadata["support_ticket_resolution_evidence_count"] == 2
    assert metadata["support_ticket_resolution_evidence_present"] is True
    assert metadata["cluster_quality"]["largest_cluster_count"] >= 2
    assert metadata["uploaded_bytes"] == len(csv_data)
    assert payload["input_provider"]["warnings"] == []

    snapshot = payload["steps"][0]["result"]["snapshot"]
    assert snapshot["summary"]["support_ticket_resolution_evidence_count"] == 2
    assert snapshot["summary"]["support_ticket_resolution_evidence_present"] is True
    assert snapshot["summary"]["drafted_answer_count"] >= 1
    assert "<p>" not in str(snapshot)
    assert "maya@example.test" not in str(payload)
    assert payload["steps"][0]["result"]["full_report"] == {
        "status": "locked",
        "reason": "payment_required",
    }


def test_deflection_submit_fastapi_dispatch_accepts_multipart_csv_bytes() -> None:
    csv_data = _csv_bytes([
        "ticket_id,subject,message,resolution_text,pain_category",
        (
            "ticket-export-1,Export attribution,"
            "How do I export attribution reports?,"
            "Open Analytics and click Download report.,exports"
        ),
        (
            "ticket-export-2,Report download,"
            "Where is the report download for attribution exports?,"
            "Open Analytics and click Download report.,exports"
        ),
        "ticket-sso-1,SSO setup,How do I enable SSO for my team?,,auth",
    ])
    store = InMemoryDeflectionReportArtifactStore()
    app = FastAPI()
    app.include_router(_router(store))

    with TestClient(app) as client:
        response = client.post(
            "/ops/deflection-reports/submit",
            data={
                "support_platform": "zendesk",
                "company_name": "Acme Co.",
                "contact_email": "lead@acme.example",
                "limit": "2",
            },
            files={"csv_file": ("tickets.csv", csv_data, "text/csv")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["request_id"].startswith("content-ops-")
    gated_result = payload["steps"][0]["result"]
    assert gated_result["full_report"] == {
        "status": "locked",
        "reason": "payment_required",
    }
    assert payload["input_provider"]["metadata"]["uploaded_bytes"] == len(csv_data)
    assert payload["input_provider"]["metadata"]["support_platform"] == "zendesk"
    assert "markdown" not in str(gated_result)


@pytest.mark.asyncio
async def test_deflection_submit_rejects_multipart_without_csv_file() -> None:
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint(_FormRequest({
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        }))

    assert exc.value.status_code == 422
    assert exc.value.detail == "csv_file is required"


@pytest.mark.asyncio
async def test_deflection_submit_accepts_full_thread_multipart_json_upload() -> None:
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    artifact_data = ZENDESK_THREAD_SAMPLE.read_bytes()
    upload = _Upload(artifact_data, filename="zendesk-thread.json")
    request = _FormRequest({
        "json_file": upload,
        "support_platform": "zendesk",
        "company_name": "Acme Co.",
        "contact_email": "lead@acme.example",
        "importer_mode": "full_thread",
    })

    payload = await submit.endpoint(request)

    assert request.form_kwargs == {"max_part_size": api_module._MAX_DEFLECTION_SUBMIT_BLOB_BYTES}
    assert upload.read_sizes
    assert all(
        size == api_module._DEFLECTION_SUBMIT_UPLOAD_CHUNK_BYTES
        for size in upload.read_sizes
    )
    assert api_module._MAX_DEFLECTION_SUBMIT_BLOB_BYTES + 1 not in upload.read_sizes
    assert payload["status"] == "completed"
    metadata = payload["input_provider"]["metadata"]
    assert metadata["source_row_count"] == 4
    assert metadata["submitted_row_count"] == 4
    assert metadata["included_row_count"] == 4
    assert metadata["uploaded_bytes"] == len(artifact_data)
    assert metadata["importer_mode"] == "full_thread"
    assert metadata["support_platform"] == "zendesk"
    assert metadata["support_ticket_resolution_evidence_present"] is True
    assert metadata["support_ticket_resolution_evidence_count"] == 2
    assert metadata["ticket_status_present"] is True
    assert metadata["ticket_status_summary"] == {"resolved": 1, "open": 3}
    assert metadata["csat_present"] is True
    assert metadata["csat_present_count"] == 1
    assert metadata["csat_score_count"] == 0
    assert metadata["csat_score_average"] is None
    assert payload["input_provider"]["warnings"] == []
    assert "Internal note" not in json.dumps(payload)
    assert "A member of the support team will get back to you" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_deflection_submit_rejects_full_thread_multipart_csv_file() -> None:
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint(_FormRequest({
            "csv_file": _Upload(b"ticket_id,subject,message\n1,Question,How?\n"),
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
            "importer_mode": "full_thread",
        }))

    assert exc.value.status_code == 422
    assert exc.value.detail == (
        "csv_file is not accepted with importer_mode=full_thread"
    )


@pytest.mark.asyncio
async def test_deflection_submit_rejects_full_thread_without_json_file() -> None:
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint(_FormRequest({
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
            "importer_mode": "full_thread",
        }))

    assert exc.value.status_code == 422
    assert exc.value.detail == "json_file is required"


@pytest.mark.asyncio
async def test_deflection_submit_rejects_json_file_without_full_thread_mode() -> None:
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint(_FormRequest({
            "json_file": _Upload(ZENDESK_THREAD_SAMPLE.read_bytes(), filename="zendesk.json"),
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        }))

    assert exc.value.status_code == 422
    assert exc.value.detail == "json_file requires importer_mode=full_thread"


@pytest.mark.asyncio
async def test_deflection_submit_rejects_oversize_uploaded_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module, "_MAX_DEFLECTION_SUBMIT_BLOB_BYTES", 10)
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint(_FormRequest({
            "json_file": _Upload(b"01234567890", filename="zendesk-thread.json"),
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
            "importer_mode": "full_thread",
        }))

    assert exc.value.status_code == 413
    assert exc.value.detail == {
        "reason": "deflection_submit_full_thread_too_large",
        "max_file_bytes": 10,
    }


@pytest.mark.asyncio
async def test_deflection_submit_rejects_empty_uploaded_json() -> None:
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint(_FormRequest({
            "json_file": _Upload(b"", filename="zendesk-thread.json"),
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
            "importer_mode": "full_thread",
        }))

    assert exc.value.status_code == 400
    assert exc.value.detail == "Uploaded Zendesk full-thread JSON is empty."


@pytest.mark.asyncio
async def test_deflection_submit_rejects_oversize_uploaded_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module, "_MAX_DEFLECTION_SUBMIT_BLOB_BYTES", 10)
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint(_FormRequest({
            "csv_file": _Upload(b"01234567890"),
            "support_platform": "intercom",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        }))

    assert exc.value.status_code == 413
    assert exc.value.detail == {
        "reason": "deflection_submit_csv_too_large",
        "max_file_bytes": 10,
    }


@pytest.mark.asyncio
async def test_deflection_submit_rejects_oversize_multipart_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module, "_MAX_DEFLECTION_SUBMIT_BLOB_BYTES", 10)
    monkeypatch.setattr(api_module, "_MAX_DEFLECTION_SUBMIT_MULTIPART_OVERHEAD_BYTES", 3)
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")
    request = _FormRequest(
        {
            "csv_file": _Upload(b"small"),
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        },
        headers={
            "content-type": "multipart/form-data; boundary=test",
            "content-length": "14",
        },
    )

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint(request)

    assert exc.value.status_code == 413
    assert exc.value.detail == {
        "reason": "deflection_submit_csv_too_large",
        "max_file_bytes": 10,
    }
    assert request.form_kwargs is None


@pytest.mark.asyncio
async def test_deflection_submit_rejects_non_https_blob_url() -> None:
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint({
            "blob_url": "http://portfolio.example/blob/tickets.csv",
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        })

    assert exc.value.status_code == 422
    assert "blob_url must be an https URL" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_deflection_submit_rejects_unknown_importer_mode() -> None:
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint({
            "blob_url": "https://portfolio.example/blob/tickets.csv",
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
            "importer_mode": "xml",
        })

    assert exc.value.status_code == 422
    assert "importer_mode must be one of: csv, full_thread" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_deflection_submit_rejects_blob_url_with_invalid_port() -> None:
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint({
            "blob_url": "https://portfolio.example:99999/blob/tickets.csv",
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        })

    assert exc.value.status_code == 422
    assert "blob_url must be an https URL" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_deflection_submit_rejects_oversize_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    monkeypatch.setattr(api_module, "_MAX_DEFLECTION_SUBMIT_BLOB_BYTES", 10)
    _install_blob(monkeypatch, b"01234567890")
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint({
            "blob_url": "https://portfolio.example/blob/tickets.csv",
            "support_platform": "intercom",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        })

    assert exc.value.status_code == 413
    assert exc.value.detail == {
        "reason": "deflection_submit_blob_too_large",
        "max_file_bytes": 10,
    }


@pytest.mark.asyncio
async def test_deflection_submit_rejects_malformed_full_thread_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    _install_blob(monkeypatch, b"{not-json")
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint({
            "blob_url": "https://portfolio.example/blob/zendesk-thread.json",
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
            "importer_mode": "full_thread",
        })

    assert exc.value.status_code == 400
    assert exc.value.detail == "Zendesk full-thread JSON could not be parsed."


@pytest.mark.asyncio
async def test_deflection_submit_rejects_csv_without_usable_ticket_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    _install_blob(
        monkeypatch,
        _csv_bytes([
            "ticket_id,created_at",
            "ticket-empty,2026-05-01",
        ]),
    )
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint({
            "blob_url": "https://portfolio.example/blob/tickets.csv",
            "support_platform": "help_scout",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        })

    assert exc.value.status_code == 400
    assert exc.value.detail == {
        "reason": "deflection_submit_no_usable_rows",
        "message": "Blob CSV did not include usable support-ticket wording.",
        "source_row_count": 1,
        "max_source_material_rows": 1,
    }


@pytest.mark.asyncio
async def test_deflection_submit_rejects_hostname_resolving_to_private_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _getaddrinfo(host: str, port: int, *, type: int):
        assert host == "blob.attacker.example"
        return [(
            socket.AF_INET,
            type,
            6,
            "",
            ("169.254.169.254", port),
        )]

    def _open(_request: Any, *, timeout: int) -> _BlobResponse:
        raise AssertionError("private DNS target must be rejected before fetch")

    monkeypatch.setattr(api_module.socket, "getaddrinfo", _getaddrinfo)
    monkeypatch.setattr(api_module, "_open_https_blob_request", _open)
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint({
            "blob_url": "https://blob.attacker.example/tickets.csv",
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        })

    assert exc.value.status_code == 400
    assert exc.value.detail == "blob_url host is not allowed"


@pytest.mark.asyncio
async def test_deflection_submit_rejects_non_ip_literal_internal_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _getaddrinfo(host: str, port: int, *, type: int):
        assert host == "metadata.google.internal"
        return [(
            socket.AF_INET,
            type,
            6,
            "",
            ("169.254.169.254", port),
        )]

    monkeypatch.setattr(api_module.socket, "getaddrinfo", _getaddrinfo)
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint({
            "blob_url": "https://metadata.google.internal/computeMetadata/v1/",
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        })

    assert exc.value.status_code == 400
    assert exc.value.detail == "blob_url host is not allowed"


@pytest.mark.asyncio
async def test_deflection_submit_rejects_blob_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    calls: list[str] = []

    def _open(target: Any, *, timeout: int) -> _BlobResponse:
        calls.append(target.url)
        return _BlobResponse(b"", status=302)

    monkeypatch.setattr(api_module, "_open_https_blob_request", _open)
    router = _router(InMemoryDeflectionReportArtifactStore())
    submit = _route(router, "/ops/deflection-reports/submit", "POST")

    with pytest.raises(api_module.HTTPException) as exc:
        await submit.endpoint({
            "blob_url": "https://portfolio.example/blob/tickets.csv",
            "support_platform": "zendesk",
            "company_name": "Acme Co.",
            "contact_email": "lead@acme.example",
        })

    assert calls == ["https://portfolio.example/blob/tickets.csv"]
    assert exc.value.status_code == 400
    assert exc.value.detail == "Blob URL redirects are not allowed."


def test_pinned_blob_response_closes_connection_if_response_close_fails() -> None:
    class _Response:
        status = 200

        def read(self, _size: int) -> bytes:
            return b""

        def close(self) -> None:
            raise OSError("response close failed")

    class _Connection:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    connection = _Connection()
    response = api_module._PinnedBlobResponse(_Response(), connection)  # type: ignore[arg-type]

    with pytest.raises(OSError, match="response close failed"):
        response.close()

    assert connection.closed


def test_open_validated_https_blob_response_closes_and_logs_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_public_dns(monkeypatch)

    class _StatusRaisesResponse:
        closed = False

        @property
        def status(self) -> int:
            raise OSError("status failed")

        def close(self) -> None:
            self.closed = True

    response = _StatusRaisesResponse()
    caplog.set_level("WARNING", logger=api_module.logger.name)
    monkeypatch.setattr(
        api_module,
        "_open_https_blob_request",
        lambda _target, *, timeout: response,
    )

    with pytest.raises(api_module.HTTPException) as exc:
        api_module._open_validated_https_blob_response(
            "https://portfolio.example/blob/tickets.csv?sig=secret",
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Blob URL could not be fetched."
    assert response.closed
    records = [
        record
        for record in caplog.records
        if record.message == "Content Ops deflection blob fetch failed"
    ]
    assert len(records) == 1
    assert getattr(records[0], "blob_host") == "portfolio.example"
    assert getattr(records[0], "blob_path") == "/blob/tickets.csv"
    assert "secret" not in caplog.text


def test_read_bounded_https_blob_rejects_redirect_status_without_following() -> None:
    response = _BlobResponse(b"", status=302)

    with pytest.raises(api_module.HTTPException) as exc:
        with pytest.MonkeyPatch.context() as monkeypatch:
            _install_public_dns(monkeypatch)
            monkeypatch.setattr(
                api_module,
                "_open_https_blob_request",
                lambda _target, *, timeout: response,
            )
            api_module._read_bounded_https_blob(
                "https://portfolio.example/blob/tickets.csv",
                max_bytes=100,
            )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Blob URL redirects are not allowed."
    assert response.closed


def test_read_bounded_https_blob_rejects_non_success_status() -> None:
    response = _BlobResponse(b"expired", status=403)

    with pytest.raises(api_module.HTTPException) as exc:
        with pytest.MonkeyPatch.context() as monkeypatch:
            _install_public_dns(monkeypatch)
            monkeypatch.setattr(
                api_module,
                "_open_https_blob_request",
                lambda _target, *, timeout: response,
            )
            api_module._read_bounded_https_blob(
                "https://portfolio.example/blob/tickets.csv",
                max_bytes=100,
            )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Blob URL could not be fetched."
    assert response.closed


def test_read_bounded_https_blob_rejects_malformed_http_response() -> None:
    with pytest.raises(api_module.HTTPException) as exc:
        with pytest.MonkeyPatch.context() as monkeypatch:
            _install_public_dns(monkeypatch)

            def _open(_target: Any, *, timeout: int) -> _BlobResponse:
                raise http.client.BadStatusLine("not-http")

            monkeypatch.setattr(api_module, "_open_https_blob_request", _open)
            api_module._read_bounded_https_blob(
                "https://portfolio.example/blob/tickets.csv",
                max_bytes=100,
            )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Blob URL could not be fetched."


def test_read_bounded_https_blob_uses_validated_ip_for_pinned_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_dns(monkeypatch)
    csv_data = _csv_bytes(["ticket_id,message", "ticket-1,How do I export?"])
    calls: list[tuple[str, int, str, int, str, dict[str, str]]] = []

    class _PinnedConnection:
        def __init__(
            self,
            host: str,
            *,
            port: int,
            connect_host: str,
            timeout: int,
        ) -> None:
            self.host = host
            self.port = port
            self.connect_host = connect_host
            self.timeout = timeout
            self.closed = False

        def request(
            self,
            method: str,
            path: str,
            *,
            headers: dict[str, str],
        ) -> None:
            calls.append((
                self.host,
                self.port,
                self.connect_host,
                self.timeout,
                f"{method} {path}",
                headers,
            ))

        def getresponse(self) -> _BlobResponse:
            return _BlobResponse(csv_data)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(api_module, "_PINNED_HTTPS_CONNECTION_CLASS", _PinnedConnection)

    data = api_module._read_bounded_https_blob(
        "https://portfolio.example:444/blob/tickets.csv?sig=abc",
        max_bytes=1000,
    )

    assert data == csv_data
    assert calls == [(
        "portfolio.example",
        444,
        "93.184.216.34",
        api_module._DEFLECTION_SUBMIT_FETCH_TIMEOUT_SECONDS,
        "GET /blob/tickets.csv?sig=abc",
        {
            "Host": "portfolio.example:444",
            "User-Agent": "Atlas-Content-Ops/1.0",
        },
    )]


def test_pinned_https_connection_uses_validated_ip_but_original_sni() -> None:
    calls: list[tuple[str, object]] = []

    class _Context:
        def wrap_socket(self, sock: object, *, server_hostname: str) -> object:
            calls.append(("sni", server_hostname))
            return ("tls", sock, server_hostname)

    connection = api_module._PinnedHTTPSConnection(
        "portfolio.example",
        port=443,
        connect_host="93.184.216.34",
        timeout=15,
    )
    connection._context = _Context()
    connection._create_connection = lambda address, timeout, source_address: calls.append((
        "connect",
        (address, timeout, source_address),
    )) or "socket"

    connection.connect()

    assert calls == [
        ("connect", (("93.184.216.34", 443), 15, None)),
        ("sni", "portfolio.example"),
    ]
    assert connection.sock == ("tls", "socket", "portfolio.example")
