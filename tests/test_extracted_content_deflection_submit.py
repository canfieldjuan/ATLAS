from __future__ import annotations

import socket
import urllib.error
from typing import Any

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


def _route(router: Any, path: str, method: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


class _BlobResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self, _size: int) -> bytes:
        return self._data

    def __enter__(self) -> "_BlobResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _csv_bytes(rows: list[str]) -> bytes:
    return ("\n".join(rows) + "\n").encode("utf-8")


def _install_blob(
    monkeypatch: pytest.MonkeyPatch,
    data: bytes,
) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    def _open(request: Any, *, timeout: int) -> _BlobResponse:
        calls.append((request.full_url, timeout))
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
    assert payload["input_provider"]["metadata"] == {
        "blob_bytes": len(csv_data),
        "included_row_count": 2,
        "max_source_material_rows": 2,
        "skipped_row_count": 0,
        "source": "portfolio_deflection_submit",
        "source_period": "Uploaded support tickets",
        "source_row_count": 3,
        "submitted_row_count": 2,
        "support_platform": "zendesk",
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
    assert await snapshot.endpoint(request_id=payload["request_id"]) == gated_result["snapshot"]

    artifact = _route(router, "/ops/deflection-reports/{request_id}/artifact", "GET")
    with pytest.raises(api_module.HTTPException) as locked:
        await artifact.endpoint(request_id=payload["request_id"])
    assert locked.value.status_code == 403


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
        "max_source_material_rows": 1000,
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

    def _open(request: Any, *, timeout: int) -> _BlobResponse:
        calls.append(request.full_url)
        raise urllib.error.HTTPError(
            request.full_url,
            302,
            "Found",
            {"Location": "http://169.254.169.254/latest/meta-data"},
            None,
        )

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


def test_deflection_submit_opener_disables_redirects() -> None:
    handler = api_module._NoRedirectHandler()

    assert handler.redirect_request(None, None, 302, "Found", {}, "https://x") is None
