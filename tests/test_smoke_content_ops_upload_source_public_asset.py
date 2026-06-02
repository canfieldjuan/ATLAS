from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import urllib.request

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smoke_content_ops_upload_source_public_asset as smoke


class _Response:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def getcode(self) -> int:
        return self.status


class _Transport:
    def __init__(self, *payloads: Any) -> None:
        self.payloads = list(payloads)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 11.0
        self.requests.append(request)
        assert self.payloads, f"unexpected request: {request.full_url}"
        return _Response(self.payloads.pop(0))


def _args(csv_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        api_base_url="https://atlas-preview.example.com",
        token="secret-token",
        csv=csv_path,
        target_mode="vendor_retention",
        source="tickets.csv",
        timeout=11.0,
        import_path=smoke.DEFAULT_IMPORT_PATH,
        execute_path=smoke.DEFAULT_EXECUTE_PATH,
        review_path=smoke.DEFAULT_REVIEW_PATH,
        public_path_template=smoke.DEFAULT_PUBLIC_PATH_TEMPLATE,
        allow_indexed_public_artifact=True,
        output_result=None,
        json=False,
    )


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "tickets.csv"
    path.write_text(
        "ticket_id,subject,message\n"
        "ticket-1,Billing renewal,How do I confirm my renewal invoice?\n",
        encoding="utf-8",
    )
    return path


def test_upload_source_public_asset_smoke_request_sequence(tmp_path: Path) -> None:
    transport = _Transport(
        {
            "diagnostics": {"ok": True},
            "import": {
                "dry_run": False,
                "target_ids": ["ticket-1", "ticket-2"],
                "warnings": [],
            },
        },
        {
            "status": "completed",
            "steps": [
                {
                    "output": "landing_page",
                    "status": "completed",
                    "result": {"saved_ids": ["landing-1"]},
                },
                {
                    "output": "blog_post",
                    "status": "completed",
                    "result": {"saved_ids": ["blog-1"]},
                },
            ],
        },
        {"asset": "landing_page", "id": "landing-1", "status": "approved", "updated": True},
        {
            "id": "landing-1",
            "slug": "support-faq-report",
            "robots": "index,follow",
        },
    )

    result = smoke.run_smoke(_args(_csv(tmp_path)), opener=transport)

    assert result == {
        "ok": True,
        "target_ids": ["ticket-1", "ticket-2"],
        "landing_page_id": "landing-1",
        "blog_post_id": "blog-1",
        "public_slug": "support-faq-report",
        "public_robots": "index,follow",
        "public_url_path": "/lp/landing-1/support-faq-report",
    }
    assert [request.get_method() for request in transport.requests] == [
        "POST",
        "POST",
        "POST",
        "GET",
    ]
    assert [request.full_url for request in transport.requests] == [
        "https://atlas-preview.example.com/api/v1/content-ops/ingestion/files/import",
        "https://atlas-preview.example.com/api/v1/content-ops/execute",
        "https://atlas-preview.example.com/api/v1/content-assets/landing_page/drafts/review",
        "https://atlas-preview.example.com/api/v1/content-assets/landing_page/public/landing-1",
    ]

    import_request = transport.requests[0]
    import_body = import_request.data or b""
    assert import_request.headers["Authorization"] == "Bearer secret-token"
    assert b'name="file"; filename="tickets.csv"' in import_body
    assert b'name="source_rows"\r\n\r\ntrue' in import_body
    assert b'name="include_source_material"\r\n\r\nfalse' in import_body
    assert b'name="dry_run"\r\n\r\nfalse' in import_body

    execute_body = json.loads((transport.requests[1].data or b"{}").decode("utf-8"))
    assert execute_body == {
        "target_mode": "vendor_retention",
        "outputs": ["landing_page", "blog_post"],
        "limit": 1,
        "require_quality_gates": True,
        "inputs": {"source_import_target_ids": ["ticket-1", "ticket-2"]},
    }
    review_body = json.loads((transport.requests[2].data or b"{}").decode("utf-8"))
    assert review_body == {"id": "landing-1", "status": "approved"}


def test_upload_source_public_asset_smoke_fails_without_target_ids(tmp_path: Path) -> None:
    transport = _Transport({"diagnostics": {"ok": True}, "import": {"target_ids": []}})

    with pytest.raises(smoke.SmokeFailure) as exc:
        smoke.run_smoke(_args(_csv(tmp_path)), opener=transport)

    assert exc.value.errors == ("import response did not include persisted target_ids",)
    assert len(transport.requests) == 1


def test_upload_source_public_asset_smoke_fails_when_public_route_is_noindex(
    tmp_path: Path,
) -> None:
    transport = _Transport(
        {"import": {"target_ids": ["ticket-1"]}},
        {
            "status": "completed",
            "steps": [
                {"output": "landing_page", "result": {"saved_ids": ["landing-1"]}},
                {"output": "blog_post", "result": {"saved_ids": ["blog-1"]}},
            ],
        },
        {"status": "approved", "updated": True},
        {"id": "landing-1", "slug": "support-faq-report", "robots": "noindex,follow"},
    )

    with pytest.raises(smoke.SmokeFailure) as exc:
        smoke.run_smoke(_args(_csv(tmp_path)), opener=transport)

    assert exc.value.errors == ("public landing response was not indexable",)
    assert exc.value.summary["landing_page_id"] == "landing-1"


@pytest.mark.parametrize(
    ("execute_payload", "expected_errors"),
    (
        (
            {
                "status": "partial",
                "steps": [
                    {"output": "landing_page", "result": {"saved_ids": ["landing-1"]}},
                    {"output": "blog_post", "result": {"saved_ids": ["blog-1"]}},
                ],
            },
            ("execute response status was not completed",),
        ),
        (
            {
                "status": "completed",
                "steps": [
                    {"output": "blog_post", "result": {"saved_ids": ["blog-1"]}},
                ],
            },
            ("execute response did not include a landing_page saved id",),
        ),
        (
            {
                "status": "completed",
                "steps": [
                    {"output": "landing_page", "result": {"saved_ids": ["landing-1"]}},
                ],
            },
            ("execute response did not include a blog_post saved id",),
        ),
    ),
)
def test_upload_source_public_asset_smoke_fails_on_execute_envelope_drift(
    tmp_path: Path,
    execute_payload: dict[str, Any],
    expected_errors: tuple[str, ...],
) -> None:
    transport = _Transport(
        {"import": {"target_ids": ["ticket-1"]}},
        execute_payload,
    )

    with pytest.raises(smoke.SmokeFailure) as exc:
        smoke.run_smoke(_args(_csv(tmp_path)), opener=transport)

    assert exc.value.errors == expected_errors
    assert exc.value.summary["target_ids"] == ["ticket-1"]
    assert len(transport.requests) == 2


def test_upload_source_public_asset_smoke_fails_when_approval_is_not_confirmed(
    tmp_path: Path,
) -> None:
    transport = _Transport(
        {"import": {"target_ids": ["ticket-1"]}},
        {
            "status": "completed",
            "steps": [
                {"output": "landing_page", "result": {"saved_ids": ["landing-1"]}},
                {"output": "blog_post", "result": {"saved_ids": ["blog-1"]}},
            ],
        },
        {"status": "draft", "updated": False},
    )

    with pytest.raises(smoke.SmokeFailure) as exc:
        smoke.run_smoke(_args(_csv(tmp_path)), opener=transport)

    assert exc.value.errors == ("landing_page review response did not confirm approval",)
    assert exc.value.summary["landing_page_id"] == "landing-1"
    assert len(transport.requests) == 3


@pytest.mark.parametrize(
    ("public_payload", "expected_errors"),
    (
        (
            {"id": "other-landing", "slug": "support-faq-report", "robots": "index,follow"},
            ("public landing response id did not match approved draft id",),
        ),
        (
            {"id": "landing-1", "slug": "", "robots": "index,follow"},
            ("public landing response did not include a slug",),
        ),
    ),
)
def test_upload_source_public_asset_smoke_fails_on_public_envelope_drift(
    tmp_path: Path,
    public_payload: dict[str, Any],
    expected_errors: tuple[str, ...],
) -> None:
    transport = _Transport(
        {"import": {"target_ids": ["ticket-1"]}},
        {
            "status": "completed",
            "steps": [
                {"output": "landing_page", "result": {"saved_ids": ["landing-1"]}},
                {"output": "blog_post", "result": {"saved_ids": ["blog-1"]}},
            ],
        },
        {"status": "approved", "updated": True},
        public_payload,
    )

    with pytest.raises(smoke.SmokeFailure) as exc:
        smoke.run_smoke(_args(_csv(tmp_path)), opener=transport)

    assert exc.value.errors == expected_errors
    assert exc.value.summary["landing_page_id"] == "landing-1"
    assert len(transport.requests) == 4


def test_upload_source_public_asset_smoke_rejects_local_base_url(tmp_path: Path) -> None:
    args = _args(_csv(tmp_path))
    args.api_base_url = "http://localhost:8000"

    with pytest.raises(smoke.SmokeFailure) as exc:
        smoke.run_smoke(args, opener=_Transport())

    assert exc.value.errors == ("--api-base-url must be an absolute hosted HTTPS URL",)


def test_upload_source_public_asset_smoke_requires_public_artifact_confirmation(
    tmp_path: Path,
) -> None:
    args = _args(_csv(tmp_path))
    args.allow_indexed_public_artifact = False

    with pytest.raises(smoke.SmokeFailure) as exc:
        smoke.run_smoke(args, opener=_Transport())

    assert exc.value.errors == (
        "--allow-indexed-public-artifact is required because this smoke "
        "approves an indexable public landing page",
    )
