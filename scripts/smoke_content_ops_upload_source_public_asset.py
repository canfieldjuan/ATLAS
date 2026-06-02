#!/usr/bin/env python3
"""Smoke-test deployed upload-source generation through the public asset seam."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from uuid import uuid4


DEFAULT_IMPORT_PATH = "/api/v1/content-ops/ingestion/files/import"
DEFAULT_EXECUTE_PATH = "/api/v1/content-ops/execute"
DEFAULT_REVIEW_PATH = "/api/v1/content-assets/landing_page/drafts/review"
DEFAULT_PUBLIC_PATH_TEMPLATE = "/api/v1/content-assets/landing_page/public/{id}"
LOCAL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})

OpenRequest = Callable[[urllib.request.Request, float], Any]


@dataclass(frozen=True)
class HttpResult:
    status: int
    text: str
    payload: Any


class SmokeFailure(RuntimeError):
    def __init__(self, errors: Sequence[str], summary: Mapping[str, Any] | None = None) -> None:
        self.errors = tuple(errors)
        self.summary = dict(summary or {})
        super().__init__("; ".join(self.errors))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a support-ticket CSV to a deployed Content Ops API, execute "
            "landing/blog generation, approve the landing page, and verify the "
            "public landing-page route."
        )
    )
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--source", default="upload-source-public-asset-smoke.csv")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--import-path", default=DEFAULT_IMPORT_PATH)
    parser.add_argument("--execute-path", default=DEFAULT_EXECUTE_PATH)
    parser.add_argument("--review-path", default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--public-path-template", default=DEFAULT_PUBLIC_PATH_TEMPLATE)
    parser.add_argument(
        "--allow-indexed-public-artifact",
        action="store_true",
        help=(
            "Confirm that this smoke may create an approved, indexable public "
            "landing-page artifact. Use preview deployments when possible; "
            "production runs can leave crawlable smoke content until manually "
            "reviewed or removed."
        ),
    )
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def run_smoke(args: argparse.Namespace, *, opener: OpenRequest | None = None) -> dict[str, Any]:
    errors = _validate_args(args)
    if errors:
        raise SmokeFailure(errors)
    open_request = opener or _open_request
    base_url = _clean(args.api_base_url).rstrip("/")
    token = _clean(args.token)
    timeout = float(args.timeout)

    imported = _post_multipart(
        _join_url(base_url, args.import_path),
        token=token,
        fields={
            "source_rows": "true",
            "source": _clean(args.source),
            "target_mode": _clean(args.target_mode),
            "file_format": "csv",
            "include_source_material": "false",
            "replace_existing": "false",
            "dry_run": "false",
        },
        file_field="file",
        file_path=args.csv,
        timeout=timeout,
        opener=open_request,
    ).payload
    target_ids = _import_target_ids(imported)
    if not target_ids:
        raise SmokeFailure(["import response did not include persisted target_ids"], {"import": imported})

    executed = _post_json(
        _join_url(base_url, args.execute_path),
        token=token,
        payload={
            "target_mode": _clean(args.target_mode),
            "outputs": ["landing_page", "blog_post"],
            "limit": 1,
            "require_quality_gates": True,
            "inputs": {"source_import_target_ids": target_ids},
        },
        timeout=timeout,
        opener=open_request,
    ).payload
    landing_id = _saved_id_for_output(executed, "landing_page")
    blog_id = _saved_id_for_output(executed, "blog_post")
    execute_errors: list[str] = []
    if _mapping(executed).get("status") != "completed":
        execute_errors.append("execute response status was not completed")
    if not landing_id:
        execute_errors.append("execute response did not include a landing_page saved id")
    if not blog_id:
        execute_errors.append("execute response did not include a blog_post saved id")
    if execute_errors:
        raise SmokeFailure(execute_errors, {"execute": executed, "target_ids": target_ids})

    approval = _post_json(
        _join_url(base_url, args.review_path),
        token=token,
        payload={"id": landing_id, "status": "approved"},
        timeout=timeout,
        opener=open_request,
    ).payload
    approval_map = _mapping(approval)
    if approval_map.get("updated") is not True or approval_map.get("status") != "approved":
        raise SmokeFailure(
            ["landing_page review response did not confirm approval"],
            {"approval": approval, "landing_page_id": landing_id},
        )

    public_path = str(args.public_path_template).format(id=urllib.parse.quote(landing_id))
    public = _get_json(
        _join_url(base_url, public_path),
        timeout=timeout,
        opener=open_request,
    ).payload
    public_map = _mapping(public)
    public_errors: list[str] = []
    if public_map.get("id") != landing_id:
        public_errors.append("public landing response id did not match approved draft id")
    if not _clean(public_map.get("slug")):
        public_errors.append("public landing response did not include a slug")
    if public_map.get("robots") != "index,follow":
        public_errors.append("public landing response was not indexable")
    if public_errors:
        raise SmokeFailure(public_errors, {"public": public, "landing_page_id": landing_id})

    slug = _clean(public_map.get("slug"))
    return {
        "ok": True,
        "target_ids": target_ids,
        "landing_page_id": landing_id,
        "blog_post_id": blog_id,
        "public_slug": slug,
        "public_robots": public_map.get("robots"),
        "public_url_path": f"/lp/{urllib.parse.quote(landing_id)}/{urllib.parse.quote(slug)}",
    }


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not _hosted_https_url(_clean(args.api_base_url)):
        errors.append("--api-base-url must be an absolute hosted HTTPS URL")
    if not _clean(args.token):
        errors.append("--token is required")
    csv_path = Path(args.csv)
    if not csv_path.exists() or not csv_path.is_file():
        errors.append("--csv must point to an existing CSV file")
    if float(args.timeout) <= 0:
        errors.append("--timeout must be positive")
    if not bool(getattr(args, "allow_indexed_public_artifact", False)):
        errors.append(
            "--allow-indexed-public-artifact is required because this smoke "
            "approves an indexable public landing page"
        )
    for attr in ("import_path", "execute_path", "review_path", "public_path_template"):
        value = _clean(getattr(args, attr))
        if not value.startswith("/"):
            errors.append(f"--{attr.replace('_', '-')} must start with /")
    if "{id}" not in _clean(args.public_path_template):
        errors.append("--public-path-template must include {id}")
    return errors


def _hosted_https_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and bool(host) and host not in LOCAL_HOSTS and not host.startswith("127.")


def _post_json(
    url: str,
    *,
    token: str,
    payload: Mapping[str, Any],
    timeout: float,
    opener: OpenRequest,
) -> HttpResult:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    return _read_json(request, timeout=timeout, opener=opener)


def _get_json(url: str, *, timeout: float, opener: OpenRequest) -> HttpResult:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    return _read_json(request, timeout=timeout, opener=opener)


def _post_multipart(
    url: str,
    *,
    token: str,
    fields: Mapping[str, str],
    file_field: str,
    file_path: Path,
    timeout: float,
    opener: OpenRequest,
) -> HttpResult:
    boundary = f"atlas-smoke-{uuid4().hex}"
    body = _multipart_body(boundary=boundary, fields=fields, file_field=file_field, file_path=file_path)
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    return _read_json(request, timeout=timeout, opener=opener)


def _multipart_body(
    *,
    boundary: str,
    fields: Mapping[str, str],
    file_field: str,
    file_path: Path,
) -> bytes:
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    chunks.extend([
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\n'
        ).encode(),
        b"Content-Type: text/csv\r\n\r\n",
        file_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(chunks)


def _read_json(request: urllib.request.Request, *, timeout: float, opener: OpenRequest) -> HttpResult:
    try:
        with opener(request, timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            payload = json.loads(text) if text else None
            return HttpResult(status=int(response.getcode()), text=text, payload=payload)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise SmokeFailure([f"{request.full_url} returned HTTP {exc.code}"], {"body": text}) from None
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SmokeFailure([f"{request.full_url} failed: {exc}"]) from None


def _open_request(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _import_target_ids(payload: Any) -> list[str]:
    target_ids = _mapping(_mapping(payload).get("import")).get("target_ids")
    if not isinstance(target_ids, Sequence) or isinstance(target_ids, (str, bytes)):
        return []
    return [_clean(value) for value in target_ids if _clean(value)]


def _saved_id_for_output(payload: Any, output: str) -> str:
    for step in _mapping(payload).get("steps") or ():
        step_map = _mapping(step)
        if step_map.get("output") != output:
            continue
        saved_ids = _mapping(step_map.get("result")).get("saved_ids")
        if isinstance(saved_ids, Sequence) and not isinstance(saved_ids, (str, bytes)):
            for value in saved_ids:
                text = _clean(value)
                if text:
                    return text
    return ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{str(path).lstrip('/')}"


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_smoke(args)
    except SmokeFailure as exc:
        result = {"ok": False, "errors": list(exc.errors), **exc.summary}
        if args.output_result:
            args.output_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else "\n".join(exc.errors))
        return 1
    if args.output_result:
        args.output_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else f"OK {result['public_url_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
