#!/usr/bin/env python3
"""Validate the hosted FAQ deflection portfolio result page handoff."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})
SNAPSHOT_PATH_TEMPLATE = "/api/v1/content-ops/deflection-reports/{request_id}/snapshot"
ARTIFACT_PATH_TEMPLATE = "/api/v1/content-ops/deflection-reports/{request_id}/artifact"
FORBIDDEN_SNAPSHOT_KEYS = frozenset({
    "answer",
    "answers",
    "evidence",
    "faq_result",
    "full_report",
    "markdown",
    "source_id",
    "source_ids",
    "steps",
})
REQUIRED_PAGE_MARKERS = (
    "data-atlas-deflection-result",
    "data-atlas-deflection-request-id",
    "data-atlas-deflection-unlock",
    "content_ops_deflection_report",
    "request_id",
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency
    load_dotenv = None


@dataclass(frozen=True)
class HttpResult:
    status: int | None
    text: str
    payload: Any = None
    errors: tuple[str, ...] = ()


class UnlockCtaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict[str, str] | None = None

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        if "data-atlas-deflection-unlock" in attr_map and self.attrs is None:
            self.attrs = attr_map


def _load_dotenv_files() -> None:
    if os.getenv("ATLAS_DISABLE_DOTENV") == "1":
        return
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


def _env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _build_parser() -> argparse.ArgumentParser:
    _load_dotenv_files()
    parser = argparse.ArgumentParser(
        description="Validate the hosted deflection portfolio result page."
    )
    parser.add_argument("--result-url", default=_env("ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL"))
    parser.add_argument("--base-url", default=_env("ATLAS_API_BASE_URL"))
    parser.add_argument("--token", default=_env("ATLAS_B2B_JWT", "ATLAS_TOKEN"))
    parser.add_argument("--account-id", default=_env("ATLAS_ACCOUNT_ID", "ATLAS_FAQ_SEARCH_ACCOUNT_ID"))
    parser.add_argument("--request-id", default=_env("ATLAS_DEFLECTION_REQUEST_ID"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--snapshot-path-template", default=SNAPSHOT_PATH_TEMPLATE)
    parser.add_argument("--artifact-path-template", default=ARTIFACT_PATH_TEMPLATE)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _hosted_url_errors(value: str, *, label: str) -> list[str]:
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return [f"{label} must be an absolute HTTPS URL"]
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return [f"{label} must be an absolute HTTPS URL"]
    host = (parsed.hostname or "").lower()
    if not host:
        return [f"{label} must include a host"]
    if host in LOCAL_HOSTS or host.startswith("127."):
        return [f"{label} must point to a hosted URL; local hosts are not accepted"]
    return []


def _result_url_account_errors(value: str, *, account_id: str) -> list[str]:
    errors: list[str] = []
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return []
    public_location = f"{parsed.path}?{parsed.query}"
    if "account_id" in public_location:
        errors.append("--result-url must not include account_id")
    if account_id and account_id in public_location:
        errors.append("--result-url must not include the account_id value")
    return errors


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not _clean(args.result_url):
        errors.append("ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL or --result-url is required")
    else:
        errors.extend(_hosted_url_errors(_clean(args.result_url), label="--result-url"))
        errors.extend(
            _result_url_account_errors(
                _clean(args.result_url),
                account_id=_clean(args.account_id),
            )
        )
    if not _clean(args.base_url):
        errors.append("ATLAS_API_BASE_URL or --base-url is required")
    else:
        errors.extend(_hosted_url_errors(_clean(args.base_url), label="--base-url"))
    if not _clean(args.token):
        errors.append("ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required")
    if not _clean(args.account_id):
        errors.append("ATLAS_ACCOUNT_ID, ATLAS_FAQ_SEARCH_ACCOUNT_ID, or --account-id is required")
    if not _clean(args.request_id):
        errors.append("ATLAS_DEFLECTION_REQUEST_ID or --request-id is required")
    if not math.isfinite(float(args.timeout)) or float(args.timeout) <= 0:
        errors.append("--timeout must be a positive finite number")
    for attr, label in (
        ("snapshot_path_template", "--snapshot-path-template"),
        ("artifact_path_template", "--artifact-path-template"),
    ):
        value = _clean(getattr(args, attr))
        if not value.startswith("/"):
            errors.append(f"{label} must start with /")
        if "{request_id}" not in value:
            errors.append(f"{label} must include {{request_id}}")
    return errors


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _open_http_request(request: urllib.request.Request, *, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    if not exc.fp:
        return ""
    return exc.read().decode("utf-8", errors="replace")


def _fetch_text(url: str, *, timeout: float) -> HttpResult:
    request = urllib.request.Request(url, headers={"Accept": "text/html"})
    try:
        with _open_http_request(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return HttpResult(status=int(response.getcode()), text=text)
    except urllib.error.HTTPError as exc:
        body = _read_http_error(exc)
        return HttpResult(status=int(exc.code), text=body, errors=(f"HTTP {exc.code}",))
    except (OSError, urllib.error.URLError) as exc:
        return HttpResult(status=None, text="", errors=(str(exc),))


def _fetch_json(url: str, *, token: str, timeout: float) -> HttpResult:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with _open_http_request(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return HttpResult(
                status=int(response.getcode()),
                text=text,
                payload=json.loads(text) if text else None,
            )
    except urllib.error.HTTPError as exc:
        text = _read_http_error(exc)
        payload = None
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
        return HttpResult(status=int(exc.code), text=text, payload=payload, errors=(f"HTTP {exc.code}",))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return HttpResult(status=None, text="", errors=(str(exc),))


def _snapshot_errors(snapshot: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(snapshot, Mapping):
        return ["snapshot response must be an object"]
    if not isinstance(snapshot.get("summary"), Mapping):
        errors.append("snapshot.summary must be an object")
    top_questions = snapshot.get("top_questions")
    if not isinstance(top_questions, list) or not top_questions:
        errors.append("snapshot.top_questions must be a non-empty list")
    leaked = sorted(set(_forbidden_snapshot_keys(snapshot)))
    if leaked:
        errors.append(f"snapshot leaked paid-report fields: {', '.join(leaked)}")
    return errors


def _forbidden_snapshot_keys(value: Any) -> list[str]:
    leaked: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_SNAPSHOT_KEYS:
                leaked.append(key_text)
            leaked.extend(_forbidden_snapshot_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            leaked.extend(_forbidden_snapshot_keys(child))
    return leaked


def _page_errors(html: str, *, request_id: str, account_id: str) -> list[str]:
    errors: list[str] = []
    if not html:
        return ["portfolio result page returned empty HTML"]
    for marker in REQUIRED_PAGE_MARKERS:
        if marker not in html:
            errors.append(f"portfolio result page missing marker: {marker}")
    unlock_attrs = _unlock_cta_attrs(html)
    if unlock_attrs is None:
        errors.append("portfolio result page missing unlock CTA element")
    else:
        for attr, expected in (
            ("data-checkout-source", "content_ops_deflection_report"),
            ("data-checkout-request_id", request_id),
        ):
            actual = unlock_attrs.get(attr)
            if actual != expected:
                errors.append(f"portfolio result page unlock CTA {attr} must be {expected}")
        if "data-checkout-account_id" in unlock_attrs:
            errors.append("portfolio result page unlock CTA must not expose account_id")
    if request_id not in html:
        errors.append("portfolio result page missing request_id value")
    if "account_id" in html:
        errors.append("portfolio result page must not expose account_id marker")
    if account_id and account_id in html:
        errors.append("portfolio result page must not expose account_id value")
    return errors


def _unlock_cta_attrs(html: str) -> dict[str, str] | None:
    parser = UnlockCtaParser()
    parser.feed(html)
    return parser.attrs


def _result_payload(
    *,
    args: argparse.Namespace,
    preflight_errors: Sequence[str],
    page: HttpResult | None = None,
    snapshot: HttpResult | None = None,
    artifact: HttpResult | None = None,
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "ok": not preflight_errors and not errors,
        "preflight_errors": list(preflight_errors),
        "errors": list(errors),
        "inputs": {
            "result_url": _clean(args.result_url),
            "base_url": _clean(args.base_url),
            "request_id": _clean(args.request_id),
            "account_id": _clean(args.account_id),
            "token_present": bool(_clean(args.token)),
        },
        "page": {"status": page.status if page else None},
        "snapshot": {"status": snapshot.status if snapshot else None},
        "artifact": {"status": artifact.status if artifact else None},
    }


def _write_result(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    preflight_errors = _validate_args(args)
    if preflight_errors or args.preflight_only:
        payload = _result_payload(args=args, preflight_errors=preflight_errors)
        _write_result(args.output_result, payload)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print("preflight failed" if preflight_errors else "preflight passed")
        return 2 if preflight_errors else 0

    request_id = _clean(args.request_id)
    page = _fetch_text(_clean(args.result_url), timeout=float(args.timeout))
    snapshot_url = _join_url(
        _clean(args.base_url),
        _clean(args.snapshot_path_template).format(request_id=urllib.parse.quote(request_id, safe="")),
    )
    artifact_url = _join_url(
        _clean(args.base_url),
        _clean(args.artifact_path_template).format(request_id=urllib.parse.quote(request_id, safe="")),
    )
    snapshot = _fetch_json(snapshot_url, token=_clean(args.token), timeout=float(args.timeout))
    artifact = _fetch_json(artifact_url, token=_clean(args.token), timeout=float(args.timeout))

    errors: list[str] = []
    if page.status != 200:
        errors.append(f"portfolio result page returned status {page.status}")
    errors.extend(_page_errors(page.text, request_id=request_id, account_id=_clean(args.account_id)))
    if snapshot.status != 200:
        errors.append(f"snapshot endpoint returned status {snapshot.status}")
    else:
        errors.extend(_snapshot_errors(snapshot.payload))
    if artifact.status != 403:
        errors.append(f"artifact endpoint must return 403 before payment; got {artifact.status}")

    payload = _result_payload(
        args=args,
        preflight_errors=(),
        page=page,
        snapshot=snapshot,
        artifact=artifact,
        errors=errors,
    )
    _write_result(args.output_result, payload)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("portfolio result page smoke passed" if payload["ok"] else "portfolio result page smoke failed")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
