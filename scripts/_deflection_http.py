#!/usr/bin/env python3
"""Shared HTTP helpers for deflection operator scripts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
import urllib.error
import urllib.request


Redactor = Callable[[Any], str]
TextRedactor = Callable[[str], str]
JsonValueRedactor = Callable[[tuple[str, ...], Any], Any | None]
Opener = Callable[[urllib.request.Request], Any]
_KEEP_STATUS = object()


def _identity_redactor(value: Any) -> str:
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class HttpResponse:
    status: int | None
    payload: Any = None
    text: str = ""
    raw_text: str = ""
    errors: tuple[str, ...] = ()


def open_http_request(request: urllib.request.Request, *, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def read_http_error(exc: urllib.error.HTTPError) -> str:
    if not exc.fp:
        return ""
    return exc.read().decode("utf-8", errors="replace")


def bearer_header(token: str) -> str:
    value = "" if token is None else str(token).strip()
    if value.lower().startswith("bearer "):
        return value
    return f"Bearer {value}"


def _encoded_body(
    body: Mapping[str, Any] | None,
    data: bytes | None,
) -> bytes | None:
    if body is not None and data is not None:
        raise ValueError("pass either body or data, not both")
    if body is None:
        return data
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def _redact_json_string_values(
    value: Any,
    redactor: TextRedactor,
    json_value_redactor: JsonValueRedactor | None = None,
    path: tuple[str, ...] = (),
) -> Any:
    if json_value_redactor is not None:
        replacement = json_value_redactor(path, value)
        if replacement is not None:
            return replacement
    if isinstance(value, str):
        return redactor(value)
    if isinstance(value, list):
        return [
            _redact_json_string_values(item, redactor, json_value_redactor, path)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _redact_json_string_values(
                item,
                redactor,
                json_value_redactor,
                path + (str(key),),
            )
            for key, item in value.items()
        }
    return value


def _redact_error_body(
    raw: str,
    redactor: TextRedactor,
    json_value_redactor: JsonValueRedactor | None = None,
) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return redactor(raw)
    redacted = _redact_json_string_values(payload, redactor, json_value_redactor)
    return json.dumps(redacted, separators=(",", ":"))


def json_request(
    method: str,
    url: str,
    *,
    timeout: float,
    token: str = "",
    authorization: str = "",
    body: Mapping[str, Any] | None = None,
    data: bytes | None = None,
    stripe_signature: str = "",
    redactor: Redactor = _identity_redactor,
    opener: Callable[..., Any] = open_http_request,
    http_error_template: str | None = None,
    transport_error_template: str = "{method} {url} transport failed: {error}",
    invalid_json_template: str = "{method} {url} returned invalid JSON: {error}",
    invalid_json_status: int | None | object = _KEEP_STATUS,
    error_body_redactor: TextRedactor | None = None,
    error_body_json_value_redactor: JsonValueRedactor | None = None,
    truncate_text: int | None = None,
) -> HttpResponse:
    encoded_body = _encoded_body(body, data)
    headers = {"Accept": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    elif token:
        headers["Authorization"] = bearer_header(token)
    if body is not None or data is not None:
        headers["Content-Type"] = "application/json"
    if stripe_signature:
        headers["Stripe-Signature"] = stripe_signature
    request = urllib.request.Request(url, data=encoded_body, headers=headers, method=method)
    return json_response_from_request(
        method,
        url,
        request=request,
        timeout=timeout,
        redactor=redactor,
        opener=opener,
        http_error_template=http_error_template,
        transport_error_template=transport_error_template,
        invalid_json_template=invalid_json_template,
        invalid_json_status=invalid_json_status,
        error_body_redactor=error_body_redactor,
        error_body_json_value_redactor=error_body_json_value_redactor,
        truncate_text=truncate_text,
    )


def json_response_from_request(
    method: str,
    url: str,
    *,
    request: urllib.request.Request,
    timeout: float,
    redactor: Redactor = _identity_redactor,
    opener: Callable[..., Any] = open_http_request,
    http_error_template: str | None = None,
    transport_error_template: str = "{method} {url} transport failed: {error}",
    invalid_json_template: str = "{method} {url} returned invalid JSON: {error}",
    invalid_json_status: int | None | object = _KEEP_STATUS,
    error_body_redactor: TextRedactor | None = None,
    error_body_json_value_redactor: JsonValueRedactor | None = None,
    truncate_text: int | None = None,
) -> HttpResponse:
    http_errors: tuple[str, ...] = ()
    try:
        with opener(request, timeout=timeout) as response:
            status = int(getattr(response, "status", None) or response.getcode())
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = read_http_error(exc)
        if http_error_template is not None:
            http_errors = (http_error_template.format(status=status),)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        detail = redactor(exc)
        return HttpResponse(
            status=None,
            text="",
            raw_text="",
            errors=(transport_error_template.format(method=method, url=url, error=detail),),
        )

    if error_body_redactor is not None and status >= 400:
        raw = _redact_error_body(raw, error_body_redactor, error_body_json_value_redactor)
    stored_raw = raw if truncate_text is None else raw[:truncate_text]
    if not raw.strip():
        return HttpResponse(
            status=status,
            text=stored_raw,
            raw_text=stored_raw,
            errors=http_errors,
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if http_errors:
            return HttpResponse(
                status=status,
                text=stored_raw,
                raw_text=stored_raw,
                errors=http_errors,
            )
        returned_status = status if invalid_json_status is _KEEP_STATUS else invalid_json_status
        return HttpResponse(
            status=returned_status,
            text=stored_raw,
            raw_text=stored_raw,
            errors=(invalid_json_template.format(method=method, url=url, error=exc.msg),),
        )
    return HttpResponse(
        status=status,
        payload=payload,
        text=stored_raw,
        raw_text=stored_raw,
        errors=http_errors,
    )
