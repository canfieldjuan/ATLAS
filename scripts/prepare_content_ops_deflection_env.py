#!/usr/bin/env python3
"""Prepare local FAQ deflection hosted-handoff env values."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_BASE_URL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})
B2B_PRODUCTS = frozenset({"b2b_retention", "b2b_challenger"})
B2B_PLAN_ORDER = ("b2b_trial", "b2b_starter", "b2b_growth", "b2b_pro")
MIN_B2B_PLAN = "b2b_growth"
BAD_PLAN_STATUSES = frozenset({"past_due", "canceled"})
ENV_KEYS = ("ATLAS_API_BASE_URL", "ATLAS_B2B_JWT", "ATLAS_ACCOUNT_ID")


@dataclass(frozen=True)
class HttpJsonResponse:
    status: int | None
    payload: Any
    raw_text: str
    errors: tuple[str, ...] = ()


class HandoffError(Exception):
    """Raised for operator-fixable handoff preparation failures."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Log into deployed ATLAS and write local FAQ deflection env values."
    )
    parser.add_argument("--base-url", default=os.getenv("ATLAS_API_BASE_URL", ""))
    parser.add_argument("--email", default=os.getenv("ATLAS_LOGIN_EMAIL", ""))
    parser.add_argument("--password", default="")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    return parser


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _base_url_errors(base_url: str) -> list[str]:
    try:
        parsed = urllib.parse.urlparse(base_url)
    except ValueError:
        return ["--base-url must be an absolute HTTPS URL for hosted proof"]
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return ["--base-url must be an absolute HTTPS URL for hosted proof"]
    host = (parsed.hostname or "").lower()
    if not host:
        return ["--base-url must include a host for hosted proof"]
    if host in LOCAL_BASE_URL_HOSTS or host.startswith("127."):
        return ["--base-url must point to a deployed host; local hosts are not accepted"]
    if parsed.username or parsed.password:
        return ["--base-url must not include credentials"]
    return []


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not _clean(args.base_url):
        errors.append("ATLAS_API_BASE_URL or --base-url is required")
    else:
        errors.extend(_base_url_errors(_clean(args.base_url)))
    if not _clean(args.email):
        errors.append("ATLAS_LOGIN_EMAIL or --email is required")
    if float(args.timeout) <= 0:
        errors.append("--timeout must be positive")
    return errors


def _resolve_password(args: argparse.Namespace) -> str:
    password = _clean(args.password) or _clean(os.getenv("ATLAS_LOGIN_PASSWORD"))
    if password:
        return password
    if sys.stdin.isatty():
        return getpass.getpass("ATLAS login password: ")
    return ""


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _open_http_request(request: urllib.request.Request, *, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _json_request(
    method: str,
    url: str,
    *,
    timeout: float,
    body: Mapping[str, Any] | None = None,
    token: str = "",
) -> HttpJsonResponse:
    encoded_body = None
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        encoded_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=encoded_body, headers=headers, method=method)
    try:
        with _open_http_request(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        status = exc.code
    except urllib.error.URLError as exc:
        return HttpJsonResponse(
            status=None,
            payload=None,
            raw_text="",
            errors=(f"network error calling {url}: {exc.reason}",),
        )

    if not raw:
        return HttpJsonResponse(status=status, payload=None, raw_text=raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return HttpJsonResponse(
            status=status,
            payload=None,
            raw_text=raw,
            errors=(f"response from {url} was not valid JSON",),
        )
    return HttpJsonResponse(status=status, payload=payload, raw_text=raw)


def _login(base_url: str, email: str, password: str, *, timeout: float) -> tuple[str, list[str]]:
    response = _json_request(
        "POST",
        _join_url(base_url, "/api/v1/auth/login"),
        timeout=timeout,
        body={"email": email, "password": password},
    )
    errors = list(response.errors)
    if response.status != 200:
        errors.append(f"login status must be 200, got {response.status}")
    if not isinstance(response.payload, Mapping):
        errors.append("login response must be an object")
        return "", errors
    access_token = _clean(response.payload.get("access_token"))
    token_type = _clean(response.payload.get("token_type") or "bearer").lower()
    if not access_token:
        errors.append("login response must include access_token")
    if token_type != "bearer":
        errors.append("login response token_type must be bearer")
    return access_token, errors


def _plan_rank(plan: str) -> int:
    try:
        return B2B_PLAN_ORDER.index(plan)
    except ValueError:
        return -1


def _validate_me_payload(payload: Any) -> tuple[dict[str, str], list[str]]:
    if not isinstance(payload, Mapping):
        return {}, ["auth/me response must be an object"]

    account = {
        "account_id": _clean(payload.get("account_id")),
        "account_name": _clean(payload.get("account_name")),
        "product": _clean(payload.get("product")),
        "plan": _clean(payload.get("plan")),
        "plan_status": _clean(payload.get("plan_status")),
        "email": _clean(payload.get("email")),
    }
    errors: list[str] = []
    if not account["account_id"]:
        errors.append("auth/me response must include account_id")
    if account["product"] not in B2B_PRODUCTS:
        errors.append("auth/me product must be b2b_retention or b2b_challenger")
    min_rank = _plan_rank(MIN_B2B_PLAN)
    if _plan_rank(account["plan"]) < min_rank:
        errors.append("auth/me plan must be b2b_growth or higher")
    if account["plan_status"] in BAD_PLAN_STATUSES:
        errors.append("auth/me plan_status must not be past_due or canceled")
    if not account["plan_status"]:
        errors.append("auth/me response must include plan_status")
    return account, errors


def _fetch_me(base_url: str, token: str, *, timeout: float) -> tuple[dict[str, str], list[str]]:
    response = _json_request(
        "GET",
        _join_url(base_url, "/api/v1/auth/me"),
        timeout=timeout,
        token=token,
    )
    errors = list(response.errors)
    if response.status != 200:
        errors.append(f"auth/me status must be 200, got {response.status}")
    account, account_errors = _validate_me_payload(response.payload)
    errors.extend(account_errors)
    return account, errors


def _validate_env_value(key: str, value: str) -> None:
    if not value:
        raise HandoffError(f"{key} cannot be empty")
    if "\n" in value or "\r" in value:
        raise HandoffError(f"{key} cannot contain newlines")


def _format_env_line(key: str, value: str) -> str:
    _validate_env_value(key, value)
    return f"{key}={value}"


def _env_key_from_line(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return ""
    key, _value = stripped.split("=", 1)
    return key.strip()


def _merge_env_text(existing: str, updates: Mapping[str, str], *, force: bool) -> str:
    for key, value in updates.items():
        _validate_env_value(key, value)

    lines = existing.splitlines()
    present = {_env_key_from_line(line) for line in lines}
    collisions = sorted(key for key in updates if key in present)
    if collisions and not force:
        joined = ", ".join(collisions)
        raise HandoffError(f"{joined} already exist in env file; rerun with --force to replace")

    replaced: set[str] = set()
    output: list[str] = []
    for line in lines:
        key = _env_key_from_line(line)
        if key in updates:
            output.append(_format_env_line(key, updates[key]))
            replaced.add(key)
        else:
            output.append(line)

    if output and output[-1] != "":
        output.append("")
    for key in ENV_KEYS:
        if key not in replaced:
            output.append(_format_env_line(key, updates[key]))
    return "\n".join(output).rstrip() + "\n"


def _write_env_file(path: Path, updates: Mapping[str, str], *, force: bool) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    merged = _merge_env_text(existing, updates, force=force)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(merged)
        temp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_path, path)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = _clean(args.base_url)
    email = _clean(args.email).lower()
    password = _resolve_password(args)
    errors = _validate_args(args)
    if not password:
        errors.append("ATLAS_LOGIN_PASSWORD or --password is required")
    if errors:
        return {"ok": False, "stage": "preflight", "errors": errors}

    token, login_errors = _login(base_url, email, password, timeout=float(args.timeout))
    if login_errors:
        return {"ok": False, "stage": "login", "errors": login_errors}

    account, me_errors = _fetch_me(base_url, token, timeout=float(args.timeout))
    if me_errors:
        return {"ok": False, "stage": "auth_me", "errors": me_errors}

    updates = {
        "ATLAS_API_BASE_URL": base_url.rstrip("/"),
        "ATLAS_B2B_JWT": token,
        "ATLAS_ACCOUNT_ID": account["account_id"],
    }
    try:
        _write_env_file(args.env_file, updates, force=bool(args.force))
    except HandoffError as exc:
        return {"ok": False, "stage": "env_write", "errors": [str(exc)]}

    return {
        "ok": True,
        "stage": "complete",
        "env_file": str(args.env_file),
        "base_host": urllib.parse.urlparse(base_url).hostname or "",
        "account_id": account["account_id"],
        "account_name": account["account_name"],
        "product": account["product"],
        "plan": account["plan"],
        "plan_status": account["plan_status"],
        "token_written": True,
    }


def _print_human(summary: Mapping[str, Any]) -> None:
    if summary.get("ok"):
        print("Prepared FAQ deflection hosted env.")
        print(f"Env file: {summary['env_file']}")
        print(f"API host: {summary['base_host']}")
        print(f"Account: {summary['account_id']} ({summary['product']} / {summary['plan']})")
        print("Token: written (redacted)")
        return
    print(f"FAQ deflection env handoff failed at {summary.get('stage', 'unknown')}.", file=sys.stderr)
    for error in summary.get("errors", []):
        print(f"- {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human(summary)
    return 0 if summary.get("ok") else (2 if summary.get("stage") in {"preflight", "env_write"} else 1)


if __name__ == "__main__":
    raise SystemExit(main())
