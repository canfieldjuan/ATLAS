#!/usr/bin/env python3
"""Check Claude hosted OAuth root-route compatibility for Content Ops MCP."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping


DEFAULT_SCOPE = "content_ops.review.verify"
DEFAULT_REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"
ISSUER_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"
RESOURCE_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL"


def _strip(url: str) -> str:
    return str(url).strip().rstrip("/")


def _approval_path(issuer_url: str) -> str:
    path = urllib.parse.urlparse(_strip(issuer_url)).path.rstrip("/")
    return f"{path}/oauth/approve" if path else "/oauth/approve"


def _authorization_url(
    *,
    root_url: str,
    resource_url: str,
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
    code_challenge: str,
) -> str:
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "scope": scope,
            "resource": resource_url,
        }
    )
    return f"{root_url}/authorize?{query}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_no_redirect(url: str, timeout: float) -> tuple[int, Mapping[str, str]]:
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, headers={"Accept": "text/html"}, method="GET")
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} request failed: {exc.reason}") from exc


def _redirect_errors(*, issuer_url: str, status: int, headers: Mapping[str, str]) -> list[str]:
    if status not in {302, 303}:
        return [f"root authorize returned HTTP {status}, expected 302 or 303"]
    location = headers.get("location") or headers.get("Location") or ""
    if not location:
        return ["root authorize redirect missing Location header"]
    parsed = urllib.parse.urlparse(location)
    issuer = urllib.parse.urlparse(_strip(issuer_url))
    errors: list[str] = []
    if (parsed.scheme, parsed.netloc) != (issuer.scheme, issuer.netloc):
        errors.append("root authorize redirect host does not match issuer host")
    expected_path = _approval_path(issuer_url)
    if parsed.path != expected_path:
        errors.append(f"root authorize redirect path != {expected_path}")
    if not urllib.parse.parse_qs(parsed.query).get("request_id", [""])[0]:
        errors.append("root authorize redirect missing request_id")
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Claude hosted OAuth root authorize routing.")
    parser.add_argument("--issuer-url", default=os.environ.get(ISSUER_ENV, ""))
    parser.add_argument("--resource-url", default=os.environ.get(RESOURCE_ENV, ""))
    parser.add_argument("--public-root-url", default="")
    parser.add_argument("--client-id", default="")
    parser.add_argument("--redirect-uri", default=DEFAULT_REDIRECT_URI)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    issuer_url = _strip(args.issuer_url)
    resource_url = _strip(args.resource_url)
    root_source = _strip(args.public_root_url) or issuer_url
    parsed_root = urllib.parse.urlparse(root_source)
    root_url = urllib.parse.urlunparse((parsed_root.scheme, parsed_root.netloc, "", "", "", ""))
    client_id = args.client_id.strip()
    values = (
        (f"--issuer-url or {ISSUER_ENV}", issuer_url),
        (f"--resource-url or {RESOURCE_ENV}", resource_url),
        ("--public-root-url or a valid issuer host", root_url),
        ("--client-id", client_id),
        ("--redirect-uri", args.redirect_uri.strip()),
        ("--scope", args.scope.strip()),
    )
    missing = [label for label, value in values if not value]
    if missing:
        print("missing required values: " + ", ".join(missing), file=sys.stderr)
        return 2

    url = _authorization_url(
        root_url=root_url,
        resource_url=resource_url,
        client_id=client_id,
        redirect_uri=args.redirect_uri.strip(),
        scope=args.scope.strip(),
        state="content-ops-claude-hosted-smoke",
        code_challenge="content-ops-claude-hosted-smoke-challenge",
    )
    try:
        status, headers = _open_no_redirect(url, args.timeout)
    except Exception as exc:
        print(f"Claude hosted OAuth check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    errors = _redirect_errors(issuer_url=issuer_url, status=status, headers=headers)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("OK: Claude hosted OAuth root authorize route redirects to Content Ops approval")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
