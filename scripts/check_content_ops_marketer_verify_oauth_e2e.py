#!/usr/bin/env python3
"""Run a no-mutation OAuth e2e smoke for the Content Ops marketer verify MCP connector."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_draft_writer_e2e():
    path = ROOT / "scripts/check_invoicing_draft_writer_oauth_e2e.py"
    spec = importlib.util.spec_from_file_location(
        "check_invoicing_draft_writer_oauth_e2e",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load OAuth e2e helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_draft_e2e = _load_draft_writer_e2e()
AuthorizationRequest = _draft_e2e.AuthorizationRequest
OAuthE2EConfig = _draft_e2e.OAuthE2EConfig
RegisteredClient = _draft_e2e.RegisteredClient
TokenResult = _draft_e2e.TokenResult
_approve_authorization = _draft_e2e._approve_authorization
_exchange_token = _draft_e2e._exchange_token
_list_mcp_tools = _draft_e2e._list_mcp_tools
_post_json = _draft_e2e._post_json
_read_secret_file = _draft_e2e._read_secret_file
_start_authorization = _draft_e2e._start_authorization
_strip_trailing_slash = _draft_e2e._strip_trailing_slash


DEFAULT_SCOPE = "content_ops.review.verify"
DEFAULT_REDIRECT_URI = _draft_e2e.DEFAULT_REDIRECT_URI
EXPECTED_TOOLS = {"verify_draft"}
DENIED_TOOLS = {
    "add_registry_claim",
    "define_experiment",
    "generate_asset",
    "publish_asset",
    "run_gates",
    "start_brief",
    "update_registry_claim",
    "verify_and_publish",
}
ISSUER_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"
RESOURCE_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL"
APPROVAL_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a public OAuth e2e smoke for the Content Ops marketer verify "
            "MCP connector. This lists tools only and does not submit draft content."
        )
    )
    parser.add_argument(
        "--issuer-url",
        default=os.environ.get(ISSUER_ENV, ""),
        help=f"Public OAuth issuer URL. Defaults to {ISSUER_ENV}.",
    )
    parser.add_argument(
        "--resource-url",
        default=os.environ.get(RESOURCE_ENV, ""),
        help=f"Public Streamable HTTP MCP resource URL. Defaults to {RESOURCE_ENV}.",
    )
    parser.add_argument(
        "--approval-token",
        default=os.environ.get(APPROVAL_ENV, ""),
        help=f"Operator approval token. Defaults to {APPROVAL_ENV}.",
    )
    parser.add_argument(
        "--approval-token-file",
        default="",
        help=(
            "Read the operator approval token from this local file. Prefer this "
            "over passing approval secrets directly on the command line."
        ),
    )
    parser.add_argument(
        "--redirect-uri",
        default=DEFAULT_REDIRECT_URI,
        help=f"OAuth redirect URI to register. Default: {DEFAULT_REDIRECT_URI}.",
    )
    parser.add_argument(
        "--scope",
        default=DEFAULT_SCOPE,
        help=f"Required OAuth scope. Default: {DEFAULT_SCOPE}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds. Default: 10.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> Any:
    issuer_url = _strip_trailing_slash(args.issuer_url)
    resource_url = _strip_trailing_slash(args.resource_url)
    approval_token = (
        _read_secret_file(args.approval_token_file)
        if args.approval_token_file
        else args.approval_token.strip()
    )
    redirect_uri = args.redirect_uri.strip()
    scope = args.scope.strip()
    missing: list[str] = []
    if not issuer_url:
        missing.append(f"--issuer-url or {ISSUER_ENV}")
    if not resource_url:
        missing.append(f"--resource-url or {RESOURCE_ENV}")
    if not approval_token:
        missing.append(f"--approval-token-file, --approval-token, or {APPROVAL_ENV}")
    if not redirect_uri:
        missing.append("--redirect-uri")
    if not scope:
        missing.append("--scope")
    if missing:
        raise ValueError("missing required values: " + ", ".join(missing))
    return OAuthE2EConfig(
        issuer_url=issuer_url,
        resource_url=resource_url,
        approval_token=approval_token,
        redirect_uri=redirect_uri,
        scope=scope,
        timeout=args.timeout,
    )


def _register_client(config: Any) -> Any:
    status, _headers, body = _post_json(
        f"{config.issuer_url}/register",
        {
            "client_name": "Atlas Content Ops marketer verify OAuth e2e smoke",
            "redirect_uris": [config.redirect_uri],
            "token_endpoint_auth_method": "client_secret_post",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": config.scope,
        },
        config.timeout,
    )
    if status not in {200, 201}:
        raise RuntimeError(f"client registration returned HTTP {status}")
    client_id = str(body.get("client_id") or "")
    client_secret = str(body.get("client_secret") or "")
    if not client_id or not client_secret:
        raise RuntimeError("client registration response missing client_id or client_secret")
    return RegisteredClient(client_id=client_id, client_secret=client_secret)


def _tool_surface_errors(tool_names: set[str]) -> list[str]:
    errors: list[str] = []
    missing = sorted(EXPECTED_TOOLS - tool_names)
    extra = sorted(tool_names - EXPECTED_TOOLS)
    denied = sorted(tool_names & DENIED_TOOLS)
    if missing:
        errors.append("missing Content Ops marketer verify tools: " + ", ".join(missing))
    if extra:
        errors.append("unexpected tools exposed: " + ", ".join(extra))
    if denied:
        errors.append("denied tools exposed: " + ", ".join(denied))
    return errors


async def _run_smoke(config: Any) -> set[str]:
    client = _register_client(config)
    auth = _start_authorization(config, client)
    code = _approve_authorization(config, auth)
    token = _exchange_token(config, client, auth, code)
    return await _list_mcp_tools(config, token.access_token)


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        tool_names = asyncio.run(_run_smoke(config))
    except Exception as exc:
        print(f"OAuth e2e smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    errors = _tool_surface_errors(tool_names)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("OK: Content Ops marketer verify OAuth e2e smoke completed")
    print("- dynamic client registration succeeded")
    print("- operator approval and token exchange succeeded")
    print(f"- OAuth-authenticated MCP session exposes {len(tool_names)} verify-only tool")
    for name in sorted(tool_names):
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
