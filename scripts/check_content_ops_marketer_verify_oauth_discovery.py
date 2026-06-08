#!/usr/bin/env python3
"""Verify public OAuth discovery for the Content Ops marketer verify MCP server."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_draft_writer_checker():
    path = ROOT / "scripts/check_invoicing_draft_writer_oauth_discovery.py"
    spec = importlib.util.spec_from_file_location(
        "check_invoicing_draft_writer_oauth_discovery",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load OAuth discovery helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_draft_checker = _load_draft_writer_checker()
_authorization_metadata_url = _draft_checker._authorization_metadata_url
_fetch_json = _draft_checker._fetch_json
_metadata_errors = _draft_checker._metadata_errors
_probe_mcp_unauthenticated = _draft_checker._probe_mcp_unauthenticated
_protected_resource_metadata_url = _draft_checker._protected_resource_metadata_url
_strip_trailing_slash = _draft_checker._strip_trailing_slash


DEFAULT_SCOPE = "content_ops.review.verify"
ISSUER_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"
RESOURCE_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-check OAuth discovery for the Content Ops marketer verify MCP connector."
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


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    issuer_url = _strip_trailing_slash(args.issuer_url)
    resource_url = _strip_trailing_slash(args.resource_url)
    if not issuer_url:
        print(f"--issuer-url or {ISSUER_ENV} is required", file=sys.stderr)
        return 2
    if not resource_url:
        print(f"--resource-url or {RESOURCE_ENV} is required", file=sys.stderr)
        return 2

    auth_url = _authorization_metadata_url(issuer_url)
    protected_url = _protected_resource_metadata_url(resource_url)
    try:
        auth_metadata = _fetch_json(auth_url, args.timeout)
        resource_metadata = _fetch_json(protected_url, args.timeout)
        mcp_status, mcp_headers = _probe_mcp_unauthenticated(resource_url, args.timeout)
    except Exception as exc:
        print(f"OAuth discovery smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    errors = _metadata_errors(
        issuer_url=issuer_url,
        resource_url=resource_url,
        scope=args.scope,
        auth_metadata=auth_metadata,
        resource_metadata=resource_metadata,
        mcp_status=mcp_status,
        mcp_headers=mcp_headers,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("OK: Content Ops marketer verify OAuth discovery is routable")
    print(f"- authorization metadata: {auth_url}")
    print(f"- protected resource metadata: {protected_url}")
    print(f"- MCP resource: {resource_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
