#!/usr/bin/env python3
"""Verify Tailscale Funnel routes for the draft-writer invoicing OAuth connector.

This check is read-only. It inspects ``tailscale funnel status --json`` and
prints the operator commands needed to repair missing or mismatched routes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


DEFAULT_RESOURCE_URL = "https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp"
DEFAULT_PORT = "8066"
METADATA_PATH = "/.well-known/oauth-protected-resource"


@dataclass(frozen=True)
class RouteExpectation:
    host_key: str
    app_path: str
    app_proxy: str
    metadata_path: str
    metadata_proxy: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check that Tailscale Funnel has the two public routes required "
            "for the draft-writer invoicing OAuth connector."
        )
    )
    parser.add_argument(
        "--resource-url",
        default=os.environ.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL", DEFAULT_RESOURCE_URL),
        help=(
            "Public Streamable HTTP MCP resource URL. Defaults to "
            "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL or the "
            "standard draft-writer public URL."
        ),
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT", DEFAULT_PORT),
        help=(
            "Local draft-writer MCP port. Defaults to "
            "ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT or 8066."
        ),
    )
    parser.add_argument(
        "--tailscale-bin",
        default="tailscale",
        help="Tailscale executable. Default: tailscale.",
    )
    return parser


def _strip_proxy(value: Any) -> str:
    return str(value or "").rstrip("/")


def _host_key(resource_url: str) -> str:
    parsed = urlparse(resource_url.strip())
    if parsed.scheme != "https":
        raise ValueError("--resource-url must use https for public Funnel routing")
    if not parsed.netloc:
        raise ValueError("--resource-url must include a host")
    if ":" in parsed.netloc:
        return parsed.netloc
    return f"{parsed.netloc}:443"


def _app_path(resource_url: str) -> str:
    path = urlparse(resource_url.strip()).path.rstrip("/")
    if path.endswith("/mcp"):
        path = path[: -len("/mcp")]
    return path or "/"


def _validate_port(port: str) -> str:
    try:
        parsed = int(str(port).strip())
    except ValueError as exc:
        raise ValueError("--port must be an integer") from exc
    if parsed <= 0 or parsed > 65535:
        raise ValueError("--port must be between 1 and 65535")
    return str(parsed)


def _expectation(resource_url: str, port: str) -> RouteExpectation:
    parsed_port = _validate_port(port)
    app_path = _app_path(resource_url)
    metadata_path = METADATA_PATH if app_path == "/" else f"{METADATA_PATH}{app_path}"
    return RouteExpectation(
        host_key=_host_key(resource_url),
        app_path=app_path,
        app_proxy=f"http://127.0.0.1:{parsed_port}",
        metadata_path=metadata_path,
        metadata_proxy=f"http://127.0.0.1:{parsed_port}{metadata_path}",
    )


def _load_funnel_status(tailscale_bin: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [tailscale_bin, "funnel", "status", "--json"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"could not run {tailscale_bin}: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"tailscale funnel status failed: {stderr}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("tailscale funnel status did not return JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("tailscale funnel status returned non-object JSON")
    return data


def _handlers(status: Mapping[str, Any], host_key: str) -> Mapping[str, Any]:
    web = status.get("Web")
    if not isinstance(web, dict):
        return {}
    host = web.get(host_key)
    if not isinstance(host, dict):
        return {}
    handlers = host.get("Handlers")
    if not isinstance(handlers, dict):
        return {}
    return handlers


def _handler_proxy(handlers: Mapping[str, Any], path: str) -> str:
    value = handlers.get(path)
    if not isinstance(value, dict):
        return ""
    return _strip_proxy(value.get("Proxy"))


def _route_errors(status: Mapping[str, Any], expected: RouteExpectation) -> list[str]:
    handlers = _handlers(status, expected.host_key)
    errors: list[str] = []

    app_proxy = _handler_proxy(handlers, expected.app_path)
    if app_proxy != expected.app_proxy:
        errors.append(
            f"{expected.host_key} handler {expected.app_path} proxy is "
            f"{app_proxy or 'missing'}, expected {expected.app_proxy}"
        )

    metadata_proxy = _handler_proxy(handlers, expected.metadata_path)
    if metadata_proxy != expected.metadata_proxy:
        errors.append(
            f"{expected.host_key} handler {expected.metadata_path} proxy is "
            f"{metadata_proxy or 'missing'}, expected {expected.metadata_proxy}"
        )

    return errors


def _repair_commands(expected: RouteExpectation) -> list[str]:
    commands: list[str] = []
    if expected.app_path == "/":
        commands.append(f"tailscale funnel --bg --yes {expected.app_proxy}")
    else:
        commands.append(
            "tailscale funnel --bg --yes \\\n"
            f"  --set-path {expected.app_path} \\\n"
            f"  {expected.app_proxy}"
        )
    commands.append(
        "tailscale funnel --bg --yes \\\n"
        f"  --set-path {expected.metadata_path} \\\n"
        f"  {expected.metadata_proxy}"
    )
    return commands


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        expected = _expectation(args.resource_url, args.port)
        status = _load_funnel_status(args.tailscale_bin)
    except Exception as exc:
        print(f"Draft-writer Funnel route check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    errors = _route_errors(status, expected)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print("Run the required route commands:", file=sys.stderr)
        for command in _repair_commands(expected):
            print(command, file=sys.stderr)
        return 1

    print("OK: draft-writer invoicing Funnel routes are configured")
    print(f"- host: {expected.host_key}")
    print(f"- connector route: {expected.app_path} -> {expected.app_proxy}")
    print(f"- metadata route: {expected.metadata_path} -> {expected.metadata_proxy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
