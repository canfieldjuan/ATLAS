#!/usr/bin/env python3
"""Start the draft-writer invoicing MCP server in OAuth mode."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = "8066"
DEFAULT_ISSUER_URL = "https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer"
DEFAULT_RESOURCE_URL = "https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp"
PROTECTED_RESOURCE_METADATA_PATH = "/.well-known/oauth-protected-resource"
MIN_APPROVAL_TOKEN_LENGTH = 24
SECRET_KEYS = {
    "ATLAS_MCP_AUTH_TOKEN",
    "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN",
}
REQUIRED_KEYS = (
    "ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE",
    "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL",
    "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL",
    "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN",
)


@dataclass(frozen=True)
class LaunchConfig:
    env: dict[str, str]
    python: str
    host: str
    port: str
    dry_run: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load Atlas dotenv files and start the draft-writer invoicing MCP "
            "server in ChatGPT-compatible OAuth mode."
        )
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=None,
        help="Dotenv file to load. May be repeated. Defaults to .env then .env.local.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the MCP server. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Bind host. Defaults to env ATLAS_MCP_HOST or {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--port",
        default=None,
        help=f"Bind port. Defaults to env ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT or {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--issuer-url",
        default=None,
        help=f"OAuth issuer URL. Defaults to env value or {DEFAULT_ISSUER_URL}.",
    )
    parser.add_argument(
        "--resource-url",
        default=None,
        help=f"OAuth resource URL. Defaults to env value or {DEFAULT_RESOURCE_URL}.",
    )
    parser.add_argument(
        "--approval-token-file",
        default=None,
        help=(
            "Read ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN from this local file. "
            "Prefer this over putting write-approval secrets in shell history."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the launch/smoke commands without starting the server.",
    )
    return parser


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _load_dotenv_files(paths: list[Path]) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            parsed = _parse_dotenv_line(line)
            if parsed is None:
                continue
            key, value = parsed
            values[key] = value
    return values


def _read_secret_file(path: str) -> str:
    secret_path = Path(path)
    try:
        value = secret_path.read_text().strip()
    except OSError as exc:
        raise ValueError(f"could not read approval token file {secret_path}") from exc
    if not value:
        raise ValueError(f"approval token file {secret_path} is empty")
    return value


def _masked_env_report(env: Mapping[str, str], keys: tuple[str, ...] = REQUIRED_KEYS) -> list[str]:
    lines: list[str] = []
    for key in keys:
        value = env.get(key, "")
        if key in SECRET_KEYS:
            state = f"SET len={len(value)}" if value else "MISSING"
        else:
            state = value or "MISSING"
        lines.append(f"{key}={state}")
    return lines


def _validate_env(env: Mapping[str, str]) -> list[str]:
    errors: list[str] = []
    mode = env.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE", "").strip().lower()
    if mode != "oauth":
        errors.append("ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE must be oauth")
    issuer = env.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL", "").strip()
    if not issuer.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        errors.append("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL must be HTTPS or localhost HTTP")
    resource = env.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL", "").strip()
    if not resource.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        errors.append("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL must be HTTPS or localhost HTTP")
    approval_token = env.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN", "").strip()
    if len(approval_token) < MIN_APPROVAL_TOKEN_LENGTH:
        errors.append(
            "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN must be at least "
            f"{MIN_APPROVAL_TOKEN_LENGTH} characters"
        )
    port = env.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT", DEFAULT_PORT).strip()
    try:
        parsed_port = int(port)
    except ValueError:
        errors.append("ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT must be an integer")
    else:
        if parsed_port <= 0 or parsed_port > 65535:
            errors.append("ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT must be between 1 and 65535")
    return errors


def _build_launch_config(args: argparse.Namespace) -> LaunchConfig:
    env_files = [Path(path) for path in (args.env_file or [".env", ".env.local"])]
    env = _load_dotenv_files(env_files)
    env.update(os.environ)
    env["ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE"] = "oauth"
    env["ATLAS_MCP_HOST"] = (args.host or env.get("ATLAS_MCP_HOST") or DEFAULT_HOST).strip()
    env["ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT"] = (
        args.port or env.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT") or DEFAULT_PORT
    ).strip()
    env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL"] = (
        args.issuer_url
        or env.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL")
        or DEFAULT_ISSUER_URL
    ).strip()
    env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL"] = (
        args.resource_url
        or env.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL")
        or DEFAULT_RESOURCE_URL
    ).strip()
    if args.approval_token_file:
        env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN"] = _read_secret_file(
            args.approval_token_file
        )
    return LaunchConfig(
        env=env,
        python=args.python,
        host=env["ATLAS_MCP_HOST"],
        port=env["ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT"],
        dry_run=args.dry_run,
    )


def _server_command(config: LaunchConfig) -> list[str]:
    return [
        config.python,
        "-m",
        "atlas_brain.mcp.invoicing_draft_writer_server",
        "--sse",
    ]


def _funnel_app_path(resource_url: str) -> str:
    path = urlparse(resource_url.strip()).path.rstrip("/")
    if path.endswith("/mcp"):
        path = path[: -len("/mcp")]
    return path or "/"


def _funnel_metadata_path(resource_url: str) -> str:
    app_path = _funnel_app_path(resource_url)
    if app_path == "/":
        return PROTECTED_RESOURCE_METADATA_PATH
    return f"{PROTECTED_RESOURCE_METADATA_PATH}{app_path}"


def _print_operator_guidance(config: LaunchConfig) -> None:
    issuer_url = config.env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL"]
    resource_url = config.env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL"]
    app_path = _funnel_app_path(resource_url)
    metadata_path = _funnel_metadata_path(resource_url)
    print("Draft-writer invoicing OAuth launch configuration:")
    for line in _masked_env_report(config.env):
        print(f"- {line}")
    print(f"- ATLAS_MCP_HOST={config.host}")
    print(f"- ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT={config.port}")
    print()
    print("Required Funnel route for the draft-writer connector path:")
    print("tailscale funnel --bg --yes \\")
    if app_path == "/":
        print(f"  http://127.0.0.1:{config.port}")
    else:
        print(f"  --set-path {app_path} \\")
        print(f"  http://127.0.0.1:{config.port}")
    print()
    print("Required Funnel route for protected-resource metadata:")
    print("tailscale funnel --bg --yes \\")
    print(f"  --set-path {metadata_path} \\")
    print(f"  http://127.0.0.1:{config.port}{metadata_path}")
    print()
    print("After startup, verify public discovery:")
    print(".venv/bin/python scripts/check_invoicing_draft_writer_oauth_discovery.py \\")
    print(f"  --issuer-url {issuer_url} \\")
    print(f"  --resource-url {resource_url}")
    print()
    print("Then verify OAuth token exchange and draft-writer MCP tools:")
    print(".venv/bin/python scripts/check_invoicing_draft_writer_oauth_e2e.py \\")
    print(f"  --issuer-url {issuer_url} \\")
    print(f"  --resource-url {resource_url}")


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _build_launch_config(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    errors = _validate_env(config.env)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2

    _print_operator_guidance(config)
    command = _server_command(config)
    print()
    print("Starting server:")
    print(" ".join(command))
    if config.dry_run:
        print("dry-run: server not started")
        return 0

    return subprocess.call(command, env=config.env)


if __name__ == "__main__":
    raise SystemExit(_main())
