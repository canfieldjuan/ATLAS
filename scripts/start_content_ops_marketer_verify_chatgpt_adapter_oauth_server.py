#!/usr/bin/env python3
"""Start the ChatGPT Content Ops marketer verify adapter in OAuth mode."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import start_content_ops_marketer_verify_oauth_server as rich_launcher


DEFAULT_HOST = rich_launcher.DEFAULT_HOST
DEFAULT_PORT = "8069"
ADAPTER_PORT_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_CHATGPT_PORT"
RUNTIME_PORT_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT"
DEFAULT_ISSUER_URL = "https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer-chatgpt"
DEFAULT_RESOURCE_URL = (
    "https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer-chatgpt/mcp"
)
LaunchConfig = rich_launcher.LaunchConfig
REQUIRED_KEYS = rich_launcher.REQUIRED_KEYS
_build_base_parser = rich_launcher._build_parser
_funnel_app_path = rich_launcher._funnel_app_path
_funnel_metadata_path = rich_launcher._funnel_metadata_path
_load_dotenv_files = rich_launcher._load_dotenv_files
_masked_env_report = rich_launcher._masked_env_report
_read_secret_file = rich_launcher._read_secret_file
_validate_env = rich_launcher._validate_env


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_base_parser()
    parser.description = (
        "Load Atlas dotenv files and start the ChatGPT Content Ops marketer "
        "verify search/fetch adapter in OAuth connector mode."
    )
    parser.set_defaults(port=None, issuer_url=None, resource_url=None)
    _set_option_help(
        parser,
        "--port",
        f"Bind port. Defaults to env {ADAPTER_PORT_ENV} or {DEFAULT_PORT}.",
    )
    _set_option_help(
        parser,
        "--issuer-url",
        f"OAuth issuer URL. Defaults to env value or {DEFAULT_ISSUER_URL}.",
    )
    _set_option_help(
        parser,
        "--resource-url",
        f"OAuth resource URL. Defaults to env value or {DEFAULT_RESOURCE_URL}.",
    )
    return parser


def _set_option_help(parser: argparse.ArgumentParser, option: str, help_text: str) -> None:
    for action in parser._actions:
        if option in action.option_strings:
            action.help = help_text
            return
    raise RuntimeError(f"adapter launcher parser missing {option}")


def _build_launch_config(args: argparse.Namespace) -> LaunchConfig:
    env_files = [Path(path) for path in (args.env_file or [".env", ".env.local"])]
    env = _load_dotenv_files(env_files)
    env.update(rich_launcher.os.environ)
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_AUTH_MODE"] = "oauth"
    env["ATLAS_MCP_HOST"] = (args.host or env.get("ATLAS_MCP_HOST") or DEFAULT_HOST).strip()
    resolved_port = (args.port or env.get(ADAPTER_PORT_ENV) or DEFAULT_PORT).strip()
    env[ADAPTER_PORT_ENV] = resolved_port
    env[RUNTIME_PORT_ENV] = resolved_port
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"] = (
        args.issuer_url
        or env.get("ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL")
        or DEFAULT_ISSUER_URL
    ).strip()
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL"] = (
        args.resource_url
        or env.get("ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL")
        or DEFAULT_RESOURCE_URL
    ).strip()
    if args.approval_token_file:
        env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN"] = _read_secret_file(
            args.approval_token_file
        )
    return LaunchConfig(
        env=env,
        python=args.python,
        host=env["ATLAS_MCP_HOST"],
        port=env[RUNTIME_PORT_ENV],
        dry_run=args.dry_run,
        approval_token_file=args.approval_token_file,
    )


def _server_command(config: LaunchConfig) -> list[str]:
    return [
        config.python,
        "-m",
        "atlas_brain.mcp.content_ops_marketer_verify_chatgpt_adapter_server",
        "--sse",
    ]


def _print_operator_guidance(config: LaunchConfig) -> None:
    issuer_url = config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"]
    resource_url = config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL"]
    app_path = _funnel_app_path(resource_url)
    metadata_path = _funnel_metadata_path(resource_url)
    print("Content Ops marketer verify ChatGPT adapter OAuth launch configuration:")
    for line in _masked_env_report(config.env):
        print(f"- {line}")
    print(f"- ATLAS_MCP_HOST={config.host}")
    print(f"- {ADAPTER_PORT_ENV}={config.port}")
    print(f"- {RUNTIME_PORT_ENV}={config.port}")
    print()
    print("Required Funnel route for the ChatGPT adapter path:")
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
    print(".venv/bin/python scripts/check_content_ops_marketer_verify_oauth_discovery.py \\")
    print(f"  --issuer-url {issuer_url} \\")
    print(f"  --resource-url {resource_url}")
    print()
    print("OAuth e2e smoke for ChatGPT search/fetch connector profile:")
    print(".venv/bin/python scripts/check_content_ops_marketer_verify_oauth_e2e.py \\")
    print(f"  --issuer-url {issuer_url} \\")
    print(f"  --resource-url {resource_url} \\")
    print("  --client-profile chatgpt-search-fetch \\")
    token_file = config.approval_token_file or "/path/to/local-approval-token"
    print(f"  --approval-token-file {shlex.quote(token_file)}")
    if config.approval_token_file is None:
        print(
            "If the approval token came from .env/--env-file, write it to a local token "
            "file or export ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN "
            "and pass --approval-token."
        )
    print()
    print("After the rich verifier is also public, run the dual-client rollout smoke:")
    print(".venv/bin/python scripts/check_content_ops_marketer_verify_dual_client_rollout.py \\")
    print(f"  --rich-issuer-url {rich_launcher.DEFAULT_ISSUER_URL} \\")
    print(f"  --rich-resource-url {rich_launcher.DEFAULT_RESOURCE_URL} \\")
    print(f"  --chatgpt-adapter-issuer-url {issuer_url} \\")
    print(f"  --chatgpt-adapter-resource-url {resource_url} \\")
    print(f"  --approval-token-file {shlex.quote(token_file)}")


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
