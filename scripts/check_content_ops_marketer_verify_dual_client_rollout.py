#!/usr/bin/env python3
"""Run both Content Ops marketer verify public OAuth profile smokes."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_oauth_e2e():
    path = ROOT / "scripts/check_content_ops_marketer_verify_oauth_e2e.py"
    spec = importlib.util.spec_from_file_location(
        "check_content_ops_marketer_verify_oauth_e2e",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load OAuth e2e checker from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_oauth_e2e = _load_oauth_e2e()
CLAUDE_RICH_PROFILE = _oauth_e2e.CLAUDE_RICH_PROFILE
CHATGPT_SEARCH_FETCH_PROFILE = _oauth_e2e.CHATGPT_SEARCH_FETCH_PROFILE


def _strip(value: str) -> str:
    return value.strip().rstrip("/")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Claude rich plus ChatGPT search/fetch OAuth list-tools smokes."
    )
    for flag, help_text in (
        ("--rich-issuer-url", "Claude rich verifier OAuth issuer URL."),
        ("--rich-resource-url", "Claude rich verifier MCP resource URL."),
        ("--chatgpt-adapter-issuer-url", "ChatGPT adapter OAuth issuer URL."),
        ("--chatgpt-adapter-resource-url", "ChatGPT adapter MCP resource URL."),
    ):
        parser.add_argument(flag, default="", help=help_text)
    parser.add_argument(
        "--approval-token-file",
        default="",
        help="Read the operator approval token from a local file.",
    )
    parser.add_argument(
        "--approval-token",
        default="",
        help="Operator approval token. This value is not printed.",
    )
    parser.add_argument(
        "--redirect-uri",
        default=_oauth_e2e.DEFAULT_REDIRECT_URI,
        help=f"OAuth redirect URI. Default: {_oauth_e2e.DEFAULT_REDIRECT_URI}.",
    )
    parser.add_argument(
        "--scope",
        default=_oauth_e2e.DEFAULT_SCOPE,
        help=f"Required OAuth scope. Default: {_oauth_e2e.DEFAULT_SCOPE}.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser


def _value(args: argparse.Namespace, attr: str) -> str:
    value = getattr(args, attr)
    return value.strip() if isinstance(value, str) else ""


def _validate_args(args: argparse.Namespace) -> list[str]:
    required = {
        "rich_issuer_url": "--rich-issuer-url",
        "rich_resource_url": "--rich-resource-url",
        "chatgpt_adapter_issuer_url": "--chatgpt-adapter-issuer-url",
        "chatgpt_adapter_resource_url": "--chatgpt-adapter-resource-url",
    }
    missing = [flag for attr, flag in required.items() if not _value(args, attr)]
    if not _value(args, "approval_token_file") and not _value(args, "approval_token"):
        missing.append("--approval-token-file or --approval-token")
    if missing:
        return ["missing required values: " + ", ".join(missing)]
    if _strip(args.rich_resource_url) == _strip(args.chatgpt_adapter_resource_url):
        return ["rich and ChatGPT adapter resource URLs must be different"]
    return []


def _secret_argv(args: argparse.Namespace) -> tuple[str, str]:
    if _value(args, "approval_token_file"):
        return "--approval-token-file", args.approval_token_file.strip()
    return "--approval-token", args.approval_token.strip()


def _checker_argv(
    *,
    issuer_url: str,
    resource_url: str,
    profile: str,
    args: argparse.Namespace,
) -> list[str]:
    secret_flag, secret_value = _secret_argv(args)
    return [
        "--issuer-url",
        _strip(issuer_url),
        "--resource-url",
        _strip(resource_url),
        "--client-profile",
        profile,
        secret_flag,
        secret_value,
        "--redirect-uri",
        args.redirect_uri.strip(),
        "--scope",
        args.scope.strip(),
        "--timeout",
        str(args.timeout),
    ]


def _invocations_from_args(args: argparse.Namespace) -> tuple[tuple[str, str, str, str, list[str]], ...]:
    return (
        (
            "Claude rich verifier",
            CLAUDE_RICH_PROFILE,
            _strip(args.rich_issuer_url),
            _strip(args.rich_resource_url),
            _checker_argv(
                issuer_url=args.rich_issuer_url,
                resource_url=args.rich_resource_url,
                profile=CLAUDE_RICH_PROFILE,
                args=args,
            ),
        ),
        (
            "ChatGPT search/fetch adapter",
            CHATGPT_SEARCH_FETCH_PROFILE,
            _strip(args.chatgpt_adapter_issuer_url),
            _strip(args.chatgpt_adapter_resource_url),
            _checker_argv(
                issuer_url=args.chatgpt_adapter_issuer_url,
                resource_url=args.chatgpt_adapter_resource_url,
                profile=CHATGPT_SEARCH_FETCH_PROFILE,
                args=args,
            ),
        ),
    )


def _run_invocations(invocations, runner) -> int:
    for label, profile, issuer_url, resource_url, checker_argv in invocations:
        print(f"Running {label} smoke ({profile})")
        print(f"- issuer: {issuer_url}")
        print(f"- resource: {resource_url}")
        result = runner(checker_argv)
        if result != 0:
            print(f"Dual-client rollout smoke failed for {profile}", file=sys.stderr)
            return result
    print("OK: Content Ops marketer verify dual-client rollout smoke completed")
    return 0


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = _validate_args(args)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2
    return _run_invocations(_invocations_from_args(args), _oauth_e2e._main)


if __name__ == "__main__":
    raise SystemExit(_main())
