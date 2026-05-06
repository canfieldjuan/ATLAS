#!/usr/bin/env python3
"""Check standalone AI Content Ops host install readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_install_check import (  # noqa: E402
    check_campaign_install,
)


PROFILE_CHOICES = (
    "offline",
    "generation",
    "send",
    "sequence",
    "webhooks",
    "analytics",
    "export",
    "all",
)
SENDER_CHOICES = ("none", "resend", "ses")
LLM_CHOICES = ("pipeline", "offline")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect local environment readiness for standalone AI Content Ops "
            "campaign CLIs without connecting to Postgres or providers."
        )
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=PROFILE_CHOICES,
        default=[],
        help=(
            "Install profile to check. May repeat. Defaults to offline. "
            "Use all for all DB-backed worker profiles."
        ),
    )
    parser.add_argument(
        "--sender",
        choices=SENDER_CHOICES,
        default=None,
        help="Sender credentials to validate when --profile send is selected.",
    )
    parser.add_argument(
        "--llm",
        choices=LLM_CHOICES,
        default="pipeline",
        help="LLM mode to validate when --profile generation is selected.",
    )
    parser.add_argument(
        "--skip-webhook-secret",
        action="store_true",
        help="Do not require a Resend signing secret for webhook replay installs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = check_campaign_install(
        profiles=tuple(args.profile or ("offline",)),
        sender=args.sender,
        llm=args.llm,
        require_webhook_secret=not args.skip_webhook_secret,
    )
    data = report.as_dict()
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(_format_text(data))
    return 0 if report.passed else 1


def _format_text(data: dict[str, object]) -> str:
    counts = data["counts"]
    assert isinstance(counts, dict)
    lines = [
        (
            "passed={passed} profiles={profiles} sender={sender} "
            "ok={ok} warnings={warning} errors={error}"
        ).format(
            passed=str(data["passed"]).lower(),
            profiles=",".join(str(item) for item in data["profiles"]),
            sender=data["sender"],
            ok=counts["ok"],
            warning=counts["warning"],
            error=counts["error"],
        )
    ]
    checks = data["checks"]
    assert isinstance(checks, list)
    for check in checks:
        if not isinstance(check, dict):
            continue
        lines.append(
            "{status} {name}: {message}".format(
                status=str(check.get("status", "")).upper(),
                name=check.get("name", ""),
                message=check.get("message", ""),
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
