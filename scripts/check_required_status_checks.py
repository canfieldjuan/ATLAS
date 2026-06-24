#!/usr/bin/env python3
"""Validate branch-protection required status check contexts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple, Sequence


GITHUB_ACTIONS_APP_ID = 15368
DEFAULT_REQUIRED_CONTEXTS = (
    "live-reconciliation",
    "Gitleaks PR secret scan",
    "Gitleaks baseline growth guard",
)


class RequiredCheck(NamedTuple):
    context: str
    app_id: int


class RequiredCheckFailure(NamedTuple):
    context: str
    reason: str


DEFAULT_REQUIRED_CHECKS = tuple(
    RequiredCheck(context, GITHUB_ACTIONS_APP_ID)
    for context in DEFAULT_REQUIRED_CONTEXTS
)


def _required_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = payload.get("required_status_checks")
    if isinstance(required, dict):
        return required
    return payload


def required_status_contexts(payload: dict[str, Any]) -> set[str]:
    """Return required status contexts from GitHub's branch-protection payload."""
    payload = _required_status_payload(payload)

    contexts: set[str] = set()
    raw_contexts = payload.get("contexts")
    if isinstance(raw_contexts, list):
        contexts.update(item for item in raw_contexts if isinstance(item, str))

    raw_checks = payload.get("checks")
    if isinstance(raw_checks, list):
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            context = item.get("context")
            if isinstance(context, str):
                contexts.add(context)
    return contexts


def required_status_check_app_ids(payload: dict[str, Any]) -> dict[str, set[int | None]]:
    """Return required check app IDs by context; None represents legacy contexts."""
    payload = _required_status_payload(payload)

    app_ids_by_context: dict[str, set[int | None]] = {}
    raw_contexts = payload.get("contexts")
    if isinstance(raw_contexts, list):
        for item in raw_contexts:
            if isinstance(item, str):
                app_ids_by_context.setdefault(item, set()).add(None)

    raw_checks = payload.get("checks")
    if isinstance(raw_checks, list):
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            context = item.get("context")
            if not isinstance(context, str):
                continue
            app_id = item.get("app_id")
            app_ids_by_context.setdefault(context, set()).add(
                app_id
                if isinstance(app_id, int) and not isinstance(app_id, bool)
                else None
            )
    return app_ids_by_context


def missing_required_contexts(
    payload: dict[str, Any],
    required: Sequence[str] = DEFAULT_REQUIRED_CONTEXTS,
) -> list[str]:
    """Return required contexts absent from the GitHub payload."""
    present = required_status_contexts(payload)
    return [context for context in required if context not in present]


def _format_app_ids(app_ids: set[int | None]) -> str:
    if not app_ids:
        return "none"
    labels = [
        "legacy/unpinned"
        if app_id is None
        else f"app_id {app_id}"
        for app_id in sorted(app_ids, key=lambda value: -1 if value is None else value)
    ]
    return ", ".join(labels)


def required_status_check_failures(
    payload: dict[str, Any],
    required: Sequence[RequiredCheck] = DEFAULT_REQUIRED_CHECKS,
) -> list[RequiredCheckFailure]:
    """Return missing or wrong-source required check failures."""
    app_ids_by_context = required_status_check_app_ids(payload)
    failures: list[RequiredCheckFailure] = []
    for check in required:
        app_ids = app_ids_by_context.get(check.context, set())
        if not app_ids:
            failures.append(
                RequiredCheckFailure(check.context, "missing required check")
            )
            continue
        if check.app_id not in app_ids:
            failures.append(
                RequiredCheckFailure(
                    check.context,
                    (
                        f"required check is not pinned to GitHub Actions "
                        f"(expected app_id {check.app_id}; found {_format_app_ids(app_ids)})"
                    ),
                )
            )
    return failures


def _read_payload(path: str | None) -> dict[str, Any]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("required-status payload must be a JSON object")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate required branch-protection status checks."
    )
    parser.add_argument(
        "--payload-file",
        help="JSON file from GitHub's required_status_checks endpoint; defaults to stdin.",
    )
    parser.add_argument(
        "--required",
        action="append",
        default=[],
        help=(
            "required status context; may be repeated. Defaults to Atlas security "
            "guardrails. Each required context must be provided by GitHub Actions."
        ),
    )
    parser.add_argument(
        "--github-actions-app-id",
        type=int,
        default=GITHUB_ACTIONS_APP_ID,
        help="GitHub Actions app_id expected in checks[].app_id.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contexts = tuple(args.required) if args.required else DEFAULT_REQUIRED_CONTEXTS
    required = tuple(
        RequiredCheck(context, args.github_actions_app_id)
        for context in contexts
    )
    try:
        payload = _read_payload(args.payload_file)
    except (OSError, ValueError) as exc:
        print(f"required status check audit: {exc}", file=sys.stderr)
        return 2

    failures = required_status_check_failures(payload, required)
    if failures:
        print("required status check audit: FAIL")
        for failure in failures:
            print(f"- {failure.context}: {failure.reason}")
        return 1

    print("required status check audit: PASS")
    for check in required:
        print(f"- required: {check.context} (GitHub Actions app_id {check.app_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
