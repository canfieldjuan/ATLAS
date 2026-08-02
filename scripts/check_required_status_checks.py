#!/usr/bin/env python3
"""Validate branch-protection required status check contexts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple, Sequence


GITHUB_ACTIONS_APP_ID = 15368
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_REGISTRY = ROOT / "ci" / "gates.yml"
BRANCH_REQUIRED = "branch_required"
ALLOWED_ENFORCEMENTS = frozenset(
    {
        BRANCH_REQUIRED,
        "ci_blocking_not_required",
        "local_blocking",
        "advisory",
        "scheduled",
    }
)


class RequiredCheck(NamedTuple):
    context: str
    app_id: int


class RequiredCheckFailure(NamedTuple):
    context: str
    reason: str


def _decode_quoted_scalar(value: str, *, lineno: int) -> str:
    quote = value[0]
    inner = value[1:-1]
    if quote == "'":
        decoded: list[str] = []
        index = 0
        while index < len(inner):
            char = inner[index]
            if char == "'":
                if index + 1 < len(inner) and inner[index + 1] == "'":
                    decoded.append("'")
                    index += 2
                    continue
                raise ValueError(f"ci/gates.yml:{lineno}: malformed quoted scalar")
            decoded.append(char)
            index += 1
        return "".join(decoded)

    decoded = []
    escaped = False
    supported_escapes = {'"': '"', "\\": "\\"}
    for char in inner:
        if escaped:
            if char not in supported_escapes:
                raise ValueError(
                    f"ci/gates.yml:{lineno}: unsupported escape sequence: \\{char}"
                )
            decoded.append(supported_escapes[char])
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        decoded.append(char)
    if escaped:
        raise ValueError(f"ci/gates.yml:{lineno}: malformed quoted scalar")
    return "".join(decoded)


def _parse_scalar(raw: str, *, lineno: int) -> str | bool | None:
    value = raw.strip()
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith(("'", '"')) or value.endswith(("'", '"')):
        if (
            len(value) < 2
            or value[0] != value[-1]
            or not value.startswith(("'", '"'))
        ):
            raise ValueError(f"ci/gates.yml:{lineno}: malformed quoted scalar")
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value.startswith(("'", '"'))
    ):
        return _decode_quoted_scalar(value, lineno=lineno)
    return value


def _parse_key_value(text: str, *, lineno: int) -> tuple[str, str | bool | None]:
    key, separator, raw_value = text.partition(":")
    if not separator or not key.strip():
        raise ValueError(f"ci/gates.yml:{lineno}: expected key: value")
    return key.strip(), _parse_scalar(raw_value, lineno=lineno)


def _strip_inline_comment(raw_line: str, *, lineno: int) -> str:
    quote: str | None = None
    escaped = False
    index = 0
    length = len(raw_line)
    while index < length:
        char = raw_line[index]
        if quote is not None:
            if quote == '"' and escaped:
                escaped = False
                index += 1
                continue
            if quote == '"' and char == "\\":
                escaped = True
                index += 1
                continue
            if quote == "'" and char == "'" and index + 1 < length and raw_line[index + 1] == "'":
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "#":
            return raw_line[:index].rstrip()
        index += 1
    if quote is not None:
        raise ValueError(f"ci/gates.yml:{lineno}: malformed quoted scalar")
    return raw_line.rstrip()


def parse_gate_registry(text: str) -> list[dict[str, str | bool | None]]:
    """Parse the constrained YAML shape used by ci/gates.yml.

    This is deliberately not a general YAML parser. The branch-protection audit
    workflow runs this script without dependency installation, so the registry
    uses only ``gates:`` plus a list of flat key/value mappings.
    """

    gates: list[dict[str, str | bool | None]] = []
    current: dict[str, str | bool | None] | None = None
    in_gates = False
    seen_ids: set[str] = set()
    seen_contexts: set[str] = set()

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_inline_comment(raw_line, lineno=lineno)
        if not line.strip():
            continue
        if line == "gates:":
            if in_gates:
                raise ValueError(f"ci/gates.yml:{lineno}: duplicate gates section")
            in_gates = True
            continue
        if not in_gates:
            raise ValueError(f"ci/gates.yml:{lineno}: expected gates section")
        if line.startswith("  - "):
            if current is not None:
                _validate_gate(current, seen_ids, seen_contexts, len(gates) + 1)
                gates.append(current)
            current = {}
            remainder = line[4:].strip()
            if remainder:
                key, value = _parse_key_value(remainder, lineno=lineno)
                current[key] = value
            continue
        if line.startswith("    "):
            if current is None:
                raise ValueError(f"ci/gates.yml:{lineno}: gate field before gate item")
            key, value = _parse_key_value(line.strip(), lineno=lineno)
            if key in current:
                raise ValueError(f"ci/gates.yml:{lineno}: duplicate field {key!r}")
            current[key] = value
            continue
        raise ValueError(f"ci/gates.yml:{lineno}: unsupported registry shape")

    if current is not None:
        _validate_gate(current, seen_ids, seen_contexts, len(gates) + 1)
        gates.append(current)
    if not gates:
        raise ValueError("ci/gates.yml: no gates declared")
    if not any(gate.get("enforcement") == BRANCH_REQUIRED for gate in gates):
        raise ValueError("ci/gates.yml: at least one branch_required gate required")
    return gates


def _validate_gate(
    gate: dict[str, str | bool | None],
    seen_ids: set[str],
    seen_contexts: set[str],
    index: int,
) -> None:
    required_fields = {
        "id",
        "name",
        "context",
        "enforcement",
        "trusted_base",
        "workflow",
        "local_command",
    }
    missing = sorted(required_fields - set(gate))
    if missing:
        raise ValueError(f"ci/gates.yml: gate {index} missing fields: {', '.join(missing)}")
    extra = sorted(set(gate) - required_fields)
    if extra:
        raise ValueError(f"ci/gates.yml: gate {index} unsupported fields: {', '.join(extra)}")

    gate_id = gate["id"]
    if not isinstance(gate_id, str) or not gate_id:
        raise ValueError(f"ci/gates.yml: gate {index} has invalid id")
    if gate_id in seen_ids:
        raise ValueError(f"ci/gates.yml: duplicate gate id: {gate_id}")
    seen_ids.add(gate_id)

    enforcement = gate["enforcement"]
    if enforcement not in ALLOWED_ENFORCEMENTS:
        raise ValueError(
            f"ci/gates.yml: gate {gate_id} has invalid enforcement: {enforcement!r}"
        )

    name = gate["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"ci/gates.yml: gate {gate_id} has invalid name")

    trusted_base = gate["trusted_base"]
    if not isinstance(trusted_base, bool):
        raise ValueError(f"ci/gates.yml: gate {gate_id} trusted_base must be true/false")

    workflow = gate["workflow"]
    if not isinstance(workflow, str) or not workflow:
        raise ValueError(f"ci/gates.yml: gate {gate_id} has invalid workflow")

    context = gate["context"]
    if enforcement == BRANCH_REQUIRED and (not isinstance(context, str) or not context):
        raise ValueError(f"ci/gates.yml: gate {gate_id} branch_required needs context")
    if context is not None and not isinstance(context, str):
        raise ValueError(f"ci/gates.yml: gate {gate_id} has invalid context")
    if isinstance(context, str) and context:
        if context in seen_contexts:
            raise ValueError(f"ci/gates.yml: duplicate context: {context}")
        seen_contexts.add(context)

    local_command = gate["local_command"]
    if local_command is not None and not isinstance(local_command, str):
        raise ValueError(f"ci/gates.yml: gate {gate_id} has invalid local_command")


def load_gate_registry(
    path: Path = DEFAULT_GATE_REGISTRY,
) -> list[dict[str, str | bool | None]]:
    return parse_gate_registry(path.read_text(encoding="utf-8"))


def default_required_contexts(path: Path = DEFAULT_GATE_REGISTRY) -> tuple[str, ...]:
    return tuple(
        str(gate["context"])
        for gate in load_gate_registry(path)
        if gate.get("enforcement") == BRANCH_REQUIRED
    )


def default_required_checks(
    *,
    registry_path: Path = DEFAULT_GATE_REGISTRY,
    app_id: int = GITHUB_ACTIONS_APP_ID,
) -> tuple[RequiredCheck, ...]:
    return tuple(
        RequiredCheck(context, app_id)
        for context in default_required_contexts(registry_path)
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
    required: Sequence[str] | None = None,
) -> list[str]:
    """Return required contexts absent from the GitHub payload."""
    if required is None:
        required = default_required_contexts()
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
    required: Sequence[RequiredCheck] | None = None,
) -> list[RequiredCheckFailure]:
    """Return missing or wrong-source required check failures."""
    if required is None:
        required = default_required_checks()
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
            "required status context; may be repeated. Defaults to ci/gates.yml "
            "branch_required entries. Each required context must be provided by "
            "GitHub Actions."
        ),
    )
    parser.add_argument(
        "--registry-file",
        type=Path,
        default=DEFAULT_GATE_REGISTRY,
        help=(
            "gate registry used for default required contexts when --required is "
            "omitted; defaults to ci/gates.yml"
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
    try:
        contexts = (
            tuple(args.required)
            if args.required
            else default_required_contexts(args.registry_file)
        )
        required = tuple(
            RequiredCheck(context, args.github_actions_app_id)
            for context in contexts
        )
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
