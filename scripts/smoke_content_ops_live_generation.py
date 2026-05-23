#!/usr/bin/env python3
"""Smoke-test live Content Ops landing-page generation through host wiring."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency
    load_dotenv = None


DEFAULT_INPUTS: Mapping[str, Any] = {
    "campaign_name": "FAQ Report",
    "offer": "Turn repeat support tickets into customer-ready FAQ answers",
    "audience": "10-50 person SaaS support team",
    "target_keyword": "support ticket FAQ",
    "secondary_keywords": [
        "reduce repeat support tickets",
        "help center answers",
    ],
    "search_intent": (
        "Find a low-friction way to turn old support tickets into answers "
        "customers can use."
    ),
    "primary_entity": "FAQ Report",
    "audience_entity": "small SaaS support team",
    "objections": [
        "Will this publish automatically?",
        "Do we need a full-time docs person?",
    ],
    "faq_questions": [
        "What happens after I upload the CSV?",
        "Does FAQ Report publish automatically?",
    ],
    "source_period": "Last 90 days of support tickets",
    "internal_links": ["/systems/ai-content-ops/intake"],
    "cta_label": "Upload Ticket CSV -- Free Analysis",
    "cta_url": "/systems/ai-content-ops/intake",
}

AsyncCallable = Callable[[], Awaitable[None]]
ServicesFactory = Callable[[], Any]
Executor = Callable[..., Awaitable[dict[str, Any]]]


def _load_dotenv_files(extra_env_files: list[Path] | None = None) -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)
        for path in extra_env_files or []:
            if not path.exists():
                raise SystemExit(f"--env-file does not exist: {path}")
            load_dotenv(path, override=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test live AI Content Ops landing-page generation through "
            "the host DB pool, packaged skills, and pipeline-routed LLM."
        )
    )
    parser.add_argument(
        "--account-id",
        required=True,
        help="Tenant/account id used to scope the generated landing-page draft.",
    )
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional .env file to load before resolving Atlas DB and LLM "
            "settings. Repeatable."
        ),
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Optional JSON object merged over the default landing-page inputs.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "Input override as key=value. JSON values are accepted. Repeatable. "
            "Example: --input cta_url=/systems/ai-content-ops/intake"
        ),
    )
    parser.add_argument(
        "--quality-repair-attempts",
        type=int,
        default=1,
        help="Landing-page quality repair attempts for the generated draft.",
    )
    parser.add_argument(
        "--no-quality-gates",
        action="store_true",
        help="Disable landing-page quality gates for this smoke run.",
    )
    parser.add_argument(
        "--output-result",
        type=Path,
        help="Write the smoke result JSON to this path.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine JSON.")
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Unable to read --input-json: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--input-json must contain a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--input-json must contain a JSON object")
    return parsed


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise SystemExit("--input overrides must use key=value")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit("--input override key cannot be empty")
    value = value.strip()
    if not value:
        return key, ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def _payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.quality_repair_attempts) < 0:
        raise SystemExit("--quality-repair-attempts must be >= 0")
    inputs = dict(DEFAULT_INPUTS)
    if args.input_json:
        inputs.update(_load_json_object(args.input_json))
    for raw in args.input:
        key, value = _parse_override(str(raw))
        inputs[key] = value
    inputs["landing_page_quality_repair_attempts"] = int(args.quality_repair_attempts)
    return {
        "outputs": ["landing_page"],
        "limit": 1,
        "require_quality_gates": not bool(args.no_quality_gates),
        "inputs": inputs,
    }


def _resolve_runtime_dependencies() -> tuple[AsyncCallable, AsyncCallable, ServicesFactory, Executor, Any]:
    from atlas_brain._content_ops_services import (  # noqa: PLC0415
        build_content_ops_execution_services,
    )
    from atlas_brain.storage.database import (  # noqa: PLC0415
        close_database,
        init_database,
    )
    from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: PLC0415
    from extracted_content_pipeline.content_ops_execution import (  # noqa: PLC0415
        execute_content_ops_from_mapping,
    )

    return (
        init_database,
        close_database,
        lambda: build_content_ops_execution_services(enable_db_services=True),
        execute_content_ops_from_mapping,
        TenantScope,
    )


def _step_result(result: Mapping[str, Any], output: str) -> Mapping[str, Any] | None:
    for step in result.get("steps") or ():
        if isinstance(step, Mapping) and step.get("output") == output:
            return step
    return None


def _smoke_errors(
    *,
    configured_outputs: tuple[str, ...],
    result: Mapping[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if "landing_page" not in configured_outputs:
        errors.append(
            "landing_page service is not configured; check Atlas DB initialization "
            "and pipeline LLM/OpenRouter credentials"
        )
        return errors
    if result is None:
        return errors
    if result.get("status") != "completed":
        errors.append(f"execution status was {result.get('status')!r}, not 'completed'")
    step = _step_result(result, "landing_page")
    if step is None:
        errors.append("execution result did not include a landing_page step")
        return errors
    if step.get("status") != "completed":
        errors.append(f"landing_page step status was {step.get('status')!r}")
    step_payload = step.get("result") if isinstance(step.get("result"), Mapping) else {}
    if not step_payload.get("saved_ids"):
        errors.append("landing_page generation did not return saved draft ids")
    history = step_payload.get("quality_repair_history") or ()
    if history and isinstance(history[-1], Mapping) and not history[-1].get("passed"):
        errors.append("landing_page quality gate did not pass")
    return errors


async def run_content_ops_live_generation_smoke(
    args: argparse.Namespace,
    *,
    init_database_fn: AsyncCallable | None = None,
    close_database_fn: AsyncCallable | None = None,
    services_factory: ServicesFactory | None = None,
    executor: Executor | None = None,
    tenant_scope_cls: Any = None,
) -> tuple[int, dict[str, Any]]:
    _load_dotenv_files(list(args.env_file or []))
    if (
        init_database_fn is None
        or close_database_fn is None
        or services_factory is None
        or executor is None
        or tenant_scope_cls is None
    ):
        (
            init_database_fn,
            close_database_fn,
            services_factory,
            executor,
            tenant_scope_cls,
        ) = _resolve_runtime_dependencies()

    payload = _payload_from_args(args)
    configured_outputs: tuple[str, ...] = ()
    execution_result: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        await init_database_fn()
        services = services_factory()
        configured_outputs = tuple(str(item) for item in services.configured_outputs())
        errors.extend(
            _smoke_errors(configured_outputs=configured_outputs, result=None)
        )
        if not errors:
            scope = tenant_scope_cls(
                account_id=str(args.account_id or "").strip(),
                user_id=str(args.user_id or "").strip(),
            )
            execution_result = await executor(payload, services=services, scope=scope)
            errors.extend(
                _smoke_errors(
                    configured_outputs=configured_outputs,
                    result=execution_result,
                )
            )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            await close_database_fn()
        except Exception as exc:
            errors.append(f"close_database failed: {type(exc).__name__}: {exc}")

    result = {
        "ok": not errors,
        "errors": errors,
        "configured_outputs": list(configured_outputs),
        "payload": payload,
        "execution": execution_result,
    }
    return (0 if result["ok"] else 1), result


def _print_result(result: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    status = "passed" if result.get("ok") else "failed"
    print(f"Content Ops live generation smoke {status}")
    print(f"configured_outputs: {', '.join(result.get('configured_outputs') or [])}")
    for error in result.get("errors") or ():
        print(f"error: {error}")
    execution = result.get("execution")
    if isinstance(execution, Mapping):
        step = _step_result(execution, "landing_page")
        step_payload = step.get("result") if isinstance(step, Mapping) else {}
        if isinstance(step_payload, Mapping):
            saved_ids = step_payload.get("saved_ids") or []
            if saved_ids:
                print(f"saved_ids: {', '.join(str(item) for item in saved_ids)}")


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    code, result = await run_content_ops_live_generation_smoke(args)
    if args.output_result:
        args.output_result.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    _print_result(result, as_json=bool(args.json))
    return code


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
