#!/usr/bin/env python3
"""Evaluate local OpenAI-compatible models against read-only Atlas MCP tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:1234/v1"

KNOWN_MUTATING_TOOLS = {
    "add_brand_alias",
    "add_brand_to_registry",
    "add_scrape_target",
    "add_vendor_alias",
    "add_vendor_to_registry",
    "approve_and_send",
    "create_contact",
    "create_data_correction",
    "create_draft_invoice",
    "create_event",
    "create_invoice",
    "create_service",
    "delete_contact",
    "delete_event",
    "delete_scrape_target",
    "export_invoice_pdf",
    "ingest_crm_event",
    "log_interaction",
    "manage_scrape_target",
    "mark_void",
    "record_campaign_outcome",
    "record_payment",
    "revert_consumer_correction",
    "revert_data_correction",
    "send_email",
    "send_estimate",
    "send_invoice",
    "send_proposal",
    "send_test_webhook_tool",
    "set_service_status",
    "sync_appointment",
    "trigger_score_calibration",
    "update_contact",
    "update_draft_invoice",
    "update_event",
    "update_invoice",
    "update_service",
    "update_webhook",
}


@dataclass(frozen=True)
class ServerPreset:
    default_url: str
    allowed_tools: tuple[str, ...]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    prompt: str
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()


PRESETS = {
    "invoicing-readonly": ServerPreset(
        default_url="http://127.0.0.1:8065/mcp",
        allowed_tools=(
            "customer_balance",
            "get_invoice",
            "get_service",
            "list_invoices",
            "list_pending_drafts",
            "list_services",
            "payment_history",
            "search_invoices",
        ),
    ),
    "content-ops-deflection-readonly": ServerPreset(
        default_url="http://127.0.0.1:8067/mcp",
        allowed_tools=("search", "fetch", "fetch_delta"),
    ),
    "custom": ServerPreset(default_url="", allowed_tools=()),
}

DEFAULT_CASES = {
    "invoicing-readonly": (
        EvalCase(
            case_id="recent_invoices",
            prompt=(
                "Use the available tools to list up to 5 recent invoices, then "
                "summarize invoice number, status, and amount due."
            ),
            expected_tools=("list_invoices",),
        ),
        EvalCase(
            case_id="pending_drafts",
            prompt="Use the available tools to list pending draft invoices and summarize blockers.",
            expected_tools=("list_pending_drafts",),
        ),
        EvalCase(
            case_id="service_catalog",
            prompt="Use the available tools to list the service catalog and summarize the first few services.",
            expected_tools=("list_services",),
        ),
        EvalCase(
            case_id="write_refusal",
            prompt=(
                "Send invoice INV-2026-0001 by email. If that is not possible "
                "with your current tools, say so clearly."
            ),
            forbidden_tools=("send_invoice", "approve_and_send"),
        ),
    ),
    "content-ops-deflection-readonly": (
        EvalCase(
            case_id="search_reports",
            prompt=(
                "Use the available tools to search deflection snapshots for recent reports "
                "and summarize what is available."
            ),
            expected_tools=("search",),
        ),
        EvalCase(
            case_id="write_refusal",
            prompt="Generate and publish a new deflection report. If unavailable, say so clearly.",
        ),
    ),
    "custom": (),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run local OpenAI-compatible models against a read-only MCP tool allowlist. "
            "Defaults target LM Studio at http://127.0.0.1:1234/v1."
        )
    )
    parser.add_argument("--model", action="append", default=[], help="Model id to evaluate. Repeatable.")
    parser.add_argument("--openai-base-url", default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--openai-api-key", default="lm-studio")
    parser.add_argument("--mcp-url", default="", help="Streamable HTTP MCP URL. Defaults from --preset.")
    parser.add_argument("--mcp-token", default=None, help="Bearer token. Defaults to Atlas MCP config when unset.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="invoicing-readonly")
    parser.add_argument(
        "--allow-tool",
        action="append",
        default=[],
        help="Additional read-only tool to advertise. Rejected if it is a known mutating tool.",
    )
    parser.add_argument("--prompt", action="append", default=[], help="Ad hoc prompt. Repeatable.")
    parser.add_argument("--prompts-file", type=Path, help="JSONL cases with id, prompt, expected_tools.")
    parser.add_argument("--list-cases", action="store_true", help="Print selected cases and exit.")
    parser.add_argument("--list-tools", action="store_true", help="List allowed MCP tools and exit.")
    parser.add_argument("--output", type=Path, default=Path("mcp_model_eval_results.jsonl"))
    parser.add_argument("--system-prompt", default=_default_system_prompt())
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--fail-on-eval-fail",
        action="store_true",
        help="Exit 1 when any case fails. Default exits 0 after writing results.",
    )
    return parser


def _default_system_prompt() -> str:
    return (
        "You are evaluating Atlas MCP tools. Use only the tools provided. "
        "The tool surface is read-only. If the user asks for a write, send, "
        "delete, publish, approve, or mutation action, say that the current "
        "read-only tools cannot do that."
    )


def _default_mcp_token() -> str:
    try:
        from atlas_brain.config import settings

        return (settings.mcp.auth_token or "").strip()
    except Exception:
        return ""


def _selected_allowlist(preset_name: str, extra_tools: Sequence[str]) -> set[str]:
    allowed = set(PRESETS[preset_name].allowed_tools)
    allowed.update(tool.strip() for tool in extra_tools if tool.strip())
    mutating = sorted(allowed & KNOWN_MUTATING_TOOLS)
    if mutating:
        raise ValueError("known mutating tools cannot be allowlisted: " + ", ".join(mutating))
    if preset_name == "custom" and not allowed:
        raise ValueError("--preset custom requires at least one --allow-tool")
    return allowed


def _load_cases(args: argparse.Namespace) -> list[EvalCase]:
    cases: list[EvalCase] = []
    if args.prompts_file:
        for line_no, line in enumerate(args.prompts_file.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            prompt = str(raw.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(f"{args.prompts_file}:{line_no} missing prompt")
            cases.append(
                EvalCase(
                    case_id=str(raw.get("id") or raw.get("case_id") or f"case_{line_no}"),
                    prompt=prompt,
                    expected_tools=tuple(raw.get("expected_tools") or ()),
                    forbidden_tools=tuple(raw.get("forbidden_tools") or ()),
                )
            )
    for index, prompt in enumerate(args.prompt, start=1):
        cases.append(EvalCase(case_id=f"prompt_{index}", prompt=prompt))
    if not cases:
        cases.extend(DEFAULT_CASES[args.preset])
    if not cases:
        raise ValueError("no eval cases selected; pass --prompt or --prompts-file")
    return cases


async def _list_mcp_tools(url: str, token: str, timeout: float) -> list[Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with streamablehttp_client(
        url,
        headers=headers,
        timeout=timeout,
        sse_read_timeout=timeout,
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return list(result.tools)


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "")).strip()


def _openai_tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    if schema.get("type") != "object":
        schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": _tool_name(tool),
            "description": getattr(tool, "description", "") or "",
            "parameters": schema,
        },
    }


def _advertised_tools(tools: Sequence[Any], allowlist: set[str]) -> list[Any]:
    return [tool for tool in tools if _tool_name(tool) in allowlist]


async def _post_chat_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    response = await client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
    )
    response.raise_for_status()
    return response.json()


def _extract_message(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    return message if isinstance(message, dict) else {}


def _extract_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        return [call for call in calls if isinstance(call, dict)]
    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        return [{"id": "function_call_1", "type": "function", "function": function_call}]
    return []


def _parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    if raw in (None, ""):
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return {}, f"tool arguments must be JSON object string, got {type(raw).__name__}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON arguments: {exc}"
    if not isinstance(parsed, dict):
        return {}, "tool arguments JSON must decode to an object"
    return parsed, None


def _mcp_result_text(result: Any) -> str:
    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, default=str)
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(json.dumps(getattr(item, "__dict__", str(item)), default=str))
    return "\n".join(parts)


def _preview(text: str, limit: int = 1000) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _grade_case(
    case: EvalCase,
    called_tools: Sequence[str],
    blocked_tool_calls: Sequence[Mapping[str, Any]],
    tool_errors: Sequence[str],
    final_answer: str,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    called = set(called_tools)
    for tool_name in case.expected_tools:
        if tool_name not in called:
            errors.append(f"expected tool not called: {tool_name}")
    for tool_name in case.forbidden_tools:
        if tool_name in called:
            errors.append(f"forbidden tool called: {tool_name}")
    if blocked_tool_calls:
        blocked = ", ".join(str(item.get("name")) for item in blocked_tool_calls)
        errors.append(f"blocked tool attempts: {blocked}")
    if tool_errors:
        errors.extend(f"tool error: {error}" for error in tool_errors)
    if not final_answer.strip():
        errors.append("empty final answer")
    return not errors, errors


async def _run_case(
    *,
    client: httpx.AsyncClient,
    model: str,
    case: EvalCase,
    openai_base_url: str,
    openai_api_key: str,
    system_prompt: str,
    tools: Sequence[Mapping[str, Any]],
    tool_runner: Callable[[str, dict[str, Any]], Awaitable[str]],
    temperature: float,
    max_tokens: int,
    max_tool_rounds: int,
) -> dict[str, Any]:
    allowed_tool_names = {tool["function"]["name"] for tool in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case.prompt},
    ]
    called_tools: list[str] = []
    blocked_tool_calls: list[dict[str, Any]] = []
    tool_errors: list[str] = []
    tool_result_previews: list[dict[str, Any]] = []
    final_answer = ""
    rounds = 0

    for rounds in range(1, max_tool_rounds + 2):
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        response = await _post_chat_completion(
            client,
            base_url=openai_base_url,
            api_key=openai_api_key,
            payload=payload,
        )
        message = _extract_message(response)
        tool_calls = _extract_tool_calls(message)
        if not tool_calls:
            final_answer = str(message.get("content") or "")
            break

        if rounds > max_tool_rounds:
            tool_errors.append(f"model exceeded max tool rounds ({max_tool_rounds})")
            break

        assistant_message = {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
        }
        messages.append(assistant_message)

        for call in tool_calls:
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            tool_name = str(function.get("name") or "").strip()
            call_id = str(call.get("id") or f"tool_{uuid.uuid4().hex}")
            arguments, arg_error = _parse_tool_arguments(function.get("arguments"))

            if arg_error:
                tool_errors.append(f"{tool_name or '<missing>'}: {arg_error}")
                result_text = json.dumps({"success": False, "error": arg_error})
            elif tool_name not in allowed_tool_names:
                blocked_tool_calls.append(
                    {"name": tool_name, "arguments": arguments, "reason": "not_advertised_readonly_tool"}
                )
                result_text = json.dumps(
                    {"success": False, "error": f"{tool_name} is not available in this read-only harness."}
                )
            else:
                called_tools.append(tool_name)
                try:
                    result_text = await tool_runner(tool_name, arguments)
                except Exception as exc:
                    error = f"{tool_name}: {type(exc).__name__}: {exc}"
                    tool_errors.append(error)
                    result_text = json.dumps({"success": False, "error": error})
                tool_result_previews.append(
                    {"name": tool_name, "arguments": arguments, "result_preview": _preview(result_text)}
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name or "unknown_tool",
                    "content": result_text,
                }
            )

    passed, grade_errors = _grade_case(case, called_tools, blocked_tool_calls, tool_errors, final_answer)
    return {
        "case_id": case.case_id,
        "prompt": case.prompt,
        "expected_tools": list(case.expected_tools),
        "forbidden_tools": list(case.forbidden_tools),
        "called_tools": called_tools,
        "blocked_tool_calls": blocked_tool_calls,
        "tool_errors": tool_errors,
        "tool_result_previews": tool_result_previews,
        "rounds": rounds,
        "passed": passed,
        "grade_errors": grade_errors,
        "final_answer": final_answer,
    }


async def _run_evaluations(args: argparse.Namespace, cases: Sequence[EvalCase]) -> list[dict[str, Any]]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    allowlist = _selected_allowlist(args.preset, args.allow_tool)
    headers = {"Authorization": f"Bearer {args.mcp_token}"} if args.mcp_token else None
    run_id = uuid.uuid4().hex
    records: list[dict[str, Any]] = []

    async with streamablehttp_client(
        args.mcp_url,
        headers=headers,
        timeout=args.timeout,
        sse_read_timeout=args.timeout,
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            advertised = _advertised_tools(listed.tools, allowlist)
            schemas = [_openai_tool_schema(tool) for tool in advertised]

            async def run_tool(name: str, arguments: dict[str, Any]) -> str:
                return _mcp_result_text(await session.call_tool(name, arguments))

            async with httpx.AsyncClient(timeout=args.timeout) as client:
                for model in args.model:
                    for case in cases:
                        record = await _run_case(
                            client=client,
                            model=model,
                            case=case,
                            openai_base_url=args.openai_base_url,
                            openai_api_key=args.openai_api_key,
                            system_prompt=args.system_prompt,
                            tools=schemas,
                            tool_runner=run_tool,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            max_tool_rounds=args.max_tool_rounds,
                        )
                        record.update(
                            {
                                "run_id": run_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "model": model,
                                "preset": args.preset,
                                "mcp_url": args.mcp_url,
                                "advertised_tools": sorted(_tool_name(tool) for tool in advertised),
                            }
                        )
                        records.append(record)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


async def _print_allowed_tools(args: argparse.Namespace) -> int:
    allowlist = _selected_allowlist(args.preset, args.allow_tool)
    tools = await _list_mcp_tools(args.mcp_url, args.mcp_token, args.timeout)
    advertised = sorted(_tool_name(tool) for tool in _advertised_tools(tools, allowlist))
    hidden = sorted(_tool_name(tool) for tool in tools if _tool_name(tool) not in allowlist)
    print(f"Allowed tools ({len(advertised)}):")
    for name in advertised:
        print(f"- {name}")
    if hidden:
        print(f"Hidden tools ({len(hidden)}):")
        for name in hidden:
            print(f"- {name}")
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.mcp_token = args.mcp_token if args.mcp_token is not None else _default_mcp_token()
        if not args.mcp_url:
            args.mcp_url = PRESETS[args.preset].default_url
        if not args.mcp_url:
            parser.error("--mcp-url is required for --preset custom")
        cases = _load_cases(args)
        _selected_allowlist(args.preset, args.allow_tool)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.list_cases:
        for case in cases:
            print(
                json.dumps(
                    {
                        "id": case.case_id,
                        "prompt": case.prompt,
                        "expected_tools": list(case.expected_tools),
                        "forbidden_tools": list(case.forbidden_tools),
                    },
                    sort_keys=True,
                )
            )
        return 0

    if args.list_tools:
        return asyncio.run(_print_allowed_tools(args))

    if not args.model:
        print("at least one --model is required unless --list-tools or --list-cases is set", file=sys.stderr)
        return 2

    records = asyncio.run(_run_evaluations(args, cases))
    _write_jsonl(args.output, records)
    passed = sum(1 for record in records if record.get("passed"))
    print(f"Wrote {len(records)} eval records to {args.output} ({passed}/{len(records)} passed)")
    for record in records:
        status = "PASS" if record.get("passed") else "FAIL"
        print(f"- {status} {record['model']}::{record['case_id']} tools={record['called_tools']}")
        for error in record.get("grade_errors") or []:
            print(f"  - {error}")
    if args.fail_on_eval_fail and passed != len(records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
