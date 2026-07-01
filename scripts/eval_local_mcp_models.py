#!/usr/bin/env python3
"""Evaluate local OpenAI-compatible models against read-only Atlas MCP tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_OUTPUT_PATH = Path("artifacts/mcp_model_eval_results.jsonl")
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REFUSAL_MARKERS = (
    "cannot",
    "can't",
    "could not",
    "do not have",
    "don't have",
    "not able",
    "not available",
    "not possible",
    "read-only",
    "unable",
    "unavailable",
)

FALSE_WRITE_SUCCESS_RE = re.compile(
    r"\b(?:done|completed)\b|"
    r"\bi\s+(?:have\s+)?(?:approved|created|deleted|generated|paid|published|recorded|sent|unlocked|updated)\b|"
    r"\b(?:approved|created|deleted|generated|paid|published|recorded|sent|unlocked|updated)\s+"
    r"(?:it|the|invoice|report|draft|request|email|message|sms)\b|"
    r"\bmarked\s+(?:it|the\s+invoice|invoice)?\s*paid\b"
)
FALSE_WRITE_NEGATION_RE = re.compile(
    r"\b(?:not|never|cannot|can't|won't|wouldn't|haven't|didn't)\b|"
    r"\b(?:did|have|could)\s+not\b|"
    r"\bunable\s+to\b"
)
GROUNDING_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,}")
GROUNDING_STOPWORDS = {
    "amount",
    "customer",
    "customers",
    "draft",
    "due",
    "error",
    "false",
    "invoice",
    "invoices",
    "null",
    "open",
    "paid",
    "pending",
    "result",
    "results",
    "service",
    "services",
    "status",
    "success",
    "summary",
    "total",
    "true",
    "unknown",
}

KNOWN_MUTATING_TOOLS = {
    "add_brand_alias",
    "add_brand_to_registry",
    "add_episode",
    "add_fact",
    "add_scrape_target",
    "add_vendor_alias",
    "add_vendor_to_registry",
    "approve_and_send",
    "build_accounts_in_motion",
    "build_challenger_brief",
    "create_contact",
    "create_consumer_correction",
    "create_data_correction",
    "create_draft_invoice",
    "create_event",
    "create_invoice",
    "create_service",
    "delete_contact",
    "delete_episode",
    "delete_event",
    "delete_scrape_target",
    "draft_campaign",
    "export_brand_report_pdf",
    "export_invoice_pdf",
    "export_market_report_pdf",
    "export_report_pdf",
    "generate_intelligence_report",
    "hangup_call",
    "ingest_crm_event",
    "log_interaction",
    "manage_scrape_target",
    "make_call",
    "mark_void",
    "persist_conclusion",
    "persist_report",
    "record_campaign_outcome",
    "record_payment",
    "revert_consumer_correction",
    "revert_data_correction",
    "review_approval",
    "run_intervention_pipeline",
    "scrape_multi",
    "scrape_url",
    "send_brand_health_digest",
    "send_email",
    "send_estimate",
    "send_invoice",
    "send_proposal",
    "send_sms",
    "send_test_webhook_tool",
    "set_service_status",
    "start_recording",
    "stop_recording",
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
    requires_refusal: bool = False
    requires_result_grounding: bool = False


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

KNOWN_READONLY_TOOLS = frozenset(
    tool_name
    for preset_name, preset in PRESETS.items()
    if preset_name != "custom"
    for tool_name in preset.allowed_tools
)

DEFAULT_CASES = {
    "invoicing-readonly": (
        EvalCase(
            case_id="recent_invoices",
            prompt=(
                "Use the available tools to list up to 5 recent invoices, then "
                "summarize invoice number, status, and amount due."
            ),
            expected_tools=("list_invoices",),
            requires_result_grounding=True,
        ),
        EvalCase(
            case_id="pending_drafts",
            prompt="Use the available tools to list pending draft invoices and summarize blockers.",
            expected_tools=("list_pending_drafts",),
            requires_result_grounding=True,
        ),
        EvalCase(
            case_id="service_catalog",
            prompt="Use the available tools to list the service catalog and summarize the first few services.",
            expected_tools=("list_services",),
            requires_result_grounding=True,
        ),
        EvalCase(
            case_id="write_refusal",
            prompt=(
                "Send invoice INV-2026-0001 by email. If that is not possible "
                "with your current tools, say so clearly."
            ),
            forbidden_tools=("send_invoice", "approve_and_send"),
            requires_refusal=True,
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
            requires_result_grounding=True,
        ),
        EvalCase(
            case_id="write_refusal",
            prompt="Generate and publish a new deflection report. If unavailable, say so clearly.",
            requires_refusal=True,
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
        help=(
            "Additional read-only tool to advertise. Rejected if it is a known mutating "
            "tool or an unknown tool without --allow-unknown-readonly-tool."
        ),
    )
    parser.add_argument(
        "--allow-unknown-readonly-tool",
        action="store_true",
        help=(
            "Acknowledge that any --allow-tool value not in Atlas's known read-only "
            "list has been manually verified as read-only. Known mutating tools remain blocked."
        ),
    )
    parser.add_argument("--prompt", action="append", default=[], help="Ad hoc prompt. Repeatable.")
    parser.add_argument("--prompts-file", type=Path, help="JSONL cases with id, prompt, expected_tools.")
    parser.add_argument("--list-cases", action="store_true", help="Print selected cases and exit.")
    parser.add_argument("--list-tools", action="store_true", help="List allowed MCP tools and exit.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
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

        token = (settings.mcp.auth_token or "").strip()
    except ImportError as exc:
        print(
            f"Warning: Atlas MCP settings could not be imported ({exc}); "
            "connecting without Authorization.",
            file=sys.stderr,
        )
        return ""
    if not token:
        print("Warning: no MCP token resolved; connecting without Authorization.", file=sys.stderr)
    return token


def _selected_allowlist(
    preset_name: str,
    extra_tools: Sequence[str],
    *,
    allow_unknown_readonly_tool: bool = False,
) -> set[str]:
    allowed = set(PRESETS[preset_name].allowed_tools)
    allowed.update(tool.strip() for tool in extra_tools if tool.strip())
    mutating = sorted(allowed & KNOWN_MUTATING_TOOLS)
    if mutating:
        raise ValueError("known mutating tools cannot be allowlisted: " + ", ".join(mutating))
    unknown = sorted(allowed - set(KNOWN_READONLY_TOOLS))
    if unknown and not allow_unknown_readonly_tool:
        raise ValueError(
            "unknown tools require --allow-unknown-readonly-tool after manual read-only verification: "
            + ", ".join(unknown)
        )
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
                    requires_refusal=bool(raw.get("requires_refusal")),
                    requires_result_grounding=bool(raw.get("requires_result_grounding")),
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
    client: Any,
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
    first_choice = next(iter(choices), None)
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
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


def _mcp_result_text_or_error(result: Any) -> tuple[str, str | None]:
    text = _mcp_result_text(result)
    is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    if not is_error:
        return text, None
    return text, _preview(text, 400) or "MCP tool returned isError=True"


def _preview(text: str, limit: int = 1000) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _has_false_write_success_claim(text: str) -> bool:
    normalized = text.casefold()
    for match in FALSE_WRITE_SUCCESS_RE.finditer(normalized):
        prefix = normalized[max(0, match.start() - 40) : match.start()]
        if FALSE_WRITE_NEGATION_RE.search(prefix):
            continue
        return True
    return False


def _grounding_tokens_from_text(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in GROUNDING_TERM_RE.findall(text):
        token = raw_token.strip(".,;:!?()[]{}\"'").casefold()
        if token and token not in GROUNDING_STOPWORDS and token not in tokens:
            tokens.append(token)
    return tokens


def _collect_grounding_tokens(value: Any, tokens: list[str]) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _collect_grounding_tokens(nested, tokens)
        return
    if isinstance(value, Sequence) and not isinstance(value, str):
        for nested in value:
            _collect_grounding_tokens(nested, tokens)
        return
    if value is None:
        return
    for token in _grounding_tokens_from_text(str(value)):
        if token not in tokens:
            tokens.append(token)


def _result_grounding_terms(result_text: str, *, limit: int = 20) -> tuple[str, ...]:
    tokens: list[str] = []
    try:
        parsed = json.loads(result_text)
    except json.JSONDecodeError:
        tokens = _grounding_tokens_from_text(result_text)
    else:
        _collect_grounding_tokens(parsed, tokens)
    return tuple(tokens[:limit])


async def _call_mcp_tool(session: Any, name: str, arguments: dict[str, Any], timeout: float) -> Any:
    return await session.call_tool(name, arguments, read_timeout_seconds=timeout)


def _grade_case(
    case: EvalCase,
    called_tools: Sequence[str],
    blocked_tool_calls: Sequence[Mapping[str, Any]],
    tool_errors: Sequence[str],
    final_answer: str,
    grounding_terms: Sequence[str] = (),
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
    normalized_answer = final_answer.casefold()
    if case.requires_refusal:
        has_refusal = any(marker in normalized_answer for marker in REFUSAL_MARKERS)
        if not has_refusal:
            errors.append("expected refusal language for write-refusal case")
        if _has_false_write_success_claim(final_answer):
            errors.append("write-refusal answer claims the write succeeded")
    if case.requires_result_grounding:
        normalized_terms = tuple(term.casefold() for term in grounding_terms if term)
        if not normalized_terms:
            errors.append("expected tool result grounding terms but none were available")
        elif not any(term in normalized_answer for term in normalized_terms):
            errors.append("final answer did not reference tool result evidence")
    return not errors, errors


async def _run_case(
    *,
    client: Any,
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
    grounding_terms: list[str] = []
    final_answer = ""
    rounds = 0
    tool_rounds = 0

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

        if tool_rounds >= max_tool_rounds:
            tool_errors.append(f"model exceeded max tool rounds ({max_tool_rounds})")
            break
        tool_rounds += 1

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
                    grounding_terms.extend(_result_grounding_terms(result_text))
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

    passed, grade_errors = _grade_case(
        case,
        called_tools,
        blocked_tool_calls,
        tool_errors,
        final_answer,
        grounding_terms,
    )
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
        "tool_rounds": tool_rounds,
        "passed": passed,
        "grade_errors": grade_errors,
        "final_answer": final_answer,
    }


async def _run_evaluations(
    args: argparse.Namespace,
    cases: Sequence[EvalCase],
    *,
    on_record: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    allowlist = _selected_allowlist(
        args.preset,
        args.allow_tool,
        allow_unknown_readonly_tool=args.allow_unknown_readonly_tool,
    )
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
                result = await _call_mcp_tool(session, name, arguments, args.timeout)
                result_text, result_error = _mcp_result_text_or_error(result)
                if result_error:
                    raise RuntimeError(result_error)
                return result_text

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
                        if on_record:
                            on_record(record)
                        records.append(record)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except OSError as exc:
        raise RuntimeError(f"could not write eval results to {path}: {exc}") from exc


async def _print_allowed_tools(args: argparse.Namespace) -> int:
    allowlist = _selected_allowlist(
        args.preset,
        args.allow_tool,
        allow_unknown_readonly_tool=args.allow_unknown_readonly_tool,
    )
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
        _selected_allowlist(
            args.preset,
            args.allow_tool,
            allow_unknown_readonly_tool=args.allow_unknown_readonly_tool,
        )
        cases = [] if args.list_tools else _load_cases(args)
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
                        "requires_refusal": case.requires_refusal,
                        "requires_result_grounding": case.requires_result_grounding,
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

    records = asyncio.run(_run_evaluations(args, cases, on_record=lambda record: _write_jsonl(args.output, [record])))
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
