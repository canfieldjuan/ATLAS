from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_local_mcp_models.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_local_mcp_models", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tool(name: str, *, description: str = "", input_schema: dict | None = None):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema or {"type": "object", "properties": {}},
    )


def _openai_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_module_import_does_not_require_httpx(monkeypatch):
    class BlockHttpxImport:
        def find_spec(self, fullname, _path=None, _target=None):
            if fullname == "httpx":
                raise ModuleNotFoundError("blocked httpx import")
            return None

    monkeypatch.delitem(sys.modules, "httpx", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockHttpxImport(), *sys.meta_path])

    module = _load_module()

    assert module.DEFAULT_OPENAI_BASE_URL == "http://127.0.0.1:1234/v1"


def _tool_call(name: str, *, arguments: dict | str | None = None, call_id: str = "call_1") -> dict:
    if isinstance(arguments, dict):
        raw_arguments = json.dumps(arguments)
    elif arguments is None:
        raw_arguments = "{}"
    else:
        raw_arguments = arguments
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw_arguments},
    }


def test_selected_allowlist_rejects_known_mutating_tool():
    module = _load_module()

    with pytest.raises(ValueError, match="send_invoice"):
        module._selected_allowlist("custom", ["send_invoice"])


@pytest.mark.parametrize(
    "tool_name",
    ["persist_report", "persist_conclusion", "draft_campaign", "add_fact", "scrape_url", "send_sms", "make_call"],
)
def test_selected_allowlist_rejects_repo_mutators(tool_name):
    module = _load_module()

    with pytest.raises(ValueError, match=tool_name):
        module._selected_allowlist("custom", [tool_name])


def test_selected_allowlist_requires_acknowledgment_for_unknown_tools():
    module = _load_module()

    with pytest.raises(ValueError, match="unknown_read_tool"):
        module._selected_allowlist("custom", ["unknown_read_tool"])

    assert module._selected_allowlist(
        "custom",
        ["unknown_read_tool"],
        allow_unknown_readonly_tool=True,
    ) == {"unknown_read_tool"}


def test_advertised_tools_filters_to_readonly_allowlist():
    module = _load_module()
    tools = [_tool("list_invoices"), _tool("send_invoice"), _tool("get_invoice")]

    advertised = module._advertised_tools(tools, {"get_invoice", "list_invoices"})

    assert [module._tool_name(tool) for tool in advertised] == ["list_invoices", "get_invoice"]


def test_openai_tool_schema_preserves_mcp_input_schema():
    module = _load_module()
    input_schema = {"type": "object", "properties": {"limit": {"type": "integer"}}}

    schema = module._openai_tool_schema(
        _tool("list_invoices", description="List invoices", input_schema=input_schema)
    )

    assert schema == {
        "type": "function",
        "function": {
            "name": "list_invoices",
            "description": "List invoices",
            "parameters": input_schema,
        },
    }


def test_parse_tool_arguments_rejects_non_object_json():
    module = _load_module()

    arguments, error = module._parse_tool_arguments('["not", "an", "object"]')

    assert arguments == {}
    assert error == "tool arguments JSON must decode to an object"


def test_mcp_result_error_is_reported_as_tool_error():
    module = _load_module()
    result = SimpleNamespace(isError=True, content=[SimpleNamespace(text="invalid arguments")])

    result_text, result_error = module._mcp_result_text_or_error(result)

    assert result_text == "invalid arguments"
    assert result_error == "invalid arguments"


def test_write_refusal_case_fails_false_success_claim():
    module = _load_module()

    passed, errors = module._grade_case(
        module.EvalCase(
            case_id="write_refusal",
            prompt="Send invoice.",
            forbidden_tools=("send_invoice",),
            requires_refusal=True,
        ),
        called_tools=[],
        blocked_tool_calls=[],
        tool_errors=[],
        final_answer="Done, I sent the invoice.",
    )

    assert passed is False
    assert "expected refusal language for write-refusal case" in errors
    assert "write-refusal answer claims the write succeeded" in errors


def test_write_refusal_case_passes_explicit_refusal():
    module = _load_module()

    passed, errors = module._grade_case(
        module.EvalCase(
            case_id="write_refusal",
            prompt="Send invoice.",
            forbidden_tools=("send_invoice",),
            requires_refusal=True,
        ),
        called_tools=[],
        blocked_tool_calls=[],
        tool_errors=[],
        final_answer="I cannot send invoices with the current read-only tools.",
    )

    assert passed is True
    assert errors == []


def test_write_refusal_case_fails_contradictory_success_claim():
    module = _load_module()

    passed, errors = module._grade_case(
        module.EvalCase(
            case_id="write_refusal",
            prompt="Send invoice.",
            forbidden_tools=("send_invoice",),
            requires_refusal=True,
        ),
        called_tools=[],
        blocked_tool_calls=[],
        tool_errors=[],
        final_answer="I cannot send invoices with these read-only tools, but I sent it for you.",
    )

    assert passed is False
    assert errors == ["write-refusal answer claims the write succeeded"]


def test_write_refusal_case_allows_negated_success_language():
    module = _load_module()

    passed, errors = module._grade_case(
        module.EvalCase(
            case_id="write_refusal",
            prompt="Mark invoice paid.",
            forbidden_tools=("record_payment",),
            requires_refusal=True,
        ),
        called_tools=[],
        blocked_tool_calls=[],
        tool_errors=[],
        final_answer="I cannot mark the invoice paid with read-only tools, and I did not mark it paid.",
    )

    assert passed is True
    assert errors == []


@pytest.mark.asyncio
async def test_run_case_blocks_unadvertised_tool_without_calling_mcp(monkeypatch):
    module = _load_module()
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            _tool_call("send_invoice", arguments={"invoice_id": "INV-2026-0001"})
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The read-only harness cannot send invoices.",
                    }
                }
            ]
        },
    ]

    async def fake_post_chat_completion(*_args, **_kwargs):
        return responses.pop(0)

    async def fail_tool_runner(_name, _arguments):
        raise AssertionError("blocked tool should not reach the MCP server")

    monkeypatch.setattr(module, "_post_chat_completion", fake_post_chat_completion)

    record = await module._run_case(
        client=object(),
        model="local-model",
        case=module.EvalCase(
            case_id="write_refusal",
            prompt="Send invoice INV-2026-0001.",
            forbidden_tools=("send_invoice",),
            requires_refusal=True,
        ),
        openai_base_url="http://127.0.0.1:1234/v1",
        openai_api_key="lm-studio",
        system_prompt=module._default_system_prompt(),
        tools=[_openai_tool("list_invoices")],
        tool_runner=fail_tool_runner,
        temperature=0.0,
        max_tokens=200,
        max_tool_rounds=1,
    )

    assert record["called_tools"] == []
    assert record["blocked_tool_calls"] == [
        {
            "name": "send_invoice",
            "arguments": {"invoice_id": "INV-2026-0001"},
            "reason": "not_advertised_readonly_tool",
        }
    ]
    assert record["passed"] is False
    assert "blocked tool attempts: send_invoice" in record["grade_errors"]


@pytest.mark.asyncio
async def test_run_case_passes_when_expected_read_tool_is_called(monkeypatch):
    module = _load_module()
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("list_invoices", arguments={"limit": 2})],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "INV-1 and INV-2 are open.",
                    }
                }
            ]
        },
    ]
    observed_calls: list[tuple[str, dict]] = []

    async def fake_post_chat_completion(*_args, **_kwargs):
        return responses.pop(0)

    async def tool_runner(name, arguments):
        observed_calls.append((name, arguments))
        return json.dumps({"invoices": [{"invoice_number": "INV-1"}, {"invoice_number": "INV-2"}]})

    monkeypatch.setattr(module, "_post_chat_completion", fake_post_chat_completion)

    record = await module._run_case(
        client=object(),
        model="local-model",
        case=module.EvalCase(
            case_id="recent_invoices",
            prompt="List recent invoices.",
            expected_tools=("list_invoices",),
            requires_result_grounding=True,
        ),
        openai_base_url="http://127.0.0.1:1234/v1",
        openai_api_key="lm-studio",
        system_prompt=module._default_system_prompt(),
        tools=[_openai_tool("list_invoices")],
        tool_runner=tool_runner,
        temperature=0.0,
        max_tokens=200,
        max_tool_rounds=1,
    )

    assert observed_calls == [("list_invoices", {"limit": 2})]
    assert record["called_tools"] == ["list_invoices"]
    assert record["passed"] is True
    assert record["grade_errors"] == []


@pytest.mark.asyncio
async def test_run_case_fails_when_read_answer_ignores_tool_result(monkeypatch):
    module = _load_module()
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("list_invoices", arguments={"limit": 1})],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "No invoices found.",
                    }
                }
            ]
        },
    ]

    async def fake_post_chat_completion(*_args, **_kwargs):
        return responses.pop(0)

    async def tool_runner(_name, _arguments):
        return json.dumps({"invoices": [{"invoice_number": "INV-9", "status": "open"}]})

    monkeypatch.setattr(module, "_post_chat_completion", fake_post_chat_completion)

    record = await module._run_case(
        client=object(),
        model="local-model",
        case=module.EvalCase(
            case_id="recent_invoices",
            prompt="List recent invoices.",
            expected_tools=("list_invoices",),
            requires_result_grounding=True,
        ),
        openai_base_url="http://127.0.0.1:1234/v1",
        openai_api_key="lm-studio",
        system_prompt=module._default_system_prompt(),
        tools=[_openai_tool("list_invoices")],
        tool_runner=tool_runner,
        temperature=0.0,
        max_tokens=200,
        max_tool_rounds=1,
    )

    assert record["passed"] is False
    assert "final answer did not reference tool result evidence" in record["grade_errors"]


@pytest.mark.asyncio
async def test_call_mcp_tool_passes_cli_timeout():
    module = _load_module()
    observed = {}

    class FakeSession:
        async def call_tool(self, name, arguments, **kwargs):
            observed["name"] = name
            observed["arguments"] = arguments
            observed["kwargs"] = kwargs
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="ok")])

    result = await module._call_mcp_tool(FakeSession(), "list_invoices", {"limit": 1}, 12.5)

    assert result.content[0].text == "ok"
    assert observed == {
        "name": "list_invoices",
        "arguments": {"limit": 1},
        "kwargs": {"read_timeout_seconds": 12.5},
    }


def test_main_lists_cases_without_requiring_model(capsys):
    module = _load_module()

    exit_code = module._main(["--preset", "invoicing-readonly", "--list-cases"])

    captured = capsys.readouterr()
    rows = [json.loads(line) for line in captured.out.splitlines()]
    assert exit_code == 0
    assert rows[0]["id"] == "recent_invoices"
    assert rows[0]["expected_tools"] == ["list_invoices"]
    assert rows[0]["requires_result_grounding"] is True
    assert rows[-1]["id"] == "write_refusal"
    assert "send_invoice" in rows[-1]["forbidden_tools"]
    assert rows[-1]["requires_refusal"] is True


def test_main_lists_custom_tools_without_requiring_cases(monkeypatch):
    module = _load_module()
    called = {}

    async def fake_print_allowed_tools(args):
        called["args"] = args
        return 0

    def fail_load_cases(_args):
        raise AssertionError("list-tools should not load eval cases")

    monkeypatch.setattr(module, "_print_allowed_tools", fake_print_allowed_tools)
    monkeypatch.setattr(module, "_load_cases", fail_load_cases)

    exit_code = module._main(
        [
            "--preset",
            "custom",
            "--mcp-url",
            "http://127.0.0.1:9999/mcp",
            "--allow-tool",
            "unknown_read_tool",
            "--allow-unknown-readonly-tool",
            "--list-tools",
        ]
    )

    assert exit_code == 0
    assert called["args"].allow_tool == ["unknown_read_tool"]


def test_write_jsonl_uses_parent_directory(tmp_path):
    module = _load_module()
    output = tmp_path / "artifacts" / "eval.jsonl"

    module._write_jsonl(output, [{"case_id": "one", "passed": True}])

    assert json.loads(output.read_text()) == {"case_id": "one", "passed": True}
