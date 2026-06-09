from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_content_ops_marketer_verify_dual_client_rollout.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_content_ops_marketer_verify_dual_client_rollout",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _argv(*extra: str) -> list[str]:
    return [
        "--rich-issuer-url",
        "https://atlas.example.com/content-ops-marketer/",
        "--rich-resource-url",
        "https://atlas.example.com/content-ops-marketer/mcp/",
        "--chatgpt-adapter-issuer-url",
        "https://atlas.example.com/content-ops-marketer-chatgpt/",
        "--chatgpt-adapter-resource-url",
        "https://atlas.example.com/content-ops-marketer-chatgpt/mcp/",
        *extra,
    ]


def test_main_runs_both_profiles_with_token_file(monkeypatch) -> None:
    module = _load_script_module()
    calls: list[list[str]] = []
    monkeypatch.setattr(module._oauth_e2e, "_main", lambda argv: calls.append(argv) or 0)

    result = module._main(_argv("--approval-token-file", "/tmp/token-file"))

    assert result == 0
    assert [call[call.index("--client-profile") + 1] for call in calls] == [
        module.CLAUDE_RICH_PROFILE,
        module.CHATGPT_SEARCH_FETCH_PROFILE,
    ]
    assert all("--approval-token-file" in call for call in calls)
    assert all("/tmp/token-file" in call for call in calls)
    assert calls[0][calls[0].index("--resource-url") + 1] != (
        calls[1][calls[1].index("--resource-url") + 1]
    )


def test_main_requires_values_before_smoke_invocation(monkeypatch, capsys) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module, "_run_invocations", lambda *_args: 99)

    result = module._main([])

    captured = capsys.readouterr()
    assert result == 2
    for flag in (
        "--rich-issuer-url",
        "--rich-resource-url",
        "--chatgpt-adapter-issuer-url",
        "--chatgpt-adapter-resource-url",
        "--approval-token-file or --approval-token",
    ):
        assert flag in captured.err


def test_main_rejects_identical_resource_urls_before_smoke_invocation(
    monkeypatch,
    capsys,
) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module, "_run_invocations", lambda *_args: 99)

    result = module._main(
        _argv(
            "--chatgpt-adapter-resource-url",
            "https://atlas.example.com/content-ops-marketer/mcp",
            "--approval-token-file",
            "/tmp/token-file",
        )
    )

    assert result == 2
    assert "resource URLs must be different" in capsys.readouterr().err


def test_main_does_not_print_literal_token(monkeypatch, capsys) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module._oauth_e2e, "_main", lambda _argv: 0)

    result = module._main(_argv("--approval-token", "literal-secret-token-value"))

    captured = capsys.readouterr()
    assert result == 0
    assert "dual-client rollout smoke completed" in captured.out
    assert "literal-secret-token-value" not in captured.out
    assert "literal-secret-token-value" not in captured.err


@pytest.mark.parametrize(
    ("results", "expected_calls", "profile"),
    [([1], 1, "claude-rich"), ([0, 1], 2, "chatgpt-search-fetch")],
)
def test_profile_failure_blocks_success(results, expected_calls, profile, capsys) -> None:
    module = _load_script_module()
    calls: list[list[str]] = []
    args = module._build_parser().parse_args(_argv("--approval-token-file", "/tmp/token-file"))

    def fake_checker(argv: list[str]) -> int:
        calls.append(argv)
        return results[len(calls) - 1]

    result = module._run_invocations(module._invocations_from_args(args), fake_checker)

    captured = capsys.readouterr()
    assert result == 1
    assert len(calls) == expected_calls
    assert f"failed for {profile}" in captured.err
    assert "dual-client rollout smoke completed" not in captured.out
