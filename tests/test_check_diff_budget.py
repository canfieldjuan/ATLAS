"""Tests for scripts/check_diff_budget.py (pure core + CLI offline mode)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_diff_budget",
    Path(__file__).resolve().parent.parent / "scripts" / "check_diff_budget.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

BUDGET = 400


class TestUnderBudget:
    def test_under_budget_passes(self):
        code, messages = mod.evaluate(200, "", BUDGET)
        assert code == 0
        assert "within" in messages[0]

    def test_exactly_at_budget_passes(self):
        code, _ = mod.evaluate(BUDGET, "", BUDGET)
        assert code == 0

    def test_zero_additions_passes(self):
        code, _ = mod.evaluate(0, "", BUDGET)
        assert code == 0

    def test_unneeded_override_noted_but_passes(self):
        body = "Diff-budget override: just in case"
        code, messages = mod.evaluate(100, body, BUDGET)
        assert code == 0
        assert any("not needed" in m for m in messages)


class TestOverBudgetWithoutOverride:
    def test_over_budget_no_marker_fails(self):
        code, messages = mod.evaluate(401, "regular PR body text", BUDGET)
        assert code == 1
        assert any("no override marker" in m for m in messages)
        # the failure message must teach the fix
        assert any("Diff-budget override:" in m for m in messages)

    def test_prose_mention_is_not_a_marker(self):
        # retroactive prose must NOT satisfy the gate -- the loophole closed
        body = "Over the 400 LOC soft target because tests dominate."
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    @pytest.mark.parametrize("reason", ["", "   ", "TODO", "tbd", "n/a", ".", "--"])
    def test_placeholder_reasons_fail(self, reason):
        body = f"Diff-budget override: {reason}"
        code, messages = mod.evaluate(500, body, BUDGET)
        assert code == 1
        assert any("no real" in m for m in messages)


class TestOverBudgetWithOverride:
    def test_reasoned_override_passes_and_echoes_reason(self):
        body = (
            "## Why\n\n"
            "Diff-budget override: state-machine slice needs both-sides "
            "dormancy probes; splitting would separate code from mandated tests."
        )
        code, messages = mod.evaluate(1200, body, BUDGET)
        assert code == 0
        assert any("override reason:" in m for m in messages)
        assert any("dormancy" in m for m in messages)

    @pytest.mark.parametrize(
        "line",
        [
            "diff-budget override: reason here",
            "**Diff-budget override:** reason here",
            "- Diff-budget override: reason here",
            "> Diff budget override: reason here",
        ],
    )
    def test_marker_format_variants_accepted(self, line):
        code, _ = mod.evaluate(500, line, BUDGET)
        assert code == 0

    def test_marker_must_be_line_anchored(self):
        # mid-sentence mention must not count as a marker
        body = "we discussed a diff-budget override: maybe later"
        code, _ = mod.evaluate(500, body, BUDGET)
        assert code == 1


class TestFetchGuards:
    def test_non_json_gh_output_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(mod, "_gh", lambda args, gh: "not json at all")
        with pytest.raises(RuntimeError, match="non-JSON"):
            mod.fetch_pr(1, "owner/repo", "gh")


class TestFindOverrideReason:
    def test_absent_returns_none(self):
        assert mod.find_override_reason("no marker here") is None

    def test_placeholder_returns_empty(self):
        assert mod.find_override_reason("Diff-budget override: TODO") == ""

    def test_reason_extracted(self):
        got = mod.find_override_reason("Diff-budget override: real reason")
        assert got == "real reason"


class TestCliOffline:
    def test_offline_under_budget_exit_0(self, tmp_path):
        assert mod.main(["--additions", "10"]) == 0

    def test_offline_over_budget_exit_1(self, tmp_path):
        body = tmp_path / "body.md"
        body.write_text("no marker", encoding="utf-8")
        assert mod.main(["--additions", "900", "--body-file", str(body)]) == 1

    def test_offline_override_exit_0(self, tmp_path):
        body = tmp_path / "body.md"
        body.write_text("Diff-budget override: indivisible", encoding="utf-8")
        assert mod.main(["--additions", "900", "--body-file", str(body)]) == 0

    def test_no_inputs_exit_2(self, monkeypatch):
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        assert mod.main([]) == 2

    def test_bad_budget_exit_2(self):
        assert mod.main(["--additions", "10", "--budget", "0"]) == 2

    def test_missing_body_file_exit_2(self):
        assert mod.main(["--additions", "900", "--body-file", "/nonexistent/x"]) == 2
