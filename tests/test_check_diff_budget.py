"""Tests for scripts/check_diff_budget.py (pure core + CLI offline mode)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_diff_budget",
    Path(__file__).resolve().parent.parent / "scripts" / "check_diff_budget.py",
)
assert _SPEC is not None and _SPEC.loader is not None, "gate script not found"
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

    @pytest.mark.parametrize(
        "reason", ["", "   ", "TODO", "tbd", "n/a", ".", "--", "!!!", "???"]
    )
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
        ],
    )
    def test_marker_format_variants_accepted(self, line):
        code, _ = mod.evaluate(500, line, BUDGET)
        assert code == 0

    def test_blockquoted_marker_is_not_a_decision(self):
        # a quote is quoting text (e.g. the gate's failure message)
        code, _ = mod.evaluate(500, "> Diff-budget override: real-looking", BUDGET)
        assert code == 1

    def test_blockquoted_fence_marker_is_ignored(self):
        body = "> ```\n> Diff-budget override: quoted example\n> ```\n"
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_list_nested_fence_marker_is_ignored(self):
        body = "- ```\n  Diff-budget override: listed example\n- ```\n"
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_html_comment_hidden_marker_is_ignored(self):
        # invisible in the rendered body = a decision no reviewer can see
        body = "<!--\nDiff-budget override: hidden reason\n-->\n"
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_real_marker_after_html_comment_still_honored(self):
        body = ("<!-- reviewed -->\n"
                "Diff-budget override: visible reasoned decision")
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 0

    def test_four_backtick_fence_containing_triple_is_one_block(self):
        # inner ``` must not close a ```` fence (delimiter length tracked)
        body = ("````\nexample\n```\nDiff-budget override: still inside\n````\n")
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_real_marker_after_four_backtick_sample_still_honored(self):
        body = ("````\n```\n````\n"
                "Diff-budget override: real decision after the sample")
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 0

    def test_ordered_list_fence_marker_is_ignored(self):
        body = "1. ```\n   Diff-budget override: example syntax only\n   ```\n"
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_tilde_fence_not_closed_by_backticks(self):
        body = "~~~\n```\nDiff-budget override: inside tilde fence\n~~~\n"
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_fake_close_with_trailing_text_stays_inside_fence(self):
        body = "```\n``` not a real close\nDiff-budget override: hidden\n```\n"
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_unclosed_fence_fails_closed(self):
        body = "```\nDiff-budget override: after unclosed fence"
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_marker_must_be_line_anchored(self):
        # mid-sentence mention must not count as a marker
        body = "we discussed a diff-budget override: maybe later"
        code, _ = mod.evaluate(500, body, BUDGET)
        assert code == 1


class TestFencedMarkers:
    def test_marker_inside_code_fence_is_ignored(self):
        # documenting the syntax (this gate's own PR body does) is not a decision
        body = "## Mechanism\n```\nDiff-budget override: <substantive reason>\n```\n"
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_real_marker_outside_fence_still_honored(self):
        body = ("```\nDiff-budget override: fenced example\n```\n"
                "Diff-budget override: review-fix wave is indivisible from the gate")
        code, messages = mod.evaluate(900, body, BUDGET)
        assert code == 0
        assert any("indivisible" in m for m in messages)

    @pytest.mark.parametrize("reason", ["<why this slice is genuinely indivisible>",
                                        "<substantive reason>"])
    def test_copied_template_reason_fails(self, reason):
        code, messages = mod.evaluate(500, f"Diff-budget override: {reason}", BUDGET)
        assert code == 1
        assert any("no real" in m for m in messages)

    def test_early_placeholder_does_not_shadow_later_real_reason(self):
        body = ("Diff-budget override: TODO\n"
                "Diff-budget override: guard code plus mandated regression tests")
        code, messages = mod.evaluate(900, body, BUDGET)
        assert code == 0
        assert any("mandated regression" in m for m in messages)

    def test_indented_code_block_marker_is_ignored(self):
        # 4-space Markdown code block: documentation, not a decision
        body = "Example:\n\n    Diff-budget override: example syntax only\n"
        code, _ = mod.evaluate(900, body, BUDGET)
        assert code == 1

    def test_up_to_three_leading_spaces_still_counts(self):
        code, _ = mod.evaluate(900, "   Diff-budget override: real reason", BUDGET)
        assert code == 0

    @pytest.mark.parametrize("reason", ["TODO.", "n/a.", "(tbd)", "TODO -",
                                        "n/a -",
                                        "<why this slice is genuinely indivisible>.",
                                        "<why this slice is genuinely indivisible> --"])
    def test_punctuated_placeholder_reasons_fail(self, reason):
        code, _ = mod.evaluate(500, f"Diff-budget override: {reason}", BUDGET)
        assert code == 1


class TestFetchGuards:
    def test_non_json_gh_output_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(mod, "_gh", lambda args, gh: "not json at all")
        with pytest.raises(RuntimeError, match="non-JSON"):
            mod.fetch_pr(1, "owner/repo", "gh")

    @pytest.mark.parametrize("payload", ['{"body": "x"}',
                                         '{"additions": null, "body": "x"}',
                                         '{"additions": "9", "body": "x"}'])
    def test_missing_or_non_numeric_additions_raises(self, monkeypatch, payload):
        monkeypatch.setattr(mod, "_gh", lambda args, gh: payload)
        with pytest.raises(RuntimeError, match="additions"):
            mod.fetch_pr(1, "owner/repo", "gh")

    @pytest.mark.parametrize("payload", ["[]", '"str"', "3"])
    def test_non_object_json_root_raises(self, monkeypatch, payload):
        monkeypatch.setattr(mod, "_gh", lambda args, gh: payload)
        with pytest.raises(RuntimeError, match="non-object"):
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
