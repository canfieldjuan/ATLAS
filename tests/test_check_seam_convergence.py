"""Tests for the seam-convergence advisory breaker (AGENTS 3k.2).

Exercises the pure core (bot_review_rounds / window_is_flat_or_rising /
leading_path / find_trip / body_declares_seam_analysis / evaluate) with
synthetic GraphQL node shapes -- both directions per AGENTS 3i: a flat or rising
same-seam run trips, while a converging run, a strictly declining run, a single
noisy round, scattered findings, a window whose last round moved to a different
file, and a body that only mentions the phrase all stay clean.

The regression block at the end pins the five review findings from PR #2199, and
the last test replays the real ATLAS #2181 round shape.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_seam_convergence",
    Path(__file__).resolve().parent.parent / "scripts" / "check_seam_convergence.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod  # dataclass resolution needs the module registered
_SPEC.loader.exec_module(mod)


# --- helpers -----------------------------------------------------------------


def review(hour: int, paths: list[str], login: str = "chatgpt-codex-connector[bot]") -> dict:
    return {
        "submittedAt": f"2026-07-01T{hour:02d}:00:00Z",
        "author": {"login": login},
        "comments": {"nodes": [{"path": p} for p in paths]},
    }


def rounds_from(counts: list[int], path: str = "svc/classifier.py") -> list:
    """One bot review per entry, each raising `counts[i]` findings on one path."""
    return mod.bot_review_rounds(
        [review(i, [path] * n) for i, n in enumerate(counts) if n], ("codex",)
    )


# --- bot_review_rounds -------------------------------------------------------


def test_rounds_keep_codex_and_copilot() -> None:
    nodes = [review(1, ["a.py"]), review(2, ["b.py"], login="copilot-pull-request-reviewer")]
    assert len(mod.bot_review_rounds(nodes, ("copilot", "codex"))) == 2


def test_rounds_drop_human_reviews() -> None:
    assert mod.bot_review_rounds([review(1, ["a.py"], login="canfieldjuan")], ("codex",)) == []


def test_rounds_honour_bot_override() -> None:
    assert mod.bot_review_rounds([review(1, ["a.py"], login="some-other-bot")], ("some-other",))


def test_review_without_inline_comments_is_not_a_round() -> None:
    """An approval or summary-only submission is not a round of findings."""
    assert mod.bot_review_rounds([review(1, [])], ("codex",)) == []


def test_review_without_timestamp_is_skipped() -> None:
    node = review(1, ["a.py"])
    node["submittedAt"] = ""
    assert mod.bot_review_rounds([node], ("codex",)) == []


def test_rounds_are_ordered_by_submission_time() -> None:
    rounds = mod.bot_review_rounds([review(9, ["a.py"]), review(1, ["b.py"])], ("codex",))
    assert [r.submitted_at for r in rounds] == sorted(r.submitted_at for r in rounds)
    assert [r.index for r in rounds] == [1, 2]


# --- trend rule, both sides --------------------------------------------------


def test_flat_run_trips() -> None:
    trip = mod.find_trip(rounds_from([5, 4, 5]))
    assert trip is not None and trip[0] == 3


def test_rising_run_trips() -> None:
    assert mod.find_trip(rounds_from([3, 5, 7])) is not None


def test_noisy_flat_run_trips() -> None:
    """ATLAS #2181's first window: 5, 9, 4 is noise around a flat trend."""
    assert mod.find_trip(rounds_from([5, 9, 4])) is not None


def test_strictly_declining_run_does_not_trip() -> None:
    assert mod.find_trip(rounds_from([8, 6, 4])) is None


def test_converging_run_does_not_trip() -> None:
    assert mod.find_trip(rounds_from([9, 4, 1])) is None


def test_two_rounds_do_not_trip() -> None:
    assert mod.find_trip(rounds_from([9, 9])) is None


def test_single_noisy_round_does_not_trip() -> None:
    assert mod.find_trip(rounds_from([40, 3, 1])) is None


def test_scattered_findings_do_not_trip() -> None:
    """Same counts spread over many files: review breadth, not one seam."""
    nodes = [review(i, [f"file{j}.py" for j in range(5)]) for i in range(3)]
    assert mod.find_trip(mod.bot_review_rounds(nodes, ("codex",))) is None


def test_trip_names_the_leading_seam() -> None:
    nodes = [review(i, ["seam.py"] * 4 + ["other.py"]) for i in range(3)]
    trip = mod.find_trip(mod.bot_review_rounds(nodes, ("codex",)))
    assert trip is not None and trip[1] == "seam.py"


# --- leading_path ------------------------------------------------------------


def test_leading_path_allows_exact_plurality() -> None:
    """#2181 round 3 is exactly 50%; requiring a majority would miss it."""
    assert mod.leading_path(["a.py", "a.py", "b.py", "c.py"]) == "a.py"


def test_leading_path_returns_none_on_a_tie() -> None:
    assert mod.leading_path(["a.py", "b.py"]) is None


def test_leading_path_returns_none_when_empty() -> None:
    assert mod.leading_path([]) is None


# --- Decision-Seam Analysis marker -------------------------------------------

_REAL_SECTION = (
    "## Decision-Seam Analysis\n"
    "The seam is the single admit verdict for a transcript line. It is an open\n"
    "category that no pattern list closes, so we evidence-gate it and fix the\n"
    "decision structurally with a warn-by-default direction.\n"
)


def test_real_seam_section_suppresses_the_trip() -> None:
    tripped, _seam, messages = mod.evaluate(rounds_from([5, 4, 5]), _REAL_SECTION)
    assert tripped is False
    assert any("SATISFIED" in m for m in messages)


def test_bare_mention_does_not_suppress() -> None:
    assert mod.body_declares_seam_analysis("No Decision-Seam Analysis has been completed") is False


def test_promise_of_a_future_analysis_does_not_suppress() -> None:
    assert mod.body_declares_seam_analysis("Deferred: add a Decision-Seam Analysis later") is False


def test_empty_seam_section_does_not_suppress() -> None:
    assert mod.body_declares_seam_analysis("## Decision-Seam Analysis\n\nTBD\n") is False


def test_seam_section_without_a_disposition_does_not_suppress() -> None:
    body = (
        "## Decision-Seam Analysis\n"
        "The seam is the admit verdict, and it is over-broad because the category\n"
        "of inputs it must recognise cannot be enumerated by any list of patterns.\n"
    )
    assert mod.body_declares_seam_analysis(body) is False


def test_seam_section_ends_at_the_next_heading() -> None:
    body = "## Decision-Seam Analysis\n\n## Verification\nfixed the seam decision here, at length, with detail\n"
    assert mod.body_declares_seam_analysis(body) is False


# --- evaluate ----------------------------------------------------------------


def test_evaluate_reports_trip_without_seam_analysis() -> None:
    tripped, seam, messages = mod.evaluate(rounds_from([5, 4, 5]), "some body")
    assert tripped is True
    assert seam == "svc/classifier.py"
    assert any("3k.2 tripped" in m for m in messages)


def test_evaluate_clean_run_reports_ok() -> None:
    tripped, seam, messages = mod.evaluate(rounds_from([9, 4, 1]), "")
    assert tripped is False and seam is None
    assert any(m.startswith("OK:") for m in messages)


def test_evaluate_handles_empty_pr() -> None:
    tripped, seam, _messages = mod.evaluate([], "")
    assert tripped is False and seam is None


def test_annotation_points_at_the_seam_and_bans_the_next_patch() -> None:
    text = mod.annotation("svc/classifier.py", rounds_from([5, 4, 5]))
    assert text.startswith("::warning file=svc/classifier.py::")
    assert "may NOT add another token, regex, vocabulary row, or oracle fixture" in text
    assert "Decision-Seam Analysis" in text


# --- failure paths: the raises are contracts, so they are asserted -----------


def test_gh_raises_on_a_failing_command() -> None:
    """A non-zero gh exit must surface, never be read as an empty result."""
    with pytest.raises(RuntimeError):
        mod._gh(["anything"], "false")


def test_fetch_reviews_raises_on_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GitHub response that is not JSON is a retryable failure, not silence."""
    monkeypatch.setattr(mod, "_gh", lambda args, gh: "<html>rate limited</html>")
    with pytest.raises(RuntimeError, match="non-JSON"):
        mod.fetch_reviews(1, "owner", "name", "gh")


def test_main_returns_two_on_a_malformed_repo() -> None:
    """Usage error exits 2 -- never 0, which would read as a clean run."""
    assert mod.main(["--pr", "1", "--repo", "not-a-slug"]) == 2


def test_main_returns_two_when_github_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> list[dict]:
        raise RuntimeError("API down")

    monkeypatch.setattr(mod, "fetch_reviews", boom)
    assert mod.main(["--pr", "1", "--repo", "owner/name"]) == 2


def test_unreadable_pr_body_falls_toward_tripping(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body that cannot be read must not silently count as satisfied."""
    monkeypatch.delenv("ATLAS_CURRENT_PR_BODY_FILE", raising=False)

    def boom(args: object, gh: object) -> str:
        raise RuntimeError("no auth")

    monkeypatch.setattr(mod, "_gh", boom)
    assert mod._pr_body(1, "owner/name", "gh") == ""


# --- regressions for the five findings on PR #2199 ---------------------------


def test_regression_multi_commit_push_still_trips() -> None:
    """Finding 1: rounds are review submissions, so commits-per-push is moot.

    The commit-keyed model bucketed three two-commit pushes as [0,n,0,n,0,n] and
    the synthetic zeros suppressed the trip entirely.
    """
    nodes = [review(h, ["seam.py"] * 5) for h in (1, 3, 5)]
    assert mod.find_trip(mod.bot_review_rounds(nodes, ("codex",))) is not None


def test_regression_negated_marker_does_not_fail_open() -> None:
    """Finding 2: substring matching suppressed on 'No Decision-Seam Analysis'."""
    tripped, _seam, _messages = mod.evaluate(
        rounds_from([5, 4, 5]), "No Decision-Seam Analysis has been completed"
    )
    assert tripped is True


def test_regression_seam_must_lead_the_last_round() -> None:
    """Finding 4: 6a, 6a, 5b tripped and named a.py though b.py leads round 3."""
    nodes = [
        review(0, ["a.py"] * 6),
        review(1, ["a.py"] * 6),
        review(2, ["b.py"] * 5),
    ]
    assert mod.find_trip(mod.bot_review_rounds(nodes, ("codex",))) is None


def test_regression_endpoint_ratio_no_longer_trips_a_decline() -> None:
    """Finding 5a: [8,6,4] is declining, not flat or rising."""
    assert mod.window_is_flat_or_rising([8, 6, 4]) is False


def test_regression_one_noisy_round_cannot_disguise_a_collapse() -> None:
    """Finding 5b: [4,100,2] passed the endpoint test; the mean rejects it."""
    assert mod.window_is_flat_or_rising([4, 100, 2]) is False


# --- real-world replay: ATLAS #2181 ------------------------------------------


def test_atlas_2181_shape_trips_at_round_three() -> None:
    """The observed per-round finding counts on ATLAS #2181, one seam throughout.

    18 bot review rounds, dead flat. The detector must fire at round 3 -- with
    15 rounds still to come -- not somewhere near the end.
    """
    observed = [5, 9, 4, 3, 1, 5, 6, 4, 4, 5, 6, 3, 6, 5, 5, 4, 3, 5]
    seam = "atlas_brain/services/content_factory_copy_verification.py"
    rounds = rounds_from(observed, path=seam)
    assert sum(r.count for r in rounds) == 83
    trip = mod.find_trip(rounds)
    assert trip is not None
    trip_index, trip_seam, _window = trip
    assert trip_index == 3
    assert trip_seam == seam
