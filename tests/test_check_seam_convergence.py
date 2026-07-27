"""Tests for the seam-convergence advisory breaker (AGENTS 3k.2).

Exercises the pure core (bot_review_rounds / leading_path / find_trip /
recorded_seam_analysis / evaluate) with synthetic GraphQL node shapes -- both
directions per AGENTS 3i. The trip decision carries no tunable threshold, so the
tests pin facts rather than numbers: a run whose seam count does not decrease
across three consecutive led rounds trips, while a declining run, an empty round,
a tie, scattered findings, and a seam that stops leading all stay silent.

The regression block pins every review finding from PR #2199 -- both rounds --
and the replay test uses the seam-only counts observed on ATLAS #2181.
"""
from __future__ import annotations

import json
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
        [review(i, [path] * n) for i, n in enumerate(counts)], ("codex",)
    )


# --- bot_review_rounds -------------------------------------------------------


def test_rounds_keep_codex_and_copilot() -> None:
    nodes = [review(1, ["a.py"]), review(2, ["b.py"], login="copilot-pull-request-reviewer")]
    assert len(mod.bot_review_rounds(nodes, ("copilot", "codex"))) == 2


def test_rounds_drop_human_reviews() -> None:
    assert mod.bot_review_rounds([review(1, ["a.py"], login="canfieldjuan")], ("codex",)) == []


def test_rounds_honour_bot_override() -> None:
    assert mod.bot_review_rounds([review(1, ["a.py"], login="some-other-bot")], ("some-other",))


def test_empty_bot_review_is_kept_as_a_round() -> None:
    """A bot review raising nothing is convergence evidence, not a gap."""
    rounds = mod.bot_review_rounds([review(1, [])], ("codex",))
    assert len(rounds) == 1 and rounds[0].count == 0


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


def test_equal_counts_trip() -> None:
    assert mod.find_trip(rounds_from([5, 5, 5])) is not None


def test_declining_run_does_not_trip() -> None:
    assert mod.find_trip(rounds_from([8, 6, 4])) is None


def test_converging_run_does_not_trip() -> None:
    assert mod.find_trip(rounds_from([9, 4, 1])) is None


def test_two_rounds_do_not_trip() -> None:
    assert mod.find_trip(rounds_from([9, 9])) is None


def test_single_noisy_round_does_not_trip() -> None:
    assert mod.find_trip(rounds_from([40, 3, 1])) is None


def test_empty_round_breaks_the_streak() -> None:
    """A clean bot review between two noisy ones is convergence."""
    assert mod.find_trip(rounds_from([5, 0, 5])) is None


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


# --- Decision-Seam Analysis marker (machine token, not prose) ----------------


def test_marker_accepts_each_disposition() -> None:
    for word in ("fix", "waive", "rescope"):
        assert mod.recorded_seam_analysis(
            "svc/classifier.py", f"decision-seam-analysis: {word} svc/classifier.py"
        ) == word


def test_marker_is_case_insensitive() -> None:
    assert mod.recorded_seam_analysis(
        "svc/classifier.py", "Decision-Seam-Analysis: FIX svc/classifier.py"
    ) == "fix"


def test_marker_absent_returns_none() -> None:
    assert mod.recorded_seam_analysis("svc/classifier.py", "no marker here") is None


def test_prose_alone_does_not_satisfy_the_marker() -> None:
    """The old prose parser accepted this; a machine token cannot be argued with."""
    body = (
        "## Decision-Seam Analysis\n"
        "The seam is the admit verdict and it is under-broad, but we are not\n"
        "going to do anything about it in this slice.\n"
    )
    assert mod.recorded_seam_analysis(body) is None


def test_marker_requires_a_known_disposition() -> None:
    assert mod.recorded_seam_analysis("decision-seam-analysis: maybe") is None


def test_marker_is_read_from_any_supplied_text() -> None:
    """3k.2 asks for the analysis in the plan OR the PR body."""
    plan = "## Deferred\n\ndecision-seam-analysis: waive svc/classifier.py\n"
    assert mod.recorded_seam_analysis(
        "svc/classifier.py", "pr body with no marker", plan
    ) == "waive"


def test_marker_must_be_line_anchored() -> None:
    """Prose that mentions the token mid-sentence is not a recorded decision."""
    assert mod.recorded_seam_analysis("we should add decision-seam-analysis: fix later") is None


def test_recorded_marker_suppresses_the_trip() -> None:
    tripped, _seam, messages = mod.evaluate(
        rounds_from([5, 5, 5]), "decision-seam-analysis: fix svc/classifier.py"
    )
    assert tripped is False
    assert any("SATISFIED" in m for m in messages)


# --- evaluate ----------------------------------------------------------------


def test_evaluate_reports_trip_without_seam_analysis() -> None:
    tripped, seam, messages = mod.evaluate(rounds_from([5, 5, 5]), "some body")
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
        rounds_from([5, 5, 5]), "No Decision-Seam Analysis has been completed"
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
    assert mod.find_trip(rounds_from([8, 6, 4])) is None


def test_regression_one_noisy_round_cannot_disguise_a_collapse() -> None:
    """Finding 5b: [4,100,2] passed the old endpoint ratio; 2 < 4 now rejects it."""
    assert mod.find_trip(rounds_from([4, 100, 2])) is None


def test_regression_trend_is_measured_on_the_seam_not_the_total() -> None:
    """Round 2: totals hid a declining seam behind unrelated findings."""
    nodes = [
        review(0, ["seam.py"] * 5),
        review(1, ["seam.py"] * 4 + ["other.py"] * 3),
        review(2, ["seam.py"] * 2 + ["other.py"] * 6),
    ]
    assert mod.find_trip(mod.bot_review_rounds(nodes, ("codex",))) is None


def test_regression_empty_round_is_not_spliced_out() -> None:
    """Round 2: skipping a clean review made non-adjacent rounds look adjacent."""
    nodes = [review(0, ["s.py"] * 5), review(1, []), review(2, ["s.py"] * 5), review(3, ["s.py"] * 5)]
    rounds = mod.bot_review_rounds(nodes, ("codex",))
    assert [r.count for r in rounds] == [5, 0, 5, 5]
    assert mod.find_trip(rounds) is None


# --- real-world replay: ATLAS #2181 ------------------------------------------


def test_atlas_2181_replay_trips_well_before_the_end() -> None:
    """Seam-only counts observed on ATLAS #2181, one file leading throughout.

    The threshold-free rule fires at round 6 of 18 -- later than the tuned
    version did, which is the deliberate bias toward silence, and still with 12
    rounds of that loop still to come.
    """
    seam_counts = [5, 6, 2, 3, 1, 4, 5, 4, 4, 4, 5, 3, 6, 5, 4, 3, 3, 5]
    seam = "atlas_brain/services/content_factory_copy_verification.py"
    rounds = rounds_from(seam_counts, path=seam)
    trip = mod.find_trip(rounds)
    assert trip is not None
    trip_index, trip_seam, _counts = trip
    assert trip_index == 6
    assert trip_seam == seam


# --- suppression is bound, in both directions --------------------------------
#
# An unbound marker was the breaker's own off switch: plan docs live on main
# after merge, so the first merged marker suppressed every later PR's trip --
# and this slice's plan carries one, which would have made merging the breaker
# turn it off.


def test_marker_for_another_seam_does_not_suppress() -> None:
    """A disposition answering one seam is not an answer for a different one."""
    tripped, _seam, _messages = mod.evaluate(
        rounds_from([5, 5, 5]), "decision-seam-analysis: fix some/other/file.py"
    )
    assert tripped is True


def test_unbound_marker_does_not_suppress() -> None:
    """The pathless form is no longer a marker at all."""
    assert mod.recorded_seam_analysis("svc/classifier.py", "decision-seam-analysis: fix") is None
    tripped, _seam, _messages = mod.evaluate(
        rounds_from([5, 5, 5]), "decision-seam-analysis: fix"
    )
    assert tripped is True


def test_plan_texts_reads_only_the_declared_plan(tmp_path) -> None:
    """Globbing plans/ let any marker in the repository suppress the trip."""
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "PR-Mine.md").write_text("decision-seam-analysis: fix svc/classifier.py\n", encoding="utf-8")
    (plans / "PR-Unrelated.md").write_text("decision-seam-analysis: waive svc/classifier.py\n", encoding="utf-8")

    texts = mod._plan_texts(tmp_path, "Plan: plans/PR-Mine.md\n")
    assert len(texts) == 1
    assert "fix" in texts[0]

    # a body naming no plan reads nothing, rather than falling back to the glob
    assert mod._plan_texts(tmp_path, "no plan line here") == []


def test_plan_texts_rejects_paths_outside_plans(tmp_path) -> None:
    """The PR body is untrusted text; a traversal must not read arbitrary files."""
    plans = tmp_path / "plans"
    plans.mkdir()
    (tmp_path / "secret.md").write_text("decision-seam-analysis: fix svc/classifier.py\n", encoding="utf-8")
    assert mod._plan_texts(tmp_path, "Plan: plans/../secret.md\n") == []
    assert mod._plan_texts(tmp_path, "Plan: plans/PR-Missing.md\n") == []


def test_annotation_emits_a_marker_the_detector_accepts() -> None:
    """The annotation is the only place an author learns the marker's shape, so
    a round trip failure here means the instruction is unfollowable."""
    import re as _re

    rounds = rounds_from([3, 3, 3])
    text = mod.annotation("svc/classifier.py", rounds)
    found = _re.search(r"decision-seam-analysis: fix \S+", text)
    assert found is not None, "annotation must show the exact marker line"
    assert mod.recorded_seam_analysis("svc/classifier.py", found.group(0)) == "fix"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": None},
        {"data": {"repository": None}},
        {"data": {"repository": {"pullRequest": None}}},
        {"data": {"repository": {"pullRequest": {"reviews": None}}}},
        {"errors": [{"message": "rate limited"}]},
    ],
)
def test_malformed_graphql_envelope_raises(monkeypatch, payload) -> None:
    """A missing envelope is indistinguishable from 'no reviews', and no reviews
    reads as convergence -- so a transport failure would silence the breaker."""
    monkeypatch.setattr(mod, "_gh", lambda *_a, **_k: json.dumps(payload))
    with pytest.raises(RuntimeError):
        mod.fetch_reviews(1, "owner", "name", "gh")
