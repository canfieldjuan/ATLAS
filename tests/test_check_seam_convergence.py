"""Tests for the seam-convergence advisory breaker (AGENTS 3k.2).

Exercises the pure core (bot_findings / assign_findings_to_pushes /
find_trip / evaluate) with synthetic GraphQL node shapes -- both directions per
AGENTS 3i: a flat or rising same-seam run trips, while a converging run, a
single noisy push, scattered findings, and a PR that already carries a
Decision-Seam Analysis all stay clean.

The last test replays the real ATLAS #2181 shape (94 findings over 20 pushes,
dead flat, every round on one file) and pins the trip to push 3 -- the whole
point of the detector is that it fires 17 pushes before that loop ended.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_seam_convergence",
    Path(__file__).resolve().parent.parent / "scripts" / "check_seam_convergence.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod  # dataclass resolution needs the module registered
_SPEC.loader.exec_module(mod)


# --- helpers -----------------------------------------------------------------


def commit(day: int, hour: int = 0) -> dict:
    return {"commit": {"oid": f"sha{day}{hour}", "committedDate": f"2026-07-{day:02d}T{hour:02d}:00:00Z"}}


def thread(day: int, hour: int, path: str, login: str = "chatgpt-codex-connector[bot]") -> dict:
    return {
        "path": path,
        "comments": {
            "nodes": [
                {"author": {"login": login}, "createdAt": f"2026-07-{day:02d}T{hour:02d}:30:00Z"}
            ]
        },
    }


def rounds_from(counts: list[int], path: str = "svc/classifier.py") -> list:
    """Build push rounds carrying `counts[i]` findings each, all on one path."""
    commits = [commit(1, i) for i in range(len(counts))]
    findings: list[tuple[str, str]] = []
    for i, n in enumerate(counts):
        for _ in range(n):
            findings.append((f"2026-07-01T{i:02d}:30:00Z", path))
    return mod.assign_findings_to_pushes(commits, findings)


# --- bot_findings ------------------------------------------------------------


def test_bot_findings_keeps_codex_and_copilot() -> None:
    nodes = [thread(1, 1, "a.py"), thread(1, 2, "b.py", login="copilot-pull-request-reviewer")]
    assert len(mod.bot_findings(nodes, ("copilot", "codex"))) == 2


def test_bot_findings_drops_human_authors() -> None:
    nodes = [thread(1, 1, "a.py", login="canfieldjuan")]
    assert mod.bot_findings(nodes, ("copilot", "codex")) == []


def test_bot_findings_honours_bot_override() -> None:
    nodes = [thread(1, 1, "a.py", login="some-other-bot")]
    assert mod.bot_findings(nodes, ("some-other",)) != []


def test_bot_findings_skips_threads_without_comments() -> None:
    assert mod.bot_findings([{"path": "a.py", "comments": {"nodes": []}}], ("codex",)) == []


def test_bot_findings_skips_missing_timestamp() -> None:
    node = {"path": "a.py", "comments": {"nodes": [{"author": {"login": "codex"}}]}}
    assert mod.bot_findings([node], ("codex",)) == []


def test_bot_findings_counts_resolved_threads() -> None:
    """A closed thread is the instance-patch 3k.2 looks for; it must still count."""
    node = thread(1, 1, "a.py")
    node["isResolved"] = True
    assert len(mod.bot_findings([node], ("codex",))) == 1


# --- assign_findings_to_pushes -----------------------------------------------


def test_findings_bucket_into_the_preceding_push() -> None:
    commits = [commit(1, 0), commit(1, 5)]
    findings = [("2026-07-01T01:00:00Z", "a.py"), ("2026-07-01T06:00:00Z", "b.py")]
    rounds = mod.assign_findings_to_pushes(commits, findings)
    assert [r.count for r in rounds] == [1, 1]


def test_finding_before_every_push_is_dropped() -> None:
    rounds = mod.assign_findings_to_pushes([commit(2, 0)], [("2026-07-01T00:00:00Z", "a.py")])
    assert [r.count for r in rounds] == [0]


def test_no_commits_yields_no_rounds() -> None:
    assert mod.assign_findings_to_pushes([], [("2026-07-01T00:00:00Z", "a.py")]) == []


def test_commits_are_ordered_by_date_not_input_order() -> None:
    rounds = mod.assign_findings_to_pushes([commit(1, 9), commit(1, 1)], [])
    assert [r.pushed_at for r in rounds] == sorted(r.pushed_at for r in rounds)


# --- find_trip: the boundary, both sides -------------------------------------


def test_converging_run_does_not_trip() -> None:
    assert mod.find_trip(rounds_from([5, 2, 0])) is None


def test_halving_run_does_not_trip() -> None:
    assert mod.find_trip(rounds_from([8, 4, 3])) is None


def test_flat_run_trips() -> None:
    trip = mod.find_trip(rounds_from([5, 4, 5]))
    assert trip is not None and trip[0] == 3


def test_rising_run_trips() -> None:
    assert mod.find_trip(rounds_from([3, 5, 7])) is not None


def test_exactly_at_the_ratio_trips() -> None:
    """4 is exactly half of 8: not yet trending to zero, so it trips."""
    assert mod.find_trip(rounds_from([8, 5, 4])) is not None


def test_two_pushes_do_not_trip() -> None:
    assert mod.find_trip(rounds_from([9, 9])) is None


def test_zero_finding_push_breaks_the_streak() -> None:
    assert mod.find_trip(rounds_from([5, 0, 5])) is None


def test_single_noisy_push_does_not_trip() -> None:
    assert mod.find_trip(rounds_from([40, 0, 0])) is None


def test_scattered_findings_do_not_trip() -> None:
    """Same counts, but spread over many files: review breadth, not one seam."""
    commits = [commit(1, i) for i in range(3)]
    findings = []
    for i in range(3):
        for j in range(5):
            findings.append((f"2026-07-01T{i:02d}:30:00Z", f"file{j}.py"))
    assert mod.find_trip(mod.assign_findings_to_pushes(commits, findings)) is None


def test_trip_names_the_dominant_seam() -> None:
    commits = [commit(1, i) for i in range(3)]
    findings = []
    for i in range(3):
        findings += [(f"2026-07-01T{i:02d}:30:00Z", "seam.py")] * 4
        findings.append((f"2026-07-01T{i:02d}:30:00Z", "other.py"))
    trip = mod.find_trip(mod.assign_findings_to_pushes(commits, findings))
    assert trip is not None and trip[1] == "seam.py"


# --- evaluate ----------------------------------------------------------------


def test_evaluate_reports_trip_without_seam_analysis() -> None:
    tripped, seam, messages = mod.evaluate(rounds_from([5, 4, 5]), "some body")
    assert tripped is True
    assert seam == "svc/classifier.py"
    assert any("3k.2 tripped" in m for m in messages)


def test_evaluate_suppressed_by_seam_analysis_in_body() -> None:
    body = "## Decision-Seam Analysis\nThe seam is the admit verdict."
    tripped, seam, messages = mod.evaluate(rounds_from([5, 4, 5]), body)
    assert tripped is False
    assert seam == "svc/classifier.py"
    assert any("SATISFIED" in m for m in messages)


def test_seam_analysis_marker_is_case_insensitive() -> None:
    assert mod.body_has_seam_analysis("decision-seam ANALYSIS follows") is True
    assert mod.body_has_seam_analysis("no such section") is False


def test_evaluate_clean_run_reports_ok() -> None:
    tripped, seam, messages = mod.evaluate(rounds_from([5, 2, 0]), "")
    assert tripped is False and seam is None
    assert any(m.startswith("OK:") for m in messages)


def test_evaluate_handles_empty_pr() -> None:
    tripped, seam, messages = mod.evaluate([], "")
    assert tripped is False and seam is None


def test_annotation_points_at_the_seam_and_bans_the_next_patch() -> None:
    rounds = rounds_from([5, 4, 5])
    text = mod.annotation("svc/classifier.py", rounds)
    assert text.startswith("::warning file=svc/classifier.py::")
    assert "may NOT add another token, regex, vocabulary row, or oracle fixture" in text
    assert "Decision-Seam Analysis" in text


# --- real-world replay: ATLAS #2181 ------------------------------------------


def test_atlas_2181_shape_trips_at_push_three() -> None:
    """The observed per-push finding counts on ATLAS #2181, one seam throughout.

    20 pushes, 94 findings, dead flat. The detector must fire at push 3 -- with
    17 pushes and 76 findings still to come -- not somewhere near the end.
    """
    observed = [5, 9, 4, 3, 1, 5, 6, 4, 4, 5, 6, 3, 6, 5, 5, 4, 3, 5, 6, 5]
    rounds = rounds_from(observed, path="atlas_brain/services/content_factory_copy_verification.py")
    assert sum(r.count for r in rounds) == 94
    trip = mod.find_trip(rounds)
    assert trip is not None
    trip_index, seam, _window = trip
    assert trip_index == 3
    assert seam.endswith("content_factory_copy_verification.py")
