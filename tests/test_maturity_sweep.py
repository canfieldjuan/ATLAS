"""Failure-branch tests for scripts/maturity_sweep.py (AGENTS.md section 3i).

The point of these fixtures is to prove the detectors FIRE, not just that the
tool runs. The repo-style-naming case pins the dead-detector bug found in
review: with the original exact-stem matcher, tests named
test_extracted_<module>.py never matched, so HAPPY_PATH_TESTS and
NO_RAISES_TESTS could not fire at all.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "maturity_sweep.py"

SPEC = importlib.util.spec_from_file_location("maturity_sweep", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def codes(findings):
    return {f.code for f in findings}


HAPPY_ONLY_TESTS = (
    "def test_returns_rows():\n    assert build() == []\n\n"
    "def test_counts_rows():\n    assert count() == 0\n\n"
    "def test_formats_output():\n    assert fmt() == ''\n"
)

FAILURE_RICH_TESTS = (
    "import pytest\n\n"
    "def test_rejects_invalid_rows():\n"
    "    with pytest.raises(ValueError):\n        build(None)\n\n"
    "def test_missing_field_marks_invalid():\n    assert not ok({})\n\n"
    "def test_malformed_csv_fails_loud():\n"
    "    with pytest.raises(ValueError):\n        parse('x')\n"
)


def test_happy_path_detectors_fire_for_repo_style_test_naming() -> None:
    # Dead-detector pin: test_extracted_<module> naming must match the module.
    findings = MOD.score_tests(
        "support_widget_pipeline",
        {"test_extracted_support_widget_pipeline": HAPPY_ONLY_TESTS},
        all_test_text=HAPPY_ONLY_TESTS,
    )
    assert "HAPPY_PATH_TESTS" in codes(findings)
    assert "NO_RAISES_TESTS" in codes(findings)


def test_content_ops_prefix_and_multiple_test_files_union() -> None:
    findings = MOD.score_tests(
        "faq_widget_report",
        {
            "test_content_ops_faq_widget_report": HAPPY_ONLY_TESTS,
            "test_extracted_faq_widget_report": FAILURE_RICH_TESTS,
        },
        all_test_text=HAPPY_ONLY_TESTS + FAILURE_RICH_TESTS,
    )
    # The union has 6 tests, 3 of them negative (50%) and raises present:
    # neither quality finding should fire.
    assert codes(findings) == set()


def test_quality_detectors_quiet_on_failure_rich_tests() -> None:
    findings = MOD.score_tests(
        "ingest_module",
        {"test_ingest_module": FAILURE_RICH_TESTS},
        all_test_text=FAILURE_RICH_TESTS,
    )
    assert "HAPPY_PATH_TESTS" not in codes(findings)
    assert "NO_RAISES_TESTS" not in codes(findings)


def test_no_test_file_fires_when_module_is_unreferenced() -> None:
    findings = MOD.score_tests("orphan_module", {}, all_test_text="")
    assert codes(findings) == {"NO_TEST_FILE"}


def test_mentioned_anywhere_fallback_suppresses_no_test_file() -> None:
    findings = MOD.score_tests(
        "helper_module", {},
        all_test_text="from pkg import helper_module\n",
    )
    assert codes(findings) == set()


def test_unrelated_test_stems_do_not_match() -> None:
    # 'report' must not match test_extracted_reporting (segment boundary).
    assert MOD.matching_test_sources(
        "report", {"test_extracted_reporting": HAPPY_ONLY_TESTS}) == []
    assert MOD.matching_test_sources(
        "report", {"test_extracted_report_builder": HAPPY_ONLY_TESTS}) != []


def _analyze(source):
    analyzer = MOD.Analyzer(is_test=False)
    analyzer.visit(ast.parse(source))
    return analyzer.findings


def test_swallowed_except_detector_fires() -> None:
    findings = _analyze(
        "def f():\n"
        "    try:\n        risky()\n"
        "    except Exception:\n        pass\n"
    )
    assert "SWALLOWED_EXCEPT" in {f.code for f in findings}


def test_unguarded_boundary_input_detector_fires_and_guarded_is_quiet() -> None:
    unguarded = _analyze("def f(p):\n    return open(p).read()\n")
    assert "UNGUARDED_INPUT" in {f.code for f in unguarded}
    guarded = _analyze(
        "def f(p):\n"
        "    try:\n        return open(p).read()\n"
        "    except OSError:\n        raise ValueError(p)\n"
    )
    assert "UNGUARDED_INPUT" not in {f.code for f in guarded}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_reference_test(tests: Path, *module_names: str) -> None:
    _write(
        tests / "test_refs.py",
        "\n".join("import %s" % name for name in module_names) + "\n",
    )


def _module_with_swallowed_except() -> str:
    return (
        "def run():\n"
        "    try:\n"
        "        risky()\n"
        "    except Exception:\n"
        "        pass\n"
    )


def test_update_baseline_writes_expected_json_shape(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "paid_flow.py", _module_with_swallowed_except())
    _write_reference_test(tests, "paid_flow")
    baseline = tmp_path / "baseline.json"

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    payload = json.loads(baseline.read_text(encoding="utf-8"))
    entry = payload[str(lane / "paid_flow.py")]
    assert entry["score"] == 5
    assert entry["counts"] == {"SWALLOWED_EXCEPT": 1}


def test_baselined_finding_does_not_fail_ratchet(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "paid_flow.py", _module_with_swallowed_except())
    _write_reference_test(tests, "paid_flow")
    baseline = tmp_path / "baseline.json"

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "5",
    ]) == 0


def test_score_increase_in_baselined_file_fails(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "parser.py", "# TODO tighten parser\n")
    _write_reference_test(tests, "parser")
    baseline = tmp_path / "baseline.json"

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0
    _write(lane / "parser.py", "# TODO tighten parser\n" + _module_with_swallowed_except())

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
    ]) == 1


def test_new_file_min_score_ratchet(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing", "new_risky")
    baseline = tmp_path / "baseline.json"

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0
    _write(lane / "new_risky.py", _module_with_swallowed_except())

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "6",
    ]) == 0
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "5",
    ]) == 1


def test_sensitive_path_swallowed_except_fails_below_min_score(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "billing_paid.py", "VALUE = 1\n")
    _write_reference_test(tests, "billing_paid")
    baseline = tmp_path / "baseline.json"

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0
    _write(lane / "billing_paid.py", _module_with_swallowed_except())

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/billing_paid.py",
    ]) == 1


def test_update_baseline_accepts_new_sensitive_finding(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "billing_paid.py", "VALUE = 1\n")
    _write_reference_test(tests, "billing_paid")
    baseline = tmp_path / "baseline.json"

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0
    _write(lane / "billing_paid.py", _module_with_swallowed_except())
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/billing_paid.py",
    ]) == 0
