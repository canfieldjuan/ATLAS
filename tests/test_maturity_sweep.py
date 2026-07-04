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


def test_internal_mock_detector_attaches_to_mocked_first_party_module(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    module = lane / "extracted_content_pipeline" / "faq_deflection_report.py"
    _write(module, "def build():\n    return []\n")
    _write(
        tests / "test_report.py",
        "from unittest import mock\n\n"
        "def test_internal_patch():\n"
        "    with mock.patch('extracted_content_pipeline.faq_deflection_report.build'):\n"
        "        pass\n\n"
        "def test_external_patch():\n"
        "    with mock.patch('httpx.post'):\n"
        "        pass\n",
    )

    results = MOD.sweep(lane, tests)
    result = next(item for item in results if item.path == str(module))

    assert result.counts["INTERNAL_MOCK"] == 1
    assert any(
        finding.code == "INTERNAL_MOCK"
        and "extracted_content_pipeline.faq_deflection_report.build" in finding.detail
        for finding in result.findings
    )


def test_internal_mock_detector_handles_monkeypatch_module_targets(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    module = lane / "atlas_brain" / "api" / "billing.py"
    _write(module, "def mark_paid():\n    return True\n")
    _write(
        tests / "test_billing.py",
        "from unittest.mock import MagicMock\n"
        "from atlas_brain.api import billing\n\n"
        "def test_internal_monkeypatch(monkeypatch):\n"
        "    monkeypatch.setattr(billing, 'mark_paid', MagicMock())\n",
    )

    results = MOD.sweep(lane, tests)
    result = next(item for item in results if item.path == str(module))

    assert result.counts["INTERNAL_MOCK"] == 1


def test_internal_mock_detector_handles_patch_object_module_targets(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    module = lane / "atlas_brain" / "api" / "billing.py"
    _write(module, "def mark_paid():\n    return True\n")
    _write(
        tests / "test_billing.py",
        "from unittest.mock import patch\n"
        "from atlas_brain.api import billing\n\n"
        "def test_internal_patch_object():\n"
        "    with patch.object(billing, 'mark_paid'):\n"
        "        pass\n",
    )

    results = MOD.sweep(lane, tests)
    result = next(item for item in results if item.path == str(module))

    assert result.counts["INTERNAL_MOCK"] == 1
    assert any(
        finding.code == "INTERNAL_MOCK"
        and "atlas_brain.api.billing.mark_paid" in finding.detail
        and "patch.object" in finding.detail
        for finding in result.findings
    )


def test_internal_mock_detector_covers_blocking_extracted_roots(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    module = lane / "extracted_competitive_intelligence" / "worker.py"
    _write(module, "def run():\n    return True\n")
    _write(
        tests / "test_competitive_worker.py",
        "from unittest.mock import patch\n\n"
        "def test_internal_patch():\n"
        "    with patch('extracted_competitive_intelligence.worker.run'):\n"
        "        pass\n",
    )

    results = MOD.sweep(lane, tests)
    result = next(item for item in results if item.path == str(module))

    assert result.counts["INTERNAL_MOCK"] == 1


def test_internal_mock_detector_allows_wall_clock_and_randomness_seams(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    module = lane / "atlas_brain" / "jobs" / "scheduler.py"
    _write(module, "import time\nimport random\n\ndef jitter():\n    return time.perf_counter() + random.random()\n")
    _write(
        tests / "test_scheduler.py",
        "from atlas_brain.jobs import scheduler\n\n"
        "def test_clock_and_randomness(monkeypatch):\n"
        "    monkeypatch.setattr(scheduler.time, 'perf_counter', lambda: 1.0)\n"
        "    monkeypatch.setattr(scheduler.random, 'random', lambda: 0.5)\n",
    )

    results = MOD.sweep(lane, tests)
    result = next(item for item in results if item.path == str(module))

    assert "INTERNAL_MOCK" not in result.counts


def test_internal_mock_detector_keeps_no_asname_dotted_import_external_seam(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    module = lane / "extracted_content_pipeline" / "api" / "control_surfaces.py"
    _write(module, "import socket\n\ndef resolve():\n    return socket.getaddrinfo\n")
    _write(
        tests / "test_control_surfaces.py",
        "import extracted_content_pipeline.api.control_surfaces\n\n"
        "def test_external_socket_monkeypatch(monkeypatch):\n"
        "    monkeypatch.setattr(\n"
        "        extracted_content_pipeline.api.control_surfaces.socket,\n"
        "        'getaddrinfo',\n"
        "        lambda *a: [],\n"
        "    )\n",
    )

    results = MOD.sweep(lane, tests)
    result = next(item for item in results if item.path == str(module))

    assert "INTERNAL_MOCK" not in result.counts


def test_internal_mock_detector_indexes_package_init_exports(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    module = lane / "atlas_brain" / "auth" / "__init__.py"
    _write(module, "def require_auth():\n    return True\n")
    _write(
        tests / "test_auth_package.py",
        "from unittest.mock import patch\n\n"
        "def test_package_export_patch():\n"
        "    with patch('atlas_brain.auth.require_auth'):\n"
        "        pass\n",
    )

    results = MOD.sweep(lane, tests)
    result = next(item for item in results if item.path == str(module))

    assert result.counts["INTERNAL_MOCK"] == 1


def test_internal_mock_detector_allows_imported_external_seams(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    module = lane / "extracted_content_pipeline" / "api" / "control_surfaces.py"
    _write(module, "import socket\n\ndef resolve():\n    return socket.getaddrinfo\n")
    _write(
        tests / "test_control_surfaces.py",
        "from extracted_content_pipeline.api import control_surfaces\n\n"
        "def test_external_socket_monkeypatch(monkeypatch):\n"
        "    monkeypatch.setattr(control_surfaces.socket, 'getaddrinfo', lambda *a: [])\n",
    )

    results = MOD.sweep(lane, tests)
    result = next(item for item in results if item.path == str(module))

    assert "INTERNAL_MOCK" not in result.counts


def test_internal_mock_ratchet_fails_on_new_mock_target(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(
        lane / "scripts" / "build_content_ops_deflection_report.py",
        "def main():\n    return 0\n",
    )
    _write(tests / "test_report.py", "def test_clean():\n    assert True\n")
    baseline = tmp_path / "baseline.json"

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0
    _write(
        tests / "test_report.py",
        "from unittest.mock import patch\n\n"
        "def test_internal_patch():\n"
        "    with patch('scripts.build_content_ops_deflection_report.main'):\n"
        "        pass\n",
    )

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
    ]) == 1


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


def _write_happy_only_tests(tests: Path, module_stem: str) -> None:
    """A test file with enough tests to trip NO_RAISES_TESTS (>= 3) and
    zero raises assertions."""
    _write(
        tests / ("test_%s.py" % module_stem),
        "import %s\n\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_three():\n    assert True\n" % module_stem,
    )


def test_new_sensitive_module_without_raises_tests_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Negatives-presence gate (#1934 arc lesson 5): a NEW module on a
    sensitive path whose tests never assert that anything raises must
    fail the ratchet outright -- min-score cannot save it, and the same
    file off the sensitive globs still passes (the gate is scoped, not
    repo-wide noise)."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write_happy_only_tests(tests, "purge_guard")

    # Off the sensitive globs: score is far below min-score, passes.
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
    ]) == 0
    # On the sensitive globs: zero tolerance, fails regardless of score.
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1
    out = capsys.readouterr().out
    assert "sensitive-path NO_RAISES_TESTS" in out


def test_new_sensitive_module_with_raises_tests_passes(tmp_path: Path) -> None:
    """Second side: the same new sensitive-path module whose tests DO
    assert a raise sails through the zero-tolerance gate."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import pytest\n"
        "import purge_guard\n\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_rejects_bad_input():\n"
        "    with pytest.raises(ValueError):\n"
        "        raise ValueError('boundary probe')\n",
    )

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0


def test_comment_mention_of_raises_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex P2 on this PR: the raises signal must come from a REAL
    pytest.raises/assertRaises call, not from a comment or string that
    mentions one -- a '# TODO: add pytest.raises' note is exactly the
    honest-but-hasty artifact the gate exists to catch."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import purge_guard\n"
        "# TODO: add pytest.raises coverage\n"
        "NOTE = 'assertRaises would be nice'\n\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_three():\n    assert True\n",
    )

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1


def test_new_testless_sensitive_module_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Codex P2 on this PR: a sensitive module with NO tests at all is
    the zero-negatives case at its worst -- NO_TEST_FILE is zero
    tolerance too, so min-score cannot save it either."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "billing_hook.py",
           "def charge(amount):\n"
           "    if amount <= 0:\n"
           "        raise ValueError('non-positive')\n"
           "    return amount\n")
    # No test file for billing_hook anywhere.

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
    ]) == 0
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/billing_hook.py",
    ]) == 1
    out = capsys.readouterr().out
    assert "sensitive-path NO_TEST_FILE" in out


def test_unrelated_raises_helper_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-2: only the real assertion APIs count -- an arbitrary
    helper call named raises() (e.g. client.raises()) must not suppress
    the blocking signal."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import purge_guard\n\n"
        "class _Client:\n"
        "    def raises(self):\n"
        "        return 0\n\n"
        "def test_one():\n    assert _Client().raises() == 0\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_three():\n    assert True\n",
    )

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1


def test_local_raises_helper_name_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-3: a bare raises(...) call only counts when the name is
    bound by `from pytest import raises` (or an alias). A local helper or
    fixture named raises must not suppress the blocking signal; the real
    from-import (aliased or not) still does."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    module_src = ("def enforce(value):\n"
                  "    if value < 0:\n"
                  "        raise ValueError('negative')\n"
                  "    return value\n")
    _write(lane / "purge_guard.py", module_src)
    _write(
        tests / "test_purge_guard.py",
        "import purge_guard\n\n"
        "def raises(exc):\n"
        "    return exc\n\n"
        "def test_one():\n    assert raises(ValueError) is ValueError\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_three():\n    assert True\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1

    # Second side: the real from-import, aliased, still satisfies it.
    _write(
        tests / "test_purge_guard.py",
        "from pytest import raises as expect_raises\n"
        "import purge_guard\n\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_rejects():\n"
        "    with expect_raises(ValueError):\n"
        "        purge_guard.enforce(-1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0


def test_dangling_raises_statement_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-4: a dangling `pytest.raises(X)` statement builds a
    context manager and asserts nothing; only the with-context or the
    callable form counts."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import pytest\n"
        "import purge_guard\n\n"
        "def test_one():\n"
        "    pytest.raises(ValueError)\n"
        "    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_three():\n    assert True\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1

    # Second side: the callable form asserts and satisfies the gate.
    _write(
        tests / "test_purge_guard.py",
        "import pytest\n"
        "import purge_guard\n\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_rejects():\n"
        "    pytest.raises(ValueError, purge_guard.enforce, -1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0


def test_stub_test_file_counts_as_testless(tmp_path: Path) -> None:
    """Codex wave-4: a matched test file with zero collected tests is a
    placeholder, not coverage -- the sensitive zero-tolerance gate treats
    it as testless."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "billing_hook.py",
           "def charge(amount):\n"
           "    if amount <= 0:\n"
           "        raise ValueError('non-positive')\n"
           "    return amount\n")
    _write(tests / "test_billing_hook.py",
           "import billing_hook\n\nHELPER = billing_hook.charge\n")

    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
    ]) == 0
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/billing_hook.py",
    ]) == 1

def test_foreign_assert_raises_receiver_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-5: assertRaises* only counts on the unittest receivers
    self/cls -- an unrelated helper like client.assertRaises(...) must
    not suppress the blocking signal; the real self.assertRaises does."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import purge_guard\n\n"
        "class _Client:\n"
        "    def assertRaises(self, exc, fn):\n"
        "        return fn\n\n"
        "def test_one():\n"
        "    client = _Client()\n"
        "    client.assertRaises(ValueError, purge_guard.enforce)\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_three():\n    assert True\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1

    # Second side: the real unittest idiom on self satisfies the gate.
    _write(
        tests / "test_purge_guard.py",
        "import unittest\n"
        "import purge_guard\n\n"
        "class PurgeGuardTest(unittest.TestCase):\n"
        "    def test_one(self):\n        self.assertTrue(True)\n\n"
        "    def test_two(self):\n        self.assertTrue(True)\n\n"
        "    def test_rejects(self):\n"
        "        self.assertRaises(ValueError, purge_guard.enforce, -1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0


def test_dangling_assert_raises_regex_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-5: self.assertRaisesRegex(Exc, "regex") with no with
    block and no callable only builds a context object and asserts
    nothing; the callable form with a third positional arg (or the with
    form) still counts."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import unittest\n"
        "import purge_guard\n\n"
        "class PurgeGuardTest(unittest.TestCase):\n"
        "    def test_one(self):\n"
        "        self.assertRaisesRegex(ValueError, 'negative')\n"
        "        self.assertTrue(True)\n\n"
        "    def test_two(self):\n        self.assertTrue(True)\n\n"
        "    def test_three(self):\n        self.assertTrue(True)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1

    # Second side: the callable form with the function argument asserts.
    _write(
        tests / "test_purge_guard.py",
        "import unittest\n"
        "import purge_guard\n\n"
        "class PurgeGuardTest(unittest.TestCase):\n"
        "    def test_one(self):\n        self.assertTrue(True)\n\n"
        "    def test_two(self):\n        self.assertTrue(True)\n\n"
        "    def test_rejects(self):\n"
        "        self.assertRaisesRegex(ValueError, 'negative',\n"
        "                               purge_guard.enforce, -1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0

def test_assert_raises_outside_testcase_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-6: self.assertRaises* only means the unittest API when
    the enclosing class statically descends from a *TestCase base. A
    pytest-style class with a non-asserting helper named assertRaises
    must not suppress the blocking signal; the wave-5 second-side probe
    already covers the real unittest.TestCase idiom passing."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import purge_guard\n\n"
        "class TestPurgeGuard:\n"
        "    def assertRaises(self, exc, fn):\n"
        "        return fn\n\n"
        "    def test_one(self):\n"
        "        self.assertRaises(ValueError, purge_guard.enforce)\n\n"
        "    def test_two(self):\n        assert True\n\n"
        "    def test_three(self):\n        assert True\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1

    # Second side: a project base class named *TestCase still counts.
    _write(
        tests / "test_purge_guard.py",
        "import purge_guard\n"
        "from helpers import ApiTestCase\n\n"
        "class PurgeGuardTest(ApiTestCase):\n"
        "    def test_one(self):\n        self.assertTrue(True)\n\n"
        "    def test_two(self):\n        self.assertTrue(True)\n\n"
        "    def test_rejects(self):\n"
        "        self.assertRaises(ValueError, purge_guard.enforce, -1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0

def test_aliased_pytest_import_raises_satisfies_the_gate(tmp_path: Path) -> None:
    """Codex wave-7: `import pytest as pt` + `with pt.raises(...)` is a
    real assertion and must not be misreported as absent (which would
    fail an honestly-tested sensitive module)."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import pytest as pt\n"
        "import purge_guard\n\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_rejects():\n"
        "    with pt.raises(ValueError):\n"
        "        purge_guard.enforce(-1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0


def test_unparseable_test_file_fails_closed(tmp_path: Path) -> None:
    """Codex wave-7: a matched test file with a syntax error has no
    runnable tests; a comment mentioning pytest.raises inside it must
    not suppress the blocking signal (the old text-regex fallback
    failed open)."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "# with pytest.raises(ValueError): mentioned in a comment only\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_three(:\n    assert True\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1


def test_async_only_test_file_is_not_testless(tmp_path: Path) -> None:
    """Codex wave-7 refutation lock: the test counter is unanchored, so
    `async def test_*` already counts -- an async-only suite must not be
    treated as a stub file, and its raises assertion satisfies the
    gate."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import pytest\n"
        "import purge_guard\n\n"
        "async def test_one():\n    assert True\n\n"
        "async def test_two():\n    assert True\n\n"
        "async def test_rejects():\n"
        "    with pytest.raises(ValueError):\n"
        "        purge_guard.enforce(-1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0

def test_commented_out_tests_do_not_hide_a_stub_file(tmp_path: Path) -> None:
    """Codex wave-8: the test counter reads real def/async-def AST nodes,
    so commented-out `def test_*` lines no longer make a placeholder file
    look non-stub."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import purge_guard\n\n"
        "# def test_one():\n"
        "#     assert True\n\n"
        "# def test_two():\n"
        "#     assert True\n\n"
        "HELPER = purge_guard.enforce\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1


def test_fake_assert_raises_api_name_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-8: only the exact unittest APIs count (assertRaises,
    assertRaisesRegex, assertRaisesRegexp) -- a helper like
    self.assertRaisesLater must not suppress the blocking signal."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import unittest\n"
        "import purge_guard\n\n"
        "class PurgeGuardTest(unittest.TestCase):\n"
        "    def assertRaisesLater(self, exc, fn):\n"
        "        return fn\n\n"
        "    def test_one(self):\n"
        "        self.assertRaisesLater(ValueError, purge_guard.enforce)\n\n"
        "    def test_two(self):\n        self.assertTrue(True)\n\n"
        "    def test_three(self):\n        self.assertTrue(True)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1


def test_pytest_raises_without_import_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-8: `with pytest.raises(...)` only counts when the file
    actually imports pytest -- a file that never binds the name has no
    runnable assertion (it would NameError)."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import purge_guard\n\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_rejects():\n"
        "    with pytest.raises(ValueError):\n"
        "        purge_guard.enforce(-1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1


def test_argless_with_form_raises_does_not_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-8: the with-form still needs everything but the
    callable -- `with self.assertRaisesRegex(ValueError):` (missing the
    regex) errors before asserting; the complete with-form passes."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import unittest\n"
        "import purge_guard\n\n"
        "class PurgeGuardTest(unittest.TestCase):\n"
        "    def test_one(self):\n        self.assertTrue(True)\n\n"
        "    def test_two(self):\n        self.assertTrue(True)\n\n"
        "    def test_rejects(self):\n"
        "        with self.assertRaisesRegex(ValueError):\n"
        "            purge_guard.enforce(-1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1

    # Second side: the complete with-form asserts and satisfies the gate.
    _write(
        tests / "test_purge_guard.py",
        "import unittest\n"
        "import purge_guard\n\n"
        "class PurgeGuardTest(unittest.TestCase):\n"
        "    def test_one(self):\n        self.assertTrue(True)\n\n"
        "    def test_two(self):\n        self.assertTrue(True)\n\n"
        "    def test_rejects(self):\n"
        "        with self.assertRaisesRegex(ValueError, 'negative'):\n"
        "            purge_guard.enforce(-1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0

def test_nested_test_helper_does_not_hide_a_stub_file(tmp_path: Path) -> None:
    """Codex wave-9: only module-level and class-level test defs count
    (pytest's collection shape) -- a helper named test_* nested inside
    another function does not make a placeholder file look non-stub."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import purge_guard\n\n"
        "def _make_helpers():\n"
        "    def test_inner_only():\n"
        "        return purge_guard.enforce\n"
        "    return test_inner_only\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 1

    # Second side: class-level test methods still count as collected.
    _write(
        tests / "test_purge_guard.py",
        "import pytest\n"
        "import purge_guard\n\n"
        "class TestPurgeGuard:\n"
        "    def test_one(self):\n        assert True\n\n"
        "    def test_two(self):\n        assert True\n\n"
        "    def test_rejects(self):\n"
        "        with pytest.raises(ValueError):\n"
        "            purge_guard.enforce(-1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0


def test_keyword_only_raises_arguments_satisfy_the_gate(tmp_path: Path) -> None:
    """Codex wave-9: keyword args count toward the arity check --
    `with pytest.raises(expected_exception=ValueError):` is a real
    runnable assertion and must not be misreported as absent."""
    lane = tmp_path / "lane"
    tests = tmp_path / "tests"
    _write(lane / "existing.py", "VALUE = 1\n")
    _write_reference_test(tests, "existing")
    baseline = tmp_path / "baseline.json"
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--update-baseline",
    ]) == 0

    _write(lane / "purge_guard.py",
           "def enforce(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('negative')\n"
           "    return value\n")
    _write(
        tests / "test_purge_guard.py",
        "import unittest\n"
        "import purge_guard\n\n"
        "class PurgeGuardTest(unittest.TestCase):\n"
        "    def test_one(self):\n        self.assertTrue(True)\n\n"
        "    def test_two(self):\n        self.assertTrue(True)\n\n"
        "    def test_rejects(self):\n"
        "        with self.assertRaisesRegex(ValueError, expected_regex='negative'):\n"
        "            purge_guard.enforce(-1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0

    # Second side of the arity contract: the fully-keyword pytest form
    # also counts.
    _write(
        tests / "test_purge_guard.py",
        "import pytest\n"
        "import purge_guard\n\n"
        "def test_one():\n    assert True\n\n"
        "def test_two():\n    assert True\n\n"
        "def test_rejects():\n"
        "    with pytest.raises(expected_exception=ValueError):\n"
        "        purge_guard.enforce(-1)\n",
    )
    assert MOD.main([
        str(lane),
        "--tests-root", str(tests),
        "--baseline", str(baseline),
        "--min-score", "99",
        "--sensitive-glob", "**/purge_guard.py",
    ]) == 0


# --- _has_raises_assertion: only real, runnable raises forms count ---


def _raises(body: str) -> bool:
    return MOD._has_raises_assertion("import pytest\nimport unittest\n" + body)


def _unittest_body(inner: str) -> str:
    return "class T(unittest.TestCase):\n    def test_x(self):\n" + inner


def test_raises_statement_with_only_match_keyword_does_not_assert() -> None:
    # pytest.raises(X, match=...) as a bare statement builds a context
    # manager and asserts nothing -- match= is not the callable.
    assert not _raises(
        "def test_x():\n    pytest.raises(ValueError, match='bad')\n")


def test_assert_raises_with_only_msg_keyword_does_not_assert() -> None:
    # unittest msg= fills no slot; it does not supply the exception.
    assert not _raises(_unittest_body(
        "        with self.assertRaises(msg='n'):\n            pass\n"))


def test_callable_form_inside_with_item_does_not_assert() -> None:
    # `with pytest.raises(X, fn):` runs the callable before the block and
    # yields a non-context-manager, so it asserts nothing.
    assert not _raises(
        "def test_x():\n    def fn():\n        pass\n"
        "    with pytest.raises(ValueError, fn):\n        pass\n")


def test_valid_raises_forms_still_assert() -> None:
    assert _raises(
        "def test_x():\n    with pytest.raises(ValueError):\n        go()\n")
    assert _raises(
        "def test_x():\n"
        "    with pytest.raises(expected_exception=ValueError):\n        go()\n")
    assert _raises(
        "def test_x():\n    with pytest.raises(ValueError, match='m'):\n        go()\n")
    assert _raises("def test_x():\n    pytest.raises(ValueError, go)\n")
    assert _raises(_unittest_body(
        "        with self.assertRaisesRegex(ValueError, 'r'):\n            go()\n"))
    # unittest keyword regex fills the regex slot.
    assert _raises(_unittest_body(
        "        with self.assertRaisesRegex(ValueError, expected_regex='r'):\n"
        "            go()\n"))


def test_no_raises_fires_when_only_decorative_raises_present() -> None:
    # Three happy-path tests plus a non-asserting pytest.raises statement:
    # the decorative raises must NOT suppress the blocking NO_RAISES_TESTS.
    src = (
        "import pytest\n\n"
        "def test_a():\n    assert build() == []\n\n"
        "def test_b():\n    assert count() == 0\n\n"
        "def test_c():\n    pytest.raises(ValueError, match='x')\n"
    )
    findings = MOD.score_tests(
        "sensitive_mod", {"test_sensitive_mod": src}, all_test_text=src)
    assert "NO_RAISES_TESTS" in codes(findings)


# --- _collect_test_defs: pytest's real collection shape ---


def test_helper_class_test_methods_are_not_collected() -> None:
    # A non-Test / non-TestCase class does not have its test_* methods
    # collected, so it cannot disguise a stub as covered.
    assert MOD._collect_test_defs(
        ["class Helper:\n    def test_x(self):\n        pass\n"]) == []


def test_unittest_camelcase_methods_are_collected() -> None:
    # pytest collects unittest test* (camelCase) methods on TestCase classes.
    assert MOD._collect_test_defs([
        "import unittest\n"
        "class MyTest(unittest.TestCase):\n"
        "    def testRejectsBad(self):\n        pass\n"]) == ["testRejectsBad"]


def test_collected_classes_and_module_functions_still_count() -> None:
    assert MOD._collect_test_defs(
        ["def test_foo():\n    assert True\n"]) == ["test_foo"]
    assert MOD._collect_test_defs(
        ["class TestThing:\n    def test_x(self):\n        pass\n"]) == ["test_x"]


def test_no_test_file_fires_when_only_helper_class_has_test_methods() -> None:
    # A matched test file whose only test_* defs live in an uncollected
    # helper class collects zero tests, so NO_TEST_FILE still fires.
    src = "class Helper:\n    def test_x(self):\n        pass\n"
    findings = MOD.score_tests(
        "sensitive_mod", {"test_sensitive_mod": src}, all_test_text=src)
    assert "NO_TEST_FILE" in codes(findings)


# --- wave 11: match-only raises, malformed calls, exact collection shape ---


def test_pytest_match_only_with_context_asserts() -> None:
    # Modern pytest allows `with pytest.raises(match="..."):` with no
    # exception type -- it asserts SOME matching exception is raised.
    assert _raises(
        "def test_x():\n    with pytest.raises(match='boom'):\n        go()\n")
    # ... but as a bare statement it just builds a context manager.
    assert not _raises("def test_x():\n    pytest.raises(match='boom')\n")


def test_malformed_raises_calls_do_not_assert() -> None:
    # A duplicated exception slot or an unknown keyword raises TypeError
    # before asserting -- a broken test, not a runnable assertion.
    assert not _raises(
        "def test_x():\n"
        "    with pytest.raises(ValueError, expected_exception=ValueError):\n"
        "        go()\n")
    assert not _raises(
        "def test_x():\n    with pytest.raises(ValueError, bogus=1):\n        go()\n")
    assert not _raises(_unittest_body(
        "        with self.assertRaises(ValueError, bogus=1):\n            go()\n"))


def test_module_level_camelcase_is_not_collected() -> None:
    # pytest.ini sets python_functions = test_*, so a module-level camelCase
    # function is NOT collected; only test_ counts at module scope.
    assert MOD._collect_test_defs(["def testFoo():\n    pass\n"]) == []
    assert MOD._collect_test_defs(
        ["def test_foo():\n    pass\n"]) == ["test_foo"]


def test_pytest_class_uses_underscore_prefix_and_skips_constructor() -> None:
    # A Test* pytest class collects test_ methods but NOT camelCase (that is
    # a unittest-loader idiom), and pytest skips a Test* class with __init__.
    assert MOD._collect_test_defs(
        ["class TestX:\n    def test_a(self):\n        pass\n"]) == ["test_a"]
    assert MOD._collect_test_defs(
        ["class TestX:\n    def testA(self):\n        pass\n"]) == []
    assert MOD._collect_test_defs([
        "class TestX:\n    def __init__(self):\n        pass\n"
        "    def test_a(self):\n        pass\n"]) == []


def test_unittest_testcase_collects_camelcase_and_run_test() -> None:
    # unittest's loader collects the test* prefix (incl camelCase) and the
    # default runTest, regardless of an __init__.
    assert MOD._collect_test_defs([
        "import unittest\n"
        "class M(unittest.TestCase):\n"
        "    def runTest(self):\n        pass\n"]) == ["runTest"]
    assert MOD._collect_test_defs([
        "import unittest\n"
        "class M(unittest.TestCase):\n"
        "    def testRejectsBad(self):\n        pass\n"]) == ["testRejectsBad"]


# --- wave 12: async-with, check=, __test__ opt-out, None exception ---


def test_async_with_raises_context_does_not_assert() -> None:
    # pytest.raises / assertRaises return sync context managers, so
    # `async with pytest.raises(...)` errors before the block.
    assert not _raises(
        "async def test_x():\n"
        "    async with pytest.raises(ValueError):\n        await go()\n")


def test_pytest_check_predicate_context_asserts() -> None:
    # pytest accepts the `check=` predicate, including a check-only context.
    assert _raises(
        "def test_x():\n    with pytest.raises(check=_c):\n        go()\n")
    assert _raises(
        "def test_x():\n    with pytest.raises(ValueError, check=_c):\n        go()\n")


def test_pytest_raises_none_exception_does_not_assert() -> None:
    # `with pytest.raises(None):` (no matcher) raises before the block.
    assert not _raises(
        "def test_x():\n    with pytest.raises(None):\n        go()\n")
    assert not _raises(
        "def test_x():\n"
        "    with pytest.raises(expected_exception=None):\n        go()\n")


def test_module_and_class_test_optout_collect_nothing() -> None:
    # `__test__ = False` opts a module or class out of pytest collection.
    assert MOD._collect_test_defs(
        ["__test__ = False\ndef test_a():\n    pass\n"]) == []
    assert MOD._collect_test_defs(
        ["class TestX:\n    __test__ = False\n"
         "    def test_a(self):\n        pass\n"]) == []


def test_assert_raises_regexp_variant_gets_regex_arity() -> None:
    # The explicitly accepted `assertRaisesRegexp` alias needs the regex
    # (3-arg) arity: the callable form must supply exception + regex + fn.
    assert _raises(_unittest_body(
        "        self.assertRaisesRegexp(ValueError, 'r', go)\n"))
    assert not _raises(_unittest_body(
        "        self.assertRaisesRegexp(ValueError, go)\n"))
