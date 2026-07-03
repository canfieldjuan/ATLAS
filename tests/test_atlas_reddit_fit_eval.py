"""Fit evaluation harness tests (v2 S1, #1931).

The harness is pure -- no external boundary exists, so nothing is faked:
real rule catalogue, real fixture files, real CLI main() in-process. The
shipped fixture corpus is itself under test: every fail-file envelope
declares the exact checks and codes it must fire, and this suite enforces
that contract so the corpus cannot silently rot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_reddit.fit_eval import (
    FitEvalError,
    check_prediction_shape,
    evaluate_predictions,
    load_cases,
    load_predictions,
    main,
    summarize_result,
)
from atlas_reddit.fit_rules import (
    ALL_RULE_CODES,
    CLAIM_CODES,
    PII_CODES,
    POSTURE_CODES,
    RULES,
    scan_fit_text,
)

FIXTURES = Path(__file__).parent / "fixtures" / "atlas_reddit_fit_eval"
CASES = FIXTURES / "cases.jsonl"
PASS_FILE = FIXTURES / "predictions_pass.jsonl"
FAIL_FILE = FIXTURES / "predictions_fail.jsonl"


def _prediction(**overrides) -> dict:
    base = {
        "verdict": "yes",
        "reason": "They describe repeated support questions.",
        "angle": "Ask what their ticket evidence shows.",
        "risk_flags": [],
    }
    base.update(overrides)
    return base


# -- rule catalogue ---------------------------------------------------------


def test_catalogue_partitions_into_three_families() -> None:
    assert CLAIM_CODES | POSTURE_CODES | PII_CODES == ALL_RULE_CODES
    assert not (CLAIM_CODES & POSTURE_CODES)
    assert not (CLAIM_CODES & PII_CODES)
    assert not (POSTURE_CODES & PII_CODES)
    assert len({rule.code for rule in RULES}) == len(RULES)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("we guarantee a 40% reduction in tickets", "GUARANTEED_DEFLECTION"),
        ("cut their support tickets by 30", "GUARANTEED_DEFLECTION"),
        ("ticket volume will drop once this ships", "TICKET_REDUCTION_PROMISE"),
        ("clear ROI within a quarter", "ROI_SAVINGS"),
        ("hours saved every single week", "ROI_SAVINGS"),
        ("customers will churn less after this", "RETENTION_CHURN_OUTCOME"),
        ("you will rank higher on search", "RANKING_SEO_OUTCOME"),
        ("we can fix this for them", "FIX_RESOLVE_PROMISE"),
        ("it can auto-publish to the help center", "AUTO_PUBLISH"),
        ("we connect to Zendesk out of the box", "LIVE_HELPDESK_INTEGRATION"),
        ("semantic clustering groups the tickets", "SEMANTIC_CLUSTERING"),
        ("we rank by cost per ticket", "COST_RANKING"),
        ("unlimited uploads on every plan", "UNBOUNDED_HOSTED_UPLOADS"),
        ("our tool handles this and has a free trial", "SELF_PROMO_PITCH"),
        ("Hey OP, here is what worked for us", "REPLY_DRAFT"),
        ("I'd say you should start with the docs", "REPLY_DRAFT"),
        ("feel free to DM me anytime", "REPLY_DRAFT"),
        ("post this as a comment on the thread", "WRITE_ACTION_POSTURE"),
        ("reach her at jane.doe@example.com", "PII_EMAIL"),
        ("call them on (555) 123-4567", "PII_PHONE"),
        ("SSN 123-45-6789 appears in the ticket", "PII_SSN"),
        ("card 4111 1111 1111 1111 was quoted", "PII_PAYMENT_CARD"),
        ("the customer name is Jane Doe", "PII_PERSON_NAME"),
        ("order number: AB-12345 keeps failing", "PII_IDENTIFIER"),
    ],
)
def test_each_rule_family_fires(text: str, code: str) -> None:
    assert code in {finding.code for finding in scan_fit_text(text)}


def test_clean_advisory_text_passes_the_whole_catalogue() -> None:
    text = (
        "They describe repeated onboarding questions arriving despite "
        "existing documentation. Ask what their ticket history shows about "
        "which questions keep coming back; that evidence is worth an audit."
    )
    assert scan_fit_text(text) == ()


def test_pii_allowlist_suppresses_only_the_allowlisted_span() -> None:
    text = "Compare what reaches support@acme-widgets.com against jane@personal.example"
    codes = [f.code for f in scan_fit_text(text)]
    assert codes.count("PII_EMAIL") == 2
    allowed = scan_fit_text(
        text, pii_allowlist=frozenset({"support@acme-widgets.com"})
    )
    assert [f.code for f in allowed].count("PII_EMAIL") == 1


def test_findings_carry_spans_not_matched_text() -> None:
    finding = scan_fit_text("email me at jane.doe@example.com")[0]
    assert "jane.doe" not in finding.message
    assert (finding.start, finding.end) != (0, 0)
    assert not hasattr(finding, "match")


# -- prediction shape (S1-local twin of the S2 parser) -----------------------


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"confidence": 0.9}, "unknown_keys"),
        ({"verdict": "definitely"}, "verdict_invalid"),
        ({"reason": "   "}, "reason_missing"),
        ({"reason": "x" * 281}, "reason_too_long"),
        ({"angle": None}, "angle_required"),
        ({"angle": "x" * 281}, "angle_too_long"),
        ({"risk_flags": "promo_risk"}, "risk_flags_invalid"),
        ({"risk_flags": ["made_up_flag"]}, "risk_flag_unknown"),
        ({"risk_flags": ["pii_risk", "pii_risk"]}, "risk_flag_duplicate"),
    ],
)
def test_shape_checker_rejects_each_malformation(
    mutation: dict, expected_code: str
) -> None:
    assert expected_code in check_prediction_shape(_prediction(**mutation))


def test_shape_checker_missing_keys_and_no_with_angle() -> None:
    assert "missing_keys" in check_prediction_shape({"verdict": "yes"})
    bad = _prediction(verdict="no", angle="but here is an angle anyway")
    assert "angle_forbidden_for_no" in check_prediction_shape(bad)
    good = _prediction(verdict="no", angle=None)
    assert check_prediction_shape(good) == ()


# -- the shipped corpus is the contract ---------------------------------------


def test_pass_corpus_grades_fully_green() -> None:
    cases = load_cases(CASES)
    envelopes = load_predictions(
        PASS_FILE, frozenset(c.case_id for c in cases)
    )
    result = evaluate_predictions(cases, envelopes)
    assert result.ok
    assert (result.case_count, result.passed, result.failed) == (16, 16, 0)


def test_fail_corpus_fires_exactly_the_declared_checks_and_codes() -> None:
    """Every trap envelope declares its expected failing checks and codes;
    the harness must fire exactly those -- no more, no fewer."""
    cases = load_cases(CASES)
    envelopes = load_predictions(
        FAIL_FILE, frozenset(c.case_id for c in cases)
    )
    result = evaluate_predictions(cases, envelopes)
    assert result.failed == 16
    failing: dict[str, dict[str, list[str]]] = {}
    for check in result.checks:
        if not check.passed:
            failing.setdefault(check.case_id, {})[check.name] = list(check.codes)
    for envelope in envelopes.values():
        case_id = envelope["case_id"]
        got = failing.get(case_id, {})
        assert set(got) == set(envelope["expects_failing_checks"]), case_id
        got_codes = {code for codes in got.values() for code in codes}
        for code in envelope["expects_codes"]:
            assert code in got_codes, f"{case_id}: expected code {code}"


def test_corpus_covers_all_eight_categories_twice() -> None:
    cases = load_cases(CASES)
    categories: dict[str, int] = {}
    for case in cases:
        categories[case.category] = categories.get(case.category, 0) + 1
    assert set(categories) == {
        "obvious_fit", "process_gap_fit", "maybe_fit", "no_fit",
        "vendor_bait_trap", "unsupported_outcome_trap", "reply_draft_trap",
        "privacy_trap",
    }
    assert all(count == 2 for count in categories.values())


def test_missing_prediction_fails_the_case(tmp_path: Path) -> None:
    cases = load_cases(CASES)
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        PASS_FILE.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    envelopes = load_predictions(partial, frozenset(c.case_id for c in cases))
    result = evaluate_predictions(cases, envelopes)
    assert result.failed == 15
    missing = [
        check for check in result.checks
        if check.name == "prediction_present" and not check.passed
    ]
    assert len(missing) == 15


# -- structural failures are exit-2 material, not grades ----------------------


def test_duplicate_case_id_fails_closed(tmp_path: Path) -> None:
    line = CASES.read_text(encoding="utf-8").splitlines()[0]
    doubled = tmp_path / "cases.jsonl"
    doubled.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(FitEvalError, match="duplicate case_id"):
        load_cases(doubled)


def test_duplicate_and_unknown_predictions_fail_closed(tmp_path: Path) -> None:
    line = PASS_FILE.read_text(encoding="utf-8").splitlines()[0]
    case_id = json.loads(line)["case_id"]
    doubled = tmp_path / "p.jsonl"
    doubled.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(FitEvalError, match="duplicate prediction"):
        load_predictions(doubled, frozenset({case_id}))
    with pytest.raises(FitEvalError, match="unknown case"):
        load_predictions(doubled.parent / "p.jsonl", frozenset({"t3_other"}))


def test_malformed_jsonl_line_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"case_id": "x"\n', encoding="utf-8")
    with pytest.raises(FitEvalError, match="malformed JSONL"):
        load_cases(bad)


def test_unknown_expected_verdict_fails_closed(tmp_path: Path) -> None:
    record = json.loads(CASES.read_text(encoding="utf-8").splitlines()[0])
    record["expected_verdicts"] = ["definitely"]
    bad = tmp_path / "cases.jsonl"
    bad.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(FitEvalError, match="expected_verdicts"):
        load_cases(bad)


# -- CLI: the real entrypoint, in-process --------------------------------------


def test_cli_pass_file_exits_zero(tmp_path: Path, capsys) -> None:
    code = main([
        "--cases", str(CASES), "--predictions", str(PASS_FILE),
        "--summary-output", str(tmp_path / "s.json"), "--fail-on-eval-fail",
    ])
    assert code == 0
    assert "16/16 cases passed" in capsys.readouterr().out
    summary = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["schema_version"] == "atlas_reddit_fit_eval_summary.v1"
    assert summary["checks"] == []  # failures only; a green run carries none


def test_cli_fail_file_exit_gated_by_flag(tmp_path: Path) -> None:
    argv = [
        "--cases", str(CASES), "--predictions", str(FAIL_FILE),
        "--summary-output", str(tmp_path / "s.json"),
    ]
    assert main(argv) == 0  # artifacts always written; flag gates the exit
    assert main([*argv, "--fail-on-eval-fail"]) == 1


def test_cli_structural_error_exits_two(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "nope.jsonl"
    code = main(["--cases", str(missing), "--predictions", str(PASS_FILE)])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_summary_is_privacy_stripped(tmp_path: Path) -> None:
    """The privacy trap FAILS on the fail corpus -- and the PII it caught
    must still never appear in the summary artifact."""
    out = tmp_path / "s.json"
    main([
        "--cases", str(CASES), "--predictions", str(FAIL_FILE),
        "--summary-output", str(out),
    ])
    raw = out.read_text(encoding="utf-8")
    assert "jane.doe@example.com" not in raw
    assert "555" not in raw
    assert "Customers keep opening tickets" not in raw  # no candidate text
    assert "PII_EMAIL" in raw  # codes are the only payload


def test_summary_counts_match_result() -> None:
    cases = load_cases(CASES)
    envelopes = load_predictions(
        FAIL_FILE, frozenset(c.case_id for c in cases)
    )
    summary = summarize_result(evaluate_predictions(cases, envelopes))
    assert summary["case_count"] == 16
    assert summary["passed"] + summary["failed"] == 16
    assert all(not check["passed"] for check in summary["checks"])


# -- purity ---------------------------------------------------------------------


def test_harness_modules_have_no_network_or_reddit_imports() -> None:
    package = Path(__file__).parent.parent / "atlas_reddit"
    for name in ("fit_rules.py", "fit_eval.py"):
        source = (package / name).read_text(encoding="utf-8")
        for banned in ("urllib", "http.client", "socket", "requests", "praw"):
            assert banned not in source, f"{name} imports {banned}"
