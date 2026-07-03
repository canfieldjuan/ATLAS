"""Deterministic fit-evaluation harness (v2 S1, #1931).

The ruler for the whole v2 arc: fixture cases plus prediction envelopes go
in, a graded machine-readable result comes out. No model calls, no network,
no store access -- pure grading, so live model slices later must satisfy
this contract instead of merely proving JSON parsing.

Grading philosophy (borrowed from the local MCP eval harness): grade the
BEHAVIOR, not whether something was returned. Model garbage arrives as a
valid envelope with ``prediction: null`` and is graded as a case failure;
a malformed input FILE is a tooling bug and fails closed with a structural
error instead. Summaries carry codes and case ids only -- never candidate
or prediction text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .fit_rules import (
    CLAIM_CODES,
    PARSE_ERROR_CODES,
    FIT_RISK_FLAGS,
    FIT_VERDICTS,
    MAX_FIT_ANGLE_CHARS,
    MAX_FIT_REASON_CHARS,
    PII_CODES,
    POSTURE_CODES,
    scan_fit_text,
)

SUMMARY_SCHEMA_VERSION = "atlas_reddit_fit_eval_summary.v1"

_PREDICTION_KEYS = frozenset({"verdict", "reason", "angle", "risk_flags"})
_WHITESPACE_RE = re.compile(r"\s+")


class FitEvalError(Exception):
    """Structural failure in cases or predictions input: fail closed."""


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    candidate: dict
    expected_verdicts: tuple[str, ...]
    required_reason_terms: tuple[str, ...] = ()
    required_angle_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    pii_allowlist: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class CaseCheck:
    case_id: str
    name: str
    passed: bool
    level: str = "error"
    codes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "name": self.name,
            "passed": self.passed,
            "level": self.level,
            "codes": list(self.codes),
        }


@dataclass
class EvalResult:
    checks: list[CaseCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    case_count: int = 0
    passed: int = 0
    failed: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors and self.failed == 0


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FitEvalError(f"cannot read {path}: {exc}") from exc
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FitEvalError(
                f"{path}:{line_no}: malformed JSONL line: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise FitEvalError(f"{path}:{line_no}: record is not an object")
        records.append(record)
    return records


def load_cases(path: Path) -> tuple[EvalCase, ...]:
    """Load and validate the fixture corpus; any structural defect fails
    closed (a broken ruler must not silently grade)."""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for record in _load_jsonl(path):
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise FitEvalError(f"{path}: case with missing/empty case_id")
        if case_id in seen:
            raise FitEvalError(f"{path}: duplicate case_id {case_id!r}")
        seen.add(case_id)
        expected = record.get("expected_verdicts")
        if (
            not isinstance(expected, list)
            or not expected
            or any(v not in FIT_VERDICTS for v in expected)
        ):
            raise FitEvalError(
                f"{path}: case {case_id!r} expected_verdicts must be a "
                f"non-empty subset of {FIT_VERDICTS}"
            )
        candidate = record.get("candidate")
        if not isinstance(candidate, dict) or not candidate.get("title"):
            raise FitEvalError(
                f"{path}: case {case_id!r} candidate must be an object with a title"
            )

        def _str_tuple(key: str) -> tuple[str, ...]:
            value = record.get(key, [])
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise FitEvalError(
                    f"{path}: case {case_id!r} {key} must be a list of strings"
                )
            if any(not item.strip() for item in value):
                # An empty term is contained in EVERY string, so the check
                # it belongs to would pass vacuously forever -- fixture rot.
                raise FitEvalError(
                    f"{path}: case {case_id!r} {key} contains an empty term"
                )
            return tuple(value)

        cases.append(
            EvalCase(
                case_id=case_id,
                category=str(record.get("category", "")),
                candidate=candidate,
                expected_verdicts=tuple(expected),
                required_reason_terms=_str_tuple("required_reason_terms"),
                required_angle_terms=_str_tuple("required_angle_terms"),
                forbidden_terms=_str_tuple("forbidden_terms"),
                pii_allowlist=_str_tuple("pii_allowlist"),
                notes=str(record.get("notes", "")),
            )
        )
    if not cases:
        raise FitEvalError(f"{path}: no cases found")
    return tuple(cases)


def load_predictions(path: Path, case_ids: frozenset[str]) -> dict[str, dict]:
    """Load prediction envelopes keyed by case_id. Duplicate or unknown
    case ids are tooling bugs, not model behavior: fail closed."""
    envelopes: dict[str, dict] = {}
    for record in _load_jsonl(path):
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise FitEvalError(f"{path}: envelope with missing/empty case_id")
        if case_id in envelopes:
            raise FitEvalError(f"{path}: duplicate prediction for {case_id!r}")
        if case_id not in case_ids:
            raise FitEvalError(f"{path}: prediction for unknown case {case_id!r}")
        if "prediction" not in record:
            # An absent key is a malformed ENVELOPE (emitter/schema bug,
            # exit 2), distinct from an explicit null (model failure,
            # graded as a case FAIL).
            raise FitEvalError(
                f"{path}: {case_id!r} envelope is missing the prediction key"
            )
        prediction = record["prediction"]
        if prediction is not None and not isinstance(prediction, dict):
            raise FitEvalError(
                f"{path}: {case_id!r} prediction must be an object or null"
            )
        parse_error = record.get("parse_error")
        if parse_error is not None and not isinstance(parse_error, str):
            # Unhashable values would crash the frozenset membership test
            # below; a non-string parse_error is a malformed envelope.
            raise FitEvalError(
                f"{path}: {case_id!r} parse_error must be a string code or null"
            )
        if parse_error is not None and prediction is not None:
            # The contract pairs parse_error with a null prediction; a
            # stale code beside a valid prediction is an emitter bug that
            # must not silently grade as a passing case.
            raise FitEvalError(
                f"{path}: {case_id!r} carries both a prediction and a "
                "parse_error; the envelope contract allows exactly one"
            )
        if prediction is None and parse_error is None:
            # The contract pairs a null prediction with a closed parse-error
            # code; omitting it is an emitter/schema regression that must
            # not hide among graded model failures.
            raise FitEvalError(
                f"{path}: {case_id!r} null prediction requires a parse_error code"
            )
        if parse_error is not None and parse_error not in PARSE_ERROR_CODES:
            # Closed taxonomy, not just a shape check: a code-SHAPED string
            # (e.g. customer_jane_doe) can smuggle content into summaries.
            raise FitEvalError(
                f"{path}: {case_id!r} parse_error must be one of the closed "
                f"parse-error codes (got a value outside PARSE_ERROR_CODES)"
            )
        envelopes[case_id] = record
    return envelopes


def check_prediction_shape(prediction: dict) -> tuple[str, ...]:
    """Deterministic strict-shape validation of the model-output object.

    S1-local twin of the S2 parser: same constants, same rejection classes,
    machine-readable snake codes. S2 swaps this for the real parser and the
    harness must stay green through it.
    """
    problems: list[str] = []
    keys = set(prediction)
    if keys - _PREDICTION_KEYS:
        problems.append("unknown_keys")
    if _PREDICTION_KEYS - keys:
        problems.append("missing_keys")
    verdict = prediction.get("verdict")
    if "verdict" in prediction and verdict not in FIT_VERDICTS:
        problems.append("verdict_invalid")
    reason = prediction.get("reason")
    if "reason" in prediction:
        if not isinstance(reason, str) or not _collapse(reason):
            problems.append("reason_missing")
        elif len(_collapse(reason)) > MAX_FIT_REASON_CHARS:
            problems.append("reason_too_long")
    angle = prediction.get("angle")
    if "angle" in prediction and "verdict" in prediction and verdict in FIT_VERDICTS:
        if verdict in ("yes", "maybe"):
            if not isinstance(angle, str) or not _collapse(angle):
                problems.append("angle_required")
            elif len(_collapse(angle)) > MAX_FIT_ANGLE_CHARS:
                problems.append("angle_too_long")
        else:  # verdict == "no": advisory text on a rejected thread is
            # exactly where pitch language leaks; require null/empty.
            if angle is not None and _collapse(str(angle)):
                problems.append("angle_forbidden_for_no")
    flags = prediction.get("risk_flags")
    if "risk_flags" in prediction:
        if not isinstance(flags, list):
            problems.append("risk_flags_invalid")
        elif any(not isinstance(flag, str) for flag in flags):
            # Non-string elements are model garbage too: grade the case,
            # never crash on an unhashable value before dedup.
            problems.append("risk_flags_invalid")
        else:
            if any(flag not in FIT_RISK_FLAGS for flag in flags):
                problems.append("risk_flag_unknown")
            if len(set(flags)) != len(flags):
                problems.append("risk_flag_duplicate")
    return tuple(problems)


def _advisory_fields(prediction: dict) -> tuple[str, str]:
    """Reason and angle, collapsed, as SEPARATE strings: concatenating them
    would blind every start-anchored rule (e.g. reply greetings) to the
    angle, which is exactly where reply drafts appear."""
    reason = prediction.get("reason") or ""
    angle = prediction.get("angle") or ""
    return _collapse(str(reason)), _collapse(str(angle))


def _grade_case(case: EvalCase, envelope: dict | None) -> list[CaseCheck]:
    checks: list[CaseCheck] = []

    def _add(name: str, passed: bool, codes: tuple[str, ...] = ()) -> None:
        checks.append(CaseCheck(case.case_id, name, passed, codes=codes))

    if envelope is None:
        _add("prediction_present", False, ("prediction_missing",))
        return checks
    _add("prediction_present", True)

    prediction = envelope.get("prediction")
    if prediction is None:
        # load_predictions guarantees a closed-taxonomy code here.
        _add("model_output_parses", False, (envelope["parse_error"],))
        return checks
    problems = check_prediction_shape(prediction)
    if problems:
        _add("model_output_parses", False, problems)
        return checks
    _add("model_output_parses", True)

    verdict = prediction["verdict"]
    _add(
        "verdict_allowed",
        verdict in case.expected_verdicts,
        () if verdict in case.expected_verdicts else (f"verdict_{verdict}",),
    )

    reason_text, angle_text = _advisory_fields(prediction)
    reason = reason_text.lower()
    angle = angle_text.lower()

    # Term checks emit POSITIONAL codes (index into the case's public
    # fixture lists), never the raw term text: a candidate-specific term
    # in a future fixture must not leak into the privacy-stripped summary.
    missing_reason = tuple(
        f"reason_term_{index}"
        for index, term in enumerate(case.required_reason_terms)
        if term.lower() not in reason
    )
    _add("reason_grounded", not missing_reason, missing_reason)

    if angle and case.required_angle_terms:
        missing_angle = tuple(
            f"angle_term_{index}"
            for index, term in enumerate(case.required_angle_terms)
            if term.lower() not in angle
        )
        _add("angle_grounded", not missing_angle, missing_angle)

    combined = f"{reason} {angle}"
    hit_terms = tuple(
        f"forbidden_term_{index}"
        for index, term in enumerate(case.forbidden_terms)
        if term.lower() in combined
    )
    _add("no_forbidden_terms", not hit_terms, hit_terms)

    allowlist = frozenset(case.pii_allowlist)
    findings = scan_fit_text(reason_text, pii_allowlist=allowlist) + scan_fit_text(
        angle_text, pii_allowlist=allowlist
    )
    claim_codes = tuple(
        sorted({f.code for f in findings if f.code in CLAIM_CODES})
    )
    _add("no_forbidden_claims", not claim_codes, claim_codes)
    posture_codes = tuple(
        sorted({f.code for f in findings if f.code in POSTURE_CODES})
    )
    _add("no_reply_draft", not posture_codes, posture_codes)
    pii_codes = tuple(sorted({f.code for f in findings if f.code in PII_CODES}))
    _add("no_pii_echo", not pii_codes, pii_codes)
    return checks


def evaluate_predictions(
    cases: tuple[EvalCase, ...], envelopes: dict[str, dict]
) -> EvalResult:
    result = EvalResult(case_count=len(cases))
    for case in cases:
        case_checks = _grade_case(case, envelopes.get(case.case_id))
        result.checks.extend(case_checks)
        if all(check.passed for check in case_checks):
            result.passed += 1
        else:
            result.failed += 1
    return result


def summarize_result(result: EvalResult) -> dict:
    """Machine-readable summary. Privacy rule: failing checks carry case
    ids, check names, and stable codes/term labels only -- never candidate
    or prediction text."""
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "ok": result.ok,
        "case_count": result.case_count,
        "passed": result.passed,
        "failed": result.failed,
        "checks": [
            check.as_dict() for check in result.checks if not check.passed
        ],
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate_atlas_reddit_fit",
        description=(
            "Grade fit predictions against the fixture corpus. Deterministic; "
            "no model calls, no network."
        ),
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Write the machine-readable summary JSON here.",
    )
    parser.add_argument(
        "--fail-on-eval-fail",
        action="store_true",
        help="Exit 1 when any case fails. Default exits 0 after grading.",
    )
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.cases)
        envelopes = load_predictions(
            args.predictions, frozenset(case.case_id for case in cases)
        )
        result = evaluate_predictions(cases, envelopes)
        summary = summarize_result(result)
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (FitEvalError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"fit eval: {result.passed}/{result.case_count} cases passed"
        + ("" if result.ok else f" ({result.failed} failed)")
    )
    if args.fail_on_eval_fail and not result.ok:
        return 1
    return 0
