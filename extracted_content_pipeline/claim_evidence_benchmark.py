"""Deterministic benchmark scoring for claim/evidence support checks.

This module is intentionally pure. It does not call a model, read the claim
registry, or expose an MCP tool. It only scores labeled benchmark triples and
structured witness responses against the reliability thresholds from issue
#1435.
"""

from __future__ import annotations

import json
from collections.abc import Sequence as RuntimeSequence
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Mapping, Sequence


EASY = "easy"
HARD = "hard"
VALID_DIFFICULTIES = frozenset({EASY, HARD})
CONFIDENCE_COUNTS_MIN = 4
FINAL_EASY_SUPPORTS_TARGET = 15
FINAL_EASY_NOT_SUPPORTS_TARGET = 15
FINAL_HARD_TARGET = 10
FIXTURE_FORMAT_JSON = "json"
FIXTURE_FORMAT_JSONL = "jsonl"
VALID_FIXTURE_FORMATS = frozenset({FIXTURE_FORMAT_JSON, FIXTURE_FORMAT_JSONL})
VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION = "verify_claim_evidence.v1"
CLAIM_EVIDENCE_RESPONSE_FIELDS = frozenset({"supports", "confidence", "reason"})


def _required_text(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _required_bool(data: Mapping[str, object], key: str) -> bool | None:
    value = data.get(key)
    if isinstance(value, bool):
        return value
    return None


def _required_confidence(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5:
        return value
    return None


def _accuracy(correct: int, total: int) -> float | None:
    if total == 0:
        return None
    return correct / total


@dataclass(frozen=True)
class ClaimEvidenceTriple:
    """One hand-labeled benchmark item."""

    triple_id: str
    claim_id: str
    claim_text: str
    evidence_quote: str
    source_id: str
    expected_supports: bool
    difficulty: str

    @classmethod
    def from_mapping(
        cls, data: object
    ) -> tuple["ClaimEvidenceTriple | None", tuple[str, ...]]:
        if not isinstance(data, Mapping):
            return None, ("triple must be an object",)

        errors: list[str] = []
        triple_id = _required_text(data, "triple_id")
        claim_id = _required_text(data, "claim_id")
        claim_text = _required_text(data, "claim_text")
        evidence_quote = _required_text(data, "evidence_quote")
        source_id = _required_text(data, "source_id")
        expected_supports = _required_bool(data, "expected_supports")
        difficulty = _required_text(data, "difficulty")

        for key, value in (
            ("triple_id", triple_id),
            ("claim_id", claim_id),
            ("claim_text", claim_text),
            ("evidence_quote", evidence_quote),
            ("source_id", source_id),
        ):
            if value is None:
                errors.append(f"{key} missing")
        if expected_supports is None:
            errors.append("expected_supports missing")
        if difficulty not in VALID_DIFFICULTIES:
            errors.append("difficulty must be easy or hard")
        if errors:
            return None, tuple(errors)

        return (
            cls(
                triple_id=triple_id or "",
                claim_id=claim_id or "",
                claim_text=claim_text or "",
                evidence_quote=evidence_quote or "",
                source_id=source_id or "",
                expected_supports=bool(expected_supports),
                difficulty=difficulty or "",
            ),
            (),
        )


@dataclass(frozen=True)
class ClaimEvidenceResponse:
    """Structured model-witness response for one benchmark item."""

    supports: bool
    confidence: int
    reason: str

    @classmethod
    def from_mapping(
        cls, data: object
    ) -> tuple["ClaimEvidenceResponse | None", tuple[str, ...]]:
        if not isinstance(data, Mapping):
            return None, ("response must be an object",)

        errors: list[str] = []
        unexpected_fields = sorted(
            str(key) for key in data.keys() if key not in CLAIM_EVIDENCE_RESPONSE_FIELDS
        )
        if unexpected_fields:
            errors.append(f"unexpected fields: {', '.join(unexpected_fields)}")
        supports = _required_bool(data, "supports")
        confidence = _required_confidence(data, "confidence")
        reason = _required_text(data, "reason")
        if supports is None:
            errors.append("supports missing")
        if confidence is None:
            errors.append("confidence must be an integer from 1 to 5")
        if reason is None:
            errors.append("reason missing")
        if errors:
            return None, tuple(errors)

        return (
            cls(
                supports=bool(supports),
                confidence=int(confidence or 0),
                reason=reason or "",
            ),
            (),
        )


@dataclass(frozen=True)
class ClaimEvidencePromptContract:
    """Provider-neutral prompt and response schema for one witness call."""

    contract_version: str
    prompt: str
    response_schema: Mapping[str, object]


ClaimEvidenceProvider = Callable[
    [str, ClaimEvidenceTriple, ClaimEvidencePromptContract], object
]


def claim_evidence_response_json_schema() -> dict[str, object]:
    """Return the strict JSON Schema for structured witness responses."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["supports", "confidence", "reason"],
        "properties": {
            "supports": {
                "type": "boolean",
                "description": (
                    "True only when the provided evidence quote supports the "
                    "claim under test."
                ),
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": (
                    "Confidence in the support judgment. Use 4 or 5 only for "
                    "clear support or clear non-support."
                ),
            },
            "reason": {
                "type": "string",
                "pattern": "\\S",
                "description": (
                    "Short rationale grounded only in the evidence quote."
                ),
            },
        },
    }


def build_claim_evidence_prompt_contract(
    triple: ClaimEvidenceTriple,
) -> ClaimEvidencePromptContract:
    """Build the deterministic prompt/schema contract for one benchmark triple."""

    prompt = "\n".join(
        (
            f"Contract: {VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION}",
            "Task: decide whether the evidence quote supports the claim under test.",
            "Judge support only from the provided evidence quote.",
            "Do not decide whether the claim is true in general.",
            "Do not use outside knowledge or infer facts not present in the quote.",
            "Return only JSON that matches the response schema.",
            "",
            f"Claim: {triple.claim_text}",
            f"Claim id: {triple.claim_id}",
            f"Source id: {triple.source_id}",
            f"Difficulty bucket: {triple.difficulty}",
            "Evidence quote:",
            triple.evidence_quote,
        )
    )
    return ClaimEvidencePromptContract(
        contract_version=VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION,
        prompt=prompt,
        response_schema=claim_evidence_response_json_schema(),
    )


@dataclass(frozen=True)
class ClaimEvidenceRunRow:
    """One provider response attempt for one benchmark triple."""

    model_id: str
    triple_id: str
    contract_version: str
    response: ClaimEvidenceResponse | None
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.response is not None and not self.errors


@dataclass(frozen=True)
class ClaimEvidenceModelRun:
    """Decoded provider run for one model across benchmark triples."""

    model_id: str
    rows: tuple[ClaimEvidenceRunRow, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors and all(row.ok for row in self.rows)

    @property
    def responses_by_triple_id(self) -> dict[str, ClaimEvidenceResponse]:
        return {
            row.triple_id: row.response
            for row in self.rows
            if row.response is not None and not row.errors
        }


def run_claim_evidence_provider(
    model_id: object,
    triples: Sequence[object],
    provider: object,
) -> ClaimEvidenceModelRun:
    """Run the prompt/schema contract through an injected provider boundary."""

    normalized_model_id = (
        model_id.strip() if isinstance(model_id, str) and model_id.strip() else None
    )
    if normalized_model_id is None:
        return ClaimEvidenceModelRun("", (), ("model_id missing",))
    if not callable(provider):
        return ClaimEvidenceModelRun(
            normalized_model_id,
            (),
            ("provider must be callable",),
        )
    if not isinstance(triples, RuntimeSequence) or isinstance(triples, (str, bytes)):
        return ClaimEvidenceModelRun(
            normalized_model_id,
            (),
            ("triples must be a sequence",),
        )
    if not triples:
        return ClaimEvidenceModelRun(
            normalized_model_id,
            (),
            ("triples missing",),
        )

    run_errors: list[str] = []
    rows: list[ClaimEvidenceRunRow] = []
    typed_provider = provider
    for index, triple in enumerate(triples, start=1):
        if not isinstance(triple, ClaimEvidenceTriple):
            run_errors.append(f"row {index}: triple must be ClaimEvidenceTriple")
            continue

        contract = build_claim_evidence_prompt_contract(triple)
        try:
            raw_response = typed_provider(normalized_model_id, triple, contract)
        except Exception as error:
            rows.append(
                ClaimEvidenceRunRow(
                    model_id=normalized_model_id,
                    triple_id=triple.triple_id,
                    contract_version=contract.contract_version,
                    response=None,
                    errors=(f"provider error: {error.__class__.__name__}",),
                )
            )
            continue

        response, response_errors = ClaimEvidenceResponse.from_mapping(raw_response)
        rows.append(
            ClaimEvidenceRunRow(
                model_id=normalized_model_id,
                triple_id=triple.triple_id,
                contract_version=contract.contract_version,
                response=response,
                errors=response_errors,
            )
        )

    return ClaimEvidenceModelRun(
        normalized_model_id,
        tuple(rows),
        tuple(run_errors),
    )


@dataclass(frozen=True)
class BenchmarkThresholds:
    easy_accuracy_min: float = 0.95
    hard_accuracy_min: float = 0.75
    inter_model_agreement_min: float = 0.85
    intra_model_stability_min: float = 0.95
    high_confidence_accuracy_min: float = 0.90


DEFAULT_THRESHOLDS = BenchmarkThresholds()


@dataclass(frozen=True)
class ModelScore:
    model_id: str
    easy_accuracy: float | None
    hard_accuracy: float | None
    high_confidence_accuracy: float | None
    easy_total: int
    hard_total: int
    high_confidence_total: int
    missing_response_ids: tuple[str, ...]
    low_confidence_response_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseAgreement:
    left_model_id: str
    right_model_id: str
    agreement: float | None
    total: int


@dataclass(frozen=True)
class StabilityScore:
    model_id: str
    stability: float | None
    total: int


@dataclass(frozen=True)
class BenchmarkVerdict:
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkFixture:
    """Decoded operator-labeled benchmark fixture rows."""

    triples: tuple[ClaimEvidenceTriple, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def easy_supports_count(self) -> int:
        return sum(
            1
            for triple in self.triples
            if triple.difficulty == EASY and triple.expected_supports
        )

    @property
    def easy_not_supports_count(self) -> int:
        return sum(
            1
            for triple in self.triples
            if triple.difficulty == EASY and not triple.expected_supports
        )

    @property
    def hard_count(self) -> int:
        return sum(1 for triple in self.triples if triple.difficulty == HARD)


def validate_claim_evidence_fixture(
    rows: object,
    *,
    require_final_shape: bool = False,
) -> BenchmarkFixture:
    """Validate decoded operator-labeled benchmark rows.

    ``require_final_shape=False`` supports early seed sets. Final benchmark
    validation enforces the #1435 15/15/10 composition.
    """

    if not isinstance(rows, list):
        return BenchmarkFixture((), ("fixture rows must be a list of objects",))

    triples: list[ClaimEvidenceTriple] = []
    errors: list[str] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        triple, row_errors = ClaimEvidenceTriple.from_mapping(row)
        if row_errors:
            errors.extend(f"row {index}: {error}" for error in row_errors)
            continue
        assert triple is not None
        previous_index = seen.get(triple.triple_id)
        if previous_index is not None:
            errors.append(
                f"row {index}: triple_id duplicated from row {previous_index}: "
                f"{triple.triple_id}"
            )
            continue
        seen[triple.triple_id] = index
        triples.append(triple)

    fixture = BenchmarkFixture(tuple(triples), tuple(errors))
    if require_final_shape and fixture.ok:
        errors.extend(_final_shape_errors(fixture))
        fixture = BenchmarkFixture(fixture.triples, tuple(errors))
    return fixture


def load_claim_evidence_fixture_text(
    text: object,
    *,
    source_format: object = FIXTURE_FORMAT_JSON,
    require_final_shape: bool = False,
) -> BenchmarkFixture:
    """Load raw fixture text into the decoded benchmark fixture contract."""

    if not isinstance(text, str):
        return BenchmarkFixture((), ("fixture text must be a string",))
    if not isinstance(source_format, str):
        return BenchmarkFixture((), ("fixture format must be json or jsonl",))

    normalized_format = source_format.strip().lower()
    if normalized_format not in VALID_FIXTURE_FORMATS:
        return BenchmarkFixture((), ("fixture format must be json or jsonl",))

    if normalized_format == FIXTURE_FORMAT_JSON:
        rows, errors = _decode_json_fixture_text(text)
    else:
        rows, errors = _decode_jsonl_fixture_text(text)
    if errors:
        return BenchmarkFixture((), errors)
    return validate_claim_evidence_fixture(
        rows,
        require_final_shape=require_final_shape,
    )


def _decode_json_fixture_text(text: str) -> tuple[list[object], tuple[str, ...]]:
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        return [], (f"json fixture is malformed: {error.msg}",)
    if not isinstance(decoded, list):
        return [], ("json fixture must decode to a list of objects",)
    return decoded, ()


def _decode_jsonl_fixture_text(text: str) -> tuple[list[object], tuple[str, ...]]:
    rows: list[object] = []
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as error:
            errors.append(
                f"line {line_number}: jsonl fixture is malformed: {error.msg}"
            )
            continue
        if not isinstance(decoded, Mapping):
            errors.append(f"line {line_number}: jsonl fixture line must be an object")
            continue
        rows.append(decoded)
    if errors:
        return [], tuple(errors)
    return rows, ()


def _final_shape_errors(fixture: BenchmarkFixture) -> tuple[str, ...]:
    errors: list[str] = []
    if fixture.easy_supports_count != FINAL_EASY_SUPPORTS_TARGET:
        errors.append(
            "final fixture requires "
            f"{FINAL_EASY_SUPPORTS_TARGET} easy support rows; got "
            f"{fixture.easy_supports_count}"
        )
    if fixture.easy_not_supports_count != FINAL_EASY_NOT_SUPPORTS_TARGET:
        errors.append(
            "final fixture requires "
            f"{FINAL_EASY_NOT_SUPPORTS_TARGET} easy non-support rows; got "
            f"{fixture.easy_not_supports_count}"
        )
    if fixture.hard_count != FINAL_HARD_TARGET:
        errors.append(
            f"final fixture requires {FINAL_HARD_TARGET} hard rows; got "
            f"{fixture.hard_count}"
        )
    return tuple(errors)


def score_model(
    model_id: str,
    triples: Sequence[ClaimEvidenceTriple],
    responses: Mapping[str, ClaimEvidenceResponse],
) -> ModelScore:
    easy_total = hard_total = high_total = 0
    easy_correct = hard_correct = high_correct = 0
    missing: list[str] = []
    low_confidence: list[str] = []

    for triple in triples:
        if triple.difficulty == EASY:
            easy_total += 1
        elif triple.difficulty == HARD:
            hard_total += 1

        response = responses.get(triple.triple_id)
        if response is None:
            missing.append(triple.triple_id)
            continue

        correct = response.supports == triple.expected_supports
        if triple.difficulty == EASY and correct:
            easy_correct += 1
        elif triple.difficulty == HARD and correct:
            hard_correct += 1

        if response.confidence >= CONFIDENCE_COUNTS_MIN:
            high_total += 1
            if correct:
                high_correct += 1
        else:
            low_confidence.append(triple.triple_id)

    return ModelScore(
        model_id=model_id,
        easy_accuracy=_accuracy(easy_correct, easy_total),
        hard_accuracy=_accuracy(hard_correct, hard_total),
        high_confidence_accuracy=_accuracy(high_correct, high_total),
        easy_total=easy_total,
        hard_total=hard_total,
        high_confidence_total=high_total,
        missing_response_ids=tuple(missing),
        low_confidence_response_ids=tuple(low_confidence),
    )


def pairwise_agreements(
    triples: Sequence[ClaimEvidenceTriple],
    model_responses: Mapping[str, Mapping[str, ClaimEvidenceResponse]],
) -> tuple[PairwiseAgreement, ...]:
    agreements: list[PairwiseAgreement] = []
    for left, right in combinations(sorted(model_responses), 2):
        total = len(triples)
        matching = 0
        for triple in triples:
            left_response = model_responses[left].get(triple.triple_id)
            right_response = model_responses[right].get(triple.triple_id)
            if (
                left_response is not None
                and right_response is not None
                and left_response.supports == right_response.supports
            ):
                matching += 1
        agreements.append(
            PairwiseAgreement(left, right, _accuracy(matching, total), total)
        )
    return tuple(agreements)


def intra_model_stability(
    model_id: str,
    triples: Sequence[ClaimEvidenceTriple],
    runs: Sequence[Mapping[str, ClaimEvidenceResponse]],
) -> StabilityScore:
    if len(runs) < 2:
        return StabilityScore(model_id=model_id, stability=None, total=0)

    total = 0
    identical = 0
    for left_run, right_run in combinations(runs, 2):
        for triple in triples:
            total += 1
            left_response = left_run.get(triple.triple_id)
            right_response = right_run.get(triple.triple_id)
            if (
                left_response is not None
                and right_response is not None
                and left_response.supports == right_response.supports
            ):
                identical += 1
    return StabilityScore(
        model_id=model_id,
        stability=_accuracy(identical, total),
        total=total,
    )


def evaluate_thresholds(
    model_scores: Sequence[ModelScore],
    agreements: Sequence[PairwiseAgreement],
    stability_scores: Sequence[StabilityScore],
    thresholds: BenchmarkThresholds = DEFAULT_THRESHOLDS,
) -> BenchmarkVerdict:
    reasons: list[str] = []
    if not model_scores:
        reasons.append("no model scores")
    for score in model_scores:
        _require_metric(
            reasons,
            score.model_id,
            "easy accuracy",
            score.easy_accuracy,
            thresholds.easy_accuracy_min,
        )
        _require_metric(
            reasons,
            score.model_id,
            "hard accuracy",
            score.hard_accuracy,
            thresholds.hard_accuracy_min,
        )
        _require_metric(
            reasons,
            score.model_id,
            "high-confidence accuracy",
            score.high_confidence_accuracy,
            thresholds.high_confidence_accuracy_min,
            strict=True,
        )
        if score.missing_response_ids:
            reasons.append(f"{score.model_id}: missing responses")

    _require_agreement_coverage(reasons, model_scores, agreements, thresholds)
    _require_stability_coverage(reasons, model_scores, stability_scores, thresholds)
    return BenchmarkVerdict(passed=not reasons, failure_reasons=tuple(reasons))


def _require_agreement_coverage(
    reasons: list[str],
    model_scores: Sequence[ModelScore],
    agreements: Sequence[PairwiseAgreement],
    thresholds: BenchmarkThresholds,
) -> None:
    if len(model_scores) < 2:
        reasons.append("inter-model agreement missing")
        return

    expected_pairs = {
        frozenset((left.model_id, right.model_id))
        for left, right in combinations(model_scores, 2)
    }
    seen_pairs = {
        frozenset((agreement.left_model_id, agreement.right_model_id))
        for agreement in agreements
    }
    for missing_pair in sorted(expected_pairs - seen_pairs, key=lambda pair: sorted(pair)):
        left, right = sorted(missing_pair)
        reasons.append(f"{left}/{right}: inter-model agreement missing")

    for agreement in agreements:
        pair = frozenset((agreement.left_model_id, agreement.right_model_id))
        if pair not in expected_pairs:
            continue
        label = f"{agreement.left_model_id}/{agreement.right_model_id}"
        _require_metric(
            reasons,
            label,
            "inter-model agreement",
            agreement.agreement,
            thresholds.inter_model_agreement_min,
        )


def _require_stability_coverage(
    reasons: list[str],
    model_scores: Sequence[ModelScore],
    stability_scores: Sequence[StabilityScore],
    thresholds: BenchmarkThresholds,
) -> None:
    if not model_scores:
        reasons.append("intra-model stability missing")
        return

    stability_by_model = {score.model_id: score for score in stability_scores}
    for score in model_scores:
        stability = stability_by_model.get(score.model_id)
        if stability is None:
            reasons.append(f"{score.model_id}: intra-model stability missing")
            continue
        _require_metric(
            reasons,
            stability.model_id,
            "intra-model stability",
            stability.stability,
            thresholds.intra_model_stability_min,
        )


def _require_metric(
    reasons: list[str],
    owner: str,
    metric: str,
    value: float | None,
    minimum: float,
    *,
    strict: bool = False,
) -> None:
    if value is None:
        reasons.append(f"{owner}: {metric} missing")
        return
    if strict:
        if value <= minimum:
            reasons.append(f"{owner}: {metric} {value:.3f} not above {minimum:.3f}")
        return
    if value < minimum:
        reasons.append(f"{owner}: {metric} {value:.3f} below {minimum:.3f}")
