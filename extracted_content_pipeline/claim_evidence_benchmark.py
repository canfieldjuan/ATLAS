"""Deterministic benchmark scoring for claim/evidence support checks.

This module is intentionally pure. It does not call a model, read the claim
registry, or expose an MCP tool. It only scores labeled benchmark triples and
structured witness responses against the reliability thresholds from issue
#1435.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence


EASY = "easy"
HARD = "hard"
VALID_DIFFICULTIES = frozenset({EASY, HARD})
CONFIDENCE_COUNTS_MIN = 4


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
        evidence_quote = _required_text(data, "evidence_quote")
        source_id = _required_text(data, "source_id")
        expected_supports = _required_bool(data, "expected_supports")
        difficulty = _required_text(data, "difficulty")

        for key, value in (
            ("triple_id", triple_id),
            ("claim_id", claim_id),
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
