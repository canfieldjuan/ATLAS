"""Deterministic benchmark scoring for claim/evidence support checks.

This module is intentionally pure. It does not call a model, read the claim
registry, or expose an MCP tool. It only scores labeled benchmark triples and
structured witness responses against the reliability thresholds from issue
#1435.
"""

from __future__ import annotations

import json
from collections.abc import Sequence as RuntimeSequence
from dataclasses import asdict, dataclass
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
class ClaimEvidenceResultArtifact:
    """Machine-readable reliability-gate benchmark result."""

    thresholds: BenchmarkThresholds
    model_scores: tuple[ModelScore, ...]
    agreement_matrix: tuple[PairwiseAgreement, ...]
    stability_scores: tuple[StabilityScore, ...]
    failure_cases: tuple[Mapping[str, object], ...]
    verdict: BenchmarkVerdict
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors and self.verdict.passed

    @property
    def go_no_go(self) -> str:
        return "go" if self.ok else "no_go"

    def as_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["go_no_go"] = self.go_no_go
        payload["ok"] = self.ok
        return payload


@dataclass(frozen=True)
class ClaimEvidenceResultFile:
    """One deterministic operator-facing benchmark result file."""

    path: str
    content_type: str
    content: str


def claim_evidence_result_artifact_files(
    artifact: object,
) -> tuple[ClaimEvidenceResultFile, ...]:
    """Return stable JSON + Markdown result files for an existing artifact."""

    typed_artifact = _coerce_result_artifact(artifact)
    json_payload = json.dumps(
        typed_artifact.as_mapping(),
        indent=2,
        sort_keys=True,
    )
    return (
        ClaimEvidenceResultFile(
            path="claim_evidence_result.json",
            content_type="application/json",
            content=json_payload + "\n",
        ),
        ClaimEvidenceResultFile(
            path="claim_evidence_result.md",
            content_type="text/markdown",
            content=render_claim_evidence_result_markdown(typed_artifact),
        ),
    )


def render_claim_evidence_result_markdown(artifact: object) -> str:
    """Render a benchmark result artifact as an operator-facing Markdown report."""

    typed_artifact = _coerce_result_artifact(artifact)
    lines = [
        "# Claim Evidence Benchmark Result",
        "",
        "## Decision",
        "",
        f"- Go/no-go: {typed_artifact.go_no_go.upper()}",
        f"- Artifact ok: {_yes_no(typed_artifact.ok)}",
        f"- Verdict passed: {_yes_no(typed_artifact.verdict.passed)}",
        "",
        "## Thresholds",
        "",
    ]
    lines.extend(
        _markdown_table(
            ("Metric", "Threshold"),
            (
                (
                    "Easy accuracy",
                    f">= {_format_percent(typed_artifact.thresholds.easy_accuracy_min)}",
                ),
                (
                    "Hard accuracy",
                    f">= {_format_percent(typed_artifact.thresholds.hard_accuracy_min)}",
                ),
                (
                    "Inter-model agreement",
                    (
                        ">= "
                        f"{_format_percent(typed_artifact.thresholds.inter_model_agreement_min)}"
                    ),
                ),
                (
                    "Intra-model stability",
                    (
                        ">= "
                        f"{_format_percent(typed_artifact.thresholds.intra_model_stability_min)}"
                    ),
                ),
                (
                    "High-confidence accuracy",
                    (
                        "> "
                        f"{_format_percent(typed_artifact.thresholds.high_confidence_accuracy_min)}"
                    ),
                ),
            ),
        )
    )
    lines.extend(("", "## Model Scores", ""))
    lines.extend(
        _markdown_table(
            (
                "Model",
                "Easy",
                "Hard",
                "High confidence",
                "Missing responses",
                "Low confidence",
            ),
            tuple(
                (
                    score.model_id,
                    _format_percent(score.easy_accuracy),
                    _format_percent(score.hard_accuracy),
                    _format_percent(score.high_confidence_accuracy),
                    _comma_list(score.missing_response_ids),
                    _comma_list(score.low_confidence_response_ids),
                )
                for score in typed_artifact.model_scores
            ),
            empty_label="No model scores.",
        )
    )
    lines.extend(("", "## Inter-Model Agreement", ""))
    lines.extend(
        _markdown_table(
            ("Pair", "Agreement", "Total"),
            tuple(
                (
                    f"{row.left_model_id} / {row.right_model_id}",
                    _format_percent(row.agreement),
                    row.total,
                )
                for row in typed_artifact.agreement_matrix
            ),
            empty_label="No agreement rows.",
        )
    )
    lines.extend(("", "## Intra-Model Stability", ""))
    lines.extend(
        _markdown_table(
            ("Model", "Stability", "Total"),
            tuple(
                (
                    row.model_id,
                    _format_percent(row.stability),
                    row.total,
                )
                for row in typed_artifact.stability_scores
            ),
            empty_label="No stability rows.",
        )
    )
    lines.extend(("", "## Verdict Failures", ""))
    lines.extend(_markdown_bullets(typed_artifact.verdict.failure_reasons))
    lines.extend(("", "## Artifact Errors", ""))
    lines.extend(_markdown_bullets(typed_artifact.errors))
    lines.extend(("", "## Failure Cases", ""))
    lines.extend(_failure_case_table(typed_artifact.failure_cases))
    return "\n".join(lines).rstrip() + "\n"


def _coerce_result_artifact(artifact: object) -> ClaimEvidenceResultArtifact:
    if isinstance(artifact, ClaimEvidenceResultArtifact):
        return artifact
    return _failed_result_artifact(
        DEFAULT_THRESHOLDS,
        ("artifact must be ClaimEvidenceResultArtifact",),
    )


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


def build_claim_evidence_result_artifact(
    triples: Sequence[ClaimEvidenceTriple],
    model_runs: Sequence[ClaimEvidenceModelRun],
    *,
    stability_runs_by_model_id: (
        Mapping[str, Sequence[ClaimEvidenceModelRun]] | None
    ) = None,
    thresholds: BenchmarkThresholds = DEFAULT_THRESHOLDS,
) -> ClaimEvidenceResultArtifact:
    """Assemble completed model runs into a benchmark decision artifact."""

    active_thresholds = (
        thresholds if isinstance(thresholds, BenchmarkThresholds) else DEFAULT_THRESHOLDS
    )
    errors: list[str] = []
    if not isinstance(thresholds, BenchmarkThresholds):
        errors.append("thresholds must be BenchmarkThresholds")

    typed_triples: tuple[ClaimEvidenceTriple, ...] = ()
    if not _usable_sequence(triples):
        errors.append("triples must be a non-empty sequence")
    elif all(isinstance(triple, ClaimEvidenceTriple) for triple in triples):
        typed_triples = tuple(triples)
    else:
        errors.append("triples must contain ClaimEvidenceTriple")

    typed_runs: tuple[ClaimEvidenceModelRun, ...] = ()
    if not _usable_sequence(model_runs):
        errors.append("model_runs must be a non-empty sequence")
    elif all(isinstance(run, ClaimEvidenceModelRun) for run in model_runs):
        typed_runs = tuple(sorted(model_runs, key=lambda run: run.model_id))
    else:
        errors.append("model_runs must contain ClaimEvidenceModelRun")

    triple_ids = {triple.triple_id for triple in typed_triples}
    seen_models: set[str] = set()
    for run in typed_runs:
        if not run.model_id.strip():
            errors.append("model_id missing")
        elif run.model_id in seen_models:
            errors.append(f"model_id duplicated: {run.model_id}")
        else:
            seen_models.add(run.model_id)
        if run.errors:
            errors.append(f"{run.model_id}: run errors: {'; '.join(run.errors)}")
        _validate_artifact_rows(run, triple_ids, errors)

    stability_runs = (
        {} if stability_runs_by_model_id is None else stability_runs_by_model_id
    )
    stability_response_maps: dict[
        str, tuple[Mapping[str, ClaimEvidenceResponse], ...]
    ] = {}
    if not isinstance(stability_runs, Mapping):
        errors.append("stability_runs_by_model_id must be a mapping")
    else:
        stability_response_maps = _validated_stability_response_maps(
            stability_runs,
            seen_models,
            triple_ids,
            errors,
        )

    if errors:
        return _failed_result_artifact(active_thresholds, errors)

    model_responses = {
        run.model_id: run.responses_by_triple_id for run in typed_runs
    }
    model_scores = tuple(
        score_model(run.model_id, typed_triples, model_responses[run.model_id])
        for run in typed_runs
    )
    agreements = pairwise_agreements(typed_triples, model_responses)
    stability_scores = tuple(
        intra_model_stability(
            model_id,
            typed_triples,
            stability_response_maps.get(model_id, ()),
        )
        for model_id in sorted(model_responses)
    )
    verdict = evaluate_thresholds(
        model_scores,
        agreements,
        stability_scores,
        active_thresholds,
    )
    return ClaimEvidenceResultArtifact(
        thresholds=active_thresholds,
        model_scores=model_scores,
        agreement_matrix=agreements,
        stability_scores=stability_scores,
        failure_cases=_claim_evidence_failure_cases(typed_triples, typed_runs),
        verdict=verdict,
        errors=(),
    )


def _usable_sequence(value: object) -> bool:
    return (
        isinstance(value, RuntimeSequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
    )


def _validate_artifact_rows(
    run: ClaimEvidenceModelRun,
    triple_ids: set[str],
    errors: list[str],
) -> None:
    seen_rows: set[str] = set()
    for index, row in enumerate(run.rows, start=1):
        if not isinstance(row, ClaimEvidenceRunRow):
            errors.append(f"{run.model_id} row {index}: must be ClaimEvidenceRunRow")
            continue
        if row.triple_id in seen_rows:
            errors.append(f"{run.model_id}: duplicate row triple_id: {row.triple_id}")
            continue
        seen_rows.add(row.triple_id)
        if row.triple_id not in triple_ids:
            errors.append(f"{run.model_id}: unknown row triple_id: {row.triple_id}")


def _validated_stability_response_maps(
    stability_runs: Mapping[object, object],
    model_ids: set[str],
    triple_ids: set[str],
    errors: list[str],
) -> dict[str, tuple[Mapping[str, ClaimEvidenceResponse], ...]]:
    normalized: dict[str, object] = {}
    for raw_model_id, raw_runs in stability_runs.items():
        if not isinstance(raw_model_id, str) or not raw_model_id.strip():
            errors.append("stability model_id missing")
            continue
        model_id = raw_model_id.strip()
        if model_id not in model_ids:
            errors.append(f"stability model_id not in model runs: {model_id}")
            continue
        if model_id in normalized:
            errors.append(f"stability model_id duplicated: {model_id}")
            continue
        normalized[model_id] = raw_runs

    response_maps: dict[str, tuple[Mapping[str, ClaimEvidenceResponse], ...]] = {}
    for model_id in sorted(model_ids):
        raw_runs = normalized.get(model_id, ())
        if not isinstance(raw_runs, RuntimeSequence) or isinstance(
            raw_runs, (str, bytes)
        ):
            errors.append(f"stability runs for {model_id} must be a sequence")
            continue
        run_maps: list[Mapping[str, ClaimEvidenceResponse]] = []
        for index, run in enumerate(raw_runs, start=1):
            if not isinstance(run, ClaimEvidenceModelRun):
                errors.append(
                    f"stability run {index} for {model_id}: "
                    "must be ClaimEvidenceModelRun"
                )
                continue
            if run.model_id != model_id:
                errors.append(
                    f"stability run {index} for {model_id}: "
                    f"model_id mismatch: {run.model_id}"
                )
                continue
            if run.errors:
                errors.append(
                    f"stability run {index} for {model_id}: "
                    f"run errors: {'; '.join(run.errors)}"
                )
            _validate_artifact_rows(run, triple_ids, errors)
            run_maps.append(run.responses_by_triple_id)
        response_maps[model_id] = tuple(run_maps)
    return response_maps


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


def _claim_evidence_failure_cases(
    triples: Sequence[ClaimEvidenceTriple],
    model_runs: Sequence[ClaimEvidenceModelRun],
) -> tuple[Mapping[str, object], ...]:
    failures: list[Mapping[str, object]] = []
    for run in model_runs:
        rows_by_id = {row.triple_id: row for row in run.rows}
        for triple in triples:
            row = rows_by_id.get(triple.triple_id)
            if row is None:
                failures.append(
                    _failure_case(run.model_id, triple, "missing_response", None, ())
                )
                continue
            if row.response is None or row.errors:
                failures.append(
                    _failure_case(
                        run.model_id,
                        triple,
                        "malformed_response",
                        row.response,
                        row.errors,
                    )
                )
                continue
            if row.response.supports != triple.expected_supports:
                failures.append(
                    _failure_case(
                        run.model_id,
                        triple,
                        "incorrect_support",
                        row.response,
                        (),
                    )
                )
                continue
            if row.response.confidence < CONFIDENCE_COUNTS_MIN:
                failures.append(
                    _failure_case(
                        run.model_id,
                        triple,
                        "low_confidence",
                        row.response,
                        (),
                    )
                )
    return tuple(failures)


def _failure_case(
    model_id: str,
    triple: ClaimEvidenceTriple,
    failure: str,
    response: ClaimEvidenceResponse | None,
    row_errors: tuple[str, ...],
) -> Mapping[str, object]:
    return {
        "model_id": model_id,
        "triple_id": triple.triple_id,
        "failure": failure,
        "expected_supports": triple.expected_supports,
        "actual_supports": response.supports if response is not None else None,
        "confidence": response.confidence if response is not None else None,
        "response_reason": response.reason if response is not None else "",
        "row_errors": row_errors,
    }


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _comma_list(values: Sequence[object]) -> str:
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _markdown_cell(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    empty_label: str | None = None,
) -> list[str]:
    lines = [
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if not rows and empty_label is not None:
        lines.append(
            "| "
            + _markdown_cell(empty_label)
            + " | "
            + " | ".join("" for _ in headers[1:])
            + " |"
        )
        return lines
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return lines


def _markdown_bullets(values: Sequence[object]) -> list[str]:
    if not values:
        return ["- None."]
    return [f"- {_markdown_cell(value)}" for value in values]


def _failure_case_table(
    failure_cases: Sequence[Mapping[str, object]],
) -> list[str]:
    return _markdown_table(
        (
            "Model",
            "Triple",
            "Failure",
            "Expected",
            "Actual",
            "Confidence",
            "Row errors",
        ),
        tuple(
            (
                failure.get("model_id", ""),
                failure.get("triple_id", ""),
                failure.get("failure", ""),
                failure.get("expected_supports", ""),
                _markdown_optional_bool_text(failure.get("actual_supports")),
                _markdown_optional_text(failure.get("confidence")),
                _comma_list(_tuple_or_empty(failure.get("row_errors"))),
            )
            for failure in failure_cases
        ),
        empty_label="No failure cases.",
    )


def _tuple_or_empty(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _markdown_optional_bool_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return "n/a"


def _markdown_optional_text(value: object) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _failed_result_artifact(
    thresholds: BenchmarkThresholds,
    errors: Sequence[str],
) -> ClaimEvidenceResultArtifact:
    return ClaimEvidenceResultArtifact(
        thresholds=thresholds,
        model_scores=(),
        agreement_matrix=(),
        stability_scores=(),
        failure_cases=(),
        verdict=BenchmarkVerdict(False, tuple(errors)),
        errors=tuple(errors),
    )
