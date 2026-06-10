"""Tests for the claim/evidence structured-judgment benchmark core."""

from __future__ import annotations

import json

from extracted_content_pipeline.claim_evidence_benchmark import (
    EASY,
    HARD,
    BenchmarkThresholds,
    ClaimEvidenceModelRun,
    ClaimEvidenceResponse,
    ClaimEvidenceRunRow,
    ClaimEvidenceTriple,
    ModelScore,
    PairwiseAgreement,
    StabilityScore,
    build_claim_evidence_prompt_contract,
    build_claim_evidence_result_artifact,
    claim_evidence_response_json_schema,
    evaluate_thresholds,
    intra_model_stability,
    load_claim_evidence_fixture_text,
    pairwise_agreements,
    run_claim_evidence_provider,
    score_model,
    validate_claim_evidence_fixture,
)


def _triple(triple_id: str, expected: bool, difficulty: str) -> ClaimEvidenceTriple:
    return ClaimEvidenceTriple(
        triple_id=triple_id,
        claim_id=f"claim-{triple_id}",
        claim_text=f"Claim statement {triple_id}",
        evidence_quote=f"evidence {triple_id}",
        source_id=f"source-{triple_id}",
        expected_supports=expected,
        difficulty=difficulty,
    )


def _response(supports: bool, confidence: int = 5) -> ClaimEvidenceResponse:
    return ClaimEvidenceResponse(
        supports=supports,
        confidence=confidence,
        reason="quote directly supports the claim",
    )


def _passing_score(model_id: str) -> ModelScore:
    return ModelScore(model_id, 1.0, 0.8, 0.95, 30, 10, 40, (), ())


def _run_row(
    model_id: str,
    triple_id: str,
    response: ClaimEvidenceResponse | None,
    errors: tuple[str, ...] = (),
) -> ClaimEvidenceRunRow:
    return ClaimEvidenceRunRow(
        model_id=model_id,
        triple_id=triple_id,
        contract_version="verify_claim_evidence.v1",
        response=response,
        errors=errors,
    )


def _model_run(
    model_id: str,
    responses: dict[str, ClaimEvidenceResponse],
    *,
    rows: tuple[ClaimEvidenceRunRow, ...] = (),
    errors: tuple[str, ...] = (),
) -> ClaimEvidenceModelRun:
    return ClaimEvidenceModelRun(
        model_id=model_id,
        rows=tuple(
            _run_row(model_id, triple_id, response)
            for triple_id, response in responses.items()
        )
        + rows,
        errors=errors,
    )


def test_triple_decoder_accepts_valid_decoded_input() -> None:
    triple, errors = ClaimEvidenceTriple.from_mapping(
        {
            "triple_id": "t1",
            "claim_id": "c1",
            "claim_text": "The product integrates with Salesforce.",
            "evidence_quote": "The product integrates with Salesforce.",
            "source_id": "s1",
            "expected_supports": True,
            "difficulty": EASY,
        }
    )

    assert errors == ()
    assert triple is not None
    assert triple.triple_id == "t1"
    assert triple.claim_text == "The product integrates with Salesforce."
    assert triple.expected_supports is True


def test_triple_decoder_reports_missing_and_wrong_typed_fields() -> None:
    triple, errors = ClaimEvidenceTriple.from_mapping(
        {
            "triple_id": None,
            "claim_id": 10,
            "claim_text": " ",
            "evidence_quote": "",
            "source_id": "s1",
            "expected_supports": "yes",
            "difficulty": "medium",
        }
    )

    assert triple is None
    assert errors == (
        "triple_id missing",
        "claim_id missing",
        "claim_text missing",
        "evidence_quote missing",
        "expected_supports missing",
        "difficulty must be easy or hard",
    )


def test_triple_decoder_rejects_non_object_without_raising() -> None:
    triple, errors = ClaimEvidenceTriple.from_mapping(None)

    assert triple is None
    assert errors == ("triple must be an object",)


def test_fixture_validator_rejects_non_list_without_raising() -> None:
    for rows in (None, "not-json-rows", {"triple_id": "one"}, range(1)):
        fixture = validate_claim_evidence_fixture(rows)

        assert fixture.triples == ()
        assert fixture.errors == ("fixture rows must be a list of objects",)


def test_fixture_validator_prefixes_row_decoder_errors() -> None:
    fixture = validate_claim_evidence_fixture(
        [
            {
                "triple_id": "t1",
                "claim_id": "",
                "claim_text": "",
                "evidence_quote": "source quote",
                "source_id": "s1",
                "expected_supports": None,
                "difficulty": "medium",
            },
        ]
    )

    assert fixture.triples == ()
    assert fixture.errors == (
        "row 1: claim_id missing",
        "row 1: claim_text missing",
        "row 1: expected_supports missing",
        "row 1: difficulty must be easy or hard",
    )


def test_fixture_validator_rejects_duplicate_triple_ids() -> None:
    fixture = validate_claim_evidence_fixture(
        [
            _triple("same", True, EASY).__dict__,
            _triple("same", False, HARD).__dict__,
        ]
    )

    assert [triple.triple_id for triple in fixture.triples] == ["same"]
    assert fixture.errors == ("row 2: triple_id duplicated from row 1: same",)


def test_fixture_validator_allows_seed_sets_without_final_shape() -> None:
    fixture = validate_claim_evidence_fixture(
        [
            _triple("seed-1", True, EASY).__dict__,
            _triple("seed-2", False, HARD).__dict__,
        ]
    )

    assert fixture.ok is True
    assert fixture.easy_supports_count == 1
    assert fixture.easy_not_supports_count == 0
    assert fixture.hard_count == 1


def test_fixture_validator_enforces_final_issue_1435_shape() -> None:
    rows = []
    rows.extend(_triple(f"easy-yes-{idx}", True, EASY).__dict__ for idx in range(15))
    rows.extend(_triple(f"easy-no-{idx}", False, EASY).__dict__ for idx in range(15))
    rows.extend(_triple(f"hard-{idx}", idx % 2 == 0, HARD).__dict__ for idx in range(10))

    fixture = validate_claim_evidence_fixture(rows, require_final_shape=True)

    assert fixture.ok is True
    assert len(fixture.triples) == 40


def test_fixture_validator_reports_final_shape_shortfalls() -> None:
    fixture = validate_claim_evidence_fixture(
        [
            _triple("easy-yes", True, EASY).__dict__,
            _triple("hard", False, HARD).__dict__,
        ],
        require_final_shape=True,
    )

    assert fixture.errors == (
        "final fixture requires 15 easy support rows; got 1",
        "final fixture requires 15 easy non-support rows; got 0",
        "final fixture requires 10 hard rows; got 1",
    )


def test_fixture_loader_rejects_non_string_text_without_raising() -> None:
    fixture = load_claim_evidence_fixture_text(None)

    assert fixture.triples == ()
    assert fixture.errors == ("fixture text must be a string",)


def test_fixture_loader_rejects_unsupported_format_without_raising() -> None:
    for source_format in (None, "csv"):
        fixture = load_claim_evidence_fixture_text("[]", source_format=source_format)

        assert fixture.triples == ()
        assert fixture.errors == ("fixture format must be json or jsonl",)


def test_fixture_loader_accepts_json_array_text() -> None:
    fixture = load_claim_evidence_fixture_text(
        json.dumps([_triple("json-1", True, EASY).__dict__])
    )

    assert fixture.ok is True
    assert [triple.triple_id for triple in fixture.triples] == ["json-1"]


def test_fixture_loader_rejects_json_malformed_text() -> None:
    fixture = load_claim_evidence_fixture_text("[")

    assert fixture.triples == ()
    assert fixture.errors == ("json fixture is malformed: Expecting value",)


def test_fixture_loader_rejects_json_non_array_text_before_validation() -> None:
    fixture = load_claim_evidence_fixture_text(json.dumps({"triple_id": "one"}))

    assert fixture.triples == ()
    assert fixture.errors == ("json fixture must decode to a list of objects",)


def test_fixture_loader_accepts_jsonl_object_lines() -> None:
    text = "\n".join(
        [
            json.dumps(_triple("jsonl-1", True, EASY).__dict__),
            "",
            json.dumps(_triple("jsonl-2", False, HARD).__dict__),
        ]
    )

    fixture = load_claim_evidence_fixture_text(text, source_format="jsonl")

    assert fixture.ok is True
    assert [triple.triple_id for triple in fixture.triples] == [
        "jsonl-1",
        "jsonl-2",
    ]


def test_fixture_loader_reports_jsonl_malformed_line_numbers() -> None:
    text = "\n".join(
        [
            json.dumps(_triple("jsonl-1", True, EASY).__dict__),
            "{",
        ]
    )

    fixture = load_claim_evidence_fixture_text(text, source_format="jsonl")

    assert fixture.triples == ()
    assert fixture.errors == (
        "line 2: jsonl fixture is malformed: "
        "Expecting property name enclosed in double quotes",
    )


def test_fixture_loader_rejects_jsonl_array_lines() -> None:
    fixture = load_claim_evidence_fixture_text(
        json.dumps([_triple("jsonl-array", True, EASY).__dict__]),
        source_format="jsonl",
    )

    assert fixture.triples == ()
    assert fixture.errors == ("line 1: jsonl fixture line must be an object",)


def test_fixture_loader_delegates_final_shape_validation() -> None:
    fixture = load_claim_evidence_fixture_text(
        json.dumps([_triple("easy-yes", True, EASY).__dict__]),
        require_final_shape=True,
    )

    assert fixture.errors == (
        "final fixture requires 15 easy support rows; got 1",
        "final fixture requires 15 easy non-support rows; got 0",
        "final fixture requires 10 hard rows; got 0",
    )


def test_response_decoder_accepts_valid_decoded_input() -> None:
    response, errors = ClaimEvidenceResponse.from_mapping(
        {"supports": False, "confidence": 4, "reason": "quote contradicts it"}
    )

    assert errors == ()
    assert response == ClaimEvidenceResponse(
        supports=False,
        confidence=4,
        reason="quote contradicts it",
    )


def test_response_decoder_reports_missing_and_wrong_typed_fields() -> None:
    response, errors = ClaimEvidenceResponse.from_mapping(
        {"supports": "false", "confidence": True, "reason": None}
    )

    assert response is None
    assert errors == (
        "supports missing",
        "confidence must be an integer from 1 to 5",
        "reason missing",
    )


def test_response_decoder_rejects_extra_fields_like_schema() -> None:
    response, errors = ClaimEvidenceResponse.from_mapping(
        {
            "supports": True,
            "confidence": 5,
            "reason": "quote directly supports the claim",
            "internal_metadata": {"model_version": "x.y.z"},
            "leaked_chain_of_thought": "Let me think step by step.",
        }
    )

    assert response is None
    assert errors == (
        "unexpected fields: internal_metadata, leaked_chain_of_thought",
    )


def test_prompt_contract_renders_required_triple_fields_and_guardrails() -> None:
    contract = build_claim_evidence_prompt_contract(
        ClaimEvidenceTriple(
            triple_id="hard-001",
            claim_id="claim-escalation-001",
            claim_text="Customers reduced escalations by 31 percent.",
            evidence_quote="Escalation volume fell 31% after rollout.",
            source_id="case-study-acme",
            expected_supports=True,
            difficulty=HARD,
        )
    )

    assert contract.contract_version == "verify_claim_evidence.v1"
    assert "Claim: Customers reduced escalations by 31 percent." in contract.prompt
    assert "Claim id: claim-escalation-001" in contract.prompt
    assert "Source id: case-study-acme" in contract.prompt
    assert "Difficulty bucket: hard" in contract.prompt
    assert "Escalation volume fell 31% after rollout." in contract.prompt
    assert "supports the claim under test" in contract.prompt
    assert "Do not decide whether the claim is true in general." in contract.prompt
    assert "Do not use outside knowledge" in contract.prompt


def test_response_json_schema_matches_decoder_contract() -> None:
    schema = claim_evidence_response_json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["supports", "confidence", "reason"]
    assert schema["properties"]["supports"]["type"] == "boolean"
    assert schema["properties"]["confidence"]["type"] == "integer"
    assert schema["properties"]["confidence"]["minimum"] == 1
    assert schema["properties"]["confidence"]["maximum"] == 5
    assert schema["properties"]["reason"]["type"] == "string"
    assert schema["properties"]["reason"]["pattern"] == "\\S"

    response, errors = ClaimEvidenceResponse.from_mapping(
        {"supports": True, "confidence": 4, "reason": "quote states it"}
    )

    assert errors == ()
    assert response == ClaimEvidenceResponse(True, 4, "quote states it")


def test_response_json_schema_rejects_whitespace_reason_like_decoder() -> None:
    schema = claim_evidence_response_json_schema()
    whitespace_reason = {"supports": True, "confidence": 4, "reason": "   "}

    assert schema["additionalProperties"] is False
    assert schema["properties"]["reason"]["pattern"] == "\\S"
    assert not any(
        character.strip() for character in str(whitespace_reason["reason"])
    )

    response, errors = ClaimEvidenceResponse.from_mapping(whitespace_reason)

    assert response is None
    assert errors == ("reason missing",)


def test_response_json_schema_rejects_extra_fields_like_decoder() -> None:
    schema = claim_evidence_response_json_schema()
    response_with_extra = {
        "supports": True,
        "confidence": 4,
        "reason": "quote states it",
        "extra": "not in schema",
    }

    assert schema["additionalProperties"] is False

    response, errors = ClaimEvidenceResponse.from_mapping(response_with_extra)

    assert response is None
    assert errors == ("unexpected fields: extra",)


def test_response_json_schema_returns_fresh_mapping() -> None:
    schema = claim_evidence_response_json_schema()
    schema["required"] = []

    assert claim_evidence_response_json_schema()["required"] == [
        "supports",
        "confidence",
        "reason",
    ]


def test_runner_calls_provider_with_prompt_contract_for_each_triple() -> None:
    triples = (
        _triple("a", True, EASY),
        _triple("b", False, HARD),
    )
    calls = []

    def provider(model_id, triple, contract):
        calls.append((model_id, triple.triple_id, contract))
        return {
            "supports": triple.expected_supports,
            "confidence": 5,
            "reason": "quote directly supports the claim",
        }

    run = run_claim_evidence_provider(" opus ", triples, provider)

    assert run.ok is True
    assert run.errors == ()
    assert [row.triple_id for row in run.rows] == ["a", "b"]
    assert [call[0] for call in calls] == ["opus", "opus"]
    assert [call[1] for call in calls] == ["a", "b"]
    assert calls[0][2].contract_version == "verify_claim_evidence.v1"
    assert "Claim: Claim statement a" in calls[0][2].prompt
    assert calls[0][2].response_schema["required"] == [
        "supports",
        "confidence",
        "reason",
    ]
    assert run.responses_by_triple_id == {
        "a": _response(True, 5),
        "b": _response(False, 5),
    }


def test_runner_records_malformed_response_without_scoring_it() -> None:
    def provider(model_id, triple, contract):
        return {"supports": True, "confidence": 4, "reason": "   "}

    run = run_claim_evidence_provider("gpt", (_triple("bad", True, EASY),), provider)

    assert run.ok is False
    assert run.errors == ()
    assert len(run.rows) == 1
    assert run.rows[0].response is None
    assert run.rows[0].errors == ("reason missing",)
    assert run.responses_by_triple_id == {}


def test_runner_records_extra_field_response_without_scoring_it() -> None:
    def provider(model_id, triple, contract):
        return {
            "supports": True,
            "confidence": 4,
            "reason": "quote directly supports the claim",
            "extra": "provider drift",
        }

    run = run_claim_evidence_provider("gpt", (_triple("extra", True, EASY),), provider)

    assert run.ok is False
    assert run.errors == ()
    assert run.rows[0].response is None
    assert run.rows[0].errors == ("unexpected fields: extra",)
    assert run.responses_by_triple_id == {}


def test_runner_records_provider_exception_and_continues_later_triples() -> None:
    triples = (
        _triple("raises", True, EASY),
        _triple("after", False, HARD),
    )

    def provider(model_id, triple, contract):
        if triple.triple_id == "raises":
            raise RuntimeError("vendor timeout with sensitive details")
        return {"supports": False, "confidence": 4, "reason": "quote does not say it"}

    run = run_claim_evidence_provider("sonnet", triples, provider)

    assert run.ok is False
    assert run.errors == ()
    assert [row.triple_id for row in run.rows] == ["raises", "after"]
    assert run.rows[0].errors == ("provider error: RuntimeError",)
    assert "sensitive" not in run.rows[0].errors[0]
    assert run.rows[1].ok is True
    assert run.responses_by_triple_id == {
        "after": ClaimEvidenceResponse(
            supports=False,
            confidence=4,
            reason="quote does not say it",
        )
    }


def test_runner_fails_closed_on_invalid_harness_inputs() -> None:
    def provider(model_id, triple, contract):
        return {
            "supports": True,
            "confidence": 5,
            "reason": "quote directly supports the claim",
        }

    assert run_claim_evidence_provider(None, (), provider).errors == (
        "model_id missing",
    )
    assert run_claim_evidence_provider("model", (), None).errors == (
        "provider must be callable",
    )
    assert run_claim_evidence_provider("model", None, provider).errors == (
        "triples must be a sequence",
    )
    assert run_claim_evidence_provider("model", (), provider).errors == (
        "triples missing",
    )

    run = run_claim_evidence_provider(
        "model",
        ({"triple_id": "decoded"}, _triple("valid", True, EASY)),
        provider,
    )

    assert run.ok is False
    assert run.errors == ("row 1: triple must be ClaimEvidenceTriple",)
    assert [row.triple_id for row in run.rows] == ["valid"]
    assert run.responses_by_triple_id == {"valid": _response(True, 5)}


def test_result_artifact_builds_go_payload_from_completed_runs() -> None:
    triples = (
        _triple("easy", True, EASY),
        _triple("hard", False, HARD),
    )
    claude_run = _model_run(
        "claude",
        {"easy": _response(True, 5), "hard": _response(False, 4)},
    )
    gpt_run = _model_run(
        "gpt",
        {"easy": _response(True, 5), "hard": _response(False, 4)},
    )

    artifact = build_claim_evidence_result_artifact(
        triples,
        (gpt_run, claude_run),
        stability_runs_by_model_id={
            "gpt": (gpt_run, gpt_run),
            "claude": (claude_run, claude_run),
        },
        thresholds=BenchmarkThresholds(
            easy_accuracy_min=1.0,
            hard_accuracy_min=1.0,
            inter_model_agreement_min=1.0,
            intra_model_stability_min=1.0,
            high_confidence_accuracy_min=0.90,
        ),
    )

    assert artifact.ok is True
    assert artifact.go_no_go == "go"
    assert [score.model_id for score in artifact.model_scores] == ["claude", "gpt"]
    assert artifact.agreement_matrix == (PairwiseAgreement("claude", "gpt", 1.0, 2),)
    assert artifact.failure_cases == ()

    payload = artifact.as_mapping()
    assert payload["go_no_go"] == "go"
    assert payload["verdict"] == {"passed": True, "failure_reasons": ()}
    assert payload["model_scores"][0]["easy_accuracy"] == 1.0
    assert payload["agreement_matrix"][0]["agreement"] == 1.0
    assert payload["stability_scores"][0]["stability"] == 1.0


def test_result_artifact_records_failure_cases_without_raising() -> None:
    triples = (
        _triple("wrong", True, EASY),
        _triple("missing", False, HARD),
        _triple("malformed", True, EASY),
        _triple("low", True, EASY),
    )
    run = _model_run(
        "gpt",
        {
            "wrong": _response(False, 5),
            "low": _response(True, 2),
        },
        rows=(
            _run_row("gpt", "malformed", None, ("reason missing",)),
        ),
    )

    artifact = build_claim_evidence_result_artifact(triples, (run,))

    assert artifact.go_no_go == "no_go"
    assert artifact.errors == ()
    assert [case["failure"] for case in artifact.failure_cases] == [
        "incorrect_support",
        "missing_response",
        "malformed_response",
        "low_confidence",
    ]
    assert artifact.failure_cases[0]["actual_supports"] is False
    assert artifact.failure_cases[1]["actual_supports"] is None
    assert artifact.failure_cases[2]["row_errors"] == ("reason missing",)
    assert artifact.failure_cases[3]["confidence"] == 2
    assert artifact.as_mapping()["failure_cases"][2]["row_errors"] == (
        "reason missing",
    )


def test_result_artifact_fails_closed_on_malformed_primary_inputs() -> None:
    known = _triple("known", True, EASY)
    good_row = _run_row("gpt", "known", _response(True))
    run_with_bad_rows = ClaimEvidenceModelRun(
        "gpt",
        (
            _run_row("gpt", "unknown", _response(True)),
            good_row,
            good_row,
        ),
        ("setup failed",),
    )
    duplicate_model = ClaimEvidenceModelRun("gpt", (), ())
    missing_model_id = ClaimEvidenceModelRun(" ", (), ())
    bad_row_type = ClaimEvidenceModelRun("rows", ("bad-row",), ())

    empty = build_claim_evidence_result_artifact(None, None)

    assert empty.errors == (
        "triples must be a non-empty sequence",
        "model_runs must be a non-empty sequence",
    )

    bad_types = build_claim_evidence_result_artifact(
        (known, {"not": "a triple"}),
        ({"not": "a run"},),
    )

    assert bad_types.errors == (
        "triples must contain ClaimEvidenceTriple",
        "model_runs must contain ClaimEvidenceModelRun",
    )

    artifact = build_claim_evidence_result_artifact(
        (known,),
        (
            run_with_bad_rows,
            duplicate_model,
            missing_model_id,
            bad_row_type,
        ),
        thresholds="not-thresholds",
    )

    assert artifact.model_scores == ()
    assert artifact.failure_cases == ()
    for expected in (
        "gpt: run errors: setup failed",
        "gpt: unknown row triple_id: unknown",
        "gpt: duplicate row triple_id: known",
        "model_id duplicated: gpt",
        "model_id missing",
        "rows row 1: must be ClaimEvidenceRunRow",
        "thresholds must be BenchmarkThresholds",
    ):
        assert expected in artifact.errors

    bad_stability = build_claim_evidence_result_artifact(
        (known,),
        (_model_run("gpt", {"known": _response(True)}),),
        stability_runs_by_model_id=[],
    )

    assert bad_stability.errors == ("stability_runs_by_model_id must be a mapping",)


def test_result_artifact_rejects_malformed_stability_reruns() -> None:
    known = _triple("known", True, EASY)
    gpt_run = _model_run("gpt", {"known": _response(True)})
    bad_rows = ClaimEvidenceModelRun(
        "gpt",
        (_run_row("gpt", "unknown", _response(True)),),
        ("rerun setup failed",),
    )

    artifact = build_claim_evidence_result_artifact(
        (known,),
        (gpt_run,),
        stability_runs_by_model_id={
            "": (),
            "other": (),
            "gpt": (
                "bad rerun",
                ClaimEvidenceModelRun("other", (), ()),
                bad_rows,
            ),
        },
    )

    assert artifact.ok is False
    assert artifact.go_no_go == "no_go"
    assert artifact.model_scores == ()
    for expected in (
        "stability model_id missing",
        "stability model_id not in model runs: other",
        "stability run 1 for gpt: must be ClaimEvidenceModelRun",
        "stability run 2 for gpt: model_id mismatch: other",
        "stability run 3 for gpt: run errors: rerun setup failed",
        "gpt: unknown row triple_id: unknown",
    ):
        assert expected in artifact.errors


def test_model_score_counts_easy_hard_and_high_confidence_accuracy() -> None:
    triples = (
        _triple("easy-pass", True, EASY),
        _triple("easy-fail", True, EASY),
        _triple("hard-pass", False, HARD),
    )
    score = score_model(
        "claude",
        triples,
        {
            "easy-pass": _response(True, 5),
            "easy-fail": _response(False, 5),
            "hard-pass": _response(False, 4),
        },
    )

    assert score.easy_accuracy == 0.5
    assert score.hard_accuracy == 1.0
    assert score.high_confidence_accuracy == 2 / 3
    assert score.missing_response_ids == ()


def test_low_confidence_responses_do_not_count_toward_verdict_credit() -> None:
    triples = (_triple("low", True, EASY), _triple("high", True, EASY))
    score = score_model(
        "gpt",
        triples,
        {"low": _response(True, 2), "high": _response(True, 4)},
    )

    assert score.high_confidence_total == 1
    assert score.high_confidence_accuracy == 1.0
    assert score.low_confidence_response_ids == ("low",)


def test_missing_response_is_incorrect_and_recorded() -> None:
    score = score_model("gpt", (_triple("missing", True, EASY),), {})

    assert score.easy_accuracy == 0.0
    assert score.high_confidence_accuracy is None
    assert score.missing_response_ids == ("missing",)


def test_pairwise_agreement_uses_full_set_and_missing_counts_as_disagreement() -> None:
    triples = (_triple("a", True, EASY), _triple("b", False, HARD))
    agreements = pairwise_agreements(
        triples,
        {
            "claude": {"a": _response(True), "b": _response(False)},
            "gpt": {"a": _response(True)},
        },
    )

    assert len(agreements) == 1
    assert agreements[0].left_model_id == "claude"
    assert agreements[0].right_model_id == "gpt"
    assert agreements[0].agreement == 0.5


def test_intra_model_stability_scores_all_run_pairs() -> None:
    triples = (_triple("a", True, EASY), _triple("b", False, HARD))
    stability = intra_model_stability(
        "opus",
        triples,
        (
            {"a": _response(True), "b": _response(False)},
            {"a": _response(True), "b": _response(True)},
            {"a": _response(True), "b": _response(False)},
        ),
    )

    assert stability.total == 6
    assert stability.stability == 4 / 6


def test_intra_model_stability_requires_at_least_two_runs() -> None:
    stability = intra_model_stability("opus", (_triple("a", True, EASY),), ({},))

    assert stability == StabilityScore(model_id="opus", stability=None, total=0)


def test_threshold_evaluation_passes_when_all_metrics_pass() -> None:
    verdict = evaluate_thresholds(
        (
            ModelScore("claude", 1.0, 0.8, 0.95, 30, 10, 40, (), ()),
            ModelScore("gpt", 0.97, 0.8, 0.91, 30, 10, 38, (), ("t1",)),
        ),
        (
            pairwise_agreements(
                (_triple("a", True, EASY),),
                {"claude": {"a": _response(True)}, "gpt": {"a": _response(True)}},
            )[0],
        ),
        (
            StabilityScore("claude", 0.96, 120),
            StabilityScore("gpt", 0.97, 120),
        ),
    )

    assert verdict.passed is True
    assert verdict.failure_reasons == ()


def test_threshold_evaluation_fails_each_required_metric_branch() -> None:
    verdict = evaluate_thresholds(
        (
            ModelScore("claude", 0.94, 0.74, 0.90, 30, 10, 20, ("t1",), ()),
            _passing_score("gpt"),
        ),
        pairwise_agreements(
            (_triple("a", True, EASY), _triple("b", True, HARD)),
            {
                "claude": {"a": _response(True), "b": _response(True)},
                "gpt": {"a": _response(True), "b": _response(False)},
            },
        ),
        (
            StabilityScore("claude", 0.94, 120),
            StabilityScore("gpt", 0.96, 120),
        ),
    )

    assert verdict.passed is False
    assert verdict.failure_reasons == (
        "claude: easy accuracy 0.940 below 0.950",
        "claude: hard accuracy 0.740 below 0.750",
        "claude: high-confidence accuracy 0.900 not above 0.900",
        "claude: missing responses",
        "claude/gpt: inter-model agreement 0.500 below 0.850",
        "claude: intra-model stability 0.940 below 0.950",
    )


def test_threshold_evaluation_fails_closed_on_partial_coverage() -> None:
    verdict = evaluate_thresholds(
        (
            _passing_score("gpt"),
            _passing_score("opus"),
            _passing_score("sonnet"),
        ),
        (PairwiseAgreement("opus", "sonnet", 0.92, 40),),
        (StabilityScore("opus", 0.96, 120),),
    )

    assert verdict.passed is False
    assert verdict.failure_reasons == (
        "gpt/opus: inter-model agreement missing",
        "gpt/sonnet: inter-model agreement missing",
        "gpt: intra-model stability missing",
        "sonnet: intra-model stability missing",
    )


def test_threshold_evaluation_fails_closed_when_required_metrics_are_absent() -> None:
    verdict = evaluate_thresholds(
        (),
        (),
        (),
        BenchmarkThresholds(),
    )

    assert verdict.failure_reasons == (
        "no model scores",
        "inter-model agreement missing",
        "intra-model stability missing",
    )
