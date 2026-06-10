# PR-Content-Ops-Claim-Evidence-Prompt-Schema

## Why this slice exists

#1435 needs a provider runner eventually, but the runner should not invent the
prompt or response contract at call time. After #1465, operators can validate
benchmark fixtures locally; the next missing deterministic artifact is the
structured-witness prompt and JSON Schema that every future provider run will
use for the claim/evidence support slot.

This PR captures that contract without calling any model. It keeps the
reliability gate reproducible: the prompt text, response schema, and response
decoder are versioned and tested before provider execution or benchmark results
exist.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a deterministic prompt/schema helper for the `verify_claim_evidence`
   structured-witness slot.
2. Add the missing `claim_text` fixture field so the prompt carries the actual
   claim statement while `claim_id` remains traceability metadata.
3. Keep the response schema aligned with the existing decoded response
   contract: `supports`, `confidence`, and `reason`.
4. Add focused tests proving prompt content, schema required fields, no extra
   response fields, and response decoder compatibility.
5. Document the prompt/schema contract next to the fixture/operator handoff.

### Review Contract

- Acceptance criteria:
  - [ ] The fixture contract requires `claim_text` separately from `claim_id`.
  - [ ] The prompt contract includes claim text, claim id, evidence quote,
        source id, and difficulty from one benchmark triple.
  - [ ] The prompt tells the witness to judge whether the evidence supports the
        claim, not whether the claim is generally true.
  - [ ] The JSON Schema requires `supports`, `confidence`, and `reason`.
  - [ ] The JSON Schema rejects extra response fields.
  - [ ] The schema bounds confidence to integer values from 1 through 5.
  - [ ] The schema rejects whitespace-only rationale text the same way the
        decoded response validator does.
  - [ ] The schema stays compatible with the existing decoded response
        validator.
  - [ ] No provider/model runner, prompt execution, result artifact, verifier
        rubric wiring, MCP tool, database, or live-client behavior is
        introduced.
- Affected surfaces: extracted benchmark helper, focused benchmark tests, docs,
  and plan.
- Risk areas: benchmark false-green, schema drift, future provider-contract
  incompatibility, CI enrollment.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `docs/content_ops_claim_evidence_benchmark_fixtures.md`
- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `plans/PR-Content-Ops-Claim-Evidence-Prompt-Schema.md`
- `tests/test_extracted_content_claim_evidence_benchmark.py`
- `tests/test_validate_content_ops_claim_evidence_fixture.py`

## Mechanism

The benchmark module gains a pure prompt/schema contract object with a stable
contract version, a JSON Schema mapping, and a rendered prompt string for a
single `ClaimEvidenceTriple`.

The prompt is intentionally narrow: it passes only the triple fields the
benchmark owns and instructs the witness to decide support from the provided
evidence quote alone. The schema is intentionally strict: only `supports`,
`confidence`, and `reason` are allowed, `confidence` is bounded to the same
1 through 5 range accepted by the existing response decoder, and `reason`
requires non-whitespace text.

Focused tests assert the prompt includes the right triple fields and guardrail
language, then validate the schema shape against the decoded response contract.

## Intentional

- The prompt/schema helper lives in the existing benchmark module instead of a
  provider-runner module. The runner is still deferred, and keeping the contract
  adjacent to the scorer prevents a second source of truth.
- The schema remains provider-neutral JSON Schema rather than OpenAI-,
  Anthropic-, or FastMCP-specific metadata. Provider adapters can wrap it later.
- The prompt does not include ground-truth labels or benchmark thresholds. Those
  remain scorer inputs, not witness inputs.
- `claim_text` is added to the unmerged fixture contract instead of overloading
  `claim_id`. That makes existing draft examples update, but preserves
  traceability semantics before any real benchmark fixture is published.

## Deferred

- Provider/model runner and prompt execution.
- Result artifact, scoring table, agreement matrix, failure-case list, and
  go/no-go writeup.
- Batch fixture directories and operator labeling workflow.
- Verifier rubric inclusion and MCP exposure only after benchmark results pass.

Parked hardening: none.

## Verification

- pytest tests/test_extracted_content_claim_evidence_benchmark.py tests/test_validate_content_ops_claim_evidence_fixture.py - 41 passed.
- python -m py_compile extracted_content_pipeline/claim_evidence_benchmark.py tests/test_extracted_content_claim_evidence_benchmark.py tests/test_validate_content_ops_claim_evidence_fixture.py - passed.
- git diff --check - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash scripts/run_extracted_pipeline_checks.sh - 3701 passed, 10 skipped.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_prompt_schema_pr_body.md - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/content_ops_claim_evidence_benchmark_fixtures.md` | 50 |
| `extracted_content_pipeline/claim_evidence_benchmark.py` | 78 |
| `plans/PR-Content-Ops-Claim-Evidence-Prompt-Schema.md` | 123 |
| `tests/test_extracted_content_claim_evidence_benchmark.py` | 80 |
| `tests/test_validate_content_ops_claim_evidence_fixture.py` | 1 |
| **Total** | **332** |
