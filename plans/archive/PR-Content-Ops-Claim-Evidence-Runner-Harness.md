# PR-Content-Ops-Claim-Evidence-Runner-Harness

## Why this slice exists

#1469 captured the prompt and JSON Schema contract for the #1435 slot. The
benchmark still needs a provider-boundary runner that renders the prompt,
decodes the structured response, and preserves per-row failures. This PR lands
that seam with fake-provider tests only: no credentials, network calls, result
artifacts, verifier wiring, or MCP behavior.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a deterministic runner harness to the existing benchmark module.
2. Render the merged prompt/schema contract and call an injected provider.
3. Decode provider responses through the merged response validator.
4. Preserve provider exceptions and malformed responses as per-row errors.
5. Add fake-provider tests for success, malformed output, exceptions, and
   invalid harness input.

### Review Contract

- Acceptance criteria:
  - [ ] Provider is called once per valid triple with model id, triple, prompt,
        contract version, and response schema.
  - [ ] Successful responses become a `triple_id` response map.
  - [ ] Missing model id, empty/non-sequence triples, non-callable provider,
        non-triple inputs, malformed responses, and provider exceptions fail
        closed without stopping later triples.
  - [ ] No live provider adapter, credential lookup, network call, result file,
        verifier/MCP wiring, database, or live-client behavior is introduced.
- Affected surfaces: extracted benchmark helper, focused benchmark tests, docs,
  and plan.
- Risk areas: benchmark false-green, provider-contract drift, malformed output,
  future result-artifact compatibility, CI enrollment.
- Reviewer rules triggered: R1, R2, R5, R6, R10, R12.

### Files touched

- `docs/content_ops_claim_evidence_benchmark_fixtures.md`
- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `plans/PR-Content-Ops-Claim-Evidence-Runner-Harness.md`
- `tests/test_extracted_content_claim_evidence_benchmark.py`

## Mechanism

The benchmark module gains immutable per-row and per-model run results. The
runner accepts a model id, existing `ClaimEvidenceTriple` objects, and an
injected provider callable. For each valid triple, it builds the #1469
prompt/schema contract, calls the provider, decodes the returned mapping with
the response decoder, and records row errors instead of raising. The decoder
also enforces the schema's strict response field set so extra provider fields
cannot enter the scoreable response map.

## Intentional

- The provider is injected, not a concrete OpenAI, Anthropic, OpenRouter, or
  MCP client. Credential and retry policy decisions belong in a later slice.
- The runner records per-row errors but does not score or write artifacts.
- The runner catches provider exceptions by class name only. It does not log or
  persist provider messages here, which avoids accidentally storing prompt,
  token, or vendor diagnostics in this pure package layer.
- The harness stays in the existing benchmark module because this is still the
  package-owned deterministic reliability gate, not host-layer provider wiring.

## Deferred

Concrete provider adapters, live model execution, batch CLI, result artifact,
scoring table, agreement matrix, failure list, go/no-go writeup, batch fixture
directories, operator labeling workflow, verifier rubric inclusion, and MCP
exposure all remain deferred until benchmark results justify them.

Parked hardening: none.

## Verification

- pytest tests/test_extracted_content_claim_evidence_benchmark.py - 41 passed.
- python -m py_compile extracted_content_pipeline/claim_evidence_benchmark.py tests/test_extracted_content_claim_evidence_benchmark.py - passed.
- git diff --check - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash scripts/run_extracted_pipeline_checks.sh - 3708 passed, 10 skipped.
- bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file tmp/content_ops_claim_evidence_runner_harness_pr_body.md - passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_runner_harness_pr_body.md - passed.

## Estimated diff size

Estimated: 4 files, about +398 / -1 after review fixes.

| Total | 399 |
