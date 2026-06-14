# PR-Content-Ops-Claim-Evidence-Provider-Runner

## Why this slice exists

#1435 now has the manual benchmark path from labeled fixture through prompt
packets, returned response import, and result artifacts. The next remaining gap
is provider execution: concrete Claude/GPT adapters should be thin wrappers
around a deterministic runner seam, not places where coverage, stability, or
row-shape rules are reinvented.

This slice lands that seam without credentials or network calls. It takes the
already-exported prompt packets and an injected provider boundary, produces the
same returned response rows consumed by the #1530 importer, and can request
main plus stability reruns with deterministic run ids.

The estimate is slightly above the 400 LOC target because this is a new
package-level runner plus same-slice negative fixtures for each detector branch.
Splitting the tests would leave the provider seam without CI-proven failure
detection.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Add a package-owned provider runner that executes prompt packets through an
   injected provider boundary and emits returned response rows in the existing
   importer contract.
2. Support a main run plus optional stability reruns per model with stable
   run-id labels.
3. Fail closed on malformed packet rows, duplicate packet identities, invalid
   response payloads, provider exceptions, invalid stability counts, and missing
   provider results without calling live models.
4. Add focused tests and enroll them in extracted checks.

### Files touched

- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `plans/PR-Content-Ops-Claim-Evidence-Provider-Runner.md`
- `tests/test_extracted_content_claim_evidence_benchmark.py`

### Review Contract

Acceptance criteria:

- A valid prompt-packet set plus injected provider responses produces returned
  response rows accepted by the existing response importer.
- Stability reruns are explicitly labeled and do not overwrite main rows.
- Provider exceptions and malformed provider responses become row errors and no
  silent successful response is emitted.
- The runner remains deterministic and package-owned: no Atlas imports, no DB,
  no credentials, no HTTP, no MCP, and no verifier rubric inclusion.
- New or changed tests are enrolled in the extracted pipeline checks.

Affected surfaces:

- Package-owned claim/evidence benchmark module and focused tests.

Risk areas:

- Prompt-packet row validation must match the exporter/importer identity shape.
- Stability row labeling must remain deterministic.
- Provider errors must fail closed without stopping later rows.
- CI enrollment for any new test coverage.

Reviewer rules triggered: R1, R2, R10, R12, R14

## Mechanism

The runner adds a small data contract around the existing prompt packet shape:

1. Decode prompt packet mappings into typed packet records that preserve
   model id, triple id, contract version, prompt, and response schema.
2. Call an injected provider with each packet for the main run, then repeat for
   each requested stability run.
3. Decode each provider return through the existing strict claim/evidence
   response contract.
4. Emit row mappings with the same fields the prompt-response importer expects:
   model id, triple id, contract version, response, and optional stability
   run metadata.

The future live-provider PR should only adapt API clients to this provider
boundary. It should not change benchmark scoring, artifact generation, or MCP
verifier behavior.

## Intentional

- No OpenAI/Anthropic SDK usage in this slice. That keeps credentials, network
  behavior, rate limits, and model-specific JSON-mode quirks out until the
  deterministic orchestration seam is reviewed.
- No CLI in this slice. The provider runner is package-owned core; a later CLI
  can bind it to concrete provider configuration without duplicating row rules.
- No verifier/MCP inclusion. The issue still requires real benchmark results and
  threshold admission before a structured-witness slot enters the verifier.

## Deferred

- Concrete Claude/GPT provider adapters and credential loading.
- Batch CLI that binds provider adapters to fixture/prompt-packet files and
  writes returned response rows.
- Real benchmark go/no-go writeup from the final operator-labeled set.
- Verifier rubric inclusion and MCP exposure, only after #1435 thresholds pass.

Parked hardening: none.

## Verification

- Focused claim/evidence benchmark tests: 70 passed.
- Full extracted pipeline wrapper: 4093 passed, 10 skipped.
- Body-aware local PR review: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/claim_evidence_benchmark.py` | 182 |
| `plans/PR-Content-Ops-Claim-Evidence-Provider-Runner.md` | 119 |
| `tests/test_extracted_content_claim_evidence_benchmark.py` | 189 |
| **Total** | **490** |
