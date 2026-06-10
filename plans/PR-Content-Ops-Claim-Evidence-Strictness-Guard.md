# PR-Content-Ops-Claim-Evidence-Strictness-Guard

## Why this slice exists

Reviews across #1469, #1475, and #1477 surfaced the same drift class:
the contract advertised strictness while the validator accepted a looser shape.
#1484 carried this as a deferred follow-up. Before moving toward live provider
adapters or batch model-run CLI, this PR pins the response-schema strictness
claims to decoder behavior in one guard test.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Robust testing

1. Add a consolidated regression test for the `verify_claim_evidence.v1`
   response schema claims.
2. Prove required fields, boolean support, integer confidence bounds, string
   non-whitespace reason, and no-extra-fields are all enforced by the decoded
   response validator.
3. Keep production code, provider adapters, live prompt execution, model-run
   CLI, file writing, verifier rubric inclusion, MCP tools, DB, and live-client
   behavior out of scope.

### Review Contract

- Acceptance criteria:
  - [ ] Each response-schema rejection claim has a matching decoder failure
        assertion in one guard test.
  - [ ] The guard covers lookalikes that caused prior drift: boolean confidence,
        non-string/whitespace reason, and extra provider fields.
  - [ ] No production behavior changes are introduced.
- Affected surfaces: focused benchmark tests and plan.
- Risk areas: false-green contract strictness, schema/decoder drift, test
  coverage brittleness.
- Reviewer rules triggered: R1, R2, R10, R12.

### Files touched

- `plans/PR-Content-Ops-Claim-Evidence-Strictness-Guard.md`
- `tests/test_extracted_content_claim_evidence_benchmark.py`

## Mechanism

The benchmark test suite gains one table-driven guard. It reads the response
schema fields and then feeds same-class rejected values through
`ClaimEvidenceResponse.from_mapping`, asserting the specific decoder errors.
The existing per-case tests stay in place; this guard is the class-level net.

## Intentional

- This is test-only because the strict decoder behavior already exists.
- The guard lives in the existing benchmark test module so no CI enrollment
  change is needed.
- Hardening scan: no parked `HARDENING.md` entries touch this ownership lane or
  these files.

## Deferred

- Provider adapters, live prompt execution, model-run CLI, and file output.
- Verifier rubric inclusion and MCP exposure only after benchmark results pass.

Parked hardening: none.

## Verification

- Focused benchmark pytest - 49 passed.
- Python compile check and diff whitespace check - passed.
- Full extracted pipeline wrapper - 3872 passed, 10 skipped.
- Body-aware local PR review - passed.

## Estimated diff size

Estimated: 2 files, about +152 / -0. Below the 400 LOC target.

| Total | 152 |
