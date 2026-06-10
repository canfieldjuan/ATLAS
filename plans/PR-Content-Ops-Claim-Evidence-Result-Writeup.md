# PR-Content-Ops-Claim-Evidence-Result-Writeup

## Why this slice exists

#1477 merged the deterministic result artifact for #1435. The remaining issue
deliverable is the operator-facing writeup: a readable benchmark summary before
any verifier/MCP inclusion.

This PR renders an already-built artifact into deterministic Markdown so later
CLI/live-run slices can reuse the same presentation rules.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a pure Markdown renderer for `ClaimEvidenceResultArtifact`.
2. Include go/no-go, thresholds, model score table, inter-model
   agreement rows, stability rows, verdict failure reasons, artifact errors,
   and failure-case list.
3. Render malformed input as a no-go writeup with a visible artifact error.
4. Add focused tests for passing, no-go, and malformed renderer input.
5. Keep provider adapters, live prompt execution, model-run CLI,
   file writing, verifier rubric inclusion, MCP tools, database behavior, and
   live-client behavior out of scope.

### Review Contract

- Acceptance criteria:
  - [ ] Passing artifacts render go/no-go, thresholds, model scores, agreement,
        and stability sections.
  - [ ] No-go artifacts render verdict reasons, artifact errors, and failure
        cases without raising.
  - [ ] Non-artifact input renders a failed no-go writeup with an attributable
        error.
  - [ ] The renderer only presents artifact/threshold fields and does not add
        scoring policy.
  - [ ] No provider, file writing, verifier/MCP, DB, or live-client behavior is
        introduced.
- Affected surfaces: extracted benchmark helper, focused tests, and plan.
- Risk areas: false-green presentation, malformed artifact output, future
  writeup compatibility.
- Reviewer rules triggered: R1, R2, R5, R6, R10, R12.

### Files touched

- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `plans/PR-Content-Ops-Claim-Evidence-Result-Writeup.md`
- `tests/test_extracted_content_claim_evidence_benchmark.py`

## Mechanism

The benchmark module gains a renderer that accepts an artifact and returns
Markdown. It formats percentages, renders empty tables explicitly, and lists
artifact/verdict errors before failure cases. Malformed input becomes a failed
artifact-shaped report instead of an exception.

## Intentional

- The renderer stays in the existing benchmark module as artifact presentation,
  not host-layer wiring.
- The renderer does not parse JSON or write files.
- The go/no-go text derives from the artifact status only; this PR does not add
  new threshold policy or constrained-go logic.
- Hardening scan: no parked `HARDENING.md` entries touch this ownership lane or
  these files.

## Deferred

- Provider adapters, live prompt execution, model-run CLI, and file output.
- Verifier rubric inclusion and MCP exposure only after benchmark results pass.
- Follow-up drift-class guard for schema/decoder/artifact strictness claims.

Parked hardening: none.

## Verification

- Focused benchmark pytest - 48 passed; Python compile check - passed; diff whitespace check - passed.
- Extracted guardrails passed: manifest validation, Atlas reasoning import guard, standalone audit, and ASCII Python check.
- Full extracted pipeline wrapper - 3744 passed, 10 skipped.
- Body-aware local PR review - passed.

## Estimated diff size

Estimated: 3 files, about +399 / -0. Below the 400 LOC target.

| Total | 399 |
