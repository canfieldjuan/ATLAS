# PR-Deflection-Proof-Fixture-Request-ID-Redaction

## Why this slice exists

#1617 is code-complete but `extracted-checks` remains red because `origin/main`
contains a detector test fixture with a raw production-shaped deflection request
ID. The failing guard is `tests/test_docs_no_raw_deflection_request_ids.py`,
which correctly rejects committed `content-ops-<32 hex>` tokens. The source is
`tests/test_check_deflection_full_report_proof_bundle.py`, where the proof-bundle
detector used exactly that production-shaped token as its "bad request ID"
example.

Root cause: the proof-bundle detector fixture conflated "unsafe for the bundle
checker" with "production-shaped secret in the repository." That makes the
detector test itself violate the repo-wide raw-request-id guard.

This fixes the root in safe scope by changing the fixture to a clearly
non-production `content-ops-fixture-...` token that still fails the proof-bundle
checker but no longer matches the committed-secret guard. The repo-wide guard
stays strict; no allowlist is added.

## Scope (this PR)

Ownership lane: content-ops/deflection-report
Slice phase: Production hardening

1. Replace the proof-bundle request-id fixture token with a non-production
   fixture-shaped token that the proof-bundle checker still treats as unsafe.
2. Keep the repo-wide committed raw-request-id detector unchanged.
3. Add/adjust regression coverage proving the proof-bundle checker still flags
   the fixture token and the repo-wide guard no longer flags committed files.

### Files touched

- `plans/PR-Deflection-Proof-Fixture-Request-ID-Redaction.md`
- `tests/test_check_deflection_full_report_proof_bundle.py`
- `tests/test_docs_no_raw_deflection_request_ids.py`

### Review Contract

- Acceptance criteria:
  - [ ] The proof-bundle checker still fails closed for unsafe request IDs and
        result URLs.
  - [ ] No committed file contains a `content-ops-<32 hex>` request-id token.
  - [ ] The fix does not weaken the repo-wide raw-request-id detector.
  - [ ] #1617's inherited `extracted-checks` blocker is addressed at the source.
- Affected surfaces: deflection proof-bundle checker tests, repo-wide
  deflection request-id leak guard tests, plan.
- Risk areas: security test precision, false-negative detector behavior, CI
  gate integrity.
- Reviewer rules triggered: R1, R2, R3, R10, R12, R14.

## Mechanism

Use a shared test constant with a fixture-shaped token:
`content-ops-fixture-<hex>`. The proof-bundle checker's unsafe request-id
pattern still matches this token, and the token does not contain "synthetic" or
"example", so the checker still reports it as unsafe. The repo-wide committed
secret guard only rejects production-shaped `content-ops-<32 hex>` tokens, so it
no longer mistakes the detector fixture for a leaked live request ID.

## Intentional

- No allowlist is added to the repo-wide guard. A file-level allowlist would hide
  future accidental real request IDs in the same detector test.
- The fixture token intentionally remains unsafe for the proof-bundle checker.
  This preserves the detector's failure branch instead of weakening the test to
  pass on a safe placeholder.

## Deferred

- Any live-token revocation or historical artifact audit is outside this
  fixture-only remediation because the offending committed value is being
  removed from the repository surface here.

Parked hardening: none.

## Verification

- `.venv/bin/python -m pytest tests/test_check_deflection_full_report_proof_bundle.py tests/test_docs_no_raw_deflection_request_ids.py` — 37 passed.
- `bash` with `scripts/run_extracted_pipeline_checks.sh` — 4418 passed, 10 skipped.
- `bash` with `scripts/local_pr_review.sh` and the PR body file — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Proof-Fixture-Request-ID-Redaction.md` | 91 |
| `tests/test_check_deflection_full_report_proof_bundle.py` | 2 |
| `tests/test_docs_no_raw_deflection_request_ids.py` | 12 |
| **Total** | **105** |
