# PR-Deflection-Full-Volume-Smoke-Gates
## Why this slice exists
#1440 needs a real-condition proof using a near-50 MB raw CFPB CSV through hosted
submit. #1452 made the route accept that file, but the reusable smoke still lets
a 3-row fixture pass a full-volume proof command.

Root cause: the harness validates handoff shape but not real-volume evidence.
The upstream fix is hosted submit smoke gates for uploaded bytes, rows,
generated questions, repeat-ticket volume, and visible top questions.
## Scope (this PR)
Ownership lane: deflection-full-50k-e2e-proof
Slice phase: Functional validation

1. Add optional minimum-volume gates to the hosted submit handoff smoke.
2. Gate on existing submit metadata and free snapshot summary only.
3. Prove low-volume payloads fail under gates while no-gate fixtures remain valid.
### Review Contract
- Acceptance criteria:
  - Callers can require uploaded bytes, row counts, generated questions, repeat
    tickets, and top questions.
  - Missing or below-threshold metrics fail closed with a specific gate error.
  - Existing no-gate behavior stays compatible and summaries stay redacted.
- Affected surfaces: hosted deflection submit smoke and its CI-enrolled tests.
- Risk areas: skipped metrics, broken fixture behavior, or CI wording implying
  live email/payment proof.
- Reviewer rules triggered: R1, R2, R10, R13, R14.
### Files touched

- `plans/PR-Deflection-Full-Volume-Smoke-Gates.md`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

## Mechanism
The smoke already captures submit metadata and validates the free snapshot. This
slice adds parser options and a deterministic validator reading only existing
scalar fields: uploaded/blob bytes, rows, generated count, repeat-ticket count,
and top-question count.

Missing, non-numeric, or below-threshold values fail closed. Configured gates
that cannot run because submit/envelope validation failed are marked skipped
with `ok: false`, so the summary cannot imply the gates passed.

## Intentional
- No live email, PDF, Stripe, or hosted payment run in CI; this PR only makes
  hosted submit enforce full-volume evidence.
- No new orchestration script. The existing submit smoke is the upstream proof
  boundary for raw upload -> snapshot -> locked artifact.
- No generator or clustering behavior change.

## Deferred
- After deployment, run #1440 live with the near-50 MB CFPB CSV: submit raw CSV,
  confirm snapshot/email/PDF, complete payment, drain delivery, and confirm the
  full report email/PDF.
Parked hardening: none.
## Verification
- Passed: python -m pytest tests/test_smoke_content_ops_deflection_submit_handoff.py -q (37 passed).
- Passed: python -m py_compile scripts/smoke_content_ops_deflection_submit_handoff.py tests/test_smoke_content_ops_deflection_submit_handoff.py
- Passed: bash scripts/check_ascii_python.sh
- Passed: bash scripts/run_extracted_pipeline_checks.sh (4182 passed, 10 skipped, 1 warning).
## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Full-Volume-Smoke-Gates.md` | 64 |
| `scripts/smoke_content_ops_deflection_submit_handoff.py` | 128 |
| `tests/test_smoke_content_ops_deflection_submit_handoff.py` | 200 |
| **Total** | **392** |
