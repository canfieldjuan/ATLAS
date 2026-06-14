# PR-Deflection-Full-Volume-Live-Proof

## Why this slice exists

#1440 asks for the real deflection delivery proof under full-volume conditions,
not another tiny fixture or harness-only pass. #1555 closed the last vacuous
smoke gap by adding volume gates; the re-review explicitly called out that the
next action should be the live #1440 proof rather than another submit/smoke
hardening slice.

The previously generated near-50 MB CFPB CSV is no longer present locally, but
the source JSONL still exists under the operator's local `tmp/` artifacts. This
slice regenerates the upload CSV, runs the hosted submit proof with the new
volume gates, and commits only sanitized evidence.

## Scope (this PR)

Ownership lane: deflection-full-50k-e2e-proof
Slice phase: Functional validation

1. Regenerate the local near-50 MB CFPB upload CSV from the existing CFPB JSONL
   scale artifact; do not commit the raw CSV.
2. Run the hosted submit handoff smoke against the deployed ATLAS host with
   full-volume gates enabled.
3. Continue through the hosted result-page and paid-unlock proof commands to
   identify the next live blockers.
4. Commit a sanitized validation artifact that records command shapes, scalar
   counts, request identifiers where safe, and any blocked live sub-step.

### Review Contract

- Acceptance criteria:
  - The committed artifact proves the run used a regenerated CFPB CSV, not the
    3-row fixture.
  - Hosted submit evidence includes uploaded bytes, parsed/submitted row counts,
    generated question count, repeat-ticket count, and top-question count.
  - The artifact redacts bearer tokens, webhook secrets, signed URLs, raw ticket
    text, and email contents.
  - Any live sub-step that cannot run names the missing env or hosted failure
    explicitly instead of silently passing.
- Affected surfaces: validation docs/artifacts and no product runtime code.
- Risk areas: leaking source data/secrets, mistaking local build proof for
  hosted proof, or hiding a blocked payment/email step.
- Reviewer rules triggered: R1, R2, R3, R10, R14.

### Files touched

- `docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md`
- `docs/extraction/validation/fixtures/deflection_full_volume_live_proof_20260614/summary.json`
- `plans/PR-Deflection-Full-Volume-Live-Proof.md`

## Mechanism

The local CSV regeneration uses the existing CFPB source-row JSONL artifact and
standard CSV quoting to rebuild the same support-ticket-shaped upload used by
the #1452 prep proof. The file is written under `tmp/` or `/tmp`, measured, and
used only as input to the hosted smoke.

The hosted proof starts with `smoke_content_ops_deflection_submit_handoff.py`
and the #1555 gates. Successful output is reduced to a sanitized summary: scalar
counts, HTTP statuses, request id, and pass/fail state. Raw source rows, report
Markdown, token values, webhook signatures, and email bodies are excluded from
committed artifacts.

If result-page, paid-unlock, or delivery fails, the validation doc records the
blocked sub-step with the exact live error. That prevents another "submit
harness passed, full delivery unproven" ambiguity.

## Intentional

- No product runtime code changes in this slice. The purpose is live validation
  after the harness gates landed.
- No raw CSV, paid report Markdown, PDF, or email body is committed; only
  sanitized summaries are allowed.
- No embedding-booster work. That lane is paused until its enablement decision.

## Deferred

- Decide whether the repeat-volume gate should be below the observed 27,384
  repeat-ticket count for regenerated CFPB uploads, or keep 30,000 and require
  a larger/different sample.
- Restore the public portfolio result route for
  `/services/faq-deflection/results/{request_id}`; all probed likely production
  hosts returned 404 for this request.
- Align the local/operator Stripe webhook secret with the deployed `atlas-brain`
  secret, then rerun paid unlock and delivery.

Parked hardening: none.

## Verification

- Passed: regenerated full-volume CFPB CSV locally; 40,383 records,
  52,428,276 bytes, SHA-256
  `43130a9a43c2bd821a16c2025694a14a45fc2a914f79cfe5278c88736b749193`.
- Partial pass: hosted submit accepted and processed the CSV in 64.47s
  (`submit=200`, `snapshot=200`, unpaid artifact `403`, 40,383 rows submitted,
  0 truncation, 1,659 generated questions). The command exited nonzero because
  the configured repeat-ticket gate expected 30,000 and the hosted result
  reported 27,384.
- Failed live surface: portfolio result page returned `404` while ATLAS
  snapshot/artifact probes for the request remained `200`/`403`.
- Failed live surface: paid-unlock webhook returned `400`; direct error probe
  returned `Invalid signature`.
- Passed: python -m py_compile scripts/smoke_content_ops_deflection_submit_handoff.py scripts/smoke_content_ops_deflection_portfolio_result_page.py scripts/smoke_content_ops_deflection_stripe_paid_unlock.py scripts/prepare_content_ops_deflection_env.py
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-deflection-full-volume-live-proof.md

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md` | 159 |
| `docs/extraction/validation/fixtures/deflection_full_volume_live_proof_20260614/summary.json` | 78 |
| `plans/PR-Deflection-Full-Volume-Live-Proof.md` | 114 |
| **Total** | **351** |
