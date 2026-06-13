# PR-Deflection-Zendesk-API-Export-Import

## Why this slice exists

#1520 ingests Zendesk `{ticket, comments}` JSON and #1523/#1524 proved the
private-upload handoff. The remaining launch gap is producing that artifact
from Zendesk's API instead of hand-saving JSON. Zendesk's docs keep comments on
`GET /api/v2/tickets/{ticket_id}/comments` and tickets on cursor incremental
export.
This PR is over the 400 LOC soft cap because the exporter, cursor-pagination
fix, failure-branch tests, CI enrollment, and merged-plan archive need to ship
together to avoid another false-green API adapter.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a pure extracted Zendesk full-thread exporter returning
   `{tickets: [{ticket, comments}]}`.
2. Use a mocked transport in CI; no live calls, token logging, or hosted route.
3. Prove the exported artifact feeds `rows_from_zendesk_full_thread`.
4. Enroll the new exporter test in `run_extracted_pipeline_checks.sh`.
5. Archive the merged #1524 plan doc and refresh `plans/INDEX.md`.

### Review Contract
- Acceptance criteria:
  - [ ] The exporter requests tickets/comments with the existing Basic-token
        credential shape and never exposes the token.
  - [ ] Ticket/comment envelope drift fails closed.
  - [ ] Pagination/limit behavior is deterministic: cursor `after_url` is
        followed until `end_of_stream`, no more than `limit` tickets are
        exported, and comment pages are guarded against cycles.
  - [ ] The existing importer accepts the artifact and sees public/private
        comment flags.
  - [ ] The new test is enrolled in extracted CI in the same PR.
- Affected surfaces: extracted Zendesk exporter, extracted CI test list, plan
  archive/index.
- Risk areas: token leakage, live-network CI, malformed exports, cursor
  pagination, limits, accidental importer bypass.
- Reviewer rules triggered: R1, R2, R3, R10, R12, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `extracted_content_pipeline/support_ticket_zendesk_export.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Zendesk-API-Export-Import.md`
- `plans/archive/PR-Deflection-Zendesk-Live-Upload-Smoke.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_support_ticket_zendesk_export.py`

## Mechanism

The exporter accepts credentials, a limit, and a transport protocol. It calls
cursor incremental tickets with `page[size]`, follows `after_url` until
`end_of_stream`, then calls ticket comments for each selected ticket.

Tests use a fake transport that records URLs/headers and returns Zendesk-shaped
pages. Missing or malformed ticket/comment envelopes, malformed cursor
continuation, and comment-page cycles raise sanitized errors before the importer
runs.

## Intentional

- No hosted API route or UI button yet; this is the reusable artifact producer.
- No direct live Zendesk call in CI. The operator can run a local proof with
  trial credentials after merge.
- Private comments stay in the artifact because importer filtering owns that.

## Deferred

- Hosted tenant-scoped route wired to `content_ops_zendesk_credentials`.
- Browser flow that starts from stored Zendesk credentials.
- Redacted live trial-account proof artifact, only if useful and safe.

Parked hardening: none.

## Verification

- Focused exporter/importer review-fix pytest: 8 passed, 56 deselected.
- CI-enrollment pytest: 1 passed after enrolling the new test in workflow
  filters.
- Extracted pipeline check: passed on rerun, 4028 passed, 10 skipped; one
  existing torch/pynvml warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `extracted_content_pipeline/support_ticket_zendesk_export.py` | 221 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Zendesk-API-Export-Import.md` | 98 |
| `plans/archive/PR-Deflection-Zendesk-Live-Upload-Smoke.md` | 0 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_support_ticket_zendesk_export.py` | 195 |
| **Total** | **520** |
