# PR-Deflection-Zendesk-Full-Thread-Proof

## Why this slice exists

Issue #1419 still needs the publishable-answer lane proven on a real
resolution-bearing Zendesk full-thread export. The CSV/intake path can already
surface gap-list diagnostics, and #1532 wired per-tenant Zendesk credentials,
but the launch gate still needs a deterministic proof that the API-shaped
`tickets + comments` export can flow through submit -> paid artifact and
produce at least one publishable FAQ answer without leaking private notes.

This is a vertical proof slice, not a new importer: the Zendesk full-thread
normalizer, full-thread submit mode, and paid-report artifact path already
exist. This PR pins their combined behavior with the seeded Zendesk trial JSON
fixture from #1507/#1419.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Extend the existing full-thread deflection-submit test so the Zendesk
   API-shaped JSON fixture is unlocked through the paid artifact route, not
   only accepted as metadata.
2. Assert the launch-critical proof points: `drafted_answer_count > 0`,
   `resolution_evidence` answer status, refund-answer copy present, private
   note text absent, auto-ack text absent, and unresolved/reopen-style public
   text does not become publishable copy merely because status/CSAT/reopen
   diagnostics exist.
3. Add a compact validation note pointing reviewers/operators at the source
   fixture and generated proof values.

### Review Contract

Acceptance criteria:
- The seeded Zendesk full-thread fixture reaches the paid report artifact path
  and produces a summary with at least one drafted answer backed by
  `resolution_evidence`.
- The generated artifact contains the public refund-answer resolution from the
  agent reply.
- Internal/private note text and boilerplate auto-ack text are absent from the
  free payload, paid artifact, and markdown/report strings asserted by the
  test.
- Status and CSAT metadata remain exposed as diagnostics, but do not create
  extra publishable answers by themselves.
- No live Zendesk API call runs in CI; the proof uses the committed
  `tests/fixtures/zendesk_full_thread_seed_sample.json` producer-shaped
  artifact.

Affected surfaces:
- `tests/test_extracted_content_deflection_submit.py`
- `docs/extraction/validation/deflection_zendesk_full_thread_proof_2026-06-13.md`

Risk areas:
- Accidentally asserting only package metadata while leaving the paid report
  artifact unproved.
- Leaking private/internal notes into customer-facing strings.
- Treating diagnostics such as open status or CSAT as answer evidence.
- Adding a new test without enrolling it in the extracted-checks suite.

Reviewer rules triggered: R1, R2, R12, R14.

### Files touched

- `docs/extraction/validation/deflection_zendesk_full_thread_proof_2026-06-13.md`
- `docs/extraction/validation/fixtures/deflection_zendesk_full_thread_proof_20260613/report_excerpt.md`
- `docs/extraction/validation/fixtures/deflection_zendesk_full_thread_proof_20260613/summary.json`
- `plans/PR-Deflection-Zendesk-Full-Thread-Proof.md`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

The existing `test_deflection_submit_accepts_zendesk_full_thread_blob` already
submits the Zendesk seed JSON through the control-surface route and checks the
normalized package metadata. This PR extends that same enrolled route test:

1. Submit the API-shaped Zendesk full-thread blob with
   `importer_mode="full_thread"`.
2. Confirm the locked/free response still hides the full report.
3. Mark the request paid through the existing paid route.
4. Fetch the artifact route and assert the summary/markdown prove the
   publishable-answer lane while excluding private notes and boilerplate.

The validation doc records the same fixture, run target, and headline generated
values for the operator/reviewer. The test is added to an existing test file
already listed in `scripts/run_extracted_pipeline_checks.sh`, so no new CI
enrollment step is needed.

## Intentional

- No live Zendesk API call in this PR. The trial API shape was already captured
  as `tests/fixtures/zendesk_full_thread_seed_sample.json`; CI must stay
  deterministic and token-free.
- No new CLI/importer mode. The route and importer are already merged; this
  slice proves the existing full-thread path rather than expanding product
  surface.
- The fixture has one publishable answer and several non-repeat/unresolved
  tickets. That is enough for the vertical proof because #1419's remaining
  launch gate is "can a full-thread export produce publishable answers at all?"
  while preserving private-note safety.

## Deferred

- Reopen/CSAT product scoring remains deferred to the product-decision follow-up
  already called out in #1419/#1510. This slice only proves those fields remain
  diagnostics and do not create answer evidence on their own.
- Live operator run against a fresh Zendesk trial/export remains outside CI;
  this PR pins the committed producer-shaped artifact so the regression guard
  is stable.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_content_deflection_submit.py::test_deflection_submit_accepts_zendesk_full_thread_blob -q` - 1 passed.
- `pytest tests/test_extracted_content_deflection_submit.py -q` - 53 passed.
- `pytest tests/test_extracted_support_ticket_input_package.py -q` - 61 passed.
- `scripts/run_extracted_pipeline_checks.sh` via bash - extracted reasoning core 295 passed; extracted content pipeline 4080 passed, 10 skipped; all checks completed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_zendesk_full_thread_proof_2026-06-13.md` | 58 |
| `docs/extraction/validation/fixtures/deflection_zendesk_full_thread_proof_20260613/report_excerpt.md` | 16 |
| `docs/extraction/validation/fixtures/deflection_zendesk_full_thread_proof_20260613/summary.json` | 22 |
| `plans/PR-Deflection-Zendesk-Full-Thread-Proof.md` | 129 |
| `tests/test_extracted_content_deflection_submit.py` | 40 |
| **Total** | **265** |
