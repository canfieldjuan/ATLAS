# PR-Deflection-Full-Report-QA-Live-Rerun

## Why this slice exists

Issue #1612 is the full-report delivery QA arc. #1671 executed the controlled
live proof and failed safe on the hosted paid artifact contract: the hosted
report-model endpoint returned 404 and the hosted artifact lacked an object
`evidence_export`. #1674 then pinned the buyer-facing submit paid-flow contract
in source so that regression cannot silently pass CI again.

This slice exists to take the next proof step, not to build another harness:
rerun the controlled live proof against a fresh hosted paid Zendesk-shaped
request and commit only the sanitized summary. If the hosted deployment now
serves the modern contract, the committed summary should move #1612 from
source-contract proof to live artifact proof. If the hosted run still fails,
the summary records the remaining upstream production gap without committing
raw payloads or live capabilities.

Root cause for this validation slice: #1671 left the hosted live artifact proof
red, and #1674 intentionally fixed only the source contract gap. The unproven
question is whether the deployed paid artifact/report-model endpoints now
produce the same modern contract for a fresh customer-shaped request.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Functional validation

1. Create one fresh hosted Zendesk-shaped deflection request using the existing
   submit smoke and checked-in full-thread export fixture.
2. Unlock that request through the existing Stripe paid-unlock smoke using the
   deployed webhook path and replay check.
3. Render PDF bytes locally from the paid artifact using the same deflection
   delivery renderer path, then run the live QA runner without a PDF text
   override so text is derived from the PDF bytes.
4. Commit only the sanitized summary under
   `docs/extraction/validation/fixtures/deflection_full_report_qa_live_rerun_20260617/`.
5. Update #1612 with the rerun result.

### Review Contract

- Acceptance criteria:
  - [ ] The committed artifact contains no live request ID, bearer token,
        result URL, customer email, absolute local path, source ID list,
        private note, raw evidence quote, Stripe checkout session ID, or Stripe
        payment intent ID.
  - [ ] The summary records submit status, paid-unlock status, PDF-byte text
        extraction provenance, report-model fetch status, artifact fetch status,
        `evidence_export` presence, and final scorecard status.
  - [ ] Raw live request IDs, tokens, PDF bytes, report-model JSON, artifact
        JSON, email bodies, HTML, and evidence exports remain local only.
  - [ ] If the rerun fails, the failure is recorded as a sanitized live proof
        result and the plan names the upstream follow-up instead of claiming
        product success.
  - [ ] The proof-bundle redaction checker passes on the committed directory.
- Affected surfaces: #1612 proof artifacts and existing live proof scripts.
- Risk areas: committing live capabilities or PII, overstating a failed proof,
  stale hosted deployment, and drifting into new harness work.
- Reviewer rules triggered: R1, R2, R3, R6, R14.

### Files touched

- `docs/extraction/validation/fixtures/deflection_full_report_qa_live_rerun_20260617/summary.json`
- `plans/PR-Deflection-Full-Report-QA-Live-Rerun.md`

## Mechanism

The slice reuses the already-merged proof path:

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py \
  --csv-file extracted_content_pipeline/examples/support_ticket_provider_exports/zendesk_full_thread_export.csv \
  --output-result tmp/deflection_full_report_qa_live_rerun_20260617/submit-result.json \
  --json

python scripts/smoke_content_ops_deflection_stripe_paid_unlock.py \
  --request-id "$CONTROLLED_PAID_REQUEST_ID" \
  --output-result tmp/deflection_full_report_qa_live_rerun_20260617/paid-unlock.json \
  --replay-webhook \
  --json

python scripts/run_deflection_full_report_qa_live_runner.py \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "$ATLAS_B2B_SERVICE_TOKEN" \
  --request-id "$CONTROLLED_PAID_REQUEST_ID" \
  --pdf-bytes tmp/deflection_full_report_qa_live_rerun_20260617/report.pdf \
  --output-result tmp/deflection_full_report_qa_live_rerun_20260617/live-runner-result.json \
  --pretty \
  --json
```

The scripts load the operator's local environment for hosted base URL,
capability token, account id, contact details, and webhook secret. The actual
request id is extracted from the submit result and passed only through local
shell state. Raw outputs stay in `tmp/`. The committed fixture is a
hand-curated sanitized summary derived from those local outputs and then checked
with the existing proof-bundle redaction script.

The rerun produced a useful negative proof:

1. Hosted submit accepted the Zendesk full-thread fixture and created a fresh
   request, but the submit smoke still reported the parked teaser-shape drift.
2. The paid-unlock smoke exercised the live webhook path: locked artifact
   before webhook, successful webhook, idempotent replay, and unlocked artifact
   after webhook.
3. The PDF renderer produced bytes from the live paid artifact, but the live
   runner failed before scorecard because the hosted report-model endpoint
   returned 404 and the hosted artifact still omitted object `evidence_export`.

## Intentional

- No new harness code. #1612 already has the scorecard, validators, live
  runner, and PDF-byte text extraction.
- No `--pdf-text` override. The proof must use byte-derived PDF text.
- No buyer hosted-result page proof in this ATLAS slice; that page is owned by
  `atlas-portfolio/web`.
- The submit-smoke teaser false positive remains parked unless it blocks the
  rerun. The live artifact runner is the acceptance path for this slice.

## Deferred

- If the hosted artifact/report-model contract still fails, defer the smallest
  upstream production/deployment fix named by the rerun evidence.
- `atlas-portfolio/web` buyer hosted-result smoke remains separate.

Parked hardening: `Align deflection submit smoke forbidden snapshot paths with
teaser contract` remains parked unless it blocks this live proof.

## Verification

- Hosted submit smoke for a fresh Zendesk-shaped request: HTTP 200 and request
  created; smoke returned false only for the parked teaser-shape drift
  (`$.teaser.full_answer.answer`, `$.teaser.full_answer.steps`).
- Hosted paid-unlock smoke with webhook replay: before artifact 403, webhook
  200, replay 200 with `already_processed`, after artifact 200.
- Local PDF render from the live paid artifact: 7136 bytes.
- Live runner without PDF text override: failed safe with `report-model
  endpoint must return 200, got 404` and `artifact.evidence_export must be an
  object`.
- `python -m json.tool docs/extraction/validation/fixtures/deflection_full_report_qa_live_rerun_20260617/summary.json`
- `python scripts/check_deflection_full_report_proof_bundle.py docs/extraction/validation/fixtures/deflection_full_report_qa_live_rerun_20260617 --pretty`
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Full-Report-QA-Live-Rerun.md --check`

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/fixtures/deflection_full_report_qa_live_rerun_20260617/summary.json` | 83 |
| `plans/PR-Deflection-Full-Report-QA-Live-Rerun.md` | 150 |
| **Total** | **233** |
