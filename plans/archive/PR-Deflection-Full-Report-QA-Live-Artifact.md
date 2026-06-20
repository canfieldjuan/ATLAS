# PR-Deflection-Full-Report-QA-Live-Artifact

## Why this slice exists

Issue #1612 is the full-report delivery QA testing arc. The deterministic
scorecard, PDF/export validators, live runner, and PDF-byte text extraction are
now merged. The remaining proof gap is the real customer-shaped execution: a
controlled paid Zendesk-shaped request must run through the live runner and
leave a committed, sanitized proof summary. Without this slice, the lane still
proves the harness and validators but not what the deployed paid artifact
currently does.

This is a validation slice, not a fix PR. If the live run fails, this PR commits
only a sanitized failure summary and defers the code fix to the smallest
upstream slice identified by that evidence.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Functional validation

1. Run the merged live QA runner against one controlled paid Zendesk-shaped
   request using PDF bytes rendered from the live paid artifact by the same
   delivery renderer used for email attachments, with no operator-supplied PDF
   text.
2. Commit only the sanitized proof summary under
   `docs/extraction/validation/fixtures/deflection_full_report_qa_live_artifact_20260617/`.
3. Record whether the runner proves:
   - hosted report-model endpoint status;
   - hosted artifact endpoint status;
   - whether artifact evidence export was present and model-consistent;
   - whether PDF text was extracted from PDF bytes;
   - whether the PDF/export scorecard passed;
   - sanitizer kept request IDs, result URLs, tokens, local paths, customer
     email addresses, source IDs, raw evidence, and private notes out of the
     committed file.
4. Reclassify the submit-smoke `teaser.full_answer` answer/steps finding as
   smoke-checker drift because the checked-in snapshot contract permits those
   fields inside `teaser.full_answer`.
5. Do not add new harness code unless the live run exposes a blocker in the
   proof path itself.

### Review Contract

Acceptance criteria:
- [ ] The committed artifact is sanitized and contains no live request ID,
      bearer token, result URL, customer email, absolute local path, source ID
      list, private note, raw evidence quote, Stripe checkout session ID, or
      Stripe payment intent ID.
- [ ] The artifact records PDF-byte provenance. If the live runner reaches a
      passing scorecard, it records `pdf_text.source` as
      `extracted_from_pdf_bytes` and `verified_from_pdf_bytes` as true; if the
      hosted contract fails first, it records the upstream failure instead.
- [ ] The artifact records endpoint status summaries and scorecard status
      without embedding raw endpoint payloads.
- [ ] If the live runner fails, the committed artifact is still sanitized and
      the plan names the root upstream failure as deferred follow-up.
- [ ] No raw live PDF, report model JSON, email body, HTML, or evidence export
      is committed.

Affected surfaces: #1612 proof artifacts and the already-merged live QA runner
contract.

Risk areas: committing live capabilities or PII, treating a failed live run as
product proof, drifting into another harness slice instead of executing the live
proof, and stale-base plan/index churn.

Reviewer rules triggered: R1, R2, R3, R6, R14.

### Files touched

- `HARDENING.md`
- `docs/extraction/validation/fixtures/deflection_full_report_qa_live_artifact_20260617/summary.json`
- `plans/PR-Deflection-Full-Report-QA-Live-Artifact.md`

## Mechanism

The operator-run proof uses the existing submit, paid-unlock, PDF-render, and
live-runner path:

```bash
python scripts/run_deflection_full_report_qa_live_runner.py \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "$ATLAS_B2B_JWT" \
  --request-id "$CONTROLLED_PAID_REQUEST_ID" \
  --pdf-bytes tmp/deflection_full_report_qa_live_artifact_20260617/report.pdf \
  --output-result tmp/deflection_full_report_qa_live_artifact_20260617/summary.json \
  --pretty
```

The `--pdf-text` argument stays omitted so the runner derives text from the
PDF bytes. Because the actual email attachment is not recoverable from the
operator inbox in this slice, the PDF bytes are rendered locally from the live
paid artifact using `atlas_brain.deflection_pdf_renderer.render_deflection_full_report_pdf`,
which is the same renderer called by the delivery attachment path. The runner
writes a sanitized result envelope, then the slice copies only a sanitized
summary into the committed fixture directory after running sensitive-pattern
checks.

Raw live inputs remain local:

- the paid request ID;
- the bearer token;
- the locally rendered PDF bytes;
- any downloaded report model, artifact JSON, HTML, email, or evidence export.

## Intentional

- No new generic harness code. #1612 already has the scorecard, deterministic
  harness, PDF/export validators, live runner, and PDF-byte extraction.
- No `--pdf-text` override in the proof command. The point of this slice is to
  prove byte-derived PDF text, not a sidecar text file.
- The PDF bytes are renderer-equivalent rather than inbox-captured. The delivery
  worker attaches bytes from the same renderer; actual inbox attachment capture
  remains an operator-email concern.
- No raw live artifact bundle is committed. #1613 set the safety boundary:
  sanitized summary only.
- No buyer hosted-result page proof in this ATLAS PR. That surface belongs to
  `atlas-portfolio/web`; this slice proves the ATLAS report-model, artifact,
  PDF, and export side.
- The submit smoke reported `teaser.full_answer.answer` and
  `teaser.full_answer.steps` as forbidden fields, but
  `docs/frontend/content_ops_faq_report_contract.md` explicitly allows those
  fields inside `teaser.full_answer`. This PR records that as smoke-checker
  drift, not as a paywall leak or product blocker.

## Deferred

- `atlas-portfolio/web` buyer hosted-result smoke remains separate because the
  buyer URL resolves outside this repo.
- The controlled live run failed upstream: the hosted report-model endpoint
  returned 404 for the fresh paid request, the hosted artifact endpoint returned
  200 with a legacy artifact shape that lacked `evidence_export`. The next code
  slice should fix the hosted generated-asset/report artifact path so fresh paid
  deflection artifacts expose `deflection.v1` `report_model` and
  `evidence_export`, then rerun this proof.

Parked hardening: `Align deflection submit smoke forbidden snapshot paths with
teaser contract` because the stale smoke-checker rule is not required to record
this live artifact result, but it must be fixed before treating submit-smoke
forbidden-path output as paywall validation evidence.

## Verification

- Live run:
  - `python scripts/smoke_content_ops_deflection_submit_handoff.py ... --csv-file extracted_content_pipeline/examples/support_ticket_provider_exports/zendesk_full_thread_export.csv --output-result tmp/deflection_full_report_qa_live_artifact_20260617/submit-result.json --json` - submit status 200, request created, smoke reported `teaser.full_answer` answer/steps as forbidden; contract check reclassified this as smoke drift.
  - `python scripts/smoke_content_ops_deflection_stripe_paid_unlock.py ... --output-result tmp/deflection_full_report_qa_live_artifact_20260617/paid-unlock.json --replay-webhook --json` - passed; before artifact 403, webhook 200, replay 200, after artifact 200.
  - Local PDF render from live paid artifact using `render_deflection_full_report_pdf` - passed; 7139 PDF bytes, PDF text extraction 3261 chars.
  - `python scripts/run_deflection_full_report_qa_live_runner.py ... --pdf-bytes tmp/deflection_full_report_qa_live_artifact_20260617/report.pdf --output-result tmp/deflection_full_report_qa_live_artifact_20260617/live-runner-result.json --pretty --json` - failed safe; report-model 404 and artifact missing `evidence_export`.
- Artifact validation:
  - `python -m json.tool docs/extraction/validation/fixtures/deflection_full_report_qa_live_artifact_20260617/summary.json` - passed.
  - `python scripts/check_deflection_full_report_proof_bundle.py docs/extraction/validation/fixtures/deflection_full_report_qa_live_artifact_20260617 --pretty` - passed.
- Local review:
  - `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-deflection-full-report-qa-live-artifact.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 9 |
| `docs/extraction/validation/fixtures/deflection_full_report_qa_live_artifact_20260617/summary.json` | 74 |
| `plans/PR-Deflection-Full-Report-QA-Live-Artifact.md` | 163 |
| **Total** | **246** |
