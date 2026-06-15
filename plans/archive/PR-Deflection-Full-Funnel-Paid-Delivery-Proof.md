# PR-Deflection-Full-Funnel-Paid-Delivery-Proof

## Why this slice exists

#1440's title is the deliverable: test deflection delivery under real full-volume
conditions through intake, snapshot, payment, and report email. #1557 proved
hosted full-volume submit/snapshot/locked artifact, and #1569 proved offline
Zendesk product-shaped quality, but neither proved the live paid revenue path.
This docs/proof slice is slightly over the 400-line soft cap because it carries
both the human proof narrative and the sanitized machine-readable evidence
needed to review the live run without raw secrets or report payloads.

Root cause: the prior #1440 evidence stopped before the deployed Stripe webhook
secret and hosted delivery task were exercised together. This slice fixes the
evidence gap, not product code: it runs the real hosted path, records the paid
unlock and report-delivery proof, and keeps the still-failing portfolio render
as an explicit remaining blocker instead of overclaiming full closure.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation

1. Add a sanitized live proof doc for the fresh full-volume hosted funnel run.
2. Commit a redacted machine-readable summary of the run: full-volume submit,
   paid unlock, idempotent webhook replay, delivery task execution, and
   portfolio page status.
3. Do not commit raw CSV rows, paid report Markdown/PDF, email bodies, bearer
   tokens, webhook secrets, or portfolio HTML.
4. Do not change product code in this slice; the live proof identifies the next
   code-owned blocker.

### Review Contract

Acceptance criteria:

- The proof demonstrates a fresh hosted full-volume request, not only the old
  #1557 request.
- Paid unlock is proven by before-artifact 403, webhook 200, replay 200 with
  `already_processed`, and after-artifact 200.
- Report delivery is proven by the hosted delivery task scanning 1, sending 1,
  failing 0, and dry-run 0.
- The proof does not overclaim #1440 completion: portfolio production still
  renders `SNAPSHOT TEMPORARILY UNAVAILABLE`, and snapshot email/PDF remains
  unobserved.
- Committed artifacts are sanitized and exclude secrets/raw customer data.

Affected surfaces:

- `docs/extraction/validation/**`
- `docs/extraction/validation/fixtures/**`
- `plans/**`

Risk areas:

- Overclaiming a partial proof as full acceptance.
- Accidentally committing raw CSV/report/email/HTML or secret-bearing outputs.
- Confusing CFPB stress evidence with Zendesk product-quality calibration.

Reviewer rules triggered: R1, R2, R6, R14.

### Files touched

- `docs/extraction/validation/deflection_full_funnel_paid_delivery_proof_2026-06-15.md`
- `docs/extraction/validation/fixtures/deflection_full_funnel_paid_delivery_proof_20260615/summary.json`
- `plans/PR-Deflection-Full-Funnel-Paid-Delivery-Proof.md`

## Mechanism

The slice records a live run using existing hosted-proof scripts and the
deployed autonomous delivery task:

1. Regenerate a near-50 MiB CFPB stress CSV locally from the existing 50k JSONL
   source and keep it under ignored `tmp/`.
2. Run `smoke_content_ops_deflection_submit_handoff.py` against hosted ATLAS
   with `--volume-gate-profile full-volume-cfpb`.
3. Run `smoke_content_ops_deflection_stripe_paid_unlock.py` against the fresh
   request with the deployed-matching webhook secret source. The stale root env
   secret still fails with `Invalid signature`; the portfolio/backup secret
   matches deployed and passes.
4. Trigger hosted `content_ops_deflection_report_delivery` via
   `/api/v1/autonomous/content_ops_deflection_report_delivery/run` and poll the
   task execution.
5. Run the portfolio result-page smoke against the canonical paid result URL
   and record the remaining unavailable-state blocker.

Only sanitized scalar evidence is committed.

## Intentional

- This PR does not fix the portfolio result-page blocker. The proof shows the
  blocker precisely: portfolio returns 200 but renders the unavailable state
  while ATLAS snapshot/artifact endpoints return 200 for the same paid request.
- This PR keeps CFPB framed as stress/scale evidence, not Zendesk
  product-quality calibration.
- The raw generated CSV, paid artifact, email body/PDF, bearer token, webhook
  secrets, and fetched portfolio HTML stay local and uncommitted.
- Live paid request ids, result URLs, local absolute paths, and operator email
  addresses are redacted from committed proof artifacts and the PR body. The
  live request id originally used for the proof was invalidated server-side
  after review identified it as a capability.
- The proof is partial rather than greenwashing #1440; the paid revenue path is
  proven, but portfolio rendering and snapshot email/PDF are still open.

## Deferred

1. Portfolio production result rendering: fix the canonical result page so the
   paid request renders the real snapshot/report markers instead of
   `SNAPSHOT TEMPORARILY UNAVAILABLE`.
2. Snapshot email/PDF decision: either identify/build the snapshot sender and
   prove it under full-volume conditions, or update #1440 acceptance because no
   snapshot-email sender surface was found in this repo during this slice.
3. Local env cleanup: align operator env files so future paid smokes use the
   deployed Stripe webhook secret source and do not hit the stale root `.env`
   value first.

Parked hardening: none.

## Verification

- `python scripts/smoke_content_ops_deflection_submit_handoff.py --csv-file tmp/deflection_full_funnel_paid_delivery_proof_20260615/cfpb_real_upload_under_50mib.csv --company-name "CFPB Public Archive" --contact-email "<contact-email>" --support-platform other --volume-gate-profile full-volume-cfpb --timeout 360 --output-result tmp/deflection_full_funnel_paid_delivery_proof_20260615/submit-result.json --json` -> passed; 42,646 submitted rows, 52,426,845 uploaded bytes, 1,757 generated questions, 29,106 repeat tickets.
- `python scripts/smoke_content_ops_deflection_stripe_paid_unlock.py --request-id "<redacted-paid-request-id>" --timeout 180 --replay-webhook --output-result tmp/deflection_full_funnel_paid_delivery_proof_20260615/paid-unlock-fresh-request.json --json` -> passed; 403 before webhook, 200 webhook, 200 replay with `already_processed`, 200 artifact after webhook.
- `POST /api/v1/autonomous/content_ops_deflection_report_delivery/run` and execution polling -> completed; scanned 1, sent 1, failed 0, dry_run 0.
- `python scripts/smoke_content_ops_deflection_portfolio_result_page.py --result-url "<redacted-paid-result-url>" --request-id "<redacted-paid-request-id>" --timeout 120 --output-result tmp/deflection_full_funnel_paid_delivery_proof_20260615/portfolio-result-page-fresh.json --json` -> failed as expected for the remaining blocker; page 200, ATLAS snapshot 200, ATLAS artifact 200, but portfolio rendered `SNAPSHOT TEMPORARILY UNAVAILABLE` and missed result markers.
- Signed Stripe revocation webhook for the live proof request -> 200; artifact endpoint recheck after revocation -> 403 locked at 2026-06-15T13:55:05Z. The committed summary binds the proof to the leaked value by SHA-256 only: `1ebb65093faf3823d2b1413d2b34db28f786908885406acc48846f224ac1011d`.
- `python -m json.tool docs/extraction/validation/fixtures/deflection_full_funnel_paid_delivery_proof_20260615/summary.json >/tmp/deflection-paid-summary.pretty.json` -> passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_full_funnel_paid_delivery_proof_2026-06-15.md` | 218 |
| `docs/extraction/validation/fixtures/deflection_full_funnel_paid_delivery_proof_20260615/summary.json` | 103 |
| `plans/PR-Deflection-Full-Funnel-Paid-Delivery-Proof.md` | 135 |
| **Total** | **456** |
