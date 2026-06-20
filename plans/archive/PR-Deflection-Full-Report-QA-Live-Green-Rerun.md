# PR-Deflection-Full-Report-QA-Live-Green-Rerun

## Why this slice exists

#1612 has been driving the full-report delivery proof arc. #1671 and #1678
both failed safe on the live paid artifact contract: the hosted
`/report-model` endpoint returned 404 and `/artifact` omitted object
`report_model` / `evidence_export` for fresh paid requests, even after #1674
pinned that source contract.

Root cause: the public Tailscale Funnel `/api` route was still serving a
long-running Atlas API process started before the report-model/export contract
landed. The source producer path was already pinned by #1674; the hosted proof
was hitting a stale process. After restarting the API process from the current
worktree, a fresh paid Zendesk-shaped request exposed the modern contract and
the live QA runner passed.

This PR records that green live proof as a sanitized artifact. It fixes the
proof gap for the ATLAS-side paid artifact/report-model contract; it does not
pretend to add a durable deploy-freshness guard or prove the separate
atlas-portfolio buyer hosted-result page.

This PR is over the 400 LOC soft cap because the committed deliverable is a
machine-readable scorecard with the full assertion matrix. Splitting the
summary away from the plan would make the proof harder to review and would not
reduce product risk; the artifact itself is the evidence.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Functional validation

1. Commit only the sanitized green live-runner summary for the fresh paid
   after-restart run.
2. Record that the fresh hosted paid request returned HTTP 200 from both
   `/report-model` and `/artifact`.
3. Record that the artifact contained object `report_model` with
   `deflection.v1` and object `evidence_export` with
   `deflection_evidence.v1`.
4. Record that the live runner validated PDF bytes with text extracted from
   those bytes and produced an all-green scorecard.
5. Do not commit the raw request id, result URL, token, webhook payload, raw
   artifact JSON, report model JSON, PDF bytes, source ids, or evidence rows.

### Review Contract

Acceptance criteria:
- The committed proof artifact is sanitized and passes the existing proof-bundle
  checker.
- The proof artifact shows `ok: true`, report-model fetch 200, artifact fetch
  200, and PDF text verified from PDF bytes.
- The PR does not change production code or harness code; this slice is proof
  of the live path after the hosted API process restart.
- The plan clearly defers durable deploy-freshness detection and
  atlas-portfolio buyer hosted-result proof.

Affected surfaces:
- Full-report QA validation fixture only.
- Plan doc only.

Risk areas:
- Accidentally committing live request ids, URLs, source ids, raw evidence, or
  PDF bytes.
- Overclaiming that #1612 is fully complete when buyer hosted-result proof is
  still outside this ATLAS repo slice.
- Mistaking this operational fix proof for a durable deployment freshness
  guard.

- Reviewer rules triggered: R1, R2, R3, R14.

### Files touched

- `docs/extraction/validation/fixtures/deflection_full_report_qa_live_green_rerun_20260617/summary.json`
- `plans/PR-Deflection-Full-Report-QA-Live-Green-Rerun.md`

## Mechanism

The live run used the existing proof chain:

1. Submit a Zendesk-shaped CSV through the hosted API.
2. Mark the fresh request paid through the Stripe webhook smoke with the
   correct hosted webhook secret.
3. Fetch the fresh paid `/report-model` and `/artifact` endpoints through the
   public Funnel URL.
4. Render local PDF bytes from the paid artifact using the existing deflection
   delivery renderer path.
5. Run `scripts/run_deflection_full_report_qa_live_runner.py` with those PDF
   bytes and no text override, so the runner extracts PDF text from the bytes.
6. Commit only the runner's sanitized `summary.json`.

The raw local working files stay under `tmp/deflection_full_report_qa_live_after_restart_20260617/`
and are not part of this diff.

## Intentional

- No production code change: the fresh live proof passes after the hosted API
  process restart, so changing the artifact producer would be a symptom fix.
- No raw live bundle commit: the proof-bundle policy from #1613 allows only
  sanitized scorecards in git.
- No #1612 closure claim: the ATLAS-side artifact/report-model proof is green,
  but buyer hosted-result proof belongs to the `atlas-portfolio/web` surface.

## Deferred

- Durable deploy/process freshness detection so a stale public API process is
  visible before a live proof burns a run.
- Separate `atlas-portfolio/web` buyer hosted-result proof for the URL emailed
  to buyers.
- Align the submit smoke's forbidden snapshot paths with the checked-in teaser
  contract before using that smoke as paywall validation evidence.

Parked hardening: none.

## Verification

- `python scripts/smoke_content_ops_deflection_submit_handoff.py ... --csv-file extracted_content_pipeline/examples/support_ticket_provider_exports/zendesk_full_thread_export.csv --output-result tmp/deflection_full_report_qa_live_after_restart_20260617/submit-result.json --json` - created a fresh request; exited nonzero only on the known parked teaser-field smoke drift.
- `python scripts/smoke_content_ops_deflection_stripe_paid_unlock.py ... --replay-webhook --output-result tmp/deflection_full_report_qa_live_after_restart_20260617/paid-unlock-result.json --json` - passed after using the hosted portfolio webhook secret and public Funnel API base.
- Direct endpoint shape probe - report-model 200 with `deflection.v1`; artifact 200 with object `report_model` and object `evidence_export`.
- Local PDF render from the paid artifact - produced a valid PDF header and
  5436 bytes.
- `python scripts/run_deflection_full_report_qa_live_runner.py ... --pdf-bytes tmp/deflection_full_report_qa_live_after_restart_20260617/report.pdf --output-result tmp/deflection_full_report_qa_live_after_restart_20260617/live-runner-result.json --pretty --json` - passed.
- `python scripts/check_deflection_full_report_proof_bundle.py docs/extraction/validation/fixtures/deflection_full_report_qa_live_green_rerun_20260617 --pretty` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/fixtures/deflection_full_report_qa_live_green_rerun_20260617/summary.json` | 620 |
| `plans/PR-Deflection-Full-Report-QA-Live-Green-Rerun.md` | 130 |
| **Total** | **750** |
