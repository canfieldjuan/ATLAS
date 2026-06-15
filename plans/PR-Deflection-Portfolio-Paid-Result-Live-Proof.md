# PR-Deflection-Portfolio-Paid-Result-Live-Proof

## Why this slice exists

#1440 is still open because the last full-funnel proof (#1572) reached a
partial pass: hosted full-volume submit, paid unlock, idempotent webhook replay,
and paid report delivery were proven, but the production portfolio result page
rendered `SNAPSHOT TEMPORARILY UNAVAILABLE` even while ATLAS snapshot/artifact
endpoints returned 200 for the same paid request.

The portfolio code fix for that blocker landed as atlas-portfolio #309. This
slice reruns the live proof from the Atlas side after #309, commits only
sanitized evidence, and updates the #1440 proof narrative without changing
product code unless the rerun exposes a new code-owned blocker.

This slice is over the 400-line soft cap because the existing portfolio smoke
only validated the locked/free snapshot state. The paid-result proof needs the
small `--artifact-state unlocked` harness mode plus focused failure tests and
the sanitized live proof artifact in the same reviewable PR; splitting them
would make the proof depend on an unproven checker.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation

1. Rerun the hosted full-volume submit and paid-unlock path only as needed to
   create a fresh non-revoked paid request.
2. Rerun the production portfolio result-page smoke against that paid request
   after atlas-portfolio #309 is on production.
3. Commit a sanitized proof doc/summary that records the production page result
   without raw CSV rows, result URLs, request ids, report Markdown, email
   bodies, bearer tokens, webhook secrets, or portfolio HTML.
4. Archive this session's previously merged #1572 plan doc while this branch is
   already touching validation proof docs.

### Review Contract

Acceptance criteria:

- The proof uses a fresh paid request or explicitly explains why an existing
  non-revoked request is safe to reuse.
- The portfolio page smoke records page status, ATLAS snapshot status, ATLAS
  artifact status, marker presence, and unavailable-state status.
- Committed evidence is sanitized: no bearer tokens, webhook secrets, raw report
  bodies, raw portfolio HTML, operator email, request id, or result URL.
- The doc states whether #1440's production result-page blocker is closed or
  still open; it does not overclaim the snapshot-email/PDF surface.
- The merged #1572 plan is moved to `plans/archive/` and `plans/INDEX.md` is
  refreshed.

Affected surfaces:

- `docs/extraction/validation/**`
- `docs/extraction/validation/fixtures/**`
- `plans/**`

Risk areas:

- Committing capability-bearing live request identifiers.
- Mistaking a locked snapshot page for an unlocked paid report.
- Overclaiming full #1440 closure while snapshot-email/PDF remains undefined.
- Re-running live Stripe webhook proof with the wrong local secret source.

Reviewer rules triggered: R1, R2, R6, R14.

### Files touched

- `docs/extraction/validation/deflection_full_funnel_paid_delivery_proof_2026-06-15.md`
- `docs/extraction/validation/deflection_portfolio_paid_result_live_proof_2026-06-15.md`
- `docs/extraction/validation/fixtures/deflection_portfolio_paid_result_live_proof_20260615/summary.json`
- `plans/INDEX.md`
- `plans/PR-Deflection-Portfolio-Paid-Result-Live-Proof.md`
- `plans/archive/PR-Deflection-Full-Funnel-Paid-Delivery-Proof.md`
- `scripts/smoke_content_ops_deflection_portfolio_result_page.py`
- `tests/test_smoke_content_ops_deflection_portfolio_result_page.py`

## Mechanism

The slice reuses the existing hosted proof scripts:

1. Use the existing CFPB 50k JSONL stress corpus to regenerate an ignored
   near-50 MiB CSV if a fresh request is needed.
2. Run `smoke_content_ops_deflection_submit_handoff.py` with
   `--volume-gate-profile full-volume-cfpb` against hosted ATLAS.
3. Run `smoke_content_ops_deflection_stripe_paid_unlock.py` with the
   deployed-matching webhook secret source, preserving signature verification
   and webhook replay idempotency.
4. Run `smoke_content_ops_deflection_portfolio_result_page.py` against
   `https://juancanfield.com/systems/support-ticket-deflection/results/...`
   for the paid request after atlas-portfolio #309 is deployed. The script now
   supports `--artifact-state unlocked`, which validates paid report markers,
   artifact 200, no unavailable state, no locked snapshot state, and no unlock
   CTA.
5. Transform local raw outputs into committed scalar evidence: status codes,
   booleans, hashes, counts, and redaction booleans only.

## Intentional

- This is a proof/evidence slice, not a product-code slice. If the production
  rerun fails, the committed artifact will name the new blocker instead of
  hiding it.
- CFPB remains stress/scale evidence only; this slice does not use it as
  Zendesk product-quality calibration.
- Raw CSV, paid report payload, email body/PDF, portfolio HTML, webhook secret,
  bearer token, result URL, and live request id stay local and uncommitted.
- Snapshot email/PDF remains a product-surface decision unless this proof finds
  an existing sender path.

## Deferred

- If the portfolio page still fails after #309, the next slice is the new
  production result-page blocker surfaced by the smoke.
- If the portfolio page passes, #1440 still needs an operator decision on
  whether snapshot email/PDF is a real product surface or should be removed from
  acceptance language.

Parked hardening: none.

## Verification

- Regenerated ignored CSV locally from `tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl`:
  41,608 records, 52,426,203 bytes, SHA-256 recorded in the summary.
- Hosted submit smoke with `--volume-gate-profile full-volume-cfpb` and explicit
  current service token: passed; submit 200, snapshot 200, unpaid artifact 403,
  41,608 submitted rows, 26,989 repeat tickets.
- Hosted paid-unlock smoke with deployed-matching webhook secret and replay:
  passed; before artifact 403, webhook 200, replay 200 / `already_processed`,
  after artifact 200.
- Hosted portfolio result-page smoke with `--artifact-state unlocked`: passed;
  page 200, snapshot 200, artifact 200, paid markers present, unavailable state
  absent, unlock CTA absent.
- `tests/test_smoke_content_ops_deflection_portfolio_result_page.py` - focused
  pytest run passed, 14 passed.
- `scripts/smoke_content_ops_deflection_portfolio_result_page.py` - Python
  compile check passed.
- Redaction sanity check over committed proof doc/summary: passed; no live
  request id, result URL, account id, event id, or session id in committed
  artifacts.
- Pending: `bash scripts/local_pr_review.sh --current-pr-body-file <body>`.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_full_funnel_paid_delivery_proof_2026-06-15.md` | 6 |
| `docs/extraction/validation/deflection_portfolio_paid_result_live_proof_2026-06-15.md` | 160 |
| `docs/extraction/validation/fixtures/deflection_portfolio_paid_result_live_proof_20260615/summary.json` | 89 |
| `plans/INDEX.md` | 1 |
| `plans/PR-Deflection-Portfolio-Paid-Result-Live-Proof.md` | 154 |
| `plans/archive/PR-Deflection-Full-Funnel-Paid-Delivery-Proof.md` | 0 |
| `scripts/smoke_content_ops_deflection_portfolio_result_page.py` | 84 |
| `tests/test_smoke_content_ops_deflection_portfolio_result_page.py` | 93 |
| **Total** | **587** |
