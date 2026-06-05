# PR-FAQ-Scale-Hardening-Ledger-Closeout

## Why this slice exists

`HARDENING.md` still lists `FAQSCALE-1 - Large synchronous FAQ generation needs
hosted limits / backpressure / background execution`, but the repo already has
the synchronous hosted protections that item asks for: inline execute source
material is capped, file ingestion has separate byte/row caps, `/execute` has a
router-local concurrency gate, and the closeout note under
`docs/extraction/validation/content_ops_faq_stress_hardening_closeout_2026-05-23.md`
documents those controls.

Leaving the item parked makes the FAQ lane look less production-hardened than
the implementation now is. This slice reconciles the ledger with the current
code and keeps any future large-upload/background-job work framed as a new
product/runtime slice, not as unbounded synchronous request handling.

Slice size: **small**. This changes the debt ledger and closeout note only, and
verifies the existing route-limit/concurrency regression coverage.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-validation

Slice phase: Production hardening.

1. Remove the stale FAQSCALE-1 parked item from `HARDENING.md`.
2. Update the FAQ stress closeout note to say the root hardening ledger has been
   reconciled.
3. Keep the existing scale/backpressure tests as the proof; no route or
   generator behavior changes.

### Files touched

- `plans/PR-FAQ-Scale-Hardening-Ledger-Closeout.md`
- `HARDENING.md`
- `docs/extraction/validation/content_ops_faq_stress_hardening_closeout_2026-05-23.md`

## Mechanism

The implementation removes the stale root ledger item and updates the closeout
note with the explicit ledger decision:

- synchronous hosted FAQ execute is bounded by source-material row caps;
- synchronous hosted execute is bounded by router-local concurrency admission;
- larger FAQ workloads remain supported by offline/validation tooling today and
  require a future background-job boundary before becoming hosted upload
  product behavior.

## Intentional

- No code changes. The protections are already implemented and covered; this
  slice only reconciles stale documentation debt.
- No claim that Atlas has a background job system for large hosted FAQ uploads.
  The closeout is specifically for preventing unbounded synchronous hosted
  requests.
- No removal of validation docs for 50k deterministic generation; those docs
  remain useful scale evidence.

## Deferred

- A background-job upload path for >1,000 hosted FAQ source rows remains a
  future product/runtime slice if we choose to offer large hosted uploads.
- Cross-process/global admission control remains deferred until the deployment
  topology requires coordinating multiple workers against one DB pool.

## Verification

- Focused route-limit pytest for source-material row caps, accepted 1,000-row
  bundle, execute concurrency rejection, and invalid concurrency config - 4
  passed.
- Plan/code consistency audit for this plan - passed.
- Git whitespace check - passed.
- Local PR review bundle - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 83 |
| Hardening ledger | 11 |
| Closeout note | 8 |
| **Total** | **102** |
