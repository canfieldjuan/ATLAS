# PR-Content-Ops-FAQ-Continuous-Direction

## Why this slice exists

The support-ticket FAQ product is becoming credible as a one-time report and
searchable demo, but the subscription packaging question needs to be logged
before future slices accidentally frame monthly pricing as "rerun the same FAQ
generator." A monthly FAQ product is only defensible if it becomes ongoing FAQ
operations: monitor new inputs, detect deltas, and maintain a review queue.

This slice records that direction without starting implementation.

## Scope (this PR)

Ownership lane: content-ops/faq-product-direction

Slice phase: Product polish

1. Add a deferred product-direction note to the AI Content Ops backlog.
2. Clarify that the subscription shape is ongoing detection and maintenance,
   not a scheduled rerun of the same report.
3. State the trigger for implementation: after the one-time FAQ report/search
   flow and hosted proof are stable enough to support recurring comparison.

### Files touched

| File | Purpose |
|---|---|
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | Records the continuous FAQ ops product direction as deferred. |
| `plans/PR-Content-Ops-FAQ-Continuous-Direction.md` | Slice contract. |

## Mechanism

The backlog note describes the subscription model as a delta workflow:

```text
new ticket/search-log inputs -> compare against existing FAQ inventory ->
new FAQ / update FAQ / retire stale FAQ recommendations
```

It also names what not to do yet: do not build this before the current FAQ
report/search path is stable and the hosted proof is complete.

## Intentional

- No code, schema, API, or UI changes. This is a direction log only.
- No new active implementation backlog is created.
- The pricing language stays out of code/docs that operators execute; the
  backlog captures product direction and timing.

## Deferred

- Continuous FAQ operations implementation remains deferred until the one-time
  FAQ report/search flow and hosted SaaS FAQ proof are stable.
- Parked hardening: none. This slice does not surface implementation defects.

## Verification

- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Continuous-Direction.md`
  - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Continuous-Direction.md`
  - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-continuous-direction-pr-body.md`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 72 |
| Backlog note | 23 / -1 |
| **Total** | **96** |
