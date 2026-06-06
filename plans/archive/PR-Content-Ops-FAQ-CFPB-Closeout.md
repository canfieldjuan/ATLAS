# Content Ops FAQ CFPB Closeout

## Why this slice exists

PR #684 and PR #687 closed the latest FAQ/source follow-ups: complaint-shaped
CFPB rows now produce human FAQ questions, and operators have a live CFPB to
FAQ Markdown smoke command. The coordination ledger still shows the merged
CFPB FAQ smoke as in flight, and the active backlog stops at PR #673.

This slice keeps the source-of-truth docs aligned before starting another
implementation thread.

## Scope (this PR)

1. Remove the merged CFPB FAQ smoke claim from the in-flight ledger.
2. Update the AI Content Ops deferred backlog with the PR #684 and PR #687 FAQ
   closeout.
3. Leave README and STATUS alone because PR #687 already updated those product
   docs.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-CFPB-Closeout.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Remove merged CFPB FAQ smoke row. |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | Record the latest FAQ/CFPB closeout and keep future work gated on real host data or UI need. |

## Mechanism

The in-flight table becomes empty again because the CFPB FAQ smoke PR has
merged. The backlog FAQ paragraph now records the two latest shipped slices and
keeps the existing policy: future FAQ/source work should be driven by real
customer help desk exports or hosted UI needs, not speculative source shapes.

## Intentional

- Docs-only closeout; no runtime behavior changes.
- No README or STATUS edits because they already mention the CFPB FAQ smoke.

## Deferred

- Next implementation work remains dependent on a real host export, hosted UI
  need, or a deliberate shift back to extracted reasoning core productization.

## Verification

- `bash scripts/local_pr_review.sh --allow-dirty` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-CFPB-Closeout.md` | 55 |
| `docs/extraction/coordination/inflight.md` | 3 |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | 12 |
| **Total** | **70** |
