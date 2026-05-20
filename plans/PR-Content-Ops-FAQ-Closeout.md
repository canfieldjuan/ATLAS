Plan: PR-Content-Ops-FAQ-Closeout

## Why this slice exists

PR #667 shipped the FAQ output-contract proof, but the coordination ledger
still claims the merged slice as in-flight and the AI Content Ops deferred
backlog does not record the closeout. That makes the next-session state look
staler than the code.

## Scope (this PR)

1. Remove the merged PR-Content-Ops-FAQ-Output-Contract row from the in-flight
   coordination ledger.
2. Update the AI Content Ops deferred backlog with the FAQ output-contract
   closeout.

### Files touched

- `plans/PR-Content-Ops-FAQ-Closeout.md`
- `docs/extraction/coordination/inflight.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`

## Mechanism

This is a documentation/coordination-only closeout. The implementation is
already merged in PR #667, so this PR only updates the two tracking docs that
future sessions use to decide whether more AI Content Ops work is active.

## Intentional

- No runtime files are touched; `extracted_content_pipeline/STATUS.md` already
  records the FAQ CLI output-check behavior.
- The active source-breadth item remains blocked on a real host export fixture.

## Deferred

- Real customer help desk exports remain the trigger for future source alias or
  generated-asset quality slices.
- Further reasoning work belongs to the `extracted_reasoning_core`
  productization track unless a concrete AI Content Ops trigger appears.

## Verification

- git diff --check
- bash scripts/local_pr_review.sh

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Closeout.md` | +45 |
| `docs/extraction/coordination/inflight.md` | +1 / -2 |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | +9 |
| **Total** | **57** |

This is below the 400 LOC review budget.
