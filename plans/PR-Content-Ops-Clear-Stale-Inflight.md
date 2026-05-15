# PR: Clear Stale Content Ops Inflight Row

## Why This Slice Exists

The coordination ledger still shows the generated asset preview UX claim after
that PR merged. Keeping stale locks slows the next sessions down.

## Scope

Clear the stale row in `docs/extraction/coordination/inflight.md`.

### Files Touched

- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Clear-Stale-Inflight.md`

## Mechanism

- Replace the stale draft row with the ledger's explicit empty-state row.

## Intentional

- No product code changes.
- No frontend changes.
- No generated asset behavior changes.

## Deferred

- Next Content Ops product slice.

## Verification

- `scripts/local_pr_review.sh`.
- `git diff --check`.

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Coordination row | ~2 |
| Plan doc | ~45 |
| **Total** | ~47 |
