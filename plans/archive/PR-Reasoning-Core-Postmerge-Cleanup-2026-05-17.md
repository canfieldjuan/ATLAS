# Reasoning Core Post-Merge Coordination Cleanup

## Why this slice exists

PR #573 merged the reasoning graph boundary closeout, but coordination still
shows that slice as in flight. The next session would otherwise treat the graph
boundary docs as locked and might keep looking for more graph extraction work.

## Scope (this PR)

1. Remove the merged #573 row from `inflight.md`.
2. Mark PR-C9 as merged #573 in `queue.md`.
3. Advance `extracted_reasoning_core` state to #573 with no active hot zone.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Reasoning-Core-Postmerge-Cleanup-2026-05-17.md`

## Mechanism

Documentation-only cleanup. This applies the coordination protocol's
post-merge step after #573: drop the in-flight lock and update the per-product
state.

## Intentional

- No runtime code changes.
- No AI Content Ops backlog changes. Current main already says speculative
  Content Ops reasoning-policy work should pause until a real source export or
  concrete reasoning-core provider need appears.

## Deferred

- Next code work should be selected from a concrete trigger: a real source
  export fixture, a host-requested reasoning preset/output, or a product
  runtime need from `extracted_reasoning_core`.

## Verification

- `git diff --check`
- Local PR review bundle via `scripts/local_pr_review.sh`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination docs | ~5 |
| Plan doc | ~45 |
| **Total** | ~50 |
