# PR: Clear Stale LLM Usage Hardening

## Why this slice exists

`HARDENING.md` still lists `LLM usage storage schema mismatch hides per-run cost
telemetry`, but that parked item has already been closed by
`PR-LLM-Usage-Schema-Cache-Telemetry` and then proven by the live
support-ticket observed-shell telemetry validation.

Leaving the closed item in the active queue makes the support-ticket/provider
lane look more blocked than it is and can cause future sessions to re-scope a
fixed issue instead of moving to the next real hardening gap.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Production hardening

1. Remove the stale LLM usage schema mismatch entry from `HARDENING.md`.
2. Verify the merged plan and live validation doc still contain the closure
   evidence for traceability.
3. Keep this to hardening queue cleanup; no runtime code or schema changes.

### Files touched

- `HARDENING.md` - remove the closed support-ticket/provider telemetry item.
- `plans/PR-Clear-Stale-LLM-Usage-Hardening.md` - this plan.

## Mechanism

The hardening item is removed because the repository already contains:

- `plans/PR-LLM-Usage-Schema-Cache-Telemetry.md`, which documents the schema
  fallback fix and says the parked item was closed
- `docs/extraction/validation/support_ticket_blog_observed_shell_live_telemetry_2026-05-28.md`,
  which records a live support-ticket blog generation run where persisted
  `llm_usage` totals matched `generation_usage`

This slice does not change the runtime path. It only reconciles the active
hardening queue with the already-merged implementation and validation evidence.

## Intentional

- No new telemetry test is added because this slice does not alter telemetry
  behavior. Existing tests and the live validation doc are the evidence.
- The FAQ-search owned hardening entries remain untouched.
- The broader product UI for surfacing per-run cost remains a separate product
  slice.

## Deferred

- Future PR: surface per-run Content Ops usage/cost in the product UI.
- Parked hardening: none.

## Verification

- Command: if rg -n '^### LLM usage storage schema mismatch hides per-run cost telemetry' HARDENING.md; then exit 1; fi
  - Passed; no active hardening entry remains.
- Command: rg -n 'PR-LLM-Usage-Schema-Cache-Telemetry|Persisted `llm_usage` summary|cache-token telemetry survived' plans/PR-LLM-Usage-Schema-Cache-Telemetry.md docs/extraction/validation/support_ticket_blog_observed_shell_live_telemetry_2026-05-28.md
  - Passed; merged plan and live validation evidence remain in-tree.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file <PR body file>
  - Passed; advisory overlap with #1094 on `HARDENING.md`, no blocking drift.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Hardening cleanup | ~9 |
| Plan doc | ~70 |
| **Total** | **~79** |
