# Content Ops Productization Audit Refresh

## Why this slice exists

Refresh the AI Content Ops productization audit so it matches current `main`.
The old audit still describes `_b2b_pool_compression.py` and
`competitive_intelligence.py` as standalone import failures, but the current
manifest-driven smoke imports both successfully.

## Scope

1. Update `extracted_content_pipeline/docs/remaining_productization_audit.md`
   to record the current import state.
2. Keep this as documentation and coordination only. No runtime code changes.
3. Do not reopen reasoning-producer extraction work; the active policy remains
   host-owned reasoning through `CampaignReasoningContextProvider`.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/docs/remaining_productization_audit.md`
- `plans/PR-Content-Ops-Productization-Audit-Refresh.md`

## Mechanism

Replace the stale import-failure table with the current import state and keep
the ownership guidance narrow: importability is closed, but copied Atlas task
files still should not become product-owned runtime paths by default.

## Intentional

- Documentation only.
- No runtime code changes.
- No test fixture changes.

## Deferred

- Real source export fixture work remains deferred until a host supplies a
  concrete export that exposes an adapter/runtime gap.
- Reasoning producer extraction remains deferred until a separate
  `extracted_reasoning_core` product/runtime decision requires bundling a
  producer instead of consuming `CampaignReasoningContextProvider`.

## Verification

- Command: EXTRACTED_PIPELINE_STANDALONE=1 python -c "import extracted_content_pipeline.autonomous.tasks._b2b_pool_compression"; result: passed.
- Command: EXTRACTED_PIPELINE_STANDALONE=1 python -c "import extracted_content_pipeline.autonomous.tasks.competitive_intelligence"; result: passed.
- Command: EXTRACTED_PIPELINE_STANDALONE=1 python scripts/smoke_extracted_pipeline_imports.py; result: 91 imported OK, 0 decoupling failures, 0 env failures.
- Command: bash scripts/local_pr_review.sh; result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Productization audit | ~45 |
| Coordination claim | ~5 |
| Plan doc | ~45 |
| **Total** | ~95 |
