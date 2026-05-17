# Post-575 Extraction State Closeout

## Why this slice exists

PR #575 merged the Content Ops productization audit refresh. The coordination
table still lists #575 as in flight, and the reasoning-core current-state audit
still recommends the per-review enrichment pack split even though that split
already landed in code and queue as #564.

## Scope (this PR)

1. Remove merged #575 from the in-flight coordination table.
2. Update Content Ops state to show #575 as the latest merged productization
   audit refresh.
3. Update the reasoning-core current-state audit so the per-review enrichment
   pack split is marked closed instead of recommended as next work.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `docs/extraction/reasoning_core_current_state_audit_2026-05-17.md`
- `plans/PR-Post-575-Extraction-State-Closeout.md`

## Mechanism

This is documentation and coordination cleanup only. It records the verified
code state: `atlas_brain.reasoning.review_enrichment.ReviewEnrichmentMixin`
owns per-review enrichment methods, and
`atlas_brain.reasoning.evidence_engine.EvidenceEngine` wraps core while mixing
in that atlas product pack.

## Intentional

- No runtime code changes.
- No test fixture changes.
- No new reasoning-core extraction work.

## Deferred

- New reasoning-core code remains deferred until a concrete product runtime
  asks for a capability listed in the graph-boundary stop rule.
- AI Content Ops source export work remains deferred until a real host export
  fixture exposes an adapter/runtime gap.

## Verification

The first two checks sanity-check the audit's claims about the already-landed
reasoning wrapper boundary; they do not validate runtime changes in this slice.

- Command: python -m py_compile atlas_brain/reasoning/evidence_engine.py atlas_brain/reasoning/review_enrichment.py; result: passed.
- Command: pytest tests/test_atlas_reasoning_evidence_engine_aliases.py; result: 13 passed.
- Command: git diff --check; result: passed.
- Command: bash scripts/local_pr_review.sh; result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination docs | ~10 |
| Reasoning audit | ~55 |
| Plan doc | ~65 |
| **Total** | ~130 |
