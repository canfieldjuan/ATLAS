# Reasoning Enrichment Pack Split

## Why this slice exists

PR #563 reset the reasoning-core backlog and identified the next concrete gap:
atlas-side per-review enrichment still lives directly on
`atlas_brain.reasoning.evidence_engine.EvidenceEngine`. That keeps core slim,
but the product-specific enrichment policy is still embedded inside the wrapper
instead of living in an explicit pack module.

This is expected to exceed the 400 LOC soft budget because it moves an existing
method block into a new module while deleting the same block from the wrapper.
The behavioral change is intentionally narrow: composition only, no logic
rewrite.

## Scope (this PR)

1. Move the six atlas-side per-review enrichment methods into a dedicated
   `atlas_brain.reasoning.review_enrichment` module.
2. Keep `atlas_brain.reasoning.evidence_engine.EvidenceEngine` as the existing
   public object shape by mixing in the product pack.
3. Preserve the current method signatures and behavior.
4. Update tests that currently pin the wrapper/enrichment boundary.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `plans/PR-Reasoning-Enrichment-Pack-Split.md`
- `atlas_brain/reasoning/evidence_engine.py`
- `atlas_brain/reasoning/review_enrichment.py`
- `extracted_reasoning_core/evidence_engine.py`
- `tests/test_atlas_reasoning_evidence_engine_aliases.py`

## Mechanism

Add `ReviewEnrichmentMixin` in `atlas_brain.reasoning.review_enrichment`. The
mixin owns:

- regex precompile setup for recommendation and pricing phrase rules;
- `compute_urgency`;
- `override_pain`;
- `derive_recommend`;
- `derive_price_complaint`;
- `derive_budget_authority`;
- `_check_derivation_rule`.

`atlas_brain.reasoning.evidence_engine.EvidenceEngine` remains the public
factory product. It subclasses `ReviewEnrichmentMixin` and core's slim
`EvidenceEngine`, then calls the mixin setup after core loads the YAML rules.

## Intentional

- No behavior change to callers. `get_evidence_engine()` still returns
  `atlas_brain.reasoning.evidence_engine.EvidenceEngine`.
- No new public API in `extracted_reasoning_core`. The enrichment pack is
  atlas/product-owned because `derive_price_complaint` depends on atlas-only
  phrase metadata helpers.
- No function signature changes.

## Deferred

- Exporting the enrichment pack for other products. This PR only makes the
  atlas product pack explicit.
- Graph/state slimming remains a later reasoning-core slice.

## Verification

- python -m py_compile atlas_brain/reasoning/evidence_engine.py atlas_brain/reasoning/review_enrichment.py extracted_reasoning_core/evidence_engine.py tests/test_atlas_reasoning_evidence_engine_aliases.py - passed.
- pytest tests/test_atlas_reasoning_evidence_engine_aliases.py tests/test_evidence_engine.py tests/test_b2b_phase2_subject_gate.py tests/test_b2b_phase3_polarity_gate.py - 105 passed.
- git diff --check - passed.
- bash scripts/local_pr_review.sh - passed after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination docs | ~10 |
| Plan doc | ~85 |
| New review enrichment module | ~265 |
| Evidence engine wrapper trim | ~250 |
| Core docstring update | ~10 |
| Tests | ~35 |
| **Total** | ~655 |
