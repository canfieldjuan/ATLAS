# PR-Content-Ops-Output-Variations-Shared-Fanout

## Why this slice exists

PR #1347 added blog output variations and PR #1349 added landing-page output
variations. The #1349 review was LGTM, but it called out one concrete NIT before
the third generator: blog and landing-page fan-out now duplicate the same
per-angle aggregation, caught-exception isolation, all-raising failure status,
warning, saved-id collection, and error-tagging logic. Manually copying that
logic into sales briefs would triple the exact surface that already cost one
review cycle in #1347.

This hardening slice removes that duplication before sales-brief parity. It
does not add a new variation-capable output. It preserves the reviewed blog and
landing-page result contracts, but moves the shared loop into one helper so the
next generator composes with the same failure semantics instead of copying them.

## Scope (this PR)

Ownership lane: content-ops/output-variations/shared-fanout
Slice phase: Production hardening

1. Extract one shared variant fan-out helper for the already-supported blog and
   landing-page dispatchers.
2. Keep the existing single-call behavior for `variant_count == 1`.
3. Keep the public aggregate result shape unchanged for blog and landing-page
   variants: counts, saved ids, per-angle `variant_results`, tagged errors,
   warnings, and all-raising step failure status.
4. Add or keep focused tests proving blog and landing-page fan-out, partial
   failure isolation, and all-raising failure behavior through the public
   executor.

### Review Contract
- Acceptance criteria:
  - [ ] Blog and landing-page variant requests still call their services once
        per selected angle with the same kwargs they used before this refactor.
  - [ ] Partial per-angle failures remain non-fatal for both outputs.
  - [ ] All-raising blog variants still fail with `all_blog_variants_failed`.
  - [ ] All-raising landing-page variants still fail with
        `all_landing_page_variants_failed`.
  - [ ] The shared helper has one implementation of aggregate counts, saved ids,
        `variant_results`, tagged errors, warnings, and caught-exception status.
  - [ ] No sales-brief behavior changes in this PR.
- Affected surfaces: extracted package execution dispatcher and execution
  regression tests.
- Risk areas: behavior-preserving refactor, result contract drift, failure
  status regressions.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `extracted_content_pipeline/content_ops_execution.py`
- `plans/PR-Content-Ops-Output-Variations-Shared-Fanout.md`

## Mechanism

Add one private helper in `content_ops_execution.py` that accepts the request,
a per-angle async generation callback, the requested-count default, the
all-raising failure label, and the all-zero warning message. The helper owns the
loop over `selected_variant_angles`, calls the callback with each
`VariantAngle`, converts results through the existing result helpers, tags
per-angle errors, accumulates saved ids and consumed reasoning, and raises
`_ContentOpsStepResultFailure` only when `generated == 0` and at least one
variant call raised.

The blog dispatcher passes a callback that calls the blog service with
`scope`, `target_mode`, `limit`, `filters`, and the selected angle instruction.
The landing-page dispatcher passes a callback that calls the landing-page
service with its campaign kwargs and the selected angle instruction. Their
single-call branches remain unchanged. The old blog/landing-specific variant
loops collapse to thin wrappers around the helper so the warning text and
failure labels stay output-specific.

## Intentional

- This PR is a refactor/hardening slice, not sales-brief parity. It removes the
  copied logic first because the #1349 review explicitly warned against a third
  copy.
- The helper remains private to `content_ops_execution.py`; no new package port
  or public API is needed.
- Tests stay at the executor boundary instead of asserting private helper
  internals. The risk is public behavior drift, so the public result contract is
  the right proof point.

## Deferred

- Sales-brief `variant_angle` parity.
- Persistent variant grouping (`variant_group_id`) and a past-run variants view.
- Auto-ranking / recommended winner and A/B serving/analytics.

Parked hardening: none.

## Verification

- Command: pytest tests/test_extracted_content_ops_execution.py -q -- PASS, 74 tests.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- PASS.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- PASS.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- PASS.
- Command: bash scripts/check_ascii_python.sh -- PASS.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- PASS, 3,237 passed / 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/content_ops_execution.py` | 147 |
| `plans/PR-Content-Ops-Output-Variations-Shared-Fanout.md` | 108 |
| **Total** | **255** |
