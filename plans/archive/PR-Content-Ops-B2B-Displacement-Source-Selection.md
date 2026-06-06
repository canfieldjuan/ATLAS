# PR: Content Ops B2B Displacement Source Selection

## Why this slice exists

PR #1253 made operator-supplied competitive/displacement rows usable in the
Content Ops New Run path. Its first deferred product gap is direct sourcing from
Atlas's canonical B2B displacement tables: today an operator can paste or import
competitive rows, but cannot ask the competitive input provider to pull the
latest tracked-vendor displacement dynamics already computed by the B2B engine.

This slice is the thinnest data-selection step over the merged competitive
provider. It does not add new generators or output types; it turns canonical B2B
displacement dynamics into the same source-row shape the existing competitive
provider already accepts. The diff is over the 400 LOC soft cap because the
slice must ship the scoped SQL loader, dynamics-to-source-row conversion, and
failure-branch tests together; splitting out any one of those would leave a
source selector that either cannot load real data or has unproven fail-closed
behavior.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add request keys for competitive runs to select tracked B2B displacement
   vendors from canonical `b2b_displacement_dynamics`.
2. Load only rows scoped through the tenant's `tracked_vendors` account binding.
3. Convert loaded dynamics rows into competitive source rows with text,
   from/to vendors, displacement counts, driver, source ids, and dates.
4. Feed those rows through the already-merged `_build_competitive_input_package`
   path so landing/blog handoff stays unchanged.
5. Add focused host tests for successful load, missing account scope,
   unconfigured repository, and empty/missing dynamics.

### Files touched

- `plans/PR-Content-Ops-B2B-Displacement-Source-Selection.md`
- `atlas_brain/_content_ops_input_provider.py`
- `tests/test_atlas_content_ops_review_input_provider.py`

## Mechanism

For `source_type` values in the competitive alias set, the host provider will
look for vendor selectors such as `b2b_displacement_vendors`. When present, it
will use the existing `pool_provider` to query `b2b_displacement_dynamics` for
latest rows whose `from_vendor` is one of the requested vendors and is also in
`tracked_vendors` for `scope.account_id`.

Each returned dynamics row is normalized into a competitive source row:
`source_type=competitive_displacement`, `source_id`/`target_id`, `from_vendor`,
`to_vendor`, `competitor`, `competitive_alternatives`,
`displacement_mention_count`, `primary_driver`, and a compact `text` summary
derived from canonical dynamics fields such as `battle_summary`,
`migration_proof`, and `edge_metrics`. The existing competitive package builder
then applies the same fail-closed marker filter and generator context handoff as
#1253.

## Intentional

- No UI selector in this PR. This slice proves the backend/data seam through
  request JSON first; UI selection can follow once the contract is stable.
- No new generator or output skill. Landing page and blog are already wired by
  #1253.
- No global fallback. Without `scope.account_id` or a configured pool, canonical
  B2B displacement selection noops with warnings rather than reading unscoped
  rows.

## Deferred

- UI controls for picking tracked B2B displacement vendors in New Run.
- Competitive-specific output skills beyond blog and landing page, such as ad
  copy, social posts, or stat/quote cards.
- Product packaging/pricing for the marketer competitive offer.

## Parked hardening

None.

## Verification

- Command: pytest tests/test_atlas_content_ops_review_input_provider.py -q -- 22 passed.
- Command: pytest tests/test_atlas_content_ops_review_input_provider.py tests/test_atlas_content_ops_input_provider.py -q -- 46 passed, 1 warning.
- Command: python -m py_compile atlas_brain/_content_ops_input_provider.py -- passed.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main -- OK: 144 matching tests are enrolled.
- Command: git diff --check -- passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- 2948 passed, 10 skipped, 1 warning.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-b2b-displacement-source-selection-pr-body.md -- passed.

## Estimated diff size

Estimated: 581 LOC actual (3 files, +575 / -6). This exceeds the 400 LOC soft
cap for the vertical-slice reason named in **Why this slice exists**.

| Area | Estimated LOC |
|---|---:|
| Host provider B2B displacement loader | ~336 |
| Backend tests | ~146 |
| Plan doc | ~96 |
| **Total** | **~581** |
