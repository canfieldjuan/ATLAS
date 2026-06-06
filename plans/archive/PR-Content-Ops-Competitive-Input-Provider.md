# PR: Content Ops Competitive Input Provider

## Why this slice exists

PR #1247 made review-shaped source rows usable from the operator-facing New Run
path. Its deferred marketer follow-up is competitive/displacement-as-input:
marketers also need to turn switching, competitor, and alternative evidence into
landing-page and blog drafts without waiting for the autonomous B2B report path.
The campaign source adapter already preserves competitive fields on normalized
opportunities, and the generators already accept source-grounded context; the
missing product seam is explicit host routing and generator handoff for
operator-supplied competitive evidence.

The diff is over the 400 LOC soft cap because the vertical slice needs the host
source-mode package, generator context handoff, UI selector behavior, and
negative fixtures together; splitting any one of those would leave either an
unselectable source mode or a source package that does not actually reach the
generators.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add an explicit `competitive` / `displacement` source mode beside support
   tickets and reviews.
2. Package operator-supplied competitive/displacement rows through the existing
   source-row adapter, with a fail-closed filter for rows that do not carry
   competitive markers.
3. Carry the packaged competitive evidence into landing-page campaign context
   and blog `data_context`.
4. Extend the New Run source selector and behavioral selector test.
5. Add focused host tests for competitive routing, selected-target tenant
   loading, non-competitive rejection, and generator handoff.

### Files touched

- `plans/PR-Content-Ops-Competitive-Input-Provider.md`
- `atlas_brain/_content_ops_input_provider.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/landing_page_input_contract.py`
- `atlas-intel-ui/src/pages/contentOpsSourceMode.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/scripts/content-ops-review-source-selection.test.mjs`
- `tests/test_atlas_content_ops_review_input_provider.py`

## Mechanism

The host provider recognizes competitive aliases (`competitive`,
`competitive_displacement`, `competitive-signal`, `displacement`, etc.) from
`inputs.source_type` or `inputs.source_material_type`. Hyphenated aliases are
normalized server-side the same way the UI normalizes them. That path expands
competitive bundle keys such as `competitive_signals` and `displacement_edges`
before reusing
`source_material_to_source_rows(...)` and
`source_rows_to_campaign_opportunities(...)`, then keeps only opportunities with
competitive/displacement markers such as competitor, alternative, `from_vendor`,
`to_vendor`, or displacement counts. Rows that only contain generic text fail
closed with a noop warning rather than becoming customer-facing content.

Accepted rows are packaged as `source_material` plus
`competitive_source_material`, with marketer-facing campaign defaults for
landing-page and blog generation. The executor allowlist exposes
`competitive_source_material` to landing pages, and a sibling blog data-context
helper passes the same evidence into blog generation.

## Intentional

- No autonomous B2B Postgres fetcher in this PR. This slice handles
  operator-supplied or selected imported rows; live B2B displacement-table
  sourcing is a later data-selection slice.
- No adapter fork. The host package path reuses the existing source-row adapter
  and applies only host-level source-mode filtering plus competitive-specific
  bundle-key expansion.
- The UI keeps the existing `test:content-ops-review-source-selection` script
  rather than adding a new script, so no new intel-ui workflow enrollment is
  needed.

## Deferred

- Pulling canonical B2B displacement dynamics directly from the B2B tables.
- Competitive-specific output skills beyond blog and landing page, such as ad
  copy, social posts, or stat/quote cards.
- Product packaging/pricing for the competitive marketer offer.

## Parked hardening

None.

## Verification

- Command: pytest tests/test_atlas_content_ops_review_input_provider.py -q -- 16 passed.
- Review follow-up command: pytest tests/test_atlas_content_ops_review_input_provider.py -q -- 18 passed.
- Command: pytest tests/test_atlas_content_ops_review_input_provider.py tests/test_atlas_content_ops_input_provider.py -q -- 40 passed, 1 warning.
- Review follow-up command: pytest tests/test_atlas_content_ops_review_input_provider.py tests/test_atlas_content_ops_input_provider.py -q -- 42 passed, 1 warning.
- Command: cd atlas-intel-ui && npm run test:content-ops-review-source-selection -- 5 passed.
- Command: cd atlas-intel-ui && npm run lint -- passed.
- Command: cd atlas-intel-ui && npm run build -- passed.
- Command: python -m py_compile atlas_brain/_content_ops_input_provider.py extracted_content_pipeline/content_ops_execution.py extracted_content_pipeline/landing_page_input_contract.py -- passed.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main -- OK: 143 matching tests are enrolled.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- Command: bash scripts/check_ascii_python.sh -- passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- 2935 passed, 10 skipped, 1 warning.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-competitive-input-provider-pr-body.md -- passed.

## Estimated diff size

Estimated: 762 LOC actual (8 files, +756 / -6). This is over the 400 LOC soft
cap for the vertical-slice reason named in **Why this slice exists**.

| Area | Estimated LOC |
|---|---:|
| Host provider + executor context | ~409 |
| Backend tests | ~194 |
| Frontend selector + test updates | ~51 |
| Plan doc | ~119 |
| **Total** | **~762** |
