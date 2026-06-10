# PR-Deflection-Cluster-Preview-Skip

## Why this slice exists

#1454 (P0, pre-#1440 gate): removing the 1000-row submit cap in #1452 made the
support-ticket clustering pass reachable at full upload volume for the first
time. Measurement on the real submit path (recorded on #1454) showed two
things. First, the actual #1440 CFPB sample is safe: every row carries an
explicit category, so clustering takes 0.17 s at 35,386 rows. Second, the
token-set branch - the shape of any provider export WITHOUT a category column -
is quadratic (6.7 s at 2k rows, ~40 minutes extrapolated at 35k) and, on
long-form text, degenerate: every row anchors into one mega-cluster labeled by
the most frequent token. A single category-less 50 MB upload would wedge the
generation worker to produce a useless preview.

Per the operator decision on #1454, this slice implements option 1: skip the
token-set cluster preview above a row threshold and surface the skip visibly,
instead of attempting an exact-parity algorithmic rewrite (the prescribed
inverted-index/MinHash replacements do not fit the measured token
distribution; details on #1454).

## Scope (this PR)

Ownership lane: deflection-full-50k-e2e-proof
Slice phase: Production hardening

1. Gate the pairwise token-set clustering pass behind a row threshold
   (`MAX_TOKEN_SET_CLUSTER_ROWS = 2000`); above it, token-set rows stay
   uncategorized and the skip is reported. Explicit/provided/keyword hints
   always cluster (cheap key-equality bucketing).
2. Surface the skip at every layer: clustering diagnostics, input-package
   warning (`cluster_preview_skipped_large_upload`) and metadata, and the
   deflection submit response (`input_provider` metadata + warnings).
3. Regression coverage per #1454 acceptance: a 25k-row token-set input
   completes in seconds and reports the skip.

### Review Contract

- Acceptance criteria:
  - [ ] `assign_support_ticket_clusters` never runs the pairwise token-set
        scan when token-set rows exceed the threshold; those rows are
        returned without cluster annotations.
  - [ ] Rows with explicit category hints still cluster, even in uploads
        where the token-set preview is skipped.
  - [ ] The skip is visible: package warning + metadata, and submit response
        `input_provider.warnings` / metadata.
  - [ ] Below the threshold, behavior is byte-identical to before this PR.
  - [ ] A regression test proves a 25k-row token-set input completes within
        a bounded wall-time and reports the skip.
- Affected surfaces: clustering service, support-ticket input package, submit
  API diagnostics, tests.
- Risk areas: performance, diagnostic truthfulness, backward compatibility.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/support_ticket_clustering.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Deflection-Cluster-Preview-Skip.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_clustering_scale.py`

## Mechanism

`assign_support_ticket_clusters_with_diagnostics` is the new core entry
point: it counts token-set hints up front and, when the count exceeds
`max_token_set_rows` (default `MAX_TOKEN_SET_CLUSTER_ROWS`, resolved at call
time so tests can patch the module constant), appends `None` assignments for
token-set rows instead of querying `_matching_token_bucket`. It returns the
annotated rows plus a diagnostics dict (`token_set_row_count`,
`max_token_set_rows`, `cluster_preview_skipped`). The existing
`assign_support_ticket_clusters` keeps its public row-tuple contract by
delegating.

`build_support_ticket_input_package` consumes the diagnostics variant: when
the preview is skipped it appends a `cluster_preview_skipped_large_upload`
warning (row count + threshold, and an explicit statement that the rows are
still included in the report) and sets `cluster_preview_skipped` /
`cluster_preview_token_set_row_count` metadata. The metadata keys are present
only when the skip fires, so existing exact-dict assertions stay valid.
Skipped rows count as `uncategorized_row_count` in the existing
`cluster_quality` counters.

The submit response copies the two new metadata keys and propagates the skip
warning into `input_provider.warnings`, alongside the existing language-filter
and truncation warnings.

The threshold is 2000: the legacy submit path capped uploads at 1000 rows, so
no input shape that previously clustered is ever skipped, and the worst
measured cost at the threshold (~6.7 s on long-form CFPB narratives, shorter
for typical ticket subjects) stays within the few-seconds acceptance bound.

## Intentional

- This is the operator-selected option 1 from #1454, not a clustering
  rewrite. A rarity-weighted redesign of category-less clustering (which
  would also fix the mega-bucket degeneracy below the threshold) is a
  separate quality slice if category-less exports become a real lane.
- Explicit-category rows always cluster regardless of upload size; the
  threshold only gates rows that would enter the quadratic token-set path.
- `support_ticket_cluster_quality` re-derives clusters via
  `_ensure_clustered` when no row carries a cluster; with the skip active
  this re-runs the gated assignment (another linear tokenize pass, no
  quadratic work). Accepted for this slice to avoid changing the public
  quality helper.
- The threshold is a module constant with a call-time override parameter,
  not a config setting; the package has no config plumbing and the value is
  a measured engineering bound, not a tenant knob.
- The 25k regression test asserts a generous 30 s wall bound; the quadratic
  path it guards against measured ~6.7 s at just 2k rows, so the bound
  discriminates without CI flakiness.

## Deferred

- The #1440 live run (submit the CFPB sample through the hosted endpoint,
  snapshot email/PDF, payment, report email/PDF) follows this slice; the
  measurement on #1454 already proved the live input never enters the gated
  path.
- Rarity-weighted anchor redesign for category-less clustering quality
  (#1454 option 3).
- Remaining #1454 acceptance text mentions "a few seconds at 50k rows":
  with the skip, 50k token-set rows are a linear tokenize pass (the 25k
  regression test bounds it); no pairwise work runs at any size above the
  threshold.

Parked hardening: none.

## Verification

- Passed: pytest tests/test_extracted_support_ticket_clustering_scale.py
  tests/test_extracted_support_ticket_input_package.py
  tests/test_extracted_content_deflection_submit.py
  tests/test_extracted_ticket_faq_markdown.py
  tests/test_smoke_content_ops_support_ticket_package.py
  - 267 passed.
- Passed: bash scripts/run_extracted_pipeline_checks.sh
  - 3730 passed, 10 skipped (after enrolling the new test file in the
    runner + workflow filters per the CI-enrollment contract test).
- Passed: bash scripts/check_ascii_python.sh
- Passed: python scripts/audit_extracted_pipeline_ci_enrollment.py
  - OK: 165 matching tests are enrolled.
- Pending before push:
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Cluster-Preview-Skip.md --check
  - bash scripts/local_pr_review.sh

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `extracted_content_pipeline/api/control_surfaces.py` | 9 |
| `extracted_content_pipeline/support_ticket_clustering.py` | 49 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 57 |
| `plans/PR-Deflection-Cluster-Preview-Skip.md` | 161 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_content_deflection_submit.py` | 40 |
| `tests/test_extracted_support_ticket_clustering_scale.py` | 138 |
| **Total** | **457** |
