# PR: Content Ops Card Visual Export

## Why this slice exists

PR #1293 and PR #1296 completed durable review/export lifecycles for
`quote_card` and `stat_card`, but their exports are still data-only CSV/JSON
rows. The deferred product-polish item from both plans is visual
template/export generation for those cards after review rows exist.

This slice adds the first visual artifact path without changing generation,
review status, or #1268 output variations: approved or filtered quote/stat
rows can be exported as self-contained HTML cards through the existing
generated-assets export route.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Product polish

1. Add a package-owned, dependency-light card visual renderer for
   `quote_card` and `stat_card` rows.
2. Extend `/content-assets/{asset}/drafts/export` with `format=html` only for
   `quote_card` and `stat_card`, reusing the existing tenant scope, status,
   target mode, theme, and limit filters.
3. Return a downloadable `text/html` response with escaped row content and no
   JavaScript.
4. Add focused API tests proving quote/stat HTML output, HTML escaping, and
   fail-closed rejection for unsupported asset types.

### Files touched

- `plans/PR-Content-Ops-Card-Visual-Export.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/card_visual_export.py`
- `extracted_content_pipeline/api/generated_assets.py`
- `tests/test_extracted_content_asset_api.py`

## Mechanism

The new renderer accepts rows already produced by the quote/stat export
helpers:

```python
html = render_card_visual_html(asset_name, result.rows)
```

The API route continues to call `_export_for_asset(...)` first, so database
access and tenant scoping remain exactly the same as CSV/JSON export. After
that, `format=html` branches only when the asset is visual-card capable:

```python
if format_name == "html":
    if not supports_card_visual_export(asset_name):
        raise HTTPException(...)
    return Response(content=render_card_visual_html(asset_name, result.rows), ...)
```

All row text is escaped with `html.escape`, the output is static HTML/CSS, and
the renderer does not include metadata JSON blobs or script execution.

## Intentional

- No PNG/SVG rendering. Self-contained HTML is the first reviewable visual
  artifact; raster export can follow once the template is accepted.
- No atlas-intel-ui download button in this slice. The API path is enough to
  validate the artifact contract; UI polish can call the same route later.
- No change to default export status. The route keeps the existing `status`
  filter behavior, so operators can request `status=approved` without a hidden
  special case.
- No #1268 output-variations work.

## Deferred

- Add an atlas-intel-ui visual export button for quote/stat cards after this
  backend artifact contract is accepted.
- Add PNG rendering or server-side image export after the HTML template has
  been reviewed with real approved rows.
- Optional quote/stat id deep links remain deferred until a product path needs
  exact run-result links.

## Parked hardening

None.

## Verification

- Passed: `pytest tests/test_extracted_content_asset_api.py -q`
  (85 passed)
- Passed: `pytest tests/test_extracted_content_asset_api.py tests/test_atlas_content_ops_generated_assets_api.py -q`
  (99 passed, 1 warning)
- Passed: `python -m py_compile extracted_content_pipeline/card_visual_export.py extracted_content_pipeline/api/generated_assets.py tests/test_extracted_content_asset_api.py`
- Passed: `git diff --check`
- Passed: `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main`
  (OK: 146 matching tests are enrolled.)
- Passed: `bash scripts/validate_extracted_content_pipeline.sh`
- Passed: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
- Passed: `python scripts/audit_extracted_standalone.py --fail-on-debt`
- Passed: `bash scripts/check_ascii_python.sh`
- Passed: `bash scripts/run_extracted_pipeline_checks.sh`
  (3057 passed, 10 skipped, 1 warning)

## Estimated diff size

Actual git diff: 5 files, +392 / -3.

| Area | LOC |
|---|---:|
| **Total** | **395** |
