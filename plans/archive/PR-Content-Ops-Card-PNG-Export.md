# PR: Content Ops Card PNG Export

## Why this slice exists

PR #1297 added static HTML visual exports for quote/stat cards, and PR #1298
put that export behind an atlas-intel-ui action. Those artifacts are useful
for review, but marketers ultimately need image files they can upload into
social, paid, and presentation workflows.

This slice closes the next deferred visual-export step: `quote_card` and
`stat_card` can be exported as PNG images through the same generated-assets
export route, without changing generation, review state, tenant scoping, or
#1268 output variations.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Product polish

1. Add a lazy Playwright-backed PNG renderer that converts the existing static
   HTML visual card document into a full-page screenshot.
2. Extend `/content-assets/{asset}/drafts/export` with `format=png` only for
   `quote_card` and `stat_card`, reusing the same tenant scope, status,
   target mode, theme, and limit filters as CSV/JSON/HTML.
3. Fail closed with a service-unavailable response when the optional PNG
   renderer cannot run.
4. Add focused tests that mock the browser boundary, prove route shape and
   PNG response headers, and prove non-card assets cannot request PNG.

### Files touched

- `plans/PR-Content-Ops-Card-PNG-Export.md`
- `extracted_content_pipeline/card_visual_export.py`
- `extracted_content_pipeline/api/generated_assets.py`
- `tests/test_extracted_content_asset_api.py`

## Mechanism

The renderer reuses `render_card_visual_html(...)` so the PNG path inherits the
same escaped static markup accepted by #1297. Playwright is imported lazily only
when PNG rendering is requested:

```python
html = render_card_visual_html(asset, rows)
async with async_playwright() as pw:
    browser = await pw.chromium.launch()
    png = await _screenshot_card_html(browser, html, config)
```

Tests supply a fake browser to the renderer helper and monkeypatch the API
route's renderer function, so CI validates the real browser-boundary calls
without requiring a live browser install. If Playwright is missing or browser
startup fails in production, the route returns `503` rather than pretending to
export a valid PNG.

## Intentional

- No frontend button in this slice. This PR establishes the backend PNG
  contract first; the UI can add an `Export PNG` action after review accepts
  the route and failure behavior.
- No new dependency. Playwright is already in the repo requirements and stays a
  lazy import.
- No #1268 output-variations work.
- No id-filter support for quote/stat exports.

## Deferred

- atlas-intel-ui `Export PNG` action for quote/stat cards after this backend
  contract is accepted.
- Optional quote/stat id deep links remain deferred until a product path needs
  exact run-result links.

## Parked hardening

None.

## Verification

- `pytest tests/test_extracted_content_asset_api.py -q` -- 91 passed.
- `python -m py_compile extracted_content_pipeline/card_visual_export.py extracted_content_pipeline/api/generated_assets.py tests/test_extracted_content_asset_api.py && git diff --check` -- passed.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main` -- OK: 146 matching tests are enrolled.
- `bash scripts/run_extracted_pipeline_checks.sh` -- 3063 passed, 10 skipped.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan | +94 / -0 |
| Product + tests | +251 / -3 |
| **Total** | **~350** |
