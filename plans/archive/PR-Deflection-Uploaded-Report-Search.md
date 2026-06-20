# PR-Deflection-Uploaded-Report-Search

## Why this slice exists

atlas-portfolio #341 now ships the paid uploaded-report search workbench dark
behind `DEFLECTION_UPLOADED_SEARCH_ENABLED=true`. That portfolio path is honest
and paid-gated, but it cannot run end to end until ATLAS serves the matching
request-scoped search endpoint. The root cause is that ATLAS currently exposes
only tenant-scoped compact FAQ search at
`GET /api/v1/content-ops/faq-deflection-search`; there is no
`POST /api/v1/content-ops/deflection-reports/{request_id}/search` contract that
searches a buyer's uploaded report and returns full report items.

This slice closes the ATLAS side of that contract without faking shape from the
existing compact projection.

The diff is over the 400 LOC soft cap because the route is small but the
review contract requires guard-shaped tests for paid access, request scoping,
validation-before-store, limit capping, and compact-row rejection in the same
slice. Splitting the tests from the endpoint would ship an unproven paid-data
gate.

## Scope (this PR)

Ownership lane: deflection/uploaded-report-search
Slice phase: Vertical slice

1. Add `POST /deflection-reports/{request_id}/search` beside the existing
   deflection report snapshot/artifact/report-model routes.
2. Accept JSON `{ q, limit }`, validate the query before store access, cap the
   limit, and keep customer ticket phrases out of URLs.
3. Scope lookup by the current tenant/account and `request_id`, require the
   existing paid/unlocked report state, and return 404/403 with the same
   semantics as artifact/report-model access.
4. Search only the stored, scrubbed full artifact's `faq_result.items`, and
   return full item payloads under an envelope shape the portfolio parser can
   accept: `{ query, count, results: [{ item, score, rank }] }`.
5. Add focused route tests for happy path, empty result, locked/missing reports,
   malformed/overlong query, limit capping, malformed artifact rows, and no
   compact-row fabrication.

### Review Contract

Acceptance criteria:

- The route path and method match the portfolio contract:
  `POST /api/v1/content-ops/deflection-reports/{request_id}/search` once the
  control-surface router is mounted at `/api/v1/content-ops`.
- Search is request-scoped, not tenant-corpus scoped: no call to the existing
  `ticket_faq_search_documents` projection is used for uploaded report search.
- A report must exist and be paid before any full `faq_result.items` payload can
  be returned.
- The route returns full stored report items only; malformed or compact rows are
  skipped, not inflated into fake `TicketFAQItem` objects.
- Query text is accepted only in the POST body and is validated before store
  access on blank/overlong input.

Affected surfaces:

- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_deflection_submit.py`

Risk areas:

- Paid-gate leakage of full report items.
- Cross-tenant/request leakage.
- Compact-row shape adaptation that would violate portfolio's "real, not fake"
  requirement.
- Search behavior over scrubbed artifact JSON after the PII scrub slices.

Reviewer rules triggered: R1, R2, R3, R5, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Uploaded-Report-Search.md`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

The new route reuses the existing control-surface dependencies:
`_resolve_deflection_report_store` for the persisted report store and
`_resolve_scope` / `_required_scope_account_id` for account isolation. It reads
the same `DeflectionReportAccessRecord` used by `/artifact` and
`/report-model`; missing reports return 404 and unpaid reports return 403.

After paid access passes, the route searches the stored artifact's
`faq_result.items`. Each candidate must already be a full item mapping with the
fields the paid report workbench needs (`topic`, `question`, `answer`,
`when_to_contact_support`, `steps`, `action_items`, `source_ids`,
`source_labels`, `term_mappings`, count/evidence fields). Candidates that do
not meet that minimum are skipped. The matcher tokenizes the query and scores
matches across question, topic, answer, customer term mappings, steps, action
items, and source labels. Results are sorted deterministically by score, rank,
and question, capped by the configured API limit, and returned as:

```json
{
  "query": "export reports",
  "count": 1,
  "results": [
    { "item": { "...": "full stored TicketFAQItem fields" }, "score": 12, "rank": 1 }
  ]
}
```

The route does not touch the tenant-scoped compact FAQ search projection. That
projection remains useful for approved FAQ corpus search, but uploaded report
search needs the exact report artifact generated from the buyer's CSV.

## Intentional

- Keep this as a local artifact search instead of adding a new search table in
  the first ATLAS companion slice; the stored report artifact already contains
  the scrubbed full items the paid workbench must render.
- Do not hydrate compact `ticket_faq_search_documents` rows through
  `/faq-deflection-search/{faq_id}`. Those rows are tenant-corpus scoped and can
  mismatch the uploaded request.
- Skip malformed full-item candidates instead of fabricating required fields.
- Reuse the existing paid report gate rather than adding a separate unlock
  concept for search.

## Deferred

- Live portfolio enablement remains a follow-up: after this PR lands and live
  upload-search verification passes, set
  `DEFLECTION_UPLOADED_SEARCH_ENABLED=true` in the portfolio deployment.
- A future scale hardening slice can add a persisted request-scoped FTS index if
  artifact-local search becomes too slow for larger paid reports.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_content_deflection_submit.py -k "deflection_report_search" -q`
  -- 4 passed, 70 deselected.
- `pytest tests/test_extracted_content_deflection_submit.py -q` -- 74 passed.
- `python -m py_compile extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_deflection_submit.py`
  -- passed.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `bash scripts/run_extracted_pipeline_checks.sh` -- 4742 passed, 15 skipped,
  1 existing torch warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr_body_deflection_uploaded_report_search.md`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 205 |
| `plans/PR-Deflection-Uploaded-Report-Search.md` | 157 |
| `tests/test_extracted_content_deflection_submit.py` | 233 |
| **Total** | **595** |
