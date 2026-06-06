# PR-Content-Ops-FAQ-Detail-Item-Contract

## Why this slice exists

The FAQ search route now supports search-to-detail checks and concurrent detail
hydration, but the shared hosted detail checker only validates the persisted
detail wrapper. A malformed detail payload with `items: []` or item rows that
cannot render as a generated FAQ report can still pass the contract.

The landing-page/demo handoff needs confidence that detail hydration returns the
full generated FAQ shape, not just a top-level wrapper. This slice tightens the
canonical hosted detail checker so every caller that reuses it gets the same
fail-closed item validation.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Robust testing

1. Add shared validation for generated FAQ detail item rows inside the existing
   hosted FAQ search route contract checker.
2. Require hydrated detail payloads to include at least one generated item.
3. Validate the item fields required by the documented frontend/demo FAQ report
   contract: question, summary, steps/action items, evidence status, source
   proof, counts, scores, and vocabulary mappings.
4. Add focused negative fixtures for missing item fields, malformed scalar/list
   shapes, invalid enum values, and malformed term mappings.
5. Leave unrelated concurrency flag polish parked in `HARDENING.md`.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Detail-Item-Contract.md` | Plan contract for this detail item contract slice. |
| `HARDENING.md` | Park the non-blocking #1025 detail/concurrency flag polish item. |
| `scripts/check_content_ops_faq_search_route_contract.py` | Validate generated FAQ item rows in hosted detail payloads. |
| `tests/test_check_content_ops_faq_search_route_contract.py` | Cover success and fail-closed detail item validation branches. |
| `tests/test_smoke_content_ops_faq_search_route_concurrency.py` | Keep the concurrency smoke fixtures aligned with the stricter shared checker. |

## Mechanism

`_validate_detail(...)` remains the single detail validation seam used by the
single-request contract checker, seeded e2e, and concurrency smoke. After the
existing wrapper checks confirm `items` is a list, it validates each generated
FAQ item against the current documented report contract. The checker reports
field-specific paths such as `detail.items[0].question` and
`detail.items[0].term_mappings[0].customer_term` so hosted failures are
actionable.

The implementation intentionally uses small local helpers for strict string,
integer, list, mapping, and enum checks. It does not introduce a new runtime
schema dependency or a second copy of the detail checker.

## Intentional

- No hosted route, database, search projection, or generated FAQ behavior
  changes.
- No OpenAPI/schema generator. This is a focused contract checker hardening
  slice.
- The `--require-detail --allow-empty-results` polish item from #1025 is parked
  instead of folded into this item-contract slice.

## Deferred

- Parked hardening: `Reject Contradictory FAQ Route Detail Concurrency Flags`
  in `HARDENING.md`; left parked because it is operator polish for the
  concurrency runner, not required for detail item contract validation.
- Formal generated schema export remains deferred until the hosted API schema
  is generated from the FastAPI app.

## Verification

- python -m pytest tests/test_check_content_ops_faq_search_route_contract.py tests/test_smoke_content_ops_faq_search_route_concurrency.py -q — 126 passed.
- python -m py_compile scripts/check_content_ops_faq_search_route_contract.py tests/test_check_content_ops_faq_search_route_contract.py tests/test_smoke_content_ops_faq_search_route_concurrency.py — passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Detail-Item-Contract.md — passed.
- git diff --check — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . — 122 matching tests enrolled.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/run_extracted_pipeline_checks.sh — 2531 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 93 |
| HARDENING entry | 11 |
| Contract checker | 149 |
| Contract checker tests | 103 |
| Concurrency fixture alignment | 27 |
| **Total** | **383** |
