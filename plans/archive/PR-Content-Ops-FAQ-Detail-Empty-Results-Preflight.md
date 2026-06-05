# PR-Content-Ops-FAQ-Detail-Empty-Results-Preflight

## Why this slice exists

PR #1025 added opt-in hosted detail hydration to the FAQ search route
concurrency smoke. Review flagged one non-blocking polish gap: operators can
combine `--require-detail` with `--allow-empty-results`, which is
self-contradictory because detail hydration requires `results[0].faq_id`.

That gap was parked in `HARDENING.md`. This slice drains that parked item with a
small preflight guard so the runner rejects the bad configuration before making
concurrent hosted requests.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Product polish

1. Add an argument preflight error when `--require-detail` is used while result
   rows are not required.
2. Cover the negative branch directly in `_validate_args(...)`.
3. Cover the CLI `main(...)` preflight result artifact so the failure is visible
   without issuing network requests.
4. Remove the now-drained `HARDENING.md` entry.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Detail-Empty-Results-Preflight.md` | Plan contract for this polish slice. |
| `HARDENING.md` | Remove the drained parked hardening item. |
| `scripts/smoke_content_ops_faq_search_route_concurrency.py` | Add the contradictory flag preflight guard. |
| `tests/test_smoke_content_ops_faq_search_route_concurrency.py` | Add focused preflight and result-artifact coverage. |

## Mechanism

`_validate_args(...)` already owns runner-level preflight validation. This slice
adds one guard:

```python
if args.require_detail and not args.require_results:
    errors.append("--require-detail requires result rows; remove --allow-empty-results")
```

`main(...)` already writes preflight summaries before returning exit code 2, so
the existing result path will carry the new error without changing summary
shape or runtime request behavior.

## Intentional

- No search route, detail route, contract checker, database, or concurrency
  worker behavior changes.
- No case-file semantic changes. Case rows still control per-query
  `require_results`; this guard only rejects the impossible global detail mode
  with globally allowed empty results.
- No new hardening entries; this PR drains the only active FAQ/search parked
  item.

## Deferred

- Parked hardening: none.
- Per-case detail expectations remain out of scope; the route contract and
  seeded e2e cover detail-field assertions.

## Verification

- python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py -q — 48 passed.
- python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py — passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Detail-Empty-Results-Preflight.md — passed.
- git diff --check — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . — 122 matching tests enrolled.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/run_extracted_pipeline_checks.sh — 2541 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 86 |
| HARDENING cleanup | 11 |
| Runner guard | 2 |
| Tests | 41 |
| **Total** | **140** |
