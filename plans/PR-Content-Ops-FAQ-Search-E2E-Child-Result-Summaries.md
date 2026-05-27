# PR-Content-Ops-FAQ-Search-E2E-Child-Result-Summaries

## Why this slice exists

The seeded hosted FAQ search e2e runner composes the real flow: DB seed,
hosted route concurrency, detail hydration, and cleanup. Its top-level result
currently keeps subprocess return codes plus artifact paths, but when the
runner uses the default temporary artifact directory those child artifact files
are cleaned up before operators can inspect them.

This slice keeps the e2e result self-diagnostic by embedding compact seed,
route, and detail summaries before temporary artifact cleanup runs.

This slightly exceeds the 400 LOC target because the checker needs focused
negative fixtures for missing and malformed child artifacts; without those
fixtures the new fail-closed detector would be under-proven.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add compact result-artifact summaries to seed, route, and detail phases.
2. Fail closed when a child command reports success but its requested
   `--output-result` artifact is missing, malformed, has non-boolean `ok`, or
   is not an object.
3. Keep full child payload bodies out of the top-level e2e result.
4. Preserve existing seed, route, detail, cleanup, and artifact-cleanup control
   flow.
5. Add focused tests for success summaries and missing-artifact detection.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-E2E-Child-Result-Summaries.md` | Plan contract for this e2e visibility slice. |
| `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` | Attach compact child result summaries to command phase results. |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | Cover embedded summaries and missing child artifacts. |

## Mechanism

The e2e runner already passes `--output-result` to each child script. A new
helper reads those JSON files immediately after each child command returns and
attaches a compact `result_artifact` object to the corresponding phase.

Seed summaries keep request/setup/lifecycle/latency/isolation counts. Route
summaries keep request/latency/error/budget/case counts. Detail summaries keep
detail-check status, count, FAQ id, timings, and errors.

If the child command exited zero but its artifact is missing, malformed, or has
an invalid `ok` type, the phase is marked `ok=false` with a result-artifact
error so the retained e2e summary cannot falsely pass without proof.

## Intentional

- No hosted route, detail route, seeding, cleanup SQL, or search behavior
  changes.
- The top-level result remains compact; it does not duplicate full route result
  rows or full child JSON bodies.
- Missing artifacts from an already-failing child are reported but do not change
  the already-failed phase semantics.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ
  search entries touching this runner.
- A separate detail-route concurrency runner remains deferred until we have a
  concrete hosted detail latency target.

## Verification

- python -m pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q — 54 passed.
- python -m py_compile scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py — passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-E2E-Child-Result-Summaries.md — passed.
- git diff --check — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . — 122 matching tests enrolled.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/run_extracted_pipeline_checks.sh — 2508 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 89 |
| E2E runner | 121 |
| Tests | 240 |
| **Total** | **451** |
