# PR-Content-Ops-FAQ-Stress-Hardening-Closeout

## Why this slice exists

Root `HARDENING.md` still lists two FAQ stress items from the scale probe:
`FAQSTRESS-1` for oversized hosted uploads and `FAQSTRESS-2` for concurrent DB
pressure plus missing lifecycle failure artifacts.

Those entries are stale in their current form after the follow-up chain:

- PR #861 added bounded server-side file ingestion.
- PR #863 made lifecycle setup failures write result artifacts.
- PR #864 added a hosted `/content-ops/execute` concurrency gate.
- PR #867 added frontend adapter coverage proving file uploads route through
  the bounded file endpoints.

This slice drains the resolved root hardening queue without changing runtime
behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-validation

1. Remove the resolved `FAQSTRESS-1` and `FAQSTRESS-2` entries from root
   `HARDENING.md`.
2. Add a closeout note tying each removed item to the production limits and
   regression tests now on `main`.
3. Leave historical probe docs untouched; they remain accurate snapshots of the
   state when the stress probe ran.

### Files touched

- `plans/PR-Content-Ops-FAQ-Stress-Hardening-Closeout.md`
- `docs/extraction/validation/content_ops_faq_stress_hardening_closeout_2026-05-23.md`
- `HARDENING.md`

## Mechanism

No production code changes. The closeout note records the current enforcement
points:

- inline `/content-ops/execute` source material is capped at 1,000 rows;
- uploaded ingestion files are capped at 25 MB and 10,000 normalized rows;
- `/content-ops/execute` has router-local fail-fast concurrency admission;
- lifecycle setup failures now write the requested result JSON.

Relevant tests already exist for these enforcement points and are rerun in this
slice.

## Intentional

- No async job runner in this slice. Large uploads above the hosted caps are not
  accepted by the synchronous hosted path.
- No distributed/global admission controller yet. PR #864 intentionally added a
  per-router gate; a cross-process gate needs the deployed worker topology
  before it can be designed without guessing.
- No edits to the original stress probe report. It should remain a historical
  record, not a mutable status page.

## Deferred

- Parked hardening: none added.
- A future deploy-topology slice can add global admission control if production
  runs multiple API workers against a shared database pool.
- A future product slice can add background jobs if we decide to support FAQ
  uploads above the current synchronous hosted limits.

## Verification

- Passed: focused FAQ hardening closeout pytest:
  `tests/test_smoke_content_ops_faq_lifecycle.py::test_faq_lifecycle_smoke_writes_result_when_pool_creation_fails`
  plus six Content Ops control-surface limit/concurrency tests (`7 passed`).
- Passed: local PR review via `scripts/local_pr_review.sh`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~72 |
| Closeout note | ~65 |
| HARDENING cleanup | ~18 |
| **Total** | **~155** |
