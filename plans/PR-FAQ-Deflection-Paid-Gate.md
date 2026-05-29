# PR-FAQ-Deflection-Paid-Gate

## Why this slice exists

The portfolio results page is moving to a Free Snapshot -> $1,500 Backlog
Report -> $500/mo ongoing model. The current hosted `faq_deflection_report`
execute response returns the full `DeflectionReportArtifact` immediately:
Markdown, drafted answers, evidence, and the nested FAQ result all leave ATLAS
before payment. That breaks the product boundary because the free page would
hold the paid deliverable.

This slice builds the thinnest end-to-end ATLAS-owned gate: generate the real
report, persist the full artifact server-side by `request_id`, return only a
draft/evidence-free top-5 snapshot to the caller, and release the full artifact
only after a privileged paid flag is set. The diff is expected to exceed the
400 LOC soft cap because projection, storage, route gating, migration, host
wiring, alternate-door closure, and negative tests must land together or the
gate is not real.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating

Slice phase: Vertical slice

1. Add a `DeflectionSnapshot` projection with configurable top-N, default 5,
   that includes only summary counts and ranked customer questions.
2. Add a deflection report artifact store contract plus a Postgres
   implementation that keeps the full artifact and paid flag in ATLAS.
3. Gate the hosted `/content-ops/execute` response for `faq_deflection_report`
   so the browser gets only the snapshot and `request_id`.
4. Add snapshot, full-artifact, and mark-paid routes; keep snapshot/artifact
   tenant-scoped, but put mark-paid behind a trusted dependency hook.
5. Wire the host router to the Postgres store and add the migration.
6. Hide the FAQ markdown output from the hosted buyer-executable output list while
   keeping it available internally as the deflection report source generator.
7. Add focused tests proving drafts/evidence are stripped before payment and
   full release is fail-closed until paid.

### Files touched

| File | Purpose |
|---|---|
| `extracted_content_pipeline/faq_deflection_report.py` | Adds the snapshot projection contract and helpers. |
| `extracted_content_pipeline/deflection_report_access.py` | Adds store contract, in-memory test store, and Postgres adapter. |
| `extracted_content_pipeline/api/control_surfaces.py` | Persists full reports, returns snapshots, and exposes gated release routes. |
| `atlas_brain/_content_ops_services.py` | Lets the host hide the FAQ markdown output as customer-executable while retaining it internally for deflection reports. |
| `atlas_brain/api/__init__.py` | Wires the host Content Ops router to the Postgres deflection report store. |
| `atlas_brain/storage/migrations/328_content_ops_deflection_reports.sql` | Adds the ATLAS-owned report artifact/paid-flag table. |
| `tests/test_content_ops_deflection_report.py` | Proves snapshot projection strips paid content. |
| `tests/test_atlas_content_ops_execution_services.py` | Proves hosted output hiding keeps the deflection report runnable. |
| `tests/test_atlas_content_ops_generated_assets_api.py` | Proves the hosted mark-paid route uses the platform-admin gate. |
| `tests/test_extracted_content_control_surface_api.py` | Proves execute redaction and paid-gated route behavior. |
| `tests/test_extracted_content_ops_live_execute_harness.py` | Updates hosted deflection route proofs to use the paid-gated release path. |
| `plans/PR-FAQ-Deflection-Paid-Gate.md` | Documents the slice contract and verification. |

## Mechanism

`build_deflection_snapshot(artifact, top_n=5)` reads only
`artifact.summary` and ranked `faq_result.items`. Each top question contains
rank, question, weighted frequency, and customer wording. It intentionally does
not carry answer text, steps, evidence quotes, source IDs, term mappings, or
the full Markdown body.

The hosted execute route keeps generating the full report internally. For each
completed `faq_deflection_report` step it:

1. builds the snapshot,
2. saves `{account_id, request_id, snapshot, artifact, paid=false}` to the
   configured store,
3. replaces the step result with `{request_id, snapshot, full_report: locked}`.

The same router exposes:

- `GET /content-ops/deflection-reports/{request_id}/snapshot`
- `GET /content-ops/deflection-reports/{request_id}/artifact`
- `POST /content-ops/deflection-reports/{request_id}/paid`

Snapshot reads are available to the authenticated tenant regardless of paid
state. Full artifact reads return 403 until the paid flag is set. The paid
marker route is mounted behind a trusted dependency; in the Atlas host that is
the existing platform-admin usage gate, so a tenant buyer cannot self-unlock a
report.

The host execution bundle uses `expose_faq_markdown_output=False` for the
browser-mounted Content Ops router. The FAQ markdown output remains available inside
`FAQDeflectionReportService`, but it is no longer advertised or runnable as an
alternate customer-facing output on that hosted boundary.

## Intentional

- This slice chooses a privileged mark-paid seam instead of a public buyer
  route. Stripe checkout initiation remains portfolio-owned and Stripe webhook
  mapping requires a dedicated requestId/payment mapping slice.
- The library-level `execute_content_ops_from_mapping` still returns full
  artifacts for tests and internal callers. The hosted route is the browser
  boundary and applies the paid gate.
- The extracted router still supports the FAQ markdown output for internal/test mounts.
  The Atlas host hides it from the buyer-facing service bundle because it
  carries the drafted answers and evidence this paywall protects.
- If the hosted router has no deflection report store configured, it fails
  closed for `faq_deflection_report` instead of returning the paid artifact.

## Deferred

- Future production-hardening slice: Stripe webhook -> ATLAS paid-flag flip
  with requestId/payment mapping and webhook idempotency.
- Future product-polish slice: portfolio/Intel UI copy and subscription
  writeback messaging for the $500/mo ongoing offer.
- Parked hardening considered: none.

## Verification

- Command: python -m py_compile extracted_content_pipeline/faq_deflection_report.py extracted_content_pipeline/deflection_report_access.py extracted_content_pipeline/api/control_surfaces.py atlas_brain/api/__init__.py atlas_brain/_content_ops_services.py tests/test_content_ops_deflection_report.py tests/test_extracted_content_control_surface_api.py tests/test_extracted_content_ops_live_execute_harness.py tests/test_atlas_content_ops_generated_assets_api.py tests/test_atlas_content_ops_execution_services.py
  - Result: passed.
- Command: pytest tests/test_extracted_content_control_surface_api.py::test_deflection_report_paid_route_uses_trusted_dependency tests/test_atlas_content_ops_generated_assets_api.py::test_content_ops_deflection_paid_route_uses_operator_gate tests/test_atlas_content_ops_execution_services.py::test_faq_markdown_can_be_hidden_while_deflection_report_runs -q
  - Result: 3 passed, 1 warning.
- Command: pytest tests/test_content_ops_deflection_report.py tests/test_extracted_content_control_surface_api.py tests/test_extracted_content_ops_live_execute_harness.py tests/test_atlas_content_ops_execution_services.py tests/test_atlas_content_ops_generated_assets_api.py -q
  - Result: 187 passed, 1 skipped, 1 warning.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Result: passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Result: passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Result: passed.
- Command: bash scripts/check_ascii_python.sh
  - Result: passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - Result: 2776 passed, 10 skipped, 1 warning.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-paid-gate.md
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Snapshot + store contract/adapters | 356 |
| Hosted route gating | 175 |
| Host wiring + migration + hosted output hiding | 46 |
| Tests | 544 |
| Plan doc | 142 |
| **Total** | **1263** |
