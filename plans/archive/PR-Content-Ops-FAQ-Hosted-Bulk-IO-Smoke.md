# PR-Content-Ops-FAQ-Hosted-Bulk-IO-Smoke

## Why this slice exists

PR-Content-Ops-FAQ-Vocab-Gap-IO-Smoke proved the hosted FAQ route accepts
vocabulary-gap inputs and returns the expected mapped output. The next testing
gap is confidence that the hosted input/output path also handles a larger
uploaded source-material payload, not only a single-ticket example.

This slice adds a thin 1,000-row hosted execute smoke and closes the failures
surfaced while testing the real FAQ flow.

Initial testing surfaced a real hosted input-path failure: direct
inputs.source_material arrays were capped at 50 items because the generic
input-shape validator reused `_MAX_INPUT_KEYS` for every array. The ingestion
API validator was then raised to 1,000 rows because the product claim we need to
prove is 500-1,000 ticket uploads, not only a 500-row ceiling.

Follow-on testing surfaced three more real failures in this lane:

1. The local DB lifecycle smoke could not run because the generated-asset table
   was missing; applying migrations exposed that the migration runner wrapped
   CREATE INDEX CONCURRENTLY in a transaction.
2. The 1,000-row CFPB scale artifact passed checks but still emitted weak
   customer-wording questions such as all-caps fragments and
   `my complaint is about...` phrasing.
3. A live CFPB smoke grouped unrelated call-behavior evidence with a billing
   dispute.

Those are fixed here because they directly affect the tested FAQ product proof.
The resulting PR is over the 400 LOC soft cap because splitting would leave the
current proof in a knowingly misleading state: hosted scale, DB lifecycle, and
output quality all failed in the same real-flow test pass.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Add a hosted `/content-ops/execute` FAQ test with 1,000 inline support-ticket
   rows.
2. Assert the route completes, the FAQ result reports all 1,000 ticket sources,
   output checks pass, and the generated FAQ item keeps source coverage.
3. Keep the test deterministic by using generated in-test source rows and the
   real `TicketFAQMarkdownService`.
4. Allow direct inputs.source_material lists and recognized one-level
   source_material bundle lists to use the existing `_MAX_INGESTION_ROWS` cap
   and pin the 1,001-row rejection path.
5. Allow packaged extracted migrations containing CONCURRENTLY to run outside
   the per-migration transaction.
6. Reject weak CFPB customer-question candidates and split contact/call
   complaints away from billing disputes.

### Files touched

| File | Change |
|---|---|
| `extracted_content_pipeline/api/control_surfaces.py` | Raises direct and recognized one-level bundle source_material row-list caps to 1,000 while keeping descendant arrays at 50. |
| `extracted_content_pipeline/storage/migration_runner.py` | Runs CONCURRENTLY migrations outside the transaction wrapper. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Rejects weak customer questions, aligns summaries with the question source row, and separates communication/contact issues. |
| `plans/PR-Content-Ops-FAQ-Hosted-Bulk-IO-Smoke.md` | Plan doc for this bulk hosted IO smoke slice. |
| `tests/test_extracted_content_control_surface_api.py` | Pins the 1,001-row hosted source-material and ingestion rejection boundary plus scoped 50/1,000 array-cap behavior. |
| `tests/test_extracted_content_ops_live_execute_harness.py` | Adds the hosted 1,000-row FAQ route test. |
| `tests/test_extracted_content_pipeline_migration_runner.py` | Covers non-transactional CONCURRENTLY migration execution. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers weak-question rejection and contact/billing split behavior. |

## Mechanism

The existing live execute harness calls the mounted `/content-ops/execute`
endpoint directly with host-injected services. This slice reuses that shape and
injects `ContentOpsExecutionServices(faq_markdown=TicketFAQMarkdownService())`.

The test builds 1,000 support-ticket rows in memory:

```python
source_material = [
    {"ticket_id": f"ticket-bulk-{index}", ...}
    for index in range(1000)
]
```

The route should return a completed FAQ step whose result reports
`ticket_source_count == 1000`, passing output checks, and a first FAQ item whose
`source_ids` include the generated ticket IDs.

The API validator keeps the existing 50-item cap for ordinary input arrays, but
uses `_MAX_INGESTION_ROWS` when validating the direct source_material array or
a recognized one-level source_material bundle list:

```python
max_items = _input_array_max_items(path)
```

List descendants append a synthetic path segment before recursion, so nested
arrays under source_material do not inherit the 1,000-row row-list budget.

The migration runner detects SQL that cannot run inside a transaction, such as
CREATE INDEX CONCURRENTLY, executes that migration outside the transaction
wrapper, and then records the applied migration.

The FAQ generator keeps using source-policy questions when customer wording is
not publishable. This slice expands that filter to reject all-caps fragments and
`my complaint is about...` boilerplate, and it adds a communication/contact
intent so call-behavior complaints do not collapse into billing disputes.

## Intentional

- No dispatcher or UI behavior changes.
- The hosted request validator now accepts direct source_material arrays and
  recognized one-level source_material bundle arrays up to 1,000 rows because
  that is the confidence threshold being tested.
- The generator behavior changes are limited to the weak output shapes surfaced
  by the 1,000-row and live CFPB smokes.
- The source rows are generated inside the test instead of checked into a new
  fixture file.
- The real DB migrations were applied locally as verification. The migration
  state itself is not part of the repo diff.

## Deferred

- A browser file-upload test remains a future slice; this PR proves the hosted
  execute route body, the CLI scale runner, and the DB lifecycle smoke.
- Mixed-source hosted payload coverage remains separate; this slice focuses on
  bulk row count through the route.
- Current `HARDENING.md` entries were scanned; no root hardening items are
  parked for this FAQ lane after the fixes in this slice.

## Verification

- `python -m pytest tests/test_extracted_content_ops_live_execute_harness.py::test_live_execute_route_handles_bulk_faq_source_material -q` - failed initially with HTTP 422 `inputs arrays are too large`, then passed after the source-material validation cap fix.
- `python -m pytest tests/test_extracted_content_ops_live_execute_harness.py::test_live_execute_route_handles_bulk_faq_source_material tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_rejects_source_material_over_1000_as_422 tests/test_extracted_content_control_surface_api.py::test_ingestion_inspect_route_rejects_oversized_rows -q` - passed, 3 tests.
- `python -m pytest tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_rejects_source_material_over_1000_as_422 tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_keeps_50_cap_for_non_source_material_arrays tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_accepts_source_material_bundle_1000_rows tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_keeps_50_cap_for_nested_source_material_arrays -q` - passed, 4 tests.
- `python -m pytest tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_rejects_all_caps_customer_question tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_rejects_complaint_about_as_customer_question tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_summarizes_customer_question_source_row -q` - passed, 3 tests.
- `python -m pytest tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_summarizes_customer_question_source_row tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_separates_contact_complaints_from_billing_disputes -q` - passed, 2 tests.
- `python -m pytest tests/test_extracted_content_pipeline_migration_runner.py::test_apply_content_pipeline_migrations_runs_concurrent_index_outside_transaction tests/test_extracted_content_pipeline_migration_runner.py::test_apply_content_pipeline_migrations_applies_pending_files -q` - passed, 2 tests.
- `python -m pytest tests/test_smoke_content_ops_faq_scale_run.py::test_faq_scale_smoke_writes_standard_artifacts -q` - passed, 4 tests.
- `python scripts/run_extracted_content_pipeline_migrations.py --json` with the resolved local DB URL - initially failed with CREATE INDEX CONCURRENTLY cannot run inside a transaction block, then passed after the migration-runner fix: 20 applied, 8 skipped.
- `python scripts/smoke_content_ops_faq_lifecycle.py --account-id acct-faq-db-smoke-20260523 --user-id user-faq-db-smoke --title "FAQ DB Lifecycle Smoke 2026-05-23" --output-result tmp/faq_lifecycle_smoke_20260523_after_migrations.json --json` - passed, `ok=true`, 4 source rows, 1 saved FAQ, draft export and published export present.
- `python scripts/smoke_content_ops_faq_scale_run.py tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl --source-format jsonl --artifact-dir tmp/content_ops_faq_testing_20260523_scale1000_after_contact_fix --title 'CFPB 1,000 Row FAQ Test After Contact Fix 2026-05-23' --max-items 12 --max-evidence-per-item 5 --max-text-chars 1200 --default-field company_name=CFPB --default-field contact_email=cfpb-public-archive@example.invalid` - passed, 1,000/1,000 usable rows, 12 generated, 0 failed output checks, 0 warnings.
- `python scripts/smoke_content_ops_cfpb_faq_markdown.py --limit 10 --max-rows-scanned 50 --title 'CFPB Live FAQ Smoke After Contact Fix 2026-05-23' --output-source-rows tmp/content_ops_faq_testing_20260523_live_cfpb_after_contact_fix/cfpb_sources.jsonl --output-markdown tmp/content_ops_faq_testing_20260523_live_cfpb_after_contact_fix/faq.md --json` - passed, 10/10 usable live CFPB rows, 4 generated FAQ items, 0 errors.
- `python -m pytest tests/test_extracted_content_ops_live_execute_harness.py -q` - passed, 4 tests.
- `python -m pytest tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_rejects_invalid_faq_vocabulary_rules_as_400 tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_rejects_source_material_over_1000_as_422 -q` - passed, 2 tests.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed, 0 findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `git diff --check` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed, 1783 tests, 1 skipped, 1 existing `torch`/`pynvml` warning.
- `bash scripts/local_pr_review.sh origin/main --allow-dirty` - pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 169 |
| API validation boundary | 45 |
| Hosted bulk route and boundary tests | 183 |
| Migration runner and tests | 59 |
| FAQ generator quality fixes and tests | 192 |
| **Total** | 648 |

This exceeds the 400 LOC soft cap because the same real-flow test pass surfaced
inseparable hosted-scale, DB-lifecycle, and output-quality failures. Splitting
the fixes would leave the open PR proving a 1,000-row FAQ path while known
database and generated-output failures remained unresolved. Actual diff:
+612 / -25.
