# PR-Gate-A-Report-Live-Coverage

## Why this slice exists

Gate A output-quality validation has converged for `landing_page`,
`blog_post`, `sales_brief`, and `email_campaign`. The remaining gap is the
`report` generator: it is wired through `ContentOpsExecutionServices`,
`GenerationPlanStep`, `ReportGenerationService`, `PostgresReportRepository`,
review status updates, and `export_report_drafts(...)`, but it has not been
exercised end-to-end in the live Gate A harness.

This slice exists to produce the missing proof run: request only `report`
through the same service builder/database/model route used by the prior Gate A
proofs, persist the generated report draft, review it, export the exact saved
row, and commit the generated artifacts for reviewer/operator inspection.

This is validation, not a new product feature. The code change should stay
small because the generator and host service wiring already exist; the missing
piece is harness coverage and the proof artifact.

This PR exceeds the 400 LOC soft cap because the raw live-proof JSON artifacts
and the archived #1394 plan doc are part of the deliverable. The executable
harness/test diff remains narrow and focused on report selection, binding,
review, and export.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Functional validation

1. Extend `scripts/smoke_content_ops_gate_a_live_quality.py` so `--outputs report`
   is an accepted Gate A output.
2. Review and export saved `report` ids through the existing generated-assets
   status path and `export_report_drafts(PostgresReportRepository(...))`.
3. Add focused harness tests for report selection, explicit opportunity
   binding, review/export, and single-report no-variant handling.
4. Archive the already-merged #1394 plan doc as same-branch teardown
   housekeeping.
5. Run the live smoke against only `report` with local Ollama fallback disabled,
   then commit `summary.json`, `execution-result.json`, `opportunity-import.json`,
   `review-results.json`, `export-report.json`, and a markdown proof report
   pointing at the generated sample.
6. Do not modify report prompt behavior, report service semantics, or other
   generators unless the live proof exposes a correctness blocker that prevents
   report generation from functioning.

### Files touched

- `docs/extraction/validation/content_ops_gate_a_report_live_2026-06-08.md`
- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/export-report.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/opportunity-import.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/summary.json`
- `plans/INDEX.md`
- `plans/PR-Gate-A-Report-Live-Coverage.md`
- `plans/archive/PR-Gate-A-Email-Campaign-Input-Fit-Proof.md`
- `scripts/smoke_content_ops_gate_a_live_quality.py`
- `tests/test_smoke_content_ops_gate_a_live_quality.py`

## Mechanism

The existing harness already builds a Content Ops payload, resolves the host
runtime dependencies, calls the real `execute_content_ops_from_mapping(...)`
path, records saved ids, reviews the saved rows through generated-assets status
helpers, exports rows through product-specific repository adapters, and checks
that every selected output persisted.

The report extension is deliberately mechanical:

```python
DEFAULT_OUTPUTS = (..., "report")

if "report" in selected_outputs:
    inputs["opportunity_id"] = first_source_row_target_id

if "report" in selected_outputs:
    assert inputs["opportunity_id"] in imported_target_ids
    filters["target_id"] = inputs["opportunity_id"]

if report_ids:
    export = await export_report_drafts(
        PostgresReportRepository(pool),
        scope=scope,
        status=None,
        target_mode=target_mode,
        limit=max(100, len(report_ids)),
    )
    exports["report"] = _filter_saved_draft_export_rows(export.as_dict(), report_ids)
```

The proof command uses `--outputs report` so the run isolates the one unproven
generator. Report preview requires `inputs.opportunity_id`, while report
execution reads from `campaign_opportunities` using `filters`. The harness
therefore imports the support-ticket rows into the existing opportunity table,
sets `inputs.opportunity_id` from the source row, verifies that exact id is in
the import result, and adds `filters.target_id` before execution. If the
requested id is missing, no ids are returned, or multiple imported ids appear
without an explicit request, the run fails closed instead of guessing.

The existing `saved_ids_by_output(...)`, `review_saved_ids(...)`, and
`_execution_errors(...)` helpers then treat `report` as a single-run output:
the run fails if the step is missing, status is not completed, no saved id is
returned, review misses the row, or exact export filtering cannot find the
saved report id.

The live artifact report records the command, selected account, model route,
Ollama fallback-disabled setting, saved report id, title/summary/section counts,
reference ids, and the generated sample location. It does not claim the report
is product-accepted beyond the harness passing.

## Intentional

- Run only `report`. The other Gate A generators already have committed proof
  artifacts, and running them again would blur the signal for the one remaining
  gap.
- Reuse `ReportGenerationService`, `PostgresReportRepository`, review status
  updates, and `export_report_drafts(...)`. This PR does not rebuild service
  wiring that is already merged.
- Keep report prompt/content changes out of scope unless the live run exposes a
  data-truthfulness blocker. The #1394 shared grounding contract already routes
  the report prompt through the common no-fabrication seam.
- Keep committed proof artifacts in the repo so reviewer and operator can
  inspect the exact generated sample.
- Fold the #1394 plan archive into this branch as teardown housekeeping because
  #1394 was merged immediately before this slice.

## Deferred

The Gate A hardening items already parked in `HARDENING.md` for landing-page
variant distinctness, blog prose style, and second-person brand voice remain
parked. They do not block the report generator from being exercised
end-to-end and are not report-specific.

Parked hardening: none.

## Verification

- Completed before push:
  - `pytest tests/test_smoke_content_ops_gate_a_live_quality.py`
    - Result: `23 passed in 0.21s`.
  - `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_gate_a_live_quality.py --account-id 7d9c8e6a-5f42-4c91-9237-1394a0f2b681 --user-id 11111111-1111-4111-8111-111111111111 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --output-dir tmp/content_ops_gate_a_report_live_20260608 --outputs report --variant-count 3 --quality-repair-attempts 1 --max-cost-usd 20.00 --json`
    - Result: passed; 1 report saved/reviewed/exported.
  - Artifact JSON validation for
    `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/*.json`
    - Result: passed; 5 JSON files parsed.
  - Export route/sample check
    - Result: 1 row, model `anthropic/claude-sonnet-4-5`, reference
      `saas-demo-001`, 3 sections.
  - `bash scripts/run_extracted_pipeline_checks.sh`
    - Result: `3475 passed, 10 skipped, 1 warning in 57.26s`; wrapper completed all extracted checks.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/content_ops_gate_a_report_live_2026-06-08.md` | 166 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/execution-result.json` | 79 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/export-report.json` | 84 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/opportunity-import.json` | 46 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/review-results.json` | 11 |
| `docs/extraction/validation/fixtures/content_ops_gate_a_report_live_20260608/summary.json` | 106 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Gate-A-Report-Live-Coverage.md` | 168 |
| `plans/archive/PR-Gate-A-Email-Campaign-Input-Fit-Proof.md` | 0 |
| `scripts/smoke_content_ops_gate_a_live_quality.py` | 90 |
| `tests/test_smoke_content_ops_gate_a_live_quality.py` | 194 |
| **Total** | **947** |
