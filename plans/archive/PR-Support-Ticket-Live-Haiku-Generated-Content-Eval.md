# Support Ticket Live Haiku Generated Content Eval

## Why this slice exists

PR #956 wired the deterministic generated-content evaluator into the live
Content Ops smoke, but the new flag has only been proven with focused tests.
The next validation step is a recorded live Haiku run that generates real
support-ticket-backed landing-page and blog-post drafts, exports the saved
drafts, and runs the evaluator in the same smoke command.

This keeps us in the support-ticket provider lane and proves the current
end-to-end generation path before changing the generated-copy contract.

This slice is slightly over the normal diff budget because the live validation
found two false-green classes in the same path: unsupported uploaded-ticket
timeframes and unsupported future-impact percentages. Keeping the validation
report, evaluator guards, and focused regressions together makes the recorded
outcome reproducible.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Run live landing-page generation with the packaged support-ticket CSV,
   saved-draft export, and generated-content evaluation enabled.
2. Run live blog-post generation with the same support-ticket CSV,
   saved-draft export, and generated-content evaluation enabled.
3. Force OpenRouter to the Claude Haiku family for this validation run.
4. Close the unsupported temporal and percentage-claim gaps exposed by the live
   blog output.
5. Archive the observed ids, evaluator results, and any follow-up findings in a
   validation document.

### Files touched

- `plans/PR-Support-Ticket-Live-Haiku-Generated-Content-Eval.md`
- `scripts/smoke_content_ops_live_generation.py`
- `scripts/evaluate_support_ticket_generated_content.py`
- `tests/test_smoke_content_ops_live_generation.py`
- `tests/test_evaluate_support_ticket_generated_content.py`
- `atlas_brain/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/skills/digest/blog_post_generation.md`
- `docs/extraction/validation/support_ticket_live_haiku_generated_content_eval_2026-05-25.md`

## Mechanism

The smoke command loads the Atlas DB/OpenRouter env files plus a temporary
Haiku override env file. Each run uses `--support-ticket-csv` so the provider
packages the example ticket rows, `--export-saved-draft` so the exact saved row
can be inspected, and `--evaluate-generated-content` so the new evaluator runs
against the in-memory export before the command exits.

The validation doc records the command shape and summarized results. Large or
environment-specific JSON exports stay in ignored local `tmp/`.

The live blog runs exposed truthfulness gaps: an undated CSV generated copy that
said "Between May 2026 and the present" even though the source period was only
`Uploaded support tickets`, plus unsupported future-impact percentages such as
`20-40%`, `70%`, `75%`, and `30-50%`. The fix removes `report_date` from
undated support-ticket blog blueprint context, updates the blog-generation
prompt to forbid invented calendar windows and predictive ROI math for
uploaded-ticket inputs, and adds deterministic evaluator checks for unsupported
calendar-window and percentage language.

The same prompt update also names the citable-section shape required by the
blog GEO readiness gate, after one Haiku run generated usable uploaded-ticket
language but missed `geo_citable_section_structure_missing`.

## Intentional

- The prompt change is limited to uploaded-ticket truthfulness and the existing
  GEO citable-section gate shape. It does not change blog SEO fields or FAQ
  ownership.
- No FAQ generator changes. FAQ generation remains owned by the parallel FAQ
  session.
- No Sonnet spend for this validation; the temporary env override points the
  OpenRouter reasoning route at Haiku.

## Deferred

- Future PR: use generated-content evaluator failures as blog repair feedback,
  or otherwise tighten support-ticket blog generation until the live Haiku blog
  path passes without unsupported future-impact percentages.
- Future PR: align blog save-time quality gating with exported
  `geo_readiness`, or make the live smoke fail on exported readiness when
  quality gates are enabled. The latest failed blog smoke still saved a draft
  whose exported `geo_readiness.status` was `needs_review`.
- Parked hardening: none added by this slice.

## Verification

- Live landing-page smoke command recorded in
  `docs/extraction/validation/support_ticket_live_haiku_generated_content_eval_2026-05-25.md`
  - Passed; saved draft `cad10d59-d62a-4c04-a111-7036cfd73e0b`; SEO/AEO ready; GEO ready; generated-content evaluation passed.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_uses_csv_counts tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_uses_date_window_when_dates_validate -q`
  - 14 passed.
- `python -m pytest tests/test_smoke_content_ops_live_generation.py tests/test_evaluate_support_ticket_generated_content.py -q`
  - 51 passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .`
  - OK, 114 matching tests enrolled.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
  - 18 passed.
- Py compile for the changed Python scripts and tests
  - Passed.
- Generated-content evaluator against the live landing-page export
  - Passed.
- Generated-content evaluator against the earlier live blog export
  - Failed as expected after the new percentage guard, catching unsupported `20-40%`.
- Live blog-post smoke command recorded in
  `docs/extraction/validation/support_ticket_live_haiku_generated_content_eval_2026-05-25.md`
  - Failed by generated-content evaluation as expected, catching unsupported `30-50%` claims. This proves the smoke no longer gives a false green for unsupported impact percentages.
- `bash scripts/local_pr_review.sh --allow-dirty`
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan and validation report | ~220 |
| Evaluator guardrails and tests | ~370 |
| Smoke/prompt alignment | ~80 |
| **Total** | **~670** |

Over budget by design: the validation report plus focused evaluator regressions
are the bulk of the diff, and splitting them would separate the live failure
evidence from the guardrails that now catch it.
