# PR-Deflection-Email-Action-Scorecard

## Why this slice exists

Issue #1612's QA harness is supposed to prove the paid deliverable surfaces
agree with the persisted `deflection.v1` model. #1764 made the delivery email
actionable by rendering compact `priority_fix_queue` and `drafted_resolutions`
rows, but the shared scorecard still treats email as a counts-only surface.

Root cause: `DEFAULT_DEFLECTION_FULL_REPORT_SURFACE_CAPS` had row caps for the
result page and PDF, but no email caps. The deterministic harness could
therefore stay green even if the delivery email stopped rendering its action
rows, because there was no required `email.displayed_rows.*` observation to
check. The first implementation also treated email displayed rows as
`min(section_total, 3)`, which was still a symptom fix: the delivery email
filters action rows by item validity, `result_page_limit`, drafted-question
deduplication, and `Draft ready` priority exclusion before rendering.

This PR fixes the root in the shared scorecard contract by making email action
rows an observed, capped surface, moving email action-row selection into one
shared extracted-package helper, and feeding actual rendered delivery-email row
counts into the scorecard. It turns the #1764 behavior into a reusable,
load-bearing QA invariant instead of a duplicated renderer mirror.

This exceeds the 400 LOC soft cap because the re-review root fix is indivisible:
landing only the scorecard contract duplicated renderer rules, while landing
only the shared selector left no live consumer. The coherent fix needs the
shared selector, renderer rewiring, rendered-email observer, and tests together.

## Scope (this PR)

Ownership lane: issue-1612/deflection-full-report-delivery-actionability
Slice phase: Vertical slice

1. Add email action-section row caps to the shared full-report QA scorecard
   surface-cap defaults.
2. Move delivery-email action row selection into a shared lower-layer helper
   consumed by both the scorecard and the delivery renderer.
3. Add a rendered delivery-email observer that feeds sanitized action row
   counts into the scorecard.
4. Add focused scorecard/harness and delivery-renderer tests proving email
   action rows are required, capped, fail closed when mismatched, and reflect
   the actually rendered email.
5. Max files: 5.

### Review Contract

Acceptance criteria:
- The shared deterministic full-report QA harness requires the `email` surface
  to observe `priority_fix_queue` and `drafted_resolutions` displayed rows when
  the model contains those sections.
- Email expected rows mirror the delivery renderer's action selection:
  section-level `result_page_limit`, three-row email cap, invalid-row skips,
  drafted-question removal from priority rows, and `Draft ready` priority
  exclusion.
- The delivery renderer consumes the same shared action-row selector as the
  scorecard; it does not carry a second copy of the selection loop.
- A rendered delivery email can be observed as sanitized row counts and those
  counts pass/fail the scorecard.
- A missing email action-row observation fails the harness instead of silently
  passing counts-only email coverage.
- Existing PDF/result-page/evidence-export assertions remain unchanged.

Affected surfaces:
- `atlas_brain/content_ops_deflection_delivery.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_report.py`

Risk areas:
- QA false-green risk: the harness must fail on missing email action rows
  without requiring raw evidence, source IDs, or email-body content.
- Cross-surface contract drift: email rows should be bounded independently from
  PDF rows, should not inherit the larger PDF cap, and should not use raw
  section totals when the email renderer would display fewer rows.
- Renderer/scorecard drift: action-row selection must live in one shared helper
  so the scorecard does not duplicate the renderer's private rules.

Reviewer rules triggered: R1, R2, R5, R8, R10, R14.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Email-Action-Scorecard.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

Add email-specific action-section caps to
`DEFAULT_DEFLECTION_FULL_REPORT_SURFACE_CAPS`:

- `priority_fix_queue`: 3
- `drafted_resolutions`: 3

The scorecard and delivery renderer now share an extracted-package email
action-row selector, which selects the rows eligible for compact delivery email
rendering. Result page and PDF sections keep the
existing aggregate `min(total, cap)` scorecard behavior. Email action sections
use the shared selected rows: malformed action rows are skipped, each section's
`result_page_limit` can lower the three-row email cap, drafted-resolution
questions are removed from priority rows, and `Draft ready` priority rows are
excluded.

The delivery email surface observer parses the rendered text email into
sanitized `displayed_rows` counts. The delivery test renders the actual
email through the worker, feeds those counts into the scorecard, and proves a
wrong rendered row count fails.

## Intentional

- Delivery email output is unchanged; the renderer now consumes the shared row
  selector instead of selecting rows privately.
- No email checks for `top_unresolved_repeats` or
  `already_covered_still_recurring`; those sections are not `email_summary`
  surfaces today.
- Email body observation returns sanitized counts only; source IDs, quotes, and
  email text stay out of scorecards.
- Codex P2 on the first push was valid. This PR fixes it by deriving email
  expected rows from renderer-equivalent item selection instead of raw section
  counts.
- Human re-review MAJOR was valid. This update removes the duplicated
  renderer-rule mirror and proves the scorecard against actual rendered email
  counts.

## Deferred

- Hosted/live runner wiring for real inbox delivery remains deferred. This PR
  adds the reusable rendered-email observation helper and proves it against the
  worker-rendered email in tests.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_full_report_qa_scorecard_checks_action_section_caps tests/test_content_ops_deflection_report.py::test_deflection_full_report_qa_scorecard_mirrors_email_action_limits tests/test_atlas_content_ops_deflection_delivery.py::test_delivery_worker_renders_model_backed_email_summary -q -- 3 passed.
- Command: python -m pytest tests/test_content_ops_deflection_report.py tests/test_atlas_content_ops_deflection_delivery.py -q -- 167 passed.
- Command: python -m py_compile extracted_content_pipeline/faq_deflection_report.py atlas_brain/content_ops_deflection_delivery.py tests/test_content_ops_deflection_report.py tests/test_atlas_content_ops_deflection_delivery.py -- passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- Command: bash scripts/check_ascii_python.sh -- passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr_body_deflection_email_action_scorecard.md -- failed before push because the plan missed R10 for a checker/gate change; fixed in this commit.
- Pending before push: bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr_body_deflection_email_action_scorecard.md -- rerun via push wrapper.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/content_ops_deflection_delivery.py` | 105 |
| `extracted_content_pipeline/faq_deflection_report.py` | 171 |
| `plans/PR-Deflection-Email-Action-Scorecard.md` | 156 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 44 |
| `tests/test_content_ops_deflection_report.py` | 126 |
| **Total** | **602** |
