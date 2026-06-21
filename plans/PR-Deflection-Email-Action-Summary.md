# PR-Deflection-Email-Action-Summary

## Why this slice exists

Issue #1612 split the paid deflection deliverable into multiple customer
surfaces: the hosted result page, curated PDF, delivery email, and complete
evidence export. Recent slices added action-section model contracts, rendered
those sections in the PDF, and made the QA scorecard verify the PDF rows. The
paid delivery email still summarizes only legacy top-line metrics, so a buyer
can receive the report email without seeing the first actionable work queue
that the report now exists to deliver.

Root cause: `atlas_brain.content_ops_deflection_delivery._DeliveryEmailSummary`
only extracts the `support_tax` email-summary section. The report model already
marks `priority_fix_queue` and `drafted_resolutions` as `email_summary`
surfaces, but the email renderer ignores those sections and therefore cannot
name the highest-value fixes or publish-ready drafts.

This PR fixes the root at the email projection boundary by allowlist-building a
small action summary from those two email-eligible sections. It does not widen
email to dump arbitrary action-section fields or raw evidence payloads.

Review follow-up root cause: the first projection allowed `_clean()` to
stringify nested action-row containers, treated draft-ready priority rows as a
separate email action, and treated explicit zero row limits as unset. This
update keeps the fix at the projection boundary by accepting only scalar text
for row copy, de-duping draft-ready questions from the priority block, and
honoring zero as "show no rows."

Diff budget note: this slice is over the 400 LOC soft cap because the review
fix touches a privacy/export boundary and ships the same-class negative probes
with the fix. The added size is concentrated in regression fixtures for
non-scalar required text, non-scalar optional text, draft de-dupe, and explicit
zero limits rather than a broader product surface.

## Scope (this PR)

Ownership lane: issue-1612/deflection-full-report-delivery-actionability
Slice phase: Vertical slice

1. Extend the paid report delivery email summary with an allowlisted action
   summary from `priority_fix_queue` and `drafted_resolutions`.
2. Render only compact, customer-safe fields in HTML/text: question, ticket
   count, estimated handling cost, status/recommended action for priority
   fixes, and question/count/cost for publish-ready drafts.
3. Keep raw evidence, source IDs, representative phrasing, and full backlog
   details out of the email body.
4. Preserve fallback behavior for legacy/future/malformed report models.
5. Max files: 3.

### Review Contract

Acceptance criteria:
- A `deflection.v1` report model with email-summary action sections produces an
  email containing "Next actions" and "Ready to publish" sections.
- Email action rows are capped to the section's `result_page_limit` when
  present, falling back to a small fixed cap only when the limit is missing;
  an explicit zero suppresses that email block.
- The email summary is allowlist construction; it must not include
  `top_evidence`, `source_ids`, `representative_phrasing`, raw evidence quotes,
  or complete evidence rows.
- Action row text is scalar-only: malformed object/list questions are skipped,
  and malformed object/list status or recommended-action values are omitted
  rather than stringified.
- Draft-ready questions already shown in "Ready to publish" do not consume
  "Next actions" slots.
- Missing or malformed action-section data does not fail delivery and does not
  synthesize misleading zero-action copy.
- Existing support-tax email metrics, PDF attachment behavior, result URL, and
  legacy fallback email copy remain unchanged.

Affected surfaces:
- `atlas_brain/content_ops_deflection_delivery.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`

Risk areas:
- Privacy/export boundary: action items carry raw evidence fields in the paid
  model, but delivery email must remain a bounded summary.
- Report actionability: email should point the buyer at what to fix first
  without becoming a second full report or complete ticket archive.

Reviewer rules triggered: R1, R2, R3, R5, R8, R13, R14.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `plans/PR-Deflection-Email-Action-Summary.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`

## Mechanism

Add a small internal action-summary type to the delivery email module. The
summary extractor continues to require valid `support_tax` metrics for the
model-backed email path, then scans `email_summary` sections for:

- `priority_fix_queue`: top N `items`, capped by `result_page_limit`;
- `drafted_resolutions`: top N `items`, capped by `result_page_limit`.

Each row is built from an explicit allowlist. The renderers add short HTML and
plain-text blocks after the key numbers. Malformed action rows are skipped; if
no safe rows remain, that block is omitted. Optional row copy is admitted only
when it is already scalar text, so nested evidence/source containers cannot
become email text through stringification. Drafted-resolution rows are collected
first and their questions/statuses are filtered out of the priority block, so a
publish-ready answer is not duplicated as a next action.

## Intentional

- No report-model producer changes. The sections already advertise
  `email_summary`; this PR consumes that contract in the delivery email only.
- No unresolved-repeats or already-covered recurring email block. Those
  sections are `web`/`pdf` today, not `email_summary`, and adding them to email
  would be a model-contract change.
- No raw customer wording/phrasing/snippet fields in email. The email should
  orient the buyer and link to the report, not become another high-risk content
  surface.

## Deferred

- Hosted buyer result-page action-section observation remains in
  `atlas-portfolio`.
- Richer email design/layout polish remains later product polish after the
  functional delivery surface is correct.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q -- 21 passed.
- Command: python -m py_compile atlas_brain/content_ops_deflection_delivery.py tests/test_atlas_content_ops_deflection_delivery.py -- passed.
- Pending before push: bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr_body_deflection_email_action_summary.md

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/content_ops_deflection_delivery.py` | 155 |
| `plans/PR-Deflection-Email-Action-Summary.md` | 140 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 234 |
| **Total** | **529** |
