# PR-Deflection-Safe-Label-Disambiguation

## Why this slice exists

`HARDENING.md` still tracks `Safe-vocabulary representative label collisions
render duplicate FAQ headings`.

The upstream grouping is now correct: distinct topic-degraded support-ticket
subclusters stay separate, and the renderer only emits controlled-vocabulary
representative labels. The remaining defect is presentation-level label
collision. Two truthful groups can choose the same safe representative heading,
which makes the paid report look bloated even though the evidence groups should
not be merged.

Root cause: selected FAQ items do not have a collision-resolution pass after
question-label resolution. A downstream symptom fix would merge same-heading
groups or suppress one item, but that would undo the count/evidence truth fixed
by the prior slices. This slice fixes the root at the renderer boundary by
disambiguating colliding safe source-policy headings while preserving groups.

The final diff is over the 400 LOC target because review found two dependent
surface gaps in this same path: disambiguated draft answers must reference the
new heading, and service-generated drafts must preserve renderer warnings.
Splitting either fix would knowingly ship an inconsistent or silent
duplicate-heading path.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Production hardening

1. Add a deterministic post-label disambiguation pass for selected
   source-policy FAQ items whose rendered questions collide.
2. Use only already-safe representative vocabulary from each item's evidence
   group as the disambiguator; never publish raw source titles or raw ticket
   text.
3. Preserve grouping, item count, source IDs, evidence quotes, and sort order.
4. Emit a diagnostic warning when duplicate source-policy headings remain
   because no safe per-group disambiguator exists.
5. Remove the closed safe-vocabulary collision item from `HARDENING.md`.

### Files touched

- `HARDENING.md`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Safe-Label-Disambiguation.md`
- `tests/test_extracted_ticket_faq_markdown.py`

### Review Contract

- Acceptance criteria:
  - [ ] Distinct topic-degraded support-ticket subclusters that initially
        resolve to the same safe source-policy question render distinct buyer
        headings when each group has an additional safe distinguishing token.
  - [ ] Disambiguation does not merge, drop, or reorder FAQ items.
  - [ ] Disambiguation does not change customer-wording questions.
  - [ ] Disambiguation does not publish raw source titles, raw evidence text,
        emails, phone-shaped values, redaction artifacts, or identifier-context
        long numbers.
  - [ ] If a collision cannot be safely disambiguated, the report emits a
        warning diagnostic and leaves the truthful duplicate headings intact.
- Affected surfaces: ticket FAQ Markdown question labels and warning diagnostics.
- Risk areas: buyer-facing report shape, privacy boundary, grouping truth,
  deterministic ordering, and support-ticket evidence accounting.
- Reviewer rules triggered: R1, R2, R3, R10, R13, R14.

## Mechanism

After selected groups become FAQ item dictionaries, the renderer groups only
`source_policy` items by normalized question text. Singleton labels pass through.

For a duplicate source-policy question, the renderer looks for safe
controlled-vocabulary tokens that appeared in that item's evidence group but do
not already appear in the question. If every item in the collision set gets a
distinct safe suffix, the heading becomes a deterministic variant such as
`What should I do about billing issue - invoice refund?`. The suffix comes from
documentation or structured taxonomy vocabulary already admitted by the
representative-label gate, not from raw ticket text. Draft answers are refreshed
when the heading changes so downstream JSON/search consumers see one question.

If any item in the collision set cannot receive a safe unique suffix, the
renderer leaves every item unchanged and adds a warning with the duplicate
question and affected source IDs. The service path merges those renderer
warnings with source-normalization warnings before returning or saving drafts.

## Intentional

- **No group merge.** Duplicate-looking headings are a label collision, not
  proof that the evidence groups are the same customer question.
- **No raw-text fallback.** A heading stays duplicate if the only available
  distinguishing text is unsafe customer/source text.
- **No customer-wording rewrite.** Customer wording already passed the
  buyer-facing privacy gate and should not be altered by a source-policy label
  disambiguator.
- **No embedding-booster work.** The operator stopped further embedding-booster
  buildout unless a new explicit enablement/calibration decision is made.

## Deferred

- Broader report/UI copy for explaining warning diagnostics if unresolved
  duplicate labels remain visible in a live paid-funnel artifact.

Parked hardening: none expected; this slice should close the
`Safe-vocabulary representative label collisions render duplicate FAQ headings`
entry.

## Verification

- Command passed: python -m py_compile for the touched Python files.
- Command passed: pytest focused duplicate-label, privacy, and
  representative-label regressions -- 13 passed.
- Command passed: pytest review-fix regressions -- 4 passed.
- Command passed: pytest tests/test_extracted_ticket_faq_markdown.py -q -- 265 passed.
- Command passed: bash scripts/check_ascii_python.sh.
- Command passed: git diff --check.
- Command passed: bash scripts/validate_extracted_content_pipeline.sh.
- Command passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline.
- Command passed: python scripts/audit_extracted_standalone.py --fail-on-debt.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh -- 4152 passed, 10 skipped, 1 warning.
- Pending before push: bash scripts/local_pr_review.sh with the PR body file.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 9 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 156 |
| `plans/PR-Deflection-Safe-Label-Disambiguation.md` | 130 |
| `tests/test_extracted_ticket_faq_markdown.py` | 196 |
| **Total** | **491** |
