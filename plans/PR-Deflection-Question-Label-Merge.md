# PR-Deflection-Representative-Question-Labels

## Why this slice exists

The paid-funnel deflection run on June 13, 2026 showed the report shape can
still balloon when many generated FAQ items render the same generic question
label. The local archive7 run artifact available under
`tmp/deflection_live_run/` has 11,923 uploaded ticket rows, 3,566 generated
items, and 980 distinct rendered question labels. The UI snapshot captured in
`results-snapshot.yml` shows a related hosted result with a 3,427-question
headline. Both artifacts point at the same product defect: topic-degraded
support-ticket subclusters can remain distinct, but still render buyer-facing
labels like "What should I do about technical support?"

The first implementation attempted to merge same-label topic-degraded groups.
Review rejected that approach. `#1460` deliberately splits those rows because
they are genuinely different repeated issues; collapsing them again lowers the
buyer-facing question count and can hide over-cap evidence. The defect is the
fallback label, not the grouping. This re-scoped slice fixes the root cause by
deriving a representative source-policy label from the subcluster's own
support-ticket content when no extractable customer question is available.

This remains complementary to #1515. The hybrid clustering RFC improves recall
inside question subclustering; it does not change how a kept subcluster is
labeled after the split.

The diff is over the 400 LOC target because the reviewed PR was re-scoped after
operator feedback and then received a privacy BLOCKER review. Keeping the
correction in one update lets the open PR replace the rejected merge approach,
add the root-cause representative-label path, handle the PII/coherence findings,
rewrite the affected regression tests, and record the hardening/reconciliation
trail against one corrected contract.
The final privacy review chose the zero-residual path for this new
representative label code: no customer-authored free text is allowed to become a
published representative heading.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Production hardening

1. `extracted_content_pipeline/ticket_faq_markdown.py`: preserve the existing
   topic-degraded subcluster grouping and remove the rejected same-label merge
   pass.
2. Carry the provided support-ticket cluster onto normalized evidence rows so
   the fallback can distinguish the reviewed topic-degraded path from older
   source-policy fallback paths.
3. When no customer wording can be extracted for a clustered support-ticket
   subcluster, derive a deterministic representative source-policy question
   only from known-safe documentation/glossary tokens or structured
   product/issue tokens that match the known taxonomy allowlist and also appear
   in the subcluster evidence.
4. Keep email/long-number filtering as defense-in-depth for safe vocabulary
   inputs, and require representative labels to have at least two repeated safe
   tokens before rendering.
5. Add focused tests proving distinct topic-degraded repeated issues render
   distinct labels, identical repeated content stays one subcluster, cap order
   remains insertion-stable, resolution-backed groups remain separate, PII-shaped
   labels fall back safely, clean-looking source titles do not bypass the safe
   vocabulary, allowlisted structured issue fields work without an explicit
   glossary, unlisted structured issue fields fall back generically, and
   low-signal repeated evidence tokens do not become headings.
6. Record the operator/Codex review reconciliation in the PR body so the live
   reconciliation gate can verify the open automated-review finding is handled.

### Files touched

- `HARDENING.md`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Question-Label-Merge.md`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_ticket_faq_markdown.py`

### Review Contract

- Acceptance criteria:
  - [ ] No same-label merge pass remains in the ticket FAQ renderer.
  - [ ] Topic-degraded rows render distinct source-policy questions when their
        evidence intersects known-safe documentation/glossary terms or known
        product/issue taxonomy terms.
  - [ ] Identical repeated topic-degraded content still renders as one
        subcluster with one item and all source IDs preserved.
  - [ ] Existing non-clustered policy fallback paths remain unchanged.
  - [ ] Tied topic-degraded subclusters preserve creation order through the
        `max_items` cap and overflow condensation.
  - [ ] Representative fallback never emits raw source title or raw evidence
        free text.
  - [ ] Representative fallback requires at least two repeated known-safe tokens
        before rendering; otherwise it falls back to the generic topic question.
  - [ ] Safe-vocabulary inputs containing email addresses or long numeric
        identifiers are ignored as defense-in-depth.
  - [ ] Support-ticket upload normalization preserves structured product/issue
        taxonomy fields, and the renderer validates those fields against the
        known taxonomy allowlist before using them as representative labels.
  - [ ] Resolution-backed same-label groups remain separate and keep
        `answer_evidence_status = resolution_evidence`.
- Affected surfaces: ticket FAQ Markdown grouping/rendering internals and
  focused deflection FAQ tests.
- Risk areas: buyer-facing report shape, drafted-answer preservation,
  deterministic ordering, golden report stability, and fallback-label precision.
- Reviewer rules triggered: R1, R2, R10, R13, R14.

## Mechanism

Normalized evidence rows now retain `support_ticket_cluster`. The grouping
pipeline still builds `subclustered_groups` exactly once: resolution-scoped
groups pass through, and topic-degraded groups are split by question similarity.
There is no post-subcluster label merge.

The shared question resolver still prefers extractable customer wording for the
existing path. If no customer wording is usable and the rows came from the
clustered support-ticket path, the representative fallback builds a safe token
vocabulary from `documentation_terms` plus structured product/issue taxonomy
fields carried through support-ticket upload normalization. Structured fields
contribute only when they match the known taxonomy allowlist; uploads without a
matching taxonomy fall back to the generic topic question on this path. The
resolver also ignores any safe vocabulary term that contains an email address or
long numeric identifier. It then treats row text only as a signal: tokens must
appear in the subcluster evidence, repeat at least twice, and also exist in that
safe vocabulary before they can render in the heading.

The emitted label uses the safe documentation/glossary or structured taxonomy
display token, not the raw row token. Raw support-ticket `source_title` remains
excluded from the vocabulary. If no safe distinct label can be formed, the
resolver falls back to the generic topic policy question. Non-clustered rows
keep the existing topic policy fallback.

This fixes the duplicate-label presentation defect for supplied-glossary or
known-taxonomy uploads without changing which subclusters exist, lowering the
count by collapse, or reordering groups. Uploads without a matching safe
vocabulary keep the generic topic fallback by design.

## Intentional

- **No merge pass.** The operator decision rejected symptom-level collapse; this
  slice removes that code instead of trying to warn around it.
- **`source_policy` remains the source bucket.** Representative labels are
  derived from source metadata/content, not extracted customer questions, so no
  result-contract enum is added.
- **Cluster marker required.** The representative fallback is gated on retained
  `support_ticket_cluster` data so older complaint/search/source-policy cases do
  not drift.
- **Representative labels are controlled-vocabulary only.** The new
  representative source-policy path does not render raw `source_title` or raw
  evidence text; row text can only select repeated known-safe documentation
  tokens or allowlisted structured product/issue taxonomy tokens.
- **Resolution-backed groups stay separate.** This slice does not merge or
  otherwise combine evidence scopes.

## Deferred

- #1515's hybrid lexical/embedding recall booster.
- Hosted paid-funnel re-baseline after representative labels land.
- UI/report wording changes around the new label shape.
- Broader representative-label tuning for non-clustered fallback paths.
- Parked hardening: `Customer-wording FAQ headings can publish raw PII` and
  `Safe-vocabulary representative label collisions render duplicate FAQ headings`.

## Verification

- Command passed: python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py extracted_content_pipeline/support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_support_ticket_input_package.py.
- Command passed: pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_support_ticket_input_package.py -q -- 298 passed.
- Command passed: pytest tests/test_content_ops_deflection_resolution_live_proof.py::test_resolution_live_proof_regenerates_from_committed_csv -q -- 1 passed.
- Command passed: bash scripts/validate_extracted_content_pipeline.sh.
- Command passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline.
- Command passed: python scripts/audit_extracted_standalone.py --fail-on-debt.
- Command passed: bash scripts/check_ascii_python.sh.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh -- 4041 passed, 10 skipped, 1 warning.
- Command passed: git diff --check.
- Planned before push: bash scripts/local_pr_review.sh --current-pr-body-file tmp/deflection_question_label_merge_pr_body.md.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 20 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 10 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 223 |
| `plans/PR-Deflection-Question-Label-Merge.md` | 183 |
| `tests/test_extracted_support_ticket_input_package.py` | 4 |
| `tests/test_extracted_ticket_faq_markdown.py` | 438 |
| **Total** | **878** |
