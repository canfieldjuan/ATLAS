# PR-Deflection-Zendesk-Product-Proof-Eval

## Why this slice exists

`PR-Deflection-Zendesk-Product-Proof-Corpus` captured and labeled the
50-ticket Zendesk product-shaped corpus, but explicitly deferred the funnel run:
feeding the corpus through deflection and scoring publishable-answer precision,
private-note exclusion, and reopened or unresolved handling. `PR-Deflection-
Question-Label-Quality` then fixed the buyer-facing label defects that made the
first Zendesk proof artifact unsuitable as a quality pass. This slice turns
that deferred corpus into a deterministic validation run so the product-proof
lane has a fresh artifact after the label cleanup.

This PR may exceed 400 LOC because the evaluator is a checker: AGENTS.md
requires failure-branch tests for detection logic, and splitting the tests away
would leave the artifact generator looking green without proving that false
positive, private-note, or unresolved-ticket violations actually fail.

## Scope (this PR)

Ownership lane: content-ops/deflection-product-proof
Slice phase: Functional validation

1. Add an offline Zendesk product-proof evaluator that reads the committed
   sanitized corpus, imports it through the real full-thread adapter, builds the
   existing deflection report artifact, and scores generated items against the
   corpus `expected` labels by `zd-proof-NNN` source id.
2. Commit a fresh validation summary and short report excerpt generated after
   the question-label cleanup, replacing the prior artifact's known
   output-quality boundary with measured pass/fail fields.
3. Add focused tests proving the evaluator catches publishable-answer false
   positives, private-note leakage, and unresolved or reopened sources appearing
   in publishable-answer items.
4. Park the pre-existing `es`-ending depluralizer NIT as hardening only; this
   slice does not touch tokenizer behavior unless the evaluator reveals a
   current product-proof failure.
5. Fix one extracted-suite test isolation issue found during verification: the
   delivery CLI fail-closed test now clears result URL env fallbacks before
   asserting the missing-destination error.
6. Fix the review-discovered false-green root: unscoped resolution evidence is
   no longer promoted into publishable answers, and the evaluator blocks on any
   failed artifact output check.
7. Fix the remaining review-discovered safety hole: private-note leak detection
   now catches truncated token windows instead of requiring the full private
   note body to appear verbatim in Markdown.
8. Fix the published localized heading root by stripping metadata before the
   Portuguese `Como ...` question opener; the evaluator now blocks degraded
   published labels and records draft-only weak labels separately.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `HARDENING.md`
- `docs/extraction/validation/deflection_zendesk_product_proof_corpus.md`
- `docs/extraction/validation/deflection_zendesk_product_proof_eval_2026-06-14.md`
- `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/report_excerpt.md`
- `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/summary.json`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Zendesk-Product-Proof-Eval.md`
- `scripts/evaluate_zendesk_product_proof_corpus.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_evaluate_zendesk_product_proof_corpus.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_send_content_ops_deflection_report_deliveries.py`

### Review Contract

Acceptance criteria:
- The evaluator uses `rows_from_zendesk_full_thread` and the existing support
  ticket deflection report path; it must not invent a separate Zendesk parser.
- The summary records generated item counts, publishable-answer item counts,
  label-based false positives, reopened or unresolved publishable violations,
  private-note leak findings, degraded published labels, draft-only weak labels,
  and failed artifact output checks.
- The committed artifact must no longer show `[Atlas seed ...]` question
  headings, the degraded `What should I do about atla?` label, or the
  published `Localized support question ...` prefix.
- Tests include negative fixtures for each detector branch added by this PR.

Affected surfaces:
- Offline validation scripts and docs under `docs/extraction/validation`.
- Extracted pipeline CI enrollment for the new evaluator tests.
- A delivery CLI test fixture only; no delivery production code changes.
- No live Zendesk, Stripe, hosted submit, or email path is touched.

Risk areas:
- A summary-only artifact could claim quality without proving failure
  detection.
- Text-only private-note checks can be brittle, so source-id label checks must
  be the primary gate and text checks must be conservative enough to catch
  truncated evidence quotes.
- The evaluator must not mutate the committed corpus labels.
- The evaluator must not report `status=ok` when the underlying report artifact
  has failed output checks.

Reviewer rules triggered: R1, R2, R10, R13, R14

## Mechanism

The evaluator loads `zendesk_product_proof_corpus.json`, keeps the committed
`expected` labels keyed by local ticket id, and normalizes the same corpus
through `rows_from_zendesk_full_thread`. It passes those imported rows into the
existing support-ticket input package and report builder, then evaluates the
generated FAQ items by their `source_ids`.

For each generated item with `answer_evidence_status` equal to
`resolution_evidence`, the evaluator checks whether any source id is labeled as
non-publishable, reopened, or unresolved. It separately compares report
markdown against private comment text from the corpus using normalized token
windows so an internal note cannot enter the committed buyer-facing excerpt
unnoticed through quote truncation. The JSON summary stores both the raw metric
counts and the named violation lists; the script exits non-zero when any
blocking violation is present.

The report builder also fails closed one layer upstream: resolution text whose
scope cannot be tied to the selected question remains a draft-needs-review item
instead of entering the publishable-answer section. Controlled-vocabulary
disambiguated source-policy labels can still publish when the disambiguation
path safely reattaches scoped resolution evidence.

The localized-heading fix also lives upstream in the report builder: bracketed
seed metadata can now be followed by a Portuguese `Como ...` customer question
without leaving the theme prefix in the buyer-facing heading. The evaluator
then blocks degraded labels only for published answers and records draft-only
weak labels separately, so the artifact does not claim those draft labels are
clean.

## Intentional

- This is offline validation against the committed sanitized corpus. The live
  Zendesk API capture remains operator-controlled and is not rerun here.
- The evaluator scores source-id membership rather than trying to infer support
  intent from generated prose. The corpus labels are the source of truth for
  publishability.
- Private-note detection is a conservative text-leak check on committed private
  comment bodies after normalization and six-token window matching; it is not a
  broad PII scanner.
- The pre-existing `es`-ending tokenizer NIT is parked, not fixed, because the
  current proof question labels no longer depend on that behavior.
- The delivery test isolation patch is included because it blocked this slice's
  required extracted-suite verification; it does not alter delivery behavior.
- The Codex P2 finding is fixed instead of waived: failed artifact output
  checks are evaluator blockers, and the current artifact now has zero failed
  output checks after unscoped resolution evidence is held for review.
- Draft-only weak labels remain a recorded boundary, not a blocking safety
  failure: they are not published answers, and the current summary reports them
  under `degraded_draft_question_labels`.

## Deferred

Hosted end-to-end proof remains outside this slice: submit, payment unlock,
delivery email, and portfolio result-page validation stay in #1440's full
funnel lane.

Parked hardening: support-ticket tokenizer still strips one trailing `s` from
some `es`-ending words such as `series` and `kubernetes`; record in
`HARDENING.md`.

## Verification

- `PATH=/home/juan-canfield/Desktop/Atlas/.venv/bin:$PATH pytest tests/test_evaluate_zendesk_product_proof_corpus.py tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_cleans_committed_zendesk_product_corpus_labels tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_strips_seed_subject_prefix_from_customer_question -q` -- 18 passed.
- `PATH=/home/juan-canfield/Desktop/Atlas/.venv/bin:$PATH python scripts/evaluate_zendesk_product_proof_corpus.py --json` -- exited 0 with `status=ok`, 0 blocking violations, 0 private-note leaks, 0 degraded published question labels, 3 recorded degraded draft labels, 0 failed artifact output checks, 25 of 36 publishable-labeled sources covered, and 2 recorded FAQ warnings.
- `PATH=/home/juan-canfield/Desktop/Atlas/.venv/bin:$PATH pytest tests/test_evaluate_zendesk_product_proof_corpus.py tests/test_extracted_ticket_faq_markdown.py tests/test_content_ops_deflection_report.py tests/test_send_content_ops_deflection_report_deliveries.py::test_validate_fails_closed_before_pool_for_missing_destination -q` -- 338 passed.
- `PATH=/home/juan-canfield/Desktop/Atlas/.venv/bin:$PATH python -m compileall -q scripts/evaluate_zendesk_product_proof_corpus.py extracted_content_pipeline/ticket_faq_markdown.py tests/test_evaluate_zendesk_product_proof_corpus.py tests/test_extracted_ticket_faq_markdown.py && bash scripts/check_ascii_python.sh` -- passed.
- `PATH=/home/juan-canfield/Desktop/Atlas/.venv/bin:$PATH bash scripts/run_extracted_pipeline_checks.sh` -- 4248 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `HARDENING.md` | 11 |
| `docs/extraction/validation/deflection_zendesk_product_proof_corpus.md` | 9 |
| `docs/extraction/validation/deflection_zendesk_product_proof_eval_2026-06-14.md` | 52 |
| `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/report_excerpt.md` | 119 |
| `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/summary.json` | 194 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 52 |
| `plans/PR-Deflection-Zendesk-Product-Proof-Eval.md` | 184 |
| `scripts/evaluate_zendesk_product_proof_corpus.py` | 486 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_evaluate_zendesk_product_proof_corpus.py` | 280 |
| `tests/test_extracted_ticket_faq_markdown.py` | 13 |
| `tests/test_send_content_ops_deflection_report_deliveries.py` | 11 |
| **Total** | **1414** |
