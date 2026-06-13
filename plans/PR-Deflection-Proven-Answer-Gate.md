# PR-Deflection-Proven-Answer-Gate

## Why this slice exists

#1456 is a P0 launch-readiness issue: the paid deflection report currently
marks any non-empty `resolution_text` as `resolution_evidence`. That means
closure boilerplate such as "Customer did not respond, closing this out" or
internal notes such as "Escalated to T2 / refunded per policy 4.2" can become
paid, teaser-eligible, customer-facing answers.

This slice hardens the proven-answer gate before the #1440 real full-volume
paid run. It keeps the deterministic/no-LLM deflection lane intact and avoids
the open #1452 parser/full-volume submit PR.

Diff budget note: this PR is over the 400 LOC soft cap after the review-blocker
fixes because the gate, symmetric verb normalization, disposition-only note
filter, concrete-start action support, narrow synonym overlap support, and
failure-first fixtures need to land together. Splitting the past-tense
normalization, disposition-only rejection, valid "start the return"
restoration, or login/password-style synonym support from the gate would
knowingly leave either real support resolutions under-included or generic agent
status updates over-included.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Filter support-ticket resolution candidates before they can populate
   `resolution_text`, `evidence_group_key`, `resolution_evidence`, answer
   copy, or steps in `ticket_faq_markdown.py`.
2. Reject closure/disposition boilerplate, disposition-only agent update notes,
   and narrow internal-note patterns.
3. Require a minimal actionable/on-topic signal for a resolution to count as
   publishable answer evidence.
4. Add failure-first fixtures proving boilerplate/internal notes and
   disposition-only agent updates stay `draft_needs_review`, plus positive
   fixtures proving real step-wise resolution evidence still gates, including
   past-tense agent language, concrete account-review steps, and common
   support synonym pairs such as login/password reset and receipt/invoice.

### Review Contract
- Acceptance criteria:
  - [ ] Closure boilerplate does not produce `resolution_evidence`, resolution
        source counts, customer-facing steps, or leaked answer copy.
  - [ ] Internal operational notes do not produce `resolution_evidence` or
        leak internal policy/escalation details into paid answers.
  - [ ] Disposition-only agent status updates such as reviewed/replied or
        checked/sent-update notes do not produce `resolution_evidence`,
        including phrasing where the update/reply appears before the recipient.
  - [ ] A genuine step-wise, on-topic resolution still gates as
        `resolution_evidence`, including "start the return" style portal
        instructions.
  - [ ] Common support synonym pairs used by real tickets, such as
        login/password reset, receipt/invoice, connect/integration sync, and
        renewal/cancel, do not get dropped by a raw lexical-overlap check.
  - [ ] Off-topic near-misses still fail closed; synonym expansion must not let
        billing steps answer login questions or password resets answer receipt
        questions.
  - [ ] The filter is deterministic and lives in the extracted package; no
        LLM/Ollama/local model path is introduced.
- Affected surfaces: deflection FAQ markdown/report item construction and its
  extracted-checks test suite.
- Risk areas: over-filtering short legitimate resolutions, under-filtering
  generic closure text, grouping drift via bogus `evidence_group_key`.
- Reviewer rules triggered: R1, R2, R9, R10, R13.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Proven-Answer-Gate.md`
- `tests/test_build_deflection_messy_csv_fixtures.py`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

The resolution-text normalizer is the earliest point where raw evidence/opportunity
fields are normalized into the row-level `resolution_text` used by grouping,
answer status, answer copy, and resolution source counts. This slice adds a
deterministic publishability predicate there:

1. compact the candidate text;
2. reject narrow closure/disposition boilerplate;
3. reject narrow internal-note patterns such as tier escalations and numbered
   internal policy references;
4. reject disposition-only customer-update notes when the only action signal is
   weak status/review/check/send/update/start wording, including
   `sent an update to the customer` and `provided an update to the requester`;
5. require enough actionable/on-topic signal before returning the text, using
   the ticket text plus existing topic/source-title/pain-category/tag context
   so legitimate invoice/receipt wording is not lost.
6. normalize action verbs symmetrically enough that e-ending past-tense agent
   notes such as "enabled", "configured", and "updated" still match the
   publishable-action gate.
7. recognize concrete "start" instructions as customer-facing actions while
   still rejecting started/reviewed/sent-update disposition notes.
8. expand only the final topic-overlap token sets through a narrow support
   synonym map. The action-token and disposition-only checks still use the raw
   normalized tokens, so synonym support cannot manufacture action evidence or
   bypass the weak-status guard.

If a candidate fails, the normalizer returns an empty string. That keeps
the bogus text out of `evidence_group_key`, collected resolution texts,
`resolution_source_count`, resolution-evidence scope calculation, generated
steps, and paid answer summaries.

## Intentional

- No LLM judge in this slice. The launch blocker is that obvious non-answer
  text is currently trusted as proven evidence; a deterministic fail-closed
  filter closes that immediate risk without adding cost or model dependency.
- The internal-note matcher stays narrow. It rejects concrete operational
  patterns named in #1456 but avoids broad words such as "policy" or
  "escalation" by themselves, because those can appear in legitimate
  customer-facing instructions.
- The disposition-only guard is constrained to weak action-token sets plus
  customer-update/reply wording. It does not reject concrete step-wise account
  fixes such as opening invoices and updating a payment method, or concrete
  return-flow instructions such as starting a return in a portal.
- The synonym map is intentionally small and support-domain-specific. It covers
  recurring help-desk wording pairs that still point at the same customer task,
  and tests include off-topic near-misses to keep the expansion from becoming a
  broad topical bypass.
- This does not solve the separate #1460 fixed-bucket over-merge issue.

## Deferred

- #1460 remains the broader within-intent clustering/subcluster fix.
- A future robust-testing slice can add a larger resolution-quality corpus and
  calibrate thresholds against real help-desk exports.

Parked hardening: none.

## Verification

- Focused pytest for `tests/test_extracted_ticket_faq_markdown.py`.
  - Passed, 177 tests.
- Downstream pytest targets in `tests/test_content_ops_deflection_resolution_live_proof.py`,
  `tests/test_extracted_ticket_faq_macro_writeback.py`,
  `tests/test_extracted_ticket_faq_output_ingestion.py`, and
  `tests/test_extracted_content_ops_live_execute_harness.py`
  - Passed, 4 tests.
- `./scripts/validate_extracted_content_pipeline.sh`
  - Passed.
- `./extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - Passed.
- `./scripts/audit_extracted_standalone.py --fail-on-debt`
  - Passed.
- `./scripts/check_ascii_python.sh`
  - Passed.
- Python compile check for `extracted_content_pipeline/ticket_faq_markdown.py`
  and `tests/test_extracted_ticket_faq_markdown.py`
  - Passed.
- `./scripts/run_extracted_pipeline_checks.sh`
  - Passed, 3881 passed, 10 skipped; existing torch/pynvml warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | 246 |
| `plans/PR-Deflection-Proven-Answer-Gate.md` | 165 |
| `tests/test_build_deflection_messy_csv_fixtures.py` | 11 |
| `tests/test_extracted_ticket_faq_markdown.py` | 343 |
| **Total** | **765** |
