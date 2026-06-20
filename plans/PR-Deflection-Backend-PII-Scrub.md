# PR-Deflection-Backend-PII-Scrub

## Why this slice exists

Portfolio copy cannot honestly claim trustworthy PII scrubbing while ATLAS only
has narrow FAQ-heading/evidence privacy filters. The hosted deflection backend
persists a free Snapshot and paid report artifact in
`content_ops_deflection_reports`; those payloads can include customer-facing
question text, customer wording, teaser answers, Markdown, raw FAQ items,
structured report-model rows, and evidence-export rows. This slice adds the
backend scrub boundary before those payloads are stored or served.

Review found the first implementation still treated identifiers as plain text:
prefixed IDs were over-redacted to one flat token, while bare numeric IDs under
identifier keys could leak because the scrubber ignored field context. The root
cause was context-blind recursive value scrubbing, plus word-boundary regexes
that treated Markdown underscores as part of a token. This update fixes the
root in the backend scrub boundary by tokenizing non-source identifier fields,
preserving paid source-link fields for buyer traceability, and making the text
patterns delimiter-aware.

This PR is over the 400 LOC soft cap because it combines the existing storage
boundary, the field-context scrubber fix required by review, and the negative
tests that prove both sides of the privacy guard. Splitting after the review
findings would leave the backend guarantee knowingly incomplete.

## Scope (this PR)

Ownership lane: content-ops/deflection-privacy
Slice phase: Production hardening

1. Scrub supported PII classes from completed deflection report artifact
   payloads before the host stores them.
2. Build the free Snapshot from the scrubbed artifact so pre-checkout and paid
   surfaces share the same backend privacy boundary.
3. Preserve paid-report source IDs as buyer traceability/linkage keys while
   scrubbing supported PII classes from answer, step, quote, and structured
   report text.
4. Treat Markdown emphasis characters as delimiters for supported email, phone,
   and labeled-identifier detection.
5. Add focused tests proving raw email, phone, account/reference identifiers,
   and redaction-token artifacts do not survive in the Snapshot, paid artifact,
   structured report model, Markdown, FAQ payload, or evidence export.

### Review Contract

- Acceptance criteria:
  - Completed `faq_deflection_report` artifacts are scrubbed before the report
    store receives them.
  - The free Snapshot is derived from the scrubbed artifact, not the raw one.
  - Bare numeric account/case/reference values under known identifier keys are
    replaced with stable per-report refs.
  - Paid-artifact `source_id` / `source_ids` values stay available for buyer
    helpdesk cross-reference, while free Snapshot/gated output still omits
    source IDs.
  - Markdown-wrapped email and phone values are redacted.
  - Tests prove the covered PII classes are absent from persisted Snapshot and
    paid report payloads.
- Affected surfaces: hosted deflection submit, stored free Snapshot, paid
  artifact/report-model payload, report Markdown, FAQ payload, and evidence
  export.
- Risk areas: over-redacting safe support terms, under-redacting supported PII
  classes, breaking the paid-report model shape, or scrubbing delivery metadata
  that the mailer needs.
- Reviewer rules triggered: R1 (requirements match), R2 (failure-class tests),
  R10 (maintainability), R14 (codebase verification).

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Backend-PII-Scrub.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

`faq_deflection_report` gets a small deterministic scrubber for supported
customer PII classes:

- email-like tokens;
- phone-shaped numbers;
- long identifier/account/case/reference numbers in identifier context;
- existing redaction-token artifacts such as `XXXXX` and bracketed redactions.

The scrubber walks Mapping/Sequence payloads recursively and rewrites strings
with typed redaction labels. Before the recursive rewrite, it collects values
from known identifier fields such as `account_id`, `case_id`, `reference_id`,
and safe suffix variants. Each distinct non-source identifier gets a
deterministic `deflection-ref-*` token for that scrub pass; prefixed and bare
forms sharing the same long numeric body map to the same token. That closes the
bare-ID leak for account/case/reference fields without changing the paid
report's existing source-ID cross-reference contract.

Source-link keys such as `source_id`, `source_ids`, `ticket_id`, and
`first_source_id` are treated as paid-report linkage metadata: normal source IDs
are preserved in the paid artifact so buyers can cross-reference their own
helpdesk, while email/phone-shaped source IDs are still scrubbed as supported
PII. The free Snapshot remains source-free.

Text matching uses delimiter-aware boundaries instead of `\b`/`\w`, so
Markdown-wrapped values like `_jane.doe@acme.com_` and `_555-123-4567_` are
redacted. Dict keys are scrubbed as well as values, and the scrubber preserves
its own typed redaction tokens when the Snapshot gets a second defense-in-depth
scrub pass.

`_gate_deflection_report_artifacts` converts the completed report step into a
scrubbed artifact payload before persistence, then calls
`build_deflection_snapshot` on that scrubbed payload and stores both scrubbed
objects. The public Snapshot/artifact/report-model routes continue to read from
the same store, so they inherit the backend boundary instead of depending on
client-side upload behavior.

## Intentional

- The slice backs a deterministic server-side guarantee for the PII classes
  listed above. It does not claim arbitrary human-name detection; that needs a
  separate classifier/metadata-backed path before the portfolio copy can say
  broader things.
- The scrubber rewrites rather than fails the whole report so support-cost and
  opportunity counts remain usable when a few labels contain identifiers.
- `deflection-ref-*` tokens are stable only inside one scrubbed report payload.
  They are for non-source account/case/reference identifiers, not cross-report
  correlation.
- Paid-report source IDs remain visible for buyer traceability. PII-bearing
  source-ID values such as emails are scrubbed, but a broader source-ID policy
  can be revisited separately if the product chooses privacy over direct
  helpdesk cross-reference later.
- `delivery_email` is not scrubbed by this slice because it is operational
  delivery metadata, not part of the customer-facing report artifact.

## Deferred

Portfolio claim/copy upgrades remain deferred until this backend slice is
merged and the portfolio issue can point at the supported scrub classes.
Local NER for names/addresses and retention/deletion controls for stored
`content_ops_deflection_reports` remain follow-up work under the #1734 privacy
arc.

Parked hardening: none.

## Verification

- Command passed: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_supported_pii_before_snapshot_projection tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_identifier_fields_markdown_and_keys tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii tests/test_extracted_content_deflection_submit.py::test_deflection_submit_accepts_zendesk_full_thread_blob -q` -- 4 passed.
- Command passed: `python -m pytest tests/test_extracted_content_ops_live_execute_harness.py::test_deflection_report_execute_uncaps_paid_artifact_and_keeps_snapshot_top_n tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_supported_pii_before_snapshot_projection tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_identifier_fields_markdown_and_keys tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii tests/test_extracted_content_deflection_submit.py::test_deflection_submit_accepts_zendesk_full_thread_blob -q` -- 5 passed.
- Command passed: Python compile check for the touched Python modules and test files.
- Command passed: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_supported_pii_before_snapshot_projection tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_identifier_fields_markdown_and_keys tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii -q` -- 3 passed.
- Command passed: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_snapshot_strips_answers_evidence_and_sources tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_supported_pii_before_snapshot_projection -q` -- 2 passed.
- Command passed: `python -m pytest tests/test_extracted_content_deflection_submit.py::test_deflection_submit_fetches_blob_and_returns_locked_report tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii -q` -- 2 passed.
- Command passed: repository ASCII Python policy script.
- Recurring value/copy sweep: not applicable; this slice adds a backend scrub
  boundary and tests, not a repeated user-facing copy/string replacement.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 7 |
| `extracted_content_pipeline/faq_deflection_report.py` | 359 |
| `plans/PR-Deflection-Backend-PII-Scrub.md` | 163 |
| `tests/test_content_ops_deflection_report.py` | 113 |
| `tests/test_extracted_content_deflection_submit.py` | 149 |
| **Total** | **791** |
