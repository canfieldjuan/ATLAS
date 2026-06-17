# Deflection CSV Admission Threshold Evidence

Date: 2026-06-15

Issue: #1467

## What Ran

This validation projects the committed sanitized Zendesk product-proof corpus
into source-row CSV shapes and inspects each CSV through the real ingestion
diagnostics path with source-row mode enabled.

## Result

| Case | Case status | Raw rows | Usable rows | Usable ratio | Admission | Coverage warnings |
|---|---|---:|---:|---:|---|---:|
| zendesk_public_comments_csv | ok | 50 | 50 | 1.0 | ACCEPT | 0 |
| zendesk_description_csv | ok | 50 | 50 | 1.0 | ACCEPT | 0 |

Status: `ok`

Observed minimum usable source ratio: `1.0`

Observed evidence cases: `0`

## Interpretation

The committed product-shaped Zendesk CSV projections are accepted at full coverage. This supports the clean ACCEPT path, but does not justify a hard low-coverage reject threshold by itself. Operator-supplied observed CSVs are evidence only until a later policy slice promotes a threshold.

This artifact therefore supports keeping clean Zendesk-shaped CSV uploads on
the ACCEPT path. It does not choose a low non-zero reject threshold. That
threshold still needs observed partial-coverage exports, not a synthetic ratio
derived from a clean corpus. Operator-supplied observed CSVs are recorded as
evidence and do not block this runner until a later policy slice promotes a
threshold.

## Parser Breakage Matrix

These synthetic cases break parser mechanics through the same CSV diagnostics
path. They prove whether current guards fail closed, warn, or expose a known
fail-open gap. They do not justify a low-coverage threshold.

| Case | Case status | Expected outcome | Observed outcome | Raw rows | Usable rows | Admission | Decision reason | Decision location | Coverage warnings |
|---|---|---|---|---:|---:|---|---|---|---:|
| unknown_body_like_column_rejects_zero_usable | ok | REJECT | REJECT | 1 | 0 | REJECT | no_usable_source_rows | source_row_csv | 0 |
| private_note_only_rejects_zero_usable | ok | REJECT | REJECT | 1 | 0 | REJECT | no_usable_source_rows | source_row_csv | 0 |
| status_timestamp_only_rejects_zero_usable | ok | REJECT | REJECT | 1 | 0 | REJECT | no_usable_source_rows | source_row_csv | 0 |
| partial_blank_rows_warns_without_rejecting | ok | ACCEPT_WITH_WARNING | ACCEPT_WITH_WARNING | 2 | 1 | ACCEPT |  |  | 1 |
| header_only_csv_has_no_policy_decision | ok | NO_POLICY_DECISION | NO_POLICY_DECISION | 0 | 0 | None |  |  | 0 |
| json_blob_message_known_fail_open | known_gap | ACCEPT_CLEAN | ACCEPT_CLEAN | 1 | 1 | ACCEPT |  |  | 0 |

Known fail-open gaps: `1`

Synthetic breakage cases prove parser mechanics only. Fail-closed and warning expectations are blocking; known fail-open cases are recorded as explicit gaps and do not set low-coverage reject policy.

## Artifact Links

- Summary JSON:
  `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json`

## Boundary

This is offline validation. It does not call live Zendesk, upload a customer
file, mutate data, charge Stripe, or change parser policy.
