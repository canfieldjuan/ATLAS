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

## Artifact Links

- Summary JSON:
  `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json`

## Boundary

This is offline validation. It does not call live Zendesk, upload a customer
file, mutate data, charge Stripe, or change parser policy.
