# Deflection CSV Observed Text Alias Evidence

Date: 2026-06-17

Issues: #1467, #1582

## What Ran

Observed local CSV exports were inspected through the real source-row CSV
diagnostics path after adding body-shaped aliases for `actionbody` and
`Ticket Description`.

Command:

```bash
python scripts/evaluate_csv_admission_threshold_evidence.py \
  --out-dir <tmp-observed-evidence-dir> \
  --doc <tmp-observed-evidence-doc> \
  --observed-csv jira_utterances=<observed-csv: sample_utterances.csv> \
  --observed-csv support_archive=<observed-csv: customer_support_tickets.csv> \
  --json
```

The raw CSV files are local observed evidence and are not committed.

## Result

| Case | Raw rows | Usable rows | Usable ratio | Admission | Mapped source text | Notable warnings |
|---|---:|---:|---:|---|---|---|
| `jira_utterances` | 30104 | 21396 | 0.710736 | ACCEPT | `actionbody` | `private_source_text`: 8707; `missing_source_text`: 1 |
| `support_archive` | 8469 | 8469 | 1.0 | ACCEPT | `Ticket Description` | none affecting source text |

Status: `ok`

Observed minimum usable source ratio: `0.710736`

## Interpretation

This is the first real non-zero partial-coverage evidence recorded for the
parser-admission policy. The partial ratio is explained by rows explicitly
marked private/internal in the observed Jira-style utterance export, not by
parser uncertainty. The correct behavior is therefore:

- map public `actionbody` values as source text;
- skip rows marked private/internal before source-text extraction;
- keep the upload on the ACCEPT path with coverage warnings;
- do not set a hard low non-zero reject threshold from this evidence alone.

The support archive evidence confirms that `Ticket Description` is a real
body-shaped support-export alias and should map to source text.

## Boundary

This is offline validation. It does not call live Zendesk, upload a customer
file, mutate data, charge Stripe, or change parser policy. It commits only the
sanitized count/header summary above, not raw CSV content.
