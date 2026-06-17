# Deflection Parser Lane Closeout

Date: 2026-06-17

Issues: #1467, #1582, #1463

## Status

The parser-testing lane has completed the deterministic parser/admission work
that can be finished without inventing a low non-zero reject threshold from
synthetic fixtures.

Completed slices:

| PR | Slice | Outcome |
|---|---|---|
| #1624 | Parser invariant test pack | Added invariant coverage for parser folds, resolution action terms, and snapshot field deletion. |
| #1626 | Parser breakage evidence runner | Added repeatable breakage/admission evidence so parser cliffs are visible and named. |
| #1657 | JSON message guard | Machine JSON in mapped message fields no longer counts as usable customer wording. |
| #1662 | Observed text aliases | Real observed body aliases map to source text; private/internal rows are skipped before source-text extraction. |
| #1667 | JSONL line diagnostics | Malformed JSONL lines are non-fatal row warnings instead of whole-file crashes. |
| #1673 | CSV field limit | Large quoted support-ticket body cells parse through the real admission path. |
| #1675 | Diagnostics parse errors | Parser-level failures surface as structured diagnostics instead of tracebacks. |

## Real Evidence Recorded

The observed CSV evidence is recorded in
`docs/extraction/validation/deflection_csv_observed_text_alias_evidence_2026-06-17.md`.

Summary:

| Case | Raw rows | Usable rows | Ratio | Admission | Interpretation |
|---|---:|---:|---:|---|---|
| `jira_utterances` | 30104 | 21396 | 0.710736 | ACCEPT | Partial coverage is explained by private/internal rows, not parser uncertainty. |
| `support_archive` | 8469 | 8469 | 1.0 | ACCEPT | `Ticket Description` is a real body-shaped export alias. |

The raw observed CSVs are not committed.

## Remaining Policy Boundary

Do not set a hard low non-zero reject threshold from the current evidence.
The only real partial case found so far should accept with a warning because
the missing rows are explicitly private/internal. A reject threshold needs a
real provider/export sample where low coverage is caused by parser uncertainty,
not by intentional private-row filtering.

The next parser/admission change should therefore require one of:

1. a real provider CSV with low non-zero public/source coverage caused by
   ambiguous or unmapped fields; or
2. a product decision that the preview should warn more aggressively even when
   the parser can explain the missing rows as private/internal.

Until then, the correct product behavior is ACCEPT-with-warning for explained
partial coverage and REJECT only for zero usable public/source rows or
malformed inputs that fail closed with diagnostics.
