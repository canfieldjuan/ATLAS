# PR-Deflection-Parser-Observed-Text-Aliases

## Why this slice exists

#1582 asks for real observed CSV admission evidence before setting any low
non-zero usable-ratio threshold. A read-only probe against local observed
exports found no partial non-zero threshold case, but it did expose a more
upstream deterministic reject: populated body fields from real exports are not
recognized as source text.

Root cause: `_TEXT_KEYS` does not include two observed producer aliases:
Jira-style `actionbody` / `action_body`, and support-export
`Ticket Description` / `ticket_description`. Because source-row conversion
never maps those fields to source text, non-empty exports with usable customer
wording can reject as `no_usable_source_rows`. The observed `actionbody`
export also carries row-level `is_private` markers, so admitting the alias must
be paired with a row-level private/internal skip before source-text extraction.

This fixes the root at the shared source-text alias boundary. It is not a
threshold-policy PR: #1582/#1467 low non-zero reject policy remains blocked
until a real partial-coverage provider export exists.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-testing
Slice phase: Production hardening

1. Add observed source-text aliases for `actionbody`, `action_body`, and
   `ticket_description`.
2. Add row-level private/internal source-row skip for fields such as
   `is_private=1.0` and `public=0.0`.
3. Add focused fixtures proving public aliases produce usable source rows and
   private rows do not.
4. Add diagnostics fixtures proving these aliases do not reject as
   zero-usable CSVs.
5. Commit a sanitized observed-evidence note with counts/headers, without
   committing raw downloaded CSVs.

### Review Contract

Acceptance criteria:
- Jira-style `actionbody` rows become usable source text.
- Support-export `Ticket Description` rows become usable source text.
- Rows marked private/internal do not become usable source text even when they
  contain `actionbody`.
- Existing zero-usable rejection remains intact for unknown body-like columns.
- No low non-zero usable-ratio reject threshold is introduced.
- No raw local observed CSV is committed.

Affected surfaces:
- Extracted source-row text alias mapping, source adapter tests, ingestion
  diagnostics tests, and the #1582 observed evidence trail.

Risk areas:
- Over-widening source-text aliases so metadata starts counting as customer
  wording.
- Leaking private/internal row text from observed exports.
- Accidentally turning this alias fix into threshold policy.
- Committing raw observed CSV data or PII.

Reviewer rules triggered: R1, R2, R10, R13, R14.

### Files touched

- `docs/extraction/validation/deflection_csv_observed_text_alias_evidence_2026-06-17.md`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `plans/PR-Deflection-Parser-Observed-Text-Aliases.md`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_ingestion_diagnostics.py`

## Mechanism

- Extend `_TEXT_KEYS` with exact observed aliases:
  `ticket_description`, `actionbody`, and `action_body`.
- Rely on the existing normalized/compact field lookup so producer casing like
  `Ticket Description` maps through the same path.
- Add row-level private/public marker detection before source-text extraction.
  Truthy `is_private`/`is_internal` and falsey `public`/`is_public` skip the
  row with `private_source_text`; ambiguous bare `private`/`internal` columns
  are intentionally not visibility markers.
- Keep the current admission envelope and coverage policy unchanged.

## Intentional

- No threshold change: this PR fixes deterministic alias misses only.
- No committed observed CSV fixture: the observed files are local and may
  contain customer-like data. Tests use minimal synthetic rows that model the
  discovered headers.
- `actionbody` is intentionally scoped to a body-like source-text alias, not a
  generic `action` alias.
- Row-level private skips are intentionally fail-closed for common boolean
  spellings (`true`, `1`, `1.0`, `yes`; and `false`, `0`, `0.0`, `no` for
  public markers).

## Deferred

- #1582 threshold policy remains open; this PR records one real non-zero
  partial-coverage observed export, but the partial coverage is explained by
  private rows and does not justify a hard reject threshold.
- #1467 low non-zero reject threshold remains blocked on observed partial
  evidence.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_campaign_source_adapters.py tests/test_extracted_content_ingestion_diagnostics.py tests/test_evaluate_csv_admission_threshold_evidence.py -q`
  - 180 passed in 0.86s.
- `python scripts/evaluate_csv_admission_threshold_evidence.py --out-dir <tmp-observed-evidence-dir> --doc <tmp-observed-evidence-doc> --observed-csv jira_utterances=<observed-csv: sample_utterances.csv> --observed-csv support_archive=<observed-csv: customer_support_tickets.csv> --json`
  - Passed; observed `jira_utterances` at 30104 raw / 21396 usable / ratio
    0.710736 with 8707 `private_source_text` skips, and `support_archive` at
    8469 raw / 8469 usable / ratio 1.0.
- `./scripts/run_extracted_pipeline_checks.sh`
  - 4610 passed, 10 skipped, 1 warning in 74.69s.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_csv_observed_text_alias_evidence_2026-06-17.md` | 56 |
| `extracted_content_pipeline/campaign_source_adapters.py` | 43 |
| `plans/PR-Deflection-Parser-Observed-Text-Aliases.md` | 125 |
| `tests/test_extracted_campaign_source_adapters.py` | 85 |
| `tests/test_extracted_content_ingestion_diagnostics.py` | 86 |
| **Total** | **395** |
