# PR-Deflection-Delimiter-Vote-Reject

## Why this slice exists

Issue #1459 is the next launch-readiness parser gap after #1455: the unified CSV loader must choose common help-desk delimiters deterministically and fail loud when no delimiter produces a consistent table. Current `main` already routes campaign and deflection source CSVs through `_load_csv_dict_rows`, but delimiter choice is still delegated to `csv.Sniffer` without a reported/reject branch, so a long free-text support-ticket body can still make delimiter behavior hard to reason about.

Root cause: the CSV loader modeled delimiter detection as an opaque Sniffer choice rather than an explicit candidate contract, so it could not distinguish true delimiter evidence from free-text delimiter characters or report collapse. The review follow-up found the first implementation still treated a collapsed competing-delimiter row as a soft ratio miss, and modeled delimiter without quotechar, which was a symptom fix rather than the full parser contract. This PR fixes the root by making delimiter, quotechar, row-width fit, and collapsed-row detection part of one deterministic candidate model.

Diff-size note: the review fix brings the slice above the 400 LOC soft cap because the root fix needs parser-state changes plus two regression fixtures: the >=90% collapsed-row boundary and single-quote quotechar preservation. Splitting those would leave one reviewed regression unprotected on the same import path.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Replace the `csv.Sniffer` delimiter selection inside the shared `_load_csv_dict_rows` path with deterministic bounded delimiter voting over the parsed sample.
2. Reject CSVs whose non-empty data rows cannot reach a >=90% delimiter-consistency threshold under the chosen delimiter, with the first offending line named in the error; hard-reject any collapsed competing-delimiter row regardless of the ratio.
3. Prove that semicolon/tab text inside quoted support-ticket bodies does not override comma headers, and that genuinely inconsistent delimiter shapes fail before producing collapsed rows.

### Review Contract

- Acceptance criteria:
  - A comma-delimited support-ticket CSV whose quoted body contains semicolons and tabs still parses as comma-delimited rows.
  - A semicolon-delimited support-ticket CSV still parses correctly through `load_source_rows_from_file`.
  - A malformed mixed-delimiter CSV fails with a clear delimiter/column-consistency error instead of returning one-column or misaligned rows.
  - A >=90% mostly-consistent CSV still fails if one row collapses into a competing-delimiter one-cell row.
  - Single-quote-quoted CSV fields with embedded commas continue to parse through the shared import path.
  - Campaign CSV loading continues to support semicolon exports through the same shared loader.
- Affected surfaces:
  - `extracted_content_pipeline/campaign_customer_data.py`
  - `tests/test_extracted_campaign_source_adapters.py`
  - `tests/test_extracted_campaign_customer_data.py`
- Risk areas:
  - Breaking quoted multiline CSV rows.
  - Rejecting valid sparse rows that have fewer cells than the header.
  - Treating a later mixed-delimiter data row as a fallback header candidate.
  - Reintroducing nondeterministic or hard-to-debug delimiter behavior.
  - Regressing quotechar support that `csv.Sniffer` previously inferred.
- Reviewer rules triggered: R1, R2, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/campaign_customer_data.py`
- `plans/PR-Deflection-Delimiter-Vote-Reject.md`
- `tests/test_extracted_campaign_source_adapters.py`

## Mechanism

The loader parses the same bounded text sample against `,`, `;`, tab, and `|`, crossed with double-quote and single-quote dialects. For each candidate it scores non-empty rows by expected header width, exact-width fit, delimiter consistency, and hard collapsed-row detection, ignoring blank rows and keeping quoted fields intact through Python's CSV parser. The best candidate wins only if it has a plausible header, at least 90% of non-empty data rows are consistent with the chosen delimiter/quotechar, and no one-cell data row still contains a competing delimiter. Short ragged rows remain allowed unless a one-cell row still contains a competing delimiter, which is the mixed-delimiter collapse case; extra cells keep the existing "more cells than the header" failure unless another quotechar candidate parses them exactly. If no candidate reaches the threshold, `_load_csv_dict_rows` raises `ValueError` with the selected/rejected delimiter context and the first offending row number. The rest of row cleaning stays unchanged.

## Intentional

- This does not add caller-supplied delimiter hints in this slice; there is no current upload UI/control surface for one, and the launch blocker is silent autodetection/collapse.
- This does not build the full unmapped-column report from #1457; that issue is claimed by another developer and should remain separate.
- The reject branch is row-shape based rather than semantic. Header/body role mapping remains the job of the existing source-row adapter and the #1457 follow-up.

## Deferred

- #1457 remains the header/body-column admission report and Zendesk column mapping fix, currently claimed by another developer.
- Caller-supplied delimiter overrides can be added later if the paid-tier UI exposes an explicit advanced import control.

Parked hardening: none.

## Verification

- python -m py_compile extracted_content_pipeline/campaign_customer_data.py tests/test_extracted_campaign_source_adapters.py (pass)
- pytest tests/test_extracted_campaign_source_adapters.py -q -k "delimiter or quote or multiline or semicolon" (6 passed, 79 deselected)
- pytest tests/test_build_deflection_messy_csv_fixtures.py -q -k "ragged_short_rows or ragged_extra_cells" (2 passed, 9 deselected)
- pytest tests/test_smoke_content_ops_support_ticket_package.py -q -k "no_rows_survive" (1 passed, 17 deselected)
- pytest tests/test_extracted_campaign_source_adapters.py -q (85 passed)
- pytest tests/test_extracted_campaign_customer_data.py -q (10 passed)
- pytest tests/test_build_deflection_messy_csv_fixtures.py -q (11 passed)
- pytest tests/test_smoke_content_ops_support_ticket_package.py -q (18 passed)
- pytest tests/test_extracted_content_deflection_submit.py -q -k "embedded_quotes_and_newlines" (1 passed, 53 deselected)
- pytest tests/test_extracted_content_deflection_submit.py -q (54 passed)
- bash scripts/check_ascii_python.sh (pass)
- bash scripts/run_extracted_pipeline_checks.sh (4138 passed, 10 skipped, 1 warning)

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_customer_data.py` | 256 |
| `plans/PR-Deflection-Delimiter-Vote-Reject.md` | 86 |
| `tests/test_extracted_campaign_source_adapters.py` | 99 |
| **Total** | **441** |
