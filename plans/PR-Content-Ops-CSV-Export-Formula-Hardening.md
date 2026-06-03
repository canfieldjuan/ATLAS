# PR-Content-Ops-CSV-Export-Formula-Hardening

## Why this slice exists

PR #1283's review surfaced a recurring Content Ops CSV export hardening gap:
the duplicated `_csv_value` helpers serialize customer/scraped/generated text
directly into CSV cells. If an operator opens an exported CSV in a spreadsheet,
cells beginning with `=`, `+`, `-`, or `@` can be interpreted as formulas.
The generated-assets export path now includes marketer-provided review input and
LLM-shaped copy, so this belongs as a small production-hardening follow-up
before building more output variants.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input-hardening

Slice phase: Production hardening

1. Add one package-owned CSV cell serialization helper for
   `extracted_content_pipeline` exports.
2. Replace the duplicated `_csv_value` helpers in the Content Ops export
   result classes with the shared helper.
3. Keep existing JSON serialization and `None` handling stable while escaping
   formula-leading string cells.
4. Add focused tests for all risky prefixes, leading-whitespace formulas,
   normal strings, non-string numerics, and JSON/list cells.
5. Enroll the new helper test in the extracted pipeline CI wrapper.

### Files touched

- `extracted_content_pipeline/csv_export.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/campaign_postgres_export.py`
- `extracted_content_pipeline/campaign_reasoning_postgres.py`
- `extracted_content_pipeline/report_export.py`
- `extracted_content_pipeline/blog_post_export.py`
- `extracted_content_pipeline/landing_page_export.py`
- `extracted_content_pipeline/sales_brief_export.py`
- `extracted_content_pipeline/ticket_faq_export.py`
- `extracted_content_pipeline/social_post_export.py`
- `extracted_content_pipeline/ad_copy_export.py`
- `tests/test_extracted_csv_export.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `plans/PR-Content-Ops-CSV-Export-Formula-Hardening.md`

## Mechanism

The helper keeps the previous export contract first, then applies the
spreadsheet guard only to string cells:

```python
def csv_cell_value(value):
    if isinstance(value, (Mapping, list, tuple)):
        text = json.dumps(value, default=str, separators=(",", ":"))
    elif value is None:
        text = ""
    else:
        text = value
    if isinstance(text, str) and text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text
```

Each export module imports `csv_cell_value` as its local `_csv_value`, so the
row-writing loops stay unchanged while the guard applies uniformly to generated
asset exports, legacy campaign draft exports, and campaign reasoning context
CSV inventory.

## Intentional

- CSV import parsers (`campaign_customer_data.py` and
  `podcast_transcript_data.py`) are not in scope. They parse operator-supplied
  input into structured rows; the risk here is exported cells being opened by a
  spreadsheet.
- Numeric values remain numeric. The guard applies to strings only, so an actual
  integer or float does not become text; a string value such as `"-SUM(A1:A2)"`
  is escaped.
- Nested JSON values are not individually walked. Existing exports already
  serialize mappings/lists into JSON text cells; the cell is escaped only when
  the serialized cell itself starts with a formula prefix.

## Deferred

Parked hardening: none.

Atlas-wide CSV surfaces outside `extracted_content_pipeline` are not audited in
this slice. This PR drains the exact Content Ops export family raised by the
#1283 review NIT; a future repo-wide CSV hardening audit can cover unrelated
B2B/prospect exports if needed.

## Verification

- `pytest tests/test_extracted_csv_export.py -q` (8 passed)
- `pytest tests/test_extracted_campaign_postgres_export.py tests/test_extracted_campaign_reasoning_list_cli.py tests/test_extracted_report_export.py tests/test_extracted_blog_post_export.py tests/test_extracted_landing_page_export.py tests/test_extracted_sales_brief_export.py tests/test_extracted_ticket_faq_export.py tests/test_extracted_content_asset_api.py -q` (140 passed)
- `bash scripts/validate_extracted_content_pipeline.sh` (passed)
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` (passed)
- `python scripts/audit_extracted_standalone.py --fail-on-debt` (passed; findings: 0)
- `bash scripts/check_ascii_python.sh` (passed)
- `bash scripts/run_extracted_pipeline_checks.sh` (3000 passed, 10 skipped)
- `bash scripts/local_pr_review.sh --current-pr-body-file <process-substitution>` (passed)

## Estimated diff size

| Metric | LOC |
|---|---:|
| Added | 227 |
| Deleted | 58 |
| Total | 285 |

Under the 400 LOC soft cap.
