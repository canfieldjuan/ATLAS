# Content Selling URL Source CLI

## Why this slice exists

PR-Content-Selling-Context-Inputs made `inputs.selling.booking_url` a real
hosted Content Ops execution input, but deliberately deferred generic CLI sugar.
Source-row operators still have to remember nested `--default-field` plumbing
or use the review-source smoke script. This slice makes the common source-row
CLI path accept the same selling URL without changing generation behavior.

## Scope (this PR)

1. Add a shared source-adapter helper that combines repeatable flat
   `--default-field` values with an optional `--booking-url` into
   `selling.booking_url`.
2. Wire the helper into the source-row builder CLI and offline campaign
   generation example CLI.
3. Replace the review-source Postgres smoke script's local helper with the
   shared helper so the booking URL behavior has one implementation.
4. Add focused tests for the helper and both generic CLIs.

### Files touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `scripts/build_extracted_campaign_opportunities_from_sources.py`
- `scripts/run_extracted_campaign_generation_example.py`
- `scripts/smoke_content_ops_review_source_postgres.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_campaign_generation_example.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Selling-Url-Source-Cli.md`

## Mechanism

`parse_default_fields_with_booking_url_or_exit(values, booking_url=...)` reuses
the existing `parse_default_fields_or_exit` validation, trims the booking URL,
and merges it into a nested `selling` mapping:

```python
{"selling": {"booking_url": "https://example.test/book"}}
```

If existing defaults already include a mapping at `selling`, the helper
preserves it and adds or replaces only `booking_url`. Blank booking URLs are
ignored. Generic source-row CLIs pass the resulting mapping into the existing
`load_source_campaign_opportunities_from_file(..., default_fields=...)` seam,
so row values and existing source normalization precedence stay unchanged.

## Intentional

- No core generation change: PR-Content-Selling-Context-Inputs already made
  generation consume `selling.booking_url` when it appears on opportunities.
- No nested `--default-field selling.booking_url=...` parser. The product only
  needs a clear operator flag for the high-value selling URL path.
- The helper lives in the source adapter module because the value is applied as
  source-row fallback metadata before opportunity normalization.

## Deferred

- Adding `--booking-url` to every Postgres/source smoke command remains a
  follow-up if operators need it. This slice covers the generic builder and
  offline generation commands that host users are most likely to run directly.
- Richer selling context flags such as sender identity or published asset URLs
  remain deferred until a host asks for them.

## Verification

- `pytest tests/test_extracted_campaign_source_adapters.py tests/test_extracted_campaign_generation_example.py tests/test_smoke_content_ops_review_source_postgres.py` -> 99 passed
- `python -m py_compile extracted_content_pipeline/campaign_source_adapters.py scripts/build_extracted_campaign_opportunities_from_sources.py scripts/run_extracted_campaign_generation_example.py scripts/smoke_content_ops_review_source_postgres.py tests/test_extracted_campaign_source_adapters.py tests/test_extracted_campaign_generation_example.py tests/test_smoke_content_ops_review_source_postgres.py` -> passed
- `bash scripts/validate_extracted_content_pipeline.sh` -> passed
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -> passed
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -> passed
- `bash scripts/check_ascii_python.sh` -> passed
- `bash scripts/local_pr_review.sh` -> passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Production + scripts | ~60 |
| Tests | ~55 |
| Docs + plan + coordination | ~90 |
| **Total** | **~205** |

Under the 400 LOC review budget.
