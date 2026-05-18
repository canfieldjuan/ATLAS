# PR: Content Ops Review Source Readiness

## Why this slice exists

PR #588 and #589 made G2 review evidence usable for AI Content Ops source-row generation. A live Trustpilot check showed the next risk: a review source can have thousands of enriched rows but still export zero usable source rows when v4 quote-grade phrase metadata is missing.

This slice adds a small readiness summary to the existing review-source exporter so operators can choose G2, Capterra, TrustRadius, or Trustpilot based on quote-grade availability before running an export.

## Scope (this PR)

1. Add a read-only source-readiness summary mode to the existing review-source exporter.
2. Count canonical enriched rows, export-candidate rows, and quote-grade exportable rows per source.
3. Document the readiness check before the G2/source-row export workflow.

### Files touched

- `scripts/export_content_ops_review_sources.py`
- `tests/test_export_content_ops_review_sources.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Review-Source-Readiness.md`

## Mechanism

- Add a summary SQL builder that groups requested review sources and checks the same canonical/enriched/text/url/quote-grade conditions used by export.
- Add `--source-summary` plus `--summary-sources` to print JSON readiness rows without writing source-row JSONL.
- Keep normal export behavior unchanged.

## Intentional

- This does not loosen quote-grade export policy.
- This does not make Trustpilot exportable when phrase metadata is absent.
- This does not add a second exporter script.

## Deferred

- Re-enriching Trustpilot rows with v4 phrase metadata.
- Adding browser UI for source-readiness diagnostics.
- Adding new non-review source families.

## Verification

- python -m pytest tests/test_export_content_ops_review_sources.py -q -> 12 passed.
- python -m py_compile scripts/export_content_ops_review_sources.py tests/test_export_content_ops_review_sources.py -> passed.
- python scripts/export_content_ops_review_sources.py --source-summary --summary-sources g2,capterra,trustradius,trustpilot -> passed; reported quote_grade_rows: g2=364, capterra=154, trustradius=31, trustpilot=0.
- python scripts/export_content_ops_review_sources.py --source capterra --limit 3 --output /tmp/capterra_content_ops_sources.jsonl -> exported 3 rows.
- git diff --check -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Exporter summary mode | ~140 |
| Tests | ~80 |
| Docs and coordination | ~80 |
| Total | ~300 |
