# Content Ops CFPB FAQ Live Smoke

## Why this slice exists

The CFPB exporter can fetch real public complaint narratives, and the FAQ
Markdown builder can render grounded article-style answers. Operators still
need one command that proves those two seams work together: live CFPB source
rows in, source-agnostic FAQ Markdown out, with the FAQ output checks enforced.

This closes the gap between exporting CFPB rows and showing an actual grounded
FAQ artifact from real public support-ticket-like data.

## Scope (this PR)

1. Add a host-facing CFPB FAQ Markdown smoke command.
2. Keep CFPB-specific logic at the source-export boundary; FAQ generation
   continues to consume generic source rows.
3. Add tests for successful Markdown generation, too-few-row failure,
   output-check failure, and JSON stdout behavior without live network calls.
4. Document the smoke command in the Content Ops README and status notes.
5. Replace the merged FAQ narrative coordination row with this slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-CFPB-FAQ-Live-Smoke.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Claim this CFPB FAQ smoke slice. |
| `scripts/smoke_content_ops_cfpb_faq_markdown.py` | New live-source to FAQ Markdown smoke command. |
| `tests/test_smoke_content_ops_cfpb_faq_markdown.py` | Smoke command regression tests with fake CFPB fetches. |
| `scripts/run_extracted_pipeline_checks.sh` | Include the new smoke tests in the extracted gauntlet. |
| `extracted_content_pipeline/README.md` | Add the CFPB FAQ Markdown smoke example. |
| `extracted_content_pipeline/STATUS.md` | Record the new live-source FAQ smoke capability. |

## Mechanism

The new smoke command imports the existing CFPB exporter and calls its fetch
function with the same filter and request-header options already used by the
source-row exporter. It writes fetched rows to JSONL, loads that file through
the generic source-row adapter, and calls the existing FAQ Markdown builder.

The smoke fails closed when fewer rows are fetched than requested, the FAQ
builder produces no items, or any FAQ output check is false. JSON mode keeps
stdout machine-readable on both success and failure.

## Intentional

- No CFPB-specific branch is added to the FAQ renderer.
- No LLM, database, or provider dependency is added.
- Tests monkeypatch the CFPB fetch function instead of calling the public CFPB
  endpoint in CI.

## Deferred

- A hosted UI button for generating a FAQ from a CFPB sample remains separate
  UI work.
- Live network execution remains an operator command, not a CI requirement.

## Verification

- pytest tests/test_smoke_content_ops_cfpb_faq_markdown.py - 4 passed.
- python -m py_compile scripts/smoke_content_ops_cfpb_faq_markdown.py tests/test_smoke_content_ops_cfpb_faq_markdown.py - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- Local PR review: pending.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-CFPB-FAQ-Live-Smoke.md` | 70 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `scripts/smoke_content_ops_cfpb_faq_markdown.py` | 160 |
| `tests/test_smoke_content_ops_cfpb_faq_markdown.py` | 120 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `extracted_content_pipeline/README.md` | 12 |
| `extracted_content_pipeline/STATUS.md` | 4 |
| **Total** | **371** |
