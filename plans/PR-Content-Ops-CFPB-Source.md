# PR-Content-Ops-CFPB-Source

## Why this slice exists

AI Content Ops can already normalize source rows such as support tickets,
complaints, conversations, and cases, but the review-source smoke still proves
only scraped review rows. We need a real public support-ticket-like dataset
without coupling seller/product context to the source evidence. CFPB complaint
narratives are public, stable, and can be represented as support-ticket source
rows while keeping target account/contact fields supplied separately by host
defaults.

This slice exceeds the 400 LOC target because the exporter, DB-backed smoke,
tests, and host docs are indivisible for a useful end-to-end proof. Shipping
only the exporter would leave the support-ticket ingestion path unproven.

## Scope (this PR)

1. Add a read-only CFPB complaint exporter that writes Content Ops source-row
   JSONL.
2. Add a Postgres/offline smoke that fetches CFPB rows, inspects ingestion,
   imports opportunities, and generates deterministic drafts.
3. Document CFPB as a real support-ticket-like ingestion source.

### Files touched

- `scripts/export_content_ops_cfpb_sources.py`
- `scripts/smoke_content_ops_cfpb_source_postgres.py`
- `tests/test_export_content_ops_cfpb_sources.py`
- `tests/test_smoke_content_ops_cfpb_source_postgres.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-CFPB-Source.md`

## Mechanism

The exporter reads CFPB's public complaint CSV endpoint, stops after enough
narrative-bearing rows are collected, and emits source rows with:

```python
source_type = "support_ticket"
source_system = "cfpb"
vendor_name = row["Company"]
text = row["Consumer complaint narrative"]
pain_category = row["Issue"]
```

The smoke reuses existing source-row ingestion, `campaign_opportunities`
import, and offline `generate_campaign_drafts_from_postgres` paths. Target
account/contact fields continue to come from `--default-field`, not CFPB data.

## Intentional

- CFPB rows are support-ticket-like public complaint evidence, not seller-side
  offer data.
- The exporter uses standard-library HTTP/CSV only; no new dependency.
- Live provider generation is not part of this PR. Offline generation is enough
  to prove the ingestion seam without spending provider tokens.
- The stale merged `PR-LLM-Registry-Service-Export` coordination row is removed
  while claiming this slice.

## Deferred

- Quality gates for unsupported claims and CTA policy stay separate.
- A second non-commercial support-tweet adapter can follow after this CFPB path
  is proven.

## Verification

- `pytest tests/test_export_content_ops_cfpb_sources.py tests/test_smoke_content_ops_cfpb_source_postgres.py tests/test_extracted_campaign_source_adapters.py -q` -> 68 passed.
- `python -m py_compile scripts/export_content_ops_cfpb_sources.py scripts/smoke_content_ops_cfpb_source_postgres.py tests/test_export_content_ops_cfpb_sources.py tests/test_smoke_content_ops_cfpb_source_postgres.py` -> passed.
- `bash scripts/check_ascii_python.sh` -> passed.
- `grep -nP '[^\x00-\x7F]' <edited files>` -> no matches.
- `bash scripts/local_pr_review.sh` -> passed.
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1413 passed for `extracted_content_pipeline`, 295 passed for `extracted_reasoning_core`, 1 existing torch/pynvml warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| CFPB exporter | ~275 |
| CFPB Postgres smoke | ~505 |
| Tests | ~485 |
| CI runner + docs + coordination | ~160 |
| **Total** | **~1425** |
