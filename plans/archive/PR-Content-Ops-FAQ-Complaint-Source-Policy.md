# Content Ops FAQ Complaint Source Policy

## Why this slice exists

A local CFPB complaint export at `/home/juan-canfield/Downloads/archive (1)/rows.csv` exposed two source-level gaps in the generic FAQ path:

1. Raw complaint exports with provider-style fields such as `Consumer complaint narrative`, `Complaint ID`, and `Issue` do not normalize cleanly through the generic source adapter without a CFPB-specific conversion step.
2. CFPB-like financial complaint rows can be misclassified by generic B2B/support keywords (`report`, `account`, `payment`) and receive wrong next steps such as dashboard export or profile-setting guidance.

This slice fixes those issues at the generic source/FAQ policy layer so CFPB, bank complaint, help desk complaint, and similar source rows improve together.

## Scope (this PR)

1. Extend generic source-row aliases for complaint narratives and complaint IDs.
2. Preserve `issue`-style fields as pain points so FAQ grouping can use the source's own issue taxonomy.
3. Add generic financial complaint/debt/credit-report FAQ intent and action guidance before narrower B2B/support rules.
4. Add regression tests using CFPB-shaped rows, without adding CFPB-specific branches to the FAQ renderer.
5. Log this active slice in the coordination ledger.

### Files touched

- `plans/PR-Content-Ops-FAQ-Complaint-Source-Policy.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

The source adapter already normalizes provider-style headings by comparing normalized and compact field names. Adding `complaint_id`, `consumer_complaint_narrative`, `complaint_narrative`, and `issue` to the existing generic alias tables lets raw complaint CSV/JSON rows become normal source opportunities.

The FAQ renderer keeps the same `intent_rules` mechanism but orders broad financial complaint categories before generic reporting/account rules. The action guidance remains deterministic and extractive: debt/credit-report/payment issues get dispute/documentation/credit-report steps instead of SaaS dashboard/profile steps.

## Intentional

- No CFPB-specific renderer branch. CFPB remains one source producer; FAQ generation stays generic.
- No new live CFPB dependency in tests. Tests use small CFPB-shaped fixtures so CI is stable.
- No source adapter signature changes. Hosts can keep using existing CSV/JSON/JSONL ingestion.
- The downloaded 737 MB dataset is not committed. It is local validation data only.

## Deferred

- A frozen, real CFPB fixture can land later if we want repeatable large-sample quality tracking in CI.
- More financial-domain FAQ action packs can follow if another real dataset exposes distinct categories not covered here.
- Live CFPB API retry/fallback remains separate from local dataset processing because this slice fixes local/raw-source quality.

## Verification

Commands run:

```bash
pytest tests/test_extracted_campaign_source_adapters.py tests/test_extracted_ticket_faq_markdown.py -q  # 112 passed
python -m py_compile extracted_content_pipeline/campaign_source_adapters.py extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_campaign_source_adapters.py tests/test_extracted_ticket_faq_markdown.py  # passed
python scripts/build_extracted_ticket_faq_markdown.py /tmp/cfpb_raw_debt_collection_50.csv --source-format csv --title 'Debt Collection Complaint FAQ' --max-items 8 --max-evidence-per-item 3 --support-contact 'https://example.com/support' --require-output-checks --output /tmp/cfpb_raw_debt_collection_50_faq.md  # passed
bash scripts/validate_extracted_content_pipeline.sh  # passed
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline  # passed
python scripts/audit_extracted_standalone.py --fail-on-debt  # passed
bash scripts/check_ascii_python.sh  # passed
bash scripts/local_pr_review.sh  # passed
bash scripts/run_extracted_pipeline_checks.sh  # 1554 passed, 1 existing torch/pynvml warning
```

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Complaint-Source-Policy.md` | 65 |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | 10 |
| `docs/extraction/coordination/inflight.md` | 1 |
| `extracted_content_pipeline/campaign_source_adapters.py` | 8 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 70 |
| `tests/test_extracted_campaign_source_adapters.py` | 26 |
| `tests/test_extracted_ticket_faq_markdown.py` | 60 |
| **Total** | **240** |
