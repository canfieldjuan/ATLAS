# Deflection Mxbai Merge Spot-Check

Date: 2026-06-14

Issue: #1504

## What Ran

This validation reran the live CFPB fees sample with the pinned local mxbai
embedding path and the semantic-edge recorder enabled by this slice.

```bash
python scripts/smoke_content_ops_cfpb_faq_markdown.py \
  --search-term fees \
  --limit 1000 \
  --max-rows-scanned 5000 \
  --max-items 20 \
  --support-contact "https://support.example.com" \
  --compare-embedding-booster \
  --output-source-rows tmp/cfpb_mxbai_merge_spotcheck/cfpb_sources_1000.jsonl \
  --output-markdown tmp/cfpb_mxbai_merge_spotcheck/cfpb_faq_1000_boosted.md \
  --json > tmp/cfpb_mxbai_merge_spotcheck/summary_raw.json
```

## Proof Artifact

- Sanitized summary:
  `docs/extraction/validation/fixtures/deflection_mxbai_merge_spotcheck_20260614/summary.json`
- Excluded raw working files:
  `tmp/cfpb_mxbai_merge_spotcheck/cfpb_sources_1000.jsonl`,
  `tmp/cfpb_mxbai_merge_spotcheck/cfpb_faq_1000_boosted.md`, and
  `tmp/cfpb_mxbai_merge_spotcheck/summary_raw.json`

## Result

| Metric | Value |
|---|---:|
| Exit status | 0 |
| Live CFPB source rows | 1,000 |
| Embedding probe calls | 14 |
| Embedding valid batches | 14 |
| Accepted semantic merge edges | 9 |
| Non-repeat ticket delta | -18 |
| Spot-checked plausible same-issue edges | 9 |
| Spot-checked unclear edges | 0 |
| Spot-checked likely false merges | 0 |

The accepted semantic edges are the same mutual-nearest-neighbor pairs the
booster unioned. All 9 edges looked like plausible same-issue merges in this
sample, including mortgage servicing transfer, health-related card-payment
hardship, travel booking disputes, rental scam recovery, payday loan wording,
negative-balance charges, credit-card fraud disputes, recurring checking fees,
and rent-payment transfer loss.

## Launch Implication

This removes the review's immediate merge-quality blocker for the 18-ticket
lift observed in #1544. It still does not flip
`ATLAS_CONTENT_OPS_FAQ_EMBEDDING_BOOSTER_ENABLED`: the operator still owns the
ROI decision for enabling a small CFPB lift with a local CPU mxbai dependency.
