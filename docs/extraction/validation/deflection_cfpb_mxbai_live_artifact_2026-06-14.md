# Deflection CFPB Mxbai Live Artifact

Date: 2026-06-14

Issue: #1504

## What Ran

This validation ran live CFPB source rows through the #1542 compare harness with
the host mxbai embedding adapter enabled. The raw CFPB JSONL and full generated
Markdown were written under `tmp/` and are intentionally not committed.

```bash
python scripts/smoke_content_ops_cfpb_faq_markdown.py \
  --search-term fees \
  --limit 1000 \
  --max-rows-scanned 5000 \
  --max-items 20 \
  --support-contact "https://support.example.com" \
  --compare-embedding-booster \
  --output-source-rows tmp/cfpb_mxbai_live_artifact/cfpb_sources_1000.jsonl \
  --output-markdown tmp/cfpb_mxbai_live_artifact/cfpb_faq_1000_boosted.md \
  --json > tmp/cfpb_mxbai_live_artifact/summary_raw.json
```

## Proof Artifacts

- Sanitized summary:
  `docs/extraction/validation/fixtures/deflection_cfpb_mxbai_live_artifact_20260614/summary.json`
- Excluded raw working files:
  `tmp/cfpb_mxbai_live_artifact/cfpb_sources_1000.jsonl`,
  `tmp/cfpb_mxbai_live_artifact/cfpb_faq_1000_boosted.md`, and
  `tmp/cfpb_mxbai_live_artifact/summary_raw.json`

## Result

| Metric | Value |
|---|---:|
| Exit status | 0 |
| Live CFPB source rows | 1,000 |
| Usable source ratio | 1.0 |
| Embedding comparison primary | boosted |
| Embedding probe calls | 14 |
| Embedding valid batches | 14 |
| Baseline ranked questions | 20 |
| Boosted ranked questions | 20 |
| Baseline non-repeat tickets | 843 |
| Boosted non-repeat tickets | 825 |
| Non-repeat ticket delta | -18 |
| Non-repeat ticket delta share | 1.8% |
| Top question changed | false |
| Output checks passed | true |

The run used the host factory pinned to
`mixedbread-ai/mxbai-embed-large-v1` at revision
`b33106f585b9ce46904ad7443a3b52b7a63e231c`, CPU, local-files-only.

## Interpretation

The live CFPB path is not a stub: the host adapter constructed the pinned mxbai
port, inference produced 14 valid batches, and the compare harness selected the
boosted result with no errors. On this 1,000-row CFPB fees sample, the semantic
booster reduced the non-repeat ticket count by 18 while leaving the top question
unchanged. That is a small lift, consistent with the earlier #1504 note that
CFPB's dense financial vocabulary leaves less semantic-only long-tail inventory
than more varied support corpora.

This artifact does not approve production enablement. It records no threshold
change, because the current relative defaults produced a bounded delta and all
output checks passed. It also confirms presentation debt remains: duplicate
safe labels still make the ranked list look repetitive, and customer-wording
labels can still surface weak raw phrases. Those are the existing follow-up
items in the deflection/clustering lane, not changes bundled into this
validation PR.
