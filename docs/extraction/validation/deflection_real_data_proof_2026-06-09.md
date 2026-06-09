# Deflection Real-Data Proof - CFPB + Messy CSV

Date: 2026-06-09

Issue: #1408

## What Ran

This validation used the local CFPB source-row exports already present under
`/home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/`.

Large CFPB report runs:

```bash
python scripts/build_content_ops_deflection_report.py \
  /home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl \
  --source-format jsonl \
  --result-output /tmp/deflection-real-data-proof/cfpb_10000/result.json \
  --summary-output /tmp/deflection-real-data-proof/cfpb_10000/summary.json \
  --output /tmp/deflection-real-data-proof/cfpb_10000/report.md \
  --require-output-checks

python scripts/build_content_ops_deflection_report.py \
  /home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl \
  --source-format jsonl \
  --result-output /tmp/deflection-real-data-proof/cfpb_50000/result.json \
  --summary-output /tmp/deflection-real-data-proof/cfpb_50000/summary.json \
  --output /tmp/deflection-real-data-proof/cfpb_50000/report.md \
  --require-output-checks
```

Messy CSV derivative run:

```bash
python scripts/build_deflection_messy_csv_fixtures.py \
  /home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl \
  --output-dir /tmp/deflection-real-data-proof/messy_cfpb \
  --limit 8 \
  --json
```

## CFPB Quality And Scale

| Input | Status | Runtime | Max RSS | Rows | Ranked questions | Top question | Top ticket count | Singleton ranked items |
|---|---|---:|---:|---:|---:|---|---:|---:|
| CFPB 10k JSONL | Passed output checks | 14.74s | 144,056 KB | 10,000 | 33 | What should I do if information on my credit report is wrong? | 5,381 | 7 |
| CFPB 50k JSONL | Passed output checks | 74.36s | 619,932 KB | 50,000 | 39 | What should I do if information on my credit report is wrong? | 27,527 | 3 |

Quality findings:

- Clustering is no longer singleton-dominated on CFPB. The 50k run grouped the
  largest credit-report dispute cluster at 27,527 rows, plus billing/payments,
  debt collection disputes, mortgage servicing, and account-management clusters.
- The generated ranked questions are sensible for complaint data. The top five
  50k questions map to recognizable support topics: credit report disputes,
  billing/payments, debt collection disputes, mortgage servicing, and managing
  an account.
- All generated answers are `draft_needs_review`: 33 of 33 for 10k and 39 of 39
  for 50k. This is correct for CFPB because the corpus has complaint narratives
  but no uploaded resolution evidence. The report should not invent publishable
  answers from complaint text alone.
- Scale is acceptable for local deterministic generation. The 50k run completed
  in 74.36 seconds and stayed under 620 MB max RSS. The generated 50k Markdown
  artifact was 764,398 bytes.

Launch implication:

The report quality is usable as an inspect/preview signal on real untagged
complaint text, but CFPB proves the "no proven answer yet" lane more than the
publishable-answer lane. A provider export with actual agent resolutions is
still needed before claiming the full paid report generates publishable drafts
from real customer data.

## Messy CSV Outcomes

The messy CSV generator produced eight bounded fixtures from CFPB source rows.
Each case was parsed through
`load_source_campaign_opportunities_from_file(..., file_format="csv")`; parsed
cases were also passed into `build_ticket_faq_markdown(...)`.

| Case | Expected | Outcome | Usable rows | Notes |
|---|---|---|---:|---|
| `bom_utf8.csv` | parsed | parsed | 4 | UTF-8 BOM header parsed. |
| `cp1252_semicolon.csv` | parsed | parsed | 4 | cp1252 text and semicolon delimiter parsed. |
| `tab_delimited.csv` | parsed | parsed | 4 | Tab delimiter parsed. |
| `html_bodies.csv` | parsed | parsed | 4 | HTML body rows parsed, but raw tags can still reach clustered source text; the small FAQ render did not expose them. |
| `leading_metadata_row.csv` | fail_loud | failed_loud | 0 | Raised `ValueError: CSV row 2 has more cells than the header; check the delimiter and header row.` |
| `ragged_extra_cells.csv` | fail_loud | failed_loud | 0 | Raised `ValueError: CSV row 2 has more cells than the header; check the delimiter and header row.` |
| `ragged_short_rows.csv` | parsed_partial | parsed_partial | 1 | Did not crash; missing-body row surfaced as `missing_source_text` and was excluded from usable opportunities. |
| `quoted_multiline.csv` | parsed | parsed | 3 | Quoted multiline body with delimiters parsed. |

The small CFPB-derived parsed cases failed only
`resolution_evidence_scoped` when passed directly to FAQ Markdown generation,
because the selected CFPB rows do not contain verified resolution evidence.
That is the same quality signal observed in the large CFPB report runs and is
not an ingestion crash.

## Follow-Up

- Run the complete hosted upload -> snapshot -> full report -> email cycle with
  a real or sanitized provider export that includes agent resolutions. CFPB does
  not prove publishable-answer quality.
- Strip or normalize HTML before clustering/evidence for HTML-heavy provider
  exports; this fixture proves the parser does not crash, not that tags are
  fully removed from every downstream representation.
- If future real provider exports show high singleton counts despite #1410's
  corpus-derived anchors, improve deterministic clustering before paid traffic.
