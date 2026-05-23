# Content Ops FAQ 50K Gated Validation - 2026-05-23

## Summary

The FAQ scale smoke was rerun against the existing real CFPB-derived 50,000-row
support-ticket source artifact using the scale gates added in
PR-Content-Ops-FAQ-Scale-Gates.

Result: passed.

- smoke exit code: `0`
- FAQ CLI exit code: `0`
- raw source rows: `50,000`
- accepted ticket source rows: `50,000`
- rendered ticket source rows: `50,000`
- generated FAQ items: `12`
- output checks: `3 / 3` passed
- normalization warnings: `0`
- failure: `null`

This proves the deterministic generator can produce bounded FAQ Markdown from
50,000 real support-ticket-shaped rows while enforcing the row-volume and
source-coverage promises by script exit code.

## Source

- Source artifact:
  `tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl`
- Source size: `102M`
- Source line count: `50,000`
- Source type: `support_ticket`
- Defaults used to suppress CFPB fixture noise:
  - `company_name=CFPB`
  - `contact_email=cfpb-public-archive@example.invalid`
  - `vendor_name=CFPB`

## Command

```bash
/usr/bin/time -v -o tmp/faq_scale_stress_20260523/scale_50000_gated_time.txt \
  python scripts/smoke_content_ops_faq_scale_run.py \
  tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl \
  --source-format jsonl \
  --artifact-dir tmp/faq_scale_stress_20260523/scale_50000_gated \
  --title 'CFPB 50,000 Row FAQ Gated Scale Validation 2026-05-23' \
  --max-items 12 \
  --max-evidence-per-item 5 \
  --max-text-chars 1200 \
  --default-field company_name=CFPB \
  --default-field contact_email=cfpb-public-archive@example.invalid \
  --default-field vendor_name=CFPB \
  --min-raw-source-rows 50000 \
  --min-ticket-source-rows 50000 \
  --require-all-ticket-sources-rendered
```

Console result:

```text
Content Ops FAQ scale smoke passed: input_status=ok source_rows=50000/50000 faq=available generated=12 weighted_volume=50000 checks_failed=0/3 score_max=137635 summary=tmp/faq_scale_stress_20260523/scale_50000_gated/run_summary.json
```

## Gate Results

From `tmp/faq_scale_stress_20260523/scale_50000_gated/run_summary.json`:

| Gate | Expected | Actual | Result |
|---|---:|---:|---|
| `min_raw_source_rows` | 50,000 | 50,000 | passed |
| `min_ticket_source_rows` | 50,000 | 50,000 | passed |
| `all_ticket_sources_rendered` | 50,000 | 50,000 | passed |

The full `scale_gates` payload reported:

- `configured=true`
- `passed=true`
- `failed=[]`

## FAQ Summary

From the same run summary:

| Metric | Value |
|---|---:|
| `source_count` | 50,000 |
| `ticket_source_count` | 50,000 |
| `weighted_source_volume` | 50,000 |
| `source_channel_counts.support_tickets` | 50,000 |
| `generated` | 12 |
| `output_checks.passed` | 3 |
| `output_checks.failed` | 0 |
| `warning_count` | 0 |
| `item_score_distribution.max` | 137,635 |
| `item_score_distribution.min` | 980 |
| `item_score_distribution.average` | 20,833.33 |

The generated Markdown remained bounded at `29K` because `max_items=12` and
`max_evidence_per_item=5` cap the rendered artifact even when the source input
is large.

## Timing And Memory

From `tmp/faq_scale_stress_20260523/scale_50000_gated_time.txt`:

| Metric | Value |
|---|---:|
| Wall time | `1:41.86` |
| User CPU | `100.95s` |
| System CPU | `0.86s` |
| CPU utilization | `99%` |
| Maximum RSS | `592,764 KB` |
| Exit status | `0` |

## Artifacts

Generated local artifacts:

| Artifact | Path | Size |
|---|---|---:|
| Markdown | `tmp/faq_scale_stress_20260523/scale_50000_gated/faq.md` | `29K` |
| Result JSON | `tmp/faq_scale_stress_20260523/scale_50000_gated/faq_result.json` | `16K` |
| Run summary | `tmp/faq_scale_stress_20260523/scale_50000_gated/run_summary.json` | `21K` |
| Time output | `tmp/faq_scale_stress_20260523/scale_50000_gated_time.txt` | `1.2K` |

These artifacts are intentionally not checked in.

## Issues Surfaced

No new correctness issue surfaced in this gated run.

The operational concern from the earlier stress probe still stands: a
50,000-row synchronous run takes `1:41.86` and about `593 MB` RSS on this host.
That is acceptable as an operator validation run, but hosted production paths
still need explicit limits, backpressure, or background job execution before
large uploads are exposed as synchronous customer requests.
