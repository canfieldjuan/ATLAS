# Support-Ticket Package CLI Scale Validation - 2026-05-24

## Summary

This validation reruns the support-ticket package scale proof through the
operator-facing package-smoke CLI merged in #934. It confirms the CLI reports the
same bounded package behavior as the direct package proof from #935:

- 1,000 source rows package fully.
- 10,000 source rows keep the honest total count visible, include 1,000 rows in
  generation inputs, and report 9,000 truncated rows.

Ownership lane: `content-ops/support-ticket-input-provider`

## Source Artifacts

The run used existing ignored local CFPB-derived support-ticket-shaped JSONL
artifacts:

- `/home/juan-canfield/Desktop/Atlas/tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl`
- `/home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl`

CLI JSON summaries were written under ignored local `tmp/`:

- `tmp/support_ticket_package_cli_scale_20260524/cfpb_1000_package_summary.json`
- `tmp/support_ticket_package_cli_scale_20260524/cfpb_10000_package_summary.json`

The source and summary artifacts are not committed because they are local
validation artifacts. The commands and observed metrics are recorded below.

## Commands

```bash
mkdir -p tmp/support_ticket_package_cli_scale_20260524

/usr/bin/time -v python scripts/smoke_content_ops_support_ticket_package.py \
  /home/juan-canfield/Desktop/Atlas/tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl \
  --pretty \
  --require-included-rows \
  > tmp/support_ticket_package_cli_scale_20260524/cfpb_1000_package_summary.json

/usr/bin/time -v python scripts/smoke_content_ops_support_ticket_package.py \
  /home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl \
  --pretty \
  --require-included-rows \
  > tmp/support_ticket_package_cli_scale_20260524/cfpb_10000_package_summary.json
```

## Results

| Source file | Source rows | Included rows | Truncated rows | Skipped rows | Warnings | Wall time | Max RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cfpb_1000_source_rows.jsonl` | 1,000 | 1,000 | 0 | 0 | 0 | 0:00.36 | 22,728 KB |
| `cfpb_10000_source_rows.jsonl` | 10,000 | 1,000 | 9,000 | 0 | 1 | 0:00.48 | 87,972 KB |

Both runs reported:

- `source_period`: `Uploaded support tickets`
- `has_window_filter`: `false`
- `question_like_ticket_count`: `58`
- `customer_wording_example_count`: `6`
- top clusters:
  - `Incorrect information on your report` (`258`)
  - `Attempts to collect debt not owed` (`133`)
  - `Problem with a credit reporting company's investigation into an existing problem` (`130`)

The 10,000-row CLI summary emitted the expected truncation warning:

```json
{
  "code": "ticket_rows_truncated",
  "max_rows": 1000,
  "message": "Used first 1000 ticket rows out of 10000.",
  "row_count": 10000,
  "truncated_row_count": 9000
}
```

## Interpretation

The shipped package-smoke CLI is suitable as a cheap pre-LLM validation step for
large support-ticket exports. It catches and reports the same bounded package
behavior as the direct package path: operators can see the original source row
count and truncation warning before spending model calls, while generation
inputs remain capped at 1,000 rows by default.

This does not replace hosted upload policy. The hosted intake path still needs a
product decision on whether to show truncation warnings to users, reject larger
files, or move oversized work into a background job.
