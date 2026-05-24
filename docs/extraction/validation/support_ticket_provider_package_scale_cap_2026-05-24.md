# Support-Ticket Provider Package Scale Cap Validation - 2026-05-24

## Summary

This validation proves the support-ticket input package keeps generation inputs
bounded before any DB write or LLM call. A 1,000-row source file packages all
rows. A 10,000-row source file keeps the original count visible, includes the
first 1,000 rows in `source_material`, and reports the remaining 9,000 rows as
truncated.

Ownership lane: `content-ops/support-ticket-input-provider`

## Source Artifacts

The run used existing ignored local CFPB-derived support-ticket-shaped JSONL
artifacts:

- `/home/juan-canfield/Desktop/Atlas/tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl`
- `/home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl`

These artifacts were previously used by the FAQ scale validation trail. They
are not committed because they are large local validation inputs.

## Command

```bash
/usr/bin/time -v python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)

paths = [
    Path('/home/juan-canfield/Desktop/Atlas/tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl'),
    Path('/home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl'),
]
results = []
for path in paths:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    started = perf_counter()
    package = build_support_ticket_input_package(rows)
    elapsed = perf_counter() - started
    inputs = package.inputs
    results.append({
        'path': str(path),
        'loaded_rows': len(rows),
        'source_row_count': inputs['source_row_count'],
        'included_ticket_row_count': inputs['included_ticket_row_count'],
        'skipped_ticket_row_count': inputs['skipped_ticket_row_count'],
        'truncated_ticket_row_count': inputs['truncated_ticket_row_count'],
        'source_material_count': len(inputs['source_material']),
        'source_period': inputs['source_period'],
        'has_window_filter': 'faq_window_days' in inputs,
        'question_like_ticket_count': inputs['question_like_ticket_count'],
        'top_ticket_clusters': inputs['top_ticket_clusters'][:3],
        'customer_wording_example_count': len(inputs['customer_wording_examples']),
        'warning_count': len(package.warnings),
        'warnings': list(package.warnings),
        'elapsed_seconds': round(elapsed, 4),
    })
print(json.dumps(results, indent=2, sort_keys=True))
PY
```

## Results

| Source rows | Included rows | Truncated rows | Skipped rows | Source period | Package time |
|---:|---:|---:|---:|---|---:|
| 1,000 | 1,000 | 0 | 0 | `Uploaded support tickets` | 0.3195s |
| 10,000 | 1,000 | 9,000 | 0 | `Uploaded support tickets` | 0.3061s |

The 10,000-row run emitted one package warning:

```json
{
  "code": "ticket_rows_truncated",
  "max_rows": 1000,
  "message": "Used first 1000 ticket rows out of 10000.",
  "row_count": 10000,
  "truncated_row_count": 9000
}
```

Both runs preserved representative package context:

- `source_material_count`: `1,000`
- `customer_wording_example_count`: `6`
- top clusters included `Incorrect information on your report`, `Attempts to
  collect debt not owed`, and `Problem with a credit reporting company's
  investigation into an existing problem`

Resource use for the combined 1,000-row and 10,000-row package validation:

- elapsed wall time: `0:00.78`
- maximum resident set size: `93,312 KB`
- exit status: `0`

## Interpretation

The provider package is bounded before generation. Oversized source exports keep
their original row count visible for diagnostics, but only 1,000 rows enter
`source_material` by default. That protects landing-page and blog prompt inputs
from accidentally consuming 10,000+ support tickets in one generation request.

The customer-facing question remains product policy: hosted upload/intake should
decide whether to show the truncation warning to users, lower the synchronous
limit, or route larger files into a background job path.
