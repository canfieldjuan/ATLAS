# Content Ops FAQ Scale Stress Probe - 2026-05-23

## Summary

The deterministic FAQ generator and Postgres-backed FAQ lifecycle were stress
tested with real CFPB-derived support-ticket source rows at 2,000, 5,000,
10,000, 25,000, and 50,000 rows.

Correctness held through 50,000 rows:

- deterministic generator exit code: `0` at every tested size
- database lifecycle exit code: `0` at every tested size
- output checks: `condensed=true`, `has_action_items=true`,
  `uses_user_vocabulary=true`
- normalization warnings: `0`
- lifecycle saved one FAQ draft, exported the draft, updated it to `published`,
  and exported the reviewed row

The first weaknesses are operational, not FAQ correctness:

- the 50,000-row run is too slow and memory-heavy for a synchronous
  request/response path
- the real Postgres-backed lifecycle starts failing under high concurrency when
  connection slots saturate

Those hardening items are parked in `HARDENING.md`.

## Source Data

- Archive: `/home/juan-canfield/Downloads/archive (1)/rows.csv`
- Extractor: `scripts/export_content_ops_cfpb_sources.py`
- Source type: `support_ticket`
- Metadata defaults used to suppress CFPB fixture noise:
  - `company_name=CFPB`
  - `contact_email=cfpb-public-archive@example.invalid`
  - `vendor_name=CFPB`

Generated local fixtures:

| Rows | Source artifact | Rows scanned | Size |
|---:|---|---:|---:|
| 2,000 | `tmp/faq_scale_stress_20260523/cfpb_2000_source_rows.jsonl` | 69,155 | 3.8 MB |
| 5,000 | `tmp/faq_scale_stress_20260523/cfpb_5000_source_rows.jsonl` | 69,155 | 10.1 MB |
| 10,000 | `tmp/faq_scale_stress_20260523/cfpb_10000_source_rows.jsonl` | 69,155 | 20.2 MB |
| 25,000 | `tmp/faq_scale_stress_20260523/cfpb_25000_source_rows.jsonl` | 102,311 | 50.7 MB |
| 50,000 | `tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl` | 156,985 | 101.8 MB |

## Deterministic Generator Results

Command shape:

```bash
/usr/bin/time -v -o tmp/faq_scale_stress_20260523/scale_<rows>_time.txt \
  python scripts/smoke_content_ops_faq_scale_run.py \
  tmp/faq_scale_stress_20260523/cfpb_<rows>_source_rows.jsonl \
  --source-format jsonl \
  --artifact-dir tmp/faq_scale_stress_20260523/scale_<rows> \
  --title 'CFPB <rows> Row FAQ Scale Stress' \
  --max-items 12 \
  --max-evidence-per-item 5 \
  --max-text-chars 1200 \
  --default-field company_name=CFPB \
  --default-field contact_email=cfpb-public-archive@example.invalid \
  --default-field vendor_name=CFPB
```

| Rows | Exit | Generated | Failed checks | Elapsed | Max RSS |
|---:|---:|---:|---|---:|---:|
| 2,000 | 0 | 12 | `[]` | 0:03.94 | 36,040 KB |
| 5,000 | 0 | 12 | `[]` | 0:10.02 | 66,136 KB |
| 10,000 | 0 | 12 | `[]` | 0:20.32 | 138,724 KB |
| 25,000 | 0 | 12 | `[]` | 0:51.68 | 321,148 KB |
| 50,000 | 0 | 12 | `[]` | 1:49.48 | 590,612 KB |

The generated Markdown stayed around 29 KB because `max_items=12` and
`max_evidence_per_item=5` bound the output size. Runtime and memory still grew
with input size because the source file is fully loaded and grouped before
rendering.

## Database Lifecycle Results

Command shape:

```bash
EXTRACTED_DATABASE_URL="$(python - <<'PY'
from atlas_brain.storage.config import db_settings
print(db_settings.dsn)
PY
)" /usr/bin/time -v -o tmp/faq_scale_stress_20260523/lifecycle_<rows>_time.txt \
  python scripts/smoke_content_ops_faq_lifecycle.py \
  tmp/faq_scale_stress_20260523/cfpb_<rows>_source_rows.jsonl \
  --source-format jsonl \
  --account-id acct-faq-stress-<rows>-20260523 \
  --user-id user-faq-stress \
  --title 'CFPB <rows> Row FAQ DB Lifecycle Stress 2026-05-23' \
  --min-source-rows <rows> \
  --min-saved-faqs 1 \
  --export-limit 5 \
  --max-text-chars 1200 \
  --default-field company_name=CFPB \
  --default-field contact_email=cfpb-public-archive@example.invalid \
  --default-field vendor_name=CFPB \
  --output-result tmp/faq_scale_stress_20260523/lifecycle_<rows>_result.json \
  --summary-json > tmp/faq_scale_stress_20260523/lifecycle_<rows>_stdout_summary.json
```

| Rows | Exit | Saved | Draft export | Reviewed export | Elapsed | Max RSS | Result size |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 0 | 1 | 1 | 1 | 0:02.51 | 52,676 KB | 314 KB |
| 5,000 | 0 | 1 | 1 | 1 | 0:06.23 | 84,732 KB | 566 KB |
| 10,000 | 0 | 1 | 1 | 1 | 0:13.07 | 167,620 KB | 986 KB |
| 25,000 | 0 | 1 | 1 | 1 | 0:32.59 | 374,108 KB | 2.2 MB |
| 50,000 | 0 | 1 | 1 | 1 | 1:05.32 | 681,848 KB | 4.2 MB |

For every lifecycle run:

- `stdout_summary.json` exactly matched `result.json.lifecycle_summary`
- `source_count` equaled the requested row count
- `ticket_source_count` equaled the requested row count
- `error_count=0`
- normalization warning count was `0`
- question source counts were `source_policy=8`

## Concurrent Lifecycle Results

The concurrency probe launched independent
`scripts/smoke_content_ops_faq_lifecycle.py` processes with unique account IDs,
the real database URL from `atlas_brain.storage.config.db_settings.dsn`, and
the same CFPB fixture defaults used by the single-worker runs.

Host headroom before the final probe:

- memory: 61 GiB total, 39 GiB available
- Postgres `max_connections=100`
- active Postgres connections before the final probe: `6`

| Scenario | Source rows per run | Successes | Failures | Wall time | Failure mode |
|---|---:|---:|---:|---:|---|
| 5 concurrent runs | 5,000 | 5 | 0 | 6.65s | none |
| 10 concurrent runs | 10,000 | 10 | 0 | 14.77s | none |
| 20 concurrent runs | 10,000 | 20 | 0 | 23.36s | none |
| 50 concurrent runs | 10,000 | 50 | 0 | 55.82s | none |
| 100 concurrent runs | 5,000 | 97 | 3 | 50.84s | `TooManyConnectionsError` |

The 100-way run produced three fast failures at indexes 92, 93, and 94:

```text
asyncpg.exceptions.TooManyConnectionsError: remaining connection slots are reserved for roles with the SUPERUSER attribute
```

Those failed processes returned exit code `1` and did not write their requested
`--output-result` artifacts. The failure occurs while creating the script-local
asyncpg pool, before `run_faq_lifecycle_smoke()` builds its payload and writes
the result JSON.

## Issues Surfaced

### FAQSTRESS-1 - Large uploads need an async/job boundary or explicit limit

At 50,000 rows, the deterministic generator still passed all checks, but the
run took `1:49.48` and used `590,612 KB` RSS. The full DB lifecycle also passed,
but took `1:05.32` and used `681,848 KB` RSS. This is acceptable as an operator
batch smoke, but not as a synchronous web request path.

Impact: without an async job boundary, upload limits, or backpressure, large
uploads can tie up request workers and memory even when the deterministic
generator eventually succeeds.

Resolution: parked in `HARDENING.md`. The probe's goal was to find and record
the weakness, not to introduce a job runner in this slice.

### FAQSTRESS-2 - Concurrent lifecycle runs can exhaust DB connections

At 100 concurrent 5,000-row lifecycle runs, 97 completed successfully and 3
failed with `asyncpg.exceptions.TooManyConnectionsError`. The hosted service
factory uses the host's existing shared pool, so this exact smoke shape is not
the production wiring. The production requirement still stands: hosted FAQ
execution needs bounded concurrency, queue backpressure, or explicit request
limits so customer traffic cannot consume all database connection slots.

The smoke also exposed a visibility gap: pool creation failures bypass
`--output-result`, leaving only stderr for failed runs. That should be fixed in
the smoke harness before relying on it as an automated survivability probe.

Resolution: parked in `HARDENING.md`. The next slice should add result-artifact
visibility for lifecycle pool creation failures or introduce the reusable
concurrency probe, depending on which path we want to automate first.

## Conclusion

The current deterministic FAQ path survives real-data uploads through 50,000
rows without output-check or lifecycle correctness failures. It also survives
up to 50 concurrent 10,000-row lifecycle runs on this host. The first observed
failure appears at 100 concurrent 5,000-row lifecycle runs, where Postgres
connection slots saturate.

The production hardening requirement is operational: large uploads need
explicit limits and/or background job execution, and concurrent hosted FAQ
execution needs bounded database pressure before this flow is exposed as a
synchronous hosted upload endpoint.
