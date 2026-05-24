# Content Ops Host Import Multiprocess Proof - 2026-05-23

## Summary
The uploaded-file import route was exercised through the existing multiprocess
load runner with the Atlas Postgres advisory-lock admission provider enabled.
This proves the host-owned provider bounds imports across child processes, not
only inside one extracted router instance.

Result: passed.

## Command
```bash
python scripts/smoke_content_ops_ingestion_file_route_multiprocess_load.py \
  tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl \
  --source-format jsonl \
  --source cfpb-route-host-postgres-admission-20260523 \
  --min-source-rows 1000 \
  --default-field company_name=CFPB \
  --default-field vendor_name=CFPB \
  --default-field contact_email=cfpb-public-archive@example.invalid \
  --account-id acct-host-import-admission-multiprocess-20260523 \
  --user-id user-host-import-admission \
  --processes 3 \
  --child-concurrency 2 \
  --child-import-max-concurrency 1 \
  --child-min-successes 1 \
  --child-expect-at-capacity-min 1 \
  --min-total-successes 1 \
  --expect-total-at-capacity-min 5 \
  --allow-capacity-only-children \
  --admission-provider postgres \
  --output-dir tmp/content_ops_host_import_admission_multiprocess_20260523 \
  --output-result tmp/content_ops_host_import_admission_multiprocess_20260523/result.json \
  --json
```

## Observed Result
```json
{
  "ok": true,
  "admission_provider": "postgres",
  "processes": 3,
  "child_concurrency": 2,
  "child_import_max_concurrency": 1,
  "allow_capacity_only_children": true,
  "summary": {
    "processes": 3,
    "successful_processes": 3,
    "capacity_only_processes": 2,
    "failed_processes": 0,
    "successes": 1,
    "at_capacity": 5,
    "unexpected_failures": 0,
    "inserted": 1000,
    "returncode_counts": {
      "0": 1,
      "1": 2
    }
  }
}
```

## Interpretation
With three child processes and two concurrent imports per child, the run created
six simultaneous import attempts. The shared Postgres admission provider allowed
one import to proceed and returned five
`content_ops_ingestion_import_at_capacity` responses. Two child processes
correctly had no successful writes because the single global advisory-lock slot
was already held elsewhere.

That is the expected behavior for this proof: the capacity window was shared
across processes, and there were no unexpected child failures.

## Closed Hardening
`FILECONCURRENCY-2 - Uploaded-file imports need hosted multiprocess proof` is
closed by this run. The durable background job / queue visibility half is
reframed as conditional on a future async-import design because this proof
validates the current synchronous admission cap instead of introducing
long-running import jobs.
