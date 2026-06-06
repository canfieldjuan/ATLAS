# PR-Content-Ops-File-Multiprocess-Load-Probe

## Why this slice exists

`FILECONCURRENCY-2` is now parked for the remaining uploaded-file import
deployment risk: the route admission gate is process-local. The previous
in-process runner proved one router process uses one shared pool and returns
deterministic 429 responses when its local gate is full. It does not prove what
happens when multiple app processes each own a separate route gate.

This slice adds the thinnest reusable probe for that topology. It fans out the
already-merged in-process load runner across multiple child processes and
summarizes the aggregate admitted writes, 429 responses, and unexpected
failures. That gives reviewers and future sessions a concrete way to reproduce
the remaining risk before deciding whether to build distributed admission or a
durable import job boundary.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Add a multiprocess uploaded-file import load probe script.
2. Reuse `scripts/smoke_content_ops_ingestion_file_route_inprocess_load.py` as
   the per-process worker.
3. Emit one parent JSON result plus child result artifacts.
4. Add deterministic tests for command construction, aggregate classification,
   and artifact writing without requiring Postgres.

### Files touched

- `scripts/smoke_content_ops_ingestion_file_route_multiprocess_load.py`
- `tests/test_smoke_content_ops_ingestion_file_route_multiprocess_load.py`
- `plans/PR-Content-Ops-File-Multiprocess-Load-Probe.md`

## Mechanism

The parent script launches `--processes N` child Python processes. Each child
runs the existing in-process load runner with its own account id suffix and its
own output JSON path. The parent waits for all children, reads their compact
JSON results, and summarizes:

- child process count,
- child exit codes,
- total successes,
- total admission 429 responses,
- total unexpected failures,
- total inserted rows,
- per-child artifact paths.

The probe exits successfully only if every child exits successfully and the
aggregate meets caller-provided minimums for total successes and total
admission 429 responses.

## Intentional

- This is validation tooling, not a product fix. The parked hardening item still
  owns distributed admission / durable job behavior.
- The parent does not parse database credentials or open a database connection.
  It passes through the same database URL option/env behavior used by the
  existing child runner.
- Each child gets a unique account id suffix so live runs can verify inserted
  rows without write collisions between processes.

## Deferred

- Distributed semaphore, shared queue, durable import jobs, and queue
  observability remain parked under `FILECONCURRENCY-2`.
- Automatic database cleanup is left out; live runs should use unique account
  ids, as the existing validation scripts do.

## Verification

- Focused probe tests:
  - `python -m pytest tests/test_smoke_content_ops_ingestion_file_route_multiprocess_load.py -q`
  - `4 passed`
- Live multiprocess probe:
  - `python scripts/smoke_content_ops_ingestion_file_route_multiprocess_load.py tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl --source-format jsonl --source cfpb-route-multiprocess-load-20260523 --min-source-rows 1000 --default-field company_name=CFPB --default-field vendor_name=CFPB --default-field contact_email=cfpb-public-archive@example.invalid --account-id acct-file-route-multiprocess-load-20260523 --user-id user-file-route-multiprocess-load --processes 3 --child-concurrency 2 --child-import-max-concurrency 1 --child-min-successes 1 --child-expect-at-capacity-min 1 --min-total-successes 3 --expect-total-at-capacity-min 3 --output-dir tmp/content_ops_file_route_multiprocess_load_20260523 --output-result tmp/content_ops_file_route_multiprocess_load_20260523/result.json --json`
  - Exit `0`; `3` child processes, `3` successes, `3` admission 429
    responses, `0` unexpected failures, `3000` rows inserted.
  - DB verification for `acct-file-route-multiprocess-load-20260523-p%` /
    `vendor_retention`: `3` accounts, `3000` rows, `1000` distinct target IDs.
- Focused load-runner suite and compile check:
  - `python -m pytest tests/test_smoke_content_ops_ingestion_file_route_inprocess_load.py tests/test_smoke_content_ops_ingestion_file_route_multiprocess_load.py -q`
  - `8 passed`
  - Python compile check passed for the multiprocess probe script.
- Local PR review:
  - `bash scripts/local_pr_review.sh --allow-dirty`
  - Pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 96 |
| Multiprocess probe script | 360 |
| Probe tests | 241 |
| Total | 703 |

The diff is over the 400 LOC target because the probe needs a real CLI,
subprocess fan-out, child artifact management, aggregate validation, and
deterministic fake-subprocess tests to be useful as production-hardening
evidence.
