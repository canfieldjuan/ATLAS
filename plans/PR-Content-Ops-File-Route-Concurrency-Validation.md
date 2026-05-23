# PR-Content-Ops-File-Route-Concurrency-Validation

## Why this slice exists

PR #878 proved a single uploaded-file route write at the 10,000-row cap. The
next survivability question is whether the same route write path stays correct
under concurrent customer-style uploads, and where it starts to show resource
pressure.

This slice records progressive local Postgres pressure probes for the
uploaded-file route write smoke. It is validation only unless the concurrent
flow cannot be exercised.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Run multiple concurrent `--write` route-smoke processes against local
   Postgres using CFPB-derived source-row fixtures.
2. Confirm success counts, DB row counts, and failure modes for each pressure
   level.
3. Document timing, memory, and any surfaced issues.
4. Park non-blocking hardening issues in `HARDENING.md` if the pressure run
   exposes resource ceilings.

### Files touched

- `docs/extraction/validation/content_ops_file_route_concurrency_validation_2026-05-23.md`
- `plans/PR-Content-Ops-File-Route-Concurrency-Validation.md`
- `HARDENING.md`

## Mechanism

Use the existing `scripts/smoke_content_ops_ingestion_file_route.py` command in
`--write` mode with unique account ids per process and `--replace-existing`.
Run progressive batches:

- smaller concurrent 1,000-row writes to exercise request/process pressure;
- concurrent 10,000-row writes to exercise the current route cap under load;
- a higher-concurrency probe only if earlier runs pass cleanly.

Each process writes its compact JSON result under `tmp/`. After each batch,
summarize exit codes and query `campaign_opportunities` for the scoped account
prefix so the database state matches the result artifacts.

## Intentional

- No reusable load-test runner in this slice. The pressure commands are
  validation evidence; if repeated often, a future slice can promote them into a
  checked-in harness.
- No product code changes unless the route write path cannot be exercised.
- Rows are scoped by unique account prefixes and use `--replace-existing` so
  reruns are isolated and idempotent per account.

## Deferred

- Background jobs, durable upload storage, and cross-process admission control
  remain separate hardening work if this pressure test shows they are needed.
- Cleanup tooling for validation rows remains out of scope unless DB bloat
  becomes a blocker.

## Verification

- 5 concurrent 10,000-row writes:
  - 5 successes, 0 failures, 50,000 inserted rows, 0 warnings.
  - Elapsed `0:05.93`; max RSS `154760 KB`.
- 20 concurrent 1,000-row writes:
  - 20 successes, 0 failures, 20,000 inserted rows, 0 warnings.
  - Elapsed `0:01.69`; max RSS `65868 KB`.
- 100 concurrent 1,000-row writes:
  - 100 successes, 0 failures, 100,000 inserted rows, 0 warnings.
  - Elapsed `0:07.85`; max RSS `66028 KB`.
- 150 concurrent 1,000-row writes:
  - 141 successes, 9 failures, 141,000 inserted rows, 0 warnings.
  - Failure mode: `TooManyConnectionsError: None` during pool creation.
  - Elapsed `0:12.49`; max RSS `66280 KB`.
- Database count checks matched successful run counts for all four probes.
- Local Postgres settings: `max_connections=100`,
  `superuser_reserved_connections=3`.
- `FILECONCURRENCY-1` parked in `HARDENING.md`.
- `bash scripts/local_pr_review.sh --allow-dirty`
  - Pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| Validation record | ~150 |
| HARDENING entry | ~10 |
| **Total** | ~255 |
