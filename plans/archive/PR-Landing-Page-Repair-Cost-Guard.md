# PR-Landing-Page-Repair-Cost-Guard

## Why this slice exists

Saved landing-page repair now calls the LLM from the review drawer. That is the
right operator workflow, but the endpoint should not spend twice when a user
double-clicks Repair, the browser retries, or two operators repair the same
draft at the same time.

This slice adds a backend in-flight guard at the repair endpoint so duplicate
repair requests for the same tenant-scoped landing-page draft fail before LLM
provider resolution or prompt generation.

## Scope (this PR)

Ownership lane: content-ops/landing-page-repair-cost-guard

1. Add a Postgres advisory lock around saved landing-page draft repair.
2. Return `409` for duplicate in-flight repair requests before LLM work starts.
3. Release the advisory lock after successful or failed repair handling.
4. Add API tests for lock acquisition, lock release, and duplicate-request
   rejection before LLM calls.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Repair-Cost-Guard.md` | Plan doc for this cost-control slice. |
| `HARDENING.md` | Park non-blocking review findings outside this thin slice. |
| `extracted_content_pipeline/api/generated_assets.py` | Add tenant-scoped landing-page repair advisory lock around the repair endpoint. |
| `tests/test_extracted_content_asset_api.py` | Cover successful lock release and duplicate in-flight repair rejection. |

## Mechanism

The repair endpoint already loads the draft through tenant-scoped repository
access before it resolves LLM and skill providers. After that load and the
approved-row guard, this slice acquires a Postgres session advisory lock using
a stable namespace plus an account/draft lock key. The key follows the draft's
repository scope, so two operators in the same account contend for the same
draft lock.

If the lock is already held, the endpoint returns `409` with
`landing page draft repair already in progress` and does not resolve providers
or call the LLM. If the lock is acquired, the existing repair service runs
unchanged and the lock is released in a `finally` block through the same manual
pool connection. The same connection is then released back to the pool.

The extracted router still supports host test doubles or non-asyncpg pools that
do not expose `acquire()`. Those environments get the existing behavior instead
of a new hard port requirement.

## Intentional

- No UI-only debounce. The guard lives at the backend integration point where
  spend is triggered.
- No new draft status such as `repairing`, avoiding a migration and avoiding a
  row that can get stuck if a process dies mid-request.
- No sequential daily/monthly quota in this slice. This protects duplicate
  in-flight requests, which is the immediate double-spend risk from the repair
  button.

## Deferred

- `PR-Landing-Page-Repair-Quota` can add persisted per-draft or per-tenant
  repair-attempt quotas if operators need hard spend budgets across time.
- `PR-Landing-Page-Repair-Audit-Trail` can persist repair attempt metadata for
  failed attempts, not just successful repaired drafts.

## Parked hardening

- Surface skipped landing-page repair locks when a host pool has no
  `acquire()` method.
- Consider a wider advisory-lock hash key than `hashtext()`.
- Revisit holding the repair lock connection across LLM latency if repair
  volume grows.

## Verification

- Python compile for `extracted_content_pipeline/api/generated_assets.py` ->
  passed.
- Focused pytest for `tests/test_extracted_content_asset_api.py` -> 51 passed.
- Extracted content pipeline validation -> passed.
- Extracted reasoning import guard -> passed.
- Extracted standalone audit -> passed with 0 findings.
- ASCII Python policy -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan and hardening docs | ~105 |
| API repair lock | ~115 |
| API tests | ~110 |
| Total | ~330 |
