# PR-Content-Ops-FAQ-Search-Latency-Gates

## Why this slice exists

PR-Content-Ops-FAQ-Search-Concurrency-Smoke added a real-Postgres read
concurrency smoke for FAQ deflection search and recorded baseline latency. The
next survivability step is to let that same smoke fail when a caller supplies
explicit latency budgets.

This keeps production hardening grounded in the real route repository path:
first prove tenant/corpus isolation and read concurrency, then make the smoke
usable as a repeatable latency gate for larger environments.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: robust testing.

1. Add optional p95 and single-request latency budget flags to the FAQ search
   concurrency smoke.
2. Include latency budget pass/fail details in the result JSON.
3. Fail the smoke when isolation checks fail or an explicit latency budget is
   exceeded.
4. Fix the p95 calculation now that p95 can drive a gate.
5. Add focused unit coverage for passing and failing latency budgets.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Latency-Gates.md`
- `scripts/smoke_content_ops_faq_search_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_concurrency.py`

## Mechanism

The smoke accepts optional budget flags:

- `--max-p95-ms`
- `--max-single-request-ms`

When absent, behavior remains baseline-only: the smoke records latency and fails
only for search exceptions, unexpected result rows, or isolation failures. When a
budget is present, `run_smoke` compares the computed latency summary against the
provided threshold and adds a deterministic `latency_budget` object to the JSON
summary. Any exceeded threshold sets `ok=false` and returns exit code `1`.

The p95 calculation moves to nearest-rank percentile indexing so the value used
for a budget gate is not biased low on small smoke runs.

## Intentional

- No default threshold is introduced. Different laptops, CI runners, and
  deployed databases need different budgets.
- No hosted route load test is added here. This keeps the slice on the existing
  direct repository smoke while the deployed host URL/token remains outside this
  repo.
- No retry/backoff behavior is added. A latency gate should expose slow reads,
  not mask them.

## Deferred

- HTTP-route concurrency with auth tokens remains deferred until the deployed
  backend URL and bearer token are available.
- Production SLO values remain environment-owned; this PR only adds the
  mechanism to enforce a caller-provided budget.
- Write-path replacement concurrency remains separate from this read-path smoke.
- Setup-time JSON failure artifacts for unreachable DB hosts should be parked in
  `HARDENING.md` after PR #902 releases that shared file; this slice keeps the
  gate logic focused on completed search runs and avoids a cross-session
  `HARDENING.md` collision.

## Verification

- `pytest tests/test_smoke_content_ops_faq_search_concurrency.py -q` passed with
  7 tests.
- Python compile check for the smoke script and focused test module passed.
- Live DB smoke with latency budgets was not completed because the active shell
  has no `EXTRACTED_DATABASE_URL` or `DATABASE_URL`, and `.env.backup` points at
  a DB host that does not resolve in this worktree.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 88 |
| Smoke script | 57 |
| Tests | 55 |
| **Total** | **200** |
