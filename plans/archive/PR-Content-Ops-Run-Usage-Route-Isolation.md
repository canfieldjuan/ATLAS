# PR: Content Ops Run Usage Route Isolation

## Why this slice exists

PR-Content-Ops-Run-Usage-Summary added per-run usage summaries to the hosted
Content Ops execute response. The reviewer correctly noted that the new route
path is covered by mocked-pool tests, while the real Postgres account isolation
coverage lives one layer lower in the shared usage-summary helper.

This slice adds the missing route-level proof: when the execute route stamps a
request id and reads usage for that id, the summary remains scoped to the
authenticated account even if another account has usage rows with the same
request id.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add a real-Postgres integration test for the hosted execute usage-summary
   path.
2. Keep the implementation unchanged unless the test reveals a source bug.
3. Prove the route uses scope account id plus request id when attaching
   `usage_summary`.

### Files touched

- `plans/PR-Content-Ops-Run-Usage-Route-Isolation.md` - Plan doc for this
  hardening proof.
- `tests/test_extracted_content_control_surface_api.py` - Add route-level
  Postgres isolation coverage for execute response usage summaries.

## Mechanism

The test seeds `llm_usage` rows for two different accounts with the same
request id, monkeypatches the execute route's request-id generator to that id,
and calls the hosted `/execute` route as account A. The returned
`usage_summary` must include only account A's row.

That verifies the full route wiring:

- authenticated scope resolution
- request-id stamping
- execute response usage-summary attachment
- SQL account/request filtering through the shared summary helper

## Intentional

- This is a test-only slice. The implementation from the previous PR is left
  alone because the route already passes account id and request id into the
  shared helper.
- The integration test skips when `EXTRACTED_DATABASE_URL` / `DATABASE_URL` is
  unavailable, matching the existing Postgres integration tests.

## Deferred

- No new product behavior is deferred.
- Parked hardening: none.

## Verification

- Focused hosted execute route pytest for mocked usage summary, route-level
  Postgres isolation, and trace context - 2 passed, 1 skipped. The integration
  test skipped locally because neither `EXTRACTED_DATABASE_URL` nor
  `DATABASE_URL` is exported in this clean worktree shell.
- Focused Postgres isolation pytest for the shared usage helper and the new
  hosted route isolation test - 2 skipped for the same missing database URL.
- Python compile over the touched backend test file - passed.
- Full extracted content control-surface API pytest file - 117 passed, 1
  skipped.
- Environment lookup for database URLs in repo-local env files - no
  `EXTRACTED_DATABASE_URL` / `DATABASE_URL` entries found.
- Whitespace check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| Integration test | ~135 |
| **Total** | **~210** |

This stays below the 400 LOC soft cap.
