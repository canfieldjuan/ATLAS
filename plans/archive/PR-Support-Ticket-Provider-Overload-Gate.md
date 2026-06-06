# PR-Support-Ticket-Provider-Overload-Gate

## Why this slice exists

PR #942 proved large loader-backed support-ticket inputs stay bounded through
preview, plan, and execute, including 25 concurrent successful execute calls.
The remaining survivability question in this lane is overload behavior: when
execute is already at capacity, the route should reject before it invokes a
host loader that may read or parse a large customer export.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Robust testing

1. Add a support-ticket-provider execute overload test.
2. Prove the second request gets the existing `429` capacity response.
3. Prove the rejected request does not call the support-ticket source loader.

### Files touched

- `plans/PR-Support-Ticket-Provider-Overload-Gate.md`
- `tests/test_support_ticket_provider_landing_blog_execute.py`

## Mechanism

The test mounts the real Content Ops execute route with
`execute_max_concurrency=1`, a loader-backed `SupportTicketInputProvider`, and a
blocking fake landing-page service. The first request enters the service and
holds the only execute slot. The second request is issued while the slot is
held and must fail with `content_ops_execute_at_capacity`.

The loader records each call. The assertion that the loader has only one call
after the rejected request proves overload rejection happens before large
support-ticket material is loaded or packaged.

## Intentional

- No production code changes; the route already gates before provider loading.
- Landing page is enough for this overload proof because the execute gate sits
  before output dispatch and provider packaging.
- No live LLM, DB, or hosted upload path is used.
- Local review's cross-layer caller hints are generic `__init__` and
  `generate` references from a test fake class; no production symbol or caller
  is changed.

## Deferred

- Future PR: hosted upload/background execution policy for large customer files
  once persisted support-ticket uploads are wired into the provider loader.
- Future PR: hosted HTTP-level load testing after the deployed auth/upload path
  exists.
- Parked hardening: none. `HARDENING.md` was scanned; the current FAQ scale
  entry remains owned by the FAQ generation lane.

## Verification

- `python -m pytest tests/test_support_ticket_provider_landing_blog_execute.py -q`
  - passed, 10 tests.
- `scripts/local_pr_review.sh --allow-dirty`
  - passed.
- `scripts/local_pr_review.sh`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| Route overload test | ~65 |
| **Total** | **~130** |
