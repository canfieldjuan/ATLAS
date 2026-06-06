# PR-Content-Ops-Live-Provider-Smoke-Closeout

## Why this slice exists

PR #628 and PR #630 closed the live-provider smoke gap for the two public
source-row Postgres paths: scraped review sources and CFPB support-ticket-like
sources. The coordination and backlog docs still show the CFPB slice as in
flight and still describe the live-provider run gap as open. This slice keeps
the multi-session ledger accurate before the next Content Ops or reasoning-core
work starts.

## Scope (this PR)

1. Remove the merged CFPB live-provider smoke row from the in-flight ledger.
2. Advance `extracted_content_pipeline` state to PR #630.
3. Record the review-source and CFPB live-provider smoke closeout in the active
   Content Ops backlog.
4. Refresh product status wording so the available smoke modes are clear.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `extracted_content_pipeline/STATUS.md`
- `plans/PR-Content-Ops-Live-Provider-Smoke-Closeout.md`

## Mechanism

Documentation-only closeout. No runtime code, tests, APIs, or schemas change.

## Intentional

- This does not claim that a live operator run was executed in this environment;
  provider and database credentials were not loaded here.
- This keeps the source-breadth backlog waiting on a real host export fixture
  rather than inventing more hypothetical adapters.
- This keeps reasoning product depth as a separate `extracted_reasoning_core`
  decision path.

## Deferred

- Manual live-provider smoke runs with host credentials.
- Any new source adapters or aliases driven by a real host export fixture.
- Further reasoning-core productization if a concrete runtime need appears.

## Verification

- `git diff --check` -> passed.
- Plan/docs presence check -> passed.
- Grep spot-check for #630/live-provider status wording -> passed.
- `scripts/local_pr_review.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination/state/backlog/status docs | ~45 |
| Plan doc | ~55 |
| **Total** | **~100** |

This is below the 400 LOC review budget.
