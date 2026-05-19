# PR-Content-Ops-Review-Source-Live-Closeout

## Why this slice exists

PR #597 and PR #598 closed the review-source Postgres smoke path. A local live
G2/Slack run now proves the path through source export, source-row import,
DB-backed offline draft persistence, and draft export. The coordination docs
still showed the preflight slice as pending, so other sessions could avoid a
stale hot zone.

## Scope (this PR)

- Remove the merged review-source schema-preflight row from the in-flight
  coordination ledger.
- Update per-product state to point at #598 and clear the hot zone.
- Record the live G2 Postgres smoke result in the active Content Ops backlog
  and package status.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `extracted_content_pipeline/STATUS.md`
- `plans/PR-Content-Ops-Review-Source-Live-Closeout.md`

## Mechanism

This is documentation and coordination only. It records the live run result:
1 G2/Slack source row imported into `campaign_opportunities`, 2 offline
deterministic campaign drafts persisted, and the draft export CLI returned
both rows for `account_id=content_ops_smoke`.

## Intentional

- No code changes.
- No schema changes.
- No new active Content Ops backlog item from a successful smoke.

## Deferred

- Running Capterra and TrustRadius through the same Postgres smoke.
- Live provider generation over imported review-source rows.
- Broader generated-asset runs from review-source evidence.

## Verification

- Live G2/Slack Postgres smoke -> passed locally after creating
  `campaign_opportunities`.
- Draft export CLI for `account_id=content_ops_smoke` -> returned 2 rows.
- `git diff --check` -> passed.
- `scripts/local_pr_review.sh --allow-dirty` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination docs | 7 |
| Backlog/status docs | 23 |
| Plan | 61 |
| Total | 91 |
